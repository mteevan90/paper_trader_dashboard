"""run_optuna_parallel.py — Path #3 fan-out launcher.

Runs an Optuna hypothesis study across N independent OS processes, each
single-threaded internally (`n_jobs=1`). Optuna's SQLite storage backend
serializes the trial commits but TPE sampling + the per-trial backtest
work happen in parallel processes, sidestepping the GIL bottleneck that
the in-process ThreadPoolExecutor (Optuna 4.8 default) hits.

The launcher mirrors the run_hypothesis.py CLI surface and adds:
  --n-workers      number of concurrent worker processes (default 8)
  --n-trials-total trials to distribute across workers (replaces
                   run_hypothesis's --n-trials; per-worker count is
                   ceil(total/n-workers))
  --kill-on-error  fail fast if any worker exits non-zero (default off:
                   keep waiting on healthy workers, log the failure)

This file is dual-mode. When invoked with --worker-mode it acts as a
trial-only worker (run_study() with no save). The launcher subprocess-
spawns itself in worker mode to keep everything in one file.

The dashboard save (save_hypothesis_result) is called ONCE by the
launcher after all workers finish. Workers don't save — that avoids the
race where each worker's save sees a different "best trial" snapshot
and writes a different best_<study>_<n>/ directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ---------------------------------------------------------------------------
# Worker mode
# ---------------------------------------------------------------------------

def _worker_main() -> int:
    """One worker process: run a chunk of trials against a shared study.

    Env vars (PAPER_TRADER_DATA_ROOT, PAPER_TRADER_ARCHITECTURE) are
    inherited from the launcher via subprocess env propagation. We
    DON'T set them here — that's the launcher's job pre-spawn.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--worker-mode", action="store_true")
    p.add_argument("--study-name", required=True)
    p.add_argument("--n-trials",   type=int, required=True)
    p.add_argument("--worker-id",  type=int, required=True)
    p.add_argument("--range-overrides-json", default="{}")
    p.add_argument("--fixed-values-json",    default="{}")
    args = p.parse_args()

    range_overrides = json.loads(args.range_overrides_json) or None
    fixed_values    = json.loads(args.fixed_values_json)    or None
    # JSON keys are str; range_overrides values come back as lists →
    # convert to tuple (low, high) so build_search_space's _suggest can
    # tuple-unpack like the launcher's in-process callers do.
    if range_overrides:
        range_overrides = {k: tuple(v) for k, v in range_overrides.items()}

    from optuna_runner import run_study   # noqa: E402  (after sys.path)

    t0 = time.perf_counter()
    print(f"[WORKER {args.worker_id}] PID={os.getpid()} "
          f"running {args.n_trials} trials on study={args.study_name!r}")
    run_study(
        n_trials=args.n_trials,
        n_jobs=1,
        study_name=args.study_name,
        smoke_test=False,
        range_overrides=range_overrides,
        fixed_values=fixed_values,
    )
    print(f"[WORKER {args.worker_id}] done in {time.perf_counter() - t0:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Launcher mode
# ---------------------------------------------------------------------------

# Same lists as run_hypothesis.py — kept locally so the launcher is
# self-contained and doesn't rely on internals of its sibling.
_SEARCH_SPACE_PARAMS: frozenset[str] = frozenset({
    "weight_fundamental", "weight_technical", "weight_model",
    "macro_threshold_low", "macro_threshold_gap", "atr_multiplier",
    "analyst_weight", "rebalance_frequency_days", "position_count",
})
_DIRECT_FIELDS: frozenset[str] = frozenset({
    "weight_fundamental", "weight_technical", "weight_model",
    "macro_threshold_low", "atr_multiplier", "analyst_weight",
    "rebalance_frequency_days", "position_count",
})
_TRANSLATED_FIELDS: frozenset[str] = frozenset({"macro_threshold_high"})
_DERIVED_FIELDS:    frozenset[str] = frozenset({"weight_alt"})


def _parse_override_range(raw: str) -> tuple[str, tuple[float, float]]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--override-range must be key=min,max; got {raw!r}")
    key, vals = raw.split("=", 1)
    key = key.strip()
    parts = [p.strip() for p in vals.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--override-range value must be 'min,max'; got {vals!r}")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--override-range min/max must be numbers; got {vals!r}: {e}")
    if lo > hi:
        raise argparse.ArgumentTypeError(
            f"--override-range min ({lo}) > max ({hi}) in {raw!r}")
    if key not in _SEARCH_SPACE_PARAMS:
        raise argparse.ArgumentTypeError(
            f"--override-range key {key!r} is not a search-space param. "
            f"Valid keys: {sorted(_SEARCH_SPACE_PARAMS)}")
    return key, (lo, hi)


def _translate_holds(holds_csv: str, base_config: dict) -> dict:
    holds = [h.strip() for h in holds_csv.split(",") if h.strip()]
    fixed: dict = {}
    for field in holds:
        if field in _DERIVED_FIELDS:
            raise SystemExit(
                f"--hold-tunables-fixed: {field!r} is derived, not "
                f"settable independently.")
        if field in _DIRECT_FIELDS:
            if field not in base_config:
                raise SystemExit(
                    f"--hold-tunables-fixed: {field!r} not present in "
                    f"the base config.")
            fixed[field] = base_config[field]
            continue
        if field == "macro_threshold_high":
            for needed in ("macro_threshold_low", "macro_threshold_high"):
                if needed not in base_config:
                    raise SystemExit(
                        f"macro_threshold_high requires both "
                        f"macro_threshold_low and macro_threshold_high "
                        f"in the base config; missing {needed}.")
            fixed["macro_threshold_low"] = base_config["macro_threshold_low"]
            fixed["macro_threshold_gap"] = (
                base_config["macro_threshold_high"]
                - base_config["macro_threshold_low"])
            continue
        if field == "macro_threshold_gap":
            raise SystemExit(
                "--hold-tunables-fixed: pass macro_threshold_high; "
                "the launcher translates to gap.")
        raise SystemExit(
            f"--hold-tunables-fixed: unknown field {field!r}.")
    return fixed


def _build_search_ranges_log(fixed_values: dict,
                             range_overrides: dict) -> dict:
    from optuna_runner import _DEFAULT_RANGES
    out: dict = {}
    for name in sorted(_SEARCH_SPACE_PARAMS):
        if name in fixed_values:
            out[name] = ["FIXED", fixed_values[name]]
        elif name in range_overrides:
            lo, hi = range_overrides[name]
            out[name] = [lo, hi]
        else:
            lo, hi = _DEFAULT_RANGES[name]
            out[name] = [lo, hi]
    return out


def _launcher_main() -> int:
    p = argparse.ArgumentParser(
        description="Process-fan-out launcher for Optuna hypothesis studies.")
    p.add_argument("--study-name",    required=True)
    p.add_argument("--hypothesis-id", required=True)
    p.add_argument("--base-config",   required=True)
    p.add_argument("--override-range", action="append", default=[],
                   metavar="KEY=MIN,MAX")
    p.add_argument("--hold-tunables-fixed", default="", metavar="CSV")
    p.add_argument("--window", choices=["train", "validate"], required=True)
    p.add_argument("--n-trials-total", type=int, required=True,
                   help="Total trials across all workers.")
    p.add_argument("--n-workers", type=int, default=8,
                   help="Number of concurrent worker processes (default 8).")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--architecture", default="legacy",
                   choices=["legacy", "regime-dependent"])
    p.add_argument("--cache-snapshot", default=None)
    p.add_argument("--kill-on-error", action="store_true",
                   help="If any worker exits non-zero, immediately "
                        "terminate all other workers. Default off: keep "
                        "waiting on the healthy workers so we don't lose "
                        "their data on one bad apple.")
    p.add_argument("--worker-log-dir", default=None,
                   help="Directory for per-worker stdout/stderr logs. "
                        "Default: models/cache/parallel_logs/<study_name>/")
    args = p.parse_args()

    if args.window == "validate":
        raise SystemExit(
            "Validation pathway not implemented; this launcher is "
            "training-only by design.")
    if args.n_workers < 1:
        raise SystemExit("--n-workers must be >= 1")
    if args.n_trials_total < 1:
        raise SystemExit("--n-trials-total must be >= 1")

    # ---- Cache snapshot env-var setup ----
    if args.cache_snapshot:
        snap_root = (_REPO_ROOT / "models" / "snapshots"
                     / args.cache_snapshot).resolve()
        if not snap_root.is_dir():
            raise SystemExit(
                f"[PARALLEL] Snapshot not found: {snap_root}")
        os.environ["PAPER_TRADER_DATA_ROOT"] = str(snap_root)
        print(f"[PARALLEL] Using cache snapshot: "
              f"{args.cache_snapshot} ({snap_root})")
    if args.architecture != "legacy":
        os.environ["PAPER_TRADER_ARCHITECTURE"] = args.architecture
        print(f"[PARALLEL] Using architecture: {args.architecture}")

    # ---- Load base config ----
    base_path = Path(args.base_config)
    if not base_path.exists():
        raise SystemExit(f"--base-config not found: {base_path}")
    with open(base_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    base_config = meta.get("config")
    if not isinstance(base_config, dict):
        raise SystemExit(f"--base-config meta.json missing 'config' key.")

    # ---- Parse overrides + holds ----
    range_overrides: dict = {}
    for raw in args.override_range:
        key, rng = _parse_override_range(raw)
        if key in range_overrides:
            raise SystemExit(f"--override-range twice for {key!r}")
        range_overrides[key] = rng
    fixed_values = _translate_holds(args.hold_tunables_fixed, base_config)
    overlap = set(fixed_values) & set(range_overrides)
    if overlap:
        raise SystemExit(
            f"Tunables both held fixed and given override range: "
            f"{sorted(overlap)}")

    # ---- Pre-create study so workers all join the same one ----
    # Optuna's create_study(load_if_exists=True) is idempotent and
    # safe to race-call from N workers, but pre-creating in the
    # launcher gives us a clean log line and avoids an N-way race for
    # the first SQLite write on study creation.
    import optuna           # noqa: E402  (after env-var setup)
    from optuna.samplers import TPESampler
    from optuna_runner import STUDY_DB_PATH
    storage_url = f"sqlite:///{STUDY_DB_PATH}"
    optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        sampler=TPESampler(),
        direction="maximize",
        load_if_exists=True,
    )
    print(f"[PARALLEL] Study {args.study_name!r} ready at {storage_url}")

    # ---- Distribute trials across workers ----
    per_worker = math.ceil(args.n_trials_total / args.n_workers)
    print(f"[PARALLEL] n_trials_total={args.n_trials_total}, "
          f"n_workers={args.n_workers}, trials_per_worker={per_worker} "
          f"(actual total = {per_worker * args.n_workers})")

    # ---- Worker log directory ----
    log_dir = Path(args.worker_log_dir) if args.worker_log_dir else (
        _REPO_ROOT / "models" / "cache" / "parallel_logs" / args.study_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[PARALLEL] Worker logs -> {log_dir}/ (run timestamp {ts})")

    # ---- Spawn workers ----
    range_overrides_json = json.dumps(
        {k: list(v) for k, v in range_overrides.items()})
    fixed_values_json = json.dumps(fixed_values)
    worker_cmd_base = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-mode",
        "--study-name", args.study_name,
        "--n-trials",   str(per_worker),
        "--range-overrides-json", range_overrides_json,
        "--fixed-values-json",    fixed_values_json,
    ]

    procs: list[tuple[int, subprocess.Popen]] = []
    log_files: list = []
    t_start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[PARALLEL] launching {args.n_workers} workers at {started_at}...")

    for w in range(args.n_workers):
        log_path = log_dir / f"worker_{w:02d}_{ts}.log"
        f = open(log_path, "w", encoding="utf-8", errors="replace")
        log_files.append(f)
        cmd = worker_cmd_base + ["--worker-id", str(w)]
        proc = subprocess.Popen(
            cmd, cwd=str(_REPO_ROOT),
            stdout=f, stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        procs.append((w, proc))
        print(f"[PARALLEL]   worker {w:>2} pid={proc.pid} "
              f"log={log_path.name}")

    # ---- Wait for completion ----
    failures: list[tuple[int, int]] = []
    remaining = list(procs)
    try:
        while remaining:
            time.sleep(1.0)
            still: list[tuple[int, subprocess.Popen]] = []
            for w, proc in remaining:
                rc = proc.poll()
                if rc is None:
                    still.append((w, proc))
                    continue
                if rc != 0:
                    failures.append((w, rc))
                    print(f"[PARALLEL]   worker {w} exited rc={rc}")
                    if args.kill_on_error:
                        print("[PARALLEL] --kill-on-error: terminating "
                              "remaining workers")
                        for w2, p2 in still:
                            try: p2.terminate()
                            except Exception: pass
                        still = []
                        break
                else:
                    print(f"[PARALLEL]   worker {w} done OK")
            remaining = still
    finally:
        for f in log_files:
            try: f.close()
            except Exception: pass

    wall_s = time.perf_counter() - t_start

    # ---- Summary across all workers (read from the SQLite study) ----
    study = optuna.load_study(study_name=args.study_name, storage=storage_url)
    n_complete = sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.PRUNED)
    n_fail = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.FAIL)
    n_total = len(study.trials)

    print()
    print(f"[PARALLEL] === fan-out done in {wall_s:.1f}s ===")
    print(f"[PARALLEL] workers: ok={args.n_workers - len(failures)}, "
          f"failed={len(failures)} {failures or ''}")
    print(f"[PARALLEL] trials in study: total={n_total}, "
          f"complete={n_complete}, pruned={n_pruned}, fail={n_fail}")
    if n_complete:
        print(f"[PARALLEL] mean wall per trial slot: "
              f"{wall_s / max(n_complete, 1):.2f}s "
              f"(wall_s / n_complete; throughput metric)")

    # ---- Single save at the end ----
    if n_complete == 0:
        print("[PARALLEL] No completed trials — skipping save.")
        return 1 if failures else 0

    from optuna_runner import save_hypothesis_result   # noqa: E402
    search_ranges_meta = _build_search_ranges_log(fixed_values, range_overrides)
    base_config_ref = str(base_path).replace("\\", "/")
    print(f"\n[PARALLEL] Saving dashboard_results for best trial...")
    save_hypothesis_result(
        study_name=args.study_name,
        hypothesis_id=args.hypothesis_id,
        search_ranges=search_ranges_meta,
        fixed_tunables=fixed_values,
        base_config_ref=base_config_ref,
        window=args.window,
        output_dir=args.output_dir,
    )

    best = study.best_trial
    out_root = args.output_dir or "models/cache/dashboard_results"
    print(f"\n[PARALLEL] Done. study={args.study_name}")
    print(f"[PARALLEL] best_trial=#{best.number}, "
          f"best_score={best.value:.6f}")
    print(f"[PARALLEL] dashboard_results: "
          f"{out_root}/best_{args.study_name}_{best.number}/")
    return 1 if failures else 0


def main() -> int:
    if "--worker-mode" in sys.argv:
        return _worker_main()
    return _launcher_main()


if __name__ == "__main__":
    sys.exit(main())
