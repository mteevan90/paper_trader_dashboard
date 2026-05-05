"""run_hypothesis.py - parameterized hypothesis-test launcher (Archetype 3).

Wraps optuna_runner.run_study so a hypothesis (V3, V4, V5+) can be
expressed as: a baseline config + range overrides + tunables to hold
fixed. The launcher records the hypothesis metadata in
dashboard_results/.../meta.json so any future analysis can reconstruct
exactly what was tested.

Existing optuna_runner CLI flags (--smoke, --full, --resume, --report,
--save-results) are untouched; they continue to work bit-identically
because run_study's new range_overrides / fixed_values kwargs default
to None.

Example V3 invocation:

    python src/run_hypothesis.py \\
        --study-name optuna_v3_20260505_HHMMSS \\
        --hypothesis-id v3-rebalance-frequency \\
        --base-config models/cache/dashboard_results/best_optuna_v1_20260504_103429_706/meta.json \\
        --override-range rebalance_frequency_days=5,60 \\
        --hold-tunables-fixed weight_fundamental,weight_technical,weight_model,macro_threshold_low,macro_threshold_high,atr_multiplier,analyst_weight,position_count \\
        --window train \\
        --n-trials 1000

Translation rules for --hold-tunables-fixed (uses BacktestConfig field
names; the launcher translates to optuna_runner's search-space form):

  1. macro_threshold_high: BacktestConfig stores high directly, but the
     search-space samples (macro_threshold_low, macro_threshold_gap)
     where high = low + gap. Holding macro_threshold_high pins BOTH
     macro_threshold_low and macro_threshold_gap (gap derived from
     base.high - base.low).

  2. weight_alt: derived in BacktestConfig.__post_init__ as
     1.0 - (wf + wt + wm). Holding the three free weights implicitly
     holds weight_alt; passing weight_alt directly is rejected since
     it can't be set independently.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is importable when this script is run from the repo root
# or from inside src/. Add the script's own directory to sys.path so
# `from optuna_runner import ...` works regardless of cwd.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from optuna_runner import (    # noqa: E402  (after sys.path manipulation)
    _DEFAULT_RANGES,
    run_study,
    save_hypothesis_result,
)


# Search-space param names — what build_search_space samples on. Holds
# and overrides MUST be expressed in these terms internally.
_SEARCH_SPACE_PARAMS: frozenset[str] = frozenset({
    "weight_fundamental",
    "weight_technical",
    "weight_model",
    "macro_threshold_low",
    "macro_threshold_gap",
    "atr_multiplier",
    "analyst_weight",
    "rebalance_frequency_days",
    "position_count",
})

# BacktestConfig field names that map 1:1 to a search-space param name.
_DIRECT_FIELDS: frozenset[str] = frozenset({
    "weight_fundamental",
    "weight_technical",
    "weight_model",
    "macro_threshold_low",
    "atr_multiplier",
    "analyst_weight",
    "rebalance_frequency_days",
    "position_count",
})

# BacktestConfig field names that need translation. high -> (low, gap).
_TRANSLATED_FIELDS: frozenset[str] = frozenset({
    "macro_threshold_high",
})

# BacktestConfig field names that cannot be held directly because they
# are themselves derived from other fields.
_DERIVED_FIELDS: frozenset[str] = frozenset({
    "weight_alt",
})


def _parse_override_range(raw: str) -> tuple[str, tuple[float, float]]:
    """'rebalance_frequency_days=5,60' -> ('rebalance_frequency_days', (5.0, 60.0))"""
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
        lo = float(parts[0])
        hi = float(parts[1])
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
    """Convert --hold-tunables-fixed list (BacktestConfig field names)
    to a fixed_values dict in search-space terms.

    See module docstring for translation rules. Errors out with clear
    messages on derived fields (weight_alt) and unknown keys.
    """
    holds = [h.strip() for h in holds_csv.split(",") if h.strip()]
    fixed: dict = {}
    for field in holds:
        if field in _DERIVED_FIELDS:
            raise SystemExit(
                f"--hold-tunables-fixed: {field!r} is derived, not "
                f"settable independently. To hold weight_alt, hold the "
                f"three free weights (weight_fundamental, weight_technical, "
                f"weight_model) — weight_alt is then implicit via "
                f"BacktestConfig.__post_init__.")
        if field in _DIRECT_FIELDS:
            if field not in base_config:
                raise SystemExit(
                    f"--hold-tunables-fixed: {field!r} not present in "
                    f"the base config — cannot pin to base value.")
            fixed[field] = base_config[field]
            continue
        if field == "macro_threshold_high":
            # Translation: high = low + gap; we pin BOTH search-space
            # params so the search exactly reproduces base low and high.
            for needed in ("macro_threshold_low", "macro_threshold_high"):
                if needed not in base_config:
                    raise SystemExit(
                        f"--hold-tunables-fixed macro_threshold_high "
                        f"requires both macro_threshold_low and "
                        f"macro_threshold_high in the base config; "
                        f"missing {needed}.")
            fixed["macro_threshold_low"] = base_config["macro_threshold_low"]
            fixed["macro_threshold_gap"] = (
                base_config["macro_threshold_high"]
                - base_config["macro_threshold_low"])
            continue
        if field == "macro_threshold_gap":
            raise SystemExit(
                f"--hold-tunables-fixed: pass macro_threshold_high (the "
                f"BacktestConfig field name); the launcher translates "
                f"to gap internally.")
        raise SystemExit(
            f"--hold-tunables-fixed: unknown field {field!r}. Valid "
            f"BacktestConfig tunable fields: "
            f"{sorted(_DIRECT_FIELDS | _TRANSLATED_FIELDS)}")
    return fixed


def _build_search_ranges_log(fixed_values: dict,
                             range_overrides: dict) -> dict:
    """Build the search_ranges dict that gets recorded to meta.json.

    For every search-space param: report either ['FIXED', value] or
    [min, max]. This is what makes the hypothesis fully reproducible
    from the saved meta.json alone.
    """
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Hypothesis-driven Optuna launcher (Archetype 3). "
                    "Wraps optuna_runner.run_study with parameterized "
                    "search-space overrides and held-fixed tunables.")
    p.add_argument("--study-name", required=True,
                   help="Full study name written to optuna_studies.db, "
                        "e.g. optuna_v3_20260505_201500")
    p.add_argument("--hypothesis-id", required=True,
                   help="Short id written to meta.json, "
                        "e.g. v3-rebalance-frequency")
    p.add_argument("--base-config", required=True,
                   help="Path to a meta.json whose 'config' key holds "
                        "the baseline BacktestConfig values")
    p.add_argument("--override-range", action="append", default=[],
                   metavar="KEY=MIN,MAX",
                   help="Repeatable. Override one tunable's search range. "
                        "Key uses the search-space name (e.g. "
                        "rebalance_frequency_days, macro_threshold_gap)")
    p.add_argument("--hold-tunables-fixed", default="",
                   metavar="CSV",
                   help="Comma-separated BacktestConfig field names to "
                        "hold at the base config's values. macro_threshold_high "
                        "translates to holding low + gap; weight_alt cannot "
                        "be held directly (hold the 3 free weights).")
    p.add_argument("--window", choices=["train", "validate"], required=True,
                   help="Which window the study runs on. Only 'train' is "
                        "currently implemented.")
    p.add_argument("--n-trials", type=int, required=True,
                   help="Number of Optuna trials to run.")
    p.add_argument("--output-dir", default=None,
                   help="dashboard_results destination root. Defaults to "
                        "models/cache/dashboard_results/.")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Optuna parallel workers (default 1). Hypothesis "
                        "runs with --hold-tunables-fixed produce homogeneous "
                        "workloads that contend on shared state inside "
                        "run_backtest, so n_jobs>1 typically delivers no "
                        "real speedup. Override only if you understand the "
                        "tradeoff (e.g., when running multiple varying "
                        "params).")
    args = p.parse_args()

    # ---- Validate window ----
    if args.window == "validate":
        raise SystemExit(
            "Validation pathway not implemented; this launcher is "
            "training-only by design. Run training first, manually "
            "graduate per v3 hypothesis spec.")

    # ---- Load base config ----
    base_path = Path(args.base_config)
    if not base_path.exists():
        raise SystemExit(f"--base-config not found: {base_path}")
    try:
        with open(base_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--base-config is not valid JSON: {e}")
    base_config = meta.get("config")
    if not isinstance(base_config, dict):
        raise SystemExit(
            f"--base-config meta.json missing top-level 'config' key "
            f"(or non-dict). Path: {base_path}")

    # ---- Parse overrides ----
    range_overrides: dict = {}
    for raw in args.override_range:
        key, rng = _parse_override_range(raw)
        if key in range_overrides:
            raise SystemExit(
                f"--override-range specified twice for {key!r}")
        range_overrides[key] = rng

    # ---- Translate holds ----
    fixed_values = _translate_holds(args.hold_tunables_fixed, base_config)

    # ---- Conflict check ----
    overlap = set(fixed_values) & set(range_overrides)
    if overlap:
        raise SystemExit(
            f"Tunables cannot be both held fixed and given an override "
            f"range: {sorted(overlap)}")

    # ---- Pre-launch summary ----
    print(f"[HYPOTHESIS] study={args.study_name}")
    print(f"[HYPOTHESIS] hypothesis_id={args.hypothesis_id}")
    print(f"[HYPOTHESIS] base_config={base_path}")
    print(f"[HYPOTHESIS] window={args.window}, n_trials={args.n_trials}, "
          f"n_jobs={args.n_jobs}")
    if range_overrides:
        print("[HYPOTHESIS] range_overrides:")
        for k, (lo, hi) in sorted(range_overrides.items()):
            print(f"    {k}: [{lo}, {hi}]")
    if fixed_values:
        print("[HYPOTHESIS] fixed_values (search-space form):")
        for k, v in sorted(fixed_values.items()):
            print(f"    {k}: {v}")

    # ---- Run study ----
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[HYPOTHESIS] launching at {started_at}...")
    study = run_study(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        smoke_test=False,
        range_overrides=range_overrides,
        fixed_values=fixed_values,
    )

    # ---- Refuse to save if no completed trials ----
    n_complete = sum(1 for t in study.trials
                     if t.state.name == "COMPLETE")
    if n_complete == 0:
        raise SystemExit(
            "No completed trials — refusing to save dashboard_results. "
            "Inspect optuna_trials.jsonl to debug.")

    # ---- Save dashboard_results with hypothesis metadata ----
    search_ranges_meta = _build_search_ranges_log(fixed_values, range_overrides)
    base_config_ref = str(base_path).replace("\\", "/")
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
    print(f"\n[HYPOTHESIS] Done. study={args.study_name}")
    print(f"[HYPOTHESIS] n_complete={n_complete}, best_trial=#{best.number}, "
          f"best_score={best.value:.6f}")
    print(f"[HYPOTHESIS] dashboard_results written under "
          f"{out_root}/best_{args.study_name}_{best.number}/")


if __name__ == "__main__":
    main()
