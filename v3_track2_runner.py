"""V3 Track 2 — Local Perturbation around Trial #325.

Generates 33 perturbed configs by varying one tunable at a time around #325's
values, writes each as a synthetic meta.json under
models/cache/dashboard_results/v3_track2_perturbation/, then runs rescore_baseline
against each and aggregates results into a CSV.

Run with PAPER_TRADER_MACRO_VERSION=v2 (matches Trial #325's training environment).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent
BASE_META = REPO / "models" / "cache" / "dashboard_results" / "best_regime_dependent_v1_20260505_2240_325" / "meta.json"
OUT_ROOT = REPO / "models" / "cache" / "dashboard_results" / "v3_track2_perturbation"
SNAPSHOT = "pre_v2_20260505"

# Perturbation grid. Each axis: (config_key, [test_values])
# #325 value is included in each list as the reference point.
PERTURBATIONS = {
    "position_count_offensive":              [3, 4, 5, 7, 10],
    "rebalance_frequency_days_offensive":    [10, 14, 17, 21, 30],
    "atr_multiplier_offensive":              [1.0, 1.25, 1.4557473205817137, 1.75, 2.25],
    "weight_fundamental_offensive":          [0.40, 0.55, 0.6661930511520805, 0.80, 0.90],
    "weight_technical_offensive":            [0.00, 0.025, 0.05143062279547574, 0.10, 0.20],
    "weight_model_offensive":                [0.05, 0.075, 0.10426036905147223, 0.15, 0.25],
    "regime_threshold":                      [0.30, 0.40, 0.525365732863876, 0.60, 0.70],
    "macro_threshold_low":                   [0.10, 0.18, 0.26138259447974194, 0.35, 0.45],
}

def normalize_weight_triple(wf, wt, wm):
    """Match optuna_runner._normalize_weight_triple exactly."""
    total = wf + wt + wm
    if total >= 0.99:
        scale = 0.99 / total
        return wf * scale, wt * scale, wm * scale
    return wf, wt, wm

def make_perturbed_config(base_cfg: dict, axis: str, value):
    """Apply one perturbation to base config; return new config dict.

    For weight_*_offensive: renormalize via the same rule the optuna runner uses
    and recompute weight_alt_offensive as the residual.
    For macro_threshold_low: preserve gap (high - low constant).
    For regime_threshold: direct substitution.
    Other axes: direct substitution.
    """
    cfg = dict(base_cfg)
    if axis.startswith("weight_") and axis.endswith("_offensive"):
        wf = cfg["weight_fundamental_offensive"]
        wt = cfg["weight_technical_offensive"]
        wm = cfg["weight_model_offensive"]
        if axis == "weight_fundamental_offensive": wf = value
        elif axis == "weight_technical_offensive": wt = value
        elif axis == "weight_model_offensive":     wm = value
        wf, wt, wm = normalize_weight_triple(wf, wt, wm)
        cfg["weight_fundamental_offensive"] = wf
        cfg["weight_technical_offensive"]   = wt
        cfg["weight_model_offensive"]       = wm
        cfg["weight_alt_offensive"]         = max(0.0, 1.0 - (wf + wt + wm))
    elif axis == "macro_threshold_low":
        gap = base_cfg["macro_threshold_high"] - base_cfg["macro_threshold_low"]
        cfg["macro_threshold_low"]  = value
        cfg["macro_threshold_high"] = value + gap
    else:
        cfg[axis] = value
    return cfg

def write_perturbation_meta(base_meta: dict, axis: str, value, out_dir: Path):
    """Write a synthetic meta.json with perturbed config. Preserves all other
    meta keys so rescore_baseline can read it without choking."""
    new_meta = dict(base_meta)
    new_meta["config"] = make_perturbed_config(base_meta["config"], axis, value)
    new_meta["label"]  = f"v3t2_{axis}__{value}"
    new_meta.pop("rolling_metrics", None)  # don't carry stale metrics over
    new_meta.pop("regime_stats", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(new_meta, f, indent=2, default=str)
    return meta_path

def run_one(axis: str, value, base_meta: dict, is_reference: bool):
    """Run rescore_baseline for one perturbation. Returns result dict."""
    label = f"axis={axis}__value={value}"
    if is_reference:
        label += "__REFERENCE"
    out_dir = OUT_ROOT / label
    meta_path = write_perturbation_meta(base_meta, axis, value, out_dir)
    env = dict(os.environ)
    env["PAPER_TRADER_MACRO_VERSION"] = "v2"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "src/rescore_baseline.py",
         "--base-config", str(meta_path),
         "--cache-snapshot", SNAPSHOT,
         "--window", "validate"],
        capture_output=True, text=True, env=env, cwd=str(REPO))
    dur = time.time() - t0
    rescore_path = REPO / "models" / "snapshots" / SNAPSHOT / f"v3t2_{axis}__{value}_rescore.json"
    result = {"axis": axis, "value": value, "is_reference": is_reference,
              "duration_s": round(dur, 1), "returncode": proc.returncode}
    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout)[-500:]
        return result
    if rescore_path.exists():
        rs = json.load(open(rescore_path))
        v = rs.get("validation", {})
        result["validation_score"]            = v.get("score")
        result["rolling_12mo_objective"]      = (v.get("components") or {}).get("rolling_12mo_objective_score")
        result["validation_n_trades"]         = v.get("n_trades")
        result["validation_n_holdings"]       = v.get("n_holdings")
        # copy rescore file into perturbation dir for cleanup
        shutil.copy(rescore_path, out_dir / "rescore.json")
    else:
        result["error"] = f"rescore output not found at {rescore_path}"
    return result

def main():
    if not BASE_META.exists():
        sys.exit(f"Base meta.json not found at {BASE_META}")
    base_meta = json.load(open(BASE_META))
    base_cfg = base_meta["config"]
    print(f"=== V3 Track 2 — Local Perturbation around Trial #325 ===")
    print(f"Base config from: {BASE_META}")
    print(f"Snapshot:         {SNAPSHOT}")
    print(f"Macro version:    v2 (env var)")
    print(f"Output dir:       {OUT_ROOT}")
    print()

    # Build job list. Each axis-value pair is one job. The #325 value on each
    # axis is marked as reference (will dedupe in the summary).
    jobs = []
    for axis, values in PERTURBATIONS.items():
        ref_val = base_cfg.get(axis)
        for v in values:
            is_ref = (ref_val is not None and abs(v - ref_val) < 1e-9)
            jobs.append((axis, v, is_ref))
    print(f"Total jobs: {len(jobs)}  (reference points marked separately)")
    print()

    # Run with 8 workers.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    n_workers = 8
    print(f"Running with {n_workers} workers...")
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_one, axis, v, base_meta, is_ref): (axis, v)
                   for axis, v, is_ref in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            axis, v = futures[fut]
            try:
                r = fut.result()
                status = "OK" if r["returncode"] == 0 and "error" not in r else "FAIL"
                score = r.get("rolling_12mo_objective")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
                print(f"  [{i:2d}/{len(jobs)}] {status} {axis:38s} v={v}  "
                      f"score={score_str}  trades={r.get('validation_n_trades')}  "
                      f"({r['duration_s']}s)")
                results.append(r)
            except Exception as e:
                print(f"  [{i:2d}/{len(jobs)}] CRASH {axis} v={v}: {e}")
                results.append({"axis": axis, "value": v, "error": str(e)})
    print()
    print(f"Total wall-clock: {time.time() - t_start:.1f}s")

    # Write summary CSV.
    import csv
    csv_path = OUT_ROOT / "summary.csv"
    fields = ["axis", "value", "is_reference", "validation_score",
              "rolling_12mo_objective", "validation_n_trades",
              "validation_n_holdings", "duration_s", "returncode", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Summary CSV: {csv_path}")

    # Quick stats.
    refs = [r for r in results if r.get("is_reference")]
    nonrefs = [r for r in results if not r.get("is_reference")]
    ref_scores = [r["rolling_12mo_objective"] for r in refs
                  if isinstance(r.get("rolling_12mo_objective"), (int, float))]
    nonref_scores = [r["rolling_12mo_objective"] for r in nonrefs
                     if isinstance(r.get("rolling_12mo_objective"), (int, float))]
    print()
    print(f"=== QUICK SUMMARY ===")
    print(f"Reference (#325) runs:    {len(refs)}, scores: {ref_scores}")
    print(f"Non-reference runs:       {len(nonrefs)}, success: {len(nonref_scores)}")
    if nonref_scores:
        beat_threshold = 0.30  # rough proxy for "meaningfully outperforms"
        beating = sum(1 for s in nonref_scores if s > beat_threshold)
        print(f"  Score > {beat_threshold}:  {beating}/{len(nonref_scores)} = {100*beating/len(nonref_scores):.0f}%")
        print(f"  Score min:    {min(nonref_scores):.4f}")
        print(f"  Score median: {sorted(nonref_scores)[len(nonref_scores)//2]:.4f}")
        print(f"  Score max:    {max(nonref_scores):.4f}")

if __name__ == "__main__":
    main()
