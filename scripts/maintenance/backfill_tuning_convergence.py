"""Back-fill tuning_convergence.parquet + tuning_summary.json for a study.

Purpose
-------
The dashboard contract v1 adds two OPTIONAL tuning artifacts (see
`docs/architecture/dashboard_contract_v1.md` § tuning_convergence.parquet and
§ tuning_summary.json). Future studies will produce these natively as part
of their Phase 3 (hyperparameter tuning) step. This script is a one-time
retroactive build for studies that were tuned before the contract addition
landed — currently just Larger Universe v1.

It reads the study's existing `contract_v1/trial_log.parquet` and derives:

  - `tuning_convergence.parquet` — one row per trial per model, with the
    per-trial score, running-best score, best-so-far trial number, and
    cumulative wall-clock ms since trial 0.
  - `tuning_summary.json` — a JSON object keyed by model name containing
    headline tuning numbers (total trials, winning trial, winning score,
    mean / std / z-score, convergence-plateau metrics).

The plateau metric `trials_to_95pct_winning` assumes a positive
winning_score (the convention is "first trial where running_best reaches
95% of the winning score"). For studies with negative objectives the
interpretation flips — surface that case as a manual review item before
relying on the metric.

Usage
-----
    python scripts/maintenance/backfill_tuning_convergence.py \\
        --study larger_universe_v1

Idempotent: re-running overwrites both output files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _contract_dir(study_name: str) -> Path:
    return REPO_ROOT / "models" / "studies" / study_name / "contract_v1"


def build_convergence(trial_log: pd.DataFrame) -> pd.DataFrame:
    """Derive the convergence dataframe from a trial_log.parquet frame.

    Input columns (per the v1 contract): tuning_study, trial_number, state,
    value, duration_s, plus param_* columns we ignore here.
    """
    needed = {"tuning_study", "trial_number", "state", "value", "duration_s"}
    missing = needed - set(trial_log.columns)
    if missing:
        raise ValueError(f"trial_log.parquet missing required columns: {missing}")

    rows: list[pd.DataFrame] = []
    for model_name, g in trial_log.groupby("tuning_study"):
        g = g.sort_values("trial_number").reset_index(drop=True)
        # COMPLETE-only feed for running-best (non-COMPLETE scores are not
        # trustworthy — per contract spec, NaN value for FAIL / PRUNED).
        is_complete = g["state"] == "COMPLETE"
        scores = g["value"].where(is_complete, np.nan)

        # Running best: cumulative max over the non-NaN scores, holding the
        # last best forward through NaN gaps.
        running_best = scores.cummax().ffill()

        # best_so_far_trial: the trial_number at which the current best was
        # last beaten. We track this by walking forward.
        best_trial_nums: list[int] = []
        current_best = float("-inf")
        current_best_trial = -1
        for tn, s in zip(g["trial_number"].astype(int), scores):
            if pd.notna(s) and s > current_best:
                current_best = s
                current_best_trial = int(tn)
            best_trial_nums.append(current_best_trial)

        # Cumulative wall-clock: duration_s.cumsum() converted to ms.
        # Treat NaN durations as 0 (don't penalize the running clock for
        # trials that didn't record duration).
        cum_ms = (g["duration_s"].fillna(0).cumsum() * 1000).round().astype("int64")

        rows.append(pd.DataFrame({
            "model": model_name,
            "trial_number": g["trial_number"].astype("int32"),
            "score": g["value"].astype(float),
            "running_best_score": running_best.astype(float),
            "best_so_far_trial": pd.array(best_trial_nums, dtype="int32"),
            "ms_since_start": cum_ms,
        }))

    return pd.concat(rows, ignore_index=True)


def build_summary(convergence: pd.DataFrame, trial_log: pd.DataFrame) -> dict:
    """Build the per-model summary dict."""
    summary: dict[str, dict] = {}
    for model_name in convergence["model"].unique():
        c = convergence[convergence["model"] == model_name]
        log = trial_log[trial_log["tuning_study"] == model_name]
        complete_scores = log.loc[log["state"] == "COMPLETE", "value"].dropna()

        total = int(len(complete_scores))
        if total == 0:
            summary[model_name] = {
                "total_trials": 0,
                "winning_trial": None,
                "winning_score": None,
                "mean_score": None,
                "std_score": None,
                "winner_zscore": None,
                "trials_to_95pct_winning": None,
                "pct_trials_to_plateau": None,
                "_note": "No COMPLETE trials; convergence stats undefined.",
            }
            continue

        winning_score = float(complete_scores.max())
        winning_idx = int(log.loc[log["value"].idxmax(), "trial_number"])
        mean_score = float(complete_scores.mean())
        std_score = float(complete_scores.std(ddof=1)) if total > 1 else 0.0
        if std_score > 0:
            winner_z = (winning_score - mean_score) / std_score
        else:
            winner_z = float("nan")

        # Plateau: first trial_number where running_best >= 0.95 * winning_score.
        threshold = 0.95 * winning_score
        c_sorted = c.sort_values("trial_number")
        plateau_rows = c_sorted[c_sorted["running_best_score"] >= threshold]
        if plateau_rows.empty:
            # Defensive fallback: should never happen since the winning
            # trial itself meets the threshold.
            plateau_trial = winning_idx
        else:
            plateau_trial = int(plateau_rows["trial_number"].iloc[0])

        # pct_trials_to_plateau: count of COMPLETE trials with trial_number
        # <= plateau_trial, divided by total COMPLETE. Robust to studies
        # where early trial numbers are FAIL — the denominator is what
        # the narrative quotes ("X% of useful trials needed to plateau").
        complete_log = log[log["state"] == "COMPLETE"]
        complete_at_or_before_plateau = int(
            (complete_log["trial_number"] <= plateau_trial).sum()
        )
        pct_to_plateau = complete_at_or_before_plateau / total

        summary[model_name] = {
            "total_trials": total,
            "winning_trial": winning_idx,
            "winning_score": winning_score,
            "mean_score": mean_score,
            "std_score": std_score,
            "winner_zscore": (None if np.isnan(winner_z) else float(winner_z)),
            "trials_to_95pct_winning": plateau_trial,
            "pct_trials_to_plateau": float(pct_to_plateau),
        }

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True,
                   help="Study name under models/studies/<name>/")
    args = p.parse_args()

    contract = _contract_dir(args.study)
    trial_log_path = contract / "trial_log.parquet"
    if not trial_log_path.exists():
        raise SystemExit(f"trial_log.parquet not found at {trial_log_path}")

    trial_log = pd.read_parquet(trial_log_path)
    convergence = build_convergence(trial_log)
    summary = build_summary(convergence, trial_log)

    out_parquet = contract / "tuning_convergence.parquet"
    out_json = contract / "tuning_summary.json"

    convergence.to_parquet(out_parquet, index=False)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {out_parquet}  ({len(convergence)} rows, "
          f"{convergence['model'].nunique()} models)")
    print(f"wrote {out_json}")
    print()
    print("Per-model summary:")
    for model_name, s in summary.items():
        print(f"  {model_name}:")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"    {k:<28} {v:.6f}")
            else:
                print(f"    {k:<28} {v}")


if __name__ == "__main__":
    main()
