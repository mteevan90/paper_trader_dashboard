"""Top-level production v1 study orchestrator (Phase 2 Section 8).

:func:`run_v1_study` runs the locked Light scope:

1. Fetch SPY total return for the backtest window (cached benchmark).
2. Fetch BXM for the backtest window (Tradier primary, yfinance fallback).
3. For each strategy class (CSP, CC):
   a. Run primary Optuna study (100 trials).
   b. Load best trial's StudyResults from disk.
   c. Run concentration analysis ablation.
   d. Evaluate the automated promotion gate.
   e. Write ``promotion_decision.json`` with the automated recommendation.
   f. If interactive: prompt for human override and re-write the file.
4. Snapshot the full run to ``models/snapshots/options/pre_options_v1_<date>/``
   per memo §3 row 15 — the reproducibility lock at promotion.
"""

from __future__ import annotations

import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

from src.options.backtest_config import DEFAULT_UNIVERSE
from src.options.benchmarks import fetch_bxm, fetch_spy_total_return
from src.options.concentration import run_concentration_analysis
from src.options.engine import EngineDeps, StudyResults
from src.options.optuna_runner import run_optuna_study
from src.options.promotion import (
    evaluate_promotion,
    write_promotion_decision,
)


__all__ = [
    "run_v1_study",
    "STRATEGY_CLASSES",
    "SNAPSHOTS_BASE_DIR",
]


logger = logging.getLogger(__name__)


STRATEGY_CLASSES: tuple[str, ...] = ("cash_secured_put", "covered_call")
SNAPSHOTS_BASE_DIR: Path = Path("models") / "snapshots" / "options"


def _prompt_human_override(
    strategy_class: str,
    automated_recommendation: str,
    summary: str,
) -> Optional[dict]:
    """Interactive prompt for the human override decision. Returns None
    if the user accepts the automated recommendation."""
    print(f"\n[{strategy_class}] Automated recommendation: {automated_recommendation}")
    print(f"  Summary: {summary}")
    raw = input(
        "Override? [y/N or 'promote'/'do_not_promote'/'borderline']: "
    ).strip().lower()
    if raw in ("", "n", "no"):
        return None
    if raw in ("y", "yes"):
        decision = input(
            "  Decision (promote/do_not_promote): "
        ).strip().lower()
    else:
        decision = raw
    if decision not in ("promote", "do_not_promote", "borderline"):
        print(f"  Unrecognized decision {decision!r}; skipping override")
        return None
    reasoning = input("  Reasoning: ").strip()
    return {"decision": decision, "reasoning": reasoning}


def _snapshot_run(
    csp_dir: Path, cc_dir: Path, run_id: str,
) -> Path:
    """Copy per-strategy outputs into a snapshot directory.

    Snapshot path mirrors the equity convention: a dated, immutable
    folder under ``models/snapshots/options/``. Reproducibility lock
    per memo §3 row 15.
    """
    today = date.today().isoformat()
    snapshot_dir = (
        SNAPSHOTS_BASE_DIR / f"pre_options_v1_{today}_{run_id}"
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if csp_dir.exists():
        target = snapshot_dir / "cash_secured_put"
        shutil.copytree(csp_dir, target, dirs_exist_ok=True)
    if cc_dir.exists():
        target = snapshot_dir / "covered_call"
        shutil.copytree(cc_dir, target, dirs_exist_ok=True)
    return snapshot_dir


def _load_best_trial_results(
    output_dir: Path, best_trial_number: int,
) -> Optional[StudyResults]:
    """Load the best-trial StudyResults from disk."""
    if best_trial_number < 0:
        return None
    trial_dir = output_dir / f"trial_{best_trial_number:04d}"
    if not trial_dir.exists():
        return None
    return StudyResults.from_parquet(trial_dir)


def run_v1_study(
    *,
    run_id: str,
    start_date: date,
    end_date: date,
    train_val_split_date: date,
    starting_capital: float = 100_000.0,
    n_trials_primary: int = 100,
    n_trials_per_ablation: int = 25,
    output_base_dir: Path = Path("models") / "cache" / "options" / "v1_study",
    deps: Optional[EngineDeps] = None,
    seed: int = 42,
    interactive: bool = True,
    universe: tuple[str, ...] = DEFAULT_UNIVERSE,
    storage_path: Optional[Path] = None,
) -> dict[str, Path]:
    """Top-level production v1 study orchestrator.

    Returns a dict with paths to per-strategy output dirs and the
    snapshot dir: ``{"csp_dir": ..., "cc_dir": ..., "snapshot_dir": ...}``.

    ``storage_path`` defaults to ``<output_base_dir>/<run_id>/optuna_studies.db``
    so each run_id gets isolated Optuna state — no collision when
    re-running the same script with a fresh ``run_id``.
    """
    output_base_dir = Path(output_base_dir) / run_id
    output_base_dir.mkdir(parents=True, exist_ok=True)
    if storage_path is None:
        storage_path = output_base_dir / "optuna_studies.db"

    # 1. SPY total return.
    spy = fetch_spy_total_return(start_date, end_date)

    # 2. BXM (Tradier primary, yfinance fallback).
    bxm_df = fetch_bxm(start_date, end_date)
    if bxm_df.empty:
        logger.warning(
            "BXM unavailable for [%s, %s]; CC promotion check will fail",
            start_date, end_date,
        )
        bxm_df = None  # treat as unavailable downstream

    paths: dict[str, Path] = {}
    csp_dir: Optional[Path] = None
    cc_dir: Optional[Path] = None

    for strategy_class in STRATEGY_CLASSES:
        study_label = f"v1_{run_id}_{strategy_class}"
        strategy_dir = output_base_dir / strategy_class
        strategy_dir.mkdir(parents=True, exist_ok=True)
        primary_dir = strategy_dir / "primary"
        ablation_dir = strategy_dir / "ablations"

        # 3a. Primary Optuna study.
        primary = run_optuna_study(
            study_label=study_label,
            strategy_class=strategy_class,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            train_val_split_date=train_val_split_date,
            n_trials=n_trials_primary,
            starting_capital=starting_capital,
            promotable=False,
            output_dir=primary_dir,
            top_k=5,
            deps=deps,
            seed=seed,
            storage_path=storage_path,
        )
        primary.to_json(strategy_dir / "primary_summary.json")

        # 3b. Load best-trial StudyResults.
        best_results = _load_best_trial_results(
            primary_dir, primary.best_trial_number,
        )
        if best_results is None:
            logger.error(
                "v1_study: no best-trial StudyResults for %s — "
                "skipping concentration + promotion gate",
                strategy_class,
            )
            if strategy_class == "cash_secured_put":
                csp_dir = strategy_dir
            else:
                cc_dir = strategy_dir
            continue

        # 3c. Concentration analysis.
        concentration = run_concentration_analysis(
            base_study_label=study_label,
            strategy_class=strategy_class,
            base_calmar=primary.best_value,
            full_universe=universe,
            n_trials_per_ablation=n_trials_per_ablation,
            start_date=start_date,
            end_date=end_date,
            train_val_split_date=train_val_split_date,
            starting_capital=starting_capital,
            output_dir=ablation_dir,
            deps=deps,
            seed=seed,
            storage_path=storage_path,
        )

        # 3d. Promotion gate.
        recommendation = evaluate_promotion(
            strategy_class=strategy_class,
            primary_study=primary,
            primary_results=best_results,
            spy_total_return=spy,
            bxm=bxm_df,
            concentration_results=concentration,
        )

        # 3e. Persist automated recommendation.
        write_promotion_decision(strategy_dir, recommendation)

        # 3f. Optional human override.
        if interactive:
            override = _prompt_human_override(
                strategy_class,
                recommendation.automated_recommendation,
                recommendation.summary,
            )
            if override is not None:
                write_promotion_decision(
                    strategy_dir, recommendation,
                    human_override=override,
                )

        if strategy_class == "cash_secured_put":
            csp_dir = strategy_dir
        else:
            cc_dir = strategy_dir

    # 4. Snapshot.
    snapshot_dir = _snapshot_run(
        csp_dir or output_base_dir / "cash_secured_put",
        cc_dir or output_base_dir / "covered_call",
        run_id,
    )

    paths["csp_dir"] = csp_dir or output_base_dir / "cash_secured_put"
    paths["cc_dir"] = cc_dir or output_base_dir / "covered_call"
    paths["snapshot_dir"] = snapshot_dir
    return paths
