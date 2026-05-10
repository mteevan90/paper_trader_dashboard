"""Concentration analysis ablation orchestrator (Phase 2 Section 8).

Re-runs the primary Optuna study with ablation filters across three
dimensions (per-underlying, per-DTE-band, per-IV-regime) so the
promotion gate can ask "is this study's edge concentrated in one place,
or broadly distributed?" — the NVDA/META template adapted from equities
per memo §7.

Each ablation runs a smaller Optuna study (``n_trials_per_ablation``,
default 25) than the primary so the total compute is bounded. The
result is a flat tuple of :class:`ConcentrationResult` records — one
per ablation — comparing each ablated Calmar to the unrestricted base.

DTE-band and IV-regime ablations rely on the Section 6 amendment that
landed with this PR (``EntryFilters``); the per-underlying ablation
uses :meth:`BacktestConfig.evolve` to drop one ticker at a time without
needing engine changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import optuna

from src.options.backtest_config import BacktestConfig
from src.options.engine import EngineDeps, EntryFilters, run_backtest
from src.options.optuna_runner import (
    DEFAULT_OPTUNA_STORAGE_PATH,
    FAILED_TRIAL_SENTINEL,
    calmar_objective,
)


__all__ = [
    "ConcentrationResult",
    "run_concentration_analysis",
    "DTE_BANDS",
    "IV_REGIMES",
]


logger = logging.getLogger(__name__)


DTE_BANDS: tuple[tuple[int, int], ...] = (
    (25, 30),
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
)
IV_REGIMES: tuple[str, ...] = ("high", "low")


@dataclass(frozen=True, slots=True)
class ConcentrationResult:
    """Per-ablation Calmar comparison vs the unrestricted study.

    ``pct_alpha_attribution`` answers "what fraction of the base
    study's Calmar disappears when this dimension is ablated?". A
    value of 0.6 means 60% of the edge came from this slice; >0.5 is
    the promotion-gate threshold for "single dimension dominates".
    """

    ablation_dimension: str
    ablation_value: str
    base_calmar: float
    ablated_calmar: float
    delta_calmar: float
    pct_alpha_attribution: float


def _pct_alpha_attribution(
    base_calmar: float, ablated_calmar: float
) -> float:
    """``(base - ablated) / base``. Defends against base ≤ 0 by
    returning 0.0 (no meaningful attribution if the base study made no
    money)."""
    if base_calmar <= 0:
        return 0.0
    return (base_calmar - ablated_calmar) / base_calmar


def _run_ablation_study(
    *,
    base_study_label: str,
    strategy_class: str,
    universe: tuple[str, ...],
    start_date: date,
    end_date: date,
    train_val_split_date: date,
    starting_capital: float,
    n_trials: int,
    output_dir: Path,
    deps: Optional[EngineDeps],
    seed: int,
    entry_filters: Optional[EntryFilters],
    storage_path: Optional[Path] = None,
) -> float:
    """Run a small Optuna study with the given filter and return the
    best-trial val Calmar.

    Each ablation gets its own Optuna study_name (``<base>_ablation_<...>``)
    so studies don't collide in the SQLite DB.
    """
    storage_path = storage_path or DEFAULT_OPTUNA_STORAGE_PATH
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=base_study_label,
        storage=storage_url,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_results = {"obj": None, "val_calmar": FAILED_TRIAL_SENTINEL}

    def objective(trial: optuna.Trial) -> float:
        try:
            config = BacktestConfig.suggest(
                trial,
                study_label=base_study_label,
                strategy_class=strategy_class,
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                train_val_split_date=train_val_split_date,
                starting_capital=starting_capital,
                random_seed=seed,
            )
            results = run_backtest(
                config, deps=deps, entry_filters=entry_filters,
            )
            score = calmar_objective(results)
            # Track best by objective value for val-Calmar surfacing.
            if best_results["obj"] is None or score > best_results["obj"]:
                best_results["obj"] = score
                best_results["val_calmar"] = _val_calmar(results)
            return score
        except Exception as exc:
            logger.error(
                "Ablation %s trial failed: %r",
                base_study_label, exc,
            )
            return FAILED_TRIAL_SENTINEL

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    if best_results["obj"] is None:
        return FAILED_TRIAL_SENTINEL
    return float(best_results["val_calmar"])


def _val_calmar(results) -> float:
    """Calmar over only the validation snapshots — mirrors
    :func:`calmar_objective` but flips the train-vs-val filter."""
    val_snaps = [
        s for s in results.daily_snapshots
        if s.train_val_label == "val"
    ]
    if len(val_snaps) < 30:
        return 0.0
    initial = val_snaps[0].portfolio_total
    if initial <= 0:
        return 0.0
    final = val_snaps[-1].portfolio_total
    days = (val_snaps[-1].sim_date - val_snaps[0].sim_date).days
    years = days / 365.25
    if years <= 0:
        return 0.0
    if final <= 0:
        compound_return = -1.0
    else:
        compound_return = (final / initial) ** (1.0 / years) - 1.0
    peak = initial
    max_dd = 0.0
    for s in val_snaps:
        if s.portfolio_total > peak:
            peak = s.portfolio_total
        if peak > 0:
            dd = (peak - s.portfolio_total) / peak
            if dd > max_dd:
                max_dd = dd
    if max_dd == 0.0:
        if compound_return > 0:
            return 1e9
        return compound_return
    return compound_return / max_dd


def run_concentration_analysis(
    *,
    base_study_label: str,
    strategy_class: str,
    base_calmar: float,
    full_universe: tuple[str, ...],
    n_trials_per_ablation: int = 25,
    start_date: date,
    end_date: date,
    train_val_split_date: date,
    starting_capital: float,
    output_dir: Path,
    deps: Optional[EngineDeps] = None,
    seed: int = 42,
    storage_path: Optional[Path] = None,
) -> tuple[ConcentrationResult, ...]:
    """Run all ablation dimensions and return a flat tuple of results.

    Three dimensions:

    1. **Per-underlying**: for each ticker in ``full_universe``, re-run
       with that ticker dropped from the universe.
    2. **Per-DTE-band**: 5 bands (25-30, 30-35, 35-40, 40-45, 45-50) —
       each ablation excludes that band via :class:`EntryFilters`.
    3. **Per-IV-regime**: two ablations (``high`` and ``low``) using
       :class:`EntryFilters` and the engine's ``fetch_iv_regime``
       deps callable.

    Per-strategy-variant ablation is degenerate when called per
    strategy_class — the v1_study orchestrator handles that dimension
    separately by running CSP and CC studies independently.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ConcentrationResult] = []

    # 1. Per-underlying.
    for ticker in full_universe:
        ablated_universe = tuple(t for t in full_universe if t != ticker)
        if not ablated_universe:
            continue
        label = f"{base_study_label}_ablation_underlying_{ticker}"
        sub_dir = output_dir / f"ablation_underlying_{ticker}"
        ablated = _run_ablation_study(
            base_study_label=label,
            strategy_class=strategy_class,
            universe=ablated_universe,
            start_date=start_date,
            end_date=end_date,
            train_val_split_date=train_val_split_date,
            starting_capital=starting_capital,
            n_trials=n_trials_per_ablation,
            output_dir=sub_dir,
            deps=deps,
            seed=seed,
            entry_filters=None,
            storage_path=storage_path,
        )
        results.append(
            ConcentrationResult(
                ablation_dimension="underlying",
                ablation_value=ticker,
                base_calmar=base_calmar,
                ablated_calmar=ablated,
                delta_calmar=ablated - base_calmar,
                pct_alpha_attribution=_pct_alpha_attribution(
                    base_calmar, ablated,
                ),
            )
        )

    # 2. Per-DTE-band.
    for low, high in DTE_BANDS:
        band_value = f"{low}-{high}dte"
        label = f"{base_study_label}_ablation_dte_{band_value}"
        sub_dir = output_dir / f"ablation_dte_{band_value}"
        ablated = _run_ablation_study(
            base_study_label=label,
            strategy_class=strategy_class,
            universe=full_universe,
            start_date=start_date,
            end_date=end_date,
            train_val_split_date=train_val_split_date,
            starting_capital=starting_capital,
            n_trials=n_trials_per_ablation,
            output_dir=sub_dir,
            deps=deps,
            seed=seed,
            entry_filters=EntryFilters(dte_exclude_range=(low, high)),
            storage_path=storage_path,
        )
        results.append(
            ConcentrationResult(
                ablation_dimension="dte_band",
                ablation_value=band_value,
                base_calmar=base_calmar,
                ablated_calmar=ablated,
                delta_calmar=ablated - base_calmar,
                pct_alpha_attribution=_pct_alpha_attribution(
                    base_calmar, ablated,
                ),
            )
        )

    # 3. Per-IV-regime.
    for regime in IV_REGIMES:
        label = f"{base_study_label}_ablation_iv_{regime}"
        sub_dir = output_dir / f"ablation_iv_{regime}"
        ablated = _run_ablation_study(
            base_study_label=label,
            strategy_class=strategy_class,
            universe=full_universe,
            start_date=start_date,
            end_date=end_date,
            train_val_split_date=train_val_split_date,
            starting_capital=starting_capital,
            n_trials=n_trials_per_ablation,
            output_dir=sub_dir,
            deps=deps,
            seed=seed,
            entry_filters=EntryFilters(iv_regime_exclude=regime),
            storage_path=storage_path,
        )
        results.append(
            ConcentrationResult(
                ablation_dimension="iv_regime",
                ablation_value=regime,
                base_calmar=base_calmar,
                ablated_calmar=ablated,
                delta_calmar=ablated - base_calmar,
                pct_alpha_attribution=_pct_alpha_attribution(
                    base_calmar, ablated,
                ),
            )
        )

    return tuple(results)
