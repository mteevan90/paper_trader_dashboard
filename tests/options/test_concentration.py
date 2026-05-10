"""Tests for ``src/options/concentration.py`` (Phase 2 Section 8).

The Optuna runner inside the orchestrator hits real Optuna (small
trials, in-memory tmp DB), but :func:`run_backtest` is monkeypatched
to a fast deterministic stub so each ablation completes quickly. The
focus is on whether the orchestrator dispatches the right ablations
and assembles results correctly — not on the engine semantics.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.options import concentration as concentration_mod
from src.options.concentration import (
    DTE_BANDS,
    IV_REGIMES,
    ConcentrationResult,
    run_concentration_analysis,
)
from src.options.engine import (
    DailySnapshot,
    EntryFilters,
    StudyResults,
)


def _stub_results(config) -> StudyResults:
    """Fast deterministic StudyResults for any config; flat portfolio."""
    snaps: list[DailySnapshot] = []
    d = config.start_date
    train_count = 0
    val_count = 0
    while d <= config.end_date:
        label = "train" if d <= config.train_val_split_date else "val"
        snaps.append(DailySnapshot(
            sim_date=d,
            train_val_label=label,
            cash=config.starting_capital,
            stock_value=0.0,
            open_positions_count=0,
            open_positions_mark=0.0,
            realized_pnl_to_date=0.0,
            portfolio_total=config.starting_capital,
            portfolio_delta=0.0, portfolio_gamma=0.0,
            portfolio_theta_per_day=0.0, portfolio_vega_per_pct=0.0,
        ))
        if label == "train":
            train_count += 1
        else:
            val_count += 1
        d += timedelta(days=1)
    return StudyResults(
        config=config,
        daily_snapshots=tuple(snaps),
        closed_positions=(),
        spawned_equity_closes=(),
        skip_counters={},
        wall_time_seconds=0.0,
        run_id="stub",
    )


def _ablation_kwargs(tmp_path):
    return dict(
        base_study_label="concentration_test",
        strategy_class="cash_secured_put",
        base_calmar=1.0,
        full_universe=("AAPL", "MSFT", "NVDA"),
        n_trials_per_ablation=1,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 6, 1),
        train_val_split_date=date(2024, 4, 1),
        starting_capital=250_000.0,
        output_dir=tmp_path / "ablations",
        seed=42,
        storage_path=tmp_path / "concentration.db",
    )


def test_returns_results_per_dimension(tmp_path):
    captured_filters: list[EntryFilters | None] = []
    captured_universes: list[tuple[str, ...]] = []

    def stub_run_backtest(config, *, deps=None, entry_filters=None):
        captured_filters.append(entry_filters)
        captured_universes.append(config.universe)
        return _stub_results(config)

    with patch.object(
        concentration_mod, "run_backtest", side_effect=stub_run_backtest,
    ):
        results = run_concentration_analysis(**_ablation_kwargs(tmp_path))

    # 3 underlyings + 5 DTE bands + 2 IV regimes = 10 ablations.
    assert len(results) == 10
    dimensions = {r.ablation_dimension for r in results}
    assert dimensions == {"underlying", "dte_band", "iv_regime"}


def test_per_underlying_blacklists_each_in_turn(tmp_path):
    captured_universes: list[tuple[str, ...]] = []

    def stub_run_backtest(config, *, deps=None, entry_filters=None):
        captured_universes.append(config.universe)
        return _stub_results(config)

    with patch.object(
        concentration_mod, "run_backtest", side_effect=stub_run_backtest,
    ):
        run_concentration_analysis(**_ablation_kwargs(tmp_path))

    full = ("AAPL", "MSFT", "NVDA")
    # First 3 calls (underlying ablations) drop one ticker each.
    underlying_runs = captured_universes[:3]
    assert len(underlying_runs) == 3
    dropped = [
        set(full) - set(univ) for univ in underlying_runs
    ]
    # Each ablation should drop exactly one ticker, and across the
    # three runs the union of dropped tickers is the full universe.
    assert all(len(d) == 1 for d in dropped)
    assert set.union(*dropped) == set(full)


def test_dte_band_filter_applied(tmp_path):
    captured_filters: list[EntryFilters | None] = []

    def stub_run_backtest(config, *, deps=None, entry_filters=None):
        captured_filters.append(entry_filters)
        return _stub_results(config)

    with patch.object(
        concentration_mod, "run_backtest", side_effect=stub_run_backtest,
    ):
        run_concentration_analysis(**_ablation_kwargs(tmp_path))

    # The 4th-8th captured filters (after 3 underlying ablations) should
    # have non-None dte_exclude_range; bands should match DTE_BANDS in order.
    dte_filters = [
        f for f in captured_filters
        if f is not None and f.dte_exclude_range is not None
    ]
    assert len(dte_filters) == len(DTE_BANDS)
    actual_bands = [f.dte_exclude_range for f in dte_filters]
    assert actual_bands == list(DTE_BANDS)


def test_iv_regime_classification(tmp_path):
    captured_filters: list[EntryFilters | None] = []

    def stub_run_backtest(config, *, deps=None, entry_filters=None):
        captured_filters.append(entry_filters)
        return _stub_results(config)

    with patch.object(
        concentration_mod, "run_backtest", side_effect=stub_run_backtest,
    ):
        run_concentration_analysis(**_ablation_kwargs(tmp_path))

    iv_filters = [
        f for f in captured_filters
        if f is not None and f.iv_regime_exclude is not None
    ]
    assert len(iv_filters) == len(IV_REGIMES)
    actual_regimes = [f.iv_regime_exclude for f in iv_filters]
    assert sorted(actual_regimes) == sorted(IV_REGIMES)


def test_pct_alpha_attribution_bounded(tmp_path):
    """Spec: pct_alpha_attribution = (base - ablated) / base.
    Result is bounded above at 1.0 (when ablated_calmar = 0)."""

    def stub_run_backtest(config, *, deps=None, entry_filters=None):
        return _stub_results(config)

    with patch.object(
        concentration_mod, "run_backtest", side_effect=stub_run_backtest,
    ):
        results = run_concentration_analysis(**_ablation_kwargs(tmp_path))

    for r in results:
        assert r.pct_alpha_attribution <= 1.0


def test_persists_per_ablation_outputs(tmp_path):
    def stub_run_backtest(config, *, deps=None, entry_filters=None):
        return _stub_results(config)

    with patch.object(
        concentration_mod, "run_backtest", side_effect=stub_run_backtest,
    ):
        run_concentration_analysis(**_ablation_kwargs(tmp_path))

    output_dir = tmp_path / "ablations"
    # Per-underlying directories.
    assert (output_dir / "ablation_underlying_AAPL").exists()
    assert (output_dir / "ablation_underlying_MSFT").exists()
    assert (output_dir / "ablation_underlying_NVDA").exists()
    # DTE-band directories.
    for low, high in DTE_BANDS:
        assert (output_dir / f"ablation_dte_{low}-{high}dte").exists()
    # IV-regime directories.
    for regime in IV_REGIMES:
        assert (output_dir / f"ablation_iv_{regime}").exists()


def test_pct_alpha_attribution_zero_when_base_negative(tmp_path):
    """Defensive: base_calmar <= 0 → attribution defaults to 0."""

    def stub_run_backtest(config, *, deps=None, entry_filters=None):
        return _stub_results(config)

    kwargs = _ablation_kwargs(tmp_path)
    kwargs["base_calmar"] = -0.5

    with patch.object(
        concentration_mod, "run_backtest", side_effect=stub_run_backtest,
    ):
        results = run_concentration_analysis(**kwargs)

    for r in results:
        assert r.pct_alpha_attribution == 0.0
