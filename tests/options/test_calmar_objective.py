"""Pure unit tests for ``calmar_objective`` (Phase 2 Section 7).

No engine, no Optuna — synthetic ``StudyResults`` are constructed
directly so the math is hand-verifiable.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.options.backtest_config import BacktestConfig, FeeModel
from src.options.engine import DailySnapshot, StudyResults
from src.options.optuna_runner import (
    FAILED_TRIAL_SENTINEL,
    ZERO_DD_POSITIVE_RETURN_SENTINEL,
    calmar_objective,
)
from src.options.positions import ExitRules


# ----------------- helpers -----------------


def _stub_config() -> BacktestConfig:
    return BacktestConfig(
        study_label="calmar_test",
        strategy_class="cash_secured_put",
        universe=("AAPL",),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        train_val_split_date=date(2025, 7, 1),
        dte_target=30,
        strike_selector_target_delta=0.30,
        max_concurrent_positions=3,
        earnings_window_avoid=False,
        max_loss_pct_of_portfolio=0.10,
        exit_rules=ExitRules(0.50, 21, 2.0),
        fees=FeeModel(),
    )


def _build_snapshots(
    portfolio_totals: list[float],
    *,
    train_val_labels: list[str] | None = None,
    start_date: date = date(2025, 1, 1),
) -> list[DailySnapshot]:
    """Build a list of DailySnapshot from portfolio_total values.

    One snapshot per consecutive calendar day starting at ``start_date``.
    ``train_val_labels`` defaults to all "train".
    """
    if train_val_labels is None:
        train_val_labels = ["train"] * len(portfolio_totals)
    if len(train_val_labels) != len(portfolio_totals):
        raise ValueError("labels and portfolio totals length mismatch")

    snaps: list[DailySnapshot] = []
    for i, (total, label) in enumerate(
        zip(portfolio_totals, train_val_labels)
    ):
        snaps.append(
            DailySnapshot(
                sim_date=start_date + timedelta(days=i),
                train_val_label=label,
                cash=total,
                stock_value=0.0,
                open_positions_count=0,
                open_positions_mark=0.0,
                realized_pnl_to_date=0.0,
                portfolio_total=total,
                portfolio_delta=0.0,
                portfolio_gamma=0.0,
                portfolio_theta_per_day=0.0,
                portfolio_vega_per_pct=0.0,
            )
        )
    return snaps


def _study_results(snapshots: list[DailySnapshot]) -> StudyResults:
    return StudyResults(
        config=_stub_config(),
        daily_snapshots=tuple(snapshots),
        closed_positions=(),
        spawned_equity_closes=(),
        skip_counters={},
        wall_time_seconds=0.0,
        run_id="test",
    )


# ----------------- tests -----------------


class TestCalmarObjective:
    def test_calmar_positive_return_with_drawdown(self):
        # 100 days: ramp 100k → 110k linearly, with a 5% dip mid-way.
        # Initial = 100_000, final = 110_000, peak = 110_000, max_dd
        # is the biggest peak-to-trough drop.
        # Build: 100k for first 50 days then sudden drop to 95k, then
        # ramp to 110k at the end.
        totals = (
            [100_000.0] * 30
            + [95_000.0] * 20  # dip after running up
            + [102_000.0] * 30
            + [110_000.0] * 20
        )
        snaps = _build_snapshots(totals)
        result = calmar_objective(_study_results(snaps))

        # Compute expected: max_dd = (100_000 - 95_000) / 100_000 = 0.05
        # Years = 99 / 365.25 ≈ 0.2710
        # Compound = (110_000/100_000)^(1/0.2710) - 1 ≈ 0.4231
        # Calmar ≈ 0.4231 / 0.05 ≈ 8.46
        assert result == pytest.approx(8.46, abs=0.5)

    def test_calmar_zero_drawdown_positive_return_returns_sentinel_1e9(self):
        # Strictly monotonic increase → no drawdown.
        totals = [100_000.0 + i * 50.0 for i in range(60)]
        snaps = _build_snapshots(totals)
        result = calmar_objective(_study_results(snaps))
        assert result == ZERO_DD_POSITIVE_RETURN_SENTINEL

    def test_calmar_zero_drawdown_zero_return_returns_zero(self):
        # Flat 100k for 60 days. Compound return = 0, max_dd = 0.
        totals = [100_000.0] * 60
        snaps = _build_snapshots(totals)
        result = calmar_objective(_study_results(snaps))
        assert result == 0.0

    def test_calmar_negative_return_passes_through(self):
        # 100k → 90k linearly. No drawdown recovery, but max_dd > 0.
        totals = [
            100_000.0 - i * (10_000.0 / 59) for i in range(60)
        ]
        snaps = _build_snapshots(totals)
        result = calmar_objective(_study_results(snaps))
        # Compound return should be negative; max_dd ~ 0.10; Calmar negative.
        assert result < 0.0

    def test_calmar_empty_training_data_returns_zero(self):
        # All snapshots are val.
        totals = [100_000.0 + i * 10 for i in range(60)]
        labels = ["val"] * 60
        snaps = _build_snapshots(totals, train_val_labels=labels)
        assert calmar_objective(_study_results(snaps)) == 0.0

    def test_calmar_short_training_window_returns_zero(self):
        # Only 20 training days — under the 30-day floor.
        totals = [100_000.0] * 20
        snaps = _build_snapshots(totals)
        assert calmar_objective(_study_results(snaps)) == 0.0

    def test_calmar_excludes_validation_snapshots(self):
        # 60 train days flat at 100k, 60 val days that crash to 50k.
        # Calmar should ignore the val days and return 0.0 (flat train).
        train_totals = [100_000.0] * 60
        val_totals = [50_000.0] * 60
        snaps = _build_snapshots(
            train_totals + val_totals,
            train_val_labels=["train"] * 60 + ["val"] * 60,
        )
        result = calmar_objective(_study_results(snaps))
        assert result == 0.0  # flat train → 0 return, 0 dd

    def test_calmar_initial_zero_returns_zero_defensive(self):
        totals = [0.0] * 60
        snaps = _build_snapshots(totals)
        assert calmar_objective(_study_results(snaps)) == 0.0

    def test_calmar_complete_wipeout_returns_negative_one(self):
        # Goes from 100k to 0 — final ≤ 0.
        totals = [100_000.0] + [0.0] * 59
        snaps = _build_snapshots(totals)
        result = calmar_objective(_study_results(snaps))
        # compound_return is forced to -1.0 and max_dd = 1.0
        # Calmar = -1.0 / 1.0 = -1.0
        assert result == pytest.approx(-1.0, abs=1e-9)

    def test_calmar_ignores_val_data_when_train_present(self):
        # Train ends at +5%, val is irrelevant noise.
        train = [100_000.0] * 30 + [105_000.0] * 30
        val = [50_000.0] * 60  # noise; should not affect Calmar
        snaps = _build_snapshots(
            train + val,
            train_val_labels=["train"] * 60 + ["val"] * 60,
        )
        result = calmar_objective(_study_results(snaps))
        # Train: 100k flat, then 105k flat. Drawdown = 0; positive return.
        # → ZERO_DD_POSITIVE_RETURN_SENTINEL
        assert result == ZERO_DD_POSITIVE_RETURN_SENTINEL
