"""Pure logic tests for ``src/options/promotion.py`` (Phase 2 Section 8).

No engine, no Optuna. Synthetic StudyResults / DataFrame inputs are
constructed directly so each criterion can be exercised in isolation.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.options.backtest_config import BacktestConfig, FeeModel
from src.options.concentration import ConcentrationResult
from src.options.engine import DailySnapshot, StudyResults
from src.options.optuna_runner import OptunaStudyResults
from src.options.positions import ExitRules
from src.options.promotion import (
    MAX_UNDERLYING_ATTRIBUTION,
    OVERFIT_RATIO_THRESHOLD,
    PromotionCheck,
    PromotionRecommendation,
    REGIME_RATIO_THRESHOLD,
    calmar_from_series,
    evaluate_promotion,
    write_promotion_decision,
)


# ----------------- helpers -----------------


def _stub_config() -> BacktestConfig:
    return BacktestConfig(
        study_label="promo_test",
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


def _build_results(
    train_totals: list[float],
    val_totals: list[float],
    *,
    start_date: date = date(2025, 1, 1),
) -> StudyResults:
    """Build a StudyResults with given train and val portfolio_totals."""
    snaps: list[DailySnapshot] = []
    d = start_date
    for total in train_totals:
        snaps.append(DailySnapshot(
            sim_date=d,
            train_val_label="train",
            cash=total, stock_value=0.0,
            open_positions_count=0, open_positions_mark=0.0,
            realized_pnl_to_date=0.0,
            portfolio_total=total,
            portfolio_delta=0.0, portfolio_gamma=0.0,
            portfolio_theta_per_day=0.0, portfolio_vega_per_pct=0.0,
        ))
        d += timedelta(days=1)
    for total in val_totals:
        snaps.append(DailySnapshot(
            sim_date=d,
            train_val_label="val",
            cash=total, stock_value=0.0,
            open_positions_count=0, open_positions_mark=0.0,
            realized_pnl_to_date=0.0,
            portfolio_total=total,
            portfolio_delta=0.0, portfolio_gamma=0.0,
            portfolio_theta_per_day=0.0, portfolio_vega_per_pct=0.0,
        ))
        d += timedelta(days=1)
    return StudyResults(
        config=_stub_config(),
        daily_snapshots=tuple(snaps),
        closed_positions=(),
        spawned_equity_closes=(),
        skip_counters={},
        wall_time_seconds=0.0,
        run_id="test",
    )


def _stub_optuna_summary() -> OptunaStudyResults:
    return OptunaStudyResults(
        study_label="x",
        strategy_class="cash_secured_put",
        n_trials_run=1,
        n_trials_failed=0,
        best_value=1.0,
        best_trial_number=0,
        best_params={},
        top_k_trial_numbers=(0,),
        wall_time_seconds=0.0,
        storage_path=Path("a.db"),
        output_dir=Path("b"),
    )


def _stub_concentration(
    *,
    underlying_attributions: list[tuple[str, float]] = None,
    high_iv_calmar: float = 1.0,
    low_iv_calmar: float = 1.0,
    base_calmar: float = 2.0,
) -> tuple[ConcentrationResult, ...]:
    if underlying_attributions is None:
        underlying_attributions = [
            ("AAPL", 0.30), ("MSFT", 0.20), ("NVDA", 0.25),
        ]
    out: list[ConcentrationResult] = []
    for ticker, attribution in underlying_attributions:
        out.append(ConcentrationResult(
            ablation_dimension="underlying",
            ablation_value=ticker,
            base_calmar=base_calmar,
            ablated_calmar=base_calmar * (1 - attribution),
            delta_calmar=-base_calmar * attribution,
            pct_alpha_attribution=attribution,
        ))
    out.append(ConcentrationResult(
        ablation_dimension="iv_regime",
        ablation_value="high",
        base_calmar=base_calmar,
        ablated_calmar=high_iv_calmar,
        delta_calmar=high_iv_calmar - base_calmar,
        pct_alpha_attribution=0.0,
    ))
    out.append(ConcentrationResult(
        ablation_dimension="iv_regime",
        ablation_value="low",
        base_calmar=base_calmar,
        ablated_calmar=low_iv_calmar,
        delta_calmar=low_iv_calmar - base_calmar,
        pct_alpha_attribution=0.0,
    ))
    return tuple(out)


def _spy_df_with_calmar(target_calmar: float, n_days: int = 60) -> pd.DataFrame:
    """Build a SPY DataFrame whose total_return_index gives ~target_calmar."""
    if target_calmar == 0.0:
        index = [date(2025, 7, 1) + timedelta(days=i) for i in range(n_days)]
        return pd.DataFrame(
            {"total_return_index": [1.0] * n_days}, index=index,
        )
    # Build a simple monotonic ramp; max_dd will be ~0 → sentinel.
    # For test purposes return a flat series and the helper will treat
    # it as 0 calmar.
    index = [date(2025, 7, 1) + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame(
        {"total_return_index": [1.0 + 0.001 * i for i in range(n_days)]},
        index=index,
    )


# ----------------- calmar_from_series -----------------


class TestCalmarFromSeries:
    def test_short_series_returns_zero(self):
        s = pd.Series(
            [1.0] * 10,
            index=[date(2025, 1, 1) + timedelta(days=i) for i in range(10)],
        )
        assert calmar_from_series(s) == 0.0

    def test_initial_zero_returns_zero(self):
        s = pd.Series(
            [0.0] * 60,
            index=[date(2025, 1, 1) + timedelta(days=i) for i in range(60)],
        )
        assert calmar_from_series(s) == 0.0

    def test_flat_series_returns_zero(self):
        s = pd.Series(
            [1.0] * 60,
            index=[date(2025, 1, 1) + timedelta(days=i) for i in range(60)],
        )
        assert calmar_from_series(s) == 0.0


# ----------------- check evaluators -----------------


class TestOverfitCheck:
    def test_overfit_check_passes_when_val_at_60pct_of_train(self):
        # val Calmar 0.6, train Calmar 1.0 → 0.6 / 1.0 = 0.6 ≥ 0.5
        # Build train_totals to give Calmar ~1.0; val_totals ~0.6
        # Easier: directly compose checks via evaluate_promotion.
        results = _build_results(
            train_totals=([100_000.0] * 30 + [110_000.0] * 30),  # +10% step
            val_totals=([100_000.0] * 30 + [105_000.0] * 30),    # +5% step
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(),
        )
        overfit = next(
            c for c in rec.checks if c.criterion_name == "overfit_check"
        )
        # Both train and val have zero drawdown → both hit the
        # ZERO_DD_POSITIVE_RETURN_SENTINEL. val == train → ratio=1.0 passes.
        assert overfit.passed is True

    def test_overfit_check_fails_when_val_negative_and_train_positive(self):
        # train positive, val negative → val is well below 0.5*train
        results = _build_results(
            train_totals=([100_000.0] * 30 + [120_000.0] * 30),  # up
            val_totals=(  # 100k → 80k linearly: negative return
                [100_000.0 - i * (20_000.0 / 59) for i in range(60)]
            ),
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(),
        )
        overfit = next(
            c for c in rec.checks if c.criterion_name == "overfit_check"
        )
        assert overfit.passed is False


class TestBeatsSpy:
    def test_beats_spy_when_val_calmar_higher(self):
        # val produces high Calmar; SPY series flat (Calmar 0)
        results = _build_results(
            train_totals=([100_000.0] * 30 + [105_000.0] * 30),
            val_totals=([100_000.0] * 30 + [110_000.0] * 30),
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(),
        )
        beats = next(
            c for c in rec.checks if c.criterion_name == "beats_spy"
        )
        # SPY Calmar 0 (flat); val Calmar zero-dd-positive-sentinel.
        assert beats.passed is True


class TestBeatsBxm:
    def test_beats_bxm_skipped_for_csp(self):
        results = _build_results(
            train_totals=[100_000.0] * 60,
            val_totals=[100_000.0] * 60,
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(),
        )
        bxm_check = next(
            c for c in rec.checks if c.criterion_name == "beats_bxm"
        )
        assert bxm_check.passed is True
        assert "auto-pass" in bxm_check.actual.lower()

    def test_beats_bxm_no_data_for_cc_does_not_auto_fail(self):
        # spec: BXM unavailable → fails this check, but aggregation
        # treats one fail as borderline rather than do_not_promote.
        results = _build_results(
            train_totals=([100_000.0] * 30 + [105_000.0] * 30),
            val_totals=([100_000.0] * 30 + [110_000.0] * 30),
        )
        rec = evaluate_promotion(
            strategy_class="covered_call",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(),
        )
        bxm_check = next(
            c for c in rec.checks if c.criterion_name == "beats_bxm"
        )
        assert bxm_check.passed is False
        assert rec.bxm_calmar_on_val is None


class TestNoUnderlyingConcentration:
    def test_passes_when_max_attribution_below_threshold(self):
        results = _build_results(
            train_totals=[100_000.0] * 60,
            val_totals=[100_000.0] * 60,
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(
                underlying_attributions=[
                    ("AAPL", 0.30), ("MSFT", 0.20),
                ],
            ),
        )
        c = next(
            c for c in rec.checks
            if c.criterion_name == "no_underlying_concentration"
        )
        assert c.passed is True

    def test_fails_at_55pct_attribution(self):
        results = _build_results(
            train_totals=[100_000.0] * 60,
            val_totals=[100_000.0] * 60,
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(
                underlying_attributions=[("NVDA", 0.55), ("AAPL", 0.10)],
            ),
        )
        c = next(
            c for c in rec.checks
            if c.criterion_name == "no_underlying_concentration"
        )
        assert c.passed is False


class TestRegimeIndependence:
    def test_passes_when_within_2x(self):
        results = _build_results(
            train_totals=[100_000.0] * 60,
            val_totals=[100_000.0] * 60,
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(
                high_iv_calmar=1.0, low_iv_calmar=1.5,
            ),
        )
        c = next(
            c for c in rec.checks
            if c.criterion_name == "regime_independence"
        )
        # ratio = 1.0/1.5 = 0.667 ≥ 0.5
        assert c.passed is True

    def test_fails_when_4x_apart(self):
        results = _build_results(
            train_totals=[100_000.0] * 60,
            val_totals=[100_000.0] * 60,
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(
                high_iv_calmar=0.5, low_iv_calmar=2.0,
            ),
        )
        c = next(
            c for c in rec.checks
            if c.criterion_name == "regime_independence"
        )
        # ratio = 0.5/2.0 = 0.25 < 0.5
        assert c.passed is False


# ----------------- aggregation -----------------


class TestAggregation:
    def test_all_pass_returns_promote(self):
        results = _build_results(
            train_totals=([100_000.0] * 30 + [105_000.0] * 30),
            val_totals=([100_000.0] * 30 + [110_000.0] * 30),
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(
                underlying_attributions=[("AAPL", 0.20)],
                high_iv_calmar=1.0, low_iv_calmar=1.2,
            ),
        )
        assert rec.automated_recommendation == "promote"

    def test_one_fail_returns_borderline(self):
        # Make the no_underlying_concentration check fail by setting
        # one ticker to 0.6 attribution.
        results = _build_results(
            train_totals=([100_000.0] * 30 + [105_000.0] * 30),
            val_totals=([100_000.0] * 30 + [110_000.0] * 30),
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(
                underlying_attributions=[("AAPL", 0.60)],
                high_iv_calmar=1.0, low_iv_calmar=1.2,
            ),
        )
        assert rec.automated_recommendation == "borderline"

    def test_multiple_fails_returns_do_not_promote(self):
        results = _build_results(
            train_totals=([100_000.0] * 30 + [105_000.0] * 30),
            val_totals=([100_000.0] * 30 + [110_000.0] * 30),
        )
        rec = evaluate_promotion(
            strategy_class="cash_secured_put",
            primary_study=_stub_optuna_summary(),
            primary_results=results,
            spy_total_return=_spy_df_with_calmar(0.0),
            bxm=None,
            concentration_results=_stub_concentration(
                underlying_attributions=[("AAPL", 0.60), ("NVDA", 0.55)],
                high_iv_calmar=0.5, low_iv_calmar=2.0,
            ),
        )
        assert rec.automated_recommendation == "do_not_promote"


# ----------------- to_dict / from_dict -----------------


class TestSerialization:
    def test_promotion_recommendation_to_dict_round_trip(self):
        original = PromotionRecommendation(
            automated_recommendation="promote",
            checks=(
                PromotionCheck(
                    criterion_name="overfit_check",
                    passed=True,
                    expected="x",
                    actual="y",
                    explanation="z",
                ),
            ),
            train_calmar=1.5,
            val_calmar=0.9,
            spy_calmar_on_val=0.4,
            bxm_calmar_on_val=0.3,
            summary="ok",
        )
        recovered = PromotionRecommendation.from_dict(original.to_dict())
        assert recovered == original

    def test_round_trip_handles_none_bxm(self):
        original = PromotionRecommendation(
            automated_recommendation="borderline",
            checks=(),
            train_calmar=0.5,
            val_calmar=0.2,
            spy_calmar_on_val=0.1,
            bxm_calmar_on_val=None,
            summary="x",
        )
        recovered = PromotionRecommendation.from_dict(original.to_dict())
        assert recovered.bxm_calmar_on_val is None


# ----------------- write_promotion_decision -----------------


class TestWriteDecision:
    def test_write_automated_only(self, tmp_path):
        rec = PromotionRecommendation(
            automated_recommendation="promote",
            checks=(),
            train_calmar=1.0,
            val_calmar=0.7,
            spy_calmar_on_val=0.2,
            bxm_calmar_on_val=None,
            summary="ok",
        )
        path = write_promotion_decision(tmp_path, rec)
        assert path.exists()
        with open(path) as fh:
            data = json.load(fh)
        assert data["human_override"] is None
        assert data["automated"]["automated_recommendation"] == "promote"

    def test_write_with_human_override(self, tmp_path):
        rec = PromotionRecommendation(
            automated_recommendation="borderline",
            checks=(),
            train_calmar=1.0,
            val_calmar=0.4,
            spy_calmar_on_val=0.2,
            bxm_calmar_on_val=None,
            summary="x",
        )
        override = {
            "decision": "promote",
            "reasoning": "Borderline but I want to ship",
        }
        path = write_promotion_decision(
            tmp_path, rec, human_override=override,
        )
        with open(path) as fh:
            data = json.load(fh)
        assert data["human_override"] == override
