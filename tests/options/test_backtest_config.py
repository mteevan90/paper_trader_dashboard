"""Tests for ``src/options/backtest_config.py`` (Phase 2 Section 5).

All offline. The Optuna trial is faked via ``unittest.mock.MagicMock``
with ``side_effect`` keyed by parameter name — no real Optuna study or
sampler is constructed at test time.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.options.backtest_config import (
    DEFAULT_UNIVERSE,
    VALID_STRATEGY_CLASSES,
    BacktestConfig,
    FeeModel,
)
from src.options.positions import ExitRules


# ----------------- helpers -----------------


def _exit_rules() -> ExitRules:
    return ExitRules(
        profit_target_pct=0.50,
        time_stop_dte=21,
        stop_loss_pct=2.0,
    )


def _config(**overrides) -> BacktestConfig:
    """Build a valid BacktestConfig; overrides replace any field."""
    base = dict(
        study_label="smoke",
        strategy_class="cash_secured_put",
        universe=DEFAULT_UNIVERSE,
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        train_val_split_date=date(2025, 1, 1),
        dte_target=35,
        strike_selector_target_delta=0.30,
        max_concurrent_positions=5,
        earnings_window_avoid=True,
        max_loss_pct_of_portfolio=0.02,
        exit_rules=_exit_rules(),
        fees=FeeModel(),
    )
    base.update(overrides)
    return BacktestConfig(**base)


def _make_trial(
    *,
    dte_target: int = 30,
    strike_selector_target_delta: float = 0.30,
    max_concurrent_positions: int = 5,
    earnings_window_avoid: bool = True,
    max_loss_pct_of_portfolio: float = 0.02,
    profit_target_pct: float = 0.50,
    time_stop_dte: int = 21,
    stop_loss_pct: float = 2.0,
) -> MagicMock:
    """MagicMock trial whose ``suggest_*`` calls return canned values keyed by name."""
    int_returns = {
        "dte_target": dte_target,
        "max_concurrent_positions": max_concurrent_positions,
        "time_stop_dte": time_stop_dte,
    }
    float_returns = {
        "strike_selector_target_delta": strike_selector_target_delta,
        "max_loss_pct_of_portfolio": max_loss_pct_of_portfolio,
        "profit_target_pct": profit_target_pct,
        "stop_loss_pct": stop_loss_pct,
    }
    cat_returns = {"earnings_window_avoid": earnings_window_avoid}
    trial = MagicMock()
    trial.suggest_int.side_effect = (
        lambda name, low, high, **kw: int_returns[name]
    )
    trial.suggest_float.side_effect = (
        lambda name, low, high, **kw: float_returns[name]
    )
    trial.suggest_categorical.side_effect = (
        lambda name, choices, **kw: cat_returns[name]
    )
    return trial


_SUGGEST_KWARGS = dict(
    study_label="smoke",
    strategy_class="cash_secured_put",
    start_date=date(2024, 1, 1),
    end_date=date(2025, 12, 31),
    train_val_split_date=date(2025, 1, 1),
)


# ----------------- FeeModel -----------------


class TestFeeModel:
    def test_fee_model_defaults(self):
        fee = FeeModel()
        assert fee.broker_fee_per_contract == 0.35
        assert fee.regulatory_fee_per_contract == 0.10

    def test_fee_model_validates_negative_broker_raises(self):
        with pytest.raises(ValueError, match="broker_fee_per_contract"):
            FeeModel(broker_fee_per_contract=-0.01)

    def test_fee_model_validates_negative_regulatory_raises(self):
        with pytest.raises(ValueError, match="regulatory_fee_per_contract"):
            FeeModel(regulatory_fee_per_contract=-0.01)

    def test_fee_model_total_per_contract_one_way(self):
        assert FeeModel().total_per_contract_one_way() == pytest.approx(0.45)
        custom = FeeModel(
            broker_fee_per_contract=0.50,
            regulatory_fee_per_contract=0.15,
        )
        assert custom.total_per_contract_one_way() == pytest.approx(0.65)

    def test_fee_model_compute_fee_round_trip(self):
        # 1 contract round-trip = 2 × 0.45 = 0.90
        assert FeeModel().compute_fee(1) == pytest.approx(0.90)

    def test_fee_model_compute_fee_one_way(self):
        # 1 contract one-way = 0.45
        assert FeeModel().compute_fee(1, round_trip=False) == pytest.approx(0.45)

    def test_fee_model_compute_fee_multiple_contracts(self):
        # 5 contracts round-trip = 5 × 2 × 0.45 = 4.50
        assert FeeModel().compute_fee(5) == pytest.approx(4.50)
        # 3 contracts one-way = 3 × 0.45 = 1.35
        assert FeeModel().compute_fee(3, round_trip=False) == pytest.approx(1.35)


# ----------------- BacktestConfig: construction + validation -----------------


class TestBacktestConfigConstruction:
    def test_backtest_config_minimal_construction(self):
        cfg = _config()
        assert cfg.study_label == "smoke"
        assert cfg.strategy_class == "cash_secured_put"
        assert cfg.dte_target == 35
        assert cfg.promotable is False
        assert cfg.random_seed is None

    def test_backtest_config_default_universe_eight_names(self):
        assert len(DEFAULT_UNIVERSE) == 8
        assert DEFAULT_UNIVERSE == (
            "SPX", "SPY", "QQQ",
            "AAPL", "JPM", "MSFT", "NVDA", "XOM",
        )

    def test_backtest_config_promotable_defaults_false(self):
        assert _config().promotable is False

    def test_backtest_config_random_seed_defaults_none(self):
        assert _config().random_seed is None

    def test_backtest_config_empty_study_label_raises(self):
        with pytest.raises(ValueError, match="study_label"):
            _config(study_label="")

    def test_backtest_config_invalid_strategy_class_raises(self):
        with pytest.raises(ValueError, match="strategy_class"):
            _config(strategy_class="iron_condor")

    def test_backtest_config_empty_universe_raises(self):
        with pytest.raises(ValueError, match="universe"):
            _config(universe=())

    def test_backtest_config_end_before_start_raises(self):
        with pytest.raises(ValueError, match="end_date"):
            _config(
                start_date=date(2025, 1, 1),
                end_date=date(2024, 1, 1),
                train_val_split_date=date(2024, 6, 1),
            )

    def test_backtest_config_split_outside_window_raises(self):
        # split == start
        with pytest.raises(ValueError, match="train_val_split_date"):
            _config(train_val_split_date=date(2024, 1, 1))
        # split == end
        with pytest.raises(ValueError, match="train_val_split_date"):
            _config(train_val_split_date=date(2025, 12, 31))
        # split before start
        with pytest.raises(ValueError, match="train_val_split_date"):
            _config(train_val_split_date=date(2023, 1, 1))
        # split after end
        with pytest.raises(ValueError, match="train_val_split_date"):
            _config(train_val_split_date=date(2026, 6, 1))

    def test_backtest_config_dte_target_out_of_range_raises(self):
        with pytest.raises(ValueError, match="dte_target"):
            _config(dte_target=5)
        with pytest.raises(ValueError, match="dte_target"):
            _config(dte_target=100)

    def test_backtest_config_delta_out_of_range_raises(self):
        with pytest.raises(ValueError, match="strike_selector_target_delta"):
            _config(strike_selector_target_delta=0.0)
        with pytest.raises(ValueError, match="strike_selector_target_delta"):
            _config(strike_selector_target_delta=1.0)

    def test_backtest_config_max_concurrent_zero_raises(self):
        with pytest.raises(ValueError, match="max_concurrent_positions"):
            _config(max_concurrent_positions=0)

    def test_backtest_config_max_loss_pct_too_large_raises(self):
        with pytest.raises(ValueError, match="max_loss_pct_of_portfolio"):
            _config(max_loss_pct_of_portfolio=0.25)
        with pytest.raises(ValueError, match="max_loss_pct_of_portfolio"):
            _config(max_loss_pct_of_portfolio=0.0)

    def test_backtest_config_embeds_exit_rules(self):
        rules = ExitRules(
            profit_target_pct=0.40,
            time_stop_dte=14,
            stop_loss_pct=2.5,
        )
        cfg = _config(exit_rules=rules)
        assert cfg.exit_rules is rules
        assert cfg.exit_rules.profit_target_pct == 0.40

    def test_backtest_config_embeds_fee_model(self):
        fees = FeeModel(
            broker_fee_per_contract=0.0,
            regulatory_fee_per_contract=0.10,
        )
        cfg = _config(fees=fees)
        assert cfg.fees is fees
        assert cfg.fees.compute_fee(1) == pytest.approx(0.20)

    def test_backtest_config_strategy_class_set_matches_constant(self):
        # Sanity check that the public constant covers v1 strategies.
        assert VALID_STRATEGY_CLASSES == frozenset(
            {"covered_call", "cash_secured_put"}
        )


# ----------------- suggest() classmethod -----------------


class TestSuggest:
    def test_suggest_constructs_valid_config(self):
        trial = _make_trial()
        cfg = BacktestConfig.suggest(trial, **_SUGGEST_KWARGS)
        assert isinstance(cfg, BacktestConfig)
        assert cfg.study_label == "smoke"
        assert cfg.strategy_class == "cash_secured_put"
        assert cfg.dte_target == 30
        assert cfg.exit_rules.profit_target_pct == 0.50
        assert cfg.exit_rules.time_stop_dte == 21

    def test_suggest_uses_default_universe_when_none_given(self):
        trial = _make_trial()
        cfg = BacktestConfig.suggest(trial, **_SUGGEST_KWARGS)
        assert cfg.universe == DEFAULT_UNIVERSE

    def test_suggest_uses_provided_universe_when_given(self):
        trial = _make_trial()
        custom = ("SPY", "QQQ")
        cfg = BacktestConfig.suggest(
            trial, universe=custom, **_SUGGEST_KWARGS
        )
        assert cfg.universe == custom

    def test_suggest_uses_default_fees_when_none_given(self):
        trial = _make_trial()
        cfg = BacktestConfig.suggest(trial, **_SUGGEST_KWARGS)
        assert cfg.fees == FeeModel()

    def test_suggest_passes_through_promotable_and_seed(self):
        trial = _make_trial()
        cfg = BacktestConfig.suggest(
            trial,
            promotable=True,
            random_seed=42,
            **_SUGGEST_KWARGS,
        )
        assert cfg.promotable is True
        assert cfg.random_seed == 42

    def test_suggest_calls_trial_with_expected_parameter_names(self):
        trial = _make_trial()
        BacktestConfig.suggest(trial, **_SUGGEST_KWARGS)
        int_names = {
            call.args[0] for call in trial.suggest_int.call_args_list
        }
        float_names = {
            call.args[0] for call in trial.suggest_float.call_args_list
        }
        cat_names = {
            call.args[0] for call in trial.suggest_categorical.call_args_list
        }
        assert int_names == {
            "dte_target", "max_concurrent_positions", "time_stop_dte",
        }
        assert float_names == {
            "strike_selector_target_delta",
            "max_loss_pct_of_portfolio",
            "profit_target_pct",
            "stop_loss_pct",
        }
        assert cat_names == {"earnings_window_avoid"}

    def test_suggest_search_ranges_match_spec(self):
        trial = _make_trial()
        BacktestConfig.suggest(trial, **_SUGGEST_KWARGS)

        int_ranges = {
            call.args[0]: (call.args[1], call.args[2])
            for call in trial.suggest_int.call_args_list
        }
        float_ranges = {
            call.args[0]: (call.args[1], call.args[2])
            for call in trial.suggest_float.call_args_list
        }
        cat_choices = {
            call.args[0]: call.args[1]
            for call in trial.suggest_categorical.call_args_list
        }

        assert int_ranges["dte_target"] == (25, 50)
        assert int_ranges["max_concurrent_positions"] == (3, 10)
        assert int_ranges["time_stop_dte"] == (7, 28)

        assert float_ranges["strike_selector_target_delta"] == (0.15, 0.40)
        assert float_ranges["max_loss_pct_of_portfolio"] == (0.01, 0.04)
        assert float_ranges["profit_target_pct"] == (0.25, 0.80)
        assert float_ranges["stop_loss_pct"] == (1.5, 3.5)

        assert cat_choices["earnings_window_avoid"] == [True, False]


# ----------------- serialization -----------------


class TestSerialization:
    def test_to_dict_round_trip_via_from_dict(self):
        cfg = _config(
            promotable=True,
            random_seed=7,
        )
        round_tripped = BacktestConfig.from_dict(cfg.to_dict())
        assert round_tripped == cfg

    def test_to_dict_dates_are_iso_strings(self):
        cfg = _config()
        data = cfg.to_dict()
        assert data["start_date"] == "2024-01-01"
        assert data["end_date"] == "2025-12-31"
        assert data["train_val_split_date"] == "2025-01-01"

    def test_to_dict_universe_is_list(self):
        cfg = _config(universe=("SPY", "QQQ"))
        data = cfg.to_dict()
        assert isinstance(data["universe"], list)
        assert data["universe"] == ["SPY", "QQQ"]
        round_tripped = BacktestConfig.from_dict(data)
        assert isinstance(round_tripped.universe, tuple)
        assert round_tripped.universe == ("SPY", "QQQ")

    def test_to_dict_nested_exit_rules_roundtrip(self):
        rules = ExitRules(
            profit_target_pct=0.42,
            time_stop_dte=14,
            stop_loss_pct=2.75,
        )
        cfg = _config(exit_rules=rules)
        data = cfg.to_dict()
        assert data["exit_rules"] == {
            "profit_target_pct": 0.42,
            "time_stop_dte": 14,
            "stop_loss_pct": 2.75,
        }
        assert BacktestConfig.from_dict(data).exit_rules == rules

    def test_to_dict_nested_fee_model_roundtrip(self):
        fees = FeeModel(
            broker_fee_per_contract=0.0,
            regulatory_fee_per_contract=0.12,
        )
        cfg = _config(fees=fees)
        data = cfg.to_dict()
        assert data["fees"] == {
            "broker_fee_per_contract": 0.0,
            "regulatory_fee_per_contract": 0.12,
        }
        assert BacktestConfig.from_dict(data).fees == fees


# ----------------- evolve() -----------------


class TestEvolve:
    def test_evolve_returns_new_instance(self):
        cfg = _config()
        evolved = cfg.evolve(study_label="another")
        assert evolved is not cfg
        assert evolved.study_label == "another"
        # Original unchanged.
        assert cfg.study_label == "smoke"

    def test_evolve_changes_universe_for_concentration_analysis(self):
        cfg = _config()
        # Concentration ablation: drop NVDA.
        ablated = tuple(t for t in cfg.universe if t != "NVDA")
        evolved = cfg.evolve(universe=ablated)
        assert "NVDA" not in evolved.universe
        assert "NVDA" in cfg.universe  # original untouched

    def test_evolve_changes_exit_rules_atomically(self):
        cfg = _config()
        new_rules = ExitRules(
            profit_target_pct=0.30,
            time_stop_dte=10,
            stop_loss_pct=1.5,
        )
        evolved = cfg.evolve(exit_rules=new_rules)
        assert evolved.exit_rules is new_rules
        assert cfg.exit_rules is not new_rules

    def test_evolve_invalid_change_raises(self):
        cfg = _config()
        with pytest.raises(ValueError, match="end_date"):
            cfg.evolve(end_date=date(2023, 1, 1))
