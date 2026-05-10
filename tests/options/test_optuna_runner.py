"""Tests for ``src/options/optuna_runner.py`` (Phase 2 Section 7).

All tests use a fully-stubbed :class:`EngineDeps` — no Tradier, no
filesystem outside of ``tmp_path``. SQLite storage is rerouted to a
per-test tmp path. ``n_trials`` is small (1–5) to keep the suite fast.

The smoke study is exercised only via constants check; the full smoke
is a manual post-merge run (network-dependent, ~30min).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from src.options.engine import EngineDeps
from src.options.greeks import price as bsm_price
from src.options.occ import parse_occ_symbol
from src.options.optuna_runner import (
    FAILED_TRIAL_SENTINEL,
    OptunaStudyResults,
    calmar_objective,
    run_optuna_study,
)
from src.options.types import ContractSpec


# ----------------- helpers -----------------


def _trading_days(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _make_stub_deps(
    *,
    universe: tuple[str, ...] = ("AAPL",),
    iv: float = 0.30,
    flat_close: float = 100.0,
) -> EngineDeps:
    """Return EngineDeps that yields deterministic flat-spot,
    BSM-priced chain data so trials always produce a valid (if boring)
    StudyResults."""

    def fetch_close(symbol: str, sim_date: date) -> Optional[float]:
        if symbol in universe:
            return flat_close
        try:
            spec = parse_occ_symbol(symbol)
        except ValueError:
            return None
        t = max((spec.expiration_date - sim_date).days / 365.0, 1e-6)
        return bsm_price(
            flat_close, spec.strike, t, 0.04, 0.0, iv, spec.option_type
        )

    def reconstruct_chain(
        underlying: str,
        sim_date: date,
        target_expiration: date,
        spot: float,
    ) -> list[tuple[ContractSpec, float]]:
        if underlying not in universe:
            return []
        out = []
        t = max((target_expiration - sim_date).days / 365.0, 1e-6)
        for k in [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]:
            for ot in ("C", "P"):
                spec = ContractSpec(
                    underlying=underlying,
                    expiration_date=target_expiration,
                    option_type=ot,
                    strike=k,
                )
                close = bsm_price(spot, k, t, 0.04, 0.0, iv, ot)
                out.append((spec, close))
        return out

    def fetch_earnings(ticker: str) -> tuple[date, ...]:
        return ()

    def trading_days_fn(start: date, end: date) -> list[date]:
        return _trading_days(start, end)

    return EngineDeps(
        fetch_close=fetch_close,
        reconstruct_chain=reconstruct_chain,
        fetch_earnings_dates=fetch_earnings,
        trading_days=trading_days_fn,
    )


def _study_kwargs(*, study_label: str = "test_study") -> dict:
    return dict(
        study_label=study_label,
        strategy_class="cash_secured_put",
        universe=("AAPL",),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 4, 30),
        train_val_split_date=date(2024, 3, 15),
        starting_capital=250_000.0,
    )


# ----------------- basic flow -----------------


class TestRunOptunaStudyBasic:
    def test_run_optuna_study_completes_n_trials(self, tmp_path):
        deps = _make_stub_deps()
        results = run_optuna_study(
            n_trials=3,
            output_dir=tmp_path / "out",
            storage_path=tmp_path / "study.db",
            deps=deps,
            **_study_kwargs(),
        )
        assert results.n_trials_run == 3

    def test_run_optuna_study_returns_results_object(self, tmp_path):
        deps = _make_stub_deps()
        results = run_optuna_study(
            n_trials=2,
            output_dir=tmp_path / "out",
            storage_path=tmp_path / "study.db",
            deps=deps,
            **_study_kwargs(),
        )
        assert isinstance(results, OptunaStudyResults)
        assert results.study_label == "test_study"
        assert results.strategy_class == "cash_secured_put"
        assert results.wall_time_seconds >= 0.0

    def test_run_optuna_study_uses_default_universe_when_none(self, tmp_path):
        deps = _make_stub_deps(
            universe=("SPX", "SPY", "QQQ", "AAPL", "JPM", "MSFT", "NVDA", "XOM"),
        )
        kwargs = _study_kwargs()
        kwargs["universe"] = None
        results = run_optuna_study(
            n_trials=1,
            output_dir=tmp_path / "out",
            storage_path=tmp_path / "study.db",
            deps=deps,
            **kwargs,
        )
        # Smoke-only: ensure it completed without error.
        assert results.n_trials_run == 1


# ----------------- top-K persistence -----------------


class TestTopKPersistence:
    def test_top_k_persists_full_output_to_disk(self, tmp_path):
        deps = _make_stub_deps()
        out = tmp_path / "out"
        results = run_optuna_study(
            n_trials=3,
            top_k=2,
            output_dir=out,
            storage_path=tmp_path / "study.db",
            deps=deps,
            **_study_kwargs(),
        )
        # Top-K trial directories should exist with parquet files.
        assert len(results.top_k_trial_numbers) <= 2
        for tn in results.top_k_trial_numbers:
            trial_dir = out / f"trial_{tn:04d}"
            assert trial_dir.exists()
            assert (trial_dir / "daily.parquet").exists()
            assert (trial_dir / "trades.parquet").exists()
            assert (trial_dir / "config.json").exists()
            assert (trial_dir / "run_meta.json").exists()

    def test_non_top_k_does_not_write_full_output(self, tmp_path):
        deps = _make_stub_deps()
        out = tmp_path / "out"
        results = run_optuna_study(
            n_trials=4,
            top_k=2,
            output_dir=out,
            storage_path=tmp_path / "study.db",
            deps=deps,
            **_study_kwargs(),
        )
        all_dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
        # Only top_k trial dirs should be present.
        assert len(all_dirs) == len(results.top_k_trial_numbers)
        assert all(d.startswith("trial_") for d in all_dirs)


# ----------------- resume -----------------


class TestResume:
    def test_resumes_from_existing_storage(self, tmp_path):
        deps = _make_stub_deps()
        storage = tmp_path / "study.db"
        out = tmp_path / "out"
        # First run: 2 trials.
        first = run_optuna_study(
            n_trials=2,
            output_dir=out,
            storage_path=storage,
            deps=deps,
            **_study_kwargs(),
        )
        assert first.n_trials_run == 2
        # Second run on same study_label + storage: 3 more trials.
        second = run_optuna_study(
            n_trials=3,
            output_dir=out,
            storage_path=storage,
            deps=deps,
            **_study_kwargs(),
        )
        assert second.n_trials_run == 5  # cumulative

    def test_creates_storage_path_if_missing(self, tmp_path):
        deps = _make_stub_deps()
        nested = tmp_path / "nested" / "path" / "study.db"
        assert not nested.parent.exists()
        results = run_optuna_study(
            n_trials=1,
            output_dir=tmp_path / "out",
            storage_path=nested,
            deps=deps,
            **_study_kwargs(),
        )
        assert nested.parent.exists()
        assert results.n_trials_run == 1


# ----------------- failure handling -----------------


class TestFailureHandling:
    def test_failed_trial_returns_minus_one_sentinel(self, tmp_path):
        # Force run_backtest to raise — runner should catch and return
        # the sentinel.
        from src.options import optuna_runner as runner_mod

        with patch.object(
            runner_mod, "run_backtest", side_effect=RuntimeError("boom"),
        ):
            results = run_optuna_study(
                n_trials=2,
                output_dir=tmp_path / "out",
                storage_path=tmp_path / "study.db",
                deps=_make_stub_deps(),
                **_study_kwargs(),
            )
        assert results.n_trials_failed == 2
        # No successful trials → defensive sentinel/empty values.
        assert results.best_value == FAILED_TRIAL_SENTINEL
        assert results.best_trial_number == -1
        assert results.best_params == {}
        assert results.top_k_trial_numbers == ()

    def test_failed_trials_excluded_from_top_k(self, tmp_path):
        # Mix failure + success: have run_backtest raise on the first
        # trial but succeed on the rest.
        from src.options import optuna_runner as runner_mod

        original_run_backtest = runner_mod.run_backtest
        call_count = {"n": 0}

        def maybe_raise(config, *, deps=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first trial fails")
            return original_run_backtest(config, deps=deps)

        with patch.object(
            runner_mod, "run_backtest", side_effect=maybe_raise,
        ):
            results = run_optuna_study(
                n_trials=3,
                output_dir=tmp_path / "out",
                storage_path=tmp_path / "study.db",
                deps=_make_stub_deps(),
                **_study_kwargs(),
            )
        assert results.n_trials_failed == 1
        # The failed trial number should not appear in top_k.
        assert 0 not in results.top_k_trial_numbers

    def test_n_trials_failed_count_accurate(self, tmp_path):
        from src.options import optuna_runner as runner_mod

        with patch.object(
            runner_mod, "run_backtest", side_effect=RuntimeError("boom"),
        ):
            results = run_optuna_study(
                n_trials=4,
                output_dir=tmp_path / "out",
                storage_path=tmp_path / "study.db",
                deps=_make_stub_deps(),
                **_study_kwargs(),
            )
        assert results.n_trials_failed == 4
        assert results.n_trials_run == 4


# ----------------- serialization -----------------


class TestOptunaStudyResultsSerialization:
    def test_to_json_round_trip(self, tmp_path):
        original = OptunaStudyResults(
            study_label="round_trip",
            strategy_class="cash_secured_put",
            n_trials_run=5,
            n_trials_failed=1,
            best_value=2.34,
            best_trial_number=2,
            best_params={"dte_target": 35, "profit_target_pct": 0.5},
            top_k_trial_numbers=(2, 0, 4),
            wall_time_seconds=12.3,
            storage_path=tmp_path / "study.db",
            output_dir=tmp_path / "out",
        )
        path = tmp_path / "summary.json"
        original.to_json(path)
        loaded = OptunaStudyResults.from_json(path)
        assert loaded == original

    def test_to_json_paths_serialized_as_strings(self, tmp_path):
        results = OptunaStudyResults(
            study_label="x",
            strategy_class="cash_secured_put",
            n_trials_run=1,
            n_trials_failed=0,
            best_value=0.5,
            best_trial_number=0,
            best_params={},
            top_k_trial_numbers=(0,),
            wall_time_seconds=1.0,
            storage_path=tmp_path / "a.db",
            output_dir=tmp_path / "b",
        )
        path = tmp_path / "s.json"
        results.to_json(path)
        with open(path) as fh:
            data = json.load(fh)
        assert isinstance(data["storage_path"], str)
        assert isinstance(data["output_dir"], str)


# ----------------- objective wiring -----------------


class TestObjectiveWiring:
    def test_uses_calmar_objective(self, tmp_path):
        # Verify the runner's score for a trial matches what
        # calmar_objective(results) would return for the same trial.
        from src.options import optuna_runner as runner_mod

        original_run_backtest = runner_mod.run_backtest
        captured_results = {}

        def capture(config, *, deps=None):
            results = original_run_backtest(config, deps=deps)
            captured_results[len(captured_results)] = results
            return results

        with patch.object(
            runner_mod, "run_backtest", side_effect=capture,
        ):
            study_results = run_optuna_study(
                n_trials=2,
                output_dir=tmp_path / "out",
                storage_path=tmp_path / "study.db",
                deps=_make_stub_deps(),
                **_study_kwargs(),
            )

        # Recompute Calmar for each captured result; it should match
        # what the runner stored as best_value (for the best trial).
        recomputed = {
            i: calmar_objective(r) for i, r in captured_results.items()
        }
        assert study_results.best_value == pytest.approx(max(recomputed.values()))


# ----------------- smoke study constants -----------------


class TestSmokeStudyConstants:
    def test_smoke_study_constants_match_locked_config(self):
        # Import the smoke script's constants directly.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "smoke_module",
            Path(__file__).resolve().parents[2]
            / "scripts" / "run_options_smoke_study.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.SMOKE_START_DATE == date(2024, 1, 2)
        assert module.SMOKE_END_DATE == date(2024, 7, 1)
        assert module.SMOKE_SPLIT_DATE == date(2024, 5, 1)
        assert module.SMOKE_TRIALS_PER_STRATEGY == 5
        assert module.SMOKE_TOP_K == 3
        assert module.SMOKE_STARTING_CAPITAL == 100_000.0
