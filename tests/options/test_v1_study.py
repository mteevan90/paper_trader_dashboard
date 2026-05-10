"""Top-level orchestrator tests for ``src/options/v1_study.py`` (Phase
2 Section 8). All Tradier deps mocked; tiny n_trials and a 2-ticker
universe so the integration runs in a few seconds.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.options import v1_study as v1_study_mod
from src.options.engine import DailySnapshot, StudyResults
from src.options.v1_study import STRATEGY_CLASSES, run_v1_study


def _stub_results(config) -> StudyResults:
    """Fast deterministic StudyResults mirroring config window."""
    snaps: list[DailySnapshot] = []
    d = config.start_date
    while d <= config.end_date:
        label = (
            "train"
            if d <= config.train_val_split_date
            else "val"
        )
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


def _spy_df(start: date, end: date) -> pd.DataFrame:
    days = []
    closes = []
    d = start
    while d <= end:
        days.append(d)
        closes.append(400.0)
        d += timedelta(days=1)
    return pd.DataFrame(
        {
            "close": closes,
            "dividend_per_share": [0.0] * len(closes),
            "total_return_index": [1.0 + i * 0.001 for i in range(len(closes))],
        },
        index=pd.Index(days, name="date"),
    )


def _bxm_df(start: date, end: date) -> pd.DataFrame:
    days = []
    closes = []
    d = start
    while d <= end:
        days.append(d)
        closes.append(400.0)
        d += timedelta(days=1)
    return pd.DataFrame({"close": closes}, index=pd.Index(days, name="date"))


@pytest.fixture(autouse=True)
def _reroute_snapshots(monkeypatch, tmp_path):
    monkeypatch.setattr(
        v1_study_mod, "SNAPSHOTS_BASE_DIR", tmp_path / "snapshots",
    )


@pytest.fixture
def _stubbed_components(monkeypatch, tmp_path):
    """Replace all the heavy components with fast stubs."""
    from src.options import concentration as concentration_mod
    from src.options import optuna_runner as runner_mod

    # Stub run_backtest in BOTH the runner and the concentration
    # module's import sites.
    monkeypatch.setattr(
        runner_mod, "run_backtest",
        lambda config, *, deps=None: _stub_results(config),
    )
    monkeypatch.setattr(
        concentration_mod, "run_backtest",
        lambda config, *, deps=None, entry_filters=None: _stub_results(config),
    )
    # Stub benchmark fetchers.
    monkeypatch.setattr(
        v1_study_mod, "fetch_spy_total_return",
        lambda start, end: _spy_df(start, end),
    )
    monkeypatch.setattr(
        v1_study_mod, "fetch_bxm",
        lambda start, end: _bxm_df(start, end),
    )


def _v1_kwargs(tmp_path):
    return dict(
        run_id="test_run",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 6, 1),
        train_val_split_date=date(2024, 4, 1),
        starting_capital=100_000.0,
        n_trials_primary=2,
        n_trials_per_ablation=1,
        output_base_dir=tmp_path / "v1_study",
        seed=42,
        interactive=False,
        universe=("AAPL", "MSFT"),
    )


class TestRunV1Study:
    def test_completes_both_strategies(self, _stubbed_components, tmp_path):
        paths = run_v1_study(**_v1_kwargs(tmp_path))
        assert "csp_dir" in paths
        assert "cc_dir" in paths
        assert "snapshot_dir" in paths

    def test_writes_csp_output_and_cc_output(
        self, _stubbed_components, tmp_path,
    ):
        paths = run_v1_study(**_v1_kwargs(tmp_path))
        assert paths["csp_dir"].exists()
        assert paths["cc_dir"].exists()
        # Each strategy directory should have a primary subdir.
        assert (paths["csp_dir"] / "primary").exists()
        assert (paths["cc_dir"] / "primary").exists()

    def test_creates_snapshot_dir(self, _stubbed_components, tmp_path):
        paths = run_v1_study(**_v1_kwargs(tmp_path))
        assert paths["snapshot_dir"].exists()
        # Snapshot has both strategy subdirs copied.
        assert (paths["snapshot_dir"] / "cash_secured_put").exists()
        assert (paths["snapshot_dir"] / "covered_call").exists()

    def test_promotion_decision_files_exist(
        self, _stubbed_components, tmp_path,
    ):
        paths = run_v1_study(**_v1_kwargs(tmp_path))
        assert (paths["csp_dir"] / "promotion_decision.json").exists()
        assert (paths["cc_dir"] / "promotion_decision.json").exists()

    def test_non_interactive_skips_prompt(
        self, _stubbed_components, tmp_path,
    ):
        # No prompt patch needed: interactive=False bypasses input().
        paths = run_v1_study(**_v1_kwargs(tmp_path))
        # File exists but human_override is None.
        import json
        with open(paths["csp_dir"] / "promotion_decision.json") as fh:
            data = json.load(fh)
        assert data["human_override"] is None
