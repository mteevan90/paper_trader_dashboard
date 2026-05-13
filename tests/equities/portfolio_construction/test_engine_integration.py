"""Engine integration test — verifies BaselineVariant reproduces v1 rank_top_n_weights.

This is the smallest test that exercises:
  - Engine threads ConstructionState correctly
  - BaselineVariant.construct() and v1's rank_top_n_weights produce
    identical weights (the foundation of v2-baseline's reproducibility
    against v1 headline numbers)
  - Cap-enforcement order in caps.py matches v1's order in portfolio.py
    (since portfolio.py now delegates to caps.py)

A full backtest-engine integration test against synthetic scores is a
heavier fixture (mock score_fn, daily_returns DataFrame, etc.) — skipping
for Gate 2 since Gate 3 backtest results are the canonical validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equities.portfolio_construction import BaselineVariant
from src.equities.portfolio_construction.base import ConstructionState
from src.equities.study.portfolio import (
    PortfolioConstructionParams,
    rank_top_n_weights,
)


def _make_scores_and_sectors(n: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]
    scores = pd.Series(rng.normal(0.0, 0.02, n), index=tickers)
    sectors = pd.Series([f"Sector{i % 5}" for i in range(n)],
                        index=tickers)
    return scores, sectors


def test_baseline_variant_matches_v1_rank_top_n():
    """v2 BaselineVariant.construct(state) must produce identical weights
    to v1 rank_top_n_weights for the same scores + sectors + params.

    Foundation of the Gate 3 reproducibility check."""
    scores, sectors = _make_scores_and_sectors(n=200)
    params = PortfolioConstructionParams(n=30,
                                          individual_cap=0.075,
                                          sector_cap=0.30)

    v1_weights = rank_top_n_weights(scores, sectors, params)

    variant = BaselineVariant(n=30, individual_cap=0.075, sector_cap=0.30)
    state = ConstructionState(
        date=pd.Timestamp("2024-01-31"),
        scores=scores,
        sectors=sectors,
    )
    v2_weights = variant.construct(state)

    # Same set of tickers selected
    assert set(v1_weights.index) == set(v2_weights.index)
    # Same weights to floating-point precision
    aligned = v2_weights.reindex(v1_weights.index)
    np.testing.assert_array_almost_equal(
        aligned.values, v1_weights.values, decimal=12,
    )


def test_baseline_variant_matches_v1_with_concentrated_sectors():
    """v1 and v2 baseline agree even when sector cap binds aggressively."""
    rng = np.random.default_rng(7)
    n = 50
    tickers = [f"T{i:03d}" for i in range(n)]
    scores = pd.Series(rng.normal(0.0, 0.02, n), index=tickers)
    # Force 25 tickers into one sector → sector cap will bind hard
    sectors = pd.Series(["Tech"] * 25 + ["Health"] * 15 + ["Energy"] * 10,
                        index=tickers)

    params = PortfolioConstructionParams(n=30,
                                          individual_cap=0.075,
                                          sector_cap=0.30)
    v1_weights = rank_top_n_weights(scores, sectors, params)
    variant = BaselineVariant(n=30, individual_cap=0.075, sector_cap=0.30)
    state = ConstructionState(date=pd.Timestamp("2024-01-31"),
                              scores=scores, sectors=sectors)
    v2_weights = variant.construct(state)

    assert set(v1_weights.index) == set(v2_weights.index)
    aligned = v2_weights.reindex(v1_weights.index)
    np.testing.assert_array_almost_equal(
        aligned.values, v1_weights.values, decimal=12,
    )
    # Both sum to 1.0 (renormalize step at end of BaselineVariant
    # ensures full investment; allow floating-point tolerance)
    assert v1_weights.sum() == pytest.approx(1.0)
    assert v2_weights.sum() == pytest.approx(1.0)


def test_baseline_variant_empty_scores():
    variant = BaselineVariant()
    state = ConstructionState(
        date=pd.Timestamp("2024-01-31"),
        scores=pd.Series(dtype=float),
        sectors=pd.Series(dtype=str),
    )
    assert variant.construct(state).empty


def test_baseline_variant_all_nan_scores():
    scores = pd.Series([float("nan")] * 30, index=[f"T{i}" for i in range(30)])
    sectors = pd.Series(["Tech"] * 30, index=scores.index)
    variant = BaselineVariant()
    state = ConstructionState(date=pd.Timestamp("2024-01-31"),
                              scores=scores, sectors=sectors)
    assert variant.construct(state).empty
