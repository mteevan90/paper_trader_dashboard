"""Tests for shared cap-enforcement utilities.

These guarantee that caps.py reproduces v1's exact behavior since v2-baseline
reproducibility (Gate 3 gate: <1% deviation) flows through these functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equities.portfolio_construction.caps import (
    enforce_caps,
    enforce_individual_cap,
    enforce_sector_cap,
)


def test_individual_cap_no_op_when_under_cap():
    # Equal-weight 1/30 = 3.33% vs 7.5% cap → no change expected
    weights = pd.Series([1 / 30] * 30, index=[f"T{i}" for i in range(30)])
    out = enforce_individual_cap(weights, cap=0.075)
    np.testing.assert_array_almost_equal(out.values, weights.values)


def test_individual_cap_redistributes_proportionally():
    # 5 names; one at 50%, four at 12.5%. Cap at 30%.
    weights = pd.Series([0.50, 0.125, 0.125, 0.125, 0.125],
                        index=["A", "B", "C", "D", "E"])
    out = enforce_individual_cap(weights, cap=0.30)
    # Excess 0.20 redistributed proportionally to under-cap names
    # (each at 0.125 = 25% of under-cap total 0.50; so each gets 0.05)
    assert out["A"] == pytest.approx(0.30)
    np.testing.assert_array_almost_equal(out.loc[["B", "C", "D", "E"]].values,
                                          [0.175, 0.175, 0.175, 0.175])
    assert out.sum() == pytest.approx(1.0)


def test_individual_cap_handles_cascade():
    # Two names well over cap; redistribution might push redistributees over
    weights = pd.Series([0.40, 0.40, 0.10, 0.10],
                        index=["A", "B", "C", "D"])
    out = enforce_individual_cap(weights, cap=0.25)
    # All weights <= cap after iteration
    assert (out <= 0.25 + 1e-9).all()
    # Total preserved
    assert out.sum() == pytest.approx(1.0)


def test_individual_cap_empty():
    out = enforce_individual_cap(pd.Series(dtype=float), cap=0.075)
    assert out.empty


def test_sector_cap_no_op_when_under_cap():
    # 5 tickers in 3 sectors; no sector exceeds 30%
    weights = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20],
                        index=["A", "B", "C", "D", "E"])
    sectors = pd.Series(["Tech", "Tech", "Health", "Energy", "Energy"],
                        index=["A", "B", "C", "D", "E"])
    out = enforce_sector_cap(weights, sectors, sector_cap=0.50,
                              individual_cap=0.075)
    # No change expected
    np.testing.assert_array_almost_equal(out.values, weights.values)


def test_sector_cap_redistributes_to_other_sectors():
    # 4 Tech at 10% each = 40% (over 30% cap), plus 6 single-ticker sectors
    # at 10% each = 60%. Only Tech is over. Redistribution lands cleanly
    # because the 6 under-sectors have abundant headroom (each at 10% vs
    # 30% cap), so the 10% excess from Tech can spread without cycling.
    weights = pd.Series([0.10] * 10,
                        index=[f"T{i}" for i in range(10)])
    sectors = pd.Series(
        ["Tech"] * 4 + ["Health", "Energy", "Materials",
                          "Consumer", "Utilities", "RealEstate"],
        index=weights.index,
    )
    out = enforce_sector_cap(weights, sectors, sector_cap=0.30,
                              individual_cap=0.20)
    # Tech scaled to 30% total (7.5% each)
    np.testing.assert_array_almost_equal(out.loc[[f"T{i}" for i in range(4)]].values,
                                          [0.075] * 4)
    # 10% excess redistributed proportionally to the 6 under-sector tickers
    # (each at 0.10 → +0.10 * 0.10/0.60 = +0.01667 → ~0.1167 each)
    np.testing.assert_array_almost_equal(
        out.loc[[f"T{i}" for i in range(4, 10)]].values,
        [0.10 + 0.10 / 6] * 6,
        decimal=10,
    )
    # Sum preserved at 1.0
    assert out.sum() == pytest.approx(1.0)


def test_sector_cap_no_redistribution_when_no_under_sector():
    # When ALL sectors are over the cap, freed weight stays as cash residual
    # (no destination for redistribution). The variant's final renormalize
    # step brings sum back to 1.0; caps.py alone leaves it < 1.0.
    weights = pd.Series([0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15],
                        index=["A", "B", "C", "D", "E", "F", "G", "H"])
    sectors = pd.Series(["Tech"] * 4 + ["Health"] * 4,
                        index=weights.index)
    out = enforce_sector_cap(weights, sectors, sector_cap=0.30,
                              individual_cap=0.20)
    # Both sectors scaled to 30% → total 0.60 (residual is cash)
    assert out.sum() == pytest.approx(0.60)


def test_enforce_caps_full_pipeline():
    weights = pd.Series([1/30] * 30, index=[f"T{i}" for i in range(30)])
    # Force one sector to dominate
    sectors = pd.Series(["Tech"] * 20 + ["Health"] * 10,
                        index=weights.index)
    out = enforce_caps(weights, sectors, individual_cap=0.075,
                       sector_cap=0.30)
    # All individual caps respected
    assert (out <= 0.075 + 1e-9).all()
    # Tech sector at ≤ 30%
    tech_total = out.loc[sectors[sectors == "Tech"].index].sum()
    assert tech_total <= 0.30 + 1e-9
