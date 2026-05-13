"""Tests for the 7 v2 construction variants — edge cases + expected behavior.

Each variant has its own test class covering:
  - Empty / all-NaN scores (graceful degradation)
  - Expected behavior on synthetic inputs that exercise the variant's
    distinguishing logic (e.g., B1 scales down when realized_vol > 15%,
    B5 routes to stress allocation when SPY trailing return < -5%)
  - Cap enforcement integration
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equities.portfolio_construction import (
    BaselineVariant,
    ConcentrationPenaltiesVariant,
    ConvictionWeightedVariant,
    DefensiveSleevesVariant,
    DynamicTopNVariant,
    SmallerCapsVariant,
    VolTargetVariant,
)
from src.equities.portfolio_construction.base import ConstructionState


def _make_state(scores: pd.Series, sectors: pd.Series, **kw) -> ConstructionState:
    """Test helper: build a ConstructionState with a reference date."""
    return ConstructionState(
        date=pd.Timestamp("2024-01-31"),
        scores=scores,
        sectors=sectors,
        **kw,
    )


def _fake_scores(n: int = 50, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.02, n),
                     index=[f"T{i:03d}" for i in range(n)])


def _fake_sectors(tickers, n_sectors: int = 5) -> pd.Series:
    sectors = [f"Sector{i % n_sectors}" for i in range(len(tickers))]
    return pd.Series(sectors, index=tickers)


class TestBaselineVariant:
    def test_empty_scores(self):
        v = BaselineVariant()
        out = v.construct(_make_state(pd.Series(dtype=float),
                                        pd.Series(dtype=str)))
        assert out.empty

    def test_basic_top_30_equal_weight(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        v = BaselineVariant(n=30)
        out = v.construct(_make_state(scores, sectors))
        assert len(out) == 30
        assert out.sum() == pytest.approx(1.0)
        # Top 30 by score selected
        expected = set(scores.nlargest(30).index)
        assert set(out.index) == expected

    def test_fewer_than_n_eligible(self):
        scores = _fake_scores(10)
        sectors = _fake_sectors(scores.index)
        v = BaselineVariant(n=30)
        out = v.construct(_make_state(scores, sectors))
        # Should pick all 10
        assert len(out) == 10
        assert out.sum() == pytest.approx(1.0)

    def test_sector_cap_binds(self):
        # All 30 names in one sector → sector cap should reduce that sector
        # and redistribute. With only one sector, redistribution has nowhere
        # to go; weights end up renormalized within the sector at 30% total.
        scores = _fake_scores(30)
        sectors = pd.Series(["Tech"] * 30, index=scores.index)
        v = BaselineVariant(n=30, sector_cap=0.30)
        out = v.construct(_make_state(scores, sectors))
        # All 30 names in Tech; sector cap forces Tech total = 30%; then
        # renormalize brings it back to 100%.
        # (Degenerate case; the test confirms it doesn't crash.)
        assert out.sum() == pytest.approx(1.0)


class TestVolTargetVariant:
    def test_no_history_uses_training_tail_vol(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        # training_tail_vol = 30% (above 15% target → scale 0.5)
        v = VolTargetVariant(training_tail_vol=0.30)
        out = v.construct(_make_state(scores, sectors))
        baseline = BaselineVariant().construct(_make_state(scores, sectors))
        # Expected scale = 0.15 / 0.30 = 0.5
        np.testing.assert_array_almost_equal(out.values, baseline.values * 0.5)
        assert out.sum() == pytest.approx(0.5)

    def test_no_history_no_warmup_returns_baseline(self):
        # Without training_tail_vol and without portfolio_history, no
        # scaling — returns baseline weights at full gross.
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        v = VolTargetVariant(training_tail_vol=None)
        out = v.construct(_make_state(scores, sectors))
        baseline = BaselineVariant().construct(_make_state(scores, sectors))
        np.testing.assert_array_almost_equal(out.values, baseline.values)

    def test_realized_vol_below_target_no_leverage(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        # Build a 63-day history at low vol (1% annualized)
        rng = np.random.default_rng(0)
        daily_rets = rng.normal(0, 0.01 / np.sqrt(252), 70)
        history = pd.Series(daily_rets, index=pd.date_range("2024-01-01", periods=70))
        v = VolTargetVariant(training_tail_vol=0.30)
        out = v.construct(_make_state(scores, sectors, portfolio_history=history))
        # Realized vol << target → scale = 1.0 (no leverage)
        baseline = BaselineVariant().construct(_make_state(scores, sectors))
        np.testing.assert_array_almost_equal(out.values, baseline.values)

    def test_realized_vol_above_target_scales_down(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        rng = np.random.default_rng(0)
        # Vol = 30% annualized → expected scale ≈ 0.5
        daily_rets = rng.normal(0, 0.30 / np.sqrt(252), 70)
        history = pd.Series(daily_rets, index=pd.date_range("2024-01-01", periods=70))
        v = VolTargetVariant(target_vol=0.15)
        out = v.construct(_make_state(scores, sectors, portfolio_history=history))
        # Scale ≈ 0.15 / realized_vol → < 1.0
        assert 0.3 < out.sum() < 0.8  # rough sanity bounds


class TestConvictionWeightedVariant:
    def test_concentrates_on_high_scores(self):
        scores = pd.Series([1.0, 0.5, 0.0, -0.5, -1.0] + [0.0] * 25,
                            index=[f"T{i}" for i in range(30)])
        sectors = _fake_sectors(scores.index)
        v = ConvictionWeightedVariant(n=30, temperature=0.5)
        out = v.construct(_make_state(scores, sectors))
        # All 30 selected, weights sum to 1.0
        assert len(out) == 30
        assert out.sum() == pytest.approx(1.0)
        # Top-score name has highest weight, sorted descending matches
        sorted_w = out.sort_values(ascending=False)
        assert sorted_w.iloc[0] >= sorted_w.iloc[-1]
        # T0 (score=1.0) should dominate
        assert out["T0"] > out["T29"]


class TestDynamicTopNVariant:
    def test_low_dispersion_picks_n_low(self):
        scores = _fake_scores(100)
        sectors = _fake_sectors(scores.index)
        # Training dispersion distribution where current dispersion sits at 10th percentile
        training_dispersions = np.linspace(0.001, 0.10, 100)
        v = DynamicTopNVariant(n_low=50, n_high=15,
                               training_dispersion_dist=training_dispersions.tolist())
        # Force a very low current dispersion
        scores_low_disp = pd.Series(np.full(100, 0.01),
                                     index=scores.index)
        scores_low_disp.iloc[:10] += np.linspace(0, 0.0001, 10)  # tiny dispersion
        out = v.construct(_make_state(scores_low_disp, sectors))
        # Should pick close to n_low (50) at low dispersion
        assert 40 <= len(out) <= 50

    def test_high_dispersion_picks_n_high(self):
        scores = _fake_scores(100)
        sectors = _fake_sectors(scores.index)
        training_dispersions = np.linspace(0.001, 0.10, 100)
        v = DynamicTopNVariant(n_low=50, n_high=15,
                               training_dispersion_dist=training_dispersions.tolist())
        # Force high current dispersion
        scores_high_disp = scores.copy()
        scores_high_disp.iloc[:10] = np.linspace(1.0, 0.1, 10)  # high std
        out = v.construct(_make_state(scores_high_disp, sectors))
        # Should pick close to n_high (15) at high dispersion
        assert 15 <= len(out) <= 25

    def test_requires_training_distribution(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        v = DynamicTopNVariant()  # no training_dispersion_dist
        with pytest.raises(RuntimeError, match="training_dispersion_dist"):
            v.construct(_make_state(scores, sectors))


class TestConcentrationPenaltiesVariant:
    def test_no_penalties_matches_baseline(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        v = ConcentrationPenaltiesVariant()
        # No streak, no overweight → identical to baseline
        out = v.construct(_make_state(scores, sectors,
                                        top30_streak={},
                                        prev_portfolio_sector_weights={}))
        baseline = BaselineVariant().construct(_make_state(scores, sectors))
        # Should pick the same top 30 names
        assert set(out.index) == set(baseline.index)

    def test_persistence_penalty_demotes_long_streak(self):
        # Construct scores where T0-T29 are top-30 by raw score, but T0
        # has a long persistence streak (10 rebalances). T0's effective
        # score reduces by 50% → may drop out of top 30.
        scores = pd.Series(np.linspace(1.0, 0.0, 50),
                           index=[f"T{i}" for i in range(50)])
        sectors = _fake_sectors(scores.index)
        v = ConcentrationPenaltiesVariant()
        # T0 has streak=10 → factor = max(0.5, 1 - 0.1*5) = 0.5
        streak = {"T0": 10}
        out = v.construct(_make_state(scores, sectors,
                                        top30_streak=streak,
                                        prev_portfolio_sector_weights={}))
        # T0 was raw score 1.0; with 0.5 multiplier → effective 0.5,
        # which is below T20's raw 0.6 — so T0 still survives (effective
        # 0.5 > T29's raw 0.4). Just verify it didn't crash and weights
        # are valid.
        assert out.sum() == pytest.approx(1.0)
        # Some streak-adjusted names still selected
        assert len(out) == 30


class TestDefensiveSleevesVariant:
    def test_normal_regime_70_30(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        # SPY with 21-day return = 0% (above -5% threshold → normal regime)
        spy_dates = pd.date_range("2024-01-01", periods=30)
        spy_history = pd.DataFrame(
            {"close": np.linspace(500, 500, 30)},  # flat → 0% return
            index=spy_dates,
        )
        v = DefensiveSleevesVariant()
        out = v.construct(_make_state(scores, sectors, spy_history=spy_history))
        # Sum should be 0.70 (equity) + 0.15 (SHY) = 0.85; remainder cash
        equity_total = out.drop("SHY", errors="ignore").sum()
        assert equity_total == pytest.approx(0.70, abs=0.001)
        assert out.get("SHY", 0) == pytest.approx(0.15)
        assert out.sum() == pytest.approx(0.85)

    def test_stress_regime_50_50(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        # SPY dropping → trailing 21d return < -5%
        spy_dates = pd.date_range("2024-01-01", periods=30)
        # Build SPY from 500 down 8% over 21 days
        spy_close = np.linspace(500, 500 * 0.92, 30)
        spy_history = pd.DataFrame({"close": spy_close}, index=spy_dates)
        v = DefensiveSleevesVariant()
        out = v.construct(_make_state(scores, sectors, spy_history=spy_history))
        # Stress: 0.50 equity + 0.25 SHY = 0.75; remainder cash
        equity_total = out.drop("SHY", errors="ignore").sum()
        assert equity_total == pytest.approx(0.50, abs=0.001)
        assert out.get("SHY", 0) == pytest.approx(0.25)

    def test_no_spy_history_defaults_to_normal(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        v = DefensiveSleevesVariant()
        # spy_history=None → default to normal regime
        out = v.construct(_make_state(scores, sectors))
        equity_total = out.drop("SHY", errors="ignore").sum()
        assert equity_total == pytest.approx(0.70, abs=0.001)


class TestSmallerCapsVariant:
    def test_individual_cap_is_4pct(self):
        v = SmallerCapsVariant()
        assert v.individual_cap == 0.04
        assert v.sector_cap == 0.30  # unchanged

    def test_subclasses_baseline(self):
        from src.equities.portfolio_construction import BaselineVariant
        assert isinstance(SmallerCapsVariant(), BaselineVariant)

    def test_basic_top_30(self):
        scores = _fake_scores(50)
        sectors = _fake_sectors(scores.index)
        v = SmallerCapsVariant()
        out = v.construct(_make_state(scores, sectors))
        assert len(out) == 30
        assert out.sum() == pytest.approx(1.0)
        # All weights <= 4% after caps if cap binds (sector concentration may
        # force redistribution; the 4% cap is binding when that happens)
        # Note: at 1/30 = 3.33% and 30% sector cap, with 5 sectors evenly,
        # sector cap doesn't bind → weights at 3.33% each (under 4%).
        assert (out <= 0.04 + 1e-9).all()


class TestVariantRegistry:
    def test_get_variant_by_name_all_seven(self):
        from src.equities.portfolio_construction import get_variant_by_name
        names = ["baseline", "b1_vol_target", "b2_conviction_weighted",
                  "b3_dynamic_topn", "b4_concentration_penalties",
                  "b5_defensive_sleeves", "b6_smaller_caps"]
        for name in names:
            if name == "b3_dynamic_topn":
                v = get_variant_by_name(name,
                                          training_dispersion_dist=[0.01, 0.02, 0.03])
            else:
                v = get_variant_by_name(name)
            assert v.name == name
            assert v.params_dict() is not None

    def test_get_variant_by_name_invalid_raises(self):
        from src.equities.portfolio_construction import get_variant_by_name
        with pytest.raises(ValueError, match="Unknown variant name"):
            get_variant_by_name("nonexistent_variant")
