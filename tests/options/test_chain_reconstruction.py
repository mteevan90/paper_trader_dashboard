"""Tests for ``src/options/chain_reconstruction.py`` (Phase 2 Section 6)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.options.chain_reconstruction import (
    DEFAULT_DELTA_TOLERANCE,
    DEFAULT_WIDTH_PCT,
    get_strike_spacing,
    reconstruct_chain,
    select_strike,
)
from src.options.greeks import price as bsm_price
from src.options.occ import generate_occ_symbol, parse_occ_symbol
from src.options.types import ContractSpec


# ----------------- helpers -----------------


def _single_day_history(close: float, sim_date: date) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [close]},
        index=pd.Index([sim_date], name="date"),
    )


def _fake_fetcher(spec_to_close: dict[ContractSpec, float]):
    """Return a fetcher matching ``fetch_history`` signature."""

    def fetcher(symbol, start, end, *, limiter=None):
        try:
            spec = parse_occ_symbol(symbol)
        except ValueError:
            return pd.DataFrame()
        close = spec_to_close.get(spec)
        if close is None:
            return pd.DataFrame()
        return _single_day_history(close, start)

    return fetcher


# ----------------- get_strike_spacing -----------------


class TestStrikeSpacing:
    def test_get_strike_spacing_spx_is_5(self):
        assert get_strike_spacing("SPX", 4500.0) == 5.0
        assert get_strike_spacing("SPX", 100.0) == 5.0

    def test_get_strike_spacing_low_price_equity_is_1(self):
        assert get_strike_spacing("FORD", 12.0) == 1.0
        assert get_strike_spacing("FORD", 24.99) == 1.0

    def test_get_strike_spacing_mid_price_equity_is_2_50(self):
        assert get_strike_spacing("AAPL", 100.0) == 2.50
        assert get_strike_spacing("AAPL", 25.0) == 2.50
        assert get_strike_spacing("AAPL", 199.99) == 2.50

    def test_get_strike_spacing_high_price_equity_is_5(self):
        assert get_strike_spacing("NVDA", 500.0) == 5.0
        assert get_strike_spacing("NVDA", 200.0) == 5.0

    def test_get_strike_spacing_zero_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            get_strike_spacing("AAPL", 0.0)


# ----------------- reconstruct_chain -----------------


class TestReconstructChain:
    def test_reconstruct_chain_returns_non_empty_only(self):
        sim_date = date(2025, 6, 2)
        target = date(2025, 7, 18)
        spot = 100.0
        # Provide closes only for two specific strikes.
        valid_specs = {
            ContractSpec(
                underlying="AAPL", expiration_date=target,
                option_type="C", strike=100.0,
            ): 2.50,
            ContractSpec(
                underlying="AAPL", expiration_date=target,
                option_type="P", strike=100.0,
            ): 2.30,
        }
        fetcher = _fake_fetcher(valid_specs)
        chain = reconstruct_chain(
            "AAPL", sim_date, target, spot, fetcher=fetcher,
        )
        assert len(chain) == 2
        for spec, close in chain:
            assert spec in valid_specs
            assert close == valid_specs[spec]

    def test_reconstruct_chain_generates_both_calls_and_puts(self):
        sim_date = date(2025, 6, 2)
        target = date(2025, 7, 18)
        spot = 100.0
        # Valid for both C and P at strike 100.
        valid_specs = {
            ContractSpec(
                underlying="AAPL", expiration_date=target,
                option_type="C", strike=100.0,
            ): 2.50,
            ContractSpec(
                underlying="AAPL", expiration_date=target,
                option_type="P", strike=100.0,
            ): 2.30,
        }
        fetcher = _fake_fetcher(valid_specs)
        chain = reconstruct_chain(
            "AAPL", sim_date, target, spot, fetcher=fetcher,
        )
        types = {spec.option_type for spec, _ in chain}
        assert types == {"C", "P"}

    def test_reconstruct_chain_strike_grid_within_width_pct(self):
        """Every strike returned must be within ±width_pct of spot."""
        sim_date = date(2025, 6, 2)
        target = date(2025, 7, 18)
        spot = 100.0
        # Make ALL fetches succeed by providing close=1.0 for any spec.
        recorded_specs: list[ContractSpec] = []

        def fetcher(symbol, start, end, *, limiter=None):
            try:
                spec = parse_occ_symbol(symbol)
            except ValueError:
                return pd.DataFrame()
            recorded_specs.append(spec)
            return _single_day_history(1.0, start)

        reconstruct_chain(
            "AAPL", sim_date, target, spot,
            width_pct=0.20, fetcher=fetcher,
        )
        # All recorded specs are AAPL with target expiration.
        assert all(s.underlying == "AAPL" for s in recorded_specs)
        assert all(s.expiration_date == target for s in recorded_specs)
        # Strikes are within ±20% of spot, on the $2.50 grid.
        strikes = sorted({s.strike for s in recorded_specs})
        spacing = get_strike_spacing("AAPL", spot)
        for k in strikes:
            assert k % spacing == pytest.approx(0.0, abs=1e-6)
        assert min(strikes) <= 80.0
        assert max(strikes) >= 120.0

    def test_reconstruct_chain_target_expiration_must_be_after_sim_date(self):
        with pytest.raises(ValueError, match="target_expiration"):
            reconstruct_chain(
                "AAPL", date(2025, 6, 2), date(2025, 6, 2), 100.0,
                fetcher=lambda *a, **k: pd.DataFrame(),
            )


# ----------------- select_strike -----------------


def _build_call_candidates(
    *,
    underlying: str,
    expiration: date,
    sim_date: date,
    spot: float,
    strikes: list[float],
    vol: float,
    r: float,
    q: float,
) -> list[tuple[ContractSpec, float]]:
    """Build candidates whose close equals BSM price at given vol — so
    select_strike's IV solver recovers ``vol`` exactly."""
    t = (expiration - sim_date).days / 365.0
    out: list[tuple[ContractSpec, float]] = []
    for k in strikes:
        spec = ContractSpec(
            underlying=underlying, expiration_date=expiration,
            option_type="C", strike=k,
        )
        close = bsm_price(spot, k, t, r, q, vol, "C")
        out.append((spec, close))
    return out


def _build_put_candidates(**kwargs) -> list[tuple[ContractSpec, float]]:
    underlying = kwargs.pop("underlying")
    expiration = kwargs.pop("expiration")
    sim_date = kwargs.pop("sim_date")
    spot = kwargs.pop("spot")
    strikes = kwargs.pop("strikes")
    vol = kwargs.pop("vol")
    r = kwargs.pop("r")
    q = kwargs.pop("q")
    t = (expiration - sim_date).days / 365.0
    out: list[tuple[ContractSpec, float]] = []
    for k in strikes:
        spec = ContractSpec(
            underlying=underlying, expiration_date=expiration,
            option_type="P", strike=k,
        )
        close = bsm_price(spot, k, t, r, q, vol, "P")
        out.append((spec, close))
    return out


class TestSelectStrike:
    def test_select_strike_picks_closest_to_target_delta(self):
        sim_date = date(2025, 6, 2)
        expiration = date(2025, 7, 2)  # 30 DTE
        spot = 100.0
        candidates = _build_call_candidates(
            underlying="AAPL",
            expiration=expiration,
            sim_date=sim_date,
            spot=spot,
            strikes=[90.0, 95.0, 100.0, 105.0, 110.0],
            vol=0.30,
            r=0.04,
            q=0.0,
        )
        chosen = select_strike(
            candidates,
            target_delta=0.30,
            option_type="C",
            spot=spot,
            sim_date=sim_date,
            r=0.04,
            q=0.0,
        )
        assert chosen is not None
        # 30-delta call on a 0.30-vol 30-DTE ATM should land in [105, 110].
        assert chosen.strike in {105.0, 110.0}

    def test_select_strike_returns_none_when_no_candidate_within_tolerance(self):
        sim_date = date(2025, 6, 2)
        expiration = date(2025, 7, 2)
        spot = 100.0
        # Only ITM (high-delta) strikes available; target 0.05-delta is far.
        candidates = _build_call_candidates(
            underlying="AAPL",
            expiration=expiration,
            sim_date=sim_date,
            spot=spot,
            strikes=[80.0, 85.0],
            vol=0.30,
            r=0.04,
            q=0.0,
        )
        chosen = select_strike(
            candidates,
            target_delta=0.05,
            option_type="C",
            spot=spot,
            sim_date=sim_date,
            r=0.04,
            q=0.0,
            delta_tolerance=0.10,
        )
        assert chosen is None

    def test_select_strike_csp_uses_negative_delta_internally(self):
        sim_date = date(2025, 6, 2)
        expiration = date(2025, 7, 2)
        spot = 100.0
        # Build put candidates; deltas will be negative.
        candidates = _build_put_candidates(
            underlying="AAPL",
            expiration=expiration,
            sim_date=sim_date,
            spot=spot,
            strikes=[85.0, 90.0, 95.0, 100.0, 105.0],
            vol=0.30,
            r=0.04,
            q=0.0,
        )
        chosen = select_strike(
            candidates,
            target_delta=0.30,  # magnitude
            option_type="P",
            spot=spot,
            sim_date=sim_date,
            r=0.04,
            q=0.0,
        )
        assert chosen is not None
        # 30-delta put with these inputs lands around the 95 strike.
        assert chosen.strike in {90.0, 95.0}

    def test_select_strike_rejects_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            select_strike([], 0.30, "X", 100.0, date(2025, 6, 2), 0.04, 0.0)

    def test_select_strike_rejects_non_positive_target(self):
        with pytest.raises(ValueError, match="target_delta"):
            select_strike([], 0.0, "C", 100.0, date(2025, 6, 2), 0.04, 0.0)

    def test_select_strike_skips_calls_when_option_type_put(self):
        sim_date = date(2025, 6, 2)
        expiration = date(2025, 7, 2)
        spot = 100.0
        # Mixed C and P candidates; only puts should be considered.
        call_candidates = _build_call_candidates(
            underlying="AAPL", expiration=expiration, sim_date=sim_date,
            spot=spot, strikes=[100.0], vol=0.30, r=0.04, q=0.0,
        )
        put_candidates = _build_put_candidates(
            underlying="AAPL", expiration=expiration, sim_date=sim_date,
            spot=spot, strikes=[95.0], vol=0.30, r=0.04, q=0.0,
        )
        chosen = select_strike(
            call_candidates + put_candidates,
            target_delta=0.30,
            option_type="P",
            spot=spot,
            sim_date=sim_date,
            r=0.04,
            q=0.0,
        )
        assert chosen is not None
        assert chosen.option_type == "P"
