"""Tests for the options universe stub (Phase 2 Section 1).

No network. No fixtures from outside this file. Validates the static
universe shape and the locked public API.
"""

from datetime import date

import pytest

from src.options.static_universe import STATIC_UNIVERSE
from src.options.types import UnderlyingMeta
from src.options.universe import (
    get_underlying_metadata,
    get_universe_at_date,
    get_universe_at_date_v2,
    is_underlying_active,
    is_underlying_active_v2,
)


def test_get_universe_at_date_returns_eight_at_top_n_eight():
    universe = get_universe_at_date(date(2024, 1, 1), top_n=8)
    assert len(universe) == 8


def test_get_universe_at_date_top_n_three_is_indexes():
    universe = get_universe_at_date(date(2024, 1, 1), top_n=3)
    assert [u.ticker for u in universe] == ["SPX", "SPY", "QQQ"]


def test_get_universe_at_date_deterministic():
    first = get_universe_at_date(date(2024, 1, 1), top_n=8)
    second = get_universe_at_date(date(2024, 1, 1), top_n=8)
    assert first == second


def test_is_underlying_active_aapl_2024_true():
    assert is_underlying_active("AAPL", date(2024, 1, 1)) is True


def test_is_underlying_active_aapl_1985_false():
    assert is_underlying_active("AAPL", date(1985, 1, 1)) is False


def test_is_underlying_active_unknown_ticker_false():
    assert is_underlying_active("NOTREAL", date(2024, 1, 1)) is False


def test_get_underlying_metadata_spx():
    meta = get_underlying_metadata("SPX")
    assert isinstance(meta, UnderlyingMeta)
    assert meta.asset_type == "index"
    assert meta.exercise_style == "european"


def test_get_underlying_metadata_unknown_raises():
    with pytest.raises(KeyError):
        get_underlying_metadata("NOTREAL")


def test_get_universe_at_date_v2_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        get_universe_at_date_v2(date(2024, 1, 1), 8)


def test_is_underlying_active_v2_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        is_underlying_active_v2("AAPL", date(2024, 1, 1))


def test_every_underlying_has_multiplier_100():
    for u in STATIC_UNIVERSE:
        assert u.multiplier == 100, u.ticker


def test_indexes_are_european_equities_are_american():
    for u in STATIC_UNIVERSE:
        if u.asset_type == "index":
            assert u.exercise_style == "european", u.ticker
        else:
            assert u.exercise_style == "american", u.ticker


def test_only_spx_is_am_settled():
    am = [u.ticker for u in STATIC_UNIVERSE if u.settlement_type == "AM"]
    pm = [u.ticker for u in STATIC_UNIVERSE if u.settlement_type == "PM"]
    assert am == ["SPX"]
    assert set(pm) == {"SPY", "QQQ", "AAPL", "JPM", "MSFT", "NVDA", "XOM"}
