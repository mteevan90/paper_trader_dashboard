"""Tests for the crypto universe stub (Phase 2 Section 1).

No network. No fixtures from outside this file. Validates the static
universe shape and the locked public API.
"""

from datetime import date

import pytest

from src.crypto.static_universe import STATIC_UNIVERSE
from src.crypto.types import TokenMeta
from src.crypto.universe import (
    get_token_metadata,
    get_universe_at_date,
    get_universe_at_date_v2,
    is_token_active,
)


def test_get_universe_at_date_returns_ten_by_default():
    universe = get_universe_at_date(date(2024, 1, 1), top_n=10)
    assert len(universe) == 10


def test_get_universe_at_date_top_n_three_is_deterministic():
    first = get_universe_at_date(date(2024, 1, 1), top_n=3)
    second = get_universe_at_date(date(2024, 1, 1), top_n=3)
    assert len(first) == 3
    assert first == second
    assert [t.coingecko_id for t in first] == [
        STATIC_UNIVERSE[0].coingecko_id,
        STATIC_UNIVERSE[1].coingecko_id,
        STATIC_UNIVERSE[2].coingecko_id,
    ]


def test_is_token_active_bitcoin_2024_true():
    assert is_token_active("bitcoin", date(2024, 1, 1)) is True


def test_is_token_active_bitcoin_2008_false():
    assert is_token_active("bitcoin", date(2008, 1, 1)) is False


def test_is_token_active_unknown_token_false():
    assert is_token_active("nonexistent-token", date(2024, 1, 1)) is False


def test_get_token_metadata_bitcoin():
    meta = get_token_metadata("bitcoin")
    assert isinstance(meta, TokenMeta)
    assert meta.symbol == "BTC"


def test_get_token_metadata_unknown_raises():
    with pytest.raises(KeyError):
        get_token_metadata("nonexistent-token")


def test_get_universe_at_date_v2_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        get_universe_at_date_v2(date(2024, 1, 1), 10)


def test_every_token_has_binance_and_coinbase_pairs():
    for token in STATIC_UNIVERSE:
        assert "binance" in token.ccxt_symbols, token.coingecko_id
        assert "coinbase" in token.ccxt_symbols, token.coingecko_id
