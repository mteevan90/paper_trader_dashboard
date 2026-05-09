"""Smoke-universe stub for the crypto extension.

This module hard-codes a 10-token universe so Sections 1, 2, 4, and 5 of
the crypto extension can build against a stable, deterministic shape
without waiting on the multi-day CoinGecko historical fetch. Section 3
will replace this with a survivorship-correct, point-in-time universe
sourced from CoinGecko, written to parquet using
``UNIVERSE_PARQUET_SCHEMA``.

CCXT pair strings and CoinGecko IDs reflect best-known values at the
time of writing; entries marked ``# TODO verify`` should be confirmed
against live API responses in Section 2 before any production use.
"""

from datetime import date
from types import MappingProxyType

from src.crypto.types import TokenMeta


def _binance_coinbase(symbol: str) -> MappingProxyType:
    return MappingProxyType(
        {
            "binance": f"{symbol}/USDT",
            "coinbase": f"{symbol}/USD",
        }
    )


STATIC_UNIVERSE: tuple[TokenMeta, ...] = (
    TokenMeta(
        coingecko_id="bitcoin",
        symbol="BTC",
        name="Bitcoin",
        listing_date=date(2009, 1, 3),
        delisting_date=None,
        categories=("L1", "store-of-value"),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("BTC"),
    ),
    TokenMeta(
        coingecko_id="ethereum",
        symbol="ETH",
        name="Ethereum",
        listing_date=date(2015, 7, 30),
        delisting_date=None,
        categories=("L1", "smart-contract-platform"),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("ETH"),
    ),
    TokenMeta(
        coingecko_id="solana",
        symbol="SOL",
        name="Solana",
        listing_date=date(2020, 3, 16),
        delisting_date=None,
        categories=("L1", "smart-contract-platform"),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("SOL"),
    ),
    TokenMeta(
        coingecko_id="ripple",
        symbol="XRP",
        name="XRP",
        listing_date=date(2012, 1, 1),  # TODO verify (XRPL launched 2012; exact date varies by source)
        delisting_date=None,
        categories=("payment",),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("XRP"),
    ),
    TokenMeta(
        coingecko_id="cardano",
        symbol="ADA",
        name="Cardano",
        listing_date=date(2017, 9, 29),
        delisting_date=None,
        categories=("L1", "smart-contract-platform"),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("ADA"),
    ),
    TokenMeta(
        coingecko_id="avalanche-2",
        symbol="AVAX",
        name="Avalanche",
        listing_date=date(2020, 9, 21),
        delisting_date=None,
        categories=("L1", "smart-contract-platform"),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("AVAX"),
    ),
    TokenMeta(
        coingecko_id="polkadot",
        symbol="DOT",
        name="Polkadot",
        listing_date=date(2020, 8, 18),
        delisting_date=None,
        categories=("L1", "interoperability"),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("DOT"),
    ),
    TokenMeta(
        coingecko_id="chainlink",
        symbol="LINK",
        name="Chainlink",
        listing_date=date(2017, 9, 19),
        delisting_date=None,
        categories=("oracle", "defi"),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("LINK"),
    ),
    # Rebranded to POL in 2024; Section 3's survivorship layer will need
    # to handle the matic-network -> polygon-ecosystem-token transition
    # and the MATIC -> POL ticker change. For the stub we keep the legacy
    # CoinGecko ID and ticker.
    TokenMeta(
        coingecko_id="matic-network",
        symbol="MATIC",
        name="Polygon",
        listing_date=date(2019, 4, 29),
        delisting_date=None,
        categories=("L2", "scaling"),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("MATIC"),  # TODO verify (binance/coinbase relisted as POL/USDT, POL/USD post-2024 rebrand)
    ),
    TokenMeta(
        coingecko_id="litecoin",
        symbol="LTC",
        name="Litecoin",
        listing_date=date(2011, 10, 13),
        delisting_date=None,
        categories=("payment",),
        exchanges=("binance", "coinbase"),
        ccxt_symbols=_binance_coinbase("LTC"),
    ),
)
