"""Type definitions for the crypto universe module.

Defines TokenMeta (the per-token metadata record carried through universe
construction and backtest setup) and UNIVERSE_PARQUET_SCHEMA (the on-disk
contract that Section 3's CoinGecko survivorship fetcher will write
against).
"""

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Mapping, Optional

import pyarrow as pa


@dataclass(frozen=True, slots=True)
class TokenMeta:
    coingecko_id: str
    symbol: str
    name: str
    listing_date: date
    delisting_date: Optional[date]
    categories: tuple[str, ...]
    exchanges: tuple[str, ...]
    ccxt_symbols: Mapping[str, str]


def _freeze_ccxt_symbols(pairs: dict[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(pairs))


UNIVERSE_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("coingecko_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("market_cap_rank", pa.int32(), nullable=True),
        pa.field("market_cap_usd", pa.float64(), nullable=True),
        pa.field("price_usd_close", pa.float64(), nullable=True),
        pa.field("listing_date", pa.date32(), nullable=False),
        pa.field("delisting_date", pa.date32(), nullable=True),
        pa.field("categories", pa.list_(pa.string()), nullable=False),
    ]
)
