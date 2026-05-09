"""Type definitions for the options universe and contract layer.

Defines ``UnderlyingMeta`` (per-underlying metadata carried through
universe construction and backtest setup), ``ContractSpec`` (the
identifying tuple for a specific option contract), and
``UNIVERSE_PARQUET_SCHEMA`` (the on-disk contract that the v1.1+
liquidity-filter fetcher will write against).
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pyarrow as pa


_VALID_ASSET_TYPES = ("index", "equity")
_VALID_EXERCISE_STYLES = ("european", "american")
_VALID_SETTLEMENT_TYPES = ("AM", "PM")
_VALID_OPTION_TYPES = ("C", "P")


@dataclass(frozen=True, slots=True)
class UnderlyingMeta:
    ticker: str
    name: str
    asset_type: str
    exercise_style: str
    settlement_type: str
    multiplier: int
    listing_date: date
    delisting_date: Optional[date]
    sectors: tuple[str, ...]
    has_weeklies: bool
    dividend_paying: bool
    data_provider_id: str

    def __post_init__(self) -> None:
        if self.asset_type not in _VALID_ASSET_TYPES:
            raise ValueError(
                f"asset_type must be one of {_VALID_ASSET_TYPES}; "
                f"got {self.asset_type!r}")
        if self.exercise_style not in _VALID_EXERCISE_STYLES:
            raise ValueError(
                f"exercise_style must be one of {_VALID_EXERCISE_STYLES}; "
                f"got {self.exercise_style!r}")
        if self.settlement_type not in _VALID_SETTLEMENT_TYPES:
            raise ValueError(
                f"settlement_type must be one of {_VALID_SETTLEMENT_TYPES}; "
                f"got {self.settlement_type!r}")


@dataclass(frozen=True, slots=True)
class ContractSpec:
    underlying: str
    expiration_date: date
    option_type: str
    strike: float

    def __post_init__(self) -> None:
        if self.option_type not in _VALID_OPTION_TYPES:
            raise ValueError(
                f"option_type must be one of {_VALID_OPTION_TYPES}; "
                f"got {self.option_type!r}")
        if self.strike <= 0:
            raise ValueError(f"strike must be > 0; got {self.strike!r}")


# Daily underlying-universe-history table. Section 1 stub does not
# write to this; the v1.1+ liquidity-filter fetcher does.
# ``options_adv_30d`` and ``chain_spread_pct_30d`` are the filter
# inputs and are nullable so the schema accommodates rows written
# before the fetcher computes them.
UNIVERSE_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("asset_type", pa.string(), nullable=False),
        pa.field("exercise_style", pa.string(), nullable=False),
        pa.field("settlement_type", pa.string(), nullable=False),
        pa.field("multiplier", pa.int32(), nullable=False),
        pa.field("has_weeklies", pa.bool_(), nullable=False),
        pa.field("dividend_paying", pa.bool_(), nullable=False),
        pa.field("sectors", pa.list_(pa.string()), nullable=False),
        pa.field("listing_date", pa.date32(), nullable=False),
        pa.field("delisting_date", pa.date32(), nullable=True),
        pa.field("data_provider_id", pa.string(), nullable=False),
        pa.field("options_adv_30d", pa.float64(), nullable=True),
        pa.field("chain_spread_pct_30d", pa.float64(), nullable=True),
    ]
)
