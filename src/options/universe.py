"""Public API for the options universe module.

The v1 functions are backed by ``STATIC_UNIVERSE`` and are real for
Section 1+ smoke work. The v2 sibling functions (``..._v2``) reserve
the seam for v1.1+ filter-based universe construction against Mike's
equity universe; their signatures are locked here so Sections 2-7 can
build against them. The v2 implementations land alongside the
liquidity-filter fetcher in v1.1+.
"""

from datetime import date

from src.options.static_universe import STATIC_UNIVERSE
from src.options.types import UnderlyingMeta


_BY_TICKER: dict[str, UnderlyingMeta] = {u.ticker: u for u in STATIC_UNIVERSE}


def get_universe_at_date(at_date: date, top_n: int) -> list[UnderlyingMeta]:
    """Return the first ``top_n`` underlyings active on ``at_date``.

    STUB: Returns from ``STATIC_UNIVERSE`` truncated to ``top_n``.

    Ordering in v1 is **static-tuple order** (indexes first, then
    equities alphabetical) — *not* liquidity-ranked. The v2 sibling
    (``get_universe_at_date_v2``) ranks by 30-day options ADV when
    implemented in v1.1+.
    """
    active = [u for u in STATIC_UNIVERSE
              if is_underlying_active(u.ticker, at_date)]
    return active[:top_n]


def is_underlying_active(ticker: str, at_date: date) -> bool:
    """True if the underlying was listed and not yet delisted on ``at_date``."""
    meta = _BY_TICKER.get(ticker)
    if meta is None:
        return False
    if at_date < meta.listing_date:
        return False
    if meta.delisting_date is not None and at_date >= meta.delisting_date:
        return False
    return True


def get_underlying_metadata(ticker: str) -> UnderlyingMeta:
    """Lookup by ticker. Raises KeyError if not in the static universe."""
    return _BY_TICKER[ticker]


def get_universe_at_date_v2(at_date: date, top_n: int) -> list[UnderlyingMeta]:
    """Liquidity-filtered universe backed by point-in-time chain history.

    Will be implemented in v1.1+ alongside the options-grade liquidity
    fetcher. Reads from ``models/cache/options/universe_history.parquet``
    (schema: ``UNIVERSE_PARQUET_SCHEMA``) and returns the indexes
    unconditionally plus the subset of Mike's equity universe whose
    ``options_adv_30d`` and ``chain_spread_pct_30d`` clear the filter
    thresholds at ``at_date``, ranked by options ADV.
    """
    raise NotImplementedError(
        "Liquidity-filtered universe construction lands in v1.1+ "
        "alongside the options-grade liquidity fetcher. Until it ships, "
        "callers should use get_universe_at_date() with the static stub "
        "universe and treat any alpha numbers as plumbing-verification "
        "only."
    )


def is_underlying_active_v2(ticker: str, at_date: date) -> bool:
    """Parquet-backed activeness check, sibling of ``get_universe_at_date_v2``.

    Will be implemented in v1.1+. Reads from
    ``models/cache/options/universe_history.parquet`` and returns True
    iff ``ticker`` was present in the filtered universe on ``at_date``.
    """
    raise NotImplementedError(
        "Liquidity-filtered active-check lands in v1.1+ alongside "
        "get_universe_at_date_v2. Until it ships, callers should use "
        "is_underlying_active() against the static stub."
    )
