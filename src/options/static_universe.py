"""Smoke-test universe stub for the options module.

Hard-codes 8 underlyings (3 indexes + 5 curated equities) so Sections
1-7 of the options Phase 2 build can run against a stable, deterministic
shape. The v1.1+ liquidity-filter fetcher will replace this with a
point-in-time list derived from Mike's equity universe filtered by
options-grade liquidity, plus the 3 index options unconditionally; that
output writes to ``UNIVERSE_PARQUET_SCHEMA``.

Listing dates marked ``# TODO verify`` are best-known approximations and
should be confirmed against an authoritative source (Tradier reference
data or OCC historical listing records) before any production use.
"""

from datetime import date

from src.options.types import UnderlyingMeta


STATIC_UNIVERSE: tuple[UnderlyingMeta, ...] = (
    # ----- Indexes (3) -----
    UnderlyingMeta(
        ticker="SPX",
        name="S&P 500 Index",
        asset_type="index",
        exercise_style="european",
        settlement_type="AM",
        multiplier=100,
        listing_date=date(1983, 7, 1),  # TODO verify (CBOE SPX options launched July 1983)
        delisting_date=None,
        sectors=("index",),
        has_weeklies=True,
        dividend_paying=False,
        data_provider_id="SPX",
    ),
    UnderlyingMeta(
        # SPY is an ETF and pays quarterly distributions. dividend_paying=True
        # here is a coarse flag — ETF distributions are NOT single-stock
        # ex-dividend events for assignment-risk purposes (no individual-name
        # ex-div math, different mechanics). Section 4 (position lifecycle)
        # refines the assignment-risk handling for ETF underlyings.
        ticker="SPY",
        name="SPDR S&P 500 ETF Trust",
        asset_type="equity",
        exercise_style="american",
        settlement_type="PM",
        multiplier=100,
        listing_date=date(1993, 1, 29),  # TODO verify (SPY listed Jan 1993; options soon after)
        delisting_date=None,
        sectors=("index", "etf"),
        has_weeklies=True,
        dividend_paying=True,
        data_provider_id="SPY",
    ),
    UnderlyingMeta(
        # See SPY note above — same coarse-flag caveat for QQQ ETF
        # distributions; refined in Section 4.
        ticker="QQQ",
        name="Invesco QQQ Trust",
        asset_type="equity",
        exercise_style="american",
        settlement_type="PM",
        multiplier=100,
        listing_date=date(1999, 3, 10),  # TODO verify (QQQ listed Mar 1999; options soon after)
        delisting_date=None,
        sectors=("index", "etf", "tech-heavy"),
        has_weeklies=True,
        dividend_paying=True,
        data_provider_id="QQQ",
    ),
    # ----- Equities (5, alphabetical) -----
    UnderlyingMeta(
        ticker="AAPL",
        name="Apple Inc.",
        asset_type="equity",
        exercise_style="american",
        settlement_type="PM",
        multiplier=100,
        listing_date=date(1990, 1, 1),  # TODO verify (AAPL options began trading early 1990s)
        delisting_date=None,
        sectors=("technology", "consumer-electronics"),
        has_weeklies=True,
        dividend_paying=True,
        data_provider_id="AAPL",
    ),
    UnderlyingMeta(
        ticker="JPM",
        name="JPMorgan Chase & Co.",
        asset_type="equity",
        exercise_style="american",
        settlement_type="PM",
        multiplier=100,
        listing_date=date(2001, 1, 1),  # TODO verify (post-2000 J.P. Morgan Chase merger)
        delisting_date=None,
        sectors=("financials", "banks"),
        has_weeklies=True,
        dividend_paying=True,
        data_provider_id="JPM",
    ),
    UnderlyingMeta(
        ticker="MSFT",
        name="Microsoft Corporation",
        asset_type="equity",
        exercise_style="american",
        settlement_type="PM",
        multiplier=100,
        listing_date=date(1986, 6, 1),  # TODO verify (MSFT IPO Mar 1986; options soon after)
        delisting_date=None,
        sectors=("technology", "software"),
        has_weeklies=True,
        dividend_paying=True,
        data_provider_id="MSFT",
    ),
    UnderlyingMeta(
        ticker="NVDA",
        name="NVIDIA Corporation",
        asset_type="equity",
        exercise_style="american",
        settlement_type="PM",
        multiplier=100,
        listing_date=date(1999, 6, 1),  # TODO verify (NVDA IPO Jan 1999; options soon after)
        delisting_date=None,
        sectors=("technology", "semiconductors"),
        has_weeklies=True,
        dividend_paying=True,
        data_provider_id="NVDA",
    ),
    UnderlyingMeta(
        ticker="XOM",
        name="Exxon Mobil Corporation",
        asset_type="equity",
        exercise_style="american",
        settlement_type="PM",
        multiplier=100,
        listing_date=date(1999, 12, 1),  # TODO verify (post-1999 Exxon-Mobil merger)
        delisting_date=None,
        sectors=("energy", "oil-gas"),
        has_weeklies=True,
        dividend_paying=True,
        data_provider_id="XOM",
    ),
)
