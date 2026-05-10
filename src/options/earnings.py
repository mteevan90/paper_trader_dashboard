"""Tradier corporate-calendar fetcher with parquet caching (Phase 2 Section 6).

Public functions:
    fetch_earnings_calendar  -- earnings dates for a ticker, cached
                                7 days at
                                ``models/cache/options/tradier/earnings/``.
    is_in_earnings_window    -- predicate: is sim_date within ±N days
                                of any earnings date for ticker.

Indexes (SPX, SPY, QQQ) have no earnings — they return empty tuples
without hitting the network.

Source priority (post-fix):

1. **Tradier fundamentals beta** at
   ``/beta/markets/fundamentals/calendars`` — the documented endpoint
   for corporate calendar events. Requires the Fundamentals product
   subscription on the Tradier account; tokens without it get a 401
   ``Invalid API call as no apiproduct match found`` from Apigee.
2. **yfinance** ``Ticker.earnings_dates`` — used as a fallback when
   Tradier returns an empty payload, raises, or yields no parseable
   dates. Lets earnings avoidance keep working when the Tradier
   subscription isn't available.

The original ``/v1/markets/calendars/corporate`` URL hardcoded in
Section 6 was wrong — that path 404s on Tradier; the real endpoint
lives under ``/beta/markets/fundamentals/...``. Fixed in this PR.

The parser is defensive: walks the payload looking for date strings
under common envelope shapes (``calendars/calendar/events``,
``results[].tables.corporate_calendars[]``, plain ``events`` lists)
and accepts both string event types ("Earnings") and integer event
codes (Tradier's fundamentals API documents specific codes for
earnings releases). When the payload doesn't match any of those, the
call falls through to yfinance.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import requests

from src.options.tradier import (
    RateLimiter,
    _http_get,
    _resolve_beta_base_url,
)


__all__ = [
    "fetch_earnings_calendar",
    "is_in_earnings_window",
    "INDEX_TICKERS",
    "EARNINGS_CACHE_TTL_HOURS",
    "EARNINGS_CACHE_DIR",
    "FUNDAMENTALS_CALENDAR_PATH",
    "EARNINGS_EVENT_TYPE_CODES",
]


logger = logging.getLogger(__name__)


INDEX_TICKERS: frozenset[str] = frozenset({"SPX", "SPY", "QQQ"})

EARNINGS_CACHE_DIR: Path = (
    Path("models") / "cache" / "options" / "tradier" / "earnings"
)
EARNINGS_CACHE_TTL_HOURS: int = 7 * 24

# Path on the Tradier beta base URL (https://<host>/beta).
# Section 6 originally used /markets/calendars/corporate on /v1, which
# 404s — the actual path is /markets/fundamentals/calendars on /beta.
FUNDAMENTALS_CALENDAR_PATH: str = "/markets/fundamentals/calendars"

# Tradier fundamentals event_type codes that represent an earnings
# release (per the documented schema; treat as a probable-set rather
# than a hard contract since the beta API's docs may shift).
EARNINGS_EVENT_TYPE_CODES: frozenset[int] = frozenset({14, 15})


HttpGet = Callable[..., dict]


def _cache_path(ticker: str) -> Path:
    return EARNINGS_CACHE_DIR / f"{ticker}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < EARNINGS_CACHE_TTL_HOURS


def _read_earnings_cache(ticker: str) -> Optional[tuple[date, ...]]:
    path = _cache_path(ticker)
    if not _is_fresh(path):
        return None
    df = pd.read_parquet(path)
    if df.empty or "earnings_date" not in df.columns:
        return ()
    return tuple(
        d if isinstance(d, date) else d.date()
        for d in df["earnings_date"].tolist()
    )


def _write_earnings_cache(
    ticker: str, dates: tuple[date, ...]
) -> Path:
    path = _cache_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"earnings_date": list(dates)})
    df.to_parquet(path)
    return path


def _coerce_iso_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _is_earnings_event_type(value) -> bool:
    """Match against the various event_type representations the Tradier
    fundamentals API uses — string ("Earnings", "Earnings Release") or
    integer code (14/15)."""
    if isinstance(value, str):
        return "earnings" in value.lower()
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value in EARNINGS_EVENT_TYPE_CODES
    return False


def _walk_for_earnings(payload) -> Iterable[date]:
    """Recursively yield earnings dates found under common envelope shapes.

    Handles the v1 ``calendars/calendar/events/event`` shape (legacy /
    pre-fundamentals) AND the fundamentals beta shape under
    ``results[].tables.corporate_calendars[]``. Walks the structure and
    yields any ``date``-shaped value whose neighbor key suggests an
    earnings event.
    """
    if isinstance(payload, dict):
        type_value = (
            payload.get("event_type")
            or payload.get("type")
            or payload.get("eventType")
            or ""
        )
        if _is_earnings_event_type(type_value):
            for key in (
                "date",
                "begin_date_time",
                "begin_date",
                "event_date",
                "report_date",
                "estimated_date_for_next_event",
            ):
                d = _coerce_iso_date(payload.get(key))
                if d is not None:
                    yield d
                    break
        for value in payload.values():
            yield from _walk_for_earnings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk_for_earnings(item)


def _fetch_earnings_from_tradier(
    ticker: str,
    *,
    fetcher: Optional[HttpGet] = None,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
) -> tuple[date, ...]:
    """Hit Tradier fundamentals beta. Returns sorted tuple of dates,
    or ``()`` on fetch error / parse failure (auth issues like 401
    'Invalid API call as no apiproduct match found' included)."""
    fetcher = fetcher or _http_get
    limiter = limiter or RateLimiter()
    try:
        payload = fetcher(
            FUNDAMENTALS_CALENDAR_PATH,
            {"symbols": ticker},
            limiter,
            session=session,
            base_url_override=_resolve_beta_base_url(),
        )
    except Exception as exc:
        logger.warning(
            "tradier earnings fetch failed for %s: %s "
            "(will try yfinance fallback)",
            ticker, exc,
        )
        return ()

    raw_dates = sorted(set(_walk_for_earnings(payload)))
    return tuple(raw_dates)


YFinanceFetcher = Callable[[str], "pd.DataFrame | None"]


def _yfinance_default_fetcher(ticker: str) -> "pd.DataFrame | None":
    """Default yfinance fetcher: ``Ticker.earnings_dates`` returns a
    DataFrame indexed by datetime with up to ~12 dates of historical +
    upcoming earnings. Tests inject a mock instead."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed; can't run earnings fallback")
        return None
    try:
        return yf.Ticker(ticker).earnings_dates
    except Exception as exc:
        logger.warning(
            "yfinance earnings fetch failed for %s: %s", ticker, exc
        )
        return None


def _fetch_earnings_from_yfinance(
    ticker: str,
    *,
    yfinance_fetcher: Optional[YFinanceFetcher] = None,
) -> tuple[date, ...]:
    """yfinance fallback. Returns sorted tuple of dates extracted from
    ``Ticker.earnings_dates``'s DataFrame index."""
    fetcher = yfinance_fetcher or _yfinance_default_fetcher
    df = fetcher(ticker)
    if df is None or len(df) == 0:
        return ()
    dates: list[date] = []
    for ts in df.index:
        if hasattr(ts, "date"):
            try:
                dates.append(ts.date())
            except Exception:
                continue
        elif isinstance(ts, date):
            dates.append(ts)
    return tuple(sorted(set(dates)))


def fetch_earnings_calendar(
    ticker: str,
    *,
    fetcher: Optional[HttpGet] = None,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
    yfinance_fetcher: Optional[YFinanceFetcher] = None,
) -> tuple[date, ...]:
    """Fetch all known earnings dates for ``ticker`` and return a sorted
    tuple.

    Indexes (SPX/SPY/QQQ) return ``()`` immediately without I/O.

    On cache hit (within :data:`EARNINGS_CACHE_TTL_HOURS`) returns the
    cached tuple. On miss, tries Tradier fundamentals beta first; if
    that returns empty (auth not subscribed, parse miss, or genuine
    no-data), falls through to yfinance ``Ticker.earnings_dates``.
    """
    if ticker in INDEX_TICKERS:
        return ()

    if use_cache:
        cached = _read_earnings_cache(ticker)
        if cached is not None:
            return cached

    dates = _fetch_earnings_from_tradier(
        ticker,
        fetcher=fetcher,
        limiter=limiter,
        session=session,
    )
    source = "tradier"

    if not dates:
        logger.info(
            "tradier returned no earnings dates for %s; "
            "falling back to yfinance", ticker,
        )
        dates = _fetch_earnings_from_yfinance(
            ticker, yfinance_fetcher=yfinance_fetcher,
        )
        source = "yfinance" if dates else "unavailable"

    if dates:
        logger.info(
            "earnings source for %s: %s (%d dates)",
            ticker, source, len(dates),
        )
    else:
        logger.warning(
            "earnings source for %s: unavailable "
            "(both Tradier and yfinance returned nothing)", ticker,
        )

    _write_earnings_cache(ticker, dates)
    return dates


def is_in_earnings_window(
    ticker: str,
    sim_date: date,
    *,
    window_days: int = 5,
    earnings_dates: Optional[tuple[date, ...]] = None,
) -> bool:
    """True if ``sim_date`` is within ±``window_days`` of any earnings
    date for ``ticker``.

    ``earnings_dates`` lets engine callers pass pre-fetched dates per
    ticker so a daily loop doesn't repeat calendar lookups. When None,
    falls back to :func:`fetch_earnings_calendar`.
    """
    if window_days < 0:
        raise ValueError(
            f"window_days must be >= 0; got {window_days!r}"
        )
    if earnings_dates is None:
        earnings_dates = fetch_earnings_calendar(ticker)
    if not earnings_dates:
        return False
    for earn in earnings_dates:
        if abs((sim_date - earn).days) <= window_days:
            return True
    return False
