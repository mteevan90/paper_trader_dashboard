"""Tradier corporate-calendar fetcher with parquet caching (Phase 2 Section 6).

Public functions:
    fetch_earnings_calendar  -- earnings dates for a ticker, cached
                                7 days at
                                ``models/cache/options/tradier/earnings/``.
    is_in_earnings_window    -- predicate: is sim_date within ±N days
                                of any earnings date for ticker.

Indexes (SPX, SPY, QQQ) have no earnings — they return empty tuples
without hitting the network.

Tradier's corporate-calendar endpoint structure has shifted between API
versions, so the parser here is defensive: walks the payload looking
for ISO date strings under common envelope shapes (``request/events``,
``calendars/calendar/events``, plain ``events`` lists). When the
payload doesn't match any of those, the call is logged and an empty
tuple is returned rather than raising — earnings avoidance becoming a
soft-no-op is acceptable for v1; missing it is preferred to crashing
the daily backtest loop. ``# TODO verify endpoint`` left as a
follow-up so the parser can be tightened against a known-good response
once Section 6 ships.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import requests

from src.options.tradier import RateLimiter, _http_get


__all__ = [
    "fetch_earnings_calendar",
    "is_in_earnings_window",
    "INDEX_TICKERS",
    "EARNINGS_CACHE_TTL_HOURS",
    "EARNINGS_CACHE_DIR",
]


logger = logging.getLogger(__name__)


INDEX_TICKERS: frozenset[str] = frozenset({"SPX", "SPY", "QQQ"})

EARNINGS_CACHE_DIR: Path = (
    Path("models") / "cache" / "options" / "tradier" / "earnings"
)
EARNINGS_CACHE_TTL_HOURS: int = 7 * 24

CORPORATE_CALENDAR_PATH: str = "/markets/calendars/corporate"


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


def _walk_for_earnings(payload) -> Iterable[date]:
    """Recursively yield earnings dates found under common envelope shapes.

    Tradier corporate calendar responses have nested under different keys
    across API versions — ``calendars/calendar/events/event``,
    ``request/events``, etc. Rather than hard-coding one path, walk the
    structure and yield any ``date``-shaped value whose neighbor key
    suggests "earnings".
    """
    if isinstance(payload, dict):
        type_value = (
            payload.get("event_type")
            or payload.get("type")
            or payload.get("eventType")
            or ""
        )
        if isinstance(type_value, str) and "earnings" in type_value.lower():
            for key in ("date", "begin_date", "event_date", "report_date"):
                d = _coerce_iso_date(payload.get(key))
                if d is not None:
                    yield d
                    break
        for value in payload.values():
            yield from _walk_for_earnings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk_for_earnings(item)


def fetch_earnings_calendar(
    ticker: str,
    *,
    fetcher: Optional[HttpGet] = None,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
) -> tuple[date, ...]:
    """Fetch all known earnings dates for ``ticker`` from Tradier's
    corporate calendar endpoint. Returns sorted tuple.

    Indexes (SPX/SPY/QQQ) return ``()`` immediately without I/O.

    On cache hit (within :data:`EARNINGS_CACHE_TTL_HOURS`) returns the
    cached tuple. On miss, fetches via Tradier, parses defensively
    (logs and returns ``()`` on shape mismatch), writes the cache, and
    returns. Set ``use_cache=False`` to force a refresh.
    """
    if ticker in INDEX_TICKERS:
        return ()

    if use_cache:
        cached = _read_earnings_cache(ticker)
        if cached is not None:
            return cached

    fetcher = fetcher or _http_get
    limiter = limiter or RateLimiter()
    try:
        payload = fetcher(
            CORPORATE_CALENDAR_PATH,
            {"symbols": ticker},
            limiter,
            session=session,
        )
    except Exception as exc:
        logger.warning(
            "tradier earnings fetch failed for %s: %s", ticker, exc
        )
        return ()

    raw_dates = sorted(set(_walk_for_earnings(payload)))
    if not raw_dates:
        # TODO verify endpoint — payload didn't match any expected
        # shape. Log once at INFO so a study run surfaces the symbol
        # but doesn't spam.
        logger.info(
            "no earnings dates parsed for %s "
            "(payload shape may have changed)",
            ticker,
        )

    dates = tuple(raw_dates)
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
