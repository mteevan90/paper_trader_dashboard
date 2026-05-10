"""Benchmark series fetchers for the options promotion gate (Phase 2 Section 8).

Two public series:

- :func:`fetch_spy_total_return` — daily SPY price plus dividend
  reinvestment, computed by walking ex-dividend dates from Tradier's
  corporate calendar and compounding into ``total_return_index``.
- :func:`fetch_bxm` — daily CBOE BuyWrite Index closes. Tries Tradier
  index history first via :func:`tradier.fetch_index_quote_history`,
  falls back to yfinance ``^BXM`` if Tradier returns empty/errors.

Both cached at ``models/cache/options/benchmarks/`` with a 7-day TTL —
benchmarks are stable but refresh weekly to pick up the latest data.
The cache filename encodes the (start, end) window so studies with
different windows don't clobber each other.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from src.options import tradier
from src.options.tradier import RateLimiter, _http_get


__all__ = [
    "fetch_spy_total_return",
    "fetch_bxm",
    "BENCHMARKS_CACHE_DIR",
    "BENCHMARKS_CACHE_TTL_HOURS",
    "BXM_TRADIER_SYMBOL",
    "BXM_YFINANCE_SYMBOL",
    "BxmSource",
]


logger = logging.getLogger(__name__)


BENCHMARKS_CACHE_DIR: Path = (
    Path("models") / "cache" / "options" / "benchmarks"
)
BENCHMARKS_CACHE_TTL_HOURS: int = 7 * 24

BXM_TRADIER_SYMBOL: str = "$BXM"
BXM_YFINANCE_SYMBOL: str = "^BXM"

CORPORATE_CALENDAR_PATH: str = "/markets/calendars/corporate"


class BxmSource:
    """Where the cached BXM data was fetched from. Surfaces in logs so
    promoted study output is auditable."""

    TRADIER = "tradier"
    YFINANCE = "yfinance"
    UNAVAILABLE = "unavailable"


# ----------------- caching helpers -----------------


def _spy_cache_path(start: date, end: date) -> Path:
    return (
        BENCHMARKS_CACHE_DIR
        / f"spy_total_return_{start.isoformat()}_{end.isoformat()}.parquet"
    )


def _bxm_cache_path(start: date, end: date) -> Path:
    return (
        BENCHMARKS_CACHE_DIR
        / f"bxm_{start.isoformat()}_{end.isoformat()}.parquet"
    )


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < BENCHMARKS_CACHE_TTL_HOURS


def _read_parquet_if_fresh(path: Path) -> Optional[pd.DataFrame]:
    if not _is_fresh(path):
        return None
    return pd.read_parquet(path)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


# ----------------- SPY total return -----------------


def _walk_for_dividends(payload) -> Iterable[tuple[date, float]]:
    """Yield ``(ex_date, amount)`` pairs from a Tradier corporate-
    calendar payload. Walks defensively to handle multiple envelope
    shapes — same approach :mod:`src.options.earnings` uses for
    earnings events."""
    from datetime import datetime

    if isinstance(payload, dict):
        type_value = (
            payload.get("event_type")
            or payload.get("type")
            or payload.get("eventType")
            or ""
        )
        if (
            isinstance(type_value, str)
            and "dividend" in type_value.lower()
        ):
            ex_date: Optional[date] = None
            for key in ("ex_date", "date", "begin_date", "event_date"):
                value = payload.get(key)
                if isinstance(value, str):
                    try:
                        ex_date = datetime.strptime(
                            value[:10], "%Y-%m-%d",
                        ).date()
                        break
                    except ValueError:
                        continue
            amount: Optional[float] = None
            for key in ("amount", "cash_amount", "dividend_amount"):
                value = payload.get(key)
                if value is None:
                    continue
                try:
                    amount = float(value)
                    break
                except (TypeError, ValueError):
                    continue
            if ex_date is not None and amount is not None and amount > 0:
                yield ex_date, amount
        for value in payload.values():
            yield from _walk_for_dividends(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk_for_dividends(item)


def _fetch_spy_dividends(
    start: date,
    end: date,
    *,
    fetcher: Optional[Callable[..., dict]] = None,
    limiter: Optional[RateLimiter] = None,
) -> dict[date, float]:
    """Best-effort dividend lookup from Tradier corporate calendar.

    Returns ``{ex_date: amount_per_share}``. Empty dict on parse
    failure or fetch error — total_return_index then equals price-only
    return for the window, with a logged warning.
    """
    fetcher = fetcher or _http_get
    limiter = limiter or RateLimiter()
    try:
        payload = fetcher(
            CORPORATE_CALENDAR_PATH,
            {"symbols": "SPY"},
            limiter,
            session=None,
        )
    except Exception as exc:
        logger.warning(
            "Tradier SPY dividend calendar fetch failed: %s "
            "(degrading to price-only)", exc,
        )
        return {}

    out: dict[date, float] = {}
    for ex_date, amount in _walk_for_dividends(payload):
        if start <= ex_date <= end:
            out[ex_date] = amount
    if not out:
        logger.info(
            "no SPY dividend events parsed in window "
            "[%s, %s] — total_return_index will equal price index",
            start, end,
        )
    return out


def fetch_spy_total_return(
    start: date,
    end: date,
    *,
    history_fetcher: Optional[Callable[..., pd.DataFrame]] = None,
    dividend_fetcher: Optional[Callable[..., dict]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily SPY total return (price + dividend reinvestment) from Tradier.

    Returns a DataFrame indexed by ``date`` with columns:
      - ``close`` — raw SPY close
      - ``dividend_per_share`` — per-share amount on ex-div dates, 0 elsewhere
      - ``total_return_index`` — compound total return starting at 1.0 on
        the first available date in the window

    On cache hit (within :data:`BENCHMARKS_CACHE_TTL_HOURS`) returns
    the cached frame.
    """
    if use_cache:
        cached = _read_parquet_if_fresh(_spy_cache_path(start, end))
        if cached is not None:
            return cached

    fetch_history = history_fetcher or tradier.fetch_history
    df = fetch_history("SPY", start, end)
    if df is None or df.empty:
        raise RuntimeError(
            f"Tradier returned empty SPY history for [{start}, {end}]"
        )
    out = pd.DataFrame(index=df.index)
    out.index.name = "date"
    out["close"] = df["close"].astype(float)

    dividends = _fetch_spy_dividends(
        start, end, fetcher=dividend_fetcher,
    )
    out["dividend_per_share"] = [
        dividends.get(d, 0.0) for d in out.index
    ]

    # Compound total return: each day's return = (close / prev_close) +
    # (dividend / prev_close). Reinvest dividends at close on ex-div.
    closes = out["close"].values
    divs = out["dividend_per_share"].values
    tri = [1.0]
    for i in range(1, len(out)):
        prev_close = closes[i - 1]
        if prev_close <= 0:
            tri.append(tri[-1])
            continue
        price_return = closes[i] / prev_close - 1.0
        div_return = divs[i] / prev_close
        tri.append(tri[-1] * (1.0 + price_return + div_return))
    out["total_return_index"] = tri

    _write_parquet(_spy_cache_path(start, end), out)
    return out


# ----------------- BXM benchmark -----------------


def _fetch_bxm_from_tradier(
    start: date,
    end: date,
    *,
    fetcher: Optional[Callable[..., pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Try Tradier index history. Returns DataFrame[close] indexed by
    date — empty if Tradier doesn't carry BXM."""
    fetcher = fetcher or tradier.fetch_index_quote_history
    try:
        df = fetcher(BXM_TRADIER_SYMBOL, start, end)
    except Exception as exc:
        logger.warning("Tradier BXM history fetch failed: %s", exc)
        return pd.DataFrame(columns=["close"]).rename_axis("date")
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame(columns=["close"]).rename_axis("date")
    out = pd.DataFrame(index=df.index)
    out.index.name = "date"
    out["close"] = df["close"].astype(float)
    return out


def _fetch_bxm_from_yfinance(
    start: date, end: date,
) -> pd.DataFrame:
    """yfinance fallback for ``^BXM``. Returns DataFrame[close] indexed
    by date — empty on yfinance error."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not available for BXM fallback")
        return pd.DataFrame(columns=["close"]).rename_axis("date")
    try:
        ticker = yf.Ticker(BXM_YFINANCE_SYMBOL)
        # yfinance end is exclusive, so add a day.
        hist = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
        )
    except Exception as exc:
        logger.warning("yfinance BXM fetch failed: %s", exc)
        return pd.DataFrame(columns=["close"]).rename_axis("date")
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.DataFrame(columns=["close"]).rename_axis("date")
    idx = [
        d.date() if hasattr(d, "date") else d for d in hist.index
    ]
    out = pd.DataFrame({"close": hist["Close"].astype(float).values}, index=idx)
    out.index.name = "date"
    return out


def fetch_bxm(
    start: date,
    end: date,
    *,
    tradier_fetcher: Optional[Callable[..., pd.DataFrame]] = None,
    yfinance_fetcher: Optional[Callable[[date, date], pd.DataFrame]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily BXM closes. Tradier first, yfinance fallback.

    Returns a DataFrame indexed by ``date`` with a single ``close``
    column. The empty DataFrame is returned (and logged) if both
    sources fail; callers (the promotion gate) skip the BXM check
    rather than failing the whole run.
    """
    if use_cache:
        cached = _read_parquet_if_fresh(_bxm_cache_path(start, end))
        if cached is not None:
            return cached

    df = _fetch_bxm_from_tradier(start, end, fetcher=tradier_fetcher)
    if not df.empty:
        logger.info(
            "BXM source: %s for [%s, %s]", BxmSource.TRADIER, start, end,
        )
        _write_parquet(_bxm_cache_path(start, end), df)
        return df

    fallback = yfinance_fetcher or _fetch_bxm_from_yfinance
    df = fallback(start, end)
    if not df.empty:
        logger.info(
            "BXM source: %s for [%s, %s]",
            BxmSource.YFINANCE, start, end,
        )
        _write_parquet(_bxm_cache_path(start, end), df)
        return df

    logger.error(
        "BXM source: %s for [%s, %s] — both Tradier and yfinance empty",
        BxmSource.UNAVAILABLE, start, end,
    )
    return pd.DataFrame(columns=["close"]).rename_axis("date")
