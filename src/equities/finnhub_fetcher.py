"""Finnhub Basic-tier equity fetcher.

Provides the Larger Universe v1 study with prices + fundamentals for
SP1500 current members and historical-delisted SP1500 members. Earnings
dates are NOT served by this module — Finnhub Basic is forward-only for
earnings, so the earnings_dates path stays on yfinance (see
``src.backtest.fetch_earnings_dates`` and the stash@{0} retry/sanity-gate
work).

Endpoints used (with Basic-tier rate-limit buckets observed in
``docs/diagnostics/larger_universe_v1_capabilities.md``):

- ``/stock/candle`` — 150/min — daily OHLCV (10y guarantee on Basic;
  some symbols return longer history opportunistically)
- ``/stock/metric?metric=all`` — 60/min — point-in-time fundamentals
- ``/stock/symbol?exchange=US`` — 150/min — universe enumeration

Auth: ``FINNHUB_API_KEY`` env var, sent as ``token`` query parameter.

Cache layout under ``models/cache/equities/finnhub/``:
- ``prices/<SYMBOL>.parquet``
- ``metrics/<SYMBOL>.json``
- ``symbol_list_<EXCHANGE>.json``

All writes are atomic (write-to-tmp + rename). 7-day TTL on prices and
metrics — re-fetch policy is governed by the caller (Phase 3 fetcher
will pass ``use_cache=True`` to avoid duplicate work across resumed
runs).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.options.tradier import RateLimiter

__all__ = [
    "fetch_candles",
    "fetch_metrics",
    "fetch_symbol_list",
    "make_candle_limiter",
    "make_metric_limiter",
    "FINNHUB_BASE_URL",
    "FINNHUB_API_KEY_ENV",
    "FINNHUB_CACHE_ROOT",
    "CANDLE_RATE_PER_MIN",
    "METRIC_RATE_PER_MIN",
    "PRICE_CACHE_TTL_DAYS",
    "METRIC_CACHE_TTL_DAYS",
]

logger = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_API_KEY_ENV = "FINNHUB_API_KEY"
FINNHUB_CACHE_ROOT = Path("models") / "cache" / "equities" / "finnhub"

CANDLE_RATE_PER_MIN = 150
METRIC_RATE_PER_MIN = 60

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

PRICE_CACHE_TTL_DAYS = 7
METRIC_CACHE_TTL_DAYS = 7


def _resolve_token() -> str:
    token = os.environ.get(FINNHUB_API_KEY_ENV)
    if not token:
        raise RuntimeError(
            f"{FINNHUB_API_KEY_ENV} not set. Get a Basic-tier ($49.99/mo) "
            "key at https://finnhub.io and add it to .env at the repo root."
        )
    return token


def make_candle_limiter() -> RateLimiter:
    """Shared rate limiter for the 150/min candle bucket. Reuse across threads."""
    return RateLimiter(CANDLE_RATE_PER_MIN)


def make_metric_limiter() -> RateLimiter:
    """Shared rate limiter for the 60/min fundamentals bucket."""
    return RateLimiter(METRIC_RATE_PER_MIN)


# ---------------- cache helpers ----------------


def _price_cache_path(symbol: str) -> Path:
    return FINNHUB_CACHE_ROOT / "prices" / f"{symbol.upper()}.parquet"


def _metric_cache_path(symbol: str) -> Path:
    return FINNHUB_CACHE_ROOT / "metrics" / f"{symbol.upper()}.json"


def _symbol_list_cache_path(exchange: str) -> Path:
    return FINNHUB_CACHE_ROOT / f"symbol_list_{exchange.upper()}.json"


def _is_fresh(path: Path, ttl_days: int) -> bool:
    if not path.exists():
        return False
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days < ttl_days


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


# ---------------- HTTP layer ----------------


def _request_with_retries(
    session: requests.Session,
    path: str,
    *,
    params: dict,
    timeout: int,
    max_retries: int,
) -> requests.Response:
    """GET with exponential backoff on 5xx and network errors.

    4xx returned to caller for status-aware handling. Honors ``Retry-After``
    when present (capped at 60s).
    """
    url = f"{FINNHUB_BASE_URL}{path}"
    last_exc: Optional[BaseException] = None
    response: Optional[requests.Response] = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = min(float(retry_after), 60.0)
                    except ValueError:
                        delay = DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                else:
                    delay = DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "finnhub: 429 on %s, sleeping %.1fs (attempt %d/%d)",
                    path, delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
                continue
            if response.status_code < 500:
                return response
            last_exc = requests.HTTPError(
                f"5xx from Finnhub: {response.status_code}",
                response=response,
            )
            logger.warning(
                "finnhub: %d on %s, attempt %d/%d, backing off",
                response.status_code, path, attempt + 1, max_retries,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning(
                "finnhub: %s on %s, attempt %d/%d, backing off",
                type(exc).__name__, path, attempt + 1, max_retries,
            )
        if attempt < max_retries - 1:
            time.sleep(DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    assert response is not None
    return response


# ---------------- result parsing ----------------


def _empty_price_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    ).rename_axis("date")


def _parse_candle_body(body: dict) -> pd.DataFrame:
    if body.get("s") != "ok":
        return _empty_price_df()
    t = body.get("t") or []
    if not t:
        return _empty_price_df()
    df = pd.DataFrame({
        "date": [datetime.fromtimestamp(ts, tz=timezone.utc).date() for ts in t],
        "open": body.get("o", []),
        "high": body.get("h", []),
        "low": body.get("l", []),
        "close": body.get("c", []),
        "volume": body.get("v", []),
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


# ---------------- public API ----------------


def fetch_candles(
    symbol: str,
    start: date,
    end: date,
    *,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
    truncate_at: Optional[date] = None,
) -> pd.DataFrame:
    """Daily OHLCV for an equity ticker from Finnhub.

    Returns a DataFrame indexed by ``date`` with columns
    ``[open, high, low, close, volume]``. Empty DataFrame if Finnhub
    returns ``s: "no_data"`` (the symbol is unknown or outside the
    Basic-tier 10y window).

    ``truncate_at`` clips the series at the given date inclusive — used
    for OTC-tail truncation on delisted tickers where Finnhub returns
    post-corporate-event pink-sheet candles that should not be included
    in the SP1500-tier dataset. The clip date should come from the
    Wikipedia-documented delisting date in the universe builder output.

    Cache: parquet at ``models/cache/equities/finnhub/prices/<SYM>.parquet``
    with ``PRICE_CACHE_TTL_DAYS``-day TTL.
    """
    sym_upper = symbol.upper()
    cache_path = _price_cache_path(sym_upper)
    if use_cache and _is_fresh(cache_path, PRICE_CACHE_TTL_DAYS):
        # Cache contains the already-truncated form (truncation is applied
        # before write below), so we just return it as-is.
        return pd.read_parquet(cache_path)

    token = _resolve_token()
    sess = session or requests.Session()
    params = {
        "symbol": sym_upper,
        "resolution": "D",
        "from": int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp()),
        "to": int(datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).timestamp()),
        "token": token,
    }
    if limiter is not None:
        limiter.wait()
    response = _request_with_retries(
        sess, "/stock/candle",
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )
    if limiter is not None:
        # /stock/candle doesn't return the same headers Tradier does, but
        # update_from_headers is defensive against absent keys.
        try:
            limiter.update_from_headers(response.headers)
        except Exception:
            pass

    if response.status_code == 401:
        raise RuntimeError(
            f"Finnhub authentication failed for {symbol}: check "
            f"{FINNHUB_API_KEY_ENV} value"
        )
    if response.status_code == 403:
        raise RuntimeError(
            f"Finnhub NOT_AUTHORIZED for {symbol} — endpoint or window "
            "outside Basic-tier entitlement"
        )
    response.raise_for_status()
    body = response.json()
    df = _parse_candle_body(body)
    # Apply OTC-tail truncation BEFORE caching so the on-disk parquet is
    # the form we want in the snapshot. Skip the cache write if the clip
    # produces an empty series (truncate_at was before the symbol's first
    # available date) — no point persisting an empty sentinel.
    if truncate_at is not None and not df.empty:
        df = df[df.index <= truncate_at]
    if not df.empty and use_cache:
        _atomic_write_parquet(cache_path, df)
    return df


def fetch_metrics(
    symbol: str,
    *,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
) -> dict:
    """Fundamentals snapshot for an equity ticker from /stock/metric.

    Returns the parsed body: ``{metric: {...}, metricType, series, symbol}``.
    Returns an empty dict ``{}`` if Finnhub responds with no metric data.

    Cache: JSON at ``models/cache/equities/finnhub/metrics/<SYM>.json``
    with ``METRIC_CACHE_TTL_DAYS``-day TTL.
    """
    sym_upper = symbol.upper()
    cache_path = _metric_cache_path(sym_upper)
    if use_cache and _is_fresh(cache_path, METRIC_CACHE_TTL_DAYS):
        return json.loads(cache_path.read_text(encoding="utf-8"))

    token = _resolve_token()
    sess = session or requests.Session()
    params = {"symbol": sym_upper, "metric": "all", "token": token}
    if limiter is not None:
        limiter.wait()
    response = _request_with_retries(
        sess, "/stock/metric",
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )
    if response.status_code == 401:
        raise RuntimeError(
            f"Finnhub authentication failed for {symbol}: check "
            f"{FINNHUB_API_KEY_ENV} value"
        )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or "metric" not in body:
        return {}
    if use_cache:
        _atomic_write_bytes(
            cache_path,
            json.dumps(body, separators=(",", ":")).encode("utf-8"),
        )
    return body


def fetch_symbol_list(
    exchange: str = "US",
    *,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
) -> list[dict]:
    """Full listing on a given exchange from /stock/symbol.

    On Basic tier, the ``delisted=true`` filter is silently ignored, so
    the returned list includes both currently-listed and historically-listed
    symbols mixed together. Callers should cross-reference with an external
    source (e.g., Wikipedia component-change tables for the SP1500 study)
    to determine which symbols are SP500/400/600 historical members.

    Cache: 7-day TTL JSON at ``models/cache/equities/finnhub/symbol_list_<EX>.json``.
    """
    cache_path = _symbol_list_cache_path(exchange)
    if use_cache and _is_fresh(cache_path, ttl_days=7):
        return json.loads(cache_path.read_text(encoding="utf-8"))

    token = _resolve_token()
    sess = session or requests.Session()
    params = {"exchange": exchange.upper(), "token": token}
    response = _request_with_retries(
        sess, "/stock/symbol",
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, list):
        return []
    if use_cache:
        _atomic_write_bytes(
            cache_path,
            json.dumps(body, separators=(",", ":")).encode("utf-8"),
        )
    return body
