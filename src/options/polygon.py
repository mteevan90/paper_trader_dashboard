"""Polygon.io / Massive.com historical OHLCV fetcher for options contracts
(Phase 2 Section 2.5).

Polygon.io rebranded to Massive.com on October 30, 2025; existing
``api.polygon.io`` URLs continue to work without interruption per the
official rebrand announcement. This module uses ``api.polygon.io`` for
stability; future v1.1+ may migrate to ``api.massive.com`` if/when DNS
aliases that endpoint.

Replaces the historical-data path of :mod:`src.options.tradier` since
May 2026 — Tradier's ``/markets/history`` endpoint returns null for
expired options at all plan tiers, which made it unsuitable for
backtests. See Section 2.5 / Appendix I in
``docs/Options_Extension_Decisions.md`` for the discovery and
discipline notes.

Auth: ``POLYGON_API_KEY`` env var, sent as ``apiKey`` query parameter.

Cache layout: ``models/cache/options/polygon/history/<symbol>.parquet``,
mirroring Section 2's parquet discipline (1-day TTL, sanity-gated at
50% coverage). The Tradier-side cache at
``models/cache/options/tradier/history/`` is preserved for the v2+ live
execution path; the two are intentionally on separate disk paths so a
backend swap can't cross-contaminate.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.options.sanity_gate import passes_sanity_gate
from src.options.tradier import RateLimiter


__all__ = [
    "fetch_history",
    "POLYGON_BASE_URL",
    "POLYGON_CACHE_ROOT",
    "POLYGON_API_KEY_ENV",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BACKOFF_SECONDS",
    "HISTORY_CACHE_TTL_HOURS",
]


logger = logging.getLogger(__name__)


POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_API_KEY_ENV = "POLYGON_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

POLYGON_CACHE_ROOT = Path("models") / "cache" / "options" / "polygon" / "history"
HISTORY_CACHE_TTL_HOURS = 24


# ----------------- token + symbol conversion -----------------


def _resolve_token() -> str:
    token = os.environ.get(POLYGON_API_KEY_ENV)
    if not token:
        raise RuntimeError(
            f"{POLYGON_API_KEY_ENV} not set. Sign up at https://massive.com "
            "(formerly polygon.io), get an Options Developer tier API key, "
            "and add it to .env at the repo root."
        )
    return token


def _occ_to_polygon_ticker(occ: str) -> str:
    """Convert an OCC symbol to Polygon's ticker format.

    Tradier-style OCC: ``"SPY   240719C00540000"`` (6-char left-padded
    underlying then 15 chars of date+type+strike).
    Polygon: ``"O:SPY240719C00540000"`` (no spaces, ``O:`` prefix).

    Idempotent on inputs that are already space-stripped or already
    prefixed (``"O:..."`` passes through unchanged).
    """
    if occ.startswith("O:"):
        return occ
    cleaned = "".join(occ.split())
    return f"O:{cleaned}"


# ----------------- cache helpers -----------------


def _cache_path(symbol: str) -> Path:
    return POLYGON_CACHE_ROOT / f"{symbol}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < HISTORY_CACHE_TTL_HOURS


def _read_cache(symbol: str) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol)
    if not _is_fresh(path):
        return None
    return pd.read_parquet(path)


def _write_cache(symbol: str, df: pd.DataFrame) -> Path:
    path = _cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def _expected_days(start: date, end: date) -> int:
    return max((end - start).days + 1, 1)


# ----------------- HTTP layer -----------------


def _request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict,
    timeout: int,
    max_retries: int,
) -> requests.Response:
    """GET with exponential backoff on 5xx and network errors.

    2xx/3xx/4xx responses are returned to the caller for status-aware
    handling (the auth/empty/etc. branches in :func:`fetch_history`
    inspect status codes directly). 5xx and transport-level exceptions
    retry up to ``max_retries`` with backoff
    ``DEFAULT_RETRY_BACKOFF_SECONDS * 2**attempt``; if the final attempt
    still fails, the last exception is re-raised.
    """
    last_exc: Optional[BaseException] = None
    response: Optional[requests.Response] = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code < 500:
                return response
            last_exc = requests.HTTPError(
                f"5xx from Polygon: {response.status_code}",
                response=response,
            )
            logger.warning(
                "polygon: %d on attempt %d/%d, backing off",
                response.status_code, attempt + 1, max_retries,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning(
                "polygon: %s on attempt %d/%d, backing off",
                type(exc).__name__, attempt + 1, max_retries,
            )
        if attempt < max_retries - 1:
            time.sleep(DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    assert response is not None  # unreachable
    return response


# ----------------- result parsing -----------------


_POLYGON_BAR_FIELDS = ("o", "h", "l", "c", "v", "t")


def _parse_polygon_results(results: list[dict]) -> pd.DataFrame:
    """Convert Polygon aggregates JSON to OHLCV DataFrame indexed by Eastern date.

    Polygon bar fields: ``o``=open, ``h``=high, ``l``=low, ``c``=close,
    ``v``=volume, ``t``=timestamp ms-epoch.
    """
    rows: list[dict] = []
    for bar in results:
        try:
            ts_ms = bar["t"]
        except KeyError:
            continue
        bar_date = (
            pd.Timestamp(ts_ms, unit="ms", tz="UTC")
            .tz_convert("America/New_York")
            .date()
        )
        rows.append({
            "date": bar_date,
            "open": bar.get("o"),
            "high": bar.get("h"),
            "low": bar.get("l"),
            "close": bar.get("c"),
            "volume": bar.get("v"),
        })
    if not rows:
        return _empty_history_df()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
    ).rename_axis("date")


# ----------------- public API -----------------


def fetch_history(
    symbol: str,
    start: date,
    end: date,
    *,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily OHLCV for an options contract from Polygon.

    Mirrors :func:`src.options.tradier.fetch_history`. Returns a
    DataFrame indexed by ``date`` with columns ``[open, high, low,
    close, volume]``, sorted ascending. Empty DataFrame if the contract
    didn't trade in the window (Polygon returns 200 with
    ``results: []`` for valid-but-untraded contracts).

    Cache: parquet at
    ``models/cache/options/polygon/history/<symbol>.parquet`` with
    1-day TTL, sanity-gated at 50% coverage on writes.

    Errors:

    - 403 NOT_AUTHORIZED → :class:`RuntimeError` with the Polygon
      message (typical: window pre-dates the plan's historical floor).
    - 401 → :class:`RuntimeError` indicating an auth/token issue.
    - 5xx and network errors retry with exponential backoff up to
      ``DEFAULT_MAX_RETRIES``, then re-raise.
    - Other 4xx (e.g., 404) → :func:`requests.Response.raise_for_status`
      raises an :class:`requests.HTTPError`.
    """
    if use_cache:
        cached = _read_cache(symbol)
        if cached is not None:
            return cached

    polygon_ticker = _occ_to_polygon_ticker(symbol)
    token = _resolve_token()
    sess = session or requests.Session()

    url = (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{polygon_ticker}"
        f"/range/1/day/{start.isoformat()}/{end.isoformat()}"
    )
    params = {"apiKey": token, "adjusted": "true", "sort": "asc"}

    if limiter is not None:
        limiter.wait()

    response = _request_with_retries(
        sess, url,
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )

    if response.status_code == 403:
        try:
            body = response.json()
        except ValueError:
            body = {}
        msg = body.get("message") or "data timeframe outside plan"
        raise RuntimeError(
            f"Polygon NOT_AUTHORIZED for {symbol} window "
            f"{start.isoformat()} to {end.isoformat()}: {msg}"
        )
    if response.status_code == 401:
        raise RuntimeError(
            f"Polygon authentication failed for {symbol}: "
            f"check {POLYGON_API_KEY_ENV} value"
        )
    response.raise_for_status()

    body = response.json()
    if body.get("status") not in ("OK", "DELAYED"):
        raise RuntimeError(
            f"Polygon returned non-OK status for {symbol}: {body.get('status')!r}"
        )

    results = body.get("results") or []
    if not results:
        return _empty_history_df()

    df = _parse_polygon_results(results)
    if df.empty:
        return df

    if use_cache:
        expected = _expected_days(start, end)
        passed, reason = passes_sanity_gate(df, expected)
        if passed:
            _write_cache(symbol, df)
        else:
            logger.warning(
                "polygon: sanity gate refused cache write for %s: %s",
                symbol, reason,
            )
    return df
