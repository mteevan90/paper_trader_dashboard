"""Tradier-backed OHLCV and option-chain fetcher (Phase 2 Section 2).

Live execution and current chain snapshot only — historical OHLCV is
fetched via :mod:`src.options.polygon` since Section 2.5 (May 2026).
Tradier's ``/markets/history`` endpoint returns null for expired option
contracts at all plan tiers, so the historical path was rerouted to
Polygon. This module remains the live data path for paper-trade
snapshots, current chain reads, and v2+ live order routing. See
Appendix I in ``docs/Options_Extension_Decisions.md`` for the
discovery and discipline notes.

Public functions:
    fetch_history          -- daily OHLCV for an OCC symbol or an
                              underlying ticker, sanity-gated and
                              cached (1-day TTL). Retained for the
                              underlying-history path (works for live
                              equities); historical per-OCC fetches
                              now go through polygon.fetch_history.
    fetch_expirations      -- current expiration dates for an underlying.
    fetch_chain_snapshot   -- current chain for an underlying +
                              expiration, optionally with bundled
                              ORATS Greeks.

Auth is bearer-token only (no OAuth flow). Tokens via env vars
``TRADIER_SANDBOX_TOKEN`` / ``TRADIER_PRODUCTION_TOKEN``; environment
selection via ``TRADIER_ENV`` (default ``sandbox``).

Tradier returns XML by default — every request sets
``Accept: application/json``. Rate limits surface in
``X-Ratelimit-*`` response headers; the :class:`RateLimiter` consumes
them and sleeps to ``X-Ratelimit-Expiry`` when ``X-Ratelimit-Available``
falls to <=1, while a fallback per-minute cap (sandbox-tight) governs
when headers are absent.

Tradier does not expose historical chain enumeration. ``fetch_history``
takes a single OCC symbol (or underlying ticker) and returns the OHLCV
series for that symbol; reconstructing the chain "as of date D" is a
Section 6 problem (candidate-OCC enumeration). See §8 of the design
memo.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import date, datetime
from typing import Mapping, Optional

import pandas as pd
import requests

from src.options.cache import cache_history, read_history

logger = logging.getLogger(__name__)

SANDBOX_BASE_URL = "https://sandbox.tradier.com/v1"
PRODUCTION_BASE_URL = "https://api.tradier.com/v1"

TRADIER_ENV_VAR = "TRADIER_ENV"
SANDBOX_TOKEN_ENV = "TRADIER_SANDBOX_TOKEN"
PRODUCTION_TOKEN_ENV = "TRADIER_PRODUCTION_TOKEN"

FALLBACK_RATE_LIMIT_PER_MIN = 60
HTTP_TIMEOUT = 30
MAX_RETRIES = 3

# Lite plan retail per-contract one-way commission, verified against
# tradier.com/individuals/pricing on 2026-05-09. Pass-throughs (clearing,
# ORF, TAF) add ~$0.10/contract and are not in the v1 fee model.
# TODO verify against the then-current schedule when Section 5 lands the fee model.
TRADIER_OPTION_FEE_PER_CONTRACT_USD = 0.35

_RATELIMIT_AVAILABLE_HEADER = "X-Ratelimit-Available"
_RATELIMIT_EXPIRY_HEADER = "X-Ratelimit-Expiry"


class RateLimiter:
    """Sliding-window rate limiter with header-driven adjustment.

    ``wait()`` enforces ``max_per_min`` as a fallback. After each response,
    ``update_from_headers(resp.headers)`` consumes Tradier's
    ``X-Ratelimit-*`` headers; if ``Available <= 1``, the next ``wait()``
    sleeps until the reported ``Expiry`` epoch (plus a small buffer).
    """

    def __init__(self, max_per_min: int = FALLBACK_RATE_LIMIT_PER_MIN):
        if max_per_min < 1:
            raise ValueError(f"max_per_min must be >= 1; got {max_per_min}")
        self.max_per_min = max_per_min
        self.calls: deque[float] = deque()
        self._sleep_until_epoch: Optional[float] = None

    def wait(self) -> None:
        if self._sleep_until_epoch is not None:
            now = time.time()
            if now < self._sleep_until_epoch:
                time.sleep(self._sleep_until_epoch - now + 0.1)
            self._sleep_until_epoch = None

        now = time.monotonic()
        while self.calls and now - self.calls[0] > 60.0:
            self.calls.popleft()
        if len(self.calls) >= self.max_per_min:
            sleep_for = 60.0 - (now - self.calls[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while self.calls and now - self.calls[0] > 60.0:
                self.calls.popleft()
        self.calls.append(time.monotonic())

    def update_from_headers(self, headers: Mapping[str, str]) -> None:
        avail = headers.get(_RATELIMIT_AVAILABLE_HEADER)
        expiry = headers.get(_RATELIMIT_EXPIRY_HEADER)
        if avail is None or expiry is None:
            return
        try:
            avail_int = int(avail)
            expiry_epoch = float(expiry)
        except ValueError:
            return
        # Tradier emits expiry as milliseconds-since-epoch on some
        # endpoints; normalize values that are clearly too large to be
        # plain seconds (anything > year 5000 in seconds).
        if expiry_epoch > 1e11:
            expiry_epoch /= 1000.0
        if avail_int <= 1:
            self._sleep_until_epoch = expiry_epoch


def _resolve_base_url() -> str:
    env = os.environ.get(TRADIER_ENV_VAR, "sandbox").lower()
    if env == "sandbox":
        return SANDBOX_BASE_URL
    if env == "production":
        return PRODUCTION_BASE_URL
    raise ValueError(
        f"{TRADIER_ENV_VAR}={env!r} is not valid; expected 'sandbox' or 'production'"
    )


def _resolve_beta_base_url() -> str:
    """Beta-product base URL (``https://<host>/beta``). The fundamentals
    calendars endpoint lives here, not under ``/v1``."""
    return _resolve_base_url().rsplit("/", 1)[0] + "/beta"


def _resolve_token() -> str:
    env = os.environ.get(TRADIER_ENV_VAR, "sandbox").lower()
    var_name = SANDBOX_TOKEN_ENV if env == "sandbox" else PRODUCTION_TOKEN_ENV
    token = os.environ.get(var_name)
    if not token:
        raise RuntimeError(
            f"{var_name} not set. Sign up at https://developer.tradier.com, "
            f"generate an access token, and add it to .env."
        )
    return token


def _http_get(
    path: str,
    params: dict,
    limiter: RateLimiter,
    session: Optional[requests.Session] = None,
    *,
    base_url_override: Optional[str] = None,
) -> dict:
    """GET ``<base_url><path>`` with bearer auth, JSON accept, retry,
    and rate-limit header consumption. Returns parsed JSON.

    Retries 429 (honoring ``Retry-After``) and 5xx with exponential
    backoff (1s, 2s, 4s). 4xx other than 429 raises immediately.

    ``base_url_override`` lets callers point at a non-v1 base (e.g.,
    :func:`_resolve_beta_base_url` for the fundamentals product). When
    None, defaults to :func:`_resolve_base_url`.
    """
    url = (base_url_override or _resolve_base_url()) + path
    headers = {
        "Authorization": f"Bearer {_resolve_token()}",
        "Accept": "application/json",
    }
    sess = session or requests
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            resp = sess.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            backoff = 2 ** attempt
            logger.warning(
                "tradier transient error (%s) attempt %d/%d, sleeping %ds",
                type(exc).__name__, attempt + 1, MAX_RETRIES, backoff,
            )
            time.sleep(backoff)
            continue

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "10"))
            logger.warning(
                "tradier 429, retry-after=%ds (attempt %d/%d)",
                retry_after, attempt + 1, MAX_RETRIES,
            )
            time.sleep(retry_after + 1)
            continue
        if resp.status_code >= 500:
            backoff = 2 ** attempt
            logger.warning(
                "tradier %d (attempt %d/%d), sleeping %ds",
                resp.status_code, attempt + 1, MAX_RETRIES, backoff,
            )
            time.sleep(backoff)
            continue
        resp.raise_for_status()
        limiter.update_from_headers(resp.headers)
        return resp.json()

    if last_exc is not None:
        raise last_exc
    raise requests.HTTPError(f"tradier exhausted {MAX_RETRIES} retries for {url}")


def _coerce_list(value) -> list:
    """Tradier's JSON envelope returns a single-element value as a dict
    instead of a list-of-one. Normalize to always be a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _history_to_df(payload: dict) -> pd.DataFrame:
    history = payload.get("history")
    if history is None or history in ({}, ""):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).rename_axis("date")
    days = _coerce_list(history.get("day"))
    if not days:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).rename_axis("date")
    df = pd.DataFrame(days)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["date", "open", "high", "low", "close", "volume"]].set_index("date").sort_index()
    return df


def fetch_history(
    symbol: str,
    start: date,
    end: date,
    *,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily OHLCV for ``symbol`` (OCC contract symbol or underlying
    ticker) over ``[start, end]``. Returns DataFrame indexed by ``date``
    with columns ``[open, high, low, close, volume]``, sorted ascending.

    On cache hit (within :data:`HISTORY_CACHE_TTL_HOURS`) returns cached
    frame. On miss: fetches, gates via the sanity gate, writes the
    cache, returns the frame.
    """
    if use_cache:
        cached = read_history(symbol)
        if cached is not None:
            return cached

    limiter = limiter or RateLimiter()
    payload = _http_get(
        "/markets/history",
        {"symbol": symbol, "interval": "daily",
         "start": start.isoformat(), "end": end.isoformat()},
        limiter,
        session=session,
    )
    df = _history_to_df(payload)
    if not df.empty:
        cache_history(symbol, df)
    return df


def fetch_index_quote_history(
    symbol: str,
    start: date,
    end: date,
    *,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily history for an index symbol (e.g., ``$BXM``, ``$VIX``).

    Section 2 amendment landed with Section 8 to feed BXM into the
    promotion gate. Wraps :func:`fetch_history` after normalizing
    ``symbol`` to the index-prefixed form Tradier expects (a leading
    ``$`` if the caller passed the bare ticker). Returns an empty
    DataFrame on miss/error so :func:`benchmarks.fetch_bxm` can fall
    back to yfinance.
    """
    normalized = symbol if symbol.startswith("$") else f"${symbol}"
    return fetch_history(
        normalized, start, end,
        limiter=limiter, session=session, use_cache=use_cache,
    )


def fetch_expirations(
    ticker: str,
    *,
    include_all_roots: bool = True,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
) -> list[date]:
    """Current expiration dates for ``ticker``. Returns sorted list.
    Empty list if Tradier returns no expirations (unknown symbol or
    no listed options)."""
    limiter = limiter or RateLimiter()
    payload = _http_get(
        "/markets/options/expirations",
        {"symbol": ticker,
         "includeAllRoots": "true" if include_all_roots else "false",
         "strikes": "false"},
        limiter,
        session=session,
    )
    expirations = payload.get("expirations")
    if expirations is None or expirations in ({}, ""):
        return []
    raw_dates = _coerce_list(expirations.get("date"))
    return sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in raw_dates)


_GREEK_FIELDS = ("delta", "gamma", "theta", "vega", "rho",
                 "bid_iv", "mid_iv", "ask_iv")
_OPTION_FIELDS = (
    "option_type", "strike", "bid", "ask", "last", "volume",
    "open_interest", "contract_size", "expiration_date",
    "expiration_type", "root_symbol",
)


def _option_to_row(option: dict) -> dict:
    row: dict = {"occ_symbol": option.get("symbol")}
    for field in _OPTION_FIELDS:
        row[field] = option.get(field)
    greeks = option.get("greeks") or {}
    for field in _GREEK_FIELDS:
        row[field] = greeks.get(field)
    return row


def fetch_chain_snapshot(
    ticker: str,
    expiration: date,
    *,
    with_greeks: bool = True,
    limiter: Optional[RateLimiter] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Current chain for ``ticker`` at ``expiration``. One row per contract.

    Columns include ``occ_symbol, option_type, strike, bid, ask, last,
    volume, open_interest, contract_size, expiration_date,
    expiration_type, root_symbol``. With ``with_greeks=True`` (default),
    appends ``delta, gamma, theta, vega, rho, bid_iv, mid_iv, ask_iv``
    columns; missing greek values are NaN.

    Library function does not auto-cache — chain snapshots are
    point-in-time, so callers (CLI, paper-trade harness) decide when
    to persist via :func:`src.options.cache.cache_chain_snapshot`.
    """
    limiter = limiter or RateLimiter()
    payload = _http_get(
        "/markets/options/chains",
        {"symbol": ticker, "expiration": expiration.isoformat(),
         "greeks": "true" if with_greeks else "false"},
        limiter,
        session=session,
    )
    options_envelope = payload.get("options")
    columns = ["occ_symbol", *_OPTION_FIELDS]
    if with_greeks:
        columns += list(_GREEK_FIELDS)
    if options_envelope is None or options_envelope in ({}, ""):
        return pd.DataFrame(columns=columns)
    raw_options = _coerce_list(options_envelope.get("option"))
    if not raw_options:
        return pd.DataFrame(columns=columns)
    rows = [_option_to_row(o) for o in raw_options]
    df = pd.DataFrame(rows)
    if not with_greeks:
        df = df.drop(columns=list(_GREEK_FIELDS), errors="ignore")
    return df[columns]
