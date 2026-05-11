"""Historical chain reconstruction for the options backtest engine
(Phase 2 Section 6, default backend swapped to Polygon in Section 2.5).

Polygon's historical endpoint returns OHLCV per OCC contract symbol;
there is no "what strikes existed for SYMBOL on DATE" endpoint. To
reconstruct a chain at backtest time, the engine enumerates candidate
OCC symbols within ±``width_pct`` of spot and fetches each. Empty
fetches discover what didn't trade — they're expected, not errors.

Default fetcher is :func:`src.options.polygon.fetch_history` since
Section 2.5 (May 2026); Tradier's per-OCC history returns null for
expired contracts at all plan tiers, so historical paths route through
Polygon while live/paper-trade paths continue to use Tradier. The
fetcher is injectable, so concentration analysis and tests can pass
deterministic stubs.

Strike spacing follows the OCC standard:
    SPX:               $5 (deep wings sometimes $25; ignored in v1)
    Equities + ETFs:   $1 below $25 spot, $2.50 in [25, 200), $5 above $200

``select_strike`` reconstructs IV from the close price via Section 3's
``implied_vol`` solver and picks the candidate whose computed delta is
closest to the caller's target. Callers pass a magnitude (positive
0.30 for "30 delta"); the function flips the sign internally per
``option_type`` so put deltas (which are negative) compare correctly.
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Callable, Optional

import pandas as pd
import requests

from src.options.greeks import delta as bsm_delta
from src.options.greeks import implied_vol as bsm_implied_vol
from src.options.greeks import time_to_expiration
from src.options.occ import generate_occ_symbol
from src.options.polygon import fetch_history
from src.options.tradier import RateLimiter
from src.options.types import ContractSpec


__all__ = [
    "get_strike_spacing",
    "reconstruct_chain",
    "select_strike",
    "DEFAULT_WIDTH_PCT",
    "DEFAULT_DELTA_TOLERANCE",
    "TRANSIENT_FETCH_EXCEPTIONS",
    "CHAIN_RECONSTRUCTION_WORKER_COUNT",
]


logger = logging.getLogger(__name__)

# Exception types we treat as expected per-OCC noise during chain
# reconstruction (a candidate strike that didn't trade, a transient
# network blip). Anything outside this tuple — RuntimeError,
# KeyError, configuration mistakes, etc. — re-raises so a
# misconfigured study fails fast instead of stalling silently.
TRANSIENT_FETCH_EXCEPTIONS: tuple[type[BaseException], ...] = (
    requests.HTTPError,
    requests.Timeout,
    requests.ConnectionError,
)


DEFAULT_WIDTH_PCT: float = 0.20
DEFAULT_DELTA_TOLERANCE: float = 0.10

# Worker count for the per-OCC fetch pool inside reconstruct_chain.
# Sequential fetches at ~0.5-2s each made the smoke run for hours;
# 16 workers brings ~92 candidate fetches into the few-seconds range.
# Polygon Options Developer plan tolerates the burst comfortably; the
# Polygon RateLimiter (in engine deps) caps total throughput.
CHAIN_RECONSTRUCTION_WORKER_COUNT: int = 16


HistoryFetcher = Callable[..., pd.DataFrame]


def get_strike_spacing(underlying: str, spot: float) -> float:
    """OCC standard strike spacing. SPX is $5 across the board; equity
    and ETF chains step at $1 / $2.50 / $5 by spot price."""
    if spot <= 0:
        raise ValueError(f"spot must be > 0; got {spot!r}")
    if underlying == "SPX":
        return 5.0
    if spot < 25.0:
        return 1.0
    if spot < 200.0:
        return 2.50
    return 5.0


def _strike_grid(
    spot: float, spacing: float, width_pct: float
) -> list[float]:
    """Return all strikes in [spot * (1 - width_pct), spot * (1 + width_pct)]
    on the spacing grid. Strikes are aligned to multiples of spacing."""
    if width_pct <= 0:
        raise ValueError(
            f"width_pct must be > 0; got {width_pct!r}"
        )
    low = spot * (1.0 - width_pct)
    high = spot * (1.0 + width_pct)
    start = math.floor(low / spacing) * spacing
    end = math.ceil(high / spacing) * spacing
    grid: list[float] = []
    k = start
    # Floating-point: build via multiplication index instead of repeated
    # addition to avoid drift over wide ranges.
    n = int(round((end - start) / spacing)) + 1
    for i in range(n):
        candidate = round(start + i * spacing, 6)
        if candidate > 0:
            grid.append(candidate)
    return grid


def reconstruct_chain(
    underlying: str,
    sim_date: date,
    target_expiration: date,
    spot: float,
    *,
    width_pct: float = DEFAULT_WIDTH_PCT,
    fetcher: Optional[HistoryFetcher] = None,
    limiter: Optional[RateLimiter] = None,
) -> list[tuple[ContractSpec, float]]:
    """Enumerate candidate OCC symbols around ``spot`` and fetch each
    from Tradier. Return non-empty hits as ``(ContractSpec, close)``.

    For each strike on the spacing grid in ``[spot * (1 - width_pct),
    spot * (1 + width_pct)]`` and each option type (call, put), build
    the OCC symbol via :func:`generate_occ_symbol` and fetch a 1-day
    history for ``sim_date``. Symbols that didn't trade return empty
    frames — those are filtered out (this is how the engine discovers
    which strikes actually existed; it's expected behavior, not an
    error).

    The fetcher is injectable for tests. The default uses
    :func:`src.options.tradier.fetch_history` (with cache; first run is
    expensive, subsequent runs hit the parquet cache).
    """
    if target_expiration <= sim_date:
        raise ValueError(
            f"target_expiration ({target_expiration.isoformat()}) must "
            f"be after sim_date ({sim_date.isoformat()})"
        )
    fetcher = fetcher or fetch_history
    spacing = get_strike_spacing(underlying, spot)
    strikes = _strike_grid(spot, spacing, width_pct)

    # Build the candidate list upfront so we can submit them all to
    # the executor and as_completed-process the results.
    candidates: list[tuple[ContractSpec, str]] = []
    for strike in strikes:
        for option_type in ("C", "P"):
            spec = ContractSpec(
                underlying=underlying,
                expiration_date=target_expiration,
                option_type=option_type,
                strike=strike,
            )
            candidates.append((spec, generate_occ_symbol(spec)))

    # Track which transient-exception types we've already INFO-logged
    # so an analyst sees the first occurrence of each condition
    # without a per-strike spam wall (~80 candidates/day). Locked
    # because workers may concurrently observe the first occurrence.
    seen_transient_types: set[str] = set()
    seen_lock = threading.Lock()

    results: list[tuple[ContractSpec, float]] = []
    with ThreadPoolExecutor(
        max_workers=CHAIN_RECONSTRUCTION_WORKER_COUNT,
    ) as executor:
        futures = {
            executor.submit(
                _fetch_candidate,
                spec, occ, sim_date, fetcher, limiter,
                seen_transient_types, seen_lock, underlying,
            ): spec
            for spec, occ in candidates
        }
        # ``future.result()`` re-raises any non-transient exception
        # (RuntimeError, KeyError, ValueError, ...) so a misconfigured
        # study fails fast — same discipline as the sequential path.
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


def _fetch_candidate(
    spec: ContractSpec,
    occ: str,
    sim_date: date,
    fetcher: HistoryFetcher,
    limiter: Optional[RateLimiter],
    seen_transient_types: set[str],
    seen_lock: threading.Lock,
    underlying: str,
) -> Optional[tuple[ContractSpec, float]]:
    """Fetch a single candidate OCC. Workers run this in parallel.

    Transient exceptions (HTTPError/Timeout/ConnectionError) get the
    "INFO once per type, DEBUG thereafter" log treatment and the
    candidate is dropped. Non-transient exceptions propagate so
    misconfigured studies fail fast rather than silently produce empty
    chains.
    """
    try:
        df = fetcher(occ, sim_date, sim_date, limiter=limiter)
    except TRANSIENT_FETCH_EXCEPTIONS as exc:
        exc_type = type(exc).__name__
        with seen_lock:
            first_time = exc_type not in seen_transient_types
            if first_time:
                seen_transient_types.add(exc_type)
        if first_time:
            logger.info(
                "chain_reconstruction: %s on %s for %s "
                "(suppressing further %s logs at DEBUG)",
                exc_type, occ, underlying, exc_type,
            )
        else:
            logger.debug(
                "chain_reconstruction: %s on %s: %s",
                exc_type, occ, exc,
            )
        return None
    if df is None or df.empty:
        return None
    close = _close_on_date(df, sim_date)
    if close is None or close <= 0:
        return None
    return spec, float(close)


def _close_on_date(df: pd.DataFrame, sim_date: date) -> Optional[float]:
    """Pull the close price for ``sim_date`` from a history frame.

    Returns None if the frame doesn't have a row for ``sim_date``.
    The earlier "last available close" fallback masked a cache-key
    correctness bug — when a multi-day frame is cached and the
    requested ``sim_date`` isn't in the index, returning some other
    day's close produced wrong inputs to the engine. Now we say
    "no data for that day" explicitly; chain_reconstruction skips the
    candidate and the engine's per-day skip counters surface the
    coverage gap.
    """
    if "close" not in df.columns:
        return None
    if sim_date not in df.index:
        return None
    value = df.loc[sim_date, "close"]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return None if pd.isna(value) else float(value)


def select_strike(
    candidates: list[tuple[ContractSpec, float]],
    target_delta: float,
    option_type: str,
    spot: float,
    sim_date: date,
    r: float,
    q: float,
    *,
    delta_tolerance: float = DEFAULT_DELTA_TOLERANCE,
) -> Optional[ContractSpec]:
    """Pick the candidate whose computed delta is closest to ``target_delta``.

    ``target_delta`` is a magnitude; the function negates it for puts
    so the comparison matches BSM convention (put deltas are
    negative). Per candidate: reconstruct IV from the close via
    :func:`bsm_implied_vol`, compute :func:`bsm_delta`, then minimize
    ``|computed_delta - signed_target|``.

    Candidates whose IV solver fails (price below no-arbitrage bound or
    outside the bracket) are silently skipped — they're not selectable
    anyway.

    Returns None if the best candidate's delta is more than
    ``delta_tolerance`` from the signed target. Caller increments the
    skip counter under ``no_strike_within_tolerance``.
    """
    if option_type not in ("C", "P"):
        raise ValueError(
            f"option_type must be 'C' or 'P'; got {option_type!r}"
        )
    if target_delta <= 0:
        raise ValueError(
            "target_delta must be a positive magnitude; "
            f"got {target_delta!r}"
        )

    signed_target = target_delta if option_type == "C" else -target_delta

    best_spec: Optional[ContractSpec] = None
    best_diff = math.inf
    for spec, close in candidates:
        if spec.option_type != option_type:
            continue
        t = time_to_expiration(sim_date, spec.expiration_date)
        if t <= 0:
            continue
        try:
            vol = bsm_implied_vol(
                option_price=close,
                s=spot,
                k=spec.strike,
                t=t,
                r=r,
                q=q,
                option_type=option_type,
            )
        except (ValueError, RuntimeError):
            continue
        try:
            d = bsm_delta(spot, spec.strike, t, r, q, vol, option_type)
        except ValueError:
            continue
        diff = abs(d - signed_target)
        if diff < best_diff:
            best_diff = diff
            best_spec = spec

    if best_spec is None or best_diff > delta_tolerance:
        return None
    return best_spec
