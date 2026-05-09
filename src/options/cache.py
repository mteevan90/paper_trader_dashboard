"""Parquet cache for Tradier OHLCV history and chain snapshots.

Layout:
    models/cache/options/tradier/history/<symbol>.parquet
    models/cache/options/tradier/chains/<ticker>_<expiration>_<run_date>.parquet

Two distinct caching semantics, deliberately documented here because the
options Section 2 design picks them rather than mirroring an existing
equity TTL pattern (equity ``data_source.py`` doesn't use TTL-based
caching; it relies on snapshot-time freezes for reproducibility).

History cache (``cache_history``):
    1-day TTL. ``read_history`` returns the cached frame only if the
    file's mtime is within ``HISTORY_CACHE_TTL_HOURS`` of now; otherwise
    it returns ``None`` and callers re-fetch. The TTL exists because the
    most recent days of a price series can be revised post-close
    (settlements, corrections); a 24h refresh window catches those
    without forcing a refetch on every backtest run. Snapshot
    promotion (Section 7+) freezes cache state into
    ``models/snapshots/options/<snapshot>/`` for reproducibility,
    bypassing the live cache entirely.

Chain snapshot cache (``cache_chain_snapshot``):
    Immutable per file. Each call writes a new file with a ``run_date``
    suffix in the filename — the chain at run_date is the chain at
    run_date and never gets updated in place. ``read_chain_snapshot``
    without an explicit ``run_date`` returns the most-recent on disk.
    Multi-run accumulation is enabled but no scheduled harvester
    exists in v1; harvester is deferred to v1.1 per §10 of the memo.

Sanity gate applies on history writes only (chain snapshots are
single-day, so the time-series coverage gate doesn't apply).
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.options.sanity_gate import passes_sanity_gate

_CACHE_DIR = Path("models") / "cache" / "options" / "tradier"
_HISTORY_DIR = _CACHE_DIR / "history"
_CHAINS_DIR = _CACHE_DIR / "chains"

HISTORY_CACHE_TTL_HOURS = 24


class SanityGateFailure(Exception):
    """Raised by :func:`cache_history` when the gate refuses a write."""


def _history_path(symbol: str) -> Path:
    return _HISTORY_DIR / f"{symbol}.parquet"


def _chain_path(ticker: str, expiration: date, run_date: date) -> Path:
    return _CHAINS_DIR / f"{ticker}_{expiration.isoformat()}_{run_date.isoformat()}.parquet"


def _expected_days_from_df(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    start, end = df.index.min(), df.index.max()
    return (end - start).days + 1


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < HISTORY_CACHE_TTL_HOURS


def cache_history(symbol: str, df: pd.DataFrame) -> Path:
    """Gate, then write ``df`` to the history cache. Returns the written path.

    Raises :class:`SanityGateFailure` and writes nothing if the gate
    rejects ``df``. Creates parent directories as needed.
    """
    expected_days = _expected_days_from_df(df)
    passed, reason = passes_sanity_gate(df, expected_days)
    if not passed:
        raise SanityGateFailure(f"{symbol}: {reason}")
    path = _history_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def read_history(symbol: str) -> Optional[pd.DataFrame]:
    """Return the cached history if present and within TTL; ``None`` otherwise.

    A cache file older than ``HISTORY_CACHE_TTL_HOURS`` is treated as a
    miss so callers re-fetch and pick up any recent-day revisions.
    """
    path = _history_path(symbol)
    if not _is_fresh(path):
        return None
    return pd.read_parquet(path)


def cache_chain_snapshot(
    ticker: str,
    expiration: date,
    df: pd.DataFrame,
    *,
    run_date: Optional[date] = None,
) -> Path:
    """Write a chain snapshot. No gate (single-day data, not a time series).

    Each call writes a new file with the ``run_date`` (defaults to today)
    in the filename so multiple runs accumulate without clobbering.
    """
    if run_date is None:
        run_date = date.today()
    path = _chain_path(ticker, expiration, run_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def read_chain_snapshot(
    ticker: str,
    expiration: date,
    run_date: Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """Return a chain snapshot. With ``run_date``, return that exact run
    or ``None``. Without, return the most-recent run for this
    (ticker, expiration) pair, or ``None`` if none exist.
    """
    if run_date is not None:
        path = _chain_path(ticker, expiration, run_date)
        return pd.read_parquet(path) if path.exists() else None

    if not _CHAINS_DIR.exists():
        return None
    prefix = f"{ticker}_{expiration.isoformat()}_"
    candidates = sorted(_CHAINS_DIR.glob(f"{prefix}*.parquet"))
    if not candidates:
        return None
    return pd.read_parquet(candidates[-1])
