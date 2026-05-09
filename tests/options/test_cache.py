"""Tests for src/options/cache.py — history TTL + chain-snapshot semantics."""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import pytest

from src.options import cache as cache_mod
from src.options.cache import (
    HISTORY_CACHE_TTL_HOURS,
    SanityGateFailure,
    cache_chain_snapshot,
    cache_history,
    read_chain_snapshot,
    read_history,
)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Reroute cache module-level paths to a per-test tmp dir."""
    cache_root = tmp_path / "tradier"
    history_dir = cache_root / "history"
    chains_dir = cache_root / "chains"
    monkeypatch.setattr(cache_mod, "_CACHE_DIR", cache_root)
    monkeypatch.setattr(cache_mod, "_HISTORY_DIR", history_dir)
    monkeypatch.setattr(cache_mod, "_CHAINS_DIR", chains_dir)
    return cache_root


def _history_df(n: int = 10, start_day: int = 1) -> pd.DataFrame:
    dates = [date(2024, 1, start_day + i) for i in range(n)]
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000_000 + i for i in range(n)],
        },
        index=pd.Index(dates, name="date"),
    )


def test_cache_history_writes_when_gate_passes(tmp_cache):
    df = _history_df(n=10)
    path = cache_history("SPY", df)
    assert path.exists()
    assert path.name == "SPY.parquet"
    roundtrip = pd.read_parquet(path)
    assert len(roundtrip) == 10


def test_cache_history_raises_and_skips_write_when_gate_fails(tmp_cache):
    df = _history_df(n=10)
    df.loc[df.index[1:], "close"] = float("nan")  # 1/10 = 10% < 50%
    with pytest.raises(SanityGateFailure):
        cache_history("SPY", df)
    assert not (tmp_cache / "history" / "SPY.parquet").exists()


def test_read_history_returns_none_when_missing(tmp_cache):
    assert read_history("SPY") is None


def test_read_history_returns_frame_when_fresh(tmp_cache):
    df = _history_df(n=10)
    cache_history("SPY", df)
    out = read_history("SPY")
    assert out is not None
    assert len(out) == 10


def test_read_history_returns_none_when_stale(tmp_cache, monkeypatch):
    """Cache file older than TTL must be treated as a miss."""
    df = _history_df(n=10)
    path = cache_history("SPY", df)
    stale_mtime = time.time() - (HISTORY_CACHE_TTL_HOURS + 1) * 3600
    import os
    os.utime(path, (stale_mtime, stale_mtime))
    assert read_history("SPY") is None


def test_cache_chain_snapshot_writes_run_dated_filename(tmp_cache):
    df = pd.DataFrame({"occ_symbol": ["SPY220617C00450000"], "strike": [450.0]})
    path = cache_chain_snapshot(
        "SPY", date(2026, 6, 19), df, run_date=date(2026, 5, 9),
    )
    assert path.name == "SPY_2026-06-19_2026-05-09.parquet"
    assert path.exists()


def test_read_chain_snapshot_returns_most_recent(tmp_cache):
    df_old = pd.DataFrame({"strike": [450.0]})
    df_new = pd.DataFrame({"strike": [455.0]})
    cache_chain_snapshot("SPY", date(2026, 6, 19), df_old, run_date=date(2026, 5, 1))
    cache_chain_snapshot("SPY", date(2026, 6, 19), df_new, run_date=date(2026, 5, 9))
    out = read_chain_snapshot("SPY", date(2026, 6, 19))
    assert out is not None
    assert out.iloc[0]["strike"] == 455.0


def test_read_chain_snapshot_returns_none_when_no_runs(tmp_cache):
    assert read_chain_snapshot("SPY", date(2026, 6, 19)) is None


def test_read_chain_snapshot_with_explicit_run_date(tmp_cache):
    df = pd.DataFrame({"strike": [450.0]})
    cache_chain_snapshot("SPY", date(2026, 6, 19), df, run_date=date(2026, 5, 9))
    out = read_chain_snapshot("SPY", date(2026, 6, 19), run_date=date(2026, 5, 9))
    assert out is not None
    out_missing = read_chain_snapshot("SPY", date(2026, 6, 19), run_date=date(2026, 1, 1))
    assert out_missing is None
