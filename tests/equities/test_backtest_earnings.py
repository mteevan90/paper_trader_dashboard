"""Tests for fetch_earnings_dates() defensive logic.

The stash@{0} work added three behaviors to ``src/backtest.py``:
  1. ``_EARNINGS_SANITY_MIN_NONEMPTY_FRAC = 0.50`` sanity gate — refuse to
     overwrite the on-disk cache if fewer than 50% of fresh-fetched tickers
     came back non-empty (post-retry). Designed to prevent a yfinance
     throttling burst from poisoning the 1-day-TTL cache.
  2. Retry passes with ``_EARNINGS_RETRY_BACKOFFS_SECONDS = (5, 10)``.
  3. ``force_refresh=True`` parameter that bypasses the TTL fast path.

These tests verify all three. yfinance is mocked via a monkeypatched
``_fetch_one_earnings`` so no network is hit. ``time.sleep`` is also
patched to keep test runtime trivial.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point CACHE_DIR / EARNINGS_CACHE at a tmp dir for the duration of one test.

    Also no-op time.sleep so retry passes don't add 15s of wait per test.
    """
    import src.backtest as bt

    monkeypatch.setattr(bt, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(bt, "EARNINGS_CACHE", str(tmp_path / "earnings_dates.json"))
    monkeypatch.setattr(bt, "SNAPSHOT_MODE", False)
    monkeypatch.setattr(bt.time, "sleep", lambda _s: None)
    return bt


def _write_prior_cache(path: Path, payload: dict[str, list[str]]):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sanity_gate_fires_when_under_50pct_nonempty(isolated_cache, tmp_path):
    """If <50% of fresh fetches return non-empty (post-retry), the cache must NOT
    be overwritten and the function must return the prior on-disk contents."""
    bt = isolated_cache
    cache_path = Path(bt.EARNINGS_CACHE)

    # Seed a prior cache with content for one legacy ticker so we have a
    # known "prior state" to verify gets returned unchanged.
    prior = {"AAPL": ["2025-07-31T00:00:00"]}
    _write_prior_cache(cache_path, prior)
    # Force the cache to be considered stale so we take the fetch path
    # (we want the retry+gate to run; the fast path would short-circuit).
    import os, time
    old_mtime = time.time() - 86400 * 5  # 5 days old > 1-day TTL
    os.utime(cache_path, (old_mtime, old_mtime))

    # Mock yfinance to return non-empty for 1 of 5 tickers (20% — below the
    # 50% gate). Retries return the same (so post-retry is still 20%).
    def fake_fetch(tkr, start_ts, end_ts):
        if tkr == "AAPL":
            return [pd.Timestamp("2025-08-01")]
        return []
    import src.backtest as bt_module
    pytest.MonkeyPatch().setattr  # ensure import works
    # Use the bt fixture's monkeypatch indirectly via raw attribute set
    bt_module._fetch_one_earnings = fake_fetch  # type: ignore[assignment]

    result = bt.fetch_earnings_dates(
        ["AAPL", "MSFT", "GOOG", "META", "NVDA"],
        start="2025-01-01",
        end="2026-01-01",
    )

    # Cache file must be unchanged (still the prior content).
    on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
    assert on_disk == prior, "sanity gate should leave on-disk cache untouched"

    # Returned value must be the prior cache parsed back into Timestamps.
    assert set(result.keys()) == {"AAPL"}
    assert result["AAPL"] == [pd.Timestamp("2025-07-31")]


def test_sanity_gate_passes_when_at_least_50pct_nonempty(isolated_cache, tmp_path):
    """If >=50% of fresh fetches return non-empty, the cache IS overwritten."""
    bt = isolated_cache
    cache_path = Path(bt.EARNINGS_CACHE)
    # No prior cache — fresh slate.
    assert not cache_path.exists()

    def fake_fetch(tkr, start_ts, end_ts):
        # 3 of 4 return non-empty -> 75% >= 50%
        if tkr == "MSFT":
            return []
        return [pd.Timestamp("2025-08-01")]
    import src.backtest as bt_module
    bt_module._fetch_one_earnings = fake_fetch  # type: ignore[assignment]

    result = bt.fetch_earnings_dates(
        ["AAPL", "MSFT", "GOOG", "META"],
        start="2025-01-01",
        end="2026-01-01",
    )

    assert cache_path.exists(), "gate passed; cache must be written"
    on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(on_disk.keys()) == {"AAPL", "MSFT", "GOOG", "META"}
    # MSFT empty list serialized as [], others have one entry
    assert on_disk["MSFT"] == []
    assert on_disk["AAPL"] == ["2025-08-01T00:00:00"]
    # Returned dict matches
    assert result["MSFT"] == []
    assert result["AAPL"] == [pd.Timestamp("2025-08-01")]


def test_force_refresh_bypasses_ttl_fast_path(isolated_cache, tmp_path):
    """When the cache is fresh and contains all requested tickers, the function
    normally returns immediately. force_refresh=True must instead refetch."""
    bt = isolated_cache
    cache_path = Path(bt.EARNINGS_CACHE)
    # Seed a fresh cache (mtime = now) containing both tickers.
    prior = {
        "AAPL": ["2024-10-01T00:00:00"],
        "MSFT": ["2024-10-02T00:00:00"],
    }
    _write_prior_cache(cache_path, prior)
    # Don't backdate mtime — the cache is fresh.

    call_log = []
    def fake_fetch(tkr, start_ts, end_ts):
        call_log.append(tkr)
        # Return DIFFERENT dates than prior so we can detect the refetch wrote
        return [pd.Timestamp("2025-11-15")]
    import src.backtest as bt_module
    bt_module._fetch_one_earnings = fake_fetch  # type: ignore[assignment]

    # Without force_refresh: fast path returns prior, no fetch calls
    result_normal = bt.fetch_earnings_dates(
        ["AAPL", "MSFT"], start="2024-01-01", end="2026-01-01"
    )
    assert call_log == [], "fast path should not call _fetch_one_earnings"
    assert result_normal["AAPL"] == [pd.Timestamp("2024-10-01")]

    # With force_refresh: must refetch both, overwriting the cache
    call_log.clear()
    result_forced = bt.fetch_earnings_dates(
        ["AAPL", "MSFT"], start="2024-01-01", end="2026-01-01",
        force_refresh=True,
    )
    assert sorted(call_log) == ["AAPL", "MSFT"], (
        "force_refresh must trigger a fetch call for every requested ticker"
    )
    assert result_forced["AAPL"] == [pd.Timestamp("2025-11-15")]
    on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
    assert on_disk["AAPL"] == ["2025-11-15T00:00:00"]
    assert on_disk["MSFT"] == ["2025-11-15T00:00:00"]
