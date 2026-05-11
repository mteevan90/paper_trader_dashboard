"""Tests for src.equities.finnhub_fetcher.fetch_candles truncation behavior.

The Phase-3 smoke surfaced a bug where the OTC-tail-truncation clip was
applied to the returned DataFrame but NOT to the on-disk cache, so the
persisted parquet always contained the full Finnhub response including
post-delisting OTC pink-sheet candles. The fix moves the clip ahead of
``_atomic_write_parquet``.

These tests verify the four cases of truncation interacting with the
fresh-fetch and cache-hit paths. Network is fully mocked.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


def _make_candle_body(start_date: date, n_days: int) -> dict:
    """Build a /stock/candle response body with n_days of contiguous candles."""
    base = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    ts = [int((base + pd.Timedelta(days=i)).timestamp()) for i in range(n_days)]
    return {
        "s": "ok",
        "t": ts,
        "o": [100.0 + i * 0.1 for i in range(n_days)],
        "h": [101.0 + i * 0.1 for i in range(n_days)],
        "l": [99.0 + i * 0.1 for i in range(n_days)],
        "c": [100.5 + i * 0.1 for i in range(n_days)],
        "v": [1_000_000 + i for i in range(n_days)],
    }


@pytest.fixture
def isolated_finnhub(tmp_path, monkeypatch):
    """Route the fetcher's cache root to a tmp dir and stub out HTTP + auth."""
    import src.equities.finnhub_fetcher as ff

    monkeypatch.setattr(ff, "FINNHUB_CACHE_ROOT", tmp_path / "finnhub")
    monkeypatch.setattr(ff, "_resolve_token", lambda: "TEST_TOKEN")
    # Keep RateLimiter sleeps from showing up
    monkeypatch.setattr(ff.time, "sleep", lambda _s: None)
    return ff


def _install_mock_session(monkeypatch, ff, response_body: dict):
    """Patch requests.Session.get to return a 200 response with the given body."""
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {}
    fake_resp.json.return_value = response_body
    fake_resp.raise_for_status = MagicMock()

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            return fake_resp

    monkeypatch.setattr(ff.requests, "Session", FakeSession)


def test_truncate_within_series_clips_on_disk(isolated_finnhub, monkeypatch):
    """When truncate_at is in the middle of the series, the cached parquet
    must store the truncated form (not the full Finnhub response)."""
    ff = isolated_finnhub
    body = _make_candle_body(date(2023, 1, 1), 60)  # Jan 1 .. Mar 1, 2023
    _install_mock_session(monkeypatch, ff, body)

    truncate_at = date(2023, 1, 15)
    df = ff.fetch_candles(
        "SYM1", date(2023, 1, 1), date(2023, 3, 1),
        truncate_at=truncate_at,
    )
    # Returned dataframe is truncated
    assert df.index.max() <= truncate_at
    assert df.index.min() == date(2023, 1, 1)
    # AND on-disk parquet is truncated to the same shape
    cache_path = ff._price_cache_path("SYM1")
    assert cache_path.exists()
    cached = pd.read_parquet(cache_path)
    assert cached.index.max() <= truncate_at
    # Returned == cached (cache stores the same form we return)
    pd.testing.assert_frame_equal(df, cached, check_freq=False)


def test_truncate_after_series_end_is_noop(isolated_finnhub, monkeypatch):
    """If truncate_at is past the last available date, the clip should
    be a no-op and the cache should hold the full series."""
    ff = isolated_finnhub
    body = _make_candle_body(date(2023, 1, 1), 30)  # ends 2023-01-30
    _install_mock_session(monkeypatch, ff, body)

    truncate_at = date(2024, 12, 31)  # well past series end
    df = ff.fetch_candles(
        "SYM2", date(2023, 1, 1), date(2024, 12, 31),
        truncate_at=truncate_at,
    )
    assert len(df) == 30
    cache_path = ff._price_cache_path("SYM2")
    cached = pd.read_parquet(cache_path)
    assert len(cached) == 30
    assert cached.index.max() == date(2023, 1, 30)


def test_truncate_before_series_start_skips_cache(isolated_finnhub, monkeypatch):
    """If truncate_at is BEFORE the series starts (e.g., ticker delisted
    in 2017 but Finnhub returned a 2018+ series), the clip produces an
    empty DataFrame and the fetcher should NOT persist an empty parquet."""
    ff = isolated_finnhub
    body = _make_candle_body(date(2023, 1, 1), 30)
    _install_mock_session(monkeypatch, ff, body)

    truncate_at = date(2017, 9, 26)  # before series start
    df = ff.fetch_candles(
        "SYM3", date(2023, 1, 1), date(2023, 3, 1),
        truncate_at=truncate_at,
    )
    assert df.empty
    cache_path = ff._price_cache_path("SYM3")
    assert not cache_path.exists(), (
        "fetcher must skip the cache write when truncation yields an empty series"
    )


def test_truncate_at_none_caches_full_series(isolated_finnhub, monkeypatch):
    """When truncate_at is None (active ticker, no clip needed), the
    cache holds the full Finnhub response."""
    ff = isolated_finnhub
    body = _make_candle_body(date(2023, 1, 1), 30)
    _install_mock_session(monkeypatch, ff, body)

    df = ff.fetch_candles(
        "SYM4", date(2023, 1, 1), date(2023, 3, 1),
        truncate_at=None,
    )
    assert len(df) == 30
    cache_path = ff._price_cache_path("SYM4")
    cached = pd.read_parquet(cache_path)
    assert len(cached) == 30
    pd.testing.assert_frame_equal(df, cached, check_freq=False)


def test_cache_hit_returns_truncated_form_as_persisted(isolated_finnhub, monkeypatch):
    """The cache-hit path should return what's on disk verbatim. Since
    the on-disk form is already truncated, no additional clip is needed.
    Regression: the earlier buggy code applied a clip on read AND on
    write — the fix simplifies to write-once-clip-once."""
    ff = isolated_finnhub
    # Seed a cache directly with already-truncated data
    cache_path = ff._price_cache_path("SYM5")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    seeded = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [101.0, 102.0],
        "low":  [99.0, 100.0],
        "close": [100.5, 101.5],
        "volume": [1000, 2000],
    }, index=pd.Index([date(2023, 1, 1), date(2023, 1, 2)], name="date"))
    seeded.to_parquet(cache_path)

    # Pass a truncate_at that would have clipped further — but the cache
    # already represents the truncated form and we just return it.
    df = ff.fetch_candles(
        "SYM5", date(2023, 1, 1), date(2023, 1, 10),
        truncate_at=date(2023, 1, 1),  # this is informational only on cache-hit
    )
    assert len(df) == 2
    assert list(df.index) == [date(2023, 1, 1), date(2023, 1, 2)]
