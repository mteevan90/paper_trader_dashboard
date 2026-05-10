"""Tests for ``src/options/polygon.py`` (Phase 2 Section 2.5).

All offline. ``requests.Session`` is mocked via a small fake; the cache
directory is rerouted to ``tmp_path`` per test by monkeypatching the
module-level ``POLYGON_CACHE_ROOT``.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from src.options import polygon as polygon_mod
from src.options.polygon import (
    DEFAULT_MAX_RETRIES,
    HISTORY_CACHE_TTL_HOURS,
    POLYGON_API_KEY_ENV,
    POLYGON_BASE_URL,
    _occ_to_polygon_ticker,
    _resolve_token,
    fetch_history,
)
from src.options.tradier import RateLimiter


# ----------------- fixtures -----------------


@pytest.fixture(autouse=True)
def _set_token(monkeypatch):
    monkeypatch.setenv(POLYGON_API_KEY_ENV, "test-polygon-key")


@pytest.fixture(autouse=True)
def _redirect_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "polygon" / "history"
    monkeypatch.setattr(polygon_mod, "POLYGON_CACHE_ROOT", cache_dir)
    return cache_dir


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Replace time.sleep so retry tests don't actually wait."""
    monkeypatch.setattr(polygon_mod.time, "sleep", lambda *_: None)


class _FakeResp:
    def __init__(
        self,
        json_payload: Optional[dict] = None,
        status_code: int = 200,
        text: Optional[str] = None,
    ):
        self._json = json_payload
        self.status_code = status_code
        self.text = text if text is not None else (
            "" if json_payload is None else "ok"
        )
        self.headers: dict[str, str] = {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(
                f"HTTP {self.status_code}", response=self,
            )


class _FakeSession:
    def __init__(self, responses):
        # Each entry is a _FakeResp or an exception type/instance.
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self._responses:
            raise AssertionError("FakeSession ran out of responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        if isinstance(nxt, type) and issubclass(nxt, BaseException):
            raise nxt("simulated")
        return nxt


def _polygon_ok_payload(closes_by_date: dict[date, float]) -> dict:
    """Build a Polygon aggregates response with one bar per date."""
    results = []
    for d, close in sorted(closes_by_date.items()):
        # Eastern noon → ms-epoch in UTC
        ts = pd.Timestamp(d).tz_localize(
            "America/New_York"
        ).tz_convert("UTC")
        ts_ms = int(ts.timestamp() * 1000)
        results.append({
            "t": ts_ms,
            "o": float(close) - 0.5,
            "h": float(close) + 1.0,
            "l": float(close) - 1.0,
            "c": float(close),
            "v": 1000,
            "vw": float(close),
            "n": 50,
        })
    return {"status": "OK", "ticker": "O:SPY", "results": results}


# ----------------- OCC → Polygon ticker -----------------


class TestOccToPolygonTicker:
    def test_strips_padding(self):
        assert (
            _occ_to_polygon_ticker("SPY   240719C00540000")
            == "O:SPY240719C00540000"
        )

    def test_already_clean(self):
        assert (
            _occ_to_polygon_ticker("AAPL240719C00210000")
            == "O:AAPL240719C00210000"
        )

    def test_index_option(self):
        # SPXW or SPX standard symbols already have no internal spaces.
        assert (
            _occ_to_polygon_ticker("SPXW230217C04000000")
            == "O:SPXW230217C04000000"
        )

    def test_idempotent_when_already_prefixed(self):
        assert (
            _occ_to_polygon_ticker("O:SPY240719C00540000")
            == "O:SPY240719C00540000"
        )


# ----------------- token resolution -----------------


class TestResolveToken:
    def test_reads_from_env(self):
        assert _resolve_token() == "test-polygon-key"

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv(POLYGON_API_KEY_ENV, raising=False)
        with pytest.raises(RuntimeError, match=POLYGON_API_KEY_ENV):
            _resolve_token()


# ----------------- successful fetches -----------------


class TestFetchHistorySuccess:
    def test_basic_returns_ohlcv_dataframe(self):
        payload = _polygon_ok_payload({
            date(2024, 6, 17): 540.0,
            date(2024, 6, 18): 541.0,
            date(2024, 6, 19): 539.5,
            date(2024, 6, 20): 542.0,
        })
        session = _FakeSession([_FakeResp(payload)])
        df = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "date"
        assert len(df) == 4

    def test_results_sorted_by_date_ascending(self):
        payload = _polygon_ok_payload({
            date(2024, 6, 19): 539.5,
            date(2024, 6, 17): 540.0,
            date(2024, 6, 20): 542.0,
            date(2024, 6, 18): 541.0,
        })
        session = _FakeSession([_FakeResp(payload)])
        df = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        idx = list(df.index)
        assert idx == sorted(idx)

    def test_timestamp_conversion_to_eastern_date(self):
        # A bar tagged at midnight UTC on 2024-06-17 is still 2024-06-16
        # in Eastern time. Verify the conversion.
        ts_utc_midnight = pd.Timestamp(
            "2024-06-17T00:00:00", tz="UTC",
        )
        ts_ms = int(ts_utc_midnight.timestamp() * 1000)
        payload = {
            "status": "OK",
            "results": [{
                "t": ts_ms, "o": 1.0, "h": 1.0, "l": 1.0,
                "c": 1.0, "v": 1, "n": 1, "vw": 1.0,
            }],
        }
        session = _FakeSession([_FakeResp(payload)])
        df = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 16), date(2024, 6, 17),
            session=session, use_cache=False,
        )
        assert list(df.index) == [date(2024, 6, 16)]


# ----------------- empty results -----------------


class TestFetchHistoryEmpty:
    def test_empty_results_returns_empty_dataframe(self):
        session = _FakeSession([_FakeResp({"status": "OK", "results": []})])
        df = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_empty_results_does_not_write_cache(self, _redirect_cache):
        session = _FakeSession([_FakeResp({"status": "OK", "results": []})])
        fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=True,
        )
        assert not (_redirect_cache / "SPY240719C00540000.parquet").exists()


# ----------------- error handling -----------------


class TestFetchHistoryErrors:
    def test_403_unauthorized_raises_runtime_error(self):
        session = _FakeSession([_FakeResp(
            {"status": "ERROR", "message": "data timeframe outside plan"},
            status_code=403,
        )])
        with pytest.raises(RuntimeError, match="NOT_AUTHORIZED"):
            fetch_history(
                "SPY220117C00400000",
                date(2022, 1, 1), date(2022, 1, 17),
                session=session, use_cache=False,
            )

    def test_401_raises_runtime_error_token_issue(self):
        session = _FakeSession([_FakeResp(
            {"status": "ERROR"}, status_code=401,
        )])
        with pytest.raises(RuntimeError, match="authentication failed"):
            fetch_history(
                "SPY240719C00540000",
                date(2024, 6, 17), date(2024, 6, 21),
                session=session, use_cache=False,
            )

    def test_500_retries_then_raises_after_max_retries(self):
        responses = [
            _FakeResp(None, status_code=500) for _ in range(DEFAULT_MAX_RETRIES)
        ]
        session = _FakeSession(responses)
        with pytest.raises(requests.HTTPError):
            fetch_history(
                "SPY240719C00540000",
                date(2024, 6, 17), date(2024, 6, 21),
                session=session, use_cache=False,
            )
        # Used all retries.
        assert len(session.calls) == DEFAULT_MAX_RETRIES

    def test_503_retries_then_succeeds_on_second_attempt(self):
        good_payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        session = _FakeSession([
            _FakeResp(None, status_code=503),
            _FakeResp(good_payload),
        ])
        df = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        assert len(df) == 1
        assert len(session.calls) == 2

    def test_timeout_retries(self):
        good_payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        session = _FakeSession([
            requests.Timeout("slow"),
            _FakeResp(good_payload),
        ])
        df = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        assert len(df) == 1
        assert len(session.calls) == 2

    def test_connection_error_retries(self):
        good_payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        session = _FakeSession([
            requests.ConnectionError("network"),
            _FakeResp(good_payload),
        ])
        df = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        assert len(df) == 1


# ----------------- caching -----------------


class TestFetchHistoryCaching:
    def test_writes_cache_on_success_above_sanity_threshold(self, _redirect_cache):
        # 5-day window with 4 bars > 50% threshold → cache write.
        payload = _polygon_ok_payload({
            date(2024, 6, 17): 540.0,
            date(2024, 6, 18): 541.0,
            date(2024, 6, 19): 539.5,
            date(2024, 6, 20): 542.0,
        })
        session = _FakeSession([_FakeResp(payload)])
        fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=True,
        )
        assert (_redirect_cache / "SPY240719C00540000.parquet").exists()

    def test_reads_from_cache_when_present_and_use_cache_true(
        self, _redirect_cache,
    ):
        path = _redirect_cache / "SPY240719C00540000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df_cached = pd.DataFrame(
            {
                "open": [1.0], "high": [1.0], "low": [1.0],
                "close": [1.0], "volume": [10],
            },
            index=pd.Index([date(2024, 6, 17)], name="date"),
        )
        df_cached.to_parquet(path)
        # Empty session — would error if any HTTP call were attempted.
        session = _FakeSession([])
        result = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=True,
        )
        assert len(result) == 1
        assert session.calls == []

    def test_bypasses_cache_when_use_cache_false(self, _redirect_cache):
        path = _redirect_cache / "SPY240719C00540000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "open": [1.0], "high": [1.0], "low": [1.0],
                "close": [1.0], "volume": [10],
            },
            index=pd.Index([date(2024, 6, 17)], name="date"),
        ).to_parquet(path)
        # use_cache=False forces an HTTP call.
        payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        session = _FakeSession([_FakeResp(payload)])
        result = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        assert len(session.calls) == 1
        assert float(result["close"].iloc[0]) == 540.0

    def test_sparse_results_still_cached_no_sanity_gate(
        self, _redirect_cache,
    ):
        """Polygon's per-OCC fetches don't run the calendar-day
        sanity gate Tradier uses — option contracts trade sparsely
        and rejecting their writes would re-fetch every call. Anything
        non-empty is cached. (See cache-write block in polygon.py for
        the rationale.)"""
        # 1 bar in a 10-day window — this used to fail the sanity
        # gate; now it should still get cached.
        payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        session = _FakeSession([_FakeResp(payload)])
        fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 26),
            session=session, use_cache=True,
        )
        assert (_redirect_cache / "SPY240719C00540000.parquet").exists()

    def test_cache_ttl_invalidates_stale_cache(self, _redirect_cache):
        path = _redirect_cache / "SPY240719C00540000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "open": [1.0], "high": [1.0], "low": [1.0],
                "close": [99.0], "volume": [10],
            },
            index=pd.Index([date(2024, 6, 17)], name="date"),
        ).to_parquet(path)
        # Stale the file.
        stale_ts = time.time() - (HISTORY_CACHE_TTL_HOURS + 1) * 3600
        os.utime(path, (stale_ts, stale_ts))
        # Re-fetch should hit the network.
        payload = _polygon_ok_payload({
            date(2024, 6, 17): 540.0,
            date(2024, 6, 18): 541.0,
            date(2024, 6, 19): 539.5,
        })
        session = _FakeSession([_FakeResp(payload)])
        result = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 19),
            session=session, use_cache=True,
        )
        assert float(result["close"].iloc[0]) == 540.0


# ----------------- auth + limiter -----------------


class TestAuthAndLimiter:
    def test_passes_apikey_in_query_params(self):
        payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        session = _FakeSession([_FakeResp(payload)])
        fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        params = session.calls[0]["params"]
        assert params["apiKey"] == "test-polygon-key"
        assert params["adjusted"] == "true"
        assert params["sort"] == "asc"

    def test_url_uses_polygon_base_with_o_prefixed_ticker(self):
        payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        session = _FakeSession([_FakeResp(payload)])
        fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        url = session.calls[0]["url"]
        assert url.startswith(POLYGON_BASE_URL)
        assert "/v2/aggs/ticker/O:SPY240719C00540000/" in url
        assert "/range/1/day/2024-06-17/2024-06-21" in url

    def test_acquires_limiter_when_provided(self):
        limiter = MagicMock(spec=RateLimiter)
        payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        session = _FakeSession([_FakeResp(payload)])
        fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session,
            limiter=limiter,
            use_cache=False,
        )
        limiter.wait.assert_called_once()

    def test_no_limiter_when_none(self):
        payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        session = _FakeSession([_FakeResp(payload)])
        # Just verify it doesn't blow up.
        fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )


# ----------------- non-OK status -----------------


class TestNonOkStatus:
    def test_status_other_than_ok_or_delayed_raises(self):
        session = _FakeSession([_FakeResp({"status": "ERROR"})])
        with pytest.raises(RuntimeError, match="non-OK"):
            fetch_history(
                "SPY240719C00540000",
                date(2024, 6, 17), date(2024, 6, 21),
                session=session, use_cache=False,
            )

    def test_status_delayed_accepted(self):
        # Polygon sometimes returns "DELAYED" for very recent windows;
        # treat as success.
        payload = _polygon_ok_payload({date(2024, 6, 17): 540.0})
        payload["status"] = "DELAYED"
        session = _FakeSession([_FakeResp(payload)])
        df = fetch_history(
            "SPY240719C00540000",
            date(2024, 6, 17), date(2024, 6, 21),
            session=session, use_cache=False,
        )
        assert len(df) == 1
