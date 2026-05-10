"""Tests for ``src/options/benchmarks.py`` (Phase 2 Section 8).

Tradier and yfinance fetchers are both mocked. Cache is rerouted to
``tmp_path`` per test via monkeypatching the module-level
``BENCHMARKS_CACHE_DIR``.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.options import benchmarks as benchmarks_mod
from src.options.benchmarks import (
    BENCHMARKS_CACHE_TTL_HOURS,
    BxmSource,
    fetch_bxm,
    fetch_spy_total_return,
)


@pytest.fixture(autouse=True)
def _redirect_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "benchmarks"
    monkeypatch.setattr(
        benchmarks_mod, "BENCHMARKS_CACHE_DIR", cache_dir,
    )
    return cache_dir


def _spy_history_df(closes: list[float], start: date) -> pd.DataFrame:
    idx = [start + timedelta(days=i) for i in range(len(closes))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=pd.Index(idx, name="date"),
    )


# ----------------- fetch_spy_total_return -----------------


class TestFetchSpyTotalReturn:
    def test_basic_returns_dataframe_with_three_columns(self):
        history = _spy_history_df(
            [400.0 + i for i in range(40)], date(2025, 6, 2),
        )
        history_fetcher = MagicMock(return_value=history)
        dividend_fetcher = MagicMock(return_value={"calendars": {}})
        out = fetch_spy_total_return(
            date(2025, 6, 2), date(2025, 7, 11),
            history_fetcher=history_fetcher,
            dividend_fetcher=dividend_fetcher,
            use_cache=False,
        )
        assert list(out.columns) == [
            "close", "dividend_per_share", "total_return_index",
        ]
        assert len(out) == 40
        # First day TRI = 1.0
        assert out["total_return_index"].iloc[0] == 1.0

    def test_dividend_reinvestment_bumps_total_return_index(self):
        # Flat closes at 400, except a $4 dividend on day 5.
        history = _spy_history_df([400.0] * 30, date(2025, 6, 2))
        history_fetcher = MagicMock(return_value=history)
        # Calendar payload with one dividend event.
        dividend_payload = {
            "calendars": {
                "calendar": {
                    "events": {
                        "event": [
                            {
                                "event_type": "Dividend",
                                "ex_date": (date(2025, 6, 2) + timedelta(days=5)).isoformat(),
                                "amount": 4.0,
                            }
                        ]
                    }
                }
            }
        }
        dividend_fetcher = MagicMock(return_value=dividend_payload)
        out = fetch_spy_total_return(
            date(2025, 6, 2), date(2025, 7, 1),
            history_fetcher=history_fetcher,
            dividend_fetcher=dividend_fetcher,
            use_cache=False,
        )
        # On the ex-div day, total_return_index should bump by ~4/400 = 1%
        # vs the prior day.
        prior_tri = out["total_return_index"].iloc[4]
        ex_div_tri = out["total_return_index"].iloc[5]
        # Closes flat (price_return=0); div_return = 4/400 = 0.01
        assert ex_div_tri == pytest.approx(prior_tri * 1.01, rel=1e-6)

    def test_caches_to_parquet(self):
        history = _spy_history_df([400.0] * 35, date(2025, 6, 2))
        history_fetcher = MagicMock(return_value=history)
        dividend_fetcher = MagicMock(return_value={})

        first = fetch_spy_total_return(
            date(2025, 6, 2), date(2025, 7, 6),
            history_fetcher=history_fetcher,
            dividend_fetcher=dividend_fetcher,
        )
        assert history_fetcher.call_count == 1
        # Second call should hit cache.
        second = fetch_spy_total_return(
            date(2025, 6, 2), date(2025, 7, 6),
            history_fetcher=history_fetcher,
            dividend_fetcher=dividend_fetcher,
        )
        assert history_fetcher.call_count == 1
        pd.testing.assert_frame_equal(first, second)

    def test_raises_when_tradier_returns_empty(self):
        history_fetcher = MagicMock(return_value=pd.DataFrame())
        dividend_fetcher = MagicMock(return_value={})
        with pytest.raises(RuntimeError, match="empty SPY history"):
            fetch_spy_total_return(
                date(2025, 6, 2), date(2025, 7, 1),
                history_fetcher=history_fetcher,
                dividend_fetcher=dividend_fetcher,
                use_cache=False,
            )


# ----------------- fetch_bxm -----------------


def _bxm_df(closes: list[float], start: date) -> pd.DataFrame:
    idx = [start + timedelta(days=i) for i in range(len(closes))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [0] * len(closes),
        },
        index=pd.Index(idx, name="date"),
    )


class TestFetchBxm:
    def test_uses_tradier_when_available(self):
        tradier_fetcher = MagicMock(
            return_value=_bxm_df([400.0, 401.0, 402.0], date(2025, 6, 2)),
        )
        yf_fetcher = MagicMock()
        out = fetch_bxm(
            date(2025, 6, 2), date(2025, 6, 4),
            tradier_fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
            use_cache=False,
        )
        assert list(out.columns) == ["close"]
        assert len(out) == 3
        tradier_fetcher.assert_called_once()
        yf_fetcher.assert_not_called()

    def test_falls_back_to_yfinance(self):
        tradier_fetcher = MagicMock(return_value=pd.DataFrame())
        yf_fetcher = MagicMock(
            return_value=pd.DataFrame(
                {"close": [400.0, 401.0]},
                index=pd.Index(
                    [date(2025, 6, 2), date(2025, 6, 3)], name="date",
                ),
            ),
        )
        out = fetch_bxm(
            date(2025, 6, 2), date(2025, 6, 3),
            tradier_fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
            use_cache=False,
        )
        assert len(out) == 2
        tradier_fetcher.assert_called_once()
        yf_fetcher.assert_called_once()

    def test_returns_empty_when_both_sources_empty(self):
        tradier_fetcher = MagicMock(return_value=pd.DataFrame())
        yf_fetcher = MagicMock(return_value=pd.DataFrame())
        out = fetch_bxm(
            date(2025, 6, 2), date(2025, 6, 3),
            tradier_fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
            use_cache=False,
        )
        assert out.empty
        assert "close" in out.columns

    def test_caches_to_parquet(self):
        tradier_fetcher = MagicMock(
            return_value=_bxm_df([400.0, 401.0], date(2025, 6, 2)),
        )
        yf_fetcher = MagicMock()
        fetch_bxm(
            date(2025, 6, 2), date(2025, 6, 3),
            tradier_fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
        )
        # Second call hits cache.
        fetch_bxm(
            date(2025, 6, 2), date(2025, 6, 3),
            tradier_fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
        )
        assert tradier_fetcher.call_count == 1

    def test_cache_ttl_invalidates_after_7_days(self, _redirect_cache):
        tradier_fetcher = MagicMock(
            return_value=_bxm_df([400.0, 401.0], date(2025, 6, 2)),
        )
        yf_fetcher = MagicMock()
        fetch_bxm(
            date(2025, 6, 2), date(2025, 6, 3),
            tradier_fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
        )
        # Find cache file and stale its mtime.
        cache_files = list(_redirect_cache.glob("bxm_*.parquet"))
        assert len(cache_files) == 1
        stale_ts = time.time() - (BENCHMARKS_CACHE_TTL_HOURS + 1) * 3600
        os.utime(cache_files[0], (stale_ts, stale_ts))
        # Next call refetches.
        fetch_bxm(
            date(2025, 6, 2), date(2025, 6, 3),
            tradier_fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
        )
        assert tradier_fetcher.call_count == 2
