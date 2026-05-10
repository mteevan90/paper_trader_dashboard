"""Tests for ``src/options/earnings.py`` (Phase 2 Section 6).

The Tradier corporate-calendar fetcher is mocked throughout — no
network. Cache is rerouted to a per-test ``tmp_path`` via
monkeypatching the module-level ``EARNINGS_CACHE_DIR``.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.options import earnings as earnings_mod
from src.options.earnings import (
    EARNINGS_CACHE_TTL_HOURS,
    INDEX_TICKERS,
    fetch_earnings_calendar,
    is_in_earnings_window,
)


@pytest.fixture(autouse=True)
def _redirect_cache(monkeypatch, tmp_path):
    """Redirect earnings cache directory to a tmp path per test."""
    cache_dir = tmp_path / "earnings"
    monkeypatch.setattr(earnings_mod, "EARNINGS_CACHE_DIR", cache_dir)
    return cache_dir


def _payload_with_dates(dates: list[date]) -> dict:
    """Build a Tradier-shaped corporate-calendar payload."""
    return {
        "request": "x",
        "calendars": {
            "calendar": {
                "events": {
                    "event": [
                        {
                            "event_type": "Earnings",
                            "date": d.isoformat(),
                        }
                        for d in dates
                    ]
                }
            }
        },
    }


# ----------------- fetch_earnings_calendar -----------------


class TestFetchEarningsCalendar:
    def test_index_returns_empty_tuple(self):
        for ticker in INDEX_TICKERS:
            assert fetch_earnings_calendar(ticker) == ()

    def test_returns_dates_from_payload(self):
        expected = [date(2026, 5, 1), date(2026, 8, 1)]
        fetcher = MagicMock(return_value=_payload_with_dates(expected))
        result = fetch_earnings_calendar(
            "AAPL", fetcher=fetcher, use_cache=False,
        )
        assert result == tuple(expected)

    def test_caches_to_parquet(self, _redirect_cache):
        expected = [date(2026, 5, 1)]
        fetcher = MagicMock(return_value=_payload_with_dates(expected))

        first = fetch_earnings_calendar("AAPL", fetcher=fetcher)
        assert fetcher.call_count == 1
        assert first == tuple(expected)

        # Cache file should exist.
        path = _redirect_cache / "AAPL.parquet"
        assert path.exists()

        # Second call uses cache; fetcher not called again.
        second = fetch_earnings_calendar("AAPL", fetcher=fetcher)
        assert fetcher.call_count == 1
        assert second == tuple(expected)

    def test_cache_ttl_invalidates_stale_cache(self, _redirect_cache):
        expected = [date(2026, 5, 1)]
        fetcher = MagicMock(return_value=_payload_with_dates(expected))

        fetch_earnings_calendar("AAPL", fetcher=fetcher)
        path = _redirect_cache / "AAPL.parquet"
        # Backdate file mtime to >TTL hours ago.
        stale_ts = time.time() - (EARNINGS_CACHE_TTL_HOURS + 1) * 3600
        os.utime(path, (stale_ts, stale_ts))

        new_dates = [date(2026, 5, 1), date(2026, 8, 1)]
        fetcher2 = MagicMock(return_value=_payload_with_dates(new_dates))
        result = fetch_earnings_calendar("AAPL", fetcher=fetcher2)
        assert fetcher2.call_count == 1
        assert result == tuple(new_dates)

    def test_handles_unexpected_payload_shape(self, _redirect_cache):
        # Payload has no recognizable date fields; expect graceful empty.
        fetcher = MagicMock(return_value={"surprise": "no data"})
        result = fetch_earnings_calendar("AAPL", fetcher=fetcher)
        assert result == ()

    def test_handles_fetcher_exception(self, _redirect_cache):
        fetcher = MagicMock(side_effect=RuntimeError("network down"))
        result = fetch_earnings_calendar("AAPL", fetcher=fetcher)
        assert result == ()

    def test_use_cache_false_forces_refetch(self, _redirect_cache):
        expected = [date(2026, 5, 1)]
        fetcher = MagicMock(return_value=_payload_with_dates(expected))

        fetch_earnings_calendar("AAPL", fetcher=fetcher)
        assert fetcher.call_count == 1
        fetch_earnings_calendar("AAPL", fetcher=fetcher, use_cache=False)
        assert fetcher.call_count == 2


# ----------------- is_in_earnings_window -----------------


class TestIsInEarningsWindow:
    def test_within_5_days_returns_true(self):
        earn = (date(2026, 5, 1),)
        assert is_in_earnings_window(
            "AAPL", date(2026, 4, 28), earnings_dates=earn
        ) is True
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 1), earnings_dates=earn
        ) is True
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 4), earnings_dates=earn
        ) is True

    def test_outside_returns_false(self):
        earn = (date(2026, 5, 1),)
        assert is_in_earnings_window(
            "AAPL", date(2026, 4, 25), earnings_dates=earn
        ) is False
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 8), earnings_dates=earn
        ) is False

    def test_uses_passed_dates_when_provided(self, monkeypatch):
        # Spy on fetch_earnings_calendar — should NOT be called when
        # earnings_dates is passed.
        sentinel = MagicMock()
        monkeypatch.setattr(
            earnings_mod, "fetch_earnings_calendar", sentinel
        )
        result = is_in_earnings_window(
            "AAPL",
            date(2026, 5, 1),
            earnings_dates=(date(2026, 5, 1),),
        )
        assert result is True
        sentinel.assert_not_called()

    def test_falls_back_to_fetch_when_dates_none(self, monkeypatch):
        sentinel = MagicMock(return_value=(date(2026, 5, 1),))
        monkeypatch.setattr(
            earnings_mod, "fetch_earnings_calendar", sentinel
        )
        result = is_in_earnings_window("AAPL", date(2026, 5, 2))
        assert result is True
        sentinel.assert_called_once_with("AAPL")

    def test_empty_dates_is_always_false(self):
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 1), earnings_dates=()
        ) is False

    def test_custom_window_size(self):
        earn = (date(2026, 5, 1),)
        # 10-day window
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 9), earnings_dates=earn, window_days=10,
        ) is True
        # 1-day window
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 3), earnings_dates=earn, window_days=1,
        ) is False

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="window_days"):
            is_in_earnings_window(
                "AAPL",
                date(2026, 5, 1),
                window_days=-1,
                earnings_dates=(date(2026, 5, 1),),
            )
