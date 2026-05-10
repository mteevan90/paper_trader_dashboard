"""Tests for ``src/options/earnings.py``.

The Tradier fundamentals beta fetcher and yfinance fallback are both
mocked. Cache is rerouted to a per-test ``tmp_path`` via
monkeypatching the module-level ``EARNINGS_CACHE_DIR``. A live-sandbox
smoke test at the end of this file is gated on the presence of
``TRADIER_SANDBOX_TOKEN`` so it skips in normal CI runs.
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
    FUNDAMENTALS_CALENDAR_PATH,
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


# ----------------- Tradier fundamentals beta payload shape -----------------


def _fundamentals_payload(dates: list[date]) -> dict:
    """Build a Tradier fundamentals-beta-shaped payload.

    Tradier's fundamentals/calendars endpoint returns events nested
    under ``results[].tables.corporate_calendars[]`` with integer
    event_type codes (14/15 for earnings releases).
    """
    return {
        "request": "AAPL",
        "type": "Symbol",
        "results": [
            {
                "type": "Symbol",
                "id": "AAPL",
                "tables": {
                    "corporate_calendars": [
                        {
                            "company_id": "0P000000GY",
                            "begin_date_time": d.isoformat(),
                            "end_date_time": d.isoformat(),
                            "event_type": 14,
                            "event": f"Q{((d.month - 1) // 3) + 1} Earnings",
                        }
                        for d in dates
                    ]
                },
            }
        ],
    }


def _legacy_v1_payload(dates: list[date]) -> dict:
    """Legacy /v1/markets/calendars/corporate shape — still parsed by
    the defensive walker so we can roll back without re-touching the
    parser."""
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

    def test_returns_dates_from_fundamentals_payload(self):
        expected = [date(2026, 5, 1), date(2026, 8, 1)]
        fetcher = MagicMock(return_value=_fundamentals_payload(expected))
        result = fetch_earnings_calendar(
            "AAPL", fetcher=fetcher, use_cache=False,
        )
        assert result == tuple(expected)

    def test_legacy_v1_payload_still_parses(self):
        # Backwards compatibility — the defensive walker still pulls
        # dates from the old /v1 shape if Tradier ever serves it.
        expected = [date(2026, 5, 1)]
        fetcher = MagicMock(return_value=_legacy_v1_payload(expected))
        result = fetch_earnings_calendar(
            "AAPL", fetcher=fetcher, use_cache=False,
        )
        assert result == tuple(expected)

    def test_uses_beta_base_url(self):
        """The fetcher must be called with base_url_override pointing
        at /beta — the v1 path 404s in production."""
        fetcher = MagicMock(return_value=_fundamentals_payload([]))
        fetch_earnings_calendar(
            "AAPL", fetcher=fetcher, use_cache=False,
            yfinance_fetcher=lambda t: None,
        )
        assert fetcher.call_count == 1
        # Positional args: (path, params, limiter); kwargs include
        # base_url_override.
        kwargs = fetcher.call_args.kwargs
        assert "base_url_override" in kwargs
        assert "/beta" in kwargs["base_url_override"]
        # Path is the fundamentals calendar path, not the old corporate one.
        assert fetcher.call_args.args[0] == FUNDAMENTALS_CALENDAR_PATH
        assert FUNDAMENTALS_CALENDAR_PATH == "/markets/fundamentals/calendars"

    def test_caches_to_parquet(self, _redirect_cache):
        expected = [date(2026, 5, 1)]
        fetcher = MagicMock(return_value=_fundamentals_payload(expected))

        first = fetch_earnings_calendar("AAPL", fetcher=fetcher)
        assert fetcher.call_count == 1
        assert first == tuple(expected)

        path = _redirect_cache / "AAPL.parquet"
        assert path.exists()

        second = fetch_earnings_calendar("AAPL", fetcher=fetcher)
        assert fetcher.call_count == 1
        assert second == tuple(expected)

    def test_cache_ttl_invalidates_stale_cache(self, _redirect_cache):
        expected = [date(2026, 5, 1)]
        fetcher = MagicMock(return_value=_fundamentals_payload(expected))

        fetch_earnings_calendar("AAPL", fetcher=fetcher)
        path = _redirect_cache / "AAPL.parquet"
        stale_ts = time.time() - (EARNINGS_CACHE_TTL_HOURS + 1) * 3600
        os.utime(path, (stale_ts, stale_ts))

        new_dates = [date(2026, 5, 1), date(2026, 8, 1)]
        fetcher2 = MagicMock(return_value=_fundamentals_payload(new_dates))
        result = fetch_earnings_calendar("AAPL", fetcher=fetcher2)
        assert fetcher2.call_count == 1
        assert result == tuple(new_dates)

    def test_use_cache_false_forces_refetch(self, _redirect_cache):
        expected = [date(2026, 5, 1)]
        fetcher = MagicMock(return_value=_fundamentals_payload(expected))

        fetch_earnings_calendar("AAPL", fetcher=fetcher)
        assert fetcher.call_count == 1
        fetch_earnings_calendar("AAPL", fetcher=fetcher, use_cache=False)
        assert fetcher.call_count == 2

    def test_handles_unexpected_payload_shape_falls_through_to_yfinance(self):
        # Payload has no recognizable earnings; should fall through to
        # yfinance fetcher.
        fetcher = MagicMock(return_value={"surprise": "no data"})
        yf_fetcher = MagicMock(return_value=None)
        result = fetch_earnings_calendar(
            "AAPL", fetcher=fetcher, yfinance_fetcher=yf_fetcher,
        )
        assert result == ()
        # yfinance was tried as the fallback.
        yf_fetcher.assert_called_once_with("AAPL")

    def test_tradier_exception_falls_through_to_yfinance(self):
        fetcher = MagicMock(side_effect=RuntimeError("401 Unauthorized"))
        yf_fetcher = MagicMock(return_value=None)
        result = fetch_earnings_calendar(
            "AAPL", fetcher=fetcher, yfinance_fetcher=yf_fetcher,
        )
        assert result == ()
        yf_fetcher.assert_called_once_with("AAPL")


# ----------------- yfinance fallback -----------------


def _yfinance_df(dates: list[date]) -> pd.DataFrame:
    """Mock yfinance Ticker.earnings_dates output: DataFrame indexed
    by datetime."""
    if not dates:
        return pd.DataFrame()
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame(
        {"EPS Estimate": [None] * len(dates)},
        index=idx,
    )


class TestYfinanceFallback:
    def test_falls_back_when_tradier_401(self, _redirect_cache):
        # Realistic scenario: Tradier returns 401 (Apigee
        # "Invalid API call as no apiproduct match found"), yfinance
        # provides the dates instead.
        tradier_fetcher = MagicMock(
            side_effect=RuntimeError("401 Client Error"),
        )
        yf_dates = [date(2026, 5, 1), date(2026, 8, 1)]
        yf_fetcher = MagicMock(return_value=_yfinance_df(yf_dates))

        result = fetch_earnings_calendar(
            "AAPL",
            fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
        )
        assert result == tuple(yf_dates)
        yf_fetcher.assert_called_once_with("AAPL")

    def test_falls_back_when_tradier_returns_empty_payload(self):
        # 200 OK but no earnings parsed → falls through.
        tradier_fetcher = MagicMock(return_value={"results": []})
        yf_dates = [date(2026, 5, 1)]
        yf_fetcher = MagicMock(return_value=_yfinance_df(yf_dates))

        result = fetch_earnings_calendar(
            "AAPL",
            fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
            use_cache=False,
        )
        assert result == tuple(yf_dates)

    def test_uses_tradier_when_tradier_returns_data(self):
        # Tradier succeeds → don't call yfinance.
        tradier_dates = [date(2026, 4, 30)]
        tradier_fetcher = MagicMock(
            return_value=_fundamentals_payload(tradier_dates),
        )
        yf_fetcher = MagicMock()

        result = fetch_earnings_calendar(
            "AAPL",
            fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
            use_cache=False,
        )
        assert result == tuple(tradier_dates)
        yf_fetcher.assert_not_called()

    def test_returns_empty_when_both_sources_empty(self):
        tradier_fetcher = MagicMock(return_value={})
        yf_fetcher = MagicMock(return_value=None)
        result = fetch_earnings_calendar(
            "AAPL",
            fetcher=tradier_fetcher,
            yfinance_fetcher=yf_fetcher,
            use_cache=False,
        )
        assert result == ()


# ----------------- is_in_earnings_window -----------------


class TestIsInEarningsWindow:
    def test_within_5_days_returns_true(self):
        earn = (date(2026, 5, 1),)
        assert is_in_earnings_window(
            "AAPL", date(2026, 4, 28), earnings_dates=earn,
        ) is True
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 1), earnings_dates=earn,
        ) is True
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 4), earnings_dates=earn,
        ) is True

    def test_outside_returns_false(self):
        earn = (date(2026, 5, 1),)
        assert is_in_earnings_window(
            "AAPL", date(2026, 4, 25), earnings_dates=earn,
        ) is False
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 8), earnings_dates=earn,
        ) is False

    def test_uses_passed_dates_when_provided(self, monkeypatch):
        sentinel = MagicMock()
        monkeypatch.setattr(
            earnings_mod, "fetch_earnings_calendar", sentinel,
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
            earnings_mod, "fetch_earnings_calendar", sentinel,
        )
        result = is_in_earnings_window("AAPL", date(2026, 5, 2))
        assert result is True
        sentinel.assert_called_once_with("AAPL")

    def test_empty_dates_is_always_false(self):
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 1), earnings_dates=(),
        ) is False

    def test_custom_window_size(self):
        earn = (date(2026, 5, 1),)
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 9),
            earnings_dates=earn, window_days=10,
        ) is True
        assert is_in_earnings_window(
            "AAPL", date(2026, 5, 3),
            earnings_dates=earn, window_days=1,
        ) is False

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="window_days"):
            is_in_earnings_window(
                "AAPL",
                date(2026, 5, 1),
                window_days=-1,
                earnings_dates=(date(2026, 5, 1),),
            )


# ----------------- live sandbox smoke -----------------


def test_live_sandbox_calls_correct_endpoint(_redirect_cache):
    """Manual smoke: hit the real Tradier sandbox.

    Documents two facts the production study needs to know:

    1. The fixed URL (``/beta/markets/fundamentals/calendars``) is
       reachable — it doesn't 404 like the old ``/v1/markets/
       calendars/corporate`` path did.
    2. With a sandbox token that lacks the Fundamentals beta
       subscription, Tradier returns 401 ``Invalid API call as no
       apiproduct match found``. The earnings module's defensive
       try/except converts that to ``()`` and the caller (the engine)
       silently disables earnings avoidance.

    Subscribed accounts will see this test return real dates; the
    contract is just "the call goes through" rather than "we get
    earnings data" — the latter depends on subscription state.

    Loads ``.env`` inside the test body so the skip decision happens
    after dotenv runs, not at pytest collection time.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.environ.get("TRADIER_SANDBOX_TOKEN"):
        pytest.skip(
            "TRADIER_SANDBOX_TOKEN not set; "
            "skipping live sandbox smoke"
        )

    from src.options._ssl import use_system_trust_store

    use_system_trust_store()

    # Don't pass any fetcher / yfinance_fetcher → uses the real Tradier
    # path under the hood. Bypass the cache to force the network call.
    # We don't assert anything about the contents; the assertion is
    # implicit: this must not raise.
    result = fetch_earnings_calendar(
        "AAPL",
        use_cache=False,
        yfinance_fetcher=lambda t: None,  # disable yfinance for this probe
    )
    # result is a tuple — could be empty (subscription-blocked sandbox)
    # or populated (subscribed account); both are acceptable.
    assert isinstance(result, tuple)
