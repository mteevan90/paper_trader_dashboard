"""Tests for src/options/tradier.py.

No real network. All HTTP calls go through a fake ``requests.Session``
or are patched at the function level. ``time.sleep`` is patched so
retry tests run in milliseconds. Cache directory is rerouted via the
``tmp_cache`` fixture so writes do not pollute the repo.
"""

from __future__ import annotations

import time as time_mod
from datetime import date

import pandas as pd
import pytest
import requests

from src.options import cache as cache_mod
from src.options import tradier
from src.options.tradier import (
    PRODUCTION_BASE_URL,
    PRODUCTION_TOKEN_ENV,
    SANDBOX_BASE_URL,
    SANDBOX_TOKEN_ENV,
    TRADIER_ENV_VAR,
    RateLimiter,
    _http_get,
    _resolve_base_url,
    _resolve_token,
    fetch_chain_snapshot,
    fetch_expirations,
    fetch_history,
)


@pytest.fixture(autouse=True)
def _sandbox_token(monkeypatch):
    monkeypatch.setenv(SANDBOX_TOKEN_ENV, "test-sandbox-token")
    monkeypatch.setenv(TRADIER_ENV_VAR, "sandbox")


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "tradier"
    monkeypatch.setattr(cache_mod, "_CACHE_DIR", cache_root)
    monkeypatch.setattr(cache_mod, "_HISTORY_DIR", cache_root / "history")
    monkeypatch.setattr(cache_mod, "_CHAINS_DIR", cache_root / "chains")
    return cache_root


class _FakeResp:
    def __init__(self, json_payload, status_code=200, headers=None):
        self._json = json_payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"status={self.status_code}", response=self)


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({
            "url": url, "params": params, "headers": headers, "timeout": timeout,
        })
        if not self._responses:
            raise AssertionError("FakeSession ran out of programmed responses")
        return self._responses.pop(0)


# --- env / base URL / token ---------------------------------------------------


def test_resolve_base_url_defaults_to_sandbox(monkeypatch):
    monkeypatch.delenv(TRADIER_ENV_VAR, raising=False)
    assert _resolve_base_url() == SANDBOX_BASE_URL


def test_resolve_base_url_switches_to_production(monkeypatch):
    monkeypatch.setenv(TRADIER_ENV_VAR, "production")
    assert _resolve_base_url() == PRODUCTION_BASE_URL


def test_resolve_base_url_rejects_unknown_env(monkeypatch):
    monkeypatch.setenv(TRADIER_ENV_VAR, "staging")
    with pytest.raises(ValueError, match="staging"):
        _resolve_base_url()


def test_resolve_token_returns_production_token_when_env_is_production(monkeypatch):
    monkeypatch.setenv(TRADIER_ENV_VAR, "production")
    monkeypatch.setenv(PRODUCTION_TOKEN_ENV, "prod-token")
    assert _resolve_token() == "prod-token"


def test_resolve_token_raises_when_missing(monkeypatch):
    monkeypatch.delenv(SANDBOX_TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match=SANDBOX_TOKEN_ENV):
        _resolve_token()


# --- _http_get headers + retry ------------------------------------------------


def test_http_get_attaches_bearer_and_json_accept_headers():
    session = _FakeSession([_FakeResp({"ok": True})])
    _http_get("/markets/quotes", {"symbols": "SPY"}, RateLimiter(), session=session)
    headers = session.calls[0]["headers"]
    assert headers["Authorization"] == "Bearer test-sandbox-token"
    assert headers["Accept"] == "application/json"


def test_http_get_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(tradier.time, "sleep", lambda *_: None)
    session = _FakeSession([
        _FakeResp({}, status_code=429, headers={"Retry-After": "1"}),
        _FakeResp({"ok": True}),
    ])
    result = _http_get("/markets/quotes", {}, RateLimiter(), session=session)
    assert result == {"ok": True}
    assert len(session.calls) == 2


def test_http_get_retries_on_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr(tradier.time, "sleep", lambda *_: None)
    session = _FakeSession([
        _FakeResp({}, status_code=503),
        _FakeResp({"ok": True}),
    ])
    result = _http_get("/markets/quotes", {}, RateLimiter(), session=session)
    assert result == {"ok": True}


def test_http_get_does_not_retry_on_401(monkeypatch):
    monkeypatch.setattr(tradier.time, "sleep", lambda *_: None)
    session = _FakeSession([_FakeResp({}, status_code=401)])
    with pytest.raises(requests.HTTPError):
        _http_get("/markets/quotes", {}, RateLimiter(), session=session)
    assert len(session.calls) == 1


def test_http_get_calls_update_from_headers_on_success():
    headers = {"X-Ratelimit-Available": "5", "X-Ratelimit-Expiry": "9999999999"}
    session = _FakeSession([_FakeResp({"ok": True}, headers=headers)])
    limiter = RateLimiter()
    _http_get("/markets/quotes", {}, limiter, session=session)
    # Available > 1, so the limiter does not arm a header-driven sleep.
    assert limiter._sleep_until_epoch is None


# --- RateLimiter --------------------------------------------------------------


def test_rate_limiter_fallback_cap_sleeps_when_window_full(monkeypatch):
    sleeps: list[float] = []
    fake_now = [0.0]

    def fake_monotonic():
        return fake_now[0]

    def fake_sleep(s):
        sleeps.append(s)
        fake_now[0] += s

    monkeypatch.setattr(tradier.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(tradier.time, "sleep", fake_sleep)
    monkeypatch.setattr(tradier.time, "time", lambda: 0.0)

    limiter = RateLimiter(max_per_min=5)
    for _ in range(5):
        limiter.wait()
    assert sleeps == []
    limiter.wait()
    assert len(sleeps) == 1 and sleeps[0] > 0


def test_rate_limiter_header_driven_sleep_when_available_low(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(tradier.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(tradier.time, "time", lambda: 1000.0)

    limiter = RateLimiter()
    limiter.update_from_headers({
        "X-Ratelimit-Available": "0",
        "X-Ratelimit-Expiry": "1100",  # 100s in future
    })
    limiter.wait()
    assert any(s >= 100.0 for s in sleeps)


def test_rate_limiter_normalizes_millisecond_expiry(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(tradier.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(tradier.time, "time", lambda: 1000.0)

    limiter = RateLimiter()
    limiter.update_from_headers({
        "X-Ratelimit-Available": "0",
        "X-Ratelimit-Expiry": "1100000",  # ms representation of 1100s
    })
    limiter.wait()
    assert any(s >= 100.0 for s in sleeps)


def test_rate_limiter_ignores_partial_or_unparseable_headers():
    limiter = RateLimiter()
    limiter.update_from_headers({"X-Ratelimit-Available": "0"})  # no Expiry
    limiter.update_from_headers({"X-Ratelimit-Available": "x", "X-Ratelimit-Expiry": "y"})
    assert limiter._sleep_until_epoch is None


# --- fetch_history parsing ----------------------------------------------------


def _history_payload_multi_day():
    return {
        "history": {
            "day": [
                {"date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0,
                 "close": 100.5, "volume": 1_000_000},
                {"date": "2024-01-03", "open": 100.5, "high": 102.0, "low": 100.0,
                 "close": 101.5, "volume": 1_100_000},
            ]
        }
    }


def test_fetch_history_parses_multi_day_envelope(tmp_cache):
    session = _FakeSession([_FakeResp(_history_payload_multi_day())])
    df = fetch_history("SPY", date(2024, 1, 2), date(2024, 1, 3),
                       session=session, use_cache=False)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert len(df) == 2


def test_fetch_history_handles_single_day_dict_envelope(tmp_cache):
    """Tradier returns the day field as a dict (not a list) when there
    is only one day in the response."""
    payload = {
        "history": {
            "day": {"date": "2024-01-02", "open": 100.0, "high": 101.0,
                    "low": 99.0, "close": 100.5, "volume": 1_000_000}
        }
    }
    session = _FakeSession([_FakeResp(payload)])
    df = fetch_history("SPY", date(2024, 1, 2), date(2024, 1, 2),
                       session=session, use_cache=False)
    assert len(df) == 1


def test_fetch_history_handles_null_history(tmp_cache):
    session = _FakeSession([_FakeResp({"history": None})])
    df = fetch_history("SPY", date(2024, 1, 2), date(2024, 1, 3),
                       session=session, use_cache=False)
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_history_writes_cache_after_successful_fetch(tmp_cache):
    session = _FakeSession([_FakeResp(_history_payload_multi_day())])
    fetch_history("SPY", date(2024, 1, 2), date(2024, 1, 3),
                  session=session, use_cache=False)
    assert (tmp_cache / "history" / "SPY.parquet").exists()


def test_fetch_history_returns_cache_on_hit_without_http(tmp_cache):
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]},
        index=pd.Index([date(2024, 1, 2)], name="date"),
    )
    cache_mod.cache_history("SPY", df)
    session = _FakeSession([])  # zero responses queued — would error if called
    out = fetch_history("SPY", date(2024, 1, 2), date(2024, 1, 2),
                        session=session, use_cache=True)
    assert len(out) == 1
    assert session.calls == []


# --- fetch_expirations parsing ------------------------------------------------


def test_fetch_expirations_parses_list_envelope():
    payload = {"expirations": {"date": ["2026-06-19", "2026-07-17", "2026-09-18"]}}
    session = _FakeSession([_FakeResp(payload)])
    out = fetch_expirations("SPY", session=session)
    assert out == [date(2026, 6, 19), date(2026, 7, 17), date(2026, 9, 18)]


def test_fetch_expirations_handles_single_date_string_envelope():
    payload = {"expirations": {"date": "2026-06-19"}}
    session = _FakeSession([_FakeResp(payload)])
    out = fetch_expirations("SPY", session=session)
    assert out == [date(2026, 6, 19)]


def test_fetch_expirations_returns_empty_on_null():
    session = _FakeSession([_FakeResp({"expirations": None})])
    out = fetch_expirations("SPY", session=session)
    assert out == []


# --- fetch_chain_snapshot parsing ---------------------------------------------


def _chain_payload(with_greeks=True):
    option_a = {
        "symbol": "SPY260619C00450000", "option_type": "call", "strike": 450.0,
        "bid": 5.10, "ask": 5.20, "last": 5.15, "volume": 100, "open_interest": 1000,
        "contract_size": 100, "expiration_date": "2026-06-19",
        "expiration_type": "standard", "root_symbol": "SPY",
    }
    option_b = {
        "symbol": "SPY260619P00450000", "option_type": "put", "strike": 450.0,
        "bid": 4.90, "ask": 5.00, "last": 4.95, "volume": 80, "open_interest": 800,
        "contract_size": 100, "expiration_date": "2026-06-19",
        "expiration_type": "standard", "root_symbol": "SPY",
    }
    if with_greeks:
        option_a["greeks"] = {"delta": 0.55, "gamma": 0.02, "theta": -0.04,
                              "vega": 0.10, "rho": 0.05,
                              "bid_iv": 0.18, "mid_iv": 0.185, "ask_iv": 0.19}
        # option_b deliberately omits greeks to test NaN-fill.
    return {"options": {"option": [option_a, option_b]}}


def test_fetch_chain_snapshot_with_greeks_includes_greek_columns():
    session = _FakeSession([_FakeResp(_chain_payload(with_greeks=True))])
    df = fetch_chain_snapshot("SPY", date(2026, 6, 19), session=session)
    assert "delta" in df.columns
    assert df.loc[df["option_type"] == "call", "delta"].iloc[0] == 0.55
    # The put row has no greeks block — must NaN-fill.
    assert pd.isna(df.loc[df["option_type"] == "put", "delta"].iloc[0])


def test_fetch_chain_snapshot_without_greeks_drops_greek_columns():
    payload = _chain_payload(with_greeks=False)
    session = _FakeSession([_FakeResp(payload)])
    df = fetch_chain_snapshot("SPY", date(2026, 6, 19),
                              with_greeks=False, session=session)
    for greek in ("delta", "gamma", "theta", "vega", "rho"):
        assert greek not in df.columns


def test_fetch_chain_snapshot_handles_single_option_dict_envelope():
    payload = {"options": {"option": {
        "symbol": "SPY260619C00450000", "option_type": "call", "strike": 450.0,
        "bid": 5.10, "ask": 5.20, "last": 5.15, "volume": 100,
        "open_interest": 1000, "contract_size": 100,
        "expiration_date": "2026-06-19", "expiration_type": "standard",
        "root_symbol": "SPY",
    }}}
    session = _FakeSession([_FakeResp(payload)])
    df = fetch_chain_snapshot("SPY", date(2026, 6, 19),
                              with_greeks=False, session=session)
    assert len(df) == 1


def test_fetch_chain_snapshot_returns_empty_on_null_options():
    session = _FakeSession([_FakeResp({"options": None})])
    df = fetch_chain_snapshot("SPY", date(2026, 6, 19), session=session)
    assert df.empty
    assert "occ_symbol" in df.columns
