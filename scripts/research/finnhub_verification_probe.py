"""Pre-Phase-2 verification probes.

Three checks Mike asked for before committing to the full Phase 2 build:

1. Delisted-ticker price coverage on /stock/candle — verify Finnhub Basic
   returns real history through the delisting date for 5 known names.
2. /calendar/earnings rate-limit bucket — confirm 60/min vs 150/min.
3. Historical earnings spot-check — verify /calendar/earnings returns
   real historical data (not just upcoming) and that dates match known
   reality.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
KEY = os.environ["FINNHUB_API_KEY"]
BASE = "https://finnhub.io/api/v1"
NOW = datetime.now(timezone.utc)


def call(path: str, params: dict, sleep: float = 0.5):
    p = dict(params)
    p["token"] = KEY
    t0 = time.perf_counter()
    r = requests.get(f"{BASE}{path}", params=p, timeout=30)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    time.sleep(sleep)
    body = None
    try:
        body = r.json()
    except Exception:
        body = r.text[:300]
    return r.status_code, body, {
        "elapsed_ms": elapsed,
        "limit": r.headers.get("X-Ratelimit-Limit"),
        "remaining": r.headers.get("X-Ratelimit-Remaining"),
        "retry_after": r.headers.get("Retry-After"),
    }


def probe_delisted():
    print("\n=== 1. Delisted ticker price coverage ===")
    targets = [
        # (symbol, expected_last_trading_around, note)
        ("BSC", "2008-05", "Bear Stearns acquired by JPM"),
        ("LEHM", "2008-09", "Lehman Brothers bankruptcy"),
        ("TWTR", "2022-10", "Twitter taken private by Musk"),
        ("ATVI", "2023-10", "Activision acquired by Microsoft"),
        ("FB",   "2022-06", "Meta renamed; old ticker retired"),
    ]
    # Request full 20-year window so we capture even the 2008 names
    far_back = int((NOW - timedelta(days=365 * 20)).timestamp())
    now_ts = int(NOW.timestamp())
    out = []
    for sym, expected, note in targets:
        status, body, meta = call(
            "/stock/candle",
            {"symbol": sym, "resolution": "D", "from": far_back, "to": now_ts},
        )
        summary = {
            "symbol": sym,
            "expected_last_trading": expected,
            "note": note,
            "http_status": status,
            "meta": meta,
        }
        if isinstance(body, dict):
            s = body.get("s")
            summary["api_s"] = s
            if s == "ok" and body.get("t"):
                ts = body["t"]
                summary["candles"] = len(ts)
                summary["first_date"] = datetime.fromtimestamp(ts[0], tz=timezone.utc).date().isoformat()
                summary["last_date"] = datetime.fromtimestamp(ts[-1], tz=timezone.utc).date().isoformat()
            else:
                summary["body_keys"] = list(body.keys())[:10]
        else:
            summary["body"] = str(body)[:200]
        out.append(summary)
        print(f"  {sym:6s} -> {summary.get('first_date','?'):10s} .. {summary.get('last_date','?'):10s}  "
              f"({summary.get('candles','?')} candles, s={summary.get('api_s','?')})  expected ~{expected}")
    return out


def probe_calendar_earnings_rate_limit():
    print("\n=== 2. /calendar/earnings rate-limit bucket ===")
    # Send 5 calls in rapid succession; observe limit header
    results = []
    cal_from = (NOW - timedelta(days=7)).strftime("%Y-%m-%d")
    cal_to = NOW.strftime("%Y-%m-%d")
    for i in range(5):
        status, body, meta = call(
            "/calendar/earnings",
            {"from": cal_from, "to": cal_to},
            sleep=0.05,
        )
        results.append({"i": i, "status": status, **meta})
        print(f"  call {i}: status={status} limit={meta['limit']} remaining={meta['remaining']}")
    return results


def probe_calendar_earnings_historical():
    print("\n=== 3. Historical earnings spot-check ===")
    # Use known AAPL earnings dates from public record:
    #  - 2024-08-01 (Q3 FY24, after-hours)
    #  - 2023-08-03 (Q3 FY23, after-hours)
    #  - 2022-07-28 (Q3 FY22)
    #  - 2021-07-27 (Q3 FY21)
    #  - 2020-07-30 (Q3 FY20)
    test_windows = [
        ("2024-07-25", "2024-08-05", "AAPL", "2024-08-01"),
        ("2023-07-25", "2023-08-10", "AAPL", "2023-08-03"),
        ("2022-07-22", "2022-08-02", "AAPL", "2022-07-28"),
        ("2021-07-22", "2021-08-02", "AAPL", "2021-07-27"),
        ("2020-07-25", "2020-08-05", "AAPL", "2020-07-30"),
    ]
    out = []
    for frm, to, sym, expected in test_windows:
        status, body, meta = call("/calendar/earnings", {"from": frm, "to": to, "symbol": sym})
        events = []
        if isinstance(body, dict):
            cal = body.get("earningsCalendar", [])
            events = [e for e in cal if e.get("symbol") == sym]
        match = any(e.get("date") == expected for e in events)
        summary = {
            "window": f"{frm}..{to}",
            "symbol": sym,
            "expected": expected,
            "events_for_symbol": events,
            "matched_expected_date": match,
            "http_status": status,
            "meta": meta,
        }
        out.append(summary)
        dates_seen = [e.get("date") for e in events]
        print(f"  {sym} {frm}..{to}: expected={expected}, got={dates_seen}, match={match}")
    return out


if __name__ == "__main__":
    delisted = probe_delisted()
    rl = probe_calendar_earnings_rate_limit()
    hist = probe_calendar_earnings_historical()
    out_path = ROOT / "docs" / "diagnostics" / "finnhub_verification_probe.json"
    out_path.write_text(
        json.dumps(
            {
                "generated_at": NOW.isoformat(),
                "delisted_coverage": delisted,
                "calendar_earnings_rate_limit": rl,
                "historical_earnings_spot_check": hist,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[probe] wrote {out_path}")
