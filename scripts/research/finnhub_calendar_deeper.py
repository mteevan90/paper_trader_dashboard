"""Deeper probe of /calendar/earnings historical reach on Basic tier."""
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


def call(path, params):
    p = dict(params)
    p["token"] = KEY
    r = requests.get(f"{BASE}{path}", params=p, timeout=30)
    time.sleep(0.8)
    try:
        body = r.json()
    except Exception:
        body = r.text
    return r.status_code, body, dict(r.headers)


def test_window(frm, to, label, symbol=None):
    params = {"from": frm, "to": to}
    if symbol:
        params["symbol"] = symbol
    status, body, hdr = call("/calendar/earnings", params)
    cal = body.get("earningsCalendar", []) if isinstance(body, dict) else []
    n = len(cal)
    aapl = [e for e in cal if e.get("symbol") == "AAPL"]
    first_date = cal[0].get("date") if cal else None
    last_date = cal[-1].get("date") if cal else None
    print(f"  [{label:35s}]  status={status} n={n:5d} aapl={len(aapl)}  range=({first_date} .. {last_date})")
    return {
        "label": label,
        "from": frm,
        "to": to,
        "symbol": symbol,
        "status": status,
        "n_events": n,
        "n_aapl_events": len(aapl),
        "aapl_events": aapl[:5],
        "first_date_in_response": first_date,
        "last_date_in_response": last_date,
    }


def main():
    out = []
    # Forward looking (sanity)
    out.append(test_window("2026-05-11", "2026-05-18", "FWD: next 7 days, no symbol"))
    out.append(test_window("2026-05-11", "2026-05-18", "FWD: next 7 days, AAPL", symbol="AAPL"))
    out.append(test_window("2026-05-11", "2026-08-11", "FWD: next 3 months, no symbol"))

    # Recent past, no symbol
    out.append(test_window("2026-04-25", "2026-05-10", "PAST: last 2 weeks, no symbol"))
    out.append(test_window("2026-04-25", "2026-05-10", "PAST: last 2 weeks, AAPL", symbol="AAPL"))
    out.append(test_window("2026-02-01", "2026-02-28", "PAST: Feb 2026, no symbol"))

    # Further back
    out.append(test_window("2025-07-25", "2025-08-05", "PAST: AAPL Q3 2025 window, no sym"))
    out.append(test_window("2024-07-25", "2024-08-05", "PAST: AAPL Q3 2024 window, no sym"))
    out.append(test_window("2024-01-01", "2024-01-31", "PAST: Jan 2024, no symbol"))
    out.append(test_window("2020-07-25", "2020-08-05", "PAST: AAPL Q3 2020, no sym"))

    # Try /stock/earnings going further back — does it return more than 4?
    print()
    p = {"symbol": "AAPL", "token": KEY}
    r = requests.get(f"{BASE}/stock/earnings", params=p, timeout=15)
    time.sleep(0.5)
    body = r.json()
    print(f"  [/stock/earnings AAPL bare]  len={len(body) if isinstance(body, list) else 'n/a'}")
    if isinstance(body, list):
        for e in body:
            print(f"     {e.get('period')}  q{e.get('quarter')} y{e.get('year')}  est={e.get('estimate')}")

    # Wider: 60-day window for symbol=AAPL going back 1y, 2y, 3y
    out.append(test_window("2024-07-01", "2024-08-31", "PAST: 60d window Jul-Aug 2024 AAPL", symbol="AAPL"))
    out.append(test_window("2023-07-01", "2023-08-31", "PAST: 60d window Jul-Aug 2023 AAPL", symbol="AAPL"))

    # Save
    p_out = ROOT / "docs" / "diagnostics" / "finnhub_calendar_deeper.json"
    p_out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[probe] wrote {p_out}")


if __name__ == "__main__":
    main()
