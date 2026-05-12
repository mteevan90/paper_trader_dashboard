"""Probe Finnhub /stock/candle for SP1500 names delisted in the last 10y."""
from __future__ import annotations

import json, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
KEY = os.environ["FINNHUB_API_KEY"]
BASE = "https://finnhub.io/api/v1"
NOW = datetime.now(timezone.utc)


def candles(sym):
    far = int((NOW - timedelta(days=365 * 12)).timestamp())
    to = int(NOW.timestamp())
    r = requests.get(
        f"{BASE}/stock/candle",
        params={"symbol": sym, "resolution": "D", "from": far, "to": to, "token": KEY},
        timeout=20,
    )
    time.sleep(0.4)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, None


targets = [
    ("SIVB", "Silicon Valley Bank", "2023-03"),
    ("FRC",  "First Republic", "2023-05"),
    ("BBBY", "Bed Bath & Beyond", "2023-04"),
    ("SBNY", "Signature Bank", "2023-03"),
    ("TIF",  "Tiffany (LVMH buyout)", "2021-01"),
    ("CTXS", "Citrix (TIBCO buyout)", "2022-09"),
    ("MNDT", "Mandiant (Google buyout)", "2022-09"),
    ("DISCA","Discovery Comm (renamed WBD)", "2022-04"),
    ("VIAC", "ViacomCBS (renamed PARA)", "2022-02"),
    ("FB",   "Facebook (renamed META)", "2022-06"),
    ("ATVI", "Activision (MSFT buyout)", "2023-10"),
    ("TWTR", "Twitter (Musk buyout)", "2022-10"),
    ("KSU",  "Kansas City Southern (CP buyout)", "2021-12"),
    ("XLNX", "Xilinx (AMD buyout)", "2022-02"),
    ("CERN", "Cerner (Oracle buyout)", "2022-06"),
    ("PFGC", "Performance Food Group — control", "(still active)"),
]

out = []
print(f"{'sym':6s} {'first':12s} {'last':12s} {'cdl':>6s}  expected~{'note':30s}")
for sym, name, expected in targets:
    status, body = candles(sym)
    summary = {"symbol": sym, "company": name, "expected_delisting": expected, "http_status": status}
    if isinstance(body, dict):
        s = body.get("s")
        summary["api_s"] = s
        if s == "ok" and body.get("t"):
            ts = body["t"]
            summary["candles"] = len(ts)
            summary["first_date"] = datetime.fromtimestamp(ts[0], tz=timezone.utc).date().isoformat()
            summary["last_date"] = datetime.fromtimestamp(ts[-1], tz=timezone.utc).date().isoformat()
            print(f"{sym:6s} {summary['first_date']:12s} {summary['last_date']:12s} {summary['candles']:>6d}  expected~{expected:8s}  {name}")
        else:
            print(f"{sym:6s} {'?':12s} {'?':12s} {'?':>6s}  expected~{expected:8s}  {name}  (s={s})")
    out.append(summary)

p = ROOT / "docs" / "diagnostics" / "finnhub_delisted_recent.json"
p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(f"\n[probe] wrote {p}")
