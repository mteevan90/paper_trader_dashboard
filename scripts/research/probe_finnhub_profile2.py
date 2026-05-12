"""Quick capability probe for /stock/profile2 on Finnhub Basic.

Spec asks for GICS Level 1 sectors. /stock/profile2 returns finnhubIndustry
+ gicsSubIndustry on supported tiers. Tested on 5 tickers across tiers.
"""
import json, os, time
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
KEY = os.environ["FINNHUB_API_KEY"]
BASE = "https://finnhub.io/api/v1"

SAMPLES = ["AAPL", "AAON", "AAMI", "SIVB", "TWTR"]
results = []
for sym in SAMPLES:
    t0 = time.perf_counter()
    r = requests.get(f"{BASE}/stock/profile2",
                     params={"symbol": sym, "token": KEY}, timeout=20)
    dt = round((time.perf_counter() - t0) * 1000, 1)
    time.sleep(1.1)  # respect 60/min bucket
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text[:200]}
    results.append({
        "symbol": sym,
        "http_status": r.status_code,
        "rate_limit": r.headers.get("X-Ratelimit-Limit"),
        "remaining": r.headers.get("X-Ratelimit-Remaining"),
        "elapsed_ms": dt,
        "body": body,
    })
    summary_keys = list(body.keys()) if isinstance(body, dict) else []
    print(f"  {sym:6s} status={r.status_code} limit={r.headers.get('X-Ratelimit-Limit')} "
          f"keys={summary_keys[:8]}")

out = ROOT / "docs" / "diagnostics" / "finnhub_profile2_probe.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nwrote {out}")
