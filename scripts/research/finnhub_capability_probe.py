"""Phase 0 probe: enumerate which Finnhub endpoints respond on a Basic-tier key.

Writes JSON results to docs/diagnostics/finnhub_basic_capabilities.json so the
md report generator can render a clean table without re-hitting the API.

Run from repo root. Reads FINNHUB_API_KEY from .env.
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

SAMPLE_LIVE = "AAPL"
SAMPLE_MIDCAP = "AAON"
SAMPLE_SMALLCAP = "AAMI"
SAMPLE_DELISTED = "TWTR"

NOW = datetime.now(timezone.utc)


def call(path: str, params: dict, *, sleep: float = 0.5):
    p = dict(params)
    p["token"] = KEY
    t0 = time.perf_counter()
    r = requests.get(f"{BASE}{path}", params=p, timeout=30)
    dt_ms = (time.perf_counter() - t0) * 1000
    time.sleep(sleep)
    try:
        body = r.json()
    except Exception:
        body = r.text[:500]
    return {
        "path": path,
        "params_sent": {k: v for k, v in params.items() if k != "token"},
        "status": r.status_code,
        "elapsed_ms": round(dt_ms, 1),
        "headers": {
            k: v
            for k, v in r.headers.items()
            if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
        },
        "body_sample": _summarize(body),
        "body_type": type(body).__name__,
    }


def _summarize(body):
    if isinstance(body, list):
        return {"kind": "list", "len": len(body), "first": body[0] if body else None}
    if isinstance(body, dict):
        keys = list(body.keys())
        summary = {"kind": "dict", "keys": keys[:25]}
        for k in ("s", "error", "no_data", "symbol", "metric"):
            if k in body:
                summary[k] = body[k]
        for k in ("c", "t", "v", "o", "h", "l"):
            if k in body and isinstance(body[k], list):
                summary[f"{k}_len"] = len(body[k])
        if "metric" in body and isinstance(body["metric"], dict):
            summary["metric_keys"] = list(body["metric"].keys())[:30]
        return summary
    return {"kind": type(body).__name__, "value": str(body)[:200]}


def probe_all():
    out = {"generated_at": NOW.isoformat(), "endpoints": []}

    # 1. /stock/symbol with exchange=US (active listing)
    out["endpoints"].append(call("/stock/symbol", {"exchange": "US"}))

    # 2. /stock/symbol with exchange=US&delisted=true
    out["endpoints"].append(call("/stock/symbol", {"exchange": "US", "delisted": "true"}))

    # 3. /stock/candle 10y daily — AAPL
    end_ts = int(NOW.timestamp())
    start_10y = int((NOW - timedelta(days=365 * 10 + 7)).timestamp())
    out["endpoints"].append(
        call("/stock/candle", {"symbol": SAMPLE_LIVE, "resolution": "D", "from": start_10y, "to": end_ts})
    )

    # 4. /stock/candle 12y daily — boundary check (should fail or truncate on Basic)
    start_12y = int((NOW - timedelta(days=365 * 12)).timestamp())
    out["endpoints"].append(
        call("/stock/candle", {"symbol": SAMPLE_LIVE, "resolution": "D", "from": start_12y, "to": end_ts})
    )

    # 5. /stock/candle SP400 sample 10y
    out["endpoints"].append(
        call("/stock/candle", {"symbol": SAMPLE_MIDCAP, "resolution": "D", "from": start_10y, "to": end_ts})
    )

    # 6. /stock/candle SP600 sample 10y
    out["endpoints"].append(
        call("/stock/candle", {"symbol": SAMPLE_SMALLCAP, "resolution": "D", "from": start_10y, "to": end_ts})
    )

    # 7. /stock/candle delisted (TWTR — went private 2022) 10y
    out["endpoints"].append(
        call("/stock/candle", {"symbol": SAMPLE_DELISTED, "resolution": "D", "from": start_10y, "to": end_ts})
    )

    # 8. /stock/earnings AAPL
    out["endpoints"].append(call("/stock/earnings", {"symbol": SAMPLE_LIVE}))

    # 9. /stock/earnings SP600 sample
    out["endpoints"].append(call("/stock/earnings", {"symbol": SAMPLE_SMALLCAP}))

    # 10. /calendar/earnings recent week
    cal_from = (NOW - timedelta(days=7)).strftime("%Y-%m-%d")
    cal_to = NOW.strftime("%Y-%m-%d")
    out["endpoints"].append(call("/calendar/earnings", {"from": cal_from, "to": cal_to}))

    # 11. /stock/recommendation AAPL
    out["endpoints"].append(call("/stock/recommendation", {"symbol": SAMPLE_LIVE}))

    # 12. /stock/recommendation SP600 sample
    out["endpoints"].append(call("/stock/recommendation", {"symbol": SAMPLE_SMALLCAP}))

    # 13. /stock/metric AAPL (all)
    out["endpoints"].append(call("/stock/metric", {"symbol": SAMPLE_LIVE, "metric": "all"}))

    # 14. /stock/metric SP600 sample
    out["endpoints"].append(call("/stock/metric", {"symbol": SAMPLE_SMALLCAP, "metric": "all"}))

    # 15. /stock/peers AAPL
    out["endpoints"].append(call("/stock/peers", {"symbol": SAMPLE_LIVE}))

    return out


def probe_rate_limit_burst(n: int = 30):
    """Send n rapid /quote calls to characterize the rate-limit ceiling."""
    results = []
    t0 = time.perf_counter()
    for i in range(n):
        r = requests.get(f"{BASE}/quote", params={"symbol": "AAPL", "token": KEY}, timeout=15)
        results.append(
            {
                "i": i,
                "status": r.status_code,
                "remaining": r.headers.get("X-Ratelimit-Remaining"),
                "limit": r.headers.get("X-Ratelimit-Limit"),
                "retry_after": r.headers.get("Retry-After"),
            }
        )
        if r.status_code == 429:
            break
    elapsed = time.perf_counter() - t0
    return {"n_sent": len(results), "elapsed_s": round(elapsed, 2), "results": results}


if __name__ == "__main__":
    print("[probe] endpoint sweep...")
    data = probe_all()
    print("[probe] rate-limit burst...")
    data["rate_limit_burst"] = probe_rate_limit_burst(n=20)
    out_path = ROOT / "docs" / "diagnostics" / "finnhub_basic_capabilities.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"[probe] wrote {out_path}")
