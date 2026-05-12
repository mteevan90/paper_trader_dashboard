"""Fetch SPY 10y daily candles + /stock/dividend2 for the Larger Universe v1 universe.

Two endpoints, two outputs:

1. SPY history (single call, /stock/candle, 150/min bucket):
   - Window: 2014-05-12 → today (covers 36mo lookback for first 2017-05-12 beta)
   - Output: models/cache/equities/finnhub/prices/SPY.parquet (mirrors universe fetcher path)

2. Per-ticker dividend history (/stock/dividend2, 150/min bucket):
   - For every symbol in the universe
   - Output: models/cache/equities/finnhub/dividends/<SAFE_SYM>.json per ticker
   - Aggregated lookup: models/features/larger_universe_v1/dividend_history.parquet
     with columns [ticker, ex_date, amount]

Resumable: skips tickers whose dividend file already exists.
"""
from __future__ import annotations

import json, logging, os, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.options.tradier import RateLimiter
from src.equities.finnhub_fetcher import fetch_candles, make_candle_limiter

KEY = os.environ["FINNHUB_API_KEY"]
BASE = "https://finnhub.io/api/v1"
PRICE_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "prices"
DIV_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "dividends"
UNIVERSE_PATH = ROOT / "docs" / "larger_universe_v1_universe.json"
DIV_AGG_OUT = ROOT / "models" / "features" / "larger_universe_v1" / "dividend_history.parquet"

logger = logging.getLogger("spy_div")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def _safe_filename(sym: str) -> str:
    return sym.replace("/", "-").replace("\\", "-")


def fetch_spy() -> None:
    """Fetch SPY history 2014-05-12 → today via Finnhub /stock/candle."""
    spy_path = PRICE_DIR / "SPY.parquet"
    if spy_path.exists():
        df = pd.read_parquet(spy_path)
        if df.index.min() <= pd.Timestamp("2014-06-01").date():
            logger.info("SPY already has %d rows from %s; skipping fetch", len(df), df.index.min())
            return
    from datetime import date
    start = date(2014, 5, 12)
    end = date.today()
    logger.info("fetching SPY candles %s -> %s", start, end)
    limiter = make_candle_limiter()
    df = fetch_candles("SPY", start, end, limiter=limiter, use_cache=True)
    logger.info("SPY: %d rows, first=%s, last=%s", len(df), df.index.min(), df.index.max())


def fetch_dividends_one(sym: str, limiter: RateLimiter) -> list[dict]:
    """Fetch /stock/dividend2 for a single ticker; cache result.

    Body shape: {symbol: str, data: [{exDate, amount, currency, ...}, ...]}
    We persist the full body and return the data list.
    """
    cache = DIV_DIR / f"{_safe_filename(sym)}.json"
    if cache.exists():
        body = json.loads(cache.read_text(encoding="utf-8"))
        return body.get("data", []) if isinstance(body, dict) else []
    limiter.wait()
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/stock/dividend2",
                             params={"symbol": sym, "from": "2014-01-01",
                                     "to": datetime.today().strftime("%Y-%m-%d"),
                                     "token": KEY},
                             timeout=20)
            if r.status_code == 429:
                logger.warning("  %s: 429 attempt %d/3", sym, attempt+1)
                time.sleep(2 ** (attempt+1))
                continue
            if r.status_code != 200:
                logger.warning("  %s: HTTP %d", sym, r.status_code)
                return []
            body = r.json() if r.text else {}
            tmp = cache.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(body), encoding="utf-8")
            tmp.replace(cache)
            return body.get("data", []) if isinstance(body, dict) else []
        except requests.RequestException as e:
            logger.warning("  %s: %s attempt %d/3", sym, type(e).__name__, attempt+1)
            time.sleep(2 ** attempt)
    return []


def main() -> int:
    DIV_DIR.mkdir(parents=True, exist_ok=True)

    # 1) SPY
    fetch_spy()

    # 2) Dividend history per ticker
    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    by_sym: dict[str, dict] = {}
    for r in universe:
        s = r["symbol"]
        if s not in by_sym or (by_sym[s]["status"] == "removed" and r["status"] == "active"):
            by_sym[s] = r
    symbols = sorted(by_sym.keys())
    logger.info("dividend fetch: %d unique symbols", len(symbols))

    limiter = RateLimiter(150)  # /stock/dividend2 is on the 150/min bucket
    n_with_divs = 0
    n_no_divs = 0
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        data = fetch_dividends_one(sym, limiter)
        if data:
            n_with_divs += 1
        else:
            n_no_divs += 1
        if i % 100 == 0 or i == len(symbols):
            logger.info("  %d/%d  with_divs=%d  no_divs=%d  (%.1fs)",
                        i, len(symbols), n_with_divs, n_no_divs, time.time() - t0)

    # 3) Aggregate into dividend_history.parquet
    rows = []
    for sym in symbols:
        p = DIV_DIR / f"{_safe_filename(sym)}.json"
        if not p.exists():
            continue
        body = json.loads(p.read_text(encoding="utf-8"))
        data = body.get("data", []) if isinstance(body, dict) else []
        for d in data:
            ex = d.get("exDate")
            amt = d.get("amount")
            if ex and amt is not None:
                rows.append({"ticker": sym, "ex_date": ex, "amount": float(amt)})
    if rows:
        df = pd.DataFrame(rows)
        df["ex_date"] = pd.to_datetime(df["ex_date"])
        df = df.sort_values(["ticker", "ex_date"]).reset_index(drop=True)
        DIV_AGG_OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DIV_AGG_OUT)
        logger.info("aggregated dividend history: %d rows from %d tickers -> %s",
                    len(df), df["ticker"].nunique(), DIV_AGG_OUT)
    else:
        logger.warning("no dividend rows aggregated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
