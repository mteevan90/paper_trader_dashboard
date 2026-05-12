"""Fetch RSP + IWM via Finnhub for Phase 4 benchmark comparison.

SPY is already cached at models/cache/equities/finnhub/prices/SPY.parquet
from the Phase 3 prep. EW-SP1500 is constructed from the universe data
during the Phase 4 run, not fetched. RSP and IWM are ETFs the Phase 4
spec requires for the four-benchmark comparison.

Idempotent — skips tickers already cached fresh.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.equities.finnhub_fetcher import fetch_candles, make_candle_limiter


def main() -> int:
    limiter = make_candle_limiter()
    start = date(2014, 5, 12)  # match SPY's 10y span for any later beta / rolling work
    end = date.today()
    for symbol in ("RSP", "IWM"):
        print(f"[bench] fetching {symbol} {start} -> {end}")
        df = fetch_candles(symbol, start, end, limiter=limiter, use_cache=True)
        if df.empty:
            print(f"  [bench] WARN: {symbol} returned empty")
        else:
            print(f"  [bench] {symbol}: {len(df)} rows, {df.index.min()} .. {df.index.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
