"""Fetch SHY 10y daily candles for Larger Universe v2's defensive-sleeves variant.

SHY (iShares 1-3 Year Treasury Bond ETF) is the defensive ETF half of v2-B5's
defensive sleeve (cash + SHY 50/50 within the defensive portion). It is not
an S&P member and was not in v1's universe.

Window: 2014-05-12 → today (covers the earliest walk-forward window's
training warmup; first walk-forward val_start is 2020-05-12).

Output: models/cache/equities/finnhub/prices/SHY.parquet (mirrors SPY path).

Mirrors fetch_spy_and_dividends.py's fetch_spy() pattern. Same Finnhub
endpoint, same fetcher infrastructure, same cache layout. Idempotent: skips
the fetch if SHY.parquet already exists with sufficient history.
"""
from __future__ import annotations

import logging, os, sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.equities.finnhub_fetcher import fetch_candles, make_candle_limiter

PRICE_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "prices"

logger = logging.getLogger("shy_fetch")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def fetch_shy() -> int:
    shy_path = PRICE_DIR / "SHY.parquet"
    if shy_path.exists():
        df = pd.read_parquet(shy_path)
        if df.index.min() <= pd.Timestamp("2014-06-01").date():
            logger.info("SHY already has %d rows from %s -> %s; skipping fetch",
                        len(df), df.index.min(), df.index.max())
            return 0
    start = date(2014, 5, 12)
    end = date.today()
    logger.info("fetching SHY candles %s -> %s", start, end)
    limiter = make_candle_limiter()
    df = fetch_candles("SHY", start, end, limiter=limiter, use_cache=True)
    logger.info("SHY: %d rows, first=%s, last=%s",
                len(df), df.index.min(), df.index.max())
    return 0


if __name__ == "__main__":
    raise SystemExit(fetch_shy())
