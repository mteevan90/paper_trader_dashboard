"""Build the extended macro_signals parquet for the Larger Universe v1 study.

Adds BAA10Y (BAA-AAA credit spread proxy), DTWEXBGS (USD index),
UNRATE (unemployment rate), DCOILWTICO (WTI oil) to the existing 6-column
macro_signals series, and extends the start date back to 2016-05-12.

Output: models/features/larger_universe_v1/macro_signals_extended.parquet

Does NOT modify the snapshot's macro_signals.parquet. The original 6-col
parquet stays as the canonical legacy artifact; the new 10-col parquet
is used by this study and any future study that wants the expanded set.
"""
from __future__ import annotations

import logging, os, sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "macro_signals_extended.parquet"
START = "2016-01-01"  # generous buffer before training window start 2016-05-12
END = pd.Timestamp.today().strftime("%Y-%m-%d")

# 6 existing + 4 new = 10 columns. Existing keep their names for downstream
# compatibility; new ones use clean descriptive names.
FRED_SERIES = {
    # Existing (carried forward from src/macro_signals.py)
    "hy_spread":   "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS
    "yc_slope":    "T10Y2Y",         # 10yr minus 2yr Treasury
    "vix":         "VIXCLS",         # CBOE VIX
    "nfci":        "NFCI",           # Chicago Fed Financial Conditions Index
    "sahm":        "SAHMREALTIME",   # Sahm Rule recession indicator
    "yc_3m":       "T10Y3M",         # 10yr minus 3m Treasury
    # New for Larger Universe v1
    "baa_spread":  "BAA10Y",         # Moody's BAA Corporate yield minus 10y
    "usd_index":   "DTWEXBGS",       # Trade-weighted USD broad index
    "unrate":      "UNRATE",         # Civilian unemployment rate (monthly)
    "wti_oil":     "DCOILWTICO",     # WTI crude oil spot price (daily, USD/bbl)
}

logger = logging.getLogger("macro_extended")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def main() -> int:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("FRED_API_KEY not in .env")
    fred = Fred(api_key=key)
    pieces = []
    for col, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=START, observation_end=END)
            s.name = col
            n_nonnull = int(s.notnull().sum())
            logger.info("  %s (%s): %d rows, %d non-null, first=%s last=%s",
                        col, series_id, len(s), n_nonnull, s.index.min(), s.index.max())
            pieces.append(s)
        except Exception as e:
            logger.warning("  %s (%s) fetch failed: %s", col, series_id, e)

    df = pd.concat(pieces, axis=1).sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.ffill()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    logger.info("wrote %s, shape=%s", OUT_PATH, df.shape)
    logger.info("date range: %s -> %s", df.index.min(), df.index.max())
    logger.info("columns: %s", list(df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
