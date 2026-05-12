"""Build point-in-time fundamentals from Finnhub /stock/metric series.quarterly.

The Finnhub /stock/metric response has a `series` field with `quarterly`
and `annual` sub-dicts: each metric maps to a list of {period, v} entries.
The raw response is cached at models/cache/equities/finnhub/metrics/<SYM>.json.
The snapshot's fundamentals.json contains only the top-level `metric` field
(a current-as-of-fetch snapshot), not the historical series.

This script extracts the quarterly history into a tall→wide point-in-time
table with proper reporting lag so downstream feature engineering can do a
merge_asof to attach AS-OF fundamentals without look-ahead bias.

Reporting lag policy:
- Quarterly: 45 days after period_end (typical 10-Q filing window)
- Annual: 90 days after period_end (typical 10-K filing window)

Output: models/features/larger_universe_v1/fundamentals_pit.parquet
Schema:
  ticker        str
  period_end    date (Finnhub period field)
  reported_date date (period_end + reporting_lag)
  freq          str ("Q" or "A")
  <metric_name> float64 (one column per Finnhub metric)
"""
from __future__ import annotations

import json, logging, sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

METRIC_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "metrics"
OUT_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "fundamentals_pit.parquet"

REPORTING_LAG_QUARTERLY_DAYS = 45
REPORTING_LAG_ANNUAL_DAYS = 90

# Quarterly metrics we want exposed for the study (broader than the spec's
# 11-feature core set so downstream studies can drop or add without re-running
# this extraction). Spec-named fundamentals map to:
#   P/E  -> peTTM
#   P/B  -> pb
#   P/S  -> psTTM
#   debt-to-equity -> totalDebtToEquity
#   ROE -> roeTTM
#   ROA -> roaTTM
#   profit margin -> netMargin
#   revenue growth -> (computed downstream from salesPerShare TTM YoY)
#   EPS growth -> (computed downstream from eps TTM YoY)
#   dividend yield -> not in series; use static value with disclaimer
#   beta -> not in series; use static value with disclaimer
QUARTERLY_METRICS = [
    "eps", "peTTM", "pb", "ps", "psTTM",
    "currentRatio", "quickRatio", "cashRatio",
    "totalDebtToEquity", "totalDebtToTotalAsset", "totalDebtToTotalCapital",
    "longtermDebtTotalEquity", "longtermDebtTotalAsset",
    "roaTTM", "roeTTM", "roicTTM",
    "netMargin", "grossMargin", "operatingMargin", "pretaxMargin", "fcfMargin",
    "evEbitdaTTM", "evRevenueTTM", "pfcfTTM", "payoutRatioTTM",
    "salesPerShare", "bookValue", "tangibleBookValue", "ebitPerShare",
    "fcfPerShareTTM", "ev",
    "assetTurnoverTTM", "inventoryTurnoverTTM", "receivablesTurnoverTTM",
    "sgaToSale", "ptbv",
]

logger = logging.getLogger("fund_pit")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def _extract_quarterly(raw_series: dict, ticker: str) -> list[dict]:
    """Pull every available quarterly metric into rows keyed by period_end."""
    qs = raw_series.get("quarterly", {})
    # Build a {period_end: {metric: value}} dict
    by_period: dict[str, dict] = {}
    for metric in QUARTERLY_METRICS:
        entries = qs.get(metric) or []
        for e in entries:
            period = e.get("period")
            v = e.get("v")
            if not period or v is None:
                continue
            by_period.setdefault(period, {})[metric] = v
    rows = []
    for period_end, fields in by_period.items():
        try:
            pe = pd.Timestamp(period_end).date()
        except Exception:
            continue
        rd = pe + timedelta(days=REPORTING_LAG_QUARTERLY_DAYS)
        rows.append({
            "ticker": ticker,
            "period_end": pe,
            "reported_date": rd,
            "freq": "Q",
            **fields,
        })
    return rows


def main() -> int:
    if not METRIC_DIR.exists():
        raise SystemExit(f"metrics dir not found: {METRIC_DIR}")
    metric_files = sorted(METRIC_DIR.glob("*.json"))
    logger.info("found %d metric JSON files", len(metric_files))

    all_rows: list[dict] = []
    n_with_quarterly = 0
    n_empty = 0
    for p in metric_files:
        sym = p.stem
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("  %s: read failed (%s)", sym, e)
            continue
        series = body.get("series")
        if not series:
            n_empty += 1
            continue
        rows = _extract_quarterly(series, sym)
        if rows:
            n_with_quarterly += 1
            all_rows.extend(rows)

    logger.info("tickers with quarterly history: %d (empty: %d)",
                n_with_quarterly, n_empty)
    logger.info("total quarterly rows: %d", len(all_rows))

    if not all_rows:
        raise SystemExit("no quarterly rows extracted; aborting")

    df = pd.DataFrame(all_rows)
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["reported_date"] = pd.to_datetime(df["reported_date"])
    df = df.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)

    # Summary stats
    logger.info("wrote %s, shape=%s", OUT_PATH, df.shape)
    logger.info("date range: %s -> %s", df["period_end"].min(), df["period_end"].max())
    logger.info("tickers covered: %d", df["ticker"].nunique())

    # Per-metric coverage
    print()
    print("Per-metric non-null counts (out of %d total quarterly rows):" % len(df))
    for m in QUARTERLY_METRICS:
        if m in df.columns:
            n_nn = int(df[m].notnull().sum())
            print(f"  {m:25s} {n_nn:>7d}  ({100*n_nn/len(df):>5.1f}%)")
        else:
            print(f"  {m:25s} (column absent — no ticker had this metric)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
