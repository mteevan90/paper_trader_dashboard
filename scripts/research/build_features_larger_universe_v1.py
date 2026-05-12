"""Phase 1: build the Larger Universe v1 feature matrix.

Inputs:
- price_cache: models/snapshots/equities/larger_universe_v1_20260511/price_cache/*.parquet
- universe:    models/snapshots/equities/larger_universe_v1_20260511/cache/universe.json
- fundamentals (point-in-time, with reporting lag):
                models/features/larger_universe_v1/fundamentals_pit.parquet
- macro (extended FRED):
                models/features/larger_universe_v1/macro_signals_extended.parquet
- sector_map + shares_outstanding (Finnhub /stock/profile2 aggregate):
                models/features/larger_universe_v1/sector_map.json

Output: models/features/larger_universe_v1/features.parquet

Features (target ~29 total, tenure dropped per Phase-1 design decision):

  Returns (6):       ret_1d, ret_5d, ret_21d, ret_63d, ret_126d, ret_252d
  Volatility (2):    vol_21d, vol_63d
  Trend (3):         price_vs_ma50, price_vs_ma200, ma50_vs_ma200
  Drawdown (1):      dd_252d
  Fundamentals (11): pe, pb, ps, debt_to_equity, roe, roa, profit_margin,
                     revenue_growth, eps_growth, dividend_yield, beta
  Macro (10):        hy_spread, yc_slope, vix, vix_5d_chg, baa_spread,
                     usd_index, unrate, wti_oil, nfci, sahm
                     (+ yc_3m available but not in spec)
  Categorical (4):   sector (one-hot expansion downstream), in_sp500,
                     in_sp400, in_sp600
  Derived (1):       log_market_cap

Notes:
- All price-derived features use snapshot prices (split-adjusted).
- vix_5d_chg = vix.diff(5) (5-day change in VIX level)
- Fundamentals merged with reported_date <= feature_date (asof) so no
  look-ahead bias.
- log_market_cap derived as: current_market_cap_2026_05_11 * (close/last_close).
  For tickers without a current market cap (delisted), uses the last available
  estimated market cap and scales backward from the last available close.
- dividend_yield and beta are STATIC (from the snapshot's current fundamentals.json);
  documented as approximations because Finnhub series.quarterly doesn't include them.
"""
from __future__ import annotations

import json, logging, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SNAPSHOT = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511"
PRICE_DIR = SNAPSHOT / "price_cache"
UNIVERSE_PATH = SNAPSHOT / "cache" / "universe.json"
STATIC_FUND_PATH = SNAPSHOT / "cache" / "fundamentals.json"

FEAT_ROOT = ROOT / "models" / "features" / "larger_universe_v1"
FUND_PIT_PATH = FEAT_ROOT / "fundamentals_pit.parquet"
MACRO_PATH = FEAT_ROOT / "macro_signals_extended.parquet"
SECTOR_PATH = FEAT_ROOT / "sector_map.json"
OUT_PATH = FEAT_ROOT / "features.parquet"
COVERAGE_OUT = ROOT / "docs" / "diagnostics" / "larger_universe_v1_features.md"

# Spec-named features mapped to fundamentals_pit columns
FUND_PIT_MAP = {
    "pe": "peTTM",
    "pb": "pb",
    "ps": "psTTM",
    "debt_to_equity": "totalDebtToEquity",
    "roe": "roeTTM",
    "roa": "roaTTM",
    "profit_margin": "netMargin",
}

# Static fundamentals (from current /stock/metric snapshot — known approximation)
STATIC_FUND_MAP = {
    "dividend_yield": "currentDividendYieldTTM",
    "beta": "beta",
}

logger = logging.getLogger("features")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def _build_price_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Compute returns, volatility, trend, drawdown for a single ticker."""
    df = pd.DataFrame(index=close.index)
    # Returns
    for w in (1, 5, 21, 63, 126, 252):
        df[f"ret_{w}d"] = close.pct_change(w)
    # Volatility (annualized stdev of daily log returns)
    lret = np.log(close / close.shift(1))
    df["vol_21d"] = lret.rolling(21).std() * math.sqrt(252)
    df["vol_63d"] = lret.rolling(63).std() * math.sqrt(252)
    # Trend
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df["price_vs_ma50"] = close / ma50 - 1.0
    df["price_vs_ma200"] = close / ma200 - 1.0
    df["ma50_vs_ma200"] = ma50 / ma200 - 1.0
    # Drawdown from rolling 252d high
    rolling_max = close.rolling(252).max()
    df["dd_252d"] = close / rolling_max - 1.0
    return df


def _load_universe() -> dict[str, dict]:
    """Dedupe universe.json by symbol, prefer-active."""
    raw = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    by_sym: dict[str, dict] = {}
    for r in raw:
        s = r["symbol"]
        if s not in by_sym or (by_sym[s]["status"] == "removed" and r["status"] == "active"):
            by_sym[s] = r
    return by_sym


def _compute_revenue_eps_growth(fund_pit: pd.DataFrame) -> pd.DataFrame:
    """Compute revenue_growth and eps_growth from quarterly fundamentals.

    Both are TTM-vs-TTM-year-ago using salesPerShare and eps. We use TTM
    (trailing 4 quarters) rather than single-quarter to smooth seasonality.
    """
    df = fund_pit.copy()
    df = df.sort_values(["ticker", "period_end"])
    # 4-quarter rolling TTM
    df["sales_ttm"] = df.groupby("ticker")["salesPerShare"].transform(
        lambda s: s.rolling(4).sum()
    )
    df["eps_ttm"] = df.groupby("ticker")["eps"].transform(
        lambda s: s.rolling(4).sum()
    )
    # YoY growth (4 quarters back)
    df["revenue_growth"] = df.groupby("ticker")["sales_ttm"].pct_change(4)
    df["eps_growth"] = df.groupby("ticker")["eps_ttm"].pct_change(4)
    return df[["ticker", "period_end", "reported_date", "revenue_growth", "eps_growth",
               *FUND_PIT_MAP.values()]]


def _build_market_cap(tickers: list[str], close_panel: pd.DataFrame,
                       sector_map: dict, static_fund: dict) -> pd.DataFrame:
    """Compute log_market_cap per (date, ticker) using current shares.

    Uses Finnhub /stock/profile2 shareOutstanding (in millions) plus
    daily close × 1e6. For tickers absent from profile2, falls back to
    static_fund's marketCapitalization (also as of fetch time) and scales
    proportionally with price; if neither is available, log_market_cap
    is NaN.
    """
    rows = []
    for sym in tickers:
        if sym not in close_panel.columns:
            continue
        closes = close_panel[sym].dropna()
        if closes.empty:
            continue
        # Get shares (in millions)
        shares = None
        prof = sector_map.get(sym, {})
        if prof and prof.get("share_outstanding_millions"):
            shares = float(prof["share_outstanding_millions"])
        # Compute market cap series
        if shares is not None and shares > 0:
            mc = closes * shares  # millions * dollars/share = millions of dollars
        else:
            # Fall back to current_mktcap × (close/last_close)
            cur_mc = None
            sf = static_fund.get(sym, {})
            if sf and sf.get("marketCapitalization"):
                cur_mc = float(sf["marketCapitalization"])  # already in millions
            if cur_mc and closes.iloc[-1] > 0:
                mc = closes * (cur_mc / closes.iloc[-1])
            else:
                continue  # can't compute; skip
        log_mc = np.log(mc.clip(lower=1e-3))
        rows.append(pd.DataFrame({"date": mc.index, "ticker": sym,
                                   "log_market_cap": log_mc.values}))
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "log_market_cap"])
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    by_sym = _load_universe()
    symbols = sorted(by_sym.keys())
    logger.info("universe: %d unique symbols", len(symbols))

    # ---- Price-based features ----
    logger.info("computing price-based features...")
    long_rows: list[pd.DataFrame] = []
    close_panel = {}
    n_priced = 0
    for sym in symbols:
        p = PRICE_DIR / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index)
        close = df["close"]
        feats = _build_price_features(close, df["volume"])
        feats["ticker"] = sym
        feats["date"] = feats.index
        feats = feats.reset_index(drop=True)
        long_rows.append(feats[["date", "ticker", "ret_1d", "ret_5d", "ret_21d",
                                "ret_63d", "ret_126d", "ret_252d",
                                "vol_21d", "vol_63d",
                                "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200",
                                "dd_252d"]])
        close_panel[sym] = close
        n_priced += 1
    logger.info("priced tickers: %d", n_priced)
    features = pd.concat(long_rows, ignore_index=True)
    features["date"] = pd.to_datetime(features["date"])
    logger.info("price features: %s", features.shape)

    # Pivot close panel for market cap
    cp = pd.DataFrame(close_panel)
    cp.index = pd.to_datetime(cp.index)

    # ---- Macro features ----
    logger.info("merging macro signals...")
    macro = pd.read_parquet(MACRO_PATH)
    macro = macro.copy()
    macro["vix_5d_chg"] = macro["vix"].diff(5)
    macro = macro.reset_index().rename(columns={"index": "date"})
    macro["date"] = pd.to_datetime(macro["date"])
    features = features.merge(macro, on="date", how="left")
    logger.info("after macro merge: %s", features.shape)

    # ---- Fundamentals (PIT, with reporting lag) ----
    logger.info("merging fundamentals (asof reported_date)...")
    fund_pit_raw = pd.read_parquet(FUND_PIT_PATH)
    # Normalize datetime precision to ns to match features["date"]
    fund_pit_raw["reported_date"] = pd.to_datetime(fund_pit_raw["reported_date"]).astype("datetime64[ns]")
    fund_pit_raw["period_end"] = pd.to_datetime(fund_pit_raw["period_end"]).astype("datetime64[ns]")
    # Add revenue/eps growth via TTM derivation
    fund_pit = _compute_revenue_eps_growth(fund_pit_raw)
    fund_pit = fund_pit.rename(columns={v: k for k, v in FUND_PIT_MAP.items()})
    fund_pit = fund_pit.sort_values(["ticker", "reported_date"])
    # Also normalize features date to ns
    features["date"] = pd.to_datetime(features["date"]).astype("datetime64[ns]")

    # Per-ticker merge_asof
    out_pieces = []
    for sym, g in features.groupby("ticker", sort=False):
        f = g.sort_values("date")
        fp = fund_pit[fund_pit["ticker"] == sym].sort_values("reported_date")
        if fp.empty:
            # No fundamentals history; just keep price features as-is
            for col in list(FUND_PIT_MAP) + ["revenue_growth", "eps_growth"]:
                f[col] = np.nan
            out_pieces.append(f)
            continue
        merged = pd.merge_asof(
            f, fp.drop(columns=["ticker", "period_end"]),
            left_on="date", right_on="reported_date",
            direction="backward",
        )
        merged = merged.drop(columns=["reported_date"])
        out_pieces.append(merged)
    features = pd.concat(out_pieces, ignore_index=True)
    logger.info("after fundamentals merge: %s", features.shape)

    # ---- Static fundamentals (dividend_yield, beta) ----
    logger.info("merging static fundamentals (dividend_yield, beta)...")
    static_fund = json.loads(STATIC_FUND_PATH.read_text(encoding="utf-8"))
    static_rows = []
    for sym, body in static_fund.items():
        static_rows.append({
            "ticker": sym,
            "dividend_yield": body.get(STATIC_FUND_MAP["dividend_yield"]),
            "beta": body.get(STATIC_FUND_MAP["beta"]),
        })
    static_df = pd.DataFrame(static_rows)
    features = features.merge(static_df, on="ticker", how="left")
    logger.info("after static fundamentals merge: %s", features.shape)

    # ---- Sector + market cap ----
    sector_map = {}
    if SECTOR_PATH.exists():
        sector_map = json.loads(SECTOR_PATH.read_text(encoding="utf-8"))
        logger.info("sector_map: %d entries", len(sector_map))
    else:
        logger.warning("sector_map.json missing — running without sector/market_cap")

    # Sector column
    sector_rows = [(s, (sector_map.get(s, {}) or {}).get("sector") or "sector_unknown")
                   for s in features["ticker"].unique()]
    sec_df = pd.DataFrame(sector_rows, columns=["ticker", "sector"])
    features = features.merge(sec_df, on="ticker", how="left")

    # Market cap (log)
    if sector_map:
        logger.info("computing log_market_cap...")
        mc_df = _build_market_cap(symbols, cp, sector_map, static_fund)
        if not mc_df.empty:
            features = features.merge(mc_df, on=["date", "ticker"], how="left")
        else:
            features["log_market_cap"] = np.nan
    else:
        features["log_market_cap"] = np.nan

    # ---- Index membership ----
    logger.info("attaching index membership...")
    tier_rows = [(s, by_sym[s]["tier"]) for s in features["ticker"].unique() if s in by_sym]
    tier_df = pd.DataFrame(tier_rows, columns=["ticker", "tier"])
    features = features.merge(tier_df, on="ticker", how="left")
    features["in_sp500"] = (features["tier"] == "SP500").astype(int)
    features["in_sp400"] = (features["tier"] == "SP400").astype(int)
    features["in_sp600"] = (features["tier"] == "SP600").astype(int)
    features = features.drop(columns=["tier"])

    # ---- Save ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Trim to training-window-and-beyond (drop pre-2016-05-12 rows since features need 252d lookback)
    # Keep all dates >= 2017-05-12 to ensure the 252d lookback is well-formed
    # Actually keep all rows — let downstream filter; we want the full record
    features.to_parquet(OUT_PATH)
    logger.info("wrote %s, shape=%s", OUT_PATH, features.shape)
    logger.info("date range: %s -> %s", features["date"].min(), features["date"].max())
    logger.info("ticker count: %d", features["ticker"].nunique())

    # Per-feature non-null counts
    print()
    print("Per-feature non-null fraction:")
    for col in features.columns:
        if col in ("date", "ticker"):
            continue
        n_nn = features[col].notnull().sum()
        print(f"  {col:25s} {n_nn:>10d}  ({100*n_nn/len(features):>5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
