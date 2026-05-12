"""Replace the static dividend_yield and beta columns in the Larger Universe v1
feature matrix with point-in-time computations.

Beta — trailing 36-month OLS of ticker daily returns vs SPY daily returns,
recomputed at every feature date. Implemented as rolling cov(T,SPY)/var(SPY)
on a 756-trading-day window — algebraically identical to OLS slope and
~100× faster than running a regression in a loop. NaN where the ticker
has < 756 days of overlap with SPY at the feature date.

Dividend yield — trailing 12-month dividends / current close. Reads
dividend events from models/features/larger_universe_v1/dividend_history.parquet
(produced by scripts/research/fetch_spy_and_dividends.py). For tickers
absent from that file, yield = 0 (interpreted as "no dividend payer in
the last 12 months" — distinct from NaN which would mean "unknown").

Input: models/features/larger_universe_v1/features.parquet
       models/cache/equities/finnhub/prices/SPY.parquet
       models/features/larger_universe_v1/dividend_history.parquet

Output: models/features/larger_universe_v1/features.parquet (in place;
        beta and dividend_yield columns replaced)
"""
from __future__ import annotations

import json, logging, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEATURES = ROOT / "models" / "features" / "larger_universe_v1" / "features.parquet"
SPY_PATH = ROOT / "models" / "cache" / "equities" / "finnhub" / "prices" / "SPY.parquet"
DIV_HIST = ROOT / "models" / "features" / "larger_universe_v1" / "dividend_history.parquet"
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"

BETA_WINDOW_DAYS = 756   # ~36 trading months
DIV_LOOKBACK_DAYS = 365  # trailing 12 months

logger = logging.getLogger("pit_beta_div")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def compute_beta_series(close: pd.Series, spy_returns: pd.Series) -> pd.Series:
    """Rolling 36-month beta of `close` vs SPY returns.

    Returns a series indexed by `close.index` with NaN where the rolling
    window has insufficient overlap.
    """
    rets = close.pct_change()
    # Align on the union of both indexes (left join on close.index)
    aligned = pd.DataFrame({"t": rets, "spy": spy_returns}).dropna(subset=["t"])
    # rolling cov / var
    cov = aligned["t"].rolling(BETA_WINDOW_DAYS, min_periods=BETA_WINDOW_DAYS).cov(aligned["spy"])
    var = aligned["spy"].rolling(BETA_WINDOW_DAYS, min_periods=BETA_WINDOW_DAYS).var()
    beta = cov / var
    return beta.reindex(close.index)


def compute_dividend_yield_series(close: pd.Series, divs_for_ticker: pd.DataFrame) -> pd.Series:
    """Trailing 12-month dividend yield series for a ticker.

    `divs_for_ticker` is a DataFrame with columns [ex_date, amount].
    """
    if divs_for_ticker is None or divs_for_ticker.empty:
        return pd.Series(0.0, index=close.index)
    # Build a daily dividend series at the ticker's price-index dates
    div_series = pd.Series(0.0, index=close.index)
    # For each ex_date, find the next available trading day at or after ex_date
    # and add the amount there. (yfinance/Finnhub adjust prices for splits and
    # divs; the ex-date in /stock/dividend2 corresponds to the actual ex-dividend
    # day. For trailing-12-mo sums, exact attribution to the next-available
    # day vs. ex-date is immaterial since we're summing in a moving window.)
    dt_idx = pd.DatetimeIndex(close.index)
    for ex_date, amount in zip(divs_for_ticker["ex_date"], divs_for_ticker["amount"]):
        # Find position of ex_date in close.index (or the next available day)
        pos = dt_idx.searchsorted(ex_date)
        if pos < len(dt_idx):
            anchor = dt_idx[pos]
            div_series.loc[anchor] += float(amount)
    # 252-trading-day rolling sum approximates trailing 12-month dividends
    rolling_sum = div_series.rolling(252, min_periods=1).sum()
    yield_series = rolling_sum / close
    return yield_series


def main() -> int:
    if not FEATURES.exists():
        raise SystemExit(f"features parquet missing: {FEATURES}")
    if not SPY_PATH.exists():
        raise SystemExit(f"SPY parquet missing: {SPY_PATH} (run fetch_spy_and_dividends.py)")
    if not DIV_HIST.exists():
        raise SystemExit(f"dividend history missing: {DIV_HIST} (run fetch_spy_and_dividends.py)")

    logger.info("loading SPY...")
    spy = pd.read_parquet(SPY_PATH)
    spy.index = pd.to_datetime(spy.index)
    spy_returns = spy["close"].pct_change()
    logger.info("SPY: %d rows, %s → %s", len(spy), spy.index.min(), spy.index.max())

    logger.info("loading dividend history...")
    divs = pd.read_parquet(DIV_HIST)
    divs["ex_date"] = pd.to_datetime(divs["ex_date"])
    divs_by_ticker = {sym: g for sym, g in divs.groupby("ticker")}
    logger.info("dividends: %d events across %d tickers",
                len(divs), divs["ticker"].nunique())

    logger.info("loading features...")
    features = pd.read_parquet(FEATURES)
    logger.info("features: %s", features.shape)
    features["date"] = pd.to_datetime(features["date"])

    # Build per-ticker beta and div_yield series, then merge
    logger.info("computing PIT beta + dividend yield per ticker...")
    tickers = sorted(features["ticker"].unique())
    t0 = time.time()
    out_pieces = []
    for i, sym in enumerate(tickers, 1):
        p = SNAPSHOT_PRICE_DIR / f"{sym}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index)
        close = df["close"]

        beta_s = compute_beta_series(close, spy_returns)
        div_s = compute_dividend_yield_series(close, divs_by_ticker.get(sym))

        piece = pd.DataFrame({
            "date": close.index,
            "ticker": sym,
            "beta_pit": beta_s.values,
            "dividend_yield_pit": div_s.values,
        })
        out_pieces.append(piece)
        if i % 200 == 0 or i == len(tickers):
            logger.info("  beta+div_yield %d/%d (%.1fs)", i, len(tickers), time.time() - t0)

    pit_df = pd.concat(out_pieces, ignore_index=True)
    pit_df["date"] = pd.to_datetime(pit_df["date"])
    logger.info("PIT panel: %s", pit_df.shape)

    # Replace the old beta and dividend_yield columns
    logger.info("merging PIT columns into features matrix...")
    features = features.drop(columns=["beta", "dividend_yield"], errors="ignore")
    features = features.merge(pit_df, on=["date", "ticker"], how="left")
    features = features.rename(columns={
        "beta_pit": "beta",
        "dividend_yield_pit": "dividend_yield",
    })

    # Coverage stats
    n = len(features)
    n_beta_nn = features["beta"].notnull().sum()
    n_dy_nn = features["dividend_yield"].notnull().sum()
    logger.info("beta non-null: %d / %d (%.1f%%)", n_beta_nn, n, 100*n_beta_nn/n)
    logger.info("dividend_yield non-null: %d / %d (%.1f%%)", n_dy_nn, n, 100*n_dy_nn/n)

    features.to_parquet(FEATURES)
    logger.info("wrote %s (%s)", FEATURES, features.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
