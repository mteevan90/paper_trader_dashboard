import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features to an OHLCV DataFrame."""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI
    df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

    # MACD (macd_signal scaled 1.25x)
    macd = MACD(close=close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal() * 1.25

    # Bollinger Band width
    bb = BollingerBands(close=close)
    df["bb_width"] = bb.bollinger_wband()

    # Simple moving averages
    df["sma_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["sma_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(close=close, window=200).sma_indicator()

    # Volume change vs 5-day average
    vol_avg_5 = volume.rolling(window=5).mean()
    df["volume_change"] = (volume - vol_avg_5) / vol_avg_5

    # Distance from 52-week high
    high_52w = close.rolling(window=252, min_periods=1).max()
    df["dist_52w_high"] = (close - high_52w) / high_52w

    # Average True Range (14-day)
    df["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # On-Balance Volume
    df["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    # Rate of Change
    df["roc_5"] = ROCIndicator(close=close, window=5).roc()
    df["roc_20"] = ROCIndicator(close=close, window=20).roc()

    # Academic 12-1 momentum: return from 12 months ago to 1 month ago
    df["momentum_12_1"] = close.shift(21) / close.shift(252) - 1

    df = df.dropna()
    return df


def normalize_features(df: pd.DataFrame, feature_cols: list[str],
                       window: int = 60) -> pd.DataFrame:
    """Rolling z-score normalize the given feature columns in place.

    Uses a 60-day rolling window with min_periods=20 so early rows still get
    a value after the warmup period. Columns not present in df are skipped.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        rm = df[col].rolling(window=window, min_periods=20).mean()
        rs = df[col].rolling(window=window, min_periods=20).std()
        df[col] = (df[col] - rm) / rs.replace(0, np.nan)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the soft next-day-return target used for training.

    Kept separate from add_features so that backtesting and live scoring
    paths never see the target column (which is computed from future data
    and would cause lookahead bias if exposed as a feature).
    """
    df = df.sort_index().copy()
    close = df["Close"]
    next_day_ret = close.shift(-5) / close - 1
    df["target"] = np.where(
        next_day_ret > 0.03, 0.9,
        np.where(next_day_ret > 0, 0.6,
                 np.where(next_day_ret > -0.03, 0.4, 0.1)))
    df.loc[next_day_ret.isna(), "target"] = np.nan
    return df.dropna(subset=["target"])


def _download_spy_close(start: str, end: str) -> pd.Series:
    """Download SPY close prices for relative strength calculation."""
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    return spy["Close"]


def _download_vix(start: str, end: str) -> pd.DataFrame:
    """Download VIX data."""
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)
    return vix


def add_market_features(
    featured_data: dict[str, pd.DataFrame],
    price_data: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Add sector momentum, VIX features, relative strength, and market breadth.

    Replaces the old add_sector_momentum function — call this instead.
    """
    if not featured_data:
        return featured_data

    # Guard: if every per-ticker frame ended up empty after add_features(),
    # all_dates is empty and the line below indexes into [0]. That's a
    # silent upstream bug — the most common cause is too short a price
    # history window (momentum_12_1 needs 252 trading days of lookback, so
    # anything under ~253 trading days leaves dropna() with zero rows).
    non_empty = {t: df for t, df in featured_data.items() if not df.empty}
    if not non_empty:
        n_price = len(price_data)
        max_rows = max((len(df) for df in price_data.values()), default=0)
        raise ValueError(
            f"add_market_features: all {len(featured_data)} featured frames "
            f"are empty after add_features(). price_data has {n_price} tickers, "
            f"longest history is {max_rows} rows. Likely cause: the price "
            f"window passed to get_stock_data_cached() is too short for the "
            f"252-day lookback in momentum_12_1 — pass at least ~1.5 years "
            f"of history (e.g. start = today - 500 days)."
        )
    featured_data = non_empty

    # Determine date range from the data
    all_dates = sorted(set().union(*(df.index for df in featured_data.values())))
    start = (all_dates[0] - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end = (all_dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # --- Download SPY and VIX once ---
    print("    Downloading SPY & VIX for market features...")
    spy_close = _download_spy_close(start, end)
    vix_df = _download_vix(start, end)
    vix_close = vix_df["Close"] if not vix_df.empty else pd.Series(dtype=float)

    # SPY returns for relative strength
    spy_ret_20d = spy_close.pct_change(periods=20)
    spy_ret_50d = spy_close.pct_change(periods=50)
    spy_ret_5d = spy_close.pct_change(periods=5)

    # VIX features (series aligned to all dates)
    vix_20d_avg = vix_close.rolling(window=20).mean()
    vix_regime = pd.cut(vix_close, bins=[-np.inf, 20, 30, np.inf],
                        labels=[0, 1, 2]).astype(float)
    vix_change = vix_close.pct_change()

    # --- Market breadth: advance-decline across all stocks ---
    print("    Computing market breadth...")
    daily_changes = {}
    for ticker, df in price_data.items():
        daily_changes[ticker] = df["Close"].pct_change()

    changes_df = pd.DataFrame(daily_changes)
    advances = (changes_df > 0).sum(axis=1)
    declines = (changes_df < 0).sum(axis=1)
    total = advances + declines
    ad_line = (advances - declines) / total.replace(0, 1)
    ad_20d_avg = ad_line.rolling(window=20).mean()
    ad_divergence = (spy_ret_5d.reindex(ad_20d_avg.index) - ad_20d_avg) * 1.25  # scaled

    # --- Sector momentum ---
    print("    Computing sector momentum...")
    returns_20d = {}
    for ticker, df in price_data.items():
        returns_20d[ticker] = df["Close"].pct_change(periods=20)

    sectors: dict[str, list[str]] = {}
    for ticker in featured_data:
        sector = sector_map.get(ticker, "other")
        sectors.setdefault(sector, []).append(ticker)

    # --- Apply all features to each ticker ---
    for ticker, df in featured_data.items():
        df = df.copy()

        # Sector momentum (scaled 0.75x)
        sector = sector_map.get(ticker, "other")
        peers = [t for t in sectors[sector] if t != ticker and t in returns_20d]
        if peers:
            peer_returns = pd.concat(
                [returns_20d[t].rename(t) for t in peers], axis=1)
            df["sector_momentum"] = peer_returns.mean(axis=1).reindex(df.index) * 0.75
        else:
            df["sector_momentum"] = 0.0

        # VIX features
        df["vix_level"] = vix_close.reindex(df.index)
        df["vix_20d_avg"] = vix_20d_avg.reindex(df.index)
        df["vix_regime"] = vix_regime.reindex(df.index) * 1.25  # scaled
        df["vix_change"] = vix_change.reindex(df.index)

        # Relative strength vs SPY (rs_spy scaled 1.25x)
        stock_ret_20d = price_data[ticker]["Close"].pct_change(periods=20)
        stock_ret_50d = price_data[ticker]["Close"].pct_change(periods=50)
        spy_20 = spy_ret_20d.reindex(df.index)
        spy_50 = spy_ret_50d.reindex(df.index)
        df["rs_spy"] = (stock_ret_20d.reindex(df.index) / spy_20.replace(0, np.nan)) * 1.25
        df["rs_spy_50d"] = stock_ret_50d.reindex(df.index) / spy_50.replace(0, np.nan)

        # Market breadth
        df["ad_line"] = ad_line.reindex(df.index)
        df["ad_20d_avg"] = ad_20d_avg.reindex(df.index)
        df["ad_divergence"] = ad_divergence.reindex(df.index)

        # Clip extreme values in relative strength
        df["rs_spy"] = df["rs_spy"].clip(-10, 10)
        df["rs_spy_50d"] = df["rs_spy_50d"].clip(-10, 10)

        # Preserve raw copies of columns used by backtest.score_technical before
        # z-scoring destroys their natural scale.
        for col in ("sma_200", "sma_50", "rsi_14", "dist_52w_high", "obv"):
            if col in df.columns:
                df[f"{col}_raw"] = df[col]

        # Rolling z-score normalize all feature columns (per-ticker).
        from model import FEATURE_COLS
        df = normalize_features(df, FEATURE_COLS)

        df = df.dropna()
        featured_data[ticker] = df

    return featured_data


if __name__ == "__main__":
    from datetime import datetime, timedelta
    from fetch_data import get_stock_data, SECTOR_MAP

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    tickers = ["TSLA", "MSFT", "AAPL"]

    data = get_stock_data(tickers, start, end)

    featured_data = {}
    for ticker, df in data.items():
        featured_data[ticker] = add_features(df)

    featured_data = add_market_features(featured_data, data, SECTOR_MAP)

    for ticker, df in featured_data.items():
        print(f"\n{ticker} — shape: {df.shape}")
        print(df[["sector_momentum", "vix_level", "rs_spy", "ad_line"]].head())
