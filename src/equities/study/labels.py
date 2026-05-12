"""Labels (prediction targets) for the Larger Universe v1 study.

The model is rebalanced weekly (every Friday close) and outputs a continuous
score per stock that gets transformed to a target weight. The natural label
is forward 5-trading-day return:

  target(D, T) = close(D+5, T) / close(D, T) - 1

The 5-day forward horizon matches the rebalance cadence; predictions at
date D inform the portfolio held over [D+1, D+5].

For the CV splitter, the 5-day label means that train rows at D=K have
labels referring to prices at D=K+5. So validation rows can only start
at D >= train_end + 5 + 1 to be leak-free. Use ``EMBARGO_TRADING_DAYS``
in the CV splitter to enforce this.

Labels are computed once per ticker against that ticker's price series
in the snapshot.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_HORIZON_TRADING_DAYS = 5
EMBARGO_TRADING_DAYS = 5  # = label horizon

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"


def compute_forward_return(close: pd.Series, horizon: int = LABEL_HORIZON_TRADING_DAYS) -> pd.Series:
    """close[t+horizon] / close[t] - 1, with NaN at the last `horizon` rows."""
    return close.shift(-horizon) / close - 1.0


def build_labels(tickers: list[str]) -> pd.DataFrame:
    """Return a long DataFrame keyed on (date, ticker) with column `target_fwd_5d`."""
    pieces = []
    for sym in tickers:
        p = SNAPSHOT_PRICE_DIR / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index)
        fwd = compute_forward_return(df["close"])
        piece = pd.DataFrame({
            "date": df.index,
            "ticker": sym,
            "target_fwd_5d": fwd.values,
        })
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out
