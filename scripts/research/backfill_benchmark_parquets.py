"""One-off backfill: add SPY_close.parquet + QQQ_close.parquet to every
existing models/cache/dashboard_results/<label>/ that has a portfolio.parquet
but no benchmark snapshot yet.

Reason: cloud Performance tab was missing SPY because Streamlit Cloud's
shared IP gets soft-throttled by Yahoo (yfinance) for SPY in particular.
The fix is to make benchmark data self-contained per label so the
dashboard never needs yfinance at view time. _save_one_backtest_result
now emits these parquets on every save; this script populates the dirs
saved before that change.

Skips dirs that already have both files. Skips the v3_track2_perturbation
aggregation dir (no portfolio.parquet to derive a window from). Logs
each label's window + the number of rows fetched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent
DR = REPO_ROOT / "models" / "cache" / "dashboard_results"


def fetch_close(ticker: str, start: pd.Timestamp,
                end: pd.Timestamp) -> "pd.Series | None":
    """Fetch ticker Close prices for [start, end] inclusive."""
    h = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
    )
    if h is None or h.empty:
        return None
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.droplevel(1)
    if "Close" not in h.columns:
        return None
    s = h["Close"]
    s = s.loc[(s.index >= start) & (s.index <= end)]
    return s


def main() -> int:
    if not DR.exists():
        print(f"[BACKFILL] No dashboard_results dir at {DR}")
        return 1
    saved_total = 0
    for label_dir in sorted(DR.iterdir()):
        if not label_dir.is_dir():
            continue
        portfolio = label_dir / "portfolio.parquet"
        if not portfolio.exists():
            print(f"[skip ] {label_dir.name}: no portfolio.parquet")
            continue
        try:
            pv = pd.read_parquet(portfolio)
        except Exception as e:
            print(f"[skip ] {label_dir.name}: portfolio.parquet unreadable ({e})")
            continue
        if pv.empty:
            print(f"[skip ] {label_dir.name}: empty portfolio")
            continue
        bm_start = pd.to_datetime(pv.index[0])
        bm_end   = pd.to_datetime(pv.index[-1])
        any_written = False
        for tkr in ("SPY", "QQQ"):
            target = label_dir / f"{tkr}_close.parquet"
            if target.exists():
                continue
            close = fetch_close(tkr, bm_start, bm_end)
            if close is None or close.empty:
                print(f"  [warn] {label_dir.name}/{tkr}: yfinance returned empty")
                continue
            close.to_frame("Close").to_parquet(target)
            print(f"  [save] {label_dir.name}/{tkr}_close.parquet "
                  f"({len(close)} rows, {bm_start.date()} to {bm_end.date()})")
            saved_total += 1
            any_written = True
        if not any_written:
            print(f"[skip ] {label_dir.name}: SPY + QQQ already present")
    print(f"\n[BACKFILL] Done. {saved_total} new files written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
