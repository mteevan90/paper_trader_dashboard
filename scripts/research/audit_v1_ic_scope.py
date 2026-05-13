"""v1 price-universe scope audit — reproducer.

This script recomputes v1's IC decomposition and decile_returns at three
different price-universe scopes, demonstrating that v1's pinned artifacts
are computed with prices restricted to held tickers (~450 across both
models) while the standard interpretation of these analytics expects the
full eligible universe (~1963 tickers).

Surfaces the comparison referenced by
`docs/studies/larger_universe_v1/ic_scope_audit.md`.

Run:
    python scripts/research/audit_v1_ic_scope.py

Outputs to stdout. Does NOT modify any v1 artifacts. The audit is
informational; corrections require explicit decisions.

The script's formulas are verbatim copies of v1's `phase5_analytics.py`
`ic_decomposition` and `decile_analysis` — the only thing this script
varies is the set of tickers contributing to the `prices` DataFrame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

V1_DIR = ROOT / "models" / "studies" / "larger_universe_v1" / "contract_v1"
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"


def _load_prices(tickers: set[str]) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for sym in tickers:
        p = SNAPSHOT_PRICE_DIR / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index)
        closes[sym] = df["close"]
    return pd.DataFrame(closes).sort_index()


def ic_decomposition_v1formula(
    scores: pd.DataFrame, prices: pd.DataFrame,
    horizon: int = 21, top_quintile_pct: float = 0.20,
) -> dict:
    """Verbatim from v1 phase5_analytics.py:167-229. The only variable across
    runs is the `prices` DataFrame's column set."""
    rebalance_dates = sorted(scores["date"].unique())
    full_ics, top_ics = [], []
    for d in rebalance_dates:
        day_scores = scores[scores["date"] == d].copy().dropna(subset=["score"])
        if len(day_scores) < 100:
            continue
        d_ts = pd.Timestamp(d)
        idx_pos = prices.index.searchsorted(d_ts)
        end_idx = min(idx_pos + horizon, len(prices.index) - 1)
        d_end = prices.index[end_idx]
        tickers = day_scores["ticker"].values
        fwd = []
        for t in tickers:
            if t in prices.columns:
                p_start = prices[t].asof(d_ts)
                p_end = prices[t].asof(d_end)
                if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                    fwd.append(p_end / p_start - 1)
                else:
                    fwd.append(np.nan)
            else:
                fwd.append(np.nan)
        day_scores["fwd_ret"] = fwd
        day_scores = day_scores.dropna(subset=["fwd_ret"])
        if len(day_scores) < 50:
            continue
        if day_scores["score"].nunique() < 2 or day_scores["fwd_ret"].nunique() < 2:
            continue
        rho, _ = spearmanr(day_scores["score"], day_scores["fwd_ret"])
        if rho == rho:
            full_ics.append(rho)
        n_top = max(int(len(day_scores) * top_quintile_pct), 30)
        top_subset = day_scores.nlargest(n_top, "score")
        if top_subset["score"].nunique() >= 2 and top_subset["fwd_ret"].nunique() >= 2:
            rho_t, _ = spearmanr(top_subset["score"], top_subset["fwd_ret"])
            if rho_t == rho_t:
                top_ics.append(rho_t)
    return {
        "full_ic_mean": float(np.mean(full_ics)) if full_ics else float("nan"),
        "full_ic_std": float(np.std(full_ics)) if full_ics else float("nan"),
        "top_quintile_ic_mean": float(np.mean(top_ics)) if top_ics else float("nan"),
        "top_quintile_ic_std": float(np.std(top_ics)) if top_ics else float("nan"),
        "n_dates_full": len(full_ics),
        "n_dates_top": len(top_ics),
    }


def decile_v1formula(
    scores: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21,
) -> pd.DataFrame:
    """Verbatim from v1 phase5_analytics.py:121-164. The only variable across
    runs is the `prices` DataFrame's column set."""
    rows = []
    for model in scores["model"].unique():
        m_scores = scores[scores["model"] == model].copy()
        rebalance_dates = sorted(m_scores["date"].unique())
        decile_returns_acc: dict[int, list[float]] = {i: [] for i in range(1, 11)}
        for d in rebalance_dates:
            day_scores = m_scores[m_scores["date"] == d].copy().dropna(subset=["score"])
            if len(day_scores) < 50:
                continue
            day_scores["decile"] = pd.qcut(
                day_scores["score"], 10, labels=range(1, 11), duplicates="drop",
            )
            d_ts = pd.Timestamp(d)
            idx_pos = prices.index.searchsorted(d_ts)
            end_idx = min(idx_pos + horizon, len(prices.index) - 1)
            d_end = prices.index[end_idx]
            for dec in range(1, 11):
                dec_tickers = day_scores[day_scores["decile"] == dec]["ticker"].tolist()
                fwd: list[float] = []
                for t in dec_tickers:
                    if t not in prices.columns:
                        continue
                    p_start = prices[t].asof(d_ts)
                    p_end = prices[t].asof(d_end)
                    if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                        fwd.append(p_end / p_start - 1)
                if fwd:
                    decile_returns_acc[dec].append(float(np.mean(fwd)))
        for dec in range(1, 11):
            if decile_returns_acc[dec]:
                rows.append({
                    "model": model, "decile": dec,
                    "mean_fwd_return": float(np.mean(decile_returns_acc[dec])),
                    "std_fwd_return": float(np.std(decile_returns_acc[dec])),
                    "n_rebalances": len(decile_returns_acc[dec]),
                })
    return pd.DataFrame(rows)


def main() -> int:
    # Load v1's scores + holdings
    v1_scores = pd.read_parquet(V1_DIR / "scores.parquet")
    v1_scores["date"] = pd.to_datetime(v1_scores["date"])
    v1_xgb_scores = v1_scores[v1_scores["model"] == "xgboost"].copy()

    v1_holdings = pd.read_parquet(V1_DIR / "holdings.parquet")
    v1_all_held = set(v1_holdings["ticker"].astype(str).unique())
    v1_xgb_held = set(
        v1_holdings[v1_holdings["model"] == "xgboost"]["ticker"].astype(str).unique()
    )

    # Full snapshot universe
    all_snapshot = sorted(p.stem for p in SNAPSHOT_PRICE_DIR.glob("*.parquet"))

    print(f"Snapshot tickers (full universe):    {len(all_snapshot)}")
    print(f"v1 held tickers (XGB + ENet):        {len(v1_all_held)}")
    print(f"v1 XGB-only held tickers:            {len(v1_xgb_held)}")
    print()

    prices_v1 = _load_prices(v1_all_held)
    prices_xgb = _load_prices(v1_xgb_held)
    prices_full = _load_prices(all_snapshot)

    print("=" * 72)
    print("IC decomposition (XGBoost, v1 formula on v1 scores, varied scope)")
    print("=" * 72)
    print()
    for name, prices in [
        (f"v1 holdings ({len(v1_all_held)} tickers)", prices_v1),
        (f"XGB held only ({len(v1_xgb_held)} tickers)", prices_xgb),
        (f"full snapshot universe ({len(all_snapshot)} tickers)", prices_full),
    ]:
        res = ic_decomposition_v1formula(v1_xgb_scores, prices)
        print(f"-- {name} --")
        print(f"   full_ic_mean:         {res['full_ic_mean']:+.6f}  (std {res['full_ic_std']:.4f})")
        print(f"   top_quintile_ic_mean: {res['top_quintile_ic_mean']:+.6f}  (std {res['top_quintile_ic_std']:.4f})")
        print(f"   n_dates_full: {res['n_dates_full']}, n_dates_top: {res['n_dates_top']}")
        print()

    v1_pinned = pd.read_parquet(V1_DIR / "ic_decomposition.parquet")
    v1_xgb_pinned = v1_pinned[v1_pinned["model"] == "xgboost"].iloc[0]
    print("Pinned v1 ic_decomposition.parquet (XGBoost) — for cross-check:")
    print(f"   full_ic_mean:         {v1_xgb_pinned['full_ic_mean']:+.6f}")
    print(f"   top_quintile_ic_mean: {v1_xgb_pinned['top_quintile_ic_mean']:+.6f}")
    print()

    print("=" * 72)
    print("Decile returns (XGBoost, v1 formula on v1 scores, varied scope)")
    print("=" * 72)
    print()
    for name, prices in [
        (f"v1 holdings ({len(v1_all_held)} tickers)", prices_v1),
        (f"XGB held only ({len(v1_xgb_held)} tickers)", prices_xgb),
        (f"full snapshot universe ({len(all_snapshot)} tickers)", prices_full),
    ]:
        df = decile_v1formula(v1_xgb_scores, prices)
        print(f"-- {name} --")
        print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        print()

    print("Pinned v1 decile_returns.parquet (XGBoost) — for cross-check:")
    v1_dec_pinned = pd.read_parquet(V1_DIR / "decile_returns.parquet")
    print(
        v1_dec_pinned[v1_dec_pinned["model"] == "xgboost"]
        .to_string(index=False, float_format=lambda v: f"{v:+.4f}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
