"""Phase 5 analytics — per-ticker alpha attribution, decile analysis,
IC-vs-top-N decomposition, 12-month rolling win rate, concentration formalization.

Reads Phase 4's contract_v1 artifacts; produces:
  - per_ticker_attribution.parquet  (each ticker's contribution to total excess return)
  - decile_returns.parquet           (per-rebalance decile-bucketed forward returns)
  - ic_decomposition.parquet         (full-cross-section IC vs top-quintile IC, per model)
  - rolling_win_rate.parquet         (12-month rolling excess-return win rate)
  - concentration_summary.json       (top 5 holdings per rebalance, sector allocation over time)

Plus an aggregated `phase5_analytics_summary.json` with the headline numbers
for the eventual writeup.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "models" / "studies" / "larger_universe_v1" / "contract_v1"
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"

TEST_START = pd.Timestamp("2023-05-12")
TEST_END = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")

logger = logging.getLogger("phase5_analytics")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def load_inputs():
    holdings = pd.read_parquet(OUT_DIR / "holdings.parquet")
    portfolio = pd.read_parquet(OUT_DIR / "portfolio.parquet")
    benchmarks = pd.read_parquet(OUT_DIR / "benchmarks.parquet")
    scores = pd.read_parquet(OUT_DIR / "scores.parquet")
    for df in (holdings, portfolio, benchmarks, scores):
        df["date"] = pd.to_datetime(df["date"])

    # Daily closes (for ticker-level returns in attribution)
    closes = {}
    for sym in holdings["ticker"].unique():
        p = SNAPSHOT_PRICE_DIR / f"{sym}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            closes[sym] = df["close"]
    prices = pd.DataFrame(closes).sort_index()
    return holdings, portfolio, benchmarks, scores, prices


def per_ticker_attribution(holdings, prices, benchmarks):
    """Compute each ticker's contribution to total excess return per model.

    Algorithm: for each holding period (between rebalance i and i+1),
      position_return_period[t] = ticker_return_in_period
      excess_period[t] = weight_at_start × (ticker_return - SPY_return)
    Sum across periods per ticker. Express as fraction of model's total excess.
    """
    spy = benchmarks[benchmarks["benchmark"] == "SPY"].set_index("date")["nav"]
    rows = []
    for model in holdings["model"].unique():
        m_holdings = holdings[holdings["model"] == model].copy()
        m_holdings = m_holdings.sort_values(["date", "ticker"])
        rebalance_dates = sorted(m_holdings["date"].unique())

        # Compute SPY return per holding period
        spy_returns_by_period = {}
        for i, d_start in enumerate(rebalance_dates):
            d_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else prices.index[-1]
            s_start = spy.asof(d_start)
            s_end = spy.asof(d_end)
            if pd.notna(s_start) and pd.notna(s_end) and s_start > 0:
                spy_returns_by_period[d_start] = s_end / s_start - 1
            else:
                spy_returns_by_period[d_start] = 0.0

        # Per-ticker contribution accumulator
        contrib = {}
        for i, d_start in enumerate(rebalance_dates):
            d_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else prices.index[-1]
            spy_ret = spy_returns_by_period[d_start]
            day_h = m_holdings[m_holdings["date"] == d_start]
            for _, row in day_h.iterrows():
                t = row["ticker"]
                w = row["weight"]
                if t not in prices.columns:
                    continue
                p_start = prices[t].asof(d_start)
                p_end = prices[t].asof(d_end)
                if pd.isna(p_start) or pd.isna(p_end) or p_start <= 0:
                    continue
                ticker_ret = p_end / p_start - 1
                excess_contrib = w * (ticker_ret - spy_ret)
                contrib[t] = contrib.get(t, 0.0) + excess_contrib

        total_alpha = sum(contrib.values())
        for t, c in contrib.items():
            rows.append({
                "model": model,
                "ticker": t,
                "total_excess_contribution": c,
                "pct_of_total_alpha": (c / total_alpha * 100) if total_alpha != 0 else 0.0,
            })

    df = pd.DataFrame(rows).sort_values(["model", "total_excess_contribution"],
                                          ascending=[True, False]).reset_index(drop=True)
    return df


def decile_analysis(scores, prices, horizon=21):
    """For each rebalance date and model, sort by score, bucket into deciles,
    compute average forward return per decile. Aggregate across dates."""
    rows = []
    for model in scores["model"].unique():
        m_scores = scores[scores["model"] == model].copy()
        rebalance_dates = sorted(m_scores["date"].unique())
        decile_returns_acc = {i: [] for i in range(1, 11)}
        for d in rebalance_dates:
            day_scores = m_scores[m_scores["date"] == d].copy()
            day_scores = day_scores.dropna(subset=["score"])
            if len(day_scores) < 50:
                continue
            day_scores["decile"] = pd.qcut(day_scores["score"], 10,
                                              labels=range(1, 11), duplicates="drop")
            # Compute forward return per ticker
            d_ts = pd.Timestamp(d)
            for dec in range(1, 11):
                dec_tickers = day_scores[day_scores["decile"] == dec]["ticker"].tolist()
                forward_rets = []
                for t in dec_tickers:
                    if t not in prices.columns:
                        continue
                    p_start = prices[t].asof(d_ts)
                    # Find date that's `horizon` trading days later
                    idx_pos = prices.index.searchsorted(d_ts)
                    end_idx = min(idx_pos + horizon, len(prices.index) - 1)
                    d_end = prices.index[end_idx]
                    p_end = prices[t].asof(d_end)
                    if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                        forward_rets.append(p_end / p_start - 1)
                if forward_rets:
                    decile_returns_acc[dec].append(np.mean(forward_rets))
        # Aggregate
        for dec in range(1, 11):
            if decile_returns_acc[dec]:
                rows.append({
                    "model": model,
                    "decile": dec,
                    "mean_fwd_return": float(np.mean(decile_returns_acc[dec])),
                    "std_fwd_return": float(np.std(decile_returns_acc[dec])),
                    "n_rebalances": len(decile_returns_acc[dec]),
                })
    return pd.DataFrame(rows)


def ic_decomposition(scores, prices, horizon=21, top_quintile_pct=0.20):
    """Compute full-cross-section IC vs top-quintile-only IC per model.

    For each rebalance date:
      - Compute forward returns
      - Full IC = Spearman(scores, returns) over all eligible tickers
      - Top-quintile IC = Spearman restricted to top 20% by score
    Aggregate mean per model.
    """
    rows = []
    for model in scores["model"].unique():
        m_scores = scores[scores["model"] == model].copy()
        rebalance_dates = sorted(m_scores["date"].unique())
        full_ics, top_ics = [], []
        for d in rebalance_dates:
            day_scores = m_scores[m_scores["date"] == d].copy().dropna(subset=["score"])
            if len(day_scores) < 100:
                continue
            d_ts = pd.Timestamp(d)
            idx_pos = prices.index.searchsorted(d_ts)
            end_idx = min(idx_pos + horizon, len(prices.index) - 1)
            d_end = prices.index[end_idx]
            # Build per-ticker forward return
            tickers = day_scores["ticker"].values
            fwd_rets = []
            for t in tickers:
                if t in prices.columns:
                    p_start = prices[t].asof(d_ts)
                    p_end = prices[t].asof(d_end)
                    if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                        fwd_rets.append(p_end / p_start - 1)
                    else:
                        fwd_rets.append(np.nan)
                else:
                    fwd_rets.append(np.nan)
            day_scores["fwd_ret"] = fwd_rets
            day_scores = day_scores.dropna(subset=["fwd_ret"])
            if len(day_scores) < 50:
                continue
            if day_scores["score"].nunique() < 2 or day_scores["fwd_ret"].nunique() < 2:
                continue

            # Full IC
            rho, _ = spearmanr(day_scores["score"], day_scores["fwd_ret"])
            if rho == rho:
                full_ics.append(rho)
            # Top-quintile IC
            n_top = max(int(len(day_scores) * top_quintile_pct), 30)
            top_subset = day_scores.nlargest(n_top, "score")
            if top_subset["score"].nunique() >= 2 and top_subset["fwd_ret"].nunique() >= 2:
                rho_t, _ = spearmanr(top_subset["score"], top_subset["fwd_ret"])
                if rho_t == rho_t:
                    top_ics.append(rho_t)
        rows.append({
            "model": model,
            "full_ic_mean": float(np.mean(full_ics)) if full_ics else float("nan"),
            "full_ic_std": float(np.std(full_ics)) if full_ics else float("nan"),
            "top_quintile_ic_mean": float(np.mean(top_ics)) if top_ics else float("nan"),
            "top_quintile_ic_std": float(np.std(top_ics)) if top_ics else float("nan"),
            "n_dates_full": len(full_ics),
            "n_dates_top": len(top_ics),
        })
    return pd.DataFrame(rows)


def rolling_win_rate(portfolio, benchmarks, window_days=252):
    """For each 252-day rolling window starting from val_start, compute
    portfolio_return - SPY_return; report distribution + mean."""
    spy = benchmarks[benchmarks["benchmark"] == "SPY"].set_index("date")["nav"]
    rows = []
    for model in portfolio["model"].unique():
        m_port = portfolio[portfolio["model"] == model].set_index("date")["nav"].sort_index()
        # Align dates
        common = m_port.index.intersection(spy.index)
        m_port = m_port.loc[common]
        spy_a = spy.loc[common]
        excess_windows = []
        for i in range(len(common) - window_days):
            d_start = common[i]
            d_end = common[i + window_days]
            port_ret = m_port.iloc[i + window_days] / m_port.iloc[i] - 1
            spy_ret = spy_a.iloc[i + window_days] / spy_a.iloc[i] - 1
            excess_windows.append({
                "window_start": d_start,
                "window_end": d_end,
                "portfolio_return": float(port_ret),
                "spy_return": float(spy_ret),
                "excess_return": float(port_ret - spy_ret),
            })
        if excess_windows:
            wdf = pd.DataFrame(excess_windows)
            win_rate = (wdf["excess_return"] > 0).mean()
            rows.append({
                "model": model,
                "n_windows": len(wdf),
                "mean_excess_return": float(wdf["excess_return"].mean()),
                "median_excess_return": float(wdf["excess_return"].median()),
                "win_rate": float(win_rate),
                "best_window_excess": float(wdf["excess_return"].max()),
                "worst_window_excess": float(wdf["excess_return"].min()),
            })
    return pd.DataFrame(rows)


def concentration_summary(holdings):
    """Top 5 holdings per rebalance, sector allocation over time,
    summary stats. Returns a dict."""
    out = {}
    for model in holdings["model"].unique():
        m = holdings[holdings["model"] == model]
        # Sector allocation per date
        sec_alloc = m.groupby(["date", "sector"])["weight"].sum().reset_index()
        sec_max = sec_alloc.groupby("date")["weight"].max()
        # Top 5 most-held tickers (by repeat-count)
        counts = m["ticker"].value_counts().head(10)
        # Max single-ticker weight ever
        max_weight = m["weight"].max()
        # Average n_positions per rebalance
        avg_n = m.groupby("date").size().mean()
        out[model] = {
            "rebalance_dates": int(m["date"].nunique()),
            "unique_tickers_held": int(m["ticker"].nunique()),
            "avg_positions_per_rebalance": float(avg_n),
            "max_single_ticker_weight": float(max_weight),
            "max_sector_weight_across_dates": float(sec_max.max()),
            "median_sector_weight_max_per_date": float(sec_max.median()),
            "top_10_repeat_holdings": counts.to_dict(),
        }
    return out


def main() -> int:
    logger.info("=== Phase 5 analytics ===")
    t0 = time.time()
    holdings, portfolio, benchmarks, scores, prices = load_inputs()
    logger.info("loaded inputs in %.1fs", time.time() - t0)

    logger.info("computing per-ticker alpha attribution...")
    t0 = time.time()
    attrib = per_ticker_attribution(holdings, prices, benchmarks)
    attrib.to_parquet(OUT_DIR / "per_ticker_attribution.parquet")
    logger.info("  attribution done in %.1fs; %d rows", time.time() - t0, len(attrib))

    logger.info("computing decile-return analysis...")
    t0 = time.time()
    deciles = decile_analysis(scores, prices)
    deciles.to_parquet(OUT_DIR / "decile_returns.parquet")
    logger.info("  decile analysis done in %.1fs; %d rows", time.time() - t0, len(deciles))

    logger.info("computing IC-vs-top-N decomposition...")
    t0 = time.time()
    ic_dec = ic_decomposition(scores, prices)
    ic_dec.to_parquet(OUT_DIR / "ic_decomposition.parquet")
    logger.info("  IC decomposition done in %.1fs; %d rows", time.time() - t0, len(ic_dec))

    logger.info("computing 12-month rolling win rate...")
    t0 = time.time()
    win = rolling_win_rate(portfolio, benchmarks)
    win.to_parquet(OUT_DIR / "rolling_win_rate.parquet")
    logger.info("  win rate done in %.1fs; %d rows", time.time() - t0, len(win))

    logger.info("computing concentration summary...")
    t0 = time.time()
    conc = concentration_summary(holdings)
    (OUT_DIR / "concentration_summary.json").write_text(
        json.dumps(conc, indent=2, default=str), encoding="utf-8")
    logger.info("  concentration summary done in %.1fs", time.time() - t0)

    # Headline summary
    print()
    print("=== Per-ticker alpha attribution (top 10 contributors per model) ===")
    for model in attrib["model"].unique():
        m = attrib[attrib["model"] == model].head(10)
        print(f"\n  {model}:")
        for _, r in m.iterrows():
            print(f"    {r['ticker']:8s}  alpha={r['total_excess_contribution']:+.4f}  "
                  f"pct_of_total={r['pct_of_total_alpha']:+.1f}%")
        # Max single-ticker share
        max_share = attrib[attrib["model"] == model]["pct_of_total_alpha"].max()
        min_share = attrib[attrib["model"] == model]["pct_of_total_alpha"].min()
        print(f"    --- max single-ticker share: {max_share:+.1f}%  min: {min_share:+.1f}% ---")

    print()
    print("=== Decile-return analysis (avg forward 21d return per decile) ===")
    print(deciles.to_string(index=False))

    print()
    print("=== IC decomposition (full-cross-section vs top-quintile) ===")
    print(ic_dec.to_string(index=False))

    print()
    print("=== 12-month rolling win rate ===")
    print(win.to_string(index=False))

    print()
    print("=== Concentration summary ===")
    print(json.dumps(conc, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
