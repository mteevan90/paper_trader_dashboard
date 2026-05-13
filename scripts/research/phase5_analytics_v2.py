"""Phase 5 analytics — Larger Universe v2 per-variant analytics + cross-variant
concentration overlap.

Produces per-variant Phase 5 artifacts under
`models/studies/larger_universe_v2/<variant_subdir>/contract_v1/`:

  decile_returns.parquet           (decile-bucketed forward returns)
  per_ticker_attribution.parquet   (each ticker's share of total alpha)
  ic_decomposition.parquet         (full-cross-section IC vs top-quintile IC)
  rolling_win_rate.parquet         (12-month rolling excess-return distribution)
  concentration_summary.json       (top holdings, sector allocation, etc.)

Plus a study-level cross-variant artifact at
`models/studies/larger_universe_v2/comparison/concentration_overlap.parquet`
answering the C5 question: is universal C5 failure concentrated in the SAME
tickers across variants (model-determined) or DIFFERENT tickers
(construction-specific)?

Compute optimization: decile_returns and ic_decomposition depend only on
scores.parquet, which is bit-identical across all v2 variants (same v1
locked XGBoost model + features). Compute once on baseline's scores and
write identical files to every variant's contract_v1/ subdir. The
variant-specific analytics (attribution, win rate, concentration summary)
depend on the per-variant holdings/portfolio and are computed per variant.

CLI:
    python scripts/research/phase5_analytics_v2.py --variants all
    python scripts/research/phase5_analytics_v2.py --variants baseline,b4_concentration_penalties

The metric formulas mirror v1's phase5_analytics.py verbatim so v2-baseline's
output is directly comparable to v1's.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

V2_OUT_DIR = ROOT / "models" / "studies" / "larger_universe_v2"
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"

TEST_START = pd.Timestamp("2023-05-12")
TEST_END = pd.Timestamp("2025-12-31")

ALL_VARIANTS = [
    "baseline",
    "b1_vol_target",
    "b2_conviction_weighted",
    "b3_dynamic_topn",
    "b4_concentration_penalties",
    "b5_defensive_sleeves",
    "b6_smaller_caps",
]

logger = logging.getLogger("phase5_analytics_v2")


# ============================================================================
# Helpers (mirror v1's phase5_analytics.py)
# ============================================================================

def _load_variant_inputs(variant_name: str, prices: pd.DataFrame) -> dict:
    base = V2_OUT_DIR / variant_name / "contract_v1"
    required = ["portfolio.parquet", "holdings.parquet",
                "benchmarks.parquet", "scores.parquet"]
    for r in required:
        if not (base / r).exists():
            raise FileNotFoundError(f"{variant_name} missing {r}")
    holdings = pd.read_parquet(base / "holdings.parquet")
    portfolio = pd.read_parquet(base / "portfolio.parquet")
    benchmarks = pd.read_parquet(base / "benchmarks.parquet")
    scores = pd.read_parquet(base / "scores.parquet")
    for df in (holdings, portfolio, benchmarks, scores):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
    return {
        "holdings": holdings, "portfolio": portfolio,
        "benchmarks": benchmarks, "scores": scores,
        "prices": prices,
    }


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


def _per_ticker_attribution(
    holdings: pd.DataFrame, prices: pd.DataFrame, benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    """Each ticker's contribution to total excess return.

    Algorithm (mirrors v1 phase5_analytics.per_ticker_attribution):
      for each holding period (rebalance i to i+1):
          excess_period[t] = weight[t] × (ticker_return - SPY_return)
      Sum across periods per ticker. Express as fraction of total excess.

    Computed across the FULL window the variant's portfolio covers (test +
    OOS), matching v1's Phase 5 scope. Filtering to test-only happens in
    build_comparison_results_v2.py (criterion 5 specifically).
    """
    spy = benchmarks[benchmarks["benchmark"] == "SPY"].set_index("date")["nav"]
    rows = []
    for model in holdings["model"].unique():
        m_holdings = holdings[holdings["model"] == model].copy()
        m_holdings = m_holdings.sort_values(["date", "ticker"])
        rebalance_dates = sorted(m_holdings["date"].unique())
        last_date = prices.index[-1]

        spy_returns_by_period: dict[pd.Timestamp, float] = {}
        for i, d_start in enumerate(rebalance_dates):
            d_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else last_date
            s_start = spy.asof(d_start)
            s_end = spy.asof(d_end)
            if pd.notna(s_start) and pd.notna(s_end) and s_start > 0:
                spy_returns_by_period[d_start] = float(s_end / s_start - 1)
            else:
                spy_returns_by_period[d_start] = 0.0

        contrib: dict[str, float] = {}
        for i, d_start in enumerate(rebalance_dates):
            d_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else last_date
            spy_ret = spy_returns_by_period[d_start]
            day_h = m_holdings[m_holdings["date"] == d_start]
            for _, row in day_h.iterrows():
                t = row["ticker"]
                w = float(row["weight"])
                if t not in prices.columns:
                    continue
                p_start = prices[t].asof(d_start)
                p_end = prices[t].asof(d_end)
                if pd.isna(p_start) or pd.isna(p_end) or p_start <= 0:
                    continue
                ticker_ret = float(p_end / p_start - 1)
                contrib[t] = contrib.get(t, 0.0) + w * (ticker_ret - spy_ret)

        total_alpha = sum(contrib.values())
        for t, c in contrib.items():
            rows.append({
                "model": model, "ticker": t,
                "total_excess_contribution": c,
                "pct_of_total_alpha": (c / total_alpha * 100) if total_alpha != 0 else 0.0,
            })

    df = pd.DataFrame(rows).sort_values(
        ["model", "total_excess_contribution"], ascending=[True, False],
    ).reset_index(drop=True)
    return df


def _decile_analysis(
    scores: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21,
) -> pd.DataFrame:
    """Per-rebalance decile-bucketed forward returns."""
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


def _ic_decomposition(
    scores: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21,
    top_quintile_pct: float = 0.20,
) -> pd.DataFrame:
    """Full-cross-section IC vs top-quintile-only IC."""
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
            tickers = day_scores["ticker"].values
            fwd: list[float] = []
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


def _rolling_win_rate(
    portfolio: pd.DataFrame, benchmarks: pd.DataFrame, window_days: int = 252,
) -> pd.DataFrame:
    """12-month rolling excess-return distribution. Full window (test+OOS),
    matching v1's Phase 5 scope. Criterion-6 filtering to test-window only
    happens in build_comparison_results_v2.py."""
    spy = benchmarks[benchmarks["benchmark"] == "SPY"].set_index("date")["nav"]
    rows = []
    for model in portfolio["model"].unique():
        m_port = portfolio[portfolio["model"] == model].set_index("date")["nav"].sort_index()
        common = m_port.index.intersection(spy.index)
        m_port = m_port.loc[common]
        spy_a = spy.loc[common]
        excess: list[dict] = []
        for i in range(len(common) - window_days):
            d_start = common[i]
            d_end = common[i + window_days]
            port_ret = m_port.iloc[i + window_days] / m_port.iloc[i] - 1
            spy_ret = spy_a.iloc[i + window_days] / spy_a.iloc[i] - 1
            excess.append({
                "window_start": d_start, "window_end": d_end,
                "portfolio_return": float(port_ret), "spy_return": float(spy_ret),
                "excess_return": float(port_ret - spy_ret),
            })
        if excess:
            wdf = pd.DataFrame(excess)
            rows.append({
                "model": model,
                "n_windows": len(wdf),
                "mean_excess_return": float(wdf["excess_return"].mean()),
                "median_excess_return": float(wdf["excess_return"].median()),
                "win_rate": float((wdf["excess_return"] > 0).mean()),
                "best_window_excess": float(wdf["excess_return"].max()),
                "worst_window_excess": float(wdf["excess_return"].min()),
            })
    return pd.DataFrame(rows)


def _concentration_summary(holdings: pd.DataFrame) -> dict:
    out: dict = {}
    for model in holdings["model"].unique():
        m = holdings[holdings["model"] == model]
        sec_alloc = m.groupby(["date", "sector"])["weight"].sum().reset_index()
        sec_max = sec_alloc.groupby("date")["weight"].max()
        counts = m["ticker"].value_counts().head(10)
        out[model] = {
            "rebalance_dates": int(m["date"].nunique()),
            "unique_tickers_held": int(m["ticker"].nunique()),
            "avg_positions_per_rebalance": float(m.groupby("date").size().mean()),
            "max_single_ticker_weight": float(m["weight"].max()),
            "max_sector_weight_across_dates": float(sec_max.max()),
            "median_sector_weight_max_per_date": float(sec_max.median()),
            "top_10_repeat_holdings": counts.to_dict(),
        }
    return out


# ============================================================================
# Cross-variant concentration overlap (the C5 question)
# ============================================================================

def _concentration_overlap(
    attributions: dict[str, pd.DataFrame], top_k: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Cross-variant analysis: are top alpha contributors the same tickers
    across variants (model-determined concentration) or different
    (construction-specific concentration)?

    Returns:
      - per_ticker_df: one row per (ticker, variant) for tickers in any
        variant's top_k. Columns: ticker, variant, rank, pct_of_total_alpha.
        Useful for dashboard rendering of the overlap structure.
      - summary: dict with cross-variant correlation matrix of
        pct_of_total_alpha vectors + counts of "appears in N variants' top_k".

    Interpretation:
      - High Spearman correlation across variant pairs ⇒ model-determined
        concentration (same tickers drive alpha in all constructions).
      - Low correlation ⇒ construction-specific concentration.
      - Tickers appearing in many variants' top_k ⇒ model-bias signal.
      - Tickers appearing in only 1-2 variants' top_k ⇒ construction-specific.
    """
    # Per-variant top_k contributors
    rows = []
    union_top_tickers: set[str] = set()
    for v_name, df in attributions.items():
        m = df[df["model"] == "xgboost"].copy()
        m = m.sort_values("total_excess_contribution", ascending=False).head(top_k)
        for rank, (_, r) in enumerate(m.iterrows(), 1):
            rows.append({
                "variant": v_name, "ticker": r["ticker"], "rank": rank,
                "total_excess_contribution": float(r["total_excess_contribution"]),
                "pct_of_total_alpha": float(r["pct_of_total_alpha"]),
            })
            union_top_tickers.add(str(r["ticker"]))
    per_variant_top = pd.DataFrame(rows)

    # For each ticker in the union: how many variants is it in top_k of?
    appearance_counts: dict[str, int] = {}
    for t in union_top_tickers:
        appearance_counts[t] = int(
            (per_variant_top["ticker"] == t).sum()
        )

    # Cross-variant Spearman correlation on pct_of_total_alpha vectors over
    # the union of all tickers (broader than top_k — uses every ticker that
    # contributed in any variant).
    all_contrib_tickers: set[str] = set()
    for df in attributions.values():
        all_contrib_tickers.update(
            df[df["model"] == "xgboost"]["ticker"].astype(str).unique()
        )
    pct_matrix = pd.DataFrame(index=sorted(all_contrib_tickers))
    for v_name, df in attributions.items():
        m = df[df["model"] == "xgboost"].set_index("ticker")["pct_of_total_alpha"]
        pct_matrix[v_name] = m.reindex(pct_matrix.index).fillna(0.0)

    variants = list(attributions.keys())
    corr_rows = []
    for i, v1 in enumerate(variants):
        for v2 in variants:
            r, _ = spearmanr(pct_matrix[v1], pct_matrix[v2])
            corr_rows.append({"variant_a": v1, "variant_b": v2,
                                "spearman_corr": float(r) if r == r else float("nan")})
    corr_df = pd.DataFrame(corr_rows)

    # Distribution: # tickers appearing in 1, 2, ..., 7 variants' top_k
    dist: dict[int, int] = {}
    for n in range(1, len(attributions) + 1):
        dist[n] = sum(1 for v in appearance_counts.values() if v == n)

    summary = {
        "top_k": top_k,
        "union_top_tickers_count": len(union_top_tickers),
        "n_total_contributing_tickers": len(all_contrib_tickers),
        "appearance_count_distribution": dist,
        "cross_variant_spearman_mean": float(
            corr_df[corr_df["variant_a"] != corr_df["variant_b"]]["spearman_corr"].mean()
        ),
        "cross_variant_spearman_median": float(
            corr_df[corr_df["variant_a"] != corr_df["variant_b"]]["spearman_corr"].median()
        ),
    }

    # Attach appearance count to per_variant_top for convenience
    per_variant_top["n_variants_in_top_k"] = per_variant_top["ticker"].map(appearance_counts)

    return per_variant_top, summary, corr_df


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", required=True,
                   help=f"Comma-separated variants or 'all'. Valid: {','.join(ALL_VARIANTS)}")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stdout,
    )

    raw = args.variants.strip()
    if raw == "all":
        variant_names = list(ALL_VARIANTS)
    else:
        variant_names = [v.strip() for v in raw.split(",") if v.strip()]
    for v in variant_names:
        if v not in ALL_VARIANTS:
            raise ValueError(f"Unknown variant: {v!r}")
    if "baseline" not in variant_names:
        raise ValueError(
            "baseline is required to anchor the shared decile/IC computation. "
            "Add 'baseline' to --variants."
        )

    logger.info("=== Phase 5 analytics v2 — variants: %s ===", variant_names)

    # ---- Gather price universe (union of all variants' holding tickers) ----
    all_tickers: set[str] = set()
    for v_name in variant_names:
        h_path = V2_OUT_DIR / v_name / "contract_v1" / "holdings.parquet"
        if h_path.exists():
            h = pd.read_parquet(h_path)
            all_tickers.update(h["ticker"].astype(str).unique())
        s_path = V2_OUT_DIR / v_name / "contract_v1" / "scores.parquet"
        if s_path.exists():
            s = pd.read_parquet(s_path)
            all_tickers.update(s["ticker"].astype(str).unique())
    all_tickers.discard("SHY")
    logger.info("loading prices for %d unique tickers...", len(all_tickers))
    t0 = time.time()
    prices = _load_prices(all_tickers)
    logger.info("  prices: %s in %.1fs", prices.shape, time.time() - t0)

    # ---- Shared computations: decile + IC (same across variants since
    # scores.parquet is identical) ----
    logger.info("computing shared decile_returns (scores identical across variants)...")
    t0 = time.time()
    baseline_inputs = _load_variant_inputs("baseline", prices)
    decile_shared = _decile_analysis(baseline_inputs["scores"], prices)
    logger.info("  decile_returns done in %.1fs; %d rows",
                time.time() - t0, len(decile_shared))

    logger.info("computing shared ic_decomposition...")
    t0 = time.time()
    ic_shared = _ic_decomposition(baseline_inputs["scores"], prices)
    logger.info("  ic_decomposition done in %.1fs; %d rows",
                time.time() - t0, len(ic_shared))

    # ---- Per-variant analytics ----
    attributions: dict[str, pd.DataFrame] = {}
    for v_name in variant_names:
        logger.info("processing variant: %s", v_name)
        t0 = time.time()
        inputs = _load_variant_inputs(v_name, prices)
        out_dir = V2_OUT_DIR / v_name / "contract_v1"

        # Decile + IC: shared (write identical files for variant-tree completeness)
        decile_shared.to_parquet(out_dir / "decile_returns.parquet")
        ic_shared.to_parquet(out_dir / "ic_decomposition.parquet")

        # Per-variant: attribution
        attrib = _per_ticker_attribution(inputs["holdings"], prices, inputs["benchmarks"])
        attrib.to_parquet(out_dir / "per_ticker_attribution.parquet")
        attributions[v_name] = attrib

        # Per-variant: rolling win rate
        win = _rolling_win_rate(inputs["portfolio"], inputs["benchmarks"])
        win.to_parquet(out_dir / "rolling_win_rate.parquet")

        # Per-variant: concentration summary
        conc = _concentration_summary(inputs["holdings"])
        (out_dir / "concentration_summary.json").write_text(
            json.dumps(conc, indent=2, default=str), encoding="utf-8",
        )

        logger.info(
            "  [%s] %d ticker attribution rows, %d rolling windows, "
            "%d sector summaries in %.1fs",
            v_name, len(attrib), int(win["n_windows"].iloc[0]) if not win.empty else 0,
            len(conc), time.time() - t0,
        )

    # ---- Cross-variant concentration overlap ----
    logger.info("computing cross-variant concentration overlap (the C5 question)...")
    t0 = time.time()
    overlap_df, overlap_summary, corr_df = _concentration_overlap(attributions, top_k=20)
    comparison_dir = V2_OUT_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    overlap_df.to_parquet(comparison_dir / "concentration_overlap.parquet")
    corr_df.to_parquet(comparison_dir / "concentration_corr_matrix.parquet")
    (comparison_dir / "concentration_overlap_summary.json").write_text(
        json.dumps(overlap_summary, indent=2, default=str), encoding="utf-8",
    )
    logger.info(
        "  cross-variant overlap done in %.1fs; "
        "%d top-k tickers in union, cross-variant Spearman mean = %.4f, median = %.4f",
        time.time() - t0, overlap_summary["union_top_tickers_count"],
        overlap_summary["cross_variant_spearman_mean"],
        overlap_summary["cross_variant_spearman_median"],
    )

    # ---- Summary outputs ----
    print()
    print("=== Cross-variant concentration overlap ===")
    print(f"Top-k = {overlap_summary['top_k']} contributors per variant")
    print(f"Union of top-k across all {len(variant_names)} variants: "
          f"{overlap_summary['union_top_tickers_count']} unique tickers")
    print(f"Total contributing tickers across all variants: "
          f"{overlap_summary['n_total_contributing_tickers']}")
    print()
    print(f"Appearance distribution (# variants' top-k a ticker appears in):")
    dist = overlap_summary["appearance_count_distribution"]
    for n in sorted(dist.keys()):
        bar = "█" * dist[n]
        print(f"  in {n} variants: {dist[n]:3d} tickers {bar}")
    print()
    print(f"Cross-variant Spearman correlation (over all contributing tickers):")
    print(f"  mean (off-diagonal):   {overlap_summary['cross_variant_spearman_mean']:.4f}")
    print(f"  median (off-diagonal): {overlap_summary['cross_variant_spearman_median']:.4f}")
    print()
    print(f"Pairwise correlation matrix:")
    cm = corr_df.pivot(index="variant_a", columns="variant_b", values="spearman_corr")
    print(cm.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("=== Decile-return analysis (XGBoost, shared across variants) ===")
    print(decile_shared.to_string(index=False))
    print()
    print("=== IC decomposition (XGBoost, shared across variants) ===")
    print(ic_shared.to_string(index=False))
    print()
    print("=== Per-variant rolling win rate (12-month windows, full coverage) ===")
    for v_name in variant_names:
        win_path = V2_OUT_DIR / v_name / "contract_v1" / "rolling_win_rate.parquet"
        if win_path.exists():
            win = pd.read_parquet(win_path)
            print(f"-- {v_name} --")
            print(win.to_string(index=False))

    logger.info("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
