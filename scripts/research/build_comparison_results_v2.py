"""Build the cross-variant comparison artifact for Larger Universe v2.

Reads each variant's Phase 4 + walk-forward artifacts and writes
`models/studies/larger_universe_v2/comparison/comparison_results.parquet`
with the seven pre-committed success criteria evaluated per variant:

  C1 std-dev reduction  (vs baseline; relative)
  C2 positive-window count  (vs baseline; relative)
  C3 mean CAGR giveback  (vs baseline; relative)
  C4 drawdown ratio  (vs SPY; absolute)
  C5 max single-ticker alpha concentration  (absolute)
  C6 12-month rolling win rate vs SPY  (absolute)
  C7 test excess CAGR > 0  (absolute)

Schema per the Gate 1 approval clarification: each criterion contributes
both a value column (`criterion_N_<value>`) and a pass boolean
(`criterion_N_pass`), plus the raw test-window metrics that feed the
criteria. Verdict per variant follows the spec's verdict framework.

Run after both phase4_run_v2.py and phase5_walk_forward_v2.py have produced
artifacts for the requested variants. Each variant directory must contain
contract_v1/portfolio.parquet, holdings.parquet, benchmarks.parquet,
walk_forward.parquet for this script to compute all seven criteria.

CLI:
    python scripts/research/build_comparison_results_v2.py --variants all
    python scripts/research/build_comparison_results_v2.py --variants baseline,b1_vol_target,...

The per-ticker attribution and 12-month rolling win-rate computations mirror
v1's phase5_analytics.py to keep methodology identical across studies. They
are filtered to the test window (2023-05-12 → 2025-12-31) per the spec's
evaluation-window section — not the full portfolio (which would include OOS).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

V2_OUT_DIR = ROOT / "models" / "studies" / "larger_universe_v2"
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"
BENCH_PRICE_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "prices"

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

# Pass thresholds (per docs/studies/larger_universe_v2/spec.md)
C1_REDUCTION_THRESHOLD = 0.20         # ≥20% std-dev reduction
C3_GIVEBACK_THRESHOLD = 0.30          # ≤30% mean CAGR giveback
C4_DRAWDOWN_RATIO_THRESHOLD = 1.5     # |variant| ≤ 1.5 × |spy|
C5_SINGLE_TICKER_THRESHOLD = 0.25     # ≤25% of total alpha
C6_WIN_RATE_THRESHOLD = 0.60          # ≥60%

logger = logging.getLogger("compare_v2")


# ============================================================================
# Loaders + per-variant metric computations
# ============================================================================

def _load_variant_artifacts(variant_name: str) -> dict:
    """Load the variant's Phase 4 + walk-forward parquets. Returns dict; raises
    FileNotFoundError if any required artifact is missing."""
    base = V2_OUT_DIR / variant_name / "contract_v1"
    required = ["portfolio.parquet", "holdings.parquet", "benchmarks.parquet",
                "walk_forward.parquet"]
    missing = [r for r in required if not (base / r).exists()]
    if missing:
        raise FileNotFoundError(
            f"variant {variant_name!r} missing artifacts: {missing} "
            f"(expected under {base})"
        )
    portfolio = pd.read_parquet(base / "portfolio.parquet")
    holdings = pd.read_parquet(base / "holdings.parquet")
    benchmarks = pd.read_parquet(base / "benchmarks.parquet")
    walk_forward = pd.read_parquet(base / "walk_forward.parquet")
    for df in (portfolio, holdings, benchmarks, walk_forward):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
    return {
        "portfolio": portfolio,
        "holdings": holdings,
        "benchmarks": benchmarks,
        "walk_forward": walk_forward,
    }


def _test_window_summary(portfolio: pd.DataFrame, benchmarks: pd.DataFrame) -> dict:
    """Test-window CAGR / excess CAGR / max drawdown / SPY max drawdown.
    Mirrors v1's _summarize formula in phase4_run.py."""
    port = portfolio[(portfolio["model"] == "xgboost")
                      & (portfolio["date"] >= TEST_START)
                      & (portfolio["date"] <= TEST_END)].copy()
    if port.empty:
        return {}
    port["nav_period"] = port["nav"] / port["nav"].iloc[0]
    spy = benchmarks[(benchmarks["benchmark"] == "SPY")
                       & (benchmarks["date"] >= TEST_START)
                       & (benchmarks["date"] <= TEST_END)].copy()
    spy["nav_period"] = spy["nav"] / spy["nav"].iloc[0]
    n_days = len(port)
    total_return = float(port["nav_period"].iloc[-1] - 1.0)
    years = n_days / 252.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    spy_total = float(spy["nav_period"].iloc[-1] - 1.0) if not spy.empty else 0.0
    spy_cagr = (1 + spy_total) ** (1 / years) - 1 if years > 0 else 0.0
    excess_cagr = cagr - spy_cagr
    rolling_max = port["nav_period"].cummax()
    drawdown = port["nav_period"] / rolling_max - 1
    max_dd = float(drawdown.min())
    spy_dd = float((spy["nav_period"] / spy["nav_period"].cummax() - 1).min()) if not spy.empty else 0.0
    return {
        "n_days": n_days,
        "total_return": total_return,
        "test_cagr": float(cagr),
        "test_spy_cagr": float(spy_cagr),
        "test_excess_cagr_vs_spy": float(excess_cagr),
        "test_max_drawdown": max_dd,
        "test_spy_max_drawdown": spy_dd,
    }


def _walk_forward_stats(walk_forward: pd.DataFrame) -> dict:
    """6-window summary stats for XGBoost only."""
    wf = walk_forward[walk_forward["model"] == "xgboost"].copy()
    excess = wf["excess_cagr_vs_spy"].dropna()
    n_pos = int((excess > 0).sum())
    n_strong = int((excess > 0.05).sum())  # >5pp excess CAGR — informational
    return {
        "mean_excess_cagr_walkforward": float(excess.mean()) if len(excess) else float("nan"),
        "std_excess_cagr_walkforward": float(excess.std(ddof=1)) if len(excess) > 1 else float("nan"),
        "median_excess_cagr_walkforward": float(excess.median()) if len(excess) else float("nan"),
        "min_excess_cagr_walkforward": float(excess.min()) if len(excess) else float("nan"),
        "max_excess_cagr_walkforward": float(excess.max()) if len(excess) else float("nan"),
        "n_windows": int(len(wf)),
        "n_windows_positive": n_pos,
        "n_windows_strong": n_strong,
    }


def _max_single_ticker_alpha_fraction(
    holdings: pd.DataFrame, benchmarks: pd.DataFrame, prices: pd.DataFrame,
) -> float:
    """Compute max single-ticker contribution to total alpha on the test window.

    Mirrors phase5_analytics.per_ticker_attribution but filtered to the test
    window. Returns the maximum ticker's share of total alpha as a fraction
    (0.0–1.0). Returns 0.0 when total alpha is 0.
    """
    h = holdings[(holdings["model"] == "xgboost")
                  & (holdings["date"] >= TEST_START)
                  & (holdings["date"] <= TEST_END)].copy().sort_values(["date", "ticker"])
    if h.empty:
        return 0.0
    spy_nav = benchmarks[(benchmarks["benchmark"] == "SPY")
                          & (benchmarks["date"] >= TEST_START)
                          & (benchmarks["date"] <= TEST_END)].set_index("date")["nav"]
    rebalance_dates = sorted(h["date"].unique())
    last_date = min(prices.index[-1], TEST_END)

    # SPY return per holding period
    spy_returns_by_period: dict[pd.Timestamp, float] = {}
    for i, d_start in enumerate(rebalance_dates):
        d_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else last_date
        s_start = spy_nav.asof(d_start)
        s_end = spy_nav.asof(d_end)
        if pd.notna(s_start) and pd.notna(s_end) and s_start > 0:
            spy_returns_by_period[d_start] = float(s_end / s_start - 1)
        else:
            spy_returns_by_period[d_start] = 0.0

    contrib: dict[str, float] = {}
    for i, d_start in enumerate(rebalance_dates):
        d_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else last_date
        spy_ret = spy_returns_by_period[d_start]
        day_h = h[h["date"] == d_start]
        for _, row in day_h.iterrows():
            t = row["ticker"]
            w = float(row["weight"])
            if t not in prices.columns:
                # SHY (B5) and other non-snapshot tickers — skip from alpha
                # attribution since the snapshot price file doesn't cover them.
                continue
            p_start = prices[t].asof(d_start)
            p_end = prices[t].asof(d_end)
            if pd.isna(p_start) or pd.isna(p_end) or p_start <= 0:
                continue
            ticker_ret = float(p_end / p_start - 1)
            contrib[t] = contrib.get(t, 0.0) + w * (ticker_ret - spy_ret)

    total_alpha = sum(contrib.values())
    if total_alpha == 0 or not contrib:
        return 0.0
    max_share = max(c / total_alpha for c in contrib.values())
    return float(max_share)


def _rolling_12mo_win_rate(
    portfolio: pd.DataFrame, benchmarks: pd.DataFrame, window_days: int = 252,
) -> float:
    """Fraction of 252-day rolling windows (test window only) where the
    portfolio's 12-month return exceeds SPY's. Mirrors v1's rolling_win_rate
    in phase5_analytics.py, filtered to the test window."""
    port = portfolio[(portfolio["model"] == "xgboost")
                      & (portfolio["date"] >= TEST_START)
                      & (portfolio["date"] <= TEST_END)].set_index("date")["nav"].sort_index()
    spy_nav = benchmarks[(benchmarks["benchmark"] == "SPY")
                          & (benchmarks["date"] >= TEST_START)
                          & (benchmarks["date"] <= TEST_END)].set_index("date")["nav"].sort_index()
    common = port.index.intersection(spy_nav.index)
    if len(common) <= window_days:
        return float("nan")
    p = port.loc[common]
    s = spy_nav.loc[common]
    n_windows = len(common) - window_days
    n_wins = 0
    for i in range(n_windows):
        port_ret = p.iloc[i + window_days] / p.iloc[i] - 1
        spy_ret = s.iloc[i + window_days] / s.iloc[i] - 1
        if port_ret > spy_ret:
            n_wins += 1
    return float(n_wins) / n_windows


def _load_prices_for_tickers(tickers: set[str]) -> pd.DataFrame:
    """Load daily close prices for a set of tickers from the v1 snapshot.
    Used by criterion 5's per-ticker attribution."""
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


# ============================================================================
# Per-variant evaluation
# ============================================================================

def _evaluate_variant(variant_name: str, prices: pd.DataFrame) -> dict:
    """Load + summarize one variant's artifacts."""
    arts = _load_variant_artifacts(variant_name)
    summary = {"variant": variant_name}
    summary.update(_test_window_summary(arts["portfolio"], arts["benchmarks"]))
    summary.update(_walk_forward_stats(arts["walk_forward"]))
    summary["test_max_single_ticker_alpha_pct"] = (
        _max_single_ticker_alpha_fraction(arts["holdings"], arts["benchmarks"], prices)
    )
    summary["test_rolling_12mo_win_rate"] = (
        _rolling_12mo_win_rate(arts["portfolio"], arts["benchmarks"])
    )
    return summary


def _compute_criteria(row: dict, baseline_row: dict) -> dict:
    """Apply the seven criteria. Returns a dict of criterion_N_<value> and
    criterion_N_pass entries."""
    c: dict = {}

    # Criterion 1: std-dev reduction across walk-forward windows
    bl_std = baseline_row.get("std_excess_cagr_walkforward")
    var_std = row.get("std_excess_cagr_walkforward")
    if bl_std and bl_std > 0 and var_std is not None and not pd.isna(var_std):
        reduction = 1.0 - (var_std / bl_std)
        c["criterion_1_std_reduction_pct"] = float(reduction)
        c["criterion_1_pass"] = bool(reduction >= C1_REDUCTION_THRESHOLD)
    else:
        c["criterion_1_std_reduction_pct"] = float("nan")
        c["criterion_1_pass"] = False

    # Criterion 2: positive-window count
    bl_pos = baseline_row.get("n_windows_positive")
    var_pos = row.get("n_windows_positive")
    c["criterion_2_positive_window_count"] = int(var_pos) if var_pos is not None else 0
    c["criterion_2_pass"] = bool(
        var_pos is not None and bl_pos is not None and var_pos >= bl_pos
    )

    # Criterion 3: mean CAGR giveback (test window)
    bl_mean = baseline_row.get("test_excess_cagr_vs_spy")
    var_mean = row.get("test_excess_cagr_vs_spy")
    if (bl_mean is not None and var_mean is not None
            and not pd.isna(bl_mean) and not pd.isna(var_mean)
            and bl_mean != 0):
        giveback = 1.0 - (var_mean / bl_mean)
        c["criterion_3_mean_cagr_giveback_pct"] = float(giveback)
        c["criterion_3_pass"] = bool(giveback <= C3_GIVEBACK_THRESHOLD)
    else:
        c["criterion_3_mean_cagr_giveback_pct"] = float("nan")
        c["criterion_3_pass"] = False

    # Criterion 4: drawdown ratio
    var_dd = row.get("test_max_drawdown")
    spy_dd = row.get("test_spy_max_drawdown")
    if var_dd is not None and spy_dd is not None and not pd.isna(var_dd) and not pd.isna(spy_dd) and spy_dd != 0:
        ratio = abs(var_dd) / abs(spy_dd)
        c["criterion_4_drawdown_ratio"] = float(ratio)
        c["criterion_4_pass"] = bool(ratio <= C4_DRAWDOWN_RATIO_THRESHOLD)
    else:
        c["criterion_4_drawdown_ratio"] = float("nan")
        c["criterion_4_pass"] = False

    # Criterion 5: max single-ticker alpha concentration
    max_alpha = row.get("test_max_single_ticker_alpha_pct")
    c["criterion_5_max_single_ticker_alpha_pct"] = (
        float(max_alpha) if max_alpha is not None and not pd.isna(max_alpha) else float("nan")
    )
    c["criterion_5_pass"] = bool(
        max_alpha is not None and not pd.isna(max_alpha)
        and max_alpha <= C5_SINGLE_TICKER_THRESHOLD
    )

    # Criterion 6: 12-month rolling win rate
    wr = row.get("test_rolling_12mo_win_rate")
    c["criterion_6_rolling_12mo_win_rate"] = (
        float(wr) if wr is not None and not pd.isna(wr) else float("nan")
    )
    c["criterion_6_pass"] = bool(
        wr is not None and not pd.isna(wr) and wr >= C6_WIN_RATE_THRESHOLD
    )

    # Criterion 7: test excess CAGR > 0
    var_excess = row.get("test_excess_cagr_vs_spy")
    c["criterion_7_test_excess_cagr"] = (
        float(var_excess) if var_excess is not None and not pd.isna(var_excess) else float("nan")
    )
    c["criterion_7_pass"] = bool(
        var_excess is not None and not pd.isna(var_excess) and var_excess > 0
    )

    pass_flags = [c[f"criterion_{i}_pass"] for i in range(1, 8)]
    c["all_pass"] = all(pass_flags)
    c["n_pass"] = int(sum(pass_flags))
    if c["all_pass"]:
        c["verdict"] = "PROMOTE"
    elif c["n_pass"] > 0:
        c["verdict"] = "METHODOLOGY FINDING"
    else:
        c["verdict"] = "NOT PROMOTED"
    return c


# ============================================================================
# Variant_meta.json update (append optional_artifacts)
# ============================================================================

def _update_variant_meta(variant_names: list[str]) -> None:
    """Append `comparison_results.parquet` to variant_meta.json's
    optional_artifacts list (idempotent)."""
    path = V2_OUT_DIR / "variant_meta.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    opt = payload.get("optional_artifacts", [])
    entry = {
        "name": "comparison_results",
        "path": "comparison/comparison_results.parquet",
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "covers_variants": variant_names,
    }
    # De-dupe by name
    opt = [e for e in opt if e.get("name") != "comparison_results"]
    opt.append(entry)
    payload["optional_artifacts"] = opt
    payload["last_updated"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
            raise ValueError(f"Unknown variant: {v!r}. Valid: {ALL_VARIANTS}")
    if "baseline" not in variant_names:
        raise ValueError(
            "baseline is required (it's the control for criteria 1-3). "
            "Add 'baseline' to --variants."
        )
    logger.info("=== Building comparison_results.parquet for variants: %s ===", variant_names)

    # Collect all tickers from each variant's holdings (for criterion 5 prices)
    tickers: set[str] = set()
    for v_name in variant_names:
        h_path = V2_OUT_DIR / v_name / "contract_v1" / "holdings.parquet"
        if h_path.exists():
            h = pd.read_parquet(h_path)
            tickers.update(h["ticker"].astype(str).unique())
    tickers.discard("SHY")  # B5; not in snapshot, intentionally excluded
    logger.info("loading prices for %d unique tickers across variants...", len(tickers))
    prices = _load_prices_for_tickers(tickers)
    logger.info("  prices shape: %s", prices.shape)

    # Compute per-variant raw metrics
    summaries: dict[str, dict] = {}
    for v_name in variant_names:
        logger.info("evaluating variant: %s", v_name)
        summaries[v_name] = _evaluate_variant(v_name, prices)

    baseline_row = summaries["baseline"]

    # Apply criteria + verdict
    rows = []
    for v_name in variant_names:
        row = dict(summaries[v_name])
        row.update(_compute_criteria(summaries[v_name], baseline_row))
        row["evaluated_at"] = datetime.now(timezone.utc).isoformat()
        rows.append(row)

    df = pd.DataFrame(rows)

    # Write artifact
    out_dir = V2_OUT_DIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_results.parquet"
    df.to_parquet(out_path)
    logger.info("wrote %d rows to %s", len(df), out_path)

    _update_variant_meta(variant_names)

    # Summary table
    print()
    print("=== Comparison results ===")
    pass_cols = [f"criterion_{i}_pass" for i in range(1, 8)]
    cols = [
        "variant", "test_cagr", "test_excess_cagr_vs_spy", "test_max_drawdown",
        "test_rolling_12mo_win_rate", "test_max_single_ticker_alpha_pct",
        "mean_excess_cagr_walkforward", "std_excess_cagr_walkforward",
        "n_windows_positive", "n_pass", "verdict",
    ]
    print(df[cols].to_string(index=False))
    print()
    print("--- pass-detail per criterion ---")
    detail_cols = ["variant"] + pass_cols + ["all_pass", "verdict"]
    print(df[detail_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
