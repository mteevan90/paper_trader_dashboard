"""run_325_sp1500_sanity.py — replay #325's tunables against pre_v3_sp1500_<date>.

Purpose: confirm the universe expansion + retraining produced something
sane before kicking off the Phase 2 optimization. NOT an apples-to-apples
re-run of #325 — the universe is bigger and the model is different — so
results will differ. The check is "same order of magnitude as the original".

Original #325 numbers (pre_v2_20260505):
  - Validation total return:  +315.7%   (range to expect: +200% .. +400%)
  - Validation alpha (CAPM):  +39.5pp   (range to expect: ~similar)
  - Validation alpha (arith): +63.7pp   (range to expect: ~similar)
  - Trade count:              91        (range to expect: similar)

If this run lands wildly outside that range (e.g. +50% return, -10%
return, 5 trades, 500 trades) STOP — likely culprits: data fetch errors,
liquidity filter calibration, or a poorly-trained sp1500 model.

Usage (PowerShell):
    venv\\Scripts\\python.exe src\\run_325_sp1500_sanity.py \\
        --snapshot pre_v3_sp1500_<YYYYMMDD>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Pre-parse --snapshot and set PAPER_TRADER_DATA_ROOT BEFORE importing
# any backtest modules — they read DATA_ROOT at import time.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--snapshot", required=False, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.snapshot:
    _snap_root = os.path.abspath(os.path.join(
        str(_SRC_DIR), "..", "models", "snapshots", _pre_args.snapshot))
    if not os.path.isdir(_snap_root):
        sys.exit(
            f"[SANITY] Snapshot not found: {_snap_root}\n"
            f"  Run snapshot_sp1500.py first.")
    os.environ["PAPER_TRADER_DATA_ROOT"] = _snap_root
    print(f"[SANITY] Using snapshot: {_pre_args.snapshot} ({_snap_root})\n")

import pandas as pd  # noqa: E402

from backtest_config import BacktestConfig                    # noqa: E402
from backtest import (run_backtest, fetch_fundamentals,        # noqa: E402
                      fetch_earnings_dates, compute_stats)
from feature_cache import build_feature_matrix                 # noqa: E402
from fetch_data import (SP1500_TICKERS, build_sector_map,      # noqa: E402
                        get_stock_data_cached)
from model import load_model                                    # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(str(_SRC_DIR), ".."))
DEFAULT_325_META = os.path.join(
    REPO_ROOT, "models", "cache", "dashboard_results",
    "best_regime_dependent_v1_20260505_2240_325", "meta.json")
RESULTS_DIR = os.path.join(
    REPO_ROOT, "models", "cache", "dashboard_results",
    "325_sp1500_sanity")
PRICE_CACHE = os.path.join(REPO_ROOT, "models", "price_cache")

# Original #325 numbers — read straight from this script's docstring above
# in case the meta.json gets re-shaped. Used for the comparison table.
ORIGINAL_325 = {
    "validation": {
        "total_return_pct":   315.7,
        "alpha_capm_pp":      39.5,
        "alpha_arith_pp":     63.7,
        "n_trades":           91,
    },
}

# Allowed range for the sanity check. Outside this, the script flags
# CONCERN. Per the user spec.
ACCEPT_VAL_RETURN_PCT = (200.0, 400.0)


def _config_from_meta(path: str) -> BacktestConfig:
    """Build a BacktestConfig from #325's saved meta.json. Adds the
    sp1500 liquidity filter — that's the one new field this run exercises."""
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg_raw = meta["config"]
    # Drop derived fields the dataclass doesn't accept directly. weight_alt
    # / weight_alt_offensive are __post_init__-derived from the free
    # weights; passing them through validates instead of silently shadowing.
    keep = {k: v for k, v in cfg_raw.items()
            if k not in ("weight_alt", "weight_alt_offensive")}
    keep["min_avg_daily_volume_usd"] = 25_000_000.0
    return BacktestConfig(**keep)


def _load_inputs(window_start: str, window_end: str,
                 universe: list[str]) -> dict:
    """Load every input run_backtest needs from the snapshot."""
    print(f"  feature matrix...")
    fm = build_feature_matrix(universe, window_start, window_end,
                              price_cache_dir=PRICE_CACHE)
    print(f"  price data...")
    price_data = get_stock_data_cached(universe, window_start, window_end,
                                       cache_dir=PRICE_CACHE)
    print(f"  sector map...")
    sector_map = build_sector_map(list(fm.keys()))
    print(f"  fundamentals + earnings...")
    fund_data = fetch_fundamentals(list(fm.keys()))
    earn_dates = fetch_earnings_dates(list(fm.keys()),
                                      window_start, window_end)
    print(f"  model (snapshot's xgb_model.json)...")
    model = load_model()
    return {
        "featured_data": fm,
        "price_data":    price_data,
        "sector_map":    sector_map,
        "fund_data":     fund_data,
        "earnings":      earn_dates,
        "model":         model,
    }


def _stats_summary(portfolio_df: pd.DataFrame, trades_df: pd.DataFrame,
                   spy_close: pd.Series) -> dict:
    """Compute the summary numbers we compare against #325."""
    if portfolio_df.empty:
        return {"empty": True}
    val_pct_total = portfolio_df["portfolio_value"]
    start_val = float(val_pct_total.iloc[0])
    end_val   = float(val_pct_total.iloc[-1])
    total_return_pct = 100.0 * (end_val / start_val - 1.0)

    spy_window = spy_close.reindex(portfolio_df.index).ffill()
    spy_start = float(spy_window.iloc[0])
    spy_end   = float(spy_window.iloc[-1])
    spy_return_pct = 100.0 * (spy_end / spy_start - 1.0)
    alpha_arith_pp = total_return_pct - spy_return_pct

    n_trades = int(len(trades_df)) if trades_df is not None else 0
    return {
        "total_return_pct": round(total_return_pct, 2),
        "spy_return_pct":   round(spy_return_pct, 2),
        "alpha_arith_pp":   round(alpha_arith_pp, 2),
        "n_trades":         n_trades,
        "n_days":           int(len(portfolio_df)),
        "start_date":       str(portfolio_df.index[0].date()),
        "end_date":         str(portfolio_df.index[-1].date()),
    }


def _run_window(name: str, cfg: BacktestConfig, inputs: dict,
                start: str, end: str) -> dict:
    print(f"\n=== Window: {name}  ({start} -> {end}) ===")
    t0 = time.time()
    portfolio_df, trades_df, scores, _ = run_backtest(
        inputs["featured_data"], inputs["price_data"],
        split_date=start,
        fund_data=inputs["fund_data"],
        sector_map=inputs["sector_map"],
        earnings_dates=inputs["earnings"],
        model=inputs["model"],
        config=cfg,
    )
    runtime = time.time() - t0
    print(f"  Runtime: {runtime:.1f}s")
    spy_close = inputs["price_data"]["SPY"]["Close"] \
        if "SPY" in inputs["price_data"] else pd.Series(dtype=float)
    if spy_close.empty:
        # SPY isn't in SP1500_TICKERS by default; fetch separately.
        spy_close = get_stock_data_cached(["SPY"], start, end,
                                          cache_dir=PRICE_CACHE)["SPY"]["Close"]
    summary = _stats_summary(portfolio_df, trades_df, spy_close)
    summary["runtime_seconds"] = round(runtime, 2)
    summary["window"] = name
    print(f"  Total return:   {summary.get('total_return_pct'):.2f}%")
    print(f"  SPY return:     {summary.get('spy_return_pct'):.2f}%")
    print(f"  Alpha (arith):  {summary.get('alpha_arith_pp'):+.2f}pp")
    print(f"  Trades:         {summary.get('n_trades')}")
    return {"portfolio_df": portfolio_df, "trades_df": trades_df,
            "summary": summary}


def _save_results(label: str, train_out: dict, val_out: dict,
                  cfg: BacktestConfig, snapshot: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not train_out["portfolio_df"].empty:
        train_out["portfolio_df"].to_parquet(
            os.path.join(RESULTS_DIR, "portfolio_train.parquet"))
        if train_out["trades_df"] is not None and not train_out["trades_df"].empty:
            train_out["trades_df"].to_parquet(
                os.path.join(RESULTS_DIR, "trades_train.parquet"))
    if not val_out["portfolio_df"].empty:
        val_out["portfolio_df"].to_parquet(
            os.path.join(RESULTS_DIR, "portfolio_val.parquet"))
        if val_out["trades_df"] is not None and not val_out["trades_df"].empty:
            val_out["trades_df"].to_parquet(
                os.path.join(RESULTS_DIR, "trades_val.parquet"))

    meta = {
        "label":         "325_sp1500_sanity",
        "saved_at":      datetime.now(timezone.utc).isoformat(),
        "snapshot":      snapshot,
        "source_meta":   "best_regime_dependent_v1_20260505_2240_325/meta.json",
        "purpose":       ("Sanity-check #325's tunables against pre_v3_sp1500. "
                          "Universe is bigger + model is retrained, so this "
                          "is NOT an apples-to-apples reproduction."),
        "config":        cfg.to_dict(),
        "training":      train_out["summary"],
        "validation":    val_out["summary"],
        "original_325":  ORIGINAL_325,
        "promoted":      False,
    }
    meta_path = os.path.join(RESULTS_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
    return meta_path


def _print_comparison(val_summary: dict) -> None:
    orig = ORIGINAL_325["validation"]
    print()
    print("=" * 64)
    print("Sanity comparison (#325 original  vs  sp1500 sanity rerun)")
    print("=" * 64)
    rows = [
        ("Validation total return %",
         orig["total_return_pct"], val_summary.get("total_return_pct")),
        ("Validation alpha (arith) pp",
         orig["alpha_arith_pp"], val_summary.get("alpha_arith_pp")),
        ("Trade count (val)",
         orig["n_trades"], val_summary.get("n_trades")),
    ]
    print(f"  {'metric':<30}  {'original':>10}  {'sp1500':>10}  {'delta':>10}")
    for name, a, b in rows:
        if b is None:
            print(f"  {name:<30}  {a!s:>10}  {'(empty)':>10}  {'(n/a)':>10}")
            continue
        d = b - a
        print(f"  {name:<30}  {a:>10}  {b:>10}  {d:+10}")

    new_ret = val_summary.get("total_return_pct")
    if new_ret is None:
        print("\n[VERDICT] No validation data — backtest produced empty output.")
        return
    lo, hi = ACCEPT_VAL_RETURN_PCT
    if lo <= new_ret <= hi:
        print(f"\n[VERDICT] PASS — validation return ({new_ret:.1f}%) "
              f"is inside the expected {lo}..{hi}% band. "
              f"Phase 2 can launch.")
    else:
        print(f"\n[VERDICT] CONCERN — validation return ({new_ret:.1f}%) "
              f"is OUTSIDE the expected {lo}..{hi}% band. "
              f"Investigate before launching Phase 2.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", required=True,
                        help="Snapshot directory name under models/snapshots/ "
                             "(e.g. pre_v3_sp1500_20260508).")
    parser.add_argument("--config", default=DEFAULT_325_META,
                        help=f"Path to #325's meta.json "
                             f"(default: {DEFAULT_325_META}).")
    args = parser.parse_args()

    print(f"[SANITY] Loading #325 config from {args.config}")
    cfg = _config_from_meta(args.config)
    print(f"[SANITY] BacktestConfig built. min_avg_daily_volume_usd="
          f"${cfg.min_avg_daily_volume_usd:,.0f}\n")

    print(f"[SANITY] Loading shared data from snapshot...")
    inputs = _load_inputs(cfg.train_start, cfg.validate_end,
                          list(SP1500_TICKERS))
    print(f"[SANITY] {len(inputs['featured_data'])} tickers featured, "
          f"{len(inputs['price_data'])} priced.\n")

    train_out = _run_window("training", cfg, inputs,
                            cfg.train_start, cfg.train_end)
    val_out   = _run_window("validation", cfg, inputs,
                            cfg.validate_start, cfg.validate_end)

    meta_path = _save_results("325_sp1500_sanity", train_out, val_out,
                              cfg, args.snapshot)
    print(f"\n[SANITY] Saved to {RESULTS_DIR}")
    print(f"[SANITY] Meta: {meta_path}")

    _print_comparison(val_out["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
