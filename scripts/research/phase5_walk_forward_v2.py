"""Phase 5 walk-forward — Larger Universe v2 variant retrains.

For each of the 6 rolling 3-year training windows, retrain XGBoost with v1's
LOCKED hyperparameters (no Phase 3 retuning), score the val window once,
then run N variant backtests against the cached scores. Each variant gets
its own walk_forward.parquet under
`models/studies/larger_universe_v2/<variant_subdir>/contract_v1/`.

Windows mirror v1 (3-year train, 1-year val):
  W1: 2017-05-12..2020-05-11 → 2020-05-12..2021-05-11
  W2: 2018-05-12..2021-05-11 → 2021-05-12..2022-05-11
  W3: 2019-05-12..2022-05-11 → 2022-05-12..2023-05-11
  W4: 2020-05-12..2023-05-11 → 2023-05-12..2024-05-11
  W5: 2021-05-12..2024-05-11 → 2024-05-12..2025-05-11
  W6: 2022-05-12..2025-05-11 → 2025-05-12..2026-05-11

CLI:
    python scripts/research/phase5_walk_forward_v2.py --variants baseline
    python scripts/research/phase5_walk_forward_v2.py --variants all

Per-window per-variant outputs (per row in each variant's walk_forward.parquet):
mean_ic, std_ic, positive_rate, n_dates_scored, n_days, total_return, cagr,
sharpe, max_drawdown, spy_total_return, spy_cagr, excess_cagr_vs_spy.

The metric computations (IC, Sharpe, CAGR, MaxDD) mirror v1's phase5_walk_forward
verbatim so v2-baseline's walk-forward output is directly comparable to v1's.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.equities.study.labels import build_labels, LABEL_HORIZON_TRADING_DAYS
from src.equities.study.training import (
    _prep_xgb_X, _split_features_target,
)
from src.equities.study.backtest import (
    month_end_trading_dates, run_backtest, TRANSACTION_COST_PCT,
)
from src.equities.portfolio_construction import get_variant_by_name

# ---- Paths ----
FEATURES_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "features.parquet"
UNIVERSE_PATH = ROOT / "docs" / "larger_universe_v1_universe.json"
SECTOR_MAP_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "sector_map.json"
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"
XGB_PARAMS_PATH = ROOT / "models" / "studies" / "larger_universe_v1" / "xgboost_best_params.json"
BENCH_PRICE_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "prices"
V2_OUT_DIR = ROOT / "models" / "studies" / "larger_universe_v2"

WINDOWS = [
    (pd.Timestamp("2017-05-12"), pd.Timestamp("2020-05-11"), pd.Timestamp("2020-05-12"), pd.Timestamp("2021-05-11")),
    (pd.Timestamp("2018-05-12"), pd.Timestamp("2021-05-11"), pd.Timestamp("2021-05-12"), pd.Timestamp("2022-05-11")),
    (pd.Timestamp("2019-05-12"), pd.Timestamp("2022-05-11"), pd.Timestamp("2022-05-12"), pd.Timestamp("2023-05-11")),
    (pd.Timestamp("2020-05-12"), pd.Timestamp("2023-05-11"), pd.Timestamp("2023-05-12"), pd.Timestamp("2024-05-11")),
    (pd.Timestamp("2021-05-12"), pd.Timestamp("2024-05-11"), pd.Timestamp("2024-05-12"), pd.Timestamp("2025-05-11")),
    (pd.Timestamp("2022-05-12"), pd.Timestamp("2025-05-11"), pd.Timestamp("2025-05-12"), pd.Timestamp("2026-05-11")),
]

ALL_VARIANTS = [
    "baseline",
    "b1_vol_target",
    "b2_conviction_weighted",
    "b3_dynamic_topn",
    "b4_concentration_penalties",
    "b5_defensive_sleeves",
    "b6_smaller_caps",
]
VARIANTS_NEEDING_SHY = {"b5_defensive_sleeves"}
VARIANTS_NEEDING_SPY_HISTORY = {"b5_defensive_sleeves"}

logger = logging.getLogger("phase5_wf_v2")


# ============================================================================
# Helpers (mirror phase5_walk_forward.py)
# ============================================================================

def _setup() -> dict:
    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    sector_map = json.loads(SECTOR_MAP_PATH.read_text(encoding="utf-8"))
    by_sym = {}
    for r in universe:
        s = r["symbol"]
        if s not in by_sym or (by_sym[s]["status"] == "removed" and r["status"] == "active"):
            by_sym[s] = r
    sectors = pd.Series({
        sym: (sector_map.get(sym, {}) or {}).get("sector") or "sector_unknown"
        for sym in by_sym
    })
    tiers = pd.Series({sym: by_sym[sym]["tier"] for sym in by_sym})
    delisting_dates = {}
    for sym, r in by_sym.items():
        if r["status"] == "removed" and r.get("removed_at"):
            try:
                delisting_dates[sym] = pd.Timestamp(r["removed_at"])
            except Exception:
                pass

    closes = {}
    for sym in by_sym:
        p = SNAPSHOT_PRICE_DIR / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index)
        closes[sym] = df["close"]
    prices = pd.DataFrame(closes).sort_index()
    daily_returns = prices.pct_change()

    features = pd.read_parquet(FEATURES_PATH)
    full_universe = list(by_sym.keys())
    labels = build_labels(full_universe, horizon=LABEL_HORIZON_TRADING_DAYS)

    spy = pd.read_parquet(BENCH_PRICE_DIR / "SPY.parquet")
    spy.index = pd.to_datetime(spy.index)

    return {
        "universe_records": universe,
        "by_sym": by_sym,
        "sectors": sectors,
        "tiers": tiers,
        "delisting_dates": delisting_dates,
        "daily_returns": daily_returns,
        "trading_dates": prices.index,
        "features": features,
        "labels": labels,
        "spy_close": spy["close"],
        "spy_history": spy,
    }


def _train_xgb(merged: pd.DataFrame, params: dict) -> xgb.XGBRegressor:
    p = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42,
        "verbosity": 0,
        **params,
    }
    X, y = _split_features_target(merged)
    X = _prep_xgb_X(X)
    model = xgb.XGBRegressor(**p)
    model.fit(X, y)
    return model


def _score(model, X_df) -> np.ndarray:
    X = _prep_xgb_X(X_df)
    return model.predict(X)


def _cross_sectional_ic_for_period(
    scores_df: pd.DataFrame, labels: pd.DataFrame,
    val_start: pd.Timestamp, val_end: pd.Timestamp,
    min_tickers: int = 30,
) -> tuple[float, float, float, int]:
    merged = scores_df.merge(labels, on=["date", "ticker"], how="left")
    merged = merged[(merged["date"] >= val_start) & (merged["date"] <= val_end)]
    merged = merged.dropna(subset=["score", "target"])
    per_date: list[float] = []
    for d, g in merged.groupby("date"):
        if len(g) < min_tickers:
            continue
        if g["score"].nunique() < 2 or g["target"].nunique() < 2:
            continue
        rho, _ = spearmanr(g["score"], g["target"])
        if rho == rho:
            per_date.append(rho)
    if not per_date:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.array(per_date)
    return float(arr.mean()), float(arr.std()), float((arr > 0).mean()), int(len(arr))


def _backtest_stats_for_variant(
    variant_name: str, score_fn, val_start: pd.Timestamp, val_end: pd.Timestamp,
    ctx: dict, shy_returns: pd.Series | None, shy_close: pd.Series | None,
) -> dict:
    """Run a 1-year mini-backtest with the named variant on the val window."""
    rebal_dates = month_end_trading_dates(ctx["trading_dates"], val_start, val_end)
    if not rebal_dates:
        return {}

    if variant_name in VARIANTS_NEEDING_SHY:
        if shy_returns is None or shy_close is None:
            raise RuntimeError(
                f"variant {variant_name} needs SHY data but it was not loaded"
            )
        dr = ctx["daily_returns"].copy()
        if "SHY" not in dr.columns:
            dr["SHY"] = shy_returns.reindex(dr.index)
        secs = ctx["sectors"].copy()
        if "SHY" not in secs.index:
            secs["SHY"] = "treasury_etf"
        tiers_v = ctx["tiers"].copy()
        if "SHY" not in tiers_v.index:
            tiers_v["SHY"] = "etf"
        shy_prices_for_engine = shy_close
    else:
        dr = ctx["daily_returns"]
        secs = ctx["sectors"]
        tiers_v = ctx["tiers"]
        shy_prices_for_engine = None

    spy_history_for_engine = (
        ctx["spy_history"] if variant_name in VARIANTS_NEEDING_SPY_HISTORY else None
    )

    variant = get_variant_by_name(variant_name)

    result = run_backtest(
        model_name="xgboost",
        score_fn=score_fn,
        rebalance_dates=rebal_dates,
        daily_returns=dr,
        delisting_dates=ctx["delisting_dates"],
        sectors=secs,
        tiers=tiers_v,
        pc_params=None,
        universe_records=ctx["universe_records"],
        construction_variant=variant,
        spy_history=spy_history_for_engine,
        shy_prices=shy_prices_for_engine,
    )
    port = result.portfolio.copy()
    if port.empty:
        return {}
    nav = port["nav"].values
    n_days = len(nav)
    total_return = float(nav[-1] / nav[0] - 1.0)
    years = n_days / 252.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    daily_ret = pd.Series(nav).pct_change().dropna()
    sharpe = (
        daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        if daily_ret.std() > 0 else 0.0
    )
    rolling_max = pd.Series(nav).cummax()
    max_dd = float((pd.Series(nav) / rolling_max - 1).min())
    spy = ctx["spy_close"]
    spy_window = spy[(spy.index >= port["date"].iloc[0])
                      & (spy.index <= port["date"].iloc[-1])]
    if not spy_window.empty:
        spy_total = float(spy_window.iloc[-1] / spy_window.iloc[0] - 1)
        spy_cagr = (1 + spy_total) ** (1 / years) - 1 if years > 0 else 0.0
    else:
        spy_total = float("nan")
        spy_cagr = float("nan")
    return {
        "n_days": n_days,
        "total_return": total_return,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "spy_total_return": spy_total,
        "spy_cagr": float(spy_cagr),
        "excess_cagr_vs_spy": (
            float(cagr - spy_cagr) if not np.isnan(spy_cagr) else float("nan")
        ),
    }


def _make_score_fn_from_lookup(score_lookup: pd.Series):
    """Build a score_fn returning Series(ticker -> score) for the given date,
    backed by a MultiIndex score lookup."""
    def fn(d):
        if d not in score_lookup.index.get_level_values(0):
            return pd.Series(dtype=float)
        rows_for_date = score_lookup.loc[d]
        if isinstance(rows_for_date, pd.Series):
            return rows_for_date
        return pd.Series(dtype=float)
    return fn


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

    logger.info("=== Phase 5 walk-forward v2 — variants: %s ===", variant_names)
    logger.info("loading inputs...")
    t0 = time.time()
    ctx = _setup()
    logger.info("  setup done in %.1fs", time.time() - t0)

    xgb_params = json.loads(XGB_PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]

    # Conditional SHY load
    shy_returns: pd.Series | None = None
    shy_close: pd.Series | None = None
    if any(v in VARIANTS_NEEDING_SHY for v in variant_names):
        logger.info("loading SHY for B5 walk-forward windows...")
        shy_df = pd.read_parquet(BENCH_PRICE_DIR / "SHY.parquet")
        shy_df.index = pd.to_datetime(shy_df.index)
        shy_df = shy_df.sort_index()
        shy_close = shy_df["close"]
        shy_returns = shy_close.pct_change()

    # Per-variant row buckets
    rows_by_variant: dict[str, list[dict]] = {v: [] for v in variant_names}

    for i, (tr_start, tr_end, va_start, va_end) in enumerate(WINDOWS, 1):
        logger.info(
            "--- Window %d/%d: train %s..%s, val %s..%s ---",
            i, len(WINDOWS), tr_start.date(), tr_end.date(),
            va_start.date(), va_end.date(),
        )
        feat = ctx["features"]
        labels = ctx["labels"]
        train_feat = feat[(feat["date"] >= tr_start) & (feat["date"] <= tr_end)]
        train_lbl = labels[(labels["date"] >= tr_start) & (labels["date"] <= tr_end)]
        train_merged = train_feat.merge(train_lbl, on=["date", "ticker"], how="left")
        train_merged = train_merged[train_merged["target"].notnull()].reset_index(drop=True)
        logger.info("  train rows: %d", len(train_merged))

        val_feat = feat[(feat["date"] >= va_start) & (feat["date"] <= va_end)]
        logger.info("  val rows (features): %d", len(val_feat))

        # Train once per window
        logger.info("  training XGBoost (locked v1 hyperparams)...")
        t1 = time.time()
        model = _train_xgb(train_merged, xgb_params)
        logger.info("    XGBoost trained in %.1fs", time.time() - t1)

        # Score every (date, ticker) row in val window (once)
        t1 = time.time()
        X_val, _ = _split_features_target(val_feat.assign(target=np.nan))
        preds = _score(model, X_val)
        scores_df = pd.DataFrame({
            "date": val_feat["date"].values,
            "ticker": val_feat["ticker"].values,
            "score": preds,
        })
        logger.info("    scored %d rows in %.1fs", len(scores_df), time.time() - t1)

        # IC for the val period — variant-independent (depends only on model)
        mean_ic, std_ic, pos_rate, n_dates = _cross_sectional_ic_for_period(
            scores_df, ctx["labels"], va_start, va_end,
        )
        logger.info(
            "    val IC: mean=%.4f std=%.4f pos_rate=%.2f n_dates=%d",
            mean_ic, std_ic, pos_rate, n_dates,
        )

        score_lookup = scores_df.set_index(["date", "ticker"])["score"]
        score_fn = _make_score_fn_from_lookup(score_lookup)

        # Run each variant on this window's val period
        for v_name in variant_names:
            logger.info("  [%s] backtest on val window...", v_name)
            t1 = time.time()
            stats = _backtest_stats_for_variant(
                v_name, score_fn, va_start, va_end, ctx, shy_returns, shy_close,
            )
            logger.info(
                "    [%s] done in %.1fs: ret=%+.1f%% CAGR=%+.2f%% "
                "excess=%+.2fpp MaxDD=%+.1f%% Sharpe=%.2f",
                v_name, time.time() - t1,
                stats.get("total_return", float("nan")) * 100,
                stats.get("cagr", float("nan")) * 100,
                stats.get("excess_cagr_vs_spy", float("nan")) * 100,
                stats.get("max_drawdown", float("nan")) * 100,
                stats.get("sharpe", float("nan")),
            )
            rows_by_variant[v_name].append({
                "window_start": tr_start.date().isoformat(),
                "window_end":   tr_end.date().isoformat(),
                "val_start":    va_start.date().isoformat(),
                "val_end":      va_end.date().isoformat(),
                "model": "xgboost",
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "positive_rate": pos_rate,
                "n_dates_scored": n_dates,
                **stats,
            })

    # ---- Write per-variant walk_forward.parquet ----
    for v_name, rows in rows_by_variant.items():
        df = pd.DataFrame(rows)
        out_dir = V2_OUT_DIR / v_name / "contract_v1"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "walk_forward.parquet"
        df.to_parquet(out_path)
        logger.info(
            "=== [%s] walk_forward.parquet written: %d rows -> %s ===",
            v_name, len(df), out_path,
        )

    # ---- Summary table ----
    print()
    print("=== Walk-forward summary across variants ===")
    for v_name in variant_names:
        df = pd.DataFrame(rows_by_variant[v_name])
        if df.empty:
            continue
        print(f"\n-- {v_name} --")
        cols = [
            "val_start", "val_end", "mean_ic", "positive_rate", "cagr",
            "excess_cagr_vs_spy", "max_drawdown", "sharpe",
        ]
        cols_present = [c for c in cols if c in df.columns]
        print(df[cols_present].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
