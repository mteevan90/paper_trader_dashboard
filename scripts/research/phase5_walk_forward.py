"""Phase 5 walk-forward analysis — does the strategy hold up year-over-year?

For each rolling 3-year training window, retrain XGBoost and ElasticNet
with the LOCKED Phase 3 hyperparameters (no re-tuning), score the next
year's monthly rebalance dates, run a mini-backtest, aggregate stats.

Windows (val period = year following training):
  W1: train 2017-05-12..2020-05-11  val 2020-05-12..2021-05-11
  W2: train 2018-05-12..2021-05-11  val 2021-05-12..2022-05-11
  W3: train 2019-05-12..2022-05-11  val 2022-05-12..2023-05-11
  W4: train 2020-05-12..2023-05-11  val 2023-05-12..2024-05-11
  W5: train 2021-05-12..2024-05-11  val 2024-05-12..2025-05-11
  W6: train 2022-05-12..2025-05-11  val 2025-05-12..2026-05-11

Per window per model: mean_ic, std_ic, positive_rate, total_return,
excess_cagr_vs_spy, max_drawdown, sharpe. Output:
  models/studies/larger_universe_v1/contract_v1/walk_forward.parquet
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.equities.study.labels import build_labels, LABEL_HORIZON_TRADING_DAYS
from src.equities.study.training import (
    _make_elasticnet_pipeline, _prep_enet_X, _prep_xgb_X, _split_features_target,
    NON_FEATURE_COLS,
)
from src.equities.study.portfolio import PortfolioConstructionParams, rank_top_n_weights
from src.equities.study.backtest import (
    month_end_trading_dates, run_backtest, TRANSACTION_COST_PCT,
)

FEATURES_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "features.parquet"
UNIVERSE_PATH = ROOT / "docs" / "larger_universe_v1_universe.json"
SECTOR_MAP_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "sector_map.json"
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"
XGB_PARAMS_PATH = ROOT / "models" / "studies" / "larger_universe_v1" / "xgboost_best_params.json"
ENET_PARAMS_PATH = ROOT / "models" / "studies" / "larger_universe_v1" / "elasticnet_best_params.json"
BENCH_PRICE_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "prices"
OUT_DIR = ROOT / "models" / "studies" / "larger_universe_v1" / "contract_v1"

WINDOWS = [
    (pd.Timestamp("2017-05-12"), pd.Timestamp("2020-05-11"), pd.Timestamp("2020-05-12"), pd.Timestamp("2021-05-11")),
    (pd.Timestamp("2018-05-12"), pd.Timestamp("2021-05-11"), pd.Timestamp("2021-05-12"), pd.Timestamp("2022-05-11")),
    (pd.Timestamp("2019-05-12"), pd.Timestamp("2022-05-11"), pd.Timestamp("2022-05-12"), pd.Timestamp("2023-05-11")),
    (pd.Timestamp("2020-05-12"), pd.Timestamp("2023-05-11"), pd.Timestamp("2023-05-12"), pd.Timestamp("2024-05-11")),
    (pd.Timestamp("2021-05-12"), pd.Timestamp("2024-05-11"), pd.Timestamp("2024-05-12"), pd.Timestamp("2025-05-11")),
    (pd.Timestamp("2022-05-12"), pd.Timestamp("2025-05-11"), pd.Timestamp("2025-05-12"), pd.Timestamp("2026-05-11")),
]

logger = logging.getLogger("phase5_wf")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def _setup() -> dict:
    """Load all the inputs once."""
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

    # Daily returns from snapshot prices
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

    # SPY for benchmark
    spy = pd.read_parquet(BENCH_PRICE_DIR / "SPY.parquet")
    spy.index = pd.to_datetime(spy.index)
    spy_close = spy["close"]

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
        "spy_close": spy_close,
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


def _train_enet(merged: pd.DataFrame, params: dict):
    X, y = _split_features_target(merged)
    X_prep, sector_levels = _prep_enet_X(X)
    pipe = _make_elasticnet_pipeline(params["alpha"], params["l1_ratio"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_prep, y)
    return pipe, sector_levels


def _score(model_name, model, X_df, sector_levels=None) -> np.ndarray:
    if model_name == "xgboost":
        X = _prep_xgb_X(X_df)
        return model.predict(X)
    else:
        X_prep, _ = _prep_enet_X(X_df, sector_levels=sector_levels)
        return model.predict(X_prep)


def _cross_sectional_ic_for_period(scores_df, labels, val_start, val_end,
                                     min_tickers=30):
    """Compute mean cross-sectional IC and positive_rate for a val period.

    scores_df: long-format DataFrame [date, ticker, score]
    labels: long-format DataFrame [date, ticker, target]
    """
    merged = scores_df.merge(labels, on=["date", "ticker"], how="left")
    merged = merged[(merged["date"] >= val_start) & (merged["date"] <= val_end)]
    merged = merged.dropna(subset=["score", "target"])
    per_date = []
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


def _backtest_stats(model_name, score_fn, val_start, val_end, ctx, pc_params):
    """Run a 1-year mini-backtest for the val window and return summary stats."""
    rebal_dates = month_end_trading_dates(ctx["trading_dates"], val_start, val_end)
    if not rebal_dates:
        return {}
    result = run_backtest(
        model_name=model_name,
        score_fn=score_fn,
        rebalance_dates=rebal_dates,
        daily_returns=ctx["daily_returns"],
        delisting_dates=ctx["delisting_dates"],
        sectors=ctx["sectors"],
        tiers=ctx["tiers"],
        pc_params=pc_params,
        universe_records=ctx["universe_records"],
    )
    port = result.portfolio.copy()
    if port.empty:
        return {}
    nav = port["nav"].values
    n_days = len(nav)
    total_return = float(nav[-1] / nav[0] - 1.0)
    years = n_days / 252.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    # Daily returns of strategy
    daily_ret = pd.Series(nav).pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)
    rolling_max = pd.Series(nav).cummax()
    max_dd = float((pd.Series(nav) / rolling_max - 1).min())
    # SPY in same window
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
        "excess_cagr_vs_spy": float(cagr - spy_cagr) if not np.isnan(spy_cagr) else float("nan"),
    }


def main() -> int:
    logger.info("=== Phase 5 walk-forward analysis ===")
    logger.info("loading inputs...")
    t0 = time.time()
    ctx = _setup()
    logger.info("  setup done in %.1fs", time.time() - t0)

    xgb_params = json.loads(XGB_PARAMS_PATH.read_text())["best_params"]
    enet_params = json.loads(ENET_PARAMS_PATH.read_text())["best_params"]
    pc_params = PortfolioConstructionParams()  # defaults match Phase 4

    rows = []
    for i, (tr_start, tr_end, va_start, va_end) in enumerate(WINDOWS, 1):
        logger.info("--- Window %d/%d: train %s..%s, val %s..%s ---",
                    i, len(WINDOWS), tr_start.date(), tr_end.date(),
                    va_start.date(), va_end.date())
        # Build train slice
        feat = ctx["features"]
        labels = ctx["labels"]
        train_feat = feat[(feat["date"] >= tr_start) & (feat["date"] <= tr_end)]
        train_lbl = labels[(labels["date"] >= tr_start) & (labels["date"] <= tr_end)]
        train_merged = train_feat.merge(train_lbl, on=["date", "ticker"], how="left")
        train_merged = train_merged[train_merged["target"].notnull()].reset_index(drop=True)
        logger.info("  train rows: %d", len(train_merged))

        val_feat = feat[(feat["date"] >= va_start) & (feat["date"] <= va_end)]
        logger.info("  val rows (features): %d", len(val_feat))

        for model_name, params, train_fn in (
            ("xgboost",    xgb_params,  _train_xgb),
            ("elasticnet", enet_params, _train_enet),
        ):
            logger.info("  training %s...", model_name)
            t1 = time.time()
            if model_name == "xgboost":
                model = train_fn(train_merged, params)
                sector_levels = None
            else:
                model, sector_levels = train_fn(train_merged, params)
            logger.info("    %s trained in %.1fs", model_name, time.time() - t1)

            # Score every (date, ticker) row in val window
            t1 = time.time()
            X_val, _ = _split_features_target(val_feat.assign(target=np.nan))
            preds = _score(model_name, model, X_val, sector_levels=sector_levels)
            scores_df = pd.DataFrame({
                "date": val_feat["date"].values,
                "ticker": val_feat["ticker"].values,
                "score": preds,
            })
            logger.info("    scored %d rows in %.1fs", len(scores_df), time.time() - t1)

            # IC for the val period
            mean_ic, std_ic, pos_rate, n_dates = _cross_sectional_ic_for_period(
                scores_df, ctx["labels"], va_start, va_end)
            logger.info("    %s val IC: mean=%.4f std=%.4f pos_rate=%.2f n_dates=%d",
                        model_name, mean_ic, std_ic, pos_rate, n_dates)

            # Mini-backtest
            score_lookup = scores_df.set_index(["date", "ticker"])["score"]

            def make_fn(score_lookup_local=score_lookup):
                def fn(d):
                    if d not in score_lookup_local.index.get_level_values(0):
                        return pd.Series(dtype=float)
                    rows_for_date = score_lookup_local.loc[d]
                    if isinstance(rows_for_date, pd.Series):
                        return rows_for_date
                    return pd.Series(dtype=float)
                return fn

            t1 = time.time()
            stats = _backtest_stats(model_name, make_fn(), va_start, va_end, ctx,
                                       pc_params)
            logger.info("    %s val backtest done in %.1fs: ret=%+.1f%% CAGR=%+.2f%% "
                        "excess=%+.2fpp MaxDD=%+.1f%% Sharpe=%.2f",
                        model_name, time.time() - t1,
                        stats.get("total_return", float("nan")) * 100,
                        stats.get("cagr", float("nan")) * 100,
                        stats.get("excess_cagr_vs_spy", float("nan")) * 100,
                        stats.get("max_drawdown", float("nan")) * 100,
                        stats.get("sharpe", float("nan")))

            rows.append({
                "window_start": tr_start.date().isoformat(),
                "window_end":   tr_end.date().isoformat(),
                "val_start":    va_start.date().isoformat(),
                "val_end":      va_end.date().isoformat(),
                "model": model_name,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "positive_rate": pos_rate,
                "n_dates_scored": n_dates,
                **stats,
            })

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "walk_forward.parquet"
    df.to_parquet(out_path)
    logger.info("=== Walk-forward DONE: %s (%d rows) ===", out_path, len(df))

    # Print summary table
    print()
    print("=== Walk-forward summary ===")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
