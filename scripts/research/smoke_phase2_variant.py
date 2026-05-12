"""Parameterized smoke runner for Phase 2 diagnostics.

Supports two configurable axes:
  --horizon   N        : label horizon in trading days (5 or 21)
  --features  STR      : 'price_macro' (22 cols) or 'full' (38 cols)
  --variant   STR      : output filename slug (e.g., 'variant_a')

Subset universe is fixed at SP500 actives for speed (~6 min wall-clock).
Saves to models/features/larger_universe_v1/phase2_smoke/<variant>_results.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.equities.study.cv import TRAIN_START, TRAIN_END, filter_to_training_window, make_folds
from src.equities.study.labels import build_labels
from src.equities.study.training import (
    cv_score, train_enet_single_fold, train_xgb_single_fold, _make_xgb_params,
)

FEATURES_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "features.parquet"
UNIVERSE_PATH = ROOT / "docs" / "larger_universe_v1_universe.json"
SMOKE_OUT_DIR = ROOT / "models" / "features" / "larger_universe_v1" / "phase2_smoke"
SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_MACRO_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d",
    "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200", "dd_252d",
    "yc_slope", "vix", "nfci", "sahm", "yc_3m",
    "baa_spread", "usd_index", "unrate", "wti_oil", "vix_5d_chg",
]

FULL_FEATURES = PRICE_MACRO_FEATURES + [
    # Fundamentals (PIT, computed via 45d reporting lag)
    "pe", "pb", "ps", "debt_to_equity", "roe", "roa", "profit_margin",
    "revenue_growth", "eps_growth",
    # Fundamentals (PIT, computed at feature date)
    "dividend_yield", "beta",
    # Categorical + index membership
    "sector", "in_sp500", "in_sp400", "in_sp600",
    # Derived
    "log_market_cap",
]


def _fold_attrs(fold_results):
    return [
        {"fold_id": r.fold_id, "mean_ic": r.mean_ic, "std_ic": r.std_ic,
         "positive_rate": r.positive_rate, "n_dates_scored": r.n_dates_scored,
         "train_rows": r.train_rows, "val_rows": r.val_rows}
        for r in fold_results
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, required=True,
                        help="Label horizon in trading days (5 or 21)")
    parser.add_argument("--features", choices=["price_macro", "full"], required=True)
    parser.add_argument("--variant", required=True, help="output slug")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    logger = logging.getLogger(args.variant)
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s %(levelname)-7s [{args.variant}] %(message)s",
                        stream=sys.stdout)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    feature_list = FULL_FEATURES if args.features == "full" else PRICE_MACRO_FEATURES
    logger.info("config: horizon=%d, features=%s (%d cols), n_trials=%d",
                args.horizon, args.features, len(feature_list), args.n_trials)
    # Embargo = horizon (label leakage prevention)
    embargo = args.horizon
    logger.info("embargo: %d trading days", embargo)

    # Universe
    u = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    smoke_tickers = sorted({r["symbol"] for r in u
                            if r["tier"] == "SP500" and r["status"] == "active"})
    logger.info("smoke universe: %d SP500 active tickers", len(smoke_tickers))

    # Features
    feat = pd.read_parquet(FEATURES_PATH)
    feat = feat[feat["ticker"].isin(smoke_tickers)]
    keep_cols = ["date", "ticker"] + feature_list
    feat = feat[keep_cols]
    feat = filter_to_training_window(feat)
    logger.info("feature subset: %s", feat.shape)

    # Labels (horizon-parameterized)
    labels = build_labels(smoke_tickers, horizon=args.horizon)
    labels = labels[(labels["date"] >= TRAIN_START) & (labels["date"] <= TRAIN_END)]
    logger.info("labels: %s (horizon=%d)", labels.shape, args.horizon)

    # Merge
    merged = feat.merge(labels, on=["date", "ticker"], how="left")
    merged = merged[merged["target"].notnull()].reset_index(drop=True)
    logger.info("merged with target: %s", merged.shape)

    unique_dates = pd.DatetimeIndex(merged["date"].unique())
    folds = make_folds(unique_dates, n_folds=args.n_folds, embargo=embargo)
    for f in folds:
        logger.info("  fold %d: train %s..%s  val %s..%s",
                    f.fold_id, f.train_start.date(), f.train_end.date(),
                    f.val_start.date(), f.val_end.date())

    # XGBoost
    logger.info("=== XGBoost (%d trials, %d folds) ===", args.n_trials, args.n_folds)
    t0 = time.time()

    def xgb_objective(trial: optuna.Trial) -> float:
        params = _make_xgb_params(trial)
        overall_mean_ic, fold_results = cv_score(train_xgb_single_fold, merged, folds, params)
        trial.set_user_attr("folds", _fold_attrs(fold_results))
        return overall_mean_ic

    xgb_study = optuna.create_study(direction="maximize", study_name=f"{args.variant}_xgb")
    xgb_study.optimize(xgb_objective, n_trials=args.n_trials, show_progress_bar=False)
    xgb_elapsed = time.time() - t0
    logger.info("XGBoost done in %.1fs; best mean IC = %.4f", xgb_elapsed, xgb_study.best_value)
    for f in xgb_study.best_trial.user_attrs.get("folds", []):
        logger.info("    fold %d  n=%d  mean_ic=%.4f  std=%.4f  pos_rate=%.2f",
                    f["fold_id"], f["n_dates_scored"], f["mean_ic"], f["std_ic"], f["positive_rate"])

    # ElasticNet
    logger.info("=== ElasticNet (%d trials, %d folds) ===", args.n_trials, args.n_folds)
    t0 = time.time()

    def enet_objective(trial: optuna.Trial) -> float:
        params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
        overall_mean_ic, fold_results = cv_score(train_enet_single_fold, merged, folds, params)
        trial.set_user_attr("folds", _fold_attrs(fold_results))
        return overall_mean_ic

    enet_study = optuna.create_study(direction="maximize", study_name=f"{args.variant}_enet")
    enet_study.optimize(enet_objective, n_trials=args.n_trials, show_progress_bar=False)
    enet_elapsed = time.time() - t0
    logger.info("ElasticNet done in %.1fs; best mean IC = %.4f", enet_elapsed, enet_study.best_value)
    for f in enet_study.best_trial.user_attrs.get("folds", []):
        logger.info("    fold %d  n=%d  mean_ic=%.4f  std=%.4f  pos_rate=%.2f",
                    f["fold_id"], f["n_dates_scored"], f["mean_ic"], f["std_ic"], f["positive_rate"])

    summary = {
        "variant": args.variant,
        "config": {
            "horizon": args.horizon,
            "features": args.features,
            "feature_list": feature_list,
            "n_features": len(feature_list),
            "embargo": embargo,
            "n_trials": args.n_trials,
            "n_folds": args.n_folds,
            "universe_tier": "SP500",
            "universe_status": "active",
            "n_tickers": len(smoke_tickers),
            "n_training_rows": int(len(merged)),
        },
        "folds": [
            {"fold_id": f.fold_id,
             "train_start": f.train_start.date().isoformat(),
             "train_end": f.train_end.date().isoformat(),
             "val_start": f.val_start.date().isoformat(),
             "val_end": f.val_end.date().isoformat()}
            for f in folds
        ],
        "xgboost": {
            "wall_clock_s": xgb_elapsed,
            "best_value_mean_ic": xgb_study.best_value,
            "best_params": xgb_study.best_params,
            "best_trial_folds": xgb_study.best_trial.user_attrs.get("folds"),
            "all_trials": [
                {"number": t.number, "value": t.value, "params": t.params,
                 "folds": t.user_attrs.get("folds")}
                for t in xgb_study.trials
            ],
        },
        "elasticnet": {
            "wall_clock_s": enet_elapsed,
            "best_value_mean_ic": enet_study.best_value,
            "best_params": enet_study.best_params,
            "best_trial_folds": enet_study.best_trial.user_attrs.get("folds"),
            "all_trials": [
                {"number": t.number, "value": t.value, "params": t.params,
                 "folds": t.user_attrs.get("folds")}
                for t in enet_study.trials
            ],
        },
    }
    out_path = SMOKE_OUT_DIR / f"{args.variant}_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
