"""Phase 2 smoke run for the Larger Universe v1 study.

10 Optuna trials of XGBoost + 10 trials of ElasticNet on a SUBSET of the
feature matrix (SP500 actives only, price+macro features only). Verifies
the pipeline plumbing: CV splitter, label join, model training, IC scoring.

NOT a real tuning run. Phase 3 will do 100-300 trials on the full universe
and feature set. Smoke output gets surfaced in the CV design doc as
sanity evidence.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.equities.study.cv import (
    TRAIN_START, TRAIN_END, filter_to_training_window, make_folds,
)
from src.equities.study.labels import build_labels
from src.equities.study.training import (
    cv_score, train_enet_single_fold, train_xgb_single_fold,
)

FEATURES_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "features.parquet"
UNIVERSE_PATH = ROOT / "docs" / "larger_universe_v1_universe.json"
SMOKE_OUT_DIR = ROOT / "models" / "features" / "larger_universe_v1" / "phase2_smoke"
SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Smoke scope
SUBSET_TIER = "SP500"
SUBSET_STATUS = "active"
SMOKE_FEATURES = [
    # Price-derived (12)
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d",
    "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200", "dd_252d",
    # Macro (10)
    "yc_slope", "vix", "nfci", "sahm", "yc_3m",
    "baa_spread", "usd_index", "unrate", "wti_oil", "vix_5d_chg",
]
N_TRIALS = 10
N_FOLDS = 5

logger = logging.getLogger("smoke_phase2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def main() -> int:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("loading universe...")
    u = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    smoke_tickers = sorted({r["symbol"] for r in u
                            if r["tier"] == SUBSET_TIER and r["status"] == SUBSET_STATUS})
    logger.info("smoke universe: %d %s %s tickers", len(smoke_tickers), SUBSET_TIER, SUBSET_STATUS)

    logger.info("loading features...")
    feat = pd.read_parquet(FEATURES_PATH)
    feat = feat[feat["ticker"].isin(smoke_tickers)]
    # Keep only smoke features + metadata
    keep_cols = ["date", "ticker"] + SMOKE_FEATURES
    feat = feat[keep_cols]
    feat = filter_to_training_window(feat)
    logger.info("feature subset: %s", feat.shape)

    logger.info("building labels...")
    labels = build_labels(smoke_tickers)
    labels = labels[(labels["date"] >= TRAIN_START) & (labels["date"] <= TRAIN_END)]
    logger.info("labels: %s", labels.shape)

    logger.info("merging features + labels...")
    merged = feat.merge(labels, on=["date", "ticker"], how="left")
    merged = merged[merged["target_fwd_5d"].notnull()].reset_index(drop=True)
    logger.info("merged with target: %s", merged.shape)

    unique_dates = pd.DatetimeIndex(merged["date"].unique())
    logger.info("unique training dates: %d", len(unique_dates))
    folds = make_folds(unique_dates, n_folds=N_FOLDS)
    for f in folds:
        logger.info("  fold %d: train %s..%s  val %s..%s",
                    f.fold_id, f.train_start.date(), f.train_end.date(),
                    f.val_start.date(), f.val_end.date())

    def _fold_attrs(fold_results):
        return [
            {"fold_id": r.fold_id, "mean_ic": r.mean_ic, "std_ic": r.std_ic,
             "positive_rate": r.positive_rate, "n_dates_scored": r.n_dates_scored}
            for r in fold_results
        ]

    # XGBoost smoke
    logger.info("=== XGBoost smoke (%d trials, %d folds) ===", N_TRIALS, N_FOLDS)
    t0 = time.time()
    from src.equities.study.training import _make_xgb_params

    def xgb_objective(trial: optuna.Trial) -> float:
        params = _make_xgb_params(trial)
        overall_mean_ic, fold_results = cv_score(train_xgb_single_fold, merged, folds, params)
        trial.set_user_attr("folds", _fold_attrs(fold_results))
        return overall_mean_ic

    xgb_study = optuna.create_study(direction="maximize", study_name="larger_universe_v1_phase2_xgb_smoke")
    xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
    logger.info("XGBoost smoke done in %.1fs", time.time() - t0)
    logger.info("  best mean cross-sectional IC: %.4f", xgb_study.best_value)
    logger.info("  best params: %s", xgb_study.best_params)
    best_xgb = xgb_study.best_trial
    for f in best_xgb.user_attrs.get("folds", []):
        logger.info("    fold %d  mean_ic=%.4f  std=%.4f  pos_rate=%.2f  n_dates=%d",
                    f["fold_id"], f["mean_ic"], f["std_ic"], f["positive_rate"], f["n_dates_scored"])

    # ElasticNet smoke
    logger.info("=== ElasticNet smoke (%d trials, %d folds) ===", N_TRIALS, N_FOLDS)
    t0 = time.time()

    def enet_objective(trial: optuna.Trial) -> float:
        params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
        overall_mean_ic, fold_results = cv_score(train_enet_single_fold, merged, folds, params)
        trial.set_user_attr("folds", _fold_attrs(fold_results))
        return overall_mean_ic

    enet_study = optuna.create_study(direction="maximize", study_name="larger_universe_v1_phase2_enet_smoke")
    enet_study.optimize(enet_objective, n_trials=N_TRIALS, show_progress_bar=False)
    logger.info("ElasticNet smoke done in %.1fs", time.time() - t0)
    logger.info("  best mean cross-sectional IC: %.4f", enet_study.best_value)
    logger.info("  best params: %s", enet_study.best_params)
    best_enet = enet_study.best_trial
    for f in best_enet.user_attrs.get("folds", []):
        logger.info("    fold %d  mean_ic=%.4f  std=%.4f  pos_rate=%.2f  n_dates=%d",
                    f["fold_id"], f["mean_ic"], f["std_ic"], f["positive_rate"], f["n_dates_scored"])

    # Persist smoke results
    summary = {
        "subset": {
            "tier": SUBSET_TIER,
            "status": SUBSET_STATUS,
            "n_tickers": len(smoke_tickers),
            "n_features": len(SMOKE_FEATURES),
            "feature_list": SMOKE_FEATURES,
            "n_training_rows": int(len(merged)),
        },
        "folds": [
            {
                "fold_id": f.fold_id,
                "train_start": f.train_start.date().isoformat(),
                "train_end": f.train_end.date().isoformat(),
                "val_start": f.val_start.date().isoformat(),
                "val_end": f.val_end.date().isoformat(),
            }
            for f in folds
        ],
        "xgboost": {
            "n_trials": len(xgb_study.trials),
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
            "n_trials": len(enet_study.trials),
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
    out_path = SMOKE_OUT_DIR / "smoke_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
