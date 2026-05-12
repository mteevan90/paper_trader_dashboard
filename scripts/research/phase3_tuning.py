"""Phase 3 — full Optuna hyperparameter tuning for the Larger Universe v1 study.

Full universe (1,963 tickers), full 38-feature set, 21-day forward-return
label, 21-day CV embargo, 5-fold expanding-window TimeSeriesSplit over the
training window. 200 trials per model (XGBoost primary + ElasticNet sanity).

Outputs:
  models/studies/larger_universe_v1/xgboost_best_params.json
  models/studies/larger_universe_v1/elasticnet_best_params.json
  models/studies/larger_universe_v1/xgboost_study.json   (full trial log)
  models/studies/larger_universe_v1/elasticnet_study.json
  models/studies/larger_universe_v1/phase3_progress.log  (line-by-line)

Idempotent: writes intermediate progress so a kill+resume could be added
later, though for the v1 study we run end-to-end.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.equities.study.cv import TRAIN_START, TRAIN_END, filter_to_training_window, make_folds
from src.equities.study.labels import build_labels, EMBARGO_TRADING_DAYS, LABEL_HORIZON_TRADING_DAYS
from src.equities.study.training import (
    cv_score, train_enet_single_fold, train_xgb_single_fold, _make_xgb_params,
)
from scripts.research.smoke_phase2_variant import FULL_FEATURES

FEATURES_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "features.parquet"
UNIVERSE_PATH = ROOT / "docs" / "larger_universe_v1_universe.json"
STUDY_DIR = ROOT / "models" / "studies" / "larger_universe_v1"
STUDY_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_LOG = STUDY_DIR / "phase3_progress.log"


def _setup_logging() -> logging.Logger:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    handlers = [
        logging.FileHandler(PROGRESS_LOG, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("phase3")


def _fold_attrs(fold_results):
    return [
        {"fold_id": r.fold_id, "mean_ic": r.mean_ic, "std_ic": r.std_ic,
         "positive_rate": r.positive_rate, "n_dates_scored": r.n_dates_scored,
         "train_rows": r.train_rows, "val_rows": r.val_rows}
        for r in fold_results
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgb-trials", type=int, default=200)
    parser.add_argument("--enet-trials", type=int, default=100,
                        help="ElasticNet trial count — smaller because the search "
                             "space (alpha + l1_ratio) is 2-D and TPE typically "
                             "plateaus by trial 50-80 on this dimensionality")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Optuna TPE sampler seed for reproducibility")
    parser.add_argument("--convergence-interval", type=int, default=25,
                        help="Log running-best IC every N trials")
    parser.add_argument("--slow-trial-threshold-s", type=float, default=600.0,
                        help="Per-trial elapsed > threshold triggers a WARNING")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--skip-enet", action="store_true")
    args = parser.parse_args()

    logger = _setup_logging()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("=== Phase 3 tuning — Larger Universe v1 ===")
    logger.info("config: xgb_trials=%d, enet_trials=%d, n_folds=%d, "
                "horizon=%d, embargo=%d, seed=%d",
                args.xgb_trials, args.enet_trials, args.n_folds,
                LABEL_HORIZON_TRADING_DAYS, EMBARGO_TRADING_DAYS, args.seed)

    # Universe — FULL (not the SP500 actives subset)
    u = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    by_sym: dict[str, dict] = {}
    for r in u:
        s = r["symbol"]
        if s not in by_sym or (by_sym[s]["status"] == "removed" and r["status"] == "active"):
            by_sym[s] = r
    full_universe = sorted(by_sym.keys())
    logger.info("full universe: %d unique symbols", len(full_universe))

    # Features
    t0 = time.time()
    logger.info("loading features (full set, full universe)...")
    feat = pd.read_parquet(FEATURES_PATH)
    feat = feat[feat["ticker"].isin(full_universe)]
    keep_cols = ["date", "ticker"] + FULL_FEATURES
    feat = feat[keep_cols]
    feat = filter_to_training_window(feat)
    logger.info("  features: %s in %.1fs", feat.shape, time.time() - t0)

    # Labels (21d horizon)
    t0 = time.time()
    logger.info("building labels (horizon=%d)...", LABEL_HORIZON_TRADING_DAYS)
    labels = build_labels(full_universe, horizon=LABEL_HORIZON_TRADING_DAYS)
    labels = labels[(labels["date"] >= TRAIN_START) & (labels["date"] <= TRAIN_END)]
    logger.info("  labels: %s in %.1fs", labels.shape, time.time() - t0)

    # Merge
    t0 = time.time()
    merged = feat.merge(labels, on=["date", "ticker"], how="left")
    merged = merged[merged["target"].notnull()].reset_index(drop=True)
    logger.info("merged + filtered: %s in %.1fs", merged.shape, time.time() - t0)

    unique_dates = pd.DatetimeIndex(merged["date"].unique())
    folds = make_folds(unique_dates, n_folds=args.n_folds, embargo=EMBARGO_TRADING_DAYS)
    for f in folds:
        logger.info("  fold %d: train %s..%s  val %s..%s",
                    f.fold_id, f.train_start.date(), f.train_end.date(),
                    f.val_start.date(), f.val_end.date())

    def _save_study(study: optuna.Study, label: str) -> None:
        out_path = STUDY_DIR / f"{label}_study.json"
        body = {
            "label": label,
            "n_trials_completed": len(study.trials),
            "best_value_mean_ic": study.best_value if study.best_trial else None,
            "best_params": study.best_params if study.best_trial else None,
            "best_trial_folds": (study.best_trial.user_attrs.get("folds")
                                 if study.best_trial else None),
            "trials": [
                {"number": t.number,
                 "state": t.state.name,
                 "value": t.value,
                 "params": t.params,
                 "folds": t.user_attrs.get("folds"),
                 "duration_s": (t.datetime_complete - t.datetime_start).total_seconds()
                               if t.datetime_complete and t.datetime_start else None}
                for t in study.trials
            ],
        }
        out_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        if study.best_trial:
            (STUDY_DIR / f"{label}_best_params.json").write_text(
                json.dumps({
                    "best_value_mean_ic": study.best_value,
                    "best_params": study.best_params,
                    "best_trial_folds": study.best_trial.user_attrs.get("folds"),
                    "label_horizon_trading_days": LABEL_HORIZON_TRADING_DAYS,
                    "embargo_trading_days": EMBARGO_TRADING_DAYS,
                    "universe_size": len(full_universe),
                    "n_training_rows": int(len(merged)),
                }, indent=2, default=str),
                encoding="utf-8",
            )

    # XGBoost
    if not args.skip_xgb:
        logger.info("=== XGBoost (%d trials, %d folds, seed=%d) ===",
                    args.xgb_trials, args.n_folds, args.seed)
        t0 = time.time()

        def xgb_objective(trial: optuna.Trial) -> float:
            t_start = time.time()
            params = _make_xgb_params(trial)
            overall_mean_ic, fold_results = cv_score(train_xgb_single_fold, merged, folds, params)
            trial.set_user_attr("folds", _fold_attrs(fold_results))
            elapsed = time.time() - t_start
            trial.set_user_attr("duration_s", elapsed)
            best_so_far = trial.study.best_value if trial.study.best_trial else float("nan")
            logger.info("  XGB trial %d/%d  ic=%.4f  (best so far %.4f)  %.1fs",
                        trial.number + 1, args.xgb_trials, overall_mean_ic,
                        best_so_far, elapsed)
            # Pathological-trial warning
            if elapsed > args.slow_trial_threshold_s:
                logger.warning("  XGB trial %d slow: %.1fs > %.0fs threshold. "
                               "Params: %s", trial.number + 1, elapsed,
                               args.slow_trial_threshold_s, trial.params)
            # Convergence checkpoint
            if (trial.number + 1) % args.convergence_interval == 0:
                logger.info("  XGB convergence @ trial %d: running_best=%.4f",
                            trial.number + 1, best_so_far)
            # Persist progress every 10 trials for safety
            if (trial.number + 1) % 10 == 0:
                _save_study(trial.study, "xgboost")
            return overall_mean_ic

        xgb_study = optuna.create_study(
            direction="maximize",
            study_name="larger_universe_v1_phase3_xgb",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        xgb_study.optimize(xgb_objective, n_trials=args.xgb_trials, show_progress_bar=False)
        logger.info("XGBoost done in %.1fs (%.2fh)", time.time() - t0, (time.time() - t0) / 3600)
        _save_study(xgb_study, "xgboost")
        logger.info("  best mean cross-sec IC = %.4f", xgb_study.best_value)
        for f in xgb_study.best_trial.user_attrs.get("folds", []):
            logger.info("    fold %d  n=%d  mean_ic=%.4f  std=%.4f  pos_rate=%.2f",
                        f["fold_id"], f["n_dates_scored"], f["mean_ic"],
                        f["std_ic"], f["positive_rate"])

    # ElasticNet
    if not args.skip_enet:
        logger.info("=== ElasticNet (%d trials, %d folds, seed=%d) ===",
                    args.enet_trials, args.n_folds, args.seed)
        t0 = time.time()

        def enet_objective(trial: optuna.Trial) -> float:
            t_start = time.time()
            params = {
                "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }
            overall_mean_ic, fold_results = cv_score(train_enet_single_fold, merged, folds, params)
            trial.set_user_attr("folds", _fold_attrs(fold_results))
            elapsed = time.time() - t_start
            trial.set_user_attr("duration_s", elapsed)
            best_so_far = trial.study.best_value if trial.study.best_trial else float("nan")
            logger.info("  ENet trial %d/%d  ic=%.4f  (best so far %.4f)  %.1fs",
                        trial.number + 1, args.enet_trials, overall_mean_ic,
                        best_so_far, elapsed)
            if elapsed > args.slow_trial_threshold_s:
                logger.warning("  ENet trial %d slow: %.1fs > %.0fs threshold. "
                               "Params: %s", trial.number + 1, elapsed,
                               args.slow_trial_threshold_s, trial.params)
            if (trial.number + 1) % args.convergence_interval == 0:
                logger.info("  ENet convergence @ trial %d: running_best=%.4f",
                            trial.number + 1, best_so_far)
            if (trial.number + 1) % 10 == 0:
                _save_study(trial.study, "elasticnet")
            return overall_mean_ic

        enet_study = optuna.create_study(
            direction="maximize",
            study_name="larger_universe_v1_phase3_enet",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        enet_study.optimize(enet_objective, n_trials=args.enet_trials, show_progress_bar=False)
        logger.info("ElasticNet done in %.1fs (%.2fh)",
                    time.time() - t0, (time.time() - t0) / 3600)
        _save_study(enet_study, "elasticnet")
        logger.info("  best mean cross-sec IC = %.4f", enet_study.best_value)
        for f in enet_study.best_trial.user_attrs.get("folds", []):
            logger.info("    fold %d  n=%d  mean_ic=%.4f  std=%.4f  pos_rate=%.2f",
                        f["fold_id"], f["n_dates_scored"], f["mean_ic"],
                        f["std_ic"], f["positive_rate"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
