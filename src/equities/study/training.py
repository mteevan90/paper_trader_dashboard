"""Model training pipelines for the Larger Universe v1 study.

Two parallel pipelines on identical features and identical folds:

  - XGBoost: native NaN handling, native categorical support (for sector).
    Predicts forward 5-day return per (date, ticker). Optuna search space
    covers tree depth, learning rate, n_estimators, subsample, colsample,
    min_child_weight, gamma, reg_alpha, reg_lambda.

  - ElasticNet: sklearn Pipeline with SimpleImputer(strategy='mean',
    add_indicator=True) + StandardScaler + ElasticNet. One-hot encodes
    sector before the pipeline. Search space: alpha, l1_ratio.

Scoring is the **cross-sectional Information Coefficient** (IC):

  For each unique date D in the validation block with >= min_tickers
  (default 30) predictions, compute Spearman(pred, realized_return)
  across the tickers available on D. Aggregate ACROSS dates: mean,
  std, and positive-rate.

Cross-sectional IC isolates the stock-ranking signal that the portfolio
actually uses, separate from any market-timing signal that panel-wise
correlation would conflate. Mean cross-sectional IC across folds is the
Optuna objective; std and positive-rate are reported but not optimized.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.equities.study.cv import Fold

logger = logging.getLogger(__name__)


# Columns that are NOT features (used to identify the feature set vs metadata)
NON_FEATURE_COLS = ("date", "ticker", "target_fwd_5d")

# Categorical columns — handled differently in the two pipelines
CATEGORICAL_COLS = ("sector",)


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop date/ticker/target columns; return (X, y)."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["target_fwd_5d"]
    return X, y


def _drop_no_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the target is NaN (last 5 days per ticker)."""
    return df[df["target_fwd_5d"].notnull()].copy()


# -------- XGBoost --------


def _prep_xgb_X(X: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to pd.Categorical dtype for XGBoost native handling."""
    X = X.copy()
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def _make_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "enable_categorical": True,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "verbosity": 0,
    }


def train_xgb_single_fold(train_df: pd.DataFrame, val_df: pd.DataFrame,
                           params: dict) -> tuple[xgb.XGBRegressor, dict]:
    """Train XGBoost on train_df, score cross-sectional IC stats on val_df."""
    train_df = _drop_no_target_rows(train_df)
    val_df = _drop_no_target_rows(val_df)
    X_tr, y_tr = _split_features_target(train_df)
    X_va, y_va = _split_features_target(val_df)
    X_tr = _prep_xgb_X(X_tr)
    X_va = _prep_xgb_X(X_va)
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    stats = cross_sectional_ic_stats(preds, val_df)
    return model, stats


# -------- ElasticNet --------


def _make_elasticnet_pipeline(alpha: float, l1_ratio: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000,
                              random_state=42)),
    ])


def _prep_enet_X(X: pd.DataFrame, *, sector_levels: Optional[list[str]] = None) -> tuple[pd.DataFrame, list[str]]:
    """One-hot the sector column; return (X_dense, sector_level_list).

    Pass sector_levels from the training fold when encoding the validation
    fold so the column set is consistent (sklearn ElasticNet rejects shape
    mismatches between fit and predict).
    """
    X = X.copy()
    if "sector" in X.columns:
        if sector_levels is None:
            sector_levels = sorted([s for s in X["sector"].unique() if isinstance(s, str)])
        for lvl in sector_levels:
            X[f"sector__{lvl}"] = (X["sector"] == lvl).astype(int)
        X = X.drop(columns=["sector"])
    return X, (sector_levels or [])


def train_enet_single_fold(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            params: dict) -> tuple[Pipeline, dict]:
    """Train ElasticNet on train_df, score cross-sectional IC stats on val_df."""
    train_df = _drop_no_target_rows(train_df)
    val_df = _drop_no_target_rows(val_df)
    X_tr, y_tr = _split_features_target(train_df)
    X_va, y_va = _split_features_target(val_df)
    X_tr, sector_levels = _prep_enet_X(X_tr)
    X_va, _ = _prep_enet_X(X_va, sector_levels=sector_levels)
    # Align columns (val may be missing some sector levels seen only in train)
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)
    pipe = _make_elasticnet_pipeline(params["alpha"], params["l1_ratio"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings noisy on small smokes
        pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_va)
    stats = cross_sectional_ic_stats(preds, val_df)
    return pipe, stats


# -------- Scoring --------


MIN_TICKERS_PER_DATE_FOR_IC = 30


def cross_sectional_ic_stats(preds: np.ndarray, val_df: pd.DataFrame,
                              min_tickers: int = MIN_TICKERS_PER_DATE_FOR_IC) -> dict:
    """Compute cross-sectional Spearman IC stats over a validation fold.

    For each unique date D in `val_df` with at least `min_tickers` rows
    (after dropping NaN preds/targets), compute Spearman correlation
    between predictions and realized forward returns AMONG TICKERS ON
    DATE D. Then aggregate the per-date correlations:

      - mean_ic         : average correlation across qualifying dates
      - std_ic          : standard deviation across qualifying dates
                          (high std => signal driven by outlier dates)
      - positive_rate   : fraction of qualifying dates with IC > 0
                          ("how often would you win?")
      - n_dates_scored  : count of qualifying dates

    Returns dict with NaN for the metrics if no date had enough tickers
    or all per-date correlations were undefined.
    """
    df = pd.DataFrame({
        "pred": np.asarray(preds, dtype=float),
        "y": val_df["target_fwd_5d"].to_numpy(dtype=float),
        "date": val_df["date"].to_numpy(),
    }).dropna()
    per_date: list[float] = []
    for _, g in df.groupby("date", sort=False):
        if len(g) < min_tickers:
            continue
        # Spearman is undefined if either input is constant
        if g["pred"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        rho, _ = spearmanr(g["pred"], g["y"])
        if rho == rho:  # filter NaN
            per_date.append(float(rho))
    if not per_date:
        return {"mean_ic": float("nan"), "std_ic": float("nan"),
                "positive_rate": float("nan"), "n_dates_scored": 0}
    arr = np.asarray(per_date, dtype=float)
    return {
        "mean_ic": float(arr.mean()),
        "std_ic": float(arr.std(ddof=0)),
        "positive_rate": float((arr > 0).mean()),
        "n_dates_scored": int(len(arr)),
    }


# -------- CV driver --------


@dataclass
class FoldResult:
    fold_id: int
    train_rows: int
    val_rows: int
    mean_ic: float
    std_ic: float
    positive_rate: float
    n_dates_scored: int


def cv_score(model_fn, features_with_target: pd.DataFrame, folds: list[Fold],
             params: dict) -> tuple[float, list[FoldResult]]:
    """Run a single set of hyperparameters across all CV folds.

    Returns (mean_of_per_fold_mean_ic, per-fold results). The Optuna
    objective is the mean across folds; std_ic and positive_rate are
    reported per fold for review but not part of the optimization target.
    """
    results: list[FoldResult] = []
    for fold in folds:
        tr = features_with_target[
            (features_with_target["date"] >= fold.train_start) &
            (features_with_target["date"] <= fold.train_end)
        ]
        va = features_with_target[
            (features_with_target["date"] >= fold.val_start) &
            (features_with_target["date"] <= fold.val_end)
        ]
        _, stats = model_fn(tr, va, params)
        results.append(FoldResult(
            fold_id=fold.fold_id,
            train_rows=len(tr),
            val_rows=len(va),
            mean_ic=stats["mean_ic"],
            std_ic=stats["std_ic"],
            positive_rate=stats["positive_rate"],
            n_dates_scored=stats["n_dates_scored"],
        ))
    valid_means = [r.mean_ic for r in results if r.mean_ic == r.mean_ic]
    overall_mean_ic = float(np.mean(valid_means)) if valid_means else float("nan")
    return overall_mean_ic, results
