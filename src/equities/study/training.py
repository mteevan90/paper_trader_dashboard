"""Model training pipelines for the Larger Universe v1 study.

Two parallel pipelines on identical features and identical folds:

  - XGBoost: native NaN handling, native categorical support (for sector).
    Predicts forward 5-day return per (date, ticker). Optuna search space
    covers tree depth, learning rate, n_estimators, subsample, colsample,
    min_child_weight, gamma, reg_alpha, reg_lambda.

  - ElasticNet: sklearn Pipeline with SimpleImputer(strategy='mean',
    add_indicator=True) + StandardScaler + ElasticNet. One-hot encodes
    sector before the pipeline. Search space: alpha, l1_ratio.

The objective scored across all CV folds is the Information Coefficient (IC) —
Spearman rank correlation between predictions and realized forward returns,
averaged over folds. IC is the standard quant-research metric for
cross-sectional alpha and is more robust than MSE for ranking-oriented
portfolio construction. Higher IC = better.
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
        "max_depth": trial.suggest_int("max_depth", 3, 10),
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
                           params: dict) -> tuple[xgb.XGBRegressor, float]:
    """Train XGBoost on train_df, score on val_df, return (model, ic)."""
    train_df = _drop_no_target_rows(train_df)
    val_df = _drop_no_target_rows(val_df)
    X_tr, y_tr = _split_features_target(train_df)
    X_va, y_va = _split_features_target(val_df)
    X_tr = _prep_xgb_X(X_tr)
    X_va = _prep_xgb_X(X_va)
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    # Spearman IC
    ic = _safe_spearman(preds, y_va.values)
    return model, ic


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
                            params: dict) -> tuple[Pipeline, float]:
    """Train ElasticNet on train_df, score on val_df, return (pipeline, ic)."""
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
    ic = _safe_spearman(preds, y_va.values)
    return pipe, ic


# -------- Scoring --------


def _safe_spearman(preds: np.ndarray, y: np.ndarray) -> float:
    """Spearman IC tolerating NaN in either input."""
    mask = np.isfinite(preds) & np.isfinite(y)
    if mask.sum() < 100:
        return float("nan")
    rho, _ = spearmanr(preds[mask], y[mask])
    return float(rho) if rho == rho else float("nan")


# -------- CV driver --------


@dataclass
class FoldResult:
    fold_id: int
    train_rows: int
    val_rows: int
    ic: float


def cv_score(model_fn, features_with_target: pd.DataFrame, folds: list[Fold],
             params: dict) -> tuple[float, list[FoldResult]]:
    """Run a single set of hyperparameters across all CV folds.

    `model_fn` is one of train_xgb_single_fold / train_enet_single_fold.
    Returns (mean_ic_across_folds, per-fold results).
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
        _, ic = model_fn(tr, va, params)
        results.append(FoldResult(
            fold_id=fold.fold_id,
            train_rows=len(tr),
            val_rows=len(va),
            ic=ic,
        ))
    valid_ics = [r.ic for r in results if r.ic == r.ic]
    mean_ic = float(np.mean(valid_ics)) if valid_ics else float("nan")
    return mean_ic, results
