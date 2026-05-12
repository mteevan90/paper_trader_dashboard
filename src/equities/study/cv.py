"""Time-series cross-validation with embargo for the Larger Universe v1 study.

5-fold expanding-window split over the training period
(2017-05-12 → 2023-05-11, ~1,500 trading days). Each fold validates on
~12 months and trains on everything before the validation block minus a
5-trading-day embargo (the label horizon).

Embargo prevents leakage: with a 5-day forward-return label, train rows
within 5 days of the validation start have labels that reference prices
inside the validation window.

The splitter operates on unique sorted dates from the feature matrix and
yields (train_dates, val_dates) date sets, NOT row indices. The caller
applies the date filter to the (date, ticker) row dataframe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.equities.study.labels import EMBARGO_TRADING_DAYS


TRAIN_START = pd.Timestamp("2017-05-12")
TRAIN_END = pd.Timestamp("2023-05-11")
TEST_START = pd.Timestamp("2023-05-12")
TEST_END = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def make_folds(unique_train_dates: pd.DatetimeIndex, n_folds: int = 5,
               embargo: int = EMBARGO_TRADING_DAYS) -> list[Fold]:
    """Build n_folds expanding-window folds with embargo.

    Each fold's validation block is ~ (1/n_folds) of the training window.
    Fold k validates on dates [val_start_k, val_end_k] and trains on every
    date < val_start_k minus the embargo gap.
    """
    dates = pd.DatetimeIndex(sorted(unique_train_dates.unique()))
    n = len(dates)
    if n < n_folds * (embargo + 10):
        raise ValueError(f"too few dates ({n}) for {n_folds} folds with embargo {embargo}")
    val_block_size = n // (n_folds + 1)  # leave first block as initial train

    folds: list[Fold] = []
    for k in range(n_folds):
        # Validation indices: [val_start_idx, val_end_idx)
        val_start_idx = (k + 1) * val_block_size
        val_end_idx = val_start_idx + val_block_size if k < n_folds - 1 else n
        # Train indices: [0, val_start_idx - embargo)
        train_end_idx = val_start_idx - embargo
        if train_end_idx <= 0:
            raise ValueError(f"fold {k}: insufficient pre-validation rows after embargo")
        train_start_ts = dates[0]
        train_end_ts = dates[train_end_idx - 1]
        val_start_ts = dates[val_start_idx]
        val_end_ts = dates[val_end_idx - 1]
        folds.append(Fold(
            fold_id=k,
            train_start=train_start_ts,
            train_end=train_end_ts,
            val_start=val_start_ts,
            val_end=val_end_ts,
        ))
    return folds


def filter_to_training_window(features: pd.DataFrame) -> pd.DataFrame:
    """Filter the feature matrix to the locked training window."""
    f = features[(features["date"] >= TRAIN_START) & (features["date"] <= TRAIN_END)]
    return f.reset_index(drop=True)


def filter_to_test_window(features: pd.DataFrame) -> pd.DataFrame:
    f = features[(features["date"] >= TEST_START) & (features["date"] <= TEST_END)]
    return f.reset_index(drop=True)


def filter_to_oos_holdout(features: pd.DataFrame) -> pd.DataFrame:
    f = features[features["date"] >= OOS_START]
    return f.reset_index(drop=True)
