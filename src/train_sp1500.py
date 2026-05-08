"""train_sp1500.py — retrain the XGBoost model on the S&P 1500 universe.

Run AFTER fetch_sp1500.py has populated the live caches. Builds a feature
matrix over the FULL sp1500 universe, applies the per-(ticker, date) $25M
ADV liquidity filter so the model learns from realistic candidates, trains
with the same hyperparameters as the production model, and saves to
xgb_model_sp1500.json (NEXT to the existing xgb_model.json — does NOT
overwrite the production model).

Hyperparameters mirror src/model.py:train_model exactly (n_estimators=200,
max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
objective=reg:logistic, random_state=42). Training window matches the
locked baseline period: train rows < 2023-01-01, eval on 2023, no
exposure to 2024+ validation data.

Outputs:
  - models/xgb_model_sp1500.json
  - models/xgb_model_sp1500.meta.json
  - models/xgb_model_sp1500.importance.json
  - docs/sp1500_model_retraining_report.txt
        (old vs new metrics + feature importance comparison)

Usage (PowerShell):
    venv\\Scripts\\python.exe src\\train_sp1500.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Repo-relative imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, mean_squared_error)

from features import add_target
from feature_cache import build_feature_matrix
from fetch_data import (SP1500_TICKERS, get_stock_data_cached)
from model import FEATURE_COLS, MODEL_PATH, MODEL_META_PATH, MODEL_DIR


REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
PRICE_CACHE_DIR = os.path.join(REPO_ROOT, "models", "price_cache")

# Locked hyperparameters — must mirror model.train_model exactly so the
# only thing changing in the comparison is the universe + ADV filter.
_HPARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:logistic", eval_metric="rmse", random_state=42,
)

# Locked training window — same as model.py:__main__ ("segment 19").
FEATURE_START = "2018-01-01"
TRAIN_CUTOFF  = "2023-01-01"   # train rows have date < this
EVAL_CUTOFF   = "2023-12-31"   # internal eval window end
UNIVERSE_LABEL = "SP1500"

# $25M ADV threshold over a trailing 30 trading days (matches
# backtest._LIQUIDITY_LOOKBACK_TRADING_DAYS).
LIQUIDITY_THRESHOLD_USD = 25_000_000
LIQUIDITY_LOOKBACK = 30

# Output paths (alongside, NOT replacing, the existing model)
NEW_MODEL_PATH       = os.path.join(MODEL_DIR, "xgb_model_sp1500.json")
NEW_MODEL_META_PATH  = os.path.join(MODEL_DIR, "xgb_model_sp1500.meta.json")
NEW_IMPORTANCE_PATH  = os.path.join(MODEL_DIR,
                                    "xgb_model_sp1500.importance.json")
REPORT_PATH = os.path.join(DOCS_DIR, "sp1500_model_retraining_report.txt")


def _build_liquidity_mask(price_data: dict[str, pd.DataFrame],
                          tickers: list[str]) -> dict[str, pd.Series]:
    """For each ticker, return a Series indexed by trading date with True
    where trailing-30d avg dollar volume >= $25M. Uses pandas rolling so
    it's fast even at 1500 tickers x ~1500 days."""
    out: dict[str, pd.Series] = {}
    for tkr in tickers:
        df = price_data.get(tkr)
        if df is None or df.empty:
            continue
        if "Close" not in df.columns or "Volume" not in df.columns:
            continue
        dollar_vol = df["Close"] * df["Volume"]
        rolling = dollar_vol.rolling(LIQUIDITY_LOOKBACK, min_periods=5).mean()
        out[tkr] = rolling >= LIQUIDITY_THRESHOLD_USD
    return out


def _apply_liquidity_filter(
    fm: dict[str, pd.DataFrame],
    liquidity_mask: dict[str, pd.Series],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Drop rows where the ticker failed the $25M ADV filter on that date."""
    filtered: dict[str, pd.DataFrame] = {}
    stats = {"rows_in": 0, "rows_out": 0, "tickers_in": len(fm),
             "tickers_out": 0}
    for tkr, df in fm.items():
        stats["rows_in"] += len(df)
        mask = liquidity_mask.get(tkr)
        if mask is None:
            # No liquidity series — keep nothing (we can't validate).
            continue
        # Reindex mask onto df's index and keep only True rows.
        aligned = mask.reindex(df.index).fillna(False)
        kept = df[aligned]
        if not kept.empty:
            filtered[tkr] = kept
            stats["rows_out"] += len(kept)
            stats["tickers_out"] += 1
    return filtered, stats


def _train_one(combined: pd.DataFrame, label: str) -> dict:
    """Train one XGBoost model on (ticker, date)-keyed combined frame.

    Returns dict with model object, metrics, feature importances, and
    training-set sizes — caller decides what to persist.
    """
    if "ticker" in combined.columns:
        combined = combined.groupby("ticker", group_keys=False).apply(add_target)
    else:
        combined = add_target(combined)

    X = combined[FEATURE_COLS]
    y = combined["target"]

    mask = combined.index < TRAIN_CUTOFF
    X_train, X_eval = X[mask], X[~mask]
    y_train, y_eval = y[mask], y[~mask]

    model = xgb.XGBRegressor(**_HPARAMS)
    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)

    preds_raw = model.predict(X_eval)
    preds_bin = (preds_raw > 0.5).astype(int)
    y_bin = (y_eval > 0.5).astype(int)

    rmse = float(np.sqrt(mean_squared_error(y_eval, preds_raw)))
    try:
        auc = float(roc_auc_score(y_bin, preds_raw))
    except ValueError:
        auc = float("nan")  # one-class eval set
    acc = float(accuracy_score(y_bin, preds_bin))
    prec = float(precision_score(y_bin, preds_bin, zero_division=0))
    rec = float(recall_score(y_bin, preds_bin, zero_division=0))

    # Top-N hit rate: of the top 10% predicted, how many were actual hits?
    n_top = max(1, int(0.10 * len(preds_raw)))
    if n_top > 0 and len(preds_raw) > 0:
        top_idx = np.argsort(preds_raw)[-n_top:]
        top_hits = float(y_bin.iloc[top_idx].mean()) if hasattr(y_bin, "iloc") \
            else float(np.asarray(y_bin)[top_idx].mean())
    else:
        top_hits = float("nan")

    importances = dict(zip(FEATURE_COLS,
                           [float(v) for v in model.feature_importances_]))
    importances_sorted = dict(sorted(importances.items(),
                                     key=lambda kv: kv[1], reverse=True))

    print(f"  [{label}] N_train={len(X_train):,}  N_eval={len(X_eval):,}")
    print(f"  [{label}] ROC-AUC={auc:.4f}  RMSE={rmse:.4f}  "
          f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
          f"Top10pct_hit={top_hits:.4f}")

    return {
        "model": model,
        "metrics": {
            "roc_auc": auc, "rmse": rmse, "accuracy": acc,
            "precision": prec, "recall": rec,
            "top_10pct_hit_rate": top_hits,
            "n_train": int(len(X_train)),
            "n_eval": int(len(X_eval)),
        },
        "importances": importances_sorted,
    }


def _serialize_model(result: dict, n_tickers: int) -> None:
    """Write xgb_model_sp1500.json + sidecar + importance JSON."""
    os.makedirs(os.path.dirname(NEW_MODEL_PATH), exist_ok=True)
    result["model"].save_model(NEW_MODEL_PATH)
    print(f"  Saved model to {NEW_MODEL_PATH}")

    meta = {
        "trained_at":     datetime.now(timezone.utc).isoformat(),
        "train_cutoff":   TRAIN_CUTOFF,
        "eval_cutoff":    EVAL_CUTOFF,
        "n_train_rows":   result["metrics"]["n_train"],
        "n_eval_rows":    result["metrics"]["n_eval"],
        "n_tickers":      n_tickers,
        "feature_cols":   list(FEATURE_COLS),
        "universe_label": UNIVERSE_LABEL,
        "model_params":   _HPARAMS,
        "liquidity_filter": {
            "applied": True,
            "threshold_usd": LIQUIDITY_THRESHOLD_USD,
            "lookback_days": LIQUIDITY_LOOKBACK,
        },
        "metrics": result["metrics"],
    }
    tmp = NEW_MODEL_META_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp, NEW_MODEL_META_PATH)
    print(f"  Saved meta to  {NEW_MODEL_META_PATH}")

    with open(NEW_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(result["importances"], f, indent=2)
    print(f"  Saved feature importance to {NEW_IMPORTANCE_PATH}")


def _evaluate_old_model_on_legacy_features() -> dict | None:
    """Load the legacy xgb_model.json and re-evaluate it on its own training
    feature matrix to produce the comparison numbers. Returns None if the
    legacy model can't be loaded — the report just notes that."""
    if not os.path.exists(MODEL_PATH):
        print(f"  [WARN] No legacy model at {MODEL_PATH}; skipping comparison.")
        return None
    if not os.path.exists(MODEL_META_PATH):
        print(f"  [WARN] No legacy sidecar at {MODEL_META_PATH}; "
              f"can't determine legacy universe.")
        return None
    with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
        legacy_meta = json.load(f)

    # Re-build the legacy feature matrix from the (smaller) legacy universe.
    from fetch_data import UNIVERSE_TICKERS as LEGACY_UNIVERSE
    print(f"  Building legacy feature matrix ({len(LEGACY_UNIVERSE)} tickers)...")
    fm = build_feature_matrix(list(LEGACY_UNIVERSE),
                              FEATURE_START, EVAL_CUTOFF)
    cutoff_ts = pd.Timestamp(EVAL_CUTOFF)
    fm = {t: df.loc[df.index <= cutoff_ts] for t, df in fm.items()}
    fm = {t: df for t, df in fm.items() if not df.empty}
    frames = []
    for ticker, df in fm.items():
        df = df.copy()
        df["ticker"] = ticker
        frames.append(df)
    combined = pd.concat(frames).sort_index()
    print(f"  Legacy combined: {combined.shape[0]:,} rows, "
          f"{len(frames)} tickers")

    if "ticker" in combined.columns:
        combined = combined.groupby("ticker", group_keys=False).apply(add_target)
    X = combined[FEATURE_COLS]
    y = combined["target"]
    mask = combined.index < TRAIN_CUTOFF
    X_eval, y_eval = X[~mask], y[~mask]

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    preds_raw = model.predict(X_eval)
    preds_bin = (preds_raw > 0.5).astype(int)
    y_bin = (y_eval > 0.5).astype(int)
    rmse = float(np.sqrt(mean_squared_error(y_eval, preds_raw)))
    try:
        auc = float(roc_auc_score(y_bin, preds_raw))
    except ValueError:
        auc = float("nan")
    acc = float(accuracy_score(y_bin, preds_bin))
    prec = float(precision_score(y_bin, preds_bin, zero_division=0))
    rec = float(recall_score(y_bin, preds_bin, zero_division=0))
    n_top = max(1, int(0.10 * len(preds_raw)))
    top_idx = np.argsort(preds_raw)[-n_top:]
    top_hits = float(y_bin.iloc[top_idx].mean()) if hasattr(y_bin, "iloc") \
        else float(np.asarray(y_bin)[top_idx].mean())

    importances = dict(zip(FEATURE_COLS,
                           [float(v) for v in model.feature_importances_]))
    importances_sorted = dict(sorted(importances.items(),
                                     key=lambda kv: kv[1], reverse=True))

    print(f"  [LEGACY] ROC-AUC={auc:.4f}  RMSE={rmse:.4f}  "
          f"Acc={acc:.4f}  Top10pct_hit={top_hits:.4f}")
    return {
        "metrics": {
            "roc_auc": auc, "rmse": rmse, "accuracy": acc,
            "precision": prec, "recall": rec,
            "top_10pct_hit_rate": top_hits,
            "n_eval": int(len(X_eval)),
        },
        "importances": importances_sorted,
        "legacy_meta": legacy_meta,
    }


def _write_report(new_result: dict, old_result: dict | None,
                  n_tickers: int, n_rows_out: int, n_rows_in: int) -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# SP1500 Model Retraining Report\n")
        f.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write(f"## Training configuration\n")
        f.write(f"  Universe:           SP1500 ({n_tickers} tickers retained "
                f"after fetch + ADV filter)\n")
        f.write(f"  Window:             {FEATURE_START} -> {EVAL_CUTOFF}\n")
        f.write(f"  Train cutoff:       {TRAIN_CUTOFF} (train < this date)\n")
        f.write(f"  Liquidity filter:   ${LIQUIDITY_THRESHOLD_USD:,} ADV / "
                f"{LIQUIDITY_LOOKBACK}-day trailing\n")
        f.write(f"  Rows pre-filter:    {n_rows_in:,}\n")
        f.write(f"  Rows post-filter:   {n_rows_out:,}  "
                f"({100.0*n_rows_out/max(1, n_rows_in):.1f}% of pre-filter)\n")
        f.write(f"  Hyperparameters:    {_HPARAMS}\n\n")

        f.write(f"## NEW model (sp1500) metrics — internal eval window "
                f"{TRAIN_CUTOFF} -> {EVAL_CUTOFF}\n")
        for k, v in new_result["metrics"].items():
            f.write(f"  {k:<22}  {v}\n")
        f.write(f"\n")

        if old_result is not None:
            f.write(f"## OLD model (legacy 490) metrics — same eval window\n")
            for k, v in old_result["metrics"].items():
                f.write(f"  {k:<22}  {v}\n")
            f.write(f"\n")
            f.write(f"## Delta (NEW - OLD)\n")
            for k in ("roc_auc", "rmse", "accuracy", "precision", "recall",
                      "top_10pct_hit_rate"):
                a = new_result["metrics"].get(k)
                b = old_result["metrics"].get(k)
                if a is None or b is None:
                    continue
                f.write(f"  {k:<22}  {a - b:+.4f}\n")
            f.write(f"\n")

        f.write(f"## Top-10 feature importances — NEW (sp1500)\n")
        for i, (k, v) in enumerate(list(new_result["importances"].items())[:10], 1):
            f.write(f"  {i:>2}. {k:<22}  {v:.4f}\n")
        f.write(f"\n")
        if old_result is not None:
            f.write(f"## Top-10 feature importances — OLD (legacy 490)\n")
            for i, (k, v) in enumerate(list(old_result["importances"].items())[:10], 1):
                f.write(f"  {i:>2}. {k:<22}  {v:.4f}\n")

    print(f"  Wrote retraining report to {REPORT_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip re-evaluating the legacy model (faster).")
    args = parser.parse_args()

    print(f"=== Retraining XGBoost on SP1500 ===\n")

    # 1. Build SP1500 feature matrix (uses feature_cache; will rebuild if
    #    the cache is for a smaller universe).
    print(f"[1/5] Building SP1500 feature matrix "
          f"({len(SP1500_TICKERS)} tickers, {FEATURE_START} -> {EVAL_CUTOFF})...")
    fm = build_feature_matrix(list(SP1500_TICKERS),
                              FEATURE_START, EVAL_CUTOFF)
    cutoff_ts = pd.Timestamp(EVAL_CUTOFF)
    fm = {t: df.loc[df.index <= cutoff_ts] for t, df in fm.items()}
    fm = {t: df for t, df in fm.items() if not df.empty}
    n_rows_pre_filter = sum(len(df) for df in fm.values())
    print(f"      {len(fm)} tickers, {n_rows_pre_filter:,} pre-filter rows\n")

    # 2. Load price/volume for the same universe so we can compute ADV.
    print(f"[2/5] Loading price/volume for ADV filter (cached parquets)...")
    price_data = get_stock_data_cached(list(fm.keys()),
                                       FEATURE_START, EVAL_CUTOFF,
                                       cache_dir=PRICE_CACHE_DIR)
    print(f"      {len(price_data)} tickers loaded\n")

    # 3. Apply per-(ticker, date) $25M ADV filter to feature matrix.
    print(f"[3/5] Applying ${LIQUIDITY_THRESHOLD_USD:,} ADV liquidity filter...")
    liquidity_mask = _build_liquidity_mask(price_data, list(fm.keys()))
    fm_filtered, stats = _apply_liquidity_filter(fm, liquidity_mask)
    print(f"      Rows: {stats['rows_in']:,} -> {stats['rows_out']:,}  "
          f"(kept {100.0*stats['rows_out']/max(1,stats['rows_in']):.1f}%)")
    print(f"      Tickers: {stats['tickers_in']} -> "
          f"{stats['tickers_out']}\n")

    # Concatenate into the (ticker, date)-indexed long form train_model expects.
    frames = []
    for tkr, df in fm_filtered.items():
        df = df.copy()
        df["ticker"] = tkr
        frames.append(df)
    if not frames:
        print("[ABORT] No rows survived the liquidity filter; nothing to train on.")
        return 1
    combined = pd.concat(frames).sort_index()
    print(f"      Combined frame: {combined.shape[0]:,} rows, "
          f"{len(frames)} tickers, "
          f"{combined.index.min().date()} -> {combined.index.max().date()}\n")

    # 4. Train.
    print(f"[4/5] Training XGBoost (locked hyperparameters)...")
    new_result = _train_one(combined, "NEW_SP1500")
    _serialize_model(new_result, n_tickers=len(frames))
    print()

    # 5. Re-evaluate legacy model on legacy features for an apples-to-apples
    #    comparison + write the diff report.
    old_result = None
    if not args.skip_comparison:
        print(f"[5/5] Re-evaluating legacy model on legacy 490-ticker matrix...")
        old_result = _evaluate_old_model_on_legacy_features()
        print()

    _write_report(new_result, old_result,
                  n_tickers=len(frames),
                  n_rows_in=stats["rows_in"],
                  n_rows_out=stats["rows_out"])
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
