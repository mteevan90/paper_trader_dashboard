import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

from features import add_target

FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "bb_width",
    "sma_10", "sma_50", "sma_200", "volume_change",
    "dist_52w_high", "atr_14", "obv", "roc_5", "roc_20",
    "momentum_12_1",
    "sector_momentum",
    "vix_level", "vix_20d_avg", "vix_regime", "vix_change",
    "rs_spy", "rs_spy_50d",
    "ad_line", "ad_20d_avg", "ad_divergence",
]
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "xgb_model.json")
MODEL_META_PATH = os.path.join(MODEL_DIR, "xgb_model.meta.json")


def train_model(df: pd.DataFrame, split_date: str | None = None,
                *, eval_cutoff: str | None = None,
                universe_label: str = "unknown") -> xgb.XGBClassifier:
    """Train an XGBoost classifier on featured OHLCV data.

    ``split_date`` is required to prevent lookahead bias: training must
    always use a strict time-based split. If None is passed, it defaults
    to two years before the end of the data.

    ``eval_cutoff`` (the latest date the model is exposed to, even via
    eval_set) and ``universe_label`` are recorded in the sidecar
    ``xgb_model.meta.json`` written next to the model file. The
    backtest's lookahead-bias guard reads this sidecar to decide whether
    to warn instead of using the file-mtime heuristic.
    """
    if split_date is None:
        end_date = df.index.max()
        split_date = (end_date - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        print(f"  No split_date provided — defaulting to {split_date} "
              f"(2 years before end of data).")

    # Snapshot ticker count before groupby/apply — pandas 2.x drops the
    # grouping column from the result, so reading df["ticker"] afterwards
    # would silently fall through to n_tickers=1.
    n_tickers_input = (int(df["ticker"].nunique())
                       if "ticker" in df.columns else 1)
    if "ticker" in df.columns:
        df = df.groupby("ticker", group_keys=False).apply(add_target)
    else:
        df = add_target(df)

    X = df[FEATURE_COLS]
    y = df["target"]

    mask = df.index < split_date
    X_train, X_test = X[mask], X[~mask]
    y_train, y_test = y[mask], y[~mask]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:logistic",
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds_raw = model.predict(X_test)
    preds = (preds_raw > 0.5).astype(int)
    y_binary = (y_test > 0.5).astype(int)
    print(f"  Accuracy:  {accuracy_score(y_binary, preds):.4f}")
    print(f"  Precision: {precision_score(y_binary, preds):.4f}")
    print(f"  Recall:    {recall_score(y_binary, preds):.4f}")

    # Feature importance chart
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(FEATURE_COLS, importances)
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    os.makedirs(MODEL_DIR, exist_ok=True)
    fig.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Feature importance chart saved to models/feature_importance.png")

    # Save model
    model.save_model(MODEL_PATH)
    print(f"  Model saved to {MODEL_PATH}")

    # Sidecar: training provenance for the lookahead-bias guard and for
    # debugging months from now. Atomic .tmp + replace so a crash mid-
    # write can't corrupt the canonical sidecar.
    eval_cutoff_str = (eval_cutoff if eval_cutoff is not None
                       else str(df.index.max().date()))
    meta = {
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "train_cutoff":    split_date,
        "eval_cutoff":     eval_cutoff_str,
        "n_train_rows":    int(mask.sum()),
        "n_eval_rows":     int((~mask).sum()),
        "n_tickers":       n_tickers_input,
        "feature_cols":    list(FEATURE_COLS),
        "universe_label":  universe_label,
        "model_params": {
            "n_estimators":     200,
            "max_depth":        4,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "objective":        "reg:logistic",
            "random_state":     42,
        },
    }
    tmp = MODEL_META_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp, MODEL_META_PATH)
    print(f"  Sidecar metadata saved to {MODEL_META_PATH}")

    return model


def walk_forward_validate(df: pd.DataFrame, n_splits: int = 5) -> list[dict]:
    """Walk-forward cross-validation with an expanding training window.

    Divides the data into n_splits sequential time folds. For each fold, trains
    a fresh XGBoost model on all data strictly before the fold and evaluates
    on the fold. Reports accuracy / precision / recall / Sharpe per fold plus
    mean and std across folds to show whether performance is consistent or
    driven by one lucky period.

    Sharpe per fold is computed on the 5-day forward returns of rows the model
    predicts above 0.5, annualized with sqrt(50) (≈50 non-overlapping 5-day
    windows/year).
    """
    df = df.sort_index().copy()

    # Attach target + raw 5-day forward return (needed for Sharpe).
    def _prep(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        fwd = g["Close"].shift(-5) / g["Close"] - 1
        g = add_target(g)
        g["_fwd_ret_5d"] = fwd.reindex(g.index)
        return g

    if "ticker" in df.columns:
        prepped = df.groupby("ticker", group_keys=False).apply(_prep)
    else:
        prepped = _prep(df)
    prepped = prepped.dropna(subset=["target", "_fwd_ret_5d"])

    unique_dates = prepped.index.unique().sort_values()
    if len(unique_dates) < n_splits + 1:
        print(f"  Not enough unique dates ({len(unique_dates)}) for {n_splits} folds.")
        return []

    edges = np.linspace(0, len(unique_dates), n_splits + 1, dtype=int)
    rows = []

    for k in range(1, n_splits + 1):
        test_start = unique_dates[edges[k - 1]]
        test_end_idx = min(edges[k] - 1, len(unique_dates) - 1)
        test_end = unique_dates[test_end_idx]

        train = prepped[prepped.index < test_start]
        test  = prepped[(prepped.index >= test_start) & (prepped.index <= test_end)]

        if train.empty or test.empty:
            continue

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:logistic", eval_metric="rmse", random_state=42,
        )
        model.fit(train[FEATURE_COLS], train["target"], verbose=False)

        preds_raw = model.predict(test[FEATURE_COLS])
        preds_bin = (preds_raw > 0.5).astype(int)
        y_bin     = (test["target"] > 0.5).astype(int)

        acc  = accuracy_score(y_bin, preds_bin)
        prec = precision_score(y_bin, preds_bin, zero_division=0)
        rec  = recall_score(y_bin, preds_bin, zero_division=0)

        selected = test["_fwd_ret_5d"][preds_raw > 0.5]
        if len(selected) > 1 and selected.std() > 0:
            sharpe = selected.mean() / selected.std() * np.sqrt(50)
        else:
            sharpe = 0.0

        rows.append({
            "fold":      k,
            "train_end": (test_start - pd.Timedelta(days=1)).date(),
            "test_end":  test_end.date(),
            "n_train":   len(train),
            "n_test":    len(test),
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "sharpe":    sharpe,
        })

    # Print summary table
    print(f"\nWalk-forward validation (n_splits = {n_splits}):")
    header = f"  {'Fold':>4}  {'Train End':>11}  {'Test End':>11}  {'N Train':>8}  {'N Test':>7}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'Sharpe':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {r['fold']:>4}  {str(r['train_end']):>11}  {str(r['test_end']):>11}  "
              f"{r['n_train']:>8}  {r['n_test']:>7}  {r['accuracy']:>6.3f}  "
              f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['sharpe']:>7.2f}")

    if rows:
        accs  = [r["accuracy"]  for r in rows]
        precs = [r["precision"] for r in rows]
        recs  = [r["recall"]    for r in rows]
        shs   = [r["sharpe"]    for r in rows]
        print("  " + "-" * (len(header) - 2))
        print(f"  {'mean':>4}  {'':>11}  {'':>11}  {'':>8}  {'':>7}  "
              f"{np.mean(accs):>6.3f}  {np.mean(precs):>6.3f}  {np.mean(recs):>6.3f}  {np.mean(shs):>7.2f}")
        print(f"  {'std':>4}  {'':>11}  {'':>11}  {'':>8}  {'':>7}  "
              f"{np.std(accs):>6.3f}  {np.std(precs):>6.3f}  {np.std(recs):>6.3f}  {np.std(shs):>7.2f}")

    return rows


def load_model() -> xgb.XGBRegressor:
    """Load and return the saved XGBoost model."""
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def should_retrain(max_age_days: int = 90) -> bool:
    """Return True if the saved model is missing or older than max_age_days.

    Used by main.py at startup to warn the user when the model is stale under
    a walk-forward retraining schedule. Does not retrain automatically.
    """
    if not os.path.exists(MODEL_PATH):
        return True
    age_seconds = pd.Timestamp.now().timestamp() - os.path.getmtime(MODEL_PATH)
    return age_seconds > max_age_days * 86400


if __name__ == "__main__":
    from feature_cache import build_feature_matrix
    from fetch_data import UNIVERSE_TICKERS

    # Locked training architecture (segment 19): combined NASDAQ-100 +
    # S&P 500 universe, train on 2018-01-01 -> 2022-12-31, internal eval
    # on 2023, model NEVER sees 2024+ validation data. The backtest's
    # lookahead-bias guard reads the sidecar emitted below to verify.
    FEATURE_START  = "2018-01-01"
    EVAL_CUTOFF    = "2023-12-31"   # latest date model is exposed to
    TRAIN_CUTOFF   = "2023-01-01"   # train rows have date < this
    UNIVERSE_LABEL = "NASDAQ_100_PLUS_SP500"

    print(f"Loading {UNIVERSE_LABEL} features ({FEATURE_START} -> "
          f"{EVAL_CUTOFF}) via feature cache...")
    fm = build_feature_matrix(list(UNIVERSE_TICKERS),
                              FEATURE_START, EVAL_CUTOFF)
    cutoff_ts = pd.Timestamp(EVAL_CUTOFF)
    fm = {t: df.loc[df.index <= cutoff_ts]
          for t, df in fm.items()}
    fm = {t: df for t, df in fm.items() if not df.empty}

    frames = []
    for ticker, df in fm.items():
        df = df.copy()
        df["ticker"] = ticker
        frames.append(df)
    combined = pd.concat(frames).sort_index()
    print(f"Combined dataset: {combined.shape[0]} rows, "
          f"{combined.shape[1]} cols, {len(frames)} tickers")
    print(f"Date range: {combined.index.min().date()} -> "
          f"{combined.index.max().date()}")

    print(f"\nTraining XGBoost (train_cutoff={TRAIN_CUTOFF}, "
          f"eval_cutoff={EVAL_CUTOFF})...")
    model = train_model(combined, split_date=TRAIN_CUTOFF,
                        eval_cutoff=EVAL_CUTOFF,
                        universe_label=UNIVERSE_LABEL)

    print("\nReloading saved model...")
    loaded = load_model()
    n_trees = loaded.get_booster().num_boosted_rounds()
    print(f"  Model loaded — {loaded.n_features_in_} features, "
          f"{n_trees} boosted rounds")
