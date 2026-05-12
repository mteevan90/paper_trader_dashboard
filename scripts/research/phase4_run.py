"""Phase 4 runner — Larger Universe v1 portfolio construction + backtest.

Produces contract v1-conformant artifacts at
models/studies/larger_universe_v1/contract_v1/.

Inline execution (~35-50 min). Doesn't background. Surface headline
results progressively (excess CAGR, drawdown, win rate) before the
full report at the gate.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.equities.study.cv import TRAIN_START, TRAIN_END, TEST_START, TEST_END, OOS_START
from src.equities.study.labels import build_labels, LABEL_HORIZON_TRADING_DAYS, EMBARGO_TRADING_DAYS
from src.equities.study.training import (
    _make_elasticnet_pipeline, _prep_enet_X, _prep_xgb_X, _split_features_target,
    NON_FEATURE_COLS, CATEGORICAL_COLS,
)
from src.equities.study.portfolio import PortfolioConstructionParams, rank_top_n_weights
from src.equities.study.backtest import (
    month_end_trading_dates, run_backtest, ew_sp1500_backtest,
    eligible_universe_on, TRANSACTION_COST_PCT,
)

# Paths
FEATURES_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "features.parquet"
UNIVERSE_PATH = ROOT / "docs" / "larger_universe_v1_universe.json"
SECTOR_MAP_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "sector_map.json"
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"
BENCH_PRICE_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "prices"
XGB_PARAMS_PATH = ROOT / "models" / "studies" / "larger_universe_v1" / "xgboost_best_params.json"
ENET_PARAMS_PATH = ROOT / "models" / "studies" / "larger_universe_v1" / "elasticnet_best_params.json"
XGB_STUDY_JSON = ROOT / "models" / "studies" / "larger_universe_v1" / "xgboost_study.json"
ENET_STUDY_JSON = ROOT / "models" / "studies" / "larger_universe_v1" / "elasticnet_study.json"
OUT_DIR = ROOT / "models" / "studies" / "larger_universe_v1" / "contract_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("phase4")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)


def _setup_universe_and_sectors() -> tuple[list[dict], dict[str, dict], pd.Series, pd.Series, dict[str, pd.Timestamp]]:
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

    delisting_dates: dict[str, pd.Timestamp] = {}
    for sym, r in by_sym.items():
        if r["status"] == "removed" and r.get("removed_at"):
            try:
                delisting_dates[sym] = pd.Timestamp(r["removed_at"])
            except Exception:
                pass
    return universe, by_sym, sectors, tiers, delisting_dates


def _load_daily_returns(universe_symbols: list[str]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Load daily close prices for every universe ticker that has a parquet
    in the snapshot. Return (daily_returns_wide_df, trading_dates_index)."""
    closes: dict[str, pd.Series] = {}
    for sym in universe_symbols:
        p = SNAPSHOT_PRICE_DIR / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index)
        closes[sym] = df["close"]
    prices = pd.DataFrame(closes).sort_index()
    returns = prices.pct_change()
    return returns, prices.index


def _train_xgb_final(merged: pd.DataFrame, best_params: dict) -> xgb.XGBRegressor:
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42,
        "verbosity": 0,
        **best_params,
    }
    X, y = _split_features_target(merged)
    X = _prep_xgb_X(X)
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def _train_enet_final(merged: pd.DataFrame, best_params: dict):
    """Return (pipeline, sector_levels) for ENet."""
    X, y = _split_features_target(merged)
    X_prep, sector_levels = _prep_enet_X(X)
    pipe = _make_elasticnet_pipeline(best_params["alpha"], best_params["l1_ratio"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_prep, y)
    return pipe, sector_levels


def _score_for_date(model_name: str, model, features_df: pd.DataFrame,
                      date_ts: pd.Timestamp, sector_levels=None) -> pd.Series:
    """Predict scores for every ticker at date_ts."""
    rows = features_df[features_df["date"] == date_ts]
    if rows.empty:
        return pd.Series(dtype=float)
    X, _ = _split_features_target(rows.assign(target=np.nan))
    if model_name == "xgboost":
        X = _prep_xgb_X(X)
        preds = model.predict(X)
    elif model_name == "elasticnet":
        X_prep, _ = _prep_enet_X(X, sector_levels=sector_levels)
        # Pipeline's preprocessing expects the same column set seen at fit
        preds = model.predict(X_prep)
    else:
        raise ValueError(f"unknown model: {model_name}")
    return pd.Series(preds, index=rows["ticker"].values)


def _compute_shap_or_fallback(xgb_model: xgb.XGBRegressor,
                                sample_X: pd.DataFrame,
                                feature_names: list[str],
                                wall_clock_budget_s: float = 600.0) -> tuple[pd.DataFrame, str]:
    """Return (importance_df_for_xgb, method_used). Tries tree-SHAP via
    XGBoost's native pred_contribs path; falls back to gain-based if too slow."""
    t0 = time.time()
    try:
        # XGBoost SHAP via pred_contribs
        sample_X_prep = _prep_xgb_X(sample_X)
        booster = xgb_model.get_booster()
        # DMatrix with categorical support
        dm = xgb.DMatrix(sample_X_prep, enable_categorical=True)
        shap_values = booster.predict(dm, pred_contribs=True)
        elapsed = time.time() - t0
        if elapsed > wall_clock_budget_s:
            logger.warning("SHAP exceeded budget (%.1fs); falling back to gain.", elapsed)
            return _gain_importance(xgb_model, feature_names), "gain_fallback"
        # Last column is bias; drop it. The rest are per-feature SHAP contributions.
        shap_no_bias = shap_values[:, :-1]
        mean_abs = np.abs(shap_no_bias).mean(axis=0)
        importance = pd.DataFrame({
            "model": "xgboost",
            "feature": feature_names,
            "importance": mean_abs,
        })
        importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
        importance["rank"] = range(1, len(importance) + 1)
        importance["importance_type"] = "shap_mean_abs"
        logger.info("SHAP computed in %.1fs over %d rows", elapsed, len(sample_X))
        return importance, "shap_tree"
    except Exception as e:
        logger.warning("SHAP failed: %s. Falling back to gain.", e)
        return _gain_importance(xgb_model, feature_names), "gain_fallback"


def _gain_importance(xgb_model: xgb.XGBRegressor, feature_names: list[str]) -> pd.DataFrame:
    imp = xgb_model.feature_importances_
    df = pd.DataFrame({
        "model": "xgboost",
        "feature": feature_names,
        "importance": imp,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["importance_type"] = "gain"
    return df


def _enet_importance(enet_pipeline, feature_names_after_prep: list[str]) -> pd.DataFrame:
    """Absolute coefficient magnitude after StandardScaler. Feature names
    include the one-hot sector columns."""
    coef = enet_pipeline.named_steps["model"].coef_
    df = pd.DataFrame({
        "model": "elasticnet",
        "feature": feature_names_after_prep[:len(coef)],  # safety
        "importance": np.abs(coef),
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["importance_type"] = "abs_coef"
    return df


def _build_trial_log(xgb_json: dict, enet_json: dict) -> pd.DataFrame:
    """Convert Phase 3 study JSONs to contract trial_log.parquet schema."""
    rows = []
    for source, name in ((xgb_json, "xgboost"), (enet_json, "elasticnet")):
        for t in source.get("trials", []):
            row = {
                "tuning_study": name,
                "trial_number": t.get("number"),
                "state": t.get("state", "UNKNOWN"),
                "value": t.get("value") if t.get("value") is not None else float("nan"),
                "duration_s": t.get("duration_s") if t.get("duration_s") is not None else float("nan"),
            }
            for pname, pval in (t.get("params") or {}).items():
                row[f"param_{pname}"] = pval
            rows.append(row)
    return pd.DataFrame(rows)


def _build_benchmark_df(symbol: str, label: str, start: pd.Timestamp,
                         daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Load a benchmark's price parquet and build a long-format NAV row set
    aligned to the daily_returns index from `start` onward."""
    p = BENCH_PRICE_DIR / f"{symbol}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"benchmark price file missing: {p}")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    close = df["close"].sort_index()
    aligned = close.reindex(daily_returns.index).ffill()
    # NAV starts at 1.0 on the first date >= start
    mask = aligned.index >= start
    aligned = aligned[mask]
    if aligned.empty:
        return pd.DataFrame(columns=["date", "benchmark", "nav"])
    nav = aligned / aligned.iloc[0]
    out = pd.DataFrame({"date": nav.index, "benchmark": label, "nav": nav.values})
    return out


def _summarize(portfolio_df: pd.DataFrame, bench_df: pd.DataFrame,
                model_name: str, period_start: pd.Timestamp,
                period_end: pd.Timestamp, period_label: str) -> dict:
    """Compute headline metrics for one model over one period."""
    port = portfolio_df[(portfolio_df["model"] == model_name)
                          & (portfolio_df["date"] >= period_start)
                          & (portfolio_df["date"] <= period_end)].copy()
    if port.empty:
        return {"period": period_label, "model": model_name, "n_days": 0}

    # Reset NAV to 1.0 at the period start for clean per-period CAGR
    port["nav_period"] = port["nav"] / port["nav"].iloc[0]
    spy = bench_df[(bench_df["benchmark"] == "SPY")
                    & (bench_df["date"] >= period_start)
                    & (bench_df["date"] <= period_end)].copy()
    spy["nav_period"] = spy["nav"] / spy["nav"].iloc[0] if not spy.empty else 1.0

    n_days = len(port)
    if n_days < 2:
        return {"period": period_label, "model": model_name, "n_days": n_days}

    total_return = float(port["nav_period"].iloc[-1] - 1.0)
    years = n_days / 252.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    spy_total = float(spy["nav_period"].iloc[-1] - 1.0) if not spy.empty else 0.0
    spy_cagr = (1 + spy_total) ** (1 / years) - 1 if years > 0 else 0.0
    excess_cagr = cagr - spy_cagr
    # Max drawdown
    rolling_max = port["nav_period"].cummax()
    drawdown = port["nav_period"] / rolling_max - 1
    max_dd = float(drawdown.min())
    spy_dd = float((spy["nav_period"] / spy["nav_period"].cummax() - 1).min()) if not spy.empty else 0.0

    return {
        "period": period_label,
        "model": model_name,
        "n_days": n_days,
        "total_return": total_return,
        "cagr": float(cagr),
        "spy_cagr": float(spy_cagr),
        "excess_cagr": float(excess_cagr),
        "max_drawdown": max_dd,
        "spy_max_drawdown": spy_dd,
    }


def main() -> int:
    logger.info("=== Phase 4 — Larger Universe v1 ===")

    # ---- Load inputs ----
    logger.info("loading universe + sectors...")
    universe_records, by_sym, sectors, tiers, delisting_dates = _setup_universe_and_sectors()
    full_universe = sorted(by_sym.keys())
    logger.info("  universe: %d unique symbols", len(full_universe))

    logger.info("loading daily returns from snapshot prices...")
    t0 = time.time()
    daily_returns, trading_dates = _load_daily_returns(full_universe)
    logger.info("  daily_returns: %s in %.1fs", daily_returns.shape, time.time() - t0)

    logger.info("loading features...")
    feat = pd.read_parquet(FEATURES_PATH)
    logger.info("  features: %s", feat.shape)

    logger.info("building labels (horizon=%d)...", LABEL_HORIZON_TRADING_DAYS)
    labels = build_labels(full_universe, horizon=LABEL_HORIZON_TRADING_DAYS)
    logger.info("  labels: %s", labels.shape)

    # Train slice
    train_feat = feat[(feat["date"] >= TRAIN_START) & (feat["date"] <= TRAIN_END)]
    train_labels = labels[(labels["date"] >= TRAIN_START) & (labels["date"] <= TRAIN_END)]
    train_merged = train_feat.merge(train_labels, on=["date", "ticker"], how="left")
    train_merged = train_merged[train_merged["target"].notnull()].reset_index(drop=True)
    logger.info("  training rows: %s", train_merged.shape)

    # ---- Train final models ----
    xgb_best = json.loads(XGB_PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    enet_best = json.loads(ENET_PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]

    logger.info("training final XGBoost on full train window...")
    t0 = time.time()
    xgb_model = _train_xgb_final(train_merged, xgb_best)
    logger.info("  XGBoost done in %.1fs", time.time() - t0)

    logger.info("training final ElasticNet on full train window...")
    t0 = time.time()
    enet_pipe, sector_levels = _train_enet_final(train_merged, enet_best)
    logger.info("  ElasticNet done in %.1fs", time.time() - t0)

    # ---- Rebalance schedule ----
    backtest_start = pd.Timestamp(TEST_START)
    backtest_end = pd.Timestamp(trading_dates.max())
    logger.info("backtest window: %s -> %s", backtest_start.date(), backtest_end.date())
    rebalance_dates = month_end_trading_dates(trading_dates, backtest_start, backtest_end)
    logger.info("  %d monthly rebalance dates", len(rebalance_dates))

    # ---- Compute scores cache for each model + each rebalance date ----
    # Filter feature matrix to backtest window
    bt_feat = feat[(feat["date"] >= backtest_start) & (feat["date"] <= backtest_end)]
    logger.info("backtest-window features: %s", bt_feat.shape)

    # XGBoost batch scoring at each rebalance date is the same as
    # running predict per-date. We do it per-date inside the backtest
    # loop via score_fn to keep the engine generic.
    def make_xgb_score_fn():
        def fn(date_ts: pd.Timestamp) -> pd.Series:
            return _score_for_date("xgboost", xgb_model, bt_feat, date_ts)
        return fn

    def make_enet_score_fn():
        def fn(date_ts: pd.Timestamp) -> pd.Series:
            return _score_for_date("elasticnet", enet_pipe, bt_feat, date_ts,
                                    sector_levels=sector_levels)
        return fn

    pc_params = PortfolioConstructionParams(method="rank_top_n", n=30,
                                              individual_cap=0.075, sector_cap=0.30)

    # ---- XGBoost backtest ----
    logger.info("running XGBoost backtest...")
    t0 = time.time()
    xgb_result = run_backtest(
        model_name="xgboost",
        score_fn=make_xgb_score_fn(),
        rebalance_dates=rebalance_dates,
        daily_returns=daily_returns,
        delisting_dates=delisting_dates,
        sectors=sectors,
        tiers=tiers,
        pc_params=pc_params,
        universe_records=universe_records,
    )
    logger.info("  XGBoost backtest done in %.1fs; final NAV=%.4f",
                time.time() - t0, xgb_result.portfolio["nav"].iloc[-1])

    # ---- ElasticNet backtest ----
    logger.info("running ElasticNet backtest...")
    t0 = time.time()
    enet_result = run_backtest(
        model_name="elasticnet",
        score_fn=make_enet_score_fn(),
        rebalance_dates=rebalance_dates,
        daily_returns=daily_returns,
        delisting_dates=delisting_dates,
        sectors=sectors,
        tiers=tiers,
        pc_params=pc_params,
        universe_records=universe_records,
    )
    logger.info("  ElasticNet backtest done in %.1fs; final NAV=%.4f",
                time.time() - t0, enet_result.portfolio["nav"].iloc[-1])

    # ---- Benchmarks ----
    logger.info("building benchmarks...")
    t0 = time.time()
    bench_frames = []
    for sym, label in (("SPY", "SPY"), ("RSP", "RSP"), ("IWM", "IWM")):
        bench_frames.append(_build_benchmark_df(sym, label, backtest_start, daily_returns))
    # EW-SP1500
    ew = ew_sp1500_backtest(rebalance_dates, daily_returns, delisting_dates,
                              universe_records)
    bench_frames.append(ew)
    benchmarks = pd.concat(bench_frames, ignore_index=True)
    benchmarks["date"] = pd.to_datetime(benchmarks["date"]).astype("datetime64[ns]")
    logger.info("  benchmarks built in %.1fs; %d rows", time.time() - t0, len(benchmarks))

    # ---- Headline metrics surfaced progressively ----
    portfolio_combined = pd.concat([xgb_result.portfolio, enet_result.portfolio],
                                     ignore_index=True)
    portfolio_combined["date"] = pd.to_datetime(portfolio_combined["date"])

    test_metrics = []
    oos_metrics = []
    for model_name in ("xgboost", "elasticnet"):
        m_test = _summarize(portfolio_combined, benchmarks, model_name,
                              backtest_start, pd.Timestamp(TEST_END), "test")
        m_oos = _summarize(portfolio_combined, benchmarks, model_name,
                             pd.Timestamp(OOS_START), backtest_end, "oos")
        test_metrics.append(m_test)
        oos_metrics.append(m_oos)

    logger.info("=== HEADLINE METRICS ===")
    for m in test_metrics + oos_metrics:
        if m.get("n_days", 0) > 0:
            logger.info(
                "  %s / %s (%dd): total_ret=%+.1f%%  CAGR=%+.2f%%  "
                "excess_vs_SPY=%+.2fpp  MaxDD=%+.1f%% (SPY MaxDD=%+.1f%%)",
                m["model"], m["period"], m["n_days"],
                m["total_return"] * 100, m["cagr"] * 100,
                m["excess_cagr"] * 100,
                m["max_drawdown"] * 100, m["spy_max_drawdown"] * 100,
            )

    # ---- Attach target_realized to scores ----
    label_lookup = labels.set_index(["date", "ticker"])["target"]
    for r in (xgb_result, enet_result):
        if not r.scores.empty:
            keys = list(zip(r.scores["date"], r.scores["ticker"]))
            r.scores["target_realized"] = [label_lookup.get((d, t), float("nan"))
                                              for d, t in keys]

    # ---- Combine model outputs ----
    portfolio_out = pd.concat([xgb_result.portfolio, enet_result.portfolio],
                                ignore_index=True)
    holdings_out = pd.concat([xgb_result.holdings, enet_result.holdings],
                               ignore_index=True)
    trades_out = pd.concat([xgb_result.trades, enet_result.trades],
                             ignore_index=True)
    scores_out = pd.concat([xgb_result.scores, enet_result.scores],
                             ignore_index=True)

    # ---- SHAP feature importance for XGBoost (sample 10K) ----
    logger.info("computing SHAP feature importance (10K sample)...")
    feature_cols = [c for c in train_merged.columns
                     if c not in NON_FEATURE_COLS]
    # Use a sample of the test-window features
    test_feat = bt_feat[(bt_feat["date"] >= backtest_start)
                          & (bt_feat["date"] <= pd.Timestamp(TEST_END))]
    if len(test_feat) > 10000:
        sample = test_feat.sample(n=10000, random_state=42)
    else:
        sample = test_feat
    sample_X = sample[feature_cols]
    shap_importance_df, shap_method = _compute_shap_or_fallback(
        xgb_model, sample_X, feature_cols)

    # ElasticNet importance — coefficient magnitude post-scaling
    # Get the column names after _prep_enet_X
    sample_for_names = train_merged.head(100)[feature_cols + ["target"]]
    X_for_names, _ = _split_features_target(sample_for_names)
    X_prep, _ = _prep_enet_X(X_for_names, sector_levels=sector_levels)
    enet_feature_names = list(X_prep.columns)
    # Apply imputer's indicator names (add_indicator=True appends them)
    enet_pipe_imputer = enet_pipe.named_steps["impute"]
    n_extra = enet_pipe.named_steps["model"].coef_.shape[0] - len(enet_feature_names)
    if n_extra > 0:
        # Imputer's missingness-indicator columns
        try:
            missing_cols = [enet_feature_names[i] + "_missing"
                            for i in enet_pipe_imputer.indicator_.features_]
            enet_feature_names = enet_feature_names + missing_cols
        except Exception:
            enet_feature_names = enet_feature_names + [f"missing_{i}" for i in range(n_extra)]
    enet_importance_df = _enet_importance(enet_pipe, enet_feature_names)
    importance_out = pd.concat([shap_importance_df, enet_importance_df],
                                  ignore_index=True)

    # ---- Trial log ----
    logger.info("building trial_log.parquet from Phase 3 study JSONs...")
    xgb_study_json = json.loads(XGB_STUDY_JSON.read_text(encoding="utf-8"))
    enet_study_json = json.loads(ENET_STUDY_JSON.read_text(encoding="utf-8"))
    trial_log = _build_trial_log(xgb_study_json, enet_study_json)

    # ---- meta.json ----
    notes = []
    if shap_method == "gain_fallback":
        notes.append("SHAP fell back to gain-based feature importance due to "
                     "wall-clock or runtime issues. See logs for detail.")
    # Pull metrics
    xgb_test = next((m for m in test_metrics if m["model"] == "xgboost"
                      and m.get("n_days", 0) > 0), {})
    meta = {
        "schema_version": "v1",
        "study_name": "larger_universe_v1",
        "display_name": "Larger Universe v1",
        "description": (
            "XGBoost monthly cross-sectional alpha on SP1500-plus-delisted "
            "universe with 21-day forward-return label. ElasticNet sanity check."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_doc": "docs/studies/larger_universe_v1/spec.md",
        "phase4_spec_doc": "docs/studies/larger_universe_v1/phase4_spec.md",
        "family": "ml_cross_sectional",
        "models": [
            {"name": "xgboost", "role": "primary",
             "params_path": "../xgboost_best_params.json"},
            {"name": "elasticnet", "role": "sanity_check",
             "params_path": "../elasticnet_best_params.json"},
        ],
        "universe": {
            "snapshot": "larger_universe_v1_20260511",
            "size_total": 2122,
            "size_priced": 1963,
        },
        "windows": {
            "train_start": str(TRAIN_START.date()),
            "train_end":   str(TRAIN_END.date()),
            "test_start":  str(TEST_START.date()),
            "test_end":    str(TEST_END.date()),
            "oos_start":   str(OOS_START.date()),
            "oos_end":     str(backtest_end.date()),
        },
        "rebalance": {
            "cadence": "monthly",
            "day": "last_trading_day_of_month",
            "execution": "close_to_close_next_trading_day",
            "threshold_pp": None,
        },
        "label": {
            "horizon_trading_days": LABEL_HORIZON_TRADING_DAYS,
            "definition": "close[t+21] / close[t] - 1",
        },
        "constraints": {
            "max_position_weight": 0.075,
            "max_sector_concentration": 0.30,
            "investment_level_range": [0.95, 1.00],
            "long_only": True,
        },
        "portfolio_construction": pc_params.to_dict(),
        "fee_model": {
            "transaction_cost_pct": TRANSACTION_COST_PCT,
            "applies": "per_trade_leg",
        },
        "benchmarks": [
            {"name": "SPY", "type": "etf",
             "source": "SPDR S&P 500 ETF Trust", "total_return": True},
            {"name": "RSP", "type": "etf",
             "source": "Invesco S&P 500 Equal Weight ETF", "total_return": True},
            {"name": "IWM", "type": "etf",
             "source": "iShares Russell 2000 ETF", "total_return": True},
            {"name": "EW-SP1500", "type": "constructed",
             "methodology": ("monthly equal-weight rebalance across SP1500 "
                              "active-on-date members, total return; "
                              "survivorship-mitigated per universe.json"),
             "total_return": True},
        ],
        "objective": {
            "training_cv": "mean_cross_sectional_spearman_ic",
            "headline": "excess_cagr_vs_spy",
        },
        "feature_importance_method": shap_method,
        "promoted": False,
        "phases": {
            "phase_3_complete": "2026-05-12T03:41Z",
            "phase_4_complete": datetime.now(timezone.utc).isoformat(),
            "phase_5_complete": None,
        },
        "summary_metrics": {
            "cv_mean_ic": json.loads(XGB_PARAMS_PATH.read_text(encoding="utf-8"))[
                "best_value_mean_ic"],
            "test": {m["model"]: {k: m[k] for k in m if k not in ("model", "period")}
                       for m in test_metrics if m.get("n_days", 0) > 0},
            "oos": {m["model"]: {k: m[k] for k in m if k not in ("model", "period")}
                      for m in oos_metrics if m.get("n_days", 0) > 0},
        },
        "notes": notes,
    }

    # ---- Write all artifacts ----
    logger.info("writing contract_v1 artifacts to %s ...", OUT_DIR)
    portfolio_out["date"] = pd.to_datetime(portfolio_out["date"]).astype("datetime64[ns]")
    holdings_out["date"] = pd.to_datetime(holdings_out["date"]).astype("datetime64[ns]")
    trades_out["date"] = pd.to_datetime(trades_out["date"]).astype("datetime64[ns]")
    scores_out["date"] = pd.to_datetime(scores_out["date"]).astype("datetime64[ns]")
    benchmarks["date"] = pd.to_datetime(benchmarks["date"]).astype("datetime64[ns]")

    portfolio_out.to_parquet(OUT_DIR / "portfolio.parquet")
    benchmarks.to_parquet(OUT_DIR / "benchmarks.parquet")
    holdings_out.to_parquet(OUT_DIR / "holdings.parquet")
    trades_out.to_parquet(OUT_DIR / "trades.parquet")
    scores_out.to_parquet(OUT_DIR / "scores.parquet")
    trial_log.to_parquet(OUT_DIR / "trial_log.parquet")
    importance_out.to_parquet(OUT_DIR / "feature_importance.parquet")
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                          encoding="utf-8")

    logger.info("DONE.")
    logger.info("  portfolio.parquet:        %d rows", len(portfolio_out))
    logger.info("  benchmarks.parquet:       %d rows", len(benchmarks))
    logger.info("  holdings.parquet:         %d rows", len(holdings_out))
    logger.info("  trades.parquet:           %d rows", len(trades_out))
    logger.info("  scores.parquet:           %d rows", len(scores_out))
    logger.info("  trial_log.parquet:        %d rows", len(trial_log))
    logger.info("  feature_importance.parquet: %d rows", len(importance_out))
    logger.info("  meta.json: written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
