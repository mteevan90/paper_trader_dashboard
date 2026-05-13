"""Phase 4 runner — Larger Universe v2 portfolio-construction variant backtests.

Architecture: shared-model optimization. v1's locked XGBoost hyperparameters
and training data are reused — no Phase 3 retuning. The model is trained
once, scores are computed at every rebalance date once, and the cached
scores are reused across all variant backtests. Per-variant artifacts are
written to `models/studies/larger_universe_v2/<variant_subdir>/contract_v1/`.

CLI:
    python scripts/research/phase4_run_v2.py --variants baseline
    python scripts/research/phase4_run_v2.py --variants baseline,b1_vol_target
    python scripts/research/phase4_run_v2.py --variants all

Variants and their output subdirs (matching docs/studies/larger_universe_v2/
spec.md Output structure section):

    baseline                    → baseline/
    b1_vol_target               → b1_vol_target/
    b2_conviction_weighted      → b2_conviction_weighted/
    b3_dynamic_topn             → b3_dynamic_topn/
    b4_concentration_penalties  → b4_concentration_penalties/
    b5_defensive_sleeves        → b5_defensive_sleeves/
    b6_smaller_caps             → b6_smaller_caps/

Gate 3 emits a minimal contract_v1 tree per variant: portfolio.parquet,
holdings.parquet, trades.parquet, scores.parquet, benchmarks.parquet, and
meta.json. Phase 5 artifacts (decile_returns, per_ticker_attribution,
ic_decomposition, rolling_win_rate, concentration_summary, walk_forward,
feature_importance) are produced by separate runners — walk_forward by
phase5_walk_forward_v2.py and the rest at Gate 4.

The helpers in this script intentionally mirror v1's phase4_run.py
verbatim (_setup_universe_and_sectors, _load_daily_returns, _train_xgb_final,
_score_for_date). If v1's runner evolves, this one must be kept in sync
for v2-baseline reproducibility to hold.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.equities.study.cv import TRAIN_START, TRAIN_END, TEST_START, TEST_END, OOS_START
from src.equities.study.labels import build_labels, LABEL_HORIZON_TRADING_DAYS
from src.equities.study.training import (
    _prep_xgb_X, _split_features_target, NON_FEATURE_COLS,
)
from src.equities.study.backtest import (
    month_end_trading_dates, run_backtest, ew_sp1500_backtest,
    TRANSACTION_COST_PCT,
)
from src.equities.portfolio_construction import get_variant_by_name

# ---- Paths ----
FEATURES_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "features.parquet"
UNIVERSE_PATH = ROOT / "docs" / "larger_universe_v1_universe.json"
SECTOR_MAP_PATH = ROOT / "models" / "features" / "larger_universe_v1" / "sector_map.json"
SNAPSHOT_PRICE_DIR = ROOT / "models" / "snapshots" / "equities" / "larger_universe_v1_20260511" / "price_cache"
BENCH_PRICE_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "prices"
XGB_PARAMS_PATH = ROOT / "models" / "studies" / "larger_universe_v1" / "xgboost_best_params.json"
V1_SCORES_PATH = ROOT / "models" / "studies" / "larger_universe_v1" / "contract_v1" / "scores.parquet"
V2_OUT_DIR = ROOT / "models" / "studies" / "larger_universe_v2"

ALL_VARIANTS = [
    "baseline",
    "b1_vol_target",
    "b2_conviction_weighted",
    "b3_dynamic_topn",
    "b4_concentration_penalties",
    "b5_defensive_sleeves",
    "b6_smaller_caps",
]
VARIANTS_NEEDING_SHY = {"b5_defensive_sleeves"}
VARIANTS_NEEDING_SPY_HISTORY = {"b5_defensive_sleeves"}

logger = logging.getLogger("phase4_v2")


# ============================================================================
# Helpers — mirror scripts/research/phase4_run.py verbatim for reproducibility
# ============================================================================

def _setup_universe_and_sectors() -> tuple[
    list[dict], dict[str, dict], pd.Series, pd.Series, dict[str, pd.Timestamp]
]:
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


def _load_daily_returns(
    universe_symbols: list[str],
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
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


def _score_for_date(
    model_name: str, model, features_df: pd.DataFrame, date_ts: pd.Timestamp,
) -> pd.Series:
    rows = features_df[features_df["date"] == date_ts]
    if rows.empty:
        return pd.Series(dtype=float)
    X, _ = _split_features_target(rows.assign(target=np.nan))
    if model_name == "xgboost":
        X = _prep_xgb_X(X)
        preds = model.predict(X)
    else:
        raise ValueError(f"unknown model: {model_name}")
    return pd.Series(preds, index=rows["ticker"].values)


def _build_benchmark_df(
    symbol: str, label: str, start: pd.Timestamp, daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    p = BENCH_PRICE_DIR / f"{symbol}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"benchmark price file missing: {p}")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    close = df["close"].sort_index()
    aligned = close.reindex(daily_returns.index).ffill()
    mask = aligned.index >= start
    aligned = aligned[mask]
    if aligned.empty:
        return pd.DataFrame(columns=["date", "benchmark", "nav"])
    nav = aligned / aligned.iloc[0]
    return pd.DataFrame({"date": nav.index, "benchmark": label, "nav": nav.values})


def _summarize(
    portfolio_df: pd.DataFrame, bench_df: pd.DataFrame, model_name: str,
    period_start: pd.Timestamp, period_end: pd.Timestamp, period_label: str,
) -> dict:
    port = portfolio_df[(portfolio_df["model"] == model_name)
                          & (portfolio_df["date"] >= period_start)
                          & (portfolio_df["date"] <= period_end)].copy()
    if port.empty:
        return {"period": period_label, "model": model_name, "n_days": 0}

    port["nav_period"] = port["nav"] / port["nav"].iloc[0]
    spy = bench_df[(bench_df["benchmark"] == "SPY")
                    & (bench_df["date"] >= period_start)
                    & (bench_df["date"] <= period_end)].copy()
    if not spy.empty:
        spy["nav_period"] = spy["nav"] / spy["nav"].iloc[0]

    n_days = len(port)
    if n_days < 2:
        return {"period": period_label, "model": model_name, "n_days": n_days}

    total_return = float(port["nav_period"].iloc[-1] - 1.0)
    years = n_days / 252.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    spy_total = float(spy["nav_period"].iloc[-1] - 1.0) if not spy.empty else 0.0
    spy_cagr = (1 + spy_total) ** (1 / years) - 1 if years > 0 else 0.0
    excess_cagr = cagr - spy_cagr
    rolling_max = port["nav_period"].cummax()
    drawdown = port["nav_period"] / rolling_max - 1
    max_dd = float(drawdown.min())
    spy_dd = float((spy["nav_period"] / spy["nav_period"].cummax() - 1).min()) if not spy.empty else 0.0

    # v2 informational addition: Sharpe using v1's walk-forward formula
    daily_ret = pd.Series(port["nav_period"].values).pct_change().dropna()
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() > 0 else 0.0
    )

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
        "sharpe_informational": sharpe,
    }


# ============================================================================
# v2-specific machinery
# ============================================================================

def _load_spy_history() -> pd.DataFrame:
    """Load SPY OHLCV for B5 regime detection. Returns DataFrame indexed by
    datetime with the 'close' column required by the engine's spy_history
    parameter (other OHLCV columns retained, ignored downstream)."""
    p = BENCH_PRICE_DIR / "SPY.parquet"
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_shy_data() -> tuple[pd.Series, pd.Series]:
    """Load SHY daily returns + close prices for B5's defensive sleeve.

    Returns:
        (shy_returns_series, shy_close_series), both indexed by datetime.
    """
    p = BENCH_PRICE_DIR / "SHY.parquet"
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    shy_returns = df["close"].pct_change()
    shy_close = df["close"]
    return shy_returns, shy_close


def _augment_for_shy(
    daily_returns: pd.DataFrame, sectors: pd.Series, tiers: pd.Series,
    delisting_dates: dict, shy_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Add SHY as a tradeable column in daily_returns + sector/tier entries.

    B5's variant returns weights that may include SHY; the engine drifts and
    rebalances SHY like any other ticker. delisting_dates intentionally
    omits SHY (engine treats absent entries as never-delisting)."""
    dr = daily_returns.copy()
    if "SHY" not in dr.columns:
        dr["SHY"] = shy_returns.reindex(dr.index)
    secs = sectors.copy()
    if "SHY" not in secs.index:
        secs["SHY"] = "treasury_etf"
    tiers_aug = tiers.copy()
    if "SHY" not in tiers_aug.index:
        tiers_aug["SHY"] = "etf"
    # delisting_dates: leave SHY out so engine treats it as never-delisting
    return dr, secs, tiers_aug, delisting_dates


def _verify_scores_parity(
    score_cache: dict[pd.Timestamp, pd.Series], universe_records: list[dict],
    v1_scores_path: Path,
) -> dict:
    """Compare v2 cached scores against v1's scores.parquet at the (date,
    ticker) level. Returns a dict summarizing parity.

    Discipline: deviation > 1e-6 (sub-0.0001%) on any (date, ticker) pair is
    a finding to surface before continuing. Bit-identical scores expected
    because:
      - Same XGBoost hyperparameters (loaded from same file)
      - Same training data + same feature pipeline + same random_state=42
      - Same model.predict() call on the same eligible-on-date features

    Any deviation indicates an integration-path or environment drift that
    would invalidate baseline reproducibility downstream.
    """
    v1_scores = pd.read_parquet(v1_scores_path)
    v1_xgb = v1_scores[v1_scores["model"] == "xgboost"].copy()
    v1_xgb["date"] = pd.to_datetime(v1_xgb["date"])

    # Build long-format v2 scores filtered to v1's eligible-on-date subset
    v2_rows = []
    from src.equities.study.backtest import eligible_universe_on
    for d_ts, scores_today in score_cache.items():
        eligible = eligible_universe_on(d_ts, universe_records)
        filtered = scores_today[scores_today.index.isin(eligible)]
        for ticker, score in filtered.items():
            v2_rows.append({"date": d_ts, "ticker": ticker, "v2_score": float(score)})
    v2_df = pd.DataFrame(v2_rows)
    v2_df["date"] = pd.to_datetime(v2_df["date"])

    merged = v1_xgb.merge(
        v2_df, on=["date", "ticker"], how="outer", indicator=True
    )
    n_v1_only = (merged["_merge"] == "left_only").sum()
    n_v2_only = (merged["_merge"] == "right_only").sum()
    both = merged[merged["_merge"] == "both"].copy()
    both["abs_diff"] = (both["score"] - both["v2_score"]).abs()
    max_abs_diff = float(both["abs_diff"].max()) if not both.empty else 0.0
    mean_abs_diff = float(both["abs_diff"].mean()) if not both.empty else 0.0
    n_within_1e6 = int((both["abs_diff"] < 1e-6).sum())

    return {
        "n_pairs_compared": int(len(both)),
        "n_v1_only": int(n_v1_only),
        "n_v2_only": int(n_v2_only),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "n_within_1e6": n_within_1e6,
        "fraction_within_1e6": (
            float(n_within_1e6) / len(both) if not both.empty else 0.0
        ),
        "passed": bool(
            max_abs_diff < 1e-6 and int(n_v1_only) == 0 and int(n_v2_only) == 0
        ),
    }


def _variant_out_dir(variant_name: str) -> Path:
    return V2_OUT_DIR / variant_name / "contract_v1"


def _compute_warmup_state(
    xgb_model, feat: pd.DataFrame, trading_dates: pd.DatetimeIndex,
    daily_returns: pd.DataFrame, sectors: pd.Series, tiers: pd.Series,
    delisting_dates: dict, universe_records: list[dict],
    train_start: pd.Timestamp, train_end: pd.Timestamp,
) -> dict:
    """Compute training-period warmup state needed by some variants.

    Returns a dict with:
      - 'training_dispersion_dist': list of top-decile score std at each
        training-period monthly rebalance. Consumed by B3.
      - 'training_tail_vol': last-63-day annualized realized vol of a
        BASELINE-construction backtest over the training period (frozen
        at study config time). Consumed by B1.

    Per Gate 1 design: B1 uses "frozen training-tail vol" and B3 uses the
    frozen training-period dispersion distribution. Both are computed
    once at study time using the same locked XGBoost model that will be
    used in the test window. Not peek-ahead: the model is used to
    characterize the training period, not to make forward-looking
    decisions during it.
    """
    train_rb_dates = month_end_trading_dates(trading_dates, train_start, train_end)
    if not train_rb_dates:
        raise RuntimeError("no training-period rebalance dates")

    bt_train_feat = feat[(feat["date"] >= train_start) & (feat["date"] <= train_end)]

    # Score every training rebalance date with the trained model
    train_score_cache: dict[pd.Timestamp, pd.Series] = {}
    for d in train_rb_dates:
        d_ts = pd.Timestamp(d)
        train_score_cache[d_ts] = _score_for_date(
            "xgboost", xgb_model, bt_train_feat, d_ts,
        )

    # B3: top-decile dispersion at each rebalance
    dispersions: list[float] = []
    for d_ts, scores in train_score_cache.items():
        valid = scores.dropna()
        if valid.empty:
            continue
        decile_n = max(1, int(round(0.1 * len(valid))))
        top = valid.nlargest(decile_n)
        if len(top) > 1:
            dispersions.append(float(top.std()))

    # B1: run a baseline backtest over the training window to derive
    # last-63-day portfolio vol. Use a fresh BaselineVariant so caps logic
    # is identical to v1's path.
    from src.equities.portfolio_construction import BaselineVariant

    def train_score_fn(d_ts: pd.Timestamp) -> pd.Series:
        return train_score_cache.get(pd.Timestamp(d_ts), pd.Series(dtype=float))

    baseline_for_warmup = BaselineVariant()
    train_result = run_backtest(
        model_name="xgboost",
        score_fn=train_score_fn,
        rebalance_dates=train_rb_dates,
        daily_returns=daily_returns,
        delisting_dates=delisting_dates,
        sectors=sectors, tiers=tiers,
        pc_params=None,
        universe_records=universe_records,
        construction_variant=baseline_for_warmup,
    )
    train_port = train_result.portfolio
    train_nav = train_port["nav"].values
    daily_ret_train = pd.Series(train_nav).pct_change().dropna()
    last63 = daily_ret_train.iloc[-63:] if len(daily_ret_train) >= 63 else daily_ret_train
    training_tail_vol = float(last63.std() * np.sqrt(252)) if len(last63) > 1 else 0.0

    return {
        "training_dispersion_dist": dispersions,
        "training_tail_vol": training_tail_vol,
        "n_training_rebalances": len(train_rb_dates),
    }


def _build_variant_with_warmup(variant_name: str, warmup: dict | None):
    """Instantiate a variant, threading warmup state into B1/B3 as needed."""
    if variant_name == "b1_vol_target":
        if warmup is None or warmup.get("training_tail_vol") is None:
            raise RuntimeError("B1 requires warmup state; run _compute_warmup_state first")
        from src.equities.portfolio_construction import VolTargetVariant
        return VolTargetVariant(training_tail_vol=warmup["training_tail_vol"])
    if variant_name == "b3_dynamic_topn":
        if warmup is None or not warmup.get("training_dispersion_dist"):
            raise RuntimeError("B3 requires warmup state; run _compute_warmup_state first")
        from src.equities.portfolio_construction import DynamicTopNVariant
        return DynamicTopNVariant(
            training_dispersion_dist=warmup["training_dispersion_dist"],
        )
    return get_variant_by_name(variant_name)


def _run_one_variant(
    variant_name: str,
    score_cache: dict[pd.Timestamp, pd.Series],
    rebalance_dates: list[pd.Timestamp],
    daily_returns_base: pd.DataFrame,
    delisting_dates: dict,
    sectors: pd.Series,
    tiers: pd.Series,
    universe_records: list[dict],
    labels: pd.DataFrame,
    benchmarks: pd.DataFrame,
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    spy_history: pd.DataFrame,
    shy_returns: pd.Series | None,
    shy_close: pd.Series | None,
    warmup: dict | None = None,
) -> dict:
    """Build the variant, run its backtest, write its artifacts. Return a
    headline-metrics summary for logging."""
    out_dir = _variant_out_dir(variant_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional SHY augmentation for B5
    if variant_name in VARIANTS_NEEDING_SHY:
        if shy_returns is None or shy_close is None:
            raise RuntimeError(
                f"variant {variant_name} needs SHY data but it was not loaded"
            )
        dr_v, sectors_v, tiers_v, delisting_v = _augment_for_shy(
            daily_returns_base, sectors, tiers, delisting_dates, shy_returns
        )
        shy_prices_for_engine = shy_close
    else:
        dr_v = daily_returns_base
        sectors_v = sectors
        tiers_v = tiers
        delisting_v = delisting_dates
        shy_prices_for_engine = None

    spy_history_for_engine = (
        spy_history if variant_name in VARIANTS_NEEDING_SPY_HISTORY else None
    )

    # Build the variant (threading warmup state for B1 / B3)
    variant = _build_variant_with_warmup(variant_name, warmup)

    # Wrap cached scores in a score_fn the engine expects
    def score_fn(d_ts: pd.Timestamp) -> pd.Series:
        return score_cache.get(pd.Timestamp(d_ts), pd.Series(dtype=float))

    # Run the backtest
    logger.info("  [%s] running backtest...", variant_name)
    t0 = time.time()
    result = run_backtest(
        model_name="xgboost",
        score_fn=score_fn,
        rebalance_dates=rebalance_dates,
        daily_returns=dr_v,
        delisting_dates=delisting_v,
        sectors=sectors_v,
        tiers=tiers_v,
        pc_params=None,
        universe_records=universe_records,
        construction_variant=variant,
        spy_history=spy_history_for_engine,
        shy_prices=shy_prices_for_engine,
    )
    elapsed = time.time() - t0
    logger.info(
        "  [%s] done in %.1fs; final NAV=%.4f",
        variant_name, elapsed, result.portfolio["nav"].iloc[-1],
    )

    # Attach target_realized to scores (mirror v1)
    label_lookup = labels.set_index(["date", "ticker"])["target"]
    if not result.scores.empty:
        keys = list(zip(result.scores["date"], result.scores["ticker"]))
        result.scores["target_realized"] = [
            label_lookup.get((d, t), float("nan")) for d, t in keys
        ]

    # Compute headline metrics (test + oos)
    portfolio_df = result.portfolio.copy()
    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
    test_m = _summarize(
        portfolio_df, benchmarks, "xgboost",
        backtest_start, pd.Timestamp(TEST_END), "test",
    )
    oos_m = _summarize(
        portfolio_df, benchmarks, "xgboost",
        pd.Timestamp(OOS_START), backtest_end, "oos",
    )

    # Build meta.json (variant-specific)
    meta = {
        "schema_version": "v1",
        "study_name": "larger_universe_v2",
        "variant_name": variant_name,
        "display_name": f"Larger Universe v2 — {variant_name}",
        "description": (
            "v2 portfolio-construction variant. Same v1 XGBoost model, "
            "features, universe, and dates; varies portfolio construction "
            "logic only. See docs/studies/larger_universe_v2/spec.md."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_doc": "docs/studies/larger_universe_v2/spec.md",
        "family": "ml_cross_sectional",
        "control_role": "control" if variant_name == "baseline" else "treatment",
        "models": [
            {
                "name": "xgboost", "role": "primary",
                "params_path": "../../../larger_universe_v1/xgboost_best_params.json",
            },
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
        "portfolio_construction": {
            "variant": variant_name,
            "params": variant.params_dict(),
        },
        "fee_model": {
            "transaction_cost_pct": TRANSACTION_COST_PCT,
            "applies": "per_trade_leg",
        },
        "objective": {
            "training_cv": "mean_cross_sectional_spearman_ic",
            "headline": "excess_cagr_vs_spy",
        },
        "promoted": False,
        "phases": {
            "phase_3_complete": "(reused from v1)",
            "phase_4_complete": datetime.now(timezone.utc).isoformat(),
            "phase_5_complete": None,
        },
        # summary_metrics shape: {slice: {model_name: {metric: value, ...}}}
        # — matches the v1 contract schema the dashboard's Overview tab
        # iterates over. v2 has a single model (xgboost) per variant; the
        # nested dict structure preserves model-keying for cross-study
        # consistency.
        "summary_metrics": {
            "test": {
                "xgboost": {
                    k: v for k, v in test_m.items() if k not in ("model", "period")
                }
            } if test_m.get("n_days", 0) > 0 else {},
            "oos": {
                "xgboost": {
                    k: v for k, v in oos_m.items() if k not in ("model", "period")
                }
            } if oos_m.get("n_days", 0) > 0 else {},
        },
        "notes": [],
    }

    # Write artifacts
    result.portfolio["date"] = pd.to_datetime(result.portfolio["date"]).astype("datetime64[ns]")
    result.holdings["date"] = pd.to_datetime(result.holdings["date"]).astype("datetime64[ns]")
    result.trades["date"] = pd.to_datetime(result.trades["date"]).astype("datetime64[ns]")
    result.scores["date"] = pd.to_datetime(result.scores["date"]).astype("datetime64[ns]")
    benchmarks_v = benchmarks.copy()
    benchmarks_v["date"] = pd.to_datetime(benchmarks_v["date"]).astype("datetime64[ns]")

    result.portfolio.to_parquet(out_dir / "portfolio.parquet")
    result.holdings.to_parquet(out_dir / "holdings.parquet")
    result.trades.to_parquet(out_dir / "trades.parquet")
    result.scores.to_parquet(out_dir / "scores.parquet")
    benchmarks_v.to_parquet(out_dir / "benchmarks.parquet")
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8",
    )

    return {"variant": variant_name, "test": test_m, "oos": oos_m, "elapsed_s": elapsed}


def _write_variant_meta_json(variants_run: list[str]) -> None:
    """Study-level variant_meta.json. Lists ALL 7 variants (registry-driven)
    with role + subdir + status. Refreshed on each runner invocation.

    `optional_artifacts` is appended by build_comparison_results_v2.py once
    the comparison artifact lands; phase 4 leaves it as []."""
    V2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v1",
        "study_name": "larger_universe_v2",
        "display_name": "Larger Universe v2",
        "spec_doc": "docs/studies/larger_universe_v2/spec.md",
        "variants": [
            {
                "name": v,
                "subdir": v,
                "role": "control" if v == "baseline" else "treatment",
                "has_phase4_artifacts": (
                    (V2_OUT_DIR / v / "contract_v1" / "portfolio.parquet").exists()
                ),
            }
            for v in ALL_VARIANTS
        ],
        "optional_artifacts": [],
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    (V2_OUT_DIR / "variant_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variants",
        required=True,
        help=(
            "Comma-separated variant names, or 'all'. "
            f"Valid names: {','.join(ALL_VARIANTS)}"
        ),
    )
    p.add_argument(
        "--skip-scores-parity-check",
        action="store_true",
        help=(
            "Skip the upstream scores-equivalence check against v1's "
            "scores.parquet. Use only if v1 artifacts are unavailable."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stdout,
    )

    raw = args.variants.strip()
    if raw == "all":
        variant_names = list(ALL_VARIANTS)
    else:
        variant_names = [v.strip() for v in raw.split(",") if v.strip()]
    for v in variant_names:
        if v not in ALL_VARIANTS:
            raise ValueError(f"Unknown variant: {v!r}. Valid: {ALL_VARIANTS}")

    logger.info("=== Phase 4 v2 — variants: %s ===", variant_names)

    # ---- Shared setup ----
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

    train_feat = feat[(feat["date"] >= TRAIN_START) & (feat["date"] <= TRAIN_END)]
    train_labels = labels[(labels["date"] >= TRAIN_START) & (labels["date"] <= TRAIN_END)]
    train_merged = train_feat.merge(train_labels, on=["date", "ticker"], how="left")
    train_merged = train_merged[train_merged["target"].notnull()].reset_index(drop=True)
    logger.info("  training rows: %s", train_merged.shape)

    # ---- Train XGBoost (shared across variants) ----
    xgb_best = json.loads(XGB_PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    logger.info("training final XGBoost on full train window (locked v1 hyperparams)...")
    t0 = time.time()
    xgb_model = _train_xgb_final(train_merged, xgb_best)
    logger.info("  XGBoost done in %.1fs", time.time() - t0)

    # ---- Rebalance schedule ----
    backtest_start = pd.Timestamp(TEST_START)
    backtest_end = pd.Timestamp(trading_dates.max())
    logger.info("backtest window: %s -> %s", backtest_start.date(), backtest_end.date())
    rebalance_dates = month_end_trading_dates(trading_dates, backtest_start, backtest_end)
    logger.info("  %d monthly rebalance dates", len(rebalance_dates))

    # ---- Pre-cache scores at each rebalance date (shared across variants) ----
    bt_feat = feat[(feat["date"] >= backtest_start) & (feat["date"] <= backtest_end)]
    logger.info("backtest-window features: %s", bt_feat.shape)

    logger.info("pre-caching scores at %d rebalance dates...", len(rebalance_dates))
    t0 = time.time()
    score_cache: dict[pd.Timestamp, pd.Series] = {}
    for d in rebalance_dates:
        d_ts = pd.Timestamp(d)
        score_cache[d_ts] = _score_for_date("xgboost", xgb_model, bt_feat, d_ts)
    logger.info(
        "  scores cached for %d dates in %.1fs (avg %d tickers/date)",
        len(score_cache), time.time() - t0,
        int(np.mean([len(s) for s in score_cache.values()]))
        if score_cache else 0,
    )

    # ---- Scores parity check vs v1 (upstream of CAGR check) ----
    parity_path = V2_OUT_DIR / "_scores_parity_vs_v1.json"
    if not args.skip_scores_parity_check:
        logger.info("verifying scores parity against v1 scores.parquet ...")
        parity = _verify_scores_parity(score_cache, universe_records, V1_SCORES_PATH)
        V2_OUT_DIR.mkdir(parents=True, exist_ok=True)
        parity_path.write_text(json.dumps(parity, indent=2), encoding="utf-8")
        logger.info(
            "  scores parity: %s | pairs=%d v1_only=%d v2_only=%d "
            "max_abs_diff=%.3e mean_abs_diff=%.3e",
            "PASS" if parity["passed"] else "FAIL",
            parity["n_pairs_compared"], parity["n_v1_only"], parity["n_v2_only"],
            parity["max_abs_diff"], parity["mean_abs_diff"],
        )
        if not parity["passed"]:
            logger.warning(
                "Scores parity check FAILED (max_abs_diff=%.3e >= 1e-6). "
                "This is a finding — surfacing rather than blocking, so the "
                "variant runs proceed; baseline reproducibility check "
                "downstream will detect any consequent CAGR deviation. "
                "Report saved to %s.",
                parity["max_abs_diff"], parity_path,
            )
    else:
        logger.info("scores parity check SKIPPED per --skip-scores-parity-check")

    # ---- Load SPY OHLCV + (conditionally) SHY ----
    spy_history = _load_spy_history()
    shy_returns: pd.Series | None = None
    shy_close: pd.Series | None = None
    if any(v in VARIANTS_NEEDING_SHY for v in variant_names):
        logger.info("loading SHY data for B5 defensive sleeves...")
        shy_returns, shy_close = _load_shy_data()

    # ---- Benchmarks (shared across variants) ----
    logger.info("building benchmarks (SPY, RSP, IWM, EW-SP1500)...")
    t0 = time.time()
    bench_frames = []
    for sym, label in (("SPY", "SPY"), ("RSP", "RSP"), ("IWM", "IWM")):
        bench_frames.append(_build_benchmark_df(sym, label, backtest_start, daily_returns))
    ew = ew_sp1500_backtest(rebalance_dates, daily_returns, delisting_dates, universe_records)
    bench_frames.append(ew)
    benchmarks = pd.concat(bench_frames, ignore_index=True)
    benchmarks["date"] = pd.to_datetime(benchmarks["date"]).astype("datetime64[ns]")
    logger.info("  benchmarks built in %.1fs; %d rows", time.time() - t0, len(benchmarks))

    # ---- Compute warmup state for B1 / B3 (if requested) ----
    warmup: dict | None = None
    if any(v in {"b1_vol_target", "b3_dynamic_topn"} for v in variant_names):
        logger.info("computing training-period warmup state (B1 vol + B3 dispersion)...")
        t0 = time.time()
        warmup = _compute_warmup_state(
            xgb_model=xgb_model, feat=feat, trading_dates=trading_dates,
            daily_returns=daily_returns, sectors=sectors, tiers=tiers,
            delisting_dates=delisting_dates, universe_records=universe_records,
            train_start=TRAIN_START, train_end=TRAIN_END,
        )
        logger.info(
            "  warmup state computed in %.1fs: %d training rebalances, "
            "training_tail_vol=%.4f (annualized), n_dispersions=%d",
            time.time() - t0,
            warmup["n_training_rebalances"], warmup["training_tail_vol"],
            len(warmup["training_dispersion_dist"]),
        )

    # ---- Run variants ----
    summaries = []
    for v_name in variant_names:
        s = _run_one_variant(
            variant_name=v_name,
            score_cache=score_cache,
            rebalance_dates=rebalance_dates,
            daily_returns_base=daily_returns,
            delisting_dates=delisting_dates,
            sectors=sectors, tiers=tiers,
            universe_records=universe_records,
            labels=labels,
            benchmarks=benchmarks,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            spy_history=spy_history,
            shy_returns=shy_returns,
            shy_close=shy_close,
            warmup=warmup,
        )
        summaries.append(s)

    # ---- Headline log ----
    logger.info("=== HEADLINE METRICS ===")
    for s in summaries:
        for period_key in ("test", "oos"):
            m = s[period_key]
            if m.get("n_days", 0) > 0:
                logger.info(
                    "  %s / %s (%dd): CAGR=%+.2f%%  excess_vs_SPY=%+.2fpp  "
                    "MaxDD=%+.1f%% (SPY MaxDD=%+.1f%%)  Sharpe=%.4f",
                    s["variant"], m["period"], m["n_days"],
                    m["cagr"] * 100, m["excess_cagr"] * 100,
                    m["max_drawdown"] * 100, m["spy_max_drawdown"] * 100,
                    m.get("sharpe_informational", float("nan")),
                )

    # ---- Refresh study-level variant_meta.json ----
    _write_variant_meta_json(variant_names)
    logger.info("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
