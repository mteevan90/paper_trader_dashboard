"""optuna_runner.py - Optuna optimizer for the paper trader.

Searches the BacktestConfig parameter space against the locked objective
function from objective.py. Trains on the locked train window
[2018-01-01, 2023-12-31] only — the validation window is held out and
never seen here.

Runs:
  python optuna_runner.py --smoke                      # 30 trials, n_jobs=1
  python optuna_runner.py --full                       # 1000 trials, n_jobs=4
  python optuna_runner.py --resume STUDY --trials N    # extend a study
  python optuna_runner.py --report STUDY               # top-10 + best config

Each trial appends one JSON line to ../models/cache/optuna_trials.jsonl
(study_name, trial_number, state, score, config, components, duration).
The Optuna study itself is persisted to a SQLite DB at
../models/cache/optuna_studies.db so studies are resumable.

This module pre-loads the feature matrix, price data, sector map,
fundamentals, earnings, and SPY closes ONCE before the trial loop
starts and passes references through closures. Per-trial work is the
backtest loop + scoring; no I/O reload between trials.
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

# Cap BLAS/OpenMP thread pools BEFORE pandas/numpy/xgboost import. Default
# numpy uses all cores; combined with Optuna's n_jobs that's CPU
# oversubscription on per-row XGBoost predict and pandas .loc lookups,
# which thrashes more than it helps. n_jobs=8 was tried and regressed the
# wall clock (50min vs 35min at n_jobs=4) — we reverted to n_jobs=4 but
# kept these caps in place since they don't hurt at 4 workers and protect
# any future bump.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from backtest import fetch_earnings_dates, fetch_fundamentals, run_backtest
from backtest_config import BacktestConfig
from feature_cache import build_feature_matrix
from fetch_data import (UNIVERSE_TICKERS, build_sector_map,
                        get_stock_data_cached)
from model import load_model
from objective import (compute_objective, compute_objective_components,
                       summarize_backtest)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "cache"))
PRICE_CACHE = os.path.abspath(os.path.join(BASE_DIR, "..", "models",
                                           "price_cache"))
STUDY_DB_PATH    = os.path.join(CACHE_DIR, "optuna_studies.db")
TRIALS_LOG_PATH  = os.path.join(CACHE_DIR, "optuna_trials.jsonl")

_FAILURE_SENTINEL = -1e6

# Train window from the locked architecture decisions. Pulled from
# BacktestConfig() defaults rather than hardcoded so a future config
# revision flows through without missing this file.
_DEFAULTS    = BacktestConfig()
TRAIN_START  = _DEFAULTS.train_start
TRAIN_END    = _DEFAULTS.train_end


# ---------------------------------------------------------------------------
# Shared-data preload
# ---------------------------------------------------------------------------

def _filter_to_window(data: dict[str, pd.DataFrame],
                      end_date: str) -> dict[str, pd.DataFrame]:
    """Return a new dict with each DataFrame sliced to index <= end_date."""
    end_ts = pd.Timestamp(end_date)
    out: dict[str, pd.DataFrame] = {}
    for tkr, df in data.items():
        sliced = df.loc[df.index <= end_ts]
        if not sliced.empty:
            out[tkr] = sliced
    return out


def _load_shared_data(train_start: str, train_end: str) -> dict:
    """Load every input the trial loop needs, ONCE.

    Feature matrix and price data are fetched for the full archive window
    (so a single warm cache serves both training and any future
    held-out validation runs) and then filtered to the training window.
    Returns a dict the trial closure reads from.
    """
    print(f"[OPTUNA] Pre-loading shared data for "
          f"{train_start} -> {train_end}...")

    print("[OPTUNA]   feature matrix...")
    feature_matrix = build_feature_matrix(
        list(UNIVERSE_TICKERS),
        _DEFAULTS.train_start, _DEFAULTS.validate_end,
        price_cache_dir=PRICE_CACHE,
    )
    feature_matrix = _filter_to_window(feature_matrix, train_end)

    print("[OPTUNA]   price data...")
    price_data = get_stock_data_cached(
        list(UNIVERSE_TICKERS),
        _DEFAULTS.train_start, _DEFAULTS.validate_end,
        cache_dir=PRICE_CACHE,
    )
    price_data = _filter_to_window(price_data, train_end)

    print("[OPTUNA]   SPY close...")
    market = get_stock_data_cached(
        ["SPY"], _DEFAULTS.train_start, _DEFAULTS.validate_end,
        cache_dir=PRICE_CACHE,
    )
    spy_close = market["SPY"]["Close"]
    spy_close = spy_close.loc[spy_close.index <= pd.Timestamp(train_end)]

    print("[OPTUNA]   sector map...")
    sector_map = build_sector_map(list(feature_matrix.keys()))

    print("[OPTUNA]   fundamentals + earnings...")
    fund_data  = fetch_fundamentals(list(feature_matrix.keys()))
    earn_dates = fetch_earnings_dates(list(feature_matrix.keys()),
                                      train_start, train_end)

    # Pre-load model once. Passing it into run_backtest also bypasses the
    # lookahead-bias guard (intended: Optuna's training-window evaluation
    # is by design in-sample for the model — the warning would fire on
    # every trial otherwise, despite the model having been correctly cut
    # off at eval_cutoff via the segment-19 sidecar).
    print("[OPTUNA]   model...")
    model = load_model()

    # Pre-warm pandas Index._engine hash tables on every shared frame.
    # pandas builds Index._engine lazily on the first .loc lookup; the
    # construction is not thread-safe, so under n_jobs>1 the first few
    # workers can race on the same Index object.
    #
    # NOTE: This pre-warm is INCOMPLETE. The v2 study ran with this fix
    # and still saw 7.7% trial failures (vs 8.3% without). The actual
    # race lives inside run_backtest's vectorized predict block, where
    # each trial constructs FRESH Index objects:
    #   stacked = pd.concat({tkr: df[FEATURE_COLS] ...})
    #   stacked = stacked[list(FEATURE_COLS)]
    # The MultiIndex of `stacked` and the FEATURE_COLS column index are
    # both built fresh per trial, and their engines build lazily under
    # thread contention. Pre-warming the *shared* feature_matrix/
    # price_data row indexes (below) doesn't help those.
    #
    # TODO(real fix, requires backtest.py changes):
    #   (a) Build one shared stacked DataFrame in _load_shared_data,
    #       eagerly call stacked.index._engine to build its engine, and
    #       pass it through to run_backtest (new kwarg). Per-trial
    #       run_backtest just slices by date instead of re-stacking.
    #       OR
    #   (b) In run_backtest's vectorized block, after pd.concat, force
    #       eager engine construction with `_ = stacked.index._engine;
    #       _ = stacked.columns._engine` before any .loc/[] access.
    # Either lives outside this file's scope; tracked for follow-up.
    print("[OPTUNA]   pre-warming Index engines...")
    for tkr, df in feature_matrix.items():
        if not df.empty:
            _ = df.loc[df.index[0]]
    for tkr, df in price_data.items():
        if not df.empty:
            _ = df.loc[df.index[0]]
    if not spy_close.empty:
        _ = spy_close.loc[spy_close.index[0]]

    n_days = max((len(df) for df in feature_matrix.values()), default=0)
    print(f"[OPTUNA] Shared data ready: {len(feature_matrix)} tickers, "
          f"~{n_days} trading days through {train_end}")

    return {
        "featured_data":  feature_matrix,
        "price_data":     price_data,
        "spy_close":      spy_close,
        "sector_map":     sector_map,
        "fund_data":      fund_data,
        "earnings_dates": earn_dates,
        "model":          model,
        "train_start":    train_start,
        "train_end":      train_end,
    }


# ---------------------------------------------------------------------------
# Search space + trial callable
# ---------------------------------------------------------------------------

def build_search_space(trial: optuna.Trial) -> dict:
    """Suggest a set of BacktestConfig kwargs for this trial.

    macro_threshold_high is structured as low + gap (gap in [0.10, 0.40])
    so the high > low constraint is enforced by construction — no
    rejection sampling, no wasted trials.
    weight_alt is NOT suggested here: BacktestConfig.__post_init__
    derives it as 1.0 - sum_of_others.
    """
    weight_fundamental = trial.suggest_float("weight_fundamental", 0.20, 0.55)
    weight_technical   = trial.suggest_float("weight_technical",   0.10, 0.45)
    weight_model       = trial.suggest_float("weight_model",       0.10, 0.45)

    macro_threshold_low = trial.suggest_float("macro_threshold_low",
                                              0.25, 0.50)
    macro_threshold_gap = trial.suggest_float("macro_threshold_gap",
                                              0.10, 0.40)
    macro_threshold_high = macro_threshold_low + macro_threshold_gap

    atr_multiplier   = trial.suggest_float("atr_multiplier",  1.5, 4.0)
    analyst_weight   = trial.suggest_float("analyst_weight",  0.0, 0.15)
    rebalance_freq   = trial.suggest_int("rebalance_frequency_days", 14, 63)
    position_count   = trial.suggest_int("position_count",     10, 25)

    return {
        "weight_fundamental":       weight_fundamental,
        "weight_technical":         weight_technical,
        "weight_model":             weight_model,
        "macro_threshold_low":      macro_threshold_low,
        "macro_threshold_high":     macro_threshold_high,
        "atr_multiplier":           atr_multiplier,
        "analyst_weight":           analyst_weight,
        "rebalance_frequency_days": rebalance_freq,
        "position_count":           position_count,
    }


def _append_trial_log(jsonl_path: str, log_lock: threading.Lock,
                      record: dict) -> None:
    line = json.dumps(record, default=str) + "\n"
    with log_lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)


def objective_fn(trial: optuna.Trial,
                 *,
                 train_start: str,
                 train_end: str,
                 shared_data: dict,
                 study_name: str,
                 jsonl_path: str,
                 log_lock: threading.Lock) -> float:
    """One Optuna trial: build a config, run a backtest, score it.

    Pruned when the three free weights sum > 1.0 (BacktestConfig raises
    ValueError; we re-raise as TrialPruned so Optuna registers the trial
    as failed without polluting the TPE prior). Any other exception is
    swallowed and the failure sentinel is returned.
    """
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    config_kwargs = build_search_space(trial)
    try:
        config = BacktestConfig(**config_kwargs)
    except ValueError as e:
        # Weight-sum violation is the only validation BacktestConfig
        # raises. Mark the trial pruned and log so the experiment record
        # still contains every attempted point.
        record = {
            "study_name":   study_name,
            "trial_number": trial.number,
            "state":        "PRUNED",
            "score":        None,
            "reason":       str(e),
            "config_kwargs": config_kwargs,
            "started_at":   started_at.isoformat(),
            "duration_seconds": round(time.perf_counter() - t0, 3),
        }
        _append_trial_log(jsonl_path, log_lock, record)
        raise optuna.TrialPruned(str(e))

    try:
        portfolio_df, _trades_df, _scores, _holdings = run_backtest(
            shared_data["featured_data"],
            shared_data["price_data"],
            split_date=train_start,
            fund_data=shared_data["fund_data"],
            sector_map=shared_data["sector_map"],
            earnings_dates=shared_data["earnings_dates"],
            model=shared_data["model"],
            config=config,
            legacy_predict=False,
        )
        summary    = summarize_backtest(portfolio_df, shared_data["spy_close"])
        score      = compute_objective(summary)
        components = compute_objective_components(summary)

        record = {
            "study_name":   study_name,
            "trial_number": trial.number,
            "state":        "COMPLETE",
            "score":        score,
            "config":       config.to_dict(),
            "components":   components,
            "started_at":   started_at.isoformat(),
            "duration_seconds": round(time.perf_counter() - t0, 3),
        }
        _append_trial_log(jsonl_path, log_lock, record)
        return float(score)
    except Exception as e:
        # Anything else (bad data, unexpected math, NaN flow): record and
        # return the failure sentinel so the study keeps running.
        record = {
            "study_name":   study_name,
            "trial_number": trial.number,
            "state":        "FAIL",
            "score":        _FAILURE_SENTINEL,
            "error":        f"{type(e).__name__}: {e}",
            "config":       config.to_dict(),
            "started_at":   started_at.isoformat(),
            "duration_seconds": round(time.perf_counter() - t0, 3),
        }
        _append_trial_log(jsonl_path, log_lock, record)
        return _FAILURE_SENTINEL


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(n_trials: int, n_jobs: int, study_name: str,
              smoke_test: bool = False) -> optuna.Study:
    """Create-or-load a study and run ``n_trials`` more trials on it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    storage_url = f"sqlite:///{STUDY_DB_PATH}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=TPESampler(),
        direction="maximize",
        load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"[OPTUNA] Loaded existing study {study_name!r} with "
              f"{n_existing} trials")
    else:
        print(f"[OPTUNA] Created study {study_name!r}")

    shared_data = _load_shared_data(TRAIN_START, TRAIN_END)

    log_lock = threading.Lock()

    def _trial(t: optuna.Trial) -> float:
        return objective_fn(
            t,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            shared_data=shared_data,
            study_name=study_name,
            jsonl_path=TRIALS_LOG_PATH,
            log_lock=log_lock,
        )

    print(f"[OPTUNA] Running {n_trials} trial(s) with n_jobs={n_jobs}...")
    t_start = time.perf_counter()
    study.optimize(_trial, n_trials=n_trials, n_jobs=n_jobs)
    elapsed = time.perf_counter() - t_start

    n_complete = sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.PRUNED)
    n_fail = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.FAIL)

    print(f"\n[OPTUNA] Done in {elapsed:.1f}s "
          f"({elapsed/max(n_trials,1):.1f}s/trial average over the new "
          f"{n_trials})")
    print(f"[OPTUNA] Total trials in study: {len(study.trials)} "
          f"(complete={n_complete}, pruned={n_pruned}, fail={n_fail})")

    if n_complete:
        best = study.best_trial
        print(f"\n[OPTUNA] Best trial #{best.number}: score={best.value:.6f}")
        print("[OPTUNA] Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
    else:
        print("[OPTUNA] No completed trials yet — nothing to report.")

    return study


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(study_name: str, top_n: int = 10) -> None:
    storage_url = f"sqlite:///{STUDY_DB_PATH}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError:
        print(f"[OPTUNA] Study {study_name!r} not found in {STUDY_DB_PATH}")
        sys.exit(1)

    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n=== Study {study_name} ===")
    print(f"Total trials: {len(study.trials)}  "
          f"(complete={len(complete)}, "
          f"pruned={sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}, "
          f"fail={sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)})")

    if not complete:
        print("No completed trials.")
        return

    ranked = sorted(complete, key=lambda t: t.value or _FAILURE_SENTINEL,
                    reverse=True)
    print(f"\nTop {min(top_n, len(ranked))} trials:")
    print(f"  {'rank':>4} {'trial':>5} {'score':>10}  params")
    print("  " + "-" * 70)
    for i, t in enumerate(ranked[:top_n], 1):
        params_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float)
                               else f"{k}={v}"
                               for k, v in t.params.items())
        print(f"  {i:>4} {t.number:>5} {t.value:>10.6f}  {params_str}")

    print("\nBest trial full breakdown:")
    best = ranked[0]
    print(f"  Trial #{best.number}, score = {best.value:.6f}")
    print("  Suggested params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    if best.user_attrs:
        print("  User attrs:")
        for k, v in best.user_attrs.items():
            print(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Dashboard pre-save
# ---------------------------------------------------------------------------

DASHBOARD_RESULTS_DIR = os.path.join(CACHE_DIR, "dashboard_results")
LOCKED_BEST_STUDY = "optuna_v1_20260504_103429"
LOCKED_BEST_TRIAL = 706


def _save_one_backtest_result(label: str, config: BacktestConfig,
                              shared: dict) -> None:
    """Run one backtest with ``config`` over the validation window and save
    the full portfolio + trades + scores + holdings to
    ``dashboard_results/{label}/``. The dashboard reads this for
    instant-load Positions/Trades/Overview tabs."""
    out_dir = os.path.join(DASHBOARD_RESULTS_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[SAVE] Running backtest for label={label!r} "
          f"(split_date={config.validate_start})...")
    t0 = time.perf_counter()
    portfolio_df, trades_df, scores, holdings = run_backtest(
        shared["featured_data"], shared["price_data"],
        split_date=config.validate_start,
        fund_data=shared["fund_data"],
        sector_map=shared["sector_map"],
        earnings_dates=shared["earnings_dates"],
        model=shared["model"],
        config=config,
    )
    elapsed = time.perf_counter() - t0
    print(f"[SAVE]   backtest done in {elapsed:.1f}s, "
          f"{len(portfolio_df)} days, {len(trades_df)} trades, "
          f"{len(holdings)} open positions")

    # Full frames as parquet (the dashboard's helper functions need the
    # complete portfolio_df / trades_df, not just summary stats).
    portfolio_df.to_parquet(os.path.join(out_dir, "portfolio.parquet"))
    if not trades_df.empty:
        trades_df.to_parquet(os.path.join(out_dir, "trades.parquet"))
    else:
        # Empty parquet with the right columns so dashboard reads cleanly.
        pd.DataFrame(columns=["date","ticker","action","shares","price","fee"]
                     ).to_parquet(os.path.join(out_dir, "trades.parquet"))

    # Per-rebalance composite scores (dict ticker -> {fundamental, technical,
    # model, composite, [analyst_score]}) and final holdings (dict ticker
    # -> {shares, entry_price, stop_price}). JSON since they're small dicts.
    with open(os.path.join(out_dir, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, default=str, indent=2)
    holdings_serial = {
        tkr: {"shares": float(v["shares"]),
              "entry_price": float(v["entry_price"]),
              "stop_price": float(v.get("stop_price", 0.0))}
        for tkr, v in holdings.items()
    }
    with open(os.path.join(out_dir, "holdings.json"), "w", encoding="utf-8") as f:
        json.dump(holdings_serial, f, indent=2)

    meta = {
        "label":       label,
        "saved_at":    datetime.now(timezone.utc).isoformat(),
        "config":      config.to_dict(),
        "split_date":  config.validate_start,
        "n_days":      int(len(portfolio_df)),
        "n_trades":    int(len(trades_df)),
        "n_holdings":  int(len(holdings)),
        "runtime_seconds": round(elapsed, 3),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVE]   wrote {out_dir}")


def _trial_to_config(trial: optuna.trial.FrozenTrial) -> BacktestConfig:
    p = trial.params
    return BacktestConfig(
        weight_fundamental       = p["weight_fundamental"],
        weight_technical         = p["weight_technical"],
        weight_model             = p["weight_model"],
        macro_threshold_low      = p["macro_threshold_low"],
        macro_threshold_high     = p["macro_threshold_low"] + p["macro_threshold_gap"],
        atr_multiplier           = p["atr_multiplier"],
        analyst_weight           = p["analyst_weight"],
        rebalance_frequency_days = p["rebalance_frequency_days"],
        position_count           = p["position_count"],
    )


def _load_full_shared_data() -> dict:
    """Like _load_shared_data but does NOT filter to train_end. Used for
    dashboard saves where the backtest tests the validation window and
    needs feature/price coverage through validate_end."""
    cfg = BacktestConfig()
    print(f"[SAVE] Loading shared data {cfg.train_start} -> {cfg.validate_end}...")

    feature_matrix = build_feature_matrix(
        list(UNIVERSE_TICKERS), cfg.train_start, cfg.validate_end,
        price_cache_dir=PRICE_CACHE,
    )
    price_data = get_stock_data_cached(
        list(UNIVERSE_TICKERS), cfg.train_start, cfg.validate_end,
        cache_dir=PRICE_CACHE,
    )
    market = get_stock_data_cached(
        ["SPY"], cfg.train_start, cfg.validate_end,
        cache_dir=PRICE_CACHE,
    )
    spy_close = market["SPY"]["Close"]
    sector_map = build_sector_map(list(feature_matrix.keys()))
    fund_data  = fetch_fundamentals(list(feature_matrix.keys()))
    earn_dates = fetch_earnings_dates(list(feature_matrix.keys()),
                                      cfg.validate_start, cfg.validate_end)
    model = load_model()

    # Pre-warm Index engines (single-threaded; though save_results doesn't
    # use n_jobs>1, kept for consistency with the trial path).
    for tkr, df in feature_matrix.items():
        if not df.empty:
            _ = df.loc[df.index[0]]
    for tkr, df in price_data.items():
        if not df.empty:
            _ = df.loc[df.index[0]]

    return {
        "featured_data":  feature_matrix,
        "price_data":     price_data,
        "spy_close":      spy_close,
        "sector_map":     sector_map,
        "fund_data":      fund_data,
        "earnings_dates": earn_dates,
        "model":          model,
    }


def save_dashboard_results() -> None:
    """Save the two configs the dashboard primarily displays: default and
    the locked best trial. Each gets its own subdirectory. Re-running
    overwrites — fine, this is a derived artifact."""
    shared = _load_full_shared_data()

    # 1. Default config
    _save_one_backtest_result("default", BacktestConfig(), shared)

    # 2. Locked best trial (#706 of optuna_v1_20260504_103429)
    storage_url = f"sqlite:///{STUDY_DB_PATH}"
    try:
        study = optuna.load_study(study_name=LOCKED_BEST_STUDY,
                                  storage=storage_url)
        trial = study.trials[LOCKED_BEST_TRIAL]
        cfg_best = _trial_to_config(trial)
        label = f"best_{LOCKED_BEST_STUDY}_{LOCKED_BEST_TRIAL}"
        _save_one_backtest_result(label, cfg_best, shared)
    except Exception as e:
        print(f"[SAVE] Could not save locked best trial: {e}")

    print(f"\n[SAVE] Done. Dashboard reads from {DASHBOARD_RESULTS_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna optimizer for the paper trader",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true",
                   help="Quick 30-trial pass with n_jobs=1")
    g.add_argument("--full", action="store_true",
                   help="Real run: 1000 trials with n_jobs=4")
    g.add_argument("--resume", metavar="STUDY_NAME",
                   help="Continue an existing study; pair with --trials N")
    g.add_argument("--report", metavar="STUDY_NAME",
                   help="Print top-10 trials + best config breakdown")
    g.add_argument("--save-results", action="store_true",
                   help="Pre-save backtest results (full portfolio + trades "
                        "frames + scores + holdings) for the dashboard at "
                        "models/cache/dashboard_results/. Saves default "
                        "config + locked trial #706 of "
                        "optuna_v1_20260504_103429.")
    parser.add_argument("--trials", type=int, default=None,
                        help="Trial count for --resume (required there)")
    args = parser.parse_args()

    if args.smoke:
        study_name = f"smoke_{_timestamp()}"
        run_study(n_trials=30, n_jobs=1, study_name=study_name,
                  smoke_test=True)
    elif args.full:
        study_name = f"optuna_v1_{_timestamp()}"
        run_study(n_trials=1000, n_jobs=4, study_name=study_name)
    elif args.resume:
        if args.trials is None:
            parser.error("--resume requires --trials N")
        # Heuristic: smoke-named studies stay single-threaded, real
        # studies get the parallel default.
        n_jobs = 1 if args.resume.startswith("smoke_") else 4
        run_study(n_trials=args.trials, n_jobs=n_jobs,
                  study_name=args.resume,
                  smoke_test=args.resume.startswith("smoke_"))
    elif args.report:
        print_report(args.report)
    elif args.save_results:
        save_dashboard_results()
    else:
        parser.print_usage()
        sys.exit(0)


if __name__ == "__main__":
    _main()
