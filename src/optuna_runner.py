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

# --- INPUT caches: respect PAPER_TRADER_DATA_ROOT (snapshot mode) ----------
_DEFAULT_DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
DATA_ROOT = os.environ.get("PAPER_TRADER_DATA_ROOT", _DEFAULT_DATA_ROOT)
PRICE_CACHE = os.path.join(DATA_ROOT, "price_cache")

# --- OUTPUT stores: ALWAYS LIVE — never redirected via DATA_ROOT -----------
# These are RESULTS, not inputs. They must persist across snapshot runs so
# studies and trial logs accumulate in one canonical place. Snapshot mode
# only redirects input caches (fundamentals, earnings, prices, features,
# macro, analyst, sector_map, model). Output stores stay at the live tree.
_LIVE_DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
_LIVE_CACHE_DIR = os.path.join(_LIVE_DATA_ROOT, "cache")
CACHE_DIR        = _LIVE_CACHE_DIR  # back-compat for any external readers
STUDY_DB_PATH    = os.path.join(_LIVE_CACHE_DIR, "optuna_studies.db")
JOURNAL_LOG_PATH = os.path.join(_LIVE_CACHE_DIR, "optuna_journal.log")
TRIALS_LOG_PATH  = os.path.join(_LIVE_CACHE_DIR, "optuna_trials.jsonl")


def get_storage():
    """Return the Optuna storage to use, resolved at call-time.

    PAPER_TRADER_STORAGE=journal switches to JournalStorage backed by a
    file at JOURNAL_LOG_PATH (lower SQLite-style read-lock contention
    under N-way fan-out). Default and any other value falls back to the
    SQLite RDBStorage URL string. Returning either a string URL or a
    BaseStorage instance is supported by every Optuna call site
    (create_study, load_study, get_all_study_summaries, delete_study).

    Resolved per-call so a single Python process can switch between
    backends across studies if needed (e.g. dashboard reads vs parallel
    workers in the same launcher invocation).
    """
    if os.environ.get("PAPER_TRADER_STORAGE", "sqlite").lower() == "journal":
        from optuna.storages import JournalStorage
        from optuna.storages.journal import (JournalFileBackend,
                                             JournalFileOpenLock)
        os.makedirs(os.path.dirname(JOURNAL_LOG_PATH), exist_ok=True)
        # JournalFileBackend's default lock is JournalFileSymlinkLock,
        # which calls os.symlink() — that needs Developer Mode or admin
        # on Windows. JournalFileOpenLock uses exclusive file-open
        # semantics that work on every platform without elevation.
        return JournalStorage(JournalFileBackend(
            JOURNAL_LOG_PATH,
            lock_obj=JournalFileOpenLock(JOURNAL_LOG_PATH),
        ))
    return f"sqlite:///{STUDY_DB_PATH}"


def make_sampler():
    """Build the Optuna sampler used by run_study and the parallel launcher.

    PAPER_TRADER_TPE_STARTUP env var controls n_startup_trials (default
    200). The first n_startup_trials samples use Optuna's internal
    random sampling — O(1) per sample — before TPE refinement kicks
    in. This sidesteps TPE's O(n_complete) per-sample fitting cost
    during the early phase when the history is too small to inform
    refinement anyway. With 8 fan-out workers each instantiating their
    own TPESampler against the shared trial history, the first ~N
    completed trials globally end up random; subsequent trials use
    TPE on a warmed-up sample population.

    seed=42 keeps the random sequence reproducible across smokes; for
    the resulting study to be reproducible end-to-end the sampling
    side is now deterministic, which the previous seedless TPESampler()
    was not.
    """
    n_startup = int(os.environ.get("PAPER_TRADER_TPE_STARTUP", "200"))
    return TPESampler(n_startup_trials=n_startup, seed=42)


_FAILURE_SENTINEL = -1e6

# --- Objective version selector --------------------------------------------
# Default "legacy" preserves pre-rolling-framework behavior:
#   alpha_annualized − 1.5 * max(0, drawdown − 0.15)
#   (computed by objective.compute_objective on summarize_backtest output)
# Set PAPER_TRADER_OBJECTIVE=rolling_p75_p25 to switch to the new framework:
#   p75(rolling_12mo_alpha) − 0.5 * max(0, −p25(rolling_12mo_alpha))
# Read at trial-time inside objective_fn so it can be flipped between
# studies in the same process if needed.
_OBJECTIVE_LEGACY = "legacy"
_OBJECTIVE_ROLLING = "rolling_p75_p25"
_VALID_OBJECTIVES = (_OBJECTIVE_LEGACY, _OBJECTIVE_ROLLING)

# --- Architecture selector (regime_dependent_v1_spec) ----------------------
# "legacy"           — single-value tunables (existing behavior).
# "regime-dependent" — dual-tunable sets (defensive + offensive) switched
#                      per rebalance based on macro_signal vs regime_threshold.
# Read at trial-time inside objective_fn so it can be flipped between
# studies in the same process if needed.
_ARCH_LEGACY  = "legacy"
_ARCH_REGIME  = "regime-dependent"
# V2 Track B: structurally identical to regime-dependent (same field
# schema), but BacktestConfig.single_regime_mode=True suppresses the
# switch — every rebalance uses the offensive tunable set, regardless
# of macro_signal. The objective_fn routes "single-regime" to a
# dedicated, narrower search space (no defensive set, no
# regime_threshold) so TPE doesn't waste samples on dead dimensions.
_ARCH_SINGLE_REGIME = "single-regime"
_VALID_ARCHITECTURES = (_ARCH_LEGACY, _ARCH_REGIME, _ARCH_SINGLE_REGIME)

# Train window from the locked architecture decisions. Pulled from
# BacktestConfig() defaults rather than hardcoded so a future config
# revision flows through without missing this file.
_DEFAULTS    = BacktestConfig()
TRAIN_START  = _DEFAULTS.train_start
TRAIN_END    = _DEFAULTS.train_end


# Search-space param names that get suggest_int (rest get suggest_float).
_INT_PARAMS: frozenset[str] = frozenset({
    "rebalance_frequency_days",
    "position_count",
})

# Default search ranges per param. Imported by run_hypothesis.py to log
# the actually-used range (default vs override vs FIXED) into meta.json.
_DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "weight_fundamental":       (0.20, 0.55),
    "weight_technical":         (0.10, 0.45),
    "weight_model":             (0.10, 0.45),
    "macro_threshold_low":      (0.25, 0.50),
    "macro_threshold_gap":      (0.10, 0.40),
    "atr_multiplier":           (1.5, 4.0),
    "analyst_weight":           (0.0, 0.15),
    "rebalance_frequency_days": (14, 63),
    "position_count":           (10, 25),
}


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

    print("[OPTUNA]   SPY + VIX market data...")
    market = get_stock_data_cached(
        ["SPY", "^VIX"], _DEFAULTS.train_start, _DEFAULTS.validate_end,
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
    #
    # UPDATE 2026-05-05: BOTH option (a) and option (b+) tried and
    # reverted. (b+) caused the lazy-build serialization to disappear
    # and 50% of homogeneous-workload trials FAILED with the same
    # InvalidIndexError. (a) — pre-stacking once with eager engine
    # build, passing through run_backtest as a kwarg — STILL produced
    # 20% failures with the same error on a rebal=33 fixed smoke. The
    # race is downstream of the pd.concat itself; eliminating per-trial
    # concat doesn't eliminate the race. Likely lives inside model.predict
    # or the per-day .loc accesses on shared per-ticker dfs. Both failed
    # attempts also passed Smoke B (varied rebal, heterogeneous trials)
    # with 0% failures, confirming workload heterogeneity is what masks
    # the race for v1-style runs. Real fix requires understanding the
    # downstream race — deferred for a future session with deeper
    # instrumentation.
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
        "market_data":    market,
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

def _suggest_one(trial: optuna.Trial, name: str,
                 lo: float, hi: float):
    """Dispatch to suggest_int or suggest_float based on param type."""
    if name in _INT_PARAMS:
        return trial.suggest_int(name, int(lo), int(hi))
    return trial.suggest_float(name, float(lo), float(hi))


def build_search_space(trial: optuna.Trial,
                       *,
                       range_overrides: dict | None = None,
                       fixed_values: dict | None = None) -> dict:
    """Suggest a set of BacktestConfig kwargs for this trial.

    macro_threshold_high is structured as low + gap (gap in [0.10, 0.40])
    so the high > low constraint is enforced by construction — no
    rejection sampling, no wasted trials.
    weight_alt is NOT suggested here: BacktestConfig.__post_init__
    derives it as 1.0 - sum_of_others.

    Hypothesis-launcher kwargs (run_hypothesis.py uses these; defaults
    of None preserve the pre-Archetype-3 behavior bit-identically):

      fixed_values: dict mapping search-space param name -> pinned value.
        The named param is NOT passed to trial.suggest_*; Optuna doesn't
        record it as a search dimension. This is what we want when
        isolating one tunable's effect from the others — recording a
        degenerate (lo == hi) range would still inflate Optuna's
        dimension count and confuse downstream analysis.

      range_overrides: dict mapping search-space param name -> (lo, hi).
        The param IS sampled, just over the new range instead of the
        default in _DEFAULT_RANGES.

    A param appearing in both fixed_values and range_overrides is a
    programmer error — run_hypothesis.py rejects this before the call.
    If both are passed, fixed_values wins here.
    """
    range_overrides = range_overrides or {}
    fixed_values = fixed_values or {}

    def _val(name: str):
        if name in fixed_values:
            return fixed_values[name]
        lo, hi = range_overrides.get(name, _DEFAULT_RANGES[name])
        return _suggest_one(trial, name, lo, hi)

    weight_fundamental = _val("weight_fundamental")
    weight_technical   = _val("weight_technical")
    weight_model       = _val("weight_model")

    macro_threshold_low = _val("macro_threshold_low")
    macro_threshold_gap = _val("macro_threshold_gap")
    macro_threshold_high = macro_threshold_low + macro_threshold_gap

    atr_multiplier  = _val("atr_multiplier")
    analyst_weight  = _val("analyst_weight")
    rebalance_freq  = _val("rebalance_frequency_days")
    position_count  = _val("position_count")

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


# Search ranges for regime-dependent architecture (per regime_dependent_v1_spec).
_REGIME_DEPENDENT_RANGES: dict[str, tuple[float, float]] = {
    # Composite weights (per-regime; weight_alt derived as 1-sum)
    "weight_fundamental":           (0.05, 0.70),
    "weight_technical":             (0.05, 0.60),
    "weight_model":                 (0.05, 0.60),
    "atr_multiplier":               (1.0, 5.0),
    "position_count":               (5, 20),
    "rebalance_frequency_days":     (3, 90),
    # Shared
    "analyst_weight":               (0.0, 0.30),
    "macro_threshold_low":          (0.10, 0.40),
    "macro_threshold_gap":          (0.0, 0.30),
    "regime_threshold":             (0.20, 0.60),
}
_REGIME_INT_PARAMS: frozenset[str] = frozenset({
    "position_count", "position_count_offensive",
    "rebalance_frequency_days", "rebalance_frequency_days_offensive",
})

# Normalize the free weight triple so the derived weight_alt is at least
# 0.01. The wide D3 ranges (e.g. weight_fundamental [0.05, 0.70]) make
# triples summing >1.0 a frequent TPE sample (~80% prune rate at smoke
# scale). Scaling preserves relative shape while keeping the free-sum<=1
# invariant in BacktestConfig. Used by both the search-space sampler and
# _trial_to_config so the round-trip from raw trial.params back to a
# config matches the values the trial actually ran with.
def _normalize_weight_triple(wf: float, wt: float, wm: float
                             ) -> tuple[float, float, float]:
    total = wf + wt + wm
    if total >= 0.99:
        scale = 0.99 / total
        return wf * scale, wt * scale, wm * scale
    return wf, wt, wm


def build_regime_dependent_search_space(
    trial: optuna.Trial,
    *,
    range_overrides: dict | None = None,
    fixed_values: dict | None = None,
) -> dict:
    """Build a BacktestConfig kwargs dict for the regime-dependent
    architecture. Returns kwargs that can be passed directly to
    BacktestConfig(**kwargs) — architecture is set to 'regime-dependent'
    and weight_alt / weight_alt_offensive are derived (sum-to-one).

    Hypothesis-launcher kwargs (range_overrides, fixed_values) follow
    the same semantics as build_search_space — search-space param names
    that map to BacktestConfig fields. fixed_values takes precedence
    over range_overrides; both default to None for unconstrained search.
    """
    range_overrides = range_overrides or {}
    fixed_values    = fixed_values or {}

    def _suggest(name: str, default_lo: float, default_hi: float):
        if name in fixed_values:
            return fixed_values[name]
        lo, hi = range_overrides.get(name, (default_lo, default_hi))
        if name in _REGIME_INT_PARAMS:
            return trial.suggest_int(name, int(lo), int(hi))
        return trial.suggest_float(name, float(lo), float(hi))

    # Defensive set (the "regime A" the backtest enters when macro is weak)
    wf_d = _suggest("weight_fundamental",   *_REGIME_DEPENDENT_RANGES["weight_fundamental"])
    wt_d = _suggest("weight_technical",     *_REGIME_DEPENDENT_RANGES["weight_technical"])
    wm_d = _suggest("weight_model",         *_REGIME_DEPENDENT_RANGES["weight_model"])
    wf_d, wt_d, wm_d = _normalize_weight_triple(wf_d, wt_d, wm_d)
    atr_d = _suggest("atr_multiplier",      *_REGIME_DEPENDENT_RANGES["atr_multiplier"])
    pc_d  = _suggest("position_count",      *_REGIME_DEPENDENT_RANGES["position_count"])
    rf_d  = _suggest("rebalance_frequency_days",
                     *_REGIME_DEPENDENT_RANGES["rebalance_frequency_days"])

    # Offensive set (regime B, macro strong)
    wf_o = _suggest("weight_fundamental_offensive",   *_REGIME_DEPENDENT_RANGES["weight_fundamental"])
    wt_o = _suggest("weight_technical_offensive",     *_REGIME_DEPENDENT_RANGES["weight_technical"])
    wm_o = _suggest("weight_model_offensive",         *_REGIME_DEPENDENT_RANGES["weight_model"])
    wf_o, wt_o, wm_o = _normalize_weight_triple(wf_o, wt_o, wm_o)
    atr_o = _suggest("atr_multiplier_offensive",      *_REGIME_DEPENDENT_RANGES["atr_multiplier"])
    pc_o  = _suggest("position_count_offensive",      *_REGIME_DEPENDENT_RANGES["position_count"])
    rf_o  = _suggest("rebalance_frequency_days_offensive",
                     *_REGIME_DEPENDENT_RANGES["rebalance_frequency_days"])

    # Shared
    aw    = _suggest("analyst_weight",        *_REGIME_DEPENDENT_RANGES["analyst_weight"])
    ml    = _suggest("macro_threshold_low",   *_REGIME_DEPENDENT_RANGES["macro_threshold_low"])
    mg    = _suggest("macro_threshold_gap",   *_REGIME_DEPENDENT_RANGES["macro_threshold_gap"])
    mh    = ml + mg
    rt    = _suggest("regime_threshold",      *_REGIME_DEPENDENT_RANGES["regime_threshold"])
    return {
        "architecture":     _ARCH_REGIME,
        "regime_threshold": rt,
        # Defensive (the existing legacy field names hold the defensive set)
        "weight_fundamental":       wf_d,
        "weight_technical":         wt_d,
        "weight_model":             wm_d,
        "atr_multiplier":           atr_d,
        "position_count":           pc_d,
        "rebalance_frequency_days": rf_d,
        # Offensive
        "weight_fundamental_offensive":       wf_o,
        "weight_technical_offensive":         wt_o,
        "weight_model_offensive":             wm_o,
        "atr_multiplier_offensive":           atr_o,
        "position_count_offensive":           pc_o,
        "rebalance_frequency_days_offensive": rf_o,
        # Shared
        "analyst_weight":       aw,
        "macro_threshold_low":  ml,
        "macro_threshold_high": mh,
    }


def build_single_regime_search_space(
    trial: optuna.Trial,
    *,
    range_overrides: dict | None = None,
    fixed_values: dict | None = None,
) -> dict:
    """Build a BacktestConfig kwargs dict for V2 Track B (single-regime).

    Samples ONE tunable set (no defensive duplicate, no regime_threshold)
    and returns kwargs with single_regime_mode=True. The offensive_*
    fields hold the sampled values; the legacy fields are mirrored to
    the same values so BacktestConfig.__post_init__'s legacy weight-sum
    validation passes (the legacy fields are unused at runtime when
    single_regime_mode is True).

    Sampled params (9 unique): weight_{fundamental,technical,model}_offensive
    + atr_multiplier_offensive + position_count_offensive +
    rebalance_frequency_days_offensive + analyst_weight +
    macro_threshold_low + macro_threshold_gap. Param names use the
    _offensive suffix so _trial_to_config can distinguish Track B
    (has offensive params, no regime_threshold) from legacy (no
    offensive params) and Track A (has both).
    """
    range_overrides = range_overrides or {}
    fixed_values    = fixed_values or {}

    def _suggest(name: str, default_lo: float, default_hi: float):
        if name in fixed_values:
            return fixed_values[name]
        lo, hi = range_overrides.get(name, (default_lo, default_hi))
        if name in _REGIME_INT_PARAMS:
            return trial.suggest_int(name, int(lo), int(hi))
        return trial.suggest_float(name, float(lo), float(hi))

    wf = _suggest("weight_fundamental_offensive",
                  *_REGIME_DEPENDENT_RANGES["weight_fundamental"])
    wt = _suggest("weight_technical_offensive",
                  *_REGIME_DEPENDENT_RANGES["weight_technical"])
    wm = _suggest("weight_model_offensive",
                  *_REGIME_DEPENDENT_RANGES["weight_model"])
    wf, wt, wm = _normalize_weight_triple(wf, wt, wm)
    atr = _suggest("atr_multiplier_offensive",
                   *_REGIME_DEPENDENT_RANGES["atr_multiplier"])
    pc  = _suggest("position_count_offensive",
                   *_REGIME_DEPENDENT_RANGES["position_count"])
    rfd = _suggest("rebalance_frequency_days_offensive",
                   *_REGIME_DEPENDENT_RANGES["rebalance_frequency_days"])
    aw  = _suggest("analyst_weight",
                   *_REGIME_DEPENDENT_RANGES["analyst_weight"])
    ml  = _suggest("macro_threshold_low",
                   *_REGIME_DEPENDENT_RANGES["macro_threshold_low"])
    mg  = _suggest("macro_threshold_gap",
                   *_REGIME_DEPENDENT_RANGES["macro_threshold_gap"])

    return {
        "architecture":          _ARCH_REGIME,
        "single_regime_mode":    True,
        "regime_threshold":      None,  # ignored at runtime
        # Offensive set holds THE tunables in single-regime mode
        "weight_fundamental_offensive":       wf,
        "weight_technical_offensive":         wt,
        "weight_model_offensive":             wm,
        "atr_multiplier_offensive":           atr,
        "position_count_offensive":           pc,
        "rebalance_frequency_days_offensive": rfd,
        # Legacy fields mirrored so BacktestConfig.__post_init__'s
        # legacy weight-sum validation passes; never read at runtime
        # because get_active_tunables short-circuits on single_regime_mode.
        "weight_fundamental":       wf,
        "weight_technical":         wt,
        "weight_model":             wm,
        "atr_multiplier":           atr,
        "position_count":           pc,
        "rebalance_frequency_days": rfd,
        # Shared
        "analyst_weight":       aw,
        "macro_threshold_low":  ml,
        "macro_threshold_high": ml + mg,
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
                 log_lock: threading.Lock,
                 range_overrides: dict | None = None,
                 fixed_values: dict | None = None) -> float:
    """One Optuna trial: build a config, run a backtest, score it.

    Pruned when the three free weights sum > 1.0 (BacktestConfig raises
    ValueError; we re-raise as TrialPruned so Optuna registers the trial
    as failed without polluting the TPE prior). Any other exception is
    swallowed and the failure sentinel is returned.
    """
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    architecture = os.environ.get("PAPER_TRADER_ARCHITECTURE", _ARCH_LEGACY)
    if architecture not in _VALID_ARCHITECTURES:
        raise ValueError(
            f"PAPER_TRADER_ARCHITECTURE={architecture!r} not in "
            f"{_VALID_ARCHITECTURES}")

    if architecture == _ARCH_REGIME:
        config_kwargs = build_regime_dependent_search_space(
            trial,
            range_overrides=range_overrides,
            fixed_values=fixed_values,
        )
    elif architecture == _ARCH_SINGLE_REGIME:
        config_kwargs = build_single_regime_search_space(
            trial,
            range_overrides=range_overrides,
            fixed_values=fixed_values,
        )
    else:
        config_kwargs = build_search_space(
            trial,
            range_overrides=range_overrides,
            fixed_values=fixed_values,
        )
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

    # Per spec D6: regime-dependent runs default to the rolling objective.
    # Legacy runs keep the legacy objective default. Either can be
    # overridden by setting PAPER_TRADER_OBJECTIVE explicitly. Single-
    # regime (Track B control) uses the same rolling objective as
    # Track A so the two tracks compare apples-to-apples.
    obj_default = (_OBJECTIVE_ROLLING
                   if architecture in (_ARCH_REGIME, _ARCH_SINGLE_REGIME)
                   else _OBJECTIVE_LEGACY)
    objective_version = os.environ.get("PAPER_TRADER_OBJECTIVE", obj_default)
    if objective_version not in _VALID_OBJECTIVES:
        raise ValueError(
            f"PAPER_TRADER_OBJECTIVE={objective_version!r} not in "
            f"{_VALID_OBJECTIVES}")

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
            market_data=shared_data.get("market_data"),
            compute_rolling_metrics=(
                objective_version == _OBJECTIVE_ROLLING),
        )
        summary    = summarize_backtest(portfolio_df, shared_data["spy_close"])
        legacy_score      = compute_objective(summary)
        components = compute_objective_components(summary)

        if objective_version == _OBJECTIVE_ROLLING:
            from rolling_metrics import compute_objective_score  # local import
            rolling_bundle = portfolio_df.attrs.get("rolling_metrics") or {}
            rolling_12mo = rolling_bundle.get("rolling_12mo", {})
            # Use the precomputed objective_score from the bundle.
            score = float(rolling_12mo.get("objective_score", _FAILURE_SENTINEL))
        else:
            score = legacy_score

        # ---- V2 Track A activation constraint ----
        # PAPER_TRADER_REQUIRE_DEFENSIVE_PCT (default 0 = no gate) sets a
        # minimum percentage of training-window rebalances that must
        # occur in the defensive regime; configs whose regime_threshold
        # is too low to meaningfully exercise the defensive set get
        # hard-rejected with the failure sentinel so TPE learns to
        # avoid that region. Track A sets this to 10. Track B has
        # single_regime_mode=True so the gate is a no-op (no defensive
        # set to exercise). Legacy architecture is also exempt.
        require_pct = float(os.environ.get(
            "PAPER_TRADER_REQUIRE_DEFENSIVE_PCT", "0"))
        if (require_pct > 0
                and architecture == _ARCH_REGIME
                and not config.single_regime_mode):
            regime_stats = portfolio_df.attrs.get("regime_stats") or {}
            def_pct_reb = regime_stats.get("defensive_pct_rebalances")
            if def_pct_reb is None or def_pct_reb < require_pct:
                record = {
                    "study_name":      study_name,
                    "trial_number":    trial.number,
                    "state":           "REJECTED_ACTIVATION_PCT",
                    "score":           _FAILURE_SENTINEL,
                    "rejection_reason": (
                        f"defensive_pct_rebalances="
                        f"{def_pct_reb!r} < required {require_pct}"),
                    "objective_version": objective_version,
                    "architecture":      architecture,
                    "legacy_score":      legacy_score,
                    "config":            config.to_dict(),
                    "regime_stats":      regime_stats,
                    "started_at":        started_at.isoformat(),
                    "duration_seconds":  round(time.perf_counter() - t0, 3),
                }
                _append_trial_log(jsonl_path, log_lock, record)
                return float(_FAILURE_SENTINEL)

        record = {
            "study_name":   study_name,
            "trial_number": trial.number,
            "state":        "COMPLETE",
            "score":        score,
            "objective_version": objective_version,
            "architecture":      architecture,
            "legacy_score": legacy_score,
            "config":       config.to_dict(),
            "components":   components,
            "started_at":   started_at.isoformat(),
            "duration_seconds": round(time.perf_counter() - t0, 3),
        }
        if architecture in (_ARCH_REGIME, _ARCH_SINGLE_REGIME):
            record["regime_stats"] = portfolio_df.attrs.get("regime_stats")
        if objective_version == _OBJECTIVE_ROLLING:
            # Trim the windows lists out of the JSONL record (verbose); keep
            # only summary stats. Full bundle still ends up in dashboard
            # meta.json via _save_one_backtest_result.
            r12 = portfolio_df.attrs.get("rolling_metrics", {}).get("rolling_12mo", {})
            record["rolling_12mo_summary"] = {
                "alpha_distribution_stats": r12.get("alpha_distribution_stats"),
                "objective_score":          r12.get("objective_score"),
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
              smoke_test: bool = False,
              *,
              range_overrides: dict | None = None,
              fixed_values: dict | None = None) -> optuna.Study:
    """Create-or-load a study and run ``n_trials`` more trials on it.

    range_overrides / fixed_values are forwarded through objective_fn to
    build_search_space. Defaults of None preserve the pre-Archetype-3
    behavior (existing v1/smoke/resume callers see no change)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    storage = get_storage()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=make_sampler(),
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
            range_overrides=range_overrides,
            fixed_values=fixed_values,
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
    storage = get_storage()
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f"[OPTUNA] Study {study_name!r} not found in storage")
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
                              shared: dict,
                              *,
                              extra_meta: dict | None = None,
                              output_dir: str | None = None,
                              split_date: str | None = None) -> None:
    """Run one backtest with ``config`` and save the full portfolio +
    trades + scores + holdings to ``<output_dir>/{label}/``. The dashboard
    reads this for instant-load Positions/Trades/Overview tabs.

    Defaults preserve the v1 save behavior:
      output_dir defaults to DASHBOARD_RESULTS_DIR.
      split_date defaults to config.validate_start (so the dashboard's
        equity curve shows validation-window performance).
    Hypothesis runs (run_hypothesis.py) pass split_date=config.train_start
    for training-only saves.
    extra_meta gets merged into meta.json after the standard fields. Used
    by the hypothesis launcher to thread hypothesis_id, search_ranges,
    fixed_tunables, base_config_ref, promoted: false, etc."""
    base_dir = output_dir if output_dir is not None else DASHBOARD_RESULTS_DIR
    out_dir = os.path.join(base_dir, label)
    os.makedirs(out_dir, exist_ok=True)
    effective_split = (split_date if split_date is not None
                       else config.validate_start)
    print(f"[SAVE] Running backtest for label={label!r} "
          f"(split_date={effective_split})...")
    t0 = time.perf_counter()
    portfolio_df, trades_df, scores, holdings = run_backtest(
        shared["featured_data"], shared["price_data"],
        split_date=effective_split,
        fund_data=shared["fund_data"],
        sector_map=shared["sector_map"],
        earnings_dates=shared["earnings_dates"],
        model=shared["model"],
        config=config,
        market_data=shared.get("market_data"),
        compute_rolling_metrics=True,
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

    # Snapshot the SPY + QQQ close series for the result's date range.
    # The dashboard's Performance tab plots Strategy vs SPY vs QQQ and
    # computes alpha/beta — all from these benchmark series. Saving them
    # here makes the dashboard self-contained: no yfinance call at view
    # time, which matters in cloud mode where Streamlit Cloud's IP can
    # be soft-throttled by Yahoo (SPY in particular). market_data is
    # already loaded for compute_rolling_metrics → reuse it.
    market_data = shared.get("market_data") or {}
    if not portfolio_df.empty:
        bm_start = portfolio_df.index[0]
        bm_end   = portfolio_df.index[-1]
        for tkr in ("SPY", "QQQ"):
            df = market_data.get(tkr)
            if df is None or df.empty or "Close" not in df.columns:
                # Live-mode fallback (snapshot mode hard-fails by design).
                try:
                    import yfinance as yf
                    h = yf.download(
                        tkr,
                        start=bm_start.strftime("%Y-%m-%d"),
                        end=(bm_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        auto_adjust=True, progress=False)
                    if isinstance(h.columns, pd.MultiIndex):
                        h.columns = h.columns.droplevel(1)
                    close = h["Close"]
                except Exception as e:
                    print(f"[SAVE]   skipping {tkr}_close save ({e})")
                    continue
            else:
                close = df["Close"]
            close = close.loc[(close.index >= bm_start)
                               & (close.index <= bm_end)]
            close.to_frame("Close").to_parquet(
                os.path.join(out_dir, f"{tkr}_close.parquet"))

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

    # Record which cache snapshot the run used, or "live" if env var unset.
    # Stored both at top-level meta for any save (v1 path included), AND
    # available inside extra_meta for hypothesis runs that may want it.
    _snap_root = os.environ.get("PAPER_TRADER_DATA_ROOT")
    if _snap_root:
        cache_snapshot = os.path.basename(os.path.abspath(_snap_root))
    else:
        cache_snapshot = "live"

    meta = {
        "label":       label,
        "saved_at":    datetime.now(timezone.utc).isoformat(),
        "config":      config.to_dict(),
        "split_date":  effective_split,
        "n_days":      int(len(portfolio_df)),
        "n_trades":    int(len(trades_df)),
        "n_holdings":  int(len(holdings)),
        "runtime_seconds": round(elapsed, 3),
        "cache_snapshot":  cache_snapshot,
    }
    # Rolling-window evaluation framework — present if run_backtest was
    # called with compute_rolling_metrics=True (which we always do here).
    rolling_bundle = portfolio_df.attrs.get("rolling_metrics")
    if rolling_bundle:
        meta["rolling_metrics"] = rolling_bundle
    rolling_err = portfolio_df.attrs.get("rolling_metrics_error")
    if rolling_err:
        meta["rolling_metrics_error"] = rolling_err

    # Architecture + per-regime decomposition for regime-dependent runs.
    # Legacy runs just record architecture="legacy" and skip the rest.
    # Single-regime mode (Track B) sits inside architecture="regime-
    # dependent" — we record single_regime_mode flag + the offensive
    # tunables (which are the only ones that mattered) but skip the
    # defensive_tunables block since those values are runtime-unused.
    meta["architecture"] = config.architecture
    if config.architecture == _ARCH_REGIME:
        meta["single_regime_mode"] = bool(config.single_regime_mode)
        meta["regime_threshold"]   = config.regime_threshold
        if not config.single_regime_mode:
            meta["defensive_tunables"] = {
                "weight_fundamental":       config.weight_fundamental,
                "weight_technical":         config.weight_technical,
                "weight_model":             config.weight_model,
                "weight_alt":               config.weight_alt,
                "atr_multiplier":           config.atr_multiplier,
                "position_count":           config.position_count,
                "rebalance_frequency_days": config.rebalance_frequency_days,
            }
        meta["offensive_tunables"] = {
            "weight_fundamental":       config.weight_fundamental_offensive,
            "weight_technical":         config.weight_technical_offensive,
            "weight_model":             config.weight_model_offensive,
            "weight_alt":               config.weight_alt_offensive,
            "atr_multiplier":           config.atr_multiplier_offensive,
            "position_count":           config.position_count_offensive,
            "rebalance_frequency_days": config.rebalance_frequency_days_offensive,
        }
        meta["shared_tunables"] = {
            "analyst_weight":       config.analyst_weight,
            "macro_threshold_low":  config.macro_threshold_low,
            "macro_threshold_high": config.macro_threshold_high,
        }
        regime_stats = portfolio_df.attrs.get("regime_stats")
        if regime_stats:
            meta["regime_stats"] = regime_stats

    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVE]   wrote {out_dir}")


def _trial_to_config(trial: optuna.trial.FrozenTrial,
                     fixed_values: dict | None = None) -> BacktestConfig:
    """Reconstruct a BacktestConfig from a completed trial. fixed_values
    fills in search-space params that weren't sampled (the hypothesis
    launcher pins them via build_search_space's fixed_values kwarg,
    which skips trial.suggest_* — those keys are absent from
    trial.params). Default None preserves bit-identical behavior for
    v1 callers — fixed_values is only needed for hypothesis runs.

    Architecture detection (param-name based, no env-var dependency):
      * 'regime_threshold' in p          -> Track A regime-dependent
      * 'weight_fundamental_offensive' in p (no regime_threshold)
                                         -> Track B single-regime
      * neither                          -> legacy"""
    # Merge with trial.params winning on conflict. By construction the
    # two sets are disjoint (fixed names skip suggest_*), but defensive.
    p = {**(fixed_values or {}), **trial.params}

    if ("weight_fundamental_offensive" in p
            and "regime_threshold" not in p):
        # Track B: single-regime control. Offensive set holds the
        # tunables; legacy fields mirror the same values to keep
        # BacktestConfig.__post_init__ legacy validation happy.
        wf, wt, wm = _normalize_weight_triple(
            p["weight_fundamental_offensive"],
            p["weight_technical_offensive"],
            p["weight_model_offensive"])
        return BacktestConfig(
            architecture          = "regime-dependent",
            single_regime_mode    = True,
            regime_threshold      = None,
            weight_fundamental_offensive       = wf,
            weight_technical_offensive         = wt,
            weight_model_offensive             = wm,
            atr_multiplier_offensive           = p["atr_multiplier_offensive"],
            position_count_offensive           = p["position_count_offensive"],
            rebalance_frequency_days_offensive = p["rebalance_frequency_days_offensive"],
            # Mirror legacy fields (unused at runtime, validated only)
            weight_fundamental       = wf,
            weight_technical         = wt,
            weight_model             = wm,
            atr_multiplier           = p["atr_multiplier_offensive"],
            position_count           = p["position_count_offensive"],
            rebalance_frequency_days = p["rebalance_frequency_days_offensive"],
            # Shared
            analyst_weight       = p["analyst_weight"],
            macro_threshold_low  = p["macro_threshold_low"],
            macro_threshold_high = p["macro_threshold_low"] + p["macro_threshold_gap"],
        )

    if "regime_threshold" in p:
        # Regime-dependent: 7 defensive + 7 offensive + 4 shared params.
        # Apply the same renormalization the search-space sampler used
        # (raw trial.params hold the unnormalized suggestions; the trial
        # actually ran with normalized weights).
        wf_d, wt_d, wm_d = _normalize_weight_triple(
            p["weight_fundamental"], p["weight_technical"], p["weight_model"])
        wf_o, wt_o, wm_o = _normalize_weight_triple(
            p["weight_fundamental_offensive"],
            p["weight_technical_offensive"],
            p["weight_model_offensive"])
        return BacktestConfig(
            architecture="regime-dependent",
            regime_threshold=p["regime_threshold"],
            # Defensive (the legacy-named fields hold the defensive set)
            weight_fundamental       = wf_d,
            weight_technical         = wt_d,
            weight_model             = wm_d,
            atr_multiplier           = p["atr_multiplier"],
            position_count           = p["position_count"],
            rebalance_frequency_days = p["rebalance_frequency_days"],
            # Offensive
            weight_fundamental_offensive       = wf_o,
            weight_technical_offensive         = wt_o,
            weight_model_offensive             = wm_o,
            atr_multiplier_offensive           = p["atr_multiplier_offensive"],
            position_count_offensive           = p["position_count_offensive"],
            rebalance_frequency_days_offensive = p["rebalance_frequency_days_offensive"],
            # Shared
            analyst_weight       = p["analyst_weight"],
            macro_threshold_low  = p["macro_threshold_low"],
            macro_threshold_high = p["macro_threshold_low"] + p["macro_threshold_gap"],
        )

    # Legacy
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
        ["SPY", "^VIX"], cfg.train_start, cfg.validate_end,
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
        "market_data":    market,
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
    storage = get_storage()
    try:
        study = optuna.load_study(study_name=LOCKED_BEST_STUDY,
                                  storage=storage)
        trial = study.trials[LOCKED_BEST_TRIAL]
        cfg_best = _trial_to_config(trial)
        label = f"best_{LOCKED_BEST_STUDY}_{LOCKED_BEST_TRIAL}"
        _save_one_backtest_result(label, cfg_best, shared)
    except Exception as e:
        print(f"[SAVE] Could not save locked best trial: {e}")

    print(f"\n[SAVE] Done. Dashboard reads from {DASHBOARD_RESULTS_DIR}")


# ---------------------------------------------------------------------------
# Hypothesis save (Archetype 3 — invoked by run_hypothesis.py)
# ---------------------------------------------------------------------------

def save_hypothesis_result(
    study_name: str,
    hypothesis_id: str,
    search_ranges: dict,
    fixed_tunables: dict,
    base_config_ref: str,
    window: str,
    output_dir: str | None = None,
    trial_number: int | None = None,
) -> None:
    """Save dashboard_results for a hypothesis study's best (or specified)
    trial. Wraps _save_one_backtest_result with the hypothesis metadata
    that run_hypothesis.py threads through.

    The label format f"best_{study_name}_{trial_number}" matches the
    convention save_dashboard_results uses for v1, so the dashboard's
    Best-Trial picker discovers the new entry without code changes.

    Two-backtest save (matches Phase 0's save_dashboard_results pattern):
      1. Training-window verification backtest (only when window=="train").
         Captures objective-verification metrics into
         extra_meta["training_window"] for cross-window comparison. No
         disk artefacts beyond the meta entry — the parquet/json files
         from this run are intentionally discarded.
      2. Validation-window backtest. This becomes the dashboard payload
         (portfolio.parquet, trades.parquet, scores.json, holdings.json)
         so the dashboard shows the held-out 2024+ data — same convention
         as Phase 0. window=="train" hypothesis runs no longer save
         training-window data on disk.

    The window parameter still records which window the trial was
    optimized on (for provenance in meta.json), independent of the
    dashboard payload, which is now always validation.

    promoted is forced to False. snapshot_for_cloud._meta_with_promoted
    honors any explicit promoted key, so this guarantees experimental
    studies stay hidden from the cloud dashboard's Best-Trial picker
    until manual graduation.
    """
    storage = get_storage()
    study = optuna.load_study(study_name=study_name, storage=storage)
    if trial_number is None:
        trial = study.best_trial
        trial_number = trial.number
    else:
        trial = study.trials[trial_number]
    cfg = _trial_to_config(trial, fixed_values=fixed_tunables)
    label = f"best_{study_name}_{trial_number}"

    extra_meta = {
        "hypothesis_id":   hypothesis_id,
        "study_name":      study_name,
        "trial_number":    int(trial_number),
        "trial_score":     (float(trial.value) if trial.value is not None
                            else None),
        "search_ranges":   search_ranges,
        "fixed_tunables":  fixed_tunables,
        "base_config_ref": base_config_ref,
        "window":          window,
        "promoted":        False,
    }

    if window == "train":
        train_shared = _load_shared_data(cfg.train_start, cfg.train_end)
        print(f"[HYPOTHESIS SAVE] Training-window verification backtest "
              f"(split_date={cfg.train_start})...")
        t0 = time.perf_counter()
        train_pf, train_tr, _scores, train_hold = run_backtest(
            train_shared["featured_data"], train_shared["price_data"],
            split_date=cfg.train_start,
            fund_data=train_shared["fund_data"],
            sector_map=train_shared["sector_map"],
            earnings_dates=train_shared["earnings_dates"],
            model=train_shared["model"],
            config=cfg,
            market_data=train_shared.get("market_data"),
            compute_rolling_metrics=True,
        )
        train_summary = summarize_backtest(train_pf, train_shared["spy_close"])
        train_block = {
            "split_date":      cfg.train_start,
            "n_days":          int(len(train_pf)),
            "n_trades":        int(len(train_tr)),
            "n_holdings":      int(len(train_hold)),
            "score":           float(compute_objective(train_summary)),
            "components":      compute_objective_components(train_summary),
            "runtime_seconds": round(time.perf_counter() - t0, 3),
        }
        train_rolling = train_pf.attrs.get("rolling_metrics")
        if train_rolling:
            train_block["rolling_metrics"] = train_rolling
        train_regime = train_pf.attrs.get("regime_stats")
        if train_regime:
            train_block["regime_stats"] = train_regime
        extra_meta["training_window"] = train_block

    # Validation-window run is always the dashboard payload — the on-disk
    # parquet/json artefacts come from THIS run, not the training one.
    val_shared = _load_full_shared_data()
    _save_one_backtest_result(
        label, cfg, val_shared,
        extra_meta=extra_meta,
        output_dir=output_dir,
        split_date=cfg.validate_start,
    )
    out_root = output_dir if output_dir is not None else DASHBOARD_RESULTS_DIR
    print(f"[HYPOTHESIS SAVE] label={label} -> {out_root}")


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
