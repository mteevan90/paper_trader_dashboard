import json
import os
import socket
import sys
import time
from datetime import datetime, timedelta

# Force UTF-8 so non-ASCII characters in earnings messages (e.g. en-dashes)
# print cleanly on Windows consoles and when redirected to log files.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# Process-wide socket timeout — same rationale as fetch_data.py. Set here
# in addition to fetch_data.py so the timeout is in effect regardless of
# which module hits its top-level imports first; idempotent on the second
# call. yfinance calls in this module (fetch_fundamentals,
# fetch_earnings_dates) need the same protection.
socket.setdefaulttimeout(10.0)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from alt_signals import score_alt_bucket
from backtest_config import BacktestConfig
from features import add_features, add_market_features
from fetch_data import (NASDAQ_100_TICKERS, SP500_TICKERS, SECTOR_MAP,
                        UNIVERSE_TICKERS, build_sector_map,
                        get_stock_data, get_stock_data_cached)
from macro_signals import (fetch_macro_signals, compute_macro_score,
                           fetch_analyst_targets)
from model import FEATURE_COLS, MODEL_PATH, load_model

# Legacy module-level aliases — derived from BacktestConfig() defaults so
# there's a single source of truth. main.py still imports several of these
# names directly; once main.py is migrated to take a BacktestConfig the
# aliases can be deleted.
_DEFAULT_CONFIG        = BacktestConfig()
INITIAL_CASH           = _DEFAULT_CONFIG.starting_capital
TOP_N                  = _DEFAULT_CONFIG.position_count
REBAL_DAYS             = _DEFAULT_CONFIG.rebalance_frequency_days
ATR_STOP_MULT          = _DEFAULT_CONFIG.atr_multiplier
MAX_STOP_PCT           = _DEFAULT_CONFIG.atr_cap_pct
MIN_ATR_STOP_PCT       = _DEFAULT_CONFIG.atr_floor_pct
TX_COST                = _DEFAULT_CONFIG.transaction_cost_pct
MAX_PER_SECTOR         = _DEFAULT_CONFIG.sector_cap
MIN_HOLD_DAYS          = _DEFAULT_CONFIG.min_hold_days
EARNINGS_BLACKOUT_DAYS = _DEFAULT_CONFIG.earnings_blackout_days

# --- Cache snapshot system: env-driven data root --------------------------
# PAPER_TRADER_DATA_ROOT (when set) redirects all input caches to a
# snapshot directory (typically models/snapshots/<name>/). Default is the
# live models/ tree. Constants are computed at module import; run_hypothesis.py
# pre-parses --cache-snapshot and sets the env var BEFORE importing this.
_DEFAULT_DATA_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models"))
DATA_ROOT = os.environ.get("PAPER_TRADER_DATA_ROOT", _DEFAULT_DATA_ROOT)
SNAPSHOT_MODE = os.environ.get("PAPER_TRADER_DATA_ROOT") is not None

# --- Fundamentals / earnings caches -----------------------------------------
# Fundamentals change slowly (quarterly-ish), so a 7-day TTL is safe.
# Earnings calendars need to be current — daily TTL.
CACHE_DIR             = os.path.join(DATA_ROOT, "cache")
FUNDAMENTALS_CACHE    = os.path.join(CACHE_DIR, "fundamentals.json")
EARNINGS_CACHE        = os.path.join(CACHE_DIR, "earnings_dates.json")
FUNDAMENTALS_TTL_DAYS = 7
EARNINGS_TTL_DAYS     = 1


def _snapshot_miss(cache_name: str, path: str, required: str) -> "RuntimeError":
    """Build a [CACHE_SNAPSHOT_MISS] error for hard-fail in snapshot mode."""
    return RuntimeError(
        f"[CACHE_SNAPSHOT_MISS] Cache miss in snapshot mode.\n"
        f"  Cache: {cache_name}\n"
        f"  Path: {path}\n"
        f"  Snapshot: {os.environ.get('PAPER_TRADER_DATA_ROOT')}\n"
        f"  Required: {required}\n"
        f"  Action: re-create snapshot with current data, or fix snapshot to "
        f"include this data."
    )


def _cache_age_days(path: str) -> float | None:
    """Return the age of ``path`` in days, or None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 86400.0


def clear_data_cache() -> int:
    """Delete fundamentals.json and earnings_dates.json if present.
    Returns the number of files removed."""
    removed = 0
    for path in (FUNDAMENTALS_CACHE, EARNINGS_CACHE):
        if os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except OSError as e:
                print(f"  [CACHE] could not delete {path}: {e}")
    print(f"  [CACHE] cleared {removed} data cache file(s) from {CACHE_DIR}")
    return removed


# Composite score weights — legacy aliases (single source of truth lives
# in BacktestConfig). Inside run_backtest we read config.weight_* directly.
W_FUNDAMENTAL = _DEFAULT_CONFIG.weight_fundamental
W_TECHNICAL   = _DEFAULT_CONFIG.weight_technical
W_MODEL       = _DEFAULT_CONFIG.weight_model


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def _fetch_one_fundamental(tkr: str) -> dict:
    """Single-ticker yfinance .info pull, normalized to the 5 metrics
    score_fundamentals expects. Failures fall back to neutral defaults
    (matches pre-existing behavior)."""
    try:
        info = yf.Ticker(tkr).info
        return {
            "revenue_growth": info.get("revenueGrowth") or 0.0,
            "profit_margin": info.get("profitMargins") or 0.0,
            "pe_ratio": min(info.get("trailingPE") or 100, 100),
            "debt_to_equity": info.get("debtToEquity") or 0.0,
            "roe": info.get("returnOnEquity") or 0.0,
        }
    except Exception:
        return {
            "revenue_growth": 0.0, "profit_margin": 0.0,
            "pe_ratio": 100.0, "debt_to_equity": 0.0, "roe": 0.0,
        }


def fetch_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """Fetch fundamental data for all tickers via yfinance.

    Disk-cached at ``FUNDAMENTALS_CACHE`` with a 7-day TTL.

    Cache semantics:
      - If the cache is FRESH (age < TTL) and every requested ticker is
        already in it, return the cache unchanged (no yfinance calls).
      - If the cache is FRESH but some requested tickers are missing,
        fetch ONLY those missing tickers, merge into the cache, persist.
        Existing entries are preserved verbatim.
      - If the cache is STALE (age >= TTL) or absent, refetch every
        requested ticker (full refresh — pre-existing behavior).

    The "fresh-but-incomplete" branch fixes a bug where adding new
    tickers to the universe (e.g. the sp1500 expansion) silently left
    them without fundamentals coverage indefinitely: the cache was fresh
    so the early-return fired, never querying yfinance for the new
    names. See fetch_sp1500.py.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    age = _cache_age_days(FUNDAMENTALS_CACHE)
    cache_present = age is not None
    cache_fresh = cache_present and age < FUNDAMENTALS_TTL_DAYS

    cached: dict[str, dict] = {}
    if cache_present:
        try:
            with open(FUNDAMENTALS_CACHE, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except Exception as e:
            print(f"  [CACHE] Fundamentals cache unreadable ({e}), refetching")
            cached = {}
            cache_fresh = False

    requested = list(dict.fromkeys(tickers))   # dedupe, preserve order
    missing = [t for t in requested if t not in cached]

    # Fast path: fresh cache, every requested ticker already populated.
    if cache_fresh and not missing:
        print(f"  [CACHE] Fundamentals loaded from disk "
              f"({age:.1f} days old, {len(cached)} entries, "
              f"{len(requested)} requested)")
        return cached

    if SNAPSHOT_MODE:
        raise _snapshot_miss(
            "fundamentals", FUNDAMENTALS_CACHE,
            f"file present, readable, age < {FUNDAMENTALS_TTL_DAYS} days, "
            f"and every requested ticker present "
            f"(missing {len(missing)}, e.g. {missing[:3]})")

    if cache_fresh:
        # Fresh cache + new tickers — fetch ONLY the missing ones, keep
        # the rest verbatim. This is the bug-fix branch.
        fund = dict(cached)
        to_fetch = missing
        print(f"  [CACHE] Fundamentals fresh ({age:.1f} days old): "
              f"{len(cached)} cached, fetching {len(to_fetch)} new "
              f"tickers...")
    else:
        # Stale or missing cache — full refresh of the requested set.
        # Discard the old cache (matches pre-existing behavior).
        fund = {}
        to_fetch = requested
        if cache_present:
            print(f"  [CACHE] Fundamentals stale ({age:.1f} days old); "
                  f"refetching {len(to_fetch)} tickers...")
        else:
            print(f"  Fetching fundamentals for {len(to_fetch)} tickers "
                  f"(no cache)...")

    for tkr in to_fetch:
        fund[tkr] = _fetch_one_fundamental(tkr)

    try:
        with open(FUNDAMENTALS_CACHE, "w", encoding="utf-8") as f:
            json.dump(fund, f)
        print(f"  [CACHE] Fundamentals written: {len(fund)} total entries "
              f"({len(to_fetch)} fresh, "
              f"{len(fund) - len(to_fetch)} pre-existing)")
    except Exception as e:
        print(f"  [CACHE] Failed to write fundamentals cache ({e})")
    return fund


def _fetch_one_earnings(tkr: str, start_ts: pd.Timestamp,
                        end_ts: pd.Timestamp) -> list[pd.Timestamp]:
    """Single-ticker earnings calendar fetch, filtered to [start, end].
    Empty list on failure (matches pre-existing behavior)."""
    try:
        cal = yf.Ticker(tkr).get_earnings_dates(limit=12)
        if cal is not None and not cal.empty:
            idx = pd.to_datetime(cal.index).tz_localize(None).normalize()
            return sorted(d for d in idx if start_ts <= d <= end_ts)
    except Exception:
        pass
    return []


def fetch_earnings_dates(
    tickers: list[str], start: str, end: str
) -> dict[str, list[pd.Timestamp]]:
    """Fetch upcoming earnings dates for each ticker in the given window.

    Returns a dict mapping ticker -> sorted list of earnings Timestamps
    (normalized to midnight) that fall between start and end. Tickers with
    no earnings schedule available return an empty list.

    Disk-cached at ``EARNINGS_CACHE`` with a 1-day TTL. Timestamps are
    serialized to ISO-format strings since JSON can't handle Timestamp
    objects directly, and converted back on load.

    Cache semantics: same shape as fetch_fundamentals — fresh cache +
    missing tickers triggers a partial fetch that populates only the new
    names, preserving existing entries. Stale/absent cache triggers a
    full refresh of the request list. Fixes the same "fresh cache hides
    new tickers" bug that fetch_fundamentals had.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    age = _cache_age_days(EARNINGS_CACHE)
    cache_present = age is not None
    cache_fresh = cache_present and age < EARNINGS_TTL_DAYS

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    cached: dict[str, list[pd.Timestamp]] = {}
    if cache_present:
        try:
            with open(EARNINGS_CACHE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cached = {t: [pd.Timestamp(s) for s in dates]
                      for t, dates in raw.items()}
        except Exception as e:
            print(f"  [CACHE] Earnings cache unreadable ({e}), refetching")
            cached = {}
            cache_fresh = False

    requested = list(dict.fromkeys(tickers))   # dedupe, preserve order
    missing = [t for t in requested if t not in cached]

    # Fast path: fresh cache, every requested ticker already populated.
    if cache_fresh and not missing:
        print(f"  [CACHE] Earnings dates loaded from disk "
              f"({age:.1f} days old, {len(cached)} entries, "
              f"{len(requested)} requested)")
        return cached

    if SNAPSHOT_MODE:
        raise _snapshot_miss(
            "earnings_dates", EARNINGS_CACHE,
            f"file present, readable, age < {EARNINGS_TTL_DAYS} day, "
            f"and every requested ticker present "
            f"(missing {len(missing)}, e.g. {missing[:3]})")

    if cache_fresh:
        earnings = dict(cached)
        to_fetch = missing
        print(f"  [CACHE] Earnings fresh ({age:.1f} days old): "
              f"{len(cached)} cached, fetching {len(to_fetch)} new "
              f"tickers...")
    else:
        earnings = {}
        to_fetch = requested
        if cache_present:
            print(f"  [CACHE] Earnings stale ({age:.1f} days old); "
                  f"refetching {len(to_fetch)} tickers...")
        else:
            print(f"  Fetching earnings dates for {len(to_fetch)} "
                  f"tickers (no cache)...")

    for tkr in to_fetch:
        earnings[tkr] = _fetch_one_earnings(tkr, start_ts, end_ts)

    try:
        serializable = {
            t: [d.isoformat() for d in dates]
            for t, dates in earnings.items()
        }
        with open(EARNINGS_CACHE, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
        print(f"  [CACHE] Earnings written: {len(earnings)} total entries "
              f"({len(to_fetch)} fresh, "
              f"{len(earnings) - len(to_fetch)} pre-existing)")
    except Exception as e:
        print(f"  [CACHE] Failed to write earnings cache ({e})")
    return earnings


def _has_earnings_within(
    earnings_dates: dict[str, list[pd.Timestamp]],
    ticker: str,
    as_of: pd.Timestamp,
    window_days: int,
) -> int | None:
    """If ticker has an earnings date within window_days (calendar) after
    as_of, return the number of days until it; otherwise return None."""
    dates = earnings_dates.get(ticker, [])
    if not dates:
        return None
    as_of = pd.Timestamp(as_of).normalize()
    window_end = as_of + pd.Timedelta(days=window_days)
    for d in dates:
        if as_of <= d <= window_end:
            return (d - as_of).days
    return None


def _rank_normalize(values: dict[str, float], higher_is_better: bool = True) -> dict[str, float]:
    """Rank-normalize values to [0, 1]."""
    if not values:
        return {}
    sorted_items = sorted(values.items(), key=lambda x: x[1],
                          reverse=higher_is_better)
    n = len(sorted_items)
    return {tkr: 1 - i / max(n - 1, 1) for i, (tkr, _) in enumerate(sorted_items)}


def score_fundamentals(fund_data: dict[str, dict], tickers: list[str]) -> dict[str, float]:
    """Score tickers 0-1 based on fundamentals (equal-weighted 5 metrics)."""
    rev = _rank_normalize({t: fund_data[t]["revenue_growth"] for t in tickers})
    margin = _rank_normalize({t: fund_data[t]["profit_margin"] for t in tickers})
    pe = _rank_normalize({t: fund_data[t]["pe_ratio"] for t in tickers},
                         higher_is_better=False)
    de = _rank_normalize({t: fund_data[t]["debt_to_equity"] for t in tickers},
                         higher_is_better=False)
    roe = _rank_normalize({t: fund_data[t]["roe"] for t in tickers})
    return {t: (rev[t] + margin[t] + pe[t] + de[t] + roe[t]) / 5 for t in tickers}


def score_technical(
    price_data: dict[str, pd.DataFrame],
    featured_data: dict[str, pd.DataFrame],
    date,
    tickers: list[str],
) -> dict[str, float]:
    """Score tickers 0-1 based on 5 technical signals."""
    scores = {}
    for tkr in tickers:
        pts = 0
        if tkr not in featured_data or date not in featured_data[tkr].index:
            scores[tkr] = 0.0
            continue
        row = featured_data[tkr].loc[date]
        price = price_data[tkr].loc[date, "Close"] if (
            tkr in price_data and date in price_data[tkr].index) else 0

        if "sma_200_raw" in row and price > row["sma_200_raw"]:
            pts += 1
        if "sma_50_raw" in row and "sma_200_raw" in row and row["sma_50_raw"] > row["sma_200_raw"]:
            pts += 1
        if "rsi_14_raw" in row and 40 <= row["rsi_14_raw"] <= 60:
            pts += 1
        if "dist_52w_high_raw" in row and row["dist_52w_high_raw"] >= -0.10:
            pts += 1
        if tkr in featured_data and "obv_raw" in featured_data[tkr].columns:
            obv_series = featured_data[tkr]["obv_raw"]
            if date in obv_series.index:
                idx = obv_series.index.get_loc(date)
                if idx >= 20 and obv_series.iloc[idx] > obv_series.iloc[idx - 20]:
                    pts += 1
        scores[tkr] = pts / 5.0
    return scores


def select_top_tickers(
    comp_scores: dict[str, dict],
    sector_map: dict[str, str],
    top_n: int = TOP_N,
    max_per_sector: int = MAX_PER_SECTOR,
    currently_held: set[str] | None = None,
) -> list[str]:
    """Pick the top_n tickers by composite score with two filters:

    1. GOOG/GOOGL dedup: if GOOG is already held, skip GOOGL (and vice versa)
       and take the next highest-scoring ticker instead.
    2. Sector cap: no more than max_per_sector tickers from the same sector.
    """
    held = currently_held or set()
    ranked = sorted(comp_scores, key=lambda t: comp_scores[t]["composite"], reverse=True)

    selected: list[str] = []
    sector_counts: dict[str, int] = {}
    for tkr in ranked:
        if len(selected) >= top_n:
            break
        # GOOG/GOOGL dedup
        if tkr == "GOOGL" and ("GOOG" in held or "GOOG" in selected):
            continue
        if tkr == "GOOG" and ("GOOGL" in held or "GOOGL" in selected):
            continue
        # Sector cap
        sector = sector_map.get(tkr, "other")
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(tkr)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    return selected


# =============================================================================
# LIQUIDITY FILTER (sp1500 universe expansion)
# =============================================================================

# 30 trading days ≈ ~6 calendar weeks. Long enough to smooth single-day
# volume spikes (earnings, index rebalance), short enough to track recent
# liquidity rather than 6-month-old reality.
_LIQUIDITY_LOOKBACK_TRADING_DAYS = 30


def filter_candidates_by_liquidity(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    date,
    threshold_usd: float,
    lookback_days: int = _LIQUIDITY_LOOKBACK_TRADING_DAYS,
) -> list[str]:
    """Return the subset of ``tickers`` whose trailing dollar-volume meets
    the threshold on ``date``.

    Average dollar volume = mean(Close × Volume) over the last
    ``lookback_days`` trading days ending at (and including) ``date``.
    Tickers missing from ``price_data`` or with no rows up to ``date`` are
    dropped silently — the rebalance loop already requires a price row at
    ``date`` before scoring, so this matches existing behavior.

    Per-day filter (the user can be in the universe but excluded today
    for low recent volume) — preserves the rest of the strategy's logic
    bit-identically when ``threshold_usd`` is 0 or no tickers are below it.

    Tested independently — see test_liquidity_filter() in tests/.
    """
    if threshold_usd is None or threshold_usd <= 0:
        return list(tickers)

    keep: list[str] = []
    date_ts = pd.Timestamp(date)
    for tkr in tickers:
        df = price_data.get(tkr)
        if df is None or df.empty:
            continue
        if "Close" not in df.columns or "Volume" not in df.columns:
            continue
        # Window ends at/including `date`. Tickers with <lookback rows in
        # the window get judged on whatever they have — the universe data
        # fetch backfills to 2018 so this only fires for very new IPOs.
        window = df.loc[:date_ts].tail(lookback_days)
        if window.empty:
            continue
        dollar_vol = (window["Close"] * window["Volume"]).dropna()
        if dollar_vol.empty:
            continue
        avg_dollar_vol = float(dollar_vol.mean())
        if avg_dollar_vol >= threshold_usd:
            keep.append(tkr)
    return keep


def compute_composite_scores(
    tickers: list[str],
    fund_data: dict[str, dict],
    price_data: dict[str, pd.DataFrame],
    featured_data: dict[str, pd.DataFrame],
    model_scores: dict[str, float],
    date,
    config: BacktestConfig | None = None,
    active_tunables: dict | None = None,
) -> dict[str, dict]:
    """Compute composite scores with full breakdown for each ticker.

    ``config`` defaults to ``BacktestConfig()`` so existing callers (e.g.
    main.py) that don't supply one keep their previous behavior.

    ``active_tunables`` (regime-dependent runs) supplies the 4 composite
    weights to apply at this rebalance. If None (legacy mode), the
    weights come straight from config.weight_*. The keys expected on the
    dict are weight_fundamental / weight_technical / weight_model /
    weight_alt — the same shape config.get_active_tunables returns.
    """
    if config is None:
        config = BacktestConfig()
    if active_tunables is None:
        active_tunables = {
            "weight_fundamental": config.weight_fundamental,
            "weight_technical":   config.weight_technical,
            "weight_model":       config.weight_model,
            "weight_alt":         config.weight_alt,
        }
    # fund_data filter: skip tickers missing fundamentals (e.g., recent
    # universe entrants like ARM/GEHC/GEV/VLTO that yfinance hasn't
    # surfaced via .info yet). Without this, score_fundamentals raises
    # KeyError on the dict comprehension over missing tickers.
    available = [t for t in tickers if t in model_scores and t in fund_data]
    # Universe blacklist (research-time): drop named tickers from the
    # candidate pool BEFORE scoring so they cannot appear in the top-N
    # selection. Default empty list = no-op (bit-identical to pre-blacklist
    # behavior). Held positions in a blacklisted name (irrelevant for
    # fresh runs) fall out at the next rebalance because they're not in
    # the new top-N. See BacktestConfig.universe_blacklist.
    if config.universe_blacklist:
        bl = set(config.universe_blacklist)
        available = [t for t in available if t not in bl]
    f_scores = score_fundamentals(fund_data, available)
    t_scores = score_technical(price_data, featured_data, date, available)
    # Alt bucket: equal-weight mean of registered alt signals (currently
    # empty registry → 0.5 neutral for every ticker, contributing a
    # constant offset that doesn't affect ranking). Adding Group D
    # signals later doesn't require changes here — they appear via the
    # registry in alt_signals.py.
    a_scores = score_alt_bucket(available, date)

    result = {}
    for tkr in available:
        f = f_scores.get(tkr, 0)
        t = t_scores.get(tkr, 0)
        m = model_scores.get(tkr, 0.5)
        a = a_scores.get(tkr, 0.5)
        composite = (f * active_tunables["weight_fundamental"]
                     + t * active_tunables["weight_technical"]
                     + m * active_tunables["weight_model"]
                     + a * active_tunables["weight_alt"])
        result[tkr] = {
            "fundamental": f,
            "technical":   t,
            "model":       m,
            "alt":         a,
            "composite":   composite,
        }
    return result


# =============================================================================
# BACKTEST
# =============================================================================

# --- Continuous-sizing endpoints (V3 study mechanic) -----------------------
# Hardcoded per spec: position count interpolates between these two values
# when both BacktestConfig.high_macro_threshold and low_macro_threshold are
# set. Future studies can promote these to tunables; this study is testing
# the mechanic.
_CONTINUOUS_SIZING_HIGH_POS = 5    # signal >= high_macro_threshold
_CONTINUOUS_SIZING_LOW_POS  = 15   # signal <= low_macro_threshold


def is_continuous_sizing(config: BacktestConfig) -> bool:
    """True iff continuous-sizing mode is active for this config (i.e.,
    BOTH high_macro_threshold and low_macro_threshold are set). When False,
    the rebalance loop falls through to the existing
    active_tunables["position_count"] path — bit-identical to pre-this-
    commit behavior for any config that doesn't set the new fields."""
    return (config.high_macro_threshold is not None
            and config.low_macro_threshold is not None)


def compute_target_position_count(macro_signal: float,
                                  config: BacktestConfig) -> int:
    """Map today's macro_signal to a target position count.

    - signal >= high_macro_threshold:  return _CONTINUOUS_SIZING_HIGH_POS (5)
    - signal <= low_macro_threshold:   return _CONTINUOUS_SIZING_LOW_POS (15)
    - in between: linear interpolation on the line through
      (low, LOW_POS) and (high, HIGH_POS), rounded to int.

    Defensive against the degenerate hi <= lo case (returns the high
    endpoint). The objective_fn rejects such configs with sentinel before
    the backtest runs, so this branch should never fire in production —
    but the helper is callable from tests/ad-hoc scripts that bypass
    objective_fn, hence the guard.
    """
    hi = config.high_macro_threshold
    lo = config.low_macro_threshold
    if hi is None or lo is None or hi <= lo:
        return _CONTINUOUS_SIZING_HIGH_POS
    if macro_signal >= hi:
        return _CONTINUOUS_SIZING_HIGH_POS
    if macro_signal <= lo:
        return _CONTINUOUS_SIZING_LOW_POS
    frac = (macro_signal - lo) / (hi - lo)   # 0 at lo (15 pos), 1 at hi (5 pos)
    n = (_CONTINUOUS_SIZING_LOW_POS
         + frac * (_CONTINUOUS_SIZING_HIGH_POS - _CONTINUOUS_SIZING_LOW_POS))
    return int(round(n))


def _sizing_regime_bucket(target: int) -> str:
    """Bucket label for sizing_decisions.parquet. Matches the spec's
    high_conviction_5pos / interpolated_8_to_12 / diversified_15pos
    naming, with intermediate buckets named to make grep-able analysis
    obvious."""
    if target <= 6:  return "high_conviction_5pos"
    if target >= 13: return "diversified_15pos"
    return "interpolated_8_to_12"


def run_backtest(
    featured_data: dict[str, pd.DataFrame],
    price_data: dict[str, pd.DataFrame],
    split_date: str | None = None,
    fund_data: dict[str, dict] | None = None,
    sector_map: dict[str, str] | None = None,
    earnings_dates: dict[str, list[pd.Timestamp]] | None = None,
    model=None,
    config: BacktestConfig | None = None,
    legacy_predict: bool = False,
    legacy_scoring: bool = False,
    market_data: dict[str, pd.DataFrame] | None = None,
    compute_rolling_metrics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Run a single-sleeve monthly-rebalance backtest.

    100% of starting_capital allocated to top ``position_count`` stocks
    by composite score. Rebalances every ``rebalance_frequency_days``
    trading days. Dynamic SPY sizing reduces position sizes by 75% when
    SPY 20-day return is negative; macro overlay further sizes down to
    50% when the macro score drops below ``macro_threshold_low``.

    If ``model`` is provided it is used directly (preferred: it was trained
    in-memory this process, so there's no chance of disk state from a future
    training run bleeding in). If None, falls back to loading from disk and
    runs the file-mtime lookahead guard.

    ``config`` defaults to ``BacktestConfig()`` (production parameters).
    All previously-hardcoded values (cash, weights, thresholds, fees,
    stop-loss bounds) read from the config object.

    Returns (portfolio_df, trades_df, latest_scores, final_holdings).
    """
    if config is None:
        config = BacktestConfig()
    if sector_map is None:
        sector_map = SECTOR_MAP
    if earnings_dates is None:
        earnings_dates = {}

    all_dates = sorted(
        set().union(*(df.index for df in featured_data.values()))
    )
    if split_date is not None:
        split_dt = pd.Timestamp(split_date)
        test_dates = [d for d in all_dates if d >= split_dt]
        split_idx = all_dates.index(test_dates[0])
    else:
        split_idx = int(len(all_dates) * 0.8)
        test_dates = all_dates[split_idx:]

    if model is None:
        model = load_model()
        # --- Lookahead bias guard ---
        # Prefer the sidecar (records the actual training+eval cutoff)
        # over the file-mtime heuristic, which fires after every retrain
        # regardless of what data the model actually saw. Fall back to
        # mtime when no sidecar exists, so legacy models still get the
        # rough check.
        earliest_test = test_dates[0]
        meta_path = os.path.join(os.path.dirname(MODEL_PATH),
                                 "xgb_model.meta.json")
        leak_reason = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    model_meta = json.load(f)
                eval_cutoff = pd.Timestamp(model_meta["eval_cutoff"])
                if eval_cutoff >= earliest_test:
                    leak_reason = (
                        f"model eval_cutoff {eval_cutoff.date()} >= backtest "
                        f"start {earliest_test.date()} (sidecar)")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                leak_reason = f"sidecar unreadable ({e})"
        else:
            model_mtime = pd.Timestamp(os.path.getmtime(MODEL_PATH), unit="s")
            if model_mtime > earliest_test:
                leak_reason = (
                    f"model mtime {model_mtime.date()} > backtest start "
                    f"{earliest_test.date()} (no sidecar to verify cutoff)")

        if leak_reason:
            print("=" * 70)
            print("WARNING: POSSIBLE LOOKAHEAD BIAS")
            print(f"  {leak_reason}")
            print("  Retrain the model with an earlier cutoff, or move the")
            print("  backtest split_date forward past the model's eval_cutoff.")
            print("=" * 70)

    # SPY + VIX for dynamic sizing / crisis gating. Routed through the
    # per-ticker price cache so back-to-back backtests read bit-identical
    # market data instead of hitting yfinance fresh each time.
    spy_start = (all_dates[max(0, split_idx - 30)]).strftime("%Y-%m-%d")
    spy_end = (all_dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    market_cache_dir = os.path.join(os.path.dirname(__file__),
                                    "..", "models", "price_cache")
    if market_data is not None:
        # Reuse preloaded SPY+VIX from optuna_runner._load_shared_data.
        # Avoids per-trial yfinance.download which is not thread-safe and
        # under n_jobs>1 produces InvalidIndexError races at fetch_data.py:438.
        market = market_data
    else:
        market = get_stock_data_cached(["SPY", "^VIX"], spy_start, spy_end,
                                       cache_dir=market_cache_dir)
    spy = market.get("SPY", pd.DataFrame())
    vix = market.get("^VIX", pd.DataFrame())
    if not spy.empty:
        spy_ret_20d = spy["Close"].pct_change(periods=20)
        spy_ret_5d  = spy["Close"].pct_change(periods=5)
    else:
        spy_ret_20d = spy_ret_5d = pd.Series(dtype=float)
    vix_close = vix["Close"] if not vix.empty else pd.Series(dtype=float)
    vix_20d_peak = vix_close.rolling(window=20).max()

    # Macro signals: fetched once, scored per rebalance as a pure position-
    # sizing overlay (never a model feature — we keep it out of FEATURE_COLS
    # to avoid leaking macro state into the composite's model component).
    macro_df = fetch_macro_signals(spy_start, spy_end)

    # Analyst consensus: small composite-score tiebreaker. We fetch once per
    # backtest (7-day TTL cache); targets shift slowly so using today's
    # consensus for historical rebalances introduces minor lookahead on
    # recent dates — acceptable given the 0.05 weight and the overwhelming
    # cost of trying to reconstruct historical analyst snapshots.
    analyst_data = fetch_analyst_targets(list(featured_data.keys()))

    if fund_data is None:
        fund_data = {t: {"revenue_growth": 0, "profit_margin": 0,
                         "pe_ratio": 100, "debt_to_equity": 0, "roe": 0}
                     for t in featured_data}

    cash = config.starting_capital
    positions: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    entry_dates: dict[str, pd.Timestamp] = {}   # ticker -> entry date (for min hold)
    stop_prices: dict[str, float] = {}   # per-position ATR-based stop level
    fees_total = 0.0
    days_since_rebal = config.rebalance_frequency_days  # force rebalance on day 1
    daily_log = []
    trades = []
    latest_scores: dict[str, dict] = {}

    # Regime-dependent accounting (zero in legacy mode; populated in
    # regime-dependent runs and surfaced via portfolio_df.attrs at the end).
    regime_rebalances = {"defensive": 0, "offensive": 0}
    regime_trades     = {"defensive": 0, "offensive": 0}

    # Continuous-sizing decision log (one row per rebalance when active;
    # empty otherwise). Surfaced via portfolio_df.attrs["sizing_decisions"]
    # at the end and persisted as sizing_decisions.parquet by the save path.
    sizing_decisions: list[dict] = []
    continuous_sizing = is_continuous_sizing(config)

    # Trading-day index lookup so min-hold check counts trading days, not calendar days
    date_index = {d: i for i, d in enumerate(test_dates)}

    # --- Tier 1 vectorization: pre-stack features and pre-compute every
    # model prediction in one batched call. Predictions are pure
    # functions of features (XGBoost is per-row independent in
    # inference), so order is irrelevant and the result is bit-identical
    # to the per-day path. The per-day inner loop becomes a dict lookup,
    # eliminating the ~571K .loc[date, FEATURE_COLS] calls that the
    # profile showed dominating runtime (94% of total time). ---
    predictions_by_date: dict[pd.Timestamp, dict[str, float]] | None = None
    if not legacy_scoring:
        non_empty = {tkr: df[FEATURE_COLS]
                     for tkr, df in featured_data.items()
                     if df is not None and not df.empty}
        if non_empty:
            # Concat preserves dict iteration order, so the resulting
            # MultiIndex layout is (ticker, date) grouped by ticker in
            # featured_data order. DO NOT sort_index — per-date dict
            # insertion order must match the legacy per-ticker-loop's
            # order, otherwise rank-normalize tie-breaking in
            # score_fundamentals diverges (Python's sorted() is stable,
            # so equal values resolve by input order) and trade
            # selection drifts at marginal ranks.
            stacked = pd.concat(non_empty)
            stacked = stacked[list(FEATURE_COLS)]   # lock column order
            all_preds = model.predict(stacked)
            predictions_by_date = {}
            for (ticker, date), score in zip(stacked.index, all_preds):
                predictions_by_date.setdefault(date, {})[ticker] = float(score)
        else:
            predictions_by_date = {}

    for date in test_dates:
        # Regime decision (legacy mode is a no-op: get_active_tunables
        # ignores the macro_signal arg and returns the single-value
        # tunables). Computed once per day so both the rebalance gate
        # and the macro-sizing block downstream can reuse the value.
        macro_score_today = compute_macro_score(macro_df, date)
        active = config.get_active_tunables(macro_score_today)
        if config.single_regime_mode:
            # All rebalances/trades attributed to "offensive" in single-
            # regime mode so regime_stats accounting stays well-formed.
            regime_label = "offensive"
        elif (config.architecture == "regime-dependent"
              and config.regime_threshold is not None):
            regime_label = ("defensive"
                            if macro_score_today < config.regime_threshold
                            else "offensive")
        else:
            regime_label = None

        # --- score every ticker with XGBoost (soft target regression) ---
        if not legacy_scoring:
            # Fast path: predictions were precomputed above. O(1) dict lookup.
            model_scores = predictions_by_date.get(date, {})
        elif legacy_predict:
            # Old per-ticker loop. Kept for verification — bit-identical
            # to the vectorized path below.
            model_scores = {}
            for ticker, df in featured_data.items():
                if date not in df.index:
                    continue
                row = df.loc[[date], FEATURE_COLS]
                prob_up = float(model.predict(row)[0])
                model_scores[ticker] = prob_up
        else:
            # Per-day vectorized batch predict (pre-Tier-1 behavior).
            # Still pays per-ticker .loc[] cost to assemble the batch;
            # kept reachable via --legacy-scoring for verification.
            tickers_this_date: list[str] = []
            rows_data: list = []
            for ticker, df in featured_data.items():
                if date not in df.index:
                    continue
                tickers_this_date.append(ticker)
                rows_data.append(df.loc[date, FEATURE_COLS].values)
            if rows_data:
                batch_df = pd.DataFrame(rows_data,
                                        columns=FEATURE_COLS,
                                        index=tickers_this_date)
                preds = model.predict(batch_df)
                model_scores = {t: float(p)
                                for t, p in zip(tickers_this_date, preds)}
            else:
                model_scores = {}

        # --- Volatility-adjusted stop-loss (2.5x ATR, floor -15%, ceiling -5%).
        # Skip the sell if earnings are inside the blackout window — we'd be
        # locked out of re-entering the position right after anyway. ---
        for tkr in list(positions):
            if tkr in price_data and date in price_data[tkr].index:
                price = price_data[tkr].loc[date, "Close"]
                if tkr in stop_prices and price <= stop_prices[tkr]:
                    days_to_earn = _has_earnings_within(
                        earnings_dates, tkr, date, config.earnings_blackout_days)
                    if days_to_earn is not None:
                        print(f"    [EARNINGS] Skipping stop on {tkr} — "
                              f"earnings in {days_to_earn} days")
                        continue
                    proceeds = positions[tkr] * price
                    fee = proceeds * config.transaction_cost_pct
                    trades.append({
                        "date": date, "ticker": tkr, "action": "STOP_ATR",
                        "shares": positions[tkr], "price": price, "fee": fee,
                    })
                    cash += proceeds - fee
                    fees_total += fee
                    del positions[tkr]
                    del entry_prices[tkr]
                    del stop_prices[tkr]
                    entry_dates.pop(tkr, None)

        # --- Dynamic sizing: defensive + recovery override ---
        regime_scale = 1.0
        spy_20d_neg = (date in spy_ret_20d.index and spy_ret_20d.loc[date] < 0)

        if spy_20d_neg:
            # Check recovery signal: SPY 5d positive AND VIX dropped 15%+ from 20d peak
            spy_5d_pos = (date in spy_ret_5d.index and spy_ret_5d.loc[date] > 0)
            vix_fading = False
            if date in vix_close.index and date in vix_20d_peak.index:
                peak = vix_20d_peak.loc[date]
                if peak > 0:
                    vix_fading = (vix_close.loc[date] < peak * 0.85)

            if spy_5d_pos and vix_fading:
                regime_scale = 1.0   # recovery: cancel defensive, full size
            else:
                regime_scale = 0.25  # defensive: 75% reduction

        if date in vix_close.index and vix_close.loc[date] > 30:
            regime_scale *= 0.5  # additional 50% cut in crisis

        # --- Rebalance every active rebalance_frequency_days trading days
        # (regime-dependent runs use the active set; legacy uses the single
        # config.rebalance_frequency_days via get_active_tunables) ---
        days_since_rebal += 1
        if days_since_rebal >= active["rebalance_frequency_days"] and model_scores:
            days_since_rebal = 0
            if regime_label is not None:
                regime_rebalances[regime_label] += 1

            available = [t for t in model_scores if t in price_data
                         and date in price_data[t].index]
            # Liquidity filter (sp1500 universe expansion): drop names
            # below the trailing-30d ADV threshold on this rebalance date.
            # No-op when min_avg_daily_volume_usd is None (every pre-sp1500
            # config), so #325 and existing studies stay bit-identical.
            if config.min_avg_daily_volume_usd is not None:
                available = filter_candidates_by_liquidity(
                    available, price_data, date,
                    config.min_avg_daily_volume_usd)
            comp_scores = compute_composite_scores(
                available, fund_data, price_data, featured_data,
                model_scores, date, config=config, active_tunables=active)

            # Analyst tiebreaker: composite += analyst_weight * analyst_score
            # (default 0.05). Applied post-composite so the main weighting
            # isn't disturbed.
            for tkr, s in comp_scores.items():
                a = analyst_data.get(tkr)
                if a is not None:
                    s["analyst_score"] = a["analyst_score"]
                    s["composite"]     = (s["composite"]
                                          + config.analyst_weight
                                          * a["analyst_score"])

            latest_scores = comp_scores

            # Continuous-sizing override: when both thresholds are set,
            # target N is interpolated from today's macro signal instead
            # of using the regime-active position_count tunable. Default
            # path (continuous_sizing=False) preserves all existing
            # behavior bit-identically.
            if continuous_sizing:
                target_n = compute_target_position_count(
                    macro_score_today, config)
            else:
                target_n = active["position_count"]
            # Snapshot the pre-rebalance state for the decision log
            before_count = len(positions)
            top_tickers = select_top_tickers(
                comp_scores, sector_map,
                top_n=target_n,
                max_per_sector=config.sector_cap,
                currently_held=set(positions),
            )
            target_set = set(top_tickers)

            # Sell positions no longer in top N — unless they're still inside
            # the 5-trading-day minimum hold window (ATR stops already fired above).
            today_idx = date_index.get(date, 0)
            for tkr in list(positions):
                if tkr in target_set:
                    continue
                entry_idx = date_index.get(entry_dates.get(tkr), today_idx)
                if today_idx - entry_idx < config.min_hold_days:
                    continue  # hold through next rebalance
                if tkr in price_data and date in price_data[tkr].index:
                    sell_price = price_data[tkr].loc[date, "Close"]
                    proceeds = positions[tkr] * sell_price
                    fee = proceeds * config.transaction_cost_pct
                    trades.append({
                        "date": date, "ticker": tkr, "action": "SELL",
                        "shares": positions[tkr], "price": sell_price, "fee": fee,
                    })
                    cash += proceeds - fee
                    fees_total += fee
                    del positions[tkr]
                    del entry_prices[tkr]
                    if tkr in stop_prices:
                        del stop_prices[tkr]
                    entry_dates.pop(tkr, None)

            # Macro overlay: position-sizing based on credit + curve + VIX
            # at this date. > high: 100%, [low, high]: 75%, < low: 50%.
            # macro_score_today is already computed at top of loop.
            if macro_score_today > config.macro_threshold_high:
                macro_scale = 1.00
            elif macro_score_today >= config.macro_threshold_low:
                macro_scale = 0.75
            else:
                macro_scale = 0.50
            print(f"    [MACRO] {date.date()}: score {macro_score_today:.2f} "
                  f"- buying at {int(macro_scale * 100)}% size")

            # Buy new targets (equal-weight, scaled by regime + macro)
            new_targets = [t for t in top_tickers if t not in positions]
            if new_targets:
                budget_each = (cash * regime_scale * macro_scale) / len(new_targets)
                for tkr in new_targets:
                    days_to_earn = _has_earnings_within(
                        earnings_dates, tkr, date, config.earnings_blackout_days)
                    if days_to_earn is not None:
                        print(f"    [EARNINGS] Skipping {tkr} — "
                              f"earnings in {days_to_earn} days")
                        continue
                    buy_price = price_data[tkr].loc[date, "Close"]
                    shares = budget_each // buy_price
                    if shares < 1:
                        continue
                    cost = shares * buy_price
                    fee = cost * config.transaction_cost_pct
                    cash -= cost + fee
                    fees_total += fee
                    positions[tkr] = shares
                    entry_prices[tkr] = buy_price
                    entry_dates[tkr] = date

                    # ATR-based stop: atr_multiplier * ATR below entry,
                    # clamped to [atr_floor_pct, atr_cap_pct] of entry price.
                    min_stop = buy_price * (1 - config.atr_floor_pct)
                    max_stop = buy_price * (1 - config.atr_cap_pct)
                    fdf = featured_data.get(tkr)
                    if fdf is not None and "atr_14" in fdf.columns and date in fdf.index:
                        atr = fdf.loc[date, "atr_14"]
                        atr_stop = buy_price - active["atr_multiplier"] * atr
                        stop_prices[tkr] = min(max(atr_stop, max_stop), min_stop)
                    else:
                        stop_prices[tkr] = max_stop

                    trades.append({
                        "date": date, "ticker": tkr, "action": "BUY",
                        "shares": shares, "price": buy_price, "fee": fee,
                    })

            # Emit one row to the sizing-decision log per rebalance
            # (fires on every rebalance, not just continuous-sizing
            # runs, so analysis can compare across strategies in a
            # unified format). top_score / min_held_score read AFTER
            # this rebalance's buys+sells; top_unheld_score reads from
            # the same comp_scores snapshot but excludes current holdings.
            held_now = set(positions.keys())
            held_composites = [
                comp_scores[t]["composite"] for t in held_now
                if t in comp_scores
            ]
            unheld_composites = [
                s["composite"] for t, s in comp_scores.items()
                if t not in held_now
            ]
            sizing_decisions.append({
                "date":                          date,
                "macro_signal":                  float(macro_score_today),
                "target_position_count":         int(target_n),
                "actual_position_count_before":  int(before_count),
                "actual_position_count_after":   int(len(positions)),
                "top_score": (max(held_composites)
                              if held_composites else float("nan")),
                "min_held_score": (min(held_composites)
                                   if held_composites else float("nan")),
                "top_unheld_score": (max(unheld_composites)
                                     if unheld_composites else float("nan")),
                "regime": _sizing_regime_bucket(int(target_n)),
            })

        # --- End-of-day valuation ---
        port_value = cash
        for tkr, shares in positions.items():
            if tkr in price_data and date in price_data[tkr].index:
                port_value += shares * price_data[tkr].loc[date, "Close"]

        daily_log.append({
            "date": date,
            "portfolio_value": port_value,
            "cash": cash,
            "total_fees": fees_total,
            "n_positions": len(positions),
        })

    portfolio_df = pd.DataFrame(daily_log).set_index("date")
    trades_df = pd.DataFrame(trades)
    final_holdings = {tkr: {"shares": shares, "entry_price": entry_prices[tkr],
                            "stop_price": stop_prices.get(tkr, 0)}
                      for tkr, shares in positions.items()}

    # Regime-dependent accounting (attached only for regime-dependent
    # architecture; legacy runs leave portfolio_df.attrs without these
    # keys, so save_hypothesis_result conditionally serializes them).
    # Single-regime mode (Track B) records all rebalances/trades as
    # offensive and reports defensive_pct_* as 0 — the architecture
    # field structure exists but the switch never fires.
    if config.single_regime_mode:
        if not trades_df.empty:
            regime_trades = {"defensive": 0,
                             "offensive": int(len(trades_df))}
        portfolio_df.attrs["regime_stats"] = {
            "regime_threshold": None,
            "single_regime_mode": True,
            "rebalances": dict(regime_rebalances),
            "trades":     dict(regime_trades),
            "defensive_pct_rebalances": 0.0,
            "defensive_pct_trades":     0.0,
        }
    elif (config.architecture == "regime-dependent"
          and config.regime_threshold is not None):
        # Per-trade regime classification: each trade row gets attributed
        # to the regime active on its date, via macro_score lookup.
        if not trades_df.empty:
            trade_dates = pd.to_datetime(trades_df["date"])
            n_def = sum(
                compute_macro_score(macro_df, d) < config.regime_threshold
                for d in trade_dates)
            regime_trades = {"defensive": int(n_def),
                             "offensive": int(len(trade_dates) - n_def)}
        n_total_reb = (regime_rebalances["defensive"]
                       + regime_rebalances["offensive"])
        n_total_tr  = (regime_trades["defensive"]
                       + regime_trades["offensive"])
        portfolio_df.attrs["regime_stats"] = {
            "regime_threshold": float(config.regime_threshold),
            "rebalances": dict(regime_rebalances),
            "trades":     dict(regime_trades),
            "defensive_pct_rebalances": (
                regime_rebalances["defensive"] / n_total_reb * 100.0
                if n_total_reb else None),
            "defensive_pct_trades": (
                regime_trades["defensive"] / n_total_tr * 100.0
                if n_total_tr else None),
        }

    # Sizing decision log (one row per rebalance). Always attached, even
    # in non-continuous-sizing runs, so downstream analysis can compare
    # across strategies in a single unified format. Empty list → no
    # save-path action; non-empty → persisted as sizing_decisions.parquet.
    if sizing_decisions:
        portfolio_df.attrs["sizing_decisions"] = sizing_decisions

    # Rolling-window metrics: opt-in via kwarg. Result attached to
    # portfolio_df.attrs to avoid changing the return-tuple shape (every
    # existing caller stays bit-identical when this is False).
    if compute_rolling_metrics and not portfolio_df.empty:
        try:
            from rolling_metrics import compute_full_rolling_bundle
            from benchmarks import get_benchmark_returns
            strat_ret = portfolio_df["portfolio_value"].pct_change().dropna()
            bench_ret = get_benchmark_returns(
                "SPY",
                strat_ret.index[0] if len(strat_ret) else split_date,
                strat_ret.index[-1] if len(strat_ret) else split_date,
                market_data=market_data if market_data is not None else {"SPY": spy},
            )
            portfolio_df.attrs["rolling_metrics"] = compute_full_rolling_bundle(
                strat_ret, bench_ret)
        except Exception as e:
            # Don't crash the backtest just because rolling-metrics failed.
            # Record the error so callers can detect it.
            portfolio_df.attrs["rolling_metrics_error"] = (
                f"{type(e).__name__}: {e}")

    return portfolio_df, trades_df, latest_scores, final_holdings


def compute_stats(portfolio_df: pd.DataFrame, trades_df: pd.DataFrame):
    """Print summary statistics."""
    values = portfolio_df["portfolio_value"]
    returns = values.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    total_ret = values.iloc[-1] / values.iloc[0] - 1
    dd = ((values / values.cummax()) - 1).min()

    wins = 0
    total_closed = 0
    if not trades_df.empty:
        for ticker in trades_df["ticker"].unique():
            tkr_trades = trades_df[trades_df["ticker"] == ticker].sort_values("date")
            buys = tkr_trades[tkr_trades["action"] == "BUY"]
            sells = tkr_trades[tkr_trades["action"].isin(["SELL", "STOP", "STOP10", "STOP_ATR"])]
            pairs = min(len(buys), len(sells))
            for i in range(pairs):
                if sells.iloc[i]["price"] > buys.iloc[i]["price"]:
                    wins += 1
                total_closed += 1
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"  Period: {values.index[0].date()} to {values.index[-1].date()}")
    print(f"{'='*60}")
    print(f"  Start:       ${values.iloc[0]:,.2f}")
    print(f"  End:         ${values.iloc[-1]:,.2f}")
    print(f"  Return:      {total_ret:+.2%}")
    print(f"  Sharpe:      {sharpe:.2f}")
    print(f"  Max DD:      {dd:.2%}")
    atr_stops = len(trades_df[trades_df["action"] == "STOP_ATR"]) if not trades_df.empty else 0
    print(f"  Trades:      {len(trades_df)}")
    print(f"  Win rate:    {win_rate:.1f}%")
    print(f"  ATR stops:   {atr_stops}")
    print(f"  Total fees:  ${portfolio_df['total_fees'].iloc[-1]:,.2f}")
    print(f"  Positions:   {portfolio_df['n_positions'].iloc[-1]}")
    print(f"{'='*60}")


def plot_vs_spy(portfolio_df: pd.DataFrame):
    """Plot portfolio vs SPY."""
    start = portfolio_df.index[0].strftime("%Y-%m-%d")
    end = portfolio_df.index[-1].strftime("%Y-%m-%d")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    fig, ax = plt.subplots(figsize=(12, 5))
    port_norm = portfolio_df["portfolio_value"] / portfolio_df["portfolio_value"].iloc[0] * INITIAL_CASH
    spy_norm = spy["Close"] / spy["Close"].iloc[0] * INITIAL_CASH
    ax.plot(port_norm.index, port_norm.values,
            label="Portfolio", linewidth=1.5, color="#2563eb")
    ax.plot(spy_norm.index, spy_norm.values,
            label="SPY (Buy & Hold)", linewidth=1.5, color="#f59e0b", alpha=0.8)
    ax.set_title("Portfolio vs SPY Benchmark")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "backtest_vs_spy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a paper-trader backtest")
    parser.add_argument("--start", default="2018-01-01",
                        help="Feature window start (default 2018-01-01: full "
                             "train+validate). add_features needs ~252 trading "
                             "days of lookback so the first ~year of features "
                             "is dropped by dropna().")
    parser.add_argument("--end", default="2026-04-30",
                        help="Feature window end (default 2026-04-30)")
    parser.add_argument("--split-date", default="2024-01-01",
                        help="Backtest test-window start (default 2024-01-01)")
    parser.add_argument("--use-feature-cache", action="store_true",
                        help="Load features from "
                             "models/cache/feature_matrix.parquet; rebuilds "
                             "on cache miss.")
    parser.add_argument("--rebuild-feature-cache", action="store_true",
                        help="Force a fresh feature-cache rebuild before "
                             "running.")
    parser.add_argument("--legacy-bench", action="store_true",
                        help="Use the original NASDAQ-100 / 3-year window. "
                             "Overrides --start, --end, and the universe.")
    parser.add_argument("--legacy-predict", action="store_true",
                        help="Use the old per-ticker XGBoost predict loop "
                             "instead of the vectorized batch predict. "
                             "Bit-identical output; flag is for verification.")
    parser.add_argument("--legacy-scoring", action="store_true",
                        help="Use the per-day predict-batch assembly path "
                             "instead of the pre-stacked all-at-once "
                             "predict. Bit-identical; flag is for "
                             "verification of Tier 1 vectorization.")
    args = parser.parse_args()

    if args.legacy_bench:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
        tickers           = NASDAQ_100_TICKERS
        split_date        = None
        static_sector_map = SECTOR_MAP
        print(f"[--legacy-bench] NASDAQ-100, {start} -> {end}")
    else:
        start             = args.start
        end               = args.end
        tickers           = UNIVERSE_TICKERS
        split_date        = args.split_date
        static_sector_map = None  # combined path: build via build_sector_map
        print(f"Combined universe ({len(tickers)} tickers), "
              f"{start} -> {end}, test from {split_date}")

    use_cache = args.use_feature_cache or args.rebuild_feature_cache
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "models", "price_cache")

    if use_cache:
        from feature_cache import build_feature_matrix
        featured_data = build_feature_matrix(
            tickers, start, end,
            force=args.rebuild_feature_cache,
            price_cache_dir=cache_dir,
        )
        # Backtest still needs raw OHLCV for trade execution + valuation.
        print("Loading price data for backtest execution...")
        price_data = get_stock_data_cached(tickers, start, end,
                                           cache_dir=cache_dir)
        sector_map = (static_sector_map if static_sector_map is not None
                      else build_sector_map(list(featured_data.keys())))
    else:
        # Original code path: download fresh, compute features in-process.
        print(f"Downloading {len(tickers)} tickers...")
        price_data = get_stock_data_cached(tickers, start, end,
                                           cache_dir=cache_dir)
        print(f"Loaded {len(price_data)} tickers\n")

        print("Computing features...")
        featured_data = {}
        for ticker, df in price_data.items():
            try:
                featured_data[ticker] = add_features(df)
            except Exception as e:
                print(f"  [SKIP] {ticker}: {e}")

        sector_map = (static_sector_map if static_sector_map is not None
                      else build_sector_map(list(price_data.keys())))
        print("Adding market features...")
        featured_data = add_market_features(featured_data, price_data,
                                            sector_map)
        print(f"Features ready for {len(featured_data)} tickers\n")

    print("Fetching fundamentals...")
    fund_data = fetch_fundamentals(list(featured_data.keys()))

    print("Fetching earnings dates...")
    earn_dates = fetch_earnings_dates(list(featured_data.keys()), start, end)

    print("Running backtest...")
    portfolio_df, trades_df, scores, _ = run_backtest(
        featured_data, price_data, split_date=split_date,
        fund_data=fund_data, sector_map=sector_map,
        earnings_dates=earn_dates,
        legacy_predict=args.legacy_predict,
        legacy_scoring=args.legacy_scoring)
    compute_stats(portfolio_df, trades_df)

    print("\nPlotting vs SPY...")
    plot_vs_spy(portfolio_df)
