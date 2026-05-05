"""
macro_signals.py -FRED + CBOE macro regime signals with local caching.

Exposes:
  fetch_macro_signals(start, end)  -> DataFrame of HY spread, 10y-2y, VIX
  compute_macro_score(macro_df, date) -> float in [0, 1] (1 = very bullish)
  fetch_cboe_putcall()             -> DataFrame of total put/call ratio

Uses the FRED_API_KEY from ../.env (one directory above src/) via python-dotenv.
All data is cached under ../models/cache/ and fetched incrementally where
possible so repeat calls are cheap.
"""

import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred


# --- Paths ------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, "..", ".env")

# --- Cache snapshot system (PAPER_TRADER_DATA_ROOT) -----------------------
_DEFAULT_DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
DATA_ROOT = os.environ.get("PAPER_TRADER_DATA_ROOT", _DEFAULT_DATA_ROOT)
SNAPSHOT_MODE = os.environ.get("PAPER_TRADER_DATA_ROOT") is not None

CACHE_DIR   = os.path.join(DATA_ROOT, "cache")
MACRO_CACHE     = os.path.join(CACHE_DIR, "macro_signals.parquet")
MACRO_META      = os.path.join(CACHE_DIR, "macro_signals.meta.json")
PUTCALL_CACHE   = os.path.join(CACHE_DIR, "putcall.csv")
ANALYST_CACHE   = os.path.join(CACHE_DIR, "analyst_targets.json")
ANALYST_META    = os.path.join(CACHE_DIR, "analyst_targets.meta.json")


def _snapshot_miss(cache_name: str, path: str, required: str) -> "RuntimeError":
    return RuntimeError(
        f"[CACHE_SNAPSHOT_MISS] Cache miss in snapshot mode.\n"
        f"  Cache: {cache_name}\n"
        f"  Path: {path}\n"
        f"  Snapshot: {os.environ.get('PAPER_TRADER_DATA_ROOT')}\n"
        f"  Required: {required}\n"
        f"  Action: re-create snapshot with current data, or fix snapshot to "
        f"include this data."
    )

# Bump these when the cache schema or transform definitions change so old
# caches invalidate explicitly. Manual rather than auto-derived so a rebuild
# is always traceable to a human-edited number.
# v2 (segment 11): added NFCI, SAHMREALTIME, T10Y3M to FRED_SERIES;
# compute_macro_score now averages 7 components instead of 4.
MACRO_VERSION   = "v2"
ANALYST_VERSION = "v1"

FRED_SERIES = {
    "hy_spread": "BAMLH0A0HYM2",   # ICE BofA US High Yield OAS (daily)
    "yc_slope":  "T10Y2Y",         # 10yr minus 2yr Treasury spread (daily)
    "vix":       "VIXCLS",         # CBOE VIX (daily, FRED mirror)
    "nfci":      "NFCI",           # Chicago Fed Financial Conditions Index
                                   # (weekly, Wednesdays).  Negative = loose.
    "sahm":      "SAHMREALTIME",   # Sahm Rule recession indicator (monthly).
                                   # Triggers at >=0.5.
    "yc_3m":     "T10Y3M",         # 10yr minus 3-month Treasury spread
                                   # (daily). Per NY Fed, a stronger
                                   # recession predictor than 10y-2y.
}

# CBOE has been gating these CSVs intermittently; we try TOTAL first, then
# EQUITY, and if both 403 we synthesize a proxy from VIX so downstream callers
# never see an empty series.
PUTCALL_URLS = [
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/PC_1_CBOE_TOTAL.csv",
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/PC_1_CBOE_EQUITY.csv",
]

ANALYST_TTL_DAYS = 7  # Analyst targets update weekly at best

load_dotenv(DOTENV_PATH)
_FRED_API_KEY = os.getenv("FRED_API_KEY")


def _cache_age_days(path: str) -> float | None:
    """Return age of ``path`` in days, or None if missing."""
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 86400.0


# --- FRED fetch + cache -----------------------------------------------------

def _get_fred() -> Fred:
    if not _FRED_API_KEY:
        raise RuntimeError(
            f"FRED_API_KEY missing. Add it to {DOTENV_PATH}")
    return Fred(api_key=_FRED_API_KEY)


def _fetch_fred_series(fred: Fred, start: str, end: str) -> pd.DataFrame:
    """Download every FRED series in one pass, align on date index."""
    pieces = []
    for col, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id,
                                observation_start=start,
                                observation_end=end)
            s.name = col
            pieces.append(s)
        except Exception as e:
            print(f"  [MACRO] {series_id} fetch failed ({e})")
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, axis=1).sort_index()
    df.index = pd.to_datetime(df.index)
    # Forward-fill only. Daily series (hy_spread, yc_slope, vix, yc_3m)
    # have a value every business day so ffill is a no-op for them.
    # Weekly NFCI (Wednesdays) and monthly SAHMREALTIME (first of month)
    # leave NaN on most business days; ffill propagates the most recent
    # *published* value forward so any business-day lookup returns a
    # non-NaN reading. Backfill is deliberately NOT used — it would
    # propagate future readings to past dates (lookahead bias).
    df = df.ffill()
    return df


def _save_macro_meta(df: pd.DataFrame, requested_start: str,
                     requested_end: str) -> None:
    meta = {
        "version":         MACRO_VERSION,
        "fred_series":     dict(FRED_SERIES),
        "date_min":        str(df.index.min().date()) if not df.empty else None,
        "date_max":        str(df.index.max().date()) if not df.empty else None,
        "n_rows":          int(len(df)),
        "last_request_start": requested_start,
        "last_request_end":   requested_end,
        "built_at":        datetime.utcnow().isoformat() + "Z",
    }
    tmp = MACRO_META + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(tmp, MACRO_META)
    except Exception as e:
        print(f"  [MACRO] Failed to write meta ({e})")


def _load_macro_meta() -> dict | None:
    if not os.path.exists(MACRO_META):
        return None
    try:
        with open(MACRO_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [MACRO] meta unreadable ({e})")
        return None


def fetch_macro_signals(start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame of HY spread, 10y-2y slope, and VIX over [start, end].

    Cached to ``models/cache/macro_signals.parquet`` with sidecar
    ``macro_signals.meta.json``. The sidecar pins ``MACRO_VERSION`` and the
    FRED series IDs the cache was built with — version mismatch forces a
    rebuild from scratch so a series swap can never silently use stale data.
    Within the same version: backfill earlier history if requested, forward-
    fetch newer rows if requested ``end`` is past the cached tail.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    # Version + schema check via sidecar — invalidate if either drifts.
    meta = _load_macro_meta()
    version_ok = meta is not None and meta.get("version") == MACRO_VERSION \
        and meta.get("fred_series") == dict(FRED_SERIES)

    cached: pd.DataFrame | None = None
    if version_ok and os.path.exists(MACRO_CACHE):
        try:
            cached = pd.read_parquet(MACRO_CACHE)
            cached.index = pd.to_datetime(cached.index)
            cached = cached.sort_index()
        except Exception as e:
            print(f"  [MACRO] Cache unreadable ({e}) -refetching")
            cached = None
    elif meta is not None and not version_ok:
        if SNAPSHOT_MODE:
            raise _snapshot_miss(
                "macro_signals", MACRO_CACHE,
                f"version match (cache {meta.get('version')!r} != current "
                f"{MACRO_VERSION!r})")
        print(f"  [MACRO] Version/schema mismatch (cache "
              f"{meta.get('version')!r} vs {MACRO_VERSION!r}) - rebuilding")

    if cached is not None and not cached.empty:
        # Backfill earlier history if the requested start is before the cache.
        if start_ts < cached.index.min():
            if SNAPSHOT_MODE:
                raise _snapshot_miss(
                    "macro_signals", MACRO_CACHE,
                    f"date range covers requested start {start_ts.date()} "
                    f"(cache starts {cached.index.min().date()})")
            fred = _get_fred()
            back_end = (cached.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            back = _fetch_fred_series(fred, start, back_end)
            if not back.empty:
                cached = pd.concat([back, cached])
                cached = cached[~cached.index.duplicated(keep="last")].sort_index()
                print(f"  [MACRO] Backfilled {len(back)} rows "
                      f"(from {cached.index.min().date()})")

        # Forward-fetch new rows since the last cached date.
        last_date = cached.index.max()
        fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if pd.Timestamp(fetch_start) <= end_ts:
            if SNAPSHOT_MODE:
                raise _snapshot_miss(
                    "macro_signals", MACRO_CACHE,
                    f"date range covers requested end {end_ts.date()} "
                    f"(cache ends {last_date.date()})")
            fred = _get_fred()
            fresh = _fetch_fred_series(fred, fetch_start, end)
            if not fresh.empty:
                cached = pd.concat([cached, fresh])
                cached = cached[~cached.index.duplicated(keep="last")].sort_index()
                print(f"  [MACRO] Appended {len(fresh)} new rows "
                      f"(through {cached.index.max().date()})")

        # Live mode may have updated cached above; persist if so.
        if not SNAPSHOT_MODE:
            try:
                cached.to_parquet(MACRO_CACHE)
                _save_macro_meta(cached, start, end)
            except Exception:
                pass
        print(f"  [MACRO] Loaded from disk ({len(cached)} rows, "
              f"{cached.index.min().date()} to {cached.index.max().date()})")
        return cached.loc[start_ts:end_ts]

    # First run — full download.
    if SNAPSHOT_MODE:
        raise _snapshot_miss(
            "macro_signals", MACRO_CACHE,
            "parquet file present and readable")
    fred = _get_fred()
    df = _fetch_fred_series(fred, start, end)
    if df.empty:
        return df
    try:
        df.to_parquet(MACRO_CACHE)
        _save_macro_meta(df, start, end)
    except Exception as e:
        print(f"  [MACRO] Failed to write cache ({e})")
    print(f"  [MACRO] Fetched fresh ({len(df)} rows, "
          f"{df.index.min().date()} -> {df.index.max().date()})")
    return df


# --- Macro score ------------------------------------------------------------

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_macro_score(macro_df: pd.DataFrame, date) -> float:
    """Map macro conditions at ``date`` to a single score in [0, 1].

    1.0 = very bullish, 0.0 = very bearish. Seven equal-weight components:
      * HY spread level: ≤3% -> 1, ≥8% -> 0 (tight credit = bullish)
      * HY spread 20-day change: ≤-0.5pp -> 1, ≥+1pp -> 0 (tightening = bullish)
      * 10y-2y slope: ≥+1% -> 1, ≤-1% -> 0 (steep = bullish, inverted = bearish)
      * VIX level: ≤12 -> 1, ≥35 -> 0 (calm = bullish)
      * NFCI: ≤-0.5 -> 1, ≥+1.0 -> 0 (loose financial conditions = bullish)
      * Sahm Rule: ≤0.0 -> 1, ≥+0.5 -> 0 (recession trigger fires at 0.5)
      * 10y-3m slope: ≥+1% -> 1, ≤-1% -> 0 (NY-Fed-preferred recession signal)

    TODO: 10y-3m anchors may widen to ±1.5pp if score saturates during
    extreme inversions like 2023's -1.5pp; observe distribution before
    tuning.

    If data is missing a component defaults to 0.5 (neutral).
    """
    if macro_df is None or macro_df.empty:
        return 0.5
    date_ts = pd.Timestamp(date)
    hist = macro_df[macro_df.index <= date_ts]
    if hist.empty:
        return 0.5
    row = hist.iloc[-1]

    # 1. HY spread level
    hy = row.get("hy_spread", np.nan)
    hy_level_score = (1 - (hy - 3) / 5) if pd.notna(hy) else 0.5

    # 2. HY spread 20d change
    hy_change_score = 0.5
    hy_hist = hist["hy_spread"].dropna() if "hy_spread" in hist.columns else pd.Series(dtype=float)
    if len(hy_hist) > 20:
        delta = hy_hist.iloc[-1] - hy_hist.iloc[-21]
        # +1pp rise -> 0 (bearish); -0.5pp fall -> 1 (bullish).
        hy_change_score = 0.5 - delta / 1.5 * 0.5  # linear between bounds

    # 3. Yield curve slope (10y-2y)
    yc = row.get("yc_slope", np.nan)
    yc_score = (0.5 + yc / 2) if pd.notna(yc) else 0.5

    # 4. VIX level
    vix = row.get("vix", np.nan)
    vix_score = (1 - (vix - 12) / 23) if pd.notna(vix) else 0.5

    # 5. NFCI (Chicago Fed Financial Conditions). Negative = loose.
    # -0.5 -> 1.0, +1.0 -> 0.0 (linear).
    nfci = row.get("nfci", np.nan)
    nfci_score = (1.0 - (nfci + 0.5) / 1.5) if pd.notna(nfci) else 0.5

    # 6. Sahm Rule. Recession trigger fires at 0.5; we map 0->1, 0.5->0.
    sahm = row.get("sahm", np.nan)
    sahm_score = (1.0 - sahm / 0.5) if pd.notna(sahm) else 0.5

    # 7. 10y-3m yield spread (mirrors yc_slope formula; same pp scale).
    yc3m = row.get("yc_3m", np.nan)
    yc3m_score = (0.5 + yc3m / 2) if pd.notna(yc3m) else 0.5

    components = [hy_level_score, hy_change_score, yc_score, vix_score,
                  nfci_score, sahm_score, yc3m_score]
    return _clip01(sum(_clip01(c) for c in components) / 7)


# --- CBOE put/call ratio ----------------------------------------------------

def _try_cboe_url(url: str) -> pd.DataFrame | None:
    """Attempt one CBOE URL with a spoofed User-Agent; None on any failure."""
    try:
        import io, urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (paper_trader macro_signals)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw), skiprows=3)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        return df
    except Exception as e:
        print(f"  [MACRO] CBOE {url.rsplit('/', 1)[-1]} failed ({e})")
        return None


def _vix_putcall_proxy() -> pd.DataFrame:
    """Synthesize a put/call-style fear proxy from VIX / VIX_20d_avg.

    When both CBOE URLs fail we still want *something* consumers can look at
    for mean-reversion signals. A VIX ratio > 1 means current fear is above
    its 20-day average — structurally similar to an elevated put/call ratio.
    """
    try:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
        macro = fetch_macro_signals(start, end)
        if macro.empty or "vix" not in macro.columns:
            return pd.DataFrame()
        vix = macro["vix"].dropna()
        proxy = vix / vix.rolling(20, min_periods=5).mean()
        out = pd.DataFrame({"putcall_proxy": proxy}).dropna()
        return out
    except Exception as e:
        print(f"  [MACRO] VIX proxy failed ({e})")
        return pd.DataFrame()


def fetch_cboe_putcall() -> pd.DataFrame:
    """Download CBOE put/call ratio with graceful fallback.

    Order of attempts:
      1. CBOE TOTAL put/call CSV
      2. CBOE EQUITY put/call CSV (fallback, user-requested)
      3. VIX / VIX_20d_avg proxy (column 'putcall_proxy')

    Cached daily to ``putcall.csv`` regardless of which source succeeded.
    Never raises; returns an empty DataFrame only if *all* sources fail —
    never breaks the rest of the macro pipeline.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Respect the 1-day TTL on the cached file.
    age = _cache_age_days(PUTCALL_CACHE)
    if age is not None and age < 1.0:
        try:
            df = pd.read_csv(PUTCALL_CACHE, index_col=0, parse_dates=True)
            print(f"  [MACRO] Put/call loaded from disk ({age*24:.1f}h old)")
            return df
        except Exception as e:
            print(f"  [MACRO] Put/call cache unreadable ({e}) - refetching")

    for url in PUTCALL_URLS:
        df = _try_cboe_url(url)
        if df is not None and not df.empty:
            try:
                df.to_csv(PUTCALL_CACHE)
            except Exception as e:
                print(f"  [MACRO] Failed to write put/call cache ({e})")
            source = url.rsplit("/", 1)[-1]
            print(f"  [MACRO] Put/call fetched from {source} ({len(df)} rows)")
            return df

    print("  [MACRO] CBOE feeds unavailable - using VIX/VIX_20d_avg proxy")
    proxy = _vix_putcall_proxy()
    if not proxy.empty:
        try:
            proxy.to_csv(PUTCALL_CACHE)
        except Exception as e:
            print(f"  [MACRO] Failed to write put/call cache ({e})")
        print(f"  [MACRO] Put/call proxy ready ({len(proxy)} rows)")
    else:
        print("  [MACRO] Put/call unavailable - returning empty DataFrame")
    return proxy


# --- Analyst price targets --------------------------------------------------

def _upside_score(current: float, mean_target: float) -> float:
    """Linear map from upside% to [0, 1], bounded at +/-50%.

    Anchors (from user's spec):
      current $100, target $120 (+20%)  -> ~0.70
      current == target                 -> 0.50
      price above target                -> < 0.50 (bearish overhang)
    """
    if current is None or mean_target is None or current <= 0:
        return 0.5
    upside = (mean_target - current) / current
    upside = max(min(upside, 0.5), -0.5)
    return float(np.clip(0.5 + upside, 0.0, 1.0))


def _conviction_score(rec_mean: float | None) -> float:
    """Map yfinance's recommendationMean (1=strong buy, 5=sell) to [0, 1]."""
    if rec_mean is None or (isinstance(rec_mean, float) and np.isnan(rec_mean)):
        return 0.5
    if rec_mean <= 1.5:  return 1.0
    if rec_mean <= 2.0:  return 0.85
    if rec_mean <= 2.5:  return 0.7
    if rec_mean <= 3.0:  return 0.5
    return 0.2


def _save_analyst_meta(result: dict, attempted_tickers: list[str]) -> None:
    meta = {
        "version":            ANALYST_VERSION,
        "ttl_days":           ANALYST_TTL_DAYS,
        "score_formula":      "0.6*upside_score + 0.4*conviction_score",
        "min_analysts":       3,
        "attempted_tickers":  sorted(set(attempted_tickers)),
        "n_attempted":        len(set(attempted_tickers)),
        "n_with_targets":     len(result),
        "built_at":           datetime.utcnow().isoformat() + "Z",
    }
    tmp = ANALYST_META + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(tmp, ANALYST_META)
    except Exception as e:
        print(f"  [ANALYST] Failed to write meta ({e})")


def _load_analyst_meta() -> dict | None:
    if not os.path.exists(ANALYST_META):
        return None
    try:
        with open(ANALYST_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [ANALYST] meta unreadable ({e})")
        return None


def fetch_analyst_targets(tickers: list[str]) -> dict[str, dict]:
    """Fetch per-ticker analyst price targets + recommendation from yfinance.

    Returns ``{ticker: {current_price, mean_target, high_target, low_target,
    n_analysts, recommendation_mean, upside_pct, upside_score,
    conviction_score, analyst_score}}``. Tickers with fewer than 3 analyst
    opinions are skipped entirely.

    Cached to ``models/cache/analyst_targets.json`` with sidecar
    ``analyst_targets.meta.json`` (version, score formula, attempted ticker
    set, build time) and a 7-day TTL. Cache hit requires: file age < TTL,
    sidecar version matches, and the requested ticker set is a subset of
    ``attempted_tickers`` (so a universe expansion forces a refresh).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    age = _cache_age_days(ANALYST_CACHE)
    meta = _load_analyst_meta()

    version_ok = meta is not None and meta.get("version") == ANALYST_VERSION
    coverage_ok = (meta is not None
                   and set(tickers).issubset(set(meta.get("attempted_tickers", []))))

    if (age is not None and age < ANALYST_TTL_DAYS
            and version_ok and coverage_ok):
        try:
            with open(ANALYST_CACHE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"  [ANALYST] Loaded from disk ({age:.1f} days old, "
                  f"{len(cached)} tickers, version {meta.get('version')})")
            return cached
        except Exception as e:
            print(f"  [ANALYST] Cache unreadable ({e}) - refetching")
    elif meta is not None:
        if not version_ok:
            print(f"  [ANALYST] Version mismatch "
                  f"({meta.get('version')!r} != {ANALYST_VERSION!r}) "
                  f"- refetching")
        elif not coverage_ok:
            missing = sorted(set(tickers) -
                             set(meta.get("attempted_tickers", [])))
            print(f"  [ANALYST] Universe expanded "
                  f"({len(missing)} new tickers, e.g. {missing[:3]}) "
                  f"- refetching")

    if SNAPSHOT_MODE:
        raise _snapshot_miss(
            "analyst_targets", ANALYST_CACHE,
            f"file present, age < {ANALYST_TTL_DAYS} days, version match, "
            f"coverage of all {len(tickers)} tickers")

    print(f"  [ANALYST] Fetching targets for {len(tickers)} tickers...")
    result: dict[str, dict] = {}
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).info
            n = info.get("numberOfAnalystOpinions") or 0
            if n < 3:
                continue
            current = info.get("currentPrice")
            mean_t  = info.get("targetMeanPrice")
            if current is None or mean_t is None or current <= 0:
                continue

            rec      = info.get("recommendationMean")
            upside_pct = (mean_t - current) / current
            upside   = _upside_score(current, mean_t)
            convict  = _conviction_score(rec)
            analyst  = 0.6 * upside + 0.4 * convict

            result[tkr] = {
                "current_price":       float(current),
                "mean_target":         float(mean_t),
                "high_target":         float(info.get("targetHighPrice") or 0) or None,
                "low_target":          float(info.get("targetLowPrice") or 0) or None,
                "n_analysts":          int(n),
                "recommendation_mean": float(rec) if rec else None,
                "upside_pct":          float(upside_pct),
                "upside_score":        float(upside),
                "conviction_score":    float(convict),
                "analyst_score":       float(analyst),
            }
        except Exception:
            continue

    try:
        with open(ANALYST_CACHE, "w", encoding="utf-8") as f:
            json.dump(result, f)
        _save_analyst_meta(result, list(tickers))
        print(f"  [ANALYST] Fetched fresh "
              f"({len(result)}/{len(tickers)} tickers with >=3 analysts)")
    except Exception as e:
        print(f"  [ANALYST] Failed to write cache ({e})")
    return result


if __name__ == "__main__":
    print("--- FRED macro signals ---")
    df = fetch_macro_signals("2024-01-01", datetime.today().strftime("%Y-%m-%d"))
    print(df.tail())
    if not df.empty:
        today = df.index.max()
        score = compute_macro_score(df, today)
        print(f"Macro score on {today.date()}: {score:.3f}")

    print("\n--- CBOE put/call ---")
    pc = fetch_cboe_putcall()
    print(pc.tail() if not pc.empty else "(empty)")
