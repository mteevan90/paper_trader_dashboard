"""feature_cache.py - Persistent feature matrix cache.

Stores the full 24-feature matrix (z-scored) for the universe across the
requested date range as a single long-format parquet file under
``models/cache/feature_matrix.parquet`` plus a sidecar metadata JSON.

A read returns the same ``dict[ticker, DataFrame]`` shape that
``add_features`` + ``add_market_features`` produce, so backtest / Optuna
trials can swap in the cache load with no downstream changes.

TODO(post-Optuna): wire main.py (single-day score path) and dashboard.py
(full pipeline + model training) to this cache. Their data-flow patterns
differ enough that pulling them in now expands scope -- handle in a
follow-up segment.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

from features import add_features, add_market_features
from fetch_data import build_sector_map, get_stock_data_cached
from model import FEATURE_COLS

# Bump this constant whenever feature definitions change so old caches
# invalidate explicitly. Manual rather than hashing FEATURE_COLS so the
# failure mode is obvious -- when the cache invalidates, it's because a
# human edited this number, not because something subtle drifted.
FEATURE_VERSION = "v1"

# Columns preserved alongside FEATURE_COLS so backtest.score_technical can
# read raw (non-z-scored) values. Mirrors features.add_market_features.
RAW_COLS = ("sma_200_raw", "sma_50_raw", "rsi_14_raw",
            "dist_52w_high_raw", "obv_raw")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "cache"))
FEATURE_MATRIX_PATH = os.path.join(CACHE_DIR, "feature_matrix.parquet")
FEATURE_MATRIX_META = os.path.join(CACHE_DIR, "feature_matrix.meta.json")
PRICE_CACHE_DIR     = os.path.abspath(os.path.join(
    BASE_DIR, "..", "models", "price_cache"))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_feature_matrix(featured_data: dict[str, pd.DataFrame],
                         price_start: str, price_end: str,
                         attempted_tickers: list[str]) -> None:
    """Concatenate per-ticker frames into one long-format parquet plus a
    sidecar metadata JSON. Both are written through ``.tmp`` rename so a
    crash mid-write can't corrupt the canonical files.

    ``price_start`` / ``price_end`` are the raw price-data window the caller
    asked for. We store these alongside the actual feature ``date_min`` /
    ``date_max`` so cache-hit checks compare against the *requested* window
    (post-dropna feature data always starts ~252 trading days after the
    price window because of the momentum_12_1 lookback)."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    frames = []
    for tkr, df in featured_data.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df["ticker"] = tkr
        frames.append(df)
    if not frames:
        print("  [FEATURE CACHE] nothing to save (no non-empty frames)")
        return
    long_df = pd.concat(frames).sort_index()

    tmp_parquet = FEATURE_MATRIX_PATH + ".tmp"
    long_df.to_parquet(tmp_parquet, engine="pyarrow")
    os.replace(tmp_parquet, FEATURE_MATRIX_PATH)

    successful = sorted(t for t in featured_data
                        if featured_data[t] is not None
                        and not featured_data[t].empty)
    meta = {
        "feature_version": FEATURE_VERSION,
        # Tickers we tried to build features for (the caller's request).
        # Cache-hit checks compare against this — tickers that yfinance
        # can't fetch (delisted, renamed: e.g. FISV -> FI) won't appear in
        # ``successful`` but counting them as "missing" would force a
        # rebuild on every run.
        "attempted_tickers": sorted(set(attempted_tickers)),
        # Tickers that actually have rows in the parquet.
        "tickers":   successful,
        # Requested price window — used by _cache_covers to decide hit/miss.
        "price_start_req": price_start,
        "price_end_req":   price_end,
        # Actual feature range (informational; trails price window by the
        # 252-day lookback that add_features.dropna() removes).
        "date_min":  str(long_df.index.min().date()),
        "date_max":  str(long_df.index.max().date()),
        "n_rows":    int(len(long_df)),
        "n_tickers": int(long_df["ticker"].nunique()),
        "feature_cols": list(FEATURE_COLS),
        "raw_cols":     list(RAW_COLS),
        "built_at":  datetime.now(timezone.utc).isoformat(),
    }
    tmp_meta = FEATURE_MATRIX_META + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp_meta, FEATURE_MATRIX_META)

    size_mb = os.path.getsize(FEATURE_MATRIX_PATH) / 1024 / 1024
    print(f"  [FEATURE CACHE] Saved {len(long_df):,} rows x "
          f"{long_df.shape[1]} cols ({size_mb:.1f} MB) -> "
          f"{FEATURE_MATRIX_PATH}")


def _load_meta() -> dict | None:
    if not os.path.exists(FEATURE_MATRIX_META):
        return None
    try:
        with open(FEATURE_MATRIX_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [FEATURE CACHE] meta unreadable ({e})")
        return None


def _cache_covers(meta: dict, tickers: list[str],
                  start: str, end: str) -> tuple[bool, str]:
    """Return (covers, reason) for whether the cache satisfies the request.

    Compares against the *requested* price window stored in metadata, not
    the actual feature ``date_min``/``date_max`` — those trail the price
    window by ~252 trading days (the momentum_12_1 lookback) and using them
    here would force a rebuild on every call."""
    if meta.get("feature_version") != FEATURE_VERSION:
        return False, (f"version mismatch ("
                       f"{meta.get('feature_version')!r} != "
                       f"{FEATURE_VERSION!r})")
    if "price_start_req" not in meta or "price_end_req" not in meta:
        # Pre-fix cache (no requested-window metadata) — force a rebuild
        # so the new fields are populated.
        return False, "metadata missing price_start_req/price_end_req"
    cached_start = pd.Timestamp(meta["price_start_req"])
    cached_end   = pd.Timestamp(meta["price_end_req"])
    req_start    = pd.Timestamp(start)
    req_end      = pd.Timestamp(end)
    if cached_start > req_start:
        return False, (f"cache built from {cached_start.date()} "
                       f"> requested {req_start.date()}")
    if cached_end < req_end:
        return False, (f"cache built through {cached_end.date()} "
                       f"< requested {req_end.date()}")
    # Compare against attempted_tickers (what the cache was built for),
    # NOT the tickers that ended up with data. Permanently-unfetchable
    # tickers like FISV stay out of ``tickers`` but were attempted, so
    # repeating the same request shouldn't trigger a rebuild.
    attempted = set(meta.get("attempted_tickers", []))
    if not attempted:
        # Older metadata without attempted_tickers — force rebuild.
        return False, "metadata missing attempted_tickers"
    missing = sorted(set(tickers) - attempted)
    if missing:
        sample = missing[:3]
        return False, f"{len(missing)} tickers not attempted (e.g. {sample})"
    return True, ""


def _load_feature_matrix(tickers: list[str],
                         start: str, end: str
                         ) -> dict[str, pd.DataFrame] | None:
    """Load the cached matrix sliced to the requested universe / window.
    Returns None on any miss."""
    meta = _load_meta()
    if meta is None:
        return None
    covers, reason = _cache_covers(meta, tickers, start, end)
    if not covers:
        print(f"  [FEATURE CACHE] miss: {reason}")
        return None
    if not os.path.exists(FEATURE_MATRIX_PATH):
        print("  [FEATURE CACHE] meta present but parquet missing - rebuilding")
        return None

    df = pd.read_parquet(FEATURE_MATRIX_PATH, engine="pyarrow")
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    df = df[df["ticker"].isin(tickers)]

    out: dict[str, pd.DataFrame] = {}
    for tkr, g in df.groupby("ticker", sort=False):
        out[tkr] = g.drop(columns="ticker").sort_index()

    try:
        built_at = datetime.fromisoformat(meta["built_at"])
        age_h = (datetime.now(timezone.utc) - built_at).total_seconds() / 3600.0
        age_str = f", built {age_h:.1f}h ago"
    except Exception:
        age_str = ""
    print(f"  [FEATURE CACHE] hit: {len(out)} tickers, "
          f"price window {meta.get('price_start_req')} -> "
          f"{meta.get('price_end_req')} "
          f"(features {meta['date_min']} -> {meta['date_max']})"
          f"{age_str}, version {meta['feature_version']}")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    tickers: list[str],
    start: str,
    end: str,
    *,
    force: bool = False,
    price_cache_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Top-level entry point: return the feature matrix as
    ``dict[ticker, DataFrame]``.

    On cache hit (matching version + cache covers requested window + ticker
    set), loads from parquet and returns instantly. On miss (or
    ``force=True``) rebuilds via the existing ``add_features`` +
    ``add_market_features`` pipeline and persists the result.

    Note: ``add_features`` requires ~252 trading days of price-history
    lookback. Pass a ``start`` that gives that headroom -- the price cache
    already stores deeper history, so the actual fetch is just a slice.
    """
    if price_cache_dir is None:
        price_cache_dir = PRICE_CACHE_DIR

    if not force:
        cached = _load_feature_matrix(tickers, start, end)
        if cached is not None:
            return cached
    else:
        print("  [FEATURE CACHE] force=True - rebuilding from scratch")

    print(f"  [FEATURE CACHE] Building feature matrix: "
          f"{len(tickers)} tickers, {start} -> {end}")
    price_data = get_stock_data_cached(tickers, start, end,
                                       cache_dir=price_cache_dir)

    featured: dict[str, pd.DataFrame] = {}
    for tkr, df in price_data.items():
        try:
            featured[tkr] = add_features(df)
        except Exception as e:
            print(f"  [FEATURE CACHE] add_features failed for {tkr}: {e}")

    sector_map = build_sector_map(list(price_data.keys()))
    featured = add_market_features(featured, price_data, sector_map)

    _save_feature_matrix(featured, price_start=start, price_end=end,
                         attempted_tickers=list(tickers))
    return featured


def clear_feature_cache() -> int:
    """Delete the parquet + metadata files. Returns the number of files
    actually removed (0, 1, or 2)."""
    removed = 0
    for path in (FEATURE_MATRIX_PATH, FEATURE_MATRIX_META):
        if os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except OSError as e:
                print(f"  [FEATURE CACHE] could not delete {path}: {e}")
    print(f"  [FEATURE CACHE] cleared {removed} cache file(s) from {CACHE_DIR}")
    return removed
