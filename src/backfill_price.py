"""backfill_price.py — extend a per-ticker price cache backwards in time.

The project's price cache layer (fetch_data.get_stock_data_cached) does
forward-fetch and backfill on demand, but in snapshot mode those paths
are disabled to keep snapshots reproducible. When you discover that an
existing live cache is missing earlier history (e.g., SPY only goes back
to 2023 when training needs 2018+), use this tool to extend the cache
in a single, deliberate step.

Usage:

    python src/backfill_price.py <ticker> --start <YYYY-MM-DD> [--cache-dir PATH] [--dry-run]
    python src/backfill_price.py update-snapshot-manifest <snapshot> <ticker>

The first form fetches yfinance data for [start, current_cache_min) and
prepends to the existing parquet. The second updates a snapshot's
manifest.json after you've copied the backfilled parquet into a snapshot
(otherwise the manifest's recorded size/mtime drift from reality).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BASE_DIR.parent
_DEFAULT_CACHE_DIR = str(_REPO_ROOT / "models" / "price_cache")
_SNAPSHOTS_DIR = _REPO_ROOT / "models" / "snapshots"

_OHLCV_REQUIRED = {"Open", "High", "Low", "Close", "Volume"}
_MIN_REASONABLE_DATE = pd.Timestamp("1990-01-01")


# ---------------------------------------------------------------------------
# Subcommand: backfill (default mode)
# ---------------------------------------------------------------------------

def _validate_backfilled(df: pd.DataFrame, label: str) -> None:
    """Sanity-check the merged DataFrame before we overwrite the cache.

    Fails loudly so we never write a corrupted parquet."""
    if df.empty:
        raise RuntimeError(f"[{label}] result is empty")
    if not df.index.is_monotonic_increasing:
        raise RuntimeError(f"[{label}] index is not monotonically increasing")
    if df.index.has_duplicates:
        raise RuntimeError(f"[{label}] index has duplicate dates")
    missing_cols = _OHLCV_REQUIRED - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"[{label}] missing OHLCV columns: {missing_cols}")
    if df["Close"].isna().any():
        n_nan = int(df["Close"].isna().sum())
        raise RuntimeError(f"[{label}] {n_nan} NaN value(s) in Close column")


def _yf_download_with_retry(ticker: str, start: str, end: str,
                            attempts: int = 2, backoff_seconds: float = 5.0
                            ) -> pd.DataFrame:
    """Wrap yf.download with one retry on failure. Match the project's
    standard auto_adjust=True so split/dividend adjustments are consistent
    with the existing cache."""
    import yfinance as yf
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df is None or df.empty:
                raise RuntimeError(f"yfinance returned empty for "
                                   f"{ticker} {start}->{end}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = df.ffill().dropna(how="any")
            if df.empty:
                raise RuntimeError(f"yfinance result became empty after "
                                   f"ffill().dropna() for {ticker}")
            return df
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                print(f"  [yfinance] attempt {attempt} failed: {e}; "
                      f"retrying in {backoff_seconds}s")
                time.sleep(backoff_seconds)
    assert last_exc is not None
    raise RuntimeError(
        f"yfinance.download({ticker!r}, {start}, {end}) failed after "
        f"{attempts} attempts: {last_exc}")


def _cmd_backfill(args: argparse.Namespace) -> int:
    ticker = args.ticker
    cache_dir = Path(args.cache_dir).resolve()
    cache_path = cache_dir / f"{ticker}.parquet"

    if not cache_path.exists():
        print(f"[BACKFILL] No existing cache at {cache_path}")
        print(f"  This tool only EXTENDS existing caches. For new tickers, "
              f"use fetch_data.get_stock_data_cached(...) which downloads "
              f"a full range from scratch on cache miss (live mode only).")
        return 1

    try:
        cached = pd.read_parquet(cache_path)
    except Exception as e:
        print(f"[BACKFILL] Failed to read existing cache: {e}")
        return 1
    cached.index = pd.to_datetime(cached.index)
    cached = cached.sort_index()

    cached_min = cached.index.min()
    cached_max = cached.index.max()
    target_start = pd.Timestamp(args.start)

    if target_start < _MIN_REASONABLE_DATE:
        print(f"[BACKFILL] Refusing: --start {args.start} is before "
              f"{_MIN_REASONABLE_DATE.date()}. Pick a sensible date.")
        return 1
    if target_start > pd.Timestamp.today():
        print(f"[BACKFILL] Refusing: --start {args.start} is in the future.")
        return 1
    if target_start >= cached_min:
        print(f"[BACKFILL] No backfill needed.")
        print(f"  cache covers {cached_min.date()} -> {cached_max.date()}, "
              f"requested start {target_start.date()} is already covered.")
        return 0

    fetch_end = (cached_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_start = args.start

    print(f"[BACKFILL] {ticker}: existing cache {cached_min.date()} -> "
          f"{cached_max.date()} ({len(cached)} rows)")
    print(f"[BACKFILL] Will fetch [{fetch_start}, {fetch_end}] from yfinance "
          f"(auto_adjust=True)")

    if args.dry_run:
        print(f"[BACKFILL] --dry-run: not fetching, not writing.")
        return 0

    new_old = _yf_download_with_retry(ticker, fetch_start, fetch_end)
    new_old.index = pd.to_datetime(new_old.index)
    print(f"[BACKFILL] yfinance returned {len(new_old)} rows "
          f"({new_old.index.min().date()} -> {new_old.index.max().date()})")

    # Align columns: keep only columns also in existing cache (drop extras
    # if yfinance ever adds Adj Close back, etc.)
    new_old = new_old.reindex(columns=cached.columns)

    merged = pd.concat([new_old, cached])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    _validate_backfilled(merged, f"{ticker} merged")

    # Atomic write via .tmp rename
    tmp = cache_path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp)
    os.replace(tmp, cache_path)

    rows_added = len(merged) - len(cached)
    print(f"[BACKFILL] Wrote {cache_path}")
    print(f"[BACKFILL] {len(cached)} -> {len(merged)} rows "
          f"(+{rows_added}); range now "
          f"{merged.index.min().date()} -> {merged.index.max().date()}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: update-snapshot-manifest
# ---------------------------------------------------------------------------

def _cmd_update_manifest(args: argparse.Namespace) -> int:
    snap_name = args.snapshot_name
    ticker = args.ticker
    snap_root = _SNAPSHOTS_DIR / snap_name
    manifest_path = snap_root / "manifest.json"
    parquet_rel = f"price_cache/{ticker}.parquet"
    parquet_path = snap_root / parquet_rel

    if not manifest_path.exists():
        print(f"[MANIFEST] Not found: {manifest_path}")
        return 1
    if not parquet_path.exists():
        print(f"[MANIFEST] Snapshot file not found: {parquet_path}\n"
              f"  Copy the backfilled parquet into the snapshot before "
              f"running this command.")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    new_size = parquet_path.stat().st_size
    new_mtime = parquet_path.stat().st_mtime

    files = manifest.get("files", [])
    found = False
    for entry in files:
        if entry.get("key") == parquet_rel:
            old_size = entry.get("size")
            old_mtime = entry.get("mtime")
            entry["size"] = new_size
            entry["mtime"] = new_mtime
            found = True
            print(f"[MANIFEST] Updated {parquet_rel}: "
                  f"size {old_size} -> {new_size}, mtime {old_mtime} -> {new_mtime}")
            break
    if not found:
        files.append({
            "key": parquet_rel,
            "size": new_size,
            "mtime": new_mtime,
        })
        manifest["files"] = files
        print(f"[MANIFEST] Added new entry for {parquet_rel}: "
              f"size={new_size} mtime={new_mtime}")

    # Recalculate total_size_bytes if present
    if "total_size_bytes" in manifest:
        manifest["total_size_bytes"] = sum(e.get("size", 0) for e in files)

    updates = manifest.get("updates", [])
    updates.append({
        "file":       parquet_rel,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "reason":     f"backfilled (per-ticker price cache update)",
    })
    manifest["updates"] = updates

    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path)

    print(f"[MANIFEST] Wrote {manifest_path}")
    print(f"[MANIFEST] manifest.updates entries: {len(updates)}")
    return 0


# ---------------------------------------------------------------------------
# Top-level CLI dispatch
# ---------------------------------------------------------------------------

def main() -> int:
    # Manual dispatch so we can keep "ticker --start ..." as the default
    # mode without forcing a "backfill" subcommand verb.
    if len(sys.argv) >= 2 and sys.argv[1] == "update-snapshot-manifest":
        p = argparse.ArgumentParser(
            prog="backfill_price.py update-snapshot-manifest",
            description="Update a snapshot's manifest.json after a "
                        "backfilled parquet has been copied in.")
        p.add_argument("snapshot_name",
                       help="Name under models/snapshots/.")
        p.add_argument("ticker",
                       help="Ticker whose price_cache/<ticker>.parquet "
                            "was updated.")
        args = p.parse_args(sys.argv[2:])
        return _cmd_update_manifest(args)

    p = argparse.ArgumentParser(
        prog="backfill_price.py",
        description="Extend a per-ticker price cache backwards in time "
                    "via yfinance. New data is prepended to the existing "
                    "parquet, deduped, and validated before atomic write.")
    p.add_argument("ticker",
                   help="Ticker whose cache to extend (e.g. SPY).")
    p.add_argument("--start", required=True,
                   help="New earliest date (YYYY-MM-DD). Must be earlier "
                        "than the existing cache's min date.")
    p.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR,
                   help=f"Per-ticker parquet directory. "
                        f"Default: {_DEFAULT_CACHE_DIR}")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without fetching/writing.")
    args = p.parse_args()
    return _cmd_backfill(args)


if __name__ == "__main__":
    sys.exit(main())
