"""fetch_sp1500.py — populate the live data caches for the S&P 1500 universe.

Run AFTER the code changes (this is the long-running step). Pulls price
history (2018-01-01 onward), fundamentals, earnings dates, and sector
mappings for every ticker in fetch_data.SP1500_TICKERS, writing into the
canonical live-tree caches under models/. Does NOT touch a snapshot —
src/snapshot_sp1500.py captures the result into pre_v3_sp1500_<date>/
once this completes.

Outputs:
  - models/price_cache/<TICKER>.parquet           (one per ticker)
  - models/cache/fundamentals.json                (merged with existing)
  - models/cache/earnings_dates.json              (merged with existing)
  - models/cache/sector_map.json                  (merged with existing)
  - docs/sp1500_fetch_failures.txt                (failure list w/ reasons)
  - docs/sp1500_coverage_report.txt               (full coverage breakdown)

This is a research utility, not part of the runtime path. Re-runnable —
already-cached tickers are skipped or only incrementally updated.

Usage (PowerShell):
    venv\\Scripts\\python.exe src\\fetch_sp1500.py [--start 2018-01-01]
        [--end 2026-04-30] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

# Make sure CWD-independent imports work whether the script is invoked from
# the repo root or from src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import fetch_data
from fetch_data import (SP1500_TICKERS, UNIVERSE_TICKERS, build_sector_map,
                        get_stock_data_cached)
from backtest import fetch_fundamentals, fetch_earnings_dates


REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DOCS_DIR  = os.path.join(REPO_ROOT, "docs")
PRICE_CACHE_DIR = os.path.join(REPO_ROOT, "models", "price_cache")

# yfinance occasionally returns no data for a real, actively-traded
# ticker due to rate limiting or transient network blips (ABNB, ACI in
# the smoke test). Retry the missing tickers a couple of times with a
# short backoff before declaring them hard failures.
_PRICE_RETRY_BACKOFFS_SECONDS = (5, 10)


def _fetch_prices_with_retry(universe: list[str], start: str, end: str,
                             cache_dir: str) -> dict:
    """get_stock_data_cached + retry pass for tickers that came back empty.

    Returns the merged dict. Tickers still missing after all retries are
    surfaced via the caller's coverage classification (they end up in the
    'failed' bucket as before)."""
    price_data = get_stock_data_cached(universe, start, end,
                                       cache_dir=cache_dir)
    missing = [t for t in universe if t not in price_data]
    if not missing:
        return price_data

    for attempt, backoff in enumerate(_PRICE_RETRY_BACKOFFS_SECONDS, 1):
        max_attempts = len(_PRICE_RETRY_BACKOFFS_SECONDS)
        print(f"  [RETRY {attempt}/{max_attempts}] {len(missing)} tickers "
              f"had no data; waiting {backoff}s before retry...")
        time.sleep(backoff)
        retry = get_stock_data_cached(missing, start, end,
                                      cache_dir=cache_dir)
        if retry:
            price_data.update(retry)
            print(f"  [RETRY {attempt}/{max_attempts}] recovered "
                  f"{len(retry)} of {len(missing)}")
        missing = [t for t in missing if t not in price_data]
        if not missing:
            print(f"  [RETRY] All tickers recovered after attempt {attempt}.")
            break

    if missing:
        print(f"  [RETRY] {len(missing)} tickers still missing after "
              f"{len(_PRICE_RETRY_BACKOFFS_SECONDS)} retries: "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    return price_data


def _classify_coverage(
    tickers: list[str],
    price_data: dict,
    fund_data: dict,
    earn_data: dict,
    sector_map: dict,
    min_price_start: str = "2019-01-01",
) -> tuple[list[str], list[str], list[str], dict[str, list[str]]]:
    """Return (complete, partial, failed, gap_breakdown).

    - complete: price OK + fundamentals + earnings + sector mapped
    - partial:  price OK but missing one or more of fundamentals/earnings/sector
    - failed:   no price data at all (or price ends too early to be usable)
    - gap_breakdown: {ticker: [list of missing data sources]} for partial set
    """
    complete: list[str] = []
    partial: list[str] = []
    failed: list[str] = []
    gaps: dict[str, list[str]] = {}
    min_start_ts = None
    try:
        import pandas as pd
        min_start_ts = pd.Timestamp(min_price_start)
    except Exception:
        pass

    for tkr in tickers:
        df = price_data.get(tkr)
        price_ok = df is not None and not df.empty
        if price_ok and min_start_ts is not None:
            # Require at least some history before min_price_start so the
            # backtest's ~252-day feature warmup has data to chew on.
            price_ok = df.index.min() <= min_start_ts
        if not price_ok:
            failed.append(tkr)
            continue

        missing = []
        if tkr not in fund_data or not fund_data[tkr]:
            missing.append("fundamentals")
        if tkr not in earn_data or not earn_data[tkr]:
            missing.append("earnings")
        if tkr not in sector_map:
            missing.append("sector")

        if missing:
            partial.append(tkr)
            gaps[tkr] = missing
        else:
            complete.append(tkr)

    return complete, partial, failed, gaps


def _write_failures(failed: list[str], gaps: dict[str, list[str]],
                    extra_failures: dict[str, str]) -> str:
    os.makedirs(DOCS_DIR, exist_ok=True)
    path = os.path.join(DOCS_DIR, "sp1500_fetch_failures.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# SP1500 fetch failures — generated "
                f"{datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# Format: TICKER  REASON\n\n")
        f.write(f"## Hard failures (no price data — likely delisted / "
                f"wrong symbol)\n")
        for t in sorted(failed):
            reason = extra_failures.get(t, "no price data after fetch")
            f.write(f"{t:<10}  {reason}\n")
        f.write(f"\n## Partial — price OK but missing one or more "
                f"non-price datasets\n")
        for t in sorted(gaps):
            f.write(f"{t:<10}  missing: {', '.join(gaps[t])}\n")
    return path


def _write_coverage(
    universe: list[str], complete: list[str], partial: list[str],
    failed: list[str], gaps: dict[str, list[str]],
    new_vs_legacy: list[str],
) -> str:
    os.makedirs(DOCS_DIR, exist_ok=True)
    path = os.path.join(DOCS_DIR, "sp1500_coverage_report.txt")
    n = len(universe)
    pct = lambda k: (k * 100.0 / n) if n else 0.0

    by_gap: dict[str, list[str]] = {}
    for t, miss in gaps.items():
        by_gap.setdefault(", ".join(miss), []).append(t)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# SP1500 Coverage Report — generated "
                f"{datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"Universe size:                       {n}\n")
        f.write(f"  - Already in legacy 490 universe:  "
                f"{n - len(new_vs_legacy)}\n")
        f.write(f"  - NEW from S&P 400/600 expansion:  "
                f"{len(new_vs_legacy)}\n\n")
        f.write(f"Complete data (price+fund+earn+sec): {len(complete)}  "
                f"({pct(len(complete)):.1f}%)\n")
        f.write(f"Partial data (price OK, gaps elsewhere): {len(partial)}  "
                f"({pct(len(partial)):.1f}%)\n")
        f.write(f"Failed (no price data):              {len(failed)}  "
                f"({pct(len(failed)):.1f}%)\n\n")
        f.write(f"## Partial-data breakdown\n")
        for k in sorted(by_gap):
            f.write(f"  missing [{k}]: {len(by_gap[k])} tickers\n")
        f.write(f"\n## First 30 failed tickers (full list in "
                f"sp1500_fetch_failures.txt)\n")
        for t in sorted(failed)[:30]:
            f.write(f"  {t}\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2018-01-01",
                        help="Price history start (default 2018-01-01).")
    parser.add_argument("--end", default=None,
                        help="Price history end (default: today).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-ticker yfinance failure messages.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the universe to first N tickers (smoke test).")
    args = parser.parse_args()

    fetch_data.set_verbose(args.verbose)

    end = args.end or datetime.today().strftime("%Y-%m-%d")
    start = args.start

    universe = list(SP1500_TICKERS)
    if args.limit:
        universe = universe[:args.limit]
    new_vs_legacy = [t for t in universe if t not in set(UNIVERSE_TICKERS)]

    print(f"[FETCH_SP1500] Universe: {len(universe)} tickers "
          f"({len(new_vs_legacy)} new vs legacy 490)")
    print(f"[FETCH_SP1500] Window:   {start} -> {end}")
    print(f"[FETCH_SP1500] Estimated wall-clock: 30-90 min on a fresh fetch.")
    print()

    # --- Prices ---------------------------------------------------------
    t0 = time.time()
    print(f"[1/4] Price history (yfinance, cached parquets, with retries)...")
    os.makedirs(PRICE_CACHE_DIR, exist_ok=True)
    price_data = _fetch_prices_with_retry(universe, start, end,
                                          PRICE_CACHE_DIR)
    print(f"      {len(price_data)}/{len(universe)} tickers loaded "
          f"in {time.time()-t0:.1f}s\n")

    # --- Fundamentals ---------------------------------------------------
    t0 = time.time()
    print(f"[2/4] Fundamentals (yfinance .info, 7-day TTL cache)...")
    fund_data = fetch_fundamentals(list(price_data.keys()))
    print(f"      {len(fund_data)} entries in {time.time()-t0:.1f}s\n")

    # --- Earnings dates -------------------------------------------------
    t0 = time.time()
    print(f"[3/4] Earnings calendar (yfinance, 1-day TTL cache)...")
    earn_data = fetch_earnings_dates(list(price_data.keys()), start, end)
    print(f"      {len(earn_data)} entries in {time.time()-t0:.1f}s\n")

    # --- Sector map -----------------------------------------------------
    t0 = time.time()
    print(f"[4/4] Sector map (yfinance .info -> 30-day TTL cache)...")
    sector_map = build_sector_map(list(price_data.keys()))
    print(f"      {len(sector_map)} entries in {time.time()-t0:.1f}s\n")

    # --- Coverage classification + reports ------------------------------
    print("Classifying coverage...")
    complete, partial, failed, gaps = _classify_coverage(
        universe, price_data, fund_data, earn_data, sector_map)

    # Anything in `universe` but not in `price_data` is hard-failed (no
    # parquet on disk, no rows fetched). We also surface yfinance "fresh
    # download failed" cases through the get_stock_data_cached printout.
    extra_failures = {t: "no rows from yfinance (delisted / bad symbol?)"
                      for t in universe if t not in price_data}

    failures_path = _write_failures(failed, gaps, extra_failures)
    coverage_path = _write_coverage(universe, complete, partial, failed,
                                    gaps, new_vs_legacy)

    # --- Stdout summary -------------------------------------------------
    n = len(universe)
    print()
    print(f"=== SP1500 Coverage Summary ===")
    print(f"  Universe:                 {n}")
    print(f"  NEW vs legacy 490:        {len(new_vs_legacy)}")
    print(f"  Complete data:            {len(complete):>5}  "
          f"({100.0*len(complete)/n:.1f}%)")
    print(f"  Partial (price OK, gaps): {len(partial):>5}  "
          f"({100.0*len(partial)/n:.1f}%)")
    print(f"  Failed (no price):        {len(failed):>5}  "
          f"({100.0*len(failed)/n:.1f}%)")
    print()
    print(f"Coverage report: {coverage_path}")
    print(f"Failures list:   {failures_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
