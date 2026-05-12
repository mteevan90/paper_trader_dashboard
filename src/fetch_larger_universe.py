"""fetch_larger_universe.py — Phase 3 fetcher for the Larger Universe v1 study.

Populates the asset-class-namespaced cache at
``models/cache/equities/finnhub/`` for the union of:

  - Current SP500 + SP400 + SP600 constituents
  - SP500/400/600 historical members removed in the last 10 years

with daily OHLCV (Finnhub) and fundamentals (Finnhub /stock/metric).
Earnings dates remain on yfinance via :func:`src.backtest.fetch_earnings_dates`
because Finnhub Basic is forward-only for earnings — see
``docs/diagnostics/larger_universe_v1_capabilities.md``.

Universe input: ``docs/larger_universe_v1_universe.json`` (produced by
``src/equities/larger_universe_v1_builder.py``).

Outputs:
  - ``models/cache/equities/finnhub/prices/<SYM>.parquet`` (Finnhub fetcher)
  - ``models/cache/equities/finnhub/metrics/<SYM>.json`` (Finnhub fetcher)
  - ``models/cache/earnings_dates.json`` (legacy yfinance path, with stash defenses)
  - ``docs/diagnostics/larger_universe_v1_finnhub_fetch.log`` (this run's progress log)
  - ``docs/diagnostics/larger_universe_v1_fetch_summary.json`` (final report)

Re-runnable: per-ticker Finnhub caches are 7-day-TTL parquet/JSON — already-fetched
tickers are skipped on resume. yfinance earnings cache has its own TTL handling
in :mod:`src.backtest`.

Usage::

    venv\\Scripts\\python.exe -m src.fetch_larger_universe \\
        --start 2016-05-12 --end 2026-05-11 [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Strip Finnhub-style ?token=... query params from anything we log or
# persist (exception strings can carry the full request URL including
# the API key — see requests.HTTPError formatting).
_TOKEN_PARAM_RE = re.compile(r"(token=)[^&\s]+", re.IGNORECASE)


def _redact(s: object) -> str:
    return _TOKEN_PARAM_RE.sub(r"\1<REDACTED>", str(s))

# Make sure imports work whether run as script or module.
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from src.equities.finnhub_fetcher import (
    fetch_candles, fetch_metrics, make_candle_limiter, make_metric_limiter,
)


UNIVERSE_PATH = _ROOT / "docs" / "larger_universe_v1_universe.json"
FETCH_LOG_PATH = _ROOT / "docs" / "diagnostics" / "larger_universe_v1_finnhub_fetch.log"
SUMMARY_PATH = _ROOT / "docs" / "diagnostics" / "larger_universe_v1_fetch_summary.json"


logger = logging.getLogger("fetch_larger_universe")


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _load_universe(universe_path: Path | None = None) -> list[dict]:
    """Read the universe JSON and dedupe by symbol.

    When the same symbol appears as both active and removed (rebrand cases
    like ATI=Allegheny Technologies → ATI Inc.), prefer the active record
    — its date range covers both the historical and current periods.

    For true reuse cases (VAL=Valspar → Valaris), the active record's
    truncate_at remains None and we get the most-recent entity's full
    history. Whatever pre-2017 Valspar history existed under VAL is
    irretrievable from Finnhub (queries return Valaris). This is a
    documented residual gap.
    """
    path = Path(universe_path) if universe_path else UNIVERSE_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_symbol: dict[str, dict] = {}
    for rec in raw:
        sym = rec["symbol"]
        existing = by_symbol.get(sym)
        if existing is None:
            by_symbol[sym] = rec
            continue
        # Prefer active over removed
        if existing["status"] == "removed" and rec["status"] == "active":
            by_symbol[sym] = rec
        elif existing["status"] == "active" and rec["status"] == "removed":
            pass  # keep active
        else:
            # Tiebreaker: prefer the more-recent removed_at (deterministic).
            if (rec.get("removed_at") or "") > (existing.get("removed_at") or ""):
                by_symbol[sym] = rec
    return list(by_symbol.values())


def _truncate_at(rec: dict) -> date | None:
    """OTC-tail truncation date for a removed record, or None for active."""
    if rec["status"] != "removed":
        return None
    rd = rec.get("removed_at")
    if not rd:
        return None
    try:
        return datetime.strptime(rd, "%Y-%m-%d").date()
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", default="2016-05-12",
                        help="Price history start (default: 10y before today)")
    parser.add_argument("--end", default=None,
                        help="Price history end (default: today)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the universe to first N tickers (smoke test)")
    parser.add_argument("--universe-path", default=None,
                        help="Override the universe JSON path (default: "
                             "docs/larger_universe_v1_universe.json). Use this "
                             "for smoke tests against a curated subset.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be fetched without making any "
                             "Finnhub API calls or writing caches.")
    parser.add_argument("--skip-earnings", action="store_true",
                        help="Skip the yfinance earnings_dates step. Use when "
                             "iterating on the Finnhub path only.")
    parser.add_argument("--force-refresh-earnings", action="store_true",
                        help="Bypass the earnings cache TTL and re-fetch every "
                             "requested ticker via yfinance.")
    args = parser.parse_args()

    _setup_logging(FETCH_LOG_PATH)

    logger.info("=== Larger Universe v1 — Finnhub fetch ===")
    logger.info("dry_run=%s, limit=%s, skip_earnings=%s, force_refresh_earnings=%s",
                args.dry_run, args.limit, args.skip_earnings,
                args.force_refresh_earnings)

    universe_records = _load_universe(args.universe_path)
    if args.limit:
        universe_records = universe_records[:args.limit]
    n = len(universe_records)
    symbols = [r["symbol"] for r in universe_records]

    n_active = sum(1 for r in universe_records if r["status"] == "active")
    n_removed = n - n_active
    n_truncate = sum(1 for r in universe_records if _truncate_at(r) is not None)
    logger.info("universe loaded: %d symbols (%d active, %d removed; %d with OTC-tail truncation)",
                n, n_active, n_removed, n_truncate)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = (datetime.strptime(args.end, "%Y-%m-%d").date()
           if args.end else date.today())
    logger.info("window: %s -> %s", start, end)

    if args.dry_run:
        logger.info("[DRY-RUN] would call /stock/candle and /stock/metric for "
                    "%d symbols.", n)
        logger.info("[DRY-RUN] candle rate-limit: 150/min  -> ~%.1f min minimum",
                    n / 150)
        logger.info("[DRY-RUN] metric rate-limit: 60/min   -> ~%.1f min minimum",
                    n / 60)
        logger.info("[DRY-RUN] earnings (yfinance): unbounded; budget 30-60 min")
        return 0

    candle_limiter = make_candle_limiter()
    metric_limiter = make_metric_limiter()

    # --- Prices ---------------------------------------------------------
    t0 = time.time()
    logger.info("[1/3] Finnhub /stock/candle daily OHLCV ...")
    price_results = {"ok": 0, "empty": 0, "error": 0, "errors": []}
    for i, rec in enumerate(universe_records, 1):
        sym = rec["symbol"]
        trunc = _truncate_at(rec)
        try:
            df = fetch_candles(sym, start, end,
                               limiter=candle_limiter,
                               truncate_at=trunc)
            if df.empty:
                price_results["empty"] += 1
            else:
                price_results["ok"] += 1
            if i % 50 == 0 or i == n:
                logger.info("  prices %d/%d  (ok=%d, empty=%d, err=%d)",
                            i, n, price_results["ok"], price_results["empty"],
                            price_results["error"])
        except Exception as exc:
            price_results["error"] += 1
            price_results["errors"].append({"symbol": sym, "error": _redact(exc)[:160]})
            logger.warning("  prices %s: %s", sym, _redact(exc))
    logger.info("  prices done in %.1fs: ok=%d empty=%d err=%d",
                time.time() - t0,
                price_results["ok"], price_results["empty"],
                price_results["error"])

    # --- Fundamentals ---------------------------------------------------
    t0 = time.time()
    logger.info("[2/3] Finnhub /stock/metric fundamentals ...")
    metric_results = {"ok": 0, "empty": 0, "error": 0, "errors": []}
    for i, rec in enumerate(universe_records, 1):
        sym = rec["symbol"]
        try:
            body = fetch_metrics(sym, limiter=metric_limiter)
            if body and body.get("metric"):
                metric_results["ok"] += 1
            else:
                metric_results["empty"] += 1
            if i % 50 == 0 or i == n:
                logger.info("  metrics %d/%d (ok=%d empty=%d err=%d)",
                            i, n, metric_results["ok"], metric_results["empty"],
                            metric_results["error"])
        except Exception as exc:
            metric_results["error"] += 1
            metric_results["errors"].append({"symbol": sym, "error": _redact(exc)[:160]})
            logger.warning("  metrics %s: %s", sym, _redact(exc))
    logger.info("  metrics done in %.1fs: ok=%d empty=%d err=%d",
                time.time() - t0,
                metric_results["ok"], metric_results["empty"],
                metric_results["error"])

    # --- Earnings (yfinance, with stash defenses) -----------------------
    earnings_results: dict = {"skipped": True}
    if not args.skip_earnings:
        t0 = time.time()
        logger.info("[3/3] yfinance earnings_dates (with retry + sanity gate) ...")
        from backtest import fetch_earnings_dates  # import after sys.path setup
        try:
            earn = fetch_earnings_dates(
                symbols,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                force_refresh=args.force_refresh_earnings,
            )
            n_nonempty = sum(1 for s in symbols if earn.get(s))
            earnings_results = {
                "skipped": False,
                "total_entries": len(earn),
                "nonempty": n_nonempty,
                "nonempty_frac": (n_nonempty / n) if n else 0.0,
                "elapsed_s": round(time.time() - t0, 1),
            }
            logger.info("  earnings done in %.1fs: %d/%d non-empty (%.1f%%)",
                        time.time() - t0, n_nonempty, n,
                        100 * n_nonempty / n if n else 0)
        except Exception as exc:
            earnings_results = {"skipped": False, "error": _redact(exc)[:300]}
            logger.error("  earnings step failed: %s", _redact(exc))

    # --- Summary --------------------------------------------------------
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "universe": {
            "n_total": n,
            "n_active": n_active,
            "n_removed": n_removed,
            "n_with_truncate": n_truncate,
        },
        "prices": price_results,
        "metrics": metric_results,
        "earnings": earnings_results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("summary written to %s", SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
