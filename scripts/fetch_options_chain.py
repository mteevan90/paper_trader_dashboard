"""CLI: fetch a current option-chain snapshot from Tradier.

Writes a point-in-time snapshot to
``models/cache/options/tradier/chains/<ticker>_<expiration>_<run_date>.parquet``.
Each call produces a new file; multiple invocations accumulate snapshots
without clobbering. With ``--expiration`` omitted the script picks the
next monthly-cycle expiration on or after today.

Examples:
    python scripts/fetch_options_chain.py --ticker SPY
    python scripts/fetch_options_chain.py --ticker AAPL --expiration 2026-06-19
    python scripts/fetch_options_chain.py --ticker SPY --no-greeks --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.options._ssl import use_system_trust_store  # noqa: E402
use_system_trust_store()

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

import argparse  # noqa: E402
import logging  # noqa: E402
from datetime import date, datetime  # noqa: E402

from src.options.cache import cache_chain_snapshot  # noqa: E402
from src.options.tradier import fetch_chain_snapshot, fetch_expirations  # noqa: E402


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ticker", required=True)
    p.add_argument("--expiration", type=_parse_date, default=None,
                   help="Expiration date (YYYY-MM-DD). Default: next monthly on/after today.")
    p.add_argument("--no-greeks", action="store_true",
                   help="Omit ORATS Greeks from the snapshot.")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch but do not write the snapshot file.")
    return p.parse_args()


def _next_monthly(expirations: list[date], today: date) -> date | None:
    """Pick the next standard-monthly expiration on/after today.

    Standard monthlies fall on the third Friday of the month. Approximate
    by selecting the earliest expiration >= today whose day is in 15..21.
    """
    for d in expirations:
        if d >= today and 15 <= d.day <= 21 and d.weekday() == 4:
            return d
    # Fallback: just the earliest future expiration.
    return next((d for d in expirations if d >= today), None)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    args = _parse_args()

    expiration = args.expiration
    if expiration is None:
        expirations = fetch_expirations(args.ticker)
        if not expirations:
            print(f"  NO_EXPIRATIONS  ticker={args.ticker}")
            return 1
        expiration = _next_monthly(expirations, date.today())
        if expiration is None:
            print(f"  NO_FUTURE_EXPIRATION  ticker={args.ticker}")
            return 1
        print(f"  resolved next monthly expiration: {expiration}")

    suffix = " [DRY RUN]" if args.dry_run else ""
    print(
        f"Fetching chain {args.ticker} @ {expiration} "
        f"(greeks={not args.no_greeks}){suffix}",
    )

    try:
        df = fetch_chain_snapshot(
            args.ticker, expiration, with_greeks=not args.no_greeks,
        )
    except Exception as exc:
        print(f"  FETCH_ERROR  {type(exc).__name__}: {exc}")
        return 1

    if df.empty:
        print(f"  EMPTY_CHAIN  ticker={args.ticker} expiration={expiration}")
        return 1

    if args.dry_run:
        print(f"  DRY_OK  rows={len(df)}")
        return 0

    path = cache_chain_snapshot(args.ticker, expiration, df)
    print(f"  OK  rows={len(df)}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
