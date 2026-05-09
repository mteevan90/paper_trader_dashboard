"""CLI: fetch daily OHLCV from Tradier for one OCC symbol or underlying.

Pulls from ``/markets/history`` (Tradier accepts both an OCC contract
symbol and a plain ticker at this endpoint). Runs the Section 15.4
sanity gate against the requested date range and writes to
``models/cache/options/tradier/history/<symbol>.parquet`` on pass.

Examples:
    python scripts/fetch_options_history.py --symbol SPY --start 2025-01-02 --end 2025-01-31
    python scripts/fetch_options_history.py --symbol SPY220617C00450000 --start 2022-01-01 --end 2022-06-17 --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.options._ssl import use_system_trust_store  # noqa: E402
use_system_trust_store()

from dotenv import load_dotenv  # noqa: E402
load_dotenv()  # TRADIER_SANDBOX_TOKEN etc. — must precede tradier import

import argparse  # noqa: E402
import logging  # noqa: E402
from datetime import date, datetime  # noqa: E402

from src.options.cache import SanityGateFailure, cache_history  # noqa: E402
from src.options.sanity_gate import passes_sanity_gate  # noqa: E402
from src.options.tradier import fetch_history  # noqa: E402


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbol", required=True,
                   help="OCC symbol (e.g. SPY220617C00450000) or underlying ticker (e.g. SPY).")
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and run the sanity gate but do not write the cache.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    args = _parse_args()
    expected_days = (args.end - args.start).days + 1
    suffix = " [DRY RUN]" if args.dry_run else ""
    print(
        f"Fetching {args.symbol} from {args.start} to {args.end} "
        f"({expected_days} expected days){suffix}",
    )

    try:
        df = fetch_history(args.symbol, args.start, args.end, use_cache=False)
    except Exception as exc:
        print(f"  FETCH_ERROR  {type(exc).__name__}: {exc}")
        return 1

    passed, reason = passes_sanity_gate(df, expected_days)
    if not passed:
        print(f"  GATE_FAIL  rows={len(df)}  {reason}")
        return 1

    if args.dry_run:
        print(f"  DRY_OK  rows={len(df)}  {reason}")
        return 0

    try:
        path = cache_history(args.symbol, df)
    except SanityGateFailure as exc:
        print(f"  CACHE_GATE_FAIL  {exc}")
        return 1

    print(f"  OK  rows={len(df)}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
