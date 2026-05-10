"""CLI: production v1 options study (CSP + CC primary + concentration + promotion).

Roughly 6-hour wall time on the locked Light scope (100 trials × 2
strategies × 2-year window + concentration ablations). The CLI prompts
for a human override after each strategy's automated promotion check
unless ``--non-interactive`` is set.

Usage:
    python scripts/run_options_v1_study.py
    python scripts/run_options_v1_study.py --non-interactive
    python scripts/run_options_v1_study.py \\
        --run-id my_run \\
        --start-date 2023-01-02 --end-date 2026-05-08 \\
        --train-val-split-date 2025-01-02
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.options._ssl import use_system_trust_store  # noqa: E402

use_system_trust_store()

from dotenv import load_dotenv  # noqa: E402

# Explicit path so .env loads regardless of the script's invocation CWD.
# load_dotenv() with no args walks up from CWD and silently no-ops if
# .env isn't found — that hid an 8-hour production stall once.
_DOTENV_PATH = REPO_ROOT / ".env"
if not _DOTENV_PATH.exists():
    print(
        f"WARNING: {_DOTENV_PATH} not found; "
        "TRADIER_SANDBOX_TOKEN may not be set",
        file=sys.stderr,
    )
load_dotenv(_DOTENV_PATH)

from src.options.v1_study import run_v1_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the production v1 options study.",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        # Polygon Options Developer tier's enforced historical floor —
        # see Appendix I and §10 in Options_Extension_Decisions.md.
        default=date(2023, 1, 2),
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date(2026, 5, 8),
    )
    parser.add_argument(
        "--train-val-split-date",
        type=date.fromisoformat,
        default=date(2025, 1, 2),
    )
    parser.add_argument("--starting-capital", type=float, default=100_000.0)
    parser.add_argument("--n-trials-primary", type=int, default=100)
    parser.add_argument(
        "--n-trials-per-ablation", type=int, default=25,
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Skip human override prompt; promotion decision is "
            "automated only."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Running v1 study: {args.run_id}")
    print(
        f"  Backtest window: {args.start_date} -> {args.end_date}"
    )
    print(f"  Train/val split: {args.train_val_split_date}")
    print(
        f"  Trials: {args.n_trials_primary} primary + "
        f"{args.n_trials_per_ablation}/ablation"
    )

    paths = run_v1_study(
        run_id=args.run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        train_val_split_date=args.train_val_split_date,
        starting_capital=args.starting_capital,
        n_trials_primary=args.n_trials_primary,
        n_trials_per_ablation=args.n_trials_per_ablation,
        seed=args.seed,
        interactive=not args.non_interactive,
    )

    print("\n=== v1 study complete ===")
    print(f"  CSP output: {paths['csp_dir']}")
    print(f"  CC output: {paths['cc_dir']}")
    print(f"  Snapshot: {paths['snapshot_dir']}")


if __name__ == "__main__":
    main()
