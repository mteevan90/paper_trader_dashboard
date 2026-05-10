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
        --start-date 2023-01-03 --end-date 2025-12-31 \\
        --train-val-split-date 2024-12-31
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

load_dotenv()

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
        default=date(2023, 1, 3),
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date(2025, 12, 31),
    )
    parser.add_argument(
        "--train-val-split-date",
        type=date.fromisoformat,
        default=date(2024, 12, 31),
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
