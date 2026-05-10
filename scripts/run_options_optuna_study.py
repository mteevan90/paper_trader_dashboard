"""CLI: arbitrary Optuna study runner for the options module.

Used by Section 7's smoke study and Section 8's production studies. The
smoke wrapper (``scripts/run_options_smoke_study.py``) calls
``run_optuna_study`` directly so it can lock parameters that this CLI
exposes as flags.

Usage:
    python scripts/run_options_optuna_study.py \\
        --study-label my_study \\
        --strategy-class cash_secured_put \\
        --n-trials 25 \\
        --start-date 2024-01-02 \\
        --end-date 2024-12-31 \\
        --train-val-split-date 2024-09-01 \\
        --output-dir models/cache/options/optuna_studies/my_study
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
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

from src.options.optuna_runner import run_optuna_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an Optuna study for the options module.",
    )
    parser.add_argument("--study-label", required=True)
    parser.add_argument(
        "--strategy-class",
        choices=["cash_secured_put", "covered_call"],
        required=True,
    )
    parser.add_argument("--n-trials", type=int, required=True)
    parser.add_argument(
        "--start-date", required=True, type=date.fromisoformat,
    )
    parser.add_argument(
        "--end-date", required=True, type=date.fromisoformat,
    )
    parser.add_argument(
        "--train-val-split-date",
        required=True,
        type=date.fromisoformat,
    )
    parser.add_argument(
        "--starting-capital", type=float, default=100_000.0,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--promotable",
        action="store_true",
        help=(
            "Allow snapshot upload of this study "
            "(smoke studies should never set this)"
        ),
    )
    parser.add_argument(
        "--universe",
        nargs="*",
        default=None,
        help="Override default universe; e.g. --universe SPY AAPL",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    universe = tuple(args.universe) if args.universe else None

    results = run_optuna_study(
        study_label=args.study_label,
        strategy_class=args.strategy_class,
        universe=universe,
        start_date=args.start_date,
        end_date=args.end_date,
        train_val_split_date=args.train_val_split_date,
        n_trials=args.n_trials,
        starting_capital=args.starting_capital,
        promotable=args.promotable,
        output_dir=args.output_dir,
        top_k=args.top_k,
        seed=args.seed,
    )

    print(f"\n=== Study {args.study_label} complete ===")
    print(f"  Strategy: {args.strategy_class}")
    print(
        f"  Trials run: {results.n_trials_run} "
        f"({results.n_trials_failed} failed)"
    )
    print(f"  Best Calmar: {results.best_value:.4f}")
    print(f"  Best trial: #{results.best_trial_number}")
    print(
        "  Best params: "
        + json.dumps(results.best_params, indent=2, default=str)
    )
    print(
        f"  Top-{args.top_k} trial numbers: "
        f"{results.top_k_trial_numbers}"
    )
    print(f"  Wall time: {results.wall_time_seconds:.1f}s")
    print(f"  Output dir: {results.output_dir}")

    summary_path = args.output_dir / "study_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_json(summary_path)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
