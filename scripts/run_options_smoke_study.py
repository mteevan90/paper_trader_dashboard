"""CLI: compressed-full options smoke study (CSP + CC sequential).

Runs the locked smoke configuration end-to-end against real Tradier
sandbox data. Validates that Sections 1-6 work together before
Section 8 launches the production study. Roughly 30 minutes wall time.

Locked config (do not change without updating §9 row 7 of the design
memo):
    universe         : DEFAULT_UNIVERSE (8 v1 names)
    strategies       : cash_secured_put, then covered_call
    window           : 2024-01-02 → 2024-07-01
    train/val split  : 2024-05-01
    trials/strategy  : 5
    top_k            : 3
    starting_capital : $100k
    promotable       : False (smoke is never promotable)

Usage:
    python scripts/run_options_smoke_study.py
    python scripts/run_options_smoke_study.py \\
        --output-base-dir /tmp/options-smoke
"""

from __future__ import annotations

import argparse
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

from src.options.backtest_config import DEFAULT_UNIVERSE  # noqa: E402
from src.options.optuna_runner import run_optuna_study  # noqa: E402


SMOKE_START_DATE = date(2024, 1, 2)
SMOKE_END_DATE = date(2024, 7, 1)
SMOKE_SPLIT_DATE = date(2024, 5, 1)
SMOKE_TRIALS_PER_STRATEGY = 5
SMOKE_TOP_K = 3
SMOKE_STARTING_CAPITAL = 100_000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compressed-full options smoke study (CSP + CC).",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=Path("models/cache/options/optuna_studies"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = args.output_base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Running compressed-full options smoke study")
    print(f"  Universe: {DEFAULT_UNIVERSE}")
    print(f"  Window: {SMOKE_START_DATE} to {SMOKE_END_DATE}")
    print(f"  Train/val split: {SMOKE_SPLIT_DATE}")
    print(f"  Trials per strategy: {SMOKE_TRIALS_PER_STRATEGY}")
    print()

    for strategy_class in ("cash_secured_put", "covered_call"):
        study_label = f"smoke_compressed_full_{strategy_class}"
        output_dir = base_dir / study_label

        results = run_optuna_study(
            study_label=study_label,
            strategy_class=strategy_class,
            universe=DEFAULT_UNIVERSE,
            start_date=SMOKE_START_DATE,
            end_date=SMOKE_END_DATE,
            train_val_split_date=SMOKE_SPLIT_DATE,
            n_trials=SMOKE_TRIALS_PER_STRATEGY,
            starting_capital=SMOKE_STARTING_CAPITAL,
            promotable=False,
            output_dir=output_dir,
            top_k=SMOKE_TOP_K,
            seed=args.seed,
        )

        print(f"\n=== {strategy_class} smoke complete ===")
        print(f"  Best Calmar: {results.best_value:.4f}")
        print(f"  Best params: {results.best_params}")
        print(
            f"  Failed trials: {results.n_trials_failed}/"
            f"{results.n_trials_run}"
        )
        print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
