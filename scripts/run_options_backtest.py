"""CLI: run a backtest from a JSON :class:`BacktestConfig` and persist
results to parquet.

Examples:
    python scripts/run_options_backtest.py \\
        --config-path configs/smoke_csp.json \\
        --output-dir models/cache/options/study_results/smoke_csp/run_001
    python scripts/run_options_backtest.py \\
        --config-path configs/smoke_csp.json \\
        --output-dir /tmp/dryrun --dry-run

The config JSON is what :func:`BacktestConfig.to_dict` produces — see
``tests/options/test_backtest_config.py::TestSerialization`` for the
shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.options._ssl import use_system_trust_store  # noqa: E402

use_system_trust_store()

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.options.backtest_config import BacktestConfig  # noqa: E402
from src.options.engine import run_backtest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to BacktestConfig JSON",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for StudyResults output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run engine but don't write parquet output",
    )
    args = parser.parse_args()

    with open(args.config_path) as fh:
        config = BacktestConfig.from_dict(json.load(fh))

    print(
        f"Running backtest: {config.study_label} ({config.strategy_class})"
    )
    print(f"  Window: {config.start_date} to {config.end_date}")
    print(f"  Universe: {config.universe}")

    results = run_backtest(config)

    print(f"  Wall time: {results.wall_time_seconds:.1f}s")
    print(f"  Closed positions: {len(results.closed_positions)}")
    print(
        f"  Spawned equity closes: {len(results.spawned_equity_closes)}"
    )
    print(f"  Skip counters: {dict(results.skip_counters)}")

    if not args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_parquet(output_dir)
        print(f"  Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
