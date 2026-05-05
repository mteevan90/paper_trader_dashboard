"""rescore_baseline.py — re-score a saved baseline against a specific cache snapshot.

Use case: cache files (fundamentals, earnings, macro_signals) auto-refresh
by TTL between runs, so a baseline scored on Day-N caches is not directly
comparable to a hypothesis run on Day-N+M caches. This script lets you
re-score an existing baseline (any meta.json with a "config" key) against
a frozen snapshot, producing a comparable score on the same data context.

Output is written to <snapshot>/phase0_baseline.json (or with a different
name via --output-name) so the snapshot becomes self-contained: hypothesis
runs against the snapshot can compare directly against the baseline file
sitting alongside.

Usage:

    python src/rescore_baseline.py \\
        --base-config models/cache/dashboard_results/best_optuna_v1_20260504_103429_706/meta.json \\
        --cache-snapshot pre_v3_20260505 \\
        --window train

The --window flag accepts {train, validate, both}. Each window runs a
separate backtest and adds its metrics to the baseline JSON.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Pre-parse + env-var setup BEFORE importing any cache module.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--cache-snapshot", required=True)
_pre_args, _ = _pre.parse_known_args()

_snap_root = os.path.abspath(os.path.join(
    str(_SRC_DIR), "..", "models", "snapshots", _pre_args.cache_snapshot))
if not os.path.isdir(_snap_root):
    sys.exit(f"[RESCORE] Snapshot not found: {_snap_root}")
os.environ["PAPER_TRADER_DATA_ROOT"] = _snap_root
print(f"[RESCORE] Using cache snapshot: "
      f"{_pre_args.cache_snapshot} ({_snap_root})")

# Cache modules now import with the right paths.
from backtest_config import BacktestConfig    # noqa: E402
from backtest import run_backtest             # noqa: E402
from feature_cache import build_feature_matrix # noqa: E402
from fetch_data import (UNIVERSE_TICKERS, build_sector_map,    # noqa: E402
                        get_stock_data_cached)
from macro_signals import fetch_macro_signals  # noqa: E402  (not used directly,
                                                # but importing forces the
                                                # module to register its
                                                # snapshot-aware paths)
from model import load_model                   # noqa: E402
from objective import (compute_objective,      # noqa: E402
                       compute_objective_components,
                       summarize_backtest)
from optuna_runner import (PRICE_CACHE,        # noqa: E402
                           _load_full_shared_data,
                           _load_shared_data)
from backtest import fetch_earnings_dates, fetch_fundamentals  # noqa: E402


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_SRC_DIR.parent),
            stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return "unknown"


def _score_window(cfg: BacktestConfig, window: str) -> dict:
    """Run a single backtest in the chosen window, return summary metrics.

    "train": backtest with split_date=cfg.train_start, training-only data.
    "validate": backtest with split_date=cfg.validate_start, full-window data.
    """
    if window == "train":
        shared = _load_shared_data(cfg.train_start, cfg.train_end)
        split = cfg.train_start
        suffix = "training"
    else:
        shared = _load_full_shared_data()
        split = cfg.validate_start
        suffix = "validation"

    print(f"[RESCORE] Running {suffix}-window backtest "
          f"(split_date={split})...")
    portfolio_df, trades_df, _scores, holdings = run_backtest(
        shared["featured_data"], shared["price_data"],
        split_date=split,
        fund_data=shared["fund_data"],
        sector_map=shared["sector_map"],
        earnings_dates=shared["earnings_dates"],
        model=shared["model"],
        config=cfg,
    )

    summary = summarize_backtest(portfolio_df, shared["spy_close"])
    score = compute_objective(summary)
    components = compute_objective_components(summary)
    return {
        f"{suffix}_score": float(score),
        f"{suffix}_components": components,
        f"{suffix}_n_days": int(len(portfolio_df)),
        f"{suffix}_n_trades": int(len(trades_df)),
        f"{suffix}_n_holdings": int(len(holdings)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-score a saved baseline against a frozen cache "
                    "snapshot for apples-to-apples hypothesis comparison.")
    p.add_argument("--base-config", required=True,
                   help="Path to a meta.json whose 'config' key holds "
                        "the baseline BacktestConfig.")
    p.add_argument("--cache-snapshot", required=True,
                   help="Snapshot name under models/snapshots/<name>/.")
    p.add_argument("--window", choices=["train", "validate", "both"],
                   required=True,
                   help="Which window(s) to score. 'both' runs both and "
                        "stores all metrics together.")
    p.add_argument("--output-name", default="phase0_baseline.json",
                   help="Filename inside the snapshot directory to write "
                        "the rescored baseline JSON. Default: "
                        "phase0_baseline.json.")
    args = p.parse_args()

    base_path = Path(args.base_config)
    if not base_path.exists():
        sys.exit(f"[RESCORE] --base-config not found: {base_path}")
    with open(base_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    base_config_dict = meta.get("config")
    if not isinstance(base_config_dict, dict):
        sys.exit(f"[RESCORE] meta.json has no 'config' key: {base_path}")

    cfg = BacktestConfig(**{
        k: v for k, v in base_config_dict.items()
        if k in BacktestConfig.__dataclass_fields__
    })

    out: dict = {
        "base_config_ref":  str(base_path).replace("\\", "/"),
        "cache_snapshot":   args.cache_snapshot,
        "window":           args.window,
        "rescored_at":      datetime.now(timezone.utc).isoformat(),
        "rescored_git_sha": _git_sha(),
        "config":           cfg.to_dict(),
    }

    if args.window in ("train", "both"):
        out.update(_score_window(cfg, "train"))
    if args.window in ("validate", "both"):
        out.update(_score_window(cfg, "validate"))

    out_path = os.path.join(_snap_root, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"\n[RESCORE] Done. Wrote {out_path}")
    if "training_score" in out:
        print(f"[RESCORE] training_score = {out['training_score']:.6f}")
    if "validation_score" in out:
        print(f"[RESCORE] validation_score = {out['validation_score']:.6f}")


if __name__ == "__main__":
    main()
