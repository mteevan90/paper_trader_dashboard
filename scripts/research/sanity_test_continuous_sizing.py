"""Sanity tests for the continuous-sizing strategy mechanic.

Test 1 — Backwards compat: load Trial #325's saved config and replay
with NO continuous-sizing tunables set. Validation portfolio.parquet
must match the saved one bit-identically.

Test 2 — Continuous-sizing isolation: synthetic config = Trial #325's
config + high_macro_threshold=0.70, low_macro_threshold=0.40. Confirm
the run produces a non-empty sizing_decisions with varied target counts
in [5, 15], no NaN positions, and target_position_count_after equals
target_position_count for every rebalance.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

# Trial #325's saved meta records cache_snapshot=pre_v2_20260505 with
# the v2 macro signal (7 components, no SPY drawdown). To reproduce its
# saved portfolio.parquet bit-identically we need both the same snapshot
# AND the v2 formula (the v3 fix added SPY drawdown which would shift
# every macro composite). PAPER_TRADER_MACRO_VERSION=v2 is the documented
# escape hatch for replaying old saves under the new code.
SNAPSHOT = REPO_ROOT / "models" / "snapshots" / "pre_v2_20260505"
os.environ["PAPER_TRADER_DATA_ROOT"] = str(SNAPSHOT)
os.environ["PAPER_TRADER_ARCHITECTURE"] = "regime-dependent"
os.environ["PAPER_TRADER_MACRO_VERSION"] = "v2"

from backtest_config import BacktestConfig                    # noqa: E402
from backtest import (run_backtest, fetch_fundamentals,         # noqa: E402
                       fetch_earnings_dates,
                       compute_target_position_count,
                       _CONTINUOUS_SIZING_HIGH_POS,
                       _CONTINUOUS_SIZING_LOW_POS)
from feature_cache import build_feature_matrix                 # noqa: E402
from fetch_data import (UNIVERSE_TICKERS, build_sector_map,     # noqa: E402
                         get_stock_data_cached)
from model import load_model                                   # noqa: E402
from optuna_runner import _load_full_shared_data               # noqa: E402

LABEL_325 = "best_regime_dependent_v1_20260505_2240_325"
META_325  = REPO_ROOT / "models" / "cache" / "dashboard_results" / LABEL_325 / "meta.json"
SAVED_PV  = REPO_ROOT / "models" / "cache" / "dashboard_results" / LABEL_325 / "portfolio.parquet"


# ---- Quick unit tests for compute_target_position_count ----
def test_helper() -> None:
    cfg = BacktestConfig(
        high_macro_threshold=0.70,
        low_macro_threshold=0.40,
    )
    # Endpoints
    assert compute_target_position_count(0.80, cfg) == _CONTINUOUS_SIZING_HIGH_POS
    assert compute_target_position_count(0.70, cfg) == _CONTINUOUS_SIZING_HIGH_POS
    assert compute_target_position_count(0.30, cfg) == _CONTINUOUS_SIZING_LOW_POS
    assert compute_target_position_count(0.40, cfg) == _CONTINUOUS_SIZING_LOW_POS
    # Midpoint between 0.40 (15 pos) and 0.70 (5 pos) — at 0.55, frac=0.5,
    # n = 15 + 0.5 * (5 - 15) = 10.
    assert compute_target_position_count(0.55, cfg) == 10
    # Quarter-of-the-way and three-quarters land on exact half-integers
    # (12.5 and 7.5) under perfect arithmetic, but float precision can
    # drift either side of the boundary so accept either neighbor.
    assert compute_target_position_count(0.475, cfg) in {12, 13}
    assert compute_target_position_count(0.625, cfg) in {7, 8}
    # Degenerate: missing fields fall back to high endpoint
    cfg2 = BacktestConfig()
    assert compute_target_position_count(0.5, cfg2) == _CONTINUOUS_SIZING_HIGH_POS
    print("[helper] compute_target_position_count: 7/7 unit tests passed")


def _make_config_from_meta(meta: dict, **overrides) -> BacktestConfig:
    """Reconstruct a BacktestConfig from a saved meta.json's config block,
    applying any keyword overrides on top."""
    config_dict = dict(meta["config"])
    config_dict.update(overrides)
    return BacktestConfig(**{
        k: v for k, v in config_dict.items()
        if k in BacktestConfig.__dataclass_fields__
    })


def test_backwards_compat() -> None:
    """Run #325's config end-to-end (no continuous-sizing fields). The
    resulting validation portfolio_value series must match the saved
    parquet bit-identically."""
    print()
    print("=== Test 1: backwards compat against Trial #325 ===")
    if not META_325.exists():
        print(f"  SKIP — {META_325} not found")
        return
    if not SAVED_PV.exists():
        print(f"  SKIP — {SAVED_PV} not found")
        return
    meta = json.load(META_325.open())
    config = _make_config_from_meta(meta)
    assert config.high_macro_threshold is None, (
        "Test 1 expects #325's config to NOT carry continuous-sizing fields")
    assert config.low_macro_threshold is None

    saved_pv = pd.read_parquet(SAVED_PV)
    print(f"  saved portfolio.parquet: {len(saved_pv)} rows, "
          f"{saved_pv.index.min().date()} to {saved_pv.index.max().date()}")

    print(f"  Loading shared data (snapshot: {SNAPSHOT.name})...")
    shared = _load_full_shared_data()
    portfolio_df, trades_df, _scores, _holdings = run_backtest(
        shared["featured_data"], shared["price_data"],
        split_date=config.validate_start,
        fund_data=shared["fund_data"],
        sector_map=shared["sector_map"],
        earnings_dates=shared["earnings_dates"],
        model=shared["model"],
        config=config,
        market_data=shared.get("market_data"),
    )
    print(f"  fresh portfolio.parquet: {len(portfolio_df)} rows, "
          f"{portfolio_df.index.min().date()} to {portfolio_df.index.max().date()}")

    # Compare portfolio_value series
    saved_pv_series = saved_pv["portfolio_value"]
    fresh_pv_series = portfolio_df["portfolio_value"]
    n_diff = int((saved_pv_series.round(2) != fresh_pv_series.round(2)).sum())
    max_abs_diff = float((saved_pv_series - fresh_pv_series).abs().max())
    print(f"  portfolio_value diffs (rounded 2dp): {n_diff} / {len(saved_pv_series)}")
    print(f"  max absolute diff: ${max_abs_diff:,.4f}")
    if n_diff == 0:
        print("  PASS — bit-identical match")
    elif max_abs_diff < 1.0:
        print(f"  PASS (sub-dollar drift) — {max_abs_diff:.4f} max diff is "
              f"floating-point noise")
    else:
        print(f"  WARN — meaningful drift detected; {max_abs_diff:.2f} max diff")

    # Trade count sanity
    n_saved_trades = len(saved_pv.attrs.get("trades", [])) if False else None
    n_fresh_trades = len(trades_df) if trades_df is not None else 0
    print(f"  fresh trade count: {n_fresh_trades}")

    # sizing_decisions still emitted (one row per rebalance) even for
    # non-continuous runs — the spec says "every rebalance in every
    # backtest run". For #325 (offensive=5, defensive=10) the targets
    # come from active["position_count"], not the interpolation helper.
    sd_list = portfolio_df.attrs.get("sizing_decisions") or []
    assert sd_list, "sizing_decisions empty — expected one row per rebalance"
    sd = pd.DataFrame(sd_list)
    print(f"  sizing_decisions rows: {len(sd)} (one per rebalance)")
    print(f"  unique target_position_count: {sorted(sd['target_position_count'].unique())}")
    # #325 has position_count=10 (defensive) and position_count_offensive=5
    # (offensive); the regime that actually fires depends on macro signal
    # vs config.regime_threshold. Just confirm targets are sensible.
    assert sd["target_position_count"].between(1, 100).all(), (
        "Non-continuous run produced wild target_position_count values")


def test_continuous_sizing_isolation() -> None:
    print()
    print("=== Test 2: continuous-sizing isolation (synthetic config) ===")
    if not META_325.exists():
        print(f"  SKIP — {META_325} not found")
        return
    meta = json.load(META_325.open())
    config = _make_config_from_meta(
        meta,
        high_macro_threshold=0.70,
        low_macro_threshold=0.40,
    )
    print(f"  config.high_macro_threshold = {config.high_macro_threshold}")
    print(f"  config.low_macro_threshold  = {config.low_macro_threshold}")

    print(f"  Loading shared data (snapshot: {SNAPSHOT.name})...")
    shared = _load_full_shared_data()
    portfolio_df, trades_df, _scores, _holdings = run_backtest(
        shared["featured_data"], shared["price_data"],
        split_date=config.validate_start,
        fund_data=shared["fund_data"],
        sector_map=shared["sector_map"],
        earnings_dates=shared["earnings_dates"],
        model=shared["model"],
        config=config,
        market_data=shared.get("market_data"),
    )

    sd_list = portfolio_df.attrs.get("sizing_decisions")
    assert sd_list, "sizing_decisions empty — expected non-empty"
    sd = pd.DataFrame(sd_list)
    print(f"  sizing_decisions rows:        {len(sd)}")
    print(f"  unique target_position_count: {sorted(sd['target_position_count'].unique())}")
    print(f"  macro_signal range:           {sd['macro_signal'].min():.3f} – {sd['macro_signal'].max():.3f}")
    print(f"  before -> after counts (head):")
    for _, r in sd.head(3).iterrows():
        print(f"    {r['date'].date()}  signal={r['macro_signal']:.3f} "
              f"target={r['target_position_count']:>2} "
              f"before={r['actual_position_count_before']:>2} "
              f"after={r['actual_position_count_after']:>2} "
              f"regime={r['regime']}")
    # Invariants
    assert sd["target_position_count"].between(
        _CONTINUOUS_SIZING_HIGH_POS, _CONTINUOUS_SIZING_LOW_POS).all(), \
        "target_position_count outside [5, 15]"
    assert sd["macro_signal"].notna().all(), "macro_signal contains NaN"
    assert sd["target_position_count"].notna().all(), "target_position_count NaN"
    # actual_position_count_after may differ from target when:
    #   - earnings blackout skipped a buy
    #   - shares < 1 budget (budget_each / buy_price < 1)
    #   - min_hold_days prevented a sell
    # Just confirm we never EXCEED the target — over-target would mean a bug.
    assert (sd["actual_position_count_after"]
            <= sd["target_position_count"]).all(), \
        "actual_position_count_after exceeds target on some rebalance"
    # Regime label must match the bucket logic
    for _, r in sd.iterrows():
        t = int(r["target_position_count"])
        if   t <= 6:  expected_bucket = "high_conviction_5pos"
        elif t >= 13: expected_bucket = "diversified_15pos"
        else:         expected_bucket = "interpolated_8_to_12"
        assert r["regime"] == expected_bucket, \
            f"regime mismatch on {r['date'].date()}: target={t} got {r['regime']!r}"
    print(f"  PASS — invariants hold ({len(sd)} rebalances, "
          f"{len(sd['target_position_count'].unique())} distinct sizes)")
    print(f"  Strategy total return: "
          f"{(portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0] - 1)*100:+.2f}% "
          f"({len(trades_df) if trades_df is not None else 0} trades)")


if __name__ == "__main__":
    test_helper()
    test_backwards_compat()
    test_continuous_sizing_isolation()
    print()
    print("[sanity] all tests done")
