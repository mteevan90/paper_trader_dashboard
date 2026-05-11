"""Tests for backtest.filter_candidates_by_liquidity (equity-side).

Run under pytest from repo root:
    venv\\Scripts\\pytest tests/equities/test_liquidity_filter.py -v

The bare ``from backtest import ...`` style matches the equity codebase's
internal imports — src/backtest.py itself imports its siblings via
``from alt_signals import ...`` etc., not ``from src.alt_signals``. The
pyproject.toml at repo root puts src/ on pytest's pythonpath so this
resolves cleanly.

The ``main()`` block below is a legacy hand-rolled runner kept for the
sp1500 diagnostic workflow. Under pytest, ``main()`` is ignored (no
``test_`` prefix) and each ``test_*`` function runs independently.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from backtest import filter_candidates_by_liquidity


def _synthetic_price_df(closes: list[float], volumes: list[float],
                        end_date: str = "2024-01-15") -> pd.DataFrame:
    """Build a DataFrame indexed by trading days ending at end_date."""
    n = len(closes)
    idx = pd.bdate_range(end=end_date, periods=n)
    return pd.DataFrame({"Close": closes, "Volume": volumes}, index=idx)


def test_keeps_above_threshold_drops_below() -> None:
    """A $100M ADV stock passes a $25M filter; a $5M ADV stock fails."""
    high_volume = _synthetic_price_df(
        closes=[100.0] * 40, volumes=[1_000_000.0] * 40)   # $100M/day
    low_volume = _synthetic_price_df(
        closes=[50.0] * 40,  volumes=[100_000.0] * 40)     # $5M/day
    medium = _synthetic_price_df(
        closes=[20.0] * 40,  volumes=[2_000_000.0] * 40)   # $40M/day

    price_data = {"HIGH": high_volume, "LOW": low_volume, "MED": medium}
    date = pd.Timestamp("2024-01-15")
    kept = filter_candidates_by_liquidity(
        ["HIGH", "LOW", "MED"], price_data, date,
        threshold_usd=25_000_000)
    assert kept == ["HIGH", "MED"], (
        f"expected HIGH+MED kept, got {kept}")
    print(f"  PASS  high/medium kept, low dropped: {kept}")


def test_threshold_zero_passes_everything() -> None:
    """threshold_usd=0 (or None) is a no-op."""
    df = _synthetic_price_df([10.0] * 5, [100.0] * 5)
    out_zero = filter_candidates_by_liquidity(
        ["A"], {"A": df}, pd.Timestamp("2024-01-15"), threshold_usd=0)
    out_none = filter_candidates_by_liquidity(
        ["A"], {"A": df}, pd.Timestamp("2024-01-15"), threshold_usd=None)
    assert out_zero == ["A"], f"threshold=0 should keep all, got {out_zero}"
    assert out_none == ["A"], f"threshold=None should keep all, got {out_none}"
    print("  PASS  threshold 0 / None passes all candidates")


def test_per_day_filter_changes_with_date() -> None:
    """A ticker that USED to be liquid but isn't anymore gets dropped on
    a later date but kept on an earlier date — confirms the filter is
    truly per-day, not static over the run."""
    # First 35 days: $50M/day. Last 30 days: $5M/day.
    closes = [100.0] * 65
    volumes = [500_000.0] * 35 + [50_000.0] * 30
    df = pd.DataFrame({"Close": closes, "Volume": volumes},
                      index=pd.bdate_range(end="2024-04-15", periods=65))
    price_data = {"X": df}

    # Early date: still in the high-liquidity period.
    early = filter_candidates_by_liquidity(
        ["X"], price_data, df.index[34],   # last day of high vol
        threshold_usd=25_000_000)
    # Late date: trailing 30d are all low-volume.
    late = filter_candidates_by_liquidity(
        ["X"], price_data, df.index[-1],
        threshold_usd=25_000_000)

    assert early == ["X"], f"early date should keep X, got {early}"
    assert late == [], f"late date should drop X, got {late}"
    print("  PASS  per-day filter — same ticker kept early, dropped late")


def test_handles_missing_ticker() -> None:
    """A ticker present in `tickers` but missing from price_data is skipped."""
    df = _synthetic_price_df([100.0] * 40, [1_000_000.0] * 40)
    out = filter_candidates_by_liquidity(
        ["X", "MISSING"], {"X": df}, pd.Timestamp("2024-01-15"),
        threshold_usd=25_000_000)
    assert out == ["X"], f"expected X kept, MISSING skipped, got {out}"
    print("  PASS  missing tickers silently skipped")


def test_handles_short_history() -> None:
    """A ticker with only a few days of history is judged on what it has."""
    df = _synthetic_price_df([100.0] * 5, [1_000_000.0] * 5)  # $100M/day x 5
    out = filter_candidates_by_liquidity(
        ["NEW"], {"NEW": df}, df.index[-1], threshold_usd=25_000_000)
    assert out == ["NEW"], f"5-day history should still pass, got {out}"
    print("  PASS  short-history ticker judged on available rows")


def test_handles_nans() -> None:
    """NaN Close or Volume rows are dropped before averaging — mid-run
    yfinance gaps shouldn't poison the average to NaN."""
    closes = [100.0] * 40
    volumes = [1_000_000.0] * 40
    closes[5] = np.nan
    volumes[10] = np.nan
    df = _synthetic_price_df(closes, volumes)
    out = filter_candidates_by_liquidity(
        ["X"], {"X": df}, df.index[-1], threshold_usd=25_000_000)
    assert out == ["X"], (
        f"NaN rows should be dropped, ticker still passes, got {out}")
    print("  PASS  NaN Close/Volume rows handled gracefully")


def test_25m_threshold_boundary() -> None:
    """A ticker exactly at $25M/day passes (>=, not >)."""
    df = _synthetic_price_df([25.0] * 40, [1_000_000.0] * 40)  # exactly $25M
    out = filter_candidates_by_liquidity(
        ["B"], {"B": df}, df.index[-1], threshold_usd=25_000_000)
    assert out == ["B"], f"boundary case should pass with >=, got {out}"
    print("  PASS  boundary case (exactly threshold) passes")


def main() -> int:
    print("Running liquidity filter unit tests...")
    print()
    test_keeps_above_threshold_drops_below()
    test_threshold_zero_passes_everything()
    test_per_day_filter_changes_with_date()
    test_handles_missing_ticker()
    test_handles_short_history()
    test_handles_nans()
    test_25m_threshold_boundary()
    print()
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
