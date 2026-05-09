"""Section 15.4-style sanity gate for fetched OHLCV.

Refuse to cache data that's mostly missing — equity yfinance learned
this the hard way when partial-coverage caches silently poisoned
downstream backtests. The 50% default carries here until we have
options-specific evidence to retune. Mirrors the crypto sibling at
``src/crypto/sanity_gate.py``.
"""

from __future__ import annotations

import pandas as pd


def passes_sanity_gate(df: pd.DataFrame, expected_days: int,
                       threshold: float = 0.5) -> tuple[bool, str]:
    """Return ``(passed, reason)``.

    Passes if the count of non-empty rows divided by ``expected_days`` is
    at least ``threshold``. A row counts as non-empty when its ``close``
    is not NaN; a missing ``close`` column or a non-positive
    ``expected_days`` always fails. ``reason`` is a human-readable string
    suitable for logging.
    """
    if expected_days <= 0:
        return False, f"expected_days must be positive, got {expected_days}"
    actual = int(df["close"].notna().sum()) if "close" in df.columns else 0
    coverage = actual / expected_days
    passed = coverage >= threshold
    status = "pass" if passed else "fail"
    reason = (
        f"coverage={actual}/{expected_days}={coverage:.1%} "
        f"(threshold {threshold:.0%}, {status})"
    )
    return passed, reason
