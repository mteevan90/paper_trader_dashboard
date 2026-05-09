"""Tests for src/options/sanity_gate.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.options.sanity_gate import passes_sanity_gate


def _df_with_close_count(non_null: int, total: int) -> pd.DataFrame:
    closes = [100.0] * non_null + [float("nan")] * (total - non_null)
    return pd.DataFrame({"close": closes})


def test_below_threshold_fails():
    df = _df_with_close_count(non_null=4, total=10)
    passed, reason = passes_sanity_gate(df, expected_days=10, threshold=0.5)
    assert passed is False
    assert "fail" in reason


def test_at_threshold_passes():
    df = _df_with_close_count(non_null=5, total=10)
    passed, reason = passes_sanity_gate(df, expected_days=10, threshold=0.5)
    assert passed is True
    assert "pass" in reason


def test_missing_close_column_fails():
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
    passed, reason = passes_sanity_gate(df, expected_days=10, threshold=0.5)
    assert passed is False
    assert "0/10" in reason


@pytest.mark.parametrize("expected", [0, -1, -100])
def test_non_positive_expected_days_fails(expected):
    df = _df_with_close_count(non_null=5, total=5)
    passed, reason = passes_sanity_gate(df, expected_days=expected, threshold=0.5)
    assert passed is False
    assert "must be positive" in reason
