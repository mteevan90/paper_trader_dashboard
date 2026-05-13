"""VolTargetVariant — B1. 15% annualized vol target via gross scaling.

Same selection as baseline; scales gross exposure based on trailing
63-day realized portfolio volatility:
  - If realized_vol > 15% annualized: scale down (cash buffer increases).
  - If realized_vol <= 15%: scale stays at 1.0 (no leverage).

Warmup (first 63 trading days have no prior portfolio history): use the
`training_tail_vol` parameter — the baseline portfolio's last-63-day
realized vol from the training period, computed once at training time
and frozen into the variant config. No peek-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equities.portfolio_construction.base import (
    ConstructionState,
    ConstructionVariant,
)
from src.equities.portfolio_construction.baseline import BaselineVariant


class VolTargetVariant(ConstructionVariant):
    """Vol-targeted gross exposure scaling. Selection unchanged from baseline."""

    name = "b1_vol_target"

    def __init__(self, target_vol: float = 0.15, lookback_days: int = 63,
                 training_tail_vol: float | None = None,
                 baseline: BaselineVariant | None = None):
        """
        Args:
            target_vol: Annualized vol target (0.15 = 15%).
            lookback_days: Trailing window for realized vol calc (63 trading
                days ≈ 3 calendar months).
            training_tail_vol: Frozen warmup vol from training-period-tail
                baseline portfolio. Used until the running portfolio has
                `lookback_days` of history. None disables warmup (scale=1
                until lookback fills, which has slight peek-ahead risk in
                the first few months; pin this at study config time).
            baseline: Optional BaselineVariant instance for selection.
                Defaults to a fresh BaselineVariant().
        """
        self.target_vol = target_vol
        self.lookback_days = lookback_days
        self.training_tail_vol = training_tail_vol
        self._baseline = baseline if baseline is not None else BaselineVariant()

    def construct(self, state: ConstructionState) -> pd.Series:
        # Selection unchanged from baseline
        base = self._baseline.construct(state)
        if base.empty:
            return base

        # Compute realized portfolio vol
        if (state.portfolio_history is None
                or len(state.portfolio_history) < self.lookback_days):
            realized_vol = self.training_tail_vol
            if realized_vol is None:
                # No history, no frozen warmup → no scaling (scale=1)
                return base
        else:
            recent = state.portfolio_history.iloc[-self.lookback_days:]
            realized_vol = float(recent.std() * np.sqrt(252))

        # Scale gross exposure (long-only, no leverage)
        if realized_vol > 1e-6:
            scale = min(1.0, self.target_vol / realized_vol)
        else:
            scale = 1.0
        return base * scale

    def params_dict(self) -> dict:
        return {
            "method": "vol_targeted_baseline",
            "target_vol": self.target_vol,
            "lookback_days": self.lookback_days,
            "training_tail_vol": self.training_tail_vol,
            "baseline_params": self._baseline.params_dict(),
        }
