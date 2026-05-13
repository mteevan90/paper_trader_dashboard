"""DefensiveSleevesVariant — B5. 70/30 equity/defensive; 50/50 in stress.

Equity sleeve (constructed via BaselineVariant) takes a fixed allocation;
defensive sleeve splits the remainder between cash and SHY 50/50.

Regime detection: **hard threshold** on trailing 21-day SPY return.

    spy_21d_return = SPY.close[d] / SPY.close[d - 21 trading days] - 1
    if spy_21d_return < -0.05:
        equity, defensive = 0.50, 0.50      # stress
    else:
        equity, defensive = 0.70, 0.30      # normal

The discontinuity at the −5% threshold is INTENTIONAL. v2 tests whether
a step-function regime overlay improves consistency; smoothing the
trigger is a v3 refinement question that this study deliberately does
not address.

The variant returns a weights Series that may include the synthetic
ticker "SHY". The engine must handle SHY price lookups outside the
v1 universe (loaded from `models/cache/equities/finnhub/prices/SHY.parquet`).
"""
from __future__ import annotations

import pandas as pd

from src.equities.portfolio_construction.base import (
    ConstructionState,
    ConstructionVariant,
)
from src.equities.portfolio_construction.baseline import BaselineVariant


SHY_TICKER = "SHY"


class DefensiveSleevesVariant(ConstructionVariant):
    """70/30 equity/defensive normally; 50/50 in stress regime."""

    name = "b5_defensive_sleeves"

    def __init__(self, normal_equity_alloc: float = 0.70,
                 stress_equity_alloc: float = 0.50,
                 stress_trigger_spy_return: float = -0.05,
                 lookback_trading_days: int = 21,
                 baseline: BaselineVariant | None = None):
        """
        Args:
            normal_equity_alloc: Equity sleeve allocation in normal regime
                (0.70 → 30% defensive).
            stress_equity_alloc: Equity sleeve allocation in stress regime
                (0.50 → 50% defensive).
            stress_trigger_spy_return: Hard threshold on trailing 21d SPY
                return. < this value → stress regime. Default -0.05 (−5%).
            lookback_trading_days: 21 trading days for the trigger.
            baseline: BaselineVariant for the equity sleeve. Defaults to
                a fresh BaselineVariant().
        """
        self.normal_equity_alloc = normal_equity_alloc
        self.stress_equity_alloc = stress_equity_alloc
        self.stress_trigger_spy_return = stress_trigger_spy_return
        self.lookback_trading_days = lookback_trading_days
        self._baseline = baseline if baseline is not None else BaselineVariant()

    def _detect_regime(self, spy_history: pd.DataFrame | None) -> tuple[float, float]:
        """Return (equity_alloc, defensive_alloc) based on SPY trailing return."""
        if (spy_history is None
                or len(spy_history) < self.lookback_trading_days + 1):
            # Insufficient history (e.g., first rebalance) — default to normal
            return self.normal_equity_alloc, 1.0 - self.normal_equity_alloc
        spy_close = spy_history["close"]
        last = float(spy_close.iloc[-1])
        ref = float(spy_close.iloc[-(self.lookback_trading_days + 1)])
        spy_21d_return = (last / ref) - 1.0
        if spy_21d_return < self.stress_trigger_spy_return:
            return self.stress_equity_alloc, 1.0 - self.stress_equity_alloc
        return self.normal_equity_alloc, 1.0 - self.normal_equity_alloc

    def construct(self, state: ConstructionState) -> pd.Series:
        # Equity sleeve = baseline construction. Sums to ~1.0 internally.
        equity_weights = self._baseline.construct(state)
        if equity_weights.empty:
            # No equity selection possible; weights become cash-only.
            equity_weights = pd.Series(dtype=float)

        equity_alloc, defensive_alloc = self._detect_regime(state.spy_history)

        # Scale equity sleeve
        scaled_equity = equity_weights * equity_alloc

        # Defensive sleeve: 50/50 cash + SHY
        shy_alloc = defensive_alloc * 0.5
        # cash_alloc = defensive_alloc * 0.5 is implicit (residual)

        weights = scaled_equity.copy()
        if shy_alloc > 0:
            weights[SHY_TICKER] = shy_alloc
        return weights

    def params_dict(self) -> dict:
        return {
            "method": "defensive_sleeves",
            "normal_equity_alloc": self.normal_equity_alloc,
            "stress_equity_alloc": self.stress_equity_alloc,
            "stress_trigger_spy_return": self.stress_trigger_spy_return,
            "lookback_trading_days": self.lookback_trading_days,
            "trigger_discontinuity": "hard_threshold_intentional",
            "defensive_asset_split": {"cash": 0.5, "SHY": 0.5},
            "baseline_params": self._baseline.params_dict(),
        }
