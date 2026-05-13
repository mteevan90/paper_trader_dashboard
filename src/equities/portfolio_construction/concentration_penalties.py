"""ConcentrationPenaltiesVariant — B4. Persistence + sector-overweight penalties.

Two penalties applied to scores BEFORE top-30 selection so concentration
forms get rotated out rather than persisting:

  1. Persistence penalty: if a ticker has been in top-30 for ≥6
     consecutive rebalances, multiply its score by:
         factor = max(0.50, 1.0 - 0.10 * (streak - 5))
     Streak resets to 0 when the ticker exits top-30.

  2. Sector-overweight penalty: if a sector accounts for >20% of the
     PRE-REBALANCE portfolio, all tickers in that sector get their
     scores multiplied by 0.80.

Then standard rank top-30 equal-weight + caps applies on the effective
scores.

The engine tracks `top30_streak` (per-ticker consecutive count) and
`prev_portfolio_sector_weights` (sector totals from last rebalance)
and threads them via ConstructionState.
"""
from __future__ import annotations

import pandas as pd

from src.equities.portfolio_construction.base import (
    ConstructionState,
    ConstructionVariant,
)
from src.equities.portfolio_construction.caps import enforce_caps


class ConcentrationPenaltiesVariant(ConstructionVariant):
    """Persistence + sector-overweight penalties applied at score level."""

    name = "b4_concentration_penalties"

    def __init__(self, n: int = 30,
                 individual_cap: float = 0.075, sector_cap: float = 0.30,
                 persistence_streak_threshold: int = 6,
                 persistence_step: float = 0.10,
                 persistence_floor: float = 0.50,
                 sector_overweight_threshold: float = 0.20,
                 sector_overweight_factor: float = 0.80):
        self.n = n
        self.individual_cap = individual_cap
        self.sector_cap = sector_cap
        self.persistence_streak_threshold = persistence_streak_threshold
        self.persistence_step = persistence_step
        self.persistence_floor = persistence_floor
        self.sector_overweight_threshold = sector_overweight_threshold
        self.sector_overweight_factor = sector_overweight_factor

    def _persistence_factor(self, ticker: str, streak: dict) -> float:
        s = streak.get(ticker, 0)
        if s < self.persistence_streak_threshold:
            return 1.0
        # 0.90 at streak=6, 0.80 at 7, ..., capped at persistence_floor (0.50)
        reduction = self.persistence_step * (s - (self.persistence_streak_threshold - 1))
        factor = 1.0 - reduction
        return max(self.persistence_floor, factor)

    def construct(self, state: ConstructionState) -> pd.Series:
        valid = state.scores.dropna()
        if valid.empty:
            return pd.Series(dtype=float)

        # Persistence penalty
        persistence_factors = pd.Series(
            {t: self._persistence_factor(t, state.top30_streak)
             for t in valid.index},
            index=valid.index,
        )

        # Sector-overweight penalty
        overweight = {
            sec for sec, w in state.prev_portfolio_sector_weights.items()
            if w > self.sector_overweight_threshold
        }
        sector_factors = pd.Series(
            {t: (self.sector_overweight_factor
                 if state.sectors.get(t) in overweight else 1.0)
             for t in valid.index},
            index=valid.index,
        )

        effective = valid * persistence_factors * sector_factors
        top = effective.nlargest(self.n)
        if top.empty:
            return pd.Series(dtype=float)

        weights = pd.Series(1.0 / len(top), index=top.index)
        weights = enforce_caps(weights, state.sectors,
                               self.individual_cap, self.sector_cap)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights

    def params_dict(self) -> dict:
        return {
            "method": "concentration_penalties_top_n",
            "n": self.n,
            "individual_cap": self.individual_cap,
            "sector_cap": self.sector_cap,
            "persistence_streak_threshold": self.persistence_streak_threshold,
            "persistence_step": self.persistence_step,
            "persistence_floor": self.persistence_floor,
            "sector_overweight_threshold": self.sector_overweight_threshold,
            "sector_overweight_factor": self.sector_overweight_factor,
        }
