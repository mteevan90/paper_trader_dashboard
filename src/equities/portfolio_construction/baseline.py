"""BaselineVariant — v1 reproduction. Rank top-30 equal-weight + caps.

v2-baseline serves as both:
  (a) Reproducibility check against v1's headline numbers (Gate 3 gate:
      <1% deviation from v1 results).
  (b) Control against which the six treatment variants (B1-B6) are
      measured for the comparative success criteria.

The algorithm is unchanged from v1's `rank_top_n_weights`:
  1. Drop NaN scores; take top N by score (default 30).
  2. Equal-weight at 1/N.
  3. Enforce individual cap, then sector cap (with re-individual-cap
     inside the sector-cap iteration).
  4. Renormalize to sum=1.0 (fully invested).
"""
from __future__ import annotations

import pandas as pd

from src.equities.portfolio_construction.base import (
    ConstructionState,
    ConstructionVariant,
)
from src.equities.portfolio_construction.caps import enforce_caps


class BaselineVariant(ConstructionVariant):
    """v1 rank-top-N equal-weight + caps + renormalize-to-100%."""

    name = "baseline"

    def __init__(self, n: int = 30, individual_cap: float = 0.075,
                 sector_cap: float = 0.30):
        self.n = n
        self.individual_cap = individual_cap
        self.sector_cap = sector_cap

    def construct(self, state: ConstructionState) -> pd.Series:
        valid = state.scores.dropna()
        if valid.empty:
            return pd.Series(dtype=float)

        top = valid.nlargest(self.n)
        if top.empty:
            return pd.Series(dtype=float)

        weights = pd.Series(1.0 / len(top), index=top.index)
        weights = enforce_caps(weights, state.sectors,
                               self.individual_cap, self.sector_cap)
        # Renormalize to 100% invested (v1 convention)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights

    def params_dict(self) -> dict:
        return {
            "method": "rank_top_n",
            "n": self.n,
            "individual_cap": self.individual_cap,
            "sector_cap": self.sector_cap,
        }
