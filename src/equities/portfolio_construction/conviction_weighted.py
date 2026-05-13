"""ConvictionWeightedVariant — B2. Softmax weighting within top-30.

Same top-30 selection as baseline. Within those 30 names, weight by
softmax of model scores with temperature T=0.5:

    weights_i = exp(score_i / T) / sum_j(exp(score_j / T))

T=0.5 is sharper than naive softmax (T=1), giving the highest-score
names noticeably more weight. Caps (7.5% individual, 30% sector) then
apply, which typically bind on the top 1-2 names under this temperature.

Tests whether the model's relative scores within the top tier carry
information beyond rank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equities.portfolio_construction.base import (
    ConstructionState,
    ConstructionVariant,
)
from src.equities.portfolio_construction.caps import enforce_caps


class ConvictionWeightedVariant(ConstructionVariant):
    """Top-N selection + softmax(score/T) weighting + caps."""

    name = "b2_conviction_weighted"

    def __init__(self, n: int = 30, temperature: float = 0.5,
                 individual_cap: float = 0.075, sector_cap: float = 0.30):
        self.n = n
        self.temperature = temperature
        self.individual_cap = individual_cap
        self.sector_cap = sector_cap

    def construct(self, state: ConstructionState) -> pd.Series:
        valid = state.scores.dropna()
        if valid.empty:
            return pd.Series(dtype=float)

        top = valid.nlargest(self.n)
        if top.empty:
            return pd.Series(dtype=float)

        # Softmax with temperature. Subtract max before exp for numerical
        # stability (doesn't change the relative weights).
        scaled = top.values / self.temperature
        scaled = scaled - scaled.max()
        raw = np.exp(scaled)
        normalized = raw / raw.sum()
        weights = pd.Series(normalized, index=top.index)

        weights = enforce_caps(weights, state.sectors,
                               self.individual_cap, self.sector_cap)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights

    def params_dict(self) -> dict:
        return {
            "method": "softmax_top_n",
            "n": self.n,
            "temperature": self.temperature,
            "individual_cap": self.individual_cap,
            "sector_cap": self.sector_cap,
        }
