"""DynamicTopNVariant — B3. N varies 15-50 based on score-dispersion percentile.

At each rebalance:
  1. Compute std dev of model scores in the top decile of eligible
     tickers (today's "dispersion").
  2. Look up this dispersion's percentile in the frozen training-period
     distribution of top-decile dispersions.
  3. Linearly interpolate N between 50 (low conviction, broad portfolio)
     and 15 (high conviction, concentrated portfolio):
       N=50 when current dispersion is at the training 10th percentile
       N=15 when at the 90th percentile
       Linearly between; clamped at endpoints.
  4. Select top-N by score, equal-weight at 1/N, enforce caps.

The training_dispersion_dist is computed once at variant-config time
from the training-period top-decile dispersions across all training
rebalance dates. Frozen; no peek-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equities.portfolio_construction.base import (
    ConstructionState,
    ConstructionVariant,
)
from src.equities.portfolio_construction.caps import enforce_caps


class DynamicTopNVariant(ConstructionVariant):
    """Dynamic top-N based on score-dispersion percentile."""

    name = "b3_dynamic_topn"

    def __init__(self, n_low: int = 50, n_high: int = 15,
                 pct_low: float = 0.10, pct_high: float = 0.90,
                 training_dispersion_dist: list[float] | None = None,
                 individual_cap: float = 0.075, sector_cap: float = 0.30):
        """
        Args:
            n_low: N value at the low-dispersion percentile (broad portfolio).
            n_high: N value at the high-dispersion percentile (concentrated).
            pct_low / pct_high: clamp endpoints for the dispersion percentile.
            training_dispersion_dist: list/Series of training-period
                top-decile dispersions. Frozen at variant-config time.
                Required at construct() call time; init can defer.
            individual_cap / sector_cap: passed to enforce_caps.
        """
        self.n_low = n_low
        self.n_high = n_high
        self.pct_low = pct_low
        self.pct_high = pct_high
        self.individual_cap = individual_cap
        self.sector_cap = sector_cap
        self._training_dispersion_dist = (
            np.asarray(training_dispersion_dist, dtype=float)
            if training_dispersion_dist is not None else None
        )

    def set_training_dispersion(self, dispersions: list[float]) -> None:
        """Pin the training-period dispersion distribution (post-init)."""
        self._training_dispersion_dist = np.asarray(dispersions, dtype=float)

    def _compute_n(self, current_dispersion: float) -> int:
        if (self._training_dispersion_dist is None
                or len(self._training_dispersion_dist) == 0):
            raise RuntimeError(
                "B3 requires training_dispersion_dist; set it via "
                "set_training_dispersion() or pass at __init__"
            )
        # Percentile of current_dispersion in the training distribution
        pct = float((self._training_dispersion_dist <= current_dispersion).mean())
        # Clamp to [pct_low, pct_high]
        pct_clamped = max(self.pct_low, min(self.pct_high, pct))
        # Linear interpolation
        n_float = (self.n_low + (self.n_high - self.n_low)
                   * (pct_clamped - self.pct_low)
                   / (self.pct_high - self.pct_low))
        return int(round(n_float))

    def construct(self, state: ConstructionState) -> pd.Series:
        valid = state.scores.dropna()
        if valid.empty:
            return pd.Series(dtype=float)

        # Top decile
        decile_n = max(1, int(round(0.1 * len(valid))))
        top_decile = valid.nlargest(decile_n)
        current_dispersion = float(top_decile.std()) if len(top_decile) > 1 else 0.0

        n = self._compute_n(current_dispersion)
        top = valid.nlargest(n)
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
            "method": "dynamic_top_n",
            "n_low": self.n_low,
            "n_high": self.n_high,
            "pct_low": self.pct_low,
            "pct_high": self.pct_high,
            "training_dispersion_n_obs": (
                len(self._training_dispersion_dist)
                if self._training_dispersion_dist is not None else None
            ),
            "individual_cap": self.individual_cap,
            "sector_cap": self.sector_cap,
        }
