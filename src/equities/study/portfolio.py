"""Portfolio construction for the Larger Universe v1 study.

Implements the rank-based top-N transformation locked at the Phase 4
spec gate: select the top N stocks by predicted score, equal-weight them,
then enforce individual and sector caps iteratively.

Per Mike's reasoning at the Phase 4 gate: at our IC magnitude (0.028),
score-magnitude differences within the top decile carry more noise than
signal. Rank-based top-N is more robust than softmax-weighted approaches.

v2 NOTE (2026-05-12): the cap-enforcement primitives that lived here as
`_enforce_individual_cap` and `_enforce_sector_cap` have been moved to
`src/equities/portfolio_construction/caps.py` so v2's variants can share
them. This module imports them back as private aliases for v1
backward-compat; `rank_top_n_weights` is unchanged in behavior. The v2
study's `BaselineVariant` produces bit-identical results via the same
shared `caps` utility — Gate 3 validates with the <1% reproducibility
tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.equities.portfolio_construction.caps import (
    enforce_individual_cap as _enforce_individual_cap,
    enforce_sector_cap as _enforce_sector_cap,
)


@dataclass(frozen=True)
class PortfolioConstructionParams:
    """Locked Phase 4 portfolio construction parameters (Larger Universe v1)."""
    method: str = "rank_top_n"
    n: int = 30
    individual_cap: float = 0.075
    sector_cap: float = 0.30

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n": self.n,
            "individual_cap": self.individual_cap,
            "sector_cap": self.sector_cap,
        }


def rank_top_n_weights(scores: pd.Series,
                       sectors: pd.Series,
                       params: PortfolioConstructionParams) -> pd.Series:
    """Build target weights from cross-sectional scores at a single rebalance date.

    Algorithm:
      1. Rank tickers by score (descending); take top N
      2. Equal-weight at 1/N
      3. Apply individual cap (mostly a no-op at 1/30 = 3.33% < 7.5%, but
         enforced for forward-compatibility with smaller N choices)
      4. Apply sector cap iteratively until no sector exceeds the limit
      5. Re-normalize to sum to 1.0

    Args:
      scores: Series indexed by ticker; predicted forward returns for the
              cross-section on the rebalance date. NaN scores are dropped.
      sectors: Series indexed by ticker; sector classification (incl.
               "sector_unknown" for tickers absent from sector_map).
      params: PortfolioConstructionParams; locks the algorithm's knobs.

    Returns:
      Series indexed by ticker; weights sum to 1.0, all non-negative.
      Tickers not in the top-N (after cap enforcement) are omitted.
    """
    valid = scores.dropna()
    if valid.empty:
        return pd.Series(dtype=float)

    # Step 1: top N by score
    top = valid.nlargest(params.n)
    if top.empty:
        return pd.Series(dtype=float)

    # Step 2: equal weight
    weights = pd.Series(1.0 / len(top), index=top.index)

    # Step 3: individual cap (iterative redistribution)
    weights = _enforce_individual_cap(weights, params.individual_cap)

    # Step 4: sector cap (iterative)
    sec = sectors.reindex(weights.index).fillna("sector_unknown")
    weights = _enforce_sector_cap(weights, sec, params.sector_cap,
                                    params.individual_cap)

    # Step 5: renormalize (caps may leave residual capital; redistribute
    # within the existing position set to keep the portfolio fully invested
    # at 100%).
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return weights
