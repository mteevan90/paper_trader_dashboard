"""Portfolio construction for the Larger Universe v1 study.

Implements the rank-based top-N transformation locked at the Phase 4
spec gate: select the top N stocks by predicted score, equal-weight them,
then enforce individual and sector caps iteratively.

Per Mike's reasoning at the Phase 4 gate: at our IC magnitude (0.028),
score-magnitude differences within the top decile carry more noise than
signal. Rank-based top-N is more robust than softmax-weighted approaches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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


def _enforce_individual_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Iteratively cap weights at `cap`, redistributing excess to uncapped
    positions proportionally to their current weight."""
    w = weights.copy()
    for _ in range(10):  # bounded iteration; should converge in 1-2 passes
        over = w[w > cap]
        if over.empty:
            break
        excess = (over - cap).sum()
        w.loc[over.index] = cap
        under = w[w < cap]
        if under.empty or under.sum() == 0:
            break
        w.loc[under.index] += excess * (under / under.sum())
    return w


def _enforce_sector_cap(weights: pd.Series, sectors: pd.Series,
                        sector_cap: float, individual_cap: float) -> pd.Series:
    """Iteratively cap each sector's total weight at `sector_cap`,
    redistributing excess to under-cap sectors proportionally.

    Sector cap interaction with individual cap: when an over-sector is
    scaled down, the freed weight goes to other sectors' existing
    positions; if those tickers exceed the individual cap as a result,
    the individual-cap pass below corrects them. Two-pass converges in
    practice for any sane top-N + cap combination.
    """
    w = weights.copy()
    for _ in range(10):
        sec_totals = w.groupby(sectors).sum()
        over_sectors = sec_totals[sec_totals > sector_cap]
        if over_sectors.empty:
            break
        # For each over-sector: scale its tickers down so the sector total = cap
        for sec_name, total in over_sectors.items():
            sec_mask = (sectors == sec_name)
            scale = sector_cap / total
            w.loc[sec_mask] = w.loc[sec_mask] * scale
        # Redistribute the freed weight to under-sectors' tickers
        # (proportional to their current weight)
        excess = (over_sectors - sector_cap).sum()
        under_mask = ~sectors.isin(over_sectors.index)
        under_weight_sum = w.loc[under_mask].sum()
        if under_weight_sum > 0 and excess > 0:
            w.loc[under_mask] += excess * (w.loc[under_mask] / under_weight_sum)
        # Re-enforce individual cap after redistribution
        w = _enforce_individual_cap(w, individual_cap)
    return w
