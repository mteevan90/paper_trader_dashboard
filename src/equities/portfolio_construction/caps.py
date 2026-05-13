"""Cap-enforcement utilities shared across all v2 construction variants.

Extracted verbatim from v1's `src/equities/study/portfolio.py` so that
v2-baseline reproduces v1's headline numbers exactly. The Gate 3
reproducibility check (<1% deviation) validates this.

v1's algorithm:
    1. Individual cap pass — cap any weight above `individual_cap`,
       redistribute the excess proportionally to under-cap positions
       (iterative; bounded by max_iter to handle cascades).
    2. Sector cap pass — for each over-cap sector, scale all its
       weights down so the sector total = cap. Redistribute the freed
       weight to OTHER sectors' positions (proportionally to their
       current weight). Then re-enforce individual cap inside the
       sector-cap iteration.

Final renormalization to sum=1.0 happens in the *variant* (typically
BaselineVariant.construct) so different variants can choose whether to
renormalize (baseline=yes, vol-target=no since gross scales below 1).
"""
from __future__ import annotations

import pandas as pd


def enforce_individual_cap(weights: pd.Series, cap: float,
                            max_iter: int = 10) -> pd.Series:
    """Iteratively cap weights at `cap`; redistribute excess to uncapped
    positions proportionally to their current weight.

    Mirrors v1's _enforce_individual_cap exactly.
    """
    w = weights.copy()
    for _ in range(max_iter):
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


def enforce_sector_cap(weights: pd.Series, sectors: pd.Series,
                       sector_cap: float, individual_cap: float,
                       max_iter: int = 10) -> pd.Series:
    """Iteratively cap each sector's total at `sector_cap`; redistribute
    the freed weight to other sectors' positions; re-enforce individual
    cap inside each iteration.

    Mirrors v1's _enforce_sector_cap exactly.
    """
    w = weights.copy()
    for _ in range(max_iter):
        sec_totals = w.groupby(sectors).sum()
        over_sectors = sec_totals[sec_totals > sector_cap]
        if over_sectors.empty:
            break
        # Scale each over-sector's tickers so the sector total = cap
        for sec_name, total in over_sectors.items():
            sec_mask = (sectors == sec_name)
            scale = sector_cap / total
            w.loc[sec_mask] = w.loc[sec_mask] * scale
        # Redistribute the freed weight to under-sectors' tickers,
        # proportional to their current weight
        excess = (over_sectors - sector_cap).sum()
        under_mask = ~sectors.isin(over_sectors.index)
        under_weight_sum = w.loc[under_mask].sum()
        if under_weight_sum > 0 and excess > 0:
            w.loc[under_mask] += excess * (w.loc[under_mask] / under_weight_sum)
        # Re-enforce individual cap after redistribution
        w = enforce_individual_cap(w, individual_cap, max_iter=max_iter)
    return w


def enforce_caps(weights: pd.Series, sectors: pd.Series,
                 individual_cap: float, sector_cap: float,
                 max_iter: int = 10) -> pd.Series:
    """Apply individual cap then sector cap in v1's exact order.

    Convenience wrapper. The sector cap pass internally re-enforces the
    individual cap; the initial individual cap pass handles the case
    where raw weights (pre-sector-cap) already exceed the individual
    cap (e.g., conviction-weighted softmax with concentrated scores).
    """
    aligned = sectors.reindex(weights.index).fillna("sector_unknown")
    w = enforce_individual_cap(weights, individual_cap, max_iter=max_iter)
    w = enforce_sector_cap(w, aligned, sector_cap, individual_cap,
                            max_iter=max_iter)
    return w
