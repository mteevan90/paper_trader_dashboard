"""SmallerCapsVariant — B6. Baseline with individual cap tightened to 4%.

Identical to BaselineVariant except `individual_cap=0.04` instead of
`0.075`. At equal-weight 1/30 = 3.33%, the 4% cap doesn't bind on the
initial allocation, but it DOES bind during sector-cap redistribution
when sector overflow flows to other names — limiting how concentrated
any single name can become after that flow.

Tests whether stricter individual caps improve effective diversification
during sector-cap-driven redistribution.
"""
from __future__ import annotations

from src.equities.portfolio_construction.baseline import BaselineVariant


class SmallerCapsVariant(BaselineVariant):
    """Baseline with individual cap reduced from 7.5% to 4%."""

    name = "b6_smaller_caps"

    def __init__(self, n: int = 30, individual_cap: float = 0.04,
                 sector_cap: float = 0.30):
        super().__init__(n=n, individual_cap=individual_cap,
                         sector_cap=sector_cap)
