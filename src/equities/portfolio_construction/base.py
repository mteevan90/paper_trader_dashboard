"""Base types for portfolio construction variants.

Each variant subclasses `ConstructionVariant` and implements `construct()`.
The backtest engine passes a `ConstructionState` containing per-rebalance
context (scores, sectors, current portfolio state, optional history data
like SPY/SHY for variants that need them).

Variants pull what they need from the state; unused fields are ignored.
This keeps the contract narrow while letting heterogeneous variants share
one engine entry point.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ConstructionState:
    """Per-rebalance context passed to ConstructionVariant.construct().

    Engine fills these fields from running backtest state at each rebalance
    date. Variants that don't need a field can ignore it.

    Required (every variant uses these):
      date            — the rebalance timestamp
      scores          — Series(ticker -> score), already filtered to
                        eligible-on-date tickers; NaN scores dropped
      sectors         — Series(ticker -> sector), full universe sector map

    Optional (variants pull these on demand):
      current_weights — Series(ticker -> pre-rebalance weight); empty
                        Series at the first rebalance
      portfolio_history — Series(date -> daily portfolio return) for the
                        deployed portfolio so far; None at warmup
      spy_history     — DataFrame indexed by date with at least 'close'
                        column; SPY prices up to and including `date`
      shy_price       — float; SHY close on `date` (for B5's defensive
                        sleeve allocation)
      top30_streak    — dict(ticker -> int); consecutive-rebalance count
                        of being in top-30 (for B4 persistence penalty)
      prev_portfolio_sector_weights — dict(sector -> total weight) for
                        the pre-rebalance portfolio (for B4 sector
                        overweight penalty)
    """
    date: pd.Timestamp
    scores: pd.Series
    sectors: pd.Series
    current_weights: pd.Series = field(default_factory=pd.Series)
    portfolio_history: Optional[pd.Series] = None
    spy_history: Optional[pd.DataFrame] = None
    shy_price: Optional[float] = None
    top30_streak: dict = field(default_factory=dict)
    prev_portfolio_sector_weights: dict = field(default_factory=dict)


class ConstructionVariant(ABC):
    """Abstract base for portfolio construction variants.

    Subclasses declare their `name` (matching `variant_meta.json`) and
    implement `construct()`. Variants may carry instance state (e.g.,
    frozen training distributions) set at __init__.

    State that mutates across rebalances (e.g., B4's top30_streak) lives
    in `ConstructionState` and is managed by the engine, not the variant.
    """

    name: str = ""

    @abstractmethod
    def construct(self, state: ConstructionState) -> pd.Series:
        """Return target weights at this rebalance.

        Returns:
          Series indexed by ticker; weights are non-negative and typically
          sum to 1.0 (fully invested). Variants that intentionally leave
          cash (e.g., B1 vol-targeting when realized_vol > target, or B5's
          defensive sleeve) return a Series summing to < 1.0; the engine
          treats the residual as cash.

          For B5 specifically, the index may include the SHY ticker (a
          non-universe asset). The engine must handle this case.
        """
        ...

    def params_dict(self) -> dict:
        """Variant-specific parameters for persistence in meta.json.

        Default returns the instance's __dict__; variants can override to
        exclude implementation details or transform internal state.
        """
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_")}
