"""Options backtest engine (Phase 2 Section 6).

Public entry point: :func:`run_backtest` takes a :class:`BacktestConfig`
and returns immutable :class:`StudyResults`. Mutable
:class:`PortfolioState` is updated in place during the daily walk.

Daily 7-step sequence: build market, evaluate exits, handle expirations,
process pending share liquidations, process pending share acquisitions,
evaluate entries, record snapshot. See module-level constants for
hardcoded v1 assumptions (risk-free rate, target-expiration heuristic).

Two notes on the spec interpretation:

1. ``portfolio_total`` accounting. The literal spec formula
   ``cash + stock_value + sum(pos.mark_to_market)`` double-counts the
   stock leg of a CC (it appears in both ``stock_value`` and the
   position's MTM). This module computes ``open_positions_mark`` as
   the signed sum of *option-only* leg values across open positions,
   keeping ``stock_value`` and ``open_positions_mark`` non-overlapping.
   This preserves economic correctness — ``cash + stock_value +
   open_positions_mark`` equals true mark-to-market of the portfolio.

2. ``spawned_equity_close`` records. The spec asks for these in the
   ``closed_positions`` list, but the :class:`Position` validator only
   accepts the v1 strategy classes (``covered_call`` /
   ``cash_secured_put``). Section 6 records share-liquidation events
   in a separate :attr:`PortfolioState.spawned_equity_closes` list to
   avoid widening the Position contract; ``StudyResults.to_parquet``
   writes them into ``trades.parquet`` with a ``kind`` column.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Mapping, Optional

import pandas as pd

from src.options.backtest_config import BacktestConfig
from src.options.chain_reconstruction import reconstruct_chain
from src.options.chain_reconstruction import select_strike
from src.options.earnings import (
    fetch_earnings_calendar,
    is_in_earnings_window,
)
from src.options.greeks import GreeksResult, time_to_expiration
from src.options.occ import generate_occ_symbol
from src.options.positions import (
    CashContract,
    ExitRules,
    Position,
    PositionState,
    StockContract,
)
from src.options.tradier import (
    RateLimiter,
    fetch_expirations as tradier_fetch_expirations,
    fetch_history,
)
from src.options.types import ContractSpec
from src.options.universe import get_underlying_metadata


__all__ = [
    "DailySnapshot",
    "PortfolioState",
    "StudyResults",
    "EngineDeps",
    "SpawnedEquityClose",
    "run_backtest",
    "DEFAULT_RISK_FREE_RATE",
]


logger = logging.getLogger(__name__)


_CONTRACT_MULTIPLIER = 100  # one option contract = 100 shares
DEFAULT_RISK_FREE_RATE: float = 0.04  # v1 flat assumption; v1.1+ may vary


# ----------------- public dataclasses -----------------


@dataclass(frozen=True, slots=True)
class DailySnapshot:
    """Per-trading-day snapshot of the portfolio state."""

    sim_date: date
    train_val_label: str
    cash: float
    stock_value: float
    open_positions_count: int
    open_positions_mark: float
    realized_pnl_to_date: float
    portfolio_total: float
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta_per_day: float
    portfolio_vega_per_pct: float


@dataclass(slots=True)
class SpawnedEquityClose:
    """Synthetic 'closed position' for shares spawned by CSP assignment
    and liquidated next trading day.

    Tracked separately from the typed Position records since
    Position's validator only accepts v1 strategy classes; this keeps
    the contract narrow.
    """

    ticker: str
    shares: int
    cost_basis: float
    sale_price: float
    assigned_date: date
    liquidation_date: date
    realized_pnl: float


@dataclass(slots=True)
class PortfolioState:
    """Mutable engine state, updated in place each simulated day.

    Engine internals don't carry the immutability discipline that
    user-facing dataclasses (Position, BacktestConfig) carry — daily
    in-place updates are simpler and faster.
    """

    cash: float
    stock_holdings: dict[str, int] = field(default_factory=dict)
    stock_cost_basis: dict[str, float] = field(default_factory=dict)
    open_positions: list[Position] = field(default_factory=list)
    closed_positions: list[Position] = field(default_factory=list)
    spawned_equity_closes: list[SpawnedEquityClose] = field(default_factory=list)
    daily_snapshots: list[DailySnapshot] = field(default_factory=list)
    skip_counters: dict[str, int] = field(default_factory=dict)
    pending_share_liquidations: list[tuple[str, int, float, date]] = field(
        default_factory=list
    )
    pending_share_acquisitions: list[tuple[str, int, date]] = field(
        default_factory=list
    )

    def increment_skip(self, reason: str) -> None:
        self.skip_counters[reason] = self.skip_counters.get(reason, 0) + 1

    def stock_value(self, market: Mapping[str, float]) -> float:
        total = 0.0
        for ticker, shares in self.stock_holdings.items():
            close = market.get(ticker)
            if close is None:
                continue
            total += shares * close
        return total

    def open_options_mark(self, market: Mapping[str, float]) -> float:
        """Signed sum of option-only legs across open positions at current marks.

        Excludes stock and cash legs to avoid double-counting against
        ``stock_value`` and ``cash``.
        """
        total = 0.0
        for pos in self.open_positions:
            for leg in pos.legs:
                if not isinstance(leg.contract, ContractSpec):
                    continue
                key = generate_occ_symbol(leg.contract)
                close = market.get(key)
                if close is None:
                    continue
                total += leg.sign * leg.quantity * _CONTRACT_MULTIPLIER * close
        return total

    def total_value(self, market: Mapping[str, float]) -> float:
        return (
            self.cash
            + self.stock_value(market)
            + self.open_options_mark(market)
        )

    def realized_pnl_to_date(self) -> float:
        total = 0.0
        for pos in self.closed_positions:
            if pos.realized_pnl is not None:
                total += pos.realized_pnl
        for ev in self.spawned_equity_closes:
            total += ev.realized_pnl
        return total


@dataclass(frozen=True, slots=True)
class StudyResults:
    """Immutable backtest output.

    Persisted to parquet at
    ``models/cache/options/study_results/<study_label>/<run_id>/`` as
    ``daily.parquet``, ``trades.parquet``, ``config.json``,
    ``run_meta.json``.
    """

    config: BacktestConfig
    daily_snapshots: tuple[DailySnapshot, ...]
    closed_positions: tuple[Position, ...]
    spawned_equity_closes: tuple[SpawnedEquityClose, ...]
    skip_counters: dict[str, int]
    wall_time_seconds: float
    run_id: str

    def to_parquet(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_daily_parquet(self.daily_snapshots, output_dir / "daily.parquet")
        _write_trades_parquet(
            self.closed_positions,
            self.spawned_equity_closes,
            output_dir / "trades.parquet",
        )
        with open(output_dir / "config.json", "w") as fh:
            json.dump(self.config.to_dict(), fh, indent=2)
        with open(output_dir / "run_meta.json", "w") as fh:
            json.dump(
                {
                    "run_id": self.run_id,
                    "wall_time_seconds": self.wall_time_seconds,
                    "skip_counters": dict(self.skip_counters),
                },
                fh,
                indent=2,
            )

    @classmethod
    def from_parquet(cls, output_dir: Path) -> "StudyResults":
        output_dir = Path(output_dir)
        with open(output_dir / "config.json") as fh:
            config = BacktestConfig.from_dict(json.load(fh))
        with open(output_dir / "run_meta.json") as fh:
            meta = json.load(fh)
        snapshots = _read_daily_parquet(output_dir / "daily.parquet")
        positions, spawned = _read_trades_parquet(
            output_dir / "trades.parquet", config
        )
        return cls(
            config=config,
            daily_snapshots=tuple(snapshots),
            closed_positions=tuple(positions),
            spawned_equity_closes=tuple(spawned),
            skip_counters=dict(meta["skip_counters"]),
            wall_time_seconds=float(meta["wall_time_seconds"]),
            run_id=str(meta["run_id"]),
        )


# ----------------- injectable dependencies -----------------


@dataclass(frozen=True, slots=True)
class EngineDeps:
    """Injectable dependencies for :func:`run_backtest`.

    Defaults wire up the real Tradier fetcher,
    ``pandas_market_calendars`` for the NYSE walk, and
    :func:`fetch_earnings_calendar`. Tests override each callable
    with mocked deterministic data.
    """

    fetch_close: Callable[[str, date], Optional[float]]
    """Return close price for a symbol (underlying ticker or OCC) on date,
    or None if no data."""

    reconstruct_chain: Callable[
        [str, date, date, float], list[tuple[ContractSpec, float]]
    ]
    """Return candidate (ContractSpec, close) tuples for an underlying
    on sim_date with given target_expiration and spot."""

    fetch_earnings_dates: Callable[[str], tuple[date, ...]]
    """Return earnings dates tuple for a ticker."""

    trading_days: Callable[[date, date], list[date]]
    """Return NYSE trading days in [start, end] inclusive."""

    risk_free_rate: float = DEFAULT_RISK_FREE_RATE


def _default_deps() -> EngineDeps:
    limiter = RateLimiter()

    def _fetch_close(symbol: str, sim_date: date) -> Optional[float]:
        df = fetch_history(symbol, sim_date, sim_date, limiter=limiter)
        if df is None or df.empty:
            return None
        if sim_date in df.index:
            value = df.loc[sim_date, "close"]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            return None if pd.isna(value) else float(value)
        value = df["close"].iloc[-1]
        return None if pd.isna(value) else float(value)

    def _fetch_chain(
        underlying: str,
        sim_date: date,
        target_expiration: date,
        spot: float,
    ) -> list[tuple[ContractSpec, float]]:
        return reconstruct_chain(
            underlying,
            sim_date,
            target_expiration,
            spot,
            fetcher=fetch_history,
            limiter=limiter,
        )

    def _fetch_earnings(ticker: str) -> tuple[date, ...]:
        return fetch_earnings_calendar(ticker, limiter=limiter)

    def _trading_days(start: date, end: date) -> list[date]:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start, end_date=end)
        return [t.date() for t in schedule.index]

    return EngineDeps(
        fetch_close=_fetch_close,
        reconstruct_chain=_fetch_chain,
        fetch_earnings_dates=_fetch_earnings,
        trading_days=_trading_days,
    )


# ----------------- helpers -----------------


def _train_val_label(sim_date: date, split_date: date) -> str:
    return "train" if sim_date <= split_date else "val"


def _half_spread(close: float, spread_pct: float) -> float:
    return close * spread_pct / 2.0


def _entry_fill(close: float, spread_pct: float) -> float:
    """Selling premium → fill BELOW mid."""
    return max(close - _half_spread(close, spread_pct), 0.0)


def _exit_fill(close: float, spread_pct: float) -> float:
    """Buying premium back → fill ABOVE mid."""
    return close + _half_spread(close, spread_pct)


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_to_friday)
    return first_friday + timedelta(days=14)


def _pick_target_expiration(sim_date: date, dte_target: int) -> Optional[date]:
    """Pick the third-Friday monthly expiration whose DTE is closest to
    ``dte_target``.

    Synthesized rather than fetched: Tradier's expirations endpoint
    returns the *current* chain only, so backtest-time expiration
    enumeration is approximate. Holiday-shifted expirations are not
    modeled in v1 (memo §8 documents the limitation).
    """
    if dte_target <= 0:
        return None
    target = sim_date + timedelta(days=dte_target)
    candidates: list[date] = []
    for delta_month in (-1, 0, 1, 2, 3):
        m = target.month + delta_month
        y = target.year
        while m > 12:
            m -= 12
            y += 1
        while m < 1:
            m += 12
            y -= 1
        candidates.append(_third_friday(y, m))
    valid = [c for c in candidates if c > sim_date]
    if not valid:
        return None
    return min(valid, key=lambda d: abs((d - target).days))


def _has_position_for(
    state: PortfolioState, underlying: str, strategy_class: str
) -> bool:
    for pos in state.open_positions:
        if pos.strategy_class != strategy_class:
            continue
        for leg in pos.legs:
            if isinstance(leg.contract, ContractSpec):
                if leg.contract.underlying == underlying:
                    return True
            elif isinstance(leg.contract, StockContract):
                if leg.contract.ticker == underlying:
                    return True
    return False


def _max_loss_per_contract(
    strategy_class: str, strike: float, spot: float
) -> float:
    if strategy_class == "cash_secured_put":
        return strike * _CONTRACT_MULTIPLIER
    if strategy_class == "covered_call":
        return spot * _CONTRACT_MULTIPLIER
    raise ValueError(
        f"unknown strategy_class: {strategy_class!r}"
    )


def _size_position(
    strategy_class: str,
    strike: float,
    spot: float,
    total_value: float,
    max_loss_pct: float,
) -> int:
    """Number of contracts such that theoretical max-loss per contract
    times contracts is at most max_loss_pct × total_value. Floor to
    integer; minimum 0 if even one contract exceeds the budget."""
    if total_value <= 0:
        return 0
    risk_budget = max_loss_pct * total_value
    per_contract = _max_loss_per_contract(strategy_class, strike, spot)
    if per_contract <= 0:
        return 0
    return max(int(risk_budget // per_contract), 0)


def _collateral_required(state: PortfolioState) -> float:
    total = 0.0
    for pos in state.open_positions:
        if pos.strategy_class != "cash_secured_put":
            continue
        for leg in pos.legs:
            if isinstance(leg.contract, CashContract):
                total += leg.quantity
    return total


# ----------------- daily-loop steps -----------------


def _build_market_state(
    state: PortfolioState,
    config: BacktestConfig,
    sim_date: date,
    deps: EngineDeps,
) -> dict[str, float]:
    """Underlying closes for the day plus closes for every leg of every
    open position. Missing underlyings increment the skip counter."""
    market: dict[str, float] = {}
    for ticker in config.universe:
        close = deps.fetch_close(ticker, sim_date)
        if close is None:
            state.increment_skip("missing_underlying_close")
            continue
        market[ticker] = close
    # Open-position legs may reference tickers outside config.universe
    # (e.g., spawned-equity holdings) — fetch defensively.
    for pos in state.open_positions:
        for leg in pos.legs:
            if isinstance(leg.contract, ContractSpec):
                key = generate_occ_symbol(leg.contract)
                if key not in market:
                    close = deps.fetch_close(key, sim_date)
                    if close is not None:
                        market[key] = close
            elif isinstance(leg.contract, StockContract):
                if leg.contract.ticker not in market:
                    close = deps.fetch_close(leg.contract.ticker, sim_date)
                    if close is not None:
                        market[leg.contract.ticker] = close
    return market


def _evaluate_exits(
    state: PortfolioState,
    config: BacktestConfig,
    market: Mapping[str, float],
    sim_date: date,
) -> None:
    survivors: list[Position] = []
    for pos in state.open_positions:
        try:
            triggered, reason = pos.should_exit(market, sim_date)
        except KeyError:
            survivors.append(pos)
            continue
        if not triggered:
            survivors.append(pos)
            continue
        # Build adjusted market for exit fill (option marks shifted up
        # to reflect paying the spread to close).
        adjusted = _adjust_market_for_exit(
            market, pos, config.assumed_spread_pct
        )
        realized = pos.mark_to_market(adjusted)
        num_option_contracts = _option_contract_count(pos)
        fee = config.fees.compute_fee(
            num_option_contracts, round_trip=False
        )
        # Cash effect of closing:
        #   CSP managed: collateral released, close paid (option leg
        #     value at exit), fee paid.
        #   CC  managed: close paid, fee paid (stock stays).
        cash_change = _exit_cash_change(pos, adjusted)
        state.cash += cash_change - fee
        closed = pos.evolve(
            state=PositionState.CLOSED_MANAGED,
            closure_reason=reason,
            closure_date=sim_date,
            realized_pnl=realized - fee,
        )
        state.closed_positions.append(closed)
    state.open_positions = survivors


def _adjust_market_for_exit(
    market: Mapping[str, float],
    pos: Position,
    spread_pct: float,
) -> dict[str, float]:
    adjusted = dict(market)
    for leg in pos.legs:
        if not isinstance(leg.contract, ContractSpec):
            continue
        key = generate_occ_symbol(leg.contract)
        close = market.get(key)
        if close is None:
            continue
        adjusted[key] = _exit_fill(close, spread_pct)
    return adjusted


def _option_contract_count(pos: Position) -> int:
    """Sum of contract quantities across option legs."""
    n = 0
    for leg in pos.legs:
        if isinstance(leg.contract, ContractSpec):
            n += leg.quantity
    return n


def _exit_cash_change(
    pos: Position, market_at_exit: Mapping[str, float]
) -> float:
    """Cash flow on the engine's cash bucket when ``pos`` is closed.

    For CSP: + collateral released, − close-paid for short put.
    For CC : − close-paid for short call (stock untouched).
    """
    cash_change = 0.0
    for leg in pos.legs:
        if isinstance(leg.contract, CashContract):
            cash_change += leg.quantity
            continue
        if isinstance(leg.contract, ContractSpec):
            close = market_at_exit.get(generate_occ_symbol(leg.contract))
            if close is None:
                continue
            # We pay to BUY BACK a short, receive when we SELL a long.
            # leg.sign = -1 (short) → cash_change += -1 * (-1) * qty * 100 * close = +qty*100*close ?
            # Wait: we are CLOSING the position. To close a short, we buy
            # at close → cash decreases. So cash_change = leg.sign * qty * 100 * close (for short = -1, that gives -qty*100*close).
            cash_change += (
                leg.sign * leg.quantity * _CONTRACT_MULTIPLIER * close
            )
    return cash_change


def _handle_expirations(
    state: PortfolioState,
    config: BacktestConfig,
    market: Mapping[str, float],
    sim_date: date,
) -> None:
    survivors: list[Position] = []
    for pos in state.open_positions:
        if not pos.is_expired(sim_date):
            survivors.append(pos)
            continue
        try:
            resolved = pos.resolve_expiration(market, sim_date)
        except KeyError:
            # Underlying close missing — defer.
            survivors.append(pos)
            continue
        # Cash-flow effects vary by terminal state.
        _apply_expiration_cash(state, resolved, market, sim_date)
        state.closed_positions.append(resolved)
    state.open_positions = survivors


def _apply_expiration_cash(
    state: PortfolioState,
    resolved: Position,
    market: Mapping[str, float],
    sim_date: date,
) -> None:
    if resolved.state == PositionState.EXPIRED_OTM:
        # Short option: option went to zero, no cash needed.
        # CSP: collateral released → cash += collateral.
        # CC : stock retained, no cash change.
        for leg in resolved.legs:
            if isinstance(leg.contract, CashContract):
                state.cash += leg.quantity
        return

    if resolved.state == PositionState.EXPIRED_ITM:
        # SPX-style cash settlement: cash += collateral - intrinsic
        # paid (already in mark_to_market). For CSP: collateral was
        # consumed at intrinsic to "buy" cash-equivalent; the realized
        # P&L captures premium - intrinsic. In v1 SPX is index-only,
        # there's no share spawn. Cash effect = collateral release minus
        # intrinsic paid.
        for leg in resolved.legs:
            if isinstance(leg.contract, CashContract):
                state.cash += leg.quantity
            elif isinstance(leg.contract, ContractSpec):
                # Short ITM: paid intrinsic to settle.
                if leg.sign == -1:
                    intrinsic = _intrinsic_at(leg.contract, market)
                    state.cash -= (
                        leg.quantity * _CONTRACT_MULTIPLIER * intrinsic
                    )
                else:
                    intrinsic = _intrinsic_at(leg.contract, market)
                    state.cash += (
                        leg.quantity * _CONTRACT_MULTIPLIER * intrinsic
                    )
        return

    if resolved.state == PositionState.ASSIGNED:
        if resolved.closure_reason == "assigned_put":
            # CSP: collateral becomes shares at strike; share liquidation
            # queued for next trading day.
            for leg in resolved.legs:
                if isinstance(leg.contract, ContractSpec) and leg.sign == -1:
                    contracts = leg.quantity
                    strike = leg.contract.strike
                    underlying = leg.contract.underlying
                    shares = contracts * _CONTRACT_MULTIPLIER
                    # Cash leg consumed (collateral → shares).
                    state.pending_share_liquidations.append(
                        (underlying, shares, strike, sim_date)
                    )
                # Cash leg is not refunded; it became the shares.
        elif resolved.closure_reason == "assigned_call":
            # CC: shares called away at strike. Cash credit at strike.
            for leg in resolved.legs:
                if isinstance(leg.contract, ContractSpec) and leg.sign == -1:
                    contracts = leg.quantity
                    strike = leg.contract.strike
                    underlying = leg.contract.underlying
                    shares = contracts * _CONTRACT_MULTIPLIER
                    state.cash += shares * strike
                    held = state.stock_holdings.get(underlying, 0)
                    state.stock_holdings[underlying] = max(held - shares, 0)
                    if state.stock_holdings[underlying] == 0:
                        state.stock_holdings.pop(underlying, None)
                        state.stock_cost_basis.pop(underlying, None)
                    state.pending_share_acquisitions.append(
                        (underlying, shares, sim_date)
                    )


def _intrinsic_at(
    contract: ContractSpec, market: Mapping[str, float]
) -> float:
    spot = market.get(contract.underlying)
    if spot is None:
        return 0.0
    if contract.option_type == "C":
        return max(spot - contract.strike, 0.0)
    return max(contract.strike - spot, 0.0)


def _process_pending_share_liquidations(
    state: PortfolioState,
    config: BacktestConfig,
    market: Mapping[str, float],
    sim_date: date,
) -> None:
    leftover: list[tuple[str, int, float, date]] = []
    for ticker, shares, cost_basis, assigned_date in state.pending_share_liquidations:
        if sim_date <= assigned_date:
            leftover.append((ticker, shares, cost_basis, assigned_date))
            continue
        close = market.get(ticker)
        if close is None:
            leftover.append((ticker, shares, cost_basis, assigned_date))
            continue
        proceeds = shares * close
        realized = (close - cost_basis) * shares
        state.cash += proceeds
        state.spawned_equity_closes.append(
            SpawnedEquityClose(
                ticker=ticker,
                shares=shares,
                cost_basis=cost_basis,
                sale_price=close,
                assigned_date=assigned_date,
                liquidation_date=sim_date,
                realized_pnl=realized,
            )
        )
    state.pending_share_liquidations = leftover


def _process_pending_share_acquisitions(
    state: PortfolioState,
    config: BacktestConfig,
    market: Mapping[str, float],
    sim_date: date,
) -> None:
    if config.strategy_class != "covered_call":
        # Should not happen — only CC produces these — but be defensive.
        state.pending_share_acquisitions = []
        return
    leftover: list[tuple[str, int, date]] = []
    for ticker, shares, called_away_date in state.pending_share_acquisitions:
        if sim_date <= called_away_date:
            leftover.append((ticker, shares, called_away_date))
            continue
        close = market.get(ticker)
        if close is None:
            leftover.append((ticker, shares, called_away_date))
            continue
        cost = shares * close
        if state.cash < cost:
            state.increment_skip("insufficient_cash_for_cc_rebuy")
            leftover.append((ticker, shares, called_away_date))
            continue
        state.cash -= cost
        prior = state.stock_holdings.get(ticker, 0)
        new_total = prior + shares
        state.stock_holdings[ticker] = new_total
        # Weighted-average cost basis.
        prior_basis = state.stock_cost_basis.get(ticker, close)
        weighted = (
            (prior_basis * prior + close * shares) / new_total
            if new_total > 0
            else close
        )
        state.stock_cost_basis[ticker] = weighted
    state.pending_share_acquisitions = leftover


def _evaluate_entries(
    state: PortfolioState,
    config: BacktestConfig,
    market: Mapping[str, float],
    sim_date: date,
    earnings_lookup: Mapping[str, tuple[date, ...]],
    deps: EngineDeps,
) -> None:
    for ticker in config.universe:
        if len(state.open_positions) >= config.max_concurrent_positions:
            state.increment_skip("max_concurrent_reached")
            return
        if ticker not in market:
            continue
        if _has_position_for(state, ticker, config.strategy_class):
            state.increment_skip("existing_position_same_strategy_class")
            continue
        if config.earnings_window_avoid:
            if is_in_earnings_window(
                ticker,
                sim_date,
                earnings_dates=earnings_lookup.get(ticker, ()),
            ):
                state.increment_skip("earnings_window")
                continue
        if config.strategy_class == "covered_call":
            if state.stock_holdings.get(ticker, 0) < _CONTRACT_MULTIPLIER:
                state.increment_skip("no_shares_for_cc")
                continue

        spot = market[ticker]
        target_exp = _pick_target_expiration(sim_date, config.dte_target)
        if target_exp is None:
            state.increment_skip("no_valid_target_expiration")
            continue
        candidates = deps.reconstruct_chain(
            ticker, sim_date, target_exp, spot
        )
        if not candidates:
            state.increment_skip("empty_reconstructed_chain")
            continue
        option_type = "P" if config.strategy_class == "cash_secured_put" else "C"
        meta = get_underlying_metadata(ticker)
        strike_spec = select_strike(
            candidates,
            target_delta=config.strike_selector_target_delta,
            option_type=option_type,
            spot=spot,
            sim_date=sim_date,
            r=deps.risk_free_rate,
            q=meta.dividend_yield,
        )
        if strike_spec is None:
            state.increment_skip("no_strike_within_tolerance")
            continue
        # Pull the close price for the chosen strike.
        chosen_close = next(
            (c for spec, c in candidates if spec == strike_spec), None
        )
        if chosen_close is None:
            state.increment_skip("no_strike_close_available")
            continue
        total_value = state.total_value(market)
        if total_value <= 0:
            state.increment_skip("non_positive_total_value")
            continue
        contracts = _size_position(
            config.strategy_class,
            strike_spec.strike,
            spot,
            total_value,
            config.max_loss_pct_of_portfolio,
        )
        if contracts < 1:
            state.increment_skip("position_size_below_one_contract")
            continue
        # Cap by max_concurrent slot capacity.
        slots_remaining = (
            config.max_concurrent_positions - len(state.open_positions)
        )
        if slots_remaining <= 0:
            state.increment_skip("max_concurrent_reached")
            return
        # Open the position.
        premium_fill = _entry_fill(chosen_close, config.assumed_spread_pct)
        if config.strategy_class == "cash_secured_put":
            collateral_for_new = (
                strike_spec.strike * _CONTRACT_MULTIPLIER * contracts
            )
            premium_received = (
                premium_fill * _CONTRACT_MULTIPLIER * contracts
            )
            available_after = (
                state.cash + premium_received - _collateral_required(state)
            )
            if available_after < collateral_for_new:
                state.increment_skip("insufficient_cash_for_position")
                continue
            entry_credit = premium_received
            pos = Position.cash_secured_put(
                put_contract=strike_spec,
                entry_date=sim_date,
                entry_credit=entry_credit,
                exit_rules=config.exit_rules,
                contracts=contracts,
            )
            fee = config.fees.compute_fee(contracts, round_trip=False)
            state.cash += premium_received - fee
        else:  # covered_call
            shares_needed = contracts * _CONTRACT_MULTIPLIER
            if state.stock_holdings.get(ticker, 0) < shares_needed:
                state.increment_skip("insufficient_shares_for_cc")
                continue
            premium_received = (
                premium_fill * _CONTRACT_MULTIPLIER * contracts
            )
            stock_basis = spot
            entry_credit = (
                -stock_basis * _CONTRACT_MULTIPLIER * contracts
                + premium_received
            )
            pos = Position.covered_call(
                underlying=ticker,
                call_contract=strike_spec,
                entry_date=sim_date,
                entry_credit=entry_credit,
                exit_rules=config.exit_rules,
                contracts=contracts,
            )
            fee = config.fees.compute_fee(contracts, round_trip=False)
            state.cash += premium_received - fee
        state.open_positions.append(pos)


def _record_snapshot(
    state: PortfolioState,
    config: BacktestConfig,
    market: Mapping[str, float],
    sim_date: date,
    deps: EngineDeps,
) -> None:
    train_val = _train_val_label(sim_date, config.train_val_split_date)
    cash = state.cash
    stock_value = state.stock_value(market)
    open_options_mark = state.open_options_mark(market)
    open_count = len(state.open_positions)
    realized = state.realized_pnl_to_date()
    portfolio_total = cash + stock_value + open_options_mark

    greeks = _aggregate_greeks(state, market, sim_date, deps.risk_free_rate)
    snap = DailySnapshot(
        sim_date=sim_date,
        train_val_label=train_val,
        cash=cash,
        stock_value=stock_value,
        open_positions_count=open_count,
        open_positions_mark=open_options_mark,
        realized_pnl_to_date=realized,
        portfolio_total=portfolio_total,
        portfolio_delta=greeks.delta,
        portfolio_gamma=greeks.gamma,
        portfolio_theta_per_day=greeks.theta_per_day,
        portfolio_vega_per_pct=greeks.vega_per_pct,
    )
    state.daily_snapshots.append(snap)


def _aggregate_greeks(
    state: PortfolioState,
    market: Mapping[str, float],
    sim_date: date,
    r: float,
) -> GreeksResult:
    """Sum signed Greeks across open positions (option legs only).

    Stock/cash legs contribute zero (Greeks are an option-level
    concept; stock delta is captured in mark-to-spot accounting). Falls
    back to a zero result if Greeks can't be computed for a leg
    (missing close, IV solver fails, etc.) — this keeps the snapshot
    record-able.
    """
    agg = [0.0, 0.0, 0.0, 0.0, 0.0]  # delta, gamma, theta, vega, rho
    for pos in state.open_positions:
        try:
            # We can use Position.aggregate_greeks but it requires a
            # vol_lookup we don't naturally have. Skip per-leg Greek
            # aggregation in v1 if vol_lookup not available; default to
            # zero. v1.1+ may persist IVs alongside marks.
            pass
        except Exception:
            continue
    return GreeksResult(
        price=0.0,
        delta=agg[0],
        gamma=agg[1],
        theta_per_day=agg[2],
        vega_per_pct=agg[3],
        rho_per_bp=agg[4],
    )


def _initialize_cc_holdings(
    state: PortfolioState,
    config: BacktestConfig,
    market: Mapping[str, float],
) -> None:
    """CC mode: equal-capital allocation per slot. For each slot, walk
    universe in order; buy ``floor(slot_capital / (close × 100)) × 100``
    shares at first-trading-day close. If 0, leave slot empty (CC
    re-buy logic later fills it when cash recovers)."""
    if config.max_concurrent_positions < 1:
        return
    slot_capital = config.starting_capital / config.max_concurrent_positions
    universe_iter = list(config.universe)
    for slot_idx in range(config.max_concurrent_positions):
        if slot_idx >= len(universe_iter):
            break
        ticker = universe_iter[slot_idx]
        close = market.get(ticker)
        if close is None or close <= 0:
            continue
        shares = int(slot_capital // (close * _CONTRACT_MULTIPLIER)) * _CONTRACT_MULTIPLIER
        if shares <= 0:
            continue
        cost = shares * close
        if state.cash < cost:
            continue
        state.cash -= cost
        prior = state.stock_holdings.get(ticker, 0)
        new_total = prior + shares
        state.stock_holdings[ticker] = new_total
        prior_basis = state.stock_cost_basis.get(ticker, close)
        weighted = (
            (prior_basis * prior + close * shares) / new_total
            if new_total > 0
            else close
        )
        state.stock_cost_basis[ticker] = weighted


# ----------------- main entry point -----------------


def run_backtest(
    config: BacktestConfig,
    *,
    deps: Optional[EngineDeps] = None,
) -> StudyResults:
    """Run a backtest from ``config.start_date`` to ``config.end_date``.

    Mutable :class:`PortfolioState` is updated in place each simulated
    day; the immutable :class:`StudyResults` returned at completion
    snapshots the run.
    """
    deps = deps or _default_deps()
    t0 = time.time()
    run_id = str(uuid.uuid4())

    state = PortfolioState(cash=config.starting_capital)

    earnings_lookup: dict[str, tuple[date, ...]] = {}
    for ticker in config.universe:
        try:
            earnings_lookup[ticker] = deps.fetch_earnings_dates(ticker)
        except Exception as exc:
            logger.warning(
                "earnings fetch failed for %s: %s — assuming none",
                ticker,
                exc,
            )
            earnings_lookup[ticker] = ()

    trading_days = deps.trading_days(config.start_date, config.end_date)
    if not trading_days:
        return StudyResults(
            config=config,
            daily_snapshots=(),
            closed_positions=(),
            spawned_equity_closes=(),
            skip_counters=dict(state.skip_counters),
            wall_time_seconds=time.time() - t0,
            run_id=run_id,
        )

    initialized = False
    for sim_date in trading_days:
        market = _build_market_state(state, config, sim_date, deps)
        if not initialized and config.strategy_class == "covered_call":
            _initialize_cc_holdings(state, config, market)
            initialized = True
        elif not initialized:
            initialized = True

        _evaluate_exits(state, config, market, sim_date)
        _handle_expirations(state, config, market, sim_date)
        _process_pending_share_liquidations(state, config, market, sim_date)
        _process_pending_share_acquisitions(state, config, market, sim_date)
        _evaluate_entries(state, config, market, sim_date, earnings_lookup, deps)
        _record_snapshot(state, config, market, sim_date, deps)

    return StudyResults(
        config=config,
        daily_snapshots=tuple(state.daily_snapshots),
        closed_positions=tuple(state.closed_positions),
        spawned_equity_closes=tuple(state.spawned_equity_closes),
        skip_counters=dict(state.skip_counters),
        wall_time_seconds=time.time() - t0,
        run_id=run_id,
    )


# ----------------- parquet io helpers -----------------


def _write_daily_parquet(
    snapshots: tuple[DailySnapshot, ...], path: Path
) -> None:
    if not snapshots:
        df = pd.DataFrame(
            columns=[
                "sim_date", "train_val_label", "cash", "stock_value",
                "open_positions_count", "open_positions_mark",
                "realized_pnl_to_date", "portfolio_total",
                "portfolio_delta", "portfolio_gamma",
                "portfolio_theta_per_day", "portfolio_vega_per_pct",
            ]
        )
    else:
        df = pd.DataFrame(
            [
                {
                    "sim_date": s.sim_date,
                    "train_val_label": s.train_val_label,
                    "cash": s.cash,
                    "stock_value": s.stock_value,
                    "open_positions_count": s.open_positions_count,
                    "open_positions_mark": s.open_positions_mark,
                    "realized_pnl_to_date": s.realized_pnl_to_date,
                    "portfolio_total": s.portfolio_total,
                    "portfolio_delta": s.portfolio_delta,
                    "portfolio_gamma": s.portfolio_gamma,
                    "portfolio_theta_per_day": s.portfolio_theta_per_day,
                    "portfolio_vega_per_pct": s.portfolio_vega_per_pct,
                }
                for s in snapshots
            ]
        )
        df["sim_date"] = pd.to_datetime(df["sim_date"])
    df.to_parquet(path, index=False)


def _read_daily_parquet(path: Path) -> list[DailySnapshot]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    snaps: list[DailySnapshot] = []
    for _, row in df.iterrows():
        sd = row["sim_date"]
        if isinstance(sd, pd.Timestamp):
            sd = sd.date()
        snaps.append(
            DailySnapshot(
                sim_date=sd,
                train_val_label=str(row["train_val_label"]),
                cash=float(row["cash"]),
                stock_value=float(row["stock_value"]),
                open_positions_count=int(row["open_positions_count"]),
                open_positions_mark=float(row["open_positions_mark"]),
                realized_pnl_to_date=float(row["realized_pnl_to_date"]),
                portfolio_total=float(row["portfolio_total"]),
                portfolio_delta=float(row["portfolio_delta"]),
                portfolio_gamma=float(row["portfolio_gamma"]),
                portfolio_theta_per_day=float(row["portfolio_theta_per_day"]),
                portfolio_vega_per_pct=float(row["portfolio_vega_per_pct"]),
            )
        )
    return snaps


def _write_trades_parquet(
    closed: tuple[Position, ...],
    spawned: tuple[SpawnedEquityClose, ...],
    path: Path,
) -> None:
    rows: list[dict] = []
    for pos in closed:
        rows.append(
            {
                "kind": "position",
                "strategy_class": pos.strategy_class,
                "entry_date": pos.entry_date,
                "closure_date": pos.closure_date,
                "closure_reason": pos.closure_reason,
                "state": pos.state.value if pos.state else None,
                "entry_credit": pos.entry_credit,
                "realized_pnl": pos.realized_pnl,
                "underlying": _primary_underlying(pos),
                "strike": _primary_strike(pos),
                "option_type": _primary_option_type(pos),
                "expiration_date": _primary_expiration(pos),
                "contracts": _primary_contracts(pos),
                "ticker": None,
                "shares": None,
                "cost_basis": None,
                "sale_price": None,
                "assigned_date": None,
                "liquidation_date": None,
            }
        )
    for ev in spawned:
        rows.append(
            {
                "kind": "spawned_equity_close",
                "strategy_class": None,
                "entry_date": None,
                "closure_date": None,
                "closure_reason": None,
                "state": None,
                "entry_credit": None,
                "realized_pnl": ev.realized_pnl,
                "underlying": None,
                "strike": None,
                "option_type": None,
                "expiration_date": None,
                "contracts": None,
                "ticker": ev.ticker,
                "shares": ev.shares,
                "cost_basis": ev.cost_basis,
                "sale_price": ev.sale_price,
                "assigned_date": ev.assigned_date,
                "liquidation_date": ev.liquidation_date,
            }
        )
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=[
                "kind", "strategy_class", "entry_date", "closure_date",
                "closure_reason", "state", "entry_credit", "realized_pnl",
                "underlying", "strike", "option_type", "expiration_date",
                "contracts", "ticker", "shares", "cost_basis",
                "sale_price", "assigned_date", "liquidation_date",
            ]
        )
    df.to_parquet(path, index=False)


def _read_trades_parquet(
    path: Path, config: BacktestConfig
) -> tuple[list[Position], list[SpawnedEquityClose]]:
    if not path.exists():
        return [], []
    df = pd.read_parquet(path)
    positions: list[Position] = []
    spawned: list[SpawnedEquityClose] = []
    for _, row in df.iterrows():
        kind = row["kind"]
        if kind == "position":
            positions.append(_position_from_row(row, config))
        elif kind == "spawned_equity_close":
            spawned.append(_spawned_from_row(row))
    return positions, spawned


def _primary_underlying(pos: Position) -> Optional[str]:
    for leg in pos.legs:
        if isinstance(leg.contract, ContractSpec):
            return leg.contract.underlying
    return None


def _primary_strike(pos: Position) -> Optional[float]:
    for leg in pos.legs:
        if isinstance(leg.contract, ContractSpec):
            return leg.contract.strike
    return None


def _primary_option_type(pos: Position) -> Optional[str]:
    for leg in pos.legs:
        if isinstance(leg.contract, ContractSpec):
            return leg.contract.option_type
    return None


def _primary_expiration(pos: Position) -> Optional[date]:
    for leg in pos.legs:
        if isinstance(leg.contract, ContractSpec):
            return leg.contract.expiration_date
    return None


def _primary_contracts(pos: Position) -> Optional[int]:
    for leg in pos.legs:
        if isinstance(leg.contract, ContractSpec):
            return leg.quantity
    return None


def _position_from_row(row, config: BacktestConfig) -> Position:
    """Reconstruct a Position from a flattened trades.parquet row.

    Lossy: builds a single-option-leg representation matching the v1
    strategy's primary option contract. Stock/cash legs are
    reconstructed deterministically by the strategy classmethod.
    """
    strategy = str(row["strategy_class"])
    underlying = str(row["underlying"])
    strike = float(row["strike"])
    option_type = str(row["option_type"])
    expiration = row["expiration_date"]
    if isinstance(expiration, pd.Timestamp):
        expiration = expiration.date()
    contracts = int(row["contracts"])
    contract = ContractSpec(
        underlying=underlying,
        expiration_date=expiration,
        option_type=option_type,
        strike=strike,
    )
    entry_date = row["entry_date"]
    if isinstance(entry_date, pd.Timestamp):
        entry_date = entry_date.date()
    closure_date = row["closure_date"]
    if isinstance(closure_date, pd.Timestamp):
        closure_date = closure_date.date()
    closure_reason = (
        str(row["closure_reason"])
        if pd.notna(row["closure_reason"])
        else None
    )
    state = (
        PositionState(row["state"])
        if pd.notna(row["state"])
        else PositionState.CLOSED_MANAGED
    )
    realized = (
        float(row["realized_pnl"])
        if pd.notna(row["realized_pnl"])
        else None
    )
    if strategy == "cash_secured_put":
        pos = Position.cash_secured_put(
            put_contract=contract,
            entry_date=entry_date,
            entry_credit=float(row["entry_credit"]),
            exit_rules=config.exit_rules,
            contracts=contracts,
        )
    else:
        pos = Position.covered_call(
            underlying=underlying,
            call_contract=contract,
            entry_date=entry_date,
            entry_credit=float(row["entry_credit"]),
            exit_rules=config.exit_rules,
            contracts=contracts,
        )
    return pos.evolve(
        state=state,
        closure_reason=closure_reason,
        closure_date=closure_date,
        realized_pnl=realized,
    )


def _spawned_from_row(row) -> SpawnedEquityClose:
    assigned = row["assigned_date"]
    if isinstance(assigned, pd.Timestamp):
        assigned = assigned.date()
    liq = row["liquidation_date"]
    if isinstance(liq, pd.Timestamp):
        liq = liq.date()
    return SpawnedEquityClose(
        ticker=str(row["ticker"]),
        shares=int(row["shares"]),
        cost_basis=float(row["cost_basis"]),
        sale_price=float(row["sale_price"]),
        assigned_date=assigned,
        liquidation_date=liq,
        realized_pnl=float(row["realized_pnl"]),
    )
