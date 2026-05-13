"""Monthly-rebalance backtest engine for the Larger Universe studies.

Separate from the legacy src/backtest.py engine (which is composite-weighted
+ regime-dependent ATR stops). This engine is purpose-built for the
contract v1 architecture: monthly rebalance, score-weighted continuous
sizing, close-to-close attribution, forced-exit on delisting with cash
drift until next rebalance.

v2 extension (2026-05-12): supports pluggable `ConstructionVariant`
instances from `src/equities/portfolio_construction/`. v1's call path
(via `pc_params: PortfolioConstructionParams`) is preserved for backward
compat — if both `construction_variant` and `pc_params` are provided,
`construction_variant` wins. If neither is provided, the engine errors.

State threaded into the variant via `ConstructionState` includes:
  - portfolio_history (daily portfolio returns, for B1 vol-targeting)
  - spy_history (SPY OHLCV up to and including current date, for B5
    regime detection)
  - top30_streak (per-ticker consecutive-rebalance count, for B4)
  - prev_portfolio_sector_weights (last-rebalance sector totals, for B4)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.equities.portfolio_construction.base import (
    ConstructionState,
    ConstructionVariant,
)
from src.equities.study.portfolio import PortfolioConstructionParams, rank_top_n_weights


TRANSACTION_COST_PCT = 0.0005  # 0.05% per trade leg


@dataclass
class BacktestResult:
    """Outputs of a single-model backtest. Long-format dataframes ready
    for contract v1 persistence."""
    portfolio: pd.DataFrame   # date, model, nav, cash_pct, n_positions, gross_exposure
    holdings:  pd.DataFrame   # date, model, ticker, weight, value_usd, sector, tier
    trades:    pd.DataFrame   # date, model, ticker, action, weight_change, price, notional_usd, fee_usd, reason
    scores:    pd.DataFrame   # date, model, ticker, score, rank, target_realized


def month_end_trading_dates(trading_dates: pd.DatetimeIndex,
                             start: pd.Timestamp,
                             end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return the last trading day of each calendar month between
    [start, end] inclusive, intersected with the provided trading_dates."""
    tdates = trading_dates[(trading_dates >= start) & (trading_dates <= end)]
    df = pd.DataFrame({"date": tdates})
    df["year_month"] = df["date"].dt.to_period("M")
    last_of_month = df.groupby("year_month")["date"].max()
    return list(last_of_month.values)


def eligible_universe_on(date_ts: pd.Timestamp,
                          universe_records: list[dict]) -> set[str]:
    """Return tickers active on `date_ts`: status==active OR
    (status==removed AND removed_at > date_ts)."""
    date_str = date_ts.strftime("%Y-%m-%d")
    eligible = set()
    seen = set()
    # Dedupe by symbol, prefer-active (mirrors fetcher's universe loader)
    by_sym = {}
    for r in universe_records:
        s = r["symbol"]
        if s not in by_sym or (by_sym[s]["status"] == "removed" and r["status"] == "active"):
            by_sym[s] = r
    for sym, r in by_sym.items():
        if r["status"] == "active":
            eligible.add(sym)
        elif r["status"] == "removed":
            rd = r.get("removed_at")
            if rd and rd > date_str:
                eligible.add(sym)
    return eligible


def run_backtest(
    model_name: str,
    score_fn,                          # callable: date -> Series(ticker -> score)
    rebalance_dates: list[pd.Timestamp],
    daily_returns: pd.DataFrame,       # index=date, columns=ticker; daily simple returns
    delisting_dates: dict[str, pd.Timestamp],  # ticker -> last-tradable date (None if active)
    sectors: pd.Series,                # index=ticker; sector classification
    tiers: pd.Series,                  # index=ticker; SP500/SP400/SP600/removed
    pc_params: Optional[PortfolioConstructionParams] = None,
    universe_records: Optional[list[dict]] = None,
    starting_capital: float = 1_000_000.0,
    fee_pct: float = TRANSACTION_COST_PCT,
    construction_variant: Optional[ConstructionVariant] = None,
    spy_history: Optional[pd.DataFrame] = None,
    shy_prices: Optional[pd.Series] = None,
) -> BacktestResult:
    """Run the monthly-rebalance backtest for a single model.

    Args:
      model_name: identifier written into the long-format dataframes
      score_fn:   callable returning a Series of (ticker -> predicted score)
                  for the rebalance date passed in
      rebalance_dates: month-end trading dates within the backtest window
      daily_returns: wide DataFrame of daily simple returns, indexed by date.
                     For v2-B5 (defensive sleeves), the caller adds SHY as
                     a column so the engine drifts the SHY position naturally.
      delisting_dates: ticker -> last-tradable date (Wikipedia removed_at);
                       None for active tickers. SHY entry should be None.
      sectors / tiers: per-ticker metadata. For v2-B5, the caller adds
                       SHY → "treasury_etf" (or similar) to sectors and
                       a tier label so the engine handles it cleanly.
      pc_params: legacy v1 path; portfolio construction parameters. Used
                 only when `construction_variant` is None.
      universe_records: full universe.json for active-on-date filtering.
                        Required.
      starting_capital: portfolio start value in USD
      fee_pct: per-trade-leg transaction cost
      construction_variant: v2 path; a ConstructionVariant instance whose
                            `construct(state)` produces the target weights
                            at each rebalance. Takes precedence over
                            `pc_params` when both are passed.
      spy_history: SPY OHLCV up to the snapshot's last date. Only required
                   for variants that read it (B5 defensive sleeves).
      shy_prices: SHY close prices indexed by date. Only required for B5.

    Returns:
      BacktestResult with the four long-format DataFrames.
    """
    if not rebalance_dates:
        raise ValueError("rebalance_dates is empty")
    if construction_variant is None and pc_params is None:
        raise ValueError("must pass construction_variant or pc_params")
    if universe_records is None:
        raise ValueError("universe_records is required")

    first_rb = pd.Timestamp(rebalance_dates[0])
    last_date = pd.Timestamp(daily_returns.index[-1])

    # Initialize state
    nav = 1.0
    cash_frac = 0.0  # cash held as a fraction of portfolio (0 -> 1)
    weights = pd.Series(dtype=float)  # current target weights by ticker

    # v2 state threaded into ConstructionVariant
    portfolio_daily_returns: list[tuple[pd.Timestamp, float]] = []
    top30_streak: dict[str, int] = {}
    prev_portfolio_sector_weights: dict[str, float] = {}

    portfolio_rows = []
    holdings_rows = []
    trades_rows = []
    scores_rows = []

    # Walk daily from first rebalance to last available daily-return date
    all_dates = daily_returns.index[(daily_returns.index >= first_rb)
                                     & (daily_returns.index <= last_date)]
    if len(all_dates) == 0:
        raise ValueError("no daily-return dates in the requested window")

    rebalance_idx = 0
    n_rebalances = len(rebalance_dates)

    for d in all_dates:
        d_ts = pd.Timestamp(d)

        # Apply between-rebalance drift: weights drift with returns; cash stays flat
        if not weights.empty and d_ts != first_rb:
            ret_d = daily_returns.loc[d_ts]
            # Per-position daily return; missing tickers (no data on d) get 0 return
            position_returns = ret_d.reindex(weights.index).fillna(0.0)
            pos_value_change = (weights * (1 + position_returns)).sum() - weights.sum()
            # Renormalize the weights to reflect drift (still represent fractions of equity)
            old_invested = weights.sum()
            new_position_values = weights * (1 + position_returns)
            new_invested = new_position_values.sum()
            day_return = pos_value_change  # cash earns 0
            nav = nav * (1 + day_return)
            # Track running daily portfolio returns for variants that need
            # them (e.g., B1 vol-targeting).
            portfolio_daily_returns.append((d_ts, float(day_return)))
            # Update weights to drifted form (as fraction of new NAV)
            if new_invested > 0:
                weights = new_position_values * (old_invested / new_invested)

            # Forced-exit any tickers that delisted today
            delisted_today = []
            for ticker in list(weights.index):
                dl = delisting_dates.get(ticker)
                if dl is not None and pd.Timestamp(dl) <= d_ts:
                    delisted_today.append(ticker)
            for ticker in delisted_today:
                w_exit = weights[ticker]
                # Sell at today's close (already reflected in nav via the
                # drift step above). Apply exit fee.
                fee_frac = w_exit * fee_pct
                nav = nav * (1 - fee_frac)
                trades_rows.append({
                    "date": d_ts,
                    "model": model_name,
                    "ticker": ticker,
                    "action": "sell",
                    "weight_change": -float(w_exit),
                    "price": float("nan"),
                    "notional_usd": float(w_exit * nav * starting_capital),
                    "fee_usd": float(fee_frac * nav * starting_capital),
                    "reason": "delisting_truncation",
                })
                cash_frac += float(w_exit)
                weights = weights.drop(ticker)

        # Rebalance if today is a rebalance date
        if rebalance_idx < n_rebalances and d_ts == pd.Timestamp(rebalance_dates[rebalance_idx]):
            eligible_today = eligible_universe_on(d_ts, universe_records)

            # Get model scores for eligible tickers on this date
            scores_today = score_fn(d_ts)  # Series(ticker -> score)
            scores_today = scores_today[scores_today.index.isin(eligible_today)]

            # Record all scores (for scores.parquet)
            if not scores_today.empty:
                # Compute rank within the cross-section (1 = highest score)
                rank_series = scores_today.rank(ascending=False, method="min").astype(int)
                for ticker, score in scores_today.items():
                    scores_rows.append({
                        "date": d_ts,
                        "model": model_name,
                        "ticker": ticker,
                        "score": float(score),
                        "rank": int(rank_series[ticker]),
                    })

            # Compute target weights — variant path or legacy path
            if construction_variant is not None:
                # v2: build ConstructionState and call variant.construct()
                # portfolio_history: a pd.Series of daily portfolio returns
                ph = (pd.Series(dict(portfolio_daily_returns))
                      if portfolio_daily_returns else None)
                # spy_history: slice up to and including today
                sh = (spy_history.loc[:d_ts] if spy_history is not None else None)
                # shy_price on this date (for variants that ask for it
                # explicitly; B5 uses spy_history for regime detection
                # and routes SHY as a tradeable in daily_returns)
                sp = (float(shy_prices.loc[d_ts])
                      if shy_prices is not None and d_ts in shy_prices.index
                      else None)
                state = ConstructionState(
                    date=d_ts,
                    scores=scores_today,
                    sectors=sectors,
                    current_weights=weights,
                    portfolio_history=ph,
                    spy_history=sh,
                    shy_price=sp,
                    top30_streak=dict(top30_streak),
                    prev_portfolio_sector_weights=dict(prev_portfolio_sector_weights),
                )
                new_weights = construction_variant.construct(state)
            else:
                # v1 legacy path
                new_weights = rank_top_n_weights(scores_today, sectors, pc_params)

            # Compute trades from old -> new
            all_tickers = set(weights.index).union(new_weights.index)
            for ticker in sorted(all_tickers):
                old_w = float(weights.get(ticker, 0.0))
                new_w = float(new_weights.get(ticker, 0.0))
                delta = new_w - old_w
                if abs(delta) < 1e-9:
                    continue
                action = "buy" if delta > 0 else "sell"
                fee_frac = abs(delta) * fee_pct
                nav = nav * (1 - fee_frac)
                trades_rows.append({
                    "date": d_ts,
                    "model": model_name,
                    "ticker": ticker,
                    "action": action,
                    "weight_change": delta,
                    "price": float("nan"),
                    "notional_usd": float(abs(delta) * nav * starting_capital),
                    "fee_usd": float(fee_frac * nav * starting_capital),
                    "reason": "rebalance",
                })

            # Update state
            weights = new_weights
            cash_frac = max(0.0, 1.0 - weights.sum())

            # v2 state updates for variants that need them
            if construction_variant is not None:
                # top30_streak: increment for tickers still in top-30,
                # reset for tickers that exited. Use the top-30 of the
                # new selection (independent of any cap-adjusted weights).
                # Concretely we use weights' index, which equals the
                # variant's selected names (cap enforcement may have
                # changed weights but not the selected set).
                new_held = set(weights.index)
                # Drop SHY from streak tracking — it's a synthetic
                # defensive-sleeve ticker, not a model-scored stock.
                new_held.discard("SHY")
                next_streak: dict[str, int] = {}
                for t in new_held:
                    next_streak[t] = top30_streak.get(t, 0) + 1
                top30_streak = next_streak
                # prev_portfolio_sector_weights: compute by-sector totals
                # of the just-rebalanced portfolio (becomes "prev" at the
                # next rebalance).
                sector_totals: dict[str, float] = {}
                for t, w_val in weights.items():
                    sec = str(sectors.get(t, "sector_unknown"))
                    sector_totals[sec] = sector_totals.get(sec, 0.0) + float(w_val)
                prev_portfolio_sector_weights = sector_totals

            # Record holdings for this rebalance date
            for ticker, w in weights.items():
                holdings_rows.append({
                    "date": d_ts,
                    "model": model_name,
                    "ticker": ticker,
                    "weight": float(w),
                    "value_usd": float(w * nav * starting_capital),
                    "sector": str(sectors.get(ticker, "sector_unknown")),
                    "tier": str(tiers.get(ticker, "unknown")),
                })
            rebalance_idx += 1

        # Daily portfolio row
        portfolio_rows.append({
            "date": d_ts,
            "model": model_name,
            "nav": float(nav),
            "cash_pct": float(cash_frac),
            "n_positions": int(len(weights)),
            "gross_exposure": float(weights.sum()) if not weights.empty else 0.0,
        })

    portfolio = pd.DataFrame(portfolio_rows)
    holdings = pd.DataFrame(holdings_rows) if holdings_rows else pd.DataFrame(
        columns=["date", "model", "ticker", "weight", "value_usd", "sector", "tier"]
    )
    trades = pd.DataFrame(trades_rows) if trades_rows else pd.DataFrame(
        columns=["date", "model", "ticker", "action", "weight_change", "price",
                 "notional_usd", "fee_usd", "reason"]
    )
    scores = pd.DataFrame(scores_rows) if scores_rows else pd.DataFrame(
        columns=["date", "model", "ticker", "score", "rank"]
    )
    return BacktestResult(portfolio=portfolio, holdings=holdings,
                          trades=trades, scores=scores)


def ew_sp1500_backtest(
    rebalance_dates: list[pd.Timestamp],
    daily_returns: pd.DataFrame,
    delisting_dates: dict[str, pd.Timestamp],
    universe_records: list[dict],
    fee_pct: float = TRANSACTION_COST_PCT,
) -> pd.DataFrame:
    """Build the EW-SP1500 benchmark: monthly equal-weight rebalance over
    all SP1500 active-on-date members per the universe map.

    Returns a long-format DataFrame [date, benchmark="EW-SP1500", nav]
    suitable for inclusion in benchmarks.parquet.
    """
    if not rebalance_dates:
        return pd.DataFrame(columns=["date", "benchmark", "nav"])
    first_rb = pd.Timestamp(rebalance_dates[0])
    last_date = pd.Timestamp(daily_returns.index[-1])

    nav = 1.0
    weights = pd.Series(dtype=float)
    rows = []

    all_dates = daily_returns.index[(daily_returns.index >= first_rb)
                                     & (daily_returns.index <= last_date)]
    rebalance_idx = 0
    n_rebalances = len(rebalance_dates)

    for d in all_dates:
        d_ts = pd.Timestamp(d)

        if not weights.empty and d_ts != first_rb:
            ret_d = daily_returns.loc[d_ts]
            position_returns = ret_d.reindex(weights.index).fillna(0.0)
            new_position_values = weights * (1 + position_returns)
            old_invested = weights.sum()
            new_invested = new_position_values.sum()
            day_return = new_invested - old_invested
            nav = nav * (1 + day_return)
            if new_invested > 0:
                weights = new_position_values * (old_invested / new_invested)

            # Forced-exit delisted
            delisted_today = [t for t in weights.index
                              if delisting_dates.get(t) is not None
                              and pd.Timestamp(delisting_dates[t]) <= d_ts]
            for t in delisted_today:
                w_exit = weights[t]
                fee_frac = w_exit * fee_pct
                nav = nav * (1 - fee_frac)
                weights = weights.drop(t)

        if rebalance_idx < n_rebalances and d_ts == pd.Timestamp(rebalance_dates[rebalance_idx]):
            eligible_today = eligible_universe_on(d_ts, universe_records)
            # Need tickers with daily-return data on this date too
            eligible_today = {t for t in eligible_today
                              if t in daily_returns.columns
                              and not pd.isna(daily_returns.loc[d_ts, t])}
            if not eligible_today:
                # Skip rebalance; carry forward
                weights = weights
            else:
                new_w = 1.0 / len(eligible_today)
                new_weights = pd.Series(new_w, index=list(eligible_today))
                # Trades + fees on the diff
                old_w_series = weights.reindex(new_weights.index, fill_value=0.0)
                gross_delta = (new_weights - old_w_series).abs().sum()
                # Also subtract positions being fully closed
                closed = weights[~weights.index.isin(new_weights.index)]
                gross_delta += closed.abs().sum()
                fee_frac = gross_delta * fee_pct
                nav = nav * (1 - fee_frac)
                weights = new_weights
            rebalance_idx += 1

        rows.append({"date": d_ts, "benchmark": "EW-SP1500", "nav": float(nav)})

    return pd.DataFrame(rows)
