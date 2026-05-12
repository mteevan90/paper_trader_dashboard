"""Monthly-rebalance backtest engine for the Larger Universe v1 study.

Separate from the legacy src/backtest.py engine (which is composite-weighted
+ regime-dependent ATR stops). This engine is purpose-built for the
contract v1 architecture: monthly rebalance, score-weighted continuous
sizing via rank_top_n, close-to-close attribution, forced-exit on
delisting with cash drift until next rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

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
    pc_params: PortfolioConstructionParams,
    universe_records: list[dict],
    starting_capital: float = 1_000_000.0,
    fee_pct: float = TRANSACTION_COST_PCT,
) -> BacktestResult:
    """Run the monthly-rebalance backtest for a single model.

    Args:
      model_name: identifier written into the long-format dataframes
      score_fn:   callable returning a Series of (ticker -> predicted score)
                  for the rebalance date passed in
      rebalance_dates: month-end trading dates within the backtest window
      daily_returns: wide DataFrame of daily simple returns, indexed by date
      delisting_dates: ticker -> last-tradable date (Wikipedia removed_at);
                       None for active tickers
      sectors / tiers: per-ticker metadata
      pc_params: portfolio construction parameters (rank_top_n + caps)
      universe_records: full universe.json for active-on-date filtering
      starting_capital: portfolio start value in USD
      fee_pct: per-trade-leg transaction cost

    Returns:
      BacktestResult with the four long-format DataFrames.
    """
    if not rebalance_dates:
        raise ValueError("rebalance_dates is empty")

    first_rb = pd.Timestamp(rebalance_dates[0])
    last_date = pd.Timestamp(daily_returns.index[-1])

    # Initialize state
    nav = 1.0
    cash_frac = 0.0  # cash held as a fraction of portfolio (0 -> 1)
    weights = pd.Series(dtype=float)  # current target weights by ticker

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

            # Compute target weights via rank_top_n
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
