"""objective.py - Locked Optuna objective for the paper trader.

Score formula (Optuna maximizes):

    score = alpha_vs_spy_annualized - 1.5 * max(0, drawdown_magnitude - 0.15)

Where:
  * alpha_vs_spy_annualized = strategy annualized return - SPY annualized
    return over the same backtest window. Decimal (0.05 == +5pp).
  * drawdown_magnitude = abs(max_drawdown). The penalty kicks in only once
    drawdown exceeds 15%; below that threshold the trial pays no penalty.

Sign convention - CRITICAL:
``run_backtest`` reports ``max_drawdown`` as a NEGATIVE number (e.g.,
-0.1162 for an 11.62% drawdown). The penalty formula needs the
*magnitude*, so we ``abs()`` it on entry. Do NOT flip the sign anywhere
upstream - the consumer here is the only place sign-handling lives.

Public API:
  * ``summarize_backtest(portfolio_df, spy_close) -> dict`` - turns
    ``run_backtest``'s 4-tuple-portion into the metric dict the scorer
    consumes. Lives here (not in backtest.py) so this segment stays
    new-file-only.
  * ``compute_objective(summary) -> float`` - the pure scoring function.
    Returns -1e6 on any missing field or invalid input so Optuna treats
    failed trials as worst-case rather than crashing.
  * ``compute_objective_components(summary) -> dict`` - diagnostic
    breakdown for the experiment log. All floats rounded to 6 decimals.
"""

import pandas as pd


# Constants from the locked formula. Kept module-level so both the scorer
# and the components helper share one source of truth.
_DRAWDOWN_THRESHOLD       = 0.15  # penalty kicks in above 15% magnitude
_DRAWDOWN_PENALTY_WEIGHT  = 1.5
_TRADING_DAYS_PER_YEAR    = 252
_FAILURE_SENTINEL         = -1e6
_ROUND_DP                 = 6


def summarize_backtest(portfolio_df: pd.DataFrame,
                       spy_close: pd.Series) -> dict:
    """Build the metric dict consumed by ``compute_objective``.

    Args:
        portfolio_df: ``portfolio_df`` from ``run_backtest``. Must have a
            ``portfolio_value`` column and a DatetimeIndex.
        spy_close: ``pd.Series`` of SPY close prices indexed by date.
            Caller passes ``market["SPY"]["Close"]`` from the existing
            ``get_stock_data_cached`` fetch.

    Annualization uses the *actual* trading-day count from
    ``len(portfolio_df)`` rather than a flat 252 baseline, so partial-year
    windows are exact rather than approximate. Both strategy and SPY use
    the same denominator since they cover the same calendar window.

    Returns: dict with unrounded metric values. ``compute_objective``
    consumes this directly; ``compute_objective_components`` enriches +
    rounds for logging.

    Raises ``ValueError`` if inputs are degenerate. ``compute_objective``
    catches that and returns the failure sentinel.
    """
    if portfolio_df is None or portfolio_df.empty:
        raise ValueError("portfolio_df is empty")
    if "portfolio_value" not in portfolio_df.columns:
        raise ValueError("portfolio_df missing 'portfolio_value' column")

    values = portfolio_df["portfolio_value"]
    n_trading_days = len(values)
    if n_trading_days < 2:
        raise ValueError(f"need >=2 days for returns, got {n_trading_days}")

    start_value = float(values.iloc[0])
    end_value   = float(values.iloc[-1])
    if start_value <= 0:
        raise ValueError(f"non-positive starting portfolio value: {start_value}")

    strategy_total_return      = end_value / start_value - 1
    strategy_annualized_return = (end_value / start_value) ** (
        _TRADING_DAYS_PER_YEAR / n_trading_days) - 1

    # Max drawdown: NEGATIVE (e.g., -0.1162 == 11.62% drawdown).
    # Do not flip the sign here. compute_objective takes abs() at use.
    max_drawdown = float(((values / values.cummax()) - 1).min())

    if spy_close is None or spy_close.empty:
        raise ValueError("spy_close is empty")

    start_date = values.index[0]
    end_date   = values.index[-1]
    spy_window = spy_close.loc[(spy_close.index >= start_date)
                               & (spy_close.index <= end_date)].dropna()
    if len(spy_window) < 2:
        raise ValueError(
            f"SPY series has <2 observations in window "
            f"{start_date.date()} -> {end_date.date()}"
        )

    spy_start = float(spy_window.iloc[0])
    spy_end   = float(spy_window.iloc[-1])
    if spy_start <= 0:
        raise ValueError(f"non-positive starting SPY price: {spy_start}")

    # Use the strategy's n_trading_days denominator so alpha is computed
    # over a single shared time basis. SPY trades on the same exchange
    # calendar so the counts should match within rounding anyway.
    spy_total_return      = spy_end / spy_start - 1
    spy_annualized_return = (spy_end / spy_start) ** (
        _TRADING_DAYS_PER_YEAR / n_trading_days) - 1

    return {
        "strategy_total_return":      strategy_total_return,
        "strategy_annualized_return": strategy_annualized_return,
        "spy_total_return":           spy_total_return,
        "spy_annualized_return":      spy_annualized_return,
        "max_drawdown":               max_drawdown,
        "n_trading_days":             int(n_trading_days),
        "start_date":                 str(start_date.date()),
        "end_date":                   str(end_date.date()),
    }


def compute_objective(summary: dict) -> float:
    """Pure scoring function. Returns the locked objective score.

    Optuna will be configured to maximize this in segment 5. Any missing
    field, wrong type, or arithmetic error returns ``-1e6`` so a failed
    trial registers as worst-case rather than crashing the study.
    """
    try:
        alpha = (summary["strategy_annualized_return"]
                 - summary["spy_annualized_return"])
        # abs() because max_drawdown is signed negative — see module
        # docstring.
        dd_magnitude = abs(summary["max_drawdown"])
        excess  = max(0.0, dd_magnitude - _DRAWDOWN_THRESHOLD)
        penalty = _DRAWDOWN_PENALTY_WEIGHT * excess
        return float(alpha - penalty)
    except (KeyError, TypeError, ValueError):
        return _FAILURE_SENTINEL


def compute_objective_components(summary: dict) -> dict:
    """Return the full diagnostic breakdown for the experiment log.

    Structure mirrors the locked formula so a future reader can trace
    *why* a config scored what it did, not just what the score was.
    Floats are rounded to 6 decimal places — well below any precision we
    care about for the objective and keeps the log readable.

    On missing/invalid fields returns ``{"score": -1e6, "error": "..."}``
    so the log entry still parses.
    """
    try:
        strategy_ann = summary["strategy_annualized_return"]
        spy_ann      = summary["spy_annualized_return"]
        max_dd       = summary["max_drawdown"]

        alpha        = strategy_ann - spy_ann
        dd_magnitude = abs(max_dd)
        excess       = max(0.0, dd_magnitude - _DRAWDOWN_THRESHOLD)
        penalty      = _DRAWDOWN_PENALTY_WEIGHT * excess
        score        = alpha - penalty

        return {
            "strategy_total_return":      round(summary["strategy_total_return"], _ROUND_DP),
            "strategy_annualized_return": round(strategy_ann, _ROUND_DP),
            "spy_total_return":           round(summary["spy_total_return"], _ROUND_DP),
            "spy_annualized_return":      round(spy_ann, _ROUND_DP),
            "alpha_annualized":           round(alpha, _ROUND_DP),
            "max_drawdown":               round(max_dd, _ROUND_DP),
            "drawdown_magnitude":         round(dd_magnitude, _ROUND_DP),
            "drawdown_excess":            round(excess, _ROUND_DP),
            "drawdown_penalty":           round(penalty, _ROUND_DP),
            "score":                      round(score, _ROUND_DP),
            "n_trading_days":             summary["n_trading_days"],
            "start_date":                 summary["start_date"],
            "end_date":                   summary["end_date"],
        }
    except (KeyError, TypeError, ValueError) as e:
        return {"score": _FAILURE_SENTINEL, "error": str(e)}


if __name__ == "__main__":
    # Verification: run a backtest with default config, score it, print
    # the components breakdown. All caches should hit on a warm system so
    # the only real cost is the backtest loop itself.
    import json
    import os

    from backtest import run_backtest, fetch_fundamentals, fetch_earnings_dates
    from backtest_config import BacktestConfig
    from feature_cache import build_feature_matrix
    from fetch_data import (UNIVERSE_TICKERS, build_sector_map,
                            get_stock_data_cached)

    config = BacktestConfig()
    cache_dir = os.path.join(os.path.dirname(__file__),
                             "..", "models", "price_cache")

    print(f"[OBJECTIVE] Loading feature matrix "
          f"({config.train_start} -> {config.validate_end})...")
    featured = build_feature_matrix(
        list(UNIVERSE_TICKERS), config.train_start, config.validate_end,
        price_cache_dir=cache_dir,
    )

    print("[OBJECTIVE] Loading price data...")
    price_data = get_stock_data_cached(
        list(UNIVERSE_TICKERS), config.train_start, config.validate_end,
        cache_dir=cache_dir)

    sector_map = build_sector_map(list(featured.keys()))

    print("[OBJECTIVE] Loading SPY close prices...")
    market = get_stock_data_cached(["SPY"], config.train_start,
                                   config.validate_end, cache_dir=cache_dir)
    spy_close = market["SPY"]["Close"]

    print("[OBJECTIVE] Fetching fundamentals + earnings...")
    fund_data  = fetch_fundamentals(list(featured.keys()))
    earn_dates = fetch_earnings_dates(list(featured.keys()),
                                      config.train_start, config.validate_end)

    print(f"[OBJECTIVE] Running backtest "
          f"(test from {config.validate_start})...")
    portfolio_df, trades_df, latest_scores, final_holdings = run_backtest(
        featured, price_data,
        split_date=config.validate_start,
        fund_data=fund_data, sector_map=sector_map,
        earnings_dates=earn_dates, config=config,
    )

    summary    = summarize_backtest(portfolio_df, spy_close)
    score      = compute_objective(summary)
    components = compute_objective_components(summary)

    print("\n" + "=" * 60)
    print(f"  OBJECTIVE SCORE: {score}")
    print("=" * 60)
    print("Components:")
    print(json.dumps(components, indent=2))
