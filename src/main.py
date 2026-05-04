"""
main.py — Daily paper trading orchestrator.

Run each morning before market open:
    python main.py

Optional flags:
    python main.py --dry-run        # show signals without updating state
    python main.py --reset          # wipe portfolio state and start fresh
    python main.py --status         # just show current holdings, no signals
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

# Force UTF-8 stdout so the bearish warning glyph renders on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import pandas as pd
import yfinance as yf

from backtest import (
    ATR_STOP_MULT, INITIAL_CASH, MAX_STOP_PCT, MIN_ATR_STOP_PCT, TOP_N, TX_COST,
    compute_composite_scores, fetch_fundamentals,
)
from features import add_features, add_market_features
from fetch_data import UNIVERSE_TICKERS, build_sector_map, get_stock_data_cached
from macro_signals import (fetch_macro_signals, compute_macro_score,
                           fetch_analyst_targets)
from model import FEATURE_COLS, load_model, should_retrain

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
STATE_PATH = os.path.join(MODELS_DIR, "portfolio_state.json")

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load portfolio state from disk. Returns fresh state if file missing."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {
        "cash": INITIAL_CASH,
        "positions": {},      # ticker -> {shares, entry_price, stop_price, entry_date}
        "trade_log": [],      # list of trade dicts
        "last_rebal_date": None,
        "total_fees": 0.0,
    }


def save_state(state: dict):
    """Persist portfolio state to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"  State saved to {STATE_PATH}")


def reset_state():
    """Wipe state and start fresh."""
    fresh = {
        "cash": INITIAL_CASH,
        "positions": {},
        "trade_log": [],
        "last_rebal_date": None,
        "total_fees": 0.0,
    }
    save_state(fresh)
    print(f"  Portfolio reset. Starting cash: ${INITIAL_CASH:,.2f}")
    return fresh

# ---------------------------------------------------------------------------
# Portfolio valuation
# ---------------------------------------------------------------------------

def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest close prices for a list of tickers."""
    if not tickers:
        return {}
    data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
    prices = {}
    for tkr in tickers:
        try:
            if len(tickers) == 1:
                prices[tkr] = float(data["Close"].iloc[-1])
            else:
                prices[tkr] = float(data["Close"][tkr].dropna().iloc[-1])
        except Exception:
            pass
    return prices


def portfolio_value(state: dict, prices: dict[str, float]) -> float:
    """Compute total portfolio value (cash + positions)."""
    val = state["cash"]
    for tkr, pos in state["positions"].items():
        price = prices.get(tkr, pos["entry_price"])
        val += pos["shares"] * price
    return val


def print_status(state: dict, prices: dict[str, float]):
    """Print current holdings table."""
    print("\n" + "=" * 65)
    print("  CURRENT HOLDINGS")
    print("=" * 65)

    if not state["positions"]:
        print("  No open positions.")
    else:
        print(f"  {'Ticker':<8} {'Shares':>6} {'Entry':>8} {'Current':>8} "
              f"{'P/L $':>9} {'P/L %':>7} {'Stop':>8}")
        print("  " + "-" * 63)
        total_cost = total_val = 0.0
        for tkr, pos in sorted(state["positions"].items()):
            price   = prices.get(tkr, pos["entry_price"])
            cost    = pos["shares"] * pos["entry_price"]
            value   = pos["shares"] * price
            pl_d    = value - cost
            pl_p    = (price / pos["entry_price"] - 1) * 100
            stop    = pos.get("stop_price", 0)
            warn    = " ⚠" if price <= stop * 1.03 else ""
            total_cost += cost
            total_val  += value
            print(f"  {tkr:<8} {pos['shares']:>6.0f} ${pos['entry_price']:>7.2f} "
                  f"${price:>7.2f} ${pl_d:>+8.2f} {pl_p:>+6.1f}%  ${stop:>7.2f}{warn}")
        total_pl   = total_val - total_cost
        total_pl_p = (total_val / total_cost - 1) * 100 if total_cost else 0
        print("  " + "-" * 63)
        print(f"  {'TOTAL':<8} {'':>6} {'':>8} {'':>8} "
              f"${total_pl:>+8.2f} {total_pl_p:>+6.1f}%")

    print(f"\n  Cash:            ${state['cash']:>12,.2f}")
    total = portfolio_value(state, prices)
    print(f"  Portfolio value: ${total:>12,.2f}")
    start = state.get("starting_value", INITIAL_CASH)
    ret   = (total / start - 1) * 100
    print(f"  Total return:    {ret:>+11.1f}%")
    print(f"  Total fees paid: ${state['total_fees']:>12,.2f}")
    print("=" * 65)

# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def check_stop_losses(state: dict, prices: dict[str, float], today: str,
                      dry_run: bool) -> list[str]:
    """Check ATR stop losses. Returns list of tickers stopped out."""
    stopped = []
    for tkr, pos in list(state["positions"].items()):
        price = prices.get(tkr)
        if price is None:
            continue
        stop = pos.get("stop_price", 0)
        if stop and price <= stop:
            proceeds = pos["shares"] * price
            fee      = proceeds * TX_COST
            pl_pct   = (price / pos["entry_price"] - 1) * 100
            print(f"  🔴 STOP-LOSS  {tkr:<6}  ${price:.2f}  "
                  f"(entry ${pos['entry_price']:.2f}, {pl_pct:+.1f}%)")
            if not dry_run:
                state["cash"]        += proceeds - fee
                state["total_fees"]  += fee
                state["trade_log"].append({
                    "date": today, "ticker": tkr, "action": "STOP_ATR",
                    "shares": pos["shares"], "price": price, "fee": fee,
                    "return_pct": pl_pct,
                })
                del state["positions"][tkr]
            stopped.append(tkr)
    return stopped


def should_rebalance(state: dict, today: str, force: bool = False) -> bool:
    """Return True if it's time to rebalance (~monthly, every 21 trading days)."""
    if force or state["last_rebal_date"] is None:
        return True
    last = pd.Timestamp(state["last_rebal_date"])
    now  = pd.Timestamp(today)
    # Approximate: rebalance if >= 21 calendar days since last rebal
    return (now - last).days >= 28


def compute_signals(
    top_tickers: list[str],
    state: dict,
    prices: dict[str, float],
    featured_data: dict,
    comp_scores: dict,
    today: str,
    dry_run: bool,
    analyst_data: dict | None = None,
):
    """Generate BUY / SELL signals and execute them (unless dry_run)."""
    analyst_data = analyst_data or {}
    current_holdings = set(state["positions"].keys())
    target_set       = set(top_tickers)

    sells = current_holdings - target_set
    buys  = target_set - current_holdings
    holds = current_holdings & target_set

    print(f"\n{'='*65}")
    print(f"  TRADE SIGNALS  —  {today}")
    print(f"  Top {TOP_N} tickers by composite score")
    print(f"{'='*65}")

    # --- SELLS ---
    if sells:
        print(f"\n  SELLS ({len(sells)}):")
        for tkr in sorted(sells):
            price  = prices.get(tkr, 0)
            pos    = state["positions"].get(tkr, {})
            entry  = pos.get("entry_price", price)
            pl_pct = (price / entry - 1) * 100 if entry else 0
            print(f"    🔴 SELL  {tkr:<6}  ${price:.2f}  ({pl_pct:+.1f}% since entry)")
            if not dry_run and tkr in state["positions"]:
                shares   = pos["shares"]
                proceeds = shares * price
                fee      = proceeds * TX_COST
                state["cash"]       += proceeds - fee
                state["total_fees"] += fee
                state["trade_log"].append({
                    "date": today, "ticker": tkr, "action": "SELL",
                    "shares": shares, "price": price, "fee": fee,
                    "return_pct": pl_pct,
                })
                del state["positions"][tkr]
    else:
        print("\n  SELLS: none")

    # --- BUYS ---
    if buys:
        print(f"\n  BUYS ({len(buys)}):")
        n_buys     = len(buys)
        budget_each = state["cash"] / max(n_buys, 1)
        for tkr in sorted(buys):
            price = prices.get(tkr)
            if price is None or price == 0:
                print(f"    ⚠  SKIP  {tkr:<6}  (no price data)")
                continue
            shares = int(budget_each // price)
            if shares < 1:
                print(f"    ⚠  SKIP  {tkr:<6}  (insufficient cash for 1 share @ ${price:.2f})")
                continue
            cost  = shares * price
            fee   = cost * TX_COST
            score = comp_scores.get(tkr, {}).get("composite", 0)
            a = analyst_data.get(tkr)
            if a is not None:
                analyst_tag = (f"  [analyst: {a['upside_pct']*100:+.1f}% upside, "
                               f"rec {a['recommendation_mean']:.2f}, "
                               f"n={a['n_analysts']}]")
            else:
                analyst_tag = "  [analyst: n/a]"
            print(f"    🟢 BUY   {tkr:<6}  {shares} shares @ ${price:.2f}  "
                  f"(cost ${cost:,.0f}, score {score:.3f}){analyst_tag}")
            if not dry_run:
                state["cash"]       -= cost + fee
                state["total_fees"] += fee

                # ATR-based stop loss: 2.5x ATR, floor -15%, ceiling -5%
                min_stop = price * (1 - MIN_ATR_STOP_PCT)
                max_stop = price * (1 - MAX_STOP_PCT)
                stop_price = max_stop
                fdf = featured_data.get(tkr)
                if fdf is not None and "atr_14" in fdf.columns:
                    last_date = fdf.index[fdf.index <= pd.Timestamp(today)]
                    if len(last_date):
                        atr = fdf.loc[last_date[-1], "atr_14"]
                        atr_stop = price - ATR_STOP_MULT * atr
                        stop_price = min(max(atr_stop, max_stop), min_stop)

                state["positions"][tkr] = {
                    "shares":      shares,
                    "entry_price": price,
                    "stop_price":  stop_price,
                    "entry_date":  today,
                }
                state["trade_log"].append({
                    "date": today, "ticker": tkr, "action": "BUY",
                    "shares": shares, "price": price, "fee": fee,
                })
    else:
        print("\n  BUYS: none")

    # --- HOLDS ---
    if holds:
        print(f"\n  HOLDS ({len(holds)}):")
        for tkr in sorted(holds):
            price  = prices.get(tkr, 0)
            pos    = state["positions"].get(tkr, {})
            entry  = pos.get("entry_price", price)
            pl_pct = (price / entry - 1) * 100 if entry else 0
            score  = comp_scores.get(tkr, {}).get("composite", 0)
            print(f"    🔵 HOLD  {tkr:<6}  ${price:.2f}  ({pl_pct:+.1f}%, score {score:.3f})")

    print(f"\n{'='*65}")


def print_top_scores(comp_scores: dict, top_n: int = 15):
    """Print the top N composite scores."""
    print(f"\n  TOP {top_n} BY COMPOSITE SCORE:")
    print(f"  {'Rank':<5} {'Ticker':<8} {'Composite':>9} {'Fund':>6} "
          f"{'Tech':>6} {'Model':>6}")
    print("  " + "-" * 48)
    sorted_tickers = sorted(comp_scores, key=lambda t: comp_scores[t]["composite"],
                            reverse=True)[:top_n]
    for i, tkr in enumerate(sorted_tickers, 1):
        s = comp_scores[tkr]
        print(f"  {i:<5} {tkr:<8} {s['composite']:>9.3f} {s['fundamental']:>6.2f} "
              f"{s['technical']:>6.2f} {s['model']:>6.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper Trader — daily signals")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show signals without updating state")
    parser.add_argument("--reset",    action="store_true",
                        help="Reset portfolio to starting cash")
    parser.add_argument("--status",   action="store_true",
                        help="Show current holdings only")
    parser.add_argument("--force-rebal", action="store_true",
                        help="Force a rebalance even if not due")
    args = parser.parse_args()

    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n{'='*65}")
    print(f"  PAPER TRADER  —  {today}")
    if args.dry_run:
        print("  [DRY RUN — no state changes]")
    print(f"{'='*65}\n")

    # Walk-forward retraining check (warn only; do not retrain automatically)
    if should_retrain():
        print("!" * 65)
        print("  MODEL IS STALE — last trained more than 90 days ago.")
        print("  Run: python dashboard.py 2022-01-01 2024-01-01 [TODAY] "
              "--universe combined")
        print("!" * 65 + "\n")

    # --- Load or reset state ---
    state = reset_state() if args.reset else load_state()
    if "starting_value" not in state:
        state["starting_value"] = state["cash"]

    # --- Fetch current prices for existing holdings ---
    held_tickers = list(state["positions"].keys())
    print("Fetching current prices for holdings...")
    prices_now = get_current_prices(held_tickers) if held_tickers else {}

    # --- Status only mode ---
    if args.status:
        print_status(state, prices_now)
        return

    # --- Data download ---
    # Always request a year+ of history regardless of cache state. The cache
    # serves this from disk; the cost is just a bigger DataFrame slice. A
    # shorter window silently dies in add_features(): momentum_12_1 needs a
    # 252-day lookback (close.shift(252)), so 365 calendar days (~252
    # trading days) is the crash edge — dropna() wipes every row.
    print(f"Downloading price data for {len(UNIVERSE_TICKERS)} tickers...")
    end   = today
    start = (datetime.today() - timedelta(days=500)).strftime("%Y-%m-%d")
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "models", "price_cache")
    price_data = get_stock_data_cached(UNIVERSE_TICKERS, start, end,
                                       cache_dir=cache_dir)
    print(f"Loaded {len(price_data)} tickers\n")

    # Update prices_now with fresh data
    for tkr in UNIVERSE_TICKERS:
        if tkr in price_data and not price_data[tkr].empty:
            prices_now[tkr] = float(price_data[tkr]["Close"].iloc[-1])

    # --- Stop loss check (before rebalance) ---
    print("Checking stop losses...")
    stopped = check_stop_losses(state, prices_now, today, args.dry_run)
    if not stopped:
        print("  No stop losses triggered.")

    # --- Features ---
    print("\nComputing features...")
    featured_data = {}
    for tkr, df in price_data.items():
        try:
            featured_data[tkr] = add_features(df)
        except Exception as e:
            print(f"  [SKIP] {tkr}: {e}")

    print("Adding market features...")
    sector_map    = build_sector_map(list(price_data.keys()))
    featured_data = add_market_features(featured_data, price_data, sector_map)
    print(f"Features ready for {len(featured_data)} tickers\n")

    # --- Model scoring ---
    print("Scoring tickers with XGBoost model...")
    model = load_model()
    today_ts = pd.Timestamp(today)
    model_scores = {}
    for tkr, df in featured_data.items():
        available_dates = df.index[df.index <= today_ts]
        if len(available_dates) == 0:
            continue
        last_date = available_dates[-1]
        row = df.loc[[last_date], FEATURE_COLS]
        model_scores[tkr] = float(model.predict(row)[0])
    print(f"Scored {len(model_scores)} tickers\n")

    # --- Fundamentals ---
    print("Fetching fundamentals (this may take a minute)...")
    fund_data = fetch_fundamentals(list(featured_data.keys()))

    # --- Composite scores ---
    print("\nComputing composite scores...")
    last_date_ts = max(
        df.index[df.index <= today_ts].max()
        for df in featured_data.values()
        if len(df.index[df.index <= today_ts]) > 0
    )
    comp_scores = compute_composite_scores(
        list(model_scores.keys()),
        fund_data, price_data, featured_data,
        model_scores, last_date_ts,
    )

    # --- Analyst tiebreaker: +5% of analyst_score added to composite ---
    analyst_data = fetch_analyst_targets(list(comp_scores.keys()))
    for tkr, s in comp_scores.items():
        a = analyst_data.get(tkr)
        if a is not None:
            s["analyst_score"] = a["analyst_score"]
            s["composite"]     = s["composite"] + 0.05 * a["analyst_score"]

    top_tickers = sorted(comp_scores, key=lambda t: comp_scores[t]["composite"],
                         reverse=True)[:TOP_N]

    # --- Macro sentiment (print before trade signals) ---
    macro_start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    macro_df = fetch_macro_signals(macro_start, today)
    macro_score = compute_macro_score(macro_df, today_ts)
    print(f"\nMacro sentiment score: {macro_score:.2f} "
          f"(0 = bearish, 1 = bullish)")
    if macro_score < 0.4:
        print("⚠ MACRO BEARISH - position sizes reduced to 50%")

    print_top_scores(comp_scores, TOP_N)

    # --- Rebalance check ---
    if should_rebalance(state, today, force=args.force_rebal):
        print(f"\nRebalancing portfolio...")
        compute_signals(
            top_tickers, state, prices_now, featured_data,
            comp_scores, today, args.dry_run,
            analyst_data=analyst_data,
        )
        if not args.dry_run:
            state["last_rebal_date"] = today
    else:
        last = state["last_rebal_date"]
        print(f"\nNo rebalance due (last: {last}). Use --force-rebal to override.")

    # --- Final status ---
    print_status(state, prices_now)

    # --- Save state ---
    if not args.dry_run:
        save_state(state)
    else:
        print("\n  [DRY RUN: state not saved]")


if __name__ == "__main__":
    main()
