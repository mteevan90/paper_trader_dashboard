"""
portfolio.py — Portfolio state management utilities.

Handles reading, writing, and summarising the portfolio state JSON.
Can also be run directly for a quick status check:

    python portfolio.py                  # show current holdings
    python portfolio.py --history        # show full trade log
    python portfolio.py --performance    # show P&L breakdown per ticker
"""

import argparse
import json
import os
from datetime import datetime

import yfinance as yf

from backtest import INITIAL_CASH, TX_COST

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(BASE_DIR, "..", "models", "portfolio_state.json")


# ---------------------------------------------------------------------------
# Core read/write
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "cash": INITIAL_CASH,
        "positions": {},
        "trade_log": [],
        "last_rebal_date": None,
        "total_fees": 0.0,
        "starting_value": INITIAL_CASH,
    }


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def fetch_prices(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
    out = {}
    for tkr in tickers:
        try:
            if len(tickers) == 1:
                out[tkr] = float(data["Close"].dropna().iloc[-1])
            else:
                out[tkr] = float(data["Close"][tkr].dropna().iloc[-1])
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_holdings(state: dict):
    """Print current open positions."""
    positions = state.get("positions", {})
    if not positions:
        print("\n  No open positions.")
        return

    tickers = list(positions.keys())
    prices  = fetch_prices(tickers)

    print(f"\n{'='*70}")
    print("  CURRENT HOLDINGS")
    print(f"{'='*70}")
    print(f"  {'Ticker':<8} {'Shares':>6} {'Entry':>9} {'Now':>9} "
          f"{'P/L $':>10} {'P/L %':>7} {'Stop':>9} {'Date':>12}")
    print("  " + "-" * 68)

    total_cost = total_val = 0.0
    for tkr in sorted(positions):
        pos   = positions[tkr]
        price = prices.get(tkr, pos["entry_price"])
        cost  = pos["shares"] * pos["entry_price"]
        val   = pos["shares"] * price
        pl_d  = val - cost
        pl_p  = (price / pos["entry_price"] - 1) * 100
        stop  = pos.get("stop_price", 0)
        edate = pos.get("entry_date", "—")
        warn  = " ⚠" if price <= stop * 1.03 else ""
        total_cost += cost
        total_val  += val
        print(f"  {tkr:<8} {pos['shares']:>6.0f} ${pos['entry_price']:>8.2f} "
              f"${price:>8.2f} ${pl_d:>+9.2f} {pl_p:>+6.1f}%  ${stop:>8.2f}  "
              f"{edate}{warn}")

    total_pl   = total_val - total_cost
    total_pl_p = (total_val / total_cost - 1) * 100 if total_cost else 0.0
    print("  " + "-" * 68)
    print(f"  {'TOTAL':<8} {'':>6} {'':>9} {'':>9} "
          f"${total_pl:>+9.2f} {total_pl_p:>+6.1f}%")

    start   = state.get("starting_value", INITIAL_CASH)
    port_v  = state["cash"] + total_val
    overall = (port_v / start - 1) * 100

    print(f"\n  Cash:              ${state['cash']:>12,.2f}")
    print(f"  Invested:          ${total_val:>12,.2f}")
    print(f"  Total portfolio:   ${port_v:>12,.2f}")
    print(f"  Overall return:    {overall:>+11.1f}%")
    print(f"  Fees paid:         ${state.get('total_fees', 0):>12,.2f}")
    print(f"  Last rebalance:    {state.get('last_rebal_date', 'never'):>12}")
    print(f"{'='*70}")


def show_trade_history(state: dict):
    """Print full trade log."""
    log = state.get("trade_log", [])
    if not log:
        print("\n  No trades recorded.")
        return

    print(f"\n{'='*70}")
    print(f"  TRADE LOG  ({len(log)} trades)")
    print(f"{'='*70}")
    print(f"  {'Date':<12} {'Ticker':<8} {'Action':<10} {'Shares':>6} "
          f"{'Price':>9} {'Fee':>7} {'Return':>8}")
    print("  " + "-" * 68)

    for t in log:
        ret_str = f"{t.get('return_pct', 0):>+7.1f}%" if "return_pct" in t else "      —"
        print(f"  {str(t['date']):<12} {t['ticker']:<8} {t['action']:<10} "
              f"{t['shares']:>6.0f} ${t['price']:>8.2f} ${t.get('fee', 0):>6.2f}  "
              f"{ret_str}")

    print(f"{'='*70}")


def show_performance(state: dict):
    """Show P/L breakdown by ticker across closed trades."""
    log = state.get("trade_log", [])
    if not log:
        print("\n  No trades to analyse.")
        return

    # Match buy-sell round trips per ticker
    from collections import defaultdict
    buys_by_ticker  = defaultdict(list)
    sells_by_ticker = defaultdict(list)

    for t in log:
        if t["action"] == "BUY":
            buys_by_ticker[t["ticker"]].append(t)
        elif t["action"] in ("SELL", "STOP_ATR", "STOP", "STOP10"):
            sells_by_ticker[t["ticker"]].append(t)

    all_tickers = sorted(set(buys_by_ticker) | set(sells_by_ticker))

    print(f"\n{'='*65}")
    print("  PERFORMANCE BY TICKER  (closed round trips)")
    print(f"{'='*65}")
    print(f"  {'Ticker':<8} {'Trades':>6} {'Wins':>5} {'Win%':>6} "
          f"{'Avg Ret':>8} {'Total P/L':>11}")
    print("  " + "-" * 55)

    grand_wins = grand_total = 0
    grand_pl   = 0.0

    for tkr in all_tickers:
        buys  = sorted(buys_by_ticker[tkr],  key=lambda x: x["date"])
        sells = sorted(sells_by_ticker[tkr], key=lambda x: x["date"])
        pairs = min(len(buys), len(sells))
        if pairs == 0:
            continue

        wins     = 0
        rets     = []
        total_pl = 0.0
        for i in range(pairs):
            b, s    = buys[i], sells[i]
            ret     = (s["price"] / b["price"] - 1) * 100
            pl_d    = (s["price"] - b["price"]) * b["shares"]
            rets.append(ret)
            total_pl += pl_d
            if ret > 0:
                wins += 1

        avg_ret     = sum(rets) / len(rets)
        win_pct     = wins / pairs * 100
        grand_wins += wins
        grand_total += pairs
        grand_pl    += total_pl

        print(f"  {tkr:<8} {pairs:>6} {wins:>5} {win_pct:>5.0f}%  "
              f"{avg_ret:>+7.1f}%  ${total_pl:>+10,.2f}")

    if grand_total:
        overall_win = grand_wins / grand_total * 100
        print("  " + "-" * 55)
        print(f"  {'TOTAL':<8} {grand_total:>6} {grand_wins:>5} {overall_win:>5.0f}%  "
              f"{'':>8}  ${grand_pl:>+10,.2f}")

    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Manual position management
# ---------------------------------------------------------------------------

def add_position(tkr: str, shares: float, price: float, stop: float | None = None):
    """Manually add a position (e.g. to reconcile with a real brokerage)."""
    state = load_state()
    stop_price = stop if stop else price * (1 - 0.15)
    state["positions"][tkr.upper()] = {
        "shares":      shares,
        "entry_price": price,
        "stop_price":  stop_price,
        "entry_date":  datetime.today().strftime("%Y-%m-%d"),
    }
    cost = shares * price
    fee  = cost * TX_COST
    state["cash"]       -= cost + fee
    state["total_fees"] += fee
    state["trade_log"].append({
        "date": datetime.today().strftime("%Y-%m-%d"),
        "ticker": tkr.upper(), "action": "BUY",
        "shares": shares, "price": price, "fee": fee,
    })
    save_state(state)
    print(f"  Added {shares:.0f} shares of {tkr.upper()} @ ${price:.2f}")


def remove_position(tkr: str, price: float | None = None):
    """Manually close a position."""
    state = load_state()
    tkr   = tkr.upper()
    if tkr not in state["positions"]:
        print(f"  {tkr} not in portfolio.")
        return
    pos = state["positions"][tkr]
    if price is None:
        p = fetch_prices([tkr])
        price = p.get(tkr, pos["entry_price"])
    proceeds = pos["shares"] * price
    fee      = proceeds * TX_COST
    ret_pct  = (price / pos["entry_price"] - 1) * 100
    state["cash"]       += proceeds - fee
    state["total_fees"] += fee
    state["trade_log"].append({
        "date": datetime.today().strftime("%Y-%m-%d"),
        "ticker": tkr, "action": "SELL",
        "shares": pos["shares"], "price": price, "fee": fee,
        "return_pct": ret_pct,
    })
    del state["positions"][tkr]
    save_state(state)
    print(f"  Closed {tkr}: {pos['shares']:.0f} shares @ ${price:.2f}  ({ret_pct:+.1f}%)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio state viewer")
    parser.add_argument("--history",     action="store_true", help="Show trade log")
    parser.add_argument("--performance", action="store_true", help="Show P/L by ticker")
    args = parser.parse_args()

    state = load_state()

    if args.history:
        show_trade_history(state)
    elif args.performance:
        show_performance(state)
    else:
        show_holdings(state)
