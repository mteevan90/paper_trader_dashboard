import base64
import io
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from backtest import (INITIAL_CASH, TX_COST, run_backtest,
                      fetch_fundamentals, fetch_earnings_dates,
                      W_FUNDAMENTAL, W_TECHNICAL, W_MODEL, clear_data_cache)
from features import add_features, add_market_features
import fetch_data
from fetch_data import (NASDAQ_100_TICKERS, SP500_TICKERS, SP1500_TICKERS,
                        UNIVERSE_TICKERS, SECTOR_MAP, build_sector_map,
                        get_stock_data_cached, clear_cache)
from model import FEATURE_COLS, load_model


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
REPORT_PATH = os.path.join(OUT_DIR, "report.html")  # always-latest symlink/copy
INDEX_PATH = os.path.join(REPORTS_DIR, "index.html")


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _make_equity_chart(portfolio_df: pd.DataFrame) -> str:
    """Render the portfolio-vs-SPY chart. Returns a base64 PNG for the HTML
    report, and also overwrites ``../models/backtest_vs_spy.png`` so the
    standalone PNG on disk never goes stale relative to the HTML."""
    start = portfolio_df.index[0].strftime("%Y-%m-%d")
    end = portfolio_df.index[-1].strftime("%Y-%m-%d")

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)

    fig, ax = plt.subplots(figsize=(12, 5))

    port_norm = portfolio_df["portfolio_value"] / portfolio_df["portfolio_value"].iloc[0] * INITIAL_CASH
    ax.plot(port_norm.index, port_norm.values,
            label="Portfolio", linewidth=1.5, color="#2563eb")

    spy_norm = spy["Close"] / spy["Close"].iloc[0] * INITIAL_CASH
    ax.plot(spy_norm.index, spy_norm.values,
            label="SPY (Buy & Hold)", linewidth=1.5, color="#f59e0b", alpha=0.8)

    ax.set_title("Portfolio vs SPY Benchmark", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Also persist to the standalone PNG so it doesn't drift stale.
    png_path = os.path.join(OUT_DIR, "backtest_vs_spy.png")
    try:
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
    except Exception as e:
        print(f"  [WARN] Could not save {png_path}: {e}")

    return _fig_to_base64(fig)


def _summarize_period(portfolio_df: pd.DataFrame,
                      trades_df: pd.DataFrame,
                      spy_close: pd.Series) -> dict:
    """Collapse a backtest run into (total_return, sharpe, max_dd, win_rate,
    alpha) — used for the bull-vs-bear comparison table."""
    pv = portfolio_df["portfolio_value"]
    returns = pv.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    total_ret = pv.iloc[-1] / pv.iloc[0] - 1
    max_dd = ((pv / pv.cummax()) - 1).min()

    wins = total_closed = 0
    if not trades_df.empty:
        for ticker in trades_df["ticker"].unique():
            tkr_trades = trades_df[trades_df["ticker"] == ticker].sort_values("date")
            buys  = tkr_trades[tkr_trades["action"] == "BUY"]
            sells = tkr_trades[tkr_trades["action"].isin(
                ["SELL", "STOP", "STOP10", "STOP_ATR"])]
            pairs = min(len(buys), len(sells))
            for i in range(pairs):
                if sells.iloc[i]["price"] > buys.iloc[i]["price"]:
                    wins += 1
                total_closed += 1
    win_rate = (wins / total_closed * 100) if total_closed else 0.0

    # Alpha vs SPY: daily regression over the period's overlap.
    spy_aligned = spy_close.reindex(pv.index).ffill()
    pr = pv.pct_change().dropna()
    sr = spy_aligned.pct_change().dropna()
    common = pr.index.intersection(sr.index)
    pr, sr = pr.loc[common], sr.loc[common]
    if len(pr) > 1 and sr.std() > 0:
        cov = np.cov(pr, sr)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
        alpha_ann = (pr.mean() - beta * sr.mean()) * 252
    else:
        alpha_ann = 0.0

    return {
        "total_return": float(total_ret),
        "sharpe":       float(sharpe),
        "max_dd":       float(max_dd),
        "win_rate":     float(win_rate),
        "alpha":        float(alpha_ann),
    }


def _make_feature_importance_chart() -> str:
    model = load_model()
    importances = model.feature_importances_

    sorted_idx = np.argsort(importances)
    sorted_features = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sorted_features, sorted_importances, color="#2563eb", edgecolor="#1e40af")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _sleeve_metrics(values: pd.Series, trades_df: pd.DataFrame, fees: float = 0.0) -> dict:
    returns = values.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    total_return = values.iloc[-1] / values.iloc[0] - 1
    max_drawdown = ((values / values.cummax()) - 1).min()

    wins = 0
    total_closed = 0
    if not trades_df.empty:
        for ticker in trades_df["ticker"].unique():
            tkr_trades = trades_df[trades_df["ticker"] == ticker].sort_values("date")
            buy_actions = [a for a in trades_df["action"].unique() if "BUY" in a]
            sell_actions = [a for a in trades_df["action"].unique()
                          if a in ("SELL", "STOP", "STOP10", "STOP_ATR")]
            buys = tkr_trades[tkr_trades["action"].isin(buy_actions)]
            sells = tkr_trades[tkr_trades["action"].isin(sell_actions)]
            pairs = min(len(buys), len(sells))
            for i in range(pairs):
                if sells.iloc[i]["price"] > buys.iloc[i]["price"]:
                    wins += 1
                total_closed += 1

    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
    atr_stops = len(trades_df[trades_df["action"] == "STOP_ATR"]) if not trades_df.empty else 0
    return {
        "Starting Value": f'${values.iloc[0]:,.2f}',
        "Ending Value": f'${values.iloc[-1]:,.2f}',
        "Total Return": f'{total_return:+.2%}',
        "Sharpe Ratio": f'{sharpe:.2f}',
        "Max Drawdown": f'{max_drawdown:.2%}',
        "Total Trades": f'{len(trades_df)}',
        "Win Rate": f'{win_rate:.1f}%',
        "ATR Stop-Losses": f'{atr_stops}',
        "Transaction Costs": f'${fees:,.2f}',
    }


def _top_traded_stocks(trades_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    counts = trades_df["ticker"].value_counts().head(n).reset_index()
    counts.columns = ["Ticker", "Total Trades"]

    rows = []
    for _, row in counts.iterrows():
        tkr = row["Ticker"]
        tkr_trades = trades_df[trades_df["ticker"] == tkr]
        buys = len(tkr_trades[tkr_trades["action"].str.contains("BUY")])
        sells = len(tkr_trades[tkr_trades["action"].str.contains("SELL")])
        rows.append({
            "Ticker": tkr,
            "Total Trades": row["Total Trades"],
            "Buys": buys,
            "Sells": sells,
        })
    return pd.DataFrame(rows)


def _metrics_table_html(title: str, metrics: dict) -> str:
    rows = "\n".join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in metrics.items())
    return f"""<div class="card">
      <h2>{title}</h2>
      <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def _active_trade_log_html(trades_df: pd.DataFrame, end_date) -> str:
    """Build HTML for active sleeve trade log with round-trip matching."""
    act = trades_df.copy()
    if act.empty:
        return ""

    # Match buy-sell round trips per ticker chronologically
    round_trips = []
    for ticker in act["ticker"].unique():
        tkr_trades = act[act["ticker"] == ticker].sort_values("date")
        buys = tkr_trades[tkr_trades["action"] == "BUY"].reset_index(drop=True)
        exits = tkr_trades[tkr_trades["action"].isin(["SELL", "STOP", "STOP10", "STOP_ATR"])].reset_index(drop=True)

        for i in range(min(len(buys), len(exits))):
            b = buys.iloc[i]
            s = exits.iloc[i]
            ret = (s["price"] / b["price"] - 1) * 100
            hold_days = (s["date"] - b["date"]).days
            if s["action"] == "STOP_ATR":
                reason = "ATR Stop"
            elif s["action"] in ("STOP10", "STOP"):
                reason = "Hard Stop"
            else:
                reason = "Rebalance"
            round_trips.append({
                "ticker": ticker,
                "buy_date": b["date"],
                "buy_price": b["price"],
                "sell_date": s["date"],
                "sell_price": s["price"],
                "return_pct": ret,
                "hold_days": hold_days,
                "reason": reason,
            })

        # Open positions (buys without matching sell)
        for i in range(len(exits), len(buys)):
            b = buys.iloc[i]
            round_trips.append({
                "ticker": ticker,
                "buy_date": b["date"],
                "buy_price": b["price"],
                "sell_date": end_date,
                "sell_price": 0,
                "return_pct": 0,
                "hold_days": (end_date - b["date"]).days,
                "reason": "End of Period",
            })

    if not round_trips:
        return ""

    rt_df = pd.DataFrame(round_trips).sort_values("buy_date")

    # Summary stats
    closed = rt_df[rt_df["reason"] != "End of Period"]
    if len(closed) > 0:
        avg_hold = closed["hold_days"].mean()
        avg_ret = closed["return_pct"].mean()
        best = closed.loc[closed["return_pct"].idxmax()]
        worst = closed.loc[closed["return_pct"].idxmin()]
        best_str = f'{best["ticker"]} {best["return_pct"]:+.1f}% ({best["buy_date"].strftime("%m/%d")} - {best["sell_date"].strftime("%m/%d")})'
        worst_str = f'{worst["ticker"]} {worst["return_pct"]:+.1f}% ({worst["buy_date"].strftime("%m/%d")} - {worst["sell_date"].strftime("%m/%d")})'
    else:
        avg_hold = avg_ret = 0
        best_str = worst_str = "N/A"

    summary_html = f"""<div class="trade-summary">
      <div><strong>Avg Hold Time:</strong> {avg_hold:.0f} days</div>
      <div><strong>Avg Return/Trade:</strong> {avg_ret:+.2f}%</div>
      <div><strong>Best Trade:</strong> {best_str}</div>
      <div><strong>Worst Trade:</strong> {worst_str}</div>
    </div>"""

    # Table rows
    table_rows = []
    for _, r in rt_df.iterrows():
        color = "#dcfce7" if r["return_pct"] >= 0 else "#fee2e2"
        if r["reason"] == "End of Period":
            sell_str = "Open"
            price_str = "—"
            ret_str = "—"
        else:
            sell_str = r["sell_date"].strftime("%Y-%m-%d")
            price_str = f'${r["sell_price"]:,.2f}'
            ret_str = f'{r["return_pct"]:+.1f}%'

        table_rows.append(
            f'<tr style="background:{color if r["reason"] != "End of Period" else "#f0f9ff"}">'
            f'<td>{r["ticker"]}</td>'
            f'<td>{r["buy_date"].strftime("%Y-%m-%d")}</td>'
            f'<td>${r["buy_price"]:,.2f}</td>'
            f'<td>{sell_str}</td>'
            f'<td>{price_str}</td>'
            f'<td>{ret_str}</td>'
            f'<td>{r["hold_days"]}d</td>'
            f'<td>{r["reason"]}</td></tr>'
        )

    return f"""<div class="card">
      <h2>Trade Log ({len(rt_df)} round trips)</h2>
      {summary_html}
      <div style="max-height:500px;overflow-y:auto">
      <table>
        <thead><tr><th>Ticker</th><th>Buy Date</th><th>Buy Price</th>
        <th>Sell Date</th><th>Sell Price</th><th>Return</th>
        <th>Hold</th><th>Exit Reason</th></tr></thead>
        <tbody>{"".join(table_rows)}</tbody>
      </table>
      </div>
    </div>"""


def _download_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Download a benchmark's daily close prices."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data["Close"]


def _series_stats(values: pd.Series) -> dict:
    """Compute standard stats for a price series."""
    returns = values.pct_change().dropna()
    total_ret = values.iloc[-1] / values.iloc[0] - 1
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    dd = ((values / values.cummax()) - 1).min()
    vol = returns.std() * np.sqrt(252)
    monthly = returns.resample("ME").sum()
    return {
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_dd": dd,
        "volatility": vol,
        "best_day": returns.max(),
        "worst_day": returns.min(),
        "pos_months": int((monthly > 0).sum()),
        "neg_months": int((monthly <= 0).sum()),
        "returns": returns,
    }


def _benchmark_comparison_html(port_values: pd.Series, spy_close: pd.Series,
                                qqq_close: pd.Series) -> str:
    """Build the 3-column benchmark comparison table."""
    ps = _series_stats(port_values)
    ss = _series_stats(spy_close)
    qs = _series_stats(qqq_close)

    def _row(label, key, fmt):
        return (f'<tr><td>{label}</td>'
                f'<td>{fmt.format(ps[key])}</td>'
                f'<td>{fmt.format(ss[key])}</td>'
                f'<td>{fmt.format(qs[key])}</td></tr>')

    rows = [
        _row("Total Return", "total_return", "{:+.2%}"),
        _row("Sharpe Ratio", "sharpe", "{:.2f}"),
        _row("Max Drawdown", "max_dd", "{:.2%}"),
        _row("Volatility (ann.)", "volatility", "{:.2%}"),
        _row("Best Day", "best_day", "{:+.2%}"),
        _row("Worst Day", "worst_day", "{:+.2%}"),
        _row("Positive Months", "pos_months", "{}"),
        _row("Negative Months", "neg_months", "{}"),
    ]

    return f"""<div class="card">
      <h2>Benchmark Comparison</h2>
      <table>
        <thead><tr><th>Metric</th><th>Our Strategy</th><th>SPY (S&amp;P 500)</th><th>QQQ (Nasdaq 100)</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""


def _alpha_beta_html(port_values: pd.Series, spy_close: pd.Series) -> str:
    """Compute and display alpha, beta, and correlation vs SPY."""
    port_ret = port_values.pct_change().dropna()
    spy_ret = spy_close.pct_change().dropna()

    # Align on common dates
    common = port_ret.index.intersection(spy_ret.index)
    pr = port_ret.loc[common]
    sr = spy_ret.loc[common]

    cov = np.cov(pr, sr)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
    alpha_daily = pr.mean() - beta * sr.mean()
    alpha_ann = alpha_daily * 252
    corr = pr.corr(sr)

    return f"""<div class="card">
      <h2>Alpha / Beta Analysis (vs SPY)</h2>
      <table>
        <thead><tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr></thead>
        <tbody>
          <tr><td>Alpha (annualized)</td><td>{alpha_ann:+.2%}</td>
              <td>{"Outperforming" if alpha_ann > 0 else "Underperforming"} SPY on risk-adjusted basis</td></tr>
          <tr><td>Beta</td><td>{beta:.2f}</td>
              <td>{"More" if beta > 1 else "Less"} volatile than SPY ({beta:.2f}x market moves)</td></tr>
          <tr><td>Correlation to SPY</td><td>{corr:.2f}</td>
              <td>{"High" if corr > 0.7 else "Moderate" if corr > 0.4 else "Low"} correlation</td></tr>
        </tbody>
      </table>
    </div>"""


def _monthly_heatmap_html(port_values: pd.Series, spy_close: pd.Series,
                           qqq_close: pd.Series) -> str:
    """Build a monthly returns heatmap table."""
    def _monthly_returns(values):
        daily = values.pct_change().dropna()
        return daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    pm = _monthly_returns(port_values)
    sm = _monthly_returns(spy_close)
    qm = _monthly_returns(qqq_close)

    # Align all to common months
    all_months = sorted(set(pm.index) | set(sm.index) | set(qm.index))

    rows = []
    for month in all_months:
        pv = pm.get(month, 0)
        sv = sm.get(month, 0)
        qv = qm.get(month, 0)

        def _cell(val, ref=None):
            color = "#dcfce7" if val >= 0 else "#fee2e2"
            bold = ""
            if ref is not None and val > ref:
                bold = "font-weight:600;"
            return f'<td style="background:{color};{bold}text-align:right">{val:+.2%}</td>'

        rows.append(
            f'<tr><td>{month.strftime("%b %Y")}</td>'
            f'{_cell(pv)}{_cell(sv)}{_cell(qv)}</tr>'
        )

    return f"""<div class="card">
      <h2>Monthly Returns Heatmap</h2>
      <table>
        <thead><tr><th>Month</th><th>Our Strategy</th><th>SPY</th><th>QQQ</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""


def _current_holdings_html(
    final_holdings: dict[str, dict],
    price_data: dict[str, pd.DataFrame],
    latest_scores: dict[str, dict],
    sector_map: dict[str, str],
    end_date,
) -> str:
    """Build a detailed current holdings table."""
    if not final_holdings:
        return ""

    # Get company names
    names = {}
    for tkr in final_holdings:
        try:
            names[tkr] = yf.Ticker(tkr).info.get("shortName", tkr)
        except Exception:
            names[tkr] = tkr

    rows_data = []
    total_cost = 0.0
    total_value = 0.0

    for tkr, info in final_holdings.items():
        shares = info["shares"]
        entry = info["entry_price"]
        cost_basis = shares * entry
        total_cost += cost_basis

        if tkr in price_data and end_date in price_data[tkr].index:
            cur_price = price_data[tkr].loc[end_date, "Close"]
        else:
            cur_price = entry

        cur_value = shares * cur_price
        total_value += cur_value
        pl_dollar = cur_value - cost_basis
        pl_pct = (cur_price / entry - 1) * 100

        # Purchase date: find last BUY for this ticker from trades
        comp = latest_scores.get(tkr, {}).get("composite", 0)
        sector = sector_map.get(tkr, "—")

        # Stop-loss warning: within 3% of the ATR stop price
        stop_price = info.get("stop_price", entry * 0.85)
        stop_gap = (cur_price - stop_price) / cur_price
        warning = stop_gap < 0.03  # within 3% of triggering

        rows_data.append({
            "ticker": tkr, "name": names.get(tkr, tkr), "sector": sector,
            "entry": entry, "cur_price": cur_price, "shares": shares,
            "cost_basis": cost_basis, "cur_value": cur_value,
            "pl_dollar": pl_dollar, "pl_pct": pl_pct,
            "composite": comp, "warning": warning,
        })

    # Sort by P/L % descending
    rows_data.sort(key=lambda r: r["pl_pct"], reverse=True)

    table_rows = []
    for r in rows_data:
        if r["warning"]:
            row_bg = "#fef9c3"  # yellow warning
            status = "Stop Warning"
        else:
            row_bg = ""
            status = "Active"

        pl_color = "#16a34a" if r["pl_pct"] >= 0 else "#dc2626"

        table_rows.append(
            f'<tr style="background:{row_bg}">'
            f'<td><strong>{r["ticker"]}</strong></td>'
            f'<td>{r["name"]}</td>'
            f'<td>{r["sector"].title()}</td>'
            f'<td>${r["entry"]:,.2f}</td>'
            f'<td>${r["cur_price"]:,.2f}</td>'
            f'<td>{r["shares"]:.0f}</td>'
            f'<td>${r["cost_basis"]:,.2f}</td>'
            f'<td>${r["cur_value"]:,.2f}</td>'
            f'<td style="color:{pl_color};font-weight:600">${r["pl_dollar"]:+,.2f}</td>'
            f'<td style="color:{pl_color};font-weight:600">{r["pl_pct"]:+.1f}%</td>'
            f'<td>{r["composite"]:.3f}</td>'
            f'<td>{status}</td></tr>'
        )

    # Summary row
    total_pl = total_value - total_cost
    total_pl_pct = (total_value / total_cost - 1) * 100 if total_cost > 0 else 0
    pl_color = "#16a34a" if total_pl >= 0 else "#dc2626"
    table_rows.append(
        f'<tr style="background:#f1f5f9;font-weight:600">'
        f'<td colspan="6">TOTAL ({len(rows_data)} positions)</td>'
        f'<td>${total_cost:,.2f}</td>'
        f'<td>${total_value:,.2f}</td>'
        f'<td style="color:{pl_color}">${total_pl:+,.2f}</td>'
        f'<td style="color:{pl_color}">{total_pl_pct:+.1f}%</td>'
        f'<td colspan="2"></td></tr>'
    )

    return f"""<div class="card">
      <h2>Current Holdings</h2>
      <div style="overflow-x:auto">
      <table style="font-size:0.82rem">
        <thead><tr>
          <th>Ticker</th><th>Company</th><th>Sector</th>
          <th>Buy Price</th><th>Current</th><th>Shares</th>
          <th>Cost Basis</th><th>Value</th>
          <th>P/L ($)</th><th>P/L (%)</th>
          <th>Score</th><th>Status</th>
        </tr></thead>
        <tbody>{"".join(table_rows)}</tbody>
      </table>
      </div>
    </div>"""


def _composite_scores_html(latest_scores: dict[str, dict], top_n: int = 15) -> str:
    """Build HTML table showing composite score breakdown for top holdings."""
    if not latest_scores:
        return ""

    sorted_tickers = sorted(latest_scores, key=lambda t: latest_scores[t]["composite"],
                             reverse=True)[:top_n]

    rows = []
    for tkr in sorted_tickers:
        s = latest_scores[tkr]
        # Color bar for composite score
        pct = int(s["composite"] * 100)
        bar = f'<div style="background:#e2e8f0;border-radius:4px;height:8px;width:100%"><div style="background:#2563eb;border-radius:4px;height:8px;width:{pct}%"></div></div>'
        rows.append(
            f'<tr>'
            f'<td><strong>{tkr}</strong></td>'
            f'<td>{s["fundamental"]:.2f}</td>'
            f'<td>{s["technical"]:.2f}</td>'
            f'<td>{s["model"]:.2f}</td>'
            f'<td><strong>{s["composite"]:.3f}</strong></td>'
            f'<td style="min-width:80px">{bar}</td></tr>'
        )

    weights = f"Fundamental {int(W_FUNDAMENTAL*100)}% / Technical {int(W_TECHNICAL*100)}% / Model {int(W_MODEL*100)}%"

    return f"""<div class="card">
      <h2>Composite Score Breakdown (Latest Rebalance)</h2>
      <p style="color:#64748b;font-size:0.85rem;margin-bottom:1rem">Weights: {weights}</p>
      <table>
        <thead><tr><th>Ticker</th><th>Fundamental</th><th>Technical</th>
        <th>Model</th><th>Composite</th><th></th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""


def _save_report(html: str, period: str, total_return: float, sharpe: float,
                  max_dd: float, notes: str) -> str:
    """Save timestamped report, update latest, and rebuild index."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"report_{timestamp}.html"
    report_path = os.path.join(REPORTS_DIR, filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Also save as the always-latest report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    # Append metadata to a simple JSON-lines manifest
    import json
    manifest_path = os.path.join(REPORTS_DIR, "manifest.jsonl")
    entry = {
        "timestamp": timestamp,
        "filename": filename,
        "period": period,
        "total_return": round(total_return * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 2),
        "notes": notes,
    }
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # Rebuild index
    _rebuild_index(manifest_path)

    return report_path


def _rebuild_index(manifest_path: str):
    """Rebuild the index.html from the manifest file."""
    import json
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Reverse chronological
    entries.sort(key=lambda e: e["timestamp"], reverse=True)

    # Find best return for highlighting
    best_return = max(e["total_return"] for e in entries) if entries else 0

    rows = []
    for e in entries:
        is_best = e["total_return"] == best_return
        bg = ' style="background:#dcfce7"' if is_best else ""
        rows.append(
            f'<tr{bg}>'
            f'<td><a href="{e["filename"]}">{e["timestamp"].replace("_", " ")}</a></td>'
            f'<td>{e["period"]}</td>'
            f'<td>{e["total_return"]:+.2f}%</td>'
            f'<td>{e["sharpe"]:.2f}</td>'
            f'<td>{e["max_dd"]:.2f}%</td>'
            f'<td>{e["notes"]}</td></tr>'
        )

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paper Trader - Report History</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f8fafc; color: #1e293b; padding: 2rem; }}
  .container {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #64748b; margin-bottom: 2rem; font-size: 0.95rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem;
           background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
           overflow: hidden; }}
  th, td {{ text-align: left; padding: 0.75rem 1rem; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover {{ background: #f8fafc; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  td:nth-child(3), td:nth-child(4), td:nth-child(5),
  th:nth-child(3), th:nth-child(4), th:nth-child(5) {{ text-align: right; }}
</style>
</head>
<body>
<div class="container">
  <h1>Paper Trader Report History</h1>
  <p class="subtitle">{len(entries)} runs tracked. Best run highlighted in green.</p>
  <table>
    <thead>
      <tr><th>Date Generated</th><th>Backtest Period</th><th>Total Return</th>
      <th>Sharpe Ratio</th><th>Max Drawdown</th><th>Strategy Notes</th></tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>
</body>
</html>"""

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(index_html)


def _bear_comparison_html(main_summary: dict, bear_summary: dict,
                          main_period: str, bear_period: str) -> str:
    """Produce the <div class=card> snippet comparing bull vs bear metrics."""
    def cell(v: float, pct: bool = False, sign: bool = False) -> str:
        if pct and sign:   return f"{v*100:+.2f}%"
        if pct:            return f"{v*100:.2f}%"
        if sign:           return f"{v:+.2f}"
        return f"{v:.2f}"

    rows = [
        ("Total Return",  cell(main_summary["total_return"], pct=True, sign=True),
                          cell(bear_summary["total_return"], pct=True, sign=True)),
        ("Sharpe Ratio",  cell(main_summary["sharpe"]),
                          cell(bear_summary["sharpe"])),
        ("Max Drawdown",  cell(main_summary["max_dd"], pct=True),
                          cell(bear_summary["max_dd"], pct=True)),
        ("Win Rate",      f"{main_summary['win_rate']:.1f}%",
                          f"{bear_summary['win_rate']:.1f}%"),
        ("Alpha (ann.)",  cell(main_summary["alpha"], pct=True, sign=True),
                          cell(bear_summary["alpha"], pct=True, sign=True)),
    ]
    body = "\n".join(
        f"<tr><td>{m}</td><td style='text-align:right'>{b}</td>"
        f"<td style='text-align:right'>{r}</td></tr>"
        for m, b, r in rows
    )
    return f"""<div class="card">
      <h2>Bear Market Comparison (2022-2024)</h2>
      <p class="subtitle" style="margin-bottom:0.75rem">
        Retrained on 2018 to {main_period.split(" to ")[0]} train data, tested on the 2022-2023 bear window
        vs the main bull period &mdash; shows whether recent improvements
        (FRED macro overlay, z-scored features, 12-1 momentum, analyst tiebreaker)
        help in downturns or are bull-market artifacts.
      </p>
      <table>
        <thead><tr><th>Metric</th>
          <th style="text-align:right">Bull ({main_period})</th>
          <th style="text-align:right">Bear ({bear_period})</th>
        </tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>"""


def _standalone_bear_html(bear_section: str) -> str:
    """Wrap the bear-comparison card in a minimal standalone HTML page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Paper Trader - Bear Market Comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f8fafc; color: #1e293b; padding: 2rem; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  .card {{ background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
           padding: 1.5rem; margin-bottom: 1.5rem; }}
  .card h2 {{ font-size: 1.15rem; margin-bottom: 0.5rem; color: #0f172a; }}
  .subtitle {{ color: #64748b; font-size: 0.9rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; margin-top: 0.5rem; }}
  th, td {{ text-align: left; padding: 0.6rem 1rem; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
  tr:last-child td {{ border-bottom: none; }}
</style>
</head>
<body>
<div class="container">
  <p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
  {bear_section}
</div>
</body>
</html>"""


def _build_html(
    equity_b64: str,
    importance_b64: str,
    portfolio_metrics: dict,
    top_stocks: pd.DataFrame,
    period: str,
    cost_data: dict | None = None,
    current_holdings_html_str: str = "",
    trade_log_html: str = "",
    composite_html: str = "",
    benchmark_html: str = "",
    alpha_beta_html_str: str = "",
    heatmap_html: str = "",
    universe_label: str = "Nasdaq 100",
    bear_html: str = "",
) -> str:
    stocks_rows = "\n".join(
        f'<tr><td>{r["Ticker"]}</td><td>{r["Total Trades"]}</td>'
        f'<td>{r["Buys"]}</td><td>{r["Sells"]}</td></tr>'
        for _, r in top_stocks.iterrows()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paper Trader Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f8fafc; color: #1e293b; padding: 2rem; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #64748b; margin-bottom: 2rem; font-size: 0.95rem; }}
  .card {{ background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
           padding: 1.5rem; margin-bottom: 1.5rem; }}
  .card h2 {{ font-size: 1.15rem; margin-bottom: 1rem; color: #0f172a; }}
  .card img {{ width: 100%; height: auto; border-radius: 6px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  .grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 900px) {{ .grid3 {{ grid-template-columns: 1fr; }} }}
  @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.6rem 1rem; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover {{ background: #f8fafc; }}
  td:nth-child(2), th:nth-child(2),
  td:nth-child(3), th:nth-child(3),
  td:nth-child(4), th:nth-child(4) {{ text-align: right; }}
  .trade-summary {{ display: flex; gap: 2rem; flex-wrap: wrap;
                    background: #f1f5f9; border-radius: 8px; padding: 1rem 1.25rem;
                    margin-bottom: 1rem; font-size: 0.9rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>Paper Trader Dashboard</h1>
  <p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} &mdash; {period} &mdash; {universe_label} universe, XGBoost model</p>

  <div class="card">
    <h2>Portfolio vs SPY</h2>
    <img src="data:image/png;base64,{equity_b64}" alt="Equity curve">
  </div>

  {current_holdings_html_str}

  <div class="grid">
    {_metrics_table_html("Portfolio Summary", portfolio_metrics)}
    <div class="card">
      <h2>Feature Importance</h2>
      <img src="data:image/png;base64,{importance_b64}" alt="Feature importance">
    </div>
  </div>

  {benchmark_html}

  <div class="grid">
    {alpha_beta_html_str}
    {_metrics_table_html("Transaction Cost Impact (0.05% per trade)", cost_data) if cost_data else ""}
  </div>

  {heatmap_html}

  {bear_html}

  <div class="card">
    <h2>Top 10 Most Traded Stocks</h2>
    <table>
      <thead><tr><th>Ticker</th><th>Total Trades</th><th>Buys</th><th>Sells</th></tr></thead>
      <tbody>{stocks_rows}</tbody>
    </table>
  </div>

  {composite_html}

  {trade_log_html}
</div>
</body>
</html>"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trader Dashboard")
    parser.add_argument("dates", nargs="*",
                        help="data_start split_date data_end (e.g. 2022-01-01 2025-01-01 2025-09-30)")
    parser.add_argument("--notes", default="", help="Strategy notes for this run")
    parser.add_argument("--universe", default="nasdaq100",
                        choices=["nasdaq100", "sp500", "combined", "sp1500"],
                        help="Stock universe: nasdaq100, sp500, combined "
                             "(NASDAQ 100 + S&P 500, the legacy 490-ticker "
                             "set), or sp1500 (NASDAQ 100 + S&P 500 + S&P "
                             "400 + S&P 600). sp1500 implicitly enables the "
                             "$25M ADV liquidity filter.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Wipe the local parquet price cache before running.")
    parser.add_argument("--bear-compare", action="store_true",
                        help="After the main backtest, also run a 2022-2024 "
                             "bear-window backtest and print a side-by-side "
                             "comparison. Adds several minutes — off by default.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-ticker yfinance failure messages "
                             "instead of the single-line summary.")
    args = parser.parse_args()

    fetch_data.set_verbose(args.verbose)

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "models", "price_cache")
    if args.clear_cache:
        clear_cache(cache_dir)
        clear_data_cache()

    if len(args.dates) == 3:
        start, split_date, end = args.dates
    else:
        end = datetime.today().strftime("%Y-%m-%d")
        start = "2018-01-01"
        split_date = None

    strategy_notes = args.notes
    if args.universe == "combined":
        tickers = UNIVERSE_TICKERS
        universe_label = "Nasdaq 100 + S&P 500"
    elif args.universe == "sp1500":
        tickers = SP1500_TICKERS
        universe_label = "S&P 1500 (Nasdaq 100 + S&P 500/400/600)"
    elif args.universe == "sp500":
        tickers = SP500_TICKERS
        universe_label = "S&P 500"
    else:
        tickers = NASDAQ_100_TICKERS
        universe_label = "Nasdaq 100"

    print(f"Universe: {universe_label} ({len(tickers)} tickers)")
    print(f"Data range: {start} to {end}")
    if split_date:
        print(f"Train: {start} to {split_date}  |  Backtest: {split_date} to {end}")

    print(f"\nDownloading {len(tickers)} {universe_label} tickers...")
    price_data = get_stock_data_cached(tickers, start, end, cache_dir=cache_dir)
    print(f"Loaded {len(price_data)} tickers\n")

    print("Building sector map...")
    sector_map = build_sector_map(list(price_data.keys()))

    print("Computing features...")
    featured_data = {}
    for ticker, df in price_data.items():
        try:
            featured_data[ticker] = add_features(df)
        except Exception as e:
            print(f"  [SKIP] {ticker}: {e}")

    print("Adding market features...")
    featured_data = add_market_features(featured_data, price_data, sector_map)
    print(f"Features ready for {len(featured_data)} tickers\n")

    # Train model if split_date provided
    trained_model = None
    if split_date:
        from model import train_model, walk_forward_validate
        frames = []
        for ticker, df in featured_data.items():
            d = df.copy()
            d["ticker"] = ticker
            frames.append(d)
        combined = pd.concat(frames).sort_index()
        print(f"Training on {len(combined[combined.index < split_date])} rows "
              f"(testing on {len(combined[combined.index >= split_date])} rows)\n")
        print("Training XGBoost classifier...")
        trained_model = train_model(combined, split_date=split_date)
        walk_forward_validate(combined, n_splits=5)
        print()

    print("Fetching fundamentals...")
    fund_data = fetch_fundamentals(list(featured_data.keys()))

    print("Fetching earnings dates...")
    earn_dates = fetch_earnings_dates(list(featured_data.keys()), start, end)

    print("Running backtest...")
    portfolio_df, trades_df, latest_scores, final_holdings = run_backtest(
        featured_data, price_data, split_date=split_date,
        fund_data=fund_data, sector_map=sector_map,
        earnings_dates=earn_dates, model=trained_model)

    print("Generating charts...")
    equity_b64 = _make_equity_chart(portfolio_df)
    importance_b64 = _make_feature_importance_chart()

    print("Computing metrics...")
    period = f'{portfolio_df.index[0].date()} to {portfolio_df.index[-1].date()}'
    total_fees = portfolio_df["total_fees"].iloc[-1]

    portfolio_metrics = _sleeve_metrics(portfolio_df["portfolio_value"], trades_df, total_fees)
    top_stocks = _top_traded_stocks(trades_df)

    gross_return = (portfolio_df["portfolio_value"].iloc[-1] + total_fees) / portfolio_df["portfolio_value"].iloc[0] - 1
    net_return = portfolio_df["portfolio_value"].iloc[-1] / portfolio_df["portfolio_value"].iloc[0] - 1
    cost_data = {
        "Total Fees Paid": f'${total_fees:,.2f}',
        "Gross Return (before fees)": f'{gross_return:+.2%}',
        "Net Return (after fees)": f'{net_return:+.2%}',
        "Fee Drag": f'{gross_return - net_return:+.2%}',
    }

    print("Downloading benchmarks...")
    bt_start = portfolio_df.index[0].strftime("%Y-%m-%d")
    bt_end = (portfolio_df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_close = _download_benchmark("SPY", bt_start, bt_end)
    qqq_close = _download_benchmark("QQQ", bt_start, bt_end)
    port_values = portfolio_df["portfolio_value"]

    print("Computing benchmark analysis...")
    bench_html = _benchmark_comparison_html(port_values, spy_close, qqq_close)
    ab_html = _alpha_beta_html(port_values, spy_close)
    heat_html = _monthly_heatmap_html(port_values, spy_close, qqq_close)

    print("Building holdings sections...")
    end_date = portfolio_df.index[-1]
    cur_hold_html = _current_holdings_html(
        final_holdings, price_data, latest_scores, sector_map, end_date)
    trade_log = _active_trade_log_html(trades_df, end_date)
    comp_html = _composite_scores_html(latest_scores)

    # --- Bear market comparison backtest (opt-in via --bear-compare) ------
    # Runs BEFORE HTML build so the report can embed the comparison section.
    # Adds several minutes to each run, so off by default.
    bear_html = ""
    if not args.bear_compare:
        print("\n[INFO] Bear market comparison skipped. "
              "Use --bear-compare to run it.")
    else:
        bear_split_date = "2022-01-01"
        bear_end_date   = "2024-01-01"
        bear_end_ts     = pd.Timestamp(bear_end_date)

        print("\n" + "=" * 70)
        print("  BEAR MARKET COMPARISON BACKTEST")
        print(f"  Train: {start} to {bear_split_date}  |  "
              f"Test: {bear_split_date} to {bear_end_date}")
        print("=" * 70)

        bear_featured = {t: df.loc[:bear_end_ts] for t, df in featured_data.items()}
        bear_frames = []
        for ticker, df in bear_featured.items():
            d = df.copy()
            d["ticker"] = ticker
            bear_frames.append(d)
        bear_combined = pd.concat(bear_frames).sort_index()

        from model import train_model as _train_model
        print(f"  Training bear-era model on "
              f"{len(bear_combined[bear_combined.index < bear_split_date])} rows...")
        bear_model = _train_model(bear_combined, split_date=bear_split_date)

        print("  Running bear-period backtest...")
        bear_portfolio_df, bear_trades_df, _, _ = run_backtest(
            bear_featured, price_data, split_date=bear_split_date,
            fund_data=fund_data, sector_map=sector_map,
            earnings_dates=earn_dates, model=bear_model)

        # Restore the main-period model to disk (train_model overwrote it) so
        # live trading / next dashboard run don't inherit the bear-era model.
        trained_model.save_model(os.path.join(
            os.path.dirname(__file__), "..", "models", "xgb_model.json"))

        # Fetch full SPY range so both periods' alpha can be computed.
        full_spy = _download_benchmark("SPY", start, end)

        main_summary = _summarize_period(portfolio_df, trades_df, full_spy)
        bear_summary = _summarize_period(bear_portfolio_df, bear_trades_df, full_spy)

        print("\n" + "=" * 70)
        print("  PERFORMANCE COMPARISON")
        print(f"    Bull period: {portfolio_df.index[0].date()} "
              f"to {portfolio_df.index[-1].date()}")
        print(f"    Bear period: {bear_portfolio_df.index[0].date()} "
              f"to {bear_portfolio_df.index[-1].date()}")
        print("=" * 70)
        header = f"  {'Metric':<16}  {'Bull':>14}  {'Bear':>14}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        print(f"  {'Total Return':<16}  "
              f"{main_summary['total_return']*100:>+13.2f}%  "
              f"{bear_summary['total_return']*100:>+13.2f}%")
        print(f"  {'Sharpe Ratio':<16}  "
              f"{main_summary['sharpe']:>14.2f}  "
              f"{bear_summary['sharpe']:>14.2f}")
        print(f"  {'Max Drawdown':<16}  "
              f"{main_summary['max_dd']*100:>13.2f}%  "
              f"{bear_summary['max_dd']*100:>13.2f}%")
        print(f"  {'Win Rate':<16}  "
              f"{main_summary['win_rate']:>13.1f}%  "
              f"{bear_summary['win_rate']:>13.1f}%")
        print(f"  {'Alpha (ann.)':<16}  "
              f"{main_summary['alpha']*100:>+13.2f}%  "
              f"{bear_summary['alpha']*100:>+13.2f}%")
        print("=" * 70)

        # Build the HTML section for the main report + a standalone file.
        main_period_str = f"{portfolio_df.index[0].date()} to {portfolio_df.index[-1].date()}"
        bear_period_str = f"{bear_portfolio_df.index[0].date()} to {bear_portfolio_df.index[-1].date()}"
        bear_html = _bear_comparison_html(
            main_summary, bear_summary, main_period_str, bear_period_str)

        standalone_path = os.path.join(OUT_DIR, "bear_comparison.html")
        try:
            with open(standalone_path, "w", encoding="utf-8") as f:
                f.write(_standalone_bear_html(bear_html))
            print(f"\nBear comparison saved to {standalone_path}")
        except Exception as e:
            print(f"  [WARN] Could not save bear_comparison.html: {e}")

    # Build + save the main HTML report with bear section embedded (or empty).
    print("\nBuilding HTML report...")
    html = _build_html(equity_b64, importance_b64, portfolio_metrics,
                       top_stocks, period, cost_data,
                       cur_hold_html, trade_log, comp_html,
                       bench_html, ab_html, heat_html, universe_label,
                       bear_html=bear_html)

    # Compute metrics for the index
    pv = portfolio_df["portfolio_value"]
    total_return = pv.iloc[-1] / pv.iloc[0] - 1
    returns = pv.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    max_dd = ((pv / pv.cummax()) - 1).min()

    print("Saving report...")
    report_path = _save_report(html, period, total_return, sharpe, max_dd, strategy_notes)

    print(f"\nReport saved to {report_path}")
    print(f"Latest copy at {REPORT_PATH}")
    print(f"Index at {INDEX_PATH}")
