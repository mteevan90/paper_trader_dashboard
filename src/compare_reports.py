"""
compare_reports.py — Master comparison report generator.

Aggregates all backtest runs and live portfolio performance into:
  - models/master_report.html   (open in any browser)
  - models/master_report.csv    (import into Excel)

Run anytime:
    python compare_reports.py
"""

import csv
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
REPORTS_DIR  = os.path.join(MODELS_DIR, "reports")
MANIFEST     = os.path.join(REPORTS_DIR, "manifest.jsonl")
STATE_PATH   = os.path.join(MODELS_DIR, "portfolio_state.json")
HTML_OUT     = os.path.join(MODELS_DIR, "master_report.html")
CSV_OUT      = os.path.join(MODELS_DIR, "master_report.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    if not os.path.exists(MANIFEST):
        return []
    entries = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return sorted(entries, key=lambda e: e["timestamp"])


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH) as f:
        return json.load(f)


def fetch_spy_since(start: str) -> pd.Series:
    """Download SPY closes from start date to today."""
    try:
        spy = yf.download("SPY", start=start, progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.droplevel(1)
        return spy["Close"]
    except Exception:
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Live portfolio analytics
# ---------------------------------------------------------------------------

def build_equity_curve(state: dict) -> pd.DataFrame:
    """Reconstruct daily equity curve from trade log + current prices."""
    log = state.get("trade_log", [])
    if not log:
        return pd.DataFrame()

    starting = state.get("starting_value", 100_000)

    # Build list of (date, ticker, action, shares, price)
    trades = []
    for t in log:
        trades.append({
            "date":   pd.Timestamp(t["date"]),
            "ticker": t["ticker"],
            "action": t["action"],
            "shares": t["shares"],
            "price":  t["price"],
            "fee":    t.get("fee", 0),
        })
    if not trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades).sort_values("date")
    start_date = trades_df["date"].min()
    end_date   = pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))

    # Get all tickers ever traded
    all_tickers = trades_df["ticker"].unique().tolist()

    # Download price history
    print("  Downloading price history for equity curve...")
    try:
        prices_raw = yf.download(
            all_tickers, start=start_date.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False
        )
        if isinstance(prices_raw.columns, pd.MultiIndex):
            prices = prices_raw["Close"]
        else:
            prices = prices_raw[["Close"]].rename(columns={"Close": all_tickers[0]})
    except Exception:
        return pd.DataFrame()

    # Simulate day-by-day
    cash      = starting
    positions = {}  # ticker -> shares
    curve     = []

    trading_days = pd.bdate_range(start=start_date, end=end_date)

    for day in trading_days:
        day_trades = trades_df[trades_df["date"] == day]
        for _, t in day_trades.iterrows():
            if t["action"] == "BUY":
                cash -= t["shares"] * t["price"] + t["fee"]
                positions[t["ticker"]] = positions.get(t["ticker"], 0) + t["shares"]
            elif t["action"] in ("SELL", "STOP_ATR", "STOP", "STOP10"):
                cash += t["shares"] * t["price"] - t["fee"]
                positions[t["ticker"]] = max(0, positions.get(t["ticker"], 0) - t["shares"])

        # Value positions
        port_val = cash
        for tkr, shares in positions.items():
            if shares <= 0:
                continue
            try:
                if day in prices.index:
                    if tkr in prices.columns:
                        p = prices.loc[day, tkr]
                    else:
                        continue
                    if not np.isnan(p):
                        port_val += shares * p
            except Exception:
                pass

        curve.append({"date": day, "portfolio_value": port_val})

    return pd.DataFrame(curve).set_index("date")


def live_metrics(curve: pd.DataFrame, state: dict) -> dict:
    """Compute live portfolio metrics."""
    if curve.empty:
        return {}
    vals    = curve["portfolio_value"].dropna()
    if len(vals) < 2:
        return {}
    rets    = vals.pct_change().dropna()
    sharpe  = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    total   = vals.iloc[-1] / vals.iloc[0] - 1
    dd      = ((vals / vals.cummax()) - 1).min()
    vol     = rets.std() * np.sqrt(252)
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    log     = state.get("trade_log", [])
    trades_df = pd.DataFrame(log) if log else pd.DataFrame()

    wins = total_closed = 0
    atr_stops = 0
    if not trades_df.empty:
        atr_stops = len(trades_df[trades_df["action"] == "STOP_ATR"])
        for tkr in trades_df["ticker"].unique():
            tt   = trades_df[trades_df["ticker"] == tkr].sort_values("date")
            buys = tt[tt["action"] == "BUY"].reset_index(drop=True)
            sells= tt[tt["action"].isin(["SELL","STOP_ATR","STOP","STOP10"])].reset_index(drop=True)
            for i in range(min(len(buys), len(sells))):
                if sells.iloc[i]["price"] > buys.iloc[i]["price"]:
                    wins += 1
                total_closed += 1

    win_rate = (wins / total_closed * 100) if total_closed else 0

    return {
        "start_date":   vals.index[0].date(),
        "end_date":     vals.index[-1].date(),
        "start_value":  vals.iloc[0],
        "end_value":    vals.iloc[-1],
        "total_return": total,
        "sharpe":       sharpe,
        "max_drawdown": dd,
        "volatility":   vol,
        "pos_months":   int((monthly > 0).sum()),
        "neg_months":   int((monthly <= 0).sum()),
        "total_trades": len(trades_df),
        "win_rate":     win_rate,
        "atr_stops":    atr_stops,
        "fees":         state.get("total_fees", 0),
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(manifest: list[dict], live: dict, curve: pd.DataFrame,
              spy_curve: pd.Series):
    rows = []

    # --- Section 1: Backtest runs ---
    rows.append(["BACKTEST RUNS"])
    rows.append(["Run #", "Timestamp", "Period", "Total Return %",
                 "Sharpe Ratio", "Max Drawdown %", "Notes"])
    for i, e in enumerate(manifest, 1):
        rows.append([
            i,
            e["timestamp"].replace("_", " "),
            e["period"],
            f"{e['total_return']:+.2f}",
            f"{e['sharpe']:.2f}",
            f"{e['max_dd']:.2f}",
            e.get("notes", ""),
        ])

    rows.append([])

    # --- Section 2: Live portfolio summary ---
    rows.append(["LIVE PORTFOLIO SUMMARY"])
    if live:
        rows.append(["Metric", "Value"])
        rows.append(["Start Date",       str(live["start_date"])])
        rows.append(["End Date",         str(live["end_date"])])
        rows.append(["Starting Value",   f"${live['start_value']:,.2f}"])
        rows.append(["Current Value",    f"${live['end_value']:,.2f}"])
        rows.append(["Total Return",     f"{live['total_return']:+.2%}"])
        rows.append(["Sharpe Ratio",     f"{live['sharpe']:.2f}"])
        rows.append(["Max Drawdown",     f"{live['max_drawdown']:.2%}"])
        rows.append(["Volatility (ann)", f"{live['volatility']:.2%}"])
        rows.append(["Total Trades",     live["total_trades"]])
        rows.append(["Win Rate",         f"{live['win_rate']:.1f}%"])
        rows.append(["ATR Stop-Losses",  live["atr_stops"]])
        rows.append(["Total Fees",       f"${live['fees']:,.2f}"])
        rows.append(["Positive Months",  live["pos_months"]])
        rows.append(["Negative Months",  live["neg_months"]])
    else:
        rows.append(["No live trading data yet"])

    rows.append([])

    # --- Section 3: Daily equity curve ---
    rows.append(["DAILY EQUITY CURVE"])
    rows.append(["Date", "Portfolio Value ($)", "SPY Value ($)", "Portfolio Return %", "SPY Return %"])
    if not curve.empty:
        start_val   = curve["portfolio_value"].iloc[0]
        spy_start   = spy_curve.iloc[0] if not spy_curve.empty else None
        for date, row in curve.iterrows():
            pv      = row["portfolio_value"]
            pr      = (pv / start_val - 1) * 100
            try:
                sv  = spy_curve.loc[date] if date in spy_curve.index else None
                sr  = (sv / spy_start - 1) * 100 if sv and spy_start else ""
                sv_str = f"${sv:,.2f}" if sv else ""
                sr_str = f"{sr:+.2f}" if sr != "" else ""
            except Exception:
                sv_str = sr_str = ""
            rows.append([
                date.strftime("%Y-%m-%d"),
                f"{pv:,.2f}",
                sv_str,
                f"{pr:+.2f}",
                sr_str,
            ])

    rows.append([])

    # --- Section 4: Trade log ---
    rows.append(["TRADE LOG"])
    rows.append(["Date", "Ticker", "Action", "Shares", "Price", "Fee ($)", "Return %"])
    state = load_state()
    for t in state.get("trade_log", []):
        rows.append([
            str(t["date"]),
            t["ticker"],
            t["action"],
            t["shares"],
            f"{t['price']:.2f}",
            f"{t.get('fee', 0):.2f}",
            f"{t.get('return_pct', ''):+.1f}" if t.get("return_pct") is not None and t.get("return_pct") != "" else "",
        ])

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"  CSV saved to {CSV_OUT}")


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _color(val: float, good_positive: bool = True) -> str:
    if val > 0:
        return "#16a34a" if good_positive else "#dc2626"
    elif val < 0:
        return "#dc2626" if good_positive else "#16a34a"
    return "#64748b"


def build_html(manifest: list[dict], live: dict, curve: pd.DataFrame,
               spy_curve: pd.Series) -> str:

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Backtest runs table ---
    if manifest:
        best_return = max(e["total_return"] for e in manifest)
        run_rows = ""
        for i, e in enumerate(manifest, 1):
            is_best = e["total_return"] == best_return
            bg      = "#dcfce7" if is_best else ("white" if i % 2 else "#f8fafc")
            ret_col = _color(e["total_return"])
            dd_col  = _color(e["max_dd"], good_positive=False)
            run_rows += f"""<tr style="background:{bg}">
                <td>{i}</td>
                <td>{e['timestamp'].replace('_',' ')}</td>
                <td>{e['period']}</td>
                <td style="color:{ret_col};font-weight:600">{e['total_return']:+.2f}%</td>
                <td>{e['sharpe']:.2f}</td>
                <td style="color:{dd_col}">{e['max_dd']:.2f}%</td>
                <td>{e.get('notes','—')}</td>
            </tr>"""
        backtest_section = f"""<div class="card">
            <h2>Backtest Run History ({len(manifest)} runs)</h2>
            <p class="sub">Best run highlighted in green.</p>
            <div style="overflow-x:auto">
            <table>
                <thead><tr><th>#</th><th>Generated</th><th>Period</th>
                <th>Total Return</th><th>Sharpe</th><th>Max Drawdown</th><th>Notes</th></tr></thead>
                <tbody>{run_rows}</tbody>
            </table></div></div>"""
    else:
        backtest_section = '<div class="card"><h2>Backtest Run History</h2><p>No runs recorded yet.</p></div>'

    # --- Live metrics cards ---
    if live:
        ret_col = _color(live["total_return"])
        dd_col  = _color(live["max_drawdown"], good_positive=False)

        def stat(label, val, color="#1e293b"):
            return f'<div class="stat"><div class="stat-val" style="color:{color}">{val}</div><div class="stat-label">{label}</div></div>'

        stats_html = f"""
        {stat("Total Return", f"{live['total_return']:+.2%}", ret_col)}
        {stat("Sharpe Ratio", f"{live['sharpe']:.2f}")}
        {stat("Max Drawdown", f"{live['max_drawdown']:.2%}", dd_col)}
        {stat("Volatility", f"{live['volatility']:.2%}")}
        {stat("Win Rate", f"{live['win_rate']:.1f}%")}
        {stat("Total Trades", live['total_trades'])}
        {stat("ATR Stops", live['atr_stops'])}
        {stat("Fees Paid", f"${live['fees']:,.2f}")}
        {stat("Positive Months", live['pos_months'])}
        {stat("Negative Months", live['neg_months'])}
        """

        live_section = f"""<div class="card">
            <h2>Live Portfolio Performance</h2>
            <p class="sub">{live['start_date']} to {live['end_date']} &nbsp;|&nbsp;
            Start: ${live['start_value']:,.2f} &nbsp;|&nbsp; Current: ${live['end_value']:,.2f}</p>
            <div class="stats-grid">{stats_html}</div>
        </div>"""
    else:
        live_section = '<div class="card"><h2>Live Portfolio Performance</h2><p>No live trading data yet. Run main.py to start.</p></div>'

    # --- Equity curve chart data ---
    if not curve.empty and not spy_curve.empty:
        start_val  = curve["portfolio_value"].iloc[0]
        spy_start  = None
        chart_labels = []
        port_data  = []
        spy_data   = []

        for date, row in curve.iterrows():
            pv = row["portfolio_value"]
            chart_labels.append(date.strftime("%Y-%m-%d"))
            port_data.append(round(pv, 2))
            if date in spy_curve.index:
                sv = spy_curve.loc[date]
                if spy_start is None:
                    spy_start = sv
                spy_norm = round(float(sv) / float(spy_start) * float(start_val), 2)
                spy_data.append(spy_norm)
            else:
                spy_data.append(None)

        labels_js   = json.dumps(chart_labels)
        port_js     = json.dumps(port_data)
        spy_js      = json.dumps(spy_data)

        chart_section = f"""<div class="card">
            <h2>Live Portfolio Equity Curve vs SPY</h2>
            <canvas id="equityChart" style="max-height:350px"></canvas>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
            <script>
            new Chart(document.getElementById('equityChart'), {{
                type: 'line',
                data: {{
                    labels: {labels_js},
                    datasets: [
                        {{ label: 'Paper Trader', data: {port_js},
                           borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.08)',
                           borderWidth: 2, pointRadius: 0, fill: true, tension: 0.1 }},
                        {{ label: 'SPY (normalized)', data: {spy_js},
                           borderColor: '#f59e0b', backgroundColor: 'transparent',
                           borderWidth: 1.5, pointRadius: 0, tension: 0.1 }}
                    ]
                }},
                options: {{
                    responsive: true,
                    interaction: {{ intersect: false, mode: 'index' }},
                    plugins: {{ legend: {{ position: 'top' }} }},
                    scales: {{
                        x: {{ ticks: {{ maxTicksLimit: 12 }} }},
                        y: {{ ticks: {{ callback: v => '$' + v.toLocaleString() }} }}
                    }}
                }}
            }});
            </script>
        </div>"""
    else:
        chart_section = '<div class="card"><h2>Live Portfolio Equity Curve</h2><p>Not enough data to chart yet.</p></div>'

    # --- Monthly returns table ---
    if not curve.empty:
        rets    = curve["portfolio_value"].pct_change().dropna()
        monthly = rets.resample("ME").apply(lambda x: (1+x).prod()-1)

        spy_monthly = pd.Series(dtype=float)
        if not spy_curve.empty:
            spy_rets    = spy_curve.pct_change().dropna()
            spy_monthly = spy_rets.resample("ME").apply(lambda x: (1+x).prod()-1)

        month_rows = ""
        for month in sorted(monthly.index):
            pv   = monthly.get(month, 0)
            sv   = spy_monthly.get(month, None)
            pc   = "#dcfce7" if pv >= 0 else "#fee2e2"
            sc   = "#dcfce7" if sv and sv >= 0 else "#fee2e2"
            sv_s = f"{sv:+.2%}" if sv is not None else "—"
            month_rows += f"""<tr>
                <td>{month.strftime('%b %Y')}</td>
                <td style="background:{pc};text-align:right">{pv:+.2%}</td>
                <td style="background:{sc};text-align:right">{sv_s}</td>
            </tr>"""

        monthly_section = f"""<div class="card">
            <h2>Monthly Returns</h2>
            <div style="max-height:400px;overflow-y:auto">
            <table>
                <thead><tr><th>Month</th><th style="text-align:right">Portfolio</th>
                <th style="text-align:right">SPY</th></tr></thead>
                <tbody>{month_rows}</tbody>
            </table></div></div>"""
    else:
        monthly_section = ""

    # --- Trade log table ---
    state    = load_state()
    log      = state.get("trade_log", [])
    if log:
        log_rows = ""
        for t in reversed(log):  # most recent first
            action  = t["action"]
            ret     = t.get("return_pct")
            ret_str = f"{ret:+.1f}%" if ret is not None else "—"
            if action == "BUY":
                ac = "#dcfce7"; af = "#16a34a"
            elif action == "STOP_ATR":
                ac = "#fef9c3"; af = "#92400e"
            else:
                ac = "#fee2e2"; af = "#dc2626"
            log_rows += f"""<tr style="background:{ac}">
                <td>{t['date']}</td>
                <td><strong>{t['ticker']}</strong></td>
                <td style="color:{af};font-weight:600">{action}</td>
                <td style="text-align:right">{t['shares']:.0f}</td>
                <td style="text-align:right">${t['price']:.2f}</td>
                <td style="text-align:right">${t.get('fee',0):.2f}</td>
                <td style="text-align:right">{ret_str}</td>
            </tr>"""
        trade_section = f"""<div class="card">
            <h2>Full Trade Log ({len(log)} trades, most recent first)</h2>
            <div style="max-height:500px;overflow-y:auto">
            <table>
                <thead><tr><th>Date</th><th>Ticker</th><th>Action</th>
                <th style="text-align:right">Shares</th>
                <th style="text-align:right">Price</th>
                <th style="text-align:right">Fee</th>
                <th style="text-align:right">Return</th></tr></thead>
                <tbody>{log_rows}</tbody>
            </table></div></div>"""
    else:
        trade_section = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paper Trader — Master Report</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:#f8fafc; color:#1e293b; padding:2rem; }}
  .container {{ max-width:1100px; margin:0 auto; }}
  h1 {{ font-size:1.8rem; margin-bottom:0.25rem; color:#0f172a; }}
  .sub {{ color:#64748b; font-size:0.9rem; margin-bottom:1rem; }}
  .card {{ background:#fff; border-radius:12px;
           box-shadow:0 1px 3px rgba(0,0,0,0.08);
           padding:1.5rem; margin-bottom:1.5rem; }}
  .card h2 {{ font-size:1.1rem; margin-bottom:0.5rem; color:#0f172a; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
  th,td {{ text-align:left; padding:0.55rem 0.9rem;
           border-bottom:1px solid #e2e8f0; }}
  th {{ background:#f1f5f9; font-weight:600; color:#475569; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover {{ background:#f8fafc; }}
  .stats-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
                 gap:1rem; margin-top:1rem; }}
  .stat {{ background:#f8fafc; border-radius:8px; padding:1rem; text-align:center; }}
  .stat-val {{ font-size:1.4rem; font-weight:700; }}
  .stat-label {{ font-size:0.78rem; color:#64748b; margin-top:0.25rem; }}
  .badge {{ display:inline-block; padding:0.2rem 0.6rem; border-radius:99px;
            font-size:0.75rem; font-weight:600; }}
</style>
</head>
<body>
<div class="container">
  <h1>Paper Trader — Master Report</h1>
  <p class="sub">Generated {generated} &nbsp;|&nbsp; Aggregates all backtest runs and live portfolio data</p>

  {live_section}
  {chart_section}
  {backtest_section}
  {monthly_section}
  {trade_section}

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building master report...")
    print(f"  [PATH] manifest:       {MANIFEST}")
    print(f"  [PATH] portfolio_state: {STATE_PATH}")
    print(f"  [PATH] html output:    {HTML_OUT}")
    print(f"  [PATH] csv output:     {CSV_OUT}")
    if os.path.exists(MANIFEST):
        mtime = datetime.fromtimestamp(os.path.getmtime(MANIFEST))
        size  = os.path.getsize(MANIFEST)
        print(f"  [INFO] manifest exists ({size} bytes, last modified {mtime:%Y-%m-%d %H:%M})")
    else:
        print("  [WARN] manifest.jsonl does not exist yet")
    if os.path.exists(STATE_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(STATE_PATH))
        print(f"  [INFO] portfolio_state exists (last modified {mtime:%Y-%m-%d %H:%M})")
    else:
        print("  [INFO] portfolio_state.json not found — live section will be empty "
              "(normal if main.py has never been run)")

    manifest = load_manifest()
    print(f"  Found {len(manifest)} backtest runs in manifest")
    if manifest:
        latest = manifest[-1]
        print(f"         Latest: {latest['timestamp']}  "
              f"return {latest['total_return']:+.2f}%  sharpe {latest['sharpe']:.2f}")

    state = load_state()
    print(f"  Found {len(state.get('trade_log', []))} trades in portfolio state")

    print("  Building equity curve...")
    curve = build_equity_curve(state)

    spy_curve = pd.Series(dtype=float)
    if not curve.empty:
        start_str = curve.index[0].strftime("%Y-%m-%d")
        print(f"  Downloading SPY from {start_str}...")
        spy_curve = fetch_spy_since(start_str)

    print("  Computing live metrics...")
    live = live_metrics(curve, state)

    print("  Writing HTML report...")
    html = build_html(manifest, live, curve, spy_curve)
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved to {HTML_OUT}")

    print("  Writing CSV...")
    write_csv(manifest, live, curve, spy_curve)

    print(f"\nDone.")
    print(f"  HTML: {HTML_OUT}")
    print(f"  CSV:  {CSV_OUT}")


if __name__ == "__main__":
    main()
