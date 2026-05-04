"""
morning_summary.py — Weekly dashboard rerun + terminal performance summary.

Runs the full dashboard backtest, then parses ../models/report.html and
../models/reports/manifest.jsonl and prints a concise performance summary
comparing this run to all previous runs.

Usage:
    python morning_summary.py
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
REPORT_PATH = os.path.join(MODELS_DIR, "report.html")
MANIFEST    = os.path.join(MODELS_DIR, "reports", "manifest.jsonl")

DASHBOARD_ARGS = ["2018-01-01", "2024-01-01",
                  datetime.today().strftime("%Y-%m-%d"),
                  "--universe", "combined"]


# ---------------------------------------------------------------------------
# Run dashboard
# ---------------------------------------------------------------------------

def run_dashboard() -> int:
    print(f"Running dashboard.py {' '.join(DASHBOARD_ARGS)} ...")
    proc = subprocess.run(
        [sys.executable, "dashboard.py", *DASHBOARD_ARGS],
        cwd=BASE_DIR,
    )
    return proc.returncode


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _parse_pct(s: str) -> float | None:
    m = re.search(r"([+-]?\d+\.?\d*)\s*%", s)
    return float(m.group(1)) if m else None


def _parse_float(s: str) -> float | None:
    m = re.search(r"[+-]?\d+\.?\d*", s)
    return float(m.group(0)) if m else None


def extract_benchmark_row(html: str, label: str) -> tuple[str, str, str] | None:
    """Pull (strategy, spy, qqq) cells from the 4-col benchmark comparison table."""
    pat = rf"<tr><td>{re.escape(label)}</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td></tr>"
    m = re.search(pat, html)
    return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip()) if m else None


def extract_alpha_beta(html: str) -> dict:
    """Pull alpha, beta, correlation from the 3-col alpha/beta table."""
    result = {}
    for label, key in [("Alpha \\(annualized\\)", "alpha"),
                       ("Beta", "beta"),
                       ("Correlation to SPY", "corr")]:
        pat = rf"<tr><td>{label}</td><td>([^<]+)</td>"
        m = re.search(pat, html)
        if m:
            result[key] = m.group(1).strip()
    return result


def extract_sleeve_metric(html: str, label: str) -> str | None:
    """Pull a value from the 2-col Active Sleeve Metrics table."""
    pat = rf"<tr><td>{re.escape(label)}</td><td>([^<]+)</td></tr>"
    m = re.search(pat, html)
    return m.group(1).strip() if m else None


def extract_period(html: str) -> str:
    m = re.search(r"Backtest period:\s*<strong>([^<]+)</strong>", html)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|&ndash;|-)\s*(\d{4}-\d{2}-\d{2})", html)
    return f"{m.group(1)} to {m.group(2)}" if m else "unknown"


# ---------------------------------------------------------------------------
# Manifest comparison
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
    return entries


def rank(value: float, pool: list[float], higher_is_better: bool = True) -> tuple[int, int]:
    """Return (rank, n) where rank 1 is best."""
    sorted_pool = sorted(pool, reverse=higher_is_better)
    try:
        r = sorted_pool.index(value) + 1
    except ValueError:
        r = len(sorted_pool) + 1
    return r, len(sorted_pool)


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary():
    if not os.path.exists(REPORT_PATH):
        print(f"ERROR: {REPORT_PATH} not found — dashboard may have failed.")
        return

    with open(REPORT_PATH, encoding="utf-8") as f:
        html = f.read()

    manifest = load_manifest()
    this_run = manifest[-1] if manifest else None
    history  = manifest[:-1] if len(manifest) > 1 else []

    # Benchmark row values (strategy, spy, qqq)
    ret_row  = extract_benchmark_row(html, "Total Return")
    vol_row  = extract_benchmark_row(html, "Volatility (ann.)")
    ab       = extract_alpha_beta(html)

    total_ret = extract_sleeve_metric(html, "Total Return")
    sharpe    = extract_sleeve_metric(html, "Sharpe Ratio")
    max_dd    = extract_sleeve_metric(html, "Max Drawdown")
    win_rate  = extract_sleeve_metric(html, "Win Rate")
    trades    = extract_sleeve_metric(html, "Total Trades")
    atr_stops = extract_sleeve_metric(html, "ATR Stop-Losses")
    period    = this_run["period"] if this_run else extract_period(html)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    bar = "=" * 68

    print()
    print(bar)
    print(f"  PAPER TRADER — MORNING SUMMARY     {now}")
    print(bar)
    print(f"  Period : {period}   (combined universe)")
    print()

    # Returns line — strategy vs SPY vs QQQ
    if ret_row:
        strat, spy, qqq = ret_row
        print(f"  Total Return     {strat:>10}   (SPY {spy:>8}   QQQ {qqq:>8})")
    else:
        print(f"  Total Return     {total_ret or 'n/a'}")

    print(f"  Sharpe Ratio     {sharpe or 'n/a':>10}")
    print(f"  Max Drawdown     {max_dd or 'n/a':>10}")
    if vol_row:
        print(f"  Volatility (ann) {vol_row[0]:>10}   (SPY {vol_row[1]:>8}   QQQ {vol_row[2]:>8})")
    print(f"  Alpha (ann)      {ab.get('alpha', 'n/a'):>10}   Beta {ab.get('beta', 'n/a')}   Corr {ab.get('corr', 'n/a')}")
    print(f"  Win Rate         {win_rate or 'n/a':>10}")
    print(f"  Total Trades     {trades or 'n/a':>10}   ATR Stops: {atr_stops or 'n/a'}")
    print()

    # Comparison vs history
    if this_run and history:
        all_runs = history + [this_run]
        returns = [e["total_return"] for e in all_runs]
        sharpes = [e["sharpe"]       for e in all_runs]
        dds     = [e["max_dd"]       for e in all_runs]

        r_rank, n = rank(this_run["total_return"], returns, higher_is_better=True)
        s_rank, _ = rank(this_run["sharpe"],       sharpes, higher_is_better=True)
        d_rank, _ = rank(this_run["max_dd"],       dds,     higher_is_better=True)  # less negative = better

        print(f"  vs. {len(history)} previous runs:")
        print(f"     Return   rank {r_rank}/{n}    best {max(returns):+.2f}%    median {median(returns):+.2f}%")
        print(f"     Sharpe   rank {s_rank}/{n}    best {max(sharpes):.2f}       median {median(sharpes):.2f}")
        print(f"     Drawdown rank {d_rank}/{n}    best {max(dds):+.2f}%    median {median(dds):+.2f}%")
        print()

        top = sorted(all_runs, key=lambda e: e["total_return"], reverse=True)[:3]
        print("  All-time best runs (by return):")
        for i, e in enumerate(top, 1):
            marker = "  <-- THIS RUN" if e["timestamp"] == this_run["timestamp"] else ""
            print(f"     {i}. {e['timestamp']}   {e['total_return']:+7.2f}%   Sharpe {e['sharpe']:.2f}{marker}")
    elif this_run:
        print("  First recorded run — no history to compare against.")
    else:
        print("  No manifest entries found.")

    print(bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rc = run_dashboard()
    if rc != 0:
        print(f"\nERROR: dashboard.py exited with code {rc}. Skipping summary.")
        sys.exit(rc)
    print_summary()


if __name__ == "__main__":
    main()
