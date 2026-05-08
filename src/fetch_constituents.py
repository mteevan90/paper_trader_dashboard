"""One-shot helper: pull S&P 400 and S&P 600 constituents from Wikipedia,
write a small Python module fragment that can be pasted into fetch_data.py.

Not part of the runtime path — only run by hand when refreshing the
hardcoded ticker lists. Output goes to docs/sp1500_constituents.txt.

Usage:
    python src/fetch_constituents.py
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd
import requests

# Wikipedia rejects the default urllib User-Agent (HTTP 403). Fetch the
# HTML through requests with a browser UA and feed it to pandas.read_html.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/120.0 Safari/537.36")


SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"


def _yfinance_normalize(sym: str) -> str:
    """Wikipedia uses BRK.B; yfinance wants BRK-B. Same for BF.B etc."""
    return sym.replace(".", "-").strip().upper()


def _scrape_constituents(url: str, label: str) -> list[str]:
    print(f"  [SCRAPE] {label}: {url}")
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    # The constituents table is the one that contains a "Symbol" column.
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "symbol" in cols:
            sym_col = t.columns[cols.index("symbol")]
            tickers = [_yfinance_normalize(s) for s in t[sym_col].dropna().tolist()
                       if isinstance(s, str) and s.strip()]
            print(f"  [SCRAPE] {label}: {len(tickers)} tickers")
            return sorted(set(tickers))
    raise RuntimeError(f"No 'Symbol' column found in any table at {url}")


def _format_python_list(name: str, tickers: list[str]) -> str:
    out = [f"{name} = ["]
    line = "    "
    for i, t in enumerate(tickers):
        piece = f'"{t}",'
        if len(line) + len(piece) + 1 > 76:
            out.append(line.rstrip())
            line = "    "
        line += piece + " "
    if line.strip():
        out.append(line.rstrip())
    out.append("]")
    return "\n".join(out)


def main() -> None:
    sp400 = _scrape_constituents(SP400_URL, "S&P 400")
    sp600 = _scrape_constituents(SP600_URL, "S&P 600")

    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    docs_dir = os.path.join(repo_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    out_path = os.path.join(docs_dir, "sp1500_constituents.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# S&P 400: {len(sp400)} tickers\n")
        f.write(_format_python_list("SP400_TICKERS", sp400))
        f.write("\n\n")
        f.write(f"# S&P 600: {len(sp600)} tickers\n")
        f.write(_format_python_list("SP600_TICKERS", sp600))
        f.write("\n")

    print(f"\n[SCRAPE] Wrote {out_path}")
    print(f"[SCRAPE] S&P 400: {len(sp400)}  |  S&P 600: {len(sp600)}")


if __name__ == "__main__":
    sys.exit(main())
