"""generate_ticker_names.py — one-off cache builder for ticker -> shortened
company name lookup. Output: models/cache/ticker_names.json (~491 entries).

Pulls yfinance .info["longName"] (or shortName) for every ticker in
fetch_data.UNIVERSE_TICKERS, strips common corporate suffixes, writes a
sorted JSON dict.

Usage:
    python src/generate_ticker_names.py
"""

import json
import os
import sys
from pathlib import Path

# Ensure src/ is importable when run from project root or src/.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import yfinance as yf

from fetch_data import UNIVERSE_TICKERS


_REPO_ROOT = _SRC_DIR.parent
OUTPUT_PATH = _REPO_ROOT / "models" / "cache" / "ticker_names.json"


def shorten_name(longname: str, ticker: str) -> str:
    """Convert 'NVIDIA Corporation' -> 'NVIDIA', 'Apple Inc.' -> 'Apple'."""
    if not longname:
        return ticker
    suffixes = [
        " Corporation", " Inc.", " Inc", " Corp.", " Corp",
        " Company", " Co.", " Co", " Limited", " Ltd.", " Ltd",
        " plc", " PLC", " Holdings", " Group", " International",
        " S.A.", " AG", " N.V.", " ETF Trust",
    ]
    name = longname
    # Iterate until no more suffixes strip (handles e.g. "Foo Inc Holdings")
    changed = True
    while changed:
        changed = False
        for sfx in suffixes:
            if name.endswith(sfx):
                name = name[:-len(sfx)].strip()
                changed = True
                break
    return name if name else ticker


def main() -> None:
    tickers = sorted(UNIVERSE_TICKERS)
    names: dict[str, str] = {}
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            longname = info.get("longName") or info.get("shortName") or ""
            names[ticker] = shorten_name(longname, ticker)
            if (i + 1) % 50 == 0:
                print(f"[NAMES] {i+1}/{len(tickers)}: "
                      f"{ticker} -> {names[ticker]}")
        except Exception as e:
            print(f"[NAMES] {ticker} fetch failed: {e}, fallback to ticker")
            names[ticker] = ticker

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2, sort_keys=True)
    print(f"[NAMES] Wrote {len(names)} ticker names to {OUTPUT_PATH}")

    # Spot check
    print("[NAMES] Spot checks:")
    for t in ["AAPL", "MSFT", "NVDA", "BRK.B", "GEHC", "ARM", "GEV", "VLTO"]:
        if t in names:
            print(f"  {t}: {names[t]}")


if __name__ == "__main__":
    main()
