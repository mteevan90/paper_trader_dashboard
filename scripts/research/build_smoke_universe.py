"""Build a 75-ticker composed smoke universe for Phase 3 validation.

Composition (per Mike's spec):
  - 25 SP500 active large-caps, randomly sampled across sectors
  - 25 SP400 active mid-caps
  - 15 SP600 active small-caps
  - 10 known OTC-tail-truncation cases (SIVB, FRC, BBBY, SBNY + 6 others)

Writes to docs/diagnostics/smoke_universe.json so it can be passed via
fetch_larger_universe.py --universe-path.

Deterministic via a fixed RNG seed so the smoke is reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "larger_universe_v1_universe.json"
OUT = ROOT / "docs" / "diagnostics" / "smoke_universe.json"

REQUIRED_DELISTED = [
    "SIVB", "FRC", "BBBY", "SBNY",   # OTC-tail style
    "TIF", "CTXS", "MNDT", "DISCA",  # clean delistings with dates
    "ATVI", "TWTR",                  # clean recent
]

# Large-cap SP500 anchors across sectors (verify in universe before using)
SP500_ANCHORS = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA",      # tech
    "JPM", "BAC", "GS",                            # financials
    "JNJ", "PFE", "UNH",                           # healthcare
    "WMT", "COST", "HD",                           # consumer
    "XOM", "CVX",                                  # energy
    "BA", "CAT",                                   # industrials
    "PG", "KO", "PEP",                             # staples
    "T", "VZ",                                     # telecom
    "DUK", "NEE",                                  # utilities
]


def main() -> int:
    rng = random.Random(42)
    records = json.loads(SRC.read_text(encoding="utf-8"))
    by_symbol_active: dict[str, dict] = {}
    by_symbol_removed: dict[str, dict] = {}
    for r in records:
        if r["status"] == "active":
            by_symbol_active[r["symbol"]] = r
        else:
            # Prefer the one with a removed_at date
            if r["symbol"] not in by_symbol_removed or (
                r.get("removed_at") and not by_symbol_removed[r["symbol"]].get("removed_at")
            ):
                by_symbol_removed[r["symbol"]] = r

    # 1. SP500 anchors (filter to those present in universe)
    sp500_picks: list[dict] = []
    for sym in SP500_ANCHORS:
        if sym in by_symbol_active and by_symbol_active[sym]["tier"] == "SP500":
            sp500_picks.append(by_symbol_active[sym])
    if len(sp500_picks) < 25:
        extras = [r for r in by_symbol_active.values()
                  if r["tier"] == "SP500" and r["symbol"] not in {p["symbol"] for p in sp500_picks}]
        rng.shuffle(extras)
        sp500_picks.extend(extras[: 25 - len(sp500_picks)])
    sp500_picks = sp500_picks[:25]

    # 2. SP400 actives - random 25
    sp400_pool = [r for r in by_symbol_active.values() if r["tier"] == "SP400"]
    rng.shuffle(sp400_pool)
    sp400_picks = sp400_pool[:25]

    # 3. SP600 actives - random 15
    sp600_pool = [r for r in by_symbol_active.values() if r["tier"] == "SP600"]
    rng.shuffle(sp600_pool)
    sp600_picks = sp600_pool[:15]

    # 4. Delisted required + fill to 10
    delisted_picks: list[dict] = []
    for sym in REQUIRED_DELISTED:
        if sym in by_symbol_removed:
            delisted_picks.append(by_symbol_removed[sym])
        elif sym in by_symbol_active:
            print(f"  note: {sym} is listed as active in universe; skipping in delisted bucket")
    if len(delisted_picks) < 10:
        extras = [r for r in by_symbol_removed.values()
                  if r.get("removed_at") and r["symbol"] not in {p["symbol"] for p in delisted_picks}]
        rng.shuffle(extras)
        delisted_picks.extend(extras[: 10 - len(delisted_picks)])
    delisted_picks = delisted_picks[:10]

    smoke = sp500_picks + sp400_picks + sp600_picks + delisted_picks
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    print(f"Wrote {len(smoke)} smoke tickers to {OUT}:")
    print(f"  SP500 active  : {len(sp500_picks):>3}  {[r['symbol'] for r in sp500_picks]}")
    print(f"  SP400 active  : {len(sp400_picks):>3}  {[r['symbol'] for r in sp400_picks]}")
    print(f"  SP600 active  : {len(sp600_picks):>3}  {[r['symbol'] for r in sp600_picks]}")
    print(f"  Removed (tail): {len(delisted_picks):>3}  {[r['symbol'] for r in delisted_picks]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
