"""Phase 4: Promote Larger Universe v1 Finnhub fetch outputs into a snapshot.

Mirrors the layout of models/snapshots/equities/pre_v2_20260505/ but only
populates the components that the Larger Universe v1 fetch produced:

  - price_cache/<SYM>.parquet     (Finnhub /stock/candle, truncated at delisting)
  - cache/fundamentals.json        (Finnhub /stock/metric, transformed to legacy shape)
  - cache/universe.json            (Wikipedia component-change map + SEC CIK)
  - cache/macro_signals.parquet    (carried forward from pre_v2 — FRED, ticker-independent)
  - cache/macro_signals.meta.json  (carried forward)
  - manifest.json                  (file listing in pre_v2 format)
  - README.md                      (v1 properties + known gaps)

Intentionally NOT promoted (per Phase 3 outcome + Mike's policy):
  - cache/earnings_dates.json   — sanity gate fired at 23.6%, drop from v1
  - cache/analyst_targets.json  — Finnhub Basic gives only 4 months historical
  - cache/feature_matrix.parquet — downstream artifact, rebuilt by feature pipeline
  - cache/ticker_names.json     — optional metadata, derivable from universe.json
  - cache/sector_map.json       — Finnhub /stock/metric doesn't include GICS sector

DOES NOT touch models/snapshots/equities/pre_v2_20260505/.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_PRICE = ROOT / "models" / "cache" / "equities" / "finnhub" / "prices"
SRC_METRIC = ROOT / "models" / "cache" / "equities" / "finnhub" / "metrics"
SRC_UNIVERSE = ROOT / "docs" / "larger_universe_v1_universe.json"
LEGACY_SNAPSHOT = ROOT / "models" / "snapshots" / "equities" / "pre_v2_20260505"

TODAY = date.today().strftime("%Y%m%d")
SNAP_NAME = f"larger_universe_v1_{TODAY}"
DST = ROOT / "models" / "snapshots" / "equities" / SNAP_NAME


def copy_prices(dst_price_dir: Path) -> int:
    """Copy every parquet from finnhub cache into snapshot/price_cache/."""
    dst_price_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(SRC_PRICE.glob("*.parquet")):
        shutil.copy2(src, dst_price_dir / src.name)
        n += 1
    return n


def build_fundamentals(dst_cache_dir: Path) -> tuple[int, int]:
    """Transform Finnhub /stock/metric JSON snapshots into a single
    fundamentals.json keyed by ticker.

    Legacy format expected downstream is {ticker: {field: value, ...}}.
    Finnhub bodies look like {symbol, metric: {field: value}, metricType,
    series}. We flatten by lifting metric.* up to the top level under
    the ticker key. Tickers with empty/missing metric body are skipped.
    """
    dst_cache_dir.mkdir(parents=True, exist_ok=True)
    fund: dict[str, dict] = {}
    n_attempted = 0
    n_populated = 0
    for src in sorted(SRC_METRIC.glob("*.json")):
        n_attempted += 1
        sym = src.stem
        body = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            continue
        metric = body.get("metric")
        if not metric:
            continue
        fund[sym] = metric
        n_populated += 1
    (dst_cache_dir / "fundamentals.json").write_text(
        json.dumps(fund), encoding="utf-8"
    )
    return n_attempted, n_populated


def copy_universe(dst_cache_dir: Path) -> int:
    universe = json.loads(SRC_UNIVERSE.read_text(encoding="utf-8"))
    (dst_cache_dir / "universe.json").write_text(
        json.dumps(universe, indent=2), encoding="utf-8"
    )
    return len(universe)


def carry_macro_signals(dst_cache_dir: Path) -> None:
    """FRED-sourced macro signals are ticker-independent; carry the existing
    parquet (and meta) forward as-is so feature engineering can run against
    the new snapshot without re-fetching FRED."""
    for name in ("macro_signals.parquet", "macro_signals.meta.json"):
        src = LEGACY_SNAPSHOT / "cache" / name
        if src.exists():
            shutil.copy2(src, dst_cache_dir / name)


def build_manifest(dst_root: Path) -> dict:
    """Build a manifest in the same shape as pre_v2_20260505/manifest.json."""
    files: list[dict] = []
    for path in sorted(dst_root.rglob("*")):
        if path.is_dir():
            continue
        if path.name == "manifest.json":
            continue
        rel = path.relative_to(dst_root)
        files.append({
            "key": rel.as_posix(),
            "mtime": path.stat().st_mtime,
            "size": path.stat().st_size,
        })
    manifest = {
        "comment": (
            "Larger Universe v1 snapshot — Finnhub Basic-tier prices + "
            "fundamentals for SP1500 current members + last-decade delisted. "
            "Survivorship-bias-mitigated (best-effort) with documented residual "
            "gaps. See README.md and docs/diagnostics/"
            "larger_universe_v1_snapshot_summary.md."
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "files": files,
    }
    (dst_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def write_readme(dst_root: Path, *, n_prices: int, n_fund: int, n_univ: int) -> None:
    text = f"""# Larger Universe v1 — equity snapshot

**Created:** {time.strftime('%Y-%m-%d')}
**Source vendor:** Finnhub Basic tier ($49.99/mo personal use)
**Window:** 2016-05-12 .. 2026-05-11 (10y daily)
**Universe construction:** Wikipedia S&P 500 + S&P 400 + S&P 600
component-change tables, last 10y, SEC CIK-disambiguated where possible.

## Contents

| Path | Count | Notes |
|---|---|---|
| `price_cache/*.parquet` | {n_prices} | Daily OHLCV, split-and-dividend-adjusted. OTC-tail-truncated at Wikipedia-documented delisting date for removed names. |
| `cache/fundamentals.json` | {n_fund} | Finnhub /stock/metric snapshot per ticker (point-in-time, ~130 metrics). |
| `cache/universe.json` | {n_univ} | Membership map: tier (SP500/400/600), status (active/removed), removed_at, reuse_flag. |
| `cache/macro_signals.parquet` | (carried forward) | FRED-sourced, ticker-independent — unchanged from pre_v2. |
| `manifest.json` | — | File listing in pre_v2-compatible format. |

## Intentionally absent (vs. pre_v2 layout)

- `cache/earnings_dates.json` — Finnhub Basic forward-only for earnings; yfinance refetch at scale produced only 23.6% non-empty after retries, triggering the stash sanity gate. Per Mike's <85% rule, **earnings_dates is dropped from v1**. The new study spec must be earnings-agnostic.
- `cache/analyst_targets.json` — Finnhub Basic only returns 4 months of analyst rec history; not enough for back-history.
- `cache/sector_map.json` — Finnhub /stock/metric body does not include GICS sector. Could be added later via /stock/profile2 (60/min bucket).
- `cache/feature_matrix.parquet` — built downstream by the feature pipeline; not a vendor artifact.
- `cache/ticker_names.json` — derivable from universe.json's `company` field.

## Residual survivorship bias

This is "best-effort survivorship-bias mitigation" — not survivorship-bias-free. The known systematic exclusions (per ``docs/diagnostics/larger_universe_v1_snapshot_summary.md``):

- 2008 financial-crisis-era delistings (BSC, LEHM) are beyond Finnhub Basic's 10y warranty
- A small set of ticker-reuse cases where the pre-reuse entity's history is irretrievable (VAL=Valspar's pre-2017 history under Valaris)
- 17 of 616 deduped historical-removal records lack a Wikipedia removal date (asymmetric add-only entries)

Studies using this snapshot should disclaim "best-effort survivorship-bias mitigation with documented residual gaps tilting toward overstated returns by an estimated 0.3-0.6 pp/yr" rather than "survivorship-bias-free".

## Lifecycle

The legacy snapshot at `models/snapshots/equities/pre_v2_20260505/` remains the canonical anchor for the three promoted studies. This v1 is alongside it — fresh study material, not a replacement.
"""
    (dst_root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    print(f"[snapshot] target: {DST}")
    if DST.exists():
        raise SystemExit(f"refusing to overwrite existing snapshot: {DST}")
    DST.mkdir(parents=True)

    print("[snapshot] copying prices...")
    n_prices = copy_prices(DST / "price_cache")
    print(f"           copied {n_prices} parquets")

    print("[snapshot] building fundamentals.json...")
    n_attempted, n_populated = build_fundamentals(DST / "cache")
    print(f"           {n_populated} of {n_attempted} metric bodies had data")

    print("[snapshot] copying universe.json...")
    n_univ = copy_universe(DST / "cache")
    print(f"           {n_univ} universe records")

    print("[snapshot] carrying forward macro_signals...")
    carry_macro_signals(DST / "cache")

    print("[snapshot] building manifest.json...")
    manifest = build_manifest(DST)
    print(f"           {len(manifest['files'])} files in manifest")

    print("[snapshot] writing README.md...")
    write_readme(DST, n_prices=n_prices, n_fund=n_populated, n_univ=n_univ)

    print(f"[snapshot] DONE: {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
