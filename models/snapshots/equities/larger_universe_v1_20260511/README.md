# Larger Universe v1 — equity snapshot

**Created:** 2026-05-11
**Source vendor:** Finnhub Basic tier ($49.99/mo personal use)
**Window:** 2016-05-12 .. 2026-05-11 (10y daily)
**Universe construction:** Wikipedia S&P 500 + S&P 400 + S&P 600
component-change tables, last 10y, SEC CIK-disambiguated where possible.

## Contents

| Path | Count | Notes |
|---|---|---|
| `price_cache/*.parquet` | 1963 | Daily OHLCV, split-and-dividend-adjusted. OTC-tail-truncated at Wikipedia-documented delisting date for removed names. |
| `cache/fundamentals.json` | 1919 | Finnhub /stock/metric **current-snapshot-only** (~130 metrics per ticker, as of fetch time). For point-in-time historical fundamentals see "Historical fundamentals" below. |
| `cache/universe.json` | 2180 | Membership map: tier (SP500/400/600), status (active/removed), removed_at, reuse_flag. |
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

## Historical fundamentals (added 2026-05-11 post-Phase-1)

The snapshot's `cache/fundamentals.json` is a point-in-time snapshot only — using it as a feature in a 2016–2026 backtest creates look-ahead bias by construction. **Historical fundamentals time series ARE available** at the raw Finnhub cache:

```
models/cache/equities/finnhub/metrics/<SYMBOL>.json
```

Each file contains the full /stock/metric response, including `series.annual` (~40 years) and `series.quarterly` (~150 quarters) with `{period, v}` entries per metric. The Larger Universe v1 study Phase 1 extracted these into a point-in-time table at `models/features/larger_universe_v1/fundamentals_pit.parquet` with a 45-day reporting lag applied (industry-standard for 10-Q filings). See `docs/diagnostics/larger_universe_v1_features.md` and `scripts/research/build_fundamentals_pit.py` for the extraction recipe.

The snapshot's `fundamentals.json` is kept as the convenient current-state lookup for any consumer that doesn't need historical depth; future studies that need historical fundamentals should follow the same `series.quarterly`-extraction pattern. The snapshot's *data files are unchanged* — this section is documentation only.

## Lifecycle

The legacy snapshot at `models/snapshots/equities/pre_v2_20260505/` remains the canonical anchor for the three promoted studies. This v1 is alongside it — fresh study material, not a replacement.
