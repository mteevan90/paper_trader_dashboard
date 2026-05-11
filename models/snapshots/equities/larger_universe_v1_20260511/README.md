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
| `cache/fundamentals.json` | 1919 | Finnhub /stock/metric snapshot per ticker (point-in-time, ~130 metrics). |
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

## Lifecycle

The legacy snapshot at `models/snapshots/equities/pre_v2_20260505/` remains the canonical anchor for the three promoted studies. This v1 is alongside it — fresh study material, not a replacement.
