# Larger Universe v1 study — session log

Append-only record of decisions, gate reviews, and phase transitions for the Larger Universe v1 fresh-equity study. Each entry covers one Claude Code session or one gate transition. New entries go at the bottom.

## Pre-log history

Summary of work that preceded this log (sessions before 2026-05-11 evening). Captured here as a one-time onboarding aid; future sessions get their own entries.

- **Larger Universe v1 snapshot creation.** Mike subscribed to Finnhub Basic ($49.99/mo, 150 req/min for candles, 60 req/min for fundamentals, 10y daily OHLC). Built a 2,122-ticker universe via Wikipedia S&P 500 + 400 + 600 component-change tables (current actives + last-decade removed names) with SEC CIK disambiguation for ticker-reuse. Snapshot at `models/snapshots/equities/larger_universe_v1_20260511/`. Coverage: 92.5% prices, 90.4% fundamentals. `earnings_dates` dropped per yfinance sanity-gate fire at 23.6% — Finnhub Basic is forward-only for earnings.
- **Truncation bug found and fixed during smoke.** The OTC-tail-truncation clip applied to the returned DataFrame but not the on-disk parquet — SIVB/BBBY/SBNY/FRC cache files contained months-to-years of post-bankruptcy pink-sheet candles. Fixed cache-side (clip-before-write); added 5 regression tests in `tests/equities/test_finnhub_fetcher.py`; re-fetched the affected tickers and verified on disk. All four OTC-tail cases now end at or before their Wikipedia removed_at date.
- **Polygon piggyback analysis marked OBSOLETE in tracker.** Chris's Polygon subscription is options-only with 2-year history; cannot be repurposed for stocks. Tracker Section 3 superseded by Finnhub Basic. Tracker Appendix A updated to reference both equity snapshots (legacy `pre_v2_20260505` and new `larger_universe_v1_20260511`).
- **Larger Universe v1 study spec locked.** XGBoost primary + ElasticNet sanity check, objective = excess CAGR vs SPY. Constraints: 7.5% max single position, 30% sector cap, 95-100% invested (long-only). Weekly rebalance with 1.5pp threshold-based execution. Train 2017-05-12 → 2023-05-11 (revised from 2016-05-12 at Phase 1 gate), test 2023-05-12 → 2025-12-31, OOS holdout 2026-01-01 onward. Four benchmarks (SPY, RSP, IWM, equal-weight SP1500). FeeModel matches the three promoted studies: 0.05% flat per trade leg (frequency-agnostic).
- **Phase 1 pre-work surfaced five issues; resolutions captured in this log.** Phase 1 entry below documents the design decisions executed at the gate.

## 2026-05-11 — Phase 1: feature engineering pipeline

**Phase:** Phase 1 (feature engineering + data prep)
**Branch:** `feat/larger-universe-v1-study` off `feat/larger-universe-v1`
**Commits at this gate:**
- `ec37195` — feat(features): Larger Universe v1 Phase 1 — feature engineering pipeline
- `1ea6667` — fix(features): replace static dividend_yield and beta with PIT computations
- (this log added in a follow-up commit)
**Status:** Phase 1 complete. Ready for Phase 2.

### What was built

Feature matrix at `models/features/larger_universe_v1/features.parquet`: 4,350,932 rows × 40 columns (date, ticker + 38 features), 2016-05-12 → 2026-05-11, 1,963 tickers with prices.

Supporting artifacts:
- `models/features/larger_universe_v1/fundamentals_pit.parquet` — 196,622 quarterly rows with 45-day reporting lag (industry standard for 10-Q windows)
- `models/features/larger_universe_v1/macro_signals_extended.parquet` — 10-column FRED panel, 2016-01-01 → 2026-05-11
- `models/features/larger_universe_v1/sector_map.json` — 1,782 entries from Finnhub `/stock/profile2` (sector + shareOutstanding)
- `models/features/larger_universe_v1/dividend_history.parquet` — 111,143 dividend events across 1,391 tickers
- `models/cache/equities/finnhub/prices/SPY.parquet` — 3,018 rows 2014-05-12 → 2026-05-11 (added for beta computation lookback)

### Pre-work decisions (surfaced at Phase 1 gate, before any features were built)

Five issues identified during snapshot inventory; each had a fork in the road requiring Mike's call.

1. **Fundamentals were point-in-time-2026 only in the snapshot.** Snapshot's `fundamentals.json` contained only the current `/stock/metric` snapshot per ticker. Using these as features in a 2016–2026 backtest would create universe-wide look-ahead bias. Discovered the raw `/stock/metric` response in `models/cache/equities/finnhub/metrics/<SYM>.json` contained `series.annual` + `series.quarterly` — full historical fundamentals time series with `{period, v}` entries.
   - **Decision:** Build PIT fundamentals from `series.quarterly` with 45-day reporting lag. Output to `fundamentals_pit.parquet`. Per-ticker, per-quarter lookup table used by merge_asof in feature engineering.

2. **Macro signals had partial coverage vs spec.** Snapshot's `macro_signals.parquet` had 6 columns (2018-01-01 onward); spec asked for 7 specific FRED series plus a 2016-05-12 training-window start.
   - **Decision:** Extend FRED fetch to 10 series (added `BAA10Y`, `DTWEXBGS`, `UNRATE`, `DCOILWTICO`; kept all 6 existing). Backfill start to 2016-01-01. Output to `macro_signals_extended.parquet`. Snapshot's original macro file unchanged.

3. **GICS sector missing from snapshot.** Legacy `models/cache/sector_map.json` covered 1,464 / 2,122 (69%).
   - **Decision:** Probe `/stock/profile2` capability — confirmed it works on Basic and returns `finnhubIndustry` (Finnhub's sector taxonomy, GICS Level-1-ish granularity) plus `shareOutstanding`. Fetched profile2 for all 2,122 tickers (60/min, ~33 min). 1,782 returned data; 340 (mostly delisted) returned empty body and get assigned `sector_unknown`. Used Finnhub's taxonomy as-is (no translation to legacy lowercase labels).

4. **Tenure-in-index unrecoverable for most active tickers.** Wikipedia component-change tables only track add events ~10y back; ~86% of currently-active records have null `added_at`.
   - **Decision:** **Drop tenure feature.** Feature count reduces from 30 → 29 in spec terms (actual final count is 38 after build, more than spec because all 10 macro signals are kept and `vix_5d_chg` is broken out).

5. **`log_market_cap` look-ahead.** Using current `marketCapitalization` would be biased; using `bookValue × P/B` from quarterly fundamentals doubles the noise.
   - **Decision:** Compute as `close × shareOutstanding` where shareOutstanding comes from `/stock/profile2`. Covers 97.7% of feature rows. Documented caveat: current shares used for all historical rows (buyback/issuance noise is small relative to price-driven mktcap variation for ranking purposes).

### Gate review (Phase 1 → Phase 2)

After the first Phase 1 build completed, the coverage report surfaced four additional items requiring Mike's call. Resolutions:

1. **`hy_spread` (FRED `BAMLH0A0HYM2`) only serves 2023+ data.** 27.9% coverage in the full matrix. **DROPPED** from features. `baa_spread` (BAA10Y, 100% coverage) is the credit-spread feature for Phase 2.

2. **Training window:** spec said 2016-05-12, but long-lookback features (252d return, 200d MA) need ~1y warmup. **Trimmed to 2017-05-12 → 2023-05-11** (6 years). Test and OOS unchanged. Implemented at modeling time, not at feature-matrix time; `features.parquet` retains the full 2016-05-12 → 2026-05-11 range for completeness.

3. **`sector_unknown` handling:** confirmed treated as a single normal sector by both XGBoost (native categorical) and ElasticNet (one-hot column). 30% sector concentration cap (Phase 4) treats it as one bucket — collective cap, no within-bucket limit.

4. **Static `dividend_yield` and `beta` were a look-ahead bias.** Initial build sourced these from the current `fundamentals.json` snapshot. **Replaced with PIT computations** in commit `1ea6667`:
   - `dividend_yield(D, T) = sum(amount for ex_date in (D-365, D]) / close(D, T)` from `/stock/dividend2` (1,391 of 2,122 tickers had history; 731 are non-payers and get 0, not NaN). Coverage 100%.
   - `beta(D, T) = cov(ret_T, ret_SPY) / var(ret_SPY)` over rolling 756 trading days. Coverage 63% in train, 96.7% in test, 97.8% in OOS. The 63% in train is because per-ticker snapshot prices start 2016-05-12, so the rolling 756-day window cannot fill until ~2019-05-12. Per Mike's directive ("leave NaN and let XGBoost handle it"), we accept the NaN and let XGBoost route observations through its missing-value handling.

5. **Snapshot README:** added an architectural note explaining that historical fundamentals live in the raw cache at `models/cache/equities/finnhub/metrics/<SYM>.json` (not in the snapshot's `fundamentals.json` which is current-only by design). Data files unchanged.

### Open items carried into Phase 2

- **Training window has sparse beta in 2017-2019** (63% non-null). XGBoost handles natively; ElasticNet either needs the spec's "mean-imputation + missingness indicator" treatment, or training-window trim to 2019-05-12 for the ElasticNet sanity check. To decide in Phase 2 CV-design step.
- **`sector_unknown` is 14% of the matrix** (mostly post-removal rows for historical delistings). One-hot encoding gets its own column for ElasticNet; XGBoost uses native handling. Confirmed not a problem; documented.
- **Static-fundamentals look-ahead concerns are resolved.** No remaining known look-ahead biases in the feature set.

### Files produced this gate

| Path | Notes |
|---|---|
| `docs/diagnostics/larger_universe_v1_features.md` | Full feature coverage report |
| `docs/diagnostics/finnhub_profile2_probe.json` | Capability probe (kept for audit trail) |
| `docs/sessions/larger_universe_v1/session_log.md` | This file |
| `scripts/research/build_features_larger_universe_v1.py` | Main feature builder |
| `scripts/research/build_fundamentals_pit.py` | PIT fundamentals extractor |
| `scripts/research/build_macro_signals_extended.py` | Extended FRED fetch |
| `scripts/research/build_pit_beta_div_yield.py` | Rolling beta + trailing-12mo dividend yield |
| `scripts/research/fetch_finnhub_profile2.py` | Profile2 fetch with resume + symbol sanitization |
| `scripts/research/fetch_spy_and_dividends.py` | SPY history + /stock/dividend2 batch |
| `scripts/research/probe_finnhub_profile2.py` | One-off capability probe |
| `models/features/larger_universe_v1/sector_map.json` | Aggregated profile2 results (committed) |
| `models/snapshots/equities/larger_universe_v1_20260511/README.md` | Architectural note added |

Build artifacts NOT committed (gitignored `*.parquet`; regeneratable from scripts):
- `models/features/larger_universe_v1/features.parquet` (465 MB)
- `models/features/larger_universe_v1/fundamentals_pit.parquet` (20 MB)
- `models/features/larger_universe_v1/macro_signals_extended.parquet` (89 KB)
- `models/features/larger_universe_v1/dividend_history.parquet`
- `models/cache/equities/finnhub/prices/SPY.parquet` (and per-ticker dividends/profile2 caches)

### Rebuild recipe (for cross-workstation reproducibility)

Assuming the Larger Universe v1 snapshot, FRED key, and Finnhub key are all present:

```
python scripts/research/build_macro_signals_extended.py     # ~10s
python scripts/research/fetch_finnhub_profile2.py           # ~33 min first run; instant on resume
python scripts/research/fetch_spy_and_dividends.py          # ~15 min first run; instant on resume
python scripts/research/build_fundamentals_pit.py           # ~5s
python scripts/research/build_features_larger_universe_v1.py # ~2 min (this includes the initial static beta/dy — overridden in next step)
python scripts/research/build_pit_beta_div_yield.py         # ~15s
```

Each script is idempotent against its existing outputs. Total fresh-rebuild wall-clock ~50 minutes; resume from cache is ~3 minutes.

### What's deferred

- `Project_State_Tracker.docx` is NOT updated at this gate. Per the standing rule, tracker updates land at Phase 5 completion, not at intermediate phase gates.
- R2 sync — separate decision; not in scope until after Phase 5.

### Next: Phase 2 — model training infrastructure + CV design

Standing process rule from this gate forward: at each phase gate, (1) commit Phase N work, (2) write a session_log entry for it, (3) commit + push the log, (4) report. The Phase 2 entry will cover the CV design (time-series with embargo per spec), the smoke run results, and any open items for Phase 3 tuning.
