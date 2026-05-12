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

## 2026-05-11 — Phase 2: training pipelines + CV design

**Phase:** Phase 2 (model training infrastructure + CV design)
**Branch:** `feat/larger-universe-v1-study`
**Commits at this gate:**
- `1e449f8` — feat(study): Larger Universe v1 Phase 2 — training pipelines + CV design
- (this log entry added in a follow-up commit)
**Status:** Phase 2 complete. CV scaffolding + 10-trial smoke landed. Ready for Phase 3.

### What was built

Training scaffolding for two parallel pipelines on identical features and identical folds:

- `src/equities/study/labels.py` — forward 5-trading-day return label per (date, ticker), matching the weekly rebalance cadence
- `src/equities/study/cv.py` — 5-fold expanding-window TimeSeriesSplit over 2017-05-12 → 2023-05-11 with a 5-trading-day embargo (= label horizon, prevents the leakage between training rows whose labels reference prices inside the validation window). Date-window filters for train/test/OOS also live here so Phase 4 can reuse them without re-deriving constants.
- `src/equities/study/training.py` — single-fold trainers for XGBoost (native NaN + categorical sector) and ElasticNet (SimpleImputer with add_indicator + StandardScaler + ElasticNet pipeline, sector one-hot encoded). Plus `cv_score` driver that runs a hyperparameter combo across all folds and returns mean Spearman IC.
- `scripts/research/smoke_phase2.py` — 10-trial smoke runner per model

Smoke ran on a deliberate subset (SP500 actives only, 22 price+macro features, 503 tickers, 734,646 training rows) in ~6 minutes total. Phase 3 full-universe runs are expected to take 6-7 hours.

### Smoke results — both models produce positive cross-sectional IC

| Model | Best mean IC (5-fold) | Best params |
|---|---|---|
| XGBoost | **0.0854** | max_depth=3, lr=0.20, n_est=437, subsample=0.85, colsample=0.86, min_child_weight=15, gamma=0.80, reg_alpha=4.48, reg_lambda=1.06 |
| ElasticNet | **0.0711** | alpha=0.00119, l1_ratio=0.433 |

Both fall in the 0.05-0.10 range typical for daily/weekly cross-sectional alpha in liquid US equities. Not suspiciously high (we'd flag IC > 0.20). Spearman ~0.085 ≈ correctly ranking 54.25% of pairwise comparisons (Spearman 0 = 50%, 1 = 100%) — modest but enough to drive non-trivial allocation decisions in Phase 4.

XGBoost's slight edge over ElasticNet is consistent with its ability to capture non-linear interactions. Both produced NaN on some trials/folds (constant predictions in high-regularization regions for ElasticNet, narrow training windows for low-tree-count XGBoost) — Optuna's TPE sampler avoids these regions in Phase 3 by learning from failed trials.

### Decisions made in Phase 2

- **Label horizon = rebalance cadence = 5 trading days.** Predictions at date D inform the portfolio held over (D, D+5]. Embargo equal to label horizon prevents leakage.
- **Scoring metric: cross-sectional Spearman IC** averaged across 5 folds. Picked over MSE because portfolio construction (Phase 4) is rank-driven; over Pearson because rank-correlation is robust to outliers (one ticker that triples in a week shouldn't dominate the loss).
- **5-fold expanding-window CV.** Standard for time-series finance. Each fold validates on ~1/6 of the training window (~12 months); training set grows fold-by-fold (2017-05 only → through 2022-04). Initial training-only block (1/6) is part of every fold's train set.
- **XGBoost native categorical for sector**, no one-hot. `enable_categorical=True` + `pd.Categorical` dtype.
- **ElasticNet imputation fits on train fold only** — sklearn Pipeline ensures the column mean is computed on the training rows and applied to the validation rows (no cross-fold leakage). The `add_indicator=True` flag produces the binary missingness indicator per imputed column that the spec asked for.
- **Convergence warnings suppressed** in the ElasticNet trainer — high-alpha trials don't converge to a non-constant solution and produce ConvergenceWarning + ConstantInputWarning, both expected and informative.

### Open items carried into Phase 3

1. **ElasticNet NaN-on-constant-prediction.** 5 of 10 smoke trials failed for this reason. Optuna treats NaN as a failure; the TPE sampler still adapts but it's noisy. If >40% of Phase 3 ENet trials fail, switch to returning IC=0 on constant predictions so all trials inform the surrogate. Easy 2-line change in `_safe_spearman`.
2. **Fold 0 has the narrowest training window** (1/6 of total) and the most NaN-IC trials. Phase 3's 200+ trials will fill the space densely enough that this is rare; no design change needed.
3. **Phase 3 wall-clock estimate: 6-7 hours.** XGBoost dominates: ~6h for 200 trials on full universe + features. ElasticNet ~30 min. Plan to background-run; will surface progress at the 1h mark.
4. **Beta NaN in the 2017-2019 portion of training (~37% of train rows).** Both pipelines handle this — XGBoost natively, ElasticNet via mean-impute+indicator — but the practical effect is that beta is a feature available primarily in the second half of training. Phase 5 will spot-check the feature-importance for beta to see if it's load-bearing or noise.

### Files produced this gate

| Path | Notes |
|---|---|
| `src/equities/study/__init__.py` | Namespace marker |
| `src/equities/study/labels.py` | Forward 5-day return label |
| `src/equities/study/cv.py` | TimeSeriesSplit + window filters |
| `src/equities/study/training.py` | XGBoost + ElasticNet trainers + cv_score driver + safe Spearman |
| `scripts/research/smoke_phase2.py` | 10-trial smoke runner |
| `models/features/larger_universe_v1/phase2_smoke/smoke_results.json` | Per-trial fold ICs + best params (force-added; gitignored path) |
| `docs/diagnostics/larger_universe_v1_cv_design.md` | Full CV design doc + smoke results table |

### Reproducibility — smoke

```
python scripts/research/smoke_phase2.py    # ~6 min total
```

### What's deferred

- Phase 3 full tuning (200 trials per model on full universe + features)
- Final hyperparameter persistence at `models/studies/larger_universe_v1/{xgboost,elasticnet}_best_params.json`
- `Project_State_Tracker.docx` update (Phase 5 per standing rule)

### Next: Phase 3 — full Optuna hyperparameter tuning

Run XGBoost and ElasticNet against the full feature set + full universe for 100-300 trials each. Surface convergence behavior, top hyperparameter combinations, and any overfitting red flags (IC > 0.20 would be one). Background-run since wall-clock is multi-hour. Standing process at the Phase 3 gate: commit + push + session log + report.

## 2026-05-11 (post-gate) — cross-sectional IC bug fix + re-smoke

**Phase:** Phase 2 gate review (between Phase 2 and Phase 3)
**Branch:** `feat/larger-universe-v1-study`
**Commits at this gate:**
- `8eb57f2` — fix(study): replace panel-wise IC with cross-sectional IC + report fold stats
- (this log entry added in a follow-up commit)
**Status:** Phase 3 paused. Awaiting Mike's decision on the path forward given the corrected smoke results.

### The bug

Phase 2's `_safe_spearman` computed a single Spearman across all (date, ticker) rows in the validation fold — **panel-wise IC**, not cross-sectional. The cv_design.md doc claimed cross-sectional. Mike's gate review specifically asked to verify the implementation matched the claim; I re-read my own code and surfaced the discrepancy.

Panel-wise IC conflates stock-ranking signal (what the portfolio uses) with market-timing signal (what it doesn't). For a 5-day weekly cross-sectional portfolio, only the former matters.

### The fix

Replaced `_safe_spearman` with `cross_sectional_ic_stats(preds, val_df, min_tickers=30)` in `src/equities/study/training.py`:

```
for each unique date D in val_fold with >= 30 valid rows:
    rho_D = spearmanr(preds_on_D, realized_returns_on_D)
    if rho_D is finite: per_date.append(rho_D)
return {
    mean_ic       : average across qualifying dates,
    std_ic        : std across qualifying dates,
    positive_rate : (per_date > 0).mean(),
    n_dates_scored: count of qualifying dates,
}
```

Both trainers (`train_xgb_single_fold`, `train_enet_single_fold`) return the stats dict instead of a single float. `cv_score`'s `FoldResult` gains four columns. Optuna objective is `mean_ic` aggregated across folds; std and positive-rate are surfaced but not optimized.

Also tightened `max_depth` from 3-10 to 3-8 (Mike's small suggestion at the Phase 2 gate — financial tabular data rarely needs deeper trees).

### Corrected smoke results

Same smoke setup as before (SP500 actives, 22 price+macro features, 10 trials each):

**XGBoost — degenerate "winner" trial obscures a near-zero real signal.** Best mean IC of 0.2177 came from a trial with constant predictions on 3 of 5 folds (n_dates_scored=0); the 2 surviving folds totaled 74 dates of which one fold contributed mean_ic=0.415 over just 4 dates. Well-covered XGBoost trials cluster at **-0.01 to +0.01 mean cross-sectional IC**.

**ElasticNet — clean coverage but near-zero signal.** Best mean IC 0.0131. Per-fold ICs span -0.02 to +0.03 with positive-rate ~0.50 (coin-flip per date). Every fold covered ~251 dates (no constant-prediction collapse).

Verified by direct prediction inspection: XGBoost's fold-0 model produces nunique=1 predictions per date (every ticker on a given date gets the same value), nunique~15 across dates. **The model learned a market-timing signal from macro features and ignored ticker-level price features.** Panel-wise Spearman picks up that date-level signal; cross-sectional Spearman correctly reports zero stock-ranking content.

### Decision matrix from the agreed framework

| Smoke result | Action |
|---|---|
| Corrected IC 0.05-0.08 | Real signal, proceed to Phase 3 |
| 0.02-0.04 | Weaker than panel suggested but proceed |
| **Near zero or negative** | **PAUSE — decision needed** |

We are in the "pause" zone. The smoke deliberately used a 22-feature subset (no fundamentals, no sector, no log_market_cap, no index-membership flags). The features designed to provide cross-sectional differentiation were all excluded.

### Open question for Mike at this gate

The smoke result says either:
1. **Price + macro features have no cross-sectional alpha at 5-day horizon for SP500 actives, but the full feature set will rescue it.** Reasonable expectation — fundamentals (P/E, P/B, ROE, growth metrics) and sector are the canonical cross-sectional alpha sources in factor research.
2. **The 5-day horizon is too short for the feature set we have, full stop.** Weekly cross-sectional alpha is genuinely difficult; many academic factor studies use monthly horizons.
3. **The model architecture needs to change** (e.g., predict ranks rather than returns, or use a cross-sectional loss like LambdaRank instead of MSE).

Options I'd recommend offering Mike at the gate:
- (a) Re-smoke with the full 38-feature set (still SP500-actives subset, ~6 min wall-clock) and re-evaluate — directly tests hypothesis 1
- (b) Proceed to Phase 3 anyway with the full feature set; Phase 3 is the proper test and its results will tell us definitively
- (c) Pause Phase 3 and reconsider feature set + horizon design before any further compute

### Files modified

| Path | Change |
|---|---|
| `src/equities/study/training.py` | Added `cross_sectional_ic_stats`; replaced `_safe_spearman`; updated trainers + cv_score; tightened max_depth 3-10 → 3-8 |
| `scripts/research/smoke_phase2.py` | Per-fold structured output (n_dates, mean_ic, std_ic, positive_rate) |
| `docs/diagnostics/larger_universe_v1_cv_design.md` | Replaced "Scoring metric" section + smoke results section + TL;DR with corrected story |
| `models/features/larger_universe_v1/phase2_smoke/smoke_results.json` | Re-run results with cross-sectional metric |
