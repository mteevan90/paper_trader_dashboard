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

## 2026-05-11 (later evening) — horizon/feature diagnostic + Phase 3 deferred

**Phase:** Phase 2 gate diagnostic (between Phase 2 and Phase 3)
**Branch:** `feat/larger-universe-v1-study`
**Commits at this gate:**
- `fe4cacb` — diag(study): Phase 2 horizon diagnostic — 21d horizon rescues signal
- (this log entry + tracker update committed separately)
**Status:** Phase 3 deferred to tomorrow. Diagnostic results in hand; Mike decides path forward on review.

### Why this gate happened

The IC bug fix surfaced a real concern: cross-sectional IC near zero on the price+macro-only smoke. Three causes were plausible — feature set, horizon, or model architecture. Mike commissioned a 15-min diagnostic with two variants to isolate which is binding before burning 6-7h on Phase 3.

### Diagnostic configurations

Two smoke variants on SP500-actives subset, 10 Optuna trials per model, full per-fold reporting (n_dates, mean_ic, std_ic, positive_rate):

| Variant | Features | Horizon | Embargo |
|---|---|---|---|
| Original | 22 (price+macro only) | 5d | 5d |
| A | 38 (full incl. fundamentals/sector/log_mc) | 5d | 5d |
| B | 38 (full) | 21d | 21d |

### Headline numbers (best-trial honest mean cross-sectional IC, across all 5 folds)

| Variant | XGBoost | ElasticNet | Notes |
|---|---|---|---|
| Original | ~0.000 (well-covered trials) | 0.013 | Best XGB trial was degenerate (constant predictions on most folds) |
| A | 0.009 (still degenerate F0+F1) | 0.020 | Fundamentals helped marginally; XGB still collapses on early folds |
| **B** | **0.019 (all folds covered)** | **0.031** | **No degeneracy. Real signal across all 5 folds for both models.** |

XGBoost at 21d horizon is the first configuration with no constant-prediction degeneracy — every fold scored on 251 dates. Variant B ElasticNet on fold 1 reaches mean_ic=0.116 with positive_rate=0.75.

### Common pattern flagged for Phase 5

**Fold 3 (val 2021-05-10 → 2022-05-05) is hostile for both models, both variants.** Mean IC negative (XGB-21d −0.063, ENet-21d −0.046, positive_rate ~0.4). This is the 2022 bear-market regime shift that reversed growth/momentum patterns from 2017-2021 training data. Not a bug; a real regime change. Worth disclaiming when Phase 5 writes up backtest performance — strategies that rode growth/momentum factors got hit industry-wide in 2022.

### Binding constraint identified: horizon, not feature set

At 5-day horizon, the SP500-actives cross-section doesn't have enough signal-to-noise on ticker-level features. XGBoost collapses to learning macro features only. Adding fundamentals at 5d gave ENet +0.007 (0.013 → 0.020); switching to 21d gave another +0.011 (0.020 → 0.031). The horizon shift is roughly the same magnitude lift as adding all the fundamentals — and additionally it eliminated the XGBoost constant-prediction degeneracy.

This is consistent with the academic factor-research literature: most cross-sectional alpha studies use monthly horizons. Daily/weekly cross-sectional alpha is feasible but typically needs short-horizon-tuned features (momentum, technical), not the slow-moving fundamentals the v1 spec emphasizes.

### Three options surfaced for Phase 3

1. **Change label horizon to 21d, keep weekly rebalance.** Model predicts 21-day forward return; portfolio rebalances weekly using the most recent prediction. Each prediction informs ~4 rebalances before becoming stale. Clean fix — uses signal where it lives, preserves the spec's weekly cadence. **Recommended.**
2. **Keep 5d label, run Phase 3 anyway, accept low IC.** Phase 3 burns 6-7h to tune on a near-zero signal landscape. Expected outcome: similar to Variant A. Phase 5 study disclaimer would need to acknowledge minimal cross-sectional alpha.
3. **Pivot to monthly rebalance + monthly label.** Cleanest factor-research design but violates the spec's weekly cadence. Larger architectural change.

### Code changes from this gate

- `src/equities/study/labels.py:build_labels` accepts `horizon` parameter; emits column named `target` (was `target_fwd_5d`)
- `src/equities/study/training.py` — `target_fwd_5d` references replaced with `target` everywhere
- `scripts/research/smoke_phase2_variant.py` — new parameterized runner with `--horizon` / `--features` / `--variant` CLI args
- `docs/diagnostics/larger_universe_v1_horizon_diagnostic.md` — full diagnostic writeup with per-fold tables for all three configurations

### What's next

Phase 3 paused until Mike reviews the diagnostic and chooses an option. The tracker update (separate commit, this gate's standing process) reflects the paused state so partners who check in have current context. Phase 5 still gated on Phase 3 + 4; tracker stays partial until the study completes.

## 2026-05-11 (late evening) — Phase 3 authorized with revised spec (Option 3)

**Phase:** Phase 3 (Optuna hyperparameter tuning) authorized
**Branch:** `feat/larger-universe-v1-study`
**Commits at this gate:**
- `419a2eb` — spec(study): Phase 3 spec revision — 21d label, monthly rebalance, 21d embargo
- (session log + tracker committed separately as part of standing process)
**Status:** Spec revised. Phase 3 backgrounding next, after runtime sanity check.

### Option chosen

Mike authorized **Option 3** from the diagnostic — monthly rebalance + monthly label horizon. This pivots the spec to where the data shows signal lives. The earlier "Option 1: 21d label, weekly rebalance" was on the table but Mike chose the cleaner factor-research design.

Reasoning summary: monthly rebalance + monthly label is internally consistent (the model predicts what the portfolio actually realizes between rebalances), eliminates the threshold-based execution layer (no longer needed at monthly turnover), and matches academic factor-research convention. Tradeoff vs Option 1 is responsiveness (monthly vs weekly action on new information), but at the 21-day signal horizon this gap is small and the simpler architecture wins.

### Spec changes locked

| Parameter | Original | Revised |
|---|---|---|
| Label horizon | 5 trading days | **21 trading days** |
| Rebalance cadence | Weekly (Fri close) | **Monthly (last trading day)** |
| Rebalance threshold | 1.5pp | **REMOVED (full rebalance each month)** |
| CV embargo | 5 days | **21 days** (= label horizon) |
| Execution attribution | (unspecified) | **Close-to-close at next trading day after rebalance** |

Everything else unchanged: train/test/OOS splits, 5-fold expanding-window CV, XGBoost primary + ElasticNet sanity, 7.5% position cap, 30% sector cap, four benchmarks, 0.05% flat FeeModel, score-weighted continuous sizing, long-only.

Canonical spec doc at `docs/studies/larger_universe_v1/spec.md` (newly created).

### Phase 3 execution plan

- Full universe (1,963 tickers, not the smoke's 503 SP500 actives)
- All 38 features (full feature set)
- 200 Optuna trials per model (XGBoost and ElasticNet)
- 5-fold expanding-window CV with 21d embargo
- Cross-sectional Spearman IC as objective (mean across folds)
- Save best hyperparameters to `models/studies/larger_universe_v1/{xgboost,elasticnet}_best_params.json`
- Save full trial logs

### Runtime estimate (pre-background)

Variant B smoke timings (10 trials × 5 folds × 503 tickers × 38 features):
- XGBoost: 187 s = 18.7 s per trial = 3.7 s per fit
- ElasticNet: 348 s = 34.8 s per trial = 7.0 s per fit

Scaling to Phase 3 (200 trials × 5 folds × 1,963 tickers × 38 features):
- Row count scales ~3.9× (1963/503)
- Trial count scales 20× (200/10)
- Total compute scales ~78×

Rough estimates:
- XGBoost: 78 × 187s = 14,586s ≈ **4.0 hours**
- ElasticNet: 78 × 348s = 27,144s ≈ **7.5 hours**
- Sequential total: **~11.5 hours**

This is slightly over Mike's 10-hour budget threshold. Surfacing in the report for explicit go/no-go before backgrounding. If reduction is needed, options: drop ElasticNet to 100 trials (saves ~3.5h), or drop XGBoost to 150 trials (saves ~1h), or stagger the runs across two sessions.

### What to look for in Phase 3 results

- **Convergence pattern**: did the search plateau (good — confident best), or was it still exploring at trial 200 (signal could be higher with more trials)?
- **Fold 3 (val 2021-05 → 2022-05)**: the 2022 bear-market reversal. A robust strategy should still show some signal here even if reduced; a fragile one will be strongly negative. Report fold 3 separately for both models.
- **Std_ic per fold**: high std (>0.25 per fold) means the mean is driven by outlier dates, not a steady signal. A robust strategy should show std_ic 0.15-0.20 with consistent positive sign across folds.
- **Per-fold positive_rate**: aim for ≥0.55 ("wins" more than half the dates) on at least 4 of 5 folds.
- **Smoke vs full-universe IC**: smoke variant B was 0.019 (XGB) / 0.031 (ENet) on SP500 actives. Full universe should produce different numbers — likely higher due to increased cross-sectional dispersion, possibly messier on small caps. Don't anchor on smoke numbers.

### What's NOT happening tonight beyond Phase 3 backgrounding

- Phase 4 (portfolio construction + backtest) gated on Mike's review of Phase 3 results
- Phase 5 (validation + reporting) gated on Phase 4
- No tracker re-update during Phase 3 run; the pre-run tracker update reflects "Phase 3 running with revised spec"

## 2026-05-11 (final evening) — Phase 3 trial budget locked + backgrounded

**Phase:** Phase 3 backgrounding
**Branch:** `feat/larger-universe-v1-study`
**Commits at this gate:**
- `d690a5f` — docs(tracker): Phase 3 authorized — revised spec + Phase 3 runner
- Trial-budget tracker follow-up + session-log entry committed before backgrounding (this entry)
**Status:** Phase 3 backgrounded; expected completion ~7.5h.

### Decision: Option 2 — XGBoost 200 + ElasticNet 100, sequential

Mike approved the asymmetric trial budget after I surfaced the ~11.5h estimate for symmetric 200+200. The reasoning to lock in for the eventual writeup:

> Trial counts reflect search-space dimensionality, not equal budgets. XGBoost has 9 hyperparameters and benefits from 200 trials of TPE exploration. ElasticNet has 2 hyperparameters (alpha + l1_ratio) and TPE typically plateaus by trial 50-80 on a search space that small — trials 100-200 mostly resample near the best with diminishing information return. The complexity-asymmetric budget is methodologically defensible and reads cleanly in the eventual writeup.

Sequential execution (XGBoost first, then ElasticNet) rather than parallel — avoids the CPU-contention issue I flagged in the original Option 5. Wall-clock ~7.5h: XGB ~4h, ENet ~3.5h sequentially.

### Seven refinements baked into the Phase 3 runner

1. **Trial counts**: `--xgb-trials 200`, `--enet-trials 100` (separated CLI args)
2. **Sequential execution**: ENet runs only after XGB completes
3. **Surface "ENet > XGB" finding if it happens**: don't suppress as anomalous. If ENet at 100 trials beats XGB at 200, that's a signal the underlying alpha is mostly linear and Phase 5 should consider extended ENet exploration. Captured in the per-model best-IC comparison in the Phase 3 final report.
4. **Fold 3 separate reporting** (val 2021-05-10 → 2022-05-05): the 2022 bear-market regime reversal. Even if aggregate mean IC looks good, report Fold 3 separately for both models so we can see regime sensitivity magnitude.
5. **Convergence checkpoint every 25 trials**: log running-best mean IC at trials 25/50/75/.../200 so the Phase 3 report can show whether the search plateaued or was still finding improvements. If XGB is still actively improving at trial 175, 200 was undersized; if it plateaued by trial 80, 200 was generous.
6. **Fixed TPE sampler seed**: `seed=42` for both studies — full reproducibility of the trial sequence given the same code + features.parquet. Documented in best-params JSON output.
7. **Pathological-trial warning**: if any trial elapsed time exceeds 600 seconds (10 minutes — way out of the expected 4-25 s/trial range), log a WARNING with the trial number and params. Signals a hyperparameter region that caused a pathological convergence problem.

### Tracker update for the trial budget

A small follow-up commit updates the Phase 3 status paragraph in `docs/Project_State_Tracker.docx` to include:
- Trial counts (200 XGB + 100 ENet)
- Sequential execution rationale
- Backgrounded timestamp (2026-05-12 02:20 UTC)
- Expected completion timestamp (~2026-05-12 09:50 UTC)
- Seed for reproducibility

This means partners checking in mid-run see exact run state, not just "Phase 3 running".

### Expected outputs at completion

| Path | Content |
|---|---|
| `models/studies/larger_universe_v1/xgboost_best_params.json` | best params + best mean IC + per-fold breakdown for the winning XGB trial |
| `models/studies/larger_universe_v1/elasticnet_best_params.json` | same for ENet |
| `models/studies/larger_universe_v1/xgboost_study.json` | full trial-by-trial log with per-trial duration, fold stats, and Optuna state |
| `models/studies/larger_universe_v1/elasticnet_study.json` | same for ENet |
| `models/studies/larger_universe_v1/phase3_progress.log` | line-by-line stdout (tail this for live status) |

Intermediate persistence every 10 trials so a kill+resume is recoverable if needed.

### Phase 3 final report requirements (for the next session log entry)

When Phase 3 completes, the final report should cover:
- Best hyperparameters per model (full param set, not just the headline IC)
- Full per-fold breakdown: n_dates, mean_ic, std_ic, positive_rate for the winning trial of each model
- Fold 3 separate reporting for both models (regime sensitivity)
- Convergence pattern: running-best IC at 25-trial intervals; plateau vs still-improving
- Per-trial timing distribution: min, median, max, count of >10min trials
- Any pathological-trial flags surfaced during the run
- Total wall-clock per model and combined
- One-line comparison: did the full universe lift IC above the smoke's Variant B numbers (XGB 0.019, ENet 0.031)?
- Phase 4 readiness: any blocker that would warrant another pause before portfolio construction begins

## 2026-05-12 (morning) — Phase 3 complete

**Phase:** Phase 3 completion
**Branch:** `feat/larger-universe-v1-study`
**Wall-clock:** 5h 13m (XGB 4.35h + ENet 0.86h)
**Status:** Phase 3 complete. Phase 4 gated on Mike's review.

### Two pre-run bugs caught and fixed before the real run

1. **`inf` in `revenue_growth` and `eps_growth`** — 1,787 inf values from pct_change(0). XGBoost rejects inf; trial 0 crashed immediately. Fixed in `_compute_revenue_eps_growth` + global safety net before parquet save. Commit `71207d0`.
2. **`trial.study.best_trial` raises ValueError on first trial** instead of returning None. Wrapped in try/except. Commit `72ad42c`.

Both fixes pushed; 2-trial sanity check on full universe ran clean before the full 200+100 backgrounding.

### Headline numbers

| Model | Trials | Wall-clock | Best mean cross-sec IC | Best trial # |
|---|---|---|---|---|
| **XGBoost** (primary) | 200 | 4.35 h | **0.0282** | trial 150 |
| **ElasticNet** (sanity) | 100 | 0.86 h | **0.0144** | trial ~89 |

**XGBoost wins by ~2× margin.** Mike's "if ENet > XGB, surface as alpha-is-linear finding" rule fires in the opposite direction — alpha is non-linear and XGBoost is the right primary model.

### XGBoost best params + per-fold breakdown

```
max_depth=8 (hit upper search bound — wanted deeper trees)
learning_rate=0.0196   n_estimators=678   (slow careful learning)
subsample=0.642        colsample_bytree=0.971
min_child_weight=19    gamma=0.40
reg_alpha=0.572        reg_lambda=0.000182  (basically no L2)
```

| Fold | mean_ic | std_ic | positive_rate |
|---|---|---|---|
| 0 (2018-05→2019-05) | +0.0362 | 0.106 | 0.67 |
| 1 (2019-05→2020-05) | +0.0841 | 0.152 | 0.73 |
| 2 (2020-05→2021-05) | −0.0107 | 0.173 | 0.48 |
| 3 (2021-05→2022-05) | **−0.0210** | 0.126 | 0.50 |
| 4 (2022-05→2023-05) | +0.0527 | 0.142 | 0.63 |

### ElasticNet best params + per-fold breakdown

```
alpha=1.016e-05 (HIT THE SEARCH FLOOR — wanted lower regularization)
l1_ratio=0.7703
```

| Fold | mean_ic | std_ic | positive_rate |
|---|---|---|---|
| 0 | +0.0202 | 0.130 | 0.56 |
| 1 | +0.0596 | 0.125 | 0.66 |
| 2 | +0.0130 | 0.131 | 0.56 |
| 3 (2022) | **−0.0513** | 0.137 | 0.37 |
| 4 | +0.0303 | 0.178 | 0.50 |

### Fold 3 separate reporting (2022 regime shift)

| Model | Fold 3 mean_ic | Fold 3 positive_rate |
|---|---|---|
| XGBoost | **−0.021** | 0.50 |
| ElasticNet | **−0.051** | 0.37 |

XGBoost is materially more regime-robust. Tree-based interactions can re-route around the regime change; ElasticNet's fixed coefficient set can't. The 2022 reversal will be a major Phase 5 disclaimer point.

### Convergence pattern (XGBoost)

T25 → 0.0226 ; T50 → 0.0256 ; T75 → 0.0256 ; T100 → 0.0267 ; T125 → 0.0271 ; **T150 → 0.0282** ; T175 → 0.0282 ; T200 → 0.0282.

Plateau at trial 150; last 50 trials added zero. **200 was slightly oversized** for this search space — 150 would have been enough. Useful intel for future studies.

ElasticNet's convergence trace is uninformative — running best was 0.0144 from very early on because **86 of 100 trials returned NaN** (constant-prediction collapse at high alpha). TPE kept probing failed regions because NaN-as-failure provides no gradient.

### Per-trial timing distribution

- **XGBoost**: min 18.3s, median 81.9s, max 124.3s. **0 trials >10min** (no pathological flags).
- **ElasticNet**: bimodal — 17.5s constant-collapse trials vs 100-230s real-fit trials. **0 trials >10min**.

### Smoke vs full-universe (sanity check)

| Stage | XGBoost | ElasticNet |
|---|---|---|
| Variant B smoke (SP500 actives) | 0.019 | 0.031 |
| Phase 3 (full 2,122-ticker universe) | **0.0282** | 0.0144 |

XGBoost IMPROVED on full universe (more dispersion = more signal). ElasticNet DEGRADED (linear model diluted by small-cap noise). Further evidence the alpha is non-linear.

### Phase 4 readiness

**No blockers identified.** Best hyperparameters saved to:
- `models/studies/larger_universe_v1/xgboost_best_params.json`
- `models/studies/larger_universe_v1/elasticnet_best_params.json`

The full Phase 3 results writeup is at `docs/diagnostics/larger_universe_v1_phase3_results.md` including the IR estimation back-of-envelope (annualized IR plausibly 1.0-2.0 from IC=0.028 across ~18-24K cross-sectional obs/year).

### Open questions for Mike at the Phase 3 → Phase 4 gate

1. Is **0.0282 mean cross-sectional IC** enough headline alpha to justify Phase 4 work? Plausible annualized IR 1.0-2.0 — meaningful but not extraordinary.
2. Should we do a focused ElasticNet re-run with `alpha ∈ [1e-7, 1e-2]` and NaN-replacement-with-zero before Phase 4? Low priority since XGB is the primary, but might lift ENet from 0.014 → 0.018-0.022 with ~1h of compute. Useful comparison surface for the eventual writeup.
3. Phase 4 scope: full-feature-importance analysis included, or saved for Phase 5?
4. Phase 4 output naming: `models/studies/larger_universe_v1/backtests/{xgboost,elasticnet}/` per the original spec, or a different layout?

### Files produced

| Path | Content |
|---|---|
| `docs/diagnostics/larger_universe_v1_phase3_results.md` | Full results writeup (this entry's source of truth) |
| `models/studies/larger_universe_v1/xgboost_best_params.json` | Best XGB params + per-fold winning-trial breakdown |
| `models/studies/larger_universe_v1/elasticnet_best_params.json` | Best ENet params + per-fold winning-trial breakdown |
| `models/studies/larger_universe_v1/xgboost_study.json` | Full 200-trial log with per-trial duration + fold attrs |
| `models/studies/larger_universe_v1/elasticnet_study.json` | Full 100-trial log |
| `models/studies/larger_universe_v1/phase3_progress.log` | Line-by-line stdout |

### NOT proceeding to Phase 4

Per spec: stop at gate, report, wait for Mike's review.

## 2026-05-12 (morning) — Architectural decision: dashboard contract v1 proposal

**Phase:** Pre-Phase-4 (architectural)
**Branch:** `feat/larger-universe-v1-study`
**Commits at this gate:**
- (this session log entry + the proposal doc committed together as `docs(architecture): propose dashboard contract v1 for studies`)
**Status:** Proposal written; awaiting Mike + partners' review before Phase 4 spec is finalized.

### Decision (Mike, 2026-05-12)

Investigated how the current Streamlit dashboard ingests study results. Finding: the dashboard is tightly coupled to the legacy SQLite-Optuna lineage and the `dashboard_results/<label>/` file convention. Even if Phase 3 output were dropped into the right paths, the legacy tabs' UI (composite-weight sliders, ATR controls, regime traffic-light) doesn't apply to a score-weighted XGBoost monthly-rebalance study.

Mike chose **Option (b)**: design a generic results contract, build new universal dashboard tabs for contract-conformant studies, keep legacy studies on legacy tabs unchanged. Larger Universe v1 is the first study in the new system. Future studies use the new contract. No backfill of the three promoted studies (deferred indefinitely).

### What was produced this turn

A proposal-only deliverable at `docs/architecture/dashboard_contract_v1.md` (~400 lines) covering:

1. **Current dashboard tab inventory** with universal-vs-family-specific classification of each of the 8 existing tabs
2. **Recommended new tab structure** (8 tabs, same count, parallel naming for familiarity): Performance, Holdings, Trades, Risk & Behavior, Model Diagnostics (new), Market Context, Tuning History, Glossary. Optional 9th: Sensitivity / Walk-forward when `walk_forward.parquet` is present.
3. **Data contract v1 specification** — files at `models/studies/<study_name>/contract_v1/`:
   - meta.json (required) — study identity, family, models, windows, constraints, summary metrics
   - portfolio.parquet (required) — NAV time series + benchmark NAVs
   - holdings.parquet (required) — long-format date × model × ticker × weight
   - trades.parquet (required) — execution log with fees
   - scores.parquet (ML-required) — per-(date, model, ticker) predictions + ranks
   - trial_log.parquet (tuning-required) — Optuna trials in tabular form (replaces SQLite for contract studies)
   - feature_importance.parquet (ML-required)
   - walk_forward.parquet (optional, Phase 5 produces)
   - regime_attribution.parquet (optional)
4. **Schema versioning strategy** — `meta.json.schema_version: "v1"`; additive changes don't bump; breaking changes bump to "v2" with parallel renderers in the dashboard
5. **Sidebar UX** — split into "Legacy studies" and "Contract-conformant studies" sections; clicking either loads the appropriate tab set
6. **Implementation phasing** — Phase 4 produces contract_v1/, Phase 4.5 implements the new dashboard tabs, Phase 5 adds walk_forward/regime_attribution to the same contract location, R2 sync gets a parallel walker
7. **Six open questions** flagged for the review pass

### What's NOT in this turn

- No dashboard code modified
- No Phase 4 work started
- No tracker update (per Mike: next tracker update is Phase 4 completion or partners-need-to-know-now, whichever comes first)
- The legacy three promoted studies are untouched (Mike said no backfill)

### Next steps after the proposal is reviewed/approved

1. Modify the Phase 4 spec to produce contract_v1/ artifacts for Larger Universe v1
2. Confirm the modified Phase 4 spec
3. Run Phase 4 (produces contract-conformant artifacts)
4. Phase 4.5 — implement the new universal dashboard tabs
5. Phase 5 — walk-forward + OOS; auto-appears on the new dashboard via the same contract location

## 2026-05-12 (mid-morning) — Contract approved + Phase 4 spec drafted

**Phase:** Contract v1 approval → Phase 4 spec drafting
**Branch:** `feat/larger-universe-v1-study`
**Status:** Contract approved, Phase 4 spec drafted. Awaiting Mike's review of the modified Phase 4 spec before any Phase 4 code runs.

### Contract v1 — approved with answers to the six open questions

Mike's review resolved each question:

1. **scores.parquet size cap: 1M rows**, with `scores_sampled.parquet` required above that. Larger Universe v1 ~140K rows, under cap.
2. **Benchmarks: strict declaration via `meta.json.benchmarks`** — no auto-add.
3. **`promoted` flag: match legacy semantics** — default false, manual flip after explicit promotion decision.
4. **Holdings tab default: always-show-latest with date picker** — "latest" computed dynamically.
5. **Model Diagnostics tab visibility: data-driven** — render iff both `scores.parquet` AND `feature_importance.parquet` exist for the study. No explicit flag.
6. **Multi-model studies: render primary by default**, sanity_check via sidebar dropdown. No automatic compare overlay.

Plus: **portfolio.parquet switches to long-format**, with benchmarks split into their own `benchmarks.parquet`. This was a contract-level decision (not Larger-Universe-v1-specific) since it affects all future studies' schema.

Contract doc updated at `docs/architecture/dashboard_contract_v1.md` with status `APPROVED 2026-05-12`. Open questions section replaced with "Decisions (resolved 2026-05-12)".

### Phase 4 spec drafted at `docs/studies/larger_universe_v1/phase4_spec.md`

Three Larger-Universe-v1-specific implementation decisions locked into the spec:

1. **Long-format portfolio.parquet** (one row per (date, model)) + separate `benchmarks.parquet` (one row per (date, benchmark))
2. **SHAP feature importance with gain-based fallback.** Use `xgboost.Booster.predict(pred_contribs=True)` for fast tree-SHAP on a 10K-row sample of the test window. If SHAP wall-clock exceeds 10 minutes, abort and use gain-based, document the fallback in `meta.json.notes`.
3. **Per-rebalance-date holdings only** — no daily interpolation. Dashboard handles inter-rebalance constancy on its own.

Spec includes:

- Full input/output table (Phase 3 outputs → contract_v1/ outputs)
- Concrete meta.json content (model roles, benchmarks list, schema_version, promoted:false)
- Score-to-weights algorithm (softmax with temperature T=0.1, iterative 7.5%/30% cap enforcement, normalize to sum=1.0)
- Backtest execution flow (monthly rebalance loop with forced-exit on delisting)
- Benchmark construction (SPY/RSP/IWM via Finnhub fetch; EW-SP1500 custom monthly-rebalanced equal-weight)
- Estimated wall-clock: ~35-50 minutes (no backgrounding needed; most compute happened in Phase 3)
- 5 open items for Mike's review (softmax T=0.1, EW-SP1500 active-on-date definition, SHAP sample size, forced-exit accounting, OOS slice handling)

### What this turn did NOT do

- No dashboard code modified
- No Phase 4 implementation started
- No tracker update (next is Phase 4 completion)
- No Phase 4.5 dashboard work
- No legacy study backfill

### What's pending Mike's review

The modified Phase 4 spec at `docs/studies/larger_universe_v1/phase4_spec.md`. Five specific items flagged in the spec's "Open items for review" section need confirmation before code runs.

## 2026-05-12 (afternoon) — Phase 4 complete

**Phase:** Phase 4 (portfolio construction + backtest)
**Branch:** `feat/larger-universe-v1-study`
**Wall-clock:** ~3 minutes (vs the 35-50 min estimate — final-training/backtest is much cheaper than 200-trial tuning)
**Status:** Phase 4 done. Phase 5 gated on Mike's review. Phase 4.5 (dashboard work) gated separately on Mike's go-ahead.

### Five Phase-4-gate decisions applied (Mike's directives)

1. **Score-to-weights: rank_top_n=30, individual cap 7.5%, sector cap 30%.** Locked into `meta.json.portfolio_construction`. No softmax. Reasoning: at IC ~0.028, score-magnitude differences within the top decile carry more noise than signal; rank-based is more robust ex-ante.
2. **EW-SP1500 active-on-date**: `status=="active"` OR (`status=="removed"` AND `removed_at > D`). Includes historically-active-but-since-delisted in their active periods. Confirmed.
3. **SHAP feature importance with gain fallback**: ran cleanly in 9.8s on 10K sample via `xgboost.Booster.predict(pred_contribs=True)`. `meta.json.feature_importance_method = "shap_tree"`.
4. **Forced-exit accounting**: ticker delists mid-month → close at delisting-day close, weight → cash until next rebalance. 58 forced exits for XGB, 41 for ENet in trades.parquet (rows with `reason="delisting_truncation"`).
5. **OOS slice handling**: Phase 4 produces NAV through snapshot end. Phase 5 evaluates OOS separately. Explicit date ranges in `meta.json.windows`.

### Headline results

**Test period (2023-05-12 → 2025-12-31, 650 trading days):**

| Model | Total return | CAGR | Excess vs SPY | Max DD | SPY Max DD |
|---|---|---|---|---|---|
| **XGBoost** (primary) | **+78.3%** | **+25.14%** | **+3.52pp** | **−33.5%** | −19.0% |
| **ElasticNet** (sanity) | **+150.9%** | **+42.85%** | **+21.23pp** | **−37.5%** | −19.0% |

**OOS slice (2026-01-01 → 2026-05-11, 89 trading days):**

| Model | Total return | CAGR (annualized) | Excess vs SPY | Max DD | SPY Max DD |
|---|---|---|---|---|---|
| XGBoost | +16.1% | +52.6% | +27.6pp | −12.2% | −9.1% |
| ElasticNet | +59.7% | +276.7% | +251.6pp | −9.8% | −9.1% |

OOS CAGRs are statistically thin (only ~4 months) — the annualization extrapolation makes them eye-popping but the absolute returns and drawdowns are the meaningful numbers there.

### Success criteria check (preliminary — full evaluation in Phase 5)

| Criterion | XGBoost (test) | ElasticNet (test) |
|---|---|---|
| Excess CAGR vs SPY > 0 | ✅ +3.52pp | ✅ +21.23pp |
| Max DD ≤ 1.5× SPY's DD (= −28.5%) | **❌ −33.5%** | **❌ −37.5%** |
| No single ticker > 25% of total alpha | TBD Phase 5 (requires attribution analysis) | TBD Phase 5 |
| Soft: 12-month rolling win rate ≥ 60% | TBD Phase 5 | TBD Phase 5 |

**Both models fail the drawdown constraint in the test window.** The 2022-style market regime risk identified in Phase 3's fold-3 analysis materialized in real backtest drawdowns. Mike's review will weigh: is the excess CAGR signal real enough to redesign for tighter risk control in v2, or is this a "study didn't pass success criteria — pause and rethink" moment?

### Critical finding to surface: ElasticNet > XGBoost despite lower CV IC

Mike's standing rule: "If ElasticNet at 100 trials produces a meaningfully better mean cross-sectional IC than XGBoost at 200 trials, treat that as a finding to surface." The original finding was the opposite (XGB IC 0.0282 vs ENet 0.0144). **But in the actual portfolio backtest, ElasticNet vastly outperforms XGBoost** (+21pp vs +3.5pp excess CAGR in test).

Feature importance explains the gap:
- **XGBoost's top features are macro-dominant**: SAHM (recession indicator), NFCI (credit conditions), VIX, BAA spread, USD index. These vary by DATE, not by TICKER. XGBoost is essentially doing market-timing — predicting that returns will be uniformly higher/lower based on the macro state — which doesn't translate to stock-ranking within a date.
- **ElasticNet's top features are price/trend-dominant**: price_vs_ma200, ma50_vs_ma200, price_vs_ma50, vol_63d. These vary by TICKER on a given date. Stock-ranking signal.
- ElasticNet's #7 feature is `beta_missing` (the imputer's missingness indicator) — a binary flag for "ticker has insufficient history for 36mo beta". This is a legitimate informational signal (tickers with short history often recent IPOs) that ENet correctly exploited.

**Implication for cross-sectional alpha measurement**: high overall Spearman IC doesn't translate to good top-N portfolio picks if the model's edge is at the middle/bottom of the distribution rather than the top. The IC measures the slope of rank correlation across the whole cross-section; the top-N strategy specifically exploits the top end. A linear model with strong tail discrimination can beat a tree model with smoother middle-of-distribution signal.

**Worth Phase 5 deeper investigation**: rank-by-decile return analysis would show whether each model's score signal is monotonic across deciles or concentrated at the extremes.

### Contract v1 artifacts produced

All at `models/studies/larger_universe_v1/contract_v1/`:

| File | Size | Rows |
|---|---|---|
| `meta.json` | 4.1 KB | — |
| `portfolio.parquet` | 29 KB | 1,478 (daily × 2 models) |
| `benchmarks.parquet` | 39 KB | 2,992 (daily × 4 benchmarks) |
| `holdings.parquet` | 14 KB | 2,220 (37 rebalances × ~30 positions × 2 models) |
| `trades.parquet` | 99 KB | 3,411 |
| `scores.parquet` | 2.0 MB | 118,464 |
| `trial_log.parquet` | 30 KB | 300 |
| `feature_importance.parquet` | 7.0 KB | 144 |

Stays well under the 1M-row scores cap.

### Concentration check (informal — full Phase 5 attribution still needed)

**Sector cap working**: max sector weight on latest rebalance is 20.0% (XGB Food Products) and 23.3% (ENet Health Care). Both well under the 30% cap.

**Repeat-holding patterns**:
- XGBoost most-held tickers: SEDG (54% of rebalances), CLSK (51%), BHF (49%)
- ElasticNet most-held: DBD (81% of rebalances), CLSK (68%), FTRE (59%)
- ElasticNet shows more "favorite stocks" persistence; XGBoost rotates more

ENet's 81% repeat-rate on DBD (Diebold Nixdorf) is notable — single-name dependency risk worth checking against the "25% of total alpha" constraint when Phase 5 runs.

### What's NOT done in Phase 4 (per spec)

- Walk-forward analysis (Phase 5)
- Regime attribution (Phase 5)
- Per-ticker alpha attribution / 25% constraint check (Phase 5)
- 12-month rolling win rate check (Phase 5)
- Final OOS reporting / disclaimers (Phase 5)
- Tracker update (next natural is Phase 4.5 transition per Mike's standing rule)
- Dashboard code modifications (Phase 4.5)
- Promotion decision

### Open questions for Mike at the Phase 4 → Phase 5 gate

1. **Drawdown constraint failure**: do we proceed to Phase 5 (full validation + writeup) accepting that this v1 study doesn't pass the drawdown criterion, OR pause and redesign portfolio construction (smaller N, vol-targeting, etc.) before continuing?
2. **ENet > XGBoost finding**: Phase 5 deeper investigation (decile-return analysis) to confirm the "macro-timing vs stock-ranking" explanation, OR accept the finding and move on?
3. **Promotion candidacy**: both models show positive excess CAGR but neither meets the drawdown constraint. Plausible that NEITHER passes the promotion gate even after Phase 5. Worth your call on whether the study still warrants the Phase 4.5 dashboard investment to surface what was learned, OR shelve it.
4. **Repeat-holding patterns**: ENet's 81%-of-rebalances DBD weight (3.33% individual but persistent) warrants the 25%-of-alpha check in Phase 5. Worth flagging now in case you want to pivot the spec.
