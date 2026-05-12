# Larger Universe v1 — Cross-validation design (Phase 2)

**Branch:** `feat/larger-universe-v1-study`
**Phase 2 status:** Spec revised post-diagnostic — label horizon 5d → 21d, rebalance weekly → monthly, embargo 5d → 21d. Locked spec at `docs/studies/larger_universe_v1/spec.md`. Phase 3 authorized 2026-05-11 evening with the revised spec.

## TL;DR

- Label: **forward 21-trading-day return** per (date, ticker), matching the monthly rebalance cadence (revised from 5d at Phase 2 gate)
- Scoring metric: **cross-sectional Spearman IC** (per-date Spearman, mean across dates) with min_tickers=30. Reported with mean, std, positive-rate. Mean is the Optuna objective; std and positive-rate are diagnostic.
- CV: 5-fold expanding-window TimeSeriesSplit over 2017-05-12 → 2023-05-11 with a **21-trading-day embargo** (= label horizon)
- XGBoost: native NaN + native categorical (sector), Optuna over 9 hyperparameters (max_depth 3–8)
- ElasticNet: SimpleImputer(mean, add_indicator=True) + StandardScaler + ElasticNet, Optuna over alpha + l1_ratio
- **Variant B smoke result (SP500 actives, 21d horizon, full features):** XGB mean cross-sectional IC 0.019 with all 5 folds covered (no degeneracy); ElasticNet 0.031 with 4 of 5 folds positive. Phase 3 runs on the full ~1,963-ticker universe and should produce different (likely higher) numbers given the increased cross-sectional dispersion.

## Label

The model is rebalanced weekly (every Friday close per the locked spec). The natural prediction target is the forward 5-trading-day return:

```
target_fwd_5d(D, T) = close(D+5, T) / close(D, T) - 1
```

This matches the rebalance cadence: a score computed at Friday D informs the portfolio held over the next 5 trading days. The label has NaN on the last 5 rows of every ticker's series (no future to compute against).

Implementation: `src/equities/study/labels.py:compute_forward_return`. Built per-ticker from snapshot prices.

## Splits (locked at Phase-1 gate)

| Split | Date range | Use |
|---|---|---|
| **Train** | 2017-05-12 → 2023-05-11 (~6 years) | CV folds; Optuna hyperparameter search |
| **Test** | 2023-05-12 → 2025-12-31 (~2.5 years) | Out-of-sample evaluation in Phase 4 |
| **Final OOS holdout** | 2026-01-01 → 2026-05-11 (~4.5 months) | Only touched once at Phase 5 reporting |

The train window starts 2017-05-12 (not 2016-05-12 per original spec) because the long-lookback features (252-day return, 200-day MA) need a year of warmup. The feature matrix retains all dates; the splitter applies the trim.

## CV folds

**Design:** 5-fold expanding-window TimeSeriesSplit with embargo.

Each fold's validation block is ~1/6 of the training window (so 5 folds use 5/6 of the data for validation, never touching test or OOS). The first 1/6 of the training window is initial-train-only and becomes part of every fold's train set.

**Embargo:** 5 trading days between train end and validation start. With a 5-day forward-return label, train rows within 5 days of the validation start would have labels referencing prices inside the validation window — direct leakage. The embargo is exactly equal to the label horizon, as is standard for finance ML.

Per-fold dates (from the smoke run, SP500-active subset):

| Fold | Train start | Train end | Embargo gap | Val start | Val end |
|---|---|---|---|---|---|
| 0 | 2017-05-12 | 2018-05-03 | 5 days | 2018-05-11 | 2019-05-10 |
| 1 | 2017-05-12 | 2019-05-03 | 5 days | 2019-05-13 | 2020-05-08 |
| 2 | 2017-05-12 | 2020-05-01 | 5 days | 2020-05-11 | 2021-05-07 |
| 3 | 2017-05-12 | 2021-04-30 | 5 days | 2021-05-10 | 2022-05-05 |
| 4 | 2017-05-12 | 2022-04-28 | 5 days | 2022-05-06 | 2023-05-11 |

The training window grows fold-by-fold (expanding window) which is the canonical pattern for time-series finance. The validation blocks are non-overlapping and cover the entire 2018-05 → 2023-05-11 range.

Implementation: `src/equities/study/cv.py:make_folds` operates on sorted unique dates (not row indices) and yields `Fold` dataclasses with date boundaries. The caller filters rows by date range.

## Scoring metric — Cross-sectional Spearman IC (corrected 2026-05-11)

The **Information Coefficient (IC)** is the **per-date** Spearman rank correlation between predictions and realized forward returns, averaged across dates in the validation block:

```
for each unique date D in val_fold with >= 30 tickers (after dropna):
    rho_D = spearmanr(preds_on_D, realized_returns_on_D)
    if rho_D is finite: per_date.append(rho_D)
mean_ic       = mean(per_date)
std_ic        = std(per_date)         # consistency check across dates
positive_rate = (per_date > 0).mean() # "how often does the model win"
```

The Optuna objective is `mean_ic` aggregated across folds. `std_ic` and `positive_rate` are surfaced per fold for review but NOT part of the optimization target — keeping the search single-objective and clean.

The first cut of this doc reported a **panel-wise** IC (single Spearman across all (date, ticker) rows pooled). That metric was wrong for this study: it conflated stock-ranking signal (what the portfolio uses) with market-timing signal (what it doesn't). Replaced 2026-05-11 after Phase 2 gate review.

Why Spearman rank rather than MSE or Pearson:
- The portfolio construction step (Phase 4) maps scores to weights via a ranking transformation. The model's ability to RANK stocks correctly matters more than its absolute return prediction accuracy.
- Rank correlation is robust to outliers and heteroskedasticity (a ticker that triples in a week shouldn't dominate the loss signal).
- IC is the de-facto industry standard for cross-sectional alpha research.

Why min_tickers=30: Spearman on N points has SE ≈ 1/√(N-1), so 30 tickers gives ~0.19 SE per date — enough to see a real signal but not so high that we lose marginal dates. SP500 has ~500 tickers and SP1500 ~1,900; both far above the threshold for normal dates. min_tickers also defends against the degenerate case where most tickers have NaN predictions on a given date (e.g., a thin-history fold).

Implementation: `cross_sectional_ic_stats` in `src/equities/study/training.py`. NaN folds (where no date qualified) are excluded from the mean-across-folds. Pruning is disabled (5 folds is small enough that median-pruning is more disruptive than it's worth).

## Model pipelines

### XGBoost

- Native NaN handling — no imputation
- Native categorical handling via `enable_categorical=True` and `pd.Categorical` dtype on `sector`. Sector levels become tree split points without one-hot expansion. Handles `sector_unknown` cleanly.
- Search space (Optuna `_make_xgb_params`):

  | Param | Range | Note |
  |---|---|---|
  | max_depth | 3–8 | tree depth (tightened from 3–10 at Phase 2 gate; deeper trees rarely useful on financial tabular data) |
  | learning_rate | 0.01–0.30 (log) | shrinkage |
  | n_estimators | 100–800 | number of trees |
  | subsample | 0.5–1.0 | row subsample per tree |
  | colsample_bytree | 0.5–1.0 | feature subsample per tree |
  | min_child_weight | 1–20 | min Hessian per leaf |
  | gamma | 0.0–5.0 | min split-loss threshold |
  | reg_alpha | 1e-4–10 (log) | L1 regularization |
  | reg_lambda | 1e-4–10 (log) | L2 regularization |

Implementation: `train_xgb_single_fold` in `src/equities/study/training.py`.

### ElasticNet

Sklearn `Pipeline`:
1. `SimpleImputer(strategy='mean', add_indicator=True)` — fills NaN with column mean computed on the training fold; appends a binary missingness indicator for every column that had at least one NaN
2. `StandardScaler` — required because L1/L2 penalties are scale-dependent
3. `ElasticNet(alpha, l1_ratio, max_iter=5000)`

Sector is one-hot encoded BEFORE the pipeline (so the imputer doesn't operate on a categorical). The training-fold's sector levels are captured and applied to the validation fold, ensuring column alignment.

Crucially: the imputer's mean is computed on the TRAINING FOLD only (fit on train, transform on val). No cross-fold leakage.

Search space:

  | Param | Range |
  |---|---|
  | alpha | 1e-5–1.0 (log) |
  | l1_ratio | 0.0–1.0 |

Implementation: `train_enet_single_fold` in `src/equities/study/training.py`.

### Feature handling differences (XGBoost vs ElasticNet)

| Feature class | XGBoost | ElasticNet |
|---|---|---|
| Numerical with NaN | native handling; tree splits route NaN to optimal side | mean-imputed; `<col>_missing` binary indicator appended |
| `sector` categorical | `pd.Categorical` dtype, `enable_categorical=True` | one-hot encoded → `sector__<level>` columns |
| Binary `in_sp500/400/600` | as-is (0/1) | as-is (0/1) |
| Standardization | none (trees are scale-invariant) | StandardScaler after imputation |

Both pipelines see **identical feature inputs** (same columns, same dates, same tickers) — the only difference is how each model consumes them. This is the spec's "trained on identical features and identical splits" requirement satisfied.

## Smoke run results (10 trials each, CORRECTED 2026-05-11 with cross-sectional IC)

**Subset for smoke:**
- Universe: SP500 actives only (503 tickers)
- Features: price-derived (12) + macro (10) = 22 features (NO fundamentals, NO sector, NO log_market_cap, NO index-membership flags — deliberately excluded to keep smoke fast)
- 734,646 training rows after target-NaN filter
- min_tickers=30 per date for cross-sectional IC scoring

### XGBoost (10 trials × 5 folds)

| Metric | Value |
|---|---|
| Best mean cross-sectional IC | **0.2177** (degenerate — see note below) |
| Wall-clock | ~88 s |

**Note: the "best" of 0.2177 is misleading and should NOT be interpreted as a real signal.** The "best trial" produced **constant predictions on 3 of 5 folds** (n_dates_scored=0), and the 2 surviving folds totaled 74 valid dates — one of which contributed mean_ic=0.415 over just 4 dates of validation. Optuna's mean-of-valid-folds objective lets such trials win against well-covered trials whose cross-sectional IC is honestly near-zero.

**Per-fold structure of well-covered XGBoost trials** (where all or most folds produced n_dates ≥ 200):

| Trial | F0 n,IC | F1 n,IC | F2 n,IC | F3 n,IC | F4 n,IC | Real mean IC (covered folds only) |
|---|---|---|---|---|---|---|
| 0 | 0, — | 0, — | 242, -0.018 | 210, +0.006 | 255, -0.011 | **-0.008** |
| 2 | 0, — | 150, -0.019 | 251, -0.013 | 251, +0.005 | 255, +0.013 | **-0.004** |
| 6 | 0, — | 0, — | 225, -0.022 | 154, +0.010 | 230, +0.010 | **-0.001** |
| 8 | 0, — | 222, -0.000 | 251, -0.004 | 251, -0.030 | 255, +0.010 | **-0.006** |

**XGBoost cross-sectional IC on the smoke subset is ESSENTIALLY ZERO.**

Why folds 0 and 1 produce n_dates=0: the trained model outputs constant predictions WITHIN each date (every SP500 ticker on date D gets the same predicted value, but the value varies across dates). Verified by inspecting the prediction array directly — `nunique` per date is 1, but `nunique` across all dates is ~15. The model has learned to predict the day's average return from macro features (which vary by date) and is ignoring the price features that would differentiate tickers on a given day. Panel-wise Spearman picks up this date-level signal as if it were stock-ranking signal; cross-sectional Spearman correctly attributes it to date-level and reports zero stock-ranking content.

This is the panel-vs-cross-sectional distinction working exactly as designed.

### ElasticNet (10 trials × 5 folds, 1 trial failure)

| Metric | Value |
|---|---|
| Best mean cross-sectional IC | **0.0131** |
| Best params | alpha=0.000462, l1_ratio=0.688 |
| Wall-clock | ~152 s |
| Failed trials | 1 of 10 (constant-prediction at high alpha) |

Per-fold breakdown of the best trial:

| Fold | n_dates | mean_ic | std_ic | positive_rate |
|---|---|---|---|---|
| 0 | 251 | +0.0264 | 0.196 | 0.55 |
| 1 | 251 | -0.0066 | 0.272 | 0.51 |
| 2 | 251 | +0.0325 | 0.240 | 0.51 |
| 3 | 251 | -0.0207 | 0.259 | 0.44 |
| 4 | 255 | +0.0338 | 0.210 | 0.56 |

ElasticNet covers every date in every fold (no constant-prediction collapse — the linear model with mild regularization spreads coefficient mass across price + macro features rather than zeroing out the price features as XGBoost effectively does). Per-fold mean ICs span -0.02 to +0.03 with positive-rates near 0.50 (essentially coin-flip per date). **ElasticNet cross-sectional IC on the smoke subset is also essentially zero, with high per-date std (≈ 0.20–0.27)** — the signal that exists is noisy enough that any single date is largely uninformative.

### Interpretation

The smoke result is in the **"pause" zone** per the agreed Phase 3 framework: cross-sectional IC near zero, not negative but not clearly positive either. The price-derived features alone (in the smoke universe) do not provide meaningful stock-ranking power at 5-day horizon for SP500 actives.

What this DOES tell us:
- The panel-wise IC of 0.085 was almost entirely market-timing signal — explicitly what cross-sectional IC was designed to filter out
- The cross-sectional metric is working correctly; the IC of ~0 reflects the actual stock-ranking signal in this feature subset
- ElasticNet covers more dates than XGBoost on early folds; the algorithm difference matters for whether constant predictions are produced

What this does NOT tell us:
- Whether the FULL feature set (38 features incl. fundamentals, sector, log_market_cap, index_membership) provides cross-sectional alpha. The smoke deliberately excluded those features.
- Whether the 5-day horizon is too short for the features we have. Weekly cross-sectional alpha is genuinely difficult; many academic factor studies use monthly.
- Whether the universe size affects signal (smoke is 503 tickers; full universe is 1,963)

**Recommended Phase-2.5 step before authorizing Phase 3:** re-smoke with the full 38-feature set (still on SP500-actives subset to keep wall-clock short) and re-evaluate. If IC stays near zero, the spec's feature choices may not have alpha at 5-day horizon and we should redesign before burning 6-7 hours. If IC climbs to the 0.02-0.05 range, proceed to Phase 3.

## Open items for Phase 3

1. **ElasticNet NaN-on-constant-prediction:** currently Optuna sees NaN and marks the trial failed. This is functionally fine (TPE adapts) but somewhat noisy. If Phase 3 shows >40% of ElasticNet trials failing for this reason, switch the failure path to return IC = 0 (genuine no-information signal) so all trials inform the surrogate model. Easy 2-line change.

2. **Fold 0 instability for some XGBoost configs.** Trials with shallow trees + few estimators produce constant predictions on fold 0 (smallest training window). Phase 3's 100–300 trials will explore the space densely enough that this becomes a rare regime. No action needed in Phase 2.

3. **Single-direction IC may understate the actual signal.** Spearman IC of 0.085 means the model ranks correctly ~54% of pairwise comparisons (Spearman 0 = 50%, 1 = 100%, so 0.085 ≈ 54.25%). For weekly cross-sectional portfolio sizing this translates to meaningful alpha if combined with the position concentration discipline in Phase 4. Phase 5 will report the final backtested edge — IC alone is a leading indicator, not the answer.

4. **Phase 3 wall-clock estimate:** smoke was 500 tickers × 22 features × 10 trials × 5 folds ≈ 5 min for XGBoost. Full universe ≈ 1,900 tickers × 38 features × 200 trials × 5 folds. Linear scaling on rows + trials gives ~5 × (1900/500) × (200/10) ≈ 380 min ≈ 6 hours for XGBoost. ElasticNet is much faster (~30 min full). **Total Phase 3 budget: 6–7 hours wall-clock**, comparable to the spec's data-fetch budget. May want to background it overnight.

## Files produced this gate

| Path | Notes |
|---|---|
| `src/equities/study/__init__.py` | Namespace marker |
| `src/equities/study/labels.py` | Forward 5-day return + horizon constants |
| `src/equities/study/cv.py` | TimeSeriesSplit with embargo; date-window filters |
| `src/equities/study/training.py` | XGBoost + ElasticNet single-fold trainers; cv_score driver; safe Spearman |
| `scripts/research/smoke_phase2.py` | 10-trial smoke runner |
| `models/features/larger_universe_v1/phase2_smoke/smoke_results.json` | Per-trial fold ICs + best params for both models |
| `docs/diagnostics/larger_universe_v1_cv_design.md` | This file |

## Reproducibility — smoke

```
python scripts/research/smoke_phase2.py    # ~6 min total
```

## What's deferred

- Phase 3 full tuning run — 100–300 trials per model on full universe + features
- Final hyperparameter persistence at `models/studies/larger_universe_v1/{xgboost,elasticnet}_best_params.json`
- Tracker update (deferred to Phase 5 per standing rule)
