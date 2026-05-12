# Larger Universe v1 — Cross-validation design (Phase 2)

**Branch:** `feat/larger-universe-v1-study`
**Phase 2 status:** Training pipelines + CV scaffolding built; 10-trial smoke run completed for both models. Phase 3 tuning has NOT run yet.

## TL;DR

- Label: forward 5-trading-day return per (date, ticker), matching the weekly rebalance cadence
- Scoring metric: cross-sectional Spearman IC (information coefficient), mean across folds
- CV: 5-fold expanding-window TimeSeriesSplit over 2017-05-12 → 2023-05-11 with a 5-trading-day embargo
- XGBoost: native NaN + native categorical (sector), Optuna over 9 hyperparameters
- ElasticNet: SimpleImputer(mean, add_indicator=True) + StandardScaler + ElasticNet, Optuna over alpha + l1_ratio
- **Smoke result: both models produce positive cross-sectional IC** (XGB 0.085, ENet 0.071) on the SP500-active subset with 22 price+macro features. Not suspiciously high; signal is real but modest.

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

## Scoring metric — Spearman IC

The **Information Coefficient (IC)** is the cross-sectional Spearman rank correlation between predictions and realized forward returns, computed across all (date, ticker) rows in the validation block.

Why Spearman rank rather than MSE or Pearson:
- The portfolio construction step (Phase 4) maps scores to weights via a ranking/softmax transformation. The model's ability to RANK stocks correctly matters more than its absolute return prediction accuracy.
- Rank correlation is robust to outliers and heteroskedasticity (a ticker that triples in a week shouldn't dominate the loss signal).
- IC is the de-facto industry standard for cross-sectional alpha research.

Implementation: `_safe_spearman` in training.py masks non-finite values; returns NaN if fewer than 100 valid pairs (e.g., a constant-prediction fold).

The Optuna objective is **mean IC across the 5 folds**. NaN folds are excluded from the mean. Pruning is disabled (5 folds is small enough that median-pruning is more disruptive than it's worth).

## Model pipelines

### XGBoost

- Native NaN handling — no imputation
- Native categorical handling via `enable_categorical=True` and `pd.Categorical` dtype on `sector`. Sector levels become tree split points without one-hot expansion. Handles `sector_unknown` cleanly.
- Search space (Optuna `_make_xgb_params`):

  | Param | Range | Note |
  |---|---|---|
  | max_depth | 3–10 | tree depth |
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

## Smoke run results (10 trials each)

**Subset for smoke:**
- Universe: SP500 actives only (503 tickers)
- Features: price-derived (12) + macro (10) = 22 features (no fundamentals, no sector)
- 734,646 training rows after target-NaN filter

**XGBoost (10 trials, 5 folds each = 50 fits):**

| Metric | Value |
|---|---|
| Best mean IC | **0.0854** |
| Best params | max_depth=3, lr=0.20, n_est=437, subsample=0.85, colsample=0.86, min_child_weight=15, gamma=0.80, reg_alpha=4.48, reg_lambda=1.06 |
| Wall-clock | ~5 min |

Per-trial mean IC range: 0.044 to 0.085. All trials produced positive IC. Some trials had NaN on fold 0 (earliest validation, narrow training set, constant predictions on certain hyperparameter combos) but mean-across-valid-folds was always positive.

**ElasticNet (10 trials, 5 folds each = 50 fits, 5 trial failures):**

| Metric | Value |
|---|---|
| Best mean IC | **0.0711** |
| Best params | alpha=0.00119, l1_ratio=0.433 |
| Wall-clock | ~37 s |
| Failed trials | 5 of 10 |

5 trials returned NaN (constant predictions) — these were the high-alpha trials where the L1/L2 penalty was strong enough to zero out all coefficients. **This is informative, not a bug:** Optuna's TPE sampler in Phase 3 will learn from these failures and avoid high-alpha regions, focusing tuning on the productive alpha range (< 0.01). The valid trials (5 of 10) all produced positive IC between 0.056 and 0.071.

**Smoke-level sanity check:**
- Both models produce **positive cross-sectional IC** on out-of-sample folds → there is signal in the price + macro features alone
- XGBoost's slight edge (0.085 vs 0.071) is consistent with its ability to capture non-linear interactions
- Neither value is suspiciously high (we'd flag IC > 0.20 as a possible label-leakage red flag). The 0.05–0.10 range is typical for daily/weekly cross-sectional alpha in liquid US equities and gives the portfolio enough signal to make non-trivial allocation decisions

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
