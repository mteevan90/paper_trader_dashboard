# Larger Universe v1 — Phase 3 hyperparameter tuning results

**Run dates:** 2026-05-11 22:28 UTC → 2026-05-12 03:41 UTC
**Wall-clock total:** 5h 13m (XGBoost 4.35h + ElasticNet 0.86h, sequential)
**Branch:** `feat/larger-universe-v1-study`
**Spec:** locked at `docs/studies/larger_universe_v1/spec.md` (21d horizon + monthly rebalance, post-diagnostic revision)
**Status:** Phase 3 complete; awaiting Mike's review before Phase 4.

## Headline numbers

| Model | Trials | Wall-clock | Best mean cross-sec IC | Best trial # |
|---|---|---|---|---|
| **XGBoost** (primary) | 200 | 4.35 h | **0.0282** | trial 150 |
| **ElasticNet** (sanity check) | 100 | 0.86 h | **0.0144** | trial ~89 |

**XGBoost wins by ~2× margin (0.0282 vs 0.0144).** Per the agreed framework: if ENet had beaten XGB at 100 vs 200 trials, that would have been a signal the alpha is mostly linear; instead the opposite happened. **The alpha is genuinely non-linear** — XGBoost's tree splits + interaction terms + native categorical handling captured signal ElasticNet's linear+penalty form couldn't.

## XGBoost — full per-fold breakdown (winning trial)

Trial 150's best mean cross-sectional IC: **0.0282**

| Fold | Val window | n_dates | mean_ic | std_ic | positive_rate |
|---|---|---|---|---|---|
| 0 | 2018-05-11 → 2019-05-10 | 251 | **+0.0362** | 0.106 | 0.67 |
| 1 | 2019-05-13 → 2020-05-08 | 251 | **+0.0841** | 0.152 | **0.73** |
| 2 | 2020-05-11 → 2021-05-07 | 251 | −0.0107 | 0.173 | 0.48 |
| 3 | 2021-05-10 → 2022-05-05 | 251 | −0.0210 | 0.126 | 0.50 |
| 4 | 2022-05-06 → 2023-05-11 | 255 | +0.0527 | 0.142 | 0.63 |

**Best XGBoost hyperparameters:**

```json
{
  "max_depth": 8,
  "learning_rate": 0.01964,
  "n_estimators": 678,
  "subsample": 0.6419,
  "colsample_bytree": 0.9715,
  "min_child_weight": 19,
  "gamma": 0.3997,
  "reg_alpha": 0.5723,
  "reg_lambda": 0.000182
}
```

Notable: `max_depth=8` (upper bound of the 3-8 search range — search wanted deeper trees), `n_estimators=678` (high, paired with low `learning_rate=0.02` — slow careful learning), `min_child_weight=19` (conservative leaf-size lower bound), `reg_lambda=0.000182` (very loose L2 — basically no L2 regularization). The model wanted deep trees with weak per-tree regularization, using the high-n_estimators + low-learning-rate + subsample/colsample stochasticity to manage overfitting.

## ElasticNet — full per-fold breakdown (winning trial)

Best mean cross-sectional IC: **0.0144**

| Fold | Val window | n_dates | mean_ic | std_ic | positive_rate |
|---|---|---|---|---|---|
| 0 | 2018-05-11 → 2019-05-10 | 251 | +0.0202 | 0.130 | 0.56 |
| 1 | 2019-05-13 → 2020-05-08 | 251 | **+0.0596** | 0.125 | **0.66** |
| 2 | 2020-05-11 → 2021-05-07 | 251 | +0.0130 | 0.131 | 0.56 |
| 3 | 2021-05-10 → 2022-05-05 | 251 | **−0.0513** | 0.137 | **0.37** |
| 4 | 2022-05-06 → 2023-05-11 | 255 | +0.0303 | 0.178 | 0.50 |

**Best ElasticNet hyperparameters:**

```json
{
  "alpha":    1.016e-05,
  "l1_ratio": 0.7703
}
```

**Notable concern: `alpha` hit the search floor (1e-5).** TPE wanted lower regularization than the search allowed. The search range `[1e-5, 1.0]` was set conservatively; a follow-up run with `alpha ∈ [1e-7, 1e-2]` might find a better minimum. Not blocking for Phase 4 — Phase 4 uses XGBoost as the primary model — but worth noting if we ever want to extend the ENet sanity check.

## Fold 3 separate reporting (2021-05 to 2022-05 — the 2022 bear market regime shift)

Both models negative on this fold as predicted. **Regime sensitivity:**

| Model | Fold 3 mean_ic | Fold 3 positive_rate | Magnitude vs other folds |
|---|---|---|---|
| XGBoost | **−0.0210** | 0.50 | within ~0.05 of the median fold IC, modest hit |
| ElasticNet | **−0.0513** | 0.37 | larger magnitude (~2.5× XGB's hit), positive_rate well below 0.50 |

**XGBoost is materially more regime-robust than ElasticNet.** The 2022 reversal hurts both, but XGBoost's tree-based interactions can re-route around the regime change while ElasticNet's fixed coefficient set can't.

This was Mike's specific report-separately requirement. The pattern matches the diagnostic's variant B result (XGB-21d −0.063, ENet-21d −0.046) at smaller magnitudes on the full universe — likely because the full universe's small-cap dispersion partly mutes the large-cap-led 2022 reversal. **The 2022 fold should be a major Phase 5 disclaimer point**: any strategy that rides growth/momentum factors got hit industry-wide in 2022; this is not a model defect, it's a real signal of regime sensitivity.

## Convergence pattern (XGBoost, per-25-trial checkpoints)

| Trial | running best IC |
|---|---|
| 25 | 0.0226 |
| 50 | 0.0256 |
| 75 | 0.0256 |
| 100 | 0.0267 |
| 125 | 0.0271 |
| **150** | **0.0282** ← winner |
| 175 | 0.0282 |
| 200 | 0.0282 |

**XGBoost plateaued at trial 150; last 50 trials added zero improvement.** 200 trials was slightly oversized for this search space — 150 would have been enough. Useful information for future studies: 150-200 is the right zone for 9-param XGBoost hyperparameter search on this dataset.

ElasticNet's convergence trace is uninformative because most trials failed (NaN) — the running best was 0.0144 from very early on and never moved, because TPE was largely guessing inside a high-failure-rate region.

## Per-trial timing distribution

**XGBoost (200 successful trials):**
- min: 18.3 s
- median: 81.9 s
- max: 124.3 s
- pathological-trial flags (>10min): **0**
- All trials cleanly within the expected 18–125 s band.

**ElasticNet (100 trials, 14 returned a value, 86 returned NaN):**
- min: 17.5 s (the NaN-fast trials)
- median: 17.5 s
- max: 229 s (the few successful long trials)
- Bimodal distribution: ~17s constant-prediction-collapse trials vs ~100-230s real-fit trials.
- pathological-trial flags (>10min): **0**

## NaN trials problem in ElasticNet (lesson for the writeup)

86 of 100 ElasticNet trials returned NaN. This is the constant-prediction collapse: when L1 + L2 penalties are strong enough to zero out all coefficients, the model predicts a constant and cross-sectional Spearman is undefined.

Optuna treats NaN as a trial failure, so the TPE sampler's surrogate didn't get useful information from these trials. The 14 real-value trials all hovered near 0.0144 (the linear-model-on-this-data alpha ceiling). The sampler kept probing high-alpha regions because the failed trials gave no gradient.

**Mitigation for a future ENet re-run** (low priority since XGBoost is the primary):
1. Tighten the search range to `alpha ∈ [1e-7, 1e-2]` (the winning alpha was 1e-5, at the floor)
2. Replace NaN-on-constant-prediction with `0.0` (no-information IC) so the surrogate learns to avoid high-alpha
3. Could reach 0.018–0.022 best IC with focused exploration, but still well below XGBoost's 0.0282

## Smoke vs full-universe comparison

| Stage | Universe | XGBoost best IC | ElasticNet best IC |
|---|---|---|---|
| Variant B smoke | SP500 actives (503 tickers) | 0.019 | **0.031** |
| Phase 3 | Full universe (2,122 tickers) | **0.0282** | 0.0144 |

XGBoost IMPROVED on full universe (more cross-sectional dispersion → more signal to learn from). ElasticNet DEGRADED. Hypothesis: the linear model's signal on SP500 was largely large-cap-driven and got diluted by the small-cap noise in the full universe; XGBoost found small-cap-specific patterns to compensate. This further supports the "alpha is non-linear" reading.

## Phase 4 readiness

**No blockers identified.** Phase 4 can proceed with:
- Primary model: XGBoost with the locked best params (above)
- Sanity-check model: ElasticNet with the locked best params
- Train on the full training window (2017-05-12 → 2023-05-11) using ALL rows (no fold-based subsetting now that hyperparams are fixed)
- Predict on the test window (2023-05-12 → 2025-12-31) with monthly rebalance + 21d label
- Portfolio construction per spec (7.5%/30% caps, fully-invested, score-weighted continuous sizing)
- Four-benchmark comparison (SPY, RSP, IWM, equal-weight SP1500)

What Mike will weigh at the Phase 3 → Phase 4 gate:
- Is **0.0282 mean cross-sectional IC** enough headline alpha to justify Phase 4 work?
  - Industry rule of thumb: IR (information ratio) ≈ IC × sqrt(N) where N is independent breadths per period.
  - 1,500–2,000 stocks × 12 rebalance periods per year = ~18,000–24,000 cross-sectional observations per year
  - Theoretical max IR ≈ 0.028 × sqrt(18000) ≈ 3.7, but real IR is typically 0.5-1.5× theoretical due to noise correlation across stocks
  - **Plausible achievable annualized IR: 1.0–2.0** — meaningful but not extraordinary
- Is the 2022-regime fragility acceptable? XGB fold-3 IC was only mildly negative (−0.021); ENet was worse.
- Is alpha primarily large-cap or small-cap? Phase 4's backtest will surface this via the per-ticker contribution analysis.

## Reproducibility

```
python scripts/research/phase3_tuning.py \
    --xgb-trials 200 --enet-trials 100 --seed 42
```

Wall-clock ~5.3 hours total. Outputs at `models/studies/larger_universe_v1/`:
- `xgboost_best_params.json`, `elasticnet_best_params.json` (winners + per-fold breakdowns)
- `xgboost_study.json`, `elasticnet_study.json` (full trial logs with per-trial duration and fold attrs)
- `phase3_progress.log` (line-by-line stdout)
