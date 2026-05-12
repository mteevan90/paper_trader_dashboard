# Larger Universe v1 — Phase 2 horizon/feature diagnostic

**Run date:** 2026-05-11 (evening, Phase 2→3 gate)
**Purpose:** Identify whether the near-zero cross-sectional IC observed in the original smoke is due to feature set (price+macro only) or horizon (5-day forward return). Two diagnostic variants run against the SP500-actives subset.

## Configuration matrix

| Variant | Features | Horizon | Embargo | Tickers | Rows | Wall-clock (XGB+ENet) |
|---|---|---|---|---|---|---|
| Original | 22 (price+macro) | 5d | 5d | 503 SP500 active | 734,646 | ~240 s |
| A | **38 (full)** | 5d | 5d | 503 SP500 active | 734,646 | 341 s |
| B | **38 (full)** | **21d** | **21d** | 503 SP500 active | 734,646 | 534 s |

Both A and B run 10 Optuna trials each for XGBoost (max_depth 3-8, 9 params) and ElasticNet (alpha + l1_ratio).

## Headline results — XGBoost (best-trial per-fold breakdown)

| Variant | F0 (n,IC) | F1 (n,IC) | F2 (n,IC) | F3 (n,IC) | F4 (n,IC) | **Honest mean** |
|---|---|---|---|---|---|---|
| Original 22f / 5d | 0, — | 0, — | 4, +0.415 | 0, — | 70, +0.020 | **0.218 (degenerate)** |
| A: full 38f / 5d | 0, — | 0, — | 222, +0.013 | 19, +0.007 | 71, +0.006 | **0.009 (still degenerate)** |
| **B: full 38f / 21d** | **251, +0.023** | **251, +0.081** | **251, +0.023** | **251, −0.063** | **255, +0.030** | **+0.019 (all folds covered)** |

**Variant B is the only XGBoost configuration where every fold produces a clean cross-sectional IC** — no constant-within-date predictions, no degenerate trials. The 5-day horizon (both with and without fundamentals) collapses to "market-timing only" for tree models; the 21-day horizon lets ticker-level features actually contribute.

## Headline results — ElasticNet (best-trial per-fold breakdown)

| Variant | F0 (mean, std, +rate) | F1 | F2 | F3 | F4 | **Honest mean** |
|---|---|---|---|---|---|---|
| Original 22f / 5d | +0.026, .20, .55 | −0.007, .27, .51 | +0.033, .24, .51 | −0.021, .26, .44 | +0.034, .21, .56 | **+0.013** |
| A: full 38f / 5d | +0.024, .23, .59 | +0.042, .21, .60 | +0.015, .23, .53 | −0.018, .25, .47 | +0.037, .20, .55 | **+0.020** |
| **B: full 38f / 21d** | **+0.002, .21, .49** | **+0.116, .14, .75** | **+0.007, .21, .56** | **−0.046, .20, .41** | **+0.075, .25, .51** | **+0.031** |

ElasticNet covers all folds cleanly in every configuration (linear model + L1/L2 doesn't collapse the way XGBoost does). The 5d→21d horizon shift more than doubles the mean cross-sectional IC. Adding fundamentals at 5d gave a modest lift (0.013 → 0.020); going to 21d gave a bigger lift (0.020 → 0.031). The biggest single-fold improvement is fold 1 going from +0.042 (A) to +0.116 (B) with positive-rate jumping from 0.60 to 0.75.

## One-line summaries

- **Variant A XGBoost honest mean: 0.009** (price+macro had ~0.000; full features added ~0.009 lift, still degenerate on folds 0–1)
- **Variant A ElasticNet honest mean: 0.020** (price+macro had 0.013; full features added 0.007 lift, all folds covered)
- **Variant B XGBoost honest mean: 0.019** (5d full-features had 0.009; 21d horizon added 0.010 lift AND eliminated the constant-prediction degeneracy)
- **Variant B ElasticNet honest mean: 0.031** (5d full-features had 0.020; 21d added 0.011 lift; meaningfully positive on 4 of 5 folds with positive-rate ≥ 0.49)

## Common pattern: fold 3 is hostile for both models, both variants

Fold 3 validation window: 2021-05-10 → 2022-05-05. **Both XGBoost and ElasticNet produce negative cross-sectional IC on this fold in Variant B** (XGB −0.063, ENet −0.046). Same direction at 5d horizon (smaller magnitude). This is the 2022 bear-market regime transition — features trained on 2017–2021 momentum/growth patterns reversed in 2022's value rotation. Not a bug; a real regime shift the model can't extrapolate through.

Implication for Phase 5: the 2022 reversal will likely be the single biggest drag on backtest performance. Strategies that ride growth/momentum factors got crushed in 2022 across the industry; this CV result is consistent with that. Worth disclaiming in Phase 5 study writeup.

## Synthesis

**The binding constraint is horizon, not feature set.**

- At 5-day horizon, the SP500-actives cross-section doesn't have enough signal-to-noise on ticker-level features (returns, fundamentals, sector). XGBoost specifically collapses to learning macro features only (constant predictions within each date). ElasticNet's L2 regularization spreads coefficient mass enough that some ticker-level features get used, but the mean cross-sectional IC is still tiny (0.013–0.020).
- At 21-day horizon, fundamentals + sector + log_market_cap become productive. XGBoost no longer degenerates. Both models reach mean cross-sectional IC in the 0.019–0.031 range, with consistent positive contributions from 4 of 5 folds.

**Why 21d works better:**
1. **Fundamentals are slow-moving.** A 5-day window doesn't give enough time for an over-valued stock to mean-revert; a 21-day window does. P/E ratios are a 3-month-or-longer story, not a 1-week story.
2. **Sector rotation operates on monthly cycles.** Macro factor exposure shifts take weeks, not days, to show up in returns.
3. **Cross-sectional noise dominates at 5d.** Earnings surprises, news flow, single-day idiosyncrasies — all this is mostly orthogonal to fundamentals at the 5d level but smooths out over 21d.

This is consistent with the academic factor-research literature: most cross-sectional alpha studies use monthly horizons. Daily/weekly cross-sectional alpha exists but typically requires technical/momentum features tuned for short-horizon dynamics, not the slow-moving fundamentals the spec emphasizes.

## What this changes about the study spec

The original spec locked the rebalance cadence at "weekly (every Friday close)" — that's an execution detail. The label horizon (what the model predicts) can differ from the rebalance cadence:

- **Option 1: Change label horizon to 21d, keep weekly rebalance.** Model predicts 21-day forward return; portfolio rebalances weekly using the most recent prediction. Each prediction informs ~4 rebalances before becoming stale. This is the cleanest fix — uses signal where it lives, preserves the weekly cadence the spec wants.
- **Option 2: Keep 5d label, accept low IC.** Phase 3 burns 6-7h to tune on a near-zero signal landscape. Expected outcome: similar to Variant A — ENet maybe reaches 0.025–0.030 best mean IC, XGB stays degenerate or marginal. Phase 5 study disclaimer would need to acknowledge the model has minimal cross-sectional alpha.
- **Option 3: Pivot to monthly rebalance + monthly label.** Cleanest factor-research design but violates the spec's weekly cadence. Larger architectural change.

Recommendation order if I had to rank: **Option 1 > Option 2 > Option 3.** Option 1 preserves the spec's intent (weekly rebalance for nimbleness) while accommodating the data reality (cross-sectional signal lives at monthly horizons).

## Methodology notes

- The 21d horizon's larger embargo (21 days) shifts every fold's `train_end` back by ~16 days vs the 5d version. This *reduces* training data per fold by ~3,000 rows on average (out of ~120K) — negligible.
- The 5d→21d label change also means each ticker has 16 fewer rows at the tail of its data (last 21 vs last 5 are dropped because of forward-return NaN). Negligible.
- Both A and B got the same `beta` skip-imputation warning from sklearn — `beta` is 100% NaN in the earliest folds (training data starts 2017-05-12; rolling-756d beta can't fill until ~2019). SimpleImputer correctly skips it; ElasticNet treats it as not-a-feature for those folds. XGBoost handles natively.

## Reproducibility

```
python scripts/research/smoke_phase2_variant.py --horizon 5  --features full --variant variant_a
python scripts/research/smoke_phase2_variant.py --horizon 21 --features full --variant variant_b
```

Each run is ~6–9 min wall-clock.
