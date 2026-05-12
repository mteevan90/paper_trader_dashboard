# ML study CV objectives — v1

**Status:** APPROVED 2026-05-12. v1 canonical for future ML study CV-objective choices.
**Scope:** Permanent methodology finding from the Larger Universe v1 study, applicable to all future ML-based equity studies.
**Companion study:** Larger Universe v1 — see `docs/studies/larger_universe_v1/results.md`.

## TL;DR

In Larger Universe v1, full-cross-section Spearman IC selected an XGBoost hyperparameter set for cross-sectional skill it didn't possess in deployment, while overlooking the top-quintile skill it did possess. The selected model had positive in-fold IC (+0.028) but negative held-out full-IC (−0.009); its actual deployable signal lived in the top quintile (held-out top-quintile-IC +0.048). The CV process rewarded an in-sample pattern that didn't generalize, because the metric weighted all rank positions equally when the deployment only cares about the top.

**Recommendation for future ML studies that build top-N portfolios: use top-quintile Spearman IC as the primary CV objective AND report full-cross-section IC alongside for comparison.** Dual reporting lets the recommendation harden or revise empirically as evidence accumulates over the next several studies. Methodology choices propagated uncritically across studies create correlated errors; this study's recommendation is one data point, and the dual-reporting pattern lets future studies test rather than inherit it.

## The empirical evidence (from Larger Universe v1)

Phase 3 used full-cross-section Spearman IC (per-date Spearman between predictions and forward returns across the entire eligible universe, averaged across dates) as the Optuna CV objective. The locked best hyperparameters for both models had:

| Model | Phase 3 CV mean IC | Held-out Phase 4 full-IC | Held-out Phase 4 top-quintile-IC |
|---|---|---|---|
| XGBoost (primary) | +0.0282 | **−0.009** | **+0.048** |
| ElasticNet (sanity) | +0.0144 | +0.040 | +0.052 |

**XGBoost had a positive in-fold CV IC of +0.028 but a NEGATIVE held-out full-cross-section IC of −0.009.** In other words, the model that "won" the CV objective is actively wrong across the whole cross-section when evaluated on held-out data. But its top-quintile IC is +0.048 — positive — so when restricted to the top 20% of its own predictions, it's right.

Conclusion: the CV objective rewarded an in-sample pattern (XGBoost was discriminating among the bottom-of-distribution stocks in training data) that didn't generalize. The actual deployable signal lives in the top-of-distribution, which the full-cross-section metric doesn't isolate.

Decile-return analysis on the same model in the Phase 4 combined test + reserved-validation window:

| Decile | XGBoost mean forward 21d return | ElasticNet mean forward 21d return |
|---|---|---|
| 1 (low score) | +0.357 (outlier-driven, std 2.02) | +0.008 |
| 2 | +0.008 | +0.000 |
| 3 | +0.008 | −0.003 |
| 4 | +0.011 | −0.002 |
| 5 | +0.014 | +0.002 |
| 6 | +0.014 | +0.001 |
| 7 | +0.007 | +0.002 |
| 8 | +0.011 | +0.010 |
| 9 | +0.012 | +0.009 |
| 10 (high) | +0.011 | **+0.086** (DBD-driven, std 0.347) |

XGBoost shows no monotonic structure D2–D10. The model's score has essentially zero ranking power in the middle of the distribution. Yet its held-out top-quintile IC is +0.048 (positive but mild), confirming there's some real signal at the top — just not enough that the CV process surfaced via the full-cross-section metric.

ElasticNet shows the expected step at D10 but with high outlier-driven variance.

## Mechanism — why full-cross-section IC misleads for top-N strategies

### Two plausible interpretations of the XGBoost failure

**Interpretation A (overfitting):** XGBoost overfit cross-sectional ranking in CV — memorizing 2017–2022 ticker patterns that didn't generalize to 2023–2025. The fix would be more regularization, better features, or smaller model capacity, not a different CV objective.

**Interpretation B (gradient-signal dominance):** XGBoost never actually learned cross-sectional ranking — the gradient signal during training came predominantly from macro features that vary by date but not by ticker. The model effectively learned market-timing instead. Full-cross-section Spearman IC scored this skill positively in CV because predicting the day's average return helps rank across the cross-section on average (every ticker on a "bad day" gets penalized, so the model that predicts "bad day" gets credit). But the skill doesn't transfer to top-N deployment, where the question is "which 30 tickers will outperform on this day," not "what's the average return today."

### The evidence supports Interpretation B

Phase 4 SHAP feature importance for the trained XGBoost model (top 10 features):
1. **SAHM** (recession indicator) — 0.0235 (date-level)
2. **NFCI** (Chicago Fed conditions) — 0.0129 (date-level)
3. **VIX** — 0.0117 (date-level)
4. **BAA spread** (credit) — 0.0098 (date-level)
5. **USD index** — 0.0068 (date-level)
6. log_market_cap — 0.0065 (ticker-level)
7. sector — 0.0060 (ticker-level, slow-moving)
8. **unrate** (unemployment) — 0.0053 (date-level)
9. in_sp600 — 0.0044 (ticker-level, binary)
10. **wti_oil** — 0.0043 (date-level)

**Six of the top 10 are date-level macro features**; the others are slow-moving ticker-level features (market cap, sector membership, index flag). The model's gradient signal during training was dominated by features that don't vary across the cross-section on a given day. This is the signature of Interpretation B: macro features captured most of the learnable variance in forward returns, and XGBoost allocated its capacity to predicting the time-series average rather than the cross-sectional rank.

Interpretation A (overfitting) would predict a different feature profile: ticker-level price/volatility/momentum features dominating the top, with macro features secondary. We don't see that. Interpretation A is therefore the less-likely explanation; Interpretation B is the operative mechanism.

### Why this implies a CV-objective change rather than just regularization

If the mechanism were Interpretation A, more regularization or feature engineering could fix it. But Interpretation B's diagnosis — "the model learned the wrong skill because the objective rewarded it equally for either skill" — points to the CV objective as the operative lever. No amount of regularization on XGBoost trained against full-cross-section IC will reliably steer it toward cross-sectional skill instead of time-series skill, because the objective doesn't distinguish.

Top-quintile-IC explicitly evaluates the model's skill in the deployment region (top of distribution on each date). A model that learns market-timing scores poorly on top-quintile-IC because predicting "today is up" doesn't help rank within today. A model that learns cross-sectional skill scores well. The CV process then selects for the right skill rather than averaging across both.

This is structurally similar to the classification metric mismatch problem: optimizing accuracy when the deployed system makes decisions only at one operating point is the wrong target. Top-N rank optimization is the equivalent for cross-sectional ranking models.

## Recommendation — what to use instead

For CV objectives in studies that build top-N portfolios:

### Option A: top-quintile Spearman IC (recommended default)

```
For each rebalance date D in val fold:
    Sort eligible tickers by predicted score
    Take top 20% (quintile)
    Within that quintile only:
        Compute Spearman(predicted score, realized return)
    Append to per-date list

Aggregate: mean across dates → top_quintile_ic_mean
Optuna objective = top_quintile_ic_mean
```

Why: directly measures the model's skill in the operating region the strategy uses. Easy to implement (a minor variant of the existing `cross_sectional_ic_stats` function).

Caveat: top quintile is somewhat arbitrary. If a study uses top-5% picks, evaluating top-5%-IC is more aligned but produces noisier metrics due to smaller per-date sample sizes. The 20% threshold is a reasonable compromise between fidelity to deployment and statistical reliability.

### Option B: decile spread (alternative)

```
For each rebalance date D in val fold:
    Sort eligible tickers by predicted score
    Top decile mean realized return − bottom decile mean realized return

Aggregate: mean across dates → decile_spread_mean
Optuna objective = decile_spread_mean
```

Why: most direct measure of "predictions actually separate winners from losers." Less sensitive to within-decile noise than top-quintile-IC.

Caveat: a strategy that doesn't short the bottom decile gets no benefit from bottom-decile signal. Decile spread rewards both sides; an asymmetric strategy might prefer Option A.

### Option C: top-K rank correlation (most aligned, noisiest)

Restrict the IC computation to exactly the top K = N positions the strategy will hold. Most aligned with deployment but smallest per-date sample size.

Caveat: at K=30 (Larger Universe v1's choice), per-date sample size is too small for reliable Spearman. Per-date IC distributions become extremely noisy and the CV process can't reliably distinguish hyperparameter combinations.

### What the new objective doesn't change

- The CV fold structure (5-fold expanding window with horizon-matched embargo) stays
- The dataset construction (filtering, embargo, feature alignment) stays
- The score-to-weights transformation choice (rank-top-N, softmax, etc.) is a portfolio construction decision separate from CV objective design

## Suggested implementation pattern for future studies

### Code-level: add the top-quintile scorer alongside the existing full-IC scorer

In `src/equities/study/training.py` (or wherever the study's CV scorer lives), add a `top_quintile_ic_stats` function alongside `cross_sectional_ic_stats`:

```python
def top_quintile_ic_stats(preds, val_df, top_pct=0.20, min_tickers=30):
    """Cross-sectional IC restricted to top top_pct of each date's predictions.

    Returns the same {mean_ic, std_ic, positive_rate, n_dates_scored} schema
    as cross_sectional_ic_stats so the CV driver swaps cleanly.
    """
    # ... implementation ...
```

The Optuna objective in the study's CV driver becomes `top_quintile_ic_stats(...)["mean_ic"]`. The full-cross-section variant continues to be computed but is recorded as a diagnostic, not the optimization target.

### Contract-level: enforce via the dashboard contract

The dashboard contract (`docs/architecture/dashboard_contract_v1.md`) is being updated as part of this Phase 5 coherent unit to require an `objective.training_cv` field in `meta.json`. Every contract-conformant study must populate it. Allowed values:

- `"top_quintile_spearman_ic"` — recommended default per this memo
- `"mean_cross_sectional_spearman_ic"` — legacy / explicit deviation
- `"decile_spread"` — alternative top-N-relevant metric
- `"<other>"` — study-specific; must include a rationale in `meta.json.notes`

The contract requirement forces deliberate choice: even study authors who never read this memo encounter the field when populating meta.json and have to think about it. This operationalizes the methodology: the recommendation is baked into the artifact every study must produce, not into a memo authors might or might not discover.

### Dual-reporting requirement

For every contract-conformant study going forward, both metrics should be computed and persisted (e.g., in `meta.json.summary_metrics` and/or in the trial log). The CV optimizes against `objective.training_cv`; the other metric is logged for comparison. Over several studies this builds empirical evidence about whether top-quintile-IC reliably predicts deployment performance better than full-IC.

If three subsequent studies show top-quintile IC as the better predictor of deployment performance, the recommendation hardens and this memo bumps to v2 with strengthened language. If they don't, the memo revises with the contrary evidence. This is the dual-reporting pattern's payoff: methodology choices validate or refute themselves over time rather than propagating uncritically.

## Caveats

These split into two distinct epistemological concerns — whether the finding is reliable in the setting we observed it, and whether it generalizes beyond that setting.

### Reliability caveats — whether to trust the finding within this study

1. **Single-study finding.** Larger Universe v1 is one study on one universe at one horizon. The finding is consistent with broader factor-research practice (most published factor literature uses decile spread or top-quintile metrics rather than full-cross-section IC) but isn't formally proven even for this setting from one data point. The dual-reporting pattern in the recommendation is designed to validate or refute it over future studies.
2. **In-sample optimism remains an open risk.** In-sample full-IC was positive (+0.028) but held-out full-IC was negative (−0.009). This in-sample inflation is normal — the lesson is that the CV process didn't catch it. Top-quintile-IC has the same in-sample-optimism risk; whether it's smaller in absolute terms is empirically untested.

### Scope caveats — whether the finding applies beyond this setting

3. **Generalization to non-top-N strategies is unverified.** If a future strategy uses a different construction (full-cross-section vol-targeting, market-neutral long-short with proportional sizing, anything where every rank position contributes to the realized PnL), full-IC may again be the appropriate metric. The recommendation is specifically for top-N portfolios.
4. **Top-quintile-IC has its own variance issues.** With smaller per-date sample sizes (~380 tickers in a 1,900-universe top-quintile), the IC variance is higher than full-IC. Studies should use a longer rolling window or more CV folds to compensate. At extreme top-K (K=30 in our spec), per-date sample size is too small for reliable Spearman — hence the 20% threshold rather than tighter.
5. **The XGBoost-vs-ElasticNet gap may be model-class-specific.** ElasticNet's held-out full-IC stayed positive (+0.040) where XGBoost went negative (−0.009). The CV-objective mismatch hurt XGBoost severely but ElasticNet less. Future studies using only linear models may see a smaller gap and less benefit from switching objectives.
6. **Horizon-dependence is unverified.** Phase 2's horizon diagnostic established that signal lives at monthly horizons in this universe with these features. Whether the full-IC-vs-top-quintile-IC gap is larger or smaller at daily/weekly/quarterly horizons is unknown.

## The pattern this memo establishes

Going forward, when a study surfaces a methodology finding that applies beyond that specific study, it gets a memo at `docs/architecture/<topic>_v1.md`. The study-specific writeup references the memo as the canonical source.

Reasons for the separation:
- Study-specific writeups are dense and the methodology finding gets buried
- Future studies looking for guidance on a similar problem need a discoverable doc, not a 30-page writeup from years prior
- Memos are short, focused, versionable, and reusable
- The `v1` suffix makes refinement explicit — a future study that updates this finding writes `ml_study_cv_objectives_v2.md` and the dashboard / future studies route to the latest version

### Filename convention

Architectural memos use snake-case filenames that read as a sentence when expanded: `ml_study_cv_objectives_v1.md` → "ML study CV objectives, v1." Specifically:
- **Snake-case** (underscores, not dashes), lowercase
- **Descriptive of the topic**, not generic. Avoid "guide", "framework", "patterns" as the primary noun. Concretely name the topic the memo covers.
- **Ends in `_v1.md`** (or current version). Refinements bump to `_v2.md` and the v1 stays as historical reference.
- **No timestamps** in the filename — versions handle that.

### Memo shape

Future architectural memos follow the same shape:
1. **TL;DR** — firm language about the finding within the source study; recommendation in one paragraph.
2. **Empirical evidence** — tables / numbers from the source study, not just prose.
3. **Mechanism** — which interpretation of the evidence we're claiming, why competing interpretations are less likely.
4. **Recommendation** — actionable for future studies; include code-level and contract-level enforcement where applicable.
5. **Implementation pattern** — concrete code snippets, contract changes, or process updates.
6. **Caveats** — split into reliability (within-study trust) and scope (cross-study generalization).
7. **Open questions** — explicitly marked as questions, not commitments.
8. **Sourced from** — list of files/studies the memo draws on.

Concise (1-3 pages target). Reference the source study explicitly. Don't make claims the source study didn't establish.

## Open questions worth a future study

1. Does this finding hold for **monthly-rebalanced long-short** strategies with proportional sizing on both sides? (Unverified — Larger Universe v1 is long-only.)
2. Does **top-quintile-IC** CV produce better hyperparameters in practice than full-IC, or just better-aligned ones with similar absolute performance? (Need a controlled experiment — fix everything except the CV objective and compare.)
3. Is there a **theoretical link** between IC-restricted-to-top-K and downstream top-N portfolio Sharpe? (Probably yes but not derived here.)
4. Does the **horizon** affect the top-quintile-vs-full-IC gap? Phase 2's horizon diagnostic suggested signal lives at monthly horizons for fundamentals-driven features; whether the gap is larger or smaller at daily/weekly horizons is unknown.

These are research questions, not action items. A future study that wants to invest in resolving them produces another memo at `docs/architecture/`.

## Sourced from

- `docs/studies/larger_universe_v1/results.md` (forthcoming) — full Larger Universe v1 writeup
- `docs/diagnostics/larger_universe_v1_phase3_results.md` — Phase 3 hyperparameter tuning results
- `docs/diagnostics/larger_universe_v1_cv_design.md` — Phase 2 CV design + diagnostic
- `models/studies/larger_universe_v1/contract_v1/ic_decomposition.parquet` — held-out full-IC vs top-quintile-IC, per model
- `models/studies/larger_universe_v1/contract_v1/decile_returns.parquet` — per-decile forward returns
