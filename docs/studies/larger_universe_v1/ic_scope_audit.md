# v1 analytics — price-universe scope audit

**Status:** Informational. No corrections applied to v1 artifacts; corrections require explicit approval after audit review.
**Date:** 2026-05-13
**Trigger:** v2 Gate 4 surfaced an IC discrepancy (v1 reported +0.048 top-quintile IC; v2 reported −0.0041 on bit-identical scores). Investigation traced to price-universe scope. This audit inventories the full extent.

## Summary

Two of v1's sixteen `contract_v1/` artifacts depend on a price-universe-scope choice that v1's `phase5_analytics.load_inputs()` made implicitly: **only tickers in `holdings["ticker"].unique()` (≈450 tickers across XGB + ENet) have prices loaded.** Computations that test `ticker in prices.columns` are silently restricted to that held-tickers-only cross-section.

Affected artifacts:
- `ic_decomposition.parquet` — top-quintile IC and full IC computed on held-tickers-only cross-section
- `decile_returns.parquet` — per-decile forward returns computed on held-tickers-only cross-section

All other v1 artifacts are scope-independent (engine outputs, NAV-based metrics, holdings-only summaries, or analytics that iterate over `holdings.iterrows()`).

The two affected artifacts produce **materially different findings** under standard full-cross-section scope:

| Metric | v1 reported | Full-universe scope | Material change? |
|---|---:|---:|---|
| Top-quintile IC (XGBoost) | **+0.0481** | **−0.0041** | YES — sign reversal |
| Full IC (XGBoost) | −0.0087 | −0.0089 | no (both near zero) |
| Decile 1 mean fwd return | +0.3566 (std 2.02) | +0.0579 (std 0.25) | YES — magnitude 6× smaller, dispersion 8× smaller |
| Decile 10 mean fwd return | +0.0113 | +0.0121 | no |
| Decile 1 > Decile 10 (inversion direction) | YES | YES | no (direction preserved at both scopes) |

v1 published documents that cite the affected numbers in load-bearing claims:
- `docs/architecture/ml_study_cv_objectives_v1.md` — the CV objectives architectural memo
- `docs/studies/larger_universe_v1/results.md` — the v1 study writeup (lines 13, 170)

Dashboard rendering may also display the affected artifacts; not audited in this report.

## Inventory of v1 `contract_v1/` artifacts

| Artifact | Scope-dependent? | Evidence |
|---|---|---|
| benchmarks.parquet | No | Benchmark NAVs (SPY, RSP, IWM, EW-SP1500); not ticker-level |
| concentration_summary.json | No | Computed from `holdings.parquet` only |
| **decile_returns.parquet** | **YES** | Confirmed below |
| feature_importance.parquet | No | SHAP / gain on training features; no ticker-level forward returns |
| holdings.parquet | No | Engine output |
| **ic_decomposition.parquet** | **YES** | Confirmed below |
| meta.json | No | Summary metrics from portfolio NAV |
| per_ticker_attribution.parquet | No | Iterates over `holdings.iterrows()`; only held tickers contribute regardless of `prices` width |
| portfolio.parquet | No | Engine NAV output |
| rolling_win_rate.parquet | No | Portfolio NAV vs SPY NAV; no ticker-level prices |
| scores.parquet | No | Engine output (scores at rebalance dates) |
| trades.parquet | No | Engine output |
| trial_log.parquet | No | Phase 3 Optuna trial records |
| tuning_convergence.parquet | No | Phase 3 |
| tuning_summary.json | No | Phase 3 |
| walk_forward.parquet | No | IC computed via `labels.merge(scores)` (labels built from full universe at startup) — already full-cross-section. v2 bit-reproduces. |

## Detailed findings: affected artifacts

### `ic_decomposition.parquet`

**Source code:** `scripts/research/phase5_analytics.py:42-58` (`load_inputs`) and `:167-229` (`ic_decomposition`).

**Mechanism:** `load_inputs` loads `prices` only for tickers in `holdings["ticker"].unique()` (line 52). Inside `ic_decomposition`, the loop at lines 191-201 tests `if t in prices.columns` and falls through to `NaN` for non-priced tickers. `dropna(subset=["fwd_ret"])` (line 203) then drops non-held tickers from the IC computation.

**v1 reported (held-subset, 450 tickers):**
| | full_ic_mean | full_ic_std | top_quintile_ic_mean | top_quintile_ic_std |
|---|---:|---:|---:|---:|
| XGBoost | −0.008740 | 0.111886 | **+0.048121** | 0.113320 |
| ElasticNet | +0.039742 | 0.174717 | +0.051698 | 0.135406 |

**Reproduction at three scopes** (v1's formula, v1's scores, varied `prices.columns`):

| Scope | n tickers | full_ic_mean (XGB) | top_quintile_ic_mean (XGB) |
|---|---:|---:|---:|
| v1 holdings (XGB + ENet) | 450 | −0.008740 | **+0.048121** ← matches v1 pinned |
| XGB held only | 340 | +0.006062 | +0.049366 |
| Full snapshot universe | 1963 | −0.008855 | **−0.004134** ← matches v2 Gate 4 |

The 450-ticker scope reproduces v1's pinned values to floating-point identity. The full snapshot universe scope reproduces v2's Gate 4 values to floating-point identity. The formula is mathematically identical between v1 and v2; the difference is scope alone.

### `decile_returns.parquet`

**Source code:** `scripts/research/phase5_analytics.py:121-164` (`decile_analysis`). Same `load_inputs` source for `prices`.

**Mechanism:** `pd.qcut(day_scores["score"], 10)` buckets scores from the full eligible universe into deciles. But then forward-return computation (lines 140-151) tests `if t not in prices.columns: continue` — so for each decile, only held tickers in that decile contribute to `forward_rets`.

Because held tickers are biased toward upper deciles (the model picks the top-30 for holdings), held-ticker representation across deciles is uneven:
- v1 scope avg tickers per decile per rebalance: **38.76** (~5 held tickers per decile of ~160-population, plus model selection makes deciles 9-10 contain disproportionately many held names)
- Full snapshot scope avg tickers per decile per rebalance: **160.09** (representative population per decile)

The pathological consequence is decile 1: it contains few or zero held tickers per rebalance. The mean forward return at v1's scope is +35.66% with std +201.88% — driven by single-ticker tail events rather than population averages.

**Reproduction (v1's formula, v1's scores):**

| Decile | v1 reported (held-450) | v1 XGB-only (340) | Full snapshot (1963) |
|---:|---:|---:|---:|
| 1 | **+0.3566** (std 2.02) | +0.4625 (std 2.66) | **+0.0579** (std 0.25) |
| 2 | +0.0078 | −0.0011 | +0.0083 |
| 3 | +0.0078 | +0.0044 | +0.0098 |
| 4 | +0.0110 | +0.0092 | +0.0101 |
| 5 | +0.0145 | +0.0117 | +0.0121 |
| 6 | +0.0145 | +0.0133 | +0.0114 |
| 7 | +0.0072 | +0.0022 | +0.0089 |
| 8 | +0.0109 | +0.0151 | +0.0095 |
| 9 | +0.0118 | +0.0097 | +0.0108 |
| 10 | +0.0113 | +0.0112 | +0.0121 |

v1's held-scope reproduces v1's pinned values bit-for-bit. Full-universe gives interpretable decile-1 stats (still highest, but realistic).

### `per_ticker_attribution.parquet` (verified NOT scope-dependent)

Formula iterates over `m_holdings.iterrows()`. Only held tickers contribute to the `contrib` accumulator. The `prices` DataFrame width affects which lookups succeed, but for held tickers the snapshot price file always exists (those tickers were tradeable enough to be in the universe).

Empirical verification: running v1's formula at held-scope (450) vs full-scope (1963) on v1's holdings produces **bit-identical** output. Max abs diff in `total_excess_contribution` across all 340 XGB and 299 ENet ticker rows: **0.000000e+00**. No rows added or removed.

### `walk_forward.parquet` (verified NOT scope-dependent)

`scripts/research/phase5_walk_forward.py:165-187` (`_cross_sectional_ic_for_period`) computes IC via `scores.merge(labels, on=["date","ticker"])`. Labels are built once at the top of `_setup()` via `build_labels(full_universe, ...)`. So walk-forward IC is computed across the full eligible cross-section regardless of any held-set restriction.

Empirical verification: v1 walk_forward.parquet XGB `mean_ic` values across 6 windows are bit-identical to v2-baseline walk_forward.parquet (max abs diff: 0.000000e+00). v2-baseline reproduces v1's walk-forward IC because the underlying IC formula uses labels, not prices.

This means v1's walk-forward IC is the standard full-cross-section per-window IC. The 6-window XGB `mean_ic` values are: +0.0072, −0.0201, +0.0448, +0.0370, +0.0016, −0.0084. Mean across windows: +0.0103. These are small near-zero per-period ICs, sometimes positive sometimes negative — consistent with a weak model signal.

## Material findings that change

### 1. Top-quintile IC sign reversal (XGBoost)

v1's reported claim: model has **+0.048 top-quintile IC** in deployment (interpreted in the writeup as "the model has real skill where the strategy actually places weight").

Under full-cross-section scope: top-quintile IC is **−0.0041** with std 0.0781 across 36 rebalances. Essentially zero, slightly anti-predictive.

**Implication:** The directional sign of v1's central skill claim depends on which cross-section is being correlated. Under the held-subset scope, it's mildly positive (+0.048). Under the standard full-cross-section scope, it's effectively zero.

This is not a "wrong vs right" issue at the data level — both numbers are correct under their respective definitions. It's a "which definition is the substantive one" issue. The architectural memo's prose and the writeup's framing suggest the intended definition was full-cross-section ("the model has real skill"); the actual measurement was held-subset.

### 2. Decile 1 stats are pathological at v1's scope

v1's reported claim: decile 1 has mean forward return **+0.357** with std **+2.02**. The writeup treats this as evidence of decile inversion ("the worst-scored stocks outperform on average").

Under full-cross-section scope: decile 1 has mean **+0.058** with std **+0.25**. Still the highest of all deciles, but the magnitude is realistic.

**Implication:** The direction of decile inversion (decile 1 > decile 10) is preserved at both scopes — so the finding "low-scored stocks outperform high-scored stocks on average" is robust. But the magnitude is very different. v1's +35.66% is a small-sample tail artifact; +5.79% is the real cross-sectional average.

### 3. Full IC, deciles 2-10, walk-forward IC: not materially changed

- Full IC (XGB): −0.0087 (held) vs −0.0089 (full). Both essentially zero.
- Deciles 2-10: differ at the 4th decimal place but maintain similar bunched-flat structure.
- Walk-forward IC: bit-identical (already full-cross-section).
- Per-ticker attribution: bit-identical (not scope-dependent).

## v1 documents that cite the affected numbers

### `docs/architecture/ml_study_cv_objectives_v1.md`

Load-bearing claims that cite the +0.048 number:

- **L9 (opening claim):** "full-cross-section Spearman IC selected an XGBoost hyperparameter set for cross-sectional skill it didn't possess in deployment, while overlooking the top-quintile skill it did possess."
- **L19 (data table):** XGBoost row reports CV IC +0.0282, held-out full IC −0.009, top-quintile IC **+0.048**.
- **L22:** "XGBoost had a positive in-fold CV IC of +0.028 but a NEGATIVE held-out full-cross-section IC of −0.009. In other words, the model that 'won' the CV objective is actively wrong across the whole cross-section."
- **L41:** "XGBoost shows no monotonic structure D2–D10. The model's score has essentially zero ranking power in the middle of the distribution. Yet its held-out top-quintile IC is +0.048 (positive but mild), confirming the model has skill where the strategy actually places weight."
- **L93-94 (recommended training objective):** "Aggregate: mean across dates → top_quintile_ic_mean / Optuna objective = top_quintile_ic_mean"

The memo's central claim is that XGBoost's training-objective should be top-quintile IC because the model "has skill" in the top quintile (+0.048). Under full-cross-section scope, the top-quintile IC is **−0.0041**. The "model has skill" claim does not hold under that scope.

A consistency note: the memo's pseudocode for the proposed training objective (L93-94, and the code at L135-) computes top-quintile IC within the in-fold cross-section. In-fold there are no "held" tickers (the model hasn't selected anything yet — it's still training). So the proposed Optuna objective would be top-quintile-of-full-cross-section IC, which the v1 measurement (−0.0041) suggests does NOT have positive skill. The memo's recommended objective and the memo's evidence appear to be measuring different things — one is full-cross-section top-quintile, the other is held-subset top-quintile.

### `docs/studies/larger_universe_v1/results.md`

Load-bearing claims:

- **L13 (executive summary, methodology finding 1):** "full-cross-section Spearman IC was the wrong CV objective for top-N strategies. XGBoost's positive in-fold IC (+0.028) became negative held-out (−0.009), with the real signal showing up in top-quintile IC (+0.048)..."
- **L170:** "XGBoost shows the diagnostic pattern that triggered the architectural memo: held-out full-cross-section IC of −0.009 but top-quintile IC of +0.048. The model has real skill where the strategy actually places weight."

The writeup explicitly contrasts "full-cross-section IC of −0.009" with "top-quintile IC of +0.048" in a single sentence (L170), which leads readers to assume both are computed on the same cross-section. They are not.

## What the audit does NOT establish

- Whether v1's `load_inputs` scope choice was intentional or an unrecognized bug. The code was authored once; it's been consistent since.
- Whether the architectural memo's recommendation (use top-quintile IC as CV objective) is correct or incorrect on its own merits. The memo's recommendation is for a full-cross-section computation; the evidence cited is a held-subset computation. Whether the recommendation is well-justified is a separate question from the scope finding.
- Whether dashboard rendering of `ic_decomposition.parquet` and `decile_returns.parquet` displays the held-subset numbers in user-facing surfaces. (Not investigated in this audit.)

## What v2 Gate 4's evidence adds

This audit was triggered by v2 Gate 4 surfacing the IC discrepancy. v2's analytics produce:
- `ic_decomposition.parquet` per variant at full-cross-section scope: top-quintile IC −0.0041
- `decile_returns.parquet` per variant at full-cross-section scope: decile 1 +0.058 (highest of 10)

These are the standard-definition equivalents of v1's affected artifacts. v2's numbers are interpretable as "what v1's would have said if scope had been the eligible universe rather than held names".

v2-baseline's other artifacts (portfolio.parquet, holdings.parquet, etc.) bit-reproduce v1's. So the choice of full-cross-section scope in v2 does not contaminate v2's headline reproducibility check — it's a Phase 5 analytics methodology difference, not a Phase 4 backtest difference.

## Recommended next steps

The audit is informational. Decisions about corrections are deferred to Mike's review. Candidate next steps surfaced as options:

1. **Correction notes on v1 documents.** Add a clearly-marked correction section to `docs/architecture/ml_study_cv_objectives_v1.md` and `docs/studies/larger_universe_v1/results.md` documenting the scope finding, the actual full-cross-section numbers, and the implications for the memo's central claim.
2. **Re-derivation of v1's affected artifacts at full-cross-section scope.** Write new versions of `ic_decomposition.parquet` and `decile_returns.parquet` at the standard scope and surface as v1 "v1.1" or "corrected" artifacts alongside the originals (which remain as the historical record).
3. **Architectural memo revision.** Decide whether the memo's recommendation (use top-quintile IC as CV objective) still holds given that its evidence (+0.048) was held-subset rather than full-cross-section. If the recommendation stands, the memo needs a revision explaining why it stands under correct evidence. If the recommendation does not stand, the memo needs a substantive update or retraction.
4. **v2 writeup framing.** Decide how v2's writeup characterizes the IC finding given the v1 record. (This was the original blocker that triggered this audit.)
5. **Future v2-pattern study tooling.** Ensure future Phase 5 analytics tools load prices for the full eligible universe (not just held tickers), so scope-restriction doesn't silently affect cross-sectional measurements. v2's `phase5_analytics_v2.py` already does this; the v1 pattern is the one that needs updating if v1's tooling is to be reused.

None of these are taken in this audit. The audit's purpose is to surface the scope so the corrections can be decided informationally rather than by tactical reaction.
