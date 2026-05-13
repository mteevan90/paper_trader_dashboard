# Larger Universe v2 — Gate 3 (b) Phase 4 report

**Date:** 2026-05-13
**Branch:** `feat/larger-universe-v2`
**Spec:** [docs/studies/larger_universe_v2/spec.md](spec.md)
**Baseline reproducibility:** bit-exact vs v1 ([baseline_reproducibility_check.md](baseline_reproducibility_check.md))

## Headline

**No variant passes all seven success criteria.** All seven variants — including baseline — are classified as `METHODOLOGY FINDING`. None get to `PROMOTE`. The `n_pass` count across variants:

| Variant | n_pass | Verdict |
|---|---|---|
| `b4_concentration_penalties` | **4/7** | METHODOLOGY FINDING |
| `baseline` | 3/7 | METHODOLOGY FINDING |
| `b2_conviction_weighted` | 3/7 | METHODOLOGY FINDING |
| `b5_defensive_sleeves` | 3/7 | METHODOLOGY FINDING |
| `b6_smaller_caps` | 3/7 | METHODOLOGY FINDING |
| `b1_vol_target` | 2/7 | METHODOLOGY FINDING |
| `b3_dynamic_topn` | 1/7 | METHODOLOGY FINDING |

`b4_concentration_penalties` is the only variant that beats baseline's pass count. It does so by passing criterion 6 (12-month rolling win rate, 65.08% vs baseline's 54.52%, threshold 60%). All other variants either tie or trail baseline.

## Per-criterion values per variant

All values reported regardless of pass/fail. Thresholds in parentheses.

### Criterion 1 — Std-dev reduction across walk-forward windows (≥ 20%)

| Variant | Value | Status |
|---|---:|---|
| baseline | +0.00% | FAIL (by construction; baseline vs itself = 0%) |
| b1_vol_target | **+41.57%** | **PASS** |
| b2_conviction_weighted | +6.97% | FAIL |
| b3_dynamic_topn | −38.82% | FAIL (variant std worse than baseline by 39%) |
| b4_concentration_penalties | +0.89% | FAIL |
| b5_defensive_sleeves | **+47.34%** | **PASS** |
| b6_smaller_caps | +0.33% | FAIL |

Only the two gross-exposure-reducing variants (B1 and B5) achieve meaningful std reduction. The construction-logic variants (B2, B3, B4) do not change cross-window consistency in a meaningful direction. B3 actively worsens consistency (its dynamic-N adds volatility, particularly in W6 where N drops aggressively).

### Criterion 2 — Positive-window count (≥ baseline 4)

| Variant | Value | Status |
|---|---:|---|
| baseline | 4/6 | PASS (trivially) |
| b1_vol_target | 3/6 | FAIL |
| b2_conviction_weighted | 4/6 | PASS |
| b3_dynamic_topn | **5/6** | PASS (most-consistent variant on this dimension) |
| b4_concentration_penalties | 4/6 | PASS |
| b5_defensive_sleeves | 4/6 | PASS |
| b6_smaller_caps | 4/6 | PASS |

B3 leads on regime consistency by this measure, but pairs it with the worst std (criterion 1) and worst test-window CAGR — its positive windows include the W6 outlier (+83% excess CAGR), without which the picture is less favorable. Lopsided distribution: high mean driven by one window, not steady performance.

### Criterion 3 — Mean CAGR giveback vs baseline (≤ 30%)

| Variant | Value | Status |
|---|---:|---|
| baseline | +0.00% (vs itself) | PASS (trivially) |
| b1_vol_target | +283.51% | FAIL (gave back nearly 3× of baseline's excess) |
| b2_conviction_weighted | **−34.67%** | PASS (improved on baseline by 34.7%) |
| b3_dynamic_topn | +100.61% | FAIL (gave back all of baseline's excess) |
| b4_concentration_penalties | **−37.98%** | PASS (improved on baseline by 38.0%) |
| b5_defensive_sleeves | +170.96% | FAIL (gave back +71% extra beyond baseline) |
| b6_smaller_caps | +0.00% | PASS (identical to baseline in test window) |

B2 and B4 are the only variants that genuinely improve test-window excess CAGR over baseline. B1, B3, B5 sacrifice substantial alpha. B6 is identical to baseline.

### Criterion 4 — Drawdown ratio (≤ 1.5× SPY MaxDD)

| Variant | Value | Status |
|---|---:|---|
| baseline | 1.7629× | FAIL |
| b1_vol_target | **1.3494×** | **PASS** |
| b2_conviction_weighted | 1.7495× | FAIL |
| b3_dynamic_topn | 1.7080× | FAIL |
| b4_concentration_penalties | 1.7208× | FAIL |
| b5_defensive_sleeves | **1.1564×** | **PASS** |
| b6_smaller_caps | 1.7629× | FAIL |

This criterion is essentially impossible to pass without gross-exposure reduction. Top-30 equal-weight construction on a high-beta universe produces ~1.7× SPY drawdown ratio regardless of selection logic. Only the exposure-reducing variants (B1, B5) pass.

### Criterion 5 — Max single-ticker alpha share (≤ 25%)

| Variant | Value | Status |
|---|---:|---|
| baseline | 32.45% | FAIL |
| b1_vol_target | 51.75% | FAIL |
| b2_conviction_weighted | 28.06% | FAIL (closest to threshold; misses by 3.06pp) |
| b3_dynamic_topn | 54.57% | FAIL |
| b4_concentration_penalties | 28.92% | FAIL |
| b5_defensive_sleeves | 29.22% | FAIL |
| b6_smaller_caps | 32.45% | FAIL |

**No variant passes criterion 5.** This is a universe-level or model-level concentration issue, not a construction-logic issue. The top alpha contributor accounts for >25% of total alpha in every variant. v1's results.md noted NVDA concentration as a known issue for that study; v2 inherits the same. Construction changes that explicitly fight ticker concentration (B4 persistence penalty, B6 smaller cap) reduce the share modestly but not below the threshold. B1 and B3 actively worsen it because their cap-binding edge cases concentrate weight in fewer names; B5's defensive sleeve only displaces equity allocation rather than reweighting it.

### Criterion 6 — 12-month rolling win rate vs SPY (≥ 60%)

| Variant | Value | Status |
|---|---:|---|
| baseline | 54.52% | FAIL |
| b1_vol_target | 23.37% | FAIL |
| b2_conviction_weighted | 58.79% | FAIL (closest miss at 1.21pp below) |
| b3_dynamic_topn | 35.43% | FAIL |
| b4_concentration_penalties | **65.08%** | **PASS** |
| b5_defensive_sleeves | 25.13% | FAIL |
| b6_smaller_caps | 54.52% | FAIL |

B4 is the only variant that passes criterion 6. B2's 58.79% is a close miss but a miss — at the pre-committed 60% threshold, 58.79% fails, period. The criterion was set before the study; the threshold is the threshold.

### Criterion 7 — Test excess CAGR > 0

| Variant | Value | Status |
|---|---:|---|
| baseline | +3.52% | PASS |
| b1_vol_target | −6.46% | FAIL |
| b2_conviction_weighted | +4.74% | PASS |
| b3_dynamic_topn | −0.02% | FAIL (effectively flat) |
| b4_concentration_penalties | +4.85% | PASS |
| b5_defensive_sleeves | −2.50% | FAIL |
| b6_smaller_caps | +3.52% | PASS |

The "beat SPY at all" floor. Three variants (B1, B3, B5) fail to clear it on the test window. B3 misses by a hair (effectively flat at −0.02pp).

## Walk-forward consistency stats per variant

Per-window excess CAGR vs SPY across the 6 walk-forward retrains:

| Variant | Mean | Std | Median | Min | Max | Positive | Strong (>5pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | +16.30% | 23.41% | +10.11% | −3.49% | +59.35% | 4/6 | 4/6 |
| b1_vol_target | +2.47% | 13.68% | +1.54% | −12.43% | +26.42% | 3/6 | 2/6 |
| b2_conviction_weighted | +3.59% | 21.77% | +7.06% | −33.78% | +31.59% | 4/6 | 4/6 |
| b3_dynamic_topn | +19.39% | 32.49% | +8.51% | −7.47% | +83.04% | 5/6 | 4/6 |
| b4_concentration_penalties | +16.37% | 23.20% | +9.93% | −2.21% | +59.35% | 4/6 | 4/6 |
| b5_defensive_sleeves | +4.10% | 12.32% | +1.78% | −8.63% | +24.51% | 4/6 | 2/6 |
| b6_smaller_caps | +16.56% | 23.33% | +10.72% | −3.13% | +59.35% | 4/6 | 4/6 |

Observations:

- **Baseline, B4, B6 cluster tightly.** Mean ~16%, std ~23%. B4 nudges mean up by 0.07pp and std down by 0.21pp vs baseline — essentially within noise.
- **B3 has the widest spread.** Std 32.5% (39% higher than baseline). The W6 +83.04% outlier dominates its mean; remove it and B3 looks much weaker.
- **B5 and B1 are the lowest-mean, lowest-std variants.** Both deliberately reduce gross exposure. Std reduction comes paired with mean reduction in roughly proportional fashion — vol-targeting trades return for risk, not free improvement.
- **B2's W6 outlier in the WRONG direction:** B2's W6 is −33.78% excess CAGR, the worst single-window result in the study. Conviction-weighting (concentrating weight in highest-score names) compounds badly when those names underperform — which they did in the most recent walk-forward year (2025-05 to 2026-05). B2's test-window result hides this.

## Methodology findings

These are findings worth preserving in the eventual writeup, separate from the variant-selection outcome.

### B6 (smaller individual caps) is a no-op in the test window

v1's baseline construction produces top-30 equal-weight at 1/30 ≈ 3.33% per position. B6's 4% individual cap is above this, so the cap never binds when construction completes within sector limits. Bit-equivalence with baseline in the test window:

- B6 test CAGR: 25.1396337% (same as baseline to all reported precision)
- B6 test excess: 3.5186%
- B6 test MaxDD: −33.4934%
- B6 test Sharpe: 0.9015

The test-window finding is **verified identical to baseline** at the criterion-evaluation level.

However, in walk-forward windows 1-3, B6 *does* diverge from baseline by 0.0019-0.0140 in excess CAGR per window. This is because sector-cap redistribution in those windows pushes some weights above 4%, which B6 then caps and redistributes again. The cap binds only when sector caps push weights upward; otherwise B6 is a no-op.

**Implication:** B6 fails to be a meaningful test of "smaller individual caps reduce concentration risk" at top-30 equal-weight in this universe. The hypothesis isn't falsified — it's untestable in this construction context because the cap doesn't bind. The natural follow-up question — does B6 become meaningful when combined with a non-equal-weight construction like B2 — is **explicitly out of scope for v2**. Combinations require Gate 4 evidence of multiple single-variable winners plus Mike's explicit approval, neither of which apply here.

### B1 (vol-target) is highly sensitive to warmup vol level

The training-tail vol computed at study config time is 31.62% annualized (last 63 trading days of training period 2017-05-12 to 2023-05-11). This is unusually high because the training period spans several high-vol regimes (COVID 2020, 2022 bear market). At a 15% target vs 31.62% warmup, B1 starts the test window at gross exposure ≈ 47% and stays scaled-down for the first 63 days until realized test-period vol overtakes the frozen warmup.

The result is honest given the pre-committed Gate 1 specification ("B1 warmup uses frozen training-tail vol"). We don't retroactively change the warmup design because the result is unfavorable — that's goalpost-moving. The result tells us:

- Vol-targeting strategies are sensitive to warmup design.
- Frozen training-tail vol creates conservative initial allocation when training-period realized vol exceeds test-period realized vol.
- In a study with multiple regimes in the training window, the last-63-day slice may not be representative.

A future Mechanism B study that includes vol-targeting would need to think carefully about warmup alternatives (longer median window, regime-adjusted warmup, EWMA decay). This is a **v3 or later** consideration. For v2, B1's result is as observed: −6.46pp test excess CAGR, fails criterion 7 and 3.

### Criterion 5 fails universally — universe/model-level concentration

No variant passes the 25% single-ticker alpha-share threshold. Construction-logic variants reduce the share marginally (B2 28.1%, B4 28.9%, B5 29.2%) but not enough. B1 and B3 actually worsen concentration because they tend to concentrate weight in fewer names when their respective rules activate.

This is a universe-level finding from v1 carried into v2 unchanged. v1's results.md noted NVDA concentration as the known issue; v2 confirms it persists across all seven construction variants. Construction changes alone cannot address this — the underlying alpha is concentrated in a small number of names that drive most excess return.

Approaches that could address this — universe filtering, sector neutralization at the score level, signal extraction (mechanism A in the v2 scoping document) — are **out of scope for v2** by design.

### Criterion 4 is structurally bound to gross exposure

Only B1 and B5 (the gross-exposure-reducing variants) pass the 1.5× drawdown ratio. All long-only top-30 equal-weight variants land at 1.71-1.76× drawdown ratio regardless of selection logic. This is consistent with a high-beta universe (predominantly small/mid cap) where equal-weight concentration in growth names produces structural drawdown amplification vs SPY.

**Implication:** Criterion 4 is hard to pass for any construction-only variant. Future studies that aim to pass it likely need to combine selection logic with explicit risk-budgeting (vol targeting, defensive sleeves, dynamic gross exposure).

### B3's W6 outlier dominates its summary stats

B3's W6 (val 2025-05-12 to 2026-05-11) produced +83.04% excess CAGR. This is more than 1.5× the next-best variant in that window (baseline 59.35%, b4 59.35%, b6 59.35%). B3's mean walk-forward excess of +19.39% is anchored by this outlier — remove W6 and B3's mean drops to ~−6%.

The outlier comes from a window where the model's signal happened to be strong AND the score-dispersion percentile fell in B3's "concentrate aggressively" zone (N drops toward 15). This is the construction logic working as designed, but the resulting concentration paid off in this specific window in a way that may not generalize.

**Not flagged as bug; flagged as caveat for any future writeup that highlights B3's mean walk-forward excess.** The mean is real but driven by one window's tail event, not steady performance.

## Unexpected behavior flags

Per the Gate 3 deliverables checklist: any variant that produced unexpected behavior (extreme drawdowns, allocation failures, NaN outputs, error states).

### Resolved during the run

- **B3 RuntimeError at first rebalance.** B3's `construct()` raised `RuntimeError("B3 requires training_dispersion_dist")` because the variant requires frozen warmup state and the initial runner version didn't bootstrap it. Caught and fixed in commit `0d5a537` (`gate3(v2): bootstrap training-period warmup state for B1 and B3`); re-run produced clean results. The fix added a `_compute_warmup_state` helper to phase4_run_v2.py and a per-window equivalent to phase5_walk_forward_v2.py.
- **JSON serialization of `numpy.bool_`.** Baseline run initially failed at the parity-check JSON write because newer numpy versions don't serialize `numpy.bool_` via `json.dumps`. Caught and fixed in commit `9bde239`.

### Persistent (not bugs; honest results)

- **B2 W6 catastrophic miss:** B2's W6 walk-forward result is −33.78% excess CAGR. Worst single-window result in the study. Driven by conviction-weighted concentration in the highest-score names underperforming sharply. B2's test-window result is favorable, but this W6 result is a real risk pattern — softmax weighting compounds losses on top names.
- **B6 no-op in test window:** Documented in the methodology findings section. Worth surfacing because the original spec contemplated B6 as a "concentration reduction" treatment, and the construction doesn't actually exercise that treatment on this universe + N + sector cap combination.

No unexpected NaN outputs, allocation failures, or score-pipeline drift detected.

## Reproducibility verification (rolled forward from Gate 3 (a))

The Gate 3 (a) baseline reproducibility check confirmed v2-baseline reproduces v1 bit-for-bit:

- Scores parity vs v1 scores.parquet: 0.0 max abs diff across 59,232 (date, ticker) pairs
- All 7 gating headline metrics: 0.0 abs diff vs v1 pinned reference
- Full 739-row NAV series: bit-identical

This rules out pipeline drift as a source of any variant's deviation from baseline. All differences in B1-B6 results vs baseline attribute cleanly to construction-logic differences alone.

A secondary check during Gate 3 (b): v2-baseline walk-forward excess CAGR per window (+0.0707, +0.1316, −0.0349, +0.2390, −0.0219, +0.5935) matches v1's pinned `walk_forward.parquet` for XGBoost to all reported precision.

## What Gate 3 (b) does NOT establish

In keeping with honest framing — calling out what we *don't* know is as important as what we do.

- **Whether any variant would pass criterion 5 with sector neutralization, universe filtering, or signal extraction.** v2's scope was construction-logic-only; the universe and model are held constant. Approaches that could improve criterion 5 are out of v2's scope.
- **Whether B4 (the leader at 4/7) would pass all 7 with a longer test window.** The 650-day test window is what was pre-committed; we evaluate against it. Out-of-sample generalization is not yet tested (reserved validation period covers only ~89 days post-2026-01-01, insufficient for annualized metrics per the spec).
- **Whether multiple-variant combinations would pass.** Out of v2's scope by design.
- **Whether different criterion thresholds would change the verdict landscape.** The thresholds were pre-committed at Gate 1. Re-evaluating with different thresholds would violate the pre-commitment.

## What's next

Per spec and process:

1. **Mike's review of this report.** Don't auto-proceed to Gate 4.
2. **Gate 4 (if approved):** Phase 5 analytics per variant — full per-ticker attribution, decile analysis, IC decomposition, rolling win rate distributions, concentration_summary.json. These produce the artifact tree required for the dashboard's Variant Comparison tab.
3. **Gate 5 (if approved):** Writeup + dashboard contract extension + Variant Comparison tab + variant selector wiring + tracker update.

Without a PROMOTE variant, the eventual writeup will document v2 as a methodology study with these findings:
- v2's construction-logic search does not produce a deployable strategy on this universe + model + dates.
- B4 leads on the criteria but doesn't clear them all.
- Universe-level and risk-structural issues (criteria 4, 5) cannot be addressed by construction logic alone.
- v3 (or later) candidate directions are signal-extraction (mechanism A from v2 scoping), sector-neutral scoring, or universe filtering. These are *not* in v2's scope and should not be pursued as part of v2.

The dashboard's "honest framing footer" pattern from v1 applies: v2 study is documented, not promoted. The track record stays accurate.

## Artifacts produced

Code (in git):
- `docs/studies/larger_universe_v2/spec.md` (`54c286e`)
- `scripts/research/phase4_run_v2.py` (`ce8dfdd` + `9bde239` + `0d5a537`)
- `scripts/research/phase5_walk_forward_v2.py` (`ce8dfdd` + `0d5a537`)
- `scripts/research/build_comparison_results_v2.py` (`ce8dfdd`)

Data (gitignored per `models/*` convention; local + R2):
- `models/studies/larger_universe_v2/variant_meta.json` (study-level metadata + optional_artifacts entry for comparison_results)
- `models/studies/larger_universe_v2/_scores_parity_vs_v1.json` (upstream parity check log)
- `models/studies/larger_universe_v2/<variant>/contract_v1/{portfolio,holdings,trades,scores,benchmarks,walk_forward}.parquet` and `meta.json` for each of the 7 variants
- `models/studies/larger_universe_v2/comparison/comparison_results.parquet` (this report's source)

Docs (in git):
- `docs/studies/larger_universe_v2/baseline_reproducibility_check.md` (Gate 3 (a) writeup)
- `docs/studies/larger_universe_v2/gate3_phase4_report.md` (this file)
- `docs/sessions/larger_universe_v2/session_log.md` (Gate 3 (a) + (b) entries)
