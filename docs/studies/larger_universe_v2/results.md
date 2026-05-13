# Larger Universe v2 — Study results

**Branch:** `feat/larger-universe-v2`
**Snapshot:** `models/snapshots/equities/larger_universe_v1_20260511/` (reused from v1)
**Spec:** [`docs/studies/larger_universe_v2/spec.md`](spec.md)
**Dashboard:** `/studies/larger_universe_v2` (contract-conformant, see `docs/architecture/dashboard_contract_v1.md`)
**Status:** Complete. **No variant promoted; all are documented as methodology findings.** See "Verdict" below.

## TL;DR

v2 tested six portfolio-construction variants against pre-committed criteria; **no variant met all seven, and all are documented as methodology findings**. An IC scope audit during Gate 4 revealed that the v1 XGBoost model — which v2 holds fixed by design — has near-zero cross-sectional signal under standard full-cross-section definitions on the test window (top-quintile IC **−0.004**, full IC **−0.009**, non-monotonic decile structure). v1's observed +3.5pp excess CAGR on the test window is **consistent with tail-driven returns rather than durable cross-sectional ranking skill**; the model's actual signal pattern across walk-forward windows shows substantial regime variance that v2 cannot disentangle into "real skill" versus "lucky tail events" from this data alone. The universal failures on criterion 5 (alpha concentration) and criterion 4 (drawdown ratio) are consequences of this signal structure rather than failures of construction logic — cross-variant Spearman correlation of **0.94** on top alpha contributors confirms concentration is model-determined and not construction-specific, and only gross-exposure-reducing variants (B1, B5) pass criterion 4. Together these demonstrate that the binding constraint for top-N equity strategies on this universe is **signal extraction (Mechanism A), not portfolio construction (Mechanism B)** that v2 actually tested.

## Executive summary

- **No variant promoted.** All seven (baseline + B1–B6) classified as `METHODOLOGY FINDING`. B4 (concentration penalties) leads on `n_pass` count at 4/7; all others tie or trail baseline at 3/7 or below.
- **Substantive v2 contribution is upstream of construction.** An IC scope audit triggered by v2's Gate 4 analytics surfaced that v1's reported XGBoost top-quintile IC of +0.048 was computed at held-subset scope (450 tickers), not the standard full eligible universe (1,963 tickers). Under standard scope: **top-quintile IC −0.004, full IC −0.009, non-monotonic decile structure**. The model has effectively zero cross-sectional ranking signal on the test window.
- **v1's +3.5pp excess CAGR is consistent with tail-driven returns**, not durable cross-sectional ranking skill. The IC near-zero finding combined with positive cumulative excess CAGR implies the excess comes from rare large positive events on top names rather than from reliable winner-selection across rebalances. The "real-skill plus tail-effects" alternative is logically possible but not distinguishable from "all-tail-driven" given this data.
- **C5 (alpha concentration) failure is model-determined, not construction-specific.** Cross-variant Spearman correlation of 0.94 on per-ticker `pct_of_total_alpha` over the union of all 435 contributing tickers; 14 of 28 unique top-20 contributors appear in ALL seven variants. The same names dominate alpha regardless of construction logic — construction-side interventions (caps, conviction weighting, dynamic N, penalties) do not redistribute concentration meaningfully.
- **C4 (drawdown ratio) failure is structural to long-only fully-invested equity.** All variants that stay fully invested land at 1.71–1.76× SPY MaxDD regardless of selection logic. The only variants that pass C4 (B1 vol-target at 1.35×, B5 defensive sleeves at 1.16×) do so by deploying less capital, not by deploying capital better.
- **B6 (smaller individual caps) is a no-op in the test window.** v1's top-30 equal-weight at 1/30 ≈ 3.33% per position is already below B6's 4% cap; the cap does not bind on this construction in this date range. B6 is bit-identical to baseline in the test window. The hypothesis "smaller caps reduce concentration risk" is untestable at top-30 equal-weight in this universe.
- **B4 (concentration penalties) improves on different binding constraints than v2 was designed to test.** B4 improves 12-month rolling win rate (65.1% vs baseline 54.5%) and slight mean excess CAGR — but walk-forward std barely moves (23.2% vs baseline 23.4%). The persistence/sector penalties produce within-window consistency improvement, not the regime-variance reduction v2 was scoped against.
- **Baseline reproducibility was bit-exact.** v2-baseline reproduces v1's pinned headline metrics across all 7 gating metrics (n_days, total_return, cagr, spy_cagr, excess_cagr, max_drawdown, spy_max_drawdown) and the full 739-row NAV series to floating-point identity. Construction-variant code and the backtest engine refactor preserve v1's pipeline behavior.

## What v2 set out to test

v1 surfaced two methodology findings that motivated v2's scoping:
1. v1's top-N portfolio had **inconsistent excess returns across walk-forward windows** (4 of 6 positive for XGBoost; standard deviation of 23.4pp across windows). The strategy beat SPY on average but not in any reliable per-window sense.
2. v1's signal extraction and v1's portfolio construction were intermingled in a way that did not isolate the source of regime instability.

The v2 scoping document framed two mechanisms that could plausibly improve regime consistency:
- **Mechanism A — signal extraction.** Change the model class, features, CV objective, or universe to extract a stronger or differently-distributed underlying signal.
- **Mechanism B — portfolio construction.** Hold the v1 model fixed (same XGBoost, same hyperparameters, same features, same universe, same dates, same CV objective) and vary only how scores translate to portfolio weights.

v2 chose Mechanism B explicitly. Mechanism A was parked as a v3 candidate. The choice was deliberate: Mechanism B is a smaller, more controllable change with a tighter causal interpretation (if regime variance improves, it must be from construction logic, not signal changes).

### Seven variants (full spec: [spec.md](spec.md))

- **baseline** — v1 construction unchanged (rank top-30 equal-weight, 7.5% individual cap, 30% sector cap)
- **B1 vol_target** — vol-targeting overlay, 15% annualized via 63-day realized vol
- **B2 conviction_weighted** — softmax weighting within top-30, T=0.5
- **B3 dynamic_topn** — N varies 15–50 based on top-decile score dispersion
- **B4 concentration_penalties** — persistence penalty + sector overweight penalty
- **B5 defensive_sleeves** — 70/30 equity/defensive normal, 50/50 in SPY-21d < −5% stress
- **B6 smaller_caps** — 4% individual cap (vs baseline's 7.5%)

### Seven pre-committed success criteria

Three relative-to-baseline:
- **C1** Std-dev reduction across walk-forward windows ≥ 20%
- **C2** Positive-window count ≥ baseline (4/6)
- **C3** Mean CAGR giveback ≤ 30%

Four v1-promotion-style absolute:
- **C4** Drawdown ratio vs SPY ≤ 1.5×
- **C5** Max single-ticker alpha share ≤ 25%
- **C6** 12-month rolling win rate vs SPY ≥ 60%
- **C7** Test excess CAGR > 0

A variant must meet ALL seven to promote. Variants that meet some but not all are documented as methodology findings.

## Verdict

| Variant | n_pass | Verdict | Pass-detail (C1 C2 C3 C4 C5 C6 C7) |
|---|---:|---|---|
| `b4_concentration_penalties` | **4** | METHODOLOGY FINDING | F P P F F **P** P |
| `baseline` | 3 | METHODOLOGY FINDING | F P P F F F P |
| `b2_conviction_weighted` | 3 | METHODOLOGY FINDING | F P P F F F P |
| `b5_defensive_sleeves` | 3 | METHODOLOGY FINDING | **P** P F **P** F F F |
| `b6_smaller_caps` | 3 | METHODOLOGY FINDING | F P P F F F P (= baseline) |
| `b1_vol_target` | 2 | METHODOLOGY FINDING | **P** F F **P** F F F |
| `b3_dynamic_topn` | 1 | METHODOLOGY FINDING | F P F F F F F |

**No variant achieves PROMOTE.** B4 exceeds baseline by passing C6 (12-month rolling win rate 65.1% vs baseline 54.5%, threshold 60%); all other variants either tie baseline or trail.

## What the data showed

### Per-criterion values per variant

Values reported regardless of pass/fail. Thresholds parenthesized.

**C1 — Std-dev reduction across walk-forward windows** (≥ 20%):

| Variant | Value | Pass |
|---|---:|---:|
| baseline | +0.00% | F (by construction; baseline vs itself = 0%) |
| b1_vol_target | **+41.57%** | **P** |
| b2_conviction_weighted | +6.97% | F |
| b3_dynamic_topn | −38.82% | F (variant std *worse* than baseline by 39%) |
| b4_concentration_penalties | +0.89% | F |
| b5_defensive_sleeves | **+47.34%** | **P** |
| b6_smaller_caps | +0.33% | F |

Only the gross-exposure-reducing variants (B1, B5) achieve meaningful std reduction. Construction-logic variants (B2, B3, B4) do not change cross-window consistency in a meaningful direction. B3 actively worsens consistency.

**C2 — Positive-window count** (≥ baseline 4/6):

| Variant | Value | Pass |
|---|---:|---:|
| baseline | 4/6 | P (trivially) |
| b1_vol_target | 3/6 | F |
| b2_conviction_weighted | 4/6 | P |
| b3_dynamic_topn | **5/6** | P (most consistent on this dimension) |
| b4_concentration_penalties | 4/6 | P |
| b5_defensive_sleeves | 4/6 | P |
| b6_smaller_caps | 4/6 | P |

B3 leads here but pairs it with the worst std (C1) and worst test-window CAGR. Its positive windows include the W6 outlier (+83% excess); without W6 the picture is less favorable. Lopsided distribution: high mean driven by one window, not steady performance.

**C3 — Mean CAGR giveback vs baseline** (≤ 30% giveback):

| Variant | Value | Pass |
|---|---:|---:|
| baseline | +0.00% | P (trivially; baseline vs itself) |
| b1_vol_target | +283.51% | F |
| b2_conviction_weighted | **−34.67%** | P (improved over baseline by 34.7%) |
| b3_dynamic_topn | +100.61% | F |
| b4_concentration_penalties | **−37.98%** | P (improved over baseline by 38.0%) |
| b5_defensive_sleeves | +170.96% | F |
| b6_smaller_caps | +0.00% | P (identical to baseline in test window) |

B2 and B4 are the only variants that genuinely improve test-window excess CAGR over baseline. B1, B3, B5 sacrifice substantial alpha.

**C4 — Drawdown ratio vs SPY** (≤ 1.5×):

| Variant | Value | Pass |
|---|---:|---:|
| baseline | 1.7629× | F |
| b1_vol_target | **1.3494×** | **P** |
| b2_conviction_weighted | 1.7495× | F |
| b3_dynamic_topn | 1.7080× | F |
| b4_concentration_penalties | 1.7208× | F |
| b5_defensive_sleeves | **1.1564×** | **P** |
| b6_smaller_caps | 1.7629× | F |

Essentially impossible to pass without gross-exposure reduction. Long-only fully-invested top-30 equal-weight on a high-beta universe lands at 1.7× SPY MaxDD ratio regardless of selection logic.

**C5 — Max single-ticker alpha share** (≤ 25%):

| Variant | Value | Pass |
|---|---:|---:|
| baseline | 32.45% | F |
| b1_vol_target | 51.75% | F |
| b2_conviction_weighted | 28.06% | F (closest miss at 3.06pp over) |
| b3_dynamic_topn | 54.57% | F |
| b4_concentration_penalties | 28.92% | F |
| b5_defensive_sleeves | 29.22% | F |
| b6_smaller_caps | 32.45% | F |

**No variant passes criterion 5.** Detailed analysis in §"The signal-weakness finding (the headline)" below.

**C6 — 12-month rolling win rate vs SPY** (≥ 60%):

| Variant | Value | Pass |
|---|---:|---:|
| baseline | 54.52% | F |
| b1_vol_target | 23.37% | F |
| b2_conviction_weighted | 58.79% | F (closest miss at 1.21pp under) |
| b3_dynamic_topn | 35.43% | F |
| b4_concentration_penalties | **65.08%** | **P** |
| b5_defensive_sleeves | 25.13% | F |
| b6_smaller_caps | 54.52% | F |

B4 is the only variant that passes C6. B2's 58.79% is a close miss; at the pre-committed 60% threshold it fails, period. The criteria were set before the study; the threshold is the threshold.

**C7 — Test excess CAGR > 0**:

| Variant | Value | Pass |
|---|---:|---:|
| baseline | +3.52% | P |
| b1_vol_target | −6.46% | F |
| b2_conviction_weighted | +4.74% | P |
| b3_dynamic_topn | −0.02% | F (effectively flat) |
| b4_concentration_penalties | +4.85% | P |
| b5_defensive_sleeves | −2.50% | F |
| b6_smaller_caps | +3.52% | P |

The "beat SPY at all" floor. Three variants (B1, B3, B5) fail to clear it on the test window. B3 misses by a hair (effectively flat at −0.02pp).

### Walk-forward consistency stats per variant

Per-window excess CAGR vs SPY across 6 walk-forward retrains:

| Variant | Mean | Std | Median | Min | Max | Pos. | Strong (>5pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | +16.30% | 23.41% | +10.11% | −3.49% | +59.35% | 4/6 | 4/6 |
| b1_vol_target | +2.47% | 13.68% | +1.54% | −12.43% | +26.42% | 3/6 | 2/6 |
| b2_conviction_weighted | +3.59% | 21.77% | +7.06% | **−33.78%** | +31.59% | 4/6 | 4/6 |
| b3_dynamic_topn | +19.39% | 32.49% | +8.51% | −7.47% | **+83.04%** | 5/6 | 4/6 |
| b4_concentration_penalties | +16.37% | 23.20% | +9.93% | −2.21% | +59.35% | 4/6 | 4/6 |
| b5_defensive_sleeves | +4.10% | 12.32% | +1.78% | −8.63% | +24.51% | 4/6 | 2/6 |
| b6_smaller_caps | +16.56% | 23.33% | +10.72% | −3.13% | +59.35% | 4/6 | 4/6 |

Baseline, B4, B6 cluster tightly (mean ~16%, std ~23%). B4 nudges mean by 0.07pp and std by 0.21pp vs baseline — essentially within noise. B3 has the widest spread (std 32.5%, 39% higher than baseline); its W6 +83.04% outlier dominates its mean. Removing W6 drops B3's mean to ≈−6%. B5 and B1 are the lowest-mean, lowest-std variants — vol-targeting and defensive allocation trade return for risk in roughly proportional fashion. B2's W6 = −33.78% excess CAGR is the worst single-window result in the study; conviction-weighting compounds losses when top-score names underperform, which happened in 2025-2026.

## The signal-weakness finding (the headline)

This is the substantive v2 contribution. The IC scope audit ([`ic_scope_audit.md`](../larger_universe_v1/ic_scope_audit.md)) was triggered by v2's Gate 4 analytics surfacing a discrepancy with v1's pinned numbers, and revealed a finding that revises how to read v1's model behavior under standard cross-sectional definitions.

### What the audit found

v1 reported XGBoost top-quintile IC of +0.0481, prominently cited in `docs/architecture/ml_study_cv_objectives_v1.md` and `docs/studies/larger_universe_v1/results.md`. v2's `phase5_analytics_v2.py` computed the same metric using the same formula on bit-identical scores and produced **−0.0041** — sign reversal.

Investigation traced the discrepancy to v1's `phase5_analytics.load_inputs()`, which loads prices only for tickers in `holdings["ticker"].unique()` (~450 across XGB + ENet) rather than the full eligible universe (~1,963 tickers). v1's "top-quintile IC" was effectively the correlation between scores and forward returns within the held-tickers cross-section, not the standard top-quintile-of-full-eligible-universe IC. The formulas are identical; the price-universe scope differs. The audit reproduced both numbers at three different scopes (held-450, XGB-only-340, full-1963) confirming that scope alone explains the difference. Reproduction of pinned v1 values bit-for-bit at the 450-ticker scope; reproduction of v2's full-universe values bit-for-bit at the 1,963-ticker scope.

### Standard-definition IC values on v1's test window (XGBoost)

| Metric | v1 pinned (held-subset) | Standard (full universe) |
|---|---:|---:|
| Full IC mean | −0.008740 | −0.008855 |
| **Top-quintile IC mean** | **+0.048121** | **−0.004134** |
| Top-quintile IC std | 0.113320 | 0.078111 |

The full IC barely changes (already negative). The top-quintile IC flips sign — slightly anti-predictive under standard scope.

### Decile structure on v1's test window (XGBoost, full universe scope)

Per-decile mean 21-day forward return:

| Decile | Mean fwd return | Std | n rebalances |
|---|---:|---:|---:|
| 1 (lowest) | **+5.79%** | 24.77% | 37 |
| 2 | +0.83% | 4.91% | 37 |
| 3 | +0.98% | 5.08% | 37 |
| 4 | +1.01% | 5.03% | 37 |
| 5 | +1.21% | 5.04% | 37 |
| 6 | +1.14% | 4.87% | 37 |
| 7 | +0.89% | 4.86% | 37 |
| 8 | +0.95% | 4.93% | 37 |
| 9 | +1.08% | 4.91% | 37 |
| 10 (highest) | +1.21% | 5.87% | 37 |

Decile 1 has the highest mean forward return (with the highest std). Decile 10 (highest scored) is tied with decile 5 at +1.21%. The model's top-decile does NOT systematically outperform its bottom-decile on average. Deciles 2-9 are bunched between +0.83% and +1.21% with no clear monotonic ordering.

This is the same general structure that v1's `ml_study_cv_objectives_v1.md` described as "no monotonic structure D2-D10" — but the v1 artifact's pinned decile-1 mean was +35.66% (held-subset, ~5 tickers per rebalance, small-sample tail-driven). Under standard scope, decile 1 is still highest but at +5.79% rather than +35.66% — a realistic magnitude with interpretable std.

### Logical chain from near-zero IC to "consistent with tail-driven returns"

The chain:
1. **Cross-sectional IC near zero** means the model's score-to-forward-return ranking is essentially random across the eligible universe on any given rebalance date.
2. **Yet baseline produced +3.5pp excess CAGR.** If ranking is random per-date, where does the positive cumulative excess come from?
3. **A small set of names drives most of it.** v1's per-ticker attribution (and v2's, which reproduces the same pattern) shows that 5-10 names account for a substantial fraction of total alpha. The top contributor accounts for ~32% (baseline) up to ~55% (B3) of total alpha.
4. **Combined, these point to tail-driven returns.** If the model's ranking is near-random on average but a handful of names contribute most of the cumulative excess, those names must be cases where the model happened to rank them highly when they were about to experience large positive returns. The model is not reliably picking winners; it is occasionally selecting names that experience tail events.

### The alternative interpretation (acknowledged but not distinguishable)

"Real regime-dependent skill plus tail effects" is a logically possible alternative. The walk-forward window distribution does show substantial regime variance: XGBoost's per-window excess CAGR ranges from −3.49pp (W3) to +59.35pp (W6), with the model having distinctly better windows (W4, W6) and distinctly worse ones (W3, W5).

A "real skill plus tail" interpretation would say: the model has positive cross-sectional skill in some regimes (high vol, growth-rotation, AI rally) and negative or zero in others. In the good-regime years, the IC is meaningfully positive; in the bad-regime years, negative. The TEST window happens to span both regime types so the average IC washes out to near-zero, but the model still has real per-regime skill that produces cumulative excess CAGR through the good windows.

The pure "tail-driven" interpretation says: the model has near-zero ranking skill across all regimes, but in some windows it happens to land on a few names that experience large positive idiosyncratic events, and those events compound into cumulative excess CAGR. The model is not picking up regime structure — it is occasionally getting lucky.

**v2's data cannot cleanly distinguish these.** Both produce the observed pattern of near-zero average IC + positive cumulative excess + regime variance + concentrated alpha attribution. Distinguishing them would require more granular evidence — e.g., per-regime IC, per-regime decile structure, isolation of which names the model selects in which regimes vs random selection, controlled comparison against a known-zero-skill baseline.

The writeup does NOT claim the model is provably skill-free. It claims that **on this data, under standard cross-sectional definitions, the test-window IC does not provide evidence of durable ranking skill**, and that whether v1's observed +3.5pp excess CAGR is "skill plus tail" or "all tail" cannot be answered from v2's analytics alone.

### Why this is the substantive v2 finding

If the underlying signal has near-zero standard-definition IC on the test window:
- **Construction logic cannot redistribute alpha that doesn't exist as a stable ranking property.** No matter how you slice or weight the scores, you cannot extract durable cross-sectional alpha from a near-zero-IC signal. The construction-side variants would produce non-identical results because cap binding, weighting, and N choices reshuffle, but no construction can extract durable ranking-based alpha from a near-zero-IC signal.
- **Concentration becomes inevitable.** If alpha is tail-driven (or even partially tail-driven), it concentrates on the few names that experienced the tails. This holds regardless of construction logic — the construction varies which weights those names get, not whether they dominate alpha attribution.
- **Drawdown becomes structural.** Without durable ranking skill to differentiate high-drawdown from low-drawdown names within the top-N, the strategy inherits the universe's high-beta drawdown structure. Lowering drawdown requires deploying less capital (which v2's B1 and B5 demonstrate).

These three points are the consequences the headline draws. They explain why C5 and C4 failed universally and why B4's marginal improvements don't scale into a promotion.

## The cross-variant concentration overlap finding

A separate Gate 4 analysis directly tests whether C5 failure is model-determined (same names dominate across variants) or construction-specific (different variants concentrate alpha on different names).

### What the analysis did

For each variant, take the top-20 alpha contributors (by `pct_of_total_alpha` from per-ticker attribution). Compute the cross-variant Spearman correlation matrix of `pct_of_total_alpha` over the union of all 435 contributing tickers. Count how many variants' top-20 sets each ticker appears in.

### What the analysis found

**Cross-variant Spearman correlation (off-diagonal mean across all pairs): 0.9435.** Median: 0.9687.

Pairwise correlation matrix (XGBoost):

|  | baseline | B1 | B2 | B3 | B4 | B5 | B6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.000 | 0.976 | 0.994 | 0.860 | 0.969 | 0.995 | 1.000 |
| B1 | 0.976 | 1.000 | 0.971 | 0.835 | 0.963 | 0.968 | 0.976 |
| B2 | 0.994 | 0.971 | 1.000 | 0.861 | 0.969 | 0.990 | 0.994 |
| B3 | 0.860 | 0.835 | 0.861 | 1.000 | 0.851 | 0.859 | 0.860 |
| B4 | 0.969 | 0.963 | 0.969 | 0.851 | 1.000 | 0.962 | 0.969 |
| B5 | 0.995 | 0.968 | 0.990 | 0.859 | 1.000 | 0.995 | — |
| B6 | 1.000 | 0.976 | 0.994 | 0.860 | 0.969 | 0.995 | 1.000 |

Most variant pairs at 0.96-1.00. B3 is the outlier at ~0.85 vs others, because its dynamic-N reshuffles holdings more than the other variants — but its top contributors still overlap with baseline's by ~12 of 20.

**Appearance distribution of top-20 alpha contributors across the 7 variants:**

| In N variants' top-20 | # tickers |
|---:|---:|
| 7 (all variants) | **14** |
| 6 | 3 |
| 5 | 2 |
| 4 | 1 |
| 3 | 0 |
| 2 | 2 |
| 1 | 6 |

**14 of 28 unique top-20 tickers appear in ALL seven variants' top-20.** Half of the highest-alpha contributors are the same names regardless of construction.

### What this means for the C5 failure interpretation

The C5 universal failure is **not** "each variant happens to fail criterion 5 because of its specific construction choices producing different concentrations." It is "the same names dominate alpha across all variants because the model concentrates its signal on a small set of names." Construction-side changes — caps that try to spread weight, penalties that try to avoid overweighting, smaller N that should concentrate but doesn't, larger N that should diversify — do not meaningfully redistribute which names drive alpha.

This is the direct empirical support for the headline claim that "concentration risk in this universe at this horizon is not construction-addressable."

## Methodology findings from v2

These are findings worth surfacing for future studies. They are NOT drafted as standalone architectural memos in this gate — that's a post-v2 workstream decision. They are surfaced here so the v2 record includes them.

### Finding 1 — Limits of portfolio construction for fixing model-level concentration

When alpha is concentrated on a small set of names regardless of construction logic, construction-side interventions cannot address the concentration. The cross-variant Spearman correlation of 0.94 on top alpha contributors and the 14-of-28-shared top-20 finding are direct evidence. Combined with the IC near-zero finding, this generalizes: top-N portfolio construction strategies operating on signals with weak cross-sectional ranking properties will have model-determined concentration that cannot be construction-addressed.

Candidate for an architectural memo `docs/architecture/portfolio_construction_concentration_limits_v1.md`. Not drafted in this gate.

### Finding 2 — Long-only fully-invested drawdown ceiling

In a long-only fully-invested top-N equity strategy on a high-beta universe, drawdown vs benchmark is structurally bound to roughly 1.7× SPY MaxDD regardless of selection logic. v2's evidence: five fully-invested variants (baseline, B2, B3, B4, B6) all land at 1.71-1.76× SPY MaxDD. The only variants that pass criterion 4 (B1 at 1.35×, B5 at 1.16×) do so by reducing gross exposure (B1 via vol-targeting scaling, B5 via defensive sleeve allocation). The mechanism: concentrated equity exposure on a high-beta universe inherits the universe's drawdown profile. Diversifying within the universe doesn't help if the universe itself is high-beta in stress regimes.

Candidate for an architectural memo `docs/architecture/long_only_drawdown_ceiling_v1.md`. Not drafted in this gate.

### Finding 3 — Signal extraction (Mechanism A) vs portfolio construction (Mechanism B) as binding constraints

v2's Mechanism B test combined with the IC scope audit's revelation about Mechanism A signal weakness produces a clean binding-constraint demonstration. For top-N equity strategies on this universe at this horizon:

- **Mechanism B (construction)** is not the binding constraint. v2 tested six distinct construction approaches; none promoted; the variant that came closest (B4) improved on different axes than the regime-variance v2 was designed to test.
- **Mechanism A (signal)** is the apparent binding constraint. The model has near-zero standard-definition IC on the test window. Without durable cross-sectional ranking skill at the signal layer, no construction-side intervention can extract durable alpha.

Candidate for an architectural memo `docs/architecture/mechanism_a_b_binding_constraint_v1.md`. Not drafted in this gate.

### Finding 4 — IC computation scope (audit-derived)

v1's IC and decile analytics were computed at held-subset scope (450 tickers) rather than full-cross-section scope (1,963 tickers), producing a top-quintile IC of +0.048 that does not survive correction to the standard definition (where it is −0.004). v2's `phase5_analytics_v2.py` computes at full-cross-section by default, and the `artifact_metadata` schema addition (`docs/architecture/dashboard_contract_v1.md`) lets future studies declare scope explicitly so this class of mismatch is surfaced rather than buried.

Already documented in [`ic_scope_audit.md`](../larger_universe_v1/ic_scope_audit.md). Not drafted as a standalone architectural memo because the audit itself functions as the canonical record.

## Honest framing — what v2 does NOT establish

In keeping with the honest-framing discipline:

- **Whether the v1 XGBoost model has skill in other windows, on other universes, or at other horizons.** v2 holds v1's model fixed by design and tests construction logic on the same test window. The IC finding applies to v1's model on v1's test window under standard scope. Other windows, universes, or horizons require their own analysis.
- **Whether the signal-weakness interpretation is "all tail-driven" or "real regime-dependent skill plus tails."** The data cannot cleanly distinguish these. The writeup adopts the more cautious "consistent with tail-driven returns" framing and explicitly notes the alternative as logically possible.
- **Whether v3 (Mechanism A) will succeed.** v2's findings strongly suggest signal extraction is the binding constraint, but they don't establish that any specific Mechanism A approach (different features, different model class, sector neutralization, etc.) will produce a deployable signal. v3 scoping is a separate conversation.
- **Whether the CV-objectives architectural memo's recommendation survives correction.** The memo's empirical evidence (held-subset top-quintile IC of +0.048) is scope-restricted; under standard scope the number is −0.004. Whether the memo's logical argument (operating-region matching for top-N CV) still holds without the +0.048 evidence is a re-justification question that v2 does NOT close. The memo's correction section explicitly defers this re-justification as a follow-up.
- **Whether B4's improvements would compound into a promotable variant in a different test window.** B4 improves 12-month rolling win rate from 54.5% to 65.1% and beats baseline on mean test-window excess CAGR — but the walk-forward std barely moves, and only one of seven criteria flips (C6). A longer test window or different regime mix could change the picture; v2's data does not answer that.

## Methodology robustness — what v2 DID establish defensibly

- **Pre-committed criteria evaluated against pre-committed thresholds without goalpost moving.** Near-misses (B2 C6 at 58.79% vs 60% threshold; B5 C3) reported as failures at the spec's thresholds.
- **Variant codepaths exercise the construction logic they claim to test.** Confirmed via unit tests at Gate 2 (decimal=12 weight equivalence for baseline vs v1's rank_top_n_weights) and via the Gate 3 (a) bit-exact baseline reproducibility check (all 7 gating headline metrics 0.0 abs diff vs v1 pinned).
- **Walk-forward IC is full-cross-section by construction** (uses `labels.merge(scores)` not prices) and v2-baseline bit-reproduces v1's walk-forward IC across all 6 windows. Walk-forward-window analytics are scope-stable.
- **The IC scope audit was empirically reproducible** (three scopes tested via `scripts/research/audit_v1_ic_scope.py`, each reproducing the pinned values at its respective scope). The audit script lives in the repo and can be invoked at any point.
- **B6's no-op finding was bit-verified** against baseline in the test window (CAGR/excess/MaxDD/Sharpe all identical to displayed precision) and the mechanism (1/30 < 4% cap = cap doesn't bind) is documented in code.
- **Cross-variant concentration overlap was computed across all variants on a unified ticker set**; the Spearman correlation and appearance distribution are reproducible from the `comparison_results.parquet` + `concentration_overlap.parquet` artifacts.

## Per-variant detail (supporting evidence for the headline)

The per-variant findings below are NOT the headline — they support it. The substantive v2 finding is the binding-constraint demonstration; these are the per-variant evidence backing that demonstration.

### baseline

3/7 pass (C2, C3, C7 trivially via self-comparison; plus C7 is the v1 +3.52pp excess CAGR). Bit-exactly reproduces v1's pinned headline metrics. Reproducibility check passed at Gate 3 (a). Fails C1 (vs itself = 0% reduction), C4 (1.76×), C5 (32.45%), C6 (54.52%).

### B1 (vol_target)

2/7 pass. Best at C1 std reduction (+41.57%) and C4 drawdown ratio (1.35×) — both by reducing gross exposure. Heavy CAGR sacrifice (test excess −6.46pp vs baseline's +3.52pp). Training-tail vol of 31.62% annualized (last 63 days of training period, includes high-vol regimes like COVID + 2022 bear) means B1 starts the test window at ~47% gross exposure for the first 63 days of test data. The warmup design is honest given the pre-committed Gate 1 spec; the result is the consequence of frozen training-tail vol meeting the test window's lower realized vol. Future vol-targeting studies should think carefully about warmup design — for v3 or later, not v2.

### B2 (conviction_weighted)

3/7 pass. Slightly improves test-window mean excess CAGR (+4.74pp vs baseline's +3.52pp). Closest miss on C6 at 58.79% (vs threshold 60%). W6 walk-forward result is **−33.78% excess CAGR** — worst single-window result in the study. Conviction-weighting compounds losses on top-score names in adverse regimes; W6 (2025-2026 val window) was such a regime for B2.

### B3 (dynamic_topn)

1/7 pass. Highest walk-forward positive-window count (5/6) — but driven by a single tail-event window (W6 = +83.04% excess CAGR). Removing W6 drops B3's walk-forward mean from +19.39% to ≈−6%. B3's dynamic-N produces holdings that diverge more from baseline than other variants (B3 vs baseline Spearman correlation 0.86 on top alpha contributors, vs ~0.97 for other variants). Highest concentration on a single ticker (C5 value 54.57%). The construction logic is most distinctive but also the most regime-dependent.

### B4 (concentration_penalties)

**4/7 pass — only variant to exceed baseline.** Passes C6 (rolling win rate 65.08%) and improves on C3 (mean CAGR giveback −38.0% i.e. variant beats baseline). Walk-forward std (23.20%) barely moves from baseline's (23.41%). The persistence and sector-overweight penalties improve within-window consistency (rolling win rate) but not the cross-window regime variance v2 was designed to test. B4's mechanism is real but operates on a different binding constraint than the one v2 was scoping against.

### B5 (defensive_sleeves)

3/7 pass. Best at C1 std reduction (+47.34%) and C4 drawdown ratio (1.16×) by allocating 30-50% to cash + SHY. Heavy CAGR sacrifice (test excess −2.50pp). Like B1, the gross-exposure reduction explains both the wins (C1, C4) and the losses (C3, C7).

### B6 (smaller_caps)

3/7 pass — identical to baseline in the test window. v1's top-30 equal-weight at 1/30 ≈ 3.33% per position is already below B6's 4% cap. The cap does not bind in the test window's date range. B6 IS NOT bit-identical to baseline in walk-forward windows 1-3 where sector-cap redistribution pushes some weights above 4% — but in the test window, B6 is a no-op. The hypothesis "smaller individual caps reduce concentration risk" is untestable at top-30 equal-weight in this universe. Combinations with non-equal-weight variants (e.g., B6 + B2 conviction-weighted) might make B6's cap meaningful, but combinations are explicitly out of scope for v2.

## Known follow-ups (NOT pursued in this gate)

These are tracked workstreams that follow v2 closure. They are noted here so the v2 record is honest about what remains open.

### CV-objectives architectural memo recommendation re-justification

The IC scope audit revealed that the memo's central empirical evidence (+0.048 top-quintile IC) was scope-restricted. The memo's correction section landed at commit `d502629` preserves the original prose and adds a correction noting that the recommendation requires re-justification on its logical-structure merits separate from the +0.048 number.

Decision deferred until after v2 closes. Either:
- Re-justify the recommendation under standard scope evidence (e.g., a-priori theoretical argument from factor-research practice that top-quintile IC is the right CV objective for top-N portfolios, independent of any specific empirical IC value)
- Modify the recommendation (e.g., "use top-quintile IC as primary CV objective when the model produces meaningful top-quintile signal; report dual metrics for empirical validation")
- Revise the memo to retract the recommendation pending more evidence

Future contract-conformant studies should NOT default to `top_quintile_spearman_ic` as their CV objective on the strength of the memo alone until the re-justification is resolved. The contract field `objective.training_cv` remains in place.

### v3 scoping with Mechanism A direction

v2's findings strongly suggest signal extraction (Mechanism A) is the binding constraint for top-N strategies on this universe at this horizon. Specific v3 design questions are the next conversation:

- Which Mechanism A levers to test? (Feature engineering, model class change, CV objective change, universe filtering, sector neutralization, horizon change, ...)
- Does v3 hold construction fixed (mirroring v2's discipline of varying one mechanism) or co-vary both?
- What pre-committed success criteria for v3? The v2 criteria were construction-side; some don't translate directly to signal-side studies.

These are scoping questions, not v2 closure questions. v3 scoping happens after v2 lands on main.

## Sourced from

- [`docs/studies/larger_universe_v2/spec.md`](spec.md) — pre-committed v2 spec (seven variants, seven criteria)
- [`docs/studies/larger_universe_v2/baseline_reproducibility_check.md`](baseline_reproducibility_check.md) — Gate 3 (a) bit-exact reproducibility check
- [`docs/studies/larger_universe_v2/gate3_phase4_report.md`](gate3_phase4_report.md) — Gate 3 (b) full per-criterion + walk-forward report
- [`docs/studies/larger_universe_v1/ic_scope_audit.md`](../larger_universe_v1/ic_scope_audit.md) — full audit of v1 analytics scope dependence
- [`docs/studies/larger_universe_v1/dashboard_rendering_check.md`](../larger_universe_v1/dashboard_rendering_check.md) — dashboard rendering findings
- [`docs/sessions/larger_universe_v2/session_log.md`](../../sessions/larger_universe_v2/session_log.md) — per-gate session log (Gates 1-4)
- `models/studies/larger_universe_v2/comparison/comparison_results.parquet` — per-variant criterion evaluation (one row per variant; per-criterion value + pass columns)
- `models/studies/larger_universe_v2/comparison/concentration_overlap.parquet` — cross-variant top-K overlap analysis
- `models/studies/larger_universe_v2/comparison/concentration_corr_matrix.parquet` — cross-variant Spearman correlation matrix on `pct_of_total_alpha`
- `models/studies/larger_universe_v2/comparison/concentration_overlap_summary.json` — summary stats (mean/median Spearman, appearance distribution)
- `models/studies/larger_universe_v2/<variant>/contract_v1/` — per-variant Phase 4 + Phase 5 artifacts (7 variants)
- `scripts/research/phase4_run_v2.py` — Phase 4 runner (shared-model optimization)
- `scripts/research/phase5_walk_forward_v2.py` — walk-forward retrains
- `scripts/research/phase5_analytics_v2.py` — Phase 5 analytics (decile, attribution, IC, win rate, concentration) per variant + cross-variant overlap
- `scripts/research/build_comparison_results_v2.py` — comparison artifact builder evaluating the seven criteria
- `scripts/research/audit_v1_ic_scope.py` — IC + decile re-derivation at three scopes for the audit
- `scripts/research/annotate_meta_artifact_scope.py` — one-shot meta.json scope-annotation script

## Architectural artifacts touched by v2

- `docs/architecture/dashboard_contract_v1.md` — added optional `artifact_metadata` field schema (commit `df4dc9b`)
- `docs/architecture/dashboard_operations_v1.md` — documented `artifact_metadata` as an additive-change example (commit `df4dc9b`)
- `docs/architecture/ml_study_cv_objectives_v1.md` — correction section appended preserving original prose (commit `d502629`)
