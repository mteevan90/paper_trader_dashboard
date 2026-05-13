# Larger Universe v2 study — session log

Append-only record of decisions, gate reviews, and phase transitions for the Larger Universe v2 multi-variant portfolio-construction comparison study. Each entry covers one Claude Code session or one gate transition. New entries go at the bottom.

## Pre-log history

Study spec authored 2026-05-12 by Mike. Locked spec captured the seven variants (baseline + B1-B6), the seven success criteria (3 comparative vs baseline + 4 v1 promotion), the gate structure (Gates 1-5), and the architectural constraints (reuse v1's XGBoost model, no Phase 3 retuning, additive contract change for multi-variant studies).

Prerequisites resolved before Gate 1: Overview-merge to main (commit `e178006`) and walk-forward enhancements to main (commit `726193d`). Both contained merge conflicts on the v1 session log that were resolved by keeping both H2 entries chronologically.

## 2026-05-12 — Gate 1: Pre-implementation review

**Phase:** Pre-implementation design
**Branch:** `feat/larger-universe-v2` off `main` at `726193d`
**Status:** APPROVED 2026-05-12 with refinements

### Gate 1 deliverables surfaced

1. **Variant implementation translations** for all 7 variants (pseudocode + decisions on ambiguities).
2. **Dashboard contract extension proposal** — `variant_meta.json` schema, multi-variant discovery rules, page-level variant selector defaulting to `role:"control"`.
3. **Variant Comparison tab design** — 8th tab, 10 sections (decision matrix, walk-forward consistency comparison, per-window heatmap, mean excess CAGR bar chart, etc.), `comparison_results.parquet` schema.
4. **12 flagged ambiguities** resolved with explicit decisions (cap-enforcement order matches v1 exactly; B1 warmup uses frozen training-tail vol; B4 sector overweight reads pre-rebalance portfolio; B5 SHY needs Finnhub fetch; etc.).
5. **SHY data sourcing plan** — `scripts/research/fetch_shy_history.py` mirroring `fetch_spy_and_dividends.py`.

### Refinements from Mike's approval

- **B5 trigger**: hard threshold on 21-day SPY return is intentional. Document the discontinuity at the −5% threshold so it's not mistaken for an accidental edge. No smoothing in v2 — smoothing is a v3 refinement question.
- **`variant_meta.json` schema**: add `optional_artifacts` field listing supplementary files (e.g., `comparison_summary.json` if produced). Makes them discoverable without requiring them.
- **Sidebar disambiguation**: `🔀 + (N variants)` suffix — both pieces of info.
- **Variant selector**: page-level (not per-tab) approved.
- **`comparison_results.parquet`**: add per-criterion *value* columns alongside *pass* booleans (e.g., `criterion_1_std_reduction_pct`, `criterion_3_mean_cagr_giveback_pct`). Lets the dashboard surface "passed by 22%" vs "barely passed at 0.5%" — meaningful nuance that boolean alone hides.
- **Section 9 cumulative growth curve (Variant Comparison tab)**: carries forward the synthetic-compounded pattern from v1's Walk-forward tab. Mike noted the v1 prune-review dependency: if partner review cuts v1's synthetic curve, the v2 version cuts consistently. Tracked as a session-log dependency.

### Standing follow-ups (unchanged from v1)

1. `use_container_width` deprecation sweep
2. Dashboard pytest coverage via `streamlit.testing.v1.AppTest`
3. `attempted_trials` enhancement to `tuning_summary.json`
4. Convergence-pattern methodology memo (pending third Optuna data point)
5. Synthetic compounded growth curve prune review (v1 walk-forward) — partner review pending. **Cascading dependency**: any prune decision applies consistently to v2 Variant Comparison tab's section 9.

## 2026-05-12 — Gate 2: Implementation review

**Phase:** Construction-variant implementation + backtest engine refactor + test coverage
**Branch:** `feat/larger-universe-v2` (pre-commit at time of writing)
**Status:** Gate 2 deliverables landed; awaiting Mike's review before Phase 4 backtests.

### What was built

#### 1. SHY data fetched (one-time setup)

`scripts/research/fetch_shy_history.py` (new) mirrors `fetch_spy_and_dividends.py`'s SPY fetch pattern. Fetches 10 years of daily candles from Finnhub. Saves to `models/cache/equities/finnhub/prices/SHY.parquet`. Ran successfully: 3,019 rows, 2014-05-12 → 2026-05-12.

#### 2. `src/equities/portfolio_construction/` (new package, 9 modules)

```
src/equities/portfolio_construction/
├── __init__.py                # exports + get_variant_by_name() factory
├── base.py                    # ConstructionState dataclass + ConstructionVariant ABC
├── caps.py                    # enforce_individual_cap, enforce_sector_cap, enforce_caps
├── baseline.py                # BaselineVariant (v1 reproducibility)
├── vol_target.py              # VolTargetVariant (B1)
├── conviction_weighted.py     # ConvictionWeightedVariant (B2)
├── dynamic_topn.py            # DynamicTopNVariant (B3)
├── concentration_penalties.py # ConcentrationPenaltiesVariant (B4)
├── defensive_sleeves.py       # DefensiveSleevesVariant (B5)
└── smaller_caps.py            # SmallerCapsVariant (B6)
```

`caps.py` extracts v1's cap-enforcement primitives verbatim — same algorithm, same iteration bounds, same redistribute-to-under-cap-positions logic. This is the foundation of v2-baseline's reproducibility against v1 headline numbers (Gate 3 gate: <1% deviation).

All 7 variants implement the `ConstructionVariant` ABC's `construct(state) → weights` method. Variants pull what they need from `ConstructionState`; unused fields are ignored. `params_dict()` provides per-variant metadata for `meta.json` persistence.

#### 3. `src/equities/study/portfolio.py` refactored

v1's private `_enforce_individual_cap` and `_enforce_sector_cap` moved to `portfolio_construction/caps.py` and imported back as private aliases. `rank_top_n_weights` unchanged in behavior — v1's call path stays bit-identical. The refactor adds a v2 NOTE comment block explaining the structural change.

#### 4. `src/equities/study/backtest.py` refactored for variant support

Three additions:
- New parameter `construction_variant: ConstructionVariant | None = None`.
- New parameter `spy_history: pd.DataFrame | None = None` (passed through to variants that need it, e.g., B5).
- New parameter `shy_prices: pd.Series | None = None` (B5's defensive sleeve trades SHY).

Behavior:
- If `construction_variant` is passed, build a `ConstructionState` at each rebalance from running engine state (current_weights, portfolio_history, top30_streak, prev_portfolio_sector_weights, SPY slice up to date) and call `variant.construct(state)`.
- If `construction_variant` is None and `pc_params` is passed, fall back to v1's legacy path via `rank_top_n_weights()`. v1's existing call sites work unchanged.
- New state tracking inside the engine: `portfolio_daily_returns` list (for B1), `top30_streak` dict (for B4), `prev_portfolio_sector_weights` dict (for B4). Updated post-rebalance.
- SHY handling: caller is responsible for including SHY as a column in `daily_returns`, with a non-`None` `delisting_dates["SHY"]` entry, and a sector label (e.g., "treasury_etf"). The engine doesn't special-case SHY — it's just another tradeable in the wide returns DataFrame. B5's variant adds SHY into its returned weights series; the engine drifts/rebalances it naturally.

#### 5. Tests at `tests/equities/portfolio_construction/`

- `test_caps.py` — 7 tests covering individual cap (no-op, redistribute, cascade, empty), sector cap (no-op, redistribute, no-under-sector cash-residual), and the full `enforce_caps` pipeline.
- `test_variants.py` — 20 tests across 8 test classes (one per variant + registry). Each variant has tests for empty/edge inputs and its distinguishing behavior (B1 scale-down when vol > target, B2 concentration on high scores, B3 N at low/high dispersion, B4 persistence penalty effect, B5 normal/stress regime allocations, B6 4% cap declaration).
- `test_engine_integration.py` — 4 tests verifying `BaselineVariant.construct(state)` produces identical weights to v1's `rank_top_n_weights` on the same inputs, including the concentrated-sectors edge case where the sector cap binds aggressively. This is the **direct foundation of Gate 3's <1% reproducibility tolerance** — if these tests pass, baseline reproducibility on real data should follow modulo data-pipeline details.

Total: **34 new tests added, all passing**. Pre-existing v1 equity tests (15) also pass — `49 passed in 1.76s` for `tests/equities/`.

### Two test-bugs encountered + fixed (institutional knowledge)

Surfaced and worth preserving for future similar refactors:

1. **`pytest.approx` is needed for `sum() == 1.0` comparisons** even when the math should be exact. Pandas-internal float multiplications accumulate FP errors that exact `==` rejects.
2. **Strict `>` comparisons in cap enforcement can trip on FP precision**: `0.10 * 3 == 0.30000000000000004` evaluates strictly greater than `0.30`. In test scenarios with weights that should sum exactly to the cap, the algorithm sees them as "over" and scales — producing surprising results. v1's production usage at `1/30 = 0.0333…` doesn't hit this edge because the float math lands differently. Test scenarios should use weight values that don't trigger this artifact, OR explicitly account for the "all sectors over by FP epsilon" edge case. Documented in `test_sector_cap_no_redistribution_when_no_under_sector`.

### Code diff summary

```
src/equities/portfolio_construction/   +9 files, +674 lines (new package)
tests/equities/portfolio_construction/ +4 files, +384 lines (new test suite)
scripts/research/fetch_shy_history.py  +1 file,  +51 lines (one-time fetch)
src/equities/study/backtest.py         +75 lines, -14 lines (variant path)
src/equities/study/portfolio.py        +17 lines, -57 lines (delegated to caps)
```

Net: +1,113 / -71 lines. No deletions from v1's behavior — every change is either additive (new modules / new parameters with sensible defaults) or a verbatim extraction (caps logic from portfolio.py to caps.py).

### Backtest engine surface area

Engine still produces the same `BacktestResult` (portfolio, holdings, trades, scores DataFrames). Schema and column names unchanged. Per-variant artifact tree will land in Gate 3-4 under `models/studies/larger_universe_v2/<variant_subdir>/contract_v1/`.

### What is NOT yet done (Gates 3-5)

- **Phase 4 backtests** (Gate 3) — 7 variants × 1 test-window backtest + 6 walk-forward retrains = 49 model fits across all variants. Estimate ~2-3 hours compute.
- **Per-variant alpha attribution + decile + IC decomposition** (Gate 4) — Phase 5 analytics per variant.
- **`comparison_results.parquet` + writeup** (Gate 4) — cross-variant comparison framework.
- **Dashboard contract extension** + Variant Comparison tab + variant selector wiring (Gate 5).
- **Tracker update** (at Gate 5 merge).

### Standing follow-ups (unchanged from Gate 1)

1-5 as listed in the Gate 1 entry above.

### Awaiting Mike's review

Gate 2 deliverables are surface-ready: code lands cleanly, all tests pass, the engine refactor preserves v1's call path. Awaiting authorization before kicking off Phase 4 backtests (Gate 3).

## 2026-05-13 — Gate 3 (a): pre-flight + baseline reproducibility check

**Phase:** Phase 4 — runner creation + v2-baseline backtest only (B1–B6 pending baseline approval)
**Branch:** `feat/larger-universe-v2`
**Status:** Baseline bit-exact-reproduces v1; awaiting Mike's approval before running B1–B6.

### Pre-flight (verbal authorization from Mike before any compute)

Three pre-implementation findings surfaced and resolved before runner code was written:

1. **v1 reference values locked.** Re-derived v1's pinned `summary_metrics.test.xgboost` from `portfolio.parquet` using the `_summarize` formula at `scripts/research/phase4_run.py:251–294`. Zero deviation vs `meta.json` pinned values across all six metrics (total_return, cagr, spy_cagr, excess_cagr, max_drawdown, spy_max_drawdown). The < 0.01% pre-flight tolerance was met with margin to spare. Sharpe nuance: v1 doesn't pin test-window Sharpe in `summary_metrics`; the walk-forward Sharpe formula at `phase5_walk_forward.py:215–217` applied to v1's test-window NAV gives 0.901503 as a derived informational reference. Sharpe stays informational, not gated (the seven gating metrics remain CAGR, excess CAGR, MaxDD, SPY MaxDD, n_days, total_return, SPY CAGR).

2. **v2 spec was not in the repo.** The session log referenced a "locked spec" with seven success criteria, but no `docs/studies/larger_universe_v2/spec.md` existed in version control. Git log confirmed nothing had ever been added under that path. The criteria existed only in conversation. Mike pasted the full spec and it landed at commit `54c286e` ("docs(studies): land Larger Universe v2 study spec (pre-committed criteria)") before any runner code was written — making it pre-committed per the Operating Principles' methodology requirement.

3. **Runner scripts didn't exist.** Gate 2 landed the variant package + the `backtest.py` engine refactor with `construction_variant` support, but no v2 Phase 4 runners (`phase4_run_v2.py`, `phase5_walk_forward_v2.py`, `build_comparison_results_v2.py`) existed. Treated as Gate 3 scope. Written as Commit A (`ce8dfdd`) before any backtest ran.

### Commits landed

- `54c286e` — `docs(studies): land Larger Universe v2 study spec (pre-committed criteria)`
- `ce8dfdd` — `gate3(v2): phase 4 + phase 5 + comparison runners` (1,694 insertions across 3 new scripts)
- `9bde239` — `gate3(v2): cast numpy.bool_ to native bool in scores parity result` (caught by baseline run when first attempt failed at the JSON-write step of the parity check; XGBoost training and score caching had already completed successfully)
- `(this commit)` — `gate3(v2): baseline reproducibility check` (artifacts + deviation analysis writeup + this session log entry)

### Runner architecture

`phase4_run_v2.py` uses shared-model optimization: train XGBoost once on v1's locked hyperparameters, cache scores at every rebalance date once, then loop over `--variants` calling the engine with each variant's `construction_variant` instance. Cached-score reuse means per-variant compute is dominated by the engine walk over daily returns (~0.6s observed on baseline, vs ~44s for the one-shot training step). Total Gate 3 wall-clock for all 7 variants will be dominated by walk-forward retrains (6 windows × 1 training each), not by Phase 4 itself.

`phase5_walk_forward_v2.py` does the same per-window: 1 training + cached scoring + N variant backtests. `build_comparison_results_v2.py` evaluates the seven success criteria from the spec, including criterion-5 (max single-ticker alpha share, mirrors v1's `per_ticker_attribution` filtered to the test window) and criterion-6 (12-month rolling win rate vs SPY, mirrors v1's `rolling_win_rate` filtered to the test window).

### Baseline reproducibility result

Upstream scores parity check (v2 cached scores vs v1's `scores.parquet`):

- 59,232 (date, ticker) pairs compared
- v1-only: 0; v2-only: 0 (perfect set overlap)
- max_abs_diff: 0.00e+00
- mean_abs_diff: 0.00e+00
- **PASS — bit-identical**

Headline-metric deviation (v2-baseline `summary_metrics.test` vs v1 pinned `summary_metrics.test.xgboost`):

| Metric | abs diff | rel diff |
|---|---|---|
| n_days | 0 | 0% |
| total_return | 0.00e+00 | 0% |
| cagr | 0.00e+00 | 0% |
| spy_cagr | 0.00e+00 | 0% |
| excess_cagr | 0.00e+00 | 0% |
| max_drawdown | 0.00e+00 | 0% |
| spy_max_drawdown | 0.00e+00 | 0% |

Bit-level NAV check (full 739-row series across the test + OOS windows): max abs diff 0.0, bit-identical.

Informational Sharpe: v1 derived ref 0.901503; v2 baseline 0.901503; abs diff 4.97e-07 (FP precision noise from a different computation path, not flagged).

**Verdict: BIT-EXACT REPRODUCTION.** Exceeds the < 0.1% reproducibility tolerance with margin to spare. No source-of-deviation analysis required.

Full deviation writeup: [docs/studies/larger_universe_v2/baseline_reproducibility_check.md](../../studies/larger_universe_v2/baseline_reproducibility_check.md).

### What this enables

The variant package + engine refactor preserves v1's pipeline behavior with floating-point identity. Any divergences from baseline in B1–B6 backtest results can be cleanly attributed to the construction-logic differences specified in each variant — not to pipeline drift, training-path variability, or engine refactor artifacts.

### What's next

- **Awaiting Mike's approval to run B1–B6.** Per pre-flight authorization, baseline-first sequencing with stop-on-fail; baseline reproduced with zero deviation, so the gate is clean to proceed.
- After approval: run `phase4_run_v2.py --variants b1_vol_target,b2_conviction_weighted,b3_dynamic_topn,b4_concentration_penalties,b5_defensive_sleeves,b6_smaller_caps`, then `phase5_walk_forward_v2.py --variants all`, then `build_comparison_results_v2.py --variants all`.
- Then surface the Gate 3 report (Gate 3 (b) entry): headline metrics per variant, walk-forward consistency stats, comparison_results.parquet rendering, and any variant flagged for unexpected behavior. Don't auto-proceed to Gate 4.

### Standing follow-ups (unchanged from Gate 1)

1–5 as listed in the Gate 1 entry above.

## 2026-05-13 — Gate 3 (b): B1–B6 variants + walk-forward + comparison

**Phase:** Phase 4 — all 7 variants × test backtest + 6 walk-forward retrains + cross-variant comparison
**Branch:** `feat/larger-universe-v2`
**Status:** Phase 4 complete. No variant passes all seven criteria. All seven are `METHODOLOGY FINDING`. B4 leads with 4/7. Awaiting Mike's review before Gate 4.

### Headline

**No `PROMOTE` verdict.** Variants ranked by `n_pass`:

1. `b4_concentration_penalties` — 4/7 (passes baseline + criterion 6 12-month rolling win rate)
2. `baseline`, `b2_conviction_weighted`, `b5_defensive_sleeves`, `b6_smaller_caps` — 3/7
3. `b1_vol_target` — 2/7
4. `b3_dynamic_topn` — 1/7

`b4_concentration_penalties` is the only variant that exceeds baseline's pass count.

### Two issues caught + fixed during runs

- **B3 RuntimeError** at first rebalance — B3 requires `training_dispersion_dist`. Caught when phase4_run_v2.py reached B3 after B1 and B2 completed cleanly. Fixed in commit `0d5a537` (`gate3(v2): bootstrap training-period warmup state for B1 and B3`) — added `_compute_warmup_state` to phase4_run_v2.py and a per-window equivalent to phase5_walk_forward_v2.py. Both compute training-period scores using the trained model, derive top-decile dispersion list (B3) and last-63-day baseline portfolio vol (B1). Per Gate 1 design ("B1 warmup uses frozen training-tail vol"); not peek-ahead.
- **`numpy.bool_` JSON serialization** in scores parity check. Fixed in commit `9bde239` between Commit A and Commit B.

### Commits landed in Gate 3 (b)

- `0d5a537` — `gate3(v2): bootstrap training-period warmup state for B1 and B3`
- `(this commit)` — `gate3(v2): B1–B6 variants + walk-forward + comparison results` (Gate 3 (b) report + session log entry)

Plus the Gate 3 (a) commits that preceded:
- `54c286e`, `ce8dfdd`, `9bde239`, `99ee865`

### Phase 4 test-window headline metrics

| Variant | CAGR | Excess vs SPY | MaxDD | Sharpe |
|---|---:|---:|---:|---:|
| baseline | +25.14% | +3.52pp | −33.5% | 0.9015 |
| b1_vol_target | +15.16% | −6.46pp | −25.6% | 0.7817 |
| b2_conviction_weighted | +26.36% | +4.74pp | −33.2% | 0.9106 |
| b3_dynamic_topn | +21.60% | −0.02pp | −32.5% | 0.8251 |
| b4_concentration_penalties | +26.48% | +4.85pp | −32.7% | 0.9370 |
| b5_defensive_sleeves | +19.12% | −2.50pp | −22.0% | 0.9727 |
| b6_smaller_caps | +25.14% | +3.52pp | −33.5% | 0.9015 |

### Walk-forward verification

v2-baseline walk-forward excess CAGR per window matches v1's pinned `walk_forward.parquet` for XGBoost to all reported precision: +0.0707, +0.1316, −0.0349, +0.2390, −0.0219, +0.5935. Cross-confirms reproducibility from Gate 3 (a) extends to the walk-forward pipeline.

### Methodology findings (documented in the report)

- **B6 is a no-op in the test window.** v1's top-30 equal-weight at 1/30 ≈ 3.33% per position is below B6's 4% cap. The cap never binds in the test window's date range. B6 *does* differ from baseline in walk-forward windows 1-3 where sector-cap redistribution pushes weights above 4%. The test-window verdict is identical to baseline by construction, not by coincidence. The hypothesis "smaller individual caps reduce concentration risk" is untestable at top-30 equal-weight in this universe; the test isn't a falsification, it's a non-event. Combinations with non-equal-weight variants are out of scope for v2.
- **B1 is highly sensitive to warmup vol.** Training-tail vol = 31.62% annualized (last 63 training-period trading days). At a 15% target, B1 starts the test window at ≈47% gross exposure. The result is honest given the pre-committed spec; we don't retroactively change the warmup design. The finding documents vol-targeting's sensitivity to warmup design for future studies — v3 candidate, not v2 scope.
- **Criterion 5 (single-ticker alpha concentration) fails for every variant.** Universe/model-level issue inherited from v1. Construction-logic variants reduce the share marginally (B2 28.1%, B4 28.9%, B5 29.2%) but not below the 25% threshold. Addressing this requires sector neutralization, universe filtering, or signal extraction — all out of v2 scope.
- **Criterion 4 (drawdown ratio) is structurally bound to gross exposure.** Only B1 and B5 pass. All long-only top-30 equal-weight variants land at 1.71-1.76× SPY MaxDD regardless of selection logic. Future studies aiming to pass this likely need explicit risk-budgeting.
- **B3's W6 outlier dominates its mean.** B3's W6 = +83.04% excess CAGR is more than 1.5× the next-best variant in that window. Removing W6 drops B3's walk-forward mean from +19.39% to ≈−6%. Mean is real but driven by one window's tail event; not steady performance. Flagged as caveat for any future writeup highlighting B3's regime consistency.
- **B2's W6 = −33.78% excess CAGR.** Worst single-window result in the study. Conviction-weighting compounds losses when top-score names underperform — which happened in 2025-2026. B2's test-window result hides this; the walk-forward exposes it.

### Reporting discipline applied

Per Mike's framing for Gate 3 (b):
- Every criterion's *value* per variant reported, not just pass/fail booleans
- All seven variants reported regardless of pass count — no filtering to "winners"
- Near-misses (B2 C6 at 58.79% vs 60% threshold, B5 C3 close to passing) reported as failing the criterion — no goalpost-moving language
- Unexpected behavior (B3 RuntimeError, JSON serialization) flagged in the report's "Unexpected behavior" section as resolved during the run
- B6 no-op and B1 warmup-sensitivity framed as methodology findings, not performance explanations
- Negative result on PROMOTE stated explicitly as the headline rather than buried

### What's next

Awaiting Mike's review of the Gate 3 (b) report at `docs/studies/larger_universe_v2/gate3_phase4_report.md`. Don't auto-proceed to Gate 4 (Phase 5 analytics) without explicit approval.

If approved, Gate 4 would produce per-variant Phase 5 artifacts (decile_returns, per_ticker_attribution, ic_decomposition, rolling_win_rate detail, concentration_summary) needed for the eventual Variant Comparison tab in the dashboard. The headline finding (no PROMOTE) doesn't change at Gate 4 — analytics elaborate on the findings rather than reopen verdict.

## 2026-05-13 — Gate 4 (IC scope audit + corrections; writeup pending)

**Phase:** Phase 5 analytics + cross-variant comparison framework
**Branch:** `feat/larger-universe-v2`
**Status:** Gate 4 analytics ran; an IC reproducibility discrepancy was caught, audited, and corrected at the documentation + dashboard layer. The v2 writeup is deferred pending the recommendation re-justification workstream that the audit surfaced.

### What happened

`phase5_analytics_v2.py` was written and ran in ~8 seconds across all 7 variants, producing decile_returns, per_ticker_attribution, ic_decomposition, rolling_win_rate, concentration_summary per variant, plus a cross-variant concentration_overlap artifact answering the C5 question (model-determined vs construction-specific concentration).

When the headline draft was surfaced for review, Mike caught an inconsistency: v1's pinned XGBoost top-quintile IC was +0.048 (cited prominently in `docs/architecture/ml_study_cv_objectives_v1.md` and `docs/studies/larger_universe_v1/results.md`); v2 Gate 4's same-formula computation on bit-identical scores produced −0.0041. Sign reversal.

Investigation: the IC formulas are mathematically identical between v1 (`scripts/research/phase5_analytics.py:167-229`) and v2 (`phase5_analytics_v2.py:174-227`). The difference is a price-universe scope choice. v1's `load_inputs` loads prices only for tickers in `holdings["ticker"].unique()` (~450 across XGB+ENet); v2 loads the full eligible universe (~1,963 tickers). At v1's scope the top-quintile IC is +0.0481; at full-universe scope it is −0.0041. Both numbers are correct under their respective definitions, but they measure different things, and v1's published documentation read as if it had computed the standard full-cross-section IC.

### Findings

Two of v1's 16 contract_v1 artifacts are scope-affected:
- `ic_decomposition.parquet` (top-quintile IC sign-reverses at standard scope)
- `decile_returns.parquet` (decile-1 stats pathological at v1's scope: +35.7% / std 202%; realistic at full scope: +5.8% / std 25%)

All other v1 artifacts confirmed scope-independent. Walk-forward IC (via `labels.merge(scores)` not prices) is full-cross-section by construction; v2-baseline reproduces v1's walk-forward IC bit-for-bit. Per-ticker attribution iterates over `holdings.iterrows()` and is bit-identical at any scope.

The audit also surfaced a methodological tension in `ml_study_cv_objectives_v1.md`: the memo's recommendation operationalizes full-cross-section top-quintile IC as a CV objective (at training time, in-fold, no "held" subset exists), but its empirical evidence (+0.048) is held-subset top-quintile IC. The recommendation's logical structure may still hold; the empirical chain in the memo does not.

### Commits landed in Gate 4 (so far)

- `0d5a537` — `gate3(v2): bootstrap training-period warmup state for B1 and B3` (pre-Gate-4; caught when Gate 3 (b) B1-B6 first ran)
- `d502629` — `v1 audit: price-universe scope finding + factual corrections`
  - `scripts/research/audit_v1_ic_scope.py` — reproducer for the audit numbers
  - `scripts/research/phase5_analytics_v2.py` — the v2 analytics runner that surfaced the discrepancy
  - `docs/studies/larger_universe_v1/ic_scope_audit.md` — full audit report
  - `docs/studies/larger_universe_v1/dashboard_rendering_check.md` — finds two displays in `tab_contract_diagnostics` that surface v1's scope-affected artifacts with labels implying full-cross-section scope
  - Correction section appended to `docs/architecture/ml_study_cv_objectives_v1.md` (preserves original prose)
  - Correction footnotes appended to `docs/studies/larger_universe_v1/results.md` L13 and L170 (preserves original prose)
- `45338e9` — `dashboard(option 2): scope-correction banner on v1 diagnostics tab`
  - Inline `st.warning` banner on `tab_contract_diagnostics` keyed to `study_name == "larger_universe_v1"`. Immediate partner-facing correction context.
- `(this commit)` — `dashboard(option 1): artifact_metadata contract addition + scope-aware rendering`
  - Schema addition in `docs/architecture/dashboard_contract_v1.md`: new optional top-level `artifact_metadata` field on meta.json. Per-artifact dict with `scope` (controlled vocabulary: `held_subset`, `full_cross_section`, `other`), `scope_description`, `audit_reference`. Documented as additive (no schema_version bump).
  - `scripts/research/annotate_meta_artifact_scope.py` — one-shot patch script. Annotates v1's meta.json (`held_subset` + audit_reference) and all 7 v2 variant meta.json files (`full_cross_section`). Idempotent. Ran cleanly: 8 files patched.
  - `scripts/research/phase5_analytics_v2.py` updated to write `artifact_metadata` for `ic_decomposition.parquet` and `decile_returns.parquet` in each variant's meta.json automatically going forward.
  - `src/dashboard_app.py` — new `_render_scope_callout` helper; `tab_contract_diagnostics` now reads `artifact_metadata` and surfaces scope inline: `st.warning` for `held_subset`, `st.info` for `full_cross_section`, default caption for absent. Legacy v1 fallback banner from Option 2 kicks in only when `artifact_metadata` is missing AND study is `larger_universe_v1` — once the annotation script ran, the per-artifact callouts replace the banner.
  - `docs/architecture/dashboard_operations_v1.md` updated to note `artifact_metadata` as an additive-change example.

Smoke-tested with mocked Streamlit: 4 scope-rendering paths verified (legacy v1 fallback banner; v1 annotated `held_subset` per-artifact callouts; v2 annotated `full_cross_section`; unknown study with no metadata). Dashboard boots cleanly under both Option 2 and Option 1 changes.

### What is explicitly NOT in this Gate 4 work

- **v2 writeup is not drafted.** Mike's framing decision is deferred pending the recommendation re-justification workstream that the audit surfaced. The Gate 4 analytics produced supporting data; the writeup's headline depends on whether `ml_study_cv_objectives_v1.md`'s recommendation survives correction.
- **No modifications to v1's pinned parquet artifacts.** `ic_decomposition.parquet` and `decile_returns.parquet` remain as the historical record of what v1 produced. The audit script reproduces corrected values on demand for any consumer that needs them.
- **No revision of the CV objectives architectural memo's recommendation.** The correction section explicitly states the recommendation requires re-justification on its logical-structure merits; that's an open workstream, not closed by this Gate 4 work.
- **No tracker update.** Standing rule for v2 work — tracker updates when v2 closes at Gate 5 merge.

### Standing follow-up (Mike's framing note about correction-loop risk)

The Gate 4 path so far has been correction-heavy: investigation → audit → corrections → dashboard updates. Each step was individually justified, but Mike flagged the failure mode of getting stuck in further correction loops. The Operating Principles rule out time-based reasoning, but they don't rule out re-evaluation of where the work should go next — and right now the next decisions are:

1. Finish v2's Gate 4 + Gate 5 plan (writeup + dashboard implementation per spec). Close v2 honestly.
2. Decide whether v2's findings warrant a different v3 direction than originally scoped (Mechanism A signal extraction is now more clearly the binding constraint).
3. Don't drift into more correction work without a specific consumer-of-corrections demanding it.

When the writeup gate resumes, this framing note applies — finish v2 honestly with the findings we have, not by chasing further methodology corrections.

### What's next (now pending Mike's approval)

- Resume the v2 writeup work with the headline framing decision (Option A/B/C from the earlier framing discussion). The audit + memo correction + dashboard correction provide the context the writeup needs to characterize its findings honestly.
- Or: pause v2 entirely until the CV-objectives memo's recommendation is re-justified, treating that as a prerequisite for v2's writeup. Mike's call.
