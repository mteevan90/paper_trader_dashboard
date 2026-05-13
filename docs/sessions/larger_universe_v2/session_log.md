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
