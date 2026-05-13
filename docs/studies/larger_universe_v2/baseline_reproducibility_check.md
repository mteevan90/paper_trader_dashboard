# v2-baseline reproducibility check vs v1

**Date:** 2026-05-13
**Runner SHA:** `9bde239` (gate3(v2): cast numpy.bool_ to native bool — applied on top of `ce8dfdd` which landed the three Gate 3 runners)
**Spec:** [docs/studies/larger_universe_v2/spec.md](spec.md)

## Purpose

Gate 3's reproducibility check verifies that v2's refactored backtest engine and the v2-baseline construction variant produce identical results to v1's Phase 4 pipeline on the same inputs. The check is a precondition for trusting B1–B6 variant comparisons: if the baseline doesn't reproduce, divergences in B1–B6 against the baseline can't be cleanly attributed to construction-logic differences vs pipeline drift.

## Pre-flight references (locked at session start)

v1's pinned headline metrics from `models/studies/larger_universe_v1/contract_v1/meta.json` `summary_metrics.test.xgboost`, verified bit-exact against re-derivation from `portfolio.parquet` using the `_summarize` formula at `scripts/research/phase4_run.py:251–294`:

| Metric | v1 pinned |
|---|---|
| n_days | 650 |
| total_return | 0.7832693715883139 |
| cagr | 0.2513963365441678 |
| spy_cagr | 0.2162105174499303 |
| excess_cagr | 0.0351858190942376 |
| max_drawdown | -0.3349339363516158 |
| spy_max_drawdown | -0.1899890688985689 |

Sharpe was not pinned in v1's `summary_metrics`. The walk-forward Sharpe formula at `scripts/research/phase5_walk_forward.py:215–217` (`daily_ret.mean() / daily_ret.std() * sqrt(252)`) applied to v1's test-window NAV yields **0.901503** as a derived reference. Reported informationally; not gated.

## Upstream check: scores parity

Runs `_verify_scores_parity()` in `phase4_run_v2.py` against v1's `scores.parquet`. Compares v2's cached XGBoost scores at every (date, ticker) pair in the test window against v1's pinned scores.

Result written to `models/studies/larger_universe_v2/_scores_parity_vs_v1.json`:

```json
{
  "n_pairs_compared": 59232,
  "n_v1_only": 0,
  "n_v2_only": 0,
  "max_abs_diff": 0.0,
  "mean_abs_diff": 0.0,
  "n_within_1e6": 59232,
  "fraction_within_1e6": 1.0,
  "passed": true
}
```

**Interpretation:** v2's XGBoost training pipeline + scoring code produces scores identical to v1's pinned scores across all 59,232 (date, ticker) pairs in the test window. Set overlap is perfect (zero v1-only, zero v2-only). This rules out training-path or feature-pipeline drift as a source of any downstream deviation.

## Headline metrics check (7 gating metrics)

Compares v2-baseline's `meta.json` `summary_metrics.test` against v1's `meta.json` `summary_metrics.test.xgboost`.

| Metric | v1 pinned | v2 baseline | abs diff | rel diff |
|---|---|---|---|---|
| n_days | 650 | 650 | 0 | 0.0000% |
| total_return | 0.7832693715883139 | 0.7832693715883139 | 0.00e+00 | 0.0000% |
| cagr | 0.2513963365441678 | 0.2513963365441678 | 0.00e+00 | 0.0000% |
| spy_cagr | 0.2162105174499303 | 0.2162105174499303 | 0.00e+00 | 0.0000% |
| excess_cagr | 0.0351858190942376 | 0.0351858190942376 | 0.00e+00 | 0.0000% |
| max_drawdown | -0.3349339363516158 | -0.3349339363516158 | 0.00e+00 | 0.0000% |
| spy_max_drawdown | -0.1899890688985689 | -0.1899890688985689 | 0.00e+00 | 0.0000% |

**Max relative deviation across the 7 gating metrics: 0.000000%.**

## Bit-level check: NAV time series

Direct comparison of `portfolio.parquet` NAV column between v1 (XGBoost rows) and v2-baseline:

- Date overlap: 739 rows (full backtest window from 2023-05-12 to 2026-05-11; spans test + OOS)
- v1-only rows: 0
- v2-only rows: 0
- NAV max abs diff: 0.00e+00
- NAV mean abs diff: 0.00e+00
- NAV bit-identical: **True**

## Informational: Sharpe

| | Value |
|---|---|
| v1 derived ref (Sharpe formula applied to v1 NAV) | 0.901503 |
| v2 baseline (Sharpe formula applied to v2 NAV) | 0.901503 |
| abs diff | 4.97e-07 |

The sub-1e-6 difference is FP precision noise from computing pct_change inside the `_summarize` flow vs the standalone re-derivation done at pre-flight. Inconsequential at this magnitude; not flagged.

## Verdict

**BIT-EXACT REPRODUCTION.** All 7 gating metrics deviate by 0.0. NAV series is bit-identical. Scores parity is bit-identical. The v2 refactor — `portfolio_construction/` package, `BaselineVariant`, the engine's variant-path branch in `backtest.py` — preserves v1's pipeline behavior with floating-point identity.

This exceeds the < 0.1% reproducibility tolerance with margin to spare. No source-of-deviation analysis required because there is no deviation. The reproducibility gate is satisfied.

## What this enables

With baseline locked, divergences from baseline in B1–B6 results can be cleanly attributed to the construction-logic differences specified in each variant — not to pipeline drift, training-path variability, or engine refactor artifacts.

## Standing follow-up

The runner-package fix (commit `9bde239`) — casting `numpy.bool_` to native `bool` in the parity-check JSON write — should be inherited by any future v2-pattern study that uses the same parity-check function. The fix is mechanical and localized; documented here so similar issues can be spotted and corrected without re-debugging.
