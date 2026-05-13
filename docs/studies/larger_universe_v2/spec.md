# Larger Universe v2 — Study Spec

**Status: Locked 2026-05-13. Pre-committed before Gate 3 compute. Do not modify without explicit approval from Mike.**

## Purpose

Test whether portfolio construction changes improve regime consistency over v1's baseline. Primary failure mode addressed: inconsistent excess returns across walk-forward windows (4 of 6 positive for XGBoost, 23.4pp std dev across windows). Mechanism B from v2 scoping; Mechanism A (signal extraction) parked as v3 candidate.

## Approach

Multi-variant comparison. Hold model (XGBoost), features, universe, dates, CV objective constant from v1. Vary only portfolio construction logic. Reuse v1's saved XGBoost model and locked hyperparameters — no Phase 3 retuning.

## Variants (7 total)

- **baseline**: v1 construction unchanged (rank top-30 equal-weight, 7.5% individual cap, 30% sector cap)
- **B1 vol_target**: Vol-targeting overlay, 15% annualized target via 63-day realized vol
- **B2 conviction_weighted**: Softmax weighting within top-30, T=0.5
- **B3 dynamic_topn**: N varies 15–50 based on top-decile score dispersion (linear interpolation between training-distribution 10th and 90th percentiles)
- **B4 concentration_penalties**: Persistence penalty (10% per consecutive rebalance beyond 6, capped 50%) + sector overweight penalty (20% reduction for stocks in sectors > 20% of pre-rebalance portfolio)
- **B5 defensive_sleeves**: 70/30 equity/defensive normal, 50/50 in stress (trailing 21-day SPY < −5%). Defensive sleeve is 50/50 cash/SHY
- **B6 smaller_caps**: 4% individual cap instead of 7.5%

## Pre-committed success criteria

Winning variant must meet **ALL seven** criteria. A variant meeting only some criteria is documented as a methodology finding, not a promotion candidate.

### Relative-to-baseline criteria (3)

**Criterion 1: Std dev reduction**
- Metric: std dev of excess CAGR vs SPY across the 6 walk-forward windows
- Threshold: ≥20% reduction relative to baseline's std dev (baseline is v1 XGBoost: 23.4pp)
- Pass condition: `variant_std ≤ 0.80 × baseline_std`
- Value column: `criterion_1_std_reduction_pct` (computed as `1 - variant_std/baseline_std`, expressed as percentage)
- Evaluation window: 6 walk-forward windows

**Criterion 2: Positive-window rate**
- Metric: count of walk-forward windows with excess CAGR > 0
- Threshold: maintain or improve relative to baseline (baseline is v1 XGBoost: 4 of 6 = 0.667)
- Pass condition: `variant_positive_rate ≥ baseline_positive_rate`
- Value column: `criterion_2_positive_window_count` (integer count out of 6)
- Evaluation window: 6 walk-forward windows

**Criterion 3: Mean CAGR giveback**
- Metric: mean excess CAGR vs SPY on the test window
- Threshold: ≤30% giveback relative to baseline mean excess CAGR (baseline is v1 XGBoost: 3.5pp test-window excess CAGR)
- Pass condition: `variant_mean_excess_cagr ≥ 0.70 × baseline_mean_excess_cagr`
- Value column: `criterion_3_mean_cagr_giveback_pct` (computed as `1 - variant_mean/baseline_mean`, expressed as percentage; negative values indicate improvement)
- Evaluation window: full test window (2023-05-12 → 2025-12-31)

### v1 promotion criteria (4)

**Criterion 4: Drawdown**
- Metric: variant max drawdown vs SPY max drawdown on test window
- Threshold: `variant_max_dd ≤ 1.5 × spy_max_dd` (in absolute magnitude; max_dd values are negative)
- Pass condition: `|variant_max_dd| ≤ 1.5 × |spy_max_dd|`
- Value column: `criterion_4_drawdown_ratio` (computed as `|variant_max_dd| / |spy_max_dd|`)
- Evaluation window: full test window

**Criterion 5: Single-ticker concentration**
- Metric: maximum single-ticker contribution to total alpha on test window
- Threshold: ≤25% of total alpha attributable to any single ticker
- Pass condition: `max_single_ticker_alpha_pct ≤ 0.25`
- Value column: `criterion_5_max_single_ticker_alpha_pct` (the actual percentage)
- Evaluation window: full test window

**Criterion 6: Win rate**
- Metric: 12-month rolling win rate vs SPY on test window
- Threshold: ≥60%
- Pass condition: `rolling_12mo_win_rate ≥ 0.60`
- Value column: `criterion_6_rolling_12mo_win_rate` (the actual rate)
- Evaluation window: full test window

**Criterion 7: Excess CAGR**
- Metric: total excess CAGR vs SPY on test window
- Threshold: > 0 (strategy must beat SPY at all)
- Pass condition: `test_excess_cagr_vs_spy > 0`
- Value column: `criterion_7_test_excess_cagr` (the actual excess CAGR)
- Evaluation window: full test window

## Evaluation windows

All seven criteria evaluated on the test window (2023-05-12 → 2025-12-31). Criteria 1 and 2 use the 6 walk-forward windows that each span 1 year of validation. Criteria 3, 4, 5, 6, 7 use the full ~2.5-year test window.

Reserved validation period (2026-01-01 → snapshot end, ~4 months) performance is reported alongside as context but does NOT gate variant selection. Reason: insufficient sample size for statistically reliable annualized metrics.

## Verdict framework

For each variant, compute:
- `all_pass` (bool): AND of criterion_1 through criterion_7
- `verdict` (str):
  - `"PROMOTE"` if `all_pass`
  - `"METHODOLOGY FINDING"` if any criterion passes (but not all)
  - `"NOT PROMOTED"` if no criterion passes

## Output structure

```
models/studies/larger_universe_v2/
  variant_meta.json                   (study-level metadata)
  baseline/contract_v1/
  b1_vol_target/contract_v1/
  b2_conviction_weighted/contract_v1/
  b3_dynamic_topn/contract_v1/
  b4_concentration_penalties/contract_v1/
  b5_defensive_sleeves/contract_v1/
  b6_smaller_caps/contract_v1/
  comparison/comparison_results.parquet
```

`comparison_results.parquet` schema (per Gate 1 approval): one row per variant. Columns include each variant's raw test-window metrics (`test_cagr`, `test_excess_cagr_vs_spy`, `test_max_drawdown`, `test_spy_max_drawdown`, `test_max_single_ticker_alpha_pct`, `test_rolling_12mo_win_rate`), walk-forward stats (`mean_excess_cagr_walkforward`, `std_excess_cagr_walkforward`, `n_windows_positive`, `n_windows_strong`), and BOTH `criterion_N_pass` booleans AND `criterion_N_<value>` columns for nuance ("passed by 22%" vs "barely passed at 0.5%").

## Out of scope for v2

- Phase 3 retuning of XGBoost hyperparameters
- ElasticNet or any other model class
- CV objective changes
- Universe expansion
- New features
- Combination variants (e.g., B1+B6 together) — only if Gate 4 shows multiple single-variable winners, with explicit approval
