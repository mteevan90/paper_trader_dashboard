# Regime-Dependent Tunables Model V1 — Hypothesis Spec

Authored: end of session 3 day 4 (May 5, 2026 evening). After Phase 0's regime-dependence finding from rolling-window analysis: alpha is concentrated in crisis-recovery periods (2020 COVID, 2024-2026 corrections), lost in steady bull markets (2021), with alpha-beta correlation -0.43 indicating the strategy can't generate alpha when forced into market participation.

This spec is locked once V1 is launched. Future versions get their own specs.

## Why this hypothesis exists

Phase 0's locked architecture treats all tunables as static. The composite scoring weights, ATR multiplier, position count, and rebalance cadence are single values that the strategy uses regardless of market regime. Rolling-window analysis revealed Phase 0's alpha is regime-dependent — strong in crisis-recovery, weak in steady bull markets. The pattern is structural, not a tuning issue.

The previous hypothesis track (rebal_period_v1) tested whether varying ONE tunable in isolation could improve performance. It didn't, because tunables interact. The deeper finding from V1's failure mode was that single-tunable hypotheses are inherently limited.

This hypothesis goes further: rather than tuning tunables better, change the architecture so tunables can DIFFER by market regime. The strategy becomes one configuration in defensive markets and a different configuration in offensive markets. The regime is determined by the existing macro signal.

## Hypothesis

Allowing the strategy's composite weights, ATR multiplier, position count, and rebalance frequency to differ between defensive and offensive market regimes (determined by a single threshold on the macro signal) will produce a meaningfully better risk-adjusted result than Phase 0's static configuration.

This is a no-preconception test. We are not assuming defensive tunables will look like Phase 0 plus tweaks. We are not assuming offensive tunables will look like Phase 0 with momentum tilt. We're letting Optuna find the best configuration for each regime independently, optimizing both simultaneously against the new rolling-window objective.

## What this hypothesis tests

### Tunables that vary by regime (7)

Each gets two values — one for defensive regime, one for offensive regime:

| Tunable | Defensive range | Offensive range | Notes |
|---|---|---|---|
| weight_fundamental | [0.05, 0.70] | [0.05, 0.70] | Composite weight component |
| weight_technical | [0.05, 0.60] | [0.05, 0.60] | Composite weight component |
| weight_model | [0.05, 0.60] | [0.05, 0.60] | Composite weight component |
| weight_alt | [0.0, 0.20] | [0.0, 0.20] | Composite weight component |
| atr_multiplier | [1.0, 5.0] | [1.0, 5.0] | Trailing stop aggressiveness |
| position_count | [5, 20] | [5, 20] | Number of positions held |
| rebalance_frequency_days | [3, 90] | [3, 90] | Rebalance cadence |

### Tunables shared across regimes (2)

| Tunable | Range | Notes |
|---|---|---|
| analyst_weight | [0.0, 0.30] | Analyst tiebreaker, regime-independent |
| macro_threshold_low | [0.10, 0.40] | Position sizing tier (50% threshold) |
| macro_threshold_high | [0.40, 0.70] | Position sizing tier (100% threshold) |

### New tunable (1)

| Tunable | Range | Notes |
|---|---|---|
| regime_threshold | [0.20, 0.60] | Splits macro signal into defensive vs offensive regime |

### Total search space

15 search dimensions (7 doubled tunables = 14, plus regime_threshold = 15, plus the 4 single-value tunables which add 4 more = 19 total search variables). Composite weight constraints reduce effective dimensionality but the optimization complexity is roughly 2x what Phase 0's full study explored.

## Architecture details

### Regime determination logic

At each rebalance:
1. Read current macro signal value
2. Compare to `regime_threshold`:
   - If macro_signal < regime_threshold → DEFENSIVE regime
   - If macro_signal >= regime_threshold → OFFENSIVE regime
3. Use the corresponding tunable set (defensive or offensive) for scoring, selection, sizing decisions, and stops
4. Position sizing uses the EXISTING macro_threshold_low/high logic INDEPENDENTLY:
   - macro_signal < macro_threshold_low: 50% sizing
   - macro_threshold_low <= macro_signal < macro_threshold_high: 75% sizing
   - macro_signal >= macro_threshold_high: 100% sizing
5. Both the regime tunable switch and the sizing tier apply simultaneously

### What's preserved from Phase 0

- 50/75/100% position sizing logic (the existing nuance is kept)
- Single composite scoring function structure
- Single backtest engine
- ATR-based trailing stops (just with regime-dependent multiplier)
- Earnings blackout windows
- Sector caps
- Min hold periods

### What changes

- BacktestConfig must hold defensive AND offensive variants of 7 tunables
- run_backtest's per-rebalance logic reads macro signal and selects which tunable set to use
- Optuna search space significantly larger
- Uses the new rolling-window objective: p75(rolling_12mo_CAPM_alpha) − 0.5 × max(0, -p25(rolling_12mo_CAPM_alpha))

## Optimization parameters

- TPE sampler, no pruner
- n_jobs=4 (heterogeneous workload from many varying tunables; parallelism returns)
- 1000 trials initial run
- DECISION RULE: if convergence isn't visible by trial 700 (best-score plateau, parameter consensus among top 50 trials), extend to 2000 trials
- Run on TRAINING window only: 2018-01-01 to 2023-12-31
- Cache snapshot: pre_v2_20260505 (locked input data)

## Acceptance criteria

The acceptance gates are deliberately strict. This is a major architectural change; it has to clear a meaningful bar to graduate.

A trial graduates to validation only if ALL of these hold on the training window:

### Gate 1: Material objective improvement

V1 must beat Phase 0's measured baseline by at least 20%:
- Phase 0 baseline: training objective score = +0.1543 (per snapshot pre_v2_20260505)
- V1 must achieve: training objective score >= +0.185

This validates that the architectural change actually produced a better solution under the new objective.

### Gate 2: Regime tunables actually differ

If V1's optimal defensive and offensive tunables are nearly identical, the architecture failed to find a meaningful split. Concretely:

- The composite weights for defensive vs offensive must differ by at least 0.10 (10%) on at least 2 of the 4 weight tunables
- OR atr_multiplier differs by at least 0.5 between regimes
- OR position_count differs by at least 3 between regimes
- OR rebalance_frequency_days differs by at least 14 days between regimes

This gates against "TPE just optimized harder; the regime architecture didn't help."

### Gate 3: Regime threshold lands in a usable range

The optimal regime_threshold must be in [0.25, 0.55]. If it's at either boundary (0.20 or 0.60), the optimizer pushed it to extreme — meaning the architecture is effectively single-regime. Reject in that case.

### Gate 4: Capture profile improves

Recovery capture at -10% drawdown must improve over Phase 0's measured baseline (Phase 0 had recovery_capture_-10pct = 48.6% in training). V1's training-window result must show >= 60% recovery capture at -10% drawdowns. This validates the architecture is doing what we hoped — capturing more of recoveries instead of staying defensive through them.

If ALL FOUR gates hold: V1 graduates to validation per the manual graduation step below.

If ANY gate fails: V1 does not graduate. Cache results, document specifically which gate failed and why, write a notes.md for V2 design. Phase 0 stays as locked baseline.

## Manual graduation step (only if all gates met)

Single-shot, deliberate, irreversible:

1. Identify V1's best training trial config from optuna_studies.db
2. Run validation ONCE on 2024-01-01 to 2026-04-30 window with that exact config
3. Record validation rolling-window distribution + capture metrics + recovery metrics
4. Compare against locked Phase 0 validation results (from pre_v2_20260505 phase0_baseline.json)
5. Decision tree:
   - V1 validation objective score materially better than Phase 0's (>= +20%) AND recovery capture at -10% improved AND no significant regression in other metrics: V1 graduates as new locked baseline
   - V1 validation objective comparable to Phase 0 (within ±5%) AND substantially better in capture profile (e.g., trending capture > 100% confirmed): graduates as a profile-improvement baseline even if objective tied
   - V1 validation objective worse than Phase 0 OR recovery capture regressed: V1 does NOT graduate. Cache results.

## Compute estimate

- 1000 trials × ~9-15s per trial (heterogeneous workload, n_jobs=4 with ~3-4x parallelism)
- Wall clock: ~50-90 min for 1000 trials
- If extended to 2000: ~2-3 hours

## Deliverables

- Study saved to optuna_studies.db: `regime_dependent_v1_<YYYYMMDD>_<HHMMSS>` (digits only, no special chars)
- Best-trial dashboard_results saved to `models/cache/dashboard_results/best_regime_dependent_v1_<YYYYMMDD>_<HHMMSS>_<trial>/`
- Standard 5 files plus extended meta.json with: hypothesis_id="regime-dependent-v1", regime_threshold, defensive tunables (7), offensive tunables (7), shared tunables, search_ranges, base_config_ref, window, cache_snapshot, promoted=false, rolling_metrics, capture metrics, recovery metrics

## Dashboard impact

- Cloud picker filtered to promoted-only studies (per existing discipline)
- regime_dependent_v1 studies stay invisible to dad/brother until graduation
- Once graduated, the new baseline replaces Phase 0 in the cloud picker

## What this hypothesis does NOT do

- Does NOT touch the validation window during training optimization
- Does NOT modify the universe (still 491 tickers, S&P 500 + NASDAQ-100)
- Does NOT change the underlying composite scoring or model — only the tunables that drive selection
- Does NOT add new alt signals, features, or data sources
- Does NOT chain into a different track. If results reveal interactions worth following up on, they become regime_dependent_v2 specs.

## Risks acknowledged at design time

**Risk 1: Optimizer finds defensive ≈ offensive tunables.** If TPE converges to nearly-identical values, the regime-dependence hypothesis is rejected. Gate 2 catches this.

**Risk 2: Regime threshold gets pushed to extreme.** If optimal threshold is 0.20 or 0.60 (boundary), the architecture is effectively single-regime. Gate 3 catches this.

**Risk 3: Search space explosion.** 19 search variables vs Phase 0's 9 means TPE has more landscape. 1000 trials may underconverge. Decision rule: extend to 2000 if convergence not visible by trial 700.

**Risk 4: Validation regime might not exercise the architecture.** The 2024-2026 validation window has limited offensive periods relative to training. If V1 graduates on training but the offensive tunables never fire much in validation, validation results may not reflect the architecture's full value. Mitigation: report fraction of validation trades made under each regime in the meta.json.

**Risk 5: Discovery cost.** This architecture might require multiple iterations (V1, V2, V3) before producing a meaningfully better baseline. Each iteration is a separate hypothesis run. Acceptable; budget time accordingly.

## Why this is regime_dependent_v1 (new track) not rebal_period_v2

The change in architecture is too large to fit under rebal_period. Rebal_period was about co-tuning a few related tunables. Regime_dependent fundamentally changes the strategy's structure: introduces regime detection, doubled tunable sets, regime-switching logic. New track.

Future tracks under regime_dependent might explore:
- 3-regime instead of 2-regime
- Different regime detection signals (not just macro)
- Hysteresis on regime transitions
- Multiple thresholds for different tunable subsets

## Launch command

Will require code changes in BacktestConfig, run_backtest, and optuna_runner before launch is possible. The launcher CLI may need extension to accept regime-aware tunable specifications.

```
python src\run_hypothesis.py --study-name regime_dependent_v1_<YYYYMMDD>_<HHMMSS> --hypothesis-id regime-dependent-v1 --base-config models/cache/dashboard_results/best_optuna_v1_20260504_103429_706/meta.json --architecture regime-dependent --window train --n-trials 1000 --n-jobs 4 --cache-snapshot pre_v2_20260505 --objective rolling_p75_p25
```

The `--architecture` flag is new and triggers the regime-dependent search space generation in optuna_runner.

---

*Authored: May 5, 2026 evening, end of session 3. Locked once V1 is launched. The architecture, search space, gates, and graduation criteria are committed. Implementation work follows in next session.*
