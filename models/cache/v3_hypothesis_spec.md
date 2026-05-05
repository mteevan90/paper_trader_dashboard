# Model Test V3 — Hypothesis Spec (corrected)

Locked at session 3 close (May 4, 2026 evening). Read before running V3.

This is the corrected version. The original spec had factual errors about the Phase 0 baseline (claimed 5-day rebalance; actual is 35 days) which led to biased acceptance criteria. This version uses the verified baseline values and removes the biased framing.

---

## Hypothesis

Test whether the current locked rebalance frequency (35 days, monthly) is actually optimal, or whether a different frequency in the range [5, 60] days (weekly through ~quarterly) produces a better training-window result. All other tunables held at Phase 0 trial #706 best values.

This is a no-preconception test. We are NOT assuming longer is better. We are NOT assuming shorter is better. We are isolating one tunable, expanding its search range substantially beyond the original optimization space, and asking Optuna to find an optimum.

## Why this hypothesis is worth testing

The original v1 study searched `rebalance_frequency_days` over a narrow range ([14, 63]). That study converged at 35 days. But:

- Optuna found a local optimum within that range, not necessarily a global one
- A range that excludes weekly (~5-7 day) rebalancing might be missing meaningful momentum-capture territory
- The full economic question — "what cadence works best for this composite-scored strategy?" — was never explored at the short end

V3 expands to [5, 60] specifically to surface whether the original range was too narrow. This is honest exploration, not validation of a prior belief.

## What V3 tests specifically

- Vary `rebalance_frequency_days` over [5, 60] — wider than the original [14, 63] specifically to include weekly cadences
- Hold all other tunables at Phase 0 trial #706 best values:
  - weight_fundamental: 0.4774
  - weight_technical: 0.1296
  - weight_model: 0.3841
  - weight_alt: 0.0088 (derived)
  - macro_threshold_low: 0.2552
  - macro_threshold_high: 0.4972 (gap derived = 0.2420)
  - atr_multiplier: 3.247
  - analyst_weight: 0.1116
  - position_count: 10
- Run on TRAINING window only: 2018-01-01 to 2023-12-31
- Use existing TPE sampler, no pruner, n_jobs=4
- 1000 full trials (per locked architecture)

## What V3 does NOT do

- Does NOT touch the validation window (2024-01-01 to 2026-04-30). Validation is locked, runs once after manual graduation, never automatically.
- Does NOT change composite weights, ATR floor/cap, sector cap, min hold, earnings blackout, universe, or any tunable other than rebalance_frequency_days
- Does NOT add new features or alt signals
- Does NOT chain into V4 — if V3 reveals an interaction effect (e.g., longer rebalance only works with different position count), that's V4's hypothesis, set up separately

## Acceptance criteria — what counts as "promotable"

The acceptance criteria are unbiased — V3 wins if it materially improves on the baseline OR matches it with notably different rebalance cadence. V3 fails if its training-window optimum is statistically indistinguishable from Phase 0's value.

A V3 trial graduates to validation only if at least ONE of these holds on the training window:

1. **Materially better:** V3's best trial training Sharpe is at least 5% higher than Phase 0's training Sharpe (which is 1.52 per locked baseline)
2. **Materially cheaper at comparable performance:** V3's best trial trade count is at least 30% lower than Phase 0's training trade count, AND training Sharpe is at least 95% of Phase 0
3. **Materially different cadence with comparable or better performance:** V3's best `rebalance_frequency_days` differs from baseline 35 by at least 15 days (i.e., lands at most 20 or at least 50), AND training Sharpe is at least 95% of Phase 0

If V3 doesn't meet any of these: V3 is dead, do NOT graduate. Cache the V3 study results forever for retrospective analysis but do not run validation. Phase 0 stays as locked baseline.

If V3 meets criteria: see manual graduation below.

## Manual graduation step (only if acceptance criteria met)

Single-shot, deliberate, irreversible:

1. Identify V3's best training trial config from `models/cache/optuna_studies.db`
2. Run validation ONCE on 2024-01-01 to 2026-04-30 window with that exact config
3. Record validation alpha, Sharpe, max DD, and trade count
4. Compare against locked Phase 0 validation results (-1.49pp annualized alpha, 1.29 Sharpe, -16.32% max DD, 201 trades)
5. Decision tree:
   - **V3 validation alpha is materially better than Phase 0 (+0.5pp or more) AND drawdown is comparable or better:** V3 graduates, becomes new locked baseline. Update meta.json with `"promoted": true`. Update tracker.
   - **V3 validation alpha is comparable to Phase 0 (within plus or minus 0.3pp) AND trade count is significantly lower:** V3 graduates as a comparable-but-cheaper baseline. Reasoning: lower trade count = lower real-world cost.
   - **V3 validation alpha is worse than Phase 0 by more than 0.5pp OR drawdown is materially worse:** V3 does NOT graduate. Cache results. Do NOT re-run with different parameters in pursuit of a better validation result.
6. Whatever the validation showed: do not run another validation against the same window with V3-tuned parameters. The validation is spent.

## Compute estimate

- Per the v8 tracker, full Optuna study at 1000 trials × 2-3s per trial post-Tier-1-vectorization = 30-50 min wall clock
- Single-machine, one nightly run is straightforward
- Storage: study DB grows by ~few MB per V3 run, negligible

## Deliverables (what gets generated when V3 is run)

- V3 study saved to `models/cache/optuna_studies.db` under name `optuna_v3_<YYYYMMDD>_<HHMMSS>`
- Best-trial dashboard_results saved to `models/cache/dashboard_results/best_optuna_v3_<YYYYMMDD>_<HHMMSS>_<trial_num>/`
- Each result directory contains the standard 5 files: holdings.json, meta.json, portfolio.parquet, scores.json, trades.parquet
- meta.json includes additional V3 metadata: hypothesis_id, study_name, trial_number, search_ranges, fixed_tunables, base_config_ref, window, promoted false
- Promoted flag becomes true only after manual graduation

## Dashboard impact

- Cloud dashboard's "Best trial of selected study" picker filters to promoted only (per filter shipped session 3)
- Experimental V3 studies stay invisible to dad/brother
- Only graduated, validated studies show up
- This means dad/brother will NOT see V3 results until you've made an explicit graduation decision

## What to remember when running V3

The whole point of this discipline is that the validation window stays honest. If you run V3 and the training results look beautiful, the temptation is to "just check what validation says, doesn't hurt to look." That look is exactly what destroys the validation-window discipline. Don't peek. Either commit to validation as a one-shot graduation, or don't run validation at all.

If V3 doesn't meet acceptance criteria on training: cache the data, document why it failed, move on to V4. Do not iterate against validation results.

---

## V3 launch command (Archetype 3 launcher)

Once src/run_hypothesis.py is built, V3 launches with:

```
python src\run_hypothesis.py ^
  --study-name optuna_v3_<YYYYMMDD>_<HHMMSS> ^
  --hypothesis-id v3-rebalance-frequency ^
  --base-config models/cache/dashboard_results/best_optuna_v1_20260504_103429_706/meta.json ^
  --override-range rebalance_frequency_days=5,60 ^
  --hold-tunables-fixed weight_fundamental,weight_technical,weight_model,macro_threshold_low,macro_threshold_high,atr_multiplier,analyst_weight,position_count ^
  --window train ^
  --n-trials 1000
```

The launcher reads the base-config meta.json, holds the listed tunables at those values, expands rebalance_frequency_days to [5, 60], and runs 1000 TPE trials.

---

*Authored: May 4, 2026 evening, end of session 3. Corrected version. Not to be modified once V3 is run; future Vn hypotheses get their own spec files.*
