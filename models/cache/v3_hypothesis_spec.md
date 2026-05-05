# Model Test V3 — Hypothesis Spec

Locked at session 3 close (May 4, 2026 evening). Read before running V3.

---

## Hypothesis

Reducing rebalance frequency from current locked Phase 0 baseline (high-churn, ~583 trades over 2.3 years ≈ rebalance every ~1.4 trading days) to longer intervals (weekly, biweekly, monthly) will:
- Reduce trading drag and slippage
- Stabilize positions and reduce churn-driven noise
- Possibly improve Sharpe even if absolute returns drop slightly

Trading frequency drag is a documented source of underperformance in systematic strategies. 583 trades in 2.3 years is high. If a 14-day or 21-day rebalance produces comparable training Sharpe with materially fewer trades, that's a strong signal that the current rebalance regime is too aggressive.

## What V3 tests specifically

- Vary `rebalance_frequency_days` over a wider Optuna search range than the current locked Phase 0
- Current range probably ~[3, 21]. Expand to **[5, 60]** to include weekly through monthly cadences
- Hold all other tunables at current Phase 0 best values during initial V3 sweep — isolate the frequency effect from compound interaction with other tunables
- Run on TRAINING window only: 2018-01-01 to 2023-12-31
- Use existing TPE sampler, no pruner, n_jobs=4
- 1000 full trials (per locked architecture)

## What V3 does NOT do

- Does NOT touch the validation window (2024-01-01 to 2026-04-30). Validation is locked, runs once after manual graduation, never automatically.
- Does NOT change composite weights (35/25/25/15 + analyst tiebreaker remain fixed)
- Does NOT change ATR floor/cap, sector cap, min hold, earnings blackout, or universe
- Does NOT add new features or alt signals
- Does NOT chain into V4 — if V3 reveals an interaction effect (e.g., longer rebalance only works with different position count), that's V4's hypothesis, set up separately

## Acceptance criteria — what counts as "promotable"

A V3 trial only graduates to validation if ALL of these hold on the training window:

1. The training-window optimum has a rebalance frequency notably different from Phase 0 (e.g., Phase 0 trial #706 was at ~5-day rebalance — V3 winner should land at >=10 days to be interesting)
2. Training Sharpe is at least 90% of Phase 0 training Sharpe (no catastrophic regression)
3. Trade count is substantially lower than Phase 0 (validates that the longer rebalance is actually changing behavior, not getting overridden by other rules)

If V3 doesn't meet all three: V3 is dead, do NOT graduate. Cache the V3 study results forever for retrospective analysis but do not run validation. Phase 0 stays as locked baseline.

## Manual graduation step (only if acceptance criteria met)

Single-shot, deliberate, irreversible:

1. Identify V3's best training trial config from `models/cache/optuna_studies.db`
2. Run validation ONCE on 2024-01-01 to 2026-04-30 window with that exact config
3. Record validation alpha, Sharpe, max DD, and trade count
4. Compare against locked Phase 0 validation results (-1.49pp annualized alpha, 1.29 Sharpe, -16.32% max DD, 201 trades)
5. Decision tree:
   - **V3 validation alpha is materially better than Phase 0 (+0.5pp or more) AND drawdown is comparable or better:** V3 graduates, becomes new locked baseline. Update `meta.json` with `"promoted": true`. Update tracker.
   - **V3 validation alpha is comparable to Phase 0 (within ±0.3pp) AND trade count is significantly lower:** V3 graduates as a comparable-but-cheaper baseline. Reasoning: lower trade count = lower real-world cost.
   - **V3 validation alpha is worse than Phase 0 by >0.5pp OR drawdown is materially worse:** V3 does NOT graduate. Cache results. Do NOT re-run with different parameters in pursuit of a "better" validation result.
6. Whatever the validation showed: do not run another validation against the same window with V3-tuned parameters. The validation is spent.

## Compute estimate

- Per the v8 tracker, full Optuna study at 1000 trials × 2-3s per trial post-Tier-1-vectorization = 30-50 min wall clock
- Single-machine, one nightly run is straightforward
- Storage: study DB grows by ~few MB per V3 run, negligible

## Deliverables (what gets generated when V3 is run)

- V3 study saved to `models/cache/optuna_studies.db` under name `optuna_v3_<YYYYMMDD>_<HHMMSS>`
- Best-trial dashboard_results saved to `models/cache/dashboard_results/best_optuna_v3_<YYYYMMDD>_<HHMMSS>_<trial_num>/`
- Each result directory contains the standard 5 files: `holdings.json`, `meta.json`, `portfolio.parquet`, `scores.json`, `trades.parquet`
- `meta.json` includes a `"promoted": false` flag by default (becomes true only after manual graduation)
- After validation graduation: `meta.json` is updated with `"promoted": true` AND validation results are written into `meta.json` for dashboard display

## Dashboard impact

- Cloud dashboard's "Best trial of selected study" picker filters to `promoted: true` only (per the promoted-vs-experimental filter shipped session 3)
- Experimental V3 studies stay invisible to dad/brother
- Only graduated, validated studies show up
- This means dad/brother will NOT see V3 results until you've made an explicit graduation decision

## What to remember when running V3

The whole point of this discipline is that the validation window stays honest. If you run V3 and the training results look beautiful, the temptation is to "just check what validation says, doesn't hurt to look." That look is exactly what destroys the validation-window discipline. Don't peek. Either commit to validation as a one-shot graduation, or don't run validation at all.

If V3 doesn't meet acceptance criteria on training: cache the data, document why it failed, move on to V4. Do not iterate against validation results.

---

*Authored: May 4, 2026 evening, end of session 3. Not to be modified once V3 is run; future Vn hypotheses get their own spec files.*
