# Dashboard contract v1 — design proposal

**Status:** APPROVED 2026-05-12. Six review questions resolved (see "Decisions" section). Updated post-Phase-5 to require `objective.training_cv` in meta.json — see the "objective" subsection under meta.json (required) below and the companion memo `ml_study_cv_objectives_v1.md`. Larger Universe v1 Phase 4 produced the first contract-conformant artifacts; spec at `docs/studies/larger_universe_v1/phase4_spec.md`.
**Scope:** Define the universal data contract that contract-conformant studies write, and the dashboard tab structure that renders them. Larger Universe v1 is the first study under this contract; future studies use the same contract; the three promoted legacy studies stay on their existing tabs unchanged.
**Branch:** `feat/larger-universe-v1-study`

## Background

`src/dashboard_app.py` (3,868 lines) reads from:
- `models/cache/optuna_studies.db` (SQLite, Optuna's storage) for trial history
- `models/cache/dashboard_results/<label>/` (5 files per label) for per-config backtest results
- Hardcoded constants: `LOCKED_BEST_STUDY`, `STUDY_DISPLAY_NAMES`, plus a `BacktestConfig`-tightly-coupled UI

The Larger Universe v1 Phase 3 study writes to `models/studies/larger_universe_v1/` and uses a different modeling architecture (XGBoost score → continuous-sizing weight, monthly rebalance, 21d label) than what the legacy dashboard renders. Even if its outputs landed in the right paths, the legacy tabs' UI panels (composite-weight sliders, ATR sliders, regime-tier traffic lights) don't apply.

**Decision (Mike, 2026-05-12):** design a new universal results contract; build new contract-conformant tabs alongside the legacy tabs. Legacy stays as-is for the three promoted studies. Larger Universe v1 is the first study to land under the new contract.

## Current dashboard tab inventory + universality classification

| # | Tab | Content | Universal? |
|---|---|---|---|
| 1 | **Performance** | Equity curve, total return vs SPY, CAPM alpha+beta, drawdown, win rate, year-by-year | **Universal** — every study produces a NAV time series and a benchmark to compare against |
| 2 | **Current Holdings** | Current positions table, sector allocation | **Universal** — every study has a final-or-current position set |
| 3 | **Trade History** | Round-trip trades, holding periods | **Universal** — every study generates a trade log |
| 4 | **Market Context** | Today's macro tier (traffic light), composite macro score over time, threshold-based sizing logic | **Family-specific** — the traffic-light tier uses `config.macro_threshold_low/high` which only the composite-weighted regime studies have. The macro time-series chart underneath is universal. |
| 5 | **Risk & Behavior** | Up/down capture vs SPY, recovery times, rolling-12mo alpha distribution | **Universal** — derivable from NAV + benchmark series for any study |
| 6 | **Reliability** | V3 Track 2 single-axis perturbation results | **Family-specific** — purpose-built for v1's composite-weight axes (`weight_fundamental`, `weight_technical`, `atr_multiplier`, etc.). Doesn't map to ML hyperparameters. |
| 7 | **Tuning History** | Optuna trial scatter, score distribution, parameter importance | **Universal in spirit, currently coupled to SQLite** — needs to read from a JSON trial log instead of `optuna.get_all_study_names()` |
| 8 | **Glossary & Help** | Definitions of metrics | **Universal** (with study-family-specific subsections as needed) |

**Universal: 1, 2, 3, 5, 7, 8 (with Tuning History generalized to read JSON).**
**Family-specific (composite-weighted only): 4 (partially), 6.**

## Recommended new tab structure for contract-conformant studies

Same count as legacy (8 tabs), parallel naming for user familiarity. The two family-specific legacy tabs (Market Context tier, Reliability) get replaced with study-family-agnostic equivalents.

| # | Tab | What it shows |
|---|---|---|
| 1 | **Performance** | NAV curve vs 4 benchmarks (SPY, RSP, IWM, EW-SP1500 — or whatever the study declares), total return, CAPM alpha vs SPY, beta, drawdown, win rate, year-by-year bars, exec summary auto-generated from headline metrics |
| 2 | **Holdings** | Latest target weights table, sector allocation pie, position concentration histogram, weight-distribution by tier (SP500/400/600) if available |
| 3 | **Trades** | Turnover per period, trade list, holding-duration distribution, biggest contributors to PnL |
| 4 | **Risk & Behavior** | Up/down capture vs each benchmark, drawdown duration analysis, rolling Sharpe / IR, recovery times, monthly-return distribution |
| 5 | **Model Diagnostics** | NEW. For ML studies: cross-sectional IC time series, per-fold IC bars, feature importance (top N), prediction-vs-realized scatter, model-output distribution. For non-ML studies the tab still exists but renders an "n/a, this study has no ML component" placeholder — better than hiding the tab and breaking visual consistency. |
| 6 | **Market Context** | Macro indicator time series during the validation window (VIX, yield curve slope, BAA spread, USD index, etc. from `macro_signals_extended.parquet`). NO strategy-prescriptive traffic light — just the context. For composite-weighted studies (not contract-conformant), the legacy tab keeps the tier panel. |
| 7 | **Tuning History** | Trial scatter (objective value vs trial number), score distribution histogram, parameter importance (Optuna's `get_param_importances`), convergence trace at checkpoint intervals. Reads `trial_log.parquet` instead of SQLite. |
| 8 | **Glossary & Help** | Definitions section reused; adds Model Diagnostics terms (IC, IR, walk-forward, etc.) |

**Optional 9th tab when present:**

| # | Tab | Renders only if … |
|---|---|---|
| 9 | **Sensitivity / Walk-forward** | `walk_forward.parquet` is present in the study's contract directory. Year-by-year IC bars, rolling-3y window retrains, regime-conditional metrics. Larger Universe v1 will land this in Phase 5. |

### Rationale for the structure

- **Mike's "I like the current layout" constraint is honored** — same 8-tab count, 5 of the legacy tab names retained verbatim (Performance, Holdings, Trades, Risk & Behavior, Glossary), 2 renamed (Market Context loses the tier, Tuning History stays).
- **Reliability is dropped from the universal set** rather than kept-empty: V3 Track 2 perturbation is a very-specific kind of sensitivity test that doesn't generalize without a clear definition of what "axes" mean for an ML study. The Sensitivity / Walk-forward optional tab covers the same conceptual ground in a study-family-agnostic way.
- **Model Diagnostics is the new content** — gives ML studies somewhere to surface IC, feature importance, prediction quality. For non-ML studies it renders an n/a placeholder rather than disappearing, so the tab order stays consistent across studies.
- **Tuning History becomes JSON-trial-log-based** — decouples from SQLite, allowing any Optuna study (or any tuning framework that exports a trial log) to render here.

## Data contract v1 specification

### Path

```
models/studies/<study_name>/
├── contract_v1/                       <-- the renderable artifacts (this contract)
│   ├── meta.json                      (required)
│   ├── portfolio.parquet              (required)
│   ├── holdings.parquet               (required)
│   ├── trades.parquet                 (required)
│   ├── scores.parquet                 (optional; required for ML studies)
│   ├── trial_log.parquet              (optional; required for tuned studies)
│   ├── tuning_convergence.parquet     (optional; recommended for tuned studies)
│   ├── tuning_summary.json            (optional; recommended for tuned studies)
│   ├── feature_importance.parquet     (optional; required for ML studies)
│   ├── walk_forward.parquet           (optional; rendered if present)
│   └── regime_attribution.parquet     (optional; rendered if present)
└── hyperparameter_tuning/             <-- existing path for Phase 3 outputs; not part of the contract
    ├── <model>_best_params.json
    ├── <model>_study.json
    └── ...
```

The `contract_v1/` subdir separates dashboard-renderable artifacts from intermediate study workspace. Phase 4 produces contract_v1/. Phase 5 augments with walk_forward.parquet and regime_attribution.parquet.

**Auto-discovery:** the dashboard walks `models/studies/`, looks for any subdirectory containing `contract_v1/meta.json`, and surfaces those in the sidebar's "Contract-conformant studies" section. No code change needed when a new study lands — drop the artifacts at the right path and they appear.

### meta.json (required)

Single JSON file with the study's identity, family classification, and headline numbers.

```json
{
  "schema_version": "v1",
  "study_name": "larger_universe_v1",
  "display_name": "Larger Universe v1",
  "description": "XGBoost monthly cross-sectional alpha on SP1500-plus-delisted with 21d forward-return label.",
  "created_at": "2026-05-12T...Z",
  "spec_doc": "docs/studies/larger_universe_v1/spec.md",
  "family": "ml_cross_sectional",
  "models": [
    {"name": "xgboost",    "role": "primary",
     "params_path": "../hyperparameter_tuning/xgboost_best_params.json"},
    {"name": "elasticnet", "role": "sanity_check",
     "params_path": "../hyperparameter_tuning/elasticnet_best_params.json"}
  ],
  "universe": {
    "snapshot":      "larger_universe_v1_20260511",
    "size_total":    2122,
    "size_priced":   1963
  },
  "windows": {
    "train_start": "2017-05-12", "train_end": "2023-05-11",
    "test_start":  "2023-05-12", "test_end":  "2025-12-31",
    "oos_start":   "2026-01-01", "oos_end":   null
  },
  "rebalance": {
    "cadence":        "monthly",
    "day":            "last_trading_day_of_month",
    "execution":      "close_to_close_next_trading_day",
    "threshold_pp":   null
  },
  "label": {
    "horizon_trading_days": 21,
    "definition":           "close[t+21] / close[t] - 1"
  },
  "constraints": {
    "max_position_weight":    0.075,
    "max_sector_concentration": 0.30,
    "investment_level_range": [0.95, 1.00],
    "long_only":              true
  },
  "fee_model": {
    "transaction_cost_pct": 0.0005,
    "applies":              "per_trade_leg"
  },
  "benchmarks": ["SPY", "RSP", "IWM", "EW-SP1500"],
  "objective": {
    "training_cv": "top_quintile_spearman_ic",
    "headline": "excess_cagr_vs_spy"
  },
  "promoted": false,
  "phases": {
    "phase_3_complete": "2026-05-12T03:41Z",
    "phase_4_complete": null,
    "phase_5_complete": null
  },
  "summary_metrics": {
    "cv_mean_ic":              0.0282,
    "cv_per_fold_ic":          [0.0362, 0.0841, -0.0107, -0.0210, 0.0527],
    "test_cagr":               null,
    "test_excess_cagr_vs_spy": null,
    "test_max_drawdown":       null
  }
}
```

**Family values** (controlled vocabulary):
- `ml_cross_sectional` — Larger Universe v1 and similar
- `composite_weighted` — the three legacy studies (if backfilled later)
- `rule_based` — purely deterministic strategies
- `hybrid` — combines multiple

**`objective.training_cv` values** (controlled vocabulary, added 2026-05-12 post-Phase-5):

Every contract-conformant ML study must declare which CV objective it optimized against. Allowed values:

| Value | When to use |
|---|---|
| `top_quintile_spearman_ic` | **Recommended default for top-N portfolio strategies** per `ml_study_cv_objectives_v1.md`. Per-date Spearman IC restricted to top 20% of predictions, averaged across dates. |
| `mean_cross_sectional_spearman_ic` | Legacy / explicit deviation. Per-date Spearman across the full eligible universe. Larger Universe v1 used this and surfaced the misalignment that produced the recommendation. |
| `decile_spread` | Alternative top-N-relevant metric. Mean(top decile forward return) − mean(bottom decile forward return), averaged across dates. |
| `top_k_spearman_ic` | Restrict the per-date IC to exactly the top K positions the strategy holds. Aligned with deployment but noisy at small K. |
| `<other>` | Study-specific. Requires a rationale in `meta.json.notes` explaining why the standard options weren't used. |

**Dual-reporting requirement** (post-Phase-5 contract update): every ML study using a top-N construction MUST compute and persist BOTH the chosen `objective.training_cv` metric AND `mean_cross_sectional_spearman_ic` for the held-out evaluation. Both go into `meta.json.summary_metrics` (or the trial log). The CV optimizes against the chosen objective; the other metric is logged for comparison so empirical evidence accumulates over future studies. See `ml_study_cv_objectives_v1.md` for the rationale and the dual-reporting pattern.

**schema_version:** the dashboard reads this and routes to the appropriate renderer; `"v1"` is the only allowed value at this contract version. Future breaking changes bump to `"v2"`.

### portfolio.parquet (required)

Strategy NAV time series, **long format** on model — one row per (date, model). The wide-with-benchmark-columns variant was rejected in review; benchmarks live in their own file (below) so portfolio.parquet stays clean per-model and additional benchmarks don't require schema changes.

| Column | Type | Required | Notes |
|---|---|---|---|
| `date` | datetime64[ns] | yes | daily, sorted ascending |
| `model` | string | yes | model identifier matching `meta.json.models[].name` |
| `nav` | float64 | yes | normalized to 1.0 at start of evaluation window |
| `cash_pct` | float64 | yes | fraction in cash on this date |
| `n_positions` | int32 | yes | count of open positions |
| `gross_exposure` | float64 | yes | total long exposure as fraction of portfolio |

For studies with multiple models (e.g., Larger Universe v1 has XGB primary + ENet sanity), each model gets its own rows. The dashboard renders the `meta.json.models[].role == "primary"` model by default; other models reachable via a sidebar dropdown.

### benchmarks.parquet (required)

Benchmark NAV time series, long format on benchmark.

| Column | Type | Required | Notes |
|---|---|---|---|
| `date` | datetime64[ns] | yes | daily, sorted ascending; aligned with portfolio.parquet dates |
| `benchmark` | string | yes | matches one of `meta.json.benchmarks` entries |
| `nav` | float64 | yes | normalized to 1.0 at start of evaluation window (same as portfolio.parquet) |

The dashboard strictly renders only the benchmarks listed in `meta.json.benchmarks`. No auto-add of SPY or any default.

### holdings.parquet (required)

Long format. One row per (date, model, ticker) for non-zero positions.

| Column | Type | Required | Notes |
|---|---|---|---|
| `date` | datetime64[ns] | yes | rebalance date or any specified snapshot date |
| `model` | string | yes | |
| `ticker` | string | yes | |
| `weight` | float64 | yes | target weight as fraction of portfolio (0.0–`max_position_weight`) |
| `value_usd` | float64 | no | dollar value if known; otherwise null |
| `sector` | string | no | sector classification at this date |
| `tier` | string | no | universe tier ("SP500"/"SP400"/"SP600"/"removed"/"sector_unknown") |

To render "current holdings", the dashboard filters to the most recent date per model.

### trades.parquet (required)

Long format. One row per trade execution.

| Column | Type | Required | Notes |
|---|---|---|---|
| `date` | datetime64[ns] | yes | execution date (close of trade day) |
| `model` | string | yes | |
| `ticker` | string | yes | |
| `action` | string | yes | "buy" or "sell" |
| `weight_change` | float64 | yes | signed weight delta from previous rebalance |
| `price` | float64 | yes | execution price |
| `notional_usd` | float64 | yes | dollar amount transacted |
| `fee_usd` | float64 | yes | transaction cost paid |
| `reason` | string | no | "rebalance" / "stop_loss" / "delisting" / etc. |

### scores.parquet (optional; required for ML studies)

Per-(date, model, ticker) model predictions, with realized labels attached for retrospective IC computation.

| Column | Type | Required | Notes |
|---|---|---|---|
| `date` | datetime64[ns] | yes | score / rebalance date |
| `model` | string | yes | |
| `ticker` | string | yes | |
| `score` | float64 | yes | raw model output (e.g., predicted 21d return) |
| `rank` | int32 | yes | cross-sectional rank on this date (1 = highest) |
| `target_realized` | float64 | no | actual realized forward return (for retrospective IC); null until the realization window passes |

Used by Model Diagnostics tab to plot IC over time, rank-stability, and prediction-vs-realized scatter.

**Size cap.** Soft default of 1M rows. When `scores.parquet` exceeds 1M rows, the study MUST also write `scores_sampled.parquet` with the same schema but only every Nth rebalance date (sampling rule documented in `meta.json.notes`). The dashboard reads `scores_sampled.parquet` if present and falls back to `scores.parquet` otherwise. Larger Universe v1 produces ~120K rows (~32 monthly rebalances × ~1,900 tickers × 2 models) and stays well under the cap; future daily-rebalance studies will need the sampled variant.

### trial_log.parquet (optional; required for tuned studies)

Optuna trial-by-trial log in tabular form. One row per trial per tuning study (XGB and ENet contribute separate rows).

| Column | Type | Required | Notes |
|---|---|---|---|
| `tuning_study` | string | yes | e.g., "xgboost", "elasticnet" |
| `trial_number` | int32 | yes | 0-indexed |
| `state` | string | yes | "COMPLETE" / "FAIL" / "PRUNED" |
| `value` | float64 | yes | objective value (NaN if failed) |
| `duration_s` | float64 | yes | trial elapsed time |
| `param_<name>` | float64/int/string | varies | one column per hyperparameter searched |

Used by Tuning History tab. Replaces the SQLite Optuna integration for contract-conformant studies.

### tuning_convergence.parquet (optional; recommended for tuned studies)

Per-trial running-best convergence trace. One row per trial per tuning study. Derivable from `trial_log.parquet` but pre-computed at study time so the dashboard renders without re-deriving on every page load.

| Column | Type | Required | Notes |
|---|---|---|---|
| `model` | string | yes | matches the names declared in `meta.json.models[].name` (renamed from `trial_log.tuning_study` to align with the cross-artifact `model` convention) |
| `trial_number` | int32 | yes | 0-indexed |
| `score` | float64 | yes | the trial's CV objective score (NaN for non-COMPLETE trials) |
| `running_best_score` | float64 | yes | max score seen up to and including this trial |
| `best_so_far_trial` | int32 | yes | trial number that achieved the current running best |
| `ms_since_start` | int64 | yes | cumulative wall-clock from trial 0 to this trial's completion, in milliseconds (`duration_s.cumsum() * 1000`) |

Used by Tuning tab's convergence-curve section. Skipping non-COMPLETE trials when computing running-best is the recommended convention so the curve reflects the optimizer's actual best-known state.

### tuning_summary.json (optional; recommended for tuned studies)

Per-model tuning summary — single JSON object keyed by model name. Headline numbers the Tuning tab's narrative summary box quotes directly.

```json
{
  "<model_name>": {
    "total_trials": 200,
    "winning_trial": 150,
    "winning_score": 0.0282,
    "mean_score": 0.0213,
    "std_score": 0.0055,
    "winner_zscore": 1.26,
    "trials_to_95pct_winning": 18,
    "pct_trials_to_plateau": 0.09
  },
  ...
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `total_trials` | int | yes | count of COMPLETE trials in the model's tuning study |
| `winning_trial` | int | yes | trial number with max score |
| `winning_score` | float | yes | max score |
| `mean_score` | float | yes | mean across COMPLETE trials |
| `std_score` | float | yes | sample std across COMPLETE trials (`ddof=1`) |
| `winner_zscore` | float | yes | `(winning_score - mean_score) / std_score`; NaN if `std_score == 0` |
| `trials_to_95pct_winning` | int | yes | first trial number where `running_best >= 0.95 * winning_score`; convergence-plateau measurement |
| `pct_trials_to_plateau` | float | yes | `trials_to_95pct_winning / total_trials`; convenient for narrative copy |

Used by Tuning tab's narrative summary box. The narrative format reads:

> The optimizer tested {total_trials} configurations for {model_name}. The winner was Trial #{winning_trial} with score {winning_score}. 95% of the winning score was reached after about {pct_trials_to_plateau:.0%} of the trials — the curve plateaus early, then refinement happens at the margin. Optuna is search, not proof.

**Schema versioning note.** Both `tuning_convergence.parquet` and `tuning_summary.json` are **additive v1 changes** — the contract `schema_version` stays at `"v1"`. Dashboards must treat both files as optional and render the prior `trial_log`-only Tuning view (with a graceful "convergence data not pre-computed" caption) when they're absent. Future studies should produce them as part of Phase 3; Larger Universe v1 receives them via a one-time back-fill from `trial_log.parquet`.

### feature_importance.parquet (optional; required for ML studies)

| Column | Type | Required | Notes |
|---|---|---|---|
| `model` | string | yes | |
| `feature` | string | yes | feature name |
| `importance` | float64 | yes | model-specific importance (XGBoost gain; ElasticNet abs(coef)) |
| `rank` | int32 | yes | within-model rank by importance |
| `importance_type` | string | yes | "gain" / "split" / "abs_coef" / etc. |

### walk_forward.parquet (optional; Phase 5 produces this)

Rolling-window stability check.

| Column | Type | Required | Notes |
|---|---|---|---|
| `window_start` | date | yes | |
| `window_end` | date | yes | |
| `model` | string | yes | |
| `mean_ic` | float64 | yes | |
| `std_ic` | float64 | no | |
| `total_return` | float64 | no | window-period total return |
| `sharpe` | float64 | no | annualized Sharpe within window |
| `excess_cagr_vs_spy` | float64 | no | |

### regime_attribution.parquet (optional)

Performance breakdown by named market regimes.

| Column | Type | Required | Notes |
|---|---|---|---|
| `regime` | string | yes | e.g., "covid_crash_2020Q1", "ai_rally_2023Q4" |
| `model` | string | yes | |
| `start_date` | date | yes | |
| `end_date` | date | yes | |
| `strategy_return` | float64 | yes | |
| `spy_return` | float64 | yes | |
| `excess_return` | float64 | yes | |
| `cross_sectional_ic` | float64 | no | mean IC in this regime |

## Schema versioning

`meta.json.schema_version` is the single point of truth.

- `"v1"`: this proposal. Frozen once Mike + partners approve.
- Future changes:
  - **Additive** (new optional columns, new optional files, new optional `meta.json` fields): no version bump. Dashboard treats missing optional fields as "feature not supported by this study".
  - **Breaking** (removed columns, renamed columns, changed semantics of existing fields): bump to `"v2"`. The dashboard reads `schema_version` and routes to the v1 or v2 renderer accordingly. v1 studies render under v1 renderer indefinitely.

The dashboard's contract-conformant tab loaders are gated by:

```python
if meta["schema_version"] != "v1":
    st.warning(f"Study {study_name} uses schema {meta['schema_version']}; only v1 supported in this dashboard.")
    return
```

The legacy `dashboard_results/<label>/` path is the implicit v0. It's never touched by the contract-conformant tabs.

## Sidebar UX

The sidebar gets a new top-level section. Layout:

```
=== Sidebar ===

[Legacy studies]
  Source:   [Default config / Best trial of selected study / Custom trial #N]
  Study:    [optuna_v1_20260504_103429 / ...]

[Contract-conformant studies]
  Study:    [larger_universe_v1 / ...future...]
  Model:    [xgboost (primary) / elasticnet (sanity_check)]
```

The choice of section determines which tab set renders:
- Legacy section selected → existing 8 tabs (Performance, Holdings, Trades, Market Context, Risk & Behavior, Reliability, Tuning History, Glossary)
- Contract-conformant section selected → new 8 tabs (Performance, Holdings, Trades, Risk & Behavior, Model Diagnostics, Market Context, Tuning History, Glossary) + optional Sensitivity tab

Both sections always visible; user toggles by clicking which study to load. Avoids modal/radio-button switching.

## Implementation phasing (post-approval)

1. **Phase 4 (next, after this proposal's approval)** — modify Phase 4 spec to produce contract_v1/ artifacts for Larger Universe v1. Runs the backtest, saves per-spec outputs.
2. **Phase 4.5** — add the contract-conformant dashboard tab set. Includes the discovery walker for `models/studies/<name>/contract_v1/meta.json`, the sidebar section, and the 8 tab renderers reading the contract files. Estimate ~1-2 days of dashboard code.
3. **Phase 5** — walk-forward + OOS + writeup. Adds `walk_forward.parquet` and `regime_attribution.parquet` to the existing contract_v1/ dir. The Sensitivity tab auto-appears.
4. **R2 sync** — `snapshot_for_cloud.py` already walks `models/cache/` for legacy artifacts; needs a parallel `models/studies/*/contract_v1/` walker. Small modification.

## What's out of scope for contract v1

Deliberately excluded to keep the contract narrow:

- **Live trading data** — the contract is for backtest results only. Live paper-trading state stays in its existing location.
- **Multi-asset-class artifacts** — v1 contract is equity-specific. Options/crypto studies would design their own contract or extend this one.
- **Comparative tabs across studies** — "compare Larger Universe v1 vs regime_dependent_v1 on the same period" is a separate feature. v1 contract renders one study at a time.
- **Live alerts / monitoring** — out of scope.

## Decisions (resolved 2026-05-12)

The six review questions are answered. The decisions are now part of the contract.

1. **`scores.parquet` size cap: 1M rows.** Studies producing more must also write `scores_sampled.parquet` (every Nth rebalance) per the rule documented under that file's schema. Dashboard prefers the sampled variant when present. Larger Universe v1 stays under the cap.
2. **Benchmarks declaration: strict.** Dashboard renders exactly what's in `meta.json.benchmarks` — no auto-add of SPY. Studies declare their benchmark list explicitly. Larger Universe v1 declares `["SPY", "RSP", "IWM", "EW-SP1500"]`.
3. **`promoted` flag: match legacy semantics.** Default `false`; manually flipped to `true` after explicit promotion review. Larger Universe v1 stays `promoted: false` through Phase 5; promotion is a separate decision based on the success-criteria evaluation in Phase 5.
4. **Holdings tab default: always-show-latest, with date picker.** "Latest" is computed dynamically from the holdings.parquet's max date — advances naturally as Phase 5 adds OOS data, no hard-coded sentinel.
5. **Model Diagnostics tab visibility: data-driven, no flag.** The tab renders iff BOTH `scores.parquet` AND `feature_importance.parquet` exist for the selected study. Hidden otherwise. The contract is "files determine UI"; no `has_model_diagnostics` boolean.
6. **Multi-model studies: default to primary.** `meta.json.models[].role` controls. Dashboard renders the role=`"primary"` model by default; sidebar dropdown selects others. **No overlay of multiple models in default views** — clutters the headline and forces the user into compare-mode they didn't ask for. Compare-mode is a future feature.

## Next steps post-approval

1. Modify Phase 4 spec to produce contract-conformant output at `models/studies/larger_universe_v1/contract_v1/`. Spec doc lives separately at `docs/studies/larger_universe_v1/phase4_spec.md`.
2. Run Phase 4. Produces contract-conformant artifacts.
3. Phase 4.5 — implement the new universal dashboard tabs reading from the contract location.
4. Phase 5 — walk-forward + OOS + writeup. Adds `walk_forward.parquet` and `regime_attribution.parquet` to the same contract directory; Sensitivity tab auto-appears.
5. `snapshot_for_cloud.py` — extend with parallel walker for `models/studies/*/contract_v1/`. Small modification.
