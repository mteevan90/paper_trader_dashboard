# Larger Universe v1 — Phase 4 implementation spec

**Status:** DRAFT (2026-05-12). Awaiting Mike's review before any Phase 4 code runs.
**Scope:** Portfolio construction + backtest engine + four-benchmark comparison, producing **contract v1**-conformant output that the new universal dashboard tabs (Phase 4.5) will render.
**Contract:** `docs/architecture/dashboard_contract_v1.md` (approved 2026-05-12)
**Branch:** `feat/larger-universe-v1-study`

## Inputs

| Input | Path | Notes |
|---|---|---|
| Best XGBoost params | `models/studies/larger_universe_v1/xgboost_best_params.json` | From Phase 3 |
| Best ElasticNet params | `models/studies/larger_universe_v1/elasticnet_best_params.json` | From Phase 3 |
| Feature matrix | `models/features/larger_universe_v1/features.parquet` | 4.35M rows × 40 cols |
| Universe map | `docs/larger_universe_v1_universe.json` | 2,180 records (1,506 active + 674 removed) |
| Price snapshot | `models/snapshots/equities/larger_universe_v1_20260511/price_cache/*.parquet` | For benchmarks + EW-SP1500 construction |
| Macro signals | `models/features/larger_universe_v1/macro_signals_extended.parquet` | For Market Context tab |
| Locked spec | `docs/studies/larger_universe_v1/spec.md` | Study-wide parameters |

## Output location

```
models/studies/larger_universe_v1/contract_v1/
├── meta.json                          (required)
├── portfolio.parquet                  (required, long format)
├── benchmarks.parquet                 (required, long format)
├── holdings.parquet                   (required, per-rebalance-date only)
├── trades.parquet                     (required)
├── scores.parquet                     (ML-required)
├── trial_log.parquet                  (tuning-required; derived from Phase 3 study JSONs)
└── feature_importance.parquet         (ML-required; SHAP-based with gain fallback)
```

Phase 5 will add `walk_forward.parquet` and possibly `regime_attribution.parquet` to the same directory; the Sensitivity tab in the new dashboard auto-appears when those land.

## Phase 4 implementation decisions (concrete for Larger Universe v1)

### 1. Long-format portfolio.parquet — locked

One row per (date, model). Strategy NAV only. Benchmarks live in `benchmarks.parquet` (also long-format on `benchmark` column).

Concretely for Larger Universe v1:
- `model` values: `"xgboost"` (primary) and `"elasticnet"` (sanity_check)
- Date range: test window 2023-05-12 → 2025-12-31 PLUS OOS 2026-01-01 → snapshot end (~250 weekday dates in OOS slice). Phase 4 produces both; Phase 5 evaluates OOS separately.
- NAV starts at 1.0 on 2023-05-12 (first test-window date).
- Columns: `date`, `model`, `nav`, `cash_pct`, `n_positions`, `gross_exposure`.

`benchmarks.parquet`:
- `benchmark` values: `"SPY"`, `"RSP"`, `"IWM"`, `"EW-SP1500"`
- NAV starts at 1.0 on 2023-05-12 (matching the strategy series).
- SPY/RSP/IWM: read close prices from each benchmark's parquet, normalize. Phase 4 fetches these via Finnhub if not in the snapshot (they're not — universe is SP500/400/600 constituents, indexes aren't constituents). One-time fetch outside the Phase 3 cache structure.
- **EW-SP1500**: custom benchmark. Constructed per-rebalance-date as: take all tickers in `universe.json` that are `status=="active"` on that date AND have price data in the snapshot AND have a non-null price at the rebalance date; equal-weight them; rebalance monthly using the same monthly cadence as the strategy. Cost model identical to strategy (0.05% flat per trade leg). NAV normalized to 1.0 on 2023-05-12.

### 2. SHAP feature importance with gain-based fallback

Compute SHAP values for the XGBoost final model on a sample of the test window's prediction rows (sampling because SHAP at full-universe-full-test-window scale is expensive).

Implementation:
- Train final XGBoost on full training window (2017-05-12 → 2023-05-11) with locked best params
- Use `xgboost.Booster.predict(..., pred_contribs=True)` for fast tree-SHAP (much faster than the general `shap.TreeExplainer` for XGBoost specifically)
- Compute on a sample of N=10,000 (date, ticker) rows from the test window — distributed across rebalance dates and tickers
- Aggregate: mean absolute SHAP value per feature → primary importance metric
- Persist to `feature_importance.parquet` with `importance_type="shap_mean_abs"`

**Fallback:** if SHAP compute exceeds 10 minutes wall-clock (some XGBoost interaction-depth combinations are unexpectedly slow), abort and use gain-based importance from `xgb_model.feature_importances_`. Set `importance_type="gain"` in the parquet and add a note to `meta.json.notes` explaining the fallback occurred and why.

ElasticNet feature importance: absolute coefficient magnitude (after StandardScaler standardization within the pipeline so coefficients are comparable across features). `importance_type="abs_coef"`.

For each model, write a row per feature with rank computed within-model.

### 3. Per-rebalance holdings only — locked

`holdings.parquet` contains rows only on actual rebalance dates (last trading day of each month). No daily interpolation of "weights are still these values until next rebalance" — the dashboard's holdings tab handles inter-rebalance constancy on its own. Roughly 32 rows per model in the test window + ~5 rows per model in the OOS holdout.

Each row carries:
- `date` (rebalance date)
- `model`
- `ticker`
- `weight` (target weight as fraction of portfolio)
- `value_usd` (computed from notional capital × weight on the rebalance date; assumes $1,000,000 starting capital per `BacktestConfig.starting_capital`)
- `sector` (from `models/features/larger_universe_v1/sector_map.json`; "sector_unknown" for delisted)
- `tier` (from `universe.json`: SP500/SP400/SP600/removed)

Only rows with `weight > 0` are written (long-only spec means no shorts to track).

### 4. Trades.parquet detail

One row per non-zero `weight_change` at each rebalance. Columns per the contract:

| Column | Larger Universe v1 specifics |
|---|---|
| `date` | rebalance date (= execution date under close-to-close-next-day convention; the "next trading day's close" is the realized fill price) |
| `model` | xgboost / elasticnet |
| `ticker` | |
| `action` | "buy" if weight_change > 0 else "sell" |
| `weight_change` | target_weight_new − target_weight_old; +0.025 means new position at 2.5% |
| `price` | Finnhub close on the next-trading-day post-rebalance (the close-to-close fill price) |
| `notional_usd` | abs(weight_change) × portfolio_value_at_rebalance |
| `fee_usd` | 0.0005 × notional_usd |
| `reason` | "rebalance" for all monthly rebalances; "delisting_truncation" for forced exits when a removed ticker hits its Wikipedia removed_at date |

### 5. Scores.parquet detail

One row per (rebalance_date, model, ticker) for every ticker in the eligible universe on that date. Stays under the 1M-row cap:
- 32 monthly rebalance dates in test window + 5 in OOS = 37 dates
- ~1,900 eligible tickers per date (full universe minus same-date NaN-target/NaN-feature filters)
- 2 models
- = 140K rows. Single `scores.parquet` is fine; no `scores_sampled.parquet` needed.

Columns:
- `date`, `model`, `ticker`
- `score` — raw model output (predicted 21-day forward return)
- `rank` — cross-sectional rank on that date (1 = highest predicted return)
- `target_realized` — actual realized 21-day forward return; null for the last 21 trading days of OOS where the realization window doesn't close

### 6. trial_log.parquet from Phase 3 study JSONs

Derive from the existing `models/studies/larger_universe_v1/xgboost_study.json` and `elasticnet_study.json`. Flatten the per-trial records into the contract's required schema.

Columns per the contract:
- `tuning_study`: "xgboost" or "elasticnet"
- `trial_number`, `state` ("COMPLETE"/"FAIL"/"PRUNED"), `value` (cross-sectional IC), `duration_s`
- One `param_<name>` column per searched hyperparameter (XGB has 9, ENet has 2)

This file is a one-time transformation from Phase 3 outputs to contract format. No re-tuning runs in Phase 4.

### 7. meta.json content (concrete for Larger Universe v1)

```json
{
  "schema_version": "v1",
  "study_name": "larger_universe_v1",
  "display_name": "Larger Universe v1",
  "description": "XGBoost monthly cross-sectional alpha on SP1500-plus-delisted universe with 21-day forward-return label. ElasticNet sanity check.",
  "created_at": "<set at write time, UTC ISO>",
  "spec_doc": "docs/studies/larger_universe_v1/spec.md",
  "family": "ml_cross_sectional",
  "models": [
    {"name": "xgboost",    "role": "primary",
     "params_path": "../hyperparameter_tuning/xgboost_best_params.json"},
    {"name": "elasticnet", "role": "sanity_check",
     "params_path": "../hyperparameter_tuning/elasticnet_best_params.json"}
  ],
  "universe": {
    "snapshot":    "larger_universe_v1_20260511",
    "size_total":  2122,
    "size_priced": 1963
  },
  "windows": {
    "train_start": "2017-05-12", "train_end": "2023-05-11",
    "test_start":  "2023-05-12", "test_end":  "2025-12-31",
    "oos_start":   "2026-01-01", "oos_end":   null
  },
  "rebalance": {
    "cadence":      "monthly",
    "day":          "last_trading_day_of_month",
    "execution":    "close_to_close_next_trading_day",
    "threshold_pp": null
  },
  "label": {
    "horizon_trading_days": 21,
    "definition":           "close[t+21] / close[t] - 1"
  },
  "constraints": {
    "max_position_weight":      0.075,
    "max_sector_concentration": 0.30,
    "investment_level_range":   [0.95, 1.00],
    "long_only":                true
  },
  "fee_model": {
    "transaction_cost_pct": 0.0005,
    "applies":              "per_trade_leg"
  },
  "benchmarks": ["SPY", "RSP", "IWM", "EW-SP1500"],
  "objective": {
    "training_cv":   "mean_cross_sectional_spearman_ic",
    "headline":      "excess_cagr_vs_spy"
  },
  "promoted": false,
  "phases": {
    "phase_3_complete": "2026-05-12T03:41Z",
    "phase_4_complete": "<set at Phase 4 completion>",
    "phase_5_complete": null
  },
  "summary_metrics": {
    "cv_mean_ic":              0.0282,
    "cv_per_fold_ic":          [0.0362, 0.0841, -0.0107, -0.0210, 0.0527],
    "test_cagr":               "<computed at Phase 4 completion>",
    "test_excess_cagr_vs_spy": "<computed at Phase 4 completion>",
    "test_max_drawdown":       "<computed at Phase 4 completion>"
  },
  "notes": []
}
```

`notes` is a list of free-text strings; Phase 4 appends here if SHAP fallback happened, or any other Phase-4-time discovery.

## Score-to-weights transformation (Phase 4 portfolio construction)

The spec calls for "score-weighted continuous sizing respecting caps". Concrete algorithm:

1. **On each rebalance date D**, get raw model scores for every eligible ticker (`target_fwd_5d` model output — using the model's prediction of 21d return as the score signal).
2. **Filter eligibility:** ticker must be active (per universe.json `status` and `removed_at` checks), have a non-NaN score, have a non-NaN price on D, and meet basic liquidity (already implicit via the snapshot's coverage).
3. **Rank within the cross-section** for date D. Apply softmax with temperature `T=0.1` (chosen so the top decile gets ~80% of the weight before caps — tunable but starting with T=0.1).
4. **Apply 7.5% individual cap.** If any ticker's softmax weight exceeds 7.5%, clip to 7.5% and redistribute the excess proportionally to other below-cap positions. Iterate until no ticker exceeds 7.5%.
5. **Apply 30% sector cap.** Sum weights by `meta.json.universe.snapshot`'s sector_map. If any sector exceeds 30%, scale down all tickers in that sector proportionally and redistribute the excess to under-30% sectors. Iterate until no sector exceeds 30%.
6. **Re-normalize to sum to 1.0** (fully invested at 100%; the 95-100% range exists for cases where caps prevent reaching 100%, in which case the remainder is cash).
7. **Output:** target weight per ticker for date D. Difference from prior month's weights determines trades.

**The transformation is deterministic given scores + universe state.** Score-to-weight code lives in `src/equities/study/portfolio.py` (new file Phase 4 creates).

## Backtest execution flow

```
for rebalance_date D in monthly_rebalance_dates(test_window + oos_window):
    1. Score every eligible ticker for date D with the model
    2. Apply score-to-weights transformation -> target_weights[D]
    3. Compute trades = target_weights[D] - target_weights[prev D]
    4. Compute fees = 0.0005 × |notional traded|
    5. NAV(D) = NAV(prev D) × (1 + portfolio_return_between_rebalances - fees)
    6. Record portfolio.parquet row, holdings.parquet rows, trades.parquet rows, scores.parquet rows
```

The "portfolio_return_between_rebalances" uses daily close-to-close returns aggregated. For tickers exiting via delisting between rebalances, position is force-closed at the delisting-day close (Wikipedia removed_at) and the freed weight is added to cash until the next rebalance.

## Benchmark construction

- **SPY / RSP / IWM**: fetch close prices via Finnhub for 2023-05-12 through OOS end. Normalize to NAV starting at 1.0 on 2023-05-12. One-time fetch cached at `models/cache/equities/finnhub/prices/{SPY,RSP,IWM}.parquet` (SPY already exists from Phase 3). Estimated runtime: ~3 minutes for the 2 new ones (RSP + IWM).
- **EW-SP1500 (custom)**: monthly-rebalanced equal-weight portfolio of all active SP1500 constituents per `universe.json`. Same 0.05% fee model as the strategy. Starts at NAV 1.0 on 2023-05-12.

## Estimated Phase 4 runtime

| Step | Wall-clock estimate |
|---|---|
| Final XGBoost training on full train window | ~5 minutes |
| Final ElasticNet training | ~30 seconds |
| Fetch SPY/RSP/IWM via Finnhub | ~3 minutes |
| Compute scores on test + OOS window per model | ~5 minutes per model (so 10 min total) |
| Backtest loop (monthly rebalances, ~37 dates × 2 models) | ~5 minutes |
| EW-SP1500 benchmark construction | ~2 minutes |
| SHAP feature importance (sampled 10K rows) | ~5–15 minutes |
| Convert Phase 3 study JSONs → trial_log.parquet | ~10 seconds |
| Write contract_v1 artifacts | ~1 minute |
| **Total** | **~35–50 minutes** |

Compared to Phase 3's 5h+ tuning run, Phase 4 is short — most of the compute already happened. No backgrounding needed; can run inline if Mike approves the spec.

## What Phase 4 does NOT do

- **No walk-forward analysis** — that's Phase 5
- **No regime attribution** — Phase 5
- **No final OOS reporting** — Phase 5 reports OOS numbers; Phase 4 just produces the OOS slice of artifacts for posterity (the dashboard's "Performance" tab will show test + OOS combined in NAV; Phase 5 separates them in the writeup)
- **No tracker update** — next tracker update is Phase 4 completion per the standing rule
- **No dashboard code modifications** — Phase 4.5 handles that
- **No promotion decision** — `promoted: false` in meta.json; promotion is its own decision after Phase 5

## Open items for review before Phase 4 starts

1. **Softmax temperature T=0.1**: starting choice for the score-to-weight transformation. Plausible alternative: rank-based (top-N positions, equal weight within top-N). Worth confirming the softmax-with-T=0.1 choice or specifying an alternative.
2. **EW-SP1500 active-on-date definition**: a ticker is "active on date D" if `universe.json.status == "active"` OR (`status == "removed"` AND `removed_at > D`). Confirms inclusion of historically-active-but-now-delisted tickers in their pre-delisting periods. Reasonable per the survivorship-bias-mitigation framing.
3. **SHAP sample size 10K rows**: starting choice. If SHAP is fast enough (<5 min), bump to 50K for tighter importance estimates. If SHAP is too slow, fall back per the spec.
4. **Forced-exit accounting**: a ticker that delists mid-month gets force-closed at the delisting-day close and the weight goes to cash. Alternative: distribute the freed weight proportionally to other holdings same-day. The cash-until-next-rebalance approach is simpler and matches the monthly-cadence philosophy. Confirming.
5. **OOS slice handling**: Phase 4 produces NAV through snapshot end (today, 2026-05-12) which includes the OOS window. Phase 5 evaluates OOS separately for the writeup. Phase 4 doesn't gate on OOS — it just produces the data. Confirming.

Awaiting Mike's review before any Phase 4 code runs.
