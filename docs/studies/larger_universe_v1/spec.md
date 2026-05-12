# Larger Universe v1 — Locked study spec (revised 2026-05-11 post-Phase-2 diagnostic)

This is the canonical spec for the Larger Universe v1 equity study, capturing all locked parameters after the Phase 2 horizon diagnostic revealed that cross-sectional alpha lives at monthly horizons rather than weekly.

## Why this revision

Original spec called for 5-day forward-return label + weekly rebalance. The Phase 2 horizon diagnostic (variants A and B, see `docs/diagnostics/larger_universe_v1_horizon_diagnostic.md`) demonstrated:
- At 5-day horizon: XGBoost collapses to constant-within-date predictions (learns macro features, ignores ticker-level features). Cross-sectional IC near zero.
- At 21-day horizon: full feature set produces meaningful cross-sectional IC (XGB 0.019, ENet 0.031 on SP500-actives subset) with no constant-prediction degeneracy and 4 of 5 folds positive.

Per Mike's authorization (2026-05-11 evening), the spec was revised to Option 3 from the diagnostic: monthly rebalance + monthly label horizon, which puts the modeling cadence where the signal actually lives.

## Locked parameters

### Universe
- `models/snapshots/equities/larger_universe_v1_20260511/` (snapshot, read-only)
- 1,963 tickers (current SP1500 actives + last-decade delisted with truncation at Wikipedia removed_at)
- Best-effort survivorship-bias mitigation; residual gaps documented in `docs/diagnostics/larger_universe_v1_snapshot_summary.md`

### Label
- **Forward 21-trading-day return** per (date, ticker): `close(D+21) / close(D) - 1`
- Approximates 1-calendar-month forward return (21 trading days ≈ 30 calendar days)
- NaN on last 21 rows of each ticker's series (filtered during training)
- Column name: `target` (horizon-agnostic in the training code; see `src/equities/study/labels.py`)

### Splits
| Split | Date range | Use |
|---|---|---|
| **Train** | 2017-05-12 → 2023-05-11 (~72 monthly rebalances) | CV folds; Optuna hyperparameter search |
| **Test** | 2023-05-12 → 2025-12-31 (~32 monthly rebalances) | Out-of-sample evaluation in Phase 4 |
| **Final OOS holdout** | 2026-01-01 → snapshot end (~4-5 monthly rebalances) | Only touched once at Phase 5 reporting |

### CV
- 5-fold expanding-window TimeSeriesSplit over the training window
- **Embargo: 21 trading days** (= label horizon, prevents train/val leakage)
- Scoring: **mean cross-sectional Spearman IC** across folds (per-date Spearman with min_tickers=30, averaged across dates within fold, then averaged across folds)
- Per-fold diagnostics also reported: `std_ic`, `positive_rate`, `n_dates_scored` (for review only, not part of objective)

### Models
- **XGBoost** (primary): native NaN + native categorical (sector). 9-parameter Optuna search space; max_depth 3-8, learning_rate 0.01-0.30 (log), n_estimators 100-800, etc. See `src/equities/study/training.py:_make_xgb_params`.
- **ElasticNet** (sanity check): SimpleImputer(mean, add_indicator=True) + StandardScaler + ElasticNet, sector one-hot encoded. 2-parameter Optuna search space: alpha (1e-5 to 1.0, log), l1_ratio (0.0 to 1.0).
- Both pipelines see identical features and identical folds.

### Features (38 total)
| Group | Count | Columns |
|---|---|---|
| Returns | 6 | ret_1d, ret_5d, ret_21d, ret_63d, ret_126d, ret_252d |
| Volatility | 2 | vol_21d, vol_63d |
| Trend | 3 | price_vs_ma50, price_vs_ma200, ma50_vs_ma200 |
| Drawdown | 1 | dd_252d |
| Fundamentals (PIT) | 7 | pe, pb, ps, debt_to_equity, roe, roa, profit_margin |
| Fundamentals (derived) | 2 | revenue_growth, eps_growth |
| Fundamentals (PIT computed) | 2 | dividend_yield, beta |
| Macro | 9 + 1 derived | yc_slope, vix, nfci, sahm, yc_3m, baa_spread, usd_index, unrate, wti_oil, + vix_5d_chg (`hy_spread` dropped — FRED data gap) |
| Categorical | 1 | sector (Finnhub finnhubIndustry; one-hot for ENet, native categorical for XGB) |
| Index membership | 3 | in_sp500, in_sp400, in_sp600 |
| Derived | 1 | log_market_cap |

Full coverage by feature × period in `docs/diagnostics/larger_universe_v1_features.md`.

### Portfolio construction (Phase 4)
- **Long-only**, score-weighted continuous sizing (model output → softmax-like weight transformation respecting caps)
- **Max single-position weight: 7.5%**
- **Max sector concentration (Finnhub sector): 30%** — applies per-sector including `sector_unknown` as its own bucket
- **Investment level: 95-100% (target 100%)** — no engineered cash timing
- **Soft constraint: 12-month rolling win rate vs SPY ≥ 60%** — penalty term in objective (Phase 4), not hard rejection

### Rebalancing
- **Cadence: monthly.** Rebalance day = last trading day of each month.
- **Execution convention: close-to-close** (compute target weights at month-end close, attribute the trade to the next trading day's close-to-close return). Confirmed simpler than open-attribution.
- **No threshold for trading.** Rebalance fully to target weights each month. Transaction costs accrue on actual position changes via the FeeModel (revised from original spec's 1.5pp threshold which was justified for weekly turnover; monthly cadence makes the threshold unnecessary).

### Cost model
- Match the three promoted studies' `BacktestConfig.transaction_cost_pct = 0.0005` (0.05% flat per trade leg)
- Frequency-agnostic — applies as `fee = proceeds × 0.0005` or `fee = cost × 0.0005` at every trade
- No bid-ask spread or market impact in v1 cost model

### Objective
- **Maximize excess CAGR vs SPY over the test period** as the headline study metric
- Cross-validation objective during Phase 3 tuning: mean cross-sectional Spearman IC across the 5 training-window folds

### Benchmarks (reported, not optimized against)
- SPY (S&P 500 cap-weighted)
- RSP (S&P 500 equal-weighted)
- IWM (Russell 2000)
- Equal-weight SP1500 (monthly-rebalanced custom benchmark, computed from snapshot prices)

## Revision log

| Date | Change | Reason |
|---|---|---|
| 2026-05-11 (Phase 1 gate) | Train window 2016-05-12 → 2017-05-12 | Long-lookback features need year of warmup |
| 2026-05-11 (Phase 1 gate) | Drop `hy_spread`, drop `tenure_in_index` | FRED data gap + Wikipedia coverage limit |
| 2026-05-11 (Phase 1 gate) | Move `dividend_yield` + `beta` from static to PIT | Look-ahead bias |
| 2026-05-11 (Phase 2 gate) | Fix IC: panel-wise → cross-sectional | Doc claimed cross-sectional but impl was panel |
| 2026-05-11 (Phase 2 gate) | Max_depth 3-10 → 3-8 | Deeper trees rarely useful on financial tabular data |
| **2026-05-11 (post-Phase-2 diagnostic)** | **Label 5d → 21d; rebalance weekly → monthly; embargo 5d → 21d; threshold removed** | **Cross-sectional alpha lives at monthly horizons (diagnostic Variant B)** |

## Pointers

- Snapshot: `models/snapshots/equities/larger_universe_v1_20260511/`
- Feature matrix: `models/features/larger_universe_v1/features.parquet`
- Training code: `src/equities/study/`
- CV design: `docs/diagnostics/larger_universe_v1_cv_design.md`
- Diagnostic report: `docs/diagnostics/larger_universe_v1_horizon_diagnostic.md`
- Session log: `docs/sessions/larger_universe_v1/session_log.md`
- Snapshot summary: `docs/diagnostics/larger_universe_v1_snapshot_summary.md`
