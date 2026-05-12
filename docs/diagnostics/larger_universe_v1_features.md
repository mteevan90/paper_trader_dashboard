# Larger Universe v1 — Feature engineering coverage report (Phase 1)

**Output file:** `models/features/larger_universe_v1/features.parquet`
**Shape:** 4,350,932 rows × 40 columns (date, ticker + 38 features after Phase-1-gate cleanup)
**Date range:** 2016-05-12 → 2026-05-11
**Tickers:** 1,963 priced (out of 2,122 in the universe; 159 missing prices are mostly pre-warranty delistings)

## Files produced during Phase 1

| Path | Purpose |
|---|---|
| `models/features/larger_universe_v1/features.parquet` | Main feature matrix keyed on (date, ticker) |
| `models/features/larger_universe_v1/fundamentals_pit.parquet` | Point-in-time quarterly fundamentals with 45-day reporting lag (196,622 rows × 39 cols across 1,972 tickers) |
| `models/features/larger_universe_v1/macro_signals_extended.parquet` | 10-column FRED macro panel (2016-01-01 → 2026-05-11) |
| `models/features/larger_universe_v1/sector_map.json` | Aggregated Finnhub /stock/profile2 results (1,782 entries with sector + shares_outstanding + IPO date) |

The legacy snapshot's `cache/fundamentals.json` and `cache/macro_signals.parquet` are NOT modified. The architectural note for the snapshot README is at the end of this doc.

## Feature inventory (39 features total)

The spec asked for ~29 features after dropping tenure-in-index; the final count is 39 because:
- All 10 macro signals are kept (spec named 7 + we kept the 3 extras already in the snapshot: hy_spread, nfci, sahm, yc_3m for hy_spread proxy and other extras)
- VIX 5-day-change is computed as a separate column (vix_5d_chg)

| Group | Count | Features |
|---|---|---|
| Returns | 6 | `ret_1d`, `ret_5d`, `ret_21d`, `ret_63d`, `ret_126d`, `ret_252d` |
| Volatility | 2 | `vol_21d`, `vol_63d` (annualized daily-log-return std) |
| Trend | 3 | `price_vs_ma50`, `price_vs_ma200`, `ma50_vs_ma200` |
| Drawdown | 1 | `dd_252d` (close / rolling 252d max − 1) |
| Fundamentals (PIT) | 7 | `pe` (peTTM), `pb`, `ps` (psTTM), `debt_to_equity` (totalDebtToEquity), `roe` (roeTTM), `roa` (roaTTM), `profit_margin` (netMargin) |
| Fundamentals (derived TTM) | 2 | `revenue_growth`, `eps_growth` (computed as TTM-vs-TTM-year-ago from quarterly series) |
| Fundamentals (static, point-in-2026) | 2 | `dividend_yield`, `beta` — see disclaimer below |
| Macro | 9 | `yc_slope`, `vix`, `nfci`, `sahm`, `yc_3m`, `baa_spread`, `usd_index`, `unrate`, `wti_oil` (`hy_spread` DROPPED at Phase-1 gate — FRED `BAMLH0A0HYM2` only serves 2023+ data) |
| Macro derived | 1 | `vix_5d_chg` (5-day diff of VIX) |
| Categorical | 1 | `sector` (Finnhub `finnhubIndustry` taxonomy; will be one-hot-encoded in Phase 2) |
| Index membership | 3 | `in_sp500`, `in_sp400`, `in_sp600` |
| Market cap | 1 | `log_market_cap` (derived as `close × shares_outstanding`) |

**Tenure in index — DROPPED.** Per the Phase-1 design decision: Wikipedia component-change tables only track add events ~10 years back; ~86% of currently-active records have null `added_at`. Substituting a default biases the feature; cleaner to remove.

**`ps` column** — Finnhub's `series.quarterly` doesn't have a `ps` field (only `psTTM` and annual `ps`). We use `psTTM`. Documented in the per-feature mapping.

## Per-feature non-null coverage (overall)

| Feature | Non-null rows | % of 4,350,932 |
|---|---|---|
| ret_1d | 4,348,969 | **100.0%** |
| ret_5d | 4,341,117 | 99.8% |
| ret_21d | 4,309,721 | 99.1% |
| ret_63d | 4,227,432 | 97.2% |
| ret_126d | 4,104,040 | 94.3% |
| ret_252d | 3,857,693 | 88.7% |
| vol_21d | 4,309,721 | 99.1% |
| vol_63d | 4,227,432 | 97.2% |
| price_vs_ma50 | 4,254,858 | 97.8% |
| price_vs_ma200 | 3,961,210 | 91.0% |
| ma50_vs_ma200 | 3,961,210 | 91.0% |
| dd_252d | 3,859,645 | 88.7% |
| **hy_spread** | **1,215,093** | **27.9% ⚠** |
| yc_slope, vix, nfci, sahm, yc_3m, baa_spread, usd_index, unrate, wti_oil, vix_5d_chg | 4,350,932 | 100.0% |
| revenue_growth | 3,869,125 | 88.9% |
| eps_growth | 3,914,950 | 90.0% |
| pe | 3,388,347 | 77.9% |
| pb | 3,929,661 | 90.3% |
| ps (psTTM) | 3,696,796 | 85.0% |
| debt_to_equity | 4,113,737 | 94.5% |
| roe | 4,005,067 | 92.1% |
| roa | 4,174,659 | 95.9% |
| profit_margin | 3,934,067 | 90.4% |
| **dividend_yield** | **2,936,648** | **67.5%** (mostly non-dividend-paying companies) |
| beta | 4,252,860 | 97.7% |
| sector | 4,350,932 | 100.0% (incl. `sector_unknown`) |
| log_market_cap | 4,252,965 | 97.7% |
| in_sp500, in_sp400, in_sp600 | 4,350,932 | 100.0% |

## Per-year coverage (key features)

Tickers per year and key-feature non-null fractions year-by-year:

| Year | Tickers | ret_252d | vol_63d | price_vs_ma200 | dd_252d | pe | roe | hy_spread |
|---|---|---|---|---|---|---|---|---|
| 2016 | 1,716 | 0.000 | 0.607 | 0.000 | 0.000 | 0.708 | 0.815 | 0.000 |
| 2017 | 1,758 | 0.624 | 0.994 | 0.834 | 0.624 | 0.748 | 0.864 | 0.000 |
| 2018 | 1,792 | 0.978 | 0.994 | 0.983 | 0.974 | 0.764 | 0.892 | 0.000 |
| 2019 | 1,824 | 0.980 | 0.996 | 0.984 | 0.977 | 0.770 | 0.898 | 0.000 |
| 2020 | 1,861 | 0.982 | 0.995 | 0.986 | 0.981 | 0.770 | 0.910 | 0.000 |
| 2021 | 1,886 | 0.974 | 0.993 | 0.979 | 0.974 | 0.797 | 0.928 | 0.000 |
| 2022 | 1,826 | 0.985 | 0.999 | 0.990 | 0.985 | 0.789 | 0.932 | 0.000 |
| 2023 | 1,774 | 0.992 | 0.997 | 0.993 | 0.991 | 0.797 | 0.944 | 0.634 |
| 2024 | 1,688 | 0.989 | 0.998 | 0.992 | 0.988 | 0.795 | 0.949 | 1.000 |
| 2025 | 1,611 | 0.994 | 0.999 | 0.996 | 0.994 | 0.798 | 0.953 | 1.000 |
| 2026 | 1,535 | 0.997 | 0.999 | 0.998 | 0.998 | 0.806 | 0.956 | 1.000 |

### Reading the table

- **2016 long-lookback features are 0% non-null.** The 252-day return, 200-day MA, and 252-day drawdown need 252 trading days of lookback. With data starting 2016-05-12 in the snapshot, those features become populated around 2017-05-12. The early 2016 rows are pre-warmup. **Effective training window with full features: 2017-05-12 onward.**
- **2017 is partially warm** (62.4% on ret_252d). Tickers that IPO'd after 2016-05-12 don't yet have a year of history.
- **Ticker count declines after 2021** because of delistings (SIVB, FRC, BBBY, ATVI, TWTR, CTXS, KSU, etc.) and the truncation we apply at Wikipedia removed_at.
- **`hy_spread` (FRED `BAMLH0A0HYM2`) is only populated from 2023-05-12.** FRED's HY OAS series was restructured or limited at the data source; only the last ~3 years are returned. **`hy_spread` should NOT be used as a feature in Phase 2 — it would create implicit "is this 2023+?" leakage.** The new `baa_spread` (Moody's BAA minus 10y, FRED `BAA10Y`) covers the full window and is the recommended credit-spread proxy.

## Design decisions made in Phase 1 (with rationale)

### Fundamentals — point-in-time with reporting lag

Used the Finnhub `/stock/metric` raw cache files' `series.quarterly` sub-dict to build a per-ticker quarterly history of ~35 financial metrics. Applied a 45-day reporting lag: for date `D` and ticker `T`, the lookup is the most recent quarter `Q` such that `Q.period_end + 45_days <= D`. This is industry-standard practice — companies report Q3 fundamentals in November (not on September 30) so using `period <= date` literally would create a 30-45 day look-ahead bias.

**Coverage trade-off:** the static `fundamentals.json` in the snapshot covered 1,919 / 2,122 = 90.4% of tickers. The PIT version covers **1,972 / 2,122 = 92.9%** of tickers — better, because the series field is more comprehensive than the snapshot's metric block. Per-(date, ticker) coverage of any one fundamental field hovers around 77–96% depending on the metric, with the loss attributable to (a) tickers with no quarterly history at all (123 of 2,119 metric JSONs are empty — mostly delisted entities), (b) tickers whose first reportable quarter is after the feature row's date (early-IPO history), and (c) Finnhub data gaps for specific metrics on specific companies.

### Static fundamentals — dividend_yield and beta

`dividend_yield` (Finnhub `currentDividendYieldTTM`) and `beta` are NOT in the quarterly series field. They appear only in the top-level `metric` snapshot — a current-as-of-2026-05-11 reading. We use these AS STATIC FEATURES per-ticker (constant across all dates).

**This introduces a mild look-ahead.** A company's beta and dividend yield can change over a 10-year window. For mature large-caps the values are reasonably stable but they're not literally point-in-time. **Mitigation:** documented in Phase 5 disclaimer; alternative would be to drop both features for v1. I've kept them because XGBoost should weight them low if they're unhelpful and dropping forces us to lose two of the spec's 11 fundamental features.

Phase 2 / Phase 5 disclaimer text candidate:
> "dividend_yield and beta are sourced from a single 2026-05-11 snapshot per ticker and held constant across all training dates. Mature companies' values vary slowly over a 10-year window; emergent growth stocks (e.g., recent IPOs) may have stale-looking values in early training rows. The look-ahead bias from this is bounded and small for most cases but is acknowledged as a known approximation."

### Macro signals — extended FRED fetch, 10 columns

Extended `src/macro_signals.py`'s 6-series fetch to 10 series by adding `BAA10Y` (BAA-AAA credit spread, full coverage 2016+), `DTWEXBGS` (USD trade-weighted broad index), `UNRATE` (civilian unemployment, monthly), `DCOILWTICO` (WTI oil daily). Output written to `models/features/larger_universe_v1/macro_signals_extended.parquet` — the snapshot's original `macro_signals.parquet` is unchanged (per the "don't modify the snapshot" constraint, treating the new file as an additive artifact).

The original `hy_spread` series (ICE BofA HY OAS, FRED `BAMLH0A0HYM2`) is preserved for compatibility but has the coverage limitation above. The new `baa_spread` is the recommended credit-spread feature for Phase 2.

### Sector — Finnhub /stock/profile2

Probe confirmed the endpoint works on Basic. Fetched profile2 for all 2,122 tickers (60/min, ~33 min wall-clock). 1,782 returned a populated body; 340 (mostly delisted entities) returned `{}`.

**Taxonomy:** Finnhub uses `finnhubIndustry` strings ("Technology", "Health Care", "Banking", etc.) — coarser than GICS Sub-Industry but similar in granularity to GICS Sector. We use these as-is (no translation to legacy lowercase labels) so future studies have a stable, vendor-provided taxonomy.

**Missing coverage:** the 340 unknowns are assigned `sector_unknown`. Per the Phase 1 design decision, the 30% sector concentration cap (Phase 4) treats `sector_unknown` as its own bucket: tickers in it can't collectively exceed 30% of the portfolio, but there's no within-bucket concentration limit beyond the 7.5% per-position cap. Document this in Phase 5.

### Log market cap — derived from current shares × historical price

Formula: `log_market_cap(date, ticker) = log(close(date, ticker) × shares_outstanding(ticker))` where `shares_outstanding` comes from Finnhub `/stock/profile2`'s `shareOutstanding` (in millions, current as of 2026-05-11). Falls back to the static fundamentals' `marketCapitalization` × `(close/last_close)` for tickers with no profile2 entry.

**Coverage: 97.7%** of feature rows. The 2.3% missing are tickers with no profile2 data AND no static `marketCapitalization` — a small set of pre-warranty delistings.

**Look-ahead caveat:** uses current shares outstanding for all historical rows. Buybacks and issuances change share counts over time; for the ranking purposes the model uses log_market_cap (large vs mid vs small), the buyback noise is small relative to price-driven variation. Documented.

### Architectural note for the snapshot README

The snapshot's `cache/fundamentals.json` (1,919 entries) is the current-as-of-2026-05-11 metric block per ticker. **Historical fundamentals time series ARE available at `models/cache/equities/finnhub/metrics/<SYM>.json`** in the `series.annual` and `series.quarterly` sub-fields — they just weren't promoted to the snapshot. The Phase 1 PIT extraction (`fundamentals_pit.parquet`) uses these raw files as the source.

Recommendation: add a note to the snapshot's README documenting this architecture, so the next person doesn't recreate the look-ahead-biased mistake of using the static `fundamentals.json` for backtest features.

## Phase-1-gate resolutions (applied 2026-05-11 before Phase 2)

The following changes were applied at the Phase 1 gate per Mike's review:

1. **`hy_spread` dropped from the feature matrix.** Feature count drops 39 → 38; macro count drops 10 → 9. `baa_spread` is the recommended credit-spread feature for Phase 2.
2. **Training window trimmed: 2017-05-12 → 2023-05-11** (6 years). Phase 2 train/CV code will apply this trim at modeling time; the features.parquet retains all dates for completeness and to allow the test/OOS windows to be drawn from the same artifact.
3. **`sector_unknown` treated as a single normal sector** for both XGBoost (native categorical) and ElasticNet (one-hot column). The 30% sector concentration cap (Phase 4) treats `sector_unknown` as one bucket — collective cap of 30%, no internal limit.
4. **`dividend_yield` and `beta` replaced with point-in-time computations** (separate follow-up commit). See "PIT dividend yield and beta" section below.
5. **Snapshot README annotated** with a note explaining that historical fundamentals live in the raw cache at `models/cache/equities/finnhub/metrics/<SYM>.json` (not in the snapshot's `fundamentals.json` which is the current-only metric block).

## Issues and open questions for Phase 2

1. **Early training window (2016-05-12 → 2017-05-12) has very sparse long-lookback features.** Options:
   - Trim training start to 2017-05-12 (1y of warmup) — clean cut, loses 1y of training data
   - Use XGBoost's native NaN handling and accept the sparse rows in the first year
   - Use 2017-05-12 for ElasticNet (needs imputation anyway) and 2016-05-12 for XGBoost (NaN-tolerant)

2. **`hy_spread` is non-usable as a feature** (only 27.9% coverage, all post-2023). **Recommend dropping from the Phase 2 feature set.** `baa_spread` is the replacement. Net macro feature count: 10 - 1 = 9 (or 10 including vix_5d_chg).

3. **`sector_unknown` rows constitute 14% of the feature matrix.** Most are post-removal-date rows for historical delistings — they're in the training set but flagged as unknown sector. XGBoost will treat them as a category; ElasticNet's one-hot encoding will give them their own column. Need to confirm this is the desired behavior in Phase 2.

4. **Static fundamentals look-ahead.** `dividend_yield` and `beta` are point-in-2026 values used across all training dates. Spec disclaimer needed in Phase 5. Could be dropped if you prefer a stricter PIT discipline.

5. **Per-feature coverage matrix by tier/status** is computable from the universe.json `tier`/`status` joined with the feature matrix — not included in this report for length but easy to surface in Phase 2 if needed.

## Numbers ready for Phase 2

- **Feature matrix:** 4,350,932 rows × 41 columns
- **Effective full-feature training rows:** ~3.9M (excluding 2016-2017 warmup with sparse long-lookback features)
- **Train split (2016-05-12 → 2023-05-11):** approximately 3.1M rows pre-warmup-filter, ~2.8M with full features
- **Test split (2023-05-12 → 2025-12-31):** approximately 1.1M rows
- **Final OOS holdout (2026-01-01 → snapshot end):** approximately 200K rows
- **Tickers in train: ~1,886 peak (2021); in test: ~1,774→1,688; in OOS: ~1,535**

## Reproducibility

To rebuild the feature matrix from scratch (assuming Finnhub data + FRED key + universe.json all in place):

```
python scripts/research/build_macro_signals_extended.py
python scripts/research/fetch_finnhub_profile2.py      # ~33 min wall-clock for fresh fetch
python scripts/research/build_fundamentals_pit.py
python scripts/research/build_features_larger_universe_v1.py
```

Each script is idempotent against existing outputs. Profile2 fetch is resumable from cache; the others are clean rebuilds.
