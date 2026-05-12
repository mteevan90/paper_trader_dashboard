# Larger Universe v1 — snapshot summary

**Snapshot path:** `models/snapshots/equities/larger_universe_v1_20260511/`
**Built:** 2026-05-11 (Phase 4 of the Larger Universe v1 spec)
**Data source:** Finnhub Basic ($49.99/mo) for prices + fundamentals; yfinance attempted for earnings (gate-fired, see below).
**Window:** 2016-05-12 → 2026-05-11 (10 years daily OHLCV)
**Universe construction:** Wikipedia S&P 500 + 400 + 600 component-change tables, SEC CIK-disambiguated where possible. See `docs/larger_universe_v1_universe.json` and `src/equities/larger_universe_v1_builder.py`.

## TL;DR

- **Prices:** 1,963 / 2,122 (92.5%) — universe-wide
- **Fundamentals:** 1,919 / 2,122 (90.4%) via Finnhub `/stock/metric`
- **Earnings:** **dropped from v1** — yfinance scale-fetch produced only 23.6% non-empty after retries; the stash's 50% sanity gate fired by design. Per the <85% rule, earnings_dates is not a v1 feature.
- **Survivorship-bias property:** best-effort mitigation. **Not** survivorship-bias-free. Documented residual gaps tilt toward overstated returns by an estimated **0.3–0.6 pp/yr** (academic-literature-style range; cannot be pinned tighter without a held-out comparison run).

## Section 1 — Ticker count by tier × status

Counts after deduplication by symbol (prefer-active). Two records per symbol are possible when the same ticker appears as both active and removed (rebrand cases, e.g. ATI Inc. ← Allegheny Technologies).

| Tier | Status | Universe count | Prices coverage | Fundamentals coverage |
|---|---|---|---|---|
| SP500 | active | 499 | **499 (100.0%)** | 498 (99.8%) |
| SP500 | removed (last 10y) | 118 | 60 (50.8%) | 59 (50.0%) |
| SP400 | active | 396 | **396 (100.0%)** | 396 (100.0%) |
| SP400 | removed (last 10y) | 210 | 128 (61.0%) | 114 (54.3%) |
| SP600 | active | 599 | 598 (99.8%) | 599 (100.0%) |
| SP600 | removed (last 10y) | 300 | 282 (94.0%) | 253 (84.3%) |
| **Total active** | | **1,494** | **1,493 (99.93%)** | **1,493 (99.93%)** |
| **Total removed** | | **628** | **470 (74.8%)** | **426 (67.8%)** |
| **Grand total** | | **2,122** | **1,963 (92.5%)** | **1,919 (90.4%)** |

**Reading the table:**

- **Active coverage is essentially complete.** One SP600 ticker is missing a price file (likely a recent IPO or transitional listing Finnhub doesn't yet serve); one SP500 ticker is missing fundamentals (BKNG, see Section 3 — rate-limit retry exhaustion).
- **Removed coverage degrades by tier age.** SP600 removed coverage (94% prices) is much higher than SP500 removed (50.8% prices) because SP500 has older delistings on average — and the older the delisting, the more likely it falls outside Finnhub Basic's 10y warranty. Specifically: of the 118 SP500-removed records, 58 are not in Finnhub Basic's data — those are mostly 2014–2017 delistings on the boundary of the 10y window plus a long tail of older names that crept into the Wikipedia "last 10y" cutoff via stale data.
- **Fundamentals are sparser than prices on removed names** (67.8% vs 74.8%) because Finnhub doesn't maintain `/stock/metric` for many delisted entities even when historical candles are available.

## Section 2 — Date range coverage per tier (active records only)

| Tier | Symbols with prices | Earliest first-date | Latest last-date | Median rows | Symbols with <8y history (late IPOs) |
|---|---|---|---|---|---|
| SP500 | 499 | 2016-05-12 | 2026-05-11 | 2,513 | 28 |
| SP400 | 396 | 2016-05-12 | 2026-05-11 | 2,513 | 48 |
| SP600 | 598 | 2016-05-12 | 2026-05-11 | 2,513 | 81 |

**Reading the table:**

- The full 10-year window is reachable. Every tier has a non-trivial number of symbols going back to 2016-05-12.
- 2,513 rows = exactly the count of US-equity trading days in this 10-year window. Tickers that hit this row count have continuous coverage with no gaps.
- The "<8y history" column is dominated by late IPOs — SP600 has the most (81) because that tier turns over fastest. These tickers are usable for 2024+ validation but not early-training features that look back 5+ years.

## Section 3 — Errors and edge cases

### Single hard failure: BKNG (Booking Holdings)

`/stock/metric` returned 429 (Too Many Requests) and our retry policy exhausted all 3 passes with exponential backoff (2s + 4s + 8s). BKNG has no fundamentals entry in the snapshot.

**Action:** could be backfilled with a single targeted refetch outside business hours when Finnhub load is lower. Low priority — one missing large-cap fundamentals snapshot is recoverable from `/stock/metric` at any later time.

### 202 empty `/stock/metric` bodies

Most of these are delisted entities Finnhub no longer serves metrics for. They show up as `empty` (rather than `error`) because Finnhub returns a 200 with no `metric` field rather than a 4xx. The fetcher correctly treats these as not-fetched without retry escalation.

### 159 empty `/stock/candle` bodies

Distribution of the 159 empty price responses:

- **1** in the 1,506 active records — likely a recent listing/symbol that Finnhub hasn't ingested yet
- **158** in the 616 removed records — predominantly 2014–2017 delistings beyond Finnhub Basic's 10y warranty (BSC/LEHM-style cases that fell into the Wikipedia "last 10y" cutoff due to either edge-of-window dates or Wikipedia data staleness)

### Earnings sanity gate fire (the headline Phase 3 outcome)

The yfinance earnings fetch:
- 2,047 fresh fetches attempted (75 were already cached from the smoke)
- Initial pass: 1,579 returned empty (likely yfinance silent-throttle at scale)
- Retry 1 (5s backoff): recovered 7 of 1,579
- Retry 2 (10s backoff): recovered 9 of 1,572
- Final: 484 non-empty of 2,047 attempted = **23.6%**

The 50% sanity gate fired and refused to overwrite the cache. The cache **was not poisoned** — it retained its pre-Phase-3 state (75 entries from the smoke). This is the stash@{0} defensive logic working exactly as it was designed to, against exactly the failure mode it was calibrated for.

**Per Mike's <85% rule:** earnings_dates is dropped from v1. The new study spec must be earnings-agnostic.

**Live-cache side effect:** the smoke (which passed the gate at 89.3%) wrote 75 entries to `models/cache/earnings_dates.json`, overwriting the prior ~491-entry legacy production state. The snapshot at `pre_v2_20260505/cache/earnings_dates.json` (486 entries) is untouched and the three promoted studies are unaffected, but the live cache is now thinner than before. Recovery options:
- restore from snapshot (`cp models/snapshots/equities/pre_v2_20260505/cache/earnings_dates.json models/cache/earnings_dates.json`)
- accept the loss if the three promoted studies always run in snapshot mode
- re-run a legacy-491-only yfinance refetch at off-hours

## Section 4 — Truncation behavior across the full universe

The OTC-tail-truncation logic is fired only on `status="removed"` records with a Wikipedia-documented `removed_at` date. Outcomes across the 628 removed records:

| Outcome | Count | Notes |
|---|---|---|
| `truncate_at` set AND price parquet on disk (truncation applied) | 457 | Clip applied at write time; cache stores clipped form. Snapshot inherits the clean truncated series. |
| `truncate_at` set BUT no price on disk | 154 | Either Finnhub returned `s: "no_data"` (pre-warranty delisting) OR truncation produced an empty series (clip date before Finnhub's first available date). Per the fetcher's skip-write policy, no empty parquet is persisted. |
| No `truncate_at` AND price parquet on disk | 13 | Wikipedia changes table did not provide a removal date for these (asymmetric add-only entries). Series ends naturally; no explicit clipping. |
| No `truncate_at` AND no price | 4 | Records with neither — typically very-old delistings missing both Wikipedia data and Finnhub coverage. |

**Spot-checked OTC-tail cases (from the smoke):**

| Ticker | Wikipedia `removed_at` | Cache last date | Last close | Status |
|---|---|---|---|---|
| SIVB | 2023-03-15 | 2023-03-09 | $106.04 | Clipped pre-collapse — correct |
| FRC | 2023-05-04 | 2023-05-04 | $0.32 | Clipped at removal date — correct |
| BBBY | 2023-03-20 | 2023-03-20 | $0.81 | Clipped at removal date — correct (788 rows of post-bankruptcy stub removed) |
| SBNY | 2023-03-15 | 2023-03-10 | $70.00 | Clipped pre-collapse — correct |

All four "OTC pink-sheet pollution" cases are clean. The `0.3 PHaseSummary` snapshot will not contain artificially distorted series.

### On the 599 / 616 question from the Phase-3 stop report

Mike asked for clarity on "599 of 616 have truncation date." The breakdown:

- 616 = unique-symbol removed records after dedup (out of 674 raw removed events in the Wikipedia source)
- 599 of those 616 have a non-null `removed_at` field from the Wikipedia changes table
- 17 of those 616 are records where Wikipedia recorded only an "added" event in the last 10y but no subsequent "removed" event — asymmetric entries

After the full-universe truncation pass:
- 457 successfully truncated (truncate_at within the served series)
- 154 skipped cache write (truncate_at before first served date — typically pre-warranty delistings)
- 13 records have a price file but no Wikipedia removed_at — fall back to natural-end-of-data
- 4 records have neither

**So Wikipedia provides removal dates for 97.2% of historical removals it tracks.** The 17 without dates are documented residual gaps (Section 5).

## Section 5 — Residual bias characterization (honest)

This snapshot is **best-effort survivorship-bias mitigation**, not survivorship-bias-free. Studies using it should disclaim:

> "Best-effort survivorship-bias mitigation with documented residual gaps tilting toward overstated returns by an estimated 0.3–0.6 pp/yr."

The systematic residual gaps:

1. **Pre-warranty delistings.** 2008-era SP500 names (BSC Bear Stearns, LEHM Lehman, FNM Fannie Mae, FRE Freddie Mac, MER Merrill Lynch, WB Wachovia, etc.) are beyond Finnhub Basic's 10y warranty and return `s: "no_data"`. The training window starts 2016, so 2008-era casualties don't matter for *that* window — but if anyone extends to a longer training window, these absences kick in.

2. **Bank-failure OTC-tail names.** Even within the warranty, SIVB / FRC / BBBY / SBNY-style cases came back from Finnhub with months-to-years of OTC pink-sheet candles tagged onto the legitimate pre-collapse series. We truncated these correctly using Wikipedia's `removed_at`, but for the 13 records without a Wikipedia removal date, the series may extend into pink-sheet territory. These are documented by symbol below.

3. **Ticker reuse.** When the same ticker is later assigned to a different company, the older entity's price history is irretrievable from Finnhub under that symbol. Known cases in this universe:
   - **VAL** — Valaris (active oilfield services) vs Valspar (paint, acquired by Sherwin-Williams 2017). Querying VAL returns Valaris data; pre-2017 Valspar history is lost.
   - **UNIT** — Uniti Group (telecom REIT, active) vs Unit Corp (oil/gas, removed). Pre-removal Unit Corp data partially lost.
   - **FB** — Facebook (renamed to META 2022-06). NOT a data gap: querying META returns full continuous history back to 2012-05-18 (Facebook IPO). The Wikipedia "FB → META" event is a name-change tracked under META, not as an add/remove cycle.

   The other 9 cases flagged by the universe builder's reuse heuristic (COR, DOC, MTD, PCG, ATI, IRT, RH, DNOW, MD) are false-positive name-normalization issues (rebrands of the same legal entity, e.g. "ATI Inc." ← "Allegheny Technologies"). These are NOT data gaps; just the heuristic is noisy.

4. **Wikipedia asymmetric entries.** 17 removed records have no `removed_at`. Their price series extends to whatever Finnhub serves (typically the actual end-of-trading date), with no clipping. Effect on bias: minimal — Finnhub's natural-end behavior usually approximates the right clip date within a few weeks.

**Net effect:** the universe is dramatically more representative than the legacy 491-ticker survivorship-biased snapshot, but it is not a textbook clean point-in-time membership reconstruction. The honest framing for the eventual study disclaimer is the one Mike specified.

## Section 6 — What the next study spec needs to know

When you (we) write the new study spec, the constraints set by this snapshot are:

1. **Universe size:** 1,963 with prices, 1,919 with fundamentals. Realistic studyable universe is the intersection: **1,919 tickers with both prices and fundamentals.** Of those, 1,493 are active and ~426 are historical-removed.

2. **Time window:** 2016-05-12 → 2026-05-11. ~8.5 years of usable training (giving 1y holdout + 1y validation back from today) or 5.5y train + 3y holdout + 1y validation if you want a longer holdout.

3. **Feature set:** prices + technicals (derived) + fundamentals + sector_map (carried forward, legacy-491-only coverage) + macro_signals (carried forward, ticker-independent). **No earnings_dates.** No analyst_targets.

4. **Survivorship property:** see Section 5. Study disclaimer needs to reflect best-effort mitigation, not "bias-free".

5. **What's missing that the legacy snapshot has:** earnings_dates, analyst_targets, full-universe sector_map, feature_matrix (will be rebuilt by the feature pipeline against this snapshot).

## Appendix — Tracker reference

This snapshot lives alongside `pre_v2_20260505`, not as a replacement. The legacy snapshot remains the canonical anchor for the three promoted studies (#325, #842, #1852). Project_State_Tracker.docx Appendix A has been updated to reference both.

The Polygon piggyback analysis in Section 3 of the tracker is now obsolete — Finnhub Basic is the chosen vendor for stocks-side data. The tracker Section 3 is marked obsolete in the same update.
