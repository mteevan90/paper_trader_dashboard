# Larger Universe v1 — Finnhub Basic capabilities probe

**Generated:** 2026-05-11 (probe at 20:27 UTC; revised after deeper /calendar/earnings + delisted probes)
**Probe scripts:** `scripts/research/finnhub_capability_probe.py`, `finnhub_verification_probe.py`, `finnhub_calendar_deeper.py`, `finnhub_delisted_recent.py`
**Raw JSON:** `docs/diagnostics/finnhub_basic_capabilities.json`, `finnhub_verification_probe.json`, `finnhub_calendar_deeper.json`, `finnhub_delisted_recent.json`
**Tier:** Finnhub Basic ($49.99/mo, personal use; key in `.env` as `FINNHUB_API_KEY`)
**Study this powers:** Larger Universe v1 (formerly "SP1500 v1" in earlier drafts)

## Summary

| Endpoint | Status | Rate-limit bucket | Useful for Larger Universe v1? |
|---|---|---|---|
| `/stock/symbol?exchange=US` | works | 150/min | yes — enumerate full US listing (30,384 rows) |
| `/stock/symbol?exchange=US&delisted=true` | **silently ignores filter** | 150/min | **no — returns the same 30,384 row list as without the param** |
| `/stock/candle` (10y daily) | works | 150/min | yes — core feed for prices |
| `/stock/candle` (12y daily) | works | 150/min | yes — returned 3,015 daily candles for AAPL (no 10y cap enforced for daily) |
| `/stock/candle` for delisted (TWTR) | works | 150/min | **yes — delisted ticker price history is accessible if you know the symbol** |
| `/stock/earnings` | works | 60/min | **partial — only 4 most recent quarters returned per ticker; no historical depth** |
| `/calendar/earnings` | works | 60/min | **partial — ~30-day historical reach only; forward-only otherwise; cannot be used for 10y backfill** |
| `/stock/recommendation` | works | 60/min | yes — analyst targets, 4 periods returned |
| `/stock/metric?metric=all` | works | 60/min | yes — ~130 fundamental metrics per ticker |
| `/stock/peers` | works | 60/min | low priority — peer-group lookup |

All probed endpoints returned HTTP 200 on this key. No endpoint was blocked at the auth layer. Two endpoints have notable functional limits — see details below.

## Rate-limit characterization

Two distinct buckets observed from `X-Ratelimit-Limit` headers:

- **150/min bucket** — `/stock/candle`, `/stock/symbol`, `/quote`. A 20-call burst over 4.58s (~4.4 req/s) on `/quote` ran clean with no 429s. Headers count down `X-Ratelimit-Remaining` from 150; reset is per-minute (`X-Ratelimit-Reset` epoch ~60s ahead).
- **60/min bucket** — `/stock/earnings`, `/calendar/earnings`, `/stock/recommendation`, `/stock/metric`, `/stock/peers`. These appear to be "Fundamentals" endpoints with a tighter cap.

Implication for the fetch plan: candle fetching is the bulk of the work and lives in the 150/min bucket. For 2,500 tickers at 150/min sustained, candle-only is ~17 min minimum (one call per ticker for full 10y). Even at 80 req/min throttled (50% safety margin) that's ~30 min. Earnings+metrics+recommendation per ticker, at 60/min, would be ~42 min per endpoint. **Total wall-clock for a comprehensive fetch with 50% safety margin: ~3–4 hours**, dominated by the 60/min buckets if all features are requested.

The spec's 4–8 hour budget is generous; the actual constraint is more likely Finnhub's hard daily quotas (not documented in headers) than the per-minute cap.

## Endpoint-by-endpoint detail

### `/stock/symbol` — full US listing

- **Works.** Returns 30,384 rows on `exchange=US`.
- **`delisted=true` param is silently ignored.** Probe sent both `?exchange=US` and `?exchange=US&delisted=true` — both returned 30,384 rows with the same first element (`KUASF` on OOTC).
- The returned list mixes Common Stock, ETF, ADR, REIT etc. and spans both active and historically-listed symbols.
- **Consequence:** Finnhub does not provide a clean "give me only delisted SP1500 historical members" filter on Basic. We need a separate source for SP1500 historical membership (Wikipedia revision history, S&P PR releases, or a paid index-membership dataset) and then we feed those symbols into `/stock/candle` one at a time.

### `/stock/candle` — daily OHLCV

- **Works for active, IPO'd-after-2018, and most last-decade-delisted tickers.** A follow-up probe (`scripts/research/finnhub_delisted_recent.py`) tested 15 known SP500/SP1500 delistings from the last ~10 years; results documented in detail under "Delisted coverage" below. Headline numbers:

  | Symbol | Tier | 10y window | Returned candles |
  |---|---|---|---|
  | AAPL | SP500 | 10y | 2,516 |
  | AAPL | SP500 | 12y | 3,015 (no truncation at 10y) |
  | AAON | SP400 | 10y | 2,516 |
  | AAMI | SP600 | 10y | 2,516 |
  | TWTR | delisted (private 2022) | 10y window requested | 1,631 (ends at delisting) |

- **10y daily cap is not enforced at the API level for active tickers** — AAPL returned 12y when 12y was requested. The Basic tier marketing copy says "10y daily" but the API will give more if requested. (Don't rely on this for the study spec; treat 10y as the official guarantee.)
- Returns `s: "ok"` on success, `s: "no_data"` on empty range. Fields `c/o/h/l` are floats, `t` is unix epoch seconds, `v` is volume.
- **No split/dividend adjustment indication in the response schema.** Need to verify whether OHLC is adjusted or raw. (Spot-checking AAPL closes against known split events is a Phase 2 task before promoting to snapshot.)

### Delisted coverage — detailed

15 last-decade delistings probed; raw data in `docs/diagnostics/finnhub_delisted_recent.json`:

| Symbol | Event | Last candle | Candles | Notes |
|---|---|---|---|---|
| TIF | LVMH buyout 2021-01 | 2021-01-06 | 1,674 | clean |
| KSU | CP Rail buyout 2021-12 | 2021-12-13 | 1,910 | clean |
| XLNX | AMD buyout 2022-02 | 2022-02-11 | 1,952 | clean |
| CERN | Oracle buyout 2022-06 | 2022-06-07 | 2,031 | clean |
| CTXS | TIBCO buyout 2022-09 | 2022-09-29 | 2,110 | clean |
| MNDT | Google buyout 2022-09 | 2022-09-09 | 2,096 | clean |
| TWTR | Musk buyout 2022-10 | 2022-10-27 | 2,130 | clean |
| ATVI | MSFT buyout 2023-10 | 2023-10-12 | 2,370 | clean |
| DISCA | rename to WBD 2022-04 | 2022-05-27 | 2,023 | clean |
| SIVB | SVB collapse 2023-03 | 2023-08-24 | 2,324 | OTC tail past collapse; truncate at delisting |
| FRC | First Republic 2023-05 | 2023-10-09 | 2,365 | OTC tail past collapse; truncate at delisting |
| BBBY | bankruptcy 2023-04 | 2026-05-11 (today) | 3,015 | **ticker recycled / OTC stub continues — needs explicit truncation by event date** |
| SBNY | Signature Bank 2023-03 | 2026-05-08 | 2,997 | same pattern as BBBY |
| VIAC | ViacomCBS → PARA 2022-02 | 2022-05-27 | **only 69** | partial history (last 3 months only) |
| FB | Facebook → META 2022-06 | 2026-05-11 | **220** | **wrong entity** — returns a recycled FB ticker, NOT old Facebook history |

Also tested but outside Basic's 10y window:
- BSC (Bear Stearns 2008): `s: no_data`
- LEHM (Lehman 2008): `s: no_data`

**Categorization:**

- **Clean delistings (9/15 = 60%):** Full pre-event history; series ends near actual delisting date. Drop-in usable.
- **OTC-stub-tail cases (4/15 = 27%):** SIVB, FRC, BBBY, SBNY — Finnhub returns price candles past the corporate event, often into the OTC pink-sheet phase. **Fixable** by truncating at the actual delisting date from the Wikipedia component-change table.
- **Ticker-reuse cases (1/15 = ~7%):** FB — pre-2022 Facebook history is NOT accessible under "FB"; need to query "META" for the full history. **Fixable** by maintaining a symbol-remapping table during the fetch.
- **Partial-only cases (1/15 = ~7%):** VIAC — only late history returned. Could be a Finnhub data gap or an exchange-code issue. **Not easily fixable**; this kind of ticker may be silently incomplete in v1.
- **Beyond-warranty cases:** 2008 names (BSC, LEHM) — `no_data`. **Not fixable.** Out of scope for a 10y window anyway.

**Implication for "best-effort survivorship-bias mitigation":** for a 10y SP1500 universe, expect ~85–90% of delisted-historical-member symbols to have clean or fixable price history. The 10–15% that are partial / reused / wrong-entity become residual bias the snapshot summary needs to document by name, not aggregate-only.

### `/stock/earnings` — per-symbol earnings history

- **Works**, but returns only **the 4 most recent quarters per ticker**. The probe for AAPL returned 4 rows, latest period 2026-03-31.
- This is a key gap. The promoted-studies feature `earnings_dates` needs longer history. Options:
  - Use `/calendar/earnings` over a range of historical date windows (requires many calls — 60/min bucket — but covers any horizon).
  - Treat earnings_dates as a feature with limited historical coverage in v1 and document the gap.
- Body shape: `{actual, estimate, period, quarter, surprise, surprisePercent, symbol, year}` per row.

### `/calendar/earnings` — date-range earnings calendar

- **Works for current ~30-day window; does NOT serve historical windows on Basic.** Deeper probe (`scripts/research/finnhub_calendar_deeper.py`, `docs/diagnostics/finnhub_calendar_deeper.json`):

  | Window | Symbol filter | Events returned |
  |---|---|---|
  | next 7 days (2026-05-11 → 2026-05-18) | none | 1,385 |
  | next 3 months (2026-05-11 → 2026-08-11) | none | 1,500 (response cap) |
  | last 2 weeks (2026-04-25 → 2026-05-10) | none | 1,500 (response cap) |
  | last 2 weeks | AAPL | 1 (matches AAPL's 2026-04-30 print) |
  | Feb 2026 (~3 months back) | none | **0** |
  | Q3 2024 (~21 months back) | none | **0** |
  | Q3 2024 window | AAPL | **0** |
  | Jan 2024 | none | **0** |
  | Q3 2020 | none | **0** |

- **Conclusion:** Basic tier serves only the trailing ~1 month plus forward dates. Calendar/earnings is **not viable for 10y historical backfill**. This was the assumption I made in the initial Phase 0 writeup (based on a single non-empty response from a recent-week probe) and it turns out to be wrong. The endpoint is useful for "what's the next earnings date for these tickers" but not for reconstructing the historical earnings sequence.
- Body wrapped in `{"earningsCalendar": [...]}`. Each event: `{date, epsActual, epsEstimate, hour, quarter, revenueActual, revenueEstimate, symbol, year}`.

### `/stock/recommendation` — analyst ratings counts

- **Works.** Returns 4 rows per ticker — most-recent month + 3 prior months.
- Body: `{buy, hold, period, sell, strongBuy, strongSell, symbol}` per row.
- **No historical depth beyond 4 months on Basic.** The legacy `analyst_targets.json` in `models/cache/` is therefore not directly replaceable by this endpoint for back-history; we'd only have a forward-going build.

### `/stock/metric?metric=all` — fundamentals snapshot

- **Works.** Returns ~130 metrics per ticker for AAPL and ~95 for AAMI (smaller-cap, fewer derived metrics computed).
- Schema: `{metric: {...}, metricType, series, symbol}`.
- These are point-in-time / TTM / annualized snapshots, not time series — the response is a single dict per ticker. The `series` field may carry quarterly history for some metrics but wasn't fully probed; worth a follow-up sample.
- Coverage is broad and consistent across legacy SP500 and SP600 samples. **Strong candidate to replace the legacy `fundamentals.json` for the full SP1500 universe.**

### `/stock/peers`

- **Works.** Returns a list of ~10 related tickers per symbol.
- Low priority for this study; useful for future work (sector-relative features, peer-group baselines).

## What this means for Phase 2+ design

1. **Price fetching is the easy part.** `/stock/candle` at 150/min handles 2,500 tickers in ~30 min including throttle headroom, regardless of active/delisted status. The retry logic in `stash@{0}` should be straightforward to adapt — fewer edge cases than yfinance because the API returns proper status codes and `Retry-After` headers (when present).
2. **Delisted-symbol enumeration is the hard part.** Finnhub Basic won't tell us which symbols are delisted SP1500 historical members. The realistic options:
   - **Option A (cheapest):** scrape Wikipedia's S&P 500 / 400 / 600 component-change tables (these list all additions/removals with dates over the last ~25 years). Combine with the current constituents list to produce a "ever-in-SP1500 in the last 10 years" universe. Expected size ~2,000–2,500 distinct symbols.
   - **Option B:** use a third-party index-membership dataset (Refinitiv, etc.). Outside this project's budget.
   - Option A is the path of least resistance and is what I'd recommend for Phase 2.
3. **Earnings has no historical-backfill path on Basic.** Both `/stock/earnings` (per-symbol, 4-quarter cap) and `/calendar/earnings` (~30-day historical reach) are effectively forward-only. The originally-proposed Option-2 plan to chunk `/calendar/earnings` over 10y is mechanically impossible at Basic tier — the data the endpoint serves on Basic does not extend that far back. This is a Phase-2 design-revision trigger; see "Earnings revision" section below.
4. **Analyst targets historical depth is permanently limited on Basic.** Only the trailing 4 months are returned. This effectively removes `analyst_targets.json` as a viable historical feature for any new SP1500 study. **Recommendation:** drop analyst_targets from the new study's feature set, or document it as a forward-going-only feature.
5. **Fundamentals via `/stock/metric` is the win.** Replaces the legacy `fundamentals.json` with consistent SP1500-wide coverage.

## Earnings strategy — yfinance fallback with explicit health audit

**Honest note up front:** Finnhub Basic is forward-only for earnings across all four tested endpoints (`/stock/earnings` 4-quarter cap; `/calendar/earnings` ~30-day historical reach; `/calendar/earnings?symbol=X` same; `/stock/metric.series` not validated for historical earnings dates specifically). **Historical earnings-date backfill via Finnhub at this tier is not possible.** The earlier Phase 0 finding that /calendar/earnings could be chunked over 10y was wrong — it extrapolated from a single recent-window probe.

**Decision (Mike, post-deeper-probe):** keep yfinance as a tier-mismatched supplement for `earnings_dates` only. Everything else (prices, fundamentals, sector, market cap) comes from Finnhub. The stash@{0} retry + sanity-gate logic was designed for exactly this failure mode (yfinance silent-empty under throttling at scale) and is preserved verbatim on the yfinance earnings path.

Architecture:

1. **Finnhub** handles universe-wide data: `/stock/candle` for prices (active + delisted), `/stock/metric?metric=all` for fundamentals, `/stock/symbol?exchange=US` for the full US-listed universe.
2. **yfinance** handles `earnings_dates` only, with the stash's retry policy `(5s, 10s)` backoff for empties and a 50% non-empty sanity gate that refuses to overwrite the cache on a bad batch. These constants were calibrated for yfinance throttling and stay as-is.
3. **Phase 3 coverage audit (gating logic for v1):**
   - Run the earnings fetch with full retry exhaustion against the SP1500 current members + delisted universe.
   - Measure: `n_nonempty_new982 / n_attempted_new982`. (Legacy 491 is a control — should be ~≥95% per prior production runs.)
   - If non-empty fraction on the new 982 is **≥85%** → earnings_dates is a candidate feature for the new study spec.
   - If **<85%** → drop earnings_dates from v1 entirely. The new study spec must be earnings-agnostic. **Do not ship asymmetric coverage** (where the model learns "earnings populated vs. null" as a feature itself — that is data leakage masquerading as signal).
4. **Phase 5 snapshot summary** documents the measured coverage rate explicitly so the next study spec is informed.

This architecture preserves the stash code rather than discarding it; the silent-empty defense is load-bearing on the yfinance path and stays in place. Finnhub-path code (prices, fundamentals) does not need that defense because Finnhub returns proper HTTP status codes and `Retry-After` headers rather than silent empties.

## Open questions / Phase-2 follow-ups

- Are OHLC values split-and-dividend adjusted? Spot-check AAPL against known 4-for-1 split (2020-08-31) before promoting to snapshot.
- `/stock/metric` has a `series` field that wasn't fully probed — does it carry quarterly time-series for back-history?
- Does `/calendar/earnings` accept arbitrary historical date ranges on Basic, or is there a lookback cap?
- Daily-quota behavior: per-minute caps are well-characterized, but Finnhub Basic likely also has an overall daily cap. The 20-call burst probe didn't exercise enough volume to hit it. Plan the fetch to log cumulative-call counts so we can characterize this on the first real run.

## Probe methodology notes

- Calls sent serially with a 0.5s sleep between, well under both rate-limit buckets.
- Single-key probe — no concurrency yet. Concurrency for Phase 3 should use a per-bucket RateLimiter (mirror `src/options/polygon.py` pattern, but two buckets instead of one).
- Sample tickers: AAPL (SP500), AAON (SP400), AAMI (SP600), TWTR (delisted). Limited sample; broader sweep happens in Phase 3 when the real fetch runs.
