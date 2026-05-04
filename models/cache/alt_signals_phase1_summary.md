# Alt-bucket Phase 1 — architecture answer locked

Date: 2026-05-04
Status: Phase 1 = Phase 0 bit-identical. Architecture answer reached.
Segment 15 closed without shipping a second alt signal; that's the
honest finding, not a failure.

## Locked Phase 0 baseline (default config, validation window)

```
Window:  2024-01-01 → 2026-04-30
Return:        +39.42%
Sharpe:          1.45
Max drawdown: -12.39%
Trades:           583
Win rate:        46.5%
Alpha vs SPY:   -1.49pp
Score:        -0.0346
```

## Phase 1 result — bit-identical

ALT_SIGNAL_REGISTRY populated with `finnhub_insider_clusters`
(segment 14): exactly the same numbers as Phase 0. The signal works
mechanically end-to-end (cache → score → bucket → composite →
backtest) but qualifies for only ~0.2% of (ticker, date) pairs on
the large-cap universe (1 of 491 tickers at 2025-12-01 sample).
At 15% bucket weight × 1 sparse signal, the maximum composite
differential is 0.075 — not enough to re-rank a ticker into
the top-N from a low-rank position.

## Four data sources investigated for a second alt signal

Each was probed responsibly, hit a hard limitation, and stopped before
scraping at scale. The investigation pattern itself is the artifact —
each dead-end documented separately so future-me has the diagnoses.

### 1. Finnhub `recommendation_trends` (segment 13 first attempt)
**Stop reason:** free-tier history is ~4 months, not the ~12 months
I'd estimated when planning. With a 28-month validation window,
~14% of dates would have signal; the rest fall back to neutral 0.5
defaults. Below the 0.3pp drift noise floor on overall validation.
**Documented in:** `models/cache/finnhub_free_tier_limits.md`

### 2. OpenInsider screener HTML scrape (segment 14 first attempt)
**Stop reason:** openinsider.com's screener silently ignores the
date-range parameter for historical queries. Every request returns
the same ~180-row "recent activity" default view (CDN-cached at the
path level, query string doesn't bust the cache). Two probes appeared
to work; the cache build revealed the real behavior — all 34 chunks
returned identical 180 rows. Stopped, deleted partial cache.
**Documented in:** `models/cache/openinsider_scraper_dead_end.md`

### 3. FINRA RegSHO short-interest CDN (segment 15 first attempt)
**Stop reason:** all 11 URL probes returned identical 111-byte 403
AccessDenied responses (recent dates, older dates, alternate paths,
plain UA, browser UA — all uniform). The path-level uniform 403
indicates FINRA either restructured public data behind the FINRA
Data Browse Catalog registration system or deprecated the CDN path
entirely. Not solvable from inside the session.
**Documented in:** `models/cache/finra_regsho_dead_end.md`

### 4. yfinance institutional_holders (segment 15 second attempt)
**Stop reason:** known limitation — yfinance's institutional
endpoints (`institutional_holders`, `major_holders`,
`mutualfund_holders`) all return current-snapshot-only data, no
historical time series. Cannot backfill quarter-over-quarter
changes for the 2024-2026 validation window from yfinance alone.
Skipped the live probe (95% confidence) and stopped here per
explicit user instruction.
**Documented inline in this file** (no separate dead-end doc since
no probes were spent).

## What we learned across the four attempts

Free historical alt-signal data for this specific use case (28-month
validation window starting 2024-01, large-cap universe of 491 names)
is scarce. The signal sources that exist are either:
- Paid tiers (Finnhub /stock/revisions ~$100/mo, Quiver ~$30/mo)
- Registration-walled (FINRA Data, requires account + login + key
  handling)
- Substantial parsing infrastructure (SEC EDGAR daily index → 13F-HR
  filings → XML positions tables → CIK-to-ticker mapping → quarterly
  per-ticker time-series construction)

Free time-series sources tested:
- yfinance: only current snapshots for ownership data
- Finnhub `/stock/insider-transactions`: deep history but no titles
- OpenInsider: deep history but date filter doesn't work
- FINRA RegSHO: depth unknown, public access blocked

## The architecture answer

**Locked finding:** 15% bucket weight × 1 sparse signal (insider
clusters at 0.2% qualification rate on a 491-ticker large-cap
universe) does not move composite alpha meaningfully on this
validation window. Phase 1 = Phase 0 bit-identical confirms this.

**Open question:** would 15% × dense multi-signal coverage (3-4
populated alt signals each contributing on different rebalance dates,
collectively pushing the bucket toward the ranges Optuna would tune
into) cross the noise floor where one signal alone doesn't?

That question is **deferred** until either:
- (a) Quiver paid tier ($30/month) becomes a clearer investment with
  concrete cancel criteria
- (b) SEC EDGAR 13F parsing gets dedicated segment time (~6-10 hours
  of plumbing work before any signal scoring)

Forward-looking diagnostic from segment 14: once the bucket has 3-4
populated signals, run a fresh Optuna study against the populated
composite. If TPE picks `weight_alt > 0.15` consistently, the bucket
weight is binding — meaning more weight to alt would help. If TPE
picks `weight_alt ~ 0.10-0.15` even with multiple signals, the
architecture is fine and the issue is signal sources. Lets the data
answer the architecture question instead of us guessing.

## Recommended future work — not actioned

a) **Trial Quiver paid tier for one month** with concrete cancel
   criteria. Three signals from one source (congressional trades,
   government contracts, lobbying), multi-year free history per
   their docs (paid tier may also unlock additional history /
   real-time updates). Cancel if Phase 2/3 with three Quiver signals
   stacked still produces <0.3pp drift; otherwise continue.

b) **Invest 6-10 hrs in SEC EDGAR 13F parsing** for institutional
   accumulation signal. The same infrastructure plumbing also
   unlocks insider-title enrichment for the segment-14 finnhub
   data (combining EDGAR Form 4 positions with Finnhub
   transactions resurrects the senior-cluster academic signal).
   Single 6-10 hr investment unlocks two real signals.

The two paths are roughly parallel value (both could produce
2-3 working multi-year alt signals on free or near-free terms);
choice between them is a matter of which infrastructure cost the
user prefers (recurring small subscription vs one-time engineering).

## State after stopping

- ALT_SIGNAL_REGISTRY: `[("finnhub_insider_clusters", ...)]` (1 signal)
- `src/finnhub_insider_signals.py` retained, header comment updated
  to flag it as the sole alt signal pending future bucket population
- `src/openinsider_signals.py` retained as documentation of the
  HTML-scrape dead-end approach
- No new files created this segment
- Phase 0 baseline preserved
- Workspace clean

Segment 15 marked complete in spirit: the architecture answer was
reached even though no second signal shipped. Honest stop > forced
ship.
