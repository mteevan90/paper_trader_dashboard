# Finnhub insider-transactions signal — known limitations

Date documented: 2026-05-04
Module: `src/finnhub_insider_signals.py`
Registered: yes — `("finnhub_insider_clusters", score_finnhub_insider_clusters)`
Cache: `models/cache/finnhub_insider/transactions.parquet`
       (101,347 rows, 4,632 tickers, 2018-01-31 → 2026-04-30)

## Limitation 1 — No insider role/title

Finnhub's free-tier `/stock/insider-transactions` endpoint returns:
`change, currency, filingDate, id, isDerivative, name, share, source,
symbol, transactionCode, transactionDate, transactionPrice`.

**No `position` / `title` / `role` field.** We cannot identify who is
a CEO, CFO, Director, or President — only the insider's name.

This means the signal is **all-insider clustering**, not the academically
stronger **senior-only clustering** (which segment 14's plan called for).

### Academic strength impact
- Senior-only cluster (CEO/CFO/Director/President buying together):
  ~6-12% annualized alpha over 6-month holds (Jeng/Metrick/Zeckhauser,
  Cohen/Malloy/Pomorski, Lakonishok/Lee, etc.)
- All-insider clustering proxy: ~3-5% annualized alpha (lower; includes
  middle managers and lower-conviction buys)

We expected ~50% signal-strength loss vs the original plan and accepted
that for v1 because (a) shipping a working signal beats blocking on
title enrichment, (b) the locked Phase 0 baseline gives us a clean
A/B comparison, (c) title enrichment via SEC EDGAR Form 4 parsing
would be a multi-segment effort.

## Limitation 2 — No 10% beneficial-owner flag

OpenInsider exposes `"10%"` as a title token, letting us exclude
institutional positions (Berkshire, Vanguard, large family offices).
Finnhub's insider-transactions endpoint exposes no such flag. 10%-owner
purchases enter the cluster-counting pool indistinguishably from
true insider activity.

Practical impact: probably small — institutional positions tend to
not show up as same-quarter "P" (open-market purchase) transactions
because most institutional accumulations are off-exchange or via
secondary offerings. But if a fund manager does open-market buy a
small-cap, the signal will treat it as cluster activity. Documented
limitation, not a blocker for v1.

## Limitation 3 — Universe is sparse for clustering

Empirical sanity check at 2025-12-01 against the full UNIVERSE_TICKERS
(NASDAQ-100 + S&P 500, 491 names): only **1 ticker** (EPAM, 7 unique
buyers, $52,798 combined) qualified for the ≥2-insider / ≥$10K filter
in the 90-day lookback window. **0.2% qualification rate.**

This isn't a limitation of the data — it's a structural property of
large-cap insider behavior. Cluster buying is rare on names with
broad institutional ownership and high stock prices (most large-caps
have insider activity dominated by RSU grants, option exercises,
tax-withholding sales — all coded `A`/`M`/`F`/`S`, not `P`).

Net effect on Phase 1 validation: **the signal contributed zero
practical drift** vs Phase 0 baseline. Phase 1 result was bit-
identical to Phase 0 (+39.42% / 1.45 / -12.39% / 583 trades / 46.5%)
because the rare qualifying tickers either weren't ranked near the
top-N cutoff or were already in the top-N regardless of the alt
boost. The 15% × 0.5 = 0.075-point composite boost is too small to
re-rank a ticker into the top-N from a low-rank position.

## Implications for "do alt signals help?"

Phase 1 result tells us this *one* alt signal at 15% bucket weight
on a large-cap universe doesn't measurably move alpha. It does NOT
tell us:

- Whether the senior-only version would help (we can't test it on
  free tier)
- Whether multiple alt signals stacked together would cross a
  threshold (15% / 1 signal is a small slice; 15% / 4 signals at
  3.75% each is even smaller per signal but the bucket aggregates)
- Whether the signal works on a wider universe with more small/mid
  caps where cluster buying is more common (would require expanding
  UNIVERSE_TICKERS)

These are open questions for follow-up segments.

## Future-work options

a) **SEC EDGAR title enrichment.** Parse Form 4 XML directly to get
   `<isDirector>`, `<isOfficer>`, and `<officerTitle>` fields per
   transaction. Would let us recover the senior-only signal. Cost:
   1-2 segments of EDGAR plumbing (daily index files, CIK-to-ticker
   mapping, XML parsing, dedup against Finnhub's data). Worth it
   only if Phase 1 results across multiple alt signals collectively
   suggest the bucket is a real alpha source.

b) **Universe expansion.** Add small/mid-cap tickers where cluster
   buying is more common and signal density rises. Would require
   re-running feature_cache, retraining the model on the wider set,
   etc. Significant scope outside this signal.

c) **Tighten/loosen the cluster filter.** Try ≥1 insider + $25K
   floor (more permissive — broader participation, more noise) or
   ≥3 insiders + $100K floor (more conservative — fewer qualifiers
   but stronger when they do). Might surface or kill the signal at
   different thresholds.

d) **Stack with other alt signals.** The signal mechanism works.
   Adding Quiver congressional + government contracts on top of
   this one means multiple sparse signals each contributing on
   different dates. Might collectively cross a noise floor that
   one signal alone doesn't.

## Status

Signal **registered and working** end-to-end (cache → score function
→ alt bucket aggregator → composite → backtest). Currently contributes
the structural neutral 0.5 baseline for ~99.8% of (ticker, date) pairs
on this universe. Provides occasional non-neutral lift but not enough
to change top-N selection on the locked validation window.

Phase 0 baseline preserved: +39.42% / 1.45 / -12.39% / 583 / 46.5%,
alpha vs SPY annualized -1.49pp, score -0.0346.

Phase 1 result: bit-identical (signal too sparse to drift).

Honest read per calibrated bands: **<0.3pp drift = below noise floor**.
This signal alone, at this weight, on this universe, does not move
alpha. Re-evaluate when (1) more alt signals are stacked (segment 15+),
(2) universe is widened, or (3) title enrichment unlocks the senior
filter.
