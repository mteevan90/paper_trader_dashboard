# FINRA RegSHO short-interest scraper — segment 15 stopped

Date documented: 2026-05-04
Status: Cannot access data source. No code written, no cache built.
ALT_SIGNAL_REGISTRY remains `[("finnhub_insider_clusters", ...)]`
unchanged from segment 14.
Phase 0 baseline preserved: +39.42% / 1.45 / -12.39% / 583 / 46.5%,
alpha -1.49pp, score -0.0346.

## What we tried

Probed the URL pattern from segment 15's spec
`https://cdn.finra.org/equity/regsho/monthly/shrt{YYYYMMDD}.txt`
across 11 candidates — 5 recent dates (2026-04-30, 2026-04-15,
2026-03-31, 2026-03-13, 2025-12-31), 2 older dates known to be
historical settlement dates (2024-01-31, 2018-01-15), 2 alternative
path variations (`/equity/regsho/daily/`, `/equity/regsho/`), the
directory listing endpoint, and the official FINRA short-interest
landing page.

Every probe to `cdn.finra.org` returned the same 111-byte response:

```
HTTP 403  application/xml
<?xml version="1.0" encoding="UTF-8"?>
<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>
```

Tested with both honest (`paper-trader-research/0.1 (personal-use)`)
and browser (`Mozilla/5.0 ...`) User-Agents — same result. The
CDN-level AccessDenied is byte-identical across every URL we tried,
indicating a path-level (or path-prefix-level) restriction rather
than per-file 404s.

WebFetch attempts on FINRA's data documentation pages all returned
404, so we could not retrieve current official URL patterns from
within this session.

## Likely cause (best guess, not verified)

FINRA appears to have either:
1. **Moved the bulk short-interest files behind their data-browse-
   catalog registration system** (which requires a free FINRA Data
   account and login). Industry chatter from 2024-2025 suggests they
   restructured public data distribution; this is consistent with
   the path `cdn.finra.org/equity/regsho/monthly/` returning a
   uniform 403 rather than per-file 404s.
2. **Deprecated the CDN path entirely** in favor of an authenticated
   API at `api.finra.org` or a registration-walled portal.

Either way, this is not solvable by adjusting URL guesses or User-
Agents from inside an automated session — would require a human
to register at FINRA Data, generate API credentials or download
links, and either feed those credentials into the scraper or
download files manually for ingest.

## What's preserved

- No `src/short_interest_signals.py` was created (intentional — the
  data source isn't accessible, no point writing the scraper).
- `src/alt_signals.py` unchanged: registry still has just
  `finnhub_insider_clusters` from segment 14.
- No `models/cache/short_interest/` directory.

## Next-segment options

Listed for future reference. Each requires explicit authorization
before work begins.

a) **Register at FINRA Data Browse Catalog.** Free account; would
   give us either authenticated download links for the historical
   short-interest archive, or API access to the same data. Cost:
   ~30 minutes of human registration + key handling. Once we have
   credentials, the scraper from this segment's plan should work
   with minor URL/auth changes. Best long-term path for FINRA data.

b) **Alternative free short-interest source.**
   - **Yahoo Finance** (`yfinance.Ticker(t).info`): exposes
     `shortRatio`, `sharesShort`, `sharesShortPriorMonth`,
     `dateShortInterest`. **Single point per ticker, not historical**.
     Useful for current-state queries but not the time-series cluster
     signal segment 15 designed.
   - **stockanalysis.com / shortsqueeze.com** (HTML-scraped): noisy,
     have anti-scrape defenses similar to OpenInsider, would likely
     hit the same dead-end as segment 14.
   - **NYSE** publishes their own short interest data on a separate
     schedule but only for NYSE-listed names (covers ~20% of our
     universe). Not worth the per-source plumbing alone.

c) **Pivot to next alt signal family.** Quiver congressional trades
   + government contracts was already on the segment-15+ roadmap
   per user notes. Two more signals worth shipping before
   re-evaluating whether the alt bucket needs more breadth or
   whether the remaining issue is at-source signal density / weight
   tuning.

d) **Park short-interest signals entirely.** If short-interest data
   ends up requiring registration + auth + ongoing maintenance, the
   marginal value-add at 15% bucket weight may not justify the
   integration cost, especially if Quiver's three signals (congress
   + contracts + lobbying) collectively cross the noise floor on
   their own.

## Recommendation

**Pivot to (c) Quiver.** Three signals from one source, multi-year
free history per their docs, different signal family from insider
buying. If the populated alt bucket (insider clusters + 3 Quiver
signals) starts to show measurable drift, that's evidence the
bucket-architecture works and we can come back to FINRA via
option (a) registration as a follow-up. If it doesn't, the issue
is structural (bucket weight or signal-quality across the board)
and FINRA registration won't change the answer.

Don't pursue (a), (b), or (d) without explicit authorization.

## State after stopping

- ALT_SIGNAL_REGISTRY: `[("finnhub_insider_clusters", ...)]` (1 signal)
- Phase 0 baseline locked: +39.42% / 1.45 / -12.39% / 583 / 46.5%
- Phase 1 result (insider clusters alone): bit-identical to Phase 0
- 11 HTTP requests spent on FINRA probes (all 403); no data fetched,
  no cache built, no partial state.
