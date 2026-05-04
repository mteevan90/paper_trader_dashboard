# OpenInsider scraper — segment 14 stopped

Date documented: 2026-05-04
Status: Cache build aborted. ALT_SIGNAL_REGISTRY remains `[]`.
Phase 0 baseline preserved: +39.42% / 1.45 / -12.39% / 583 / 46.5%,
alpha -1.49pp, score -0.0346.

## What happened

Segment 14's plan was a 90-day-chunk historical scrape of
`openinsider.com/screener` from 2018-01-01 to 2026-05-04. Two
verification probes were run; the first revealed the user-supplied
reference URL's `fd=0&fdr=...` does not filter by date, the second
appeared to confirm that `td=13&tdr=...` (custom trade-date range)
worked when tested on a recent 14-day window — 180 rows all within
2026-04-22 → 2026-05-01.

The cache build then issued 34 chunked GETs for ranges
2018-01-01 → 2026-05-04. **Every chunk returned the same 180 rows
covering 2026-04-22 → 2026-05-01.** The dedup pass collapsed the 34
chunks to 180 unique rows. None of the historical data was actually
fetched.

Direct verification with six probe queries (Q1 2018, COVID 2020,
Q1 2023, Mid-2024, Q1 2025, recent control) produced byte-identical
responses (205,720 bytes, 180 rows, 2026-04-22 → 2026-05-01 every
time). The server is silently returning a default "recent activity"
view regardless of the `tdr` parameter.

## Root cause hypotheses (not investigated further per scope-protection)

1. **CDN / server-side caching keyed on path, not query string.**
   The screener page may be aggressively cached by a CDN (Cloudflare,
   etc.) at the path level — varying `tdr` query params doesn't bust
   the cache. Identical byte counts across totally different ranges
   strongly suggest this.

2. **Form-submission-only date filter.** The screener page is
   primarily a form. The `td=13` "custom" mode may need additional
   hidden form fields (separate `td_start` / `td_end` pairs?) that
   are submitted via POST or a different URL path, not via the
   simple `tdr=...` GET param.

3. **Anti-scrape default.** OpenInsider may detect non-browser User-
   Agents and silently substitute a generic recent-activity view.
   Live probes used `paper-trader-research/0.1 (personal-use)` —
   honest but distinctively non-browser. A spoofed browser UA might
   work but starts down the path of evading their anti-scrape logic,
   which I won't do without explicit user authorization.

## What's preserved in the codebase

- `src/openinsider_signals.py` — full module remains. The score
  function, append-only cache logic, senior-insider regex (passes
  23/23 unit tests), HTTP throttling/backoff, and HTML parser are
  all intact. Just the URL pattern doesn't fetch what we want.
- `src/alt_signals.py` — `ALT_SIGNAL_REGISTRY` reverted to `[]` so
  the system runs as clean Phase 0. A comment block at the top
  references this file for future-me.
- `models/cache/openinsider/` — directory deleted (the partial
  build was just 180 recent rows, not historical).

## What's NOT preserved

No partial cache, no half-saved data, no pinned versions. Clean state.

## Next-segment options (not actioned tonight)

Listed for future reference. Each requires explicit authorization
before work begins.

a) **Investigate OpenInsider's form mechanics directly.** Visit
   the screener page in a browser, inspect the form, see what
   parameters/method get submitted on a custom-date submission.
   The fix may be a few lines (POST instead of GET; rename `tdr`
   to whatever the form actually uses; submit hidden fields).

b) **Browser-style User-Agent.** Try a Chrome UA. If that flips
   the behavior, the constraint becomes "OpenInsider sees us as
   a scraper and degrades us to a default view." That's evading
   their scraper detection — would not proceed without authorization.

c) **Switch data source.** Insider transaction data is also
   available from:
     - SEC EDGAR Form 4 filings (free, official, but parsing the
       filings requires nontrivial XBRL/XML work)
     - finnhub.io `/stock/insider-transactions` endpoint (free
       tier, history depth unverified — would need a probe similar
       to segment 13's recommendation_trends check)
     - quiverquant.com insider trades API (requires a key, has a
       free tier with some history)
   Each comes with its own integration cost. EDGAR is canonical
   but heaviest. Finnhub is consistent with what we already use
   (FRED + macro signals). Quiver is on the segment 14+ roadmap
   for congressional + contracts data anyway, so adding their
   insider feed is incremental.

d) **Park insider signals entirely; move to a different alt
   signal family.** OpenInsider was the cheapest path to a free,
   long-history insider signal. If the scrape can't be made to
   work, the marginal value of insider data may not justify the
   integration cost vs. testing other signals first.

## Recommendation

Pivot. Either (c) Finnhub insider-transactions probe (~5 min, free,
matches existing pattern) or (d) park insiders and move to the next
alt-signal family. Don't pursue (a) or (b) without authorization.

Phase 0 baseline +39.42% / 1.45 / -12.39% remains the locked answer.
ALT_SIGNAL_REGISTRY remains empty. The bucket scaffolding from
segment 12 still works (current behavior: 0.5 neutral for all
tickers, 15% × 0.5 = 0.075 constant offset on every composite, no
ranking effect).
