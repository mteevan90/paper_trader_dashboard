# SP1500 Data Inventory — pre-study go/no-go audit

**Date:** 2026-05-11
**Branch:** `chore/sp1500-data-inventory`
**Scope:** read-only inspection of `models/snapshots/equities/` and live `models/cache/` to determine whether a fresh independent equity study can be trained on SP1500 with a meaningful train/test split.

## Headline

**Stop-and-report finding (per spec instructions): SP1500 is not snapshot-integrated.** Mike's mental model that "SP1500 is in the snapshot" is incorrect. The data partially exists on disk in the uncommitted top-level `models/price_cache/` and `models/cache/`, but neither of the two equity snapshots (`pre_v2_20260505`, `post_macro_fix_20260506`) contains SP400 or SP600 names. The four-section coverage / survivorship / parity / split analysis the spec asked for can't be answered against the snapshot because there is no SP1500 universe in the snapshot to audit.

A secondary blocker exists independent of the SP1500 question: **all price history in the cache starts 2018-01-02**, which is ~8 years of data — insufficient for a ≥10-year training window even before universe choice.

## What is actually on disk

### Snapshots — `models/snapshots/equities/`

| Snapshot | price_cache files | earnings_dates entries | fundamentals entries |
|---|---|---|---|
| `pre_v2_20260505` (canonical) | 493 | 486 | 491 |
| `post_macro_fix_20260506` | 493 | 486 | 491 |
| `macro_signal_investigation_20260506` | (no price_cache dir — diagnostic artifacts only) | — | — |
| `parallelism_diagnostic_20260506` | (no price_cache dir — diagnostic artifacts only) | — | — |

Both real snapshots are the **legacy ~490-ticker universe** (SP500 + NDX overlap). No SP400 or SP600 names are present in any snapshot.

### Live (uncommitted) caches at repo root

These are working caches outside the snapshot system; the SP1500 fetch landed here but was never promoted into a snapshot.

| File / dir | Entries | Covers SP1500? |
|---|---|---|
| `models/price_cache/*.parquet` | 1,475 (incl. `^VIX`) | yes — full SP1500 + VIX |
| `models/cache/fundamentals.json` | 1,473 | yes |
| `models/cache/sector_map.json` | 1,473 | yes |
| `models/cache/earnings_dates.json` | 491 | **no — legacy only** |
| `models/cache/analyst_targets.json` | 484 | **no — legacy only** |
| `models/cache/macro_signals.parquet` | ticker-independent (FRED-sourced) | n/a — applies universally |
| `models/cache/feature_matrix.parquet` | 184 MB blob built from the above | inherits the legacy/SP1500 split from its inputs |

### Git state confirming "not integrated"

- `git stash list` →
  `stash@{0}: On main: deferred from sp1500/finnhub session: retry+sanity-gate+force-refresh in backtest.py + fetch_sp1500.py`
- Project State Tracker (2026-05-11): *"SP1500 universe expansion remains stashed, awaiting Finnhub TOS clarity."*
- `docs/sp1500_coverage_report.txt` (generated 2026-05-08) records the fetch outcome: 34/1473 fully complete, 1436/1473 partial (price OK, earnings missing).
- `docs/diagnostics/sp1500_fetch_failures.txt` enumerates 1,436 "missing: earnings" entries and 3 limited-history IPOs.

## Spot-checked SP1500 price coverage (live cache only)

Sample tickers across tiers (all from `models/price_cache/`, none in any snapshot):

| Ticker | Tier | First date | Last date | Rows |
|---|---|---|---|---|
| AAPL | SP500 legacy | 2018-01-02 | 2026-05-08 | 2,099 |
| MSFT | SP500 legacy | 2018-01-02 | 2026-05-08 | 2,099 |
| AAON | SP400 | 2018-01-02 | 2026-05-07 | 2,098 |
| AGCO | SP400 | 2018-01-02 | 2026-05-07 | 2,098 |
| ALGM | SP400 (post-IPO) | 2020-10-29 | 2026-05-07 | 1,386 |
| BJ | SP400 (post-IPO) | 2018-06-28 | 2026-05-07 | 1,975 |
| CART | SP400 (post-IPO) | 2023-09-19 | 2026-05-07 | 661 |
| AAMI | SP600 | 2018-01-02 | 2026-05-07 | 2,098 |
| AAT | SP600 | 2018-01-02 | 2026-05-07 | 2,098 |
| ABG | SP600 | 2018-01-02 | 2026-05-07 | 2,098 |
| ACAD | SP600 | 2018-01-02 | 2026-05-07 | 2,098 |

Observations:
- The fetch policy clearly capped earliest history at **2018-01-02** for all tickers, regardless of tier. Tickers IPO'd later (ALGM, BJ, CART, ABNB, ACI, SDGR…) start naturally on their first trading day.
- No tier-specific date-range gap. SP400 and SP600 names that were public before 2018 have the same depth as SP500 names. **The shallow history is a global cap, not an SP1500 problem.**

## Survivorship — sampled

Spot-checked 5 well-known delisted names in `models/price_cache/`:

| Ticker | Event | Present? |
|---|---|---|
| LEHM (Lehman) | Bankruptcy 2008 | no |
| BSC (Bear Stearns) | Acquired by JPM 2008 | no |
| SHLD (Sears) | Bankruptcy 2018 | no |
| TWTR (Twitter) | Taken private 2022 | no |
| ATVI (Activision) | Acquired by MSFT 2023 | no |

All five absent. The universe is built from **current** SP1500 membership (per `docs/sp1500_constituents.txt`, dated to 2025 membership) with no historical-membership reconstruction. This is consistent with how `fetch_sp1500.py` works (constituent list → per-ticker fetch). **Treat the universe as survivorship-biased — expect 1–2 pp/yr overstatement of backtest returns vs. a point-in-time membership universe.** No need to "re-audit" this; it is a hard architectural property of how the data was built.

## Feature parity

For the new study, the relevant features used by the three promoted studies (per `src/macro_signals.py` + feature_matrix construction) are: price/technicals, earnings dates, fundamentals, analyst targets, sector mapping, and macro signals.

Even if Mike promoted the **live caches** to a new snapshot today:

| Feature | Source file | SP1500 coverage if promoted |
|---|---|---|
| price + technicals (derivable from price) | `models/price_cache/` | ~1,473 of 1,473 (good) |
| fundamentals | `models/cache/fundamentals.json` | 1,473 of 1,473 (good) |
| sector_map | `models/cache/sector_map.json` | 1,473 of 1,473 (good) |
| **earnings_dates** | `models/cache/earnings_dates.json` | **491 of 1,473 (legacy only)** |
| **analyst_targets** | `models/cache/analyst_targets.json` | **484 of 1,473 (legacy only)** |
| macro signals (FRED) | `models/cache/macro_signals.parquet` | universal (ticker-independent) |

The yfinance earnings/targets fetch is what was failing — exactly the issue the Project State Tracker calls out and the reason SP1500 was paused. Until that's resolved (Polygon piggyback per the tracker's Section 3 is the candidate path), an SP1500 study would have to either:

- **drop earnings_dates and analyst_targets** as features (and confirm via Phase 0 that the three promoted studies don't have load-bearing dependence on them), or
- **restrict the universe** to the 491 legacy names (which defeats the point of running SP1500), or
- **fetch earnings via a different vendor** (Polygon/Massive — already wired up in `src/options/polygon.py`; would need an equity analog).

## Why I'm stopping rather than producing the full 4-section audit

The spec says: *"If you discover the SP1500 data isn't actually integrated yet (i.e., my mental model is wrong about what's in the snapshot), stop and report that clearly rather than proceeding."* That is this case. Producing a coverage-by-year table or a per-tier date-of-≥80%-coverage table against the snapshot would yield a misleading "looks fine" answer — because the snapshot doesn't have SP400/600 names at all, the year-by-year coverage of "current SP1500 members" in the snapshot is mechanically ~33% (the legacy 491/1473) at every year going back to 2018, and 0% before 2018. That's a meaningless tier audit.

## Recommendation — what would need to happen before a fresh SP1500 study is viable

Sequenced from least to most invasive:

1. **Decide whether 8 years (2018-01-02 → 2026-05) is enough total span.** A 10y/3y train/holdout split is not achievable. A 5y/3y or 6y/2y split is — but that's a notable downgrade from the ambition in the spec. If 8y total is acceptable, proceed; if not, the fetch policy needs to be re-run with an earlier start date (and we need to think about how to handle the IPO'd-after-start tickers — masking vs. limited-history-OK is what the current `sp1500_coverage_report.txt` already flags for 3 of them).
2. **Resolve the earnings/analyst_targets gap for the 982 new tickers.** The cheapest path per the State Tracker is Option C in Section 3: piggyback on Chris's Polygon key, write `src/polygon_equities.py` mirroring `src/options/polygon.py`, pop `stash@{0}`, and re-run the SP1500 fetch via Polygon for the earnings field. Pre-condition: confirm Chris is OK with the shared key.
3. **Promote the resulting live caches into a new snapshot** (`models/snapshots/equities/sp1500_<date>/`) and re-baseline Phase 0 against it. The three promoted studies need a parity check on the legacy 491 against the old snapshot before the new snapshot is treated as authoritative.
4. **Accept survivorship bias as a hard constraint of the universe**, document the expected overstatement (1–2 pp/yr), and frame any new study's headline metrics against that. A point-in-time membership reconstruction is out of scope for this project's data plumbing.

Once 1–3 are done, the fresh-study spec can be designed against a real SP1500 snapshot with a concrete date split. Until then, designing the study now would mean either designing against the legacy 491 (in which case it's not really "SP1500-fresh", it's "legacy with a new model") or designing against a universe that doesn't yet exist as snapshot data.

## Appendix — files referenced

- `models/snapshots/equities/pre_v2_20260505/` — canonical Phase 0 anchor (legacy 491)
- `models/snapshots/equities/post_macro_fix_20260506/` — identical universe, post-macro-fix Phase 0 re-baseline
- `models/price_cache/` (top-level, uncommitted) — 1,475 SP1500 price parquets
- `models/cache/` (top-level, uncommitted) — live feature caches
- `docs/sp1500_constituents.txt` — constituent lists (SP400_TICKERS, SP600_TICKERS)
- `docs/sp1500_coverage_report.txt` — output of the 2026-05-08 fetch
- `docs/diagnostics/sp1500_fetch_failures.txt` — 1,448-line failure manifest
- `stash@{0}` — paused fetch retry/sanity-gate work
- `docs/Project_State_Tracker.docx` — Section 3 has the Polygon piggyback analysis that gates SP1500 unblock
