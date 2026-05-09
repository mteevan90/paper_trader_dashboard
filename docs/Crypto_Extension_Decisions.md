# Crypto Extension — Decisions Memo

**Status:** Mike's responses to Chris's design memo (2026-05-09)
**Source:** Chris's design memo dated 2026-05-09 + Mike's review
**Purpose:** Single source of truth for both contributors and both Claude sessions

---

## Top-line decision

**Option C from Chris's memo is the path forward.** Sibling module (`src/crypto/`) plus namespaced R2, with a shared dashboard. NOT a fork; NOT a deep refactor.

Rationale: preserves Mike's working equity code bit-identically, isolates crypto cleanly, and bounds the cross-cutting changes to ~3 shared files. The shared dashboard is the project's value proposition — it's why the family sees both asset classes in one place.

---

## Decisions, indexed by Chris's questions

| # | Question | Decision | Reasoning |
|---|----------|----------|-----------|
| 1 | Sibling module vs fork? | **Sibling module** | Shared dashboard is the point. Forks would lose dashboard improvements that should benefit both. |
| 2 | Namespaced R2 vs separate bucket? | **Namespaced R2 in one bucket** | Same storage cost; one credential set; simpler ops. |
| 3 | One deployment vs two? | **One deployment with sidebar switcher** | Family sees both asset classes in one URL. That's the value. |
| 4 | Shared optuna DB vs separate? | **Separate** | DB schema doesn't care about asset class, but separation is simpler to reason about and back up. Per-asset paths: `models/cache/equities/optuna_studies.db` and `models/cache/crypto/optuna_studies.db`. |
| 5 | Crypto benchmark? | **BTC as primary; equal-weight top-10 as secondary** | "We beat BTC" is a clearer claim than "we beat a custom index." Show top-10 in detail views for context. |
| 6 | Universe construction? | **Dynamic by market cap rank as of rebalance day, with delisted-token handling** | Survivorship bias is brutal in crypto. Treat as first-class data engineering before any backtesting. See "Survivorship handling" below. |
| 7 | Snapshot storage budget? | **Daily OHLCV only for v1** | ~tens of MB total; minute-level can come later if research demands it. |
| 8 | Cloud deployment access? | **Mike owns Streamlit Cloud + R2 credentials.** Chris gets read access for verification; Mike does deploys for the shared dashboard. | Single point of accountability for the deployed surface. Revisit if it becomes friction. |
| 9 | CODEOWNERS strictness? | **Hard-block on shared files** | Soft convention isn't enough when Claude Code can edit files. The hard block catches accidental cross-namespace edits. |
| 10 | Architecture field naming? | **Don't reuse the `architecture` field name in crypto.** Pick a clearer name in crypto's BacktestConfig (e.g., `selection_logic`) or omit the field entirely if not needed. | The existing `architecture` field carries equity-specific semantics ("legacy" vs "regime-dependent"). Reusing the name in crypto would confuse later readers. |

---

## Things Chris didn't ask, but Mike's flagging

### Snapshot version tagging
Use per-asset version sequences, not a single global one:

```
models/snapshots/equities/pre_v2_20260505/
models/snapshots/equities/pre_v3_20260615/   (Mike's next equity snapshot)
models/snapshots/crypto/pre_v1_20260801/     (Chris's first crypto snapshot)
models/snapshots/crypto/pre_v2_20260920/
```

Independent version sequences mean Mike's equity v3 doesn't need to wait for Chris's crypto v3.

### Tab function layout
**Lean toward separate tab functions per asset class** (`tab_performance_equities()`, `tab_performance_crypto()`), not a shared function with asset-aware wrappers.

Reasoning: the equity Performance tab references concepts (earnings, sector caps, blackouts) that don't exist in crypto. Trying to unify behind a single function will create either a sprawling conditional or a leaky abstraction. Cleaner to duplicate the renderer code (each tab is ~100 lines) than pretzel it.

**Caveat:** the *visual conventions* stay shared — Layer 1 exec summary blue box, three-card KPI row, hero chart, detail expanders. Just the data binding and field semantics differ.

### Macro signal portability
Mike's macro signal is built from VIX + macroeconomic indicators (Treasury yields, etc.). **None of that is portable to crypto.** Chris will need a crypto-equivalent. Candidates:

- BTC funding rates (perpetual futures basis)
- Fear & Greed Index (free)
- Stablecoin supply changes
- Exchange net flows
- Realized volatility regime

This is a genuine research question for Chris, not a port. Worth its own design pass before crypto strategy work begins.

### Survivorship handling (high-priority before any backtesting)
Crypto's universe changes constantly — tokens delist, rebrand, fork, or simply die. If Chris's universe at backtest time is "today's top 200 by market cap" applied to historical data, every "alpha" he measures is fake (he'd be implicitly selecting only winners).

The honest approach:

1. Fetch market cap rankings as of EVERY historical rebalance day, not just today
2. Include delisted tokens with their delisting prices in the universe
3. Track listing dates so backtests don't trade tokens that didn't exist yet
4. Use CoinGecko's `/coins/{id}/history` endpoint per token per date for point-in-time market cap

This is meaningful data-engineering work. Worth budgeting a full day for the universe construction module before any strategy work.

---

## Sequencing

Three phases. Each gated on the previous one being verified-clean.

**Phase 1: Shared-edge refactor (Mike does this BEFORE Chris starts).**
- Namespace R2 layout by asset class
- Add asset-class parameter to `path_to()` / `r2_key_for()`, defaulting to `"equities"` so existing callsites are unaffected
- Add sidebar asset selector to dashboard (initially shows only "Stocks" — "Crypto" is greyed out until Chris's data exists)
- Add asset-class field to snapshot manifest
- Move existing `models/snapshots/pre_v2_20260505/` to `models/snapshots/equities/pre_v2_20260505/`
- Verify equity dashboard renders bit-identically before merging
- One PR, single commit, push to `main`

**Phase 2: Crypto smoke-test scaffolding (Chris does this).**
- Create `src/crypto/` with `backtest_config.py`, `data_source.py`, `model.py`, etc.
- Build minimal CCXT-based fetch for OHLCV
- Build CoinGecko-based universe construction with survivorship handling
- Build minimal crypto BacktestConfig and run a smoke study (~100 trials, top-10 token universe)
- Verify the smoke study produces sensible numbers (not asking for alpha; asking for plausibility)
- One PR, Mike reviews shared-file impacts (likely none if Phase 1 was done right)

**Phase 3: Real crypto research (Chris does this).**
- Universe expansion to top-200
- Real research on macro-equivalent signals, alt data, optimization
- Promote results to dashboard once meaningful

---

## Git workflow

### Branch model
- `main` — stable, deployed to Streamlit Cloud. Protected.
- `mike/equity-<topic>` — Mike's equity work
- `chris/crypto-<topic>` — Chris's crypto work
- Both contributors PR into main; squash-merge for clean history

### CODEOWNERS (lives at `.github/CODEOWNERS`)

```
# Equity-side ownership
/src/equities/             @mike
/src/dashboard_app.py      @mike
/src/data_source.py        @mike
/src/snapshot_for_cloud.py @mike
/src/backtest.py           @mike
/src/backtest_config.py    @mike
/src/optuna_runner.py      @mike
/src/macro_signals.py      @mike
/models/snapshots/equities/ @mike

# Crypto-side ownership
/src/crypto/               @chris
/models/snapshots/crypto/  @chris

# Shared files (require BOTH approvals)
/.github/CODEOWNERS        @mike @chris
/README.md                 @mike @chris
/requirements.txt          @mike @chris
/docs/                     @mike @chris
```

(Note: GitHub CODEOWNERS uses the most-specific match. Files under `/src/dashboard_app.py` are owned by Mike alone; if Chris needs to modify it, he opens a PR and Mike must approve.)

### Review rules
- Asset-only changes: self-merge after CI passes (no reviewer required)
- Shared-file changes: require the other person's approval, no exceptions
- Breaking changes to R2 layout: coordinate via direct comms before opening the PR

### Tagging promoted snapshots
- `equities-2026-05-15` for equity promotions
- `crypto-2026-08-01` for crypto promotions
- Independent tag namespaces; clean rollback targets per asset

### Deployment cadence
- Don't land equity AND crypto promotions in the same dashboard deployment window
- If the dashboard breaks after a promotion, we want to know which asset's changes caused it

---

## Open questions Mike's holding

These came up while writing this memo and don't have answers yet:

1. **CCXT exchange selection.** Chris suggested Binance + Coinbase as cross-checks. Worth asking: does he want to use both as primary sources or pick one as canonical? If both, how does he reconcile inconsistencies (different close prices on the same day)? This isn't a Mike-decides question, but it'll shape his data layer.

2. **CoinGecko free tier rate limits.** 30 calls/min × historical universe builds = potentially many hours of fetching. Worth considering whether the free tier is enough for a serious project or whether the $129/mo Pro tier is justified. Same calculation Mike did for Finnhub.

3. **Streamlit Cloud's resource limits.** The deployed app already has 8 tabs of equity content. Adding crypto rendering (more data, more charts, possibly minute-level resolution) might push memory limits. Worth a stress test before going live.

These are for Chris and his Claude to think through; Mike isn't blocking on them.

---

## Final word

This is a clean handoff because Chris's Claude did the homework — actually read the code, surfaced the real integration points, and proposed the right architectural compromise. The fact that Option C falls out cleanly when you read the code (rather than from theoretical reasoning about "asset class abstractions") is the signal that it's the right call.

Mike will do the Phase 1 shared-edge refactor. Chris waits until that's merged before starting Phase 2. After Phase 2 produces a working smoke study, both proceed in parallel with clear ownership boundaries and shared dashboard chrome.

If anything in this memo conflicts with what's in `Paper_Trader_Handoff.docx`, this memo is more recent and supersedes.
