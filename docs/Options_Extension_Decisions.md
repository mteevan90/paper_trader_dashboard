# Options Extension Decisions

| Field | Value |
| --- | --- |
| Document version | v1.4 — May 9, 2026 |
| Author | Chris Teevan |
| Repo path | `docs/Options_Extension_Decisions.md` |
| Status | Phase 1 + Sections 1, 2, 3 merged. Section 4 (Position + lifecycle model) specced and ready to ship. Sections 5–9 specced. |
| Predecessors | `docs/Crypto_Extension_Decisions.md` (sibling pattern), `docs/Comprehensive_User_Guide.docx` (Mike's authoritative reference) |

This is the canonical record of architectural decisions for the options module. When sections complete, when decisions get refined, or when v1.1 work begins, this file is the source of truth — same role the crypto decisions doc plays for crypto.

---

## 1. Strategy concept

A multi-strategy options research module focused on **active-management premium collection**. The driving research thesis:

> *Edge over hold-to-expiration comes from active position management. Close positions on profit targets (e.g., 50% of max profit for short-premium) and time stops (e.g., 21 DTE on monthlies). Don't ride theta to zero. Don't carry gamma into expiration week.*

This is the Tasty Trade methodology with honest validation discipline applied. It's where retail edge can plausibly exist if it exists at all — not from picking direction better than the market, but from systematic discipline around exit timing on premium collection.

Implications:

- The thesis applies cleanest to **short-premium** strategies (CCs, CSPs, credit spreads, iron condors). For these, profit targets and time stops compound directly into edge.
- The thesis applies *less* cleanly to **long directional** (you're paying theta, not collecting it; gamma is your friend, not your enemy). Directional gets its own engine mode in v2 — different math, different studies.
- The backtest engine must support **intra-position decision points every day**, not just open/close. Profit targets and time stops are first-class concepts.

The strategy phasing reflects this:

- **v1**: Covered Calls + Cash-Secured Puts (CCs + CSPs). Simplest, most inventory of historical data, defined max-loss profile.
- **v1.1**: Add credit spreads (verticals). Same engine, multi-leg position model exercised.
- **v1.2**: Add iron condors. Four-leg variant of v1.1 — natural extension.
- **v2**: Add long directional (calls/puts). Different engine mode. Long premium with managed-exit logic.

---

## 2. Asset interconnection

Options is the third sibling under the asset_class architecture Mike established in Phase 1 of the multi-asset extension. It uses the existing plumbing:

- `src/options/` for all options-specific code
- `path_to(local, asset_class="options")` and `r2_key_for(local, asset_class="options")` resolve paths and R2 keys
- `models/snapshots/options/<snapshot_name>/` for frozen historical data
- `models/cache/options/dashboard_results/best_<study_label>/` for promoted study output
- `snapshot_for_cloud.py --asset-class options` for upload
- Sidebar asset selector adds a third entry (Stocks / Crypto / Options)
- `.github/CODEOWNERS` extended so Chris owns `src/options/`, `tests/options/`, and the options docs; shared infrastructure files still require both contributors' approval

Adding `options` is mechanically the same shape as adding `crypto` was. The Phase 1 PR is the placeholder, the sectioned build follows.

---

## 3. Locked architectural decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Underlying market | US equity options on single names + index options (SPX/SPY/QQQ) | Single universe model — both share the same data pipeline (Tradier) and contract spec. Broadest scope without crossing into Deribit/crypto-options. |
| Execution mode | Backtest + paper-trade | Real-time chain access required for paper-trade fidelity. Live-trade deferred to v2+. |
| Driving thesis | Active-management edge over hold-to-expiration | Profit targets + time stops + early close. Tasty-Trade-style discipline. Tests where retail edge plausibly exists. |
| Strategy phasing | CCs + CSPs (v1) → verticals (v1.1) → iron condors (v1.2) → long directional (v2) | Active-management thesis applies cleanest to premium collection. Multi-leg position model gets exercised once at v1.1, reused thereafter. Directional is a different engine mode. |
| Data source (historical) | Polygon.io / Massive.com Options Developer tier ($79/mo, ~3-4 years rolling depth on expired options) | Tradier proved unsuitable for historical backtests — its `/markets/history` endpoint returns null for any expired option contract regardless of plan tier. Polygon carries full historical OHLCV including expired contracts, OCC symbol convention matches Tradier's, response shape is straightforward. Note: Polygon.io rebranded to Massive.com on October 30, 2025; existing `api.polygon.io` URLs continue to work without interruption per the official rebrand statement. The integration uses `api.polygon.io` for stability. |
| Data source (live execution, deferred) | Tradier (paper-trade and v2+ live execution) | Section 2's Tradier client retained for paper-trade snapshots and v2+ live order routing. Tradier's current chain endpoint, quotes, and earnings calendar (when fundamentals beta is available) remain useful for forward-looking work. Historical fetches now route through Polygon. |
| Greeks model | Black-Scholes (closed-form) | Closed-form, fast, well-understood. Adequate for short-dated equity/index options actively managed. Vol surface modeling (Heston, SVI) deferred to v1.1+. We compute our own Greeks for validation, but treat Tradier/ORATS Greeks as a sanity check. |
| Backtest engine | Hybrid — new options-native engine, reuse Optuna runner + config-dataclass shape | Position lifecycle differs fundamentally from equity day-walk (expirations, multi-leg atomic positions, intra-position exits). Optuna runner and BacktestConfig dataclass shape are reusable. |
| Underlying universe | SPX, SPY, QQQ + curated equity subset (5 names locked at Section 1; v1.1+ replaces with liquidity-filtered Mike's equity universe) | Indexes for clean data and high liquidity. Equity subset for diversification. Liquidity floor much harder than equity baseline (options-grade liquidity, not just stock liquidity). |
| Primary benchmark | SPY total return | Strategy-relevant baseline. The dashboard reports vs SPY total return for comparability with Mike's equity studies. |
| Secondary benchmark | CBOE BuyWrite Index (BXM) | Strategy-class-specific honesty check for covered call studies. Surfaces whether the active-management edge is real or just a beta repackaging. |
| Python version | 3.11.9 | Match Mike's documented convention. |
| TLS handling | `truststore` package, injected at script entry point | Carries the crypto lesson forward. Norton 360 TLS inspection breaks certifi-based requests. truststore reads from Windows trust store. Landed in main with Section 2; entry-point scripts call `src/options/_ssl.py:use_system_trust_store()` before any HTTPS-touching imports. |
| Sentiment / macro signals | Not in v1 architecture | Crypto needs sentiment because it's reflexive. Options have natural macro hooks (VIX, term structure, skew) — but these are options-derived signals, so wiring them into option strategies is recursive. Defer to v1.1. |
| v1 publish bar | Light: single Optuna run, train/val window split, SPY + BXM benchmarks, one promoted study | Mirror crypto v1 publish bar. Walk-forward, multi-regime studies, vol-of-vol modeling are v1.1+. |
| Snapshot for v1 study | `pre_options_v1_<date>` under `models/snapshots/options/` | Locks the data inputs at promotion. Reproducibility guarantee carries from equities. |
| Position sizing rule | Fixed-risk: `max_loss_pct_of_portfolio` field (default 2%, search 1–4%) | Locked at Section 5. Each position sized so its theoretical max loss is at most this fraction of starting portfolio value. Kelly-style and CVaR-aware sizing deferred to v1.1+ per §10. |
| Starting capital | Configurable per study via `BacktestConfig.starting_capital` (default $100k) | Snapshot-able with the config for reproducibility. Allows sensitivity analysis on capital scale (e.g., does the strategy work at $25k? At $1M?). Locked at Section 6. |
| Optimization objective | Calmar ratio (compound annualized return / max drawdown) on training window | Locked at Section 7. Drawdown-aware, no tuning knob, defensible without hyperparameter justification. SPY total return and BXM remain reporting-time benchmarks but are not used as the optimization target — Calmar is asset-class-appropriate for premium collection where return distributions are skewed and Sharpe assumptions are violated. |
| Promotion gate criteria | Automated function with five hardcoded thresholds + human override capability | Locked at Section 8. Automated check codifies discipline mechanically; human override allows nuanced judgment when results are close to thresholds. Both decisions captured in `promotion_decision.json` for audit. The five criteria: val_calmar >= 0.5 × train_calmar (overfit check), beats SPY total return on val window (basic alpha), beats BXM on val window for CC studies only (strategy-class-specific honesty), no single underlying >50% of training alpha (concentration check), high-IV-regime alpha within 2x of low-IV-regime alpha (regime independence). |
| BXM benchmark fetch path | Tradier `/markets/quotes` for `$BXM` as primary; yfinance `^BXM` as documented fallback | Locked at Section 8. Tradier index quote endpoint may or may not carry BXM — first call verifies and either succeeds or triggers the documented yfinance fallback. yfinance is already in the project's transitive dependencies via Mike's equity infrastructure. SPY total return is fetched via the existing `src/options/tradier.py` history endpoint (no new fetcher needed). |

---

## 4. Underlying universe

### Initial universe (Section 1, smoke-test scope)

- **Indexes (3)**: SPX (cash-settled, AM-settled monthlies), SPY (PM-settled), QQQ (PM-settled)
- **Curated equities (5, locked at Section 1)**: AAPL, MSFT, NVDA, JPM, XOM. Mix of tech mega-cap and non-tech to avoid pre-baking the NVDA/META concentration trap into the smoke study itself.

### Long-term universe (v1.1+)

The Section 1 smoke universe is a hand-curated stub. The long-term universe is:

- The 3 index options (SPX, SPY, QQQ), unconditionally.
- All tickers in Mike's equity universe (currently 491 — S&P 500 + Nasdaq 100; expandable to S&P 1500 if/when the SP1500 fetch unblocks) that pass an **options-grade liquidity filter** at the rebalance date.

The liquidity filter combines:

- Minimum **30-day average daily option volume** across the chain (specific threshold tuned at Section 8 against the smoke study; conservative starting point ~10k contracts/day across all expiries).
- Minimum **weekly options availability** for active-management thesis.
- Bid-ask **spread floor** at the 0.30-delta band (conservative starting point: spread < 5% of mark for monthlies).

This shifts the universe construction problem from "hand-curate ~20 names" to "filter Mike's equity universe by options-grade liquidity per-date." The Section 1 API is designed to scale to that without rework: `get_universe_at_date_v2(at_date, top_n)` and `is_underlying_active_v2(ticker, at_date)` are the seams where the filter logic lands when v1.1+ implements them. The on-disk shape (`UNIVERSE_PARQUET_SCHEMA`) reserves nullable columns (`options_adv_30d`, `chain_spread_pct_30d`) for the filter inputs.

### What the universe explicitly excludes

- Penny options (low-priced underlyings with $1 strike spacing — uneconomic)
- Underlyings with frequent special dividends (assignment risk math gets messy)
- Underlyings that have undergone splits or M&A in the backtest window (data continuity issues)
- Underlyings with no monthly cycle (weeklies-only is harder to backtest cleanly for monthly-thesis studies)

---

## 5. What carries from equities

The shared-infrastructure inheritance is substantial. These transfer cleanly:

- **`asset_class` plumbing** — `path_to()`, `r2_key_for()`, namespaced filesystem and R2 layout
- **Three-layer dashboard pattern** — exec summary / KPIs / detail
- **Train/validation window discipline** — both reported, both labeled
- **Optuna TPE optimization framework** — same runner skeleton, separate `optuna_studies.db` per asset class (mirrors crypto's separate DB)
- **Rolling 12-month outperformance objective** — adapted to vs SPY total return
- **Sanity gates on every external API fetch** — refuse cache write if <50% non-empty (mirrors equity's yfinance gate and crypto's planned CCXT/CoinGecko gates)
- **Snapshot reproducibility system** — frozen inputs at promotion, immutable thereafter
- **Concentration analysis on every promoted study** — the NVDA/META template, adapted (see Section 7)
- **`promotable: false` flag** — code-enforced gate against accidental smoke-study publication
- **CODEOWNERS multi-contributor workflow** — Chris owns options, shared infrastructure requires both approvals
- **`BacktestConfig` dataclass shape** — per the hybrid architecture choice. The fields differ but the dataclass-as-config pattern is the same.

---

## 6. What does NOT transfer from equities

These are explicitly different in options and need to be designed fresh:

- **Day-walk hold-until-rebalance loop** — dead. Positions have hard expirations. Replace with active-management lifecycle (entry rules → daily exit-rule eval → open/close).
- **Top-N selection by score** — dead. CCs and CSPs aren't "pick top 5 by composite score." Strategy is "for each underlying meeting filter criteria, place position at strike-X delta if conditions met."
- **GICS sector caps** — partially dead. Concentration matters but the taxonomy is closer to "single-name vs index" and "underlying overlap across positions" than to GICS sectors.
- **ATR-based stops** — dead. Vol-of-vol and IV regime matter more than realized-vol stops for options.
- **VIX-based macro signal** — recursive and conceptually broken. VIX *is* an options-derived signal; building option strategies on top of VIX-based macro creates a feedback loop. Replace with: term-structure shape (front-month vs 3-month VIX), realized-vs-implied vol gap, skew signals — to be addressed in v1.1.
- **Equity XGBoost model** — dead. Different feature set; if ML enters in v2 it's trained from scratch on option-native features (IV percentile, term structure, skew, gamma exposure).
- **yfinance fundamentals** — dead. Fundamentals aren't strategy-relevant for short-dated active premium collection. Tradier provides everything we need (price, chain, Greeks, earnings dates).
- **Earnings blackout flag (binary)** — refined. For single-name short premium, the question isn't binary "blackout" but "do we trade through earnings to capture the IV crush, or avoid the gap risk?" v1 default: avoid earnings windows for single-name short positions. Indexes don't have this issue. Revisit in v1.1.
- **Stop losses on underlying price** — replaced. Options have natural max-loss bounds (long premium can't lose more than premium paid; short premium has defined max loss for spreads, undefined but margin-bounded for naked short). Stop logic operates on position P&L, not underlying.
- **Fee model (5bps)** — replaced. Options fees are per-contract: Tradier's retail Lite plan is $0.35/contract one-way (verified against tradier.com/individuals/pricing during Section 2 implementation, May 2026); Pro/Pro Plus tiers are $0.00. Section 2 records the placeholder constant `TRADIER_OPTION_FEE_PER_CONTRACT_USD = 0.35` in `src/options/tradier.py` with a `# TODO verify` note; the actual fee model lives in Section 5 and re-validates against the then-current schedule. Exchange/regulatory pass-throughs add roughly $0.10/contract (clearing $0.0775, ORF $0.02295, TAF $0.00279 as of May 2026, subject to change without notice) and are not in the v1 fee model. Slippage assumption: mid-price minus half the spread on entry, mid + half-spread on exit (conservative).

---

## 7. Concentration analysis pattern (the NVDA/META template, adapted)

Mike's NVDA/META finding for equities — that ~62% of #325's CAPM alpha came from two names — is the template every promoted study must follow.

For options, the analog dimensions to ablate:

- **By underlying**: re-run study with each underlying blacklisted in turn. Surfaces single-name dependency (the direct NVDA/META analog).
- **By DTE band**: re-run with positions in narrow DTE bands excluded (e.g., "drop all entries 30–35 DTE"). Surfaces whether the edge is concentrated in a specific time-to-expiration window.
- **By IV regime**: re-run with positions opened during top-quartile IV-rank excluded vs bottom-quartile. Surfaces whether the strategy works only in high-vol regimes (a real concern for premium collection).
- **By strategy variant**: when v1.1+ has multiple strategy classes promoted, re-run with each class blacklisted. Surfaces whether one variant is doing all the work.

Every promoted study includes this analysis in its Layer 1 exec summary. The dashboard's Performance tab leads with the honest read, not the headline number. (Mirrors how Mike is updating the equity Performance exec summary to lead with CAPM alpha + NVDA/META concentration.)

---

## 8. Options-specific gotchas

These are carried forward into the build. Anything new discovered during sectioned work gets appended here.

- **Bid-ask spreads dominate fill assumptions.** OOM strikes can have 10–20% spreads. Backtest cannot assume mid execution. v1 fill model: mid − half-spread on entry (open short premium), mid + half-spread on exit (close short premium). Revisit with NBBO modeling in v1.1+.
- **Pin risk near expiration.** Strikes very close to spot can flip in/out of the money in the final hour. The 21 DTE time stop sidesteps this for monthlies. Document the limitation; don't try to model pin risk in v1.
- **Assignment risk on short positions.** American-style options can be assigned anytime. Most likely before ex-dividend on short calls. v1: avoid ex-div windows on short call positions on dividend-paying underlyings. Document the limitation. Indexes are European-style and don't have this issue.
- **IV crush around earnings.** Single-name short premium through earnings = systematic IV crush profit *and* gap risk. v1 default: avoid earnings windows for single-name short positions. Indexes unaffected.
- **Liquidity changes by DTE.** Weekly options have lower OI than monthlies. v1 sticks to monthly cycle (~30–45 DTE entry, with the 21 DTE time stop landing well before expiration).
- **Strikes are discrete.** Can't write "exactly 30-delta CSP" — choose the nearest available strike. Engine needs a `strike_selector(target_delta, chain)` function. Document the rounding choice in study output.
- **Multi-leg positions are atomic.** A vertical spread is one position with two legs, not two independent positions. The position model encodes this from day one (Section 4) so v1.1 doesn't require a refactor.
- **Options data is dense.** A single underlying has ~1000 contracts at a time (multiple expirations × strike grid × calls/puts). Storage and query patterns differ from equities. Section 2 addresses this with parquet partitioning by `underlying / expiration_date`.
- **Norton 360 TLS still applies.** Tradier API uses HTTPS. `truststore.inject_into_ssl()` at script entry point via `src/options/_ssl.py`. Landed in main with Section 2.
- **OCC symbol format.** Standard 21-character format (`AAPL220617C00270000`). Parsing/generation utility at Section 1.
- **Dividend handling affects pricing.** Equity dividends affect option pricing (forward-price adjustment). Tradier provides dividend data; we use it directly rather than recomputing.
- **Holiday and early-close calendars.** Settlement on holiday-shifted expiries needs care. Use NYSE calendar (`pandas_market_calendars`).
- **Tradier sandbox is 15-min delayed without funded account.** For backtest this doesn't matter (we use historical). For paper-trade development it means a small staleness in the sandbox feed — acceptable for v1, document it.
- **Tradier exposes per-contract history, not historical chain enumeration.** `/markets/history` accepts an OCC symbol or an underlying ticker and returns daily OHLCV. There is no endpoint for "what strikes existed for SYMBOL on DATE." Section 2 ships per-contract history, current-chain snapshot, and expirations only. Section 6 reconstructs the chain at backtest time by enumerating candidate OCC symbols and accepting that many will return empty histories. If candidate-enumeration overhead becomes binding, escalate to Polygon or ThetaData per §3 row 5.
- **Tradier rate limits surface in response headers.** `X-Ratelimit-Used`, `X-Ratelimit-Allowed`, `X-Ratelimit-Available`, and `X-Ratelimit-Expiry` are returned per request. The fetcher honors these in addition to a conservative fallback cap. Sandbox limits are tighter than production for market-data endpoints — the fallback respects sandbox.
- **Tradier API returns XML by default.** Set `Accept: application/json` on every request. Forgetting this is a silent-failure class where parsing chokes on what looks like JSON but is an XML envelope.
- **BSM treats American-style options as European.** Early-exercise premium is ignored in v1. The premium is small for non-dividend-paying single names but non-trivial for dividend-paying names near ex-div. Adequate for short-dated actively-managed positions (the v1 thesis). v1.1+ adds Barone-Adesi-Whaley approximation if Section 8 surfaces a meaningful gap. SPX is European-style and unaffected; SPY/QQQ are American-style ETFs but their distribution mechanics don't trigger the same early-exercise math as single-name ex-div windows.
- **Expiration settlement differs by underlying type.** SPX is European-style and cash-settles to intrinsic at expiration. SPY/QQQ/equity options are American-style and share-settle when ITM at expiration: short call ITM → short shares delivered (-100 per contract); short put ITM → long shares delivered (+100 per contract); long call/put ITM → cash credit equal to intrinsic. Section 4's Position model sets `state=ASSIGNED` for share-settled cases and surfaces the resulting equity exposure for Section 6 engine to handle. Long-leg ITM on share-settled options is treated as cash-settled in v1 — automatic exercise logic for retail accounts is broker-dependent and not modeled.
- **Cash legs in v1 are treated as zero-yield.** CSP collateral cash is held in the position but does not earn the risk-free rate. Cash drag is a real cost to the strategy in high-rate environments (4–5% in 2026). v1.1+ adds risk-free yield accrual to cash legs; impact on study results documented in concentration analysis.
- **Provider depth verification is a spec-time discipline, not a run-time one.** The original Section 2 spec locked Tradier as the data source citing "verified, working" — but verification only covered current-date data. Historical depth on expired contracts was assumed, not probed. The 8+ hour stalled production v1 study + ~12 hours of fix-forward debugging traced to that assumption. Going forward, every external data dependency must be probed against worst-case dates (oldest in study window, expired instruments) at spec time before the design is locked. Section 2.5 ships with this discipline applied: probes against 2022-05, 2023-01, and 2024-06 ATM strikes verified Polygon's actual coverage before integration code was written.
- **Cash-constrained sizing in v1.** The engine refuses to open positions that would push cash below zero. CSP collateral is locked while position is open. CC strategy reserves cash equal to current stock holdings' cost basis. Margin-aware sizing (allowing notional > cash up to broker margin limits) is a v1.1+ concern.
- **Strike spacing varies by underlying.** OCC standard equity option strikes are $1 below $25 spot, $2.50 between $25 and $200, $5 above $200. SPX uses $5 strikes at-the-money (sometimes $25 in deep wings, ignored in v1). Section 6's chain reconstruction encodes these conventions in `get_strike_spacing(underlying, spot)`.

---

## 9. Phase 2 — Section breakdown

Mirrors the crypto Phase 2 sectioning. Each section is a self-contained PR that lands cleanly without breaking previous sections. Branch naming: `chris/options-section-N-<topic>`.

| # | Section | Status | Notes |
| --- | --- | --- | --- |
| 1 | Universe + contract spec | MERGED (PR #4) | `UnderlyingMeta` (12 fields, frozen+slots; `dividend_yield` added in Section 3) and `ContractSpec` (4 fields) dataclasses, OCC symbol parse/generate utility (strict 21-char round-trip), `UNIVERSE_PARQUET_SCHEMA`, static 8-underlying universe (SPX, SPY, QQQ + AAPL, JPM, MSFT, NVDA, XOM), v1 public API + v2 stubs reserving the seam for v1.1+ filter-based expansion against Mike's equity universe. Mirrors crypto Section 1 shape. |
| 2 | Tradier OHLCV + chain fetcher | MERGED (PR #5) | Sandbox + production tokens via env vars (`TRADIER_SANDBOX_TOKEN`, `TRADIER_PRODUCTION_TOKEN`); bearer-token auth (no OAuth). Header-driven rate-limit handling (`X-Ratelimit-*`) with conservative fallback. `truststore.inject_into_ssl()` invoked by entry-point scripts via `src/options/_ssl.py` helper, mirroring crypto's pattern. Per-contract OHLCV by OCC symbol (or underlying ticker) + current-chain snapshot + expirations endpoint; historical chain enumeration is **not** offered by Tradier and is reconstructed at backtest time in Section 6 by candidate-OCC enumeration. 1-day TTL on history cache; chain snapshots immutable per `<run_date>` file. Sanity gate at <50% non-empty refuses history cache write. Caches to `models/cache/options/tradier/`. truststore landed in main with this section. Live smoke against Tradier sandbox: SPY history 20/30 days (66.7% coverage, gate pass), SPY chain 508 contracts, real cache write succeeded — truststore correctly bypasses Norton 360 TLS inspection. **Status update from Section 2.5 (May 2026):** Tradier's `/markets/history` endpoint was discovered to return null for expired option contracts at all plan tiers, blocking the v1 production study. Tradier client retained for current chain snapshots, quotes, and v2+ live order routing; historical OHLCV fetches now route through Polygon (Section 2.5). The Section 2 client and tests remain on main and continue to function for live data scenarios. |
| 2.5 | Polygon historical fetcher | NOT STARTED | New `src/options/polygon.py` implementing the same `HistoryFetcher` protocol as Section 2's Tradier client. Raw `requests` (no SDK) for consistency with Tradier client and to keep truststore + Norton 360 TLS handling proven. OCC symbol conversion (strip Tradier-style padding, add `O:` prefix). Auth via `apiKey` query parameter from `POLYGON_API_KEY` env var. Parquet caching at `models/cache/options/polygon/history/<symbol>.parquet`, same TTL discipline as Tradier cache. Sanity gate at <50% non-empty refuses cache write. Error discipline: 200 with empty results returns empty DataFrame (contract didn't trade); 403 NOT_AUTHORIZED re-raises with clear "data timeframe outside plan" message; 401/5xx/429 handled with retries-with-backoff or re-raise per type. `chain_reconstruction.py` switches default fetcher from Tradier to Polygon. Section 8's `run_v1_study.py` default `start_date` updated to 2023-01-02 to match Polygon's enforced historical depth. Self-merges under Chris's CODEOWNERS rule (no shared-file edits — `requests` already a transitive dependency). |
| 3 | Black-Scholes Greeks module | MERGED (PR #6) | Closed-form Black-Scholes-Merton (continuous dividend yield `q`). Pure-function module exposing `price`, `delta`, `gamma`, `theta_per_day`, `vega_per_pct`, `rho_per_bp`, `implied_vol`, plus `compute_all` returning a frozen `GreeksResult` dataclass. Trader-convention units throughout, encoded in field names so consumers don't have to remember (theta scaled to per-calendar-day, vega per 1 IV point, rho per 1 bp). Day count ACT/365 hardcoded in a `time_to_expiration` helper (basis flexibility deferred to v1.1+ if a study needs ACT/360). Caller passes `q` (SPX: 0; SPY/QQQ: distribution yield; single names: ticker-specific) — Section 1's `UnderlyingMeta` was amended with a `dividend_yield` field in the same PR so callers have a canonical lookup. American-style treated as European — early-exercise premium ignored; documented in §8. `implied_vol` solver via Brent's method ships in Section 3 because Section 6 needs it for backtest IV reconstruction (Tradier per-contract history returns OHLCV without IV). The "below intrinsic" check uses the proper European lower bound (`max(K·e^(-rT) - S·e^(-qT), 0)` for puts, call analogue for calls), not the American intrinsic — caught mid-flight; deep-ITM puts with r > q would have spuriously raised under the simpler check. Edge cases handled explicitly: `T<0`/`S<=0`/`K<=0`/`vol<0` raise; `T==0` or `vol==0` returns intrinsic + zero Greeks except delta = ±1/0 ITM indicator. Pure-math validation only: Hull reference values (`pytest.approx(abs=0.005, rel=0.001)` to absorb textbook rounding) + put-call parity + finite-difference Greeks (~1e-4 tolerance). ORATS comparison is a manual one-time post-merge sanity check via `scripts/fetch_options_chain.py`, not a permanent test. |
| 4 | Position + lifecycle model | NOT STARTED | Position dataclass (frozen + slots) representing a multi-leg options position with first-class active-management exit rules. **Hybrid representation:** canonical `legs: tuple[Leg, ...]` shape used by all engine code, plus per-strategy classmethod constructors (`Position.covered_call`, `Position.cash_secured_put`, future `Position.vertical_spread`, etc.) for self-documenting construction with validation in `__post_init__`. `Leg` carries (contract, sign, quantity); contract is `ContractSpec` (option), `StockContract` (stock), or `CashContract` (cash collateral) — discriminated union via type. Explicit cash legs on CSPs symmetric with explicit stock legs on CCs — honest portfolio accounting. `ExitRules` dataclass with hardcoded fields (`profit_target_pct`, `time_stop_dte`, `stop_loss_pct`, at least one required) — Optuna-friendly fixed parameter space. `PositionState` enum: `OPEN` → `CLOSED_MANAGED` (active-management exit) | `EXPIRED_ITM` (cash-settled, SPX) | `EXPIRED_OTM` | `ASSIGNED` (share-settled, equity options at expiration). Mark-to-mid P&L using daily chain close. Honest settlement at expiration: SPX cash-settles to intrinsic, SPY/QQQ/equities share-settle (the spawned equity position from `state=ASSIGNED` is created and managed by Section 6 engine, not in Section 4). Frozen with `evolve(**changes)` method that returns new instance via `dataclasses.replace`. Position aggregates leg-level Greeks via Section 3's `compute_all` (cash and stock legs contribute zero Greeks except stock delta). Closure reason format documented: `profit_target_<pct>`, `time_stop_<dte>`, `stop_loss_<pct>`, `expired_itm_cash_settled`, `expired_otm`, `assigned_call`, `assigned_put`. v1 ignores early assignment per §8 (avoid ex-div windows on short calls). **`entry_credit` convention (locked at Section 4 implementation):** `entry_credit` follows trader semantics — net cash received at open, positive for credits (CSP `+put_premium*100`, CC `-stock_basis*100 + call_premium*100`), negative for debits (long premium positions). `mark_to_market` returns P&L in dollars and excludes cash legs from the leg sum (cash held at par with zero yield per §8): `P&L = sum(sign × qty × multiplier × mark for non-cash legs) + entry_credit`. Stock legs **do** contribute (a CC's stock leg moves with the underlying). `should_exit` thresholds use `abs(entry_credit)` for symmetry across credit/debit positions: profit triggers when `P&L ≥ profit_target_pct × abs(entry_credit)`, stop_loss triggers when `P&L ≤ -stop_loss_pct × abs(entry_credit)`. The original Section 4 spec said "sum over legs ... minus entry_credit", which couldn't simultaneously satisfy trader-view `entry_credit` and the P&L-zero-at-open invariant once explicit cash collateral legs were introduced; the trader-view + non-cash-sum convention was chosen at implementation time. |
| 5 | Options BacktestConfig | NOT STARTED | Frozen + slots `BacktestConfig` dataclass bundling all study levers into a single immutable, serializable object the engine consumes. Embedded `ExitRules` from Section 4 (composition, no field duplication). Embedded `FeeModel` dataclass with broker_fee_per_contract and regulatory_fee_per_contract broken out for fee-sensitivity analysis (Tradier Lite defaults: $0.35 broker + $0.10 regulatory; structured rather than flat-composite for honesty in study output). Universe specification as `tuple[str, ...]` field defaulting to the 8 v1 names — smoke studies override with smaller subsets, v1.1+ replaces with liquidity-filtered selection. `BacktestConfig.suggest(trial)` classmethod owns Optuna parameter ranges (cohesive: parameter definitions and search bounds in the same place). Per-strategy_class instances — CSP and CC run as separate studies and are compared at the study level, not within a single config. Train/val split via `start_date` + `end_date` + `train_val_split_date` fields; engine processes one walk and tags days by which side of split they fall on. Position sizing as fixed-risk: `max_loss_pct_of_portfolio` (default 0.02). `promotable: bool = False` flag enforced at study upload to prevent accidental publication of smoke runs. `to_dict()` / `from_dict()` for snapshot reproducibility. |
| 6 | Options backtest engine | NOT STARTED | Daily walk loop on NYSE trading calendar (`pandas_market_calendars`). Mutable `PortfolioState` updated in-place each simulated day for performance — engine internals don't need the immutability discipline that user-facing dataclasses (Position, BacktestConfig) carry. Per-day sequence: (1) evaluate exit rules on open positions and close triggered ones at mid + half-spread; (2) handle expirations via `Position.resolve_expiration` per Section 4; (3) liquidate any CSP-spawned shares at next-day open per locked decision; (4) for CC strategy, buy back shares if any were called away on prior day to continue writing; (5) evaluate entries — for each eligible underlying (in universe, not in earnings window, no existing position of same strategy_class), reconstruct historical chain, select strike, open at mid − half-spread; (6) mark-to-market and record daily snapshot tagged with train/val label. Strategy mode asymmetry: CSP starts with cash and never holds shares except transiently after assignment (always liquidated). CC starts by buying 100 shares per concurrent slot from universe, writes CCs against holdings, re-buys shares after assignment to continue. Cash-constrained sizing — engine refuses entries that would require cash beyond what's available. Skip-and-continue error handling with per-reason counters surfaced in `StudyResults`. Output: `StudyResults` dataclass with daily snapshots, closed positions, skip counters, wall-time, persisted to parquet at `models/cache/options/study_results/<study_label>/<run_id>/`. Historical chain reconstruction lives in `src/options/chain_reconstruction.py` — candidate-OCC enumeration with strike-spacing-by-underlying conventions, IV reconstruction via Section 3's `implied_vol()`, strike selection by closest delta. Earnings calendar fetched from Tradier's corporate calendar endpoint via `src/options/earnings.py`, cached per ticker. |
| 7 | Optuna runner + smoke study | NOT STARTED | Optuna TPE optimization runner with Calmar objective. Compound annualized return divided by max drawdown computed on training-window snapshots only — validation data is tagged in StudyResults but excluded from optimization. SQLite storage at `models/cache/options/optuna_studies.db` (separate from crypto's per asset-class isolation). Resume capability via Optuna's `load_if_exists=True`. Top-K (default 5) trials persist full StudyResults parquet to `models/cache/options/optuna_studies/<study_label>/trial_<N>/`; non-top-K trials retain only Optuna's parameter-and-objective summary in the SQLite study. Skip-and-continue error handling: failed trials return -1.0 sentinel objective and the study continues. Smoke study covers all 8 underlyings × 2 strategies × 5 trials × 6-month window with `promotable=False` enforced — exercises universe iteration, both strategy modes, and parquet persistence end-to-end (~30min runtime estimate). Pruning disabled in v1 (engine doesn't report intermediate objective values during the daily walk; v1.1+ may add MedianPruner if Optuna sweeps become time-binding). |
| 8 | Real study (active-management CCs + CSPs) | NOT STARTED | Production v1 study orchestrator. Wraps `run_optuna_study` from Section 7 with: (1) primary Optuna run on full v1 universe, both strategy classes (CSP and CC as separate Optuna studies), 100 trials each, 2-year backtest window (Jan 2023 through Dec 2024 train, Jan 2025 through May 2026 val); (2) concentration analysis ablation per memo §7 — re-runs with each underlying blacklisted (8 ablations per strategy), each DTE band excluded (5 bands × 2 strategies), each IV regime excluded (top vs bottom quartile, per-underlying IV-rank against 252-day rolling distribution), strategy variant ablation (run with CSP only and CC only); (3) benchmark series fetch — SPY total return primary via existing `tradier.fetch_history` infrastructure with dividend reinvestment, BXM secondary via Tradier `/markets/quotes` index endpoint with yfinance `^BXM` fallback if Tradier doesn't carry it; (4) automated promotion gate `evaluate_promotion()` returning structured pass/fail with explanations on five hardcoded criteria (val_calmar >= 0.5 × train_calmar, beats SPY on val window by Calmar, beats BXM for CC studies, no underlying contributes >50% of training alpha, IV regime ratio within 2x); (5) human override capability — `promotion_decision.json` captures both the automated recommendation and the human's final decision with reasoning. Output structure: `models/cache/options/v1_study/<run_id>/` containing primary study output, ablation subdirectories, benchmark series, promotion decision JSON. Snapshot for v1 publish: `models/snapshots/options/pre_options_v1_<date>/` per memo §3 row 15. |
| 9 | Dashboard wiring + publish | NOT STARTED | Replace Options placeholder with three-layer view. Tab structure: **Performance**, **Open Positions**, **Trade History**, **Greeks Exposure** (portfolio-level delta/gamma/theta/vega over time), **Risk & Behavior**, **Reliability**, **Tuning History**, **Glossary**. Adapt equity dashboard chrome — same scaffolding, options-relevant content. |

---

## 10. Open questions / v1.1+ deferrals

These are not blockers for v1 but should be tracked. When v1 ships, this list is the seed for the v1.1 design conversation.

- **Earnings handling refinement.** v1 defaults to avoiding earnings windows on single-name short positions. v1.1 should evaluate trading through earnings as a deliberate IV-crush study with separate framing and risk profile.
- **Walk-forward validation.** v1 uses single train/val split. v1.1 adds rolling walk-forward.
- **Vol surface modeling.** Heston, SVI, or simpler local-vol fits. v1.1+ when there's a study that justifies the modeling overhead.
- **VIX term structure / skew as macro inputs.** Replaces the dead VIX-as-macro signal from equities. Conceptually challenging because options strategies on top of options-derived macro are recursive — needs careful framing.
- **NBBO-based fill modeling.** v1 uses mid ± half-spread. v1.1 with real NBBO history if Polygon or ThetaData is added.
- **Live-trade integration via Tradier brokerage.** v2+. Real capital, real regulatory considerations. Not without significant additional discipline.
- **ML-driven entry/exit signals.** v2+ leverage of the RTX 5080's CUDA capacity. Possible features: IV percentile time-series, gamma exposure, dealer positioning, term-structure shape. Train from scratch on option-native features.
- **Crypto options on Deribit.** Conceptually fits as a fourth asset class (`asset_class="crypto_options"`), or as a sub-strategy of crypto. Deferred — not in scope for any v1.x.
- **Position sizing rule.** v1 default: fixed-risk per position (e.g., max-loss = 2% of portfolio). Alternative: Kelly-style or CVaR-aware. Revisit once v1 study has data.
- **American-style early-exercise modeling.** v1 BSM ignores it. v1.1+ adds Barone-Adesi-Whaley closed-form approximation (or a small binomial tree) for stricter pricing on dividend-paying single names near ex-div windows. Pulls in iff Section 8 finds the v1 gap meaningful.
- **Day-count basis flexibility.** v1 hardcodes ACT/365 in `time_to_expiration`. v1.1+ adds basis selection (ACT/360, business-day) if a study or strategy class needs it.
- **Early assignment on American-style short options.** v1 avoids ex-div windows for short calls on dividend-paying single names (per §8) but does not actively model early-exercise probability. v1.1+ adds early-assignment risk modeling using BSM-derived early-exercise premium and ex-div date proximity.
- **Cash leg interest accrual.** v1 treats all cash legs as zero-yield. v1.1+ accrues risk-free rate on cash collateral; relevant for high-rate-environment realism (CSP cash drag was ~$2/contract/month in 2026's rate regime — small but compounds).
- **Long-leg automatic exercise modeling.** v1 cash-settles long ITM legs at expiration. Real retail brokers auto-exercise long ITM options on expiration day if ITM by ≥$0.01, with broker-specific opt-out windows. v1.1+ adds broker-realistic auto-exercise logic where it materially changes P&L.
- **Sizing rule alternatives.** v1 uses fixed-risk sizing (`max_loss_pct_of_portfolio`). v1.1+ may evaluate Kelly-style sizing (variable based on edge estimation) and CVaR-aware sizing (scales position size by tail-risk estimate). Either requires a real-data baseline before introducing variable sizing.
- **Margin-aware position sizing.** v1 is cash-constrained — every CSP requires full collateral, every CC requires full share ownership. v1.1+ may add margin-aware sizing where the engine permits naked-short positions sized against broker margin requirements rather than full collateral. Increases capital efficiency at the cost of ruin risk. Real but requires honest validation including stress scenarios.
- **Optuna trial pruning.** v1 runs every Optuna trial to completion (no early-stop on unpromising trials). v1.1+ may add MedianPruner or HyperbandPruner — requires the engine to report intermediate objective values during the daily walk, which is a non-trivial engine refactor. Worth it only when Optuna sweeps become time-binding.
- **Batch chain pre-fetch.** v1 engine fetches historical chain data lazily per simulated trading day during the walk loop. v1.1+ adds an upfront batch pre-fetch that populates the cache for the full backtest window before any trials begin — eliminates the "trial 1 is slow" pattern, also enables parallel chain population.
- **Vectorized BSM across candidates.** v1 computes IV and delta per-candidate iteratively (Brent's method per OCC, BSM delta per OCC). v1.1+ vectorizes both across all candidates simultaneously using numpy broadcasting — expected 5-10x speedup on the strike-selection step.
- **Smarter strike grid.** v1 enumerates ~40 candidates per underlying per entry day at uniform strike spacing across ±20% of spot. Most are wasted (we only open one position). v1.1+ uses target_delta + recent IV to bracket the relevant strike range tightly, fetching ~10 candidates instead of 40.
- **Live data path via Tradier** is preserved but unused in v1. v2+ paper-trade and live order routing will reactivate the Tradier client. Section 2's code stays on main as-is; only the historical-data path was rerouted to Polygon.
- **Polygon plan depth is enforced more conservatively than the marketing implies.** Options Developer tier advertises "4 years historical data," but the plan's actual hard floor is approximately the start of the third calendar year prior — i.e., May 2026 → 2023-01-02 floor, not 2022-05-02 as the literal 4-year boundary would suggest. Probed during Section 2.5 spec verification. Study windows must respect this floor. v1.1+ may revisit if Polygon clarifies the boundary or offers higher tiers.

---

## Appendix A — Phase 1 spec (shared-edge refactor) [archived]

The Phase 1 spec executed in PR #3. Kept here as historical reference for the shared-edge refactor pattern subsequent asset classes can copy.

### Goal

Add `options` as the third asset class in the sidebar selector with placeholder content. Establish `src/options/` skeleton and ownership boundaries. No options business logic yet — that's Section 1.

### Branch

`chris/options-phase-1-shared-edge`

### Files created

| Path | Content |
| --- | --- |
| `docs/Options_Extension_Decisions.md` | This file |
| `src/options/__init__.py` | Empty package marker |
| `tests/options/__init__.py` | Empty file |
| `tests/options/test_smoke.py` | Smoke test: import `src.options`, assert package loads |

### Files edited

| Path | Edit |
| --- | --- |
| `src/dashboard_app.py` | Added `Options` to the sidebar asset selector with placeholder branch in `main()`. |
| `.github/CODEOWNERS` | Added: `src/options/ @cmjteevan`, `tests/options/ @cmjteevan`, `docs/Options_Extension_Decisions.md @cmjteevan`. Also fixed misleading "most-specific match" comment to "last-match-wins". |
| `src/data_source.py` | Added `"options"` to `SUPPORTED_ASSET_CLASSES` so subsequent options sections can be options-only PRs without bundling shared-file edits. |
| `docs/future_work.md` | Logged Phase 1 completion. |

### Reviewer requirements

`src/options/`, `tests/options/`, `docs/Options_Extension_Decisions.md` self-merged under Chris's ownership. Edits to `src/dashboard_app.py`, `src/data_source.py`, and `.github/CODEOWNERS` were shared-file changes — Mike's approval required per the multi-contributor workflow.

---

## Appendix B — Working agreements carried forward

From the handoff doc, applicable to any chat session working on this module:

- Honest assessments. Push back when something's off.
- Coherent process — one step at a time. Don't dump multiple steps in one message.
- Bundle questions, cap 3 per turn, use pop-ups + prose.
- All design decisions stay in chat. Claude Code executes specs only.
- One Claude Code session per repo checkout. Wait for Claude Code to reach a clean stopping point before switching branches in another shell.
- Validate URLs and version pins before sharing — software versions and API tiers go stale.
- truststore in any HTTPS fetcher (Norton 360 TLS inspection) — landed in main as of Section 2.
- Concentration analysis on every promoted study (the NVDA/META template).
- Don't be falsely definitive. Research before recommending. If a real tradeoff exists, surface alternatives transparently.
- Spec hand-offs to Claude Code should be short and reference the design memo, not re-paste it.
- Don't generate filler — commit messages, PR bodies, and Claude Code prompts should be exact text Chris can paste.

---

## Appendix E — Section 5 spec (Options BacktestConfig + FeeModel)

This is the spec for the Section 5 PR. Self-contained — Claude Code should execute it without further chat context. Mirrors the Section 1/2/3/4 spec shapes.

### Goal

Land the `BacktestConfig` and `FeeModel` dataclasses that bundle all study parameters into immutable, serializable objects. Section 6 (engine) consumes a `BacktestConfig` directly. Section 7 (Optuna runner) creates `BacktestConfig` instances per trial via the `suggest` classmethod. No engine logic, no walk loop — those are Section 6.

### Branch

`chris/options-section-5-backtest-config`

### Files to create

| Path | Content |
| --- | --- |
| `src/options/backtest_config.py` | `FeeModel`, `BacktestConfig` dataclasses + `DEFAULT_UNIVERSE` constant. See "Public API" below. |
| `tests/options/test_backtest_config.py` | Comprehensive offline tests. See "Test scope" below. |

### Files to edit

| Path | Edit |
| --- | --- |
| `docs/Options_Extension_Decisions.md` | Apply §9 row 5 / §3 / §10 deltas above. |
| `docs/future_work.md` | Append: "Options Section 5 (BacktestConfig + FeeModel) merged on `<merge date>`." under the Section 4 entry. |

No edits to `src/dashboard_app.py`, `src/data_source.py`, `src/snapshot_for_cloud.py`, `.github/CODEOWNERS`, or `requirements.txt`. Section 5 is options-only — self-merges under Chris's CODEOWNERS rule.

### Public API — `src/options/backtest_config.py`

#### Module-level constants

```python
DEFAULT_UNIVERSE: tuple[str, ...] = (
    "SPX", "SPY", "QQQ",
    "AAPL", "JPM", "MSFT", "NVDA", "XOM",
)

VALID_STRATEGY_CLASSES: frozenset[str] = frozenset({"covered_call", "cash_secured_put"})
```

#### `FeeModel`

```python
@dataclass(frozen=True, slots=True)
class FeeModel:
    """Per-contract fee model. v1 defaults match Tradier Lite plan + 2026 regulatory pass-throughs."""
    broker_fee_per_contract: float = 0.35      # Tradier Lite plan one-way
    regulatory_fee_per_contract: float = 0.10  # Combined OCC + ORF + TAF estimate, May 2026
```

Validation in `__post_init__`:
- `broker_fee_per_contract >= 0` (raise `ValueError` otherwise)
- `regulatory_fee_per_contract >= 0`

Methods:
```python
def total_per_contract_one_way(self) -> float:
    return self.broker_fee_per_contract + self.regulatory_fee_per_contract

def compute_fee(self, num_contracts: int, *, round_trip: bool = True) -> float:
    """Total fee for opening (or opening+closing if round_trip) a position of N contracts."""
    multiplier = 2 if round_trip else 1
    return num_contracts * self.total_per_contract_one_way() * multiplier
```

#### `BacktestConfig`

```python
@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Immutable container for all study parameters. Section 6 engine consumes; Section 7 Optuna runner constructs via suggest()."""

    # --- identity / scope ---
    study_label: str
    strategy_class: str  # "covered_call" or "cash_secured_put"
    universe: tuple[str, ...]

    # --- backtest window ---
    start_date: date
    end_date: date
    train_val_split_date: date

    # --- entry levers ---
    dte_target: int
    strike_selector_target_delta: float
    max_concurrent_positions: int
    earnings_window_avoid: bool
    max_loss_pct_of_portfolio: float

    # --- exit levers (composed) ---
    exit_rules: ExitRules  # imported from src.options.positions

    # --- cost model ---
    fees: FeeModel

    # --- discipline ---
    promotable: bool = False
    random_seed: int | None = None
```

Validation in `__post_init__`:
- `study_label` non-empty
- `strategy_class in VALID_STRATEGY_CLASSES`
- `universe` non-empty (at least one ticker)
- `end_date > start_date`
- `start_date < train_val_split_date < end_date`
- `dte_target in range [10, 90]` (sanity bounds; not the search range — that's wider in `suggest()`)

Search ranges should be a strict subset of validation ranges. dte_target search range in suggest() is `[25, 50]`, validation is `[10, 90]`, so suggest values always pass validation. Same pattern for the other fields:

- `strike_selector_target_delta in (0.0, 1.0)` (validation); suggest range `[0.15, 0.40]`
- `max_concurrent_positions >= 1` (validation); suggest range `[3, 10]`
- `max_loss_pct_of_portfolio in (0.0, 0.20)` (validation; max 20% per position is the absolute ceiling); suggest range `[0.01, 0.04]`

Methods:

```python
@classmethod
def suggest(
    cls,
    trial,  # optuna.Trial
    *,
    study_label: str,
    strategy_class: str,
    start_date: date,
    end_date: date,
    train_val_split_date: date,
    universe: tuple[str, ...] | None = None,
    fees: FeeModel | None = None,
    promotable: bool = False,
    random_seed: int | None = None,
) -> "BacktestConfig":
    """Construct a BacktestConfig from an Optuna trial. Search ranges live here.

    Fixed values (study_label, strategy_class, universe, dates, fees) come from kwargs.
    Tunable parameters (entry levers + exit rules) are sampled from the trial.
    """
    return cls(
        study_label=study_label,
        strategy_class=strategy_class,
        universe=universe or DEFAULT_UNIVERSE,
        start_date=start_date,
        end_date=end_date,
        train_val_split_date=train_val_split_date,
        dte_target=trial.suggest_int("dte_target", 25, 50),
        strike_selector_target_delta=trial.suggest_float(
            "strike_selector_target_delta", 0.15, 0.40
        ),
        max_concurrent_positions=trial.suggest_int("max_concurrent_positions", 3, 10),
        earnings_window_avoid=trial.suggest_categorical(
            "earnings_window_avoid", [True, False]
        ),
        max_loss_pct_of_portfolio=trial.suggest_float(
            "max_loss_pct_of_portfolio", 0.01, 0.04
        ),
        exit_rules=ExitRules(
            profit_target_pct=trial.suggest_float("profit_target_pct", 0.25, 0.80),
            time_stop_dte=trial.suggest_int("time_stop_dte", 7, 28),
            stop_loss_pct=trial.suggest_float("stop_loss_pct", 1.5, 3.5),
        ),
        fees=fees or FeeModel(),
        promotable=promotable,
        random_seed=random_seed,
    )
```

```python
def to_dict(self) -> dict:
    """Serialize to a JSON-safe dict for snapshot reproducibility.

    Dates → ISO strings. Embedded ExitRules and FeeModel flattened recursively.
    """
    # use dataclasses.asdict + post-process date fields to isoformat
```

```python
@classmethod
def from_dict(cls, data: dict) -> "BacktestConfig":
    """Reverse of to_dict. Reconstructs nested ExitRules and FeeModel."""
```

```python
def evolve(self, **changes) -> "BacktestConfig":
    """Return new BacktestConfig with given fields replaced. Wraps dataclasses.replace.

    Useful for studies that want to run multiple variants (e.g., concentration analysis
    where universe is replaced with a subset).
    """
    return dataclasses.replace(self, **changes)
```

Note on `evolve` semantics: when changing `exit_rules` or `fees`, callers pass a fully-constructed replacement (e.g., `cfg.evolve(exit_rules=ExitRules(...))`). No deep-merge logic.

#### Exports

```python
__all__ = ["FeeModel", "BacktestConfig", "DEFAULT_UNIVERSE", "VALID_STRATEGY_CLASSES"]
```

### Test scope — `tests/options/test_backtest_config.py`

All offline. Uses `unittest.mock.MagicMock` for the Optuna trial in `suggest()` tests. No real Optuna dependency at test time.

#### FeeModel

- `test_fee_model_defaults` — FeeModel() has broker=0.35, regulatory=0.10
- `test_fee_model_validates_negative_broker_raises`
- `test_fee_model_validates_negative_regulatory_raises`
- `test_fee_model_total_per_contract_one_way`
- `test_fee_model_compute_fee_round_trip` — 1 contract round-trip = 2 × 0.45 = 0.90
- `test_fee_model_compute_fee_one_way` — 1 contract one-way = 0.45
- `test_fee_model_compute_fee_multiple_contracts`

#### BacktestConfig — construction and validation

- `test_backtest_config_minimal_construction` — all required fields, defaults for optional
- `test_backtest_config_default_universe_eight_names`
- `test_backtest_config_promotable_defaults_false`
- `test_backtest_config_random_seed_defaults_none`
- `test_backtest_config_empty_study_label_raises`
- `test_backtest_config_invalid_strategy_class_raises`
- `test_backtest_config_empty_universe_raises`
- `test_backtest_config_end_before_start_raises`
- `test_backtest_config_split_outside_window_raises`
- `test_backtest_config_dte_target_out_of_range_raises` (e.g., 5 or 100)
- `test_backtest_config_delta_out_of_range_raises` (e.g., 0 or 1.0)
- `test_backtest_config_max_concurrent_zero_raises`
- `test_backtest_config_max_loss_pct_too_large_raises` (e.g., 0.25)
- `test_backtest_config_embeds_exit_rules`
- `test_backtest_config_embeds_fee_model`

#### `suggest()` classmethod

- `test_suggest_constructs_valid_config` — mock trial returns sensible values for each parameter, suggest() returns BacktestConfig that passes validation
- `test_suggest_uses_default_universe_when_none_given`
- `test_suggest_uses_provided_universe_when_given`
- `test_suggest_uses_default_fees_when_none_given`
- `test_suggest_passes_through_promotable_and_seed`
- `test_suggest_calls_trial_with_expected_parameter_names` — verifies `dte_target`, `strike_selector_target_delta`, etc. are the names registered with Optuna (matters for resuming studies)
- `test_suggest_search_ranges_match_spec` — verifies the suggest_int/suggest_float bounds exactly match the spec (e.g., dte_target is `(25, 50)` not `(20, 60)`)

#### Serialization

- `test_to_dict_round_trip_via_from_dict` — `cfg == BacktestConfig.from_dict(cfg.to_dict())`
- `test_to_dict_dates_are_iso_strings` — start_date, end_date, train_val_split_date serialize as `"YYYY-MM-DD"`
- `test_to_dict_universe_is_list` — JSON doesn't support tuples; tuples serialize as lists, deserialize back to tuples
- `test_to_dict_nested_exit_rules_roundtrip`
- `test_to_dict_nested_fee_model_roundtrip`

#### `evolve()`

- `test_evolve_returns_new_instance` — original unchanged
- `test_evolve_changes_universe_for_concentration_analysis` — typical use case
- `test_evolve_changes_exit_rules_atomically` — passes a fully-constructed replacement ExitRules
- `test_evolve_invalid_change_raises` — e.g., evolve with end_date before start_date raises via __post_init__

### Verification

Before opening the PR, all of these must pass on Chris's machine:

```powershell
cd "C:\Users\cteev\AI Projects\paper_trader_dashboard"
.\venv\Scripts\activate

# All options tests pass
pytest tests\options\ -v

# Crypto + equity unaffected
pytest tests\crypto\ -v
pytest tests\unit -v
```

No live smoke needed — Section 5 has no network or external dependencies.

### Reviewer requirements

All paths Chris-owned per CODEOWNERS overrides: `src/options/`, `tests/options/`, `docs/Options_Extension_Decisions.md`, `docs/future_work.md`. Self-merge — no Mike approval required.

### Out of scope for Section 5

- Daily walk loop, position management orchestration — Section 6 (engine).
- Optuna study creation, trial execution, results storage — Section 7 (runner).
- Spawned equity-position tracking from `state=ASSIGNED` (BacktestConfig has no concept of stock-leg holdings; that's engine state, not config).
- Universe-filter implementation (the v2 path — Section 1's stubs reserve the seam).
- Kelly / CVaR-aware position sizing — v1.1+ per §10.
- Walk-forward validation (multiple train/val splits per config) — v1.1+.
- Multi-strategy-class composite configs — v1 runs CSP and CC as separate studies; combined sizing is v1.2+.
- Per-underlying parameter overrides (e.g., different DTE for SPX vs equities) — v1.1+ if Section 8 surfaces a need.

---

## Appendix F — Section 6 spec (Backtest Engine)

This is the spec for the Section 6 PR. The largest section in Phase 2 — orchestrates everything Sections 1–5 built. Self-contained, no further chat context needed.

### Section 5 amendment

Add `starting_capital: float` field to `BacktestConfig` so studies can vary it for sensitivity analysis and so it gets snapshotted with the config for reproducibility.

| Path | Edit |
| --- | --- |
| `src/options/backtest_config.py` | Add `starting_capital: float` field after `random_seed` (default 100_000.0). Validate `> 0` in `__post_init__`. Add `starting_capital` kwarg to `suggest()` classmethod (passed through, not optimized — it's a fixed study parameter, not a search variable). Update `to_dict()` / `from_dict()` to handle the field. |
| `tests/options/test_backtest_config.py` | Add tests: `test_starting_capital_default_100k`, `test_starting_capital_zero_raises`, `test_starting_capital_negative_raises`, `test_starting_capital_round_trip_via_dict`, `test_suggest_passes_starting_capital_through`. |

This amendment lands in the Section 6 PR, not as a separate PR. The dependency is one-directional — the engine needs the field, the field is small.

The Section 6 implementation also adds `assumed_spread_pct: float = 0.05` to `BacktestConfig` — Tradier history is OHLCV-only, so the spec's "mid − half-spread / mid + half-spread" fill model needs a configurable spread input rather than reading bid/ask from the data. Validate `0.0 ≤ assumed_spread_pct < 1.0` in `__post_init__`.

### Goal

Land the backtest engine: `run_backtest(config) -> StudyResults` that simulates the active-management premium-collection strategy day-by-day over a historical window. Includes mutable PortfolioState management, NYSE trading calendar walking, chain reconstruction, strike selection, entry/exit evaluation, expiration handling, P&L roll-up, parquet persistence. Section 7 (Optuna runner) calls `run_backtest()` per trial; Section 9 (dashboard) reads the persisted results.

### Branch

`chris/options-section-6-engine`

### Files to create

| Path | Content |
| --- | --- |
| `src/options/engine.py` | `PortfolioState`, `DailySnapshot`, `StudyResults`, `run_backtest`. Daily-loop logic. |
| `src/options/chain_reconstruction.py` | `reconstruct_chain`, `select_strike`, `get_strike_spacing`, OCC enumeration helpers. |
| `src/options/earnings.py` | `fetch_earnings_calendar`, `is_in_earnings_window`. Tradier corporate calendar fetcher with caching. |
| `scripts/run_options_backtest.py` | CLI wrapper: parses `--config-path` (JSON), constructs `BacktestConfig`, runs, saves results. |
| `tests/options/test_engine.py` | End-to-end engine tests with mocked Tradier. |
| `tests/options/test_chain_reconstruction.py` | Chain enumeration + strike selection tests. |
| `tests/options/test_earnings.py` | Earnings calendar fetcher + caching tests. |

### Files to edit

| Path | Edit |
| --- | --- |
| `src/options/backtest_config.py` | Section 5 amendment: add `starting_capital` and `assumed_spread_pct` fields. |
| `tests/options/test_backtest_config.py` | Section 5 amendment tests. |
| `requirements.txt` | Add `pandas_market_calendars` if not already present. **Shared-file edit — Mike's approval required.** |
| `docs/Options_Extension_Decisions.md` | Apply §9 row 6 / §3 / §8 / §10 deltas above. |
| `docs/future_work.md` | Append: "Options Section 6 (backtest engine) merged on `<merge date>`." |

### Public API — `src/options/engine.py`

#### `DailySnapshot` (frozen + slots)

```python
@dataclass(frozen=True, slots=True)
class DailySnapshot:
    sim_date: date
    train_val_label: str  # "train" or "val"
    cash: float
    stock_value: float          # for CC: shares × close price; for CSP: 0
    open_positions_count: int
    open_positions_mark: float  # sum of mark-to-mid for open positions
    realized_pnl_to_date: float
    portfolio_total: float      # cash + stock_value + open_positions_mark
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta_per_day: float
    portfolio_vega_per_pct: float
```

#### `PortfolioState` (mutable, slots only — no frozen)

`cash`, `stock_holdings: dict[str, int]`, `open_positions: list[Position]`, `closed_positions: list[Position]`, `daily_snapshots: list[DailySnapshot]`, `skip_counters: dict[str, int]`, plus `pending_share_liquidations` and `pending_share_acquisitions` as lists of `(ticker, shares, trigger_date)` tuples. Methods: `total_value(market)`, `increment_skip(reason)`, `record_snapshot(...)`.

#### `StudyResults` (frozen + slots)

`config: BacktestConfig`, `daily_snapshots: tuple[DailySnapshot, ...]`, `closed_positions: tuple[Position, ...]`, `skip_counters: dict[str, int]`, `wall_time_seconds: float`, `run_id: str`. `to_parquet(output_dir)` writes `daily.parquet`, `trades.parquet`, `config.json`, `run_meta.json`. `from_parquet` is its inverse.

#### Main entry point

`run_backtest(config: BacktestConfig) -> StudyResults` — runs from `config.start_date` to `config.end_date`. Mutable `PortfolioState` updated in-place each simulated day. Daily 7-step sequence (build market, evaluate exits, handle expirations, process pending share liquidations, process pending share acquisitions, evaluate entries, record snapshot).

### Public API — `src/options/chain_reconstruction.py`

`get_strike_spacing(underlying, spot) -> float` — SPX $5; equities $1 below $25, $2.50 in [25,200), $5 above $200.

`reconstruct_chain(underlying, sim_date, target_expiration, spot, *, width_pct=0.20, fetcher=None) -> list[tuple[ContractSpec, float]]` — enumerate candidate OCC symbols within ±width_pct of spot, fetch each, return non-empty as `(ContractSpec, close_price)`.

`select_strike(candidates, target_delta, option_type, spot, sim_date, r, q, *, delta_tolerance=0.10) -> ContractSpec | None` — caller passes magnitude, function flips sign per option_type. Returns the candidate whose computed delta (via Section 3 `implied_vol` then `delta`) is closest to target.

### Public API — `src/options/earnings.py`

`fetch_earnings_calendar(ticker, *, fetcher=None) -> tuple[date, ...]` — Tradier corporate calendar endpoint, parquet cache at `models/cache/options/tradier/earnings/<ticker>.parquet` with 7-day TTL. Empty tuple for indexes.

`is_in_earnings_window(ticker, sim_date, *, window_days=5, earnings_dates=None) -> bool` — True if sim_date within ±window_days of any earnings date. Pre-fetched dates kwarg avoids redundant calendar lookups in the engine loop.

### Daily-loop semantics

Per-day sequence:

1. **Build market state** for each underlying (close from `fetch_history`; skip with `missing_underlying_close` if empty).
2. **Evaluate exits** on open positions; close at mid + half-spread; apply one-way fee; update cash; move to closed_positions with `state=CLOSED_MANAGED`.
3. **Handle expirations** via `Position.resolve_expiration`. ASSIGNED short put → queue `pending_share_liquidations`. ASSIGNED short call → queue `pending_share_acquisitions` for next-day re-buy. EXPIRED_ITM (SPX) → cash credit at intrinsic. EXPIRED_OTM → no further action.
4. **Process pending share liquidations** at next-day close; record as `spawned_equity_close` synthetic position in closed list.
5. **Process pending share acquisitions** (CC re-buy) when cash permits; otherwise increment `insufficient_cash_for_cc_rebuy`.
6. **Evaluate entries** for each eligible underlying:
   - Skip with `existing_position_same_strategy_class`, `earnings_window`, `max_concurrent_reached`, `no_shares_for_cc` as appropriate
   - Compute target_expiration (nearest available ≥ `sim_date + dte_target`)
   - Reconstruct chain, select strike, size by `max_loss_pct_of_portfolio` (CSP: strike × 100; CC: spot × 100)
   - Skip with `insufficient_cash_for_position` if cash short; otherwise open via `Position.cash_secured_put()` / `Position.covered_call()` at mid − half-spread, apply one-way fee
7. **Record daily snapshot** tagged `train`/`val` per `train_val_split_date`.

### Strategy mode initialization

CC: equal-capital allocation per slot. `slot_capital = starting_capital / max_concurrent_positions`. For each slot, walk universe in order; buy `floor(slot_capital / (close × 100)) × 100` shares at first-trading-day close; if 0, leave slot empty (CC re-buy logic later fills it when cash recovers). CSP: no initialization, all cash.

### Verification

```powershell
pytest tests/ -v
python -c "import pandas_market_calendars; print(pandas_market_calendars.__version__)"
```

### Reviewer requirements

`src/options/`, `tests/options/`, `scripts/run_options_backtest.py`, `docs/Options_Extension_Decisions.md`, `docs/future_work.md` self-merge. `requirements.txt` is the only shared-file edit (if `pandas_market_calendars` needs adding). Mike's approval required for that one line.

### Out of scope for Section 6

Optuna parameter sweeps (Section 7); concentration analysis (Section 8); walk-forward validation (v1.1+); margin-aware sizing (v1.1+); real-time / paper-trade mode (v2+); live order routing (v2+); multi-strategy-class composite engines (v1.2+); per-underlying parameter overrides (v1.1+); performance optimization beyond cache reuse (v1.1+).

---

## Appendix G — Section 7 spec (Optuna runner + smoke study)

This is the spec for the Section 7 PR. Self-contained.

### Goal

Land the Optuna optimization runner: `run_optuna_study()` wraps the Section 6 engine in an Optuna TPE objective, persists top-K trial results, supports resume. Plus a smoke-study CLI that exercises the full v1 universe and both strategy classes against real Tradier sandbox data over a 6-month compressed window — validates Sections 1-6 work correctly end-to-end before Section 8 runs the real study.

### Branch

`chris/options-section-7-optuna-runner`

### Files to create

| Path | Content |
| --- | --- |
| `src/options/optuna_runner.py` | `calmar_objective`, `OptunaStudyResults`, `run_optuna_study`, top-K trial tracking. |
| `scripts/run_options_optuna_study.py` | CLI for arbitrary studies (used by Section 8 too). |
| `scripts/run_options_smoke_study.py` | CLI wrapper for the compressed-full smoke (CSP + CC). |
| `tests/options/test_optuna_runner.py` | Optuna runner tests with mocked engine deps. |
| `tests/options/test_calmar_objective.py` | Calmar objective math tests. |

### Files to edit

| Path | Edit |
| --- | --- |
| `docs/Options_Extension_Decisions.md` | Apply §9 row 7 / §3 / §10 deltas above. |
| `docs/future_work.md` | Append: "Options Section 7 (Optuna runner + smoke study) merged on `<merge date>`." |

No shared-file edits. Optuna is already a transitive dependency.

### Public API — `src/options/optuna_runner.py`

#### `calmar_objective(results: StudyResults) -> float`

Computes Calmar = annualized compound return / max drawdown on training-window snapshots only. Edge cases:
- Empty training data → 0.0
- Training window < 30 days → 0.0
- Initial portfolio value ≤ 0 → 0.0
- Zero drawdown with positive return → 1e9 sentinel
- Complete wipeout (final ≤ 0) → -1.0 from the compound-return path
- Failed trial → caller catches and returns -1.0 directly

#### `OptunaStudyResults` (frozen + slots)

`study_label`, `strategy_class`, `n_trials_run`, `n_trials_failed`, `best_value`, `best_trial_number`, `best_params`, `top_k_trial_numbers`, `wall_time_seconds`, `storage_path: Path`, `output_dir: Path`. `to_json(path)` and `from_json(path)` for round-trip; Path objects serialize as strings.

#### `run_optuna_study(...) -> OptunaStudyResults`

Constructs an Optuna TPE study at `models/cache/options/optuna_studies.db` (configurable via `storage_path`). Each trial calls `BacktestConfig.suggest()` to construct a config, runs `run_backtest`, computes Calmar. Top-K trial outputs persisted to `<output_dir>/trial_<NNNN>/`. Resume capability via `load_if_exists=True`. Failed trials log and return -1.0 sentinel; failure count surfaced in `n_trials_failed`.

### CLIs

- `scripts/run_options_optuna_study.py` — flexible CLI for arbitrary studies (Section 8 uses this too).
- `scripts/run_options_smoke_study.py` — locked compressed-full smoke configuration: 8 v1 underlyings × 2 strategies × 5 trials × 6 months (2024-01-02 to 2024-07-01, split 2024-05-01), `promotable=False`.

### Test scope

`test_calmar_objective.py`: synthetic StudyResults, no engine, no Optuna. Cover positive return with drawdown, zero-drawdown sentinels, validation exclusion, short-window guard, defensive zero-handling.

`test_optuna_runner.py`: mocked EngineDeps, small `n_trials`. Cover trial completion, top-K persistence, resume from existing storage, failed-trial sentinel, OptunaStudyResults JSON round-trip.

Smoke study test: only assert constants match the locked config (`SMOKE_START_DATE`, `SMOKE_TRIALS_PER_STRATEGY`, etc.). The full smoke is validated via manual run, not unit test.

### Verification

```powershell
pytest tests/ -v
# Manual smoke (post-merge, ~30min, network-dependent):
# venv\Scripts\python.exe scripts\run_options_smoke_study.py
```

### Reviewer requirements

All paths Chris-owned per CODEOWNERS overrides. Self-merge — no Mike approval required.

### Out of scope for Section 7

- Real (non-smoke) study execution — Section 8.
- Concentration analysis orchestration — Section 8 calls `run_optuna_study` multiple times with `cfg.evolve(universe=...)`.
- Walk-forward validation, multi-objective optimization, trial pruning, distributed Optuna, per-trial intermediate reporting — v1.1+.

---

## Appendix H — Section 8 spec (Production v1 study)

This is the spec for the Section 8 PR. Self-contained.

### Goal

Land the production v1 study orchestrator: takes the locked Light scope (100 trials × 2 strategies × ~2-year window), runs the primary Optuna studies, runs concentration analysis ablation, fetches benchmark series, evaluates the automated promotion gate, captures the human override decision. Outputs become Section 9's input.

### Branch

`chris/options-section-8-production-study`

### Files to create

| Path | Content |
| --- | --- |
| `src/options/benchmarks.py` | `fetch_spy_total_return`, `fetch_bxm` (Tradier primary, yfinance fallback). |
| `src/options/concentration.py` | Concentration ablation orchestrator. |
| `src/options/promotion.py` | `evaluate_promotion()` automated gate, `promotion_decision.json` schema. |
| `src/options/v1_study.py` | Top-level `run_v1_study()` orchestrator wrapping Optuna runs + concentration + benchmarks + promotion. |
| `scripts/run_options_v1_study.py` | Production CLI. |
| `tests/options/test_benchmarks.py` | Mocked fetcher tests. |
| `tests/options/test_concentration.py` | Concentration orchestration tests. |
| `tests/options/test_promotion.py` | Automated gate logic tests. |
| `tests/options/test_v1_study.py` | Top-level orchestrator integration tests with all Tradier deps mocked. |

### Files to edit

| Path | Edit |
| --- | --- |
| `src/options/tradier.py` | Add `fetch_index_quote_history(symbol, start, end)` for index history (e.g., `$BXM`). Wraps `/markets/history` with index-symbol prefix. **Section 2 amendment.** |
| `src/options/engine.py` | Add `entry_filters: EntryFilters \| None = None` parameter to `run_backtest`. Add `EntryFilters` frozen dataclass with `dte_exclude_range` and `iv_regime_exclude` fields. Add `fetch_iv_regime` callable to `EngineDeps` (default impl computes RV-percentile against 252-day rolling distribution). Engine reads filters during entry evaluation. **Section 6 amendment.** |
| `tests/options/test_tradier.py` | Add tests for `fetch_index_quote_history`. |
| `tests/options/test_engine.py` | Add tests for `entry_filters` (DTE band exclusion + IV regime exclusion). |
| `requirements.txt` | If `yfinance` is not already pinned, add it. **Verified: yfinance==1.2.0 is already in requirements.txt via the equity infrastructure — no shared-file edit needed.** |
| `docs/Options_Extension_Decisions.md` | Apply §9 row 8 / §3 / §10 deltas above. |
| `docs/future_work.md` | Append: "Options Section 8 (production v1 study) merged on `<merge date>`." |

### Public API summary

- `src/options/benchmarks.py`: `fetch_spy_total_return(start, end, *, fetcher=None)` returns DataFrame with `close`, `dividend_per_share`, `total_return_index` columns. `fetch_bxm(start, end, *, fetcher=None)` returns DataFrame with `close` column; tries Tradier index history first, falls back to yfinance `^BXM`. Both cached at `models/cache/options/benchmarks/` with 7-day TTL.

- `src/options/concentration.py`: `ConcentrationResult` (frozen + slots) records `ablation_dimension`, `ablation_value`, `base_calmar`, `ablated_calmar`, `delta_calmar`, `pct_alpha_attribution`. `run_concentration_analysis(...)` runs three ablation dimensions (per-underlying, per-DTE-band [25-30, 30-35, 35-40, 40-45, 45-50], per-IV-regime [high, low]) and returns a flat tuple of results.

- `src/options/promotion.py`: `PromotionCheck` (per criterion), `PromotionRecommendation` (aggregate, with `to_dict`/`from_dict`). `evaluate_promotion(...)` runs five hardcoded checks (overfit, beats_spy, beats_bxm for CC only, no_underlying_concentration, regime_independence). `write_promotion_decision(output_dir, recommendation, human_override=None)` persists `promotion_decision.json` with optional human override section.

- `src/options/v1_study.py`: `run_v1_study(*, run_id, ...)` returns `dict[str, Path]` with paths to per-strategy output dirs and the snapshot dir. Sequence: fetch benchmarks → for each strategy (CSP, CC) run primary Optuna + concentration + promotion gate + decision file → snapshot the run.

### Verification

```powershell
pytest tests/ -v
python -c "import yfinance; print(yfinance.__version__)"
# Manual production run (post-merge, ~6 hours):
# venv\Scripts\python.exe scripts\run_options_v1_study.py --non-interactive
```

### Reviewer requirements

Mostly Chris-owned. `src/options/tradier.py`, `src/options/engine.py`, `src/options/benchmarks.py`, `src/options/concentration.py`, `src/options/promotion.py`, `src/options/v1_study.py`, `tests/options/`, `scripts/run_options_v1_study.py`, `docs/Options_Extension_Decisions.md`, `docs/future_work.md` — all Chris-owned. yfinance is already a transitive dependency, so `requirements.txt` is not touched.

### Out of scope for Section 8

- Walk-forward validation (multiple train/val splits) — v1.1+.
- Multi-objective optimization — v1.1+.
- Live BXM real-time updates — v1.1+ if dashboard wants live-updating benchmarks.
- Cross-strategy concentration analysis — v1.2+.
- Bayesian model averaging across promoted studies — v2+.
- Automated re-promotion on data refresh — v2+. v1 is single-run-and-promote-once.
- Real-money trading hooks — v2+.

---

## Appendix I — Section 2.5 spec (Polygon historical fetcher)

This is the spec for the Section 2.5 PR. Self-contained.

### Goal

Land `src/options/polygon.py` implementing the same `HistoryFetcher` protocol as Section 2's Tradier client, but backed by Polygon.io / Massive.com. Switch `chain_reconstruction.py`'s default fetcher from Tradier to Polygon. Update Section 8's default study window to 2023-01-02 (Polygon's enforced floor). Section 2.5 unblocks the v1 production study by replacing the historical-data backend that Tradier could not provide.

### Branch

`chris/options-section-2-5-polygon-fetcher`

### Files to create

| Path | Content |
| --- | --- |
| `src/options/polygon.py` | `fetch_history`, OCC-to-Polygon-ticker conversion, parquet caching, error handling. Mirrors `src/options/tradier.py` shape and discipline. |
| `tests/options/test_polygon.py` | Comprehensive offline tests with mocked HTTP. |

### Files to edit

| Path | Edit |
| --- | --- |
| `src/options/chain_reconstruction.py` | Switch default fetcher from `src.options.tradier.fetch_history` to `src.options.polygon.fetch_history`. Tradier import retained for the `RateLimiter` type. |
| `src/options/tradier.py` | Module-level docstring note: "Live execution and current chain snapshot only — historical OHLCV is fetched via `src/options/polygon.py` since Section 2.5 (May 2026)." Public API unchanged. |
| `tests/options/test_chain_reconstruction.py` | Mocked-fetcher tests stay protocol-identical. New `test_reconstruct_chain_default_fetcher_is_polygon` confirms the swap. |
| `scripts/run_options_v1_study.py` | Default `--start-date` 2023-01-03 → 2023-01-02 (Polygon plan floor); `--train-val-split-date` 2024-12-31 → 2025-01-02; `--end-date` 2025-12-31 → 2026-05-08. |
| `scripts/run_options_smoke_study.py` | Smoke-window dates retained if within Polygon coverage; otherwise updated. |
| `docs/Options_Extension_Decisions.md` | Apply §3 / §9 / §10 / §8 deltas above. |
| `docs/future_work.md` | Append: "Options Section 2.5 (Polygon historical fetcher) merged on `<merge date>`." |

No `requirements.txt` edit needed. `requests` is already a dependency. Raw HTTP rather than the `polygon-api-client` SDK keeps truststore + Norton 360 TLS handling consistent with Section 2's pattern. **Section 2.5 self-merges under Chris's CODEOWNERS rule.**

### Public API summary — `src/options/polygon.py`

- `fetch_history(symbol, start, end, *, limiter=None, session=None, use_cache=True) -> pd.DataFrame` — daily OHLCV mirroring Tradier's signature; OCC symbol auto-converted to Polygon ticker (`O:` prefix, padding stripped). Returns empty DataFrame for valid-but-untraded contracts. Cache path `models/cache/options/polygon/history/<symbol>.parquet`, sanity-gated.
- Errors: 403 NOT_AUTHORIZED → `RuntimeError("Polygon NOT_AUTHORIZED ...")`. 401 → `RuntimeError("Polygon authentication failed ...")`. 5xx / network → retry with exponential backoff up to `DEFAULT_MAX_RETRIES`, then re-raise. 429 retried with backoff.
- `_occ_to_polygon_ticker(occ)` — strips internal whitespace, prefixes `O:`. Idempotent on already-clean symbols.
- `_resolve_token()` — reads `POLYGON_API_KEY` from env; clear error message on missing.

### Test scope

Comprehensive offline tests in `tests/options/test_polygon.py` cover OCC conversion, token resolution, success / empty / 401 / 403 / 5xx / 429 / network retry paths, cache write / read / sanity gate, `apiKey` query parameter, optional `RateLimiter` integration. Updates to `tests/options/test_chain_reconstruction.py` add a default-fetcher assertion.

### Verification

```powershell
pytest tests/ -v

# Mandatory live Polygon smoke (memo §8 discipline):
venv\Scripts\python.exe -c @"
from datetime import date
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path.cwd() / '.env')
from src.options.polygon import fetch_history
df = fetch_history('SPY240719C00540000', date(2024, 6, 17), date(2024, 6, 21))
print(f'Rows: {len(df)}')
print(df)
'@
```

### Reviewer requirements

All paths Chris-owned per CODEOWNERS overrides: `src/options/`, `tests/options/`, `scripts/`, `docs/Options_Extension_Decisions.md`, `docs/future_work.md`. **Self-merge — no Mike approval required.** No `requirements.txt` edit (raw `requests` only).

### Out of scope for Section 2.5

- Migrating `src/options/tradier.py` to deprecated/removed status — file stays on main, retained for v2+ live execution. Module docstring update only.
- Polygon's other endpoints (snapshot quotes, reference data, real-time WebSockets) — v2+ if needed.
- `polygon-api-client` SDK migration — raw `requests` is sufficient. Revisit if SDK offers material advantages.
- Polygon-based BXM / SPY benchmark fetch — current code paths fine.
- Polygon flat-file CSV bulk download — v1.1+ optimization.

---

*End of design memo. Next action: hand the Section 2.5 spec to Claude Code.*
