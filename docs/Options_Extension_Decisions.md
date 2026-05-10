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
| Data source | Tradier (brokerage-attached) | Free with account, backtest + paper-trade in a single integration, Greeks via ORATS bundled in chain response. Escalate to Polygon ($79/mo) or ThetaData if backtest depth becomes binding. |
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

---

## 9. Phase 2 — Section breakdown

Mirrors the crypto Phase 2 sectioning. Each section is a self-contained PR that lands cleanly without breaking previous sections. Branch naming: `chris/options-section-N-<topic>`.

| # | Section | Status | Notes |
| --- | --- | --- | --- |
| 1 | Universe + contract spec | MERGED (PR #4) | `UnderlyingMeta` (12 fields, frozen+slots; `dividend_yield` added in Section 3) and `ContractSpec` (4 fields) dataclasses, OCC symbol parse/generate utility (strict 21-char round-trip), `UNIVERSE_PARQUET_SCHEMA`, static 8-underlying universe (SPX, SPY, QQQ + AAPL, JPM, MSFT, NVDA, XOM), v1 public API + v2 stubs reserving the seam for v1.1+ filter-based expansion against Mike's equity universe. Mirrors crypto Section 1 shape. |
| 2 | Tradier OHLCV + chain fetcher | MERGED (PR #5) | Sandbox + production tokens via env vars (`TRADIER_SANDBOX_TOKEN`, `TRADIER_PRODUCTION_TOKEN`); bearer-token auth (no OAuth). Header-driven rate-limit handling (`X-Ratelimit-*`) with conservative fallback. `truststore.inject_into_ssl()` invoked by entry-point scripts via `src/options/_ssl.py` helper, mirroring crypto's pattern. Per-contract OHLCV by OCC symbol (or underlying ticker) + current-chain snapshot + expirations endpoint; historical chain enumeration is **not** offered by Tradier and is reconstructed at backtest time in Section 6 by candidate-OCC enumeration. 1-day TTL on history cache; chain snapshots immutable per `<run_date>` file. Sanity gate at <50% non-empty refuses history cache write. Caches to `models/cache/options/tradier/`. truststore landed in main with this section. Live smoke against Tradier sandbox: SPY history 20/30 days (66.7% coverage, gate pass), SPY chain 508 contracts, real cache write succeeded — truststore correctly bypasses Norton 360 TLS inspection. |
| 3 | Black-Scholes Greeks module | MERGED (PR #6) | Closed-form Black-Scholes-Merton (continuous dividend yield `q`). Pure-function module exposing `price`, `delta`, `gamma`, `theta_per_day`, `vega_per_pct`, `rho_per_bp`, `implied_vol`, plus `compute_all` returning a frozen `GreeksResult` dataclass. Trader-convention units throughout, encoded in field names so consumers don't have to remember (theta scaled to per-calendar-day, vega per 1 IV point, rho per 1 bp). Day count ACT/365 hardcoded in a `time_to_expiration` helper (basis flexibility deferred to v1.1+ if a study needs ACT/360). Caller passes `q` (SPX: 0; SPY/QQQ: distribution yield; single names: ticker-specific) — Section 1's `UnderlyingMeta` was amended with a `dividend_yield` field in the same PR so callers have a canonical lookup. American-style treated as European — early-exercise premium ignored; documented in §8. `implied_vol` solver via Brent's method ships in Section 3 because Section 6 needs it for backtest IV reconstruction (Tradier per-contract history returns OHLCV without IV). The "below intrinsic" check uses the proper European lower bound (`max(K·e^(-rT) - S·e^(-qT), 0)` for puts, call analogue for calls), not the American intrinsic — caught mid-flight; deep-ITM puts with r > q would have spuriously raised under the simpler check. Edge cases handled explicitly: `T<0`/`S<=0`/`K<=0`/`vol<0` raise; `T==0` or `vol==0` returns intrinsic + zero Greeks except delta = ±1/0 ITM indicator. Pure-math validation only: Hull reference values (`pytest.approx(abs=0.005, rel=0.001)` to absorb textbook rounding) + put-call parity + finite-difference Greeks (~1e-4 tolerance). ORATS comparison is a manual one-time post-merge sanity check via `scripts/fetch_options_chain.py`, not a permanent test. |
| 4 | Position + lifecycle model | NOT STARTED | Position dataclass (frozen + slots) representing a multi-leg options position with first-class active-management exit rules. **Hybrid representation:** canonical `legs: tuple[Leg, ...]` shape used by all engine code, plus per-strategy classmethod constructors (`Position.covered_call`, `Position.cash_secured_put`, future `Position.vertical_spread`, etc.) for self-documenting construction with validation in `__post_init__`. `Leg` carries (contract, sign, quantity); contract is `ContractSpec` (option), `StockContract` (stock), or `CashContract` (cash collateral) — discriminated union via type. Explicit cash legs on CSPs symmetric with explicit stock legs on CCs — honest portfolio accounting. `ExitRules` dataclass with hardcoded fields (`profit_target_pct`, `time_stop_dte`, `stop_loss_pct`, at least one required) — Optuna-friendly fixed parameter space. `PositionState` enum: `OPEN` → `CLOSED_MANAGED` (active-management exit) | `EXPIRED_ITM` (cash-settled, SPX) | `EXPIRED_OTM` | `ASSIGNED` (share-settled, equity options at expiration). Mark-to-mid P&L using daily chain close. Honest settlement at expiration: SPX cash-settles to intrinsic, SPY/QQQ/equities share-settle (the spawned equity position from `state=ASSIGNED` is created and managed by Section 6 engine, not in Section 4). Frozen with `evolve(**changes)` method that returns new instance via `dataclasses.replace`. Position aggregates leg-level Greeks via Section 3's `compute_all` (cash and stock legs contribute zero Greeks except stock delta). Closure reason format documented: `profit_target_<pct>`, `time_stop_<dte>`, `stop_loss_<pct>`, `expired_itm_cash_settled`, `expired_otm`, `assigned_call`, `assigned_put`. v1 ignores early assignment per §8 (avoid ex-div windows on short calls). **`entry_credit` convention (locked at Section 4 implementation):** `entry_credit` follows trader semantics — net cash received at open, positive for credits (CSP `+put_premium*100`, CC `-stock_basis*100 + call_premium*100`), negative for debits (long premium positions). `mark_to_market` returns P&L in dollars and excludes cash legs from the leg sum (cash held at par with zero yield per §8): `P&L = sum(sign × qty × multiplier × mark for non-cash legs) + entry_credit`. Stock legs **do** contribute (a CC's stock leg moves with the underlying). `should_exit` thresholds use `abs(entry_credit)` for symmetry across credit/debit positions: profit triggers when `P&L ≥ profit_target_pct × abs(entry_credit)`, stop_loss triggers when `P&L ≤ -stop_loss_pct × abs(entry_credit)`. The original Section 4 spec said "sum over legs ... minus entry_credit", which couldn't simultaneously satisfy trader-view `entry_credit` and the P&L-zero-at-open invariant once explicit cash collateral legs were introduced; the trader-view + non-cash-sum convention was chosen at implementation time. |
| 5 | Options BacktestConfig | NOT STARTED | Dataclass mirroring equity shape but with options fields: `dte_target`, `profit_target_pct`, `time_stop_dte`, `strategy_class` enum, position-sizing rule, fee model, slippage model, `max_concurrent_positions`, earnings-window-avoidance flag, `strike_selector_target_delta`. |
| 6 | Options backtest engine | NOT STARTED | Daily walk on NYSE trading calendar. Position management loop: evaluate exit rules on all open positions → close exits → evaluate entry rules → open new entries (subject to `max_concurrent_positions`). P&L roll-up per day. Train/val window split. Reads `state=ASSIGNED` from Section 4 positions and creates the resulting equity position for independent management. |
| 7 | Optuna runner + smoke study | NOT STARTED | Borrow runner skeleton from `src/optuna_runner.py`. Separate `optuna_studies.db` at `models/cache/options/`. Smoke study: tiny universe (1–2 underlyings), tight DTE range, single strategy class. `promotable: false` enforced at upload. |
| 8 | Real study (active-management CCs + CSPs) | NOT STARTED | Train/val split (windows TBD at Section 8 — likely train pre-2024, val 2024–2025). SPY total return primary benchmark, BXM secondary for CCs. Single Optuna run. Concentration analysis per Section 7 of this doc (by underlying, DTE band, IV regime, strategy variant). |
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

*End of design memo. Next action: hand the Section 4 spec to Claude Code.*
