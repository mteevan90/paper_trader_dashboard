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
| Position sizing rule | Fixed-risk: `max_loss_pct_of_portfolio` field (default 2%, search 1–4%) | Locked at Section 5. Each position sized so its theoretical max loss is at most this fraction of starting portfolio value. Kelly-style and CVaR-aware sizing deferred to v1.1+ per §10. |

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
| 5 | Options BacktestConfig | NOT STARTED | Frozen + slots `BacktestConfig` dataclass bundling all study levers into a single immutable, serializable object the engine consumes. Embedded `ExitRules` from Section 4 (composition, no field duplication). Embedded `FeeModel` dataclass with broker_fee_per_contract and regulatory_fee_per_contract broken out for fee-sensitivity analysis (Tradier Lite defaults: $0.35 broker + $0.10 regulatory; structured rather than flat-composite for honesty in study output). Universe specification as `tuple[str, ...]` field defaulting to the 8 v1 names — smoke studies override with smaller subsets, v1.1+ replaces with liquidity-filtered selection. `BacktestConfig.suggest(trial)` classmethod owns Optuna parameter ranges (cohesive: parameter definitions and search bounds in the same place). Per-strategy_class instances — CSP and CC run as separate studies and are compared at the study level, not within a single config. Train/val split via `start_date` + `end_date` + `train_val_split_date` fields; engine processes one walk and tags days by which side of split they fall on. Position sizing as fixed-risk: `max_loss_pct_of_portfolio` (default 0.02). `promotable: bool = False` flag enforced at study upload to prevent accidental publication of smoke runs. `to_dict()` / `from_dict()` for snapshot reproducibility. |
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
- **Sizing rule alternatives.** v1 uses fixed-risk sizing (`max_loss_pct_of_portfolio`). v1.1+ may evaluate Kelly-style sizing (variable based on edge estimation) and CVaR-aware sizing (scales position size by tail-risk estimate). Either requires a real-data baseline before introducing variable sizing.

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

*End of design memo. Next action: hand the Section 5 spec to Claude Code.*
