# Options Extension Decisions

| Field | Value |
| --- | --- |
| Document version | v1.0 — May 9, 2026 |
| Author | Chris Teevan |
| Repo path | `docs/Options_Extension_Decisions.md` |
| Status | Design phase complete. Phase 1 (shared-edge refactor) ready to ship. Sections 1–9 specced. |
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
| Underlying universe | SPX, SPY, QQQ + curated equity subset (~5–10 names initial, expand to ~20 after Section 8) | Indexes for clean data and high liquidity. Equity subset for diversification. Liquidity floor much harder than equity baseline (options-grade liquidity, not just stock liquidity). |
| Primary benchmark | SPY total return | Strategy-relevant baseline. The dashboard reports vs SPY total return for comparability with Mike's equity studies. |
| Secondary benchmark | CBOE BuyWrite Index (BXM) | Strategy-class-specific honesty check for covered call studies. Surfaces whether the active-management edge is real or just a beta repackaging. |
| Python version | 3.11.9 | Match Mike's documented convention. |
| TLS handling | `truststore` package, injected at script start | Carries the crypto lesson forward. Norton 360 TLS inspection breaks certifi-based requests. truststore reads from Windows trust store. Land this in main with Section 2. |
| Sentiment / macro signals | Not in v1 architecture | Crypto needs sentiment because it's reflexive. Options have natural macro hooks (VIX, term structure, skew) — but these are options-derived signals, so wiring them into option strategies is recursive. Defer to v1.1. |
| v1 publish bar | Light: single Optuna run, train/val window split, SPY + BXM benchmarks, one promoted study | Mirror crypto v1 publish bar. Walk-forward, multi-regime studies, vol-of-vol modeling are v1.1+. |
| Snapshot for v1 study | `pre_options_v1_<date>` under `models/snapshots/options/` | Locks the data inputs at promotion. Reproducibility guarantee carries from equities. |

---

## 4. Underlying universe

### Initial universe (Section 1, smoke-test scope)

- **Indexes (3)**: SPX (cash-settled, AM-settled monthlies), SPY (PM-settled), QQQ (PM-settled)
- **Curated equities (~5–7 to start)**: highly liquid, weekly options available, cleanly trade through earnings or have predictable IV crush patterns. Initial candidates: AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA. Final list locked at Section 1.

### Expanded universe (after Section 8)

After v1 study lands, expand to ~20 equity names. Selection criteria for the equity subset (stricter than Mike's 491-ticker equity universe — these are *options-grade liquidity* filters):

- Minimum **average daily option volume** across the chain (specific floor TBD at Section 1; conservative starting point ~10k contracts/day across all expiries)
- Minimum **weekly options availability** for active-management thesis
- Bid-ask **spread floor** at the strike levels we'll trade (the 0.30-delta band, conservative starting point: spread < 5% of mark for monthlies)
- Coverage across sectors (avoid all-tech-mega-cap concentration, repeat the NVDA/META lesson)

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
- **Fee model (5bps)** — replaced. Options fees are per-contract: Tradier is roughly $0.35/contract one-way for options (validate at Section 2 against current Tradier fee schedule). Slippage assumption: mid-price minus half the spread on entry, mid + half-spread on exit (conservative).

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
- **Norton 360 TLS still applies.** Tradier API uses HTTPS. `truststore.inject_into_ssl()` at script start. Land truststore with Section 2.
- **OCC symbol format.** Standard 21-character format (`AAPL220617C00270000`). Parsing/generation utility at Section 1.
- **Dividend handling affects pricing.** Equity dividends affect option pricing (forward-price adjustment). Tradier provides dividend data; we use it directly rather than recomputing.
- **Holiday and early-close calendars.** Settlement on holiday-shifted expiries needs care. Use NYSE calendar (`pandas_market_calendars`).
- **Tradier sandbox is 15-min delayed without funded account.** For backtest this doesn't matter (we use historical). For paper-trade development it means a small staleness in the sandbox feed — acceptable for v1, document it.

---

## 9. Phase 2 — Section breakdown

Mirrors the crypto Phase 2 sectioning. Each section is a self-contained PR that lands cleanly without breaking previous sections. Branch naming: `chris/options-section-N-<topic>`.

| # | Section | Status | Notes |
| --- | --- | --- | --- |
| 1 | Universe + contract spec | NOT STARTED | `UnderlyingMeta` and `ContractSpec` dataclasses, OCC symbol parse/generate utility, parquet schema, static initial universe (3 indexes + 5–7 curated equities). Mirror crypto Section 1 shape. |
| 2 | Tradier OHLCV + chain fetcher | NOT STARTED | Sandbox API key, OAuth flow, rate limit handling, `truststore.inject_into_ssl()` at module init. Historical chains by OCC symbol. Sanity gate at <50% non-empty refuses cache write. Caches to `models/cache/options/tradier/`. truststore lands in main with this section — will require shared-file approval from Mike. |
| 3 | Black-Scholes Greeks module | NOT STARTED | Closed-form delta/gamma/theta/vega/rho. Pure-function module, no external state. Validate against Tradier/ORATS Greeks (sanity check, not source of truth). Tests cover ATM/OTM/ITM and near-expiration edge cases. |
| 4 | Position + lifecycle model | NOT STARTED | The new architectural element. `Position` dataclass with `strategy_class`, `legs[]`, `entry_date`, `exit_rules`, `pnl()`, `is_expired()`, `should_exit_now(market)` methods. Multi-leg atomic operations from day one. Active-management exit rules (profit target, time stop, stop-loss-on-pnl) as first-class. |
| 5 | Options BacktestConfig | NOT STARTED | Dataclass mirroring equity shape but with options fields: `dte_target`, `profit_target_pct`, `time_stop_dte`, `strategy_class` enum, position-sizing rule, fee model, slippage model, `max_concurrent_positions`, earnings-window-avoidance flag, `strike_selector_target_delta`. |
| 6 | Options backtest engine | NOT STARTED | Daily walk on NYSE trading calendar. Position management loop: evaluate exit rules on all open positions → close exits → evaluate entry rules → open new entries (subject to `max_concurrent_positions`). P&L roll-up per day. Train/val window split. |
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

---

## Appendix A — Phase 1 spec (shared-edge refactor)

This is the spec to hand to Claude Code as the first PR. It's small, scope-controlled, and mirrors what Mike did for crypto Phase 1. The goal is to land the placeholder so Section 1 (universe + contract spec) can follow as a self-contained options-only PR with no shared-file edits.

### Goal

Add `options` as the third asset class in the sidebar selector with placeholder content. Establish `src/options/` skeleton and ownership boundaries. No options business logic yet — that's Section 1.

### Branch

`chris/options-phase-1-shared-edge`

### Files to create

| Path | Content |
| --- | --- |
| `docs/Options_Extension_Decisions.md` | This file (drop in as-is) |
| `src/options/__init__.py` | Empty package marker with module docstring referencing this design doc |
| `tests/options/__init__.py` | Empty file |
| `tests/options/test_smoke.py` | Smoke test: import `src.options`, assert package loads |
| `models/cache/options/.gitkeep` | Empty file (directory marker, gitignored content) |
| `models/snapshots/options/.gitkeep` | Empty file (directory marker, gitignored content) |

### Files to edit

| Path | Edit |
| --- | --- |
| `src/dashboard_app.py` | Add `Options` to the sidebar asset selector. Add `elif asset_class == "options":` branch in `main()` rendering a placeholder identical in pattern to the crypto placeholder Mike wired in Phase 1 (e.g., "Options module — Phase 2 in progress. See `docs/Options_Extension_Decisions.md`."). |
| `.github/CODEOWNERS` | Add: `src/options/ @cmjteevan`, `tests/options/ @cmjteevan`, `docs/Options_Extension_Decisions.md @cmjteevan`. |
| `docs/future_work.md` | Append: "Options Phase 1 (shared-edge) merged on `<merge date>`. Phase 2 sections 1–9 specced in `Options_Extension_Decisions.md`." |

### Verification

Before opening the PR, all of these must pass on Chris's machine:

```powershell
cd "C:\Users\cteev\AI Projects\paper_trader_dashboard"
.\venv\Scripts\activate

# Smoke test passes
pytest tests\options\ -v

# Dashboard renders, sidebar shows Options option, selecting it shows placeholder
streamlit run src\dashboard_app.py
# (manual visual check in browser, then Ctrl+C)

# R2 dry-run succeeds (uploads nothing — no data yet)
venv\Scripts\python.exe src\snapshot_for_cloud.py --asset-class options --dry-run

# Equity dashboard verifies bit-identical to before (sanity check on the shared-file edit)
streamlit run src\dashboard_app.py
# (sidebar still shows Stocks and Crypto selecting unchanged behavior)
```

### Reviewer requirements

`src/options/`, `tests/options/`, `docs/Options_Extension_Decisions.md` self-merge under Chris's ownership. The edits to `src/dashboard_app.py` and `.github/CODEOWNERS` are shared-file changes — Mike's approval required per the multi-contributor workflow.

### Out of scope for Phase 1

- Any options business logic (universe, fetcher, Greeks, position model, engine — all Section 1+)
- `requirements.txt` updates (Section 2 adds `truststore` and the Tradier client)
- R2 bucket folder creation (`snapshot_for_cloud.py` creates prefixes lazily)

---

## Appendix B — Working agreements carried forward

From the handoff doc, applicable to any chat session working on this module:

- Honest assessments. Push back when something's off.
- Bundle questions, cap 3 per turn, use pop-ups + prose.
- All design decisions stay in chat. Claude Code executes specs only.
- One Claude Code session per repo checkout.
- Validate URLs and version pins before sharing — software versions and API tiers go stale.
- truststore in any HTTPS fetcher (Norton 360 TLS inspection).
- Concentration analysis on every promoted study (the NVDA/META template).

---

*End of design memo. Next action: hand the Phase 1 spec to Claude Code.*
