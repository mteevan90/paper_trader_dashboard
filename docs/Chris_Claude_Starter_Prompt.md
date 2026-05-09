# Claude Chat Starter — Crypto Extension Phase 2

**For Chris.** Paste this entire message into a fresh Claude.ai chat session as your first message. It gives Claude the context it needs to be useful immediately. After the starter prompt, you can have a normal conversation with Claude about your crypto work.

---

## Paste this into a fresh Claude chat:

I'm working on a quant research project called Paper Trader, originally built by my brother Mike for equity strategies. I'm extending it with a crypto sibling module. Here's the situation:

**Project repository:** https://github.com/mteevan90/paper_trader_dashboard

**My role:** I own everything under `src/crypto/` and `models/snapshots/crypto/`. Mike owns the equity side (everything else under `src/`). We share approval on `dashboard_app.py`, `data_source.py`, `snapshot_for_cloud.py`, and a few other shared files.

**My hardware:** Ryzen 9 9900x, RTX 5080 (CUDA), 32GB RAM, 2TB storage. The CUDA card matters — model training and any deep-learning experiments will be much faster on my machine than on Mike's. Optuna optimization itself is CPU-bound, so my CPU performance is comparable to his.

**Current state of the project:**

1. Equity dashboard is live at https://paper-trader-mteev.streamlit.app with three promoted studies (#325 baseline, #842 15-position variant, #1852 continuous-sizing variant). Honest assessment: only #325 generalized to validation, and ~62% of its risk-adjusted alpha came from holdings in NVDA and META during the AI rally. The strategy works for the validated period but is heavily concentration-dependent.

2. **Phase 1 of the crypto extension is merged to main** (commit b83451d on May 9, 2026). Mike refactored the codebase to be asset-class-aware:
   - `data_source.py` has `path_to(local, asset_class="equities")` and `r2_key_for(local, asset_class="equities")` — defaults preserve equity behavior; passing `asset_class="crypto"` routes to crypto paths.
   - `snapshot_for_cloud.py` accepts `--asset-class` flag.
   - Dashboard sidebar has a Stocks/Crypto selector at the top. Stocks shows the existing equity dashboard; Crypto currently shows a placeholder pending my Phase 2 work.
   - `models/snapshots/pre_v2_20260505/` was moved to `models/snapshots/equities/pre_v2_20260505/`.
   - `.github/CODEOWNERS` enforces the ownership boundaries.

3. **My Phase 2 work begins now.** I'm building `src/crypto/` from scratch as a sibling to Mike's equity code. Key decisions already made (in `docs/Crypto_Extension_Decisions.md` on main):
   - Sibling module (not a fork). Shared dashboard chrome, separate strategy/data code per asset class.
   - Dynamic universe by market cap rank as of rebalance day, with explicit survivorship handling. Crypto survivorship bias is brutal.
   - Primary OHLCV from CCXT (Binance + Coinbase as cross-checks). NOT yfinance — yfinance crypto coverage is unreliable.
   - Universe metadata from CoinGecko free tier.
   - Don't reuse equity field names like `architecture`. Pick clearer names for crypto BacktestConfig.
   - BTC as primary benchmark; equal-weight top-10 as secondary.
   - Separate Optuna database per asset class.
   - Crypto macro signal needs to be designed from scratch (VIX-based equity macro doesn't transfer). Candidates: BTC funding rates, fear & greed index, stablecoin flows, exchange net flows.

**My Phase 2 deliverables:**

- `src/crypto/` package with: `backtest_config.py`, `backtest.py`, `data_source.py`, `model.py`, plus fetch utilities for CCXT and CoinGecko.
- Universe construction module that handles survivorship correctly (point-in-time market cap rankings from CoinGecko, including delisted tokens).
- Crypto-specific BacktestConfig — no `earnings_blackout_days`, no `sector_cap`, different field names where clarity benefits from it.
- Crypto-equivalent macro signal design (research question, not a port).
- Smoke study: ~100 Optuna trials on a 10-token universe to verify the plumbing works end-to-end.
- `models/snapshots/crypto/pre_v1_<date>/` snapshot once the smoke study is reproducible.

**My constraints:**

- Don't modify any equity code. Stick strictly to `src/crypto/` for new files; only modify shared files (`dashboard_app.py`, etc.) when there's no other option, and require Mike's review.
- Apply the same data-quality discipline that the equity side learned the hard way. Sanity-gate every fetch (refuse cache write if <50% non-empty results).
- The crypto dashboard (whenever I get to surfacing it) should follow the same three-layer pattern as Mike's equity dashboard: Layer 1 exec summary blue box, Layer 2 KPIs + hero chart, Layer 3 detail expanders.

**What I'd like you to help me with in this session:**

Don't write any code yet. Help me think through the Phase 2 design more concretely. Specifically:

1. **Universe construction strategy.** What's the cleanest way to fetch point-in-time market cap rankings from CoinGecko and handle delisted tokens? CoinGecko's free tier has rate limits I need to respect.

2. **Crypto macro signal design.** Of the candidate signals (BTC funding rates, fear & greed index, stablecoin flows, exchange net flows, realized volatility regime), which combination would make the most sense for a regime-aware crypto strategy? What's a reasonable v1 to ship for the smoke study?

3. **CCXT data layer.** Cross-checking Binance and Coinbase for OHLCV introduces edge cases (different close times, different daily candles for tokens listed on only one exchange, occasional outages). What's a reasonable canonicalization approach?

4. **What I should NOT try to port from Mike's strategy.** Mike's strategy uses fundamentals scoring, sector caps, earnings blackouts, and an XGBoost model trained on equity-specific features. Most of this doesn't apply to crypto. What's the equivalent feature space I should think about for a crypto candidate scorer?

5. **Smoke study scope.** What's the minimum viable smoke study that would prove the plumbing works without doing real research yet? I'm thinking: 10 top-mcap tokens, daily OHLCV from CCXT, basic technical scoring (no fundamentals, no model), 100 Optuna trials. Does that sound right?

After we work through the design, I'll come back with specific Claude Code spec drafts for the actual implementation. Goal of this session: a concrete written plan I can hand to Claude Code in subsequent sessions, not code.

---

## Notes for you (Chris) outside the prompt itself

- The above prompt gives Claude full context. You can adjust it before pasting if anything is wrong — e.g., if you have access to a CoinGecko Pro account or a paid Glassnode subscription, mention that. If you have specific tokens in mind for the smoke universe, name them.

- After the design discussion, you'll likely want to open Claude Code and feed it specs based on this conversation. Pattern: think with Claude Chat, execute with Claude Code.

- Your first practical task should be the universe construction module. Survivorship handling is genuinely the hardest part of crypto backtesting — getting it right early prevents you from chasing fake alpha later.

- Read `docs/Comprehensive_User_Guide.docx` end-to-end before or shortly after this conversation. Mike's hard-earned lessons (especially Section 15 "Gotchas") apply to crypto too, particularly around silent data failures from external APIs.

- Read `docs/Crypto_Extension_Decisions.md` carefully. That's the authoritative source for the architectural decisions; if anything in your Claude conversation contradicts it, the memo wins.

- Mike has stashed work (`stash@{0}` on main) related to the SP1500 universe expansion — leave it alone, that's his thread to pick up when Finnhub TOS clarifies.

Good luck with Phase 2.
