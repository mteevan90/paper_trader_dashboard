# Future Work

A running list of ideas, follow-ups, and "we'll get to it" items that don't have current owners or active timelines. Each item should explain *what*, *why*, and *when it becomes worth doing* — so anyone reading this doc later (including future Claude sessions) can prioritize honestly.

When something here moves to active work, move it out of this doc into the appropriate session/ticket and remove it here.

---

## Distributed Optuna across machines

**Status:** Idea, not blocking anything.

**What:**  Set up Optuna's distributed optimization mode so multiple PCs can run workers against a shared trial database. With Mike's 9900x (12 cores) and Chris's 9900x (12 cores), a single large study could run on 24 cores in parallel instead of 12.

**Why it's worth doing eventually:**  Optuna is already designed for this — TPE work-stealing across workers is supported out of the box. The synchronization point is just the trial database. For studies that take 5+ hours, halving wall-clock time matters.

**When it becomes worth doing:**

- A single study takes more than 6 hours wall-clock (e.g., walk-forward validation, which is roughly 20x compute, would take 4 days at current speeds — distributed compute brings it to 2 days)
- Or both contributors are routinely sitting idle while waiting for the other's compute job to finish

**What's required:**

- Shared Optuna database — either hosted Postgres or shared SQLite via SMB/network share
- Both machines on identical git SHA (TPE results aren't comparable across code versions)
- Snapshot data mirrored to both machines (git-lfs, network share, or each machine independently fetches and caches)
- A startup runner that points at the shared DB and adds workers from a given machine

**Estimated effort:** 1-2 Claude Code sessions to wire up. Not trivial because of the synchronization, but not large.

**What it does NOT solve:**

- Model retraining (that's still GPU-bound on Chris's machine specifically — see "Hardware-aware work allocation" below)
- Deep-learning experiments (CUDA-only, also Chris's machine)
- Backtests of a single config (those are inherently sequential per config; distributed Optuna parallelizes across configs)

---

## Hardware-aware work allocation between contributors

**Status:** Worth a conversation, not a code task.

**What:**  Explicit division of labor that plays to hardware advantages rather than treating both contributors as fungible CPU resources.

**Mike's machine:** Ryzen 9 9900x + AMD 9070 XT (no CUDA), 32GB RAM, 4TB.
**Chris's machine:** Ryzen 9 9900x + RTX 5080 (CUDA), 32GB RAM, 2TB.

**Suggested allocation:**

- Chris should own model training and retraining work. CUDA gives him 5-10x speedup on XGBoost training. If hyperparameter search on the model itself ever becomes a study, that's clearly his work.
- Chris should own any future deep-learning experiments (LSTMs, transformers, GNNs). Mike's AMD card can't realistically train them.
- Both contributors run Optuna studies independently for now; CPU is parallel by default when each runs their own.

**When this matters:** When the project gets to model-architecture experiments or walk-forward validation that requires retraining at every step. Until then, both machines are roughly equivalent for current workflows.

---

## Walk-forward validation

**Status:** Mentioned in handoff doc Section 9 (Q6); deferred.

**What:**  Replace the current fixed train/validation split with a rolling window approach. Train on months 1-24, validate on month 25. Then train on months 2-25, validate on month 26. Repeat through the entire history.

**Why it's worth doing:**  Fixed splits give one validation result; walk-forward gives many, with a confidence interval on generalization. It's the standard rigor bar for quantitative strategy validation.

**When it becomes worth doing:**

- After at least one new strategy variant has been validated under the current methodology and looks promising
- When we're confident enough in a strategy that we want to know how it generalizes across regimes, not just one validation window
- Realistically: after Phase 2 of the crypto extension is shipped, when there's headroom to invest in research methodology improvements

**Cost:** Roughly 20x current compute. A 100-minute study becomes a 33-hour study. Distributed Optuna becomes a hard prerequisite.

---

## Sizing decisions visualization

**Status:** Spec ready at `docs/specs/Sizing_Decisions_Viz_Prompt.md`; queued for implementation.

**What:**  New section on the Risk & Behavior dashboard tab visualizing per-rebalance sizing decisions logged by the continuous-sizing strategy. Hero chart of position count + macro signal over time, plus forward-return-by-sizing-bucket analysis.

**Why it's worth doing:**  The continuous-sizing study is promoted to the cloud dashboard but readers can't see *what the sizing did*. The data is in `sizing_decisions.parquet`; the dashboard just needs to surface it.

**When it becomes worth doing:**  When the cloud dashboard's continuous-sizing study gets meaningful viewer traffic (your dad/brother actually click into it and want to understand why the validation alpha is what it is). Right now it's quietly sitting at "modest underperformance"; the visualization story is most useful when someone's confused about why.

**Cost:** ~2-3 hours of Claude Code work. Spec already exists.

---

## NVDA blacklist analysis dashboard surfacing

**Status:** Research finding committed locally (`f46bd05`), not pushed; dashboard not yet updated.

**What:**  The May 2026 NVDA blacklist analysis showed ~62% of #325's risk-adjusted alpha comes from holdings in NVDA and META during the AI rally. The Performance tab's exec summary should reflect this finding.

**Suggested update:** Replace the generic "validation period was a strong bull market" caveat with: *"Approximately 62% of this strategy's risk-adjusted edge comes from holdings in NVDA and META during the AI rally. Performance in market conditions where these names underperform could be materially different."*

**Why it's worth doing:**  The current exec summary lets the reader assume the strategy's edge is broadly distributed. The NVDA analysis showed it isn't. Honest framing matters when family members are looking at the dashboard.

**When it becomes worth doing:**  After the crypto extension Phase 1 refactor lands. Bundling these into the same commit confuses the changelog.

**Cost:** Small Claude Code session, 30-45 min.

---

## Headline alpha: switch from arithmetic to CAPM-corrected

**Status:** Discussed but not actioned.

**What:**  The Performance tab currently leads with arithmetic alpha (#325 shows +63.7pp/yr). The CAPM-corrected version (+39.3pp/yr) is more honest because it removes the high-beta market amplification — NVDA's beta of 1.5+ inflated the arithmetic number.

**Why it's worth doing:**  The arithmetic figure is real but misleading. CAPM is the more defensible headline.

**When it becomes worth doing:**  Same window as the NVDA blacklist exec summary update. They're conceptually linked — both are "stop overstating the strategy."

**Cost:** Small Claude Code session, 30-45 min.

---

## Cross-study toggle UX upgrade (Tier 2)

**Status:** Mentioned during dashboard refactor; deferred.

**What:**  The dashboard's "Compare best-known values across all studies" toggle currently pools all promoted studies. With three studies now (and growing), readers should be able to select WHICH studies to pool, not all-or-nothing.

**Why it's worth doing:**  As more studies get promoted, the all-or-nothing toggle loses precision. Comparing #325 vs continuous-sizing only is useful; pooling those two with the 15-position study muddies the question.

**When it becomes worth doing:**  When promoted study count exceeds 4 or family feedback explicitly requests the granularity.

**Cost:** Medium Claude Code session, ~2 hours. Touches multiple tabs.

---

## Trade History "Past 6 months" date shortcut still rejects

**Status:** Known UX bug, deferred.

**What:**  Streamlit's date_input shortcut buttons compute the end date as today's actual real-world date, not the data's last date. The clamp logic catches it after-the-fact, but the UX of the rejection-then-snap is awkward.

**Why it's worth doing:**  Better UX. Clean.

**When it becomes worth doing:**  When a viewer reports the bug or someone has spare cycles on dashboard polish.

**Cost:** Small. Either suppress the shortcut buttons entirely or upgrade the clamp to never produce the rejection state.

---

## SP1500 universe expansion (paused)

**Status:** Phase 1 (data infrastructure) paused pending Finnhub TOS investigation.

**What:**  Expand the universe from 491 (S&P 500 + Nasdaq 100) to ~1473 tickers (S&P 1500). Required: solving the earnings data problem yfinance can't deliver at scale.

**Current state:**

- Code infrastructure built and committed (`730e3f4`, `a99e4f8`, `a1ec3a8`, `0c4b9dc`)
- Deferred retry/sanity-gate/force-refresh logic in working tree (~103 lines, uncommitted, proven correct)
- yfinance fundamentals + sectors + prices fetched for all 1473 tickers
- Earnings data blocked: yfinance returns ~16% non-empty, sanity gate correctly preserves prior cache
- Finnhub free tier insufficient (4-5 quarters per ticker; need 30+)

**Pending:** Mike's investigation of Finnhub paid tier TOS — specifically whether the "subscribe one month, cache data forever" model is allowed.

**When it resumes:**  After Finnhub TOS clarity AND after the crypto extension Phase 1 lands. Don't try to do both at once.

---

## Live cache relocation to models/cache/equities/

**Status:** Deferred. Phase 1 of crypto extension chose conservative path; this is the cleanup.

**What:**  Move all equity-side cache files into models/cache/equities/ to mirror the crypto subdirectory pattern. Currently equity caches sit at the top level (models/cache/optuna_studies.db, models/cache/dashboard_results/, etc.) while crypto goes to models/cache/crypto/. Asymmetric but functional.

**Why it's worth doing eventually:**  Symmetry is cleaner. Both asset classes under their own subdirectories means new contributors don't have to remember the convention; it's enforced by the file layout.

**When it becomes worth doing:**

- A quiet research week with no active studies running
- Or before a major version bump where path conventions matter

**What's required:**

- Modify CACHE_DIR constants in fetch_data.py, backtest.py, optuna_runner.py, feature_cache.py, and macro_signals.py
- Move ~few GB of cache files into the new subdirectory
- Update path_to() defaults and any other hardcoded path references
- Verify all studies still reproduce against the new paths

**Cost:** One focused Claude Code session, ~1-2 hours. Touches do-not-modify files in 5 locations, so needs care.

**Branch suggestion:** mike/equity-cache-relocation

---

## Snapshot earnings cache refresh for newly-added tickers

**Status:** Pre-existing issue surfaced during Phase 1 verification.

**What:**  The locked snapshot pre_v2_20260505/ was built before GEHC, ARM, and VLTO were added to UNIVERSE_TICKERS. The strict snapshot-miss check from commit a99e4f8 correctly rejects fetching data for tickers not in the snapshot, which means fresh rescore_baseline.py runs against this snapshot fail with cache-miss errors.

**Why it's worth fixing:**  Anyone trying to run a baseline rescore locally will hit this. The cloud dashboard works fine because R2 has the precomputed results, but local research workflows are broken until this is fixed.

**When it becomes worth doing:**  Next time you need to run a fresh rescore against the baseline snapshot. Or proactively if you want a smooth research workflow.

**Two possible fixes:**

1. **Refresh the snapshot's earnings cache to include the new tickers.** Cleanest answer. Snapshot stays self-contained. Requires care: the new tickers' earnings data needs to be from a date as close to 2026-05-05 (the snapshot's nominal time) as possible to maintain reproducibility.

2. **Relax the strict miss check.** Allow tickers that entered the universe after the snapshot was built to fetch from live cache. Simpler but loosens the reproducibility guarantee.

**Cost:** Small — one Claude Code session, 30-45 min for option 1. Less for option 2.

---

## Claude Code workflow improvements

**Status:** Loose collection of pain points from today's session.

Items to think about when there's time:

- **PowerShell BOM handling.** PowerShell's `Out-File -Encoding utf8` writes a UTF-8 BOM that Python tools sometimes misread. Standardize on `Out-File -Encoding utf8NoBOM` if available, or post-process to strip BOMs.
- **Tracker scripts that handle stage detection.** Today's `track_fetch_v2.py` failed because of BOM issues plus alphabetical first-N bias in smoke tests. A more robust progress tracker as a reusable utility would save time.
- **Pre-flight check for `PAPER_TRADER_DATA_ROOT`.** Multiple times today, lingering env vars from prior commands caused fetch failures. A short pre-flight script that validates env state before launching long-running jobs would catch these.
- **Random sampling in smoke tests.** `fetch_sp1500.py --limit 25` takes the alphabetical first 25, which biased multiple diagnostics today. Should be random by default.

---

## Options Phase 1 (shared-edge)

Options Phase 1 (shared-edge) merged on `<merge date>`. Phase 2 sections 1–9 specced in `Options_Extension_Decisions.md`.

---

## Options Section 4 (Position + lifecycle model)

Options Section 4 (Position + lifecycle model) merged on `<merge date>`.

## Options Section 5 (BacktestConfig + FeeModel)

Options Section 5 (BacktestConfig + FeeModel) merged on `<merge date>`.

## Options Section 6 (backtest engine)

Options Section 6 (backtest engine) merged on `<merge date>`. Included Section 5 amendment adding `starting_capital` and `assumed_spread_pct` to `BacktestConfig`. `pandas_market_calendars` added to requirements.txt.

## Options Section 7 (Optuna runner + smoke study)

Options Section 7 (Optuna runner + smoke study) merged on `<merge date>`. Calmar objective on training-window snapshots only; Optuna TPE with SQLite storage at `models/cache/options/optuna_studies.db`; resume via `load_if_exists=True`. Compressed-full smoke study CLI exercises 8 underlyings × 2 strategies × 5 trials × 6 months — manual sandbox run is post-merge (~30min, network-dependent).

## Options Section 8 (production v1 study)

Options Section 8 (production v1 study) merged on `<merge date>`. v1_study orchestrator runs primary Optuna for CSP + CC, concentration analysis (per-underlying / per-DTE-band / per-IV-regime ablations), automated promotion gate with five hardcoded checks, human-override capability, and a snapshot of the promoted run. Bundled Section 6 amendment (entry_filters + fetch_iv_regime on EngineDeps) and Section 2 amendment (fetch_index_quote_history). Manual production run is post-merge (~6 hours, network-dependent).

## Options Section 3 (Black-Scholes Greeks module)

Options Section 3 (Black-Scholes-Merton Greeks module) merged on `<merge date>`. Pure-function module with `price`, `delta`, `gamma`, `theta_per_day`, `vega_per_pct`, `rho_per_bp`, `implied_vol`, `compute_all`, plus a `time_to_expiration` ACT/365 helper. Section 1's `UnderlyingMeta` amended in this PR to carry a `dividend_yield` field for the BSM `q` input.

## Options Section 2 (Tradier OHLCV + chain fetcher)

Options Section 2 (Tradier OHLCV + chain fetcher) merged on `<merge date>`. truststore landed in main with this section.
ea534c8 (Options Section 2: Tradier OHLCV + chain fetcher)

---

## How to use this doc

- When something on this list becomes a current priority, move it out (to a session, ticket, or active work) and delete from here.
- When something new comes up that isn't blocking, add it here with the same `What / Why / When / Cost` template.
- Quarterly (or whenever felt right), review and prune anything that's no longer relevant.

Keep this doc honest. If something's on here for 6 months without being touched, it probably isn't worth doing — delete it rather than letting it pile up.
