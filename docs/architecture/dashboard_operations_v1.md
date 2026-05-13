# Dashboard operations — v1

**Status:** v1 canonical for working on the contract-conformant dashboard.
**Scope:** "How to work on the dashboard" reference. Architecture overview + concrete how-tos for adding tabs, charts, and contract fields, drawn from the history of this project. Companion to `dashboard_contract_v1.md` (the data spec) — that doc tells you *what* the dashboard reads; this doc tells you *how the dashboard reads it and how to change what it reads*.
**Inherits memo shape from:** `ml_study_cv_objectives_v1.md` (adapted for an operations reference rather than a methodology finding).

## TL;DR

The dashboard (`src/dashboard_app.py`) renders two parallel worlds: **legacy Optuna v1 studies** (composite-weighted strategies, SQLite-backed, live-fallback compute path) and **contract-conformant v1+ studies** (auto-discovered from `models/studies/<name>/contract_v1/meta.json`, pre-computed parquet artifacts, no live-fallback). The sidebar's "Study type" radio routes between them. Inside contract-conformant, 7 universal tabs read artifacts from the study's `contract_v1/` directory; each tab function lives in `dashboard_app.py` next to its siblings and follows a small set of conventions (cached loaders, role-aware defaults, graceful fallback for optional files).

Most dashboard work falls into one of four shapes: **(1) fix a render bug**, **(2) add a chart to an existing tab**, **(3) add a new optional contract artifact + section**, **(4) add a whole new tab**. The recipes below walk through each, grounded in this project's actual change history.

## Architecture overview

### Two parallel worlds, one entry point

```
src/dashboard_app.py
└── main()
    ├── sidebar_asset_picker()              → Stocks / Crypto / Options
    ├── "Study type" sidebar radio          → Legacy / Contract-conformant
    │
    ├── (Legacy branch — Stocks + Legacy)
    │   ├── sidebar_config_picker()         → Default / Best trial / Custom #
    │   └── st.tabs([...8 legacy tabs...])
    │       ├── tab_performance(...)
    │       ├── tab_holdings(...)
    │       └── ... (Risk & Behavior, Tuning History, Glossary, etc.)
    │
    └── main_contract()                     → Contract-conformant branch
        ├── sidebar_contract_picker()       → choose a study
        └── st.tabs([...7 universal tabs...])
            ├── tab_contract_overview(study_name)
            ├── tab_contract_holdings(study_name)
            ├── tab_contract_trades(study_name)
            ├── tab_contract_alpha(study_name)
            ├── tab_contract_diagnostics(study_name)
            ├── tab_contract_walk_forward(study_name)
            └── tab_contract_tuning(study_name)
```

The two worlds share no rendering code intentionally. Legacy was the original v1 dashboard built on SQLite + `BacktestConfig`-tightly-coupled UI; contract-conformant is a clean break that reads from a versioned artifact contract. **Don't retrofit legacy with contract conventions or vice versa.** Mixing them re-introduces the tight coupling the contract was designed to escape.

### Auto-discovery for contract-conformant studies

`list_contract_v1_studies()` walks `models/studies/` once per session (cached) and returns every subdirectory containing `contract_v1/meta.json`. Drop a new study's artifacts at the right path and it appears in the sidebar selector — no code changes required. This is the contract's auto-discovery promise.

### Cached loaders

Contract artifacts are read via `@st.cache_data`-decorated helpers near the top of the contract-conformant section:

- `load_contract_meta(study_name)` → dict (meta.json)
- `load_contract_concentration(study_name)` → dict | None (concentration_summary.json)
- `load_contract_tuning_summary(study_name)` → dict | None (tuning_summary.json)
- `load_contract_parquet(study_name, filename)` → DataFrame (any parquet under contract_v1/)

Cache lives until the file changes (Streamlit hashes the path). Reloads are cheap. Use these instead of `pd.read_parquet(...)` inline; they're consistent and fast.

### Universal helper: model defaults

`_default_model_index(study_name, available)` returns the index of the preferred default model inside an alphabetically sorted dropdown list. Preference order:

1. The model with `role: "primary"` in `meta.json.models[]`
2. If no model declares `role: "primary"`, the first entry in `meta.json.models[]` (array order, not alphabetical)
3. If neither applies, 0 (first in displayed list)

**Every tab with a model selectbox must call this helper as its `index=` argument.** The convention exists because a sanity-check model is one click away but should never be the default view — see the `feat/dashboard-primary-model-default` history for the partner-perception cost of getting this wrong.

## The data contract is the canonical input spec

`docs/architecture/dashboard_contract_v1.md` is the source of truth for what artifacts a contract-conformant study produces. The dashboard renders that contract; it doesn't compute its own data. Anything the dashboard needs to display comes from either:

- **`meta.json`** — study identity, model declarations with roles, date windows, summary headline metrics, objective declarations.
- **Required parquet files** — portfolio.parquet, holdings.parquet, trades.parquet.
- **Optional parquet/JSON files** — scores.parquet, trial_log.parquet, tuning_convergence.parquet, tuning_summary.json, feature_importance.parquet, walk_forward.parquet, regime_attribution.parquet, concentration_summary.json, ic_decomposition.parquet, decile_returns.parquet, rolling_win_rate.parquet, per_ticker_attribution.parquet.

The contract spec describes each file's schema. The dashboard treats every "optional" file as truly optional — if it's absent, the relevant tab section degrades gracefully (caption explaining what's missing) rather than raising.

### Schema vs UI vocabulary distinction

`dashboard_contract_v1.md` has a "Terminology convention — schema fields vs UI labels" section that's load-bearing for how text in the dashboard is constructed. Two examples:

- `meta.json.windows.oos_start` is the schema field. The UI surfaces it as "Reserved validation period" / "Reserved validation window".
- `meta.json.objective.training_cv` is the schema field. The UI surfaces its value verbatim (it's a controlled vocabulary) but contextualizes it with text like "CV objective: `top_quintile_spearman_ic` (see [memo](...))".

When in doubt, schema-field names stay short and technical (developer-facing); UI labels are partner-facing and prioritize legibility over convention. Don't rename schema fields to match UI prose.

## How studies feed the dashboard

A study writes its contract artifacts during Phases 4 and 5 (see the per-phase Phase docs in `docs/studies/<study_name>/`). After artifacts land:

1. The study's directory becomes `models/studies/<study_name>/contract_v1/`.
2. The dashboard's next page-load picks up the new study via `list_contract_v1_studies()`.
3. Tabs that find their relevant artifact render; tabs that don't (e.g., Walk-forward when there's no walk_forward.parquet) show a graceful placeholder.

No registration step. No code change. This is the contract's design payoff.

### Two deployment paths — legacy vs contract-conformant studies

The `.gitignore` rule `models/*` blocks ordinary `git add`. Studies use one of two mechanisms to get their data to the cloud dashboard:

**Legacy studies** (composite-weighted family, `models/cache/dashboard_results/<label>/`):
- Synced to R2 object storage via `src/snapshot_for_cloud.py`.
- Cloud Streamlit (in `DASHBOARD_CLOUD_MODE`) fetches files from R2 on demand using the bucket layout in `src/data_source.py` (`R2_LAYOUT_SUFFIX` + `dashboard_results/` prefix rule).
- Workflow: produce results locally → run `python src/snapshot_for_cloud.py` → cloud reads via R2 within ~5 min cache TTL.

**Contract-conformant studies** (`models/studies/<study_name>/contract_v1/`):
- Force-added to git via `git add -f models/studies/<study_name>/`. The `-f` flag is required because `.gitignore` would otherwise silently skip these files.
- Cloud Streamlit reads from the deployed git checkout — no R2 fetch for contract data. Streamlit Cloud's GitHub integration auto-deploys on push to `main`; new study data appears in the cloud dashboard within minutes of the push.
- Workflow: produce contract artifacts locally → `git add -f models/studies/<study_name>/` → commit → push → cloud auto-deploys.
- v1 established this pattern at commit `5b96fd0`. v2 followed it at commit `7ae3977` (force-added 90 files / ~10MB for `larger_universe_v2/` covering 7 variants + `comparison/` artifacts + `variant_meta.json`).

The `src/snapshot_for_cloud.py` script does NOT sync `models/studies/<study>/contract_v1/` — its bundle scope is the legacy `dashboard_results/` subtree plus the hardcoded `R2_LAYOUT_SUFFIX` files only. Running it does not affect contract-study cloud visibility.

### Deploying a new contract-conformant study to the cloud dashboard

Concrete step-by-step (matches v1 + v2 pattern):

```
# 1. Produce contract artifacts locally (Phase 4 + Phase 5 runners)
python scripts/research/phase4_run.py        # or phase4_run_v2.py for multi-variant
python scripts/research/phase5_walk_forward.py
python scripts/research/phase5_analytics.py  # or phase5_analytics_v2.py

# 2. Verify artifacts exist locally
ls models/studies/<study_name>/contract_v1/

# 3. Force-add to git (-f required due to models/* .gitignore rule)
git add -f models/studies/<study_name>/

# 4. Commit with a descriptive message naming the study
git commit -m "phase4(study): <study_name> — contract v1 artifacts"

# 5. Push to main (or via PR depending on branch protection)
git push origin main

# 6. Streamlit Cloud auto-deploys on push to main; new study
#    appears in the dashboard's "Study" sidebar within minutes.
```

**Why git instead of R2 for contract data:** the contract artifacts are version-controlled snapshots of what a study produced; deploying them via git keeps the dashboard's view bit-identical to what the writeup describes at that commit. Legacy `dashboard_results/` predates the contract and uses R2 because the legacy artifacts are larger and historically updated more frequently. Future architectural conversation may revisit this split (tracked as a follow-up post-v2); for now both paths are in use and partners should know which applies to their study type.

## Recipe 1 — Fix a render bug

The smallest type of change. Two recent examples ground the recipe.

**Example A: `feat/dashboard-add-vline-fix` (Plotly 6.7 + pandas 3.0 datetime annotation bug).**

The Performance tab's `fig.add_vline(x=oos_start, annotation_text="OOS start", ...)` raised TypeError because Plotly 6.7's annotation-positioning math adds an integer offset to the x value, and pandas 3.0 doesn't allow integer arithmetic with `Timestamp` or `datetime` or `np.datetime64`. The fix was to convert to ms-since-epoch:

```python
oos_ms = int(pd.Timestamp(oos_start).value // 10**6)
fig.add_vline(x=oos_ms, ..., annotation_text="...", annotation_position="top right")
```

**Lesson — Streamlit cascade failure mode.** An unhandled exception inside any `with tabs[i]:` block halts the entire script run, making it look like every tab failed when only one tab raised. Diagnostic shortcut: when "every tab fails", check for one exception in the earliest non-rendering tab rather than assuming wide-scope breakage.

**Example B: `feat/dashboard-primary-model-default` (model selector default).**

All 5 contract-tab model selectboxes used `index=0` (alphabetical-first) instead of role-aware defaulting. For Larger Universe v1 that meant the sanity-check ElasticNet rendered before the primary XGBoost — a partner-perception bug, not a crash bug. Fix: extract `_default_model_index()` reading `meta.json.models[].role` and wire all 5 selectboxes through it.

**Process for fixing render bugs**:

1. Branch off `main`: `feat/dashboard-<short-bug-description>`.
2. Reproduce, ideally headless via `streamlit.testing.v1.AppTest` (sample fixture: Larger Universe v1 contract artifacts in-repo at `models/studies/larger_universe_v1/contract_v1/`).
3. Fix the smallest unit possible. Don't touch unrelated code in the same commit.
4. Smoke-test via AppTest — toggle sidebar to Contract-conformant, exercise the path that raised.
5. Add a session log entry covering symptom, root cause, fix scope, smoke-test outcome, and any side observations worth preserving (institutional knowledge for future debugging).
6. Push. Don't auto-merge. Surface for review.

## Recipe 2 — Add a chart to an existing tab

A pure rendering change. No contract change.

Locate the tab function (e.g., `tab_contract_diagnostics`), load the relevant artifact via `load_contract_parquet(study_name, ...)`, build the figure with `plotly.graph_objects` or `plotly.express`, and render via `st.plotly_chart(fig, use_container_width=True)`.

**Conventions to follow**:

- **Cached loaders only.** Don't `pd.read_parquet(...)` inline; use `load_contract_parquet()`.
- **Empty-frame guard.** Every tab reads optional artifacts. If `load_contract_parquet(...)` returns an empty DataFrame, render a caption explaining what's missing and return.
- **Model selector via the helper.** If the chart is per-model, add a selectbox with `index=_default_model_index(study_name, models)` and a unique `key=` matching the convention `contract_<purpose>_<scope>` (e.g., `contract_alpha_model`).
- **Plotly time-axis gotcha.** When using `add_vline(x=<date>, annotation_text=...)`, convert the date to ms-since-epoch (`int(pd.Timestamp(d).value // 10**6)`). Without `annotation_text`, raw `Timestamp` works; with `annotation_text`, only int ms works on the current library stack.
- **Numeric vlines (e.g., 25% constraint line in Alpha Attribution) are fine as-is.** The annotation-math bug is specific to datetime axes.

## Recipe 3 — Add a new optional contract artifact + section

The most common shape of dashboard enhancement. Worked example: `feat/contract-tuning-enhancements` added `tuning_convergence.parquet` and `tuning_summary.json` plus the narrative-summary, score-histogram, convergence-curve, parameter-sensitivity sections on the Tuning tab.

Process — three commits as one coherent unit:

### Commit 1: Update the contract spec

Edit `docs/architecture/dashboard_contract_v1.md`:

- Add the new file(s) to the path-tree under the appropriate category (required / required-for-X / optional / recommended).
- Add a schema section documenting columns/fields with types and notes.
- Document graceful-fallback behavior: "dashboards must render the prior view (with a caption explaining what's missing) when the file is absent."
- Specify schema-version semantics. **Additive changes** (new optional columns, new optional files, new optional `meta.json` fields) don't bump `schema_version`. **Breaking changes** (removed columns, renamed columns, changed semantics) bump to `v2` and a v2 renderer routes alongside the v1 renderer.

The contract change lands as a single doc-only commit. This is the architectural lever — every later commit can reference the contract as already-stable.

### Commit 2: Build the artifact (or back-fill for existing studies)

For future studies, the relevant Phase X driver produces the artifact natively. For existing studies (currently just Larger Universe v1), write a one-time back-fill at `scripts/maintenance/<name>.py` and run it.

Back-fill script convention:

- Module docstring explains: what it builds, why it exists (e.g., "this is a one-time retroactive build for studies tuned before the contract addition landed; future studies produce these natively"), idempotency guarantee.
- Read from existing artifacts; write into the study's `contract_v1/` directory.
- Print a per-model summary so the runner can verify sanity at a glance.

For Larger Universe v1 the back-fill produced 300 rows of convergence data (200 XGB + 100 ENet) and a per-model summary JSON. See `scripts/maintenance/backfill_tuning_convergence.py` as the template.

### Commit 3: Add the dashboard rendering

Add a cached loader for the new artifact (matching the existing `load_contract_*` helpers). In the relevant tab function, load the artifact and render the new section(s). Guard with graceful fallback when the artifact is absent.

Example pattern (from Tuning tab):

```python
conv = load_contract_parquet(study_name, "tuning_convergence.parquet")
summary = load_contract_tuning_summary(study_name)
has_precomputed = (
    not conv.empty
    and summary is not None
    and model in summary
    and summary[model].get("total_trials", 0) > 0
)
if has_precomputed:
    # render the narrative + histogram + convergence-curve sections
    ...
else:
    st.caption(
        "Pre-computed convergence data not found. "
        "Run scripts/maintenance/backfill_tuning_convergence.py "
        f"--study {study_name} to enrich, or this study will get "
        "the narrative + histogram + convergence-curve sections "
        "once its Phase 3 produces them natively."
    )
```

The fallback caption tells the user exactly how to fix the gap. Don't show a blank section.

## Recipe 4 — Add a whole new tab

The largest type of change. Treated as a special case of Recipe 3: define the artifacts the tab consumes (add to contract spec), back-fill for existing studies if applicable, then add a `tab_contract_<name>` function and wire it into `main_contract()`'s `st.tabs([...])` list and dispatch block.

**Sizing question to ask first**: does this content belong on a new tab, or can it live as a section on an existing tab? Tabs are expensive — they take horizontal real estate and split the user's attention. Section-on-existing-tab is the default; new tab is the exception. The 7-tab structure was deliberately reduced from 8 (Performance merged into Overview) because the Performance tab held only one chart — Overview-as-section was the better home.

If a new tab is genuinely the right answer, the recipe:

1. Update `dashboard_contract_v1.md`'s tab structure table to include the new tab and its purpose.
2. Add `tab_contract_<name>(study_name)` following the conventions in this doc.
3. Add to `main_contract()`'s tab labels list and dispatch block.
4. Smoke-test that all existing tabs still render (re-indexing breaks easily).
5. Tracker update? Only if the new tab represents a study-level milestone (e.g., a new study family that needed bespoke rendering). Pure dashboard-internal changes don't trigger tracker updates.

## Updating the contract spec

The contract is versioned with `meta.json.schema_version`. Currently `"v1"`.

**Additive change** — `schema_version` stays at `"v1"`:
- New optional columns on an existing parquet file
- New optional file in `contract_v1/`
- New optional field on `meta.json` (any nested path)

Dashboards must treat missing optional fields/files as "feature not supported by this study" and render gracefully without them. Example: `objective.training_cv` was added to `meta.json` post-Phase-5; studies that pre-date it should still render. (None do in practice — Larger Universe v1's meta.json was retroactively updated when the field was added — but the dashboard's defensive coding handles the absence.)

Another example: `artifact_metadata` was added to `meta.json` 2026-05-13 to record per-artifact scope context for scope-sensitive analytics (`ic_decomposition.parquet`, `decile_returns.parquet`). The dashboard treats its absence as "no scope info; use default caption" (with a legacy-v1-specific fallback banner for the known affected study). Studies running `phase5_analytics_v2.py` populate the field automatically; legacy studies get patched via `scripts/research/annotate_meta_artifact_scope.py`. See `docs/architecture/dashboard_contract_v1.md` → `artifact_metadata` for the schema.

**Breaking change** — bumps to `"v2"`:
- Removed columns
- Renamed columns (rename = remove + add; this is a v2-level change for a published artifact)
- Changed semantics of existing fields (e.g., a percentage that was 0–100 becoming 0–1)

The dashboard reads `schema_version` and routes to the matching renderer. v1 studies render under the v1 renderer indefinitely.

In practice every change to date has been additive. When a breaking change is genuinely needed, the v1 → v2 path is to write a v2 renderer alongside the v1 renderer rather than to migrate v1 studies to the new shape.

## Process rules

These are project-wide rules but they apply with extra force to dashboard work because partners read the dashboard and partner-perception bugs are easy to ship inadvertently.

### Feature branches

Every change to `src/dashboard_app.py` or `docs/architecture/dashboard_*` goes on a feature branch. Naming convention: `feat/dashboard-<short-description>` or `fix/dashboard-<short-description>`. Examples from this project:

- `feat/larger-universe-v1-study` (Phase 4.5 — initial contract-conformant tabs landing)
- `feat/dashboard-add-vline-fix` (Plotly+pandas datetime annotation bug)
- `feat/contract-tuning-enhancements` (tuning_convergence + Tuning-tab parity work)
- `feat/dashboard-primary-model-default` (role-aware model selector defaults)
- `feat/dashboard-overview-merge-and-terminology` (Performance → Overview merge + "OOS" → "reserved validation")

### Review before merge

Surface every dashboard change for review. Don't auto-merge. The reviewer (Mike, typically) approves explicitly via "Merge approved" before the merge happens. Standard mechanics: `--no-ff` merge commit, no squash, preserve the feature branch.

### Smoke-test via AppTest

The `streamlit.testing.v1.AppTest` harness is the right shape for smoke-tests:

```python
import sys
sys.path.insert(0, "src")
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("src/dashboard_app.py", default_timeout=180)
at.run()
for r in at.sidebar.radio:
    if r.label == "Study type":
        r.set_value("Contract-conformant (v1+)").run()
        break
# Inspect at.main.tabs, at.selectbox, at.info, at.markdown for content
assert len(list(at.exception)) == 0
```

For changes that touch a model selector, also verify all 5 selectors still default to `role: "primary"`. For changes that touch a chart, verify the chart renders without exception (the harness force-serializes Plotly figures, catching the same TypeErrors a browser would).

### Session log entries for material changes

Every branch's worth of work gets an entry in the relevant session log (currently `docs/sessions/larger_universe_v1/session_log.md` for everything LU-v1-related). Entries cover: what was built, decisions made (especially decisions that have non-obvious rationale), smoke-test outcome, side observations worth preserving as institutional knowledge.

Two examples of institutional knowledge worth preserving:

1. **Plotly 6.7 + pandas 3.0 datetime annotation bug** — `add_vline` with `annotation_text` on a datetime axis requires ms-since-epoch. From `feat/dashboard-add-vline-fix`.
2. **Streamlit cascade failure mode** — one exception in any `with tabs[i]:` block halts the entire script. Diagnostic shortcut for "every tab fails" symptoms. Same source.

Both findings would be re-derived expensively if not captured.

### Tracker updates only at natural stable points

The `docs/Project_State_Tracker.docx` is for **study-level milestones and structural changes**, not routine fixes. Updates happen when:

- A study completes a phase or closes entirely
- A new study workstream starts
- A structural change to project shape lands (e.g., the introduction of the dashboard contract; the Overview-merge + terminology cleanup)
- Partner feedback changes direction

Routine bug fixes, individual selector defaults, single-tab enhancements — none of these trigger tracker updates. The tracker is partner-facing; it gets noisy if every commit produces an entry.

Standing rule from the LU-v1 session: "tracker updates wait for genuinely study-level milestones."

## Caveats

### Reliability caveats — what this doc captures correctly

1. **Code paths and conventions are current as of 2026-05-12.** The `_default_model_index` helper, the cached loaders, the 7-tab structure, the schema-vs-UI distinction — all reflect main as of this date. Future changes may make subsections obsolete; check `git log src/dashboard_app.py` before assuming a recipe is current.
2. **Examples are real but may age.** The Plotly+pandas bug is specific to versions 6.7 / 3.0; a future upgrade may fix the underlying annotation-math issue and make the ms-since-epoch workaround unnecessary. The institutional-knowledge entry will note the resolution if/when that happens.

### Scope caveats — what this doc doesn't cover

3. **Legacy tab internals.** This doc focuses on the contract-conformant world. Legacy tab functions (`tab_performance`, `tab_holdings`, `tab_market_context`, `tab_reliability`, `tab_tuning_history`, etc.) have their own conventions tied to SQLite + `BacktestConfig` + `dashboard_results/` paths. If you're modifying a legacy tab, read the existing code carefully — the conventions are different.
4. **Cloud-mode deployment.** `src/data_source.py` handles cloud-vs-local data routing; `snapshot_for_cloud.py` syncs artifacts to R2. These are out of scope for tab-level work but become relevant when adding new artifacts that need to ship to the cloud build.
5. **Streamlit / Plotly version drift.** This doc assumes Plotly 6.7 and pandas 3.0. The `use_container_width` deprecation warning currently emitted on every page load (see standing follow-up list) will eventually require a sweep. When that happens, the conventions in this doc carry forward but the specific API call pattern changes.

## Standing follow-ups

Tracked here for visibility; not actioned in this doc.

1. **`use_container_width` deprecation sweep.** Streamlit logs ~40 warnings per page load. Mechanical sweep, replace with `width="stretch"` / `width="content"`. Affects both legacy and contract code. Not blocking; sweep candidate.
2. **Dashboard pytest coverage via `streamlit.testing.v1.AppTest`.** No existing pytest coverage. The AppTest harness pattern used in smoke-tests is the right shape for formal coverage. Sample fixture: Larger Universe v1 contract artifacts in-repo.
3. **`attempted_trials` enhancement to `tuning_summary.json`.** Currently `total_trials` is COMPLETE-only; the narrative "tested 14 configurations" for ENet hides that 86 trials failed. A future enhancement adds `attempted_trials` and surfaces the attempted-vs-complete split. Revisit when another study produces similarly-high failure rates.
4. **Convergence-pattern methodology memo.** XGBoost on Larger Universe v1 plateaued at 61% of 200 trials; the legacy v1 study plateaued at ~33% of 1000 trials. Two data points. A third data point promotes this observation to a memo at `docs/architecture/`. Pending.

## Sourced from

- `docs/architecture/dashboard_contract_v1.md` — the data contract that this doc operates on top of.
- `docs/architecture/ml_study_cv_objectives_v1.md` — first architectural memo; established the shape this doc adapts.
- `docs/architecture/sanity_check_methodology_v1.md` — second architectural memo; paired with the CV-objectives memo, both derived from Larger Universe v1.
- `docs/sessions/larger_universe_v1/session_log.md` — the running log of dashboard changes since Phase 4.5 landed. Every recipe in this doc is grounded in entries from this log.
- `src/dashboard_app.py` — the implementation. Read alongside this doc for any non-trivial change.
