# Dashboard audit — 2026-05-07

**Scope**: read-only audit of `src/dashboard_app.py` (1733 lines).
**Goal**: inform "rebuild vs patch" decision for adding new tabs
(perturbation/robustness, eventually attribution).

**TL;DR**: **Recommendation (a) — keep current structure, add new tabs
in the existing pattern.** Cruft is concentrated in
`tab_diagnostics` + `_notable_observations`. The bones around them
(path layer, cache layer, tab dispatch, sidebar) are sound and new
tabs slot in cleanly without touching the cruft.

---

## 1. Structure

| metric | value |
|---|---:|
| total lines (`src/dashboard_app.py`) | **1733** |
| top-level functions | **39** |
| `@st.cache_data` decorators | 11 |
| `@st.cache_resource` decorators | 1 (live backtest at L368) |
| longest function | `tab_diagnostics` 304 lines (L1373-1676) |
| 2nd longest | `_notable_observations` 223 lines (L1148-1370) |
| `main()` | 48 lines (L1683-1730) |

**Tab dispatch is explicit, one-line-per-tab in `main()`** (L1713-1730):

```python
tabs = st.tabs(["Overview", "Diagnostics", "Optuna explorer",
                "Macro state", "Positions", "Trades log", "User Guide"])
with tabs[0]: tab_overview(label, config, result)
with tabs[1]: tab_diagnostics(label, config, result)
# ... etc
```

This is the cleanest of the patterns the spec listed (no if/elif chain,
no dispatch dict). Adding an 8th tab is a 2-line edit.

`main()` itself is appropriately thin — auth gate, header strip, sidebar
call, result load, tab dispatch. No business logic inlined. Good.

**Function-length distribution (top 10):**

| function | lines | location |
|---|---:|---|
| `tab_diagnostics` | 304 | L1373-1676 |
| `_notable_observations` | 223 | L1148-1370 |
| `sidebar_config_picker` | 114 | L543-656 |
| `tab_overview` | 106 | L663-768 |
| `tab_optuna` | 92 | L771-862 |
| `tab_macro` | 67 | L865-931 |
| `_reconstruct_sector_weights` | 63 | L1083-1145 |
| `_render_df_with_ticker_links` | 59 | L256-314 |
| `tab_positions` | 59 | L934-992 |
| `tab_trades` | 59 | L995-1053 |

Five tab functions sit in the 59-114 line range — comfortable size.
`tab_diagnostics` is the outlier at 5× a typical tab.

---

## 2. Data flow

### Per-tab data sources (loaders + payload from main())

| tab | loader calls | result-dict deps |
|---|---|---|
| Overview | `cached_benchmark`, `load_macro_df` | `portfolio_df`, `meta`, `trades_df` |
| Diagnostics | `cached_benchmark`, `load_feature_importance`, `load_sector_map` | full result |
| Optuna explorer | `load_study_trials_df`, `load_trial_jsonl_records` | — (study-only) |
| Macro state | `load_macro_df` | — (config-only) |
| Positions | (none) | `holdings`, `meta`, `scores` |
| Trades log | (none) | `trades_df` |
| User Guide | (none) | — |
| (sidebar) | `list_studies`, `_load_meta_only` | — |

`result` is loaded **once** in `main()` via `get_result_for_config(label, config)`
(L517-536) and threaded through every tab. Saved-vs-live fallback is
encapsulated there — tabs don't know which path produced the data.

### Files read by multiple tabs (sources of cache/coverage drift)

- `models/cache/macro_signals.parquet` via `load_macro_df`:
  Overview header strip + `tab_macro` body. Single cache key, OK.
- SPY benchmark via `cached_benchmark`:
  Overview equity-curve overlay + Diagnostics drawdown plot. Single
  cache key, OK.
- `models/cache/dashboard_results/<label>/{portfolio,trades}.parquet`
  + `scores.json` + `holdings.json` + `meta.json` via `load_saved_result`:
  loaded once per `(label, config)` in `main()`, threaded as `result`.
  No re-read.

No duplicated loads. Cache layer is well-factored.

### Hardcoded paths / study-name assumptions

All filesystem paths flow through `data_source.path_to(...)` (e.g.
L89, L97, L101, L105, L109, L114, L118, L122 — eight `_*_path()`
helpers, each a one-liner around `data_source.path_to(...)`). No
direct path strings bypassing the abstraction.

Study-name assumptions:
- `LOCKED_BEST_STUDY` constant at L125 — single source for the
  default-selected study in the picker. Was updated cleanly during
  Trial #325 graduation.
- `s.startswith("smoke_")` filter in sidebar (L546) — stable
  convention for hiding throwaway studies.
- `f"best_{study_name}_{trial_number}"` label format (L606, L620,
  L650) — used both to build labels and to filter promoted labels
  (L557). One naming convention spread across three sites — would
  benefit from a `_label_for(study, n)` helper but is not painful
  today.

---

## 3. Abstractions

### Data-load vs render separation

Clean. The 11 `@st.cache_data` functions (L159, 167, 182, 199, 207,
216, 228, 237, 317, 333, 363) all live above the tab functions; tab
bodies call them and render. No `pd.read_parquet` or `json.load`
hidden inside a tab body.

### Reused config-selector pattern

Yes — centralized in `sidebar_config_picker()` (L543-656). It returns
`(label, BacktestConfig, study_name, trial_number)` and `main()`
threads them into every tab. No tab re-implements selection.

The "default config vs best trial of selected study vs custom trial"
toggle is built once in the sidebar (L566-654) with three branches.
Tabs receive a single normalized `(label, config)` pair and don't need
to know which branch produced it.

### Saved-vs-live fallback

`get_result_for_config(label, config)` (L517-536) is the single
gateway — `load_saved_result` first, fall back to `run_live_backtest`
in local mode, friendly-warn-and-`st.stop()` in cloud mode. Clean
separation.

---

## 4. Rough edges

### Functions over ~100 lines

- **`tab_diagnostics` (L1373-1676, 304 lines)** — five clearly-marked
  sections (`# ----- Section 1`, ..., `# ----- Section 4`, plus an
  awkward `# ----- Section 3.5` at L1614 from a later insertion).
  Splitting into five private `_diagnostics_*` helpers would drop
  this to ~50 dispatch lines + 5 × ~50-line section helpers.
  **High-value refactor target IF you intend to edit Diagnostics.**
  Not a gate for adding new tabs.
- **`_notable_observations` (L1148-1370, 223 lines)** — 8-9
  defensive try/except blocks each computing a different bullet for
  the auto-text section in Diagnostics. Each block is independent
  ("alpha gap opening", "drawdown depth", etc.). Could move to its
  own `dashboard_observations.py` with one function per bullet.
  Pure cleanup, no behavioral risk.
- **`sidebar_config_picker` (L543-656, 114 lines)** — single
  cohesive responsibility (build the sidebar selector). The branch
  structure is "Default config / Best trial / Custom" which is
  intrinsic to the UX. Length is OK.
- **`tab_overview` (L663-768, 106 lines)** — borderline. Reasonably
  cohesive (header, equity curve, current macro state, real-portfolio
  hint). Splitting would yield four ~25-line helpers that each only
  exist to be called once. Probably leave alone.

### Magic numbers / hardcoded thresholds

- Plotly chart heights: `240, 360, 380, 420` (L713, L811, L896, L925,
  L1422, L1445, L1900-style). Same `margin=dict(l=10, r=10, t=50, b=10)`
  literal repeated 6+ times. A `_PLOT_HEIGHTS = {"main": 420, ...}`
  + shared margin constant would cut ~12 lines and lock a consistent
  look. Cosmetic, not functional.
- `* 100` for percent conversion appears 14+ times across tabs (L444,
  L696, L702, L707, L1039, L1137, L1216, L1361, etc.). A `pct(x)`
  helper would tidy. Cosmetic.
- Cache TTLs (300, 3600) are correctly chosen and applied per the
  earlier audit — not magic, *intentional*.

### Direct file path strings

**None outside the `_*_path()` helpers.** The `data_source.path_to(...)`
abstraction is used uniformly. This is one of the cleaner aspects of
the file.

### Dead code / commented-out blocks

`grep -E "^\\s*(# TODO|# FIXME|# XXX|# HACK)"` returns **zero hits**.
No commented-out code blocks. Notable for a 1733-line file — usually
these accumulate.

### Inconsistent error handling

19 `st.warning`/`st.error`/`st.info`/`st.stop` calls. Audited a few
samples — pattern is consistent: empty dataframe → `st.warning("Empty
backtest result.")` early-return, missing data in loaders → return
None or `{}` and handle in caller, custom-config in cloud →
`st.warning(...)` + `st.stop()` (L527-533). No exceptions raised
out to the user. Solid.

`_notable_observations` (L1148+) is intentionally defensive — every
bullet wraps in `try/except` so a missing data source silently skips
that bullet rather than killing the whole section. Good pattern.

---

## 5. New tabs would add

### Perturbation / robustness tab

Conceptually: take a saved Optuna result + run a small grid of
nearby configs, plot score distribution, flag fragile axes.

**Existing patterns it slots into**:
- `data_source.path_to(...)` for any new file → trivial
- `@st.cache_data(ttl=300, ...)` for the perturbation-result loader →
  trivial; mirrors `load_saved_result`
- Tab function signature `tab_perturbation(label, config, result)` →
  matches every other tab; just append to `main()` tabs list and
  `with tabs[7]: tab_perturbation(...)`
- Could reuse `cached_benchmark`, `_render_df_with_ticker_links` if
  rendering per-perturbation tables

**New things it'd need (not conflicts)**:
- A loader for whatever file format perturbation results take (most
  likely a JSON or parquet at `models/cache/perturbation_results/<label>/`)
- A new visualization helper for "score distribution across N runs" —
  no current tab has this shape

**Conflict points**: none. The pattern is well-templated.

### Attribution tab (later)

**Existing patterns it could reuse**:
- `_reconstruct_sector_weights` (L1083-1145) — already builds a
  sector-by-time matrix from holdings + sector_map. Attribution by
  sector would consume this directly.
- `load_sector_map` (L228) and `load_feature_importance` (L216) —
  both already in the cache layer.

**New things**: a per-trade attribution loader, plus per-feature
contribution viz. Same shape as Perturbation — load helper + tab
function + entry in `main()`.

**Conflict points**: none. Some _shared_ machinery between
Diagnostics' sector chart and an Attribution tab's sector view —
that's an opportunity to extract a `dashboard_sector_views.py` later
but not a blocker.

### Wiring cost per new tab

```
1. New @st.cache_data loader        ~10 lines
2. New tab_xxx(label, config, result) function       ~50-100 lines
3. main() edits: extend tabs list (1 line) + with tabs[N] (2 lines)
```

Total ~60-115 lines per new tab. Self-contained.

---

## 6. Recommendation

**(a) Keep current structure, add new tabs in the existing pattern.**

### Justification

1. **The cruft is in Diagnostics, not the structure around it.**
   `tab_diagnostics` (304L) + `_notable_observations` (223L) account
   for **30%** of the file's mass. New tabs don't touch either; they
   sit alongside as peers.

2. **Path / cache / dispatch layers are clean.** All filesystem
   access goes through `data_source.path_to(...)`. All loaders carry
   appropriate TTLs (per the prior audit). `main()` is 48 lines and
   just dispatches. No layering violations to clean up before new
   tabs slot in.

3. **The existing tab template is exactly what new tabs need.** Every
   tab is a `def tab_xxx(label, config, result) -> None:` that reads
   from already-loaded helpers and renders. Perturbation and
   Attribution map directly onto this signature.

4. **Light refactor (option b) would be overscoped.** The natural
   refactor target — splitting `tab_diagnostics` and extracting
   `_notable_observations` to its own module — does not gate
   new-tab work. Doing it as a prerequisite would delay the new
   tabs to fix a problem the new tabs don't have.

5. **Rebuild (option c) is wasted effort.** No conceptual misfit
   between current structure and planned tabs. The cruft is local
   and editable in place when you next touch Diagnostics.

### Suggested guardrails when adding the new tabs

- Cap each new tab function at **~120 lines**. If a section grows
  past that, split before merging — don't follow Diagnostics' lead.
- Add the new loader's TTL deliberately (5 min for things that
  change between sessions, 1 hour for static reference data).
- If a new viz pattern (score distribution, attribution waterfall)
  is reused in 2+ places, extract a helper from the start. The
  current file's `_render_df_with_ticker_links` (L256-314) is the
  template — a 59-line helper that earns its keep across tabs.

### When to revisit (later)

If/when you have to edit `tab_diagnostics` for a feature change, do
the section split *then* — that's the cheap moment. Until then it's
working code, well-commented, with clear section markers (`# ----- Section N -----`).

### Out of scope here, but worth noting for any future refactor

- The 8 `_*_path()` one-liner helpers (L88-122) could collapse to
  a `_PATHS = {"db": "models/cache/optuna_studies.db", ...}` dict
  + one `_path(key)` helper. Cosmetic.
- The "Section 3.5" naming in `tab_diagnostics` (L1614) is
  visible in `grep` output and grates a little — rename to
  "Section 4" and renumber (Notable observations → 5) when you
  next touch it.
- The `LOCKED_BEST_TRIAL` constant was just removed for being
  unused; the `LOCKED_BEST_STUDY` constant remains and is
  appropriate. No similar dead code observed today.
