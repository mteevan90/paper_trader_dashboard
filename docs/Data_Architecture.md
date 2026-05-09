# Paper Trader — Data Architecture & Archival Strategy

A reference document for understanding what data the project accumulates, what's worth preserving, and how to think about archival as the project grows.

**Status:** Reference only. No implementation has been done. This document captures the design thinking; future Claude Code sessions will implement archival infrastructure when storage pressure makes it worthwhile.

**Last updated:** May 7, 2026

---

## Why this document exists

Every Optuna study run produces several layers of data, ranging from trivially small (megabytes) to potentially large (gigabytes). At the current pace of research — a few studies per week — local disk can absorb everything for months. But the data is also genuinely valuable:

- The trial-level data (every config TPE has tested across all studies) is research-grade material. With enough studies, you can do meta-analyses that no single study can answer.
- The snapshots define reproducibility. A study from 6 months ago can only be re-run if the input data it used is preserved.
- The backtest outputs power the dashboard. Lose them, lose the user-facing artifact.

This document defines what's important, what's disposable, and what to do when storage starts to matter.

---

## What gets generated, by tier

### Tier 1 — Research-critical (~50-100 MB at scale)

Must be preserved indefinitely. Lose this and you lose the project's accumulated research output.

| Path | Contents | Notes |
|---|---|---|
| `models/cache/optuna_studies.db` | SQLite database of every trial across every study — params, scores, states, timestamps | Single file. Grows ~1-2 MB per 1000-trial study. |
| `models/cache/optuna_trials.jsonl` | Same data as the SQLite DB but newline-JSON format | Portable. Useful for cross-study queries that don't fit Optuna's API. |
| `models/cache/dashboard_results/*/meta.json` | Per-study config + headline results | Tiny (~50 KB each). The "what was this study" record. |
| Git history (GitHub) | All code commits | Already preserved off-machine. |

### Tier 2 — Reproducibility (~2-3 GB at scale)

Should be preserved so old studies can be re-run. Without these, the trial database tells you what was searched, but you can't reconstruct what the strategy actually did.

| Path | Contents | Per-study size |
|---|---|---|
| `models/snapshots/<snapshot_name>/` | Cached price data, fundamentals, feature matrices, macro signals, sector maps | ~500 MB per snapshot; you have ~1-2 distinct snapshots |
| `models/cache/dashboard_results/*/portfolio.parquet` | Daily portfolio value, cash, position count | ~70 KB |
| `models/cache/dashboard_results/*/trades.parquet` | One row per BUY/SELL/STOP event | ~10 KB |
| `models/cache/dashboard_results/*/scores.json` | Per-rebalance scoring data | ~100 KB |
| `models/cache/dashboard_results/*/holdings.json` | Final holdings | ~1 KB |
| `models/cache/dashboard_results/*/sizing_decisions.parquet` | Per-rebalance sizing log (continuous-sizing studies and forward) | ~5-10 KB |
| `models/cache/dashboard_results/*/SPY_close.parquet`, `QQQ_close.parquet` | Per-study benchmark data (added May 7 to fix cloud yfinance issue) | ~5 KB each |

### Tier 3 — Debugging utility (~10-15 GB at scale)

Useful for diagnosing problems but rarely needed once a study has produced its summary. Compresses ~5-10x with standard tools.

| Path | Contents | Per-study size |
|---|---|---|
| `models/cache/parallel_logs/<study_name>/worker_*.log` | Full stdout from each of the 8 workers during the study run | 50-100 MB per worker × 8 workers = 400-800 MB per study |

### Tier 4 — Throwaway

Already covered by other paths. Disposable.

| Path | Why disposable |
|---|---|
| `*_v1.log`, `*_v1_rescore.log` at repo root | Duplicates of `parallel_logs/<study_name>/` content |
| Worker stdout files older than the latest run | Optuna resume mechanic means old worker stdout from a failed-and-restarted study is irrelevant |
| `*.tmp`, `*.bak`, scratch Python files | Already in .gitignore |

---

## Total size projections

Based on the current rate (~1-2 studies/week) and current per-study artifact sizes:

| Time horizon | Studies completed | Total disk usage | Storage state |
|---|---|---|---|
| Today | 4 | ~3-4 GB | Local disk fine |
| 3 months | 15-25 | ~10-15 GB | Local disk fine, archive script worth building |
| 6 months | 30-50 | ~25-40 GB | Mix of local + NAS makes sense |
| 1 year | 50-100 | ~50-100 GB | NAS-primary with local cache for active research |

The dominant size driver is parallel_logs (Tier 3). The actual research output (Tiers 1 + 2) stays small even at year-scale — under 5 GB total.

---

## Recommended action by stage

### Stage 1 — Now (under 5 GB total)

Do nothing. Local disk handles it. Don't build archival infrastructure prematurely.

### Stage 2 — Around 5-10 GB total

Build a simple compression script. After each study finishes, compress its `parallel_logs/<study_name>/` directory to a `.zip` or `.7z`. Drops log sizes from ~500 MB per study to ~50 MB per study. Single Claude Code session, ~30 minutes work.

### Stage 3 — Around 10-15 GB total

Build a one-shot NAS archive script (`archive_to_nas.py`). Behavior:

- Copies Tier 1+2 to NAS (research data + snapshots) on demand
- Optionally copies Tier 3 (compressed logs) for debugging access
- Skips Tier 4 (throwaway)
- Skips anything that's already in git
- Writes a manifest to NAS showing what's there with timestamps

Single Claude Code session, ~60 minutes work.

### Stage 4 — Around 25+ GB total

NAS becomes primary storage. Local PC keeps:

- The current snapshot you're actively running studies against
- The most recent 5-10 studies' dashboard_results
- The optuna_studies.db (small enough to keep local always)

NAS holds everything else. Active studies pull data from NAS as needed; completed studies push to NAS automatically. Probably needs a more polished sync layer, ~half-day Claude Code session.

### Stage 5 — Cross-study analysis

Once you have ~10+ studies on disk + NAS, build a "studies index" — a dashboard tab or analysis script that asks cross-study questions:

- "Across all studies, which tunables most affect rolling_12mo_objective?"
- "Which snapshot version produced the most robust validation results?"
- "Has TPE converged on similar configs across different starting points, or has the search space been fully explored?"

This is the highest-leverage research output of having all this data. The infrastructure to do it depends on having the data already preserved cleanly.

---

## Things to keep in mind

### The actually valuable data is tiny

`optuna_studies.db` is what makes meta-research possible. Even at 100 studies, it'll be under 200 MB. That's small enough to back up to multiple places: GitHub releases, NAS, cloud storage, an external SSD. Treat it as the project's most important research asset and protect it accordingly.

### Snapshots are version-pinned

Each snapshot is named for its creation date and tied to a specific commit's data layer. Don't delete an old snapshot if any saved study used it. The `meta.json` for each study records which snapshot it ran against; that becomes a dependency graph if you ever rebuild from scratch.

### Compression is cheap, decompression is too

Standard zip/7z compression on parallel_logs gets 5-10x reduction. The only reason not to compress is the small inconvenience of `unzip` when you actually want to read a worker log — and most of the time you don't. The .docx skill repository, R2 sync, and any future archive layer should always operate on compressed data.

### Git LFS is overkill

Snapshots are large (~500 MB). Putting them in git via Git LFS would technically work but:

- LFS bandwidth is expensive
- Snapshots are reproducible from yfinance + the cache fetch logic, just slowly
- The reproducibility need is rare in practice (only if NAS + local both fail)

Leave snapshots out of git. Document in a future setup doc how to rebuild them from scratch if needed.

### The dashboard's R2 sync is separate from any archive

R2 holds the cloud dashboard's data — promoted study results + snapshots needed at view time. That's a *display* concern, not an *archive* concern. The archive system needs to preserve everything the dashboard sees plus more (failed studies, experimental studies, raw logs). Don't conflate them.

---

## Open questions to revisit later

- **What's the right NAS folder structure?** Probably mirror the local `models/` tree, but with a date-based snapshot naming so multiple project iterations can coexist.
- **Should studies be archived as immutable bundles?** A "study bundle" might be a single .zip containing all of Tier 1+2 for one study, with a manifest. Easier to move around, harder to query without unpacking.
- **Cross-study queries — Python script vs dashboard tab?** Probably both eventually. A standalone analysis script for one-off questions; a dashboard tab for ongoing comparison views the family can use.
- **Disaster recovery story.** If local disk dies, what does recovery look like? Probably: clone repo, pull snapshots from NAS, rebuild active venv, resume.

---

## Concrete TODOs (when storage pressure makes them worthwhile)

In rough priority order:

1. Add log compression to study workflow (Stage 2 trigger)
2. Write `archive_to_nas.py` with the tier system above (Stage 3 trigger)
3. Document the snapshot rebuild procedure (whenever you have time)
4. Build cross-study analysis script (after 10+ studies)
5. Build studies-index dashboard tab (after cross-study script proves useful)

Each of these is a single Claude Code session of ~30-90 minutes. None are urgent. Doing them prematurely is wasted effort.
