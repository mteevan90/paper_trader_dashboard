"""Insert a "Post-LU-v1 dashboard refinements" section into the tracker.

Additive only — inserts a new H1 section between the existing
"Larger Universe v1 study — COMPLETE" section and the "1. Top 5 Most
Important Findings" section. Captures:

- New dashboard tab structure (7 tabs, Overview includes the headline chart)
- Terminology convention ("Reserved validation period" replaces "OOS" in
  human-facing artifacts; schema fields and Python variable names retain
  the technical "OOS" name)
- New architectural reference: docs/architecture/dashboard_operations_v1.md
- Standing follow-ups list (unchanged)

Idempotent against re-runs in spirit (writes a fresh backup each time);
intended as a one-time application.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import docx

TRACKER = Path(__file__).resolve().parents[2] / "docs" / "Project_State_Tracker.docx"
BACKUP = TRACKER.with_suffix(".docx.bak_post_lu_v1_dashboard")
ANCHOR_TEXT = "1. Top 5 Most Important Findings"


def _find_anchor(doc: docx.Document):
    for p in doc.paragraphs:
        if p.style.name == "Heading 1" and p.text == ANCHOR_TEXT:
            return p
    raise SystemExit(f"Anchor heading not found: {ANCHOR_TEXT!r}")


def main() -> None:
    shutil.copy2(TRACKER, BACKUP)
    print(f"Backup written to {BACKUP}")

    doc = docx.Document(str(TRACKER))
    anchor = _find_anchor(doc)

    blocks = [
        ("Heading 1",
         "Post-LU-v1 dashboard refinements — 2026-05-12"),

        ("Normal",
         "After Larger Universe v1 closed, four narrow follow-ups landed "
         "on main as separate feature branches: an add_vline render-bug "
         "fix, a Tuning-tab parity enhancement (with contract additions), "
         "a primary-model selector-default fix, and an Overview-merge + "
         "terminology cleanup. Each was reviewed and merged via the "
         "standard --no-ff branch-policy pattern. The end-state below "
         "captures the current stable shape of the contract-conformant "
         "dashboard."),

        ("Heading 2", "Tab structure (7 tabs)"),
        ("Normal",
         "The contract-conformant dashboard now renders seven universal "
         "tabs. Overview was promoted to executive summary by absorbing "
         "the headline NAV-vs-benchmarks chart from the former Performance "
         "tab; that chart sits directly above the headline metrics and "
         "gives visual context for the numbers that follow. The Performance "
         "tab is gone."),
        ("List Bullet",
         "Overview — date-range header, NAV chart vs benchmarks with the "
         "reserved-validation marker, headline metrics per slice, "
         "concentration check, repeat-holding profile, objective + "
         "construction summary, and an explanatory note defining the "
         "reserved-validation period."),
        ("List Bullet",
         "Holdings — per-model rebalance-date selector → positions table."),
        ("List Bullet",
         "Trades — per-model round-trip trade list."),
        ("List Bullet",
         "Alpha Attribution — per-model top-N alpha contributors with the "
         "25% constraint line marked."),
        ("List Bullet",
         "Diagnostics — IC decomposition (full vs top-quintile), decile "
         "returns with error bars, rolling 12-month win rate."),
        ("List Bullet",
         "Walk-forward — per-window excess CAGR bars (rendered when "
         "walk_forward.parquet is present)."),
        ("List Bullet",
         "Tuning — narrative summary box quoting tuning_summary.json, "
         "score-distribution histogram, running-best convergence curve, "
         "per-parameter sensitivity scatter grid, collapsed trial log, "
         "preserved feature-importance section."),

        ("Heading 2", "Terminology convention"),
        ("Normal",
         "Human-facing artifacts now use \"Reserved validation period\" "
         "(or \"reserved validation window\" / \"reserved validation slice\") "
         "where they previously used \"OOS\". The phrase \"out-of-sample\" "
         "remains correct when the surrounding context distinguishes "
         "model-level out-of-sample from analyst-level analysis discretion. "
         "Schema field names and Python code variables retain the technical "
         "\"oos_*\" names — the schema is short, technical, developer-facing; "
         "the UI is partner-facing and prioritizes legibility."),
        ("List Bullet",
         "meta.json.windows.oos_start / oos_end — unchanged"),
        ("List Bullet",
         "meta.json.summary_metrics.oos — unchanged"),
        ("List Bullet",
         "Python variables (oos_start, oos_ms, etc.) — unchanged"),
        ("List Bullet",
         "Dashboard UI labels — \"Reserved validation period\""),
        ("List Bullet",
         "Architectural memos and writeup narrative — \"reserved "
         "validation\" or contextually appropriate variant"),
        ("Normal",
         "The terminology convention is formalized in "
         "docs/architecture/dashboard_contract_v1.md under the \"Terminology "
         "convention — schema fields vs UI labels\" section."),

        ("Heading 2", "New architectural reference"),
        ("Normal",
         "docs/architecture/dashboard_operations_v1.md is a how-to-work-on-"
         "the-dashboard reference for future sessions — part tutorial, "
         "part architecture overview. Covers the legacy-vs-contract-"
         "conformant routing, cached loaders, role-aware model defaults, "
         "and four recipes (fix a render bug, add a chart to an existing "
         "tab, add a new optional contract artifact + section, add a "
         "whole new tab). Grounded in concrete examples from this "
         "project's history (tuning_convergence additive change, "
         "add_vline ms-since-epoch fix, primary-model-default fix, "
         "Overview merge + terminology cleanup)."),

        ("Heading 2", "Standing follow-ups (unchanged)"),
        ("Normal",
         "All tracked in the LU-v1 session log; none urgent. Mike decides "
         "when/whether to action."),
        ("List Bullet",
         "use_container_width deprecation sweep — ~40 Streamlit warnings "
         "per page load; mechanical replace with width=\"stretch\" / "
         "width=\"content\". Affects both legacy and contract code."),
        ("List Bullet",
         "Dashboard pytest coverage via streamlit.testing.v1.AppTest — "
         "no existing pytest coverage; harness pattern is the right shape "
         "for formal tests."),
        ("List Bullet",
         "attempted_trials enhancement to tuning_summary.json — current "
         "total_trials is COMPLETE-only; future studies with high failure "
         "rates would benefit from surfacing the attempted-vs-complete split."),
        ("List Bullet",
         "Convergence-pattern methodology memo — pending a third Optuna "
         "data point. XGBoost on LU-v1 plateaued at 61% of 200 trials; "
         "legacy v1 plateaued at ~33% of 1000 trials. A third study with "
         "documented convergence promotes this observation to an "
         "architectural memo."),

        ("Heading 2", "Merge SHAs (for traceability)"),
        ("List Bullet",
         "727bc0d — add_vline ms-since-epoch fix"),
        ("List Bullet",
         "b486e65 — Tuning-tab parity enhancement (tuning_convergence + "
         "tuning_summary contract additions)"),
        ("List Bullet",
         "235c8a6 — primary-model selector-default fix"),
        ("List Bullet",
         "Overview merge + terminology cleanup — pre-merge at time of "
         "this tracker entry; SHA filled in on the post-merge session-log "
         "entry."),
    ]

    for style, text in blocks:
        anchor.insert_paragraph_before(text=text, style=style)

    doc.save(str(TRACKER))
    print(f"Updated tracker saved to {TRACKER}.")


if __name__ == "__main__":
    main()
