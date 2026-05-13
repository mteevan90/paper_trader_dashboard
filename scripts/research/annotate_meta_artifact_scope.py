"""One-shot script: annotate existing contract-conformant meta.json files
with `artifact_metadata` scope information.

Adds `artifact_metadata` entries for `ic_decomposition.parquet` and
`decile_returns.parquet`:

  - v1 (`larger_universe_v1`) → scope: "held_subset" (per ic_scope_audit.md)
  - v2 (`larger_universe_v2/<variant>` for all 7 variants) → scope:
    "full_cross_section" (per phase5_analytics_v2.py's load-prices design)

Idempotent: if `artifact_metadata` already exists in a meta.json, the
script updates only the affected artifact entries without disturbing
other entries.

Does NOT modify any parquet artifacts. Only meta.json overlays.

Run:
    python scripts/research/annotate_meta_artifact_scope.py
    python scripts/research/annotate_meta_artifact_scope.py --dry-run

This is the migration step for existing studies. Newly-generated meta.json
files from `phase5_analytics_v2.py` write `artifact_metadata` directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDIES_DIR = ROOT / "models" / "studies"

V1_HELD_SUBSET_NOTE = (
    "Prices loaded for held tickers (~450 across XGBoost and ElasticNet "
    "holdings) rather than the full eligible universe (~1,963 tickers). "
    "Cross-sectional statistics computed at this scope are not the "
    "standard interpretation; see audit_reference."
)
V2_FULL_SCOPE_NOTE = (
    "Prices loaded for the full eligible universe (~1,963 tickers) at "
    "computation time. Standard cross-sectional interpretation."
)

V1_AUDIT_REF = "docs/studies/larger_universe_v1/ic_scope_audit.md"

SCOPE_SENSITIVE_ARTIFACTS = (
    "ic_decomposition.parquet",
    "decile_returns.parquet",
)


def _patch_meta(meta_path: Path, scope: str, description: str,
                audit_reference: str | None) -> bool:
    """Patch a single meta.json with artifact_metadata entries. Returns True
    if a change was made (and written), False if the file already had the
    desired state."""
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    am = meta.get("artifact_metadata") or {}
    changed = False
    for artifact in SCOPE_SENSITIVE_ARTIFACTS:
        desired = {
            "scope": scope,
            "scope_description": description,
            "audit_reference": audit_reference,
        }
        if am.get(artifact) != desired:
            am[artifact] = desired
            changed = True
    if changed:
        meta["artifact_metadata"] = am
        meta_path.write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8",
        )
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be patched; don't write.")
    args = parser.parse_args()

    targets: list[tuple[Path, str, str, str | None]] = []

    # v1
    v1_meta = STUDIES_DIR / "larger_universe_v1" / "contract_v1" / "meta.json"
    if v1_meta.exists():
        targets.append((v1_meta, "held_subset", V1_HELD_SUBSET_NOTE, V1_AUDIT_REF))

    # v2 variants (everything under larger_universe_v2/<variant>/contract_v1)
    v2_root = STUDIES_DIR / "larger_universe_v2"
    if v2_root.exists():
        for sub in sorted(v2_root.iterdir()):
            if not sub.is_dir() or sub.name in {"comparison"}:
                continue
            meta_path = sub / "contract_v1" / "meta.json"
            if meta_path.exists():
                targets.append((meta_path, "full_cross_section", V2_FULL_SCOPE_NOTE, None))

    if not targets:
        print("No meta.json files found to annotate.")
        return 0

    print(f"Found {len(targets)} meta.json files to evaluate.")
    n_changed = 0
    for meta_path, scope, description, audit_ref in targets:
        rel = meta_path.relative_to(ROOT)
        if args.dry_run:
            # Just inspect
            cur = json.loads(meta_path.read_text(encoding="utf-8"))
            cur_am = (cur.get("artifact_metadata") or {})
            needs_change = any(
                cur_am.get(a, {}).get("scope") != scope
                for a in SCOPE_SENSITIVE_ARTIFACTS
            )
            status = "WOULD PATCH" if needs_change else "already correct"
            print(f"  [{status}] {rel} -> scope={scope}")
        else:
            changed = _patch_meta(meta_path, scope, description, audit_ref)
            status = "PATCHED" if changed else "no-op (already correct)"
            print(f"  [{status}] {rel} -> scope={scope}")
            if changed:
                n_changed += 1

    if args.dry_run:
        print("\nDRY RUN — no files modified.")
    else:
        print(f"\nDone. {n_changed} file(s) modified, {len(targets) - n_changed} no-op.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
