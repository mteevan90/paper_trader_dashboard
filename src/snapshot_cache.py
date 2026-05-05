"""snapshot_cache.py — create / list / delete / inspect frozen cache snapshots.

A snapshot is a directory under models/snapshots/<name>/ with the same
internal layout as models/ (cache/, price_cache/, xgb_model.json,
xgb_model.meta.json). It captures every INPUT cache the backtest needs
at a single moment in time so hypothesis runs are reproducible across
days, even though the live caches auto-refresh by TTL.

Usage:

    python src/snapshot_cache.py create <name> [--comment "..."] [--force]
    python src/snapshot_cache.py list
    python src/snapshot_cache.py inspect <name>
    python src/snapshot_cache.py delete <name>

Snapshots intentionally exclude OUTPUT stores:
    - models/cache/optuna_studies.db
    - models/cache/optuna_trials.jsonl
    - models/cache/dashboard_results/
    - models/cache/feature_importance.json
    - models/cache/finnhub_insider/
The runtime always reads/writes those at the live tree, regardless of
PAPER_TRADER_DATA_ROOT.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_BASE_DIR, ".."))
_LIVE_MODELS_DIR = os.path.join(_REPO_ROOT, "models")
_SNAPSHOTS_DIR = os.path.join(_LIVE_MODELS_DIR, "snapshots")

# Files / directories COPIED from live into a snapshot.
# Tuples of (live_relative_path_from_models_dir, snapshot_relative_path).
# Use a directory name with trailing slash to copy a directory recursively.
_SNAPSHOT_INCLUDE = [
    # Cache files (under models/cache/)
    ("cache/fundamentals.json",                "cache/fundamentals.json"),
    ("cache/earnings_dates.json",              "cache/earnings_dates.json"),
    ("cache/macro_signals.parquet",            "cache/macro_signals.parquet"),
    ("cache/macro_signals.meta.json",          "cache/macro_signals.meta.json"),
    ("cache/analyst_targets.json",             "cache/analyst_targets.json"),
    ("cache/analyst_targets.meta.json",        "cache/analyst_targets.meta.json"),
    ("cache/sector_map.json",                  "cache/sector_map.json"),
    ("cache/feature_matrix.parquet",           "cache/feature_matrix.parquet"),
    ("cache/feature_matrix.meta.json",         "cache/feature_matrix.meta.json"),
    # Per-ticker price parquets (entire directory)
    ("price_cache/",                           "price_cache/"),
    # Trained model + sidecar
    ("xgb_model.json",                         "xgb_model.json"),
    ("xgb_model.meta.json",                    "xgb_model.meta.json"),
]

_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]+$")


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        sys.exit(
            f"[SNAPSHOT] Invalid name: {name!r}\n"
            f"  Names must match ^[a-zA-Z][a-zA-Z0-9_]+$ "
            f"(start with letter, then letters/digits/underscores only).\n"
            f"  No spaces, hyphens, dots, or special characters.\n"
            f"  Recommended convention: <purpose>_<YYYYMMDD>, "
            f"e.g. pre_v3_20260505")


def _git_sha_and_dirty() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        sha = "unknown"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True)
        dirty = bool(status.strip())
    except Exception:
        dirty = False
    return sha, dirty


def _walk_files(root: str) -> list[str]:
    """All file paths under root, relative to root."""
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            out.append(os.path.relpath(full, root).replace("\\", "/"))
    return sorted(out)


def cmd_create(args: argparse.Namespace) -> None:
    name = args.name
    _validate_name(name)
    snap_root = os.path.join(_SNAPSHOTS_DIR, name)

    if os.path.exists(snap_root):
        if not args.force:
            sys.exit(
                f"[SNAPSHOT] {name!r} already exists at {snap_root}\n"
                f"  Pass --force to overwrite, or pick a different name.")
        print(f"[SNAPSHOT] --force: removing existing {snap_root}")
        shutil.rmtree(snap_root)

    os.makedirs(snap_root, exist_ok=True)

    file_records: list[dict] = []
    total_bytes = 0
    skipped: list[str] = []

    for src_rel, dst_rel in _SNAPSHOT_INCLUDE:
        src = os.path.join(_LIVE_MODELS_DIR, src_rel)
        dst = os.path.join(snap_root, dst_rel)

        if src_rel.endswith("/"):
            # Directory copy
            if not os.path.isdir(src):
                skipped.append(src_rel)
                print(f"  [WARN] missing dir, skipping: {src_rel}")
                continue
            os.makedirs(dst, exist_ok=True)
            n_files = 0
            for rel in _walk_files(src):
                src_f = os.path.join(src, rel)
                dst_f = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(dst_f), exist_ok=True)
                shutil.copy2(src_f, dst_f)
                size = os.path.getsize(dst_f)
                total_bytes += size
                file_records.append({
                    "key": (dst_rel + rel).replace("\\", "/"),
                    "size": size,
                    "mtime": os.path.getmtime(dst_f),
                })
                n_files += 1
            print(f"  [COPY] {src_rel} -> {dst_rel} ({n_files} files)")
        else:
            # Single file copy
            if not os.path.isfile(src):
                skipped.append(src_rel)
                print(f"  [WARN] missing file, skipping: {src_rel}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            size = os.path.getsize(dst)
            total_bytes += size
            file_records.append({
                "key": dst_rel.replace("\\", "/"),
                "size": size,
                "mtime": os.path.getmtime(dst),
            })
            print(f"  [COPY] {src_rel:<40} ({size:>12,} bytes)")

    git_sha, git_dirty = _git_sha_and_dirty()
    manifest = {
        "name":             name,
        "created_at":       datetime.now(timezone.utc).isoformat(),
        "git_sha":          git_sha,
        "git_dirty":        git_dirty,
        "comment":          args.comment or "",
        "n_files":          len(file_records),
        "total_size_bytes": total_bytes,
        "skipped":          skipped,
        "files":            file_records,
        "manifest_version": 1,
    }
    manifest_path = os.path.join(snap_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\n[SNAPSHOT] Created {name!r}: "
          f"{len(file_records):,} files, {total_bytes:,} bytes "
          f"({total_bytes/1024/1024:.1f} MB)")
    print(f"[SNAPSHOT] Path: {snap_root}")
    print(f"[SNAPSHOT] git_sha={git_sha} dirty={git_dirty}")
    if skipped:
        print(f"[SNAPSHOT] WARNING: skipped {len(skipped)} expected paths "
              f"(missing in live tree): {skipped}")


def cmd_list(_args: argparse.Namespace) -> None:
    if not os.path.isdir(_SNAPSHOTS_DIR):
        print(f"[SNAPSHOT] No snapshots directory at {_SNAPSHOTS_DIR}")
        return
    entries = sorted(d for d in os.listdir(_SNAPSHOTS_DIR)
                     if os.path.isdir(os.path.join(_SNAPSHOTS_DIR, d)))
    if not entries:
        print(f"[SNAPSHOT] No snapshots in {_SNAPSHOTS_DIR}")
        return
    print(f"[SNAPSHOT] Snapshots in {_SNAPSHOTS_DIR}:")
    print(f"  {'name':<40}  {'created_at':<25}  {'size':>10}  comment")
    print(f"  {'-'*40}  {'-'*25}  {'-'*10}  -------")
    for name in entries:
        manifest_path = os.path.join(_SNAPSHOTS_DIR, name, "manifest.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                created  = m.get("created_at", "?")[:19]
                size_mb  = m.get("total_size_bytes", 0) / 1024 / 1024
                comment  = m.get("comment", "")[:60]
                print(f"  {name:<40}  {created:<25}  {size_mb:>7.1f}MB  {comment}")
            except Exception as e:
                print(f"  {name:<40}  (manifest unreadable: {e})")
        else:
            print(f"  {name:<40}  (no manifest.json)")


def cmd_inspect(args: argparse.Namespace) -> None:
    name = args.name
    snap_root = os.path.join(_SNAPSHOTS_DIR, name)
    manifest_path = os.path.join(snap_root, "manifest.json")
    if not os.path.isfile(manifest_path):
        sys.exit(f"[SNAPSHOT] No manifest at {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    print(json.dumps(m, indent=2, sort_keys=True))


def cmd_delete(args: argparse.Namespace) -> None:
    name = args.name
    _validate_name(name)
    snap_root = os.path.join(_SNAPSHOTS_DIR, name)
    if not os.path.isdir(snap_root):
        sys.exit(f"[SNAPSHOT] Not found: {snap_root}")
    if not args.yes:
        ans = input(f"[SNAPSHOT] Delete {snap_root}? (yes/N): ").strip().lower()
        if ans != "yes":
            print("[SNAPSHOT] Aborted.")
            return
    shutil.rmtree(snap_root)
    print(f"[SNAPSHOT] Deleted {snap_root}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Manage frozen cache snapshots for reproducible "
                    "hypothesis runs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new snapshot from "
                                              "live caches.")
    p_create.add_argument("name", help="Snapshot name. Must match "
                                       "^[a-zA-Z][a-zA-Z0-9_]+$. Suggested "
                                       "convention: <purpose>_<YYYYMMDD>.")
    p_create.add_argument("--comment", default=None,
                          help="Optional human-readable note stored in "
                               "manifest.json.")
    p_create.add_argument("--force", action="store_true",
                          help="Overwrite an existing snapshot of the "
                               "same name.")
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="List existing snapshots.")
    p_list.set_defaults(func=cmd_list)

    p_inspect = sub.add_parser("inspect", help="Print a snapshot's "
                                                "manifest.json.")
    p_inspect.add_argument("name")
    p_inspect.set_defaults(func=cmd_inspect)

    p_delete = sub.add_parser("delete", help="Delete a snapshot directory.")
    p_delete.add_argument("name")
    p_delete.add_argument("--yes", "-y", action="store_true",
                          help="Skip the interactive confirmation.")
    p_delete.set_defaults(func=cmd_delete)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
