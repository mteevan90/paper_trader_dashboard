"""snapshot_sp1500.py — freeze the sp1500 caches into pre_v3_sp1500_<date>/.

Run AFTER fetch_sp1500.py + train_sp1500.py. Creates a fresh snapshot
under models/snapshots/pre_v3_sp1500_<YYYYMMDD>/ that mirrors the live
tree at the moment of the snapshot. Coexists with pre_v2_20260505 so old
studies (#325 etc.) stay reproducible against the legacy 490-ticker
snapshot.

Includes:
  - cache/fundamentals.json, earnings_dates.json, sector_map.json,
    analyst_targets.json + .meta.json, feature_matrix.parquet + .meta.json
  - cache/macro_signals.parquet + .meta.json (copied forward from
    pre_v2_20260505 — macro signals are universe-independent)
  - price_cache/<TICKER>.parquet for all SP1500_TICKERS
  - xgb_model.json (the sp1500 retrained model — saved under this name
    so the existing runtime that reads MODEL_PATH = .../xgb_model.json
    Just Works in snapshot mode)
  - xgb_model_sp1500.json (same content, named for clarity in the manifest)
  - xgb_model.meta.json (sidecar with universe_label='SP1500')
  - manifest.json (file inventory + git_sha + universe_label)

Usage (PowerShell):
    venv\\Scripts\\python.exe src\\snapshot_sp1500.py [--name pre_v3_sp1500_<YYYYMMDD>]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
LIVE_MODELS_DIR = os.path.join(REPO_ROOT, "models")
SNAPSHOTS_DIR   = os.path.join(LIVE_MODELS_DIR, "snapshots")
PRE_V2_SNAPSHOT = os.path.join(SNAPSHOTS_DIR, "pre_v2_20260505")

# Files copied from the live tree into the snapshot.
# Each tuple is (live_relative_to_models, snapshot_relative).
_INCLUDE = [
    ("cache/fundamentals.json",         "cache/fundamentals.json"),
    ("cache/earnings_dates.json",       "cache/earnings_dates.json"),
    ("cache/analyst_targets.json",      "cache/analyst_targets.json"),
    ("cache/analyst_targets.meta.json", "cache/analyst_targets.meta.json"),
    ("cache/sector_map.json",           "cache/sector_map.json"),
    ("cache/feature_matrix.parquet",    "cache/feature_matrix.parquet"),
    ("cache/feature_matrix.meta.json",  "cache/feature_matrix.meta.json"),
    ("price_cache/",                    "price_cache/"),
    # The sp1500 model is saved twice in the snapshot — once under the
    # canonical xgb_model.json name (so backtest.py's lookahead-bias guard
    # and load_model() Just Work in snapshot mode), once under
    # xgb_model_sp1500.json for clarity in the manifest. Same bytes either
    # way; the dual save happens below in cmd_create() so we don't depend
    # on the live tree having both files.
]

# Files copied from pre_v2_20260505 (macro signals are universe-
# independent, so re-using the v2 numbers is correct + much faster than
# refetching). If you ever rebuild macro signals, regenerate the live
# tree's copy first and they'll naturally flow through this snapshot too.
_COPY_FORWARD_FROM_V2 = [
    ("cache/macro_signals.parquet",   "cache/macro_signals.parquet"),
    ("cache/macro_signals.meta.json", "cache/macro_signals.meta.json"),
]


def _git_sha_and_dirty() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        sha = "unknown"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True)
        dirty = bool(status.strip())
    except Exception:
        dirty = False
    return sha, dirty


def _walk_files(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            out.append(os.path.relpath(full, root).replace("\\", "/"))
    return sorted(out)


def _copy_with_record(src: str, dst: str) -> tuple[int, list[dict]]:
    """Copy src->dst. If src is a directory (trailing slash), recurse.
    Returns (bytes_written, file_records[])."""
    records: list[dict] = []
    total = 0
    if src.endswith(os.sep) or src.endswith("/"):
        if not os.path.isdir(src):
            print(f"  [WARN] missing dir: {src}")
            return 0, []
        os.makedirs(dst, exist_ok=True)
        for rel in _walk_files(src):
            sf = os.path.join(src, rel)
            df = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(df), exist_ok=True)
            shutil.copy2(sf, df)
            sz = os.path.getsize(df)
            total += sz
            records.append({"key": (os.path.basename(dst.rstrip(os.sep))
                                    + "/" + rel).replace(os.sep, "/"),
                            "size": sz})
        return total, records
    if not os.path.isfile(src):
        print(f"  [WARN] missing file: {src}")
        return 0, []
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    sz = os.path.getsize(dst)
    return sz, [{"key": os.path.basename(dst).replace(os.sep, "/"),
                 "size": sz}]


def cmd_create(name: str, force: bool, comment: str) -> int:
    snap_root = os.path.join(SNAPSHOTS_DIR, name)
    if os.path.exists(snap_root):
        if not force:
            sys.exit(f"[SNAPSHOT] {name!r} exists at {snap_root}. "
                     f"Pass --force to overwrite.")
        print(f"[SNAPSHOT] --force: removing {snap_root}")
        shutil.rmtree(snap_root)
    os.makedirs(snap_root, exist_ok=True)

    file_records: list[dict] = []
    total_bytes = 0
    skipped: list[str] = []

    print(f"[SNAPSHOT_SP1500] Creating snapshot at {snap_root}\n")

    # 1. Copy fresh live caches.
    print(f"[1/3] Copying live caches...")
    for src_rel, dst_rel in _INCLUDE:
        src = os.path.join(LIVE_MODELS_DIR, src_rel)
        dst = os.path.join(snap_root, dst_rel)
        sz, recs = _copy_with_record(src, dst)
        if sz == 0 and not recs:
            skipped.append(src_rel)
            continue
        total_bytes += sz
        file_records.extend(recs)
        print(f"  [COPY] {src_rel:<40} ({sz:>14,} bytes / {len(recs)} files)")

    # 2. Copy macro signals forward from pre_v2_20260505.
    print(f"\n[2/3] Copying macro signals forward from pre_v2_20260505...")
    if not os.path.isdir(PRE_V2_SNAPSHOT):
        print(f"  [WARN] pre_v2_20260505 missing at {PRE_V2_SNAPSHOT} — "
              f"the new snapshot will not have macro signals. Aborting "
              f"so the snapshot is not silently incomplete.")
        sys.exit(1)
    for src_rel, dst_rel in _COPY_FORWARD_FROM_V2:
        src = os.path.join(PRE_V2_SNAPSHOT, src_rel)
        dst = os.path.join(snap_root, dst_rel)
        if not os.path.isfile(src):
            print(f"  [WARN] missing in pre_v2: {src_rel}")
            skipped.append(f"v2:{src_rel}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        sz = os.path.getsize(dst)
        total_bytes += sz
        file_records.append({"key": dst_rel.replace(os.sep, "/"), "size": sz})
        print(f"  [COPY] (from pre_v2) {src_rel:<28} "
              f"({sz:>14,} bytes)")

    # 3. Save the sp1500 model under BOTH names (xgb_model.json so the
    #    runtime in snapshot mode just works, and xgb_model_sp1500.json
    #    so the manifest reflects which model is in the snapshot).
    print(f"\n[3/3] Embedding sp1500 model (dual save)...")
    sp1500_model_src = os.path.join(LIVE_MODELS_DIR, "xgb_model_sp1500.json")
    sp1500_meta_src  = os.path.join(LIVE_MODELS_DIR, "xgb_model_sp1500.meta.json")
    if not os.path.isfile(sp1500_model_src):
        print(f"  [ABORT] {sp1500_model_src} not found. "
              f"Run train_sp1500.py first.")
        sys.exit(1)

    for dst_name in ("xgb_model.json", "xgb_model_sp1500.json"):
        dst = os.path.join(snap_root, dst_name)
        shutil.copy2(sp1500_model_src, dst)
        sz = os.path.getsize(dst)
        total_bytes += sz
        file_records.append({"key": dst_name, "size": sz})
        print(f"  [COPY] {dst_name:<30} ({sz:>14,} bytes)")

    # Sidecar — also under both names for symmetry.
    if os.path.isfile(sp1500_meta_src):
        for dst_name in ("xgb_model.meta.json", "xgb_model_sp1500.meta.json"):
            dst = os.path.join(snap_root, dst_name)
            shutil.copy2(sp1500_meta_src, dst)
            sz = os.path.getsize(dst)
            total_bytes += sz
            file_records.append({"key": dst_name, "size": sz})
            print(f"  [COPY] {dst_name:<30} ({sz:>14,} bytes)")
    else:
        print(f"  [WARN] {sp1500_meta_src} not found — "
              f"snapshot's xgb_model.meta.json missing.")

    # 4. Manifest.
    git_sha, git_dirty = _git_sha_and_dirty()
    manifest = {
        "name":             name,
        "created_at":       datetime.now(timezone.utc).isoformat(),
        "git_sha":          git_sha,
        "git_dirty":        git_dirty,
        "comment":          comment,
        "universe_label":   "SP1500",
        "model_filename":   "xgb_model_sp1500.json (also at xgb_model.json)",
        "n_files":          len(file_records),
        "total_size_bytes": total_bytes,
        "skipped":          skipped,
        "files":            file_records,
        "manifest_version": 2,
    }
    manifest_path = os.path.join(snap_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\n[SNAPSHOT_SP1500] Done.")
    print(f"  Path:       {snap_root}")
    print(f"  Files:      {len(file_records):,}")
    print(f"  Total size: {total_bytes:,} bytes "
          f"({total_bytes / 1024 / 1024:.1f} MB)")
    print(f"  git_sha:    {git_sha} (dirty={git_dirty})")
    if skipped:
        print(f"  WARNING:    skipped {len(skipped)} expected paths "
              f"(missing in live tree): {skipped}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_name = f"pre_v3_sp1500_{datetime.now().strftime('%Y%m%d')}"
    parser.add_argument("--name", default=default_name,
                        help=f"Snapshot directory name "
                             f"(default: {default_name}).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing snapshot of the same name.")
    parser.add_argument("--comment", default="SP1500 universe expansion. "
                        "Coexists with pre_v2_20260505 (used by #325 etc.).",
                        help="Optional human-readable note in manifest.json.")
    args = parser.parse_args()
    return cmd_create(args.name, args.force, args.comment)


if __name__ == "__main__":
    sys.exit(main())
