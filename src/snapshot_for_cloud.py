"""snapshot_for_cloud.py — bundle dashboard data + upload to R2.

Run after any local change you want reflected on the cloud dashboard:

    cd src
    python snapshot_for_cloud.py [--asset-class equities|crypto] [--dry-run]

What it does:
  1. Collects every file in data_source.R2_LAYOUT_SUFFIX plus every file
     under <asset_class>'s dashboard_results/<label>/ subtree.
  2. Uploads each to its corresponding asset-class-prefixed R2 key
     (mapping defined in data_source.r2_key_for — single source of truth
     shared with the reader).
  3. Reads the prior snapshot_manifest.json from R2 (if any) and DELETES
     any files no longer in this snapshot — keeps the bucket as a clean
     single-snapshot mirror with no stale leftover files. Delete is
     scoped to the asset_class prefix so a crypto sync does not erase
     equity keys (and vice versa).
  4. Writes a fresh snapshot_manifest.json to the bucket root with
     asset_class, generated_at (UTC ISO), git SHA, file list, total size.

Asset class:
  Defaults to "equities" so existing usage is unchanged. Phase 2 will
  invoke with --asset-class crypto.

Dry-run:
  --dry-run prints every key that WOULD be uploaded / deleted without
  contacting R2. No credentials required for a dry run.

Bucket layout: see ``data_source.py`` docstring (the canonical source).

Credentials: read from ../.env via python-dotenv. Required env vars:
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_ENDPOINT_URL
    R2_BUCKET_NAME
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from data_source import (R2_LAYOUT_SUFFIX, MANIFEST_KEY, REPO_ROOT,
                         DEFAULT_ASSET_CLASS, SUPPORTED_ASSET_CLASSES,
                         _local_path_for_asset, r2_key_for)


load_dotenv(REPO_ROOT / ".env")

R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL      = os.getenv("R2_ENDPOINT_URL")
R2_BUCKET_NAME       = os.getenv("R2_BUCKET_NAME")


def _check_creds() -> None:
    missing = [k for k, v in (
        ("R2_ACCESS_KEY_ID",     R2_ACCESS_KEY_ID),
        ("R2_SECRET_ACCESS_KEY", R2_SECRET_ACCESS_KEY),
        ("R2_ENDPOINT_URL",      R2_ENDPOINT_URL),
        ("R2_BUCKET_NAME",       R2_BUCKET_NAME),
    ) if not v]
    if missing:
        raise SystemExit(f"Missing R2 env vars: {missing}. Add to ../.env")


def _client():
    # Lazy import: dry-runs don't need boto3 + don't need credentials.
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def _ensure_feature_importance_json(asset_class: str) -> None:
    """Generate <asset>'s feature_importance.json from the trained
    XGBoost model + feature column list. Cheap regen on every snapshot —
    model load is fast and the file is ~1 KB. The dashboard's
    Diagnostics tab reads this for the top-features-by-importance bar
    chart.

    Equity uses the legacy flat path; non-equity asset classes write
    under their namespaced subdir."""
    if asset_class != "equities":
        # Phase 2 will own its own model; this helper is currently
        # equity-specific (imports model.load_model() which loads the
        # equity XGBoost). Skip cleanly for non-equity classes so a
        # crypto sync doesn't try to load an equity model file.
        print(f"[SNAPSHOT] Skipping feature_importance.json for "
              f"asset_class={asset_class} (not equity).")
        return
    from model import FEATURE_COLS, load_model
    model = load_model()
    importances = model.feature_importances_
    data = {
        "features": [
            {"name": str(name), "importance": float(imp)}
            for name, imp in zip(FEATURE_COLS, importances)
        ]
    }
    out = Path(_local_path_for_asset(
        "models/cache/feature_importance.json", asset_class))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[SNAPSHOT] Wrote feature importance: "
          f"{out.relative_to(REPO_ROOT)}")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return "unknown"


def _collect_files(asset_class: str) -> list[tuple[Path, str]]:
    """Return list of (local_absolute_path, r2_key) for every file the
    cloud dashboard reads, scoped to ``asset_class``. Skips missing
    files with a warning."""
    pairs: list[tuple[Path, str]] = []

    # Static layout entries — local path resolution is asset-class
    # aware (equities = legacy flat; others = namespaced subdir).
    for local_rel in R2_LAYOUT_SUFFIX.keys():
        local = Path(_local_path_for_asset(local_rel, asset_class))
        if local.exists():
            r2_key = r2_key_for(local_rel, asset_class=asset_class)
            pairs.append((local, r2_key))
        else:
            print(f"  [WARN] missing local file, skipping: "
                  f"{local.relative_to(REPO_ROOT)}")

    # Dashboard results: every file under each <label>/ subdir.
    dr_root = Path(_local_path_for_asset(
        "models/cache/dashboard_results", asset_class))
    if dr_root.exists():
        for label_dir in sorted(dr_root.iterdir()):
            if not label_dir.is_dir():
                continue
            for f in sorted(label_dir.iterdir()):
                if not f.is_file():
                    continue
                # Reconstruct the canonical "models/cache/..." form so
                # r2_key_for can prefix it correctly.
                # The on-disk path may already include <asset_class>/
                # for non-equity classes — strip it to get back to the
                # canonical prefix.
                rel = f.relative_to(REPO_ROOT).as_posix()
                if (asset_class != "equities"
                        and rel.startswith(f"models/cache/{asset_class}/")):
                    canonical = rel.replace(
                        f"models/cache/{asset_class}/",
                        "models/cache/", 1)
                else:
                    canonical = rel
                pairs.append((f, r2_key_for(canonical,
                                            asset_class=asset_class)))
    else:
        print(f"  [WARN] no dashboard_results directory at {dr_root}")

    return pairs


def _list_remote_keys(client, asset_class: str) -> set[str]:
    """List all keys in the bucket scoped to the given asset_class
    prefix. The shared snapshot_manifest.json at the bucket root is
    intentionally excluded — it's not asset-prefixed."""
    keys: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET_NAME,
                                   Prefix=f"{asset_class}/"):
        for obj in page.get("Contents", []) or []:
            keys.add(obj["Key"])
    return keys


def _is_dashboard_meta(r2_key: str,
                       asset_class: str) -> tuple[bool, str | None]:
    """Return (is_meta, label) for keys of shape
    <asset_class>/dashboard_results/<label>/meta.json. (False, None)
    otherwise."""
    parts = r2_key.split("/")
    if (len(parts) == 4
            and parts[0] == asset_class
            and parts[1] == "dashboard_results"
            and parts[3] == "meta.json"):
        return True, parts[2]
    return False, None


def _meta_with_promoted(local: Path, label: str) -> tuple[bytes, bool]:
    """Read a dashboard_results meta.json; return (bytes_to_upload,
    promoted_flag).

    If "promoted" is present, honor it verbatim (return original bytes).
    If absent, default-fill to False — every label is experimental until
    explicitly promoted, no hardcoded exceptions. To graduate a study,
    set "promoted": true in its local meta.json (the saver already does
    this for save_dashboard_results; hypothesis runs save with
    promoted=False and require manual flip).

    The local file is NOT modified — only the snapshot upload is augmented.
    """
    with open(local, "rb") as f:
        original = f.read()
    meta = json.loads(original)
    if "promoted" in meta:
        return original, bool(meta["promoted"])
    meta["promoted"] = False
    return (json.dumps(meta, indent=2).encode("utf-8"), False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset-class", default=DEFAULT_ASSET_CLASS,
                        choices=list(SUPPORTED_ASSET_CLASSES),
                        help="Which asset class to sync. Defaults to "
                             "'equities' (preserves pre-Phase-1 behavior).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every key that WOULD be uploaded / "
                             "deleted without contacting R2. Skips "
                             "credential checks and boto3 entirely. Use "
                             "this to verify R2 key shapes before a real "
                             "sync.")
    args = parser.parse_args()
    asset_class = args.asset_class
    dry_run = args.dry_run

    if not dry_run:
        _check_creds()
        client = _client()
        print(f"[SNAPSHOT] Bucket: {R2_BUCKET_NAME}")
        print(f"[SNAPSHOT] Endpoint: {R2_ENDPOINT_URL}")
    else:
        client = None
        print(f"[SNAPSHOT] *** DRY RUN *** No R2 calls will be made.")
    print(f"[SNAPSHOT] asset_class = {asset_class}")

    _ensure_feature_importance_json(asset_class)

    pairs = _collect_files(asset_class)
    new_keys = {key for _, key in pairs}
    print(f"[SNAPSHOT] Collected {len(pairs)} files for upload.")

    # Diff against current bucket contents (scoped to this asset_class
    # prefix — a crypto sync must NOT delete equity keys, and vice versa).
    if dry_run:
        existing: set[str] = set()
        print(f"[SNAPSHOT] (dry-run) skipping bucket list; treating "
              f"bucket as empty for diff purposes.")
    else:
        existing = _list_remote_keys(client, asset_class)
    to_delete = existing - new_keys - {MANIFEST_KEY}
    to_upload = pairs

    print(f"[SNAPSHOT] Existing under {asset_class}/ prefix: "
          f"{len(existing)} keys.")
    print(f"[SNAPSHOT] To delete (no longer in snapshot): {len(to_delete)}.")
    print(f"[SNAPSHOT] To upload: {len(to_upload)}.")

    # Delete stale keys first
    for key in sorted(to_delete):
        print(f"  [DELETE] {key}")
        if not dry_run:
            client.delete_object(Bucket=R2_BUCKET_NAME, Key=key)

    # Upload all current files (overwrites are normal — this is the snapshot)
    total_bytes = 0
    t0 = time.perf_counter()
    file_records = []
    promoted_log: dict[str, bool] = {}
    for local, r2_key in to_upload:
        is_meta, label = _is_dashboard_meta(r2_key, asset_class)
        if is_meta and label is not None:
            body, promoted = _meta_with_promoted(local, label)
            promoted_log[label] = promoted
            size = len(body)
        else:
            body = None
            size = local.stat().st_size
        total_bytes += size
        prefix = "[UPLOAD]" if not dry_run else "[UPLOAD-DRY]"
        print(f"  {prefix} {r2_key:<60} ({size:>10,} bytes)")
        if not dry_run:
            if body is not None:
                client.put_object(
                    Bucket=R2_BUCKET_NAME, Key=r2_key,
                    Body=body, ContentType="application/json",
                )
            else:
                client.upload_file(str(local), R2_BUCKET_NAME, r2_key)
        file_records.append({"key": r2_key, "size": size})

    if promoted_log:
        print("\n[SNAPSHOT] Promoted vs experimental dashboard_results:")
        for label, p in sorted(promoted_log.items()):
            flag = "PROMOTED" if p else "experimental"
            print(f"  [{flag:<12}] {label}")

    # Manifest
    manifest = {
        "asset_class":   asset_class,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "git_sha":       _git_sha(),
        "n_files":       len(to_upload),
        "total_bytes":   total_bytes,
        "files":         file_records,
        "manifest_version": 2,
    }
    if not dry_run:
        client.put_object(
            Bucket=R2_BUCKET_NAME, Key=MANIFEST_KEY,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    elapsed = time.perf_counter() - t0
    if dry_run:
        print(f"\n[SNAPSHOT] Dry-run complete in {elapsed:.1f}s. "
              f"Would have uploaded {total_bytes:,} bytes across "
              f"{len(file_records)} files. No R2 calls were made.")
    else:
        print(f"\n[SNAPSHOT] Done in {elapsed:.1f}s, "
              f"{total_bytes:,} bytes total.")
        print(f"[SNAPSHOT] Manifest written to {MANIFEST_KEY}")
    print(f"[SNAPSHOT] manifest.asset_class = {manifest['asset_class']}")
    print(f"[SNAPSHOT] git_sha = {manifest['git_sha']}")
    print(f"[SNAPSHOT] generated_at = {manifest['generated_at']}")


if __name__ == "__main__":
    main()
