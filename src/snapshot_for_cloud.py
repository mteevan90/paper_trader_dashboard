"""snapshot_for_cloud.py — bundle dashboard data + upload to R2.

Run after any local change you want reflected on the cloud dashboard:

    cd src
    python snapshot_for_cloud.py

What it does:
  1. Collects every file in data_source.R2_LAYOUT plus every file under
     models/cache/dashboard_results/<label>/.
  2. Uploads each to its corresponding R2 key (mapping defined in
     data_source.r2_key_for — single source of truth shared with the
     reader).
  3. Reads the prior snapshot_manifest.json from R2 (if any) and DELETES
     any files no longer in this snapshot — keeps the bucket as a clean
     single-snapshot mirror with no stale leftover files.
  4. Writes a fresh snapshot_manifest.json to the bucket root with
     generated_at (UTC ISO), git SHA, file list, total size.

Bucket layout: see ``data_source.py`` docstring (the canonical source).

Credentials: read from ../.env via python-dotenv. Required env vars:
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_ENDPOINT_URL
    R2_BUCKET_NAME
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv

from data_source import (R2_LAYOUT, MANIFEST_KEY, REPO_ROOT, r2_key_for,
                         DASHBOARD_RESULTS_PREFIX_LOCAL,
                         DASHBOARD_RESULTS_PREFIX_REMOTE)


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
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return "unknown"


def _collect_files() -> list[tuple[Path, str]]:
    """Return list of (local_absolute_path, r2_key) for every file the
    cloud dashboard reads. Skips missing files with a warning."""
    pairs: list[tuple[Path, str]] = []

    # Static layout entries
    for local_rel, r2_key in R2_LAYOUT.items():
        local = REPO_ROOT / local_rel
        if local.exists():
            pairs.append((local, r2_key))
        else:
            print(f"  [WARN] missing local file, skipping: {local_rel}")

    # Dashboard results: every file under each <label>/ subdir
    dr_root = REPO_ROOT / "models" / "cache" / "dashboard_results"
    if dr_root.exists():
        for label_dir in sorted(dr_root.iterdir()):
            if not label_dir.is_dir():
                continue
            for f in sorted(label_dir.iterdir()):
                if not f.is_file():
                    continue
                rel = f.relative_to(REPO_ROOT).as_posix()
                pairs.append((f, r2_key_for(rel)))
    else:
        print(f"  [WARN] no dashboard_results directory at {dr_root}")

    return pairs


def _list_remote_keys(client) -> set[str]:
    """List all keys currently in the bucket."""
    keys: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET_NAME):
        for obj in page.get("Contents", []) or []:
            keys.add(obj["Key"])
    return keys


def main():
    _check_creds()
    client = _client()
    print(f"[SNAPSHOT] Bucket: {R2_BUCKET_NAME}")
    print(f"[SNAPSHOT] Endpoint: {R2_ENDPOINT_URL}")

    pairs = _collect_files()
    new_keys = {key for _, key in pairs}
    print(f"[SNAPSHOT] Collected {len(pairs)} files for upload.")

    # Diff against current bucket contents
    existing = _list_remote_keys(client)
    to_delete = existing - new_keys - {MANIFEST_KEY}
    to_upload = pairs

    print(f"[SNAPSHOT] Existing in bucket: {len(existing)} keys.")
    print(f"[SNAPSHOT] To delete (no longer in snapshot): {len(to_delete)}.")
    print(f"[SNAPSHOT] To upload: {len(to_upload)}.")

    # Delete stale keys first
    for key in sorted(to_delete):
        print(f"  [DELETE] {key}")
        client.delete_object(Bucket=R2_BUCKET_NAME, Key=key)

    # Upload all current files (overwrites are normal — this is the snapshot)
    total_bytes = 0
    t0 = time.perf_counter()
    file_records = []
    for local, r2_key in to_upload:
        size = local.stat().st_size
        total_bytes += size
        print(f"  [UPLOAD] {r2_key:<60} ({size:>10,} bytes)")
        client.upload_file(str(local), R2_BUCKET_NAME, r2_key)
        file_records.append({"key": r2_key, "size": size})

    # Manifest
    manifest = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "git_sha":       _git_sha(),
        "n_files":       len(to_upload),
        "total_bytes":   total_bytes,
        "files":         file_records,
        "manifest_version": 1,
    }
    client.put_object(
        Bucket=R2_BUCKET_NAME, Key=MANIFEST_KEY,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[SNAPSHOT] Done in {elapsed:.1f}s, {total_bytes:,} bytes total.")
    print(f"[SNAPSHOT] Manifest written to {MANIFEST_KEY}")
    print(f"[SNAPSHOT] git_sha = {manifest['git_sha']}")
    print(f"[SNAPSHOT] generated_at = {manifest['generated_at']}")


if __name__ == "__main__":
    main()
