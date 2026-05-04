"""data_source.py — local/cloud abstraction for dashboard reads.

  Local mode (DASHBOARD_CLOUD_MODE unset/false): paths resolve to repo-relative
    files on disk. Existing dashboard_app.py logic is preserved unchanged.

  Cloud mode (DASHBOARD_CLOUD_MODE=true): paths resolve to files fetched
    from a Cloudflare R2 bucket into a per-session temp dir. The R2 client
    is created once per session via st.cache_resource; individual file
    fetches are cached via st.cache_data and keyed on the snapshot
    manifest's generated_at timestamp so that bumping the snapshot
    invalidates every cached fetch automatically.

============================================================================
CANONICAL R2 BUCKET LAYOUT
============================================================================
Both this file (reader) and src/snapshot_for_cloud.py (writer) MUST use the
exact same mapping below. Path drift between writer and reader is a silent-
failure bug class; this is the single source of truth.

    repo-relative path                          R2 bucket key
    ----------------------------------------  ----------------------------------
    models/cache/optuna_studies.db            optuna/optuna_studies.db
    models/cache/optuna_trials.jsonl          optuna/optuna_trials.jsonl
    models/cache/macro_signals.parquet        macro/macro_signals.parquet
    models/xgb_model.meta.json                model/xgb_model.meta.json
    models/cache/dashboard_results/<L>/<F>    dashboard_results/<L>/<F>
                                              snapshot_manifest.json   (root)
============================================================================
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env for local cloud-mode testing. In production (Streamlit Cloud)
# the secrets bridge in dashboard_app.py has already populated os.environ
# from st.secrets before this module is imported, so load_dotenv on a
# non-existent .env is a silent no-op.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


# ----- Path constants -------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent
# Per-session temp dir for cloud-fetched files. tempfile.gettempdir()
# resolves to /tmp on Linux (Streamlit Cloud) and the per-user TEMP on
# Windows, regardless of which env var the platform sets.
TMP_CACHE = Path(tempfile.gettempdir()) / "paper_trader_snapshot_cache"


# ----- Bucket layout (single source of truth) -------------------------------
R2_LAYOUT: dict[str, str] = {
    "models/cache/optuna_studies.db":     "optuna/optuna_studies.db",
    "models/cache/optuna_trials.jsonl":   "optuna/optuna_trials.jsonl",
    "models/cache/macro_signals.parquet": "macro/macro_signals.parquet",
    "models/xgb_model.meta.json":         "model/xgb_model.meta.json",
}
MANIFEST_KEY = "snapshot_manifest.json"
DASHBOARD_RESULTS_PREFIX_LOCAL  = "models/cache/dashboard_results/"
DASHBOARD_RESULTS_PREFIX_REMOTE = "dashboard_results/"


def cloud_mode() -> bool:
    """Late binding so dashboard_app.py can set the env var from
    Streamlit secrets after module import."""
    return os.getenv("DASHBOARD_CLOUD_MODE", "").lower() in ("1", "true", "yes")


def r2_key_for(local_relative: str) -> str:
    """Translate a repo-relative path to its R2 bucket key.

    Static mappings (in R2_LAYOUT) plus a dynamic rule for
    dashboard_results/<label>/<file> entries that vary by saved config.
    """
    p = local_relative.replace("\\", "/")
    if p in R2_LAYOUT:
        return R2_LAYOUT[p]
    if p.startswith(DASHBOARD_RESULTS_PREFIX_LOCAL):
        return p.replace(DASHBOARD_RESULTS_PREFIX_LOCAL,
                         DASHBOARD_RESULTS_PREFIX_REMOTE, 1)
    raise ValueError(f"No R2 mapping for {local_relative!r}")


# ----- Boto3 client (cloud-mode only) ---------------------------------------
@st.cache_resource(show_spinner=False)
def _get_r2_client():
    """Boto3 S3 client configured for Cloudflare R2. Singleton per session.
    Lazy-imports boto3 so local-mode imports of this module don't require it.

    Cloudflare R2 requires path-style addressing and SigV4. boto3 defaults
    to virtual-hosted-style when given an endpoint_url, which works
    inconsistently across R2 buckets/operations — uploads via upload_file
    may succeed while downloads via download_file fail silently. Forcing
    path-style is the safe default per R2's official guidance.
    """
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def _bucket() -> str:
    return os.environ["R2_BUCKET_NAME"]


@st.cache_data(ttl=300, show_spinner=False)
def _get_manifest_ts() -> str:
    """Fetch the snapshot manifest, return generated_at timestamp.

    Cached for 5 minutes so subsequent reads in the same session don't
    re-fetch the manifest on every tab switch. After 5 min the cache
    expires; if a new snapshot has been published, the new generated_at
    becomes the cache key for downstream fetches and they re-download
    automatically.
    """
    client = _get_r2_client()
    obj = client.get_object(Bucket=_bucket(), Key=MANIFEST_KEY)
    manifest = json.loads(obj["Body"].read())
    return manifest.get("generated_at", "")


@st.cache_data(show_spinner=False)
def _fetch_to_tmp(remote_key: str, manifest_ts: str) -> str:
    """Download a file from R2 into TMP_CACHE; return the local path.

    manifest_ts is part of the cache key so updating the manifest busts
    every cached fetch. On 404/network error: log a streamlit warning and
    return a path that doesn't exist (caller's os.path.exists() check
    falls through to its own missing-file branch)."""
    _ = manifest_ts  # used only as cache key
    client = _get_r2_client()
    local = TMP_CACHE / remote_key
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(_bucket(), remote_key, str(local))
        return str(local)
    except Exception as e:
        # Log to stderr — visible in Streamlit Cloud's app logs even when
        # st.warning calls inside @st.cache_data don't render reliably.
        # This is the diagnostic channel when the cloud dashboard goes
        # silent: tail the app's logs to see exact (key, error_type, msg).
        msg = f"R2 fetch failed for {remote_key!r}: {type(e).__name__}: {e}"
        print(f"[data_source] {msg}", file=sys.stderr, flush=True)
        st.warning(msg)   # best-effort UI surface; not always visible
        # Return a path guaranteed not to exist; caller's
        # os.path.exists() falls through to its missing-file branch.
        return str(TMP_CACHE / "_missing" / remote_key)


def path_to(local_relative: str) -> str:
    """Resolve a repo-relative path to an absolute local path.

    Local mode: returns the absolute path under REPO_ROOT (whether or not
    the file exists; caller's os.path.exists() handles missing files
    exactly as it does today).
    Cloud mode: maps to the R2 key, fetches into TMP_CACHE (cached for the
    session, manifest-timestamp keyed), returns the /tmp path."""
    if not cloud_mode():
        return str(REPO_ROOT / local_relative)
    remote_key = r2_key_for(local_relative)
    ts = _get_manifest_ts()
    return _fetch_to_tmp(remote_key, ts)


def list_dashboard_result_labels() -> list[str]:
    """Return the set of dashboard_result label names available.

    Local: scans the local dashboard_results/ directory.
    Cloud: lists keys under the R2 dashboard_results/ prefix and extracts
    the unique <label> path components."""
    if not cloud_mode():
        d = REPO_ROOT / "models" / "cache" / "dashboard_results"
        if not d.exists():
            return []
        return sorted([p.name for p in d.iterdir() if p.is_dir()])
    client = _get_r2_client()
    paginator = client.get_paginator("list_objects_v2")
    labels: set[str] = set()
    for page in paginator.paginate(Bucket=_bucket(),
                                   Prefix=DASHBOARD_RESULTS_PREFIX_REMOTE):
        for obj in page.get("Contents", []) or []:
            parts = obj["Key"].split("/")
            # key shape: dashboard_results/<label>/<file>
            if len(parts) >= 3 and parts[0] == "dashboard_results":
                labels.add(parts[1])
    return sorted(labels)
