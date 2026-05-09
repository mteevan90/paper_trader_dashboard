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
CANONICAL R2 BUCKET LAYOUT (asset-class aware as of crypto-extension Phase 1)
============================================================================
Both this file (reader) and src/snapshot_for_cloud.py (writer) MUST use the
exact same mapping below. Path drift between writer and reader is a silent-
failure bug class; this is the single source of truth.

R2 keys are now PREFIXED by asset_class. For asset_class="equities":

    repo-relative path                          R2 bucket key
    ----------------------------------------  ----------------------------------
    models/cache/optuna_studies.db            equities/optuna/optuna_studies.db
    models/cache/optuna_trials.jsonl          equities/optuna/optuna_trials.jsonl
    models/cache/macro_signals.parquet        equities/macro/macro_signals.parquet
    models/xgb_model.meta.json                equities/model/xgb_model.meta.json
    models/cache/dashboard_results/<L>/<F>    equities/dashboard_results/<L>/<F>
                                              snapshot_manifest.json   (root, shared)

For asset_class="crypto" (Phase 2 — Chris's work), the same suffixes
appear under a `crypto/` prefix instead. The manifest at the bucket
root is shared across asset classes (a single deployment reads it once).

LOCAL paths: equities keep the flat `models/cache/...` layout (preserves
every existing callsite bit-identically). Other asset classes get a
namespaced subdir, e.g. `models/cache/crypto/optuna_studies.db`. See
docs/Crypto_Extension_Decisions.md for the full rationale.
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
# The keys here are the SUFFIX after the asset_class prefix. r2_key_for()
# composes the final R2 key as f"{asset_class}/{suffix}". For example,
# "models/cache/optuna_studies.db" with asset_class="equities" maps to
# "equities/optuna/optuna_studies.db" in the bucket.
R2_LAYOUT_SUFFIX: dict[str, str] = {
    "models/cache/optuna_studies.db":       "optuna/optuna_studies.db",
    "models/cache/optuna_trials.jsonl":     "optuna/optuna_trials.jsonl",
    "models/cache/macro_signals.parquet":   "macro/macro_signals.parquet",
    "models/xgb_model.meta.json":           "model/xgb_model.meta.json",
    "models/cache/feature_importance.json": "model/feature_importance.json",
    "models/cache/sector_map.json":         "model/sector_map.json",
    "models/cache/ticker_names.json":       "model/ticker_names.json",
}
# Back-compat alias for any external readers that imported R2_LAYOUT
# directly. Same suffix mapping. snapshot_for_cloud.py is the only known
# in-repo consumer; it goes through r2_key_for() now, not this dict.
R2_LAYOUT = R2_LAYOUT_SUFFIX

DEFAULT_ASSET_CLASS = "equities"
SUPPORTED_ASSET_CLASSES = ("equities", "crypto")

MANIFEST_KEY = "snapshot_manifest.json"
DASHBOARD_RESULTS_PREFIX_LOCAL  = "models/cache/dashboard_results/"
DASHBOARD_RESULTS_PREFIX_REMOTE_SUFFIX = "dashboard_results/"
# Back-compat alias for any consumer that still imports the legacy name.
# Equity-class-prefixed remote paths are now produced via r2_key_for().
DASHBOARD_RESULTS_PREFIX_REMOTE = DASHBOARD_RESULTS_PREFIX_REMOTE_SUFFIX


def cloud_mode() -> bool:
    """Late binding so dashboard_app.py can set the env var from
    Streamlit secrets after module import."""
    return os.getenv("DASHBOARD_CLOUD_MODE", "").lower() in ("1", "true", "yes")


def _validate_asset_class(asset_class: str) -> None:
    if asset_class not in SUPPORTED_ASSET_CLASSES:
        raise ValueError(
            f"Unsupported asset_class={asset_class!r}; "
            f"valid choices: {SUPPORTED_ASSET_CLASSES}")


def r2_key_for(local_relative: str,
               asset_class: str = DEFAULT_ASSET_CLASS) -> str:
    """Translate a repo-relative path to its asset-class-prefixed R2 bucket key.

    Static mappings (in R2_LAYOUT_SUFFIX) plus a dynamic rule for
    dashboard_results/<label>/<file> entries that vary by saved config.
    The final key is always ``f"{asset_class}/{suffix}"`` so the bucket
    stays cleanly partitioned per asset class. The shared
    ``snapshot_manifest.json`` at the bucket root is the only key that
    is NOT asset-class-prefixed.
    """
    _validate_asset_class(asset_class)
    p = local_relative.replace("\\", "/")
    if p in R2_LAYOUT_SUFFIX:
        return f"{asset_class}/{R2_LAYOUT_SUFFIX[p]}"
    if p.startswith(DASHBOARD_RESULTS_PREFIX_LOCAL):
        suffix = p.replace(DASHBOARD_RESULTS_PREFIX_LOCAL,
                           DASHBOARD_RESULTS_PREFIX_REMOTE_SUFFIX, 1)
        return f"{asset_class}/{suffix}"
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
def _fetch_to_tmp(remote_key: str, manifest_ts: str,
                  _quiet: bool = False) -> str:
    """Download a file from R2 into TMP_CACHE; return the local path.

    manifest_ts is part of the cache key so updating the manifest busts
    every cached fetch. _quiet (underscore prefix excludes it from
    Streamlit's cache hash) suppresses the on-error log + st.warning
    when the caller knows a 404 is expected — e.g. scanning meta.json
    for non-study directories under dashboard_results/ such as
    v3_track2_perturbation/, which holds aggregation CSVs but no
    meta.json.

    On 404/network error: log a streamlit warning (unless _quiet=True)
    and return a path that doesn't exist (caller's os.path.exists()
    check falls through to its own missing-file branch)."""
    _ = manifest_ts  # used only as cache key
    client = _get_r2_client()
    local = TMP_CACHE / remote_key
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(_bucket(), remote_key, str(local))
        return str(local)
    except Exception as e:
        if not _quiet:
            # Log to stderr — visible in Streamlit Cloud's app logs even
            # when st.warning calls inside @st.cache_data don't render
            # reliably. This is the diagnostic channel when the cloud
            # dashboard goes silent: tail the app's logs to see exact
            # (key, error_type, msg).
            msg = f"R2 fetch failed for {remote_key!r}: {type(e).__name__}: {e}"
            print(f"[data_source] {msg}", file=sys.stderr, flush=True)
            st.warning(msg)   # best-effort UI surface; not always visible
        # Return a path guaranteed not to exist; caller's
        # os.path.exists() falls through to its missing-file branch.
        return str(TMP_CACHE / "_missing" / remote_key)


def _local_path_for_asset(local_relative: str, asset_class: str) -> str:
    """Compute the on-disk path for a repo-relative resource at a given
    asset_class. For "equities" the legacy flat layout is preserved
    (no path mutation), so every existing callsite that passes default
    arguments hits the exact same file as before. For other asset
    classes (crypto, etc.) the path is namespaced under the cache
    subtree, e.g. "models/cache/optuna_studies.db" with
    asset_class="crypto" resolves to "models/cache/crypto/optuna_studies.db".
    """
    p = local_relative.replace("\\", "/")
    if asset_class == "equities":
        return str(REPO_ROOT / p)
    # Insert <asset_class>/ inside models/cache/ for non-equity classes.
    if p.startswith("models/cache/"):
        rest = p[len("models/cache/"):]
        return str(REPO_ROOT / "models" / "cache" / asset_class / rest)
    # Outside models/cache/: leave alone (e.g. models/xgb_model.meta.json
    # has no asset-aware override yet — callers asking for a non-equity
    # variant of a top-level model file should pass the namespaced
    # local_relative explicitly).
    return str(REPO_ROOT / p)


def path_to(local_relative: str,
            asset_class: str = DEFAULT_ASSET_CLASS,
            quiet: bool = False) -> str:
    """Resolve a repo-relative path to an absolute local path,
    asset-class aware.

    Local mode: returns the absolute path under REPO_ROOT. For
    asset_class="equities" the legacy flat path is preserved
    (bit-identical to pre-Phase-1 behavior). For other asset classes
    the path is namespaced under models/cache/<asset_class>/.
    Cloud mode: maps to the asset-class-prefixed R2 key, fetches into
    TMP_CACHE (cached for the session, manifest-timestamp keyed),
    returns the /tmp path. quiet=True suppresses the warning emitted
    on a fetch miss — pass it when a missing remote file is an
    expected condition (not an error)."""
    _validate_asset_class(asset_class)
    if not cloud_mode():
        return _local_path_for_asset(local_relative, asset_class)
    remote_key = r2_key_for(local_relative, asset_class=asset_class)
    ts = _get_manifest_ts()
    return _fetch_to_tmp(remote_key, ts, _quiet=quiet)


def list_dashboard_result_labels(
        asset_class: str = DEFAULT_ASSET_CLASS) -> list[str]:
    """Return the set of dashboard_result label names available for the
    given asset_class.

    Local: scans the local dashboard_results/ directory. For "equities"
    that's the legacy `models/cache/dashboard_results/`; for other
    asset classes it's `models/cache/<asset_class>/dashboard_results/`.
    Cloud: lists keys under the asset-class-prefixed dashboard_results/
    bucket prefix and extracts the unique <label> path components."""
    _validate_asset_class(asset_class)
    if not cloud_mode():
        # Resolve via _local_path_for_asset so the equity legacy layout
        # stays flat and crypto picks up the namespaced subdir.
        d_str = _local_path_for_asset(
            "models/cache/dashboard_results", asset_class)
        d = Path(d_str)
        if not d.exists():
            return []
        return sorted([p.name for p in d.iterdir() if p.is_dir()])
    client = _get_r2_client()
    paginator = client.get_paginator("list_objects_v2")
    prefix = f"{asset_class}/{DASHBOARD_RESULTS_PREFIX_REMOTE_SUFFIX}"
    labels: set[str] = set()
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            parts = obj["Key"].split("/")
            # key shape: <asset_class>/dashboard_results/<label>/<file>
            if (len(parts) >= 4
                    and parts[0] == asset_class
                    and parts[1] == "dashboard_results"):
                labels.add(parts[2])
    return sorted(labels)


@st.cache_data(ttl=300, show_spinner=False)
def list_promoted_dashboard_result_labels(
        asset_class: str = DEFAULT_ASSET_CLASS) -> list[str]:
    """Subset of list_dashboard_result_labels() restricted to labels
    whose meta.json has promoted=true. Used by the cloud dashboard's
    Best Trial picker to hide experimental studies. Missing/malformed
    meta.json or missing "promoted" field is treated as not promoted."""
    _validate_asset_class(asset_class)
    out: list[str] = []
    for label in list_dashboard_result_labels(asset_class=asset_class):
        try:
            # quiet=True: a missing meta.json is expected for non-study
            # directories (e.g. v3_track2_perturbation/ holds aggregation
            # CSVs but no meta.json). The 404 from R2 is normal here and
            # would otherwise emit a warning banner on every render.
            meta_path = path_to(
                f"models/cache/dashboard_results/{label}/meta.json",
                asset_class=asset_class, quiet=True)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("promoted") is True:
                out.append(label)
        except Exception:
            continue
    return out
