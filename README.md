# paper_trader

Personal-research algorithmic paper-trading project.

This README covers operational details for the cloud dashboard
deployment (segment 22). For the project's architecture, locked
baselines, and segment history, see the markdown files in
`models/cache/`.

## What's in here

- `src/` — Python source: features, model, backtest, Optuna runner,
  alt-signal modules, dashboard.
- `requirements.txt` — pinned deps for both local dev and the cloud
  build.
- `models/cache/*.md` and `models/cache/*.txt` — segment summaries,
  dead-end notes, baselines (the audit trail for what was tried).
- `models/cache/streamlit_cloud_secrets_template.toml` — template for
  Streamlit Cloud secrets (real values never go in git).

Data files (`models/cache/*.parquet`, `*.db`, `*.json`, `*.jsonl`,
`models/price_cache/`, `models/xgb_model.json`, etc.) live locally
only. They are mirrored to a Cloudflare R2 bucket via
`src/snapshot_for_cloud.py` and read by the cloud dashboard via
`src/data_source.py`.

## Local development

```
cd src
..\venv\Scripts\activate         # Windows
# or: source ../venv/bin/activate  (macOS/Linux)
streamlit run dashboard_app.py
```

Local mode reads from `models/cache/` directly. No auth gate. Cloud
mode is disabled unless `DASHBOARD_CLOUD_MODE=true` is set.

## Cloud dashboard — refresh data

After re-running anything that updates local caches (training,
backtest, Optuna, save_results), publish the refreshed snapshot:

```
cd src
python snapshot_for_cloud.py
```

This:
1. Collects every file the dashboard reads (Optuna SQLite, JSONL,
   macro parquet, model meta, dashboard_results/).
2. Uploads to R2 under the canonical bucket layout (see
   `src/data_source.py` docstring).
3. Deletes any prior bucket files that aren't in this snapshot.
4. Writes a fresh `snapshot_manifest.json` with timestamp + git SHA.

The cloud dashboard's read-side cache invalidates on next page load
when the manifest's `generated_at` changes.

## Cloud dashboard — first-time deploy

1. **Generate password hashes** (interactive, getpass-based, plaintext
   never on disk):
   ```
   cd src
   python hash_passwords.py
   ```
   Copy the printed TOML block.

2. **Push to GitHub**:
   ```
   git add src/ requirements.txt README.md .gitignore models/cache/*.md models/cache/*.txt models/cache/*.toml
   git commit -m "segment 22: cloud dashboard"
   git push
   ```

3. **Connect Streamlit Cloud**: at https://share.streamlit.io, "New app",
   pick this repo, branch, main file = `src/dashboard_app.py`, Python
   3.11+, deploy.

4. **Paste secrets**: in Streamlit Cloud → App settings → Secrets, paste
   the TOML output from step 1, then add the `[r2]` section using your
   Cloudflare R2 token (Object Read permission on the bucket). See
   `models/cache/streamlit_cloud_secrets_template.toml` for the
   complete shape.

5. **Test access**: open the deployed URL, log in with one of the
   accounts you generated. The dashboard should load all 5 tabs from
   the R2 snapshot.

## Cloud dashboard — adding/removing users

1. Re-run `python hash_passwords.py` to add new users (it generates
   hashes for everyone you re-enter — old users stay valid as long as
   you copy their existing `password_hash` lines into the new TOML).
2. Paste the updated `[auth.users.*]` blocks into Streamlit Cloud
   secrets. Streamlit reloads the app with new credentials.
3. Optionally rotate `[auth.cookie] key` to invalidate all current
   sessions.
