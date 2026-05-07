"""dashboard_app.py - Streamlit dashboard for the paper trader.

Read-only personal dashboard. Two modes:

  Local mode (default): reads cached artifacts directly from
    models/cache/ and models/. Live-fallback backtests work for any
    config including custom trial numbers.

  Cloud mode (DASHBOARD_CLOUD_MODE=true, set via Streamlit Cloud
    secrets): fetches the same artifacts from a Cloudflare R2 bucket
    via src/data_source.py. Authentication required (see
    src/dashboard_auth.py). Live-fallback is disabled — cloud users
    can only view configs that were pre-saved by snapshot_for_cloud.py.

Always reads:
  - models/cache/optuna_studies.db        (SQLite study)
  - models/cache/optuna_trials.jsonl
  - models/cache/macro_signals.parquet    (v2 composite components)
  - models/cache/dashboard_results/<label>/{portfolio.parquet, trades.parquet,
                                            scores.json, holdings.json, meta.json}
  - models/xgb_model.meta.json            (model training provenance)

Run locally:
  streamlit run dashboard_app.py

Sidebar selects a config (default / best trial of study / custom trial #N);
all tabs reflect that selection. Saved results under
models/cache/dashboard_results/ load instantly. In LOCAL mode, custom
trial configs that aren't pre-saved fall back to a live backtest. In
CLOUD mode, they show a friendly "switch to a saved config" message.

TODO: After dashboard stabilizes, extract shared compute helpers
(_summarize_period, _sleeve_metrics, etc.) into src/dashboard_compute.py
and have both dashboard.py and dashboard_app.py import from there.
Cleaner long-term separation of compute from rendering.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------------------------
# Streamlit secrets → env-vars bridge (must run BEFORE data_source import).
# In Streamlit Cloud, secrets are set via the Secrets UI. data_source.py
# reads R2 credentials and the cloud-mode flag from os.environ so it can
# work in both Streamlit-Cloud and local-cloud-mode-test contexts. Local
# dev without secrets is a no-op (the .env load below handles that path).
# ---------------------------------------------------------------------------
try:
    if hasattr(st, "secrets"):
        if "r2" in st.secrets:
            for k, v in st.secrets["r2"].items():
                os.environ[f"R2_{k.upper()}"] = str(v)
        if "app" in st.secrets and st.secrets["app"].get("cloud_mode"):
            os.environ["DASHBOARD_CLOUD_MODE"] = "true"
except Exception:
    pass  # Local dev without a secrets file — fall through to .env
import data_source  # noqa: E402  must follow the secrets bridge

from backtest_config import BacktestConfig            # noqa: E402
# Import (not modify) compute helpers from the existing batch dashboard.
# Pure data/compute functions; their HTML-emitting siblings stay untouched.
from dashboard import (_download_benchmark, _series_stats, _sleeve_metrics,  # noqa: E402
                       _summarize_period, _top_traded_stocks)
from macro_signals import compute_macro_score, FRED_SERIES   # noqa: E402


CLOUD_MODE = data_source.cloud_mode()


# ---------------------------------------------------------------------------
# Path resolution (delegated to data_source so cloud reads transparently
# fetch from R2; local reads return paths under REPO_ROOT/models/).
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))


def _db_path() -> str:
    return data_source.path_to("models/cache/optuna_studies.db")


def _db_url() -> str:
    return f"sqlite:///{_db_path()}"


def _trials_jsonl_path() -> str:
    return data_source.path_to("models/cache/optuna_trials.jsonl")


def _macro_parquet_path() -> str:
    return data_source.path_to("models/cache/macro_signals.parquet")


def _model_meta_path() -> str:
    return data_source.path_to("models/xgb_model.meta.json")


def _dashboard_result_path(label: str, filename: str) -> str:
    return data_source.path_to(
        f"models/cache/dashboard_results/{label}/{filename}")


def _feature_importance_path() -> str:
    return data_source.path_to("models/cache/feature_importance.json")


def _sector_map_path() -> str:
    return data_source.path_to("models/cache/sector_map.json")


def _ticker_names_path() -> str:
    return data_source.path_to("models/cache/ticker_names.json")


LOCKED_BEST_STUDY = "regime_dependent_v1_20260505_2240"


# Human-readable display labels for SQLite study names. Forward-compatible:
# unmapped names fall through to display the raw name. Update when new
# studies get pinned as significant; smoke tests are listed for completeness.
STUDY_DISPLAY_NAMES: dict[str, str] = {
    "optuna_v1_20260504_103429": "v1 baseline — full study (1000 trials, locked best)",
    "optuna_v1_20260504_114307": "v2 macro expanded — full study (1000 trials)",
    "smoke_20260503_222050":     "v1 smoke (30 trials, leaky model)",
    "smoke_20260504_102210":     "v1 smoke (30 trials, clean model)",
    "smoke_20260504_134020":     "post-segment-12 smoke (30 trials, alt-bucket refactor)",
}


def study_display_name(name: str) -> str:
    """Human-readable label for a SQLite study name; falls through to
    the raw name if not in STUDY_DISPLAY_NAMES."""
    return STUDY_DISPLAY_NAMES.get(name, name)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Paper Trader Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def list_studies() -> list[str]:
    db_path = _db_path()
    if not os.path.exists(db_path):
        return []
    return optuna.get_all_study_names(storage=_db_url())


@st.cache_data(ttl=300, show_spinner="Loading study trials...")
def load_study_trials_df(study_name: str) -> pd.DataFrame:
    """Flat DataFrame of every trial in the study with params + value + state."""
    s = optuna.load_study(study_name=study_name, storage=_db_url())
    rows = []
    for t in s.trials:
        rows.append({
            "trial_number": t.number,
            "state":        t.state.name,
            "value":        t.value,
            **t.params,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def load_trial_jsonl_records(study_name: str) -> list[dict]:
    out: list[dict] = []
    jsonl_path = _trials_jsonl_path()
    if not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("study_name") == study_name:
                out.append(r)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_df() -> pd.DataFrame:
    macro_path = _macro_parquet_path()
    if not os.path.exists(macro_path):
        return pd.DataFrame()
    return pd.read_parquet(macro_path)


@st.cache_data(ttl=3600, show_spinner=False)
def load_model_meta() -> dict:
    meta_path = _model_meta_path()
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_feature_importance() -> list[dict]:
    """Read models/cache/feature_importance.json (written by
    snapshot_for_cloud.py before each snapshot upload)."""
    p = _feature_importance_path()
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("features", []) or []


@st.cache_data(ttl=3600, show_spinner=False)
def load_sector_map() -> dict:
    p = _sector_map_path()
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_perturbation_summary() -> "pd.DataFrame | None":
    """Read the V3 Track 2 perturbation summary CSV. Returns None if
    the file isn't present on this deployment (e.g. cloud snapshot
    predates Track 2 generation). Auto-routed via data_source so it
    transparently reads from R2 in cloud mode."""
    p = data_source.path_to(
        "models/cache/dashboard_results/v3_track2_perturbation/"
        "summary_full.csv")
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_ticker_names() -> dict:
    """Read models/cache/ticker_names.json. Returns {} on miss so callers
    fall back to plain ticker strings — never crashes the dashboard."""
    p = _ticker_names_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _yf_url(ticker: str) -> str:
    """Yahoo Finance quote URL for a ticker."""
    return f"https://finance.yahoo.com/quote/{ticker}"


def _render_df_with_ticker_links(df: pd.DataFrame, **kwargs) -> None:
    """Render a DataFrame via st.dataframe with the ticker column as
    clickable Yahoo Finance links + a wrapped Company-name column
    injected immediately after.

    Detection is case-insensitive: column named 'ticker' or 'Ticker'
    both work (the latter is what dashboard._top_traded_stocks emits).

    The Name column comes from ticker_names.json (cached via
    load_ticker_names). Missing entries fall back to the ticker
    string. If ticker_names.json is empty/missing, no Name column is
    injected (graceful fallback)."""
    if df is None or df.empty:
        st.dataframe(df, **kwargs)
        return
    # Find the ticker column (case-insensitive)
    ticker_col = next((c for c in df.columns if c.lower() == "ticker"), None)
    if ticker_col is None:
        st.dataframe(df, **kwargs)
        return

    show = df.copy()

    # Inject Name column right after the ticker column (only if names
    # cache is available — empty dict means cache missing/unbuilt).
    names = load_ticker_names()
    name_col_label = None
    if names:
        # Pick a label that won't clobber an existing column.
        name_col_label = "Name" if "Name" not in show.columns else "Company"
        if name_col_label in show.columns:
            # Both Name and Company already exist; skip injection.
            name_col_label = None
    if name_col_label is not None:
        name_series = show[ticker_col].apply(lambda t: names.get(t, t))
        cols = list(show.columns)
        idx = cols.index(ticker_col)
        cols.insert(idx + 1, name_col_label)
        show[name_col_label] = name_series
        show = show[cols]

    # Transform ticker to URL for LinkColumn rendering
    show[ticker_col] = show[ticker_col].apply(_yf_url)

    column_config = kwargs.pop("column_config", None) or {}
    column_config[ticker_col] = st.column_config.LinkColumn(
        ticker_col,
        display_text=r"quote/([\w\.\-]+)",
        help="Open Yahoo Finance for this ticker",
    )
    if name_col_label is not None:
        # width="medium" constrains the column so long names wrap rather
        # than expanding the table out wide.
        column_config[name_col_label] = st.column_config.TextColumn(
            name_col_label,
            width="medium",
            help="Company name (shortened — corporate suffixes stripped)",
        )
    st.dataframe(show, column_config=column_config, **kwargs)


@st.cache_data(ttl=300, show_spinner=False)
def _load_meta_only(label: str) -> dict | None:
    """Read just the meta.json from a dashboard_results label. Returns
    None on missing/unreadable. Used by sidebar_config_picker to extract
    fixed_tunables for hypothesis-style studies (whose Optuna trial.params
    is missing the held-fixed keys)."""
    p = _dashboard_result_path(label, "meta.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_saved_result(label: str) -> dict | None:
    """Read a pre-saved backtest result from dashboard_results/<label>/.
    Returns None if any file is missing. Cloud mode fetches each file
    from R2 on demand; local mode reads from disk."""
    portfolio_p = _dashboard_result_path(label, "portfolio.parquet")
    trades_p    = _dashboard_result_path(label, "trades.parquet")
    scores_p    = _dashboard_result_path(label, "scores.json")
    holdings_p  = _dashboard_result_path(label, "holdings.json")
    meta_p      = _dashboard_result_path(label, "meta.json")
    if not all(os.path.exists(p) for p in (portfolio_p, trades_p,
                                           scores_p, holdings_p, meta_p)):
        return None
    portfolio_df = pd.read_parquet(portfolio_p)
    trades_df = pd.read_parquet(trades_p)
    with open(scores_p, "r", encoding="utf-8") as f:
        scores = json.load(f)
    with open(holdings_p, "r", encoding="utf-8") as f:
        holdings = json.load(f)
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "portfolio_df": portfolio_df,
        "trades_df":    trades_df,
        "scores":       scores,
        "holdings":     holdings,
        "meta":         meta,
    }


@st.cache_data(ttl=3600, show_spinner="Downloading benchmark...")
def cached_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    return _download_benchmark(ticker, start, end)


@st.cache_resource(show_spinner="Running live backtest (one-off, cached)...")
def run_live_backtest(config_dict: dict) -> dict:
    """Live backtest fallback when no saved result exists for the chosen
    config. Cached at the resource level so it survives Streamlit reruns
    within a session."""
    # Heavy imports only used on the fallback path — keep dashboard cold-load fast.
    from backtest import (fetch_earnings_dates, fetch_fundamentals,
                          run_backtest)
    from feature_cache import build_feature_matrix
    from fetch_data import (UNIVERSE_TICKERS, build_sector_map,
                            get_stock_data_cached)
    from model import load_model

    cfg = BacktestConfig(**config_dict)
    price_cache = os.path.join(MODELS_DIR, "price_cache")
    fm = build_feature_matrix(list(UNIVERSE_TICKERS),
                              cfg.train_start, cfg.validate_end,
                              price_cache_dir=price_cache)
    pdata = get_stock_data_cached(list(UNIVERSE_TICKERS),
                                  cfg.train_start, cfg.validate_end,
                                  cache_dir=price_cache)
    spy = get_stock_data_cached(["SPY"], cfg.train_start, cfg.validate_end,
                                cache_dir=price_cache)
    spy_close = spy["SPY"]["Close"]
    sector_map = build_sector_map(list(fm.keys()))
    fund = fetch_fundamentals(list(fm.keys()))
    earn = fetch_earnings_dates(list(fm.keys()),
                                cfg.validate_start, cfg.validate_end)
    model = load_model()
    portfolio_df, trades_df, scores, holdings = run_backtest(
        fm, pdata, split_date=cfg.validate_start,
        fund_data=fund, sector_map=sector_map, earnings_dates=earn,
        model=model, config=cfg,
    )
    holdings_serial = {
        tkr: {"shares": float(v["shares"]),
              "entry_price": float(v["entry_price"]),
              "stop_price": float(v.get("stop_price", 0.0))}
        for tkr, v in holdings.items()
    }
    return {
        "portfolio_df": portfolio_df,
        "trades_df":    trades_df,
        "scores":       scores,
        "holdings":     holdings_serial,
        "meta":         {"label": "live", "config": cfg.to_dict()},
    }


# ---------------------------------------------------------------------------
# Helpers (lifted compute from dashboard.py, pure data; HTML stays in the
# original file). Round-trip pairing for the trades log.
# ---------------------------------------------------------------------------

def round_trip_trades(trades_df: pd.DataFrame, end_date) -> pd.DataFrame:
    """Pair BUY/exit rows per ticker chronologically. Open positions are
    represented as one row each with reason='Open'.

    Lifted from dashboard._active_trade_log_html so the dashboard can show
    a sortable Streamlit dataframe instead of a static HTML table."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    # Normalize date dtype
    df = trades_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    end_ts = pd.Timestamp(end_date)
    rows = []
    for ticker in df["ticker"].unique():
        tk = df[df["ticker"] == ticker].sort_values("date")
        buys = tk[tk["action"] == "BUY"].reset_index(drop=True)
        exits = tk[tk["action"].isin(
            ["SELL", "STOP", "STOP10", "STOP_ATR"])].reset_index(drop=True)
        n_pairs = min(len(buys), len(exits))
        for i in range(n_pairs):
            b = buys.iloc[i]
            s = exits.iloc[i]
            ret_pct = (s["price"] / b["price"] - 1) * 100
            hold_days = (s["date"] - b["date"]).days
            if s["action"] == "STOP_ATR":
                reason = "ATR Stop"
            elif s["action"] in ("STOP10", "STOP"):
                reason = "Hard Stop"
            else:
                reason = "Rebalance"
            shares = float(b.get("shares", 0))
            pnl_d  = (s["price"] - b["price"]) * shares
            rows.append({
                "ticker":      ticker,
                "buy_date":    b["date"],
                "buy_price":   float(b["price"]),
                "sell_date":   s["date"],
                "sell_price":  float(s["price"]),
                "shares":      shares,
                "pnl_dollars": float(pnl_d),
                "return_pct":  ret_pct,
                "hold_days":   hold_days,
                "reason":      reason,
            })
        # Open positions (buys without matching exit)
        for i in range(len(exits), len(buys)):
            b = buys.iloc[i]
            rows.append({
                "ticker":      ticker,
                "buy_date":    b["date"],
                "buy_price":   float(b["price"]),
                "sell_date":   end_ts,
                "sell_price":  float("nan"),
                "shares":      float(b.get("shares", 0)),
                "pnl_dollars": float("nan"),
                "return_pct":  float("nan"),
                "hold_days":   (end_ts - b["date"]).days,
                "reason":      "Open",
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("buy_date").reset_index(drop=True)


def _monthly_returns(pv: pd.Series, spy_close: pd.Series
                     ) -> tuple[pd.DataFrame, str]:
    """Resample portfolio + SPY to month-end and compute monthly pct
    change. Returns (df with columns ['Strategy', 'SPY'] indexed by
    month-end, label). The first month is dropped because pct_change
    on the resample's first observation is NaN. Used by the
    Performance tab's Layer 2 monthly-bars chart.
    """
    if pv is None or pv.empty:
        return pd.DataFrame(), ""
    pv_m = pv.resample("ME").last().pct_change().dropna() * 100
    out = pv_m.to_frame("Strategy")
    if spy_close is not None and not spy_close.empty:
        spy_m = spy_close.resample("ME").last().pct_change().dropna() * 100
        out["SPY"] = spy_m
    return out, f"{out.index[0]:%b %Y} – {out.index[-1]:%b %Y}"


def _holdings_sector_values(holdings: dict, sector_map: dict,
                            total_portfolio_value: float | None
                            ) -> pd.DataFrame:
    """Group current holdings by sector. Each holding's $ value is
    shares × entry_price (cost basis — same approximation the
    existing Positions tab uses; live prices not fetched in cloud
    mode). Returns a DataFrame indexed by sector with columns
    ['value', 'pct'] sorted by value descending.

    If total_portfolio_value is provided, pct uses that as the
    denominator (so cash + holdings = 100%). Otherwise pct sums to
    100% across only the sectors with holdings.
    """
    if not holdings:
        return pd.DataFrame()
    sec_values: dict[str, float] = {}
    for tkr, h in holdings.items():
        sec = sector_map.get(tkr) or "Other"
        sec_values[sec] = sec_values.get(sec, 0.0) + (
            float(h["shares"]) * float(h["entry_price"]))
    df = pd.DataFrame(
        [{"sector": s, "value": v} for s, v in sec_values.items()])
    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    denom = float(total_portfolio_value) if total_portfolio_value else df["value"].sum()
    df["pct"] = df["value"] / denom * 100.0 if denom else 0.0
    return df


def macro_score_series(macro_df: pd.DataFrame,
                       start: pd.Timestamp | None = None,
                       end: pd.Timestamp | None = None) -> pd.Series:
    """Compute macro composite score for every business day in the window."""
    if macro_df is None or macro_df.empty:
        return pd.Series(dtype=float)
    start = start if start is not None else macro_df.index.min()
    end   = end   if end   is not None else macro_df.index.max()
    bdays = pd.bdate_range(start, end)
    return pd.Series(
        [compute_macro_score(macro_df, d) for d in bdays],
        index=bdays,
        name="macro_score",
    )


def trial_to_config(trial: optuna.trial.FrozenTrial,
                    fixed_values: dict | None = None) -> BacktestConfig:
    """Reconstruct a BacktestConfig from a completed trial. fixed_values
    fills in search-space params that weren't sampled — needed for
    hypothesis-style studies where trial.params only contains the
    varied tunables (held-fixed ones skip trial.suggest_*). Default
    None preserves bit-identical behavior for v1 studies."""
    p = {**(fixed_values or {}), **trial.params}
    return BacktestConfig(
        weight_fundamental       = p["weight_fundamental"],
        weight_technical         = p["weight_technical"],
        weight_model             = p["weight_model"],
        macro_threshold_low      = p["macro_threshold_low"],
        macro_threshold_high     = p["macro_threshold_low"] + p["macro_threshold_gap"],
        atr_multiplier           = p["atr_multiplier"],
        analyst_weight           = p["analyst_weight"],
        rebalance_frequency_days = p["rebalance_frequency_days"],
        position_count           = p["position_count"],
    )


def get_result_for_config(label: str, config: BacktestConfig) -> dict:
    """Saved-first, live-fallback. Returns the four-key result dict.

    Cloud mode disables live-fallback (the heavyweight inputs aren't in
    the snapshot — see segment 22's architecture decision). Local mode
    runs run_backtest live for any custom config not pre-saved."""
    saved = load_saved_result(label)
    if saved is not None:
        return saved
    if CLOUD_MODE:
        st.warning(
            f"Custom trial backtests aren't available in the cloud build — "
            f"saved result for **{label}** not found. Switch the sidebar "
            f"to **Default config** or **Best trial of selected study**. "
            f"(Custom trials require the local dev environment.)"
        )
        st.stop()
    st.info(f"No pre-saved result for **{label}** — running live backtest "
            f"(~3-5s, cached for this session).")
    return run_live_backtest(config.to_dict())


# ---------------------------------------------------------------------------
# Sidebar config selector
# ---------------------------------------------------------------------------

def sidebar_config_picker() -> tuple[str, BacktestConfig, str | None, int | None]:
    st.sidebar.title("⚙️ Config")
    studies = list_studies()
    full_studies_all = sorted([s for s in studies if not s.startswith("smoke_")],
                              reverse=True)

    # Cloud mode has no live-fallback; offering a study with no pre-saved
    # best_* dashboard_results would 404 on R2 fetch and friendly-warning-
    # stop the user. Filter to studies that actually have results in the
    # snapshot. Local mode keeps the full list because live-fallback works.
    if CLOUD_MODE:
        available_labels = set(
            data_source.list_promoted_dashboard_result_labels())
        full_studies = [s for s in full_studies_all
                        if any(lbl.startswith(f"best_{s}_")
                               for lbl in available_labels)]
    else:
        full_studies = full_studies_all

    # Custom-trial selection requires the live-fallback backtest path,
    # which is disabled in cloud mode (segment 22 design). Hide the
    # option entirely in cloud builds rather than letting users click
    # it and get the "switch config" message.
    source_options = ["Default config", "Best trial of selected study"]
    if not CLOUD_MODE:
        source_options.append("Custom trial number")
    mode = st.sidebar.radio("Source", source_options, index=0)

    study_name: str | None = None
    trial_number: int | None = None

    if mode != "Default config":
        if not full_studies:
            if CLOUD_MODE and full_studies_all:
                st.sidebar.warning(
                    "No promoted studies available in the cloud "
                    "snapshot. Showing Default config only."
                )
            else:
                st.sidebar.warning("No optuna_v1_* studies found in SQLite.")
            mode = "Default config"
        else:
            default_idx = (full_studies.index(LOCKED_BEST_STUDY)
                           if LOCKED_BEST_STUDY in full_studies else 0)
            study_name = st.sidebar.selectbox(
                "Study", full_studies, index=default_idx,
                format_func=study_display_name,
            )

    if mode == "Default config":
        return "default", BacktestConfig(), None, None

    s = optuna.load_study(study_name=study_name, storage=_db_url())

    # Hypothesis-style studies record fixed_tunables in their saved
    # best-trial meta.json (study-level property — every trial in the
    # study shares the same held values). Look it up once per render so
    # both Best and Custom branches can pass it to trial_to_config.
    # v1 studies produce None here (no fixed_tunables key in meta) →
    # bit-identical to pre-Archetype-3 behavior because trial_to_config's
    # default-None merge is a no-op when trial.params has all 9 keys.
    fixed_for_study: dict | None = None
    try:
        best_label = f"best_{study_name}_{s.best_trial.number}"
        saved_meta = _load_meta_only(best_label)
        if saved_meta and isinstance(saved_meta.get("fixed_tunables"), dict):
            fixed_for_study = saved_meta["fixed_tunables"]
    except ValueError:
        pass  # study has no completed trials — handled in branches below

    if mode == "Best trial of selected study":
        try:
            t = s.best_trial
        except ValueError:
            st.sidebar.error("Study has no completed trials yet.")
            return "default", BacktestConfig(), None, None
        trial_number = t.number
        label = f"best_{study_name}_{trial_number}"
        cfg = trial_to_config(t, fixed_values=fixed_for_study)
        st.sidebar.caption(f"Best trial: **#{trial_number}** "
                           f"(score {t.value:.4f})")
    else:  # Custom
        completes = [t for t in s.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completes:
            st.sidebar.error("Study has no completed trials yet.")
            return "default", BacktestConfig(), None, None
        max_trial = max(t.number for t in completes)
        # Try to default to the best trial number for convenience
        try:
            default_n = s.best_trial.number
        except ValueError:
            default_n = completes[0].number
        trial_number = st.sidebar.number_input(
            "Trial #", min_value=0, max_value=max_trial,
            value=default_n, step=1,
        )
        try:
            t = s.trials[int(trial_number)]
        except IndexError:
            st.sidebar.error(f"Trial #{trial_number} not found.")
            return "default", BacktestConfig(), None, None
        if t.state != optuna.trial.TrialState.COMPLETE:
            st.sidebar.warning(f"Trial #{trial_number} is "
                               f"{t.state.name} — value={t.value}")
            if t.value is None:
                return "default", BacktestConfig(), None, None
        label = f"custom_{study_name}_{trial_number}"
        cfg = trial_to_config(t, fixed_values=fixed_for_study)
        st.sidebar.caption(f"Trial #{trial_number} score: "
                           f"{t.value:.4f}" if t.value is not None
                           else "Trial has no value")

    return label, cfg, study_name, trial_number


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def _summary_caveat_prefix(label: str, result: dict) -> str:
    """Shared prefix for exec-summary headlines.

    Default config and experimental (non-promoted) configs each get a
    cue prefix so the reader knows the result isn't the locked baseline.
    """
    meta = result.get("meta") or {}
    if label == "default":
        return "**Default config** — unoptimized baseline. "
    if meta.get("promoted") is False:
        return "**Experimental config** — "
    return ""


def _summary_default_caveat() -> str:
    return ("This is the unoptimized baseline configuration, not the "
            "locked V1 strategy. Numbers shown are illustrative.")


def _exec_summary_performance(label: str, config: BacktestConfig,
                              result: dict) -> str:
    """Data-driven exec summary for the Performance tab.

    Headline / detail / caveat pulled from the loaded result. The
    adjective in the headline ('strongly outperforms', 'underperforms',
    etc.) is threshold-templated so it always agrees with the KPI cards
    rendered immediately below."""
    portfolio_df = result.get("portfolio_df")
    if portfolio_df is None or portfolio_df.empty:
        return "*Performance summary will appear when results are loaded.*"
    meta = result.get("meta") or {}
    components = meta.get("components") or {}

    pv = portfolio_df["portfolio_value"]
    start = portfolio_df.index[0].strftime("%Y-%m-%d")
    end   = (portfolio_df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_close = cached_benchmark("SPY", start, end)

    total_pct = (pv.iloc[-1] / pv.iloc[0] - 1) * 100.0
    n_days = max(len(pv), 1)
    sys_ann = (pv.iloc[-1] / pv.iloc[0]) ** (252 / n_days) - 1
    spy_pct   = ((spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100.0
                 if not spy_close.empty else float("nan"))
    arith_pp  = float("nan")
    beta      = float("nan")
    if not spy_close.empty:
        spy_ann = (spy_close.iloc[-1] / spy_close.iloc[0]) ** (252 / n_days) - 1
        # Compound-annualized arithmetic alpha (matches the +63.7pp
        # number cited in project documentation for Trial #325).
        arith_pp = (sys_ann - spy_ann) * 100
        pr = pv.pct_change().dropna()
        sr = spy_close.pct_change().dropna()
        common = pr.index.intersection(sr.index)
        if len(common) > 1 and sr.loc[common].std() > 0:
            cov = np.cov(pr.loc[common], sr.loc[common])
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0.0
    max_dd = abs(float((pv / pv.cummax() - 1).min())) * 100

    if   arith_pp >= 30: strength = "strongly outperforms"
    elif arith_pp >= 10: strength = "outperforms"
    elif arith_pp >= 0:  strength = "modestly beats"
    else:                strength = "underperforms"

    if   max_dd >= 25: dd_note = f"a meaningful -{max_dd:.0f}% peak-to-trough drop"
    elif max_dd >= 15: dd_note = f"a -{max_dd:.0f}% maximum drawdown"
    else:              dd_note = f"a manageable -{max_dd:.0f}% drawdown"

    beta_phrase = ""
    if not pd.isna(beta):
        if beta >= 1.20:
            beta_phrase = (f" The strategy also moves more than the "
                           f"market (beta around {beta:.2f}) — so some "
                           f"outperformance reflects amplified market "
                           f"exposure rather than pure stock selection.")
        elif beta < 0.80:
            beta_phrase = (f" The strategy moves less than the market "
                           f"(beta around {beta:.2f}), so a slice of "
                           f"the comfort vs SPY comes from lower "
                           f"market exposure.")

    headline = (f"This strategy {strength} the S&P 500 over the "
                f"validation window: "
                f"{total_pct:+.1f}% total return vs SPY's "
                f"{spy_pct:+.1f}% (annualized {arith_pp:+.1f}pp).")
    detail = (f"Performance came with {dd_note} along the way."
              f"{beta_phrase}")

    if label == "default":
        caveat = _summary_default_caveat()
    else:
        caveat = ("The 2024–2026 validation period was a strong bull "
                  "market for tech and quality stocks, which align "
                  "with this strategy's selection criteria. Backtest "
                  "results don't include real-money frictions "
                  "(slippage, taxes), and performance in a different "
                  "market regime could be materially different.")

    return f"{_summary_caveat_prefix(label, result)}{headline}\n\n{detail}\n\n*{caveat}*"


def tab_performance(label: str, config: BacktestConfig, result: dict) -> None:
    st.header("Performance")
    st.caption(f"Showing config: **{label}** "
               f"(window {config.validate_start} → {config.validate_end})")

    portfolio_df = result["portfolio_df"]
    trades_df    = result.get("trades_df")
    meta         = result.get("meta", {}) or {}

    if portfolio_df.empty:
        st.warning("Empty backtest result.")
        return

    # Exec summary (Phase 4 will populate; show placeholder for now)
    st.info(_exec_summary_performance(label, config, result))
    st.divider()

    pv = portfolio_df["portfolio_value"]
    start = portfolio_df.index[0].strftime("%Y-%m-%d")
    end = (portfolio_df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_close = cached_benchmark("SPY", start, end)
    qqq_close = cached_benchmark("QQQ", start, end)

    # ===== Layer 1 — Quick inference =====
    total_return_pct = (pv.iloc[-1] / pv.iloc[0] - 1) * 100.0
    spy_total_pct = float("nan")
    if not spy_close.empty:
        spy_total_pct = (spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100.0

    # CAPM alpha (risk-adjusted excess return) and beta
    alpha_ann_pp = float("nan")
    arith_alpha_ann_pp = float("nan")
    beta = float("nan")
    if not spy_close.empty:
        port_ret = pv.pct_change().dropna()
        spy_ret  = spy_close.pct_change().dropna()
        common = port_ret.index.intersection(spy_ret.index)
        pr = port_ret.loc[common]
        sr = spy_ret.loc[common]
        if len(pr) > 1 and sr.std() > 0:
            cov = np.cov(pr, sr)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0.0
            alpha_ann_pp = (pr.mean() - beta * sr.mean()) * 252 * 100
            # Arithmetic alpha = compound-annualized strategy return
            # minus compound-annualized SPY return — matches the
            # +63.7pp figure cited in project docs for Trial #325.
            n_days_p = max(len(pv), 1)
            sys_ann_r = (pv.iloc[-1] / pv.iloc[0]) ** (252 / n_days_p) - 1
            spy_ann_r = (spy_close.iloc[-1] / spy_close.iloc[0]) ** (252 / n_days_p) - 1
            arith_alpha_ann_pp = (sys_ann_r - spy_ann_r) * 100

    # Worst drawdown
    sys_dd_pct = (pv / pv.cummax() - 1.0) * 100.0
    worst_dd_pct = float(sys_dd_pct.min())

    # Win rate: prefer rolling 12-month positive-alpha share if available
    win_rate_label = ""
    win_rate_value = ""
    rolling_12mo = ((meta.get("rolling_metrics") or {}).get("rolling_12mo") or {})
    ad = rolling_12mo.get("alpha_distribution_stats") or {}
    if "count_positive" in ad and "count_total" in ad and ad["count_total"]:
        pct_pos = ad["count_positive"] / ad["count_total"] * 100
        win_rate_label = "12-month windows positive"
        win_rate_value = (f"{pct_pos:.0f}% "
                          f"({ad['count_positive']}/{ad['count_total']})")
    else:
        # Fallback: closed-trade win rate (returns_pct > 0).
        rt = round_trip_trades(
            trades_df, portfolio_df.index[-1]) if trades_df is not None else None
        if rt is not None and not rt.empty:
            closed = rt[rt["reason"] != "Open"]
            if not closed.empty:
                pct_pos = (closed["return_pct"] > 0).mean() * 100
                win_rate_label = "Closed-trade win rate"
                win_rate_value = f"{pct_pos:.0f}%"

    cols = st.columns(4)
    cols[0].metric(
        "Total return",
        f"{total_return_pct:+.1f}%",
        delta=(f"SPY {spy_total_pct:+.1f}%" if not pd.isna(spy_total_pct) else None),
        delta_color="off",
        help=f"Strategy growth over {config.validate_start} to "
             f"{config.validate_end}. SPY did "
             f"{spy_total_pct:+.1f}% over the same window."
        if not pd.isna(spy_total_pct)
        else f"Strategy growth over {config.validate_start} to "
             f"{config.validate_end}.",
    )
    cols[1].metric(
        "Risk-adjusted excess return",
        f"{alpha_ann_pp:+.1f}pp/yr" if not pd.isna(alpha_ann_pp) else "—",
        help=(f"Annualized return above SPY, adjusted for the strategy "
              f"moving more than the market (beta = "
              f"{beta:.2f}). The unadjusted spread is "
              f"{arith_alpha_ann_pp:+.1f}pp."
              if not pd.isna(beta) else
              "Annualized excess return after beta adjustment (Jensen's alpha)."),
    )
    cols[2].metric(
        "Worst drawdown",
        f"{worst_dd_pct:.1f}%",
        help="The biggest peak-to-trough drop the strategy experienced "
             "during the validation window.",
    )
    cols[3].metric(
        win_rate_label or "Win rate",
        win_rate_value or "—",
        help=("Of the rolling 12-month windows in validation, this share "
              "had positive alpha vs SPY."
              if win_rate_label.startswith("12-month")
              else "Closed round-trips with positive return."),
    )

    # Hero chart — Strategy vs SPY vs QQQ normalized to 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pv.index, y=pv / pv.iloc[0] * 100.0,
        name="Strategy", mode="lines",
        line=dict(color="#2563eb", width=2),
    ))
    if not spy_close.empty:
        fig.add_trace(go.Scatter(
            x=spy_close.index, y=spy_close / spy_close.iloc[0] * 100.0,
            name="SPY", mode="lines", line=dict(color="#f59e0b", width=2),
        ))
    if not qqq_close.empty:
        fig.add_trace(go.Scatter(
            x=qqq_close.index, y=qqq_close / qqq_close.iloc[0] * 100.0,
            name="QQQ", mode="lines", line=dict(color="#10b981", width=2),
        ))
    fig.update_layout(
        title="Strategy vs SPY vs QQQ (start = 100)",
        yaxis_title="Indexed value", xaxis_title="",
        height=420, margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===== Layer 2 — Visual breakdown =====
    st.divider()
    monthly_df, monthly_label = _monthly_returns(pv, spy_close)
    if not monthly_df.empty:
        st.markdown(f"**Performance month by month — strategy returns vs "
                    f"SPY returns** ({monthly_label}).")
        bars = go.Figure()
        bars.add_trace(go.Bar(
            x=monthly_df.index, y=monthly_df["Strategy"],
            name="Strategy", marker_color="#2563eb",
        ))
        if "SPY" in monthly_df.columns:
            bars.add_trace(go.Bar(
                x=monthly_df.index, y=monthly_df["SPY"],
                name="SPY", marker_color="#f59e0b",
            ))
        bars.update_layout(
            barmode="group",
            yaxis_title="Monthly return (%)", xaxis_title="",
            height=380, margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified", legend=dict(orientation="h"),
        )
        st.plotly_chart(bars, use_container_width=True)

    # ===== Layer 3 — Detailed view =====
    st.divider()
    st.subheader("Detailed metrics")
    fees = float(portfolio_df["total_fees"].iloc[-1])
    metrics = _sleeve_metrics(pv, trades_df, fees)
    cols = st.columns(5)
    cols[0].metric("Total return (raw)", metrics["Total Return"])
    cols[1].metric("Sharpe", metrics["Sharpe Ratio"])
    cols[2].metric("Max DD", metrics["Max Drawdown"])
    cols[3].metric("Trades", metrics["Total Trades"])
    cols[4].metric("Win rate (trades)", metrics["Win Rate"])

    if not pd.isna(beta):
        # The 4 alpha cards from commit 37c5b49 — preserved per spec.
        corr = pr.corr(sr) if not pd.isna(beta) else float("nan")
        cols2 = st.columns(4)
        cols2[0].metric(
            "Excess return vs SPY (annualized)",
            f"{arith_alpha_ann_pp:+.2f}pp",
            help="Strategy annualized return minus SPY annualized "
                 "return. Does not adjust for beta. This is the headline "
                 "+63.7pp number cited in project documentation.",
        )
        cols2[1].metric(
            "Risk-adjusted excess return (CAPM)",
            f"{alpha_ann_pp:+.2f}pp",
            help="Jensen's alpha: excess annualized return after "
                 "removing the beta-amplified SPY contribution. Lower "
                 "than arithmetic alpha when beta > 1 because some "
                 "outperformance is attributed to amplified market "
                 "exposure.",
        )
        cols2[2].metric("Beta vs SPY", f"{beta:.2f}",
                        help="Sensitivity to SPY moves. >1 means the "
                             "strategy moves more than the market on "
                             "average; <1 means less.")
        cols2[3].metric("Correlation vs SPY", f"{corr:.2f}",
                        help="Day-to-day return correlation with SPY. "
                             "1.0 = identical, 0 = unrelated.")


def _exec_summary_tuning_history(label: str, config: BacktestConfig,
                                 result: dict,
                                 study_name: str | None) -> str:
    if study_name is None:
        return ("*Pick a study from the sidebar to see tuning history.*")
    try:
        df = load_study_trials_df(study_name)
    except Exception:
        return "*Tuning history isn't available for this study.*"
    completes = df[df["state"] == "COMPLETE"]
    n_total = len(df)
    n_complete = len(completes)
    if completes.empty:
        return (f"*Study **{study_name}** has {n_total} trials but none "
                f"completed yet.*")
    best_row = completes.loc[completes["value"].idxmax()]
    best_n = int(best_row["trial_number"])
    best_score = float(best_row["value"])
    headline = (f"The optimizer tested **{n_total} configurations** for "
                f"this study ({n_complete} completed). The winner was "
                f"**Trial #{best_n}** with score **{best_score:.4f}**.")
    cumulative = completes.sort_values("trial_number")["value"].cummax()
    pct_to_best = ((cumulative >= best_score * 0.95).idxmax()
                   if not cumulative.empty else None)
    if pct_to_best is not None and len(cumulative) >= 1:
        pos = (cumulative.values >= best_score * 0.95).argmax()
        share = (pos + 1) / len(cumulative) * 100
        detail = (f"95% of the winning score was reached after about "
                  f"{share:.0f}% of the trials — the curve plateaus "
                  f"early, then refinement happens at the margin.")
    else:
        detail = ("Score progression is shown in Layer 2 below.")
    if label == "default":
        caveat = _summary_default_caveat()
    else:
        caveat = ("Optuna is search, not proof. A different random seed "
                  "or longer search might find a better config or might "
                  "find that this peak doesn't generalize to other "
                  "validation windows.")
    return f"{_summary_caveat_prefix(label, result)}{headline}\n\n{detail}\n\n*{caveat}*"


def tab_tuning_history(label: str, config: BacktestConfig, result: dict,
                       study_name: str | None) -> None:
    st.header("Tuning History — every configuration tested by the optimizer")
    if study_name is None:
        st.info("Pick a study from the sidebar (set Source to "
                "'Best trial of selected study' or 'Custom trial number') "
                "to explore trial-level results.")
        return

    st.info(_exec_summary_tuning_history(label, config, result, study_name))
    st.divider()

    df = load_study_trials_df(study_name)
    completes = df[df["state"] == "COMPLETE"]
    pruned    = df[df["state"] == "PRUNED"]
    failed    = df[df["state"] == "FAIL"]

    # ===== Layer 1 — Quick inference =====
    meta = result.get("meta") or {}
    runtime_s = meta.get("runtime_seconds")
    runtime_label = "—"
    if runtime_s:
        if runtime_s >= 3600:
            runtime_label = f"{runtime_s/3600:.1f} hours"
        else:
            runtime_label = f"{runtime_s/60:.0f} min"
    if not completes.empty:
        best_n = int(completes.loc[completes["value"].idxmax(), "trial_number"])
        best_v = float(completes["value"].max())
        best_str = f"#{best_n} (score {best_v:.4f})"
    else:
        best_str = "—"

    cols = st.columns(3)
    cols[0].metric("Configurations tested", f"{len(df):,} trials",
                   help=f"{len(completes)} completed, {len(pruned)} "
                        f"pruned, {len(failed)} failed.")
    cols[1].metric("Best trial chosen", best_str)
    cols[2].metric("Time spent tuning (best-trial save)", runtime_label,
                   help="Wall-clock seconds the saved best-trial "
                        "backtest took to re-run. Total study runtime "
                        "is much higher (1000s of trials).")

    if completes.empty:
        st.warning("No completed trials in this study.")
        return
    st.divider()

    # --- Trial number vs score with running best ---
    completes_sorted = completes.sort_values("trial_number")
    running_max = completes_sorted["value"].cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=completes_sorted["trial_number"], y=completes_sorted["value"],
        mode="markers", name="Trials",
        marker=dict(color="#94a3b8", size=5),
    ))
    fig.add_trace(go.Scatter(
        x=completes_sorted["trial_number"], y=running_max,
        mode="lines", name="Running best",
        line=dict(color="#2563eb", width=2),
    ))
    fig.update_layout(
        title=f"Trial scores — {study_display_name(study_name)}",
        xaxis_title="Trial #", yaxis_title="Score",
        height=380, margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Param vs score scatter for each tunable ---
    st.subheader("Parameter sensitivity")
    tunables = [c for c in ("weight_fundamental", "weight_technical",
                            "weight_model", "macro_threshold_low",
                            "macro_threshold_gap", "atr_multiplier",
                            "analyst_weight", "rebalance_frequency_days",
                            "position_count") if c in completes.columns]
    cols_per_row = 3
    for row_start in range(0, len(tunables), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, p in enumerate(tunables[row_start: row_start + cols_per_row]):
            with cols[j]:
                fig = px.scatter(
                    completes, x=p, y="value", trendline=None,
                    title=p, height=260,
                    color_discrete_sequence=["#2563eb"],
                )
                fig.update_traces(marker=dict(size=4, opacity=0.6))
                fig.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis_title="Score",
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Top 10 ---
    st.subheader("Top 10 trials")
    top10 = completes.nlargest(10, "value").reset_index(drop=True)
    top10.index = top10.index + 1  # rank
    top10.index.name = "rank"
    st.dataframe(top10, use_container_width=True)

    # --- Best trial drilldown ---
    st.subheader(f"Best trial drilldown: #{int(top10.iloc[0]['trial_number'])}")
    best_n = int(top10.iloc[0]["trial_number"])
    records = load_trial_jsonl_records(study_name)
    best_recs = [r for r in records if r.get("trial_number") == best_n]
    if best_recs:
        rec = best_recs[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Config**")
            st.json(rec.get("config", {}))
        with c2:
            st.markdown("**Components**")
            st.json(rec.get("components", {}))
    else:
        st.caption("No JSONL record for this trial.")


def _market_tier(score: float, low: float, high: float
                 ) -> tuple[str, str, str, str]:
    """Return (label, emoji, color, sizing) for the macro score under
    the given config's tier thresholds."""
    if score > high:
        return ("Bullish", "🟢", "#16a34a", "100%")
    if score >= low:
        return ("Mixed",   "🟡", "#ca8a04", "75%")
    return ("Stressed",   "🔴", "#dc2626", "50%")


def _exec_summary_market_context(label: str, config: BacktestConfig,
                                 result: dict) -> str:
    macro_df = load_macro_df()
    if macro_df is None or macro_df.empty:
        return ("*Market Context summary will appear once macro signals "
                "are loaded.*")
    today_score = compute_macro_score(macro_df, pd.Timestamp.today())
    tier, _emoji, _color, sizing = _market_tier(
        today_score, config.macro_threshold_low,
        config.macro_threshold_high)
    headline = (f"Today's market read is **{tier}** — the strategy is "
                f"sizing positions at **{sizing}** of target weight.")
    # Identify which raw components are stressed-side outliers (low z-score)
    raw = macro_df.dropna()
    if not raw.empty:
        latest = raw.iloc[-1]
        zs = (latest - raw.mean()) / raw.std().replace(0, np.nan)
        stressed = zs.abs().sort_values(ascending=False).head(2).index.tolist()
        detail = (f"The market health score reads {today_score:.3f}"
                  f"(thresholds for this config: low {config.macro_threshold_low:.3f}, "
                  f"high {config.macro_threshold_high:.3f}). The most "
                  f"unusual underlying signals right now are "
                  f"{' and '.join(stressed)}.")
    else:
        detail = (f"The market health score reads {today_score:.3f}"
                  f"(thresholds: {config.macro_threshold_low:.3f} / "
                  f"{config.macro_threshold_high:.3f}).")

    if label == "default":
        caveat = ("*This is the **Default config** — its thresholds "
                  f"({config.macro_threshold_low:.3f} / "
                  f"{config.macro_threshold_high:.3f}) are different "
                  "from the locked Trial #325 config, so the same "
                  "macro signal can give a different traffic-light "
                  "reading on the two configs. Toggle the sidebar to "
                  "compare.*")
    else:
        caveat = ("The market health score floor in the validation "
                  "period was 0.42, meaning the 'Stressed' tier never "
                  "fired during validation. The position-sizing overlay "
                  "is structurally present but has not been exercised "
                  "in current data.")
    return f"{_summary_caveat_prefix(label, result)}{headline}\n\n{detail}\n\n*{caveat}*"


def tab_market_context(label: str, config: BacktestConfig, result: dict) -> None:
    st.header("Market Context")
    st.caption(f"Showing config: **{label}** "
               f"(window {config.validate_start} → {config.validate_end})")

    macro_df = load_macro_df()
    if macro_df.empty:
        st.warning("No macro cache. Run the macro pipeline first.")
        return

    st.info(_exec_summary_market_context(label, config, result))
    st.divider()

    # ===== Layer 1 — Quick inference (traffic light) =====
    today_score = compute_macro_score(macro_df, pd.Timestamp.today())
    tier, emoji, color, sizing = _market_tier(
        today_score, config.macro_threshold_low,
        config.macro_threshold_high)
    cols = st.columns([2, 3])
    with cols[0]:
        st.markdown(
            f"<div style='padding:18px 20px;border-radius:12px;"
            f"background:{color}22;border-left:6px solid {color};"
            f"font-size:1.7em;line-height:1.2'>"
            f"{emoji} <b>{tier}</b><br>"
            f"<span style='font-size:0.55em;color:#475569'>"
            f"score {today_score:.3f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"The strategy currently uses **{sizing} position sizing** "
            f"based on the macro signal. "
            f"Thresholds for the selected config: low "
            f"{config.macro_threshold_low:.3f}, high "
            f"{config.macro_threshold_high:.3f}."
        )

    # ===== Layer 2 — Composite over time =====
    st.divider()
    st.markdown("**Market health score over time.** Higher = healthier; "
                "lower = more stressed. Shaded bands show the sizing "
                "tiers under the selected config.")
    scores = macro_score_series(macro_df)
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=config.macro_threshold_low,
                  fillcolor="#fca5a5", opacity=0.18, line_width=0,
                  annotation_text="50% sizing",
                  annotation_position="top left")
    fig.add_hrect(y0=config.macro_threshold_low,
                  y1=config.macro_threshold_high,
                  fillcolor="#fde68a", opacity=0.18, line_width=0,
                  annotation_text="75% sizing",
                  annotation_position="top left")
    fig.add_hrect(y0=config.macro_threshold_high, y1=1,
                  fillcolor="#86efac", opacity=0.18, line_width=0,
                  annotation_text="100% sizing",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=scores.index, y=scores,
        mode="lines", name="Market health score",
        line=dict(color="#1e293b", width=1.5),
    ))
    fig.update_layout(
        xaxis_title="", yaxis_title="Score (0 = stressed, 1 = healthy)",
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===== Layer 3 — Underlying components (expander) =====
    with st.expander("Underlying components (raw FRED + SPY drawdown)",
                      expanded=False):
        st.caption("Each panel shows one raw input that feeds into the "
                   "composite score. Technical labels and FRED series "
                   "IDs preserved for reference.")
        pcts = scores.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        cols = st.columns(7)
        cols[0].metric("Mean",      f"{scores.mean():.3f}")
        cols[1].metric("Std",       f"{scores.std():.3f}")
        cols[2].metric("p10",       f"{pcts.loc[0.10]:.3f}")
        cols[3].metric("p25",       f"{pcts.loc[0.25]:.3f}")
        cols[4].metric("Median",    f"{pcts.loc[0.50]:.3f}")
        cols[5].metric("p75",       f"{pcts.loc[0.75]:.3f}")
        cols[6].metric("p90",       f"{pcts.loc[0.90]:.3f}")
        st.divider()
        raw_cols = list(macro_df.columns)
        cols_per_row = 2
        for row_start in range(0, len(raw_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(raw_cols[row_start: row_start + cols_per_row]):
                series = macro_df[col].dropna()
                if series.empty:
                    continue
                with cols[j]:
                    fred_id = FRED_SERIES.get(col, "—")
                    fig = px.line(series,
                                  title=f"{col}  ({fred_id})",
                                  height=240,
                                  color_discrete_sequence=["#2563eb"])
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=False,
                        yaxis_title="", xaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)


def _exec_summary_holdings(label: str, config: BacktestConfig,
                           result: dict) -> str:
    holdings = result.get("holdings", {}) or {}
    if not holdings:
        return ("*No open positions at the end of the backtest window. "
                "The strategy held cash through the final trading day.*")
    sector_map = load_sector_map() or {}
    sec_df = _holdings_sector_values(holdings, sector_map, None)
    n_pos = len(holdings)
    n_sec = len(sec_df) if not sec_df.empty else 0
    top_sector = sec_df.iloc[0]["sector"] if not sec_df.empty else "—"
    top_sec_pct = sec_df.iloc[0]["pct"] if not sec_df.empty else 0.0
    largest_pos = max(holdings.items(),
                      key=lambda kv: float(kv[1]["shares"]) * float(kv[1]["entry_price"]))
    largest_t = largest_pos[0]
    largest_v = float(largest_pos[1]["shares"]) * float(largest_pos[1]["entry_price"])
    total_v = sum(float(h["shares"]) * float(h["entry_price"])
                  for h in holdings.values())
    largest_pct = largest_v / total_v * 100.0 if total_v else 0.0

    if   n_sec <= 2: conc = "highly sector-concentrated"
    elif n_sec <= 4: conc = "moderately concentrated"
    else:            conc = "diversified across sectors"

    headline = (f"The portfolio holds **{n_pos} positions** across "
                f"**{n_sec} sectors** as of the last backtest day — "
                f"{conc} by design.")
    detail = (f"The largest position is **{largest_t}** at "
              f"{largest_pct:.0f}% of cost basis; the largest sector "
              f"exposure is **{top_sector}** at {top_sec_pct:.0f}%.")
    if label == "default":
        caveat = _summary_default_caveat()
    else:
        caveat = (f"A {n_pos}-position portfolio carries materially more "
                  f"single-name risk than a diversified ETF. If any one "
                  f"of these names blows up, performance suffers "
                  f"disproportionately.")
    return f"{_summary_caveat_prefix(label, result)}{headline}\n\n{detail}\n\n*{caveat}*"


def tab_holdings(label: str, config: BacktestConfig, result: dict) -> None:
    st.header("Current Holdings")
    st.caption(f"Showing config: **{label}** "
               f"(window {config.validate_start} → {config.validate_end})")

    holdings = result.get("holdings", {}) or {}
    portfolio_df = result.get("portfolio_df")
    scores = result.get("scores", {}) or {}
    trades_df = result.get("trades_df")

    st.info(_exec_summary_holdings(label, config, result))
    st.divider()

    if not holdings:
        st.info("No open positions at end of backtest window.")
        return

    end_date = portfolio_df.index[-1] if portfolio_df is not None and \
        not portfolio_df.empty else None
    sector_map = load_sector_map() or {}
    ticker_names = load_ticker_names() or {}
    total_pv = (float(portfolio_df["portfolio_value"].iloc[-1])
                if portfolio_df is not None and not portfolio_df.empty
                else None)
    sec_df = _holdings_sector_values(holdings, sector_map, total_pv)

    # ===== Layer 1 — Quick inference =====
    n_sectors_universe = len(set(sector_map.values())) or 11
    cols = st.columns(3)
    cols[0].metric("Holdings", f"{len(holdings)} positions")
    cols[1].metric("Total value (latest backtest day)",
                   f"${total_pv:,.0f}" if total_pv else "—")
    cols[2].metric("Sectors represented",
                   f"{len(sec_df)} of {n_sectors_universe}")

    if not sec_df.empty:
        pie = go.Figure(data=[go.Pie(
            labels=sec_df["sector"], values=sec_df["value"],
            hole=0.45, sort=False,
            textinfo="label+percent",
        )])
        pie.update_layout(
            title="Allocation by sector (cost basis)",
            height=380, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(pie, use_container_width=True)

    # ===== Layer 2 — Visual breakdown =====
    st.divider()
    st.markdown("**Holdings detail.** Sortable table — click any column "
                "header to sort.")
    entry_dates: dict[str, pd.Timestamp] = {}
    if trades_df is not None and not trades_df.empty:
        td = trades_df.copy()
        td["date"] = pd.to_datetime(td["date"])
        for tkr in holdings.keys():
            tk = td[(td["ticker"] == tkr) & (td["action"] == "BUY")]
            if not tk.empty:
                entry_dates[tkr] = tk["date"].max()
    rows = []
    for tkr, h in holdings.items():
        shares  = float(h["shares"])
        entry   = float(h["entry_price"])
        cost    = shares * entry
        days_h  = ((end_date - entry_dates[tkr]).days
                   if (end_date is not None and tkr in entry_dates) else None)
        score   = scores.get(tkr, {}).get("composite", float("nan"))
        sector  = sector_map.get(tkr) or "Other"
        company = ticker_names.get(tkr, "")
        pct = (cost / total_pv * 100.0) if total_pv else float("nan")
        rows.append({
            "Ticker":          tkr,
            "Company":         company,
            "Sector":          sector,
            "$ Value":         round(cost, 2),
            "% of Portfolio":  round(pct, 2) if pct == pct else None,
            "Composite Score": round(float(score), 3) if pd.notna(score) else None,
            "Days Held":       days_h,
        })
    df = pd.DataFrame(rows).sort_values("$ Value", ascending=False)
    _render_df_with_ticker_links(df, use_container_width=True, hide_index=True)

    st.caption("`$ Value` is cost basis (shares × entry price). Live "
               "mark-to-market values are not fetched in the read-only "
               "dashboard.")

    # ===== Layer 3 — Detailed view (expander) =====
    with st.expander("Per-holding score breakdown", expanded=False):
        st.markdown(
            "*Per-holding sub-score breakdown (fundamental / technical / "
            "model / alt) is not currently emitted by the backtest at "
            "save time. Composite scores above are the most granular "
            "detail available; sub-score attribution will be added when "
            "the attribution layer ships.*"
        )


def _exec_summary_trade_history(label: str, config: BacktestConfig,
                                result: dict, rt: pd.DataFrame,
                                window_label: str) -> str:
    if rt is None or rt.empty:
        return "*No trades in the selected period.*"
    closed = rt[rt["reason"] != "Open"]
    n_trades = len(rt)
    if closed.empty:
        return (f"In the selected period ({window_label}) the strategy "
                f"opened {n_trades} positions, none yet closed.")
    n_closed = len(closed)
    win_rate = (closed["return_pct"] > 0).mean() * 100.0
    # Top winner / loser by ticker (aggregate $ P&L across all that ticker's pairs)
    closed_with_pnl = closed.dropna(subset=["pnl_dollars"])
    if not closed_with_pnl.empty:
        by_ticker = closed_with_pnl.groupby("ticker")["pnl_dollars"].sum().sort_values()
        worst_t, worst_pnl = by_ticker.index[0], float(by_ticker.iloc[0])
        best_t,  best_pnl  = by_ticker.index[-1], float(by_ticker.iloc[-1])
    else:
        worst_t = best_t = "—"
        worst_pnl = best_pnl = 0.0
    activity = "steady" if n_trades >= 30 else "modest"
    headline = (f"In the selected period ({window_label}) the strategy "
                f"made **{n_trades} trades** ({n_closed} closed). "
                f"Biggest winner: **{best_t}** "
                f"(+${best_pnl:,.0f}). Biggest loser: **{worst_t}** "
                f"(${worst_pnl:,.0f}).")
    detail = (f"Win rate on closed trades: **{win_rate:.0f}%**. "
              f"Activity is {activity} for the window length.")
    if label == "default":
        caveat = _summary_default_caveat()
    else:
        caveat = ("Realized P&L only — open positions at end of the "
                  "period are not counted. FIFO matching attributes "
                  "gains/losses imperfectly when positions are "
                  "partial-sized.")
    return f"{_summary_caveat_prefix(label, result)}{headline}\n\n{detail}\n\n*{caveat}*"


def tab_trade_history(label: str, config: BacktestConfig, result: dict) -> None:
    st.header("Trade History")
    st.caption(f"Showing config: **{label}** "
               f"(window {config.validate_start} → {config.validate_end})")

    trades_df = result.get("trades_df")
    portfolio_df = result.get("portfolio_df")
    if trades_df is None or trades_df.empty:
        st.info("No trades in this backtest.")
        return

    end_date = portfolio_df.index[-1] if portfolio_df is not None and \
        not portfolio_df.empty else pd.Timestamp.today()
    rt_full = round_trip_trades(trades_df, end_date)
    if rt_full.empty:
        st.info("No round trips to display.")
        return

    # ---- Date filter (above Layer 1) ----
    full_min = rt_full["buy_date"].min().to_pydatetime().date()
    full_max = rt_full["sell_date"].max().to_pydatetime().date()
    date_pick = st.date_input(
        "Date range (filters every section below)",
        value=(full_min, full_max),
        min_value=full_min, max_value=full_max,
        format="YYYY-MM-DD",
    )
    if isinstance(date_pick, tuple) and len(date_pick) == 2:
        d_start, d_end = pd.Timestamp(date_pick[0]), pd.Timestamp(date_pick[1])
    else:
        d_start, d_end = pd.Timestamp(full_min), pd.Timestamp(full_max)
    # Filter: include trades whose buy_date is inside the window
    rt = rt_full[(rt_full["buy_date"] >= d_start)
                 & (rt_full["buy_date"] <= d_end + pd.Timedelta(days=1))].copy()
    window_label = f"{d_start.date()} → {d_end.date()}"

    st.info(_exec_summary_trade_history(label, config, result, rt, window_label))
    st.divider()

    if rt.empty:
        st.warning("No trades in the selected date range.")
        return

    # ===== Layer 1 — Quick inference (top winners + losers by ticker) =====
    closed = rt[rt["reason"] != "Open"].dropna(subset=["pnl_dollars"]).copy()
    by_ticker = (closed.groupby("ticker")["pnl_dollars"].sum().sort_values()
                 if not closed.empty else pd.Series(dtype=float))
    top_winners = by_ticker.tail(5).iloc[::-1] if not by_ticker.empty else pd.Series(dtype=float)
    top_losers  = by_ticker.head(5)            if not by_ticker.empty else pd.Series(dtype=float)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 5 winners** (by total $ P&L)")
        if not top_winners.empty:
            f = go.Figure(data=[go.Bar(
                x=top_winners.values, y=top_winners.index, orientation="h",
                marker_color="#16a34a",
                text=[f"${v:,.0f}" for v in top_winners.values],
                textposition="auto",
            )])
            f.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="$ profit", yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(f, use_container_width=True)
        else:
            st.caption("No closed winners in this period.")
    with c2:
        st.markdown("**Top 5 losers** (by total $ P&L)")
        if not top_losers.empty:
            f = go.Figure(data=[go.Bar(
                x=top_losers.values, y=top_losers.index, orientation="h",
                marker_color="#dc2626",
                text=[f"${v:,.0f}" for v in top_losers.values],
                textposition="auto",
            )])
            f.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="$ loss", yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(f, use_container_width=True)
        else:
            st.caption("No closed losers in this period.")

    # ===== Layer 2 — Visual breakdown =====
    st.divider()
    st.markdown("**Activity over time, return distribution, and average hold.**")
    avg_hold = (closed["hold_days"].mean() if not closed.empty
                else float("nan"))
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        # Trade count by month (BUY events only — counts entries, not pairs)
        td = trades_df.copy()
        td["date"] = pd.to_datetime(td["date"])
        td_filt = td[(td["date"] >= d_start) & (td["date"] <= d_end)
                     & (td["action"] == "BUY")]
        if not td_filt.empty:
            by_month = td_filt.set_index("date").resample("ME").size()
            f = go.Figure(data=[go.Bar(
                x=by_month.index, y=by_month.values,
                marker_color="#2563eb",
            )])
            f.update_layout(
                title="Trade count by month (entries)",
                height=260, margin=dict(l=10, r=10, t=40, b=10),
                yaxis_title="Trades",
            )
            st.plotly_chart(f, use_container_width=True)
    with c2:
        if not closed.empty:
            f = go.Figure(data=[go.Histogram(
                x=closed["return_pct"], nbinsx=20,
                marker_color="#475569",
            )])
            f.update_layout(
                title="Return distribution (closed trades)",
                height=260, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Return (%)", yaxis_title="Count",
            )
            st.plotly_chart(f, use_container_width=True)
    with c3:
        st.markdown("&nbsp;")  # spacer
        st.metric(
            "Average hold days",
            f"{avg_hold:.0f} days" if not pd.isna(avg_hold) else "—",
            help="Mean number of calendar days between entry and exit "
                 "for closed round trips in the selected window.",
        )

    # ===== Layer 3 — Detailed view (full filtered table) =====
    st.divider()
    st.markdown("**Full trade log for selected period.** Most recent first.")
    show = rt.copy().sort_values("buy_date", ascending=False)
    show["buy_date"]  = show["buy_date"].dt.strftime("%Y-%m-%d")
    show["sell_date"] = show["sell_date"].dt.strftime("%Y-%m-%d")
    show["buy_price"] = show["buy_price"].round(2)
    show["sell_price"] = show["sell_price"].round(2)
    show["return_pct"] = show["return_pct"].round(2)
    show["pnl_dollars"] = show["pnl_dollars"].round(2)
    _render_df_with_ticker_links(show, use_container_width=True, hide_index=True)


_ROBUSTNESS_THRESHOLD = 0.40   # rolling_12mo_objective bar for "still robust"
_ROBUSTNESS_DEAD_BAND = 0.04   # ±r12_obj range that counts as "dead axis"
_RELIABILITY_ALPHA_BAR = 0.30  # +30pp annualized alpha threshold (Reliability headline)

# Plain-English axis labels per spec section 5 translation table.
# Falls back to the raw name if not in the map.
_ROBUSTNESS_AXIS_LABELS: dict[str, str] = {
    "atr_multiplier_offensive":           "Stop-loss tightness",
    "macro_threshold_low":                "Market-stress sizing threshold",
    "position_count_offensive":           "Number of holdings",
    "rebalance_frequency_days_offensive": "Rebalance frequency (days)",
    "regime_threshold":                   "Defensive-mode trigger",
    "weight_fundamental_offensive":       "Fundamentals weight",
    "weight_model_offensive":             "ML model weight",
    "weight_technical_offensive":         "Technical analysis weight",
}

# New plain-English classification labels per spec section 5.
_SENSITIVITY_LABELS: dict[str, str] = {
    "Dead axis":              "No effect",
    "Robust plateau":         "Stable",
    "Peak with sensitivity":  "Sensitive",
    "Knife edge":             "Very sensitive",
    "Mixed":                  "Mixed sensitivity",
}


def _classify_axis(values: pd.DataFrame) -> tuple[str, str]:
    """Classify an axis's behavior + auto-generate a one-sentence note.

    values is the per-axis subframe (5 rows) sorted by `value`. Returns
    (classification, note). Rules per the V3 spec:
      - All 5 within ±dead_band of reference: "Dead axis"
      - 4-5 of 5 above threshold: "Robust plateau"
      - 2-3 of 5 above threshold with reference at peak: "Peak with sensitivity"
      - 1-2 of 5 above threshold: "Knife edge"
      - else: "Mixed"
    """
    thr = _ROBUSTNESS_THRESHOLD
    r12 = values["rolling_12mo_objective"]
    n_above = int((r12 >= thr).sum())
    ref_row = values[values["is_reference"]]
    ref_val = float(ref_row["rolling_12mo_objective"].iloc[0]) if not ref_row.empty else float(r12.max())
    spread = float(r12.max() - r12.min())
    n_trades_unique = values["n_trades"].nunique()
    alpha_spread_pp = float(values["alpha_ann"].max() - values["alpha_ann"].min()) * 100

    if spread <= _ROBUSTNESS_DEAD_BAND:
        cls = "Dead axis"
        if n_trades_unique == 1:
            note = (f"Zero change in trade count or alpha across the "
                    f"tested range — likely indicates this axis isn't "
                    f"materially exercised in the validation window.")
        else:
            note = (f"r12_obj spread only {spread:.3f} across the "
                    f"tested range — strategy is structurally insensitive "
                    f"to this axis here.")
    elif n_above >= 4:
        cls = "Robust plateau"
        note = (f"{n_above}/5 perturbations stay above the {thr:.2f} "
                f"threshold; alpha varies by {alpha_spread_pp:.1f}pp "
                f"across the range.")
    elif n_above >= 2 and abs(ref_val - r12.max()) < 1e-6:
        cls = "Peak with sensitivity"
        note = (f"Reference value sits at the peak; {5 - n_above}/5 "
                f"alternative values fall below {thr:.2f}. Alpha drops "
                f"up to {alpha_spread_pp:.1f}pp at the extremes.")
    elif n_above <= 2:
        cls = "Knife edge"
        note = (f"Only {n_above}/5 perturbations stay above {thr:.2f}; "
                f"performance is sensitive to this axis (alpha range "
                f"{alpha_spread_pp:.1f}pp).")
    else:
        cls = "Mixed"
        note = (f"{n_above}/5 perturbations above {thr:.2f}; alpha "
                f"range {alpha_spread_pp:.1f}pp.")
    return cls, note


def _reliability_axis_buckets(df: pd.DataFrame
                              ) -> tuple[list[str], list[str], list[str]]:
    """Classify each axis into 'most stable', 'most sensitive', and
    'one-sided room to move' buckets for the Reliability Layer 1 KPIs.

    - most_stable: classification 'Dead axis' (no observable effect)
    - most_sensitive: classification in {'Knife edge', 'Peak with sensitivity'}
    - one_sided: reference value sits at min or max of tested range AND
      the score holds up across the rest (>=3/5 above threshold)
    """
    most_stable, most_sensitive, one_sided = [], [], []
    for ax_name, sub in df.groupby("axis", sort=False):
        sub = sub.sort_values("value")
        cls, _note = _classify_axis(sub)
        friendly = _ROBUSTNESS_AXIS_LABELS.get(ax_name, ax_name)
        if cls == "Dead axis":
            most_stable.append(friendly)
        elif cls in ("Knife edge", "Peak with sensitivity"):
            most_sensitive.append(friendly)
        ref_sub = sub[sub["is_reference"]]
        if not ref_sub.empty:
            ref_v = float(ref_sub["value"].iloc[0])
            v_lo = float(sub["value"].min())
            v_hi = float(sub["value"].max())
            at_extreme = abs(ref_v - v_lo) < 1e-9 or abs(ref_v - v_hi) < 1e-9
            n_above = int((sub["rolling_12mo_objective"]
                           >= _ROBUSTNESS_THRESHOLD).sum())
            if at_extreme and n_above >= 3:
                one_sided.append(friendly)
    return most_stable, most_sensitive, one_sided


def _exec_summary_reliability(label: str, config: BacktestConfig,
                              result: dict) -> str:
    df = load_perturbation_summary()
    if df is None or df.empty:
        return ("*Reliability summary will appear once perturbation "
                "data is generated. See "
                "`src/v3_track2_runner.py`.*")
    df = df.copy()
    df["axis"] = df["axis"].astype(str)
    n_pert = int((~df["is_reference"]).sum())
    n_strong = int(((df["alpha_ann"] >= _RELIABILITY_ALPHA_BAR)
                    & ~df["is_reference"]).sum())
    n_close  = int(((df["rolling_12mo_objective"] >= _ROBUSTNESS_THRESHOLD)
                    & ~df["is_reference"]).sum())
    pct_strong = n_strong / n_pert * 100 if n_pert else 0.0
    pct_close  = n_close  / n_pert * 100 if n_pert else 0.0

    headline = (f"We tested the strategy with each of its 8 settings "
                f"tweaked up and down. Across **{n_pert}** tweaked "
                f"versions: **{n_strong} ({pct_strong:.0f}%)** still "
                f"substantially beat SPY (+{int(_RELIABILITY_ALPHA_BAR*100)}pp "
                f"annualized or better); "
                f"**{n_close} ({pct_close:.0f}%)** performed similarly "
                f"to the chosen version (12-month outperformance score "
                f"≥ {_ROBUSTNESS_THRESHOLD:.2f}).")
    detail = ("The strategy concept is robust. The specific peak result "
              "is sensitive to a few settings — see the surface plots "
              "and per-setting table below.")
    if label == "default":
        caveat = _summary_default_caveat()
    else:
        caveat = (f"Trial #325 sits at a TPE-found peak; some axes show "
                  f"~30pp alpha drop with small parameter shifts. The "
                  f"strategy concept is robust; the specific peak "
                  f"result is sensitive.")
    return f"{_summary_caveat_prefix(label, result)}{headline}\n\n{detail}\n\n*{caveat}*"


def tab_reliability(label: str, config: BacktestConfig, result: dict) -> None:
    st.header("Reliability")
    st.caption(f"Showing config: **{label}** "
               f"(window {config.validate_start} → {config.validate_end})")

    df = load_perturbation_summary()
    if df is None or df.empty:
        st.info(
            "No perturbation data found in this deployment. The "
            "Reliability tab surfaces V3 Track 2 single-axis "
            "perturbation results from "
            "`models/cache/dashboard_results/v3_track2_perturbation/"
            "summary_full.csv`. Generate via `python "
            "src/v3_track2_runner.py` (local) or wait for the next "
            "snapshot sync (cloud)."
        )
        return

    df = df.copy()
    df["axis"] = df["axis"].astype(str)

    st.info(_exec_summary_reliability(label, config, result))
    st.divider()

    # ===== Layer 1 — Quick inference =====
    most_stable, most_sensitive, one_sided = _reliability_axis_buckets(df)
    cols = st.columns(3)
    cols[0].metric(
        "Most stable settings (least sensitive)",
        ", ".join(most_stable) if most_stable else "—",
        help="Settings whose 5 tested values produce nearly identical "
             "results — tweaking them within the tested range has no "
             "observable effect on outperformance.",
    )
    cols[1].metric(
        "Most sensitive settings",
        ", ".join(most_sensitive) if most_sensitive else "—",
        help="Settings where moving the value drops outperformance "
             "noticeably — the chosen peak does not generalize across "
             "the whole tested range.",
    )
    cols[2].metric(
        "Settings with one-sided room to move",
        ", ".join(one_sided) if one_sided else "—",
        help="Settings where the chosen value sits at the edge of the "
             "tested range AND the score holds up across most of the "
             "range — could likely be moved further in one direction "
             "without losing alpha.",
    )

    # ===== Layer 2 — Per-axis surface grid =====
    st.divider()
    st.markdown(
        "**For each setting, we tested 5 different values.** The green "
        "star marks the chosen value; lines show how outperformance "
        "changes when that setting moves."
    )
    by_axis = df.groupby("axis", sort=False)
    thr = _ROBUSTNESS_THRESHOLD
    axes_sorted = [a for a in _ROBUSTNESS_AXIS_LABELS.keys()
                   if a in by_axis.groups]
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[_ROBUSTNESS_AXIS_LABELS.get(a, a)
                        for a in axes_sorted],
        specs=[[{"secondary_y": True}, {"secondary_y": True}]] * 4,
        vertical_spacing=0.08, horizontal_spacing=0.10,
    )
    for i, ax_name in enumerate(axes_sorted):
        sub = by_axis.get_group(ax_name).sort_values("value")
        row = i // 2 + 1
        col = i % 2 + 1
        ref_sub = sub[sub["is_reference"]]
        fig.add_trace(go.Scatter(
            x=sub["value"], y=sub["rolling_12mo_objective"],
            mode="lines+markers",
            name="Outperformance score", legendgroup="score",
            showlegend=(i == 0),
            line=dict(color="#2563eb", width=2),
            marker=dict(size=8, color="#2563eb"),
        ), row=row, col=col, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=sub["value"], y=sub["alpha_ann"],
            mode="lines+markers",
            name="Excess return vs SPY", legendgroup="alpha",
            showlegend=(i == 0),
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=7, color="#f59e0b", symbol="diamond"),
        ), row=row, col=col, secondary_y=True)
        if not ref_sub.empty:
            fig.add_trace(go.Scatter(
                x=ref_sub["value"],
                y=ref_sub["rolling_12mo_objective"],
                mode="markers", name="chosen value",
                legendgroup="ref", showlegend=(i == 0),
                marker=dict(size=14, color="#16a34a", symbol="star"),
            ), row=row, col=col, secondary_y=False)
        fig.add_hline(y=thr, line_dash="dash", line_color="#94a3b8",
                      line_width=1, row=row, col=col, secondary_y=False)
    fig.update_layout(
        height=1000, margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    for i in range(8):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.update_yaxes(
            title_text=("12-month outperformance score" if col == 1 else ""),
            row=row, col=col, secondary_y=False)
        fig.update_yaxes(
            title_text=("Excess return vs SPY" if col == 2 else ""),
            row=row, col=col, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # ===== Layer 3 — Per-axis interpretation table =====
    st.divider()
    st.subheader("Per-setting interpretation")
    rows = []
    for ax_name in axes_sorted:
        sub = by_axis.get_group(ax_name).sort_values("value")
        ref_sub = sub[sub["is_reference"]]
        ref_val = (float(ref_sub["value"].iloc[0])
                   if not ref_sub.empty else float("nan"))
        v_lo, v_hi = float(sub["value"].min()), float(sub["value"].max())
        r_lo = float(sub["rolling_12mo_objective"].min())
        r_hi = float(sub["rolling_12mo_objective"].max())
        cls, note = _classify_axis(sub)
        rows.append({
            "Setting":          _ROBUSTNESS_AXIS_LABELS.get(ax_name, ax_name),
            "Chosen value":     f"{ref_val:.3f}" if ref_val == ref_val else "—",
            "Range tested":     f"[{v_lo:.3f}, {v_hi:.3f}]",
            "Score range":      f"[{r_lo:.3f}, {r_hi:.3f}]",
            "Sensitivity":      _SENSITIVITY_LABELS.get(cls, cls),
            "Note":             note,
        })
    table_df = pd.DataFrame(rows)
    st.dataframe(
        table_df,
        use_container_width=True, hide_index=True,
        column_config={
            "Note": st.column_config.TextColumn(
                "Note", width="large", help="Auto-generated from the data"),
            "Setting":      st.column_config.TextColumn("Setting", width="medium"),
            "Sensitivity":  st.column_config.TextColumn("Sensitivity", width="small"),
        },
    )
    st.caption(
        "Sensitivity labels: **No effect** — score range "
        f"≤ {_ROBUSTNESS_DEAD_BAND} (this setting doesn't move the "
        "result in the tested range). "
        "**Stable** — 4-5/5 perturbations stay above threshold. "
        "**Sensitive** — chosen value at peak, others fall off. "
        "**Very sensitive** — only 1-2/5 above threshold."
    )


# Curated glossary covering the dashboard's user-visible jargon. Each
# entry is one sentence — meant for quick lookup, not deep explanation.
# The longer-form discussion lives in the downloadable user guide.
_GLOSSARY: dict[str, str] = {
    "Alpha (arithmetic)":
        "Annualized strategy return minus annualized SPY return — "
        "the simple performance spread, not adjusted for how much the "
        "strategy moves vs the market.",
    "Alpha (CAPM)":
        "Annualized excess return after subtracting the SPY contribution "
        "scaled by beta — i.e., the part of outperformance NOT explained "
        "by amplified market exposure. Also called Jensen's alpha.",
    "ATR multiplier":
        "Tunable that sets how much room a stop-loss gives a stock "
        "before triggering, measured in multiples of recent volatility "
        "(Average True Range). In the chart labels: 'Stop-loss tightness'.",
    "Backtest":
        "A simulation that replays a strategy over historical price "
        "data to see how it would have performed.",
    "Benchmark":
        "The reference index the strategy is compared against. Here it's "
        "SPY (the S&P 500) — sometimes also QQQ (the Nasdaq-100).",
    "Beta":
        "How much the strategy moves relative to the market. Beta = 1 "
        "means it moves in lockstep; >1 means more; <1 means less.",
    "Capture ratio (up / down)":
        "Of SPY's gains in up months, what % the strategy captures "
        "(up capture); of SPY's losses in down months, what % the "
        "strategy participates in (down capture).",
    "Composite score":
        "The blended ranking each stock gets from fundamental, technical, "
        "ML model, and alternative-data sub-scores. Higher = stronger "
        "candidate to hold.",
    "Correlation":
        "Day-to-day return correlation between the strategy and SPY. "
        "1.0 = move identically; 0 = unrelated; -1 = opposite.",
    "Defensive-mode trigger":
        "The macro-signal threshold below which the strategy switches "
        "from offensive to defensive parameter set. Inside the code: "
        "regime_threshold.",
    "Drawdown":
        "How far below the most recent peak the portfolio is at a given "
        "moment. Reported as a negative percentage; max drawdown is the "
        "worst dip ever seen during the period.",
    "Excess return vs SPY":
        "Strategy return minus SPY return over the same window. The "
        "annualized arithmetic version is shown in KPIs as 'Excess "
        "return vs SPY (annualized)'.",
    "FIFO (first-in-first-out)":
        "When a stock is bought multiple times then sold, the first "
        "buy is matched to the first sell for P&L purposes.",
    "Fundamentals weight":
        "Tunable share of the composite score that comes from company "
        "fundamentals (margins, growth, debt, valuation). Field name: "
        "weight_fundamental.",
    "Hold days":
        "Calendar days between entry and exit on a closed position.",
    "Macro overlay":
        "Position sizing tier (50% / 75% / 100%) that scales every "
        "holding's dollar exposure based on the current market health "
        "score. Conservative when stressed, full when bullish.",
    "Market health score":
        "A composite of macro signals (HY spread, yield curve, VIX, NFCI, "
        "Sahm, 10y-3m, SPY drawdown) blended into a single 0–1 number. "
        "Higher = healthier; lower = more stressed. Inside the code: "
        "the 'macro composite'.",
    "Market-stress sizing threshold":
        "Below this market-health score, the strategy drops to 50% "
        "position sizing. Inside the code: macro_threshold_low.",
    "ML model weight":
        "Tunable share of the composite score that comes from the XGBoost "
        "machine-learning model's prediction. Field name: weight_model.",
    "Number of holdings":
        "How many positions the strategy holds at any time (concentration "
        "tunable). Field name: position_count.",
    "Outperformance score (12-month)":
        "Custom optimization objective: roughly the 75th-percentile "
        "minus 25th-percentile of rolling 12-month CAPM alpha. Higher "
        "values mean consistent outperformance with limited downside. "
        "Inside the code: rolling_12mo_objective.",
    "Pruned trial":
        "An Optuna trial that was killed early before completing — "
        "either because the search-space sampling was invalid or the "
        "early scoring suggested it wouldn't be competitive.",
    "Rebalance frequency":
        "How often (in days) the strategy re-checks scores and rotates "
        "positions. Field name: rebalance_frequency_days.",
    "Recovery time":
        "How long it took the portfolio to return to peak value after "
        "a drawdown of the specified depth (e.g., -10% recovery time).",
    "Regime-dependent":
        "The architecture choice where the strategy uses different "
        "tunable values in defensive vs offensive market regimes, "
        "switched by a single macro-signal threshold.",
    "Risk-adjusted excess return":
        "Same as CAPM alpha — annualized return above SPY after removing "
        "the part attributable to amplified market exposure.",
    "Round trip":
        "A complete buy-then-sell cycle on a single ticker. Closed "
        "round trips have a realized return; open ones don't.",
    "Sharpe ratio":
        "Risk-adjusted return: annualized excess return divided by "
        "annualized volatility. Higher is better. Above 1.0 is generally "
        "good; above 2.0 is rare.",
    "Stop-loss":
        "A pre-set sell rule that closes a position if it drops to a "
        "given level, intended to cap downside on a single trade. The "
        "strategy uses an ATR-based trailing stop, clamped to a 5–15% "
        "band of entry price.",
    "Technical analysis weight":
        "Tunable share of the composite score that comes from price-"
        "based indicators (momentum, volatility, volume). Field name: "
        "weight_technical.",
    "TPE (Tree-structured Parzen Estimator)":
        "The Bayesian optimization algorithm Optuna uses by default to "
        "search for good configurations. It learns from past trial "
        "results to focus on promising regions of parameter space.",
    "Trial":
        "A single configuration tested by the optimizer. Each trial runs "
        "a full backtest with one specific set of tunable values.",
    "Win rate":
        "Either (a) the share of closed round-trip trades that ended "
        "profitable, or (b) the share of rolling 12-month windows in "
        "which the strategy beat SPY — KPI labels say which.",
}


def tab_glossary() -> None:
    st.header("Glossary & Help")

    query = st.text_input("Look up a term", value="",
                          placeholder="alpha, beta, drawdown, …")
    matches = _GLOSSARY
    if query:
        q = query.strip().lower()
        matches = {k: v for k, v in _GLOSSARY.items()
                   if q in k.lower() or q in v.lower()}
        st.caption(f"{len(matches)} of {len(_GLOSSARY)} terms match "
                   f"'{query}'.")
    else:
        st.caption(f"{len(_GLOSSARY)} terms — start typing to filter.")

    if not matches:
        st.info("No terms match that search. Try a different word, or "
                "browse the full list by clearing the search box.")
    else:
        for term in sorted(matches.keys(), key=str.lower):
            st.markdown(f"**{term}** — {matches[term]}")

    st.divider()
    guide_path = Path(__file__).parent / "paper_trader_user_guide.docx"
    if guide_path.exists():
        st.download_button(
            label="Download full User Guide (Word doc)",
            data=guide_path.read_bytes(),
            file_name="paper_trader_user_guide.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    else:
        st.warning("User guide file not found — please contact Mike.")


def _reconstruct_sector_weights(
    portfolio_df: pd.DataFrame,
    trades_df: pd.DataFrame | None,
    sector_map: dict,
) -> pd.DataFrame:
    """Reconstruct daily sector allocation by entry-cost weight.

    Walks the trades log chronologically: BUYs add a position at
    shares × buy_price (cost basis); SELL/STOP* removes the position
    entirely. At each portfolio_df date, snapshots the per-sector cost
    basis and divides by the day's total cost basis to get weights (%).

    Returns a DataFrame indexed by date with one column per sector.
    Missing sectors at a given date are 0%.

    Caveat: cost-basis weight, NOT mark-to-market. Per-position weights
    don't drift with intraday price moves; they only step when a position
    is opened or closed. Sufficient for a sector-composition diagnostic;
    not sufficient for exact daily portfolio dollar values."""
    if trades_df is None or trades_df.empty or portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame()
    td = trades_df.copy()
    td["date"] = pd.to_datetime(td["date"])
    td = td.sort_values("date").reset_index(drop=True)

    positions: dict[str, dict] = {}  # ticker -> {"shares", "cost"}
    rows: list[dict] = []
    trade_idx = 0
    n_trades = len(td)
    sell_actions = {"SELL", "STOP", "STOP10", "STOP_ATR"}

    for date in portfolio_df.index:
        # Apply all trades on or before this date
        while trade_idx < n_trades and td.iloc[trade_idx]["date"] <= date:
            tr = td.iloc[trade_idx]
            tkr = tr["ticker"]
            if tr["action"] == "BUY":
                positions[tkr] = {
                    "shares": float(tr["shares"]),
                    "cost":   float(tr["shares"]) * float(tr["price"]),
                }
            elif tr["action"] in sell_actions:
                positions.pop(tkr, None)
            trade_idx += 1

        # Snapshot per-sector cost
        sec_costs: dict[str, float] = {}
        for tkr, pos in positions.items():
            sec = sector_map.get(tkr) or "other"
            sec_costs[sec] = sec_costs.get(sec, 0.0) + pos["cost"]
        total = sum(sec_costs.values())
        rec: dict = {"date": date}
        if total > 0:
            for sec, c in sec_costs.items():
                rec[sec] = c / total * 100.0
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("date").fillna(0.0)
    # Order columns by mean weight (largest sectors first → top of legend)
    if not out.empty:
        order = out.mean().sort_values(ascending=False).index.tolist()
        out = out[order]
    return out


def _notable_observations(
    portfolio_df: pd.DataFrame,
    trades_df: pd.DataFrame | None,
    holdings: dict,
    scores: dict,
    spy_close: pd.Series,
    cutoff: float | None,
    days_uw_pct: float,
    top3_pct: float,
    n_sectors: int,
    sector_map: dict,
    feature_importance: list[dict],
) -> list[str]:
    """Generate auto-text observations for the Diagnostics tab. Each
    bullet is computed defensively — any failure (missing data, empty
    intersection) yields a silent skip, never an error message."""
    obs: list[str] = []
    pv = portfolio_df["portfolio_value"]

    # a. Alpha gap opening: first 3-consecutive-day stretch where
    #    60-day rolling alpha was negative.
    try:
        if not spy_close.empty:
            common = pv.index.intersection(spy_close.index)
            sys_60 = pv.loc[common].pct_change(60)
            spy_60 = spy_close.loc[common].pct_change(60)
            roll_alpha = sys_60 - spy_60
            neg = roll_alpha < 0
            run_sum = neg.rolling(3).sum()
            hits = run_sum[run_sum >= 3]
            if not hits.empty:
                first = hits.index[0]
                obs.append(
                    f"System began trailing SPY around "
                    f"{first.strftime('%B %Y')}"
                )
    except Exception:
        pass

    # b. Worst 90-day stretch
    # c. Best 90-day stretch (skip if max <= 0)
    try:
        if not spy_close.empty:
            common = pv.index.intersection(spy_close.index)
            sys_90 = pv.loc[common].pct_change(90)
            spy_90 = spy_close.loc[common].pct_change(90)
            r90 = (sys_90 - spy_90).dropna()
            if not r90.empty:
                worst_end = r90.idxmin()
                obs.append(
                    f"Worst 90-day stretch ended "
                    f"{worst_end.strftime('%B %Y')}: system trailed "
                    f"SPY by {abs(r90.min()) * 100:.1f}pp"
                )
                if r90.max() > 0:
                    best_end = r90.idxmax()
                    obs.append(
                        f"Best 90-day stretch ended "
                        f"{best_end.strftime('%B %Y')}: system led "
                        f"SPY by {r90.max() * 100:.1f}pp"
                    )
    except Exception:
        pass

    # d. Drawdown recovery
    try:
        dd = pv / pv.cummax() - 1
        worst_dd_date = dd.idxmin()
        worst_dd_pct = float(dd.min()) * 100
        pre_trough_max = float(pv.loc[:worst_dd_date].max())
        post = pv.loc[worst_dd_date:]
        recovered = post[post >= pre_trough_max]
        if not recovered.empty:
            recovery_date = recovered.index[0]
            days = (recovery_date - worst_dd_date).days
            obs.append(
                f"Worst drawdown {worst_dd_pct:.1f}% on "
                f"{worst_dd_date.strftime('%Y-%m-%d')}, recovered "
                f"{days} days later"
            )
        else:
            obs.append(
                f"Worst drawdown {worst_dd_pct:.1f}% on "
                f"{worst_dd_date.strftime('%Y-%m-%d')}, has not yet "
                f"recovered"
            )
    except Exception:
        pass

    # e. Underwater interpretation
    try:
        if days_uw_pct > 75:
            obs.append(
                f"System spent {days_uw_pct:.1f}% of the period below "
                f"its prior peak — significantly more than typical "
                f"buy-and-hold"
            )
        elif days_uw_pct >= 50:
            obs.append(
                f"System spent {days_uw_pct:.1f}% of the period below "
                f"its prior peak — moderately frequent recovery cycles"
            )
        elif days_uw_pct >= 25:
            obs.append(
                f"System spent {days_uw_pct:.1f}% of the period below "
                f"its prior peak — typical for an actively managed "
                f"strategy"
            )
        else:
            obs.append(
                f"System spent {days_uw_pct:.1f}% of the period at or "
                f"near all-time highs"
            )
    except Exception:
        pass

    # f. Concentration interpretation
    try:
        if top3_pct > 50:
            obs.append(
                f"High conviction: top 3 holdings represent "
                f"{top3_pct:.1f}% of portfolio"
            )
        elif top3_pct >= 25:
            obs.append(
                f"Moderate concentration: top 3 holdings represent "
                f"{top3_pct:.1f}% of portfolio"
            )
        else:
            obs.append(
                f"Diversified: top 3 holdings represent "
                f"{top3_pct:.1f}% of portfolio"
            )
    except Exception:
        pass

    # g. Score cutoff position
    try:
        if cutoff is not None and scores:
            all_scores = [s["composite"] for s in scores.values()
                          if isinstance(s, dict)
                          and s.get("composite") is not None]
            if all_scores:
                pctile = (sum(1 for s in all_scores if s <= cutoff)
                          / len(all_scores) * 100)
                top_pct = 100 - pctile
                if pctile >= 90:
                    obs.append(
                        f"System picks only from the top "
                        f"{top_pct:.0f}% of scored stocks"
                    )
                elif pctile >= 70:
                    obs.append(
                        f"System picks from the top "
                        f"{top_pct:.0f}% of scored stocks"
                    )
                else:
                    obs.append(
                        f"System picks more broadly — cutoff at "
                        f"{pctile:.0f} percentile"
                    )
    except Exception:
        pass

    # h. Top feature category
    try:
        def _categorize(name: str) -> str:
            n = name.lower()
            if "vix" in n or "vol" in n:
                return "volatility"
            if any(p in n for p in ("sma_", "ema_", "roc_", "mom_",
                                    "momentum_", "rs_", "rsi_", "macd_")):
                return "momentum/technical"
            if any(p in n for p in ("pe_", "pb_", "eps_", "_growth",
                                    "debt_", "profit_")):
                return "fundamental"
            if any(p in n for p in ("macro_", "nfci_", "sahm_", "yield_")):
                return "macro"
            return "other"

        if feature_importance:
            top5 = sorted(feature_importance,
                          key=lambda f: f["importance"], reverse=True)[:5]
            cats: dict[str, int] = {}
            for f in top5:
                c = _categorize(f["name"])
                cats[c] = cats.get(c, 0) + 1
            if cats:
                dom_cat = max(cats, key=lambda k: cats[k])
                dom_count = cats[dom_cat]
                obs.append(
                    f"Top features dominated by {dom_cat}: "
                    f"{dom_count} of top 5 features"
                )
    except Exception:
        pass

    # i. Sector spread + dominant sector if any > 40%
    try:
        if holdings and sector_map:
            obs.append(f"Holdings span {n_sectors} of 11 GICS sectors")
            costs = {tkr: float(h["shares"]) * float(h["entry_price"])
                     for tkr, h in holdings.items()}
            total_cost = sum(costs.values())
            if total_cost > 0:
                sector_costs: dict[str, float] = {}
                for tkr, c in costs.items():
                    sec = sector_map.get(tkr)
                    if sec:
                        sector_costs[sec] = sector_costs.get(sec, 0.0) + c
                if sector_costs:
                    top_sec, top_amt = max(sector_costs.items(),
                                           key=lambda kv: kv[1])
                    pct = top_amt / total_cost * 100
                    if pct > 40:
                        obs.append(
                            f"Sector concentration: {pct:.1f}% in "
                            f"{top_sec}"
                        )
    except Exception:
        pass

    return obs


def _exec_summary_risk_behavior(label: str, config: BacktestConfig,
                                result: dict) -> str:
    meta = result.get("meta") or {}
    rm = meta.get("rolling_metrics") or {}
    portfolio_df = result.get("portfolio_df")
    if portfolio_df is None or portfolio_df.empty:
        return "*Risk & Behavior summary will appear when results are loaded.*"
    capture = rm.get("capture") or {}
    up = capture.get("up_capture")
    down = capture.get("down_capture")
    pv = portfolio_df["portfolio_value"]
    max_dd = abs(float((pv / pv.cummax() - 1).min())) * 100
    rec = rm.get("recovery") or {}
    rec_10 = rec.get("-0.1") or {}
    ttr = rec_10.get("time_to_recovery_ratio_avg")
    ttr_phrase = ""
    if ttr is not None:
        if ttr <= 0.5:
            ttr_phrase = " The strategy tends to recover faster than SPY from -10% drops."
        elif ttr <= 1.2:
            ttr_phrase = " Recovery times are roughly in line with SPY."
        else:
            ttr_phrase = " Recovery from -10% drops takes longer than SPY's."

    if up is not None and down is not None:
        if up >= 110 and down <= 80:
            character = "amplifies the upside while damping the downside"
        elif up >= 110 and down >= 80:
            character = "amplifies both sides of the market"
        elif up < 90 and down < 80:
            character = "moves less than SPY in both directions"
        else:
            character = "tracks SPY in mixed fashion across regimes"
        headline = (f"In up months the strategy {character}: capturing "
                    f"**{up:.0f}%** of SPY's gains and **{down:.0f}%** "
                    f"of its losses.")
    else:
        headline = ("Capture ratios are not available in this saved "
                    "result. The detailed diagnostics below still apply.")

    detail = (f"Maximum drawdown was **-{max_dd:.0f}%** along the way."
              f"{ttr_phrase}")
    if label == "default":
        caveat = _summary_default_caveat()
    else:
        caveat = (f"A -{max_dd:.0f}% drawdown is real and would be "
                  f"psychologically difficult for a real investor; "
                  f"concentration amplifies volatility. The capture "
                  f"figures above are based on monthly returns, which "
                  f"smooth daily movement and can hide intra-month "
                  f"drawdowns.")
    return f"{_summary_caveat_prefix(label, result)}{headline}\n\n{detail}\n\n*{caveat}*"


def tab_risk_behavior(label: str, config: BacktestConfig, result: dict) -> None:
    st.header("Risk & Behavior")
    st.caption(f"Showing config: **{label}** "
               f"(window {config.validate_start} → {config.validate_end})")

    portfolio_df = result.get("portfolio_df")
    if portfolio_df is None or portfolio_df.empty:
        st.warning("Empty backtest result.")
        return

    st.info(_exec_summary_risk_behavior(label, config, result))
    st.divider()

    meta = result.get("meta") or {}
    rm = meta.get("rolling_metrics") or {}
    capture = rm.get("capture") or {}
    rec = rm.get("recovery") or {}
    rolling_12mo = rm.get("rolling_12mo") or {}
    ad = rolling_12mo.get("alpha_distribution_stats") or {}

    # ===== Layer 1 — Quick inference =====
    pv = portfolio_df["portfolio_value"]
    start = pv.index[0].strftime("%Y-%m-%d")
    end   = (pv.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_close = cached_benchmark("SPY", start, end)
    cols = st.columns(4)
    up = capture.get("up_capture")
    down = capture.get("down_capture")
    rec_10 = rec.get("-0.1") or {}
    rec_avg_days = rec_10.get("time_to_recovery_avg_days")
    rec_ratio = rec_10.get("time_to_recovery_ratio_avg")
    cols[0].metric(
        "Up capture",
        f"{up:.0f}% of SPY gains" if up is not None else "—",
        help="When SPY goes up, this number says how much of the move "
             "the strategy captures (computed monthly).",
    )
    cols[1].metric(
        "Down capture",
        f"{down:.0f}% of SPY drops" if down is not None else "—",
        help="When SPY goes down, this number says how much of the drop "
             "the strategy participates in (computed monthly).",
    )
    if rec_avg_days is not None:
        rec_label = f"{rec_avg_days:.0f} days"
    elif rec_ratio is not None:
        rec_label = f"{rec_ratio:.2f}× SPY"
    else:
        rec_label = "—"
    cols[2].metric(
        "Recovery from -10% drawdown",
        rec_label,
        help="Average time from a -10% drop to recovering peak value. "
             "Smaller is better.",
    )
    if "count_positive" in ad and "count_total" in ad and ad["count_total"]:
        cols[3].metric(
            "Months with positive alpha",
            f"{ad['count_positive']} of {ad['count_total']} "
            f"({ad['count_positive']/ad['count_total']*100:.0f}%)",
            help="Of the rolling 12-month windows in validation, this "
                 "share had positive alpha vs SPY.",
        )
    else:
        cols[3].metric("Months with positive alpha", "—")

    # ===== Layer 2 — Drawdown chart =====
    st.divider()
    st.markdown("**Drawdowns: how far below peak the strategy and SPY "
                "have been over time.**")
    sys_dd_pct = (pv / pv.cummax() - 1.0) * 100.0
    spy_dd_pct = pd.Series(dtype=float)
    if not spy_close.empty:
        spy_dd_pct = ((spy_close / spy_close.cummax()) - 1.0) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sys_dd_pct.index, y=sys_dd_pct.values,
        mode="lines", name="Strategy",
        line=dict(color="#dc2626", width=2),
        fill="tozeroy", fillcolor="rgba(220, 38, 38, 0.18)",
    ))
    if not spy_dd_pct.empty:
        fig.add_trace(go.Scatter(
            x=spy_dd_pct.index, y=spy_dd_pct.values,
            mode="lines", name="SPY",
            line=dict(color="#f59e0b", width=2, dash="dot"),
        ))
    fig.update_layout(
        yaxis_title="Drawdown (%)", xaxis_title="",
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===== Layer 3 — Detailed diagnostics (expander) =====
    st.divider()
    with st.expander("Detailed diagnostics — rolling metrics, capture "
                      "ratios, observations", expanded=False):
        _risk_behavior_detailed_diagnostics(label, config, result)


def _risk_behavior_detailed_diagnostics(
    label: str, config: BacktestConfig, result: dict) -> None:
    """Original Diagnostics tab content, now living inside the
    Risk & Behavior tab's Layer 3 expander. Keeps the cumulative-
    alpha + signal-real + snapshot + sector-allocation + notable-
    observations sections intact."""

    portfolio_df = result["portfolio_df"]
    trades_df    = result.get("trades_df")
    holdings     = result.get("holdings", {}) or {}
    scores       = result.get("scores", {}) or {}

    if portfolio_df is None or portfolio_df.empty:
        st.warning("Empty backtest result.")
        return

    pv = portfolio_df["portfolio_value"]
    start = pv.index[0].strftime("%Y-%m-%d")
    end = (pv.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_close = cached_benchmark("SPY", start, end)

    # ----- Section 1: How are we doing vs SPY? -----
    st.divider()
    st.subheader("How are we doing vs SPY?")

    sys_dd = (pv / pv.cummax() - 1) * 100
    if not spy_close.empty:
        sys_norm = pv / pv.iloc[0]
        spy_norm = spy_close / spy_close.iloc[0]
        common = sys_norm.index.intersection(spy_norm.index)
        alpha_pp = (sys_norm.loc[common] - spy_norm.loc[common]) * 100
        spy_dd = ((spy_close.loc[common] /
                   spy_close.loc[common].cummax()) - 1) * 100
    else:
        alpha_pp = pd.Series(dtype=float)
        spy_dd = pd.Series(dtype=float)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure()
        if not alpha_pp.empty:
            fig.add_trace(go.Scatter(
                x=alpha_pp.index, y=alpha_pp.values,
                mode="lines", name="Cumulative alpha",
                line=dict(color="#2563eb", width=2),
            ))
            fig.add_hline(y=0, line=dict(dash="dash",
                                         color="#475569", width=1))
        fig.update_layout(
            title="Cumulative alpha vs SPY (pp)",
            xaxis_title="", yaxis_title="Alpha (pp)",
            height=360, margin=dict(l=10, r=10, t=50, b=10),
            hovermode="x unified", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sys_dd.index, y=sys_dd.values,
            mode="lines", name="System",
            line=dict(color="#dc2626", width=2),
            fill="tozeroy", fillcolor="rgba(220, 38, 38, 0.15)",
        ))
        if not spy_dd.empty:
            fig.add_trace(go.Scatter(
                x=spy_dd.index, y=spy_dd.values,
                mode="lines", name="SPY",
                line=dict(color="#f59e0b", width=2),
                fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.15)",
            ))
        fig.update_layout(
            title="Drawdown overlay",
            xaxis_title="", yaxis_title="Drawdown (%)",
            height=360, margin=dict(l=10, r=10, t=50, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Metrics strip
    sys_total_pct = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
    if not spy_close.empty:
        spy_total_pct = (spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100
        n_days = max((pv.index[-1] - pv.index[0]).days, 1)
        years = n_days / 365.25
        sys_ann = (pv.iloc[-1] / pv.iloc[0]) ** (1 / years) - 1
        spy_ann = (spy_close.iloc[-1] / spy_close.iloc[0]) ** (1 / years) - 1
        alpha_ann_pp = (sys_ann - spy_ann) * 100
    else:
        spy_total_pct = float("nan")
        alpha_ann_pp = float("nan")

    sys_max_dd = float(sys_dd.min()) if not sys_dd.empty else float("nan")
    spy_max_dd = float(spy_dd.min()) if not spy_dd.empty else float("nan")
    days_uw_pct = (pv < pv.cummax()).mean() * 100

    m = st.columns([1, 1, 1, 1])
    m[0].metric(
        "Total return", f"{sys_total_pct:+.1f}%",
        delta=(f"SPY {spy_total_pct:+.1f}%"
               if not pd.isna(spy_total_pct) else None),
        delta_color="off",
    )
    m[1].metric(
        "Annualized alpha vs SPY",
        f"{alpha_ann_pp:+.1f}pp" if not pd.isna(alpha_ann_pp) else "—",
    )
    m[2].metric(
        "Max drawdown", f"{sys_max_dd:.1f}%",
        delta=(f"SPY {spy_max_dd:.1f}%"
               if not pd.isna(spy_max_dd) else None),
        delta_color="off",
    )
    m[3].metric("Days underwater", f"{days_uw_pct:.1f}%")
    st.caption(
        "Annualized alpha is β-adjusted (Jensen's alpha) — not the "
        "same as cumulative excess return (system total minus SPY total)."
    )

    # ----- Section 2: Is the signal real? -----
    st.divider()
    st.subheader("Is the signal real?")

    score_date = pv.index[-1].strftime("%Y-%m-%d")
    st.caption(
        f"Score histogram: as of latest snapshot date **{score_date}**. "
        "Per-date historical scores aren't yet captured by the backtest "
        "(only the most-recent rebalance is saved); date picker deferred."
    )

    # Hoisted out of the c1 block so the Notable observations section
    # below can reuse cutoff (lowest score among held stocks).
    cutoff: float | None = None
    held_vals: list[float] = []
    skip_vals: list[float] = []
    if scores:
        held_set = set(holdings.keys())
        held_vals = [s["composite"] for tkr, s in scores.items()
                     if tkr in held_set
                     and isinstance(s, dict)
                     and s.get("composite") is not None]
        skip_vals = [s["composite"] for tkr, s in scores.items()
                     if tkr not in held_set
                     and isinstance(s, dict)
                     and s.get("composite") is not None]
        cutoff = min(held_vals) if held_vals else None

    c1, c2 = st.columns([1, 1])
    with c1:
        if scores:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=skip_vals, name="Not held",
                opacity=0.55, marker=dict(color="#94a3b8"),
                xbins=dict(size=0.05),
            ))
            fig.add_trace(go.Histogram(
                x=held_vals, name="Held",
                opacity=0.75, marker=dict(color="#2563eb"),
                xbins=dict(size=0.05),
            ))
            if cutoff is not None:
                fig.add_vline(
                    x=cutoff,
                    line=dict(dash="dash", color="#dc2626", width=1.5),
                    annotation_text=f"cutoff={cutoff:.3f}",
                    annotation_position="top right",
                )
            fig.update_layout(
                title="Composite score: held vs skipped",
                xaxis_title="Composite score", yaxis_title="Count",
                height=360, margin=dict(l=10, r=10, t=50, b=10),
                barmode="overlay",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No composite scores available for this config.")

    with c2:
        feats = load_feature_importance()
        if feats:
            top20 = sorted(feats, key=lambda f: f["importance"],
                           reverse=True)[:20]
            top20.reverse()  # plotly horizontal bars render bottom-up
            fig = go.Figure(go.Bar(
                x=[f["importance"] for f in top20],
                y=[f["name"] for f in top20],
                orientation="h",
                marker=dict(color="#2563eb"),
            ))
            fig.update_layout(
                title="Top 20 features by XGBoost importance",
                xaxis_title="Importance", yaxis_title="",
                height=360, margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not found. Run "
                    "`python snapshot_for_cloud.py` to regenerate.")

    # ----- Section 3: Snapshot summary -----
    st.divider()
    st.subheader("Snapshot summary")

    n_holdings = len(holdings)
    sector_map = load_sector_map()
    sectors_held = {sector_map.get(tkr) for tkr in holdings.keys()
                    if sector_map.get(tkr)}
    n_sectors = len(sectors_held)

    if trades_df is not None and not trades_df.empty:
        last_rebal = pd.to_datetime(trades_df["date"]).max()
        last_rebal_str = last_rebal.strftime("%Y-%m-%d")
    else:
        last_rebal_str = "—"

    if holdings:
        costs = {tkr: float(h["shares"]) * float(h["entry_price"])
                 for tkr, h in holdings.items()}
        total_cost = sum(costs.values())
        if total_cost > 0:
            top3_pct = (sum(sorted(costs.values(), reverse=True)[:3])
                        / total_cost * 100)
        else:
            top3_pct = 0.0
    else:
        top3_pct = 0.0

    dd_series = (pv / pv.cummax() - 1) * 100
    worst_dd_pct = float(dd_series.min())
    worst_dd_date = dd_series.idxmin().strftime("%Y-%m-%d")

    s1, s2 = st.columns([1, 1])
    with s1:
        st.markdown(f"**Holdings:** {n_holdings} stocks across "
                    f"{n_sectors} sectors")
        st.markdown(f"**Last rebalance:** {last_rebal_str}")
    with s2:
        st.markdown(f"**Top 3 concentration:** {top3_pct:.1f}% of portfolio "
                    f"(by entry-cost weight)")
        st.markdown(f"**Worst drawdown:** {worst_dd_pct:.1f}% on "
                    f"{worst_dd_date}")

    # ----- Section 3.5: Sector allocation over time -----
    st.divider()
    st.subheader("Sector allocation over time")
    st.caption("Portfolio composition by sector across the backtest period "
               "(by entry-cost weight; positions step when opened/closed).")
    sec_df = _reconstruct_sector_weights(portfolio_df, trades_df, sector_map)
    if sec_df.empty:
        st.info("No trades recorded — sector allocation chart unavailable.")
    else:
        fig = go.Figure()
        for col in sec_df.columns:
            fig.add_trace(go.Scatter(
                x=sec_df.index, y=sec_df[col],
                mode="lines", name=col,
                hovertemplate="%{y:.1f}%<extra>" + col + "</extra>",
            ))
        fig.update_layout(
            xaxis_title="", yaxis_title="% of portfolio",
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----- Section 4: Notable observations -----
    st.divider()
    st.subheader("Notable observations")
    st.markdown("#### Strategy summary")
    st.markdown(
        "This is a long-only equity strategy that scores ~486 large-cap "
        "US stocks on a composite of fundamental, technical, "
        "model-predicted, and alternative-data signals. The strategy "
        "holds 5 positions concentrated in highest-scoring names and "
        "rebalances every ~17 days. Trailing ATR-based stops protect "
        "against drawdowns, with a relatively tight stop multiplier "
        "(1.46) that prioritizes capital preservation in adverse moves. "
        "Earnings blackout windows prevent premature exits."
    )
    st.markdown(
        "Validation testing in 2024-2026 showed strong results: the "
        "strategy returned 315.7% over 2.3 years (vs SPY's 56.2%), "
        "with annualized return of 84.9% vs SPY's 21.2%. The "
        "configuration is fundamentals-heavy in scoring and "
        "aggressively concentrated, performing well in trending bull "
        "markets where its scoring criteria align with market "
        "leadership. The strategy's beta varies meaningfully across "
        "market conditions (0.42 to 2.4 across rolling windows), "
        "indicating it amplifies market exposure when conditions "
        "favor its selections. Maximum drawdown was -22.3%."
    )
    st.markdown(
        "The strategy was discovered through a regime-dependent search "
        "space, but TPE positioned the regime threshold (0.525) such "
        "that defensive regime activation does not improve performance "
        "on the validation period. The strategy effectively runs in "
        "offensive mode throughout. See the Robustness tab for the "
        "perturbation evidence that lower thresholds (which would "
        "activate defensive) reduce alpha by ~35pp annualized."
    )
    st.markdown(
        "The macro overlay (50%/75%/100% sizing tiers) is structurally "
        "active but does not produce material changes in this "
        "validation period — perturbing macro_threshold_low across "
        "[0.10, 0.45] produces nearly identical alpha. The overlay is "
        "conservatively implemented but not currently earning its "
        "complexity."
    )
    st.markdown(
        "ATR-based stop placement is currently bounded by atr_floor_pct "
        "(0.05) and atr_cap_pct (0.15) of entry price. For typical "
        "large-cap ATR values, the floor clamp dominates: V3 Track 2 "
        "perturbation showed identical trade counts and alpha across "
        "atr_multiplier values from 1.0 to 2.25. The multiplier is "
        "structurally present as a tunable but does not materially "
        "affect stop-loss placement in the current parameter regime. "
        "Adjusting the floor/cap clamps would have more impact than "
        "adjusting the multiplier."
    )
    st.divider()
    obs = _notable_observations(
        portfolio_df, trades_df, holdings, scores, spy_close,
        cutoff, days_uw_pct, top3_pct, n_sectors, sector_map,
        load_feature_importance(),
    )
    if obs:
        st.markdown("\n".join(f"- {o}" for o in obs))
    else:
        st.caption("(No observations available — empty backtest result.)")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # Auth gate (cloud mode only). Local mode skips the gate so dev
    # iteration stays friction-free.
    if CLOUD_MODE:
        from dashboard_auth import gate
        gate()       # st.stop()s if not authenticated; renders sidebar logout

    st.title("📈 Paper Trader Dashboard")
    if CLOUD_MODE:
        st.caption("☁️ Cloud read-only build — data sourced from snapshot bucket.")

    # Header strip with model/macro provenance for orientation
    meta = load_model_meta()
    macro_df = load_macro_df()
    line_bits = []
    if meta:
        line_bits.append(f"Model: trained {meta.get('trained_at', '?')[:10]}, "
                         f"eval_cutoff = {meta.get('eval_cutoff', '?')}, "
                         f"{meta.get('n_tickers', '?')} tickers")
    if not macro_df.empty:
        line_bits.append(f"Macro cache: "
                         f"{macro_df.index.min().date()} → "
                         f"{macro_df.index.max().date()}, "
                         f"{len(macro_df.columns)} components")
    if line_bits:
        st.caption(" • ".join(line_bits))

    label, config, study_name, _trial_n = sidebar_config_picker()
    result = get_result_for_config(label, config)

    tabs = st.tabs(
        ["Performance", "Current Holdings", "Trade History",
         "Market Context", "Risk & Behavior", "Reliability",
         "Tuning History", "Glossary & Help"]
    )
    with tabs[0]:
        tab_performance(label, config, result)
    with tabs[1]:
        tab_holdings(label, config, result)
    with tabs[2]:
        tab_trade_history(label, config, result)
    with tabs[3]:
        tab_market_context(label, config, result)
    with tabs[4]:
        tab_risk_behavior(label, config, result)
    with tabs[5]:
        tab_reliability(label, config, result)
    with tabs[6]:
        tab_tuning_history(label, config, result, study_name)
    with tabs[7]:
        tab_glossary()


main()
