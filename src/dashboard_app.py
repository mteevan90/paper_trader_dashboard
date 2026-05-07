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
            rows.append({
                "ticker":     ticker,
                "buy_date":   b["date"],
                "buy_price":  float(b["price"]),
                "sell_date":  s["date"],
                "sell_price": float(s["price"]),
                "return_pct": ret_pct,
                "hold_days":  hold_days,
                "reason":     reason,
            })
        # Open positions (buys without matching exit)
        for i in range(len(exits), len(buys)):
            b = buys.iloc[i]
            rows.append({
                "ticker":     ticker,
                "buy_date":   b["date"],
                "buy_price":  float(b["price"]),
                "sell_date":  end_ts,
                "sell_price": float("nan"),
                "return_pct": float("nan"),
                "hold_days":  (end_ts - b["date"]).days,
                "reason":     "Open",
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("buy_date").reset_index(drop=True)


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

def tab_overview(label: str, config: BacktestConfig, result: dict) -> None:
    st.header("Overview")
    st.caption(f"Showing config: **{label}** "
               f"(window {config.validate_start} -> {config.validate_end})")

    # First-time-visitor context. Always rendered — st.info is mid-weight,
    # visible on first visit but unobtrusive once the reader is oriented.
    st.info(
        "This is a paper-trading research dashboard. The system runs on a "
        "491-ticker universe of large-cap US equities with a macro overlay "
        "and ML-driven composite scoring. All performance shown is paper "
        "trading, not real money. Use the sidebar to switch between "
        "**Default config** (the unoptimized baseline) and **Best trial** "
        "(the locked Optuna-tuned config from training)."
    )

    portfolio_df = result["portfolio_df"]
    trades_df    = result["trades_df"]

    if portfolio_df.empty:
        st.warning("Empty backtest result.")
        return

    # --- Equity curve: Strategy / SPY / QQQ, normalized to 100 ---
    start = portfolio_df.index[0].strftime("%Y-%m-%d")
    end = (portfolio_df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    spy_close = cached_benchmark("SPY", start, end)
    qqq_close = cached_benchmark("QQQ", start, end)

    pv = portfolio_df["portfolio_value"]
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
        title="Portfolio vs SPY vs QQQ (normalized to 100)",
        yaxis_title="Indexed value", xaxis_title="",
        height=420, margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Key stats (Streamlit metric cards) ---
    fees = float(portfolio_df["total_fees"].iloc[-1])
    metrics = _sleeve_metrics(pv, trades_df, fees)
    cols = st.columns(5)
    cols[0].metric("Total Return", metrics["Total Return"])
    cols[1].metric("Sharpe", metrics["Sharpe Ratio"])
    cols[2].metric("Max DD", metrics["Max Drawdown"])
    cols[3].metric("Trades", metrics["Total Trades"])
    cols[4].metric("Win Rate", metrics["Win Rate"])

    # Alpha / beta vs SPY
    if not spy_close.empty:
        port_ret = pv.pct_change().dropna()
        spy_ret = spy_close.pct_change().dropna()
        common = port_ret.index.intersection(spy_ret.index)
        pr = port_ret.loc[common]
        sr = spy_ret.loc[common]
        if len(pr) > 1 and sr.std() > 0:
            cov = np.cov(pr, sr)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
            alpha_ann = (pr.mean() - beta * sr.mean()) * 252
            # Arithmetic alpha: simple annualized return spread (matches
            # project documentation's headline +63.7pp for Trial #325).
            arith_alpha_ann = (pr.mean() - sr.mean()) * 252
            corr = pr.corr(sr)
            cols2 = st.columns(4)
            cols2[0].metric(
                "Alpha (arithmetic, ann.)",
                f"{arith_alpha_ann*100:+.2f}pp",
                help="Strategy annualized return minus SPY annualized "
                     "return. Does not adjust for beta. This is the "
                     "headline +63.7pp number cited in project "
                     "documentation."
            )
            cols2[1].metric(
                "Alpha (CAPM, ann.)",
                f"{alpha_ann*100:+.2f}pp",
                help="Jensen's alpha: excess annualized return after "
                     "removing the beta-amplified SPY contribution. "
                     "Lower than arithmetic alpha when beta > 1 because "
                     "some outperformance is attributed to amplified "
                     "market exposure."
            )
            cols2[2].metric("Beta vs SPY", f"{beta:.2f}")
            cols2[3].metric("Correlation vs SPY", f"{corr:.2f}")

    # --- Current macro state (today) ---
    st.subheader("Current macro state")
    macro_df = load_macro_df()
    if macro_df.empty:
        st.warning("No macro cache found.")
    else:
        today_score = compute_macro_score(macro_df, pd.Timestamp.today())
        if today_score > config.macro_threshold_high:
            tier = f"100% sizing (score > {config.macro_threshold_high:.3f})"
            color_emoji = "🟢"
        elif today_score >= config.macro_threshold_low:
            tier = f"75% sizing (score in [{config.macro_threshold_low:.3f}, {config.macro_threshold_high:.3f}])"
            color_emoji = "🟡"
        else:
            tier = f"50% sizing (score < {config.macro_threshold_low:.3f})"
            color_emoji = "🔴"
        cols3 = st.columns([1, 3])
        cols3[0].metric("Today's macro composite", f"{today_score:.3f}")
        cols3[1].markdown(f"**Tier under selected config**: {color_emoji} {tier}")

    # --- Real portfolio placeholder ---
    st.subheader("Real portfolio")
    st.info("Real portfolio tracking pending segment 4. This panel will "
            "show actual broker positions and PnL once that segment lands.")


def tab_optuna(study_name: str | None) -> None:
    st.header("Optuna explorer")
    if study_name is None:
        st.info("Pick a study from the sidebar (set Source to "
                "'Best trial of selected study' or 'Custom trial number') "
                "to explore trial-level results.")
        return

    df = load_study_trials_df(study_name)
    completes = df[df["state"] == "COMPLETE"]
    pruned    = df[df["state"] == "PRUNED"]
    failed    = df[df["state"] == "FAIL"]

    cols = st.columns(4)
    cols[0].metric("Trials", len(df))
    cols[1].metric("Complete", len(completes))
    cols[2].metric("Pruned", len(pruned))
    cols[3].metric("Failed", len(failed))

    if completes.empty:
        st.warning("No completed trials in this study.")
        return

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


def tab_macro(config: BacktestConfig) -> None:
    st.header("Macro state")
    macro_df = load_macro_df()
    if macro_df.empty:
        st.warning("No macro cache. Run the macro pipeline first.")
        return

    st.caption(f"Sizing thresholds (selected config): "
               f"low = {config.macro_threshold_low:.3f}, "
               f"high = {config.macro_threshold_high:.3f}")

    # --- Composite over time with sizing-tier bands ---
    scores = macro_score_series(macro_df)
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=config.macro_threshold_low,
                  fillcolor="#fca5a5", opacity=0.18, line_width=0,
                  annotation_text="50% sizing", annotation_position="top left")
    fig.add_hrect(y0=config.macro_threshold_low, y1=config.macro_threshold_high,
                  fillcolor="#fde68a", opacity=0.18, line_width=0,
                  annotation_text="75% sizing", annotation_position="top left")
    fig.add_hrect(y0=config.macro_threshold_high, y1=1,
                  fillcolor="#86efac", opacity=0.18, line_width=0,
                  annotation_text="100% sizing", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=scores.index, y=scores,
        mode="lines", name="Macro composite",
        line=dict(color="#1e293b", width=1.5),
    ))
    fig.update_layout(
        title="Macro composite over time (with sizing-tier bands)",
        xaxis_title="", yaxis_title="Composite score",
        height=420, margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Distribution stats ---
    st.subheader("Composite distribution (full history)")
    pcts = scores.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    cols = st.columns(7)
    cols[0].metric("Mean", f"{scores.mean():.3f}")
    cols[1].metric("Std", f"{scores.std():.3f}")
    cols[2].metric("p10", f"{pcts.loc[0.10]:.3f}")
    cols[3].metric("p25", f"{pcts.loc[0.25]:.3f}")
    cols[4].metric("p50", f"{pcts.loc[0.50]:.3f}")
    cols[5].metric("p75", f"{pcts.loc[0.75]:.3f}")
    cols[6].metric("p90", f"{pcts.loc[0.90]:.3f}")

    # --- Component signals ---
    st.subheader("Underlying components (raw FRED series)")
    raw_cols = list(macro_df.columns)
    cols_per_row = 2
    for row_start in range(0, len(raw_cols), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(raw_cols[row_start: row_start + cols_per_row]):
            series = macro_df[col].dropna()
            if series.empty:
                continue
            with cols[j]:
                fig = px.line(series, title=f"{col}  ({FRED_SERIES.get(col, '?')})",
                              height=240,
                              color_discrete_sequence=["#2563eb"])
                fig.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False, yaxis_title="", xaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True)


def tab_positions(config: BacktestConfig, result: dict) -> None:
    st.header("Positions")
    holdings = result.get("holdings", {}) or {}
    portfolio_df = result.get("portfolio_df")
    scores = result.get("scores", {}) or {}
    trades_df = result.get("trades_df")

    if not holdings:
        st.info("No open positions at end of backtest window.")
        return

    end_date = portfolio_df.index[-1] if portfolio_df is not None and \
        not portfolio_df.empty else None

    # Days-held lookup from trades
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
        shares = h["shares"]
        entry  = h["entry_price"]
        stop   = h.get("stop_price", entry * 0.85)
        cur    = entry  # fallback if portfolio_df doesn't carry per-ticker prices
        # Use last-known price from trades_df if any post-entry trade exists
        # (close approximation; precise current price would re-pull from the
        # price cache, which is overkill for this read-only view).
        cost   = shares * entry
        value  = shares * cur
        pl_d   = value - cost
        pl_p   = (cur / entry - 1) * 100
        stop_gap = (cur - stop) / cur if cur else 0
        warn = stop_gap < 0.03
        days_held = (end_date - entry_dates[tkr]).days \
            if (end_date is not None and tkr in entry_dates) else None
        score = scores.get(tkr, {}).get("composite", float("nan"))
        rows.append({
            "ticker":    tkr,
            "shares":    shares,
            "entry":     entry,
            "stop":      stop,
            "stop_gap_%": round(stop_gap * 100, 2),
            "stop_warn": "⚠ near stop" if warn else "",
            "days_held": days_held,
            "composite": round(float(score), 3) if pd.notna(score) else None,
        })
    df = pd.DataFrame(rows).sort_values("composite", ascending=False)
    _render_df_with_ticker_links(df, use_container_width=True, hide_index=True)

    st.caption("Note: `entry`/`stop` are from the saved backtest's "
               "final_holdings. Real-time current price is intentionally "
               "not fetched here — this is a read-only summary of the "
               "backtest snapshot. Live broker prices arrive in segment 4.")


def tab_trades(result: dict) -> None:
    st.header("Trades log")
    trades_df = result.get("trades_df")
    portfolio_df = result.get("portfolio_df")
    if trades_df is None or trades_df.empty:
        st.info("No trades in this backtest.")
        return

    end_date = portfolio_df.index[-1] if portfolio_df is not None and \
        not portfolio_df.empty else pd.Timestamp.today()

    rt = round_trip_trades(trades_df, end_date)
    if rt.empty:
        st.info("No round trips to display.")
        return

    # Filters
    tickers = ["All"] + sorted(rt["ticker"].unique().tolist())
    reasons = ["All"] + sorted(rt["reason"].unique().tolist())
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        ticker_pick = st.selectbox("Ticker", tickers, index=0)
    with c2:
        reason_pick = st.selectbox("Exit reason", reasons, index=0)
    with c3:
        only_winners = st.checkbox("Winners only", value=False)

    f = rt.copy()
    if ticker_pick != "All":
        f = f[f["ticker"] == ticker_pick]
    if reason_pick != "All":
        f = f[f["reason"] == reason_pick]
    if only_winners:
        f = f[f["return_pct"].fillna(-999) > 0]

    # Summary stats over the filtered set (closed trips only)
    closed = f[f["reason"] != "Open"]
    if not closed.empty:
        cols = st.columns(4)
        cols[0].metric("Closed", len(closed))
        cols[1].metric("Avg return / trade",
                       f"{closed['return_pct'].mean():+.2f}%")
        cols[2].metric("Avg hold days",
                       f"{closed['hold_days'].mean():.0f}")
        win_rate = (closed["return_pct"] > 0).mean() * 100
        cols[3].metric("Win rate", f"{win_rate:.1f}%")

    # Display
    show = f.copy()
    show["buy_date"] = show["buy_date"].dt.strftime("%Y-%m-%d")
    show["sell_date"] = show["sell_date"].dt.strftime("%Y-%m-%d")
    show["buy_price"] = show["buy_price"].round(2)
    show["sell_price"] = show["sell_price"].round(2)
    show["return_pct"] = show["return_pct"].round(2)
    _render_df_with_ticker_links(show, use_container_width=True, hide_index=True)

    st.subheader("Top traded tickers")
    _render_df_with_ticker_links(_top_traded_stocks(trades_df),
                                 use_container_width=True, hide_index=True)


_ROBUSTNESS_THRESHOLD = 0.40   # rolling_12mo_objective bar for "still robust"
_ROBUSTNESS_DEAD_BAND = 0.04   # ±r12_obj range that counts as "dead axis"

# Friendly axis labels for the per-axis table + plot titles. Falls back
# to the raw name with underscores stripped if not in the map.
_ROBUSTNESS_AXIS_LABELS: dict[str, str] = {
    "atr_multiplier_offensive":           "ATR multiplier (offensive)",
    "macro_threshold_low":                "Macro threshold low",
    "position_count_offensive":           "Position count (offensive)",
    "rebalance_frequency_days_offensive": "Rebalance freq (days, offensive)",
    "regime_threshold":                   "Regime threshold",
    "weight_fundamental_offensive":       "Weight: fundamental (offensive)",
    "weight_model_offensive":             "Weight: model (offensive)",
    "weight_technical_offensive":         "Weight: technical (offensive)",
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


def tab_robustness(label: str, config: BacktestConfig, result: dict) -> None:
    st.header("Robustness")
    df = load_perturbation_summary()
    if df is None or df.empty:
        st.info(
            "No perturbation data found in this deployment. The "
            "Robustness tab surfaces V3 Track 2 single-axis "
            "perturbation results from "
            "`models/cache/dashboard_results/v3_track2_perturbation/"
            "summary_full.csv`. Generate via `python "
            "src/v3_track2_runner.py` (local) or wait for the next "
            "snapshot sync (cloud)."
        )
        return

    df = df.copy()
    df["axis"] = df["axis"].astype(str)

    # ---- Section A: headline + KPIs ----
    n_total = int(len(df))
    n_ref = int(df["is_reference"].sum())
    n_pert = n_total - n_ref
    thr = _ROBUSTNESS_THRESHOLD
    n_pert_robust = int(((df["rolling_12mo_objective"] >= thr)
                         & ~df["is_reference"]).sum())
    pct_robust = (n_pert_robust / n_pert * 100) if n_pert else 0.0
    n_axes = int(df["axis"].nunique())

    st.markdown(
        f"Trial **{label}** was tested across **{n_pert}** single-axis "
        f"perturbations spanning **{n_axes}** axes. "
        f"**{n_pert_robust} of {n_pert}** perturbations ({pct_robust:.0f}%) "
        f"still produced rolling_12mo_objective ≥ {thr:.2f} — i.e., "
        f"still substantially outperformed SPY."
    )

    # Per-axis ranges for KPIs
    by_axis = df.groupby("axis", sort=False)
    spreads = by_axis["rolling_12mo_objective"].agg(
        lambda s: float(s.max() - s.min())).sort_values()
    tightest_axis = spreads.index[0]
    most_robust_axis = spreads.index[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total perturbations", f"{n_pert}")
    c2.metric(f"Beating r12_obj ≥ {thr:.2f}",
              f"{n_pert_robust} ({pct_robust:.0f}%)")
    c3.metric("Tightest axis (smallest r12_obj range)",
              _ROBUSTNESS_AXIS_LABELS.get(tightest_axis, tightest_axis),
              f"Δ {spreads.iloc[0]:.3f}")
    c4.metric("Widest axis (largest r12_obj range)",
              _ROBUSTNESS_AXIS_LABELS.get(most_robust_axis, most_robust_axis),
              f"Δ {spreads.iloc[-1]:.3f}")

    # ---- Section B: 4×2 subplot grid ----
    st.divider()
    st.subheader("Per-axis surface plots")
    st.caption(
        "Each panel shows the 5 tested values for one axis. Blue line is "
        "rolling_12mo_objective (left axis); orange line is alpha_ann "
        "(right axis). The reference point (Trial #325's value) is marked "
        "with a star."
    )
    axes_sorted = list(_ROBUSTNESS_AXIS_LABELS.keys())
    # Keep only axes that actually appear in the data, preserve canonical order
    axes_sorted = [a for a in axes_sorted if a in by_axis.groups]
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
        # r12_obj (primary y)
        fig.add_trace(go.Scatter(
            x=sub["value"], y=sub["rolling_12mo_objective"],
            mode="lines+markers", name="r12_obj", legendgroup="r12",
            showlegend=(i == 0),
            line=dict(color="#2563eb", width=2),
            marker=dict(size=8, color="#2563eb"),
        ), row=row, col=col, secondary_y=False)
        # alpha_ann (secondary y)
        fig.add_trace(go.Scatter(
            x=sub["value"], y=sub["alpha_ann"],
            mode="lines+markers", name="alpha_ann", legendgroup="alpha",
            showlegend=(i == 0),
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=7, color="#f59e0b", symbol="diamond"),
        ), row=row, col=col, secondary_y=True)
        # Reference markers
        if not ref_sub.empty:
            fig.add_trace(go.Scatter(
                x=ref_sub["value"],
                y=ref_sub["rolling_12mo_objective"],
                mode="markers", name="reference (#325)",
                legendgroup="ref", showlegend=(i == 0),
                marker=dict(size=14, color="#16a34a", symbol="star"),
            ), row=row, col=col, secondary_y=False)
        # Threshold line for r12_obj
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
        fig.update_yaxes(title_text="r12_obj" if col == 1 else "",
                         row=row, col=col, secondary_y=False)
        fig.update_yaxes(title_text="alpha" if col == 2 else "",
                         row=row, col=col, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Section C: per-axis classification table ----
    st.divider()
    st.subheader("Per-axis interpretation")
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
            "Axis": _ROBUSTNESS_AXIS_LABELS.get(ax_name, ax_name),
            "Reference": f"{ref_val:.3f}" if ref_val == ref_val else "—",
            "Range tested": f"[{v_lo:.3f}, {v_hi:.3f}]",
            "r12_obj range": f"[{r_lo:.3f}, {r_hi:.3f}]",
            "Robust?": cls,
            "Note": note,
        })
    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    st.caption(
        "Classifications: **Dead axis** — r12_obj spread ≤ "
        f"{_ROBUSTNESS_DEAD_BAND} (insensitive). "
        "**Robust plateau** — 4-5/5 above threshold. "
        "**Peak with sensitivity** — reference at peak, others fall off. "
        "**Knife edge** — only 1-2/5 above threshold."
    )


def tab_user_guide() -> None:
    st.header("Paper Trader Dashboard User Guide")
    st.markdown(
        "A complete walkthrough of what this dashboard shows, what each "
        "metric means, and what NOT to conclude from any of it. Recommended "
        "reading for first-time users."
    )
    st.markdown(
        "**What's in the guide:**\n"
        "- What the system is and how it picks stocks\n"
        "- Navigation of the dashboard's tabs and sidebar\n"
        "- Definitions of metrics like Sharpe, alpha, drawdown\n"
        "- Important caveats — this is **NOT** investment advice"
    )

    guide_path = Path(__file__).parent / "paper_trader_user_guide.docx"
    if guide_path.exists():
        st.download_button(
            label="Download User Guide (Word doc)",
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


def tab_diagnostics(label: str, config: BacktestConfig, result: dict) -> None:
    st.header("Diagnostics")
    st.caption(f"Showing config: **{label}** "
               f"(window {config.validate_start} -> {config.validate_end})")

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
        ["Overview", "Diagnostics", "Optuna explorer", "Macro state",
         "Positions", "Trades log", "Robustness", "User Guide"]
    )
    with tabs[0]:
        tab_overview(label, config, result)
    with tabs[1]:
        tab_diagnostics(label, config, result)
    with tabs[2]:
        tab_optuna(study_name)
    with tabs[3]:
        tab_macro(config)
    with tabs[4]:
        tab_positions(config, result)
    with tabs[5]:
        tab_trades(result)
    with tabs[6]:
        tab_robustness(label, config, result)
    with tabs[7]:
        tab_user_guide()


main()
