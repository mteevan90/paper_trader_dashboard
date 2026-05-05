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


LOCKED_BEST_STUDY = "optuna_v1_20260504_103429"
LOCKED_BEST_TRIAL = 706


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
@st.cache_data(show_spinner=False)
def list_studies() -> list[str]:
    db_path = _db_path()
    if not os.path.exists(db_path):
        return []
    return optuna.get_all_study_names(storage=_db_url())


@st.cache_data(show_spinner="Loading study trials...")
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def load_macro_df() -> pd.DataFrame:
    macro_path = _macro_parquet_path()
    if not os.path.exists(macro_path):
        return pd.DataFrame()
    return pd.read_parquet(macro_path)


@st.cache_data(show_spinner=False)
def load_model_meta() -> dict:
    meta_path = _model_meta_path()
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner="Downloading benchmark...")
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


def trial_to_config(trial: optuna.trial.FrozenTrial) -> BacktestConfig:
    p = trial.params
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
    full_studies = sorted([s for s in studies if not s.startswith("smoke_")],
                          reverse=True)

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

    if mode == "Best trial of selected study":
        try:
            t = s.best_trial
        except ValueError:
            st.sidebar.error("Study has no completed trials yet.")
            return "default", BacktestConfig(), None, None
        trial_number = t.number
        label = f"best_{study_name}_{trial_number}"
        cfg = trial_to_config(t)
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
        cfg = trial_to_config(t)
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
        "trading, not real money. The strategy is research-grade and "
        "currently does not beat the S&P 500 — see the Overview metrics "
        "for current alpha. Use the sidebar to switch between **Default "
        "config** (the unoptimized baseline) and **Best trial** (the "
        "locked Optuna-tuned config from training)."
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
            corr = pr.corr(sr)
            cols2 = st.columns(3)
            cols2[0].metric("Alpha (ann.) vs SPY", f"{alpha_ann*100:+.2f}pp")
            cols2[1].metric("Beta vs SPY", f"{beta:.2f}")
            cols2[2].metric("Correlation vs SPY", f"{corr:.2f}")

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
    st.dataframe(df, use_container_width=True, hide_index=True)

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
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Top traded tickers")
    st.dataframe(_top_traded_stocks(trades_df), use_container_width=True,
                 hide_index=True)


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
        ["Overview", "Optuna explorer", "Macro state", "Positions", "Trades log",
         "User Guide"]
    )
    with tabs[0]:
        tab_overview(label, config, result)
    with tabs[1]:
        tab_optuna(study_name)
    with tabs[2]:
        tab_macro(config)
    with tabs[3]:
        tab_positions(config, result)
    with tabs[4]:
        tab_trades(result)
    with tabs[5]:
        tab_user_guide()


main()
