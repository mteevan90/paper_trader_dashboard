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

import html
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
def compute_best_known(
    study_name: str,
    param_cols: tuple[str, ...],
    cross_study: bool = False,
) -> dict:
    """Per-parameter best-known summary across the Optuna corpus.

    For each requested param column, returns the value with the
    highest kernel-smoothed mean trial score (`best_mean_x`/`best_mean_y`),
    the single best trial's value (`best_max_x`/`best_max_y`), the
    smoothed curve points for an optional overlay, and a flag for
    discrete-parameter handling.

    Naming: "best-known" = highest-mean across the corpus, NOT the
    optimizer's chosen value. The user's spec (Section 6.2) is firm:
    don't call this "best" or "global optimum" — it's a marginal
    estimate over what was sampled, which is itself centered near
    the chosen value.

    Sentinel filter (value > -1e3) applied per spec Section 6.1 — without
    it, REJECTED_ACTIVATION_PCT trials at -1e6 dominate every kernel.

    Discrete parameters (position_count, rebalance_frequency_days)
    use a groupby-mean instead of the kernel smoother — Optuna samples
    integer values, the kernel curve would look stair-stepped, and
    the right answer is "the integer with the best mean score."

    Continuous parameters use a Gaussian-kernel local-mean smoother
    (NOT scipy.stats.gaussian_kde — that estimates p(x), the sample
    density, which would peak where Optuna sampled most. We want
    E[y|x], the conditional expected score given the parameter value).
    Bandwidth via Silverman's rule of thumb; degenerate fallback
    when std is zero. Skips smoothing entirely if <20 sane trials —
    the curve isn't informative below that.

    Cross-study (cross_study=True) pools sane trials across all
    promoted studies before computing. Per spec Section 7.4 with two
    promoted studies (regime_dependent_v1, 15_position_study_v1),
    the simple-toggle approach is fine; if more studies appear later
    a per-study checkbox UI is the right escalation.
    """
    if cross_study:
        promoted = data_source.list_promoted_dashboard_result_labels()
        # promoted labels look like best_<study>_<n>; recover the study name
        studies = sorted({lbl[len("best_"):lbl.rfind("_")]
                           for lbl in promoted
                           if lbl.startswith("best_")
                           and lbl.rfind("_") > len("best_")})
    else:
        studies = [study_name]

    pieces = []
    for s in studies:
        try:
            d = load_study_trials_df(s)
        except Exception:
            continue
        if d is None or d.empty:
            continue
        # Sentinel filter — spec Section 6.1.
        d = d[d["state"] == "COMPLETE"]
        d = d[d["value"].astype(float) > -1e3]
        if not d.empty:
            pieces.append(d)
    if not pieces:
        return {}
    pooled = pd.concat(pieces, ignore_index=True)

    out: dict = {}
    for p in param_cols:
        if p not in pooled.columns:
            continue
        sub = pooled[[p, "value"]].dropna()
        if sub.empty:
            continue
        x = sub[p].astype(float).values
        y = sub["value"].astype(float).values
        n = len(x)
        # Best-max (single best trial)
        bmax_idx = int(np.argmax(y))
        best_max_x = float(x[bmax_idx])
        best_max_y = float(y[bmax_idx])
        # Discrete heuristic: integer-valued + low cardinality
        is_discrete = (
            np.allclose(x, np.round(x))
            and len(np.unique(x)) <= 16
        )
        smooth_xs = None
        smooth_ys = None
        bandwidth = float("nan")
        if is_discrete:
            grouped = sub.groupby(p)["value"].mean()
            best_v = grouped.idxmax()
            best_mean_x = float(best_v)
            best_mean_y = float(grouped.max())
            if n >= 20:
                smooth_xs = [float(v) for v in sorted(grouped.index)]
                smooth_ys = [float(grouped[v]) for v in smooth_xs]
            bandwidth = 0.5  # discrete coincidence = same integer bucket
        elif n < 20:
            # Too few trials for a meaningful smoother — fall back to
            # best-max as the only marker; skip best-mean entirely.
            best_mean_x = float("nan")
            best_mean_y = float("nan")
        else:
            xs = np.linspace(float(x.min()), float(x.max()), 200)
            h = 1.06 * float(x.std()) * (n ** (-0.2))
            if h <= 0:
                h = max((float(x.max()) - float(x.min())) / 20.0, 1e-6)
            ys = np.empty_like(xs)
            for j, xj in enumerate(xs):
                w = np.exp(-0.5 * ((x - xj) / h) ** 2)
                wsum = float(w.sum())
                ys[j] = float((w * y).sum() / wsum) if wsum > 0 else np.nan
            argmax = int(np.nanargmax(ys))
            best_mean_x = float(xs[argmax])
            best_mean_y = float(ys[argmax])
            smooth_xs = xs.tolist()
            smooth_ys = ys.tolist()
            bandwidth = float(h)
        out[p] = {
            "best_mean_x": best_mean_x,
            "best_mean_y": best_mean_y,
            "best_max_x":  best_max_x,
            "best_max_y":  best_max_y,
            "smooth_xs":   smooth_xs,
            "smooth_ys":   smooth_ys,
            "n":           int(n),
            "is_discrete": bool(is_discrete),
            "bandwidth":   bandwidth,
        }
    return out


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


def _load_saved_benchmark(label: str, ticker: str) -> "pd.Series | None":
    """Read the saved benchmark close series for this label, if present.

    `_save_one_backtest_result` writes `<label>/SPY_close.parquet` and
    `<label>/QQQ_close.parquet` so the dashboard doesn't have to call
    yfinance at view time. That matters in cloud mode where Yahoo soft-
    throttles Streamlit Cloud's shared IP — SPY in particular intermittently
    returns empty, breaking the Performance tab's alpha/beta/chart.

    Returns None if the file isn't present (older saves, or v3_track2_*
    aggregation labels). Callers fall back to cached_benchmark() in that
    case, preserving backward compatibility for legacy dirs."""
    if not label:
        return None
    p = data_source.path_to(
        f"models/cache/dashboard_results/{label}/{ticker}_close.parquet",
        quiet=True)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p)
        s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        return None


def benchmark_for_label(label: str, ticker: str,
                        start: str, end: str) -> pd.Series:
    """Saved-first / yfinance-fallback benchmark loader.

    Prefer the saved per-label parquet (cloud-safe, no network). Fall
    back to a live yfinance download if the saved file is missing —
    typical for older labels saved before benchmark-snapshotting landed."""
    saved = _load_saved_benchmark(label, ticker)
    if saved is not None and not saved.empty:
        return saved
    return cached_benchmark(ticker, start, end)


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
    """Reconstruct a BacktestConfig from a completed trial.

    Delegates to optuna_runner._trial_to_config so the dashboard sees
    the same architecture-aware reconstruction (legacy / regime-
    dependent / single-regime) and the same weight-triple normalization
    the search-space sampler applied. Without this, regime-dependent
    studies whose raw trial.params held a free-weight triple summing
    >1.0 (the V3-spec ranges allow this; the sampler clamps via
    _normalize_weight_triple) would crash BacktestConfig validation
    when the sidebar picker tried to load them.
    """
    from optuna_runner import _trial_to_config as _runner_trial_to_config
    return _runner_trial_to_config(trial, fixed_values=fixed_values)


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

def sidebar_asset_picker() -> str:
    """Render the asset-class radio at the very top of the sidebar.

    Returns "Stocks", "Crypto", or "Options". Persisted across reruns
    via the Streamlit widget key. Phase 1 ships with Stocks rendering
    the full equity dashboard bit-identically; selecting Crypto or
    Options shows a placeholder (Phase 2 — Chris — will land the
    crypto and options renderings).
    """
    return st.sidebar.radio(
        "Asset class",
        options=["Stocks", "Crypto", "Options"],
        index=0,
        help="Switch between asset class dashboards. "
             "Crypto and Options are in development.",
        key="asset_class_selector",
    )


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

    # Cross-study pooling toggle for the best-known-value markers
    # (Tuning History per-parameter scatters + Reliability per-panel
    # overlays). When unchecked, best-known is computed within the
    # currently-selected study only. When checked, pools sane trials
    # across all promoted studies. Per viz spec Section 7.4 with the
    # current 2-promoted-study set, a single global toggle is fine;
    # if a 3rd promoted study appears, this UI shape needs revisiting.
    st.sidebar.divider()
    cross_study_pool = st.sidebar.checkbox(
        "Compare best-known values across all studies",
        value=False,
        help="When enabled, points pool across all promoted studies. "
             "Useful for spotting parameter values that worked across "
             "different strategy designs, but cross-study comparisons "
             "can be misleading when the studies use different "
             "position counts or windows.",
    )
    st.session_state["best_known_cross_study"] = cross_study_pool

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
    spy_close = benchmark_for_label(label, "SPY", start, end)

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
    spy_close = benchmark_for_label(label, "SPY", start, end)
    qqq_close = benchmark_for_label(label, "QQQ", start, end)

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
    # 12-month rolling-alpha sparkline below the windows-positive KPI.
    # Surfaces the time-series shape of outperformance: for studies where
    # alpha hovers near zero (the 15-position study at 35% positive
    # windows), the area chart crossing below zero makes the under-
    # performance visceral — the reader sees how often the strategy
    # spent time underwater vs SPY. Reads from meta directly; no
    # recomputation. Column name in capm_windows is `alpha` (verified
    # against rolling_metrics.compute_rolling_alpha L127), NOT alpha_ann.
    capm_windows = (rolling_12mo.get("capm_windows") or [])
    if capm_windows:
        sw = pd.DataFrame(capm_windows)
        if "window_end" in sw.columns and "alpha" in sw.columns:
            sw = sw.copy()
            sw["window_end"] = pd.to_datetime(sw["window_end"])
            spark = go.Figure()
            spark.add_trace(go.Scatter(
                x=sw["window_end"],
                y=sw["alpha"],
                mode="lines",
                line=dict(color="#2563eb", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.18)",
                hovertemplate=(
                    "12mo ending %{x|%b %Y}<br>"
                    "alpha %{y:+.2%}<extra></extra>"),
            ))
            spark.add_hline(y=0, line_color="#94a3b8",
                            line_width=1, line_dash="dot")
            spark.update_layout(
                height=80, margin=dict(l=4, r=4, t=4, b=4),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
            )
            cols[3].plotly_chart(spark, use_container_width=True)
            cols[3].caption(
                "12-month rolling alpha vs SPY. Below the dotted line = "
                "trailing year underperformed.")

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

    # ===== Layer 2 — Score distribution (where the winner ranks) =====
    st.markdown("**Where this strategy ranks among all configurations tested.**")
    # Filter out failure-sentinel scores. Optuna records trial state as
    # COMPLETE whenever objective_fn returns any value (including the
    # _FAILURE_SENTINEL = -1e6 that the runner uses for unrecoverable
    # backtest errors). Any legit objective output sits in [-1, 1] for
    # our scoring; anything below -1e3 is a sentinel and would skew
    # mean/std/z-score calculations beyond meaning.
    sane = completes[completes["value"].astype(float) > -1e3]
    n_sentinel = len(completes) - len(sane)
    if sane.empty:
        st.warning("All completed trials are failure sentinels — score "
                   "distribution is not meaningful.")
        st.divider()
        return
    scores = sane["value"].astype(float).values
    n_completes = len(scores)
    score_mean = float(np.mean(scores))
    score_std  = float(np.std(scores, ddof=1)) if n_completes > 1 else 0.0
    win_score  = float(sane["value"].max())
    win_n      = int(sane.loc[sane["value"].idxmax(), "trial_number"])
    win_z      = ((win_score - score_mean) / score_std
                  if score_std > 0 else float("nan"))
    # Percentile rank of the winner (ascending — 99% means "beats 99% of trials")
    win_pct_rank = float((scores < win_score).sum()) / n_completes * 100.0
    top_pct = max(100.0 - win_pct_rank, 100.0 / n_completes)  # never claim "top 0%"

    cols = st.columns(4)
    cols[0].metric(
        "Total trials completed",
        f"{n_completes:,}",
        help=(f"{n_sentinel} additional trials returned the failure "
              f"sentinel — excluded from this distribution because they "
              f"would skew mean/std beyond meaning."
              if n_sentinel else
              "Trials that returned a real (non-sentinel) score."),
    )
    cols[1].metric("Mean trial score", f"{score_mean:.3f}")
    cols[2].metric("Std dev of trial scores", f"{score_std:.3f}")
    cols[3].metric(
        "Winner's z-score",
        f"{win_z:+.2f}σ" if not pd.isna(win_z) else "—",
        help=("Standard deviations above the mean. >2 = clear peak; "
              "1–2 = outperformed but not exceptional; <1 = no "
              "obvious winner."),
    )

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=50,
        marker=dict(color="#475569", line=dict(color="#1f2937", width=0.5)),
        name="Trials", showlegend=False,
    ))
    # ±2σ shading (lighter), ±1σ shading (darker), both behind histogram
    if score_std > 0:
        fig.add_vrect(
            x0=score_mean - 2*score_std, x1=score_mean + 2*score_std,
            fillcolor="#94a3b8", opacity=0.10, line_width=0, layer="below",
            annotation_text="±2σ", annotation_position="top left",
            annotation=dict(font=dict(size=10, color="#475569")),
        )
        fig.add_vrect(
            x0=score_mean - score_std, x1=score_mean + score_std,
            fillcolor="#94a3b8", opacity=0.18, line_width=0, layer="below",
            annotation_text="±1σ", annotation_position="top left",
            annotation=dict(font=dict(size=10, color="#475569")),
        )
        fig.add_vline(x=score_mean, line_dash="dot", line_color="#475569",
                      line_width=1)
    # Winner's score — distinct red line with annotation
    fig.add_vline(
        x=win_score, line_color="#dc2626", line_width=2.5,
        annotation_text=f"Winner: Trial #{win_n} (score {win_score:.3f})",
        annotation_position="top right",
        annotation=dict(font=dict(size=11, color="#dc2626")),
    )
    fig.update_layout(
        title=f"Trial score distribution — {n_completes:,} configurations tested",
        xaxis_title="Trial score (12-month outperformance)",
        yaxis_title="Number of configurations",
        height=380, margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Live-computed interpretation sentence
    if pd.isna(win_z):
        interp = ("Trial scores are degenerate (zero variance) — "
                  "interpretation is not meaningful.")
    elif win_z >= 2:
        interp = (f"The winning configuration sits in the **top "
                  f"{top_pct:.1f}%** of trials and is "
                  f"**{win_z:.1f}σ above the mean** — a clear peak.")
    elif win_z >= 1:
        interp = (f"The winning configuration sits in the **top "
                  f"{top_pct:.1f}%** of trials, between **1 and 2σ "
                  f"above the mean** — outperformed but not exceptional.")
    else:
        interp = (f"The winning configuration is only **{win_z:.2f}σ "
                  f"above the mean** (top {top_pct:.1f}% of trials) — "
                  f"TPE may not have found a clear peak in this "
                  f"search space.")
    st.markdown(interp)
    st.divider()

    # --- Trial number vs score with running best ---
    # Use the same sentinel-filtered frame as the histogram. Without
    # this, -1e6 sentinel scores from REJECTED_ACTIVATION_PCT and
    # REJECTED_INVALID_THRESHOLD_ORDERING trials drag the y-axis to
    # -1M and crush all real scores into a thin band at the top —
    # the bug Section 3.1 of the viz spec calls out for studies with
    # any rejection-gate activity (e.g. the 15-position study with
    # 199 sentinels out of 1000 trials).
    if n_sentinel > 0:
        st.caption(
            f"Excludes {n_sentinel} trials that returned the failure "
            f"sentinel (rejected by the activation gate or other "
            f"hard-reject paths). The running-best line is computed "
            f"from real trials only — a sentinel can't be a running best."
        )
    completes_sorted = sane.sort_values("trial_number")
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
    # Tunables list — the "shared" set Tuning History always renders.
    # Note these are the legacy field names which in regime-dependent
    # mode hold the DEFENSIVE half of the tunable pair; the offensive
    # variants are surfaced separately on the Reliability tab. For
    # legacy-architecture studies (Phase 0 etc.), these ARE the tunables.
    tunables = [c for c in ("weight_fundamental", "weight_technical",
                            "weight_model", "macro_threshold_low",
                            "macro_threshold_gap", "atr_multiplier",
                            "analyst_weight", "rebalance_frequency_days",
                            "position_count") if c in sane.columns]
    # Section 3.2 — best-known-value markers from the Optuna corpus.
    # cross_study honors the sidebar toggle; cache invalidates on the
    # study_name + tuple(tunables) + cross_study args (per Streamlit's
    # default argument hashing).
    cross_study = bool(st.session_state.get("best_known_cross_study", False))
    bk = compute_best_known(study_name, tuple(tunables), cross_study=cross_study)

    # Layer-1 summary line above the scatter row — quick "is the chosen
    # value also the best-known?" read across all 8 axes. Skipped in
    # cross-study mode because "chosen" is per-study (each pooled
    # study has its own chosen value, so the comparison is ambiguous).
    if not cross_study and bk and not sane.empty:
        # Recover the chosen-trial's params from the trial DataFrame
        # (each tunable column carries that trial's sampled value).
        best_row = sane.loc[sane["value"].idxmax()]
        chosen_params = {p: best_row[p] for p in tunables
                         if p in best_row.index and pd.notna(best_row[p])}
        materially_diff = []
        DIFF_REL_PCT = 0.10  # 10% relative spread for continuous axes
        for p in tunables:
            if p not in bk or p not in chosen_params:
                continue
            chosen_x = float(chosen_params[p])
            best_x   = bk[p]["best_mean_x"]
            if pd.isna(best_x):
                continue
            if bk[p]["is_discrete"]:
                if int(round(best_x)) != int(round(chosen_x)):
                    materially_diff.append(p)
            else:
                if abs(chosen_x) > 1e-9:
                    rel = abs(best_x - chosen_x) / abs(chosen_x)
                    if rel >= DIFF_REL_PCT:
                        materially_diff.append(p)
        if materially_diff:
            names = ", ".join(materially_diff)
            st.markdown(
                f"Of **{len(tunables)} tunables**, "
                f"**{len(materially_diff)}** have a best-known value "
                f"materially different from the chosen value "
                f"(>= {int(DIFF_REL_PCT*100)}% relative spread, or any "
                f"difference for discrete axes): **{names}** — "
                f"these may be under-tuned."
            )
        else:
            st.markdown(
                f"All {len(tunables)} tunables have their best-known "
                f"value at or near the chosen value."
            )
    elif cross_study:
        st.caption(
            "Cross-study mode: best-known markers pool sane trials "
            "across every promoted study. The 'chosen-value' summary "
            "line is omitted because each study has its own chosen "
            "value — the comparison is ambiguous."
        )

    # Section intro paragraph — explains the whole grid as a unit so
    # the family-audience reader has the four shape archetypes (trend /
    # flat / bell / under-tuned) before scanning the panels.
    st.markdown(
        "These charts show how each tunable parameter relates to "
        "strategy performance across all trials Optuna explored. "
        "The pattern in each chart tells a different story:\n\n"
        "- A clear upward or downward trend means the parameter has a "
        "strong effect; the optimizer chose a value at the right end "
        "of that trend.\n"
        "- A flat scatter means the parameter doesn't matter much "
        "within the range tested.\n"
        "- A bell-shaped pattern (high in the middle, low at the "
        "edges) is what a well-tuned parameter should look like — "
        "there's a sweet spot.\n"
        "- When the best-known marker (purple) is far from the chosen "
        "value, the optimizer may have under-tuned that parameter and "
        "a different value might score higher in expectation."
    )

    cols_per_row = 3
    for row_start in range(0, len(tunables), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, p in enumerate(tunables[row_start: row_start + cols_per_row]):
            with cols[j]:
                friendly = _TUNING_AXIS_LABELS.get(p, p)
                # Use sane (sentinel-filtered) — sentinel trials don't
                # represent meaningful parameter-score relationships and
                # would otherwise drag every panel's y-axis to -1M.
                fig = px.scatter(
                    sane, x=p, y="value", trendline=None,
                    title=friendly, height=260,
                    color_discrete_sequence=["#2563eb"],
                )
                fig.update_traces(marker=dict(size=4, opacity=0.6))

                # Smoothed-curve overlay + best-mean / best-max markers
                # from the Optuna corpus. Purple (#a855f7) chosen so it
                # doesn't clash with the existing #2563eb blue trial
                # points or the #16a34a green chosen-value marker on
                # the Reliability tab. Per spec Section 7.5: skip
                # best-max if it coincides with best-mean within
                # bandwidth h (avoids drawing two markers on top of
                # each other — looks like a rendering bug).
                bki = bk.get(p)
                if bki is not None:
                    sxs, sys_ = bki.get("smooth_xs"), bki.get("smooth_ys")
                    if sxs and sys_:
                        fig.add_trace(go.Scatter(
                            x=sxs, y=sys_,
                            mode="lines", name="Smoothed mean",
                            line=dict(color="#a855f7", width=1.5),
                            hovertemplate=(
                                "param %{x}<br>"
                                "smoothed mean score %{y:.4f}"
                                "<extra></extra>"),
                            showlegend=False,
                        ))
                    if not pd.isna(bki["best_mean_x"]):
                        fig.add_trace(go.Scatter(
                            x=[bki["best_mean_x"]],
                            y=[bki["best_mean_y"]],
                            mode="markers", name="best-mean",
                            marker=dict(size=11, color="#a855f7",
                                        symbol="circle"),
                            hovertemplate=(
                                f"<b>Best-mean across "
                                f"{bki['n']} trials</b><br>"
                                f"param %{{x:.4f}}<br>"
                                f"mean score %{{y:.4f}}"
                                f"<extra></extra>"),
                            showlegend=False,
                        ))
                    # Best-max as a hollow ring; skip if it coincides
                    # with best-mean within bandwidth (continuous) or
                    # at the same integer (discrete).
                    coincide = False
                    if not pd.isna(bki["best_mean_x"]):
                        h = bki.get("bandwidth", float("nan"))
                        if bki["is_discrete"]:
                            coincide = (
                                int(round(bki["best_max_x"]))
                                == int(round(bki["best_mean_x"])))
                        elif not pd.isna(h) and h > 0:
                            coincide = (abs(bki["best_max_x"]
                                            - bki["best_mean_x"]) < h)
                    if not coincide:
                        fig.add_trace(go.Scatter(
                            x=[bki["best_max_x"]],
                            y=[bki["best_max_y"]],
                            mode="markers", name="best-max",
                            marker=dict(
                                size=11, color="#a855f7",
                                symbol="circle-open",
                                line=dict(color="#a855f7", width=2)),
                            hovertemplate=(
                                "<b>Single best trial</b><br>"
                                "param %{x:.4f}<br>"
                                "score %{y:.4f}"
                                "<extra></extra>"),
                            showlegend=False,
                        ))

                fig.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_title=friendly,
                    yaxis_title="12-month outperformance score",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Caption: generic explanation of dots + markers, plus a
                # param-specific story-line when the data shape supports
                # one (otherwise the generic line stands alone — better
                # than a forced narrative).
                specific = _tuning_param_specific_caption(p, sane, bk.get(p))
                if specific:
                    st.caption(_TUNING_PARAM_CAPTION_GENERIC + " " + specific)
                else:
                    st.caption(_TUNING_PARAM_CAPTION_GENERIC)

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
                                window_label: str,
                                ticker_label: str | None = None) -> str:
    if rt is None or rt.empty:
        return "*No trades in the selected period.*"
    closed = rt[rt["reason"] != "Open"]
    n_trades = len(rt)
    filter_clause = f", filtered to [{ticker_label}]" if ticker_label else ""
    period_phrase = f"In the selected period ({window_label}){filter_clause}"
    if closed.empty:
        if ticker_label:
            return (f"{period_phrase}: the strategy opened {n_trades} "
                    f"positions involving these tickers, none yet closed.")
        return (f"{period_phrase} the strategy opened {n_trades} "
                f"positions, none yet closed.")
    n_closed = len(closed)
    win_rate = (closed["return_pct"] > 0).mean() * 100.0
    # Top winner / loser by ticker (aggregate $ P&L across all that ticker's pairs).
    # Split into positive- and negative-PnL groups so a single-ticker filter
    # with only winners (or only losers) shows "(none in selection)" rather
    # than the same ticker on both sides of the headline.
    closed_with_pnl = closed.dropna(subset=["pnl_dollars"])
    best_str = "(none in selection)"
    worst_str = "(none in selection)"
    if not closed_with_pnl.empty:
        by_ticker = closed_with_pnl.groupby("ticker")["pnl_dollars"].sum().sort_values()
        winners = by_ticker[by_ticker > 0]
        losers = by_ticker[by_ticker < 0]
        if not winners.empty:
            best_str = f"**{winners.index[-1]}** (+${float(winners.iloc[-1]):,.0f})"
        if not losers.empty:
            worst_str = f"**{losers.index[0]}** (${float(losers.iloc[0]):,.0f})"
    activity = "steady" if n_trades >= 30 else "modest"
    if ticker_label:
        headline = (f"{period_phrase}: the strategy made **{n_trades} "
                    f"trades** ({n_closed} closed) involving these tickers. "
                    f"Biggest winner: {best_str}. Biggest loser: {worst_str}.")
    else:
        headline = (f"{period_phrase} the strategy made **{n_trades} "
                    f"trades** ({n_closed} closed). "
                    f"Biggest winner: {best_str}. Biggest loser: {worst_str}.")
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

    trades_dates = pd.to_datetime(trades_df["date"])

    # ---- Ticker filter (above date filter) ----
    # Ordered by total trade count desc so most-active tickers appear first
    # in the dropdown.
    ticker_counts = trades_df["ticker"].value_counts()
    all_tickers = list(ticker_counts.index)
    selected_tickers = st.multiselect(
        "Filter by ticker (leave empty to show all)",
        options=all_tickers,
        default=[],
    )
    is_filtered = (
        bool(selected_tickers) and len(selected_tickers) < len(all_tickers)
    )

    # ---- Date filter ----
    # Bounds derive from trades_df["date"] so the picker grays out dates
    # outside the actual data extent (otherwise Streamlit defaults end to
    # today, which exceeds the validation window for promoted studies).
    full_min = trades_dates.min().to_pydatetime().date()
    full_max = trades_dates.max().to_pydatetime().date()
    date_pick = st.date_input(
        "Date range (filters every section below)",
        value=(full_min, full_max),
        min_value=full_min, max_value=full_max,
        format="YYYY-MM-DD",
    )
    st.caption(f"Trades available: {full_min} to {full_max}")
    if isinstance(date_pick, tuple) and len(date_pick) == 2:
        d_start, d_end = pd.Timestamp(date_pick[0]), pd.Timestamp(date_pick[1])
    elif isinstance(date_pick, tuple) and len(date_pick) == 1:
        d_start = d_end = pd.Timestamp(date_pick[0])
    else:
        d_start, d_end = pd.Timestamp(full_min), pd.Timestamp(full_max)

    # Streamlit's date_input shortcut buttons ("Past Week", "Past Month",
    # etc.) compute end-date from today's real-world date, ignoring
    # max_value. Clamp to the data's actual extent so shortcut clicks
    # don't trigger a red rejection error.
    sel_start, sel_end = d_start, d_end
    d_start = max(d_start, pd.Timestamp(full_min))
    d_end = min(d_end, pd.Timestamp(full_max))
    if (d_start, d_end) != (sel_start, sel_end):
        st.caption(
            f"_Showing trades from {d_start.date()} to {d_end.date()} "
            f"(clamped to available data range)._"
        )

    # Composition: date first, then ticker. All downstream sections read
    # from `rt`, which is the doubly-filtered DataFrame.
    rt = rt_full[(rt_full["buy_date"] >= d_start)
                 & (rt_full["buy_date"] <= d_end + pd.Timedelta(days=1))].copy()
    if is_filtered:
        rt = rt[rt["ticker"].isin(selected_tickers)]
    window_label = f"{d_start.date()} → {d_end.date()}"
    ticker_label = ", ".join(selected_tickers) if is_filtered else None

    st.info(_exec_summary_trade_history(
        label, config, result, rt, window_label, ticker_label))
    st.divider()

    if rt.empty:
        if is_filtered:
            st.warning(
                "No trades for selected ticker(s) in this date range. Try "
                "expanding the date range or removing tickers from the filter."
            )
        else:
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
        if is_filtered:
            td_filt = td_filt[td_filt["ticker"].isin(selected_tickers)]
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
            # Reframe the spikes — per Section 3.5 of the viz spec, the
            # bar chart looks like noise without context. The caption
            # tells readers the spikes ARE rebalances.
            rebal_n = (config.rebalance_frequency_days_offensive
                       or config.rebalance_frequency_days)
            st.caption(
                f"Spikes correspond to rebalance days, which fall every "
                f"~{rebal_n} trading days under the current config. "
                f"Between rebalances, only stop-loss exits register as "
                f"trades."
            )
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

# Tuning History uses the legacy field names — these hold the defensive
# half of the tunable pair under regime-dependent architecture, but ARE
# the (sole) tunables under legacy architecture. Friendly names omit the
# "(defensive)" qualifier so legacy-study readers aren't confused.
_TUNING_AXIS_LABELS: dict[str, str] = {
    "weight_fundamental":       "Fundamentals weight",
    "weight_technical":         "Technical analysis weight",
    "weight_model":             "ML model weight",
    "macro_threshold_low":      "Market-stress sizing threshold",
    "macro_threshold_gap":      "Market-stress band width",
    "atr_multiplier":           "Stop-loss tightness",
    "analyst_weight":           "Analyst recommendation weight",
    "rebalance_frequency_days": "Rebalance frequency (days)",
    "position_count":           "Number of holdings",
}

# Generic caption shown under EVERY tuning-history scatter — explains
# what the dots and purple markers mean. Stays stable per chart so the
# reader builds the mental model once and re-applies it.
_TUNING_PARAM_CAPTION_GENERIC = (
    "Each dot is one Optuna trial — its parameter value (x) and the "
    "12-month outperformance score it achieved (y). The purple markers "
    "show the best-known value across all trials: filled circle = "
    "best-mean (the parameter value with the highest average score "
    "across nearby trials), ring = best-max (the single best-scoring "
    "trial)."
)

# Param-specific caption templates for the 4 axes the audience will most
# want a one-line story about. Picked when the data shape supports the
# story; gated by _tuning_param_specific_caption() below so a chart that
# doesn't actually fit the template stays generic-only.
_TUNING_PARAM_CAPTION_SPECIFIC = {
    "position_count": (
        "The strategy mostly performs at one position count; values "
        "further away tend to drop sharply, suggesting concentration "
        "is doing real work."
    ),
    "rebalance_frequency_days": (
        "Faster rebalancing (lower X) shows higher variance in scores; "
        "slower rebalancing tends to cluster more tightly. The chosen "
        "value sits in the middle range."
    ),
    "atr_multiplier": (
        "Score is relatively flat across the tested range, suggesting "
        "this parameter is robust to small adjustments."
    ),
    "macro_threshold_low": (
        "The score peak shifts based on where this threshold falls — "
        "too low and the defensive regime never fires; too high and "
        "it fires too often. The optimizer found a value in the "
        "productive band."
    ),
}


def _tuning_param_specific_caption(param: str, sane_df: pd.DataFrame,
                                   bki: dict | None) -> str | None:
    """Return a per-param story caption only when the chart's data shape
    actually supports it. The four templates in _TUNING_PARAM_CAPTION_SPECIFIC
    each describe a particular shape (concentration peak / variance funnel /
    flat / shifting peak); picking one when the chart shows a different
    pattern would mislead the reader, so we gate on simple shape checks."""
    tmpl = _TUNING_PARAM_CAPTION_SPECIFIC.get(param)
    if tmpl is None or sane_df.empty or bki is None:
        return None
    sm_xs = bki.get("smooth_xs") or []
    sm_ys = bki.get("smooth_ys") or []
    if not sm_xs or not sm_ys:
        # Discrete axis — fall back on raw spread of mean-by-x as a proxy.
        try:
            grouped = sane_df.groupby(param)["value"].mean()
            sm_y_spread = float(grouped.max() - grouped.min())
        except Exception:
            return None
    else:
        sm_y_spread = float(max(sm_ys) - min(sm_ys))
    # "Flat" template only fires when the smoothed range is genuinely small.
    if param == "atr_multiplier":
        return tmpl if sm_y_spread < 0.05 else None
    # "Concentration peak" / "shifting peak" templates need a real spread —
    # otherwise the chart looks flat and the story is wrong.
    if param in ("position_count", "macro_threshold_low"):
        return tmpl if sm_y_spread >= 0.05 else None
    # "Faster=more variance" depends on the spread of trial scores, not
    # of the smoothed mean. Approximate by std of trial scores at the
    # low half of x vs the high half.
    if param == "rebalance_frequency_days":
        try:
            x = sane_df[param].astype(float)
            v = sane_df["value"].astype(float)
            lo_mask = x <= x.median()
            std_lo = float(v[lo_mask].std()) if lo_mask.sum() > 1 else 0.0
            std_hi = float(v[~lo_mask].std()) if (~lo_mask).sum() > 1 else 0.0
            return tmpl if std_lo > std_hi * 1.15 else None
        except Exception:
            return None
    return None

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

    Returns three lists of pre-formatted strings, each "<friendly name>
    (<annotation>)":
      - most_stable / most_sensitive: annotation is the 12-month score
        swing across the 5 perturbation values, "(±X.XXX)".
      - one_sided: annotation is the parameter-value room in the
        unconstrained direction, "(room: ±X.X <below|above> chosen)".

    Bucket rules (unchanged):
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
        # Score swing across the axis's 5 perturbation values. abs() guards
        # against the (impossible-but-cheap) case max < min, plus collapses
        # any -0.0 from float arithmetic to +0.0 for clean display.
        r12 = sub["rolling_12mo_objective"]
        swing = abs(float(r12.max() - r12.min()))
        swing_str = f"{friendly} (±{swing:.3f})"
        if cls == "Dead axis":
            most_stable.append(swing_str)
        elif cls in ("Knife edge", "Peak with sensitivity"):
            most_sensitive.append(swing_str)
        ref_sub = sub[sub["is_reference"]]
        if not ref_sub.empty:
            ref_v = float(ref_sub["value"].iloc[0])
            v_lo = float(sub["value"].min())
            v_hi = float(sub["value"].max())
            at_min = abs(ref_v - v_lo) < 1e-9
            at_max = abs(ref_v - v_hi) < 1e-9
            n_above = int((sub["rolling_12mo_objective"]
                           >= _ROBUSTNESS_THRESHOLD).sum())
            if (at_min or at_max) and n_above >= 3:
                if at_min:
                    room = v_hi - ref_v
                    direction = "above"
                else:
                    room = -(ref_v - v_lo)
                    direction = "below"
                one_sided.append(
                    f"{friendly} (room: {room:+g} {direction} chosen)"
                )
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


def _render_reliability_kpi_card(label: str, items: list[str],
                                 help_text: str) -> None:
    """Bordered markdown card for a Reliability Layer 1 KPI.

    Replaces st.metric, which truncates multi-word axis names at headline
    font. Each item renders on its own line at body-text size; the help
    icon (ⓘ) carries the original tooltip via the HTML title attribute.
    """
    if items:
        body = "<br>".join(html.escape(item) for item in items)
    else:
        body = "—"
    title_attr = html.escape(help_text, quote=True)
    st.markdown(
        f"<div style='padding:14px 16px;border-radius:8px;"
        f"border:1px solid #e5e7eb;background:#f9fafb;"
        f"min-height:140px;height:100%'>"
        f"<div style='font-weight:600;color:#1f2937;font-size:0.95em;"
        f"margin-bottom:8px'>"
        f"{html.escape(label)} "
        f"<abbr title=\"{title_attr}\" "
        f"style='cursor:help;color:#6b7280;text-decoration:none;"
        f"border-bottom:none'>&#9432;</abbr>"
        f"</div>"
        f"<div style='color:#374151;font-size:0.92em;line-height:1.55'>"
        f"{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


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
    st.caption(
        "Numbers in parentheses show how much the 12-month outperformance "
        "score moved across each setting's tested range — small numbers "
        "mean the setting is robust to small changes; large numbers mean "
        "it's sensitive."
    )
    cols = st.columns(3)
    with cols[0]:
        _render_reliability_kpi_card(
            "Most stable settings (least sensitive)",
            most_stable,
            "Settings whose 5 tested values produce nearly identical "
            "results — tweaking them within the tested range has no "
            "observable effect on outperformance.",
        )
    with cols[1]:
        _render_reliability_kpi_card(
            "Most sensitive settings",
            most_sensitive,
            "Settings where moving the value drops outperformance "
            "noticeably — the chosen peak does not generalize across "
            "the whole tested range.",
        )
    with cols[2]:
        _render_reliability_kpi_card(
            "Settings with one-sided room to move",
            one_sided,
            "Settings where the chosen value sits at the edge of the "
            "tested range AND the score holds up across most of the "
            "range — could likely be moved further in one direction "
            "without losing alpha.",
        )

    # ===== Layer 2 — Per-axis surface grid =====
    st.divider()
    # Section 3.3.1 — reframe the panel caption. The previous text
    # invited the reader to ask "is the chosen value the peak?", which
    # is the wrong question (it almost always is, for grid-centering +
    # OAT-structural reasons documented in the viz spec Section 2).
    # The new caption frames robustness via the SPY-beat band instead.
    st.markdown(
        "**For each setting, we tested 4 alternative values around our "
        "choice.** The shaded green band is the threshold for 'still "
        "substantially beats SPY' (12-month outperformance score "
        f"≥ {_ROBUSTNESS_THRESHOLD:.2f}). Settings whose line stays "
        "inside the band across all values are robust to that choice; "
        "settings whose line drops outside the band are sensitive."
    )
    by_axis = df.groupby("axis", sort=False)
    thr = _ROBUSTNESS_THRESHOLD
    axes_sorted = [a for a in _ROBUSTNESS_AXIS_LABELS.keys()
                   if a in by_axis.groups]
    # Best-known-value markers from the Optuna corpus — Section 3.3.3 +
    # 3.2 of the viz spec. Plotted at TRUE x-position even when between
    # two perturbation grid points (Option 2 in the brainstorm). The
    # under-tuning signal: if the Optuna best is between two perturbation
    # values, the perturbation grid didn't sample where trials suggest
    # the real best lies.
    cross_study = bool(st.session_state.get("best_known_cross_study", False))
    rel_study_name = ""
    if isinstance(label, str) and label.startswith("best_"):
        # Recover the underlying study name from the dashboard label.
        # Format: best_<study_name>_<trial_n> — strip both ends.
        rel_study_name = label[len("best_"):]
        idx = rel_study_name.rfind("_")
        if idx > 0:
            rel_study_name = rel_study_name[:idx]
    bk_panels = compute_best_known(rel_study_name, tuple(axes_sorted),
                                    cross_study=cross_study) if rel_study_name else {}
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
        # Section 3.3.2 — replace the dashed primary-axis line with a
        # shaded band (Option b in the viz spec). The band reads as
        # "this is the safe zone" and is the visual anchor for the
        # refrazmed question. y1=2.0 gives plenty of headroom for any
        # rolling_12mo_objective value to land below it; visible y-range
        # auto-clips to the actual data extent.
        fig.add_hrect(
            y0=thr, y1=2.0,
            fillcolor="#16a34a", opacity=0.10, line_width=0,
            layer="below",
            row=row, col=col, secondary_y=False,
        )
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
            # Section 3.3.3 — demote the chosen-value marker. Previously
            # a size-14 green STAR which dominated the panel; reduced
            # to size-10 filled CIRCLE so it reads as information, not
            # a trophy. The viz spec correctly notes the chosen value is
            # structurally biased to look like a peak under OAT centering.
            fig.add_trace(go.Scatter(
                x=ref_sub["value"],
                y=ref_sub["rolling_12mo_objective"],
                mode="markers", name="chosen value",
                legendgroup="ref", showlegend=(i == 0),
                marker=dict(size=10, color="#16a34a", symbol="circle"),
            ), row=row, col=col, secondary_y=False)

        # Section 3.2 — best-known-value markers from the Optuna corpus,
        # plotted at TRUE x-position on the primary axis. y-coordinate
        # is the kernel-smoothed mean score (best-mean) or the actual
        # trial score (best-max). Purple #a855f7 — distinct from the
        # green chosen-value marker and the blue/orange data lines.
        bki = bk_panels.get(ax_name)
        if bki is not None:
            if not pd.isna(bki["best_mean_x"]):
                fig.add_trace(go.Scatter(
                    x=[bki["best_mean_x"]],
                    y=[bki["best_mean_y"]],
                    mode="markers", name="best-known (mean)",
                    legendgroup="bestmean", showlegend=(i == 0),
                    marker=dict(size=11, color="#a855f7", symbol="circle"),
                    hovertemplate=(
                        f"<b>Best-mean across {bki['n']} trials</b><br>"
                        f"param %{{x:.4f}}<br>"
                        f"smoothed mean score %{{y:.4f}}"
                        f"<extra></extra>"),
                ), row=row, col=col, secondary_y=False)
            # Best-max — skip if it coincides with best-mean within
            # bandwidth (avoids drawing two markers on top of each other,
            # per Section 7.5). Bandwidth comes from compute_best_known
            # so the test reuses the same Silverman value used to build
            # the smoother.
            coincide = False
            if not pd.isna(bki["best_mean_x"]):
                h = bki.get("bandwidth", float("nan"))
                if bki["is_discrete"]:
                    coincide = (int(round(bki["best_max_x"]))
                                == int(round(bki["best_mean_x"])))
                elif not pd.isna(h) and h > 0:
                    coincide = abs(bki["best_max_x"]
                                   - bki["best_mean_x"]) < h
            if not coincide:
                fig.add_trace(go.Scatter(
                    x=[bki["best_max_x"]],
                    y=[bki["best_max_y"]],
                    mode="markers", name="best-known (single)",
                    legendgroup="bestmax", showlegend=(i == 0),
                    marker=dict(size=11, color="#a855f7",
                                symbol="circle-open",
                                line=dict(color="#a855f7", width=2)),
                    hovertemplate=(
                        "<b>Single best trial</b><br>"
                        "param %{x:.4f}<br>"
                        "score %{y:.4f}"
                        "<extra></extra>"),
                ), row=row, col=col, secondary_y=False)
    # Legend sits in its own band ABOVE the row-1 subplot titles. The
    # earlier (t=60, y=1.02, xanchor=right) layout placed legend items
    # in the same horizontal band as the top-row panel titles, so
    # "Market-stress sizing threshold" (panel 2) collided with the
    # legend chips. Bumping top margin to 130 reserves space for one
    # horizontal legend row plus the title row beneath it; centering
    # the legend (xanchor=center, x=0.5) makes it span the full width
    # rather than clustering on the right above panel 2.
    fig.update_layout(
        height=1100, margin=dict(l=10, r=10, t=130, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="center", x=0.5),
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
    spy_close = benchmark_for_label(label, "SPY", start, end)
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
    spy_close = benchmark_for_label(label, "SPY", start, end)

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
# Phase 4.5 — Contract-conformant studies (dashboard_contract_v1).
#
# Renders studies that publish artifacts under
# `models/studies/<name>/contract_v1/`. Universal tabs read directly from
# the parquet/json artifacts; no live-fallback path. Legacy Optuna v1
# studies remain on the legacy sidebar branch unchanged.
# ---------------------------------------------------------------------------

CONTRACT_V1_DIR = Path(MODELS_DIR) / "studies"


def list_contract_v1_studies() -> list[str]:
    """Return contract-v1 study names found under models/studies/.

    A study qualifies as contract-conformant when EITHER:
      - `<study>/contract_v1/meta.json` exists (single-variant), OR
      - `<study>/variant_meta.json` exists (multi-variant, per the
        artifact_metadata + variant_meta.json schema in
        `docs/architecture/dashboard_contract_v1.md`)
    """
    if not CONTRACT_V1_DIR.exists():
        return []
    out = []
    for child in sorted(CONTRACT_V1_DIR.iterdir()):
        if not child.is_dir():
            continue
        single_variant = (child / "contract_v1" / "meta.json").exists()
        multi_variant = (child / "variant_meta.json").exists()
        if single_variant or multi_variant:
            out.append(child.name)
    return out


@st.cache_data(show_spinner=False)
def load_variant_meta(study_name: str) -> dict | None:
    """Return the study-level variant_meta.json for a multi-variant study,
    or None for single-variant studies."""
    p = CONTRACT_V1_DIR / study_name / "variant_meta.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def list_contract_v1_variants(study_name: str) -> list[dict]:
    """Return the variants[] list from variant_meta.json, or empty list for
    single-variant studies. Each entry has at least {name, subdir, role}."""
    vm = load_variant_meta(study_name)
    if vm is None:
        return []
    return vm.get("variants", []) or []


def _split_study_ref(study_ref: str) -> tuple[str, str | None]:
    """Split a study reference into (study_name, variant_name).

    Composite refs use the form "<study>/<variant>" for multi-variant
    studies; bare strings refer to single-variant studies.
    """
    parts = study_ref.split("/")
    if len(parts) == 2:
        return parts[0], parts[1]
    return study_ref, None


def _contract_dir(study_name: str) -> Path:
    """Resolve a study reference to its contract_v1 directory.

    Single-variant: `models/studies/<study>/contract_v1/`
    Multi-variant (composite ref "<study>/<variant>"):
      `models/studies/<study>/<variant>/contract_v1/`
    """
    study, variant = _split_study_ref(study_name)
    if variant is not None:
        return CONTRACT_V1_DIR / study / variant / "contract_v1"
    return CONTRACT_V1_DIR / study / "contract_v1"


@st.cache_data(show_spinner=False)
def load_contract_meta(study_name: str) -> dict:
    p = _contract_dir(study_name) / "meta.json"
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_contract_concentration(study_name: str) -> dict | None:
    p = _contract_dir(study_name) / "concentration_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_contract_tuning_summary(study_name: str) -> dict | None:
    """Per-model tuning summary (optional v1 artifact). None if absent."""
    p = _contract_dir(study_name) / "tuning_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _default_model_index(study_name: str, available: list[str]) -> int:
    """Return the index of the preferred default model within `available`.

    Preference order, per meta.json:
      1. The model with role="primary"
      2. If no model has role="primary", the first entry in meta.json.models[]
      3. If meta.json declares no models, or none of the preferred names
         appear in `available`, return 0 (first in the displayed list)

    Defensive: if multiple models declare role="primary" (data error),
    picks the first alphabetically and emits a Python warning visible in
    the Streamlit server log (not in the page body).
    """
    if not available:
        return 0
    try:
        meta = load_contract_meta(study_name)
    except Exception:
        return 0

    declared = meta.get("models") or []
    if not declared:
        return 0

    primaries = [m.get("name") for m in declared
                 if isinstance(m, dict) and m.get("role") == "primary"
                 and m.get("name")]
    if len(primaries) > 1:
        import warnings
        warnings.warn(
            f"meta.json for study '{study_name}' declares multiple models "
            f"with role='primary' ({primaries}); defaulting to first "
            "alphabetically.", stacklevel=2,
        )
        primaries.sort()

    if primaries:
        preferred = primaries[0]
    else:
        first = declared[0]
        preferred = first.get("name") if isinstance(first, dict) else None

    if preferred and preferred in available:
        return available.index(preferred)
    return 0


@st.cache_data(show_spinner=False)
def load_contract_parquet(study_name: str, name: str) -> pd.DataFrame:
    p = _contract_dir(study_name) / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _study_format(s: str) -> str:
    """Sidebar-display label for a study. Multi-variant studies get the
    '🔀 + (N variants)' suffix per the Gate 1 sidebar disambiguation rule."""
    variants = list_contract_v1_variants(s)
    if variants:
        # display_name comes from variant_meta.json for multi-variant studies
        vm = load_variant_meta(s) or {}
        display = vm.get("display_name", s)
        return f"{display} 🔀 + ({len(variants)} variants)"
    # Single-variant: read display from contract_v1/meta.json
    try:
        return load_contract_meta(s).get("display_name", s)
    except Exception:
        return s


def sidebar_contract_picker() -> str | None:
    """Render the contract-conformant study picker. Returns a study reference:
    either a bare `study_name` (single-variant) or composite `study/variant`
    (multi-variant). Variant defaults to role:'control' per Gate 1 spec.
    """
    studies = list_contract_v1_studies()
    if not studies:
        st.sidebar.warning(
            "No contract-conformant studies found under "
            "models/studies/<name>/contract_v1/ or `variant_meta.json`."
        )
        return None
    study = st.sidebar.selectbox(
        "Study", studies, index=0, key="contract_study_selector",
        format_func=_study_format,
    )

    # Variant selector for multi-variant studies — placed right below the
    # study selector. Applies globally to the 7 contract-conformant tabs.
    variants = list_contract_v1_variants(study)
    if not variants:
        return study

    variant_names = [v["name"] for v in variants]
    # Default to role:"control"; fall back to first variant
    default_idx = 0
    for i, v in enumerate(variants):
        if v.get("role") == "control":
            default_idx = i
            break

    def _variant_label(name: str) -> str:
        for v in variants:
            if v["name"] == name:
                role = v.get("role", "")
                return f"{name} ({role})" if role else name
        return name

    selected_variant = st.sidebar.selectbox(
        "Variant",
        variant_names,
        index=default_idx,
        key=f"contract_variant_selector_{study}",
        format_func=_variant_label,
    )
    return f"{study}/{selected_variant}"


def tab_contract_overview(study_name: str) -> None:
    meta = load_contract_meta(study_name)
    st.subheader(meta.get("display_name", study_name))
    st.caption(meta.get("description", ""))

    # === Date-range header ===
    cols = st.columns(4)
    win = meta.get("windows", {})
    cols[0].metric("Train window", f"{win.get('train_start', '?')} →\n"
                                    f"{win.get('train_end', '?')}")
    cols[1].metric("Test window", f"{win.get('test_start', '?')} →\n"
                                   f"{win.get('test_end', '?')}")
    # Schema field stays oos_start/oos_end; UI surfaces "Reserved validation".
    cols[2].metric("Reserved validation window",
                   f"{win.get('oos_start', '?')} →\n"
                   f"{win.get('oos_end', '?')}")
    cols[3].metric(
        "Promoted",
        "Yes" if meta.get("promoted") else "No",
    )

    # === NAV chart vs benchmarks (moved here from the former Performance tab) ===
    port = load_contract_parquet(study_name, "portfolio.parquet")
    bench = load_contract_parquet(study_name, "benchmarks.parquet")
    if not port.empty:
        port["date"] = pd.to_datetime(port["date"])
        if not bench.empty:
            bench["date"] = pd.to_datetime(bench["date"])

        fig = go.Figure()
        for model in port["model"].unique():
            m = port[port["model"] == model]
            fig.add_trace(go.Scatter(
                x=m["date"], y=m["nav"], mode="lines", name=model,
                line=dict(width=2.5),
            ))
        if not bench.empty:
            for b in bench["benchmark"].unique():
                bb = bench[bench["benchmark"] == b]
                fig.add_trace(go.Scatter(
                    x=bb["date"], y=bb["nav"], mode="lines", name=b,
                    line=dict(width=1.2, dash="dash"),
                    opacity=0.65,
                ))

        oos_start = win.get("oos_start")
        if oos_start:
            # Plotly 6.7 + pandas 3.0: add_vline with annotation_text on a
            # datetime axis requires ms-since-epoch (str / Timestamp /
            # datetime all raise TypeError because annotation-positioning
            # adds an int offset). Schema field stays `oos_start`;
            # human-facing label is "Reserved validation period".
            oos_ms = int(pd.Timestamp(oos_start).value // 10**6)
            fig.add_vline(x=oos_ms,
                          line=dict(color="red", dash="dot", width=1),
                          annotation_text="Reserved validation period →",
                          annotation_position="top right")

        fig.update_layout(
            title="Strategy NAV vs Benchmarks",
            yaxis_title="NAV (starts at 1.0)",
            xaxis_title="Date",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Headline metrics ===
    st.markdown("### Headline metrics")
    sm = meta.get("summary_metrics", {})
    SLICE_LABELS = {
        "test": "TEST slice",
        "oos": "RESERVED VALIDATION slice",
    }
    for slice_name in ("test", "oos"):
        slice_data = sm.get(slice_name, {})
        if not slice_data:
            continue
        st.markdown(f"**{SLICE_LABELS.get(slice_name, slice_name.upper())}**")
        rows = []
        for model, m in slice_data.items():
            rows.append({
                "model": model,
                "CAGR": f"{m.get('cagr', 0) * 100:.1f}%",
                "SPY CAGR": f"{m.get('spy_cagr', 0) * 100:.1f}%",
                "Excess vs SPY": f"{m.get('excess_cagr', 0) * 100:+.1f}pp",
                "Max DD": f"{m.get('max_drawdown', 0) * 100:.1f}%",
                "SPY Max DD": f"{m.get('spy_max_drawdown', 0) * 100:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

    st.markdown("### Concentration check (success criterion: ≤ 25% per ticker)")
    st.caption(
        "Per-ticker contribution to total excess return on the combined "
        "test + reserved-validation window. The contract's hard success "
        "criterion is no single ticker contributing > 25% of total alpha."
    )
    attr = load_contract_parquet(study_name, "per_ticker_attribution.parquet")
    if not attr.empty:
        rows = []
        for model in sorted(attr["model"].unique()):
            top = attr[attr["model"] == model].nlargest(1, "pct_of_total_alpha")
            if top.empty:
                continue
            ticker = top["ticker"].iloc[0]
            pct = float(top["pct_of_total_alpha"].iloc[0])
            rows.append({
                "model": model,
                "Top contributor": ticker,
                "% of total alpha": f"{pct:.1f}%",
                "≤ 25% constraint": "✅ Pass" if pct <= 25 else "❌ Fail",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

    conc = load_contract_concentration(study_name)
    if conc:
        st.markdown("### Repeat-holding profile")
        st.caption(
            "How often each top-ranked ticker was selected across rebalance "
            "dates. Persistent single-name presence is the structural "
            "driver of concentration even when per-rebalance weights stay "
            "below the individual cap."
        )
        rows = []
        for model, payload in conc.items():
            if not isinstance(payload, dict):
                continue
            top_holds = payload.get("top_10_repeat_holdings", {}) or {}
            top_str = ", ".join(f"{t}×{n}" for t, n in
                                list(top_holds.items())[:5])
            rows.append({
                "model": model,
                "Rebalances": payload.get("rebalance_dates", "—"),
                "Unique tickers": payload.get("unique_tickers_held", "—"),
                "Avg positions / rebalance":
                    f"{payload.get('avg_positions_per_rebalance', 0):.1f}",
                "Max single-ticker weight":
                    f"{payload.get('max_single_ticker_weight', 0) * 100:.2f}%",
                "Max sector weight (any date)":
                    f"{payload.get('max_sector_weight_across_dates', 0) * 100:.1f}%",
                "Top repeat holdings (×N)": top_str,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

    st.markdown("### Objective + construction")
    obj = meta.get("objective", {})
    pc = meta.get("portfolio_construction", {})
    st.markdown(
        f"- **Training CV objective**: `{obj.get('training_cv', '—')}` "
        f"(see [memo](../../docs/architecture/ml_study_cv_objectives_v1.md))\n"
        f"- **Headline objective**: `{obj.get('headline', '—')}`\n"
        f"- **Construction**: `{pc.get('method', '—')}` "
        f"n={pc.get('n', '—')}, "
        f"individual_cap={pc.get('individual_cap', '—')}, "
        f"sector_cap={pc.get('sector_cap', '—')}"
    )

    # === Reserved validation period — explanatory note ===
    st.markdown("---")
    st.markdown(
        "**About the reserved validation period.** The vertical line on "
        "the chart marks where the reserved validation period begins. "
        "Data after this line was deliberately not analyzed during the "
        "study to prevent analyst-level selection bias — it was kept "
        "untouched until the final writeup as an independent check of "
        "whether the strategy generalized beyond the period we examined. "
        "Both the test window (left of the line) and the reserved "
        "validation window (right of the line) are out-of-sample for the "
        "model; the distinction protects against subtle analyst biases "
        "like cherry-picking metrics or time slices, not model-level "
        "data contamination."
    )


def tab_contract_holdings(study_name: str) -> None:
    holdings = load_contract_parquet(study_name, "holdings.parquet")
    if holdings.empty:
        st.warning("No holdings.parquet found.")
        return
    holdings["date"] = pd.to_datetime(holdings["date"])
    models = sorted(holdings["model"].unique())
    model = st.selectbox(
        "Model", models,
        index=_default_model_index(study_name, models),
        key="contract_hold_model",
    )
    dates = sorted(holdings[holdings["model"] == model]["date"].unique(),
                   reverse=True)
    date_pick = st.selectbox(
        "Rebalance date", dates,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        key="contract_hold_date",
    )
    sub = holdings[(holdings["model"] == model) &
                   (holdings["date"] == date_pick)].copy()
    if "weight" in sub.columns:
        sub = sub.sort_values("weight", ascending=False)
        sub["weight"] = sub["weight"].apply(lambda v: f"{v * 100:.2f}%")
    st.dataframe(sub, use_container_width=True, hide_index=True)
    st.caption(f"{len(sub)} positions held on "
               f"{pd.Timestamp(date_pick).strftime('%Y-%m-%d')}.")


def tab_contract_trades(study_name: str) -> None:
    trades = load_contract_parquet(study_name, "trades.parquet")
    if trades.empty:
        st.warning("No trades.parquet found.")
        return
    for c in ("date", "exit_date"):
        if c in trades.columns:
            trades[c] = pd.to_datetime(trades[c])
    models = sorted(trades["model"].unique())
    model = st.selectbox(
        "Model", models,
        index=_default_model_index(study_name, models),
        key="contract_trade_model",
    )
    sub = trades[trades["model"] == model].copy()
    st.caption(f"{len(sub)} round-trip trades — {model}")
    st.dataframe(sub, use_container_width=True, hide_index=True)


def tab_contract_alpha(study_name: str) -> None:
    df = load_contract_parquet(study_name, "per_ticker_attribution.parquet")
    if df.empty:
        st.warning("No per_ticker_attribution.parquet found.")
        return
    models = sorted(df["model"].unique())
    model = st.selectbox(
        "Model", models,
        index=_default_model_index(study_name, models),
        key="contract_alpha_model",
    )
    sub = df[df["model"] == model].nlargest(25, "pct_of_total_alpha").copy()
    fig = go.Figure(go.Bar(
        x=sub["pct_of_total_alpha"],
        y=sub["ticker"],
        orientation="h",
        marker=dict(
            color=[
                "#c0392b" if v > 25 else "#1f4e79"
                for v in sub["pct_of_total_alpha"]
            ],
        ),
    ))
    fig.add_vline(x=25, line=dict(color="red", dash="dash", width=1.2),
                  annotation_text="25% constraint")
    fig.update_layout(
        title=f"{model} — top 25 alpha contributors (% of total excess return)",
        xaxis_title="% of total alpha",
        height=620,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sub, use_container_width=True, hide_index=True)


def _render_scope_callout(scope_info: dict, default_caption: str) -> None:
    """Render scope context for a scope-sensitive artifact.

    Uses `artifact_metadata.<file>` per `dashboard_contract_v1.md`. If
    scope info is missing or unrecognized, falls back to the
    default_caption.
    """
    if not scope_info:
        st.caption(default_caption)
        return
    scope = scope_info.get("scope")
    description = scope_info.get("scope_description") or ""
    audit_ref = scope_info.get("audit_reference")
    if scope == "held_subset":
        # Corrective callout — values are not the standard interpretation
        msg = (
            f"**Scope: held-subset.** {description}"
        )
        if audit_ref:
            msg += f" Audit: `{audit_ref}`."
        st.warning(msg)
    elif scope == "full_cross_section":
        # Informational — standard interpretation
        msg = f"**Scope: full cross-section.** {description}"
        if audit_ref:
            msg += f" See `{audit_ref}`."
        st.info(msg)
    else:
        # "other" or unknown scope value
        msg = (
            f"**Scope: {scope}.** {description}"
        )
        if audit_ref:
            msg += f" See `{audit_ref}`."
        st.caption(msg)


def tab_contract_diagnostics(study_name: str) -> None:
    # Read scope information for scope-sensitive artifacts per the
    # artifact_metadata schema in dashboard_contract_v1.md. When scope is
    # present per-artifact, the dashboard surfaces it inline above each
    # artifact's rendering and SKIPS the legacy fallback banner. When
    # absent, falls back to a legacy-v1 study-name-specific banner so
    # partners still see correction context until v1's meta.json is
    # annotated (see docs/studies/larger_universe_v1/ic_scope_audit.md).
    try:
        meta = load_contract_meta(study_name)
    except Exception:
        meta = {}
    artifact_metadata = meta.get("artifact_metadata") or {}
    ic_scope_info = artifact_metadata.get("ic_decomposition.parquet")
    dr_scope_info = artifact_metadata.get("decile_returns.parquet")

    # Legacy fallback banner — only when artifact_metadata doesn't cover
    # the scope-sensitive artifacts AND the study is the known legacy
    # v1 case. Replaced by the per-artifact callouts when annotations land.
    if (
        (ic_scope_info is None or dr_scope_info is None)
        and study_name == "larger_universe_v1"
    ):
        st.warning(
            "**Note on scope** — The IC decomposition and decile returns "
            "tables below were computed on a held-subset price universe "
            "(450 tickers across XGBoost and ElasticNet holdings) rather "
            "than the full eligible cross-section (1,963 tickers). "
            "Held-subset scope produces `top_quintile_ic_mean = +0.0481` "
            "as displayed below; the standard full-cross-section "
            "equivalent is **−0.0041**. Decile 1's `+35.7%` mean / "
            "`±202%` std is driven by ~5 held tickers per rebalance in "
            "the bottom decile (small-sample tail). Full-cross-section "
            "Decile 1 mean is +5.8% (std 25%). "
            "See `docs/studies/larger_universe_v1/ic_scope_audit.md` for "
            "the audit. v2 and future studies compute these metrics at "
            "full cross-section by default."
        )

    st.markdown("### IC decomposition")
    ic = load_contract_parquet(study_name, "ic_decomposition.parquet")
    if not ic.empty:
        _render_scope_callout(
            ic_scope_info,
            default_caption=(
                "Full-cross-section IC is the standard Spearman IC across all "
                "scored tickers per date, averaged. Top-quintile IC restricts "
                "to the top 20% of scores per date. For top-N portfolio "
                "strategies the top-quintile IC is the more deployment-aligned "
                "signal — see `docs/architecture/ml_study_cv_objectives_v1.md`."
            ),
        )
        st.dataframe(ic, use_container_width=True, hide_index=True)

    st.markdown("### Decile returns")
    dr = load_contract_parquet(study_name, "decile_returns.parquet")
    if not dr.empty:
        _render_scope_callout(
            dr_scope_info,
            default_caption=(
                "Per-decile mean forward 21d return with std error bars. "
                "Score-by-decile bucketing on the full eligible universe; "
                "forward returns from snapshot prices."
            ),
        )
        fig = go.Figure()
        for model in sorted(dr["model"].unique()):
            m = dr[dr["model"] == model].sort_values("decile")
            fig.add_trace(go.Bar(
                x=m["decile"].astype(int),
                y=m["mean_fwd_return"] * 100,
                name=model,
                error_y=dict(
                    type="data",
                    array=m["std_fwd_return"] * 100,
                    visible=True,
                    thickness=0.8,
                    width=0,
                ),
            ))
        fig.update_layout(
            barmode="group",
            title="Mean forward 21d return per score decile",
            xaxis_title="Decile (1 = lowest, 10 = highest)",
            yaxis_title="Mean fwd 21d return (%)",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Rolling 12-month win rate")
    rw = load_contract_parquet(study_name, "rolling_win_rate.parquet")
    if not rw.empty:
        st.dataframe(rw, use_container_width=True, hide_index=True)


_WALK_FORWARD_REGIME_LABELS = {
    # Keyed by val_start (YYYY-MM-DD). Descriptive only — factual market
    # events, not editorial framing. Extend when future studies push the
    # walk-forward into earlier or later windows.
    "2020-05-12": "COVID crash + recovery",
    "2021-05-12": "Late COVID recovery",
    "2022-05-12": "2022 bear market + reversal",
    "2023-05-12": "AI rally year 1",
    "2024-05-12": "AI rally year 2 + rate environment",
    "2025-05-12": "Most recent 12 months",
}


def tab_contract_walk_forward(study_name: str) -> None:
    wf = load_contract_parquet(study_name, "walk_forward.parquet")
    if wf.empty:
        st.warning("No walk_forward.parquet found.")
        return
    st.caption(
        "Walk-forward stability: each row is one rolling 3-year-train / "
        "1-year-validation window. excess_cagr_vs_spy < 0 in some windows is "
        "expected; what matters is sign and magnitude consistency."
    )

    models = sorted(wf["model"].unique())

    # === Summary statistics panel — per model ===
    st.markdown("### Window-level summary")
    cols = st.columns(len(models))
    for col, model in zip(cols, models):
        m = (wf[wf["model"] == model]
             .sort_values("val_start").reset_index(drop=True))
        excess = m["excess_cagr_vs_spy"]
        n_total = len(m)
        n_positive = int((excess > 0).sum())
        n_strong = int((excess >= 0.05).sum())
        median_excess = excess.median()
        std_excess = excess.std(ddof=1) if n_total > 1 else 0.0
        best_pos = int(excess.idxmax())
        worst_pos = int(excess.idxmin())
        best_w = best_pos + 1
        worst_w = worst_pos + 1
        best_val = float(excess.iloc[best_pos])
        worst_val = float(excess.iloc[worst_pos])

        with col:
            st.markdown(f"**{model}**")
            st.metric("Windows positive", f"{n_positive} of {n_total}")
            st.metric("Median excess CAGR",
                      f"{median_excess * 100:+.1f}pp")
            st.metric("Best window",
                      f"W{best_w}  ({best_val * 100:+.1f}pp)")
            st.metric("Worst window",
                      f"W{worst_w}  ({worst_val * 100:+.1f}pp)")
            st.metric("Std dev (excess CAGR)",
                      f"{std_excess * 100:.1f}pp")
            st.metric("Strong outperformance (≥ +5pp)",
                      f"{n_strong} of {n_total}")

    st.caption(
        "Windows are numbered W1–W6 in chronological order of validation "
        "start. \"Strong outperformance\" counts windows where the model "
        "beat SPY by ≥ 5pp CAGR in that window."
    )

    # === Bar chart with regime annotations ===
    fig = go.Figure()
    for model in models:
        m = (wf[wf["model"] == model]
             .sort_values("val_start").reset_index(drop=True))
        # Two-line tick label: year on top, regime label in smaller grey
        # text underneath. Hover tooltips also carry the regime explicitly.
        x_labels = []
        regimes = []
        for v in m["val_start"]:
            v_str = str(v)[:10]
            year = v_str[:4]
            regime = _WALK_FORWARD_REGIME_LABELS.get(v_str, "")
            regimes.append(regime)
            if regime:
                x_labels.append(
                    f"{year}<br>"
                    f"<span style='font-size:10px;color:#94a3b8'>"
                    f"{regime}</span>"
                )
            else:
                x_labels.append(year)
        fig.add_trace(go.Bar(
            x=x_labels,
            y=m["excess_cagr_vs_spy"] * 100,
            name=model,
            customdata=regimes,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Val start year: %{x}<br>"
                "Excess CAGR: %{y:+.2f}pp<br>"
                "Regime: %{customdata}"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="group",
        title="Per-window excess CAGR vs SPY",
        xaxis_title="Validation window (year + regime context)",
        yaxis_title="Excess CAGR vs SPY (pp)",
        height=480,
        margin=dict(b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # === Synthetic compounded growth curve ===
    st.markdown("### Synthetic compounded growth across windows")
    st.caption(
        "What $1 would grow to if each window's annualized CAGR held for "
        "one year, compounded across the six non-overlapping 1-year "
        "validation windows. **Different from the Overview NAV chart**, "
        "which shows the actual deployed portfolio NAV under locked Phase 3 "
        "hyperparameters over test + OOS; this view assumes per-window "
        "retrains and treats each window's CAGR as that year's realized "
        "growth. Useful for visualizing the cumulative effect of the "
        "per-window excess CAGRs in the bar chart above."
    )

    line_fig = go.Figure()

    # SPY is identical across model rows for the same window — take it
    # from either subset.
    spy_subset = (wf[wf["model"] == models[0]]
                  .sort_values("val_start").reset_index(drop=True))
    spy_growth = (1 + spy_subset["spy_cagr"]).cumprod()
    x_period_labels = [
        f"After W{i+1} ({str(v)[:7]})"
        for i, v in enumerate(spy_subset["val_end"])
    ]
    x_with_start = ["Start"] + x_period_labels
    line_fig.add_trace(go.Scatter(
        x=x_with_start,
        y=[1.0] + list(spy_growth),
        mode="lines+markers",
        name="SPY",
        line=dict(width=1.6, dash="dash", color="#888888"),
        marker=dict(size=7),
    ))
    for model in models:
        m = (wf[wf["model"] == model]
             .sort_values("val_start").reset_index(drop=True))
        growth = (1 + m["cagr"]).cumprod()
        line_fig.add_trace(go.Scatter(
            x=x_with_start,
            y=[1.0] + list(growth),
            mode="lines+markers",
            name=model,
            line=dict(width=2.5),
            marker=dict(size=8),
        ))
    line_fig.update_layout(
        title="Synthetic compounded growth — $1 invested across "
              "walk-forward windows",
        xaxis_title="Window endpoint",
        yaxis_title="Cumulative NAV multiplier",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    st.plotly_chart(line_fig, use_container_width=True)

    # === Full window-level data table ===
    st.markdown("### Full window-level data")
    st.dataframe(wf, use_container_width=True, hide_index=True)


def _render_param_sensitivity(
    trial_log: pd.DataFrame, model_name: str,
) -> None:
    """Per-parameter scatter (param value vs trial score) with winner marked.

    Detects log-scale params heuristically (max/min > 100 on positive values).
    Overlays a binned-mean line so the eye picks up monotonic trends. Only
    plots params that actually have non-null values for the selected model
    (avoids empty subplots for the other model's params).
    """
    g = trial_log[
        (trial_log["tuning_study"] == model_name)
        & (trial_log["state"] == "COMPLETE")
    ].copy()
    if g.empty:
        st.caption("No COMPLETE trials for this model — sensitivity skipped.")
        return

    param_cols = [c for c in g.columns if c.startswith("param_")
                  and g[c].notna().any()]
    if not param_cols:
        st.caption("No populated params on this trial log; sensitivity skipped.")
        return

    winner_row = g.loc[g["value"].idxmax()]

    n = len(param_cols)
    ncols = 3 if n > 4 else min(n, 2)
    nrows = (n + ncols - 1) // ncols

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[c.replace("param_", "") for c in param_cols],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    for i, col in enumerate(param_cols):
        r = (i // ncols) + 1
        c = (i % ncols) + 1
        x = g[col].astype(float)
        y = g["value"].astype(float)

        # Scatter
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=5, color="#475569", opacity=0.55),
            showlegend=False, hoverinfo="x+y",
        ), row=r, col=c)

        # Winner marker — distinct red diamond
        w = float(winner_row[col]) if pd.notna(winner_row[col]) else None
        if w is not None:
            fig.add_trace(go.Scatter(
                x=[w], y=[float(winner_row["value"])], mode="markers",
                marker=dict(size=11, color="#dc2626", symbol="diamond",
                            line=dict(color="#7f1d1d", width=1)),
                showlegend=False, hoverinfo="x+y",
                name="winner",
            ), row=r, col=c)

        # Binned-mean overlay (10 quantile bins; collapses gracefully when
        # there are fewer distinct x values)
        try:
            bins = pd.qcut(x, q=min(10, x.nunique()),
                           duplicates="drop")
            bm = pd.DataFrame({"bin": bins, "x": x, "y": y}).groupby(
                "bin", observed=True
            ).agg(x_mid=("x", "median"), y_mean=("y", "mean"))
            if len(bm) >= 2:
                bm = bm.sort_values("x_mid")
                fig.add_trace(go.Scatter(
                    x=bm["x_mid"], y=bm["y_mean"], mode="lines",
                    line=dict(color="#2563eb", width=1.8),
                    showlegend=False, hoverinfo="skip",
                ), row=r, col=c)
        except (ValueError, TypeError):
            pass  # not all params support qcut; skip overlay silently

        # Log-axis heuristic — only when ALL values are strictly positive
        # and span > 100x. Avoids silently dropping zero-valued trials on
        # log axes for params like `gamma` that often include zero.
        if (x > 0).all() and x.max() / x.min() > 100:
            fig.update_xaxes(type="log", row=r, col=c)

    fig.update_layout(
        height=240 * nrows + 60,
        title=f"{model_name} — per-parameter sensitivity "
              f"(score vs param value; red diamond = winner)",
        margin=dict(t=80, l=40, r=20, b=40),
    )
    fig.update_yaxes(title_text="score" if ncols == 1 else None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each panel shows trial score (y) against one hyperparameter (x). "
        "Blue line is a binned-mean overlay. Red diamond marks the winning "
        "trial's parameter value. Flat scatter = parameter has weak signal "
        "in this search; strong gradient = optimizer found a productive "
        "direction."
    )


def tab_contract_tuning(study_name: str) -> None:
    """Tuning tab: narrative + histogram + convergence curve + sensitivity
    + collapsed trial log + feature importance."""
    tl = load_contract_parquet(study_name, "trial_log.parquet")
    if tl.empty:
        st.info("No trial_log.parquet found — this study did not perform tuning.")
        # Still try to render feature importance below (separate artifact).
    else:
        meta = load_contract_meta(study_name)
        st.caption(
            f"CV objective: "
            f"`{meta.get('objective', {}).get('training_cv', '?')}`."
        )

        models = sorted(tl["tuning_study"].dropna().unique())
        if not models:
            st.warning("trial_log.parquet has no models.")
        else:
            model = st.selectbox(
                "Model", models,
                index=_default_model_index(study_name, models),
                key="contract_tune_model_selector",
            )

            conv = load_contract_parquet(
                study_name, "tuning_convergence.parquet",
            )
            summary = load_contract_tuning_summary(study_name)
            has_precomputed = (
                not conv.empty
                and summary is not None
                and model in summary
                and summary[model].get("total_trials", 0) > 0
            )

            if has_precomputed:
                m_summary = summary[model]
                m_conv = conv[conv["model"] == model].sort_values(
                    "trial_number",
                )

                # === Section A — narrative summary ===
                pct = m_summary.get("pct_trials_to_plateau", 0) or 0
                win_z = m_summary.get("winner_zscore")
                z_phrase = (
                    f" Winner sits **{win_z:+.2f}σ** vs the trial-score mean."
                    if win_z is not None else ""
                )
                st.info(
                    f"The optimizer tested **{m_summary['total_trials']}** "
                    f"configurations for **{model}**. The winner was "
                    f"**Trial #{m_summary['winning_trial']}** with score "
                    f"**{m_summary['winning_score']:.4f}**. 95% of the "
                    f"winning score was reached after about **{pct:.0%}** "
                    f"of the trials — the curve plateaus early, then "
                    f"refinement happens at the margin. Optuna is "
                    f"**search, not proof**: a different random seed or "
                    f"longer search might find a better config or might "
                    f"find that this peak doesn't generalize to other "
                    f"validation windows." + z_phrase
                )

                # === Section B — score distribution histogram ===
                complete_scores = tl.loc[
                    (tl["tuning_study"] == model)
                    & (tl["state"] == "COMPLETE"),
                    "value",
                ].dropna()
                mean_s = float(m_summary["mean_score"])
                std_s = float(m_summary["std_score"])
                win_score = float(m_summary["winning_score"])

                hist = go.Figure()
                hist.add_trace(go.Histogram(
                    x=complete_scores, nbinsx=min(40, max(10, len(complete_scores) // 5)),
                    marker=dict(color="#475569",
                                line=dict(color="#1f2937", width=0.5)),
                    name="Trials", showlegend=False,
                ))
                if std_s > 0:
                    hist.add_vrect(
                        x0=mean_s - 2 * std_s, x1=mean_s + 2 * std_s,
                        fillcolor="#94a3b8", opacity=0.10, line_width=0,
                        layer="below", annotation_text="±2σ",
                        annotation_position="top left",
                        annotation=dict(font=dict(size=10, color="#475569")),
                    )
                    hist.add_vrect(
                        x0=mean_s - std_s, x1=mean_s + std_s,
                        fillcolor="#94a3b8", opacity=0.18, line_width=0,
                        layer="below", annotation_text="±1σ",
                        annotation_position="top left",
                        annotation=dict(font=dict(size=10, color="#475569")),
                    )
                    hist.add_vline(x=mean_s, line_dash="dot",
                                   line_color="#475569", line_width=1)
                z_label = (
                    f" ({win_z:+.2f}σ above mean)"
                    if win_z is not None else ""
                )
                hist.add_vline(
                    x=win_score, line_color="#dc2626", line_width=2.5,
                    annotation_text=(
                        f"Winner: Trial #{m_summary['winning_trial']} "
                        f"(score {win_score:.4f}){z_label}"
                    ),
                    annotation_position="top right",
                    annotation=dict(font=dict(size=11, color="#dc2626")),
                )
                hist.update_layout(
                    title=(
                        f"Trial score distribution — "
                        f"{m_summary['total_trials']:,} configurations"
                    ),
                    xaxis_title="Trial score (CV objective value)",
                    yaxis_title="Number of trials",
                    height=380, margin=dict(l=10, r=10, t=50, b=10),
                    bargap=0.05,
                )
                st.plotly_chart(hist, use_container_width=True)

                # === Section C — running-best convergence curve ===
                conv_fig = go.Figure()
                conv_fig.add_trace(go.Scatter(
                    x=m_conv["trial_number"], y=m_conv["score"],
                    mode="markers", name="Trial score",
                    marker=dict(size=5, color="#94a3b8", opacity=0.55),
                ))
                conv_fig.add_trace(go.Scatter(
                    x=m_conv["trial_number"],
                    y=m_conv["running_best_score"],
                    mode="lines", name="Running best",
                    line=dict(color="#2563eb", width=2.5),
                ))
                plateau_trial = m_summary.get("trials_to_95pct_winning")
                if plateau_trial is not None:
                    conv_fig.add_vline(
                        x=int(plateau_trial),
                        line=dict(color="#16a34a", dash="dash", width=1.5),
                        annotation_text=(
                            f"95% plateau (trial #{int(plateau_trial)}, "
                            f"~{pct:.0%} of trials)"
                        ),
                        annotation_position="bottom right",
                        annotation=dict(font=dict(size=10, color="#16a34a")),
                    )
                conv_fig.add_hline(
                    y=win_score,
                    line=dict(color="#dc2626", dash="dot", width=1),
                    annotation_text=f"Winner: {win_score:.4f}",
                    annotation_position="top left",
                    annotation=dict(font=dict(size=10, color="#dc2626")),
                )
                conv_fig.update_layout(
                    title="Running-best convergence",
                    xaxis_title="Trial number (0-indexed)",
                    yaxis_title="Score",
                    height=400, margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                )
                st.plotly_chart(conv_fig, use_container_width=True)
            else:
                st.caption(
                    "Pre-computed convergence data not found "
                    "(`tuning_convergence.parquet` / `tuning_summary.json`). "
                    "Run `scripts/maintenance/backfill_tuning_convergence.py "
                    f"--study {study_name}` to enrich, or this study "
                    "will get the narrative + histogram + convergence-curve "
                    "sections once its Phase 3 produces them natively."
                )

            # === Section D — per-parameter sensitivity ===
            st.markdown("### Parameter sensitivity")
            _render_param_sensitivity(tl, model)

            # === Section E — trial log (collapsed, secondary position) ===
            model_tl = tl[tl["tuning_study"] == model]
            with st.expander(
                f"Trial log ({len(model_tl)} trials)", expanded=False,
            ):
                st.dataframe(
                    model_tl.head(50),
                    use_container_width=True,
                    hide_index=True,
                )
                if len(model_tl) > 50:
                    st.caption(
                        f"Showing first 50 of {len(model_tl)} trials. "
                        "Re-export the parquet for the full table."
                    )

    # === Feature importance (unchanged — separate analytical view) ===
    fi = load_contract_parquet(study_name, "feature_importance.parquet")
    if not fi.empty:
        st.markdown("### Feature importance")
        method = load_contract_meta(study_name).get(
            "feature_importance_method", "?",
        )
        st.caption(f"Method: `{method}`")
        models = sorted(fi["model"].unique())
        fi_model = st.selectbox(
            "Model", models,
            index=_default_model_index(study_name, models),
            key="contract_tune_fi_model",
        )
        sub = fi[fi["model"] == fi_model].nlargest(20, "importance").copy()
        fig = go.Figure(go.Bar(
            x=sub["importance"], y=sub["feature"],
            orientation="h",
        ))
        fig.update_layout(
            title=f"{fi_model} — top 20 features by importance",
            xaxis_title="Importance",
            height=540,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_contract_variant_comparison(study_name: str) -> None:
    """Variant Comparison tab — 8th tab for multi-variant studies.

    Renders the headline verdict + cross-variant concentration overlap as
    the prominent finding (matching the writeup's hierarchy at
    docs/studies/larger_universe_v2/results.md). Per-criterion detail is
    available in a collapsible expander below the prominent sections.
    """
    comparison_path = CONTRACT_V1_DIR / study_name / "comparison" / "comparison_results.parquet"
    if not comparison_path.exists():
        st.warning(
            "No `comparison/comparison_results.parquet` found for this "
            "multi-variant study. The Variant Comparison tab requires "
            "`scripts/research/build_comparison_results_v2.py --variants all` "
            "(or equivalent) to have run."
        )
        return

    comparison = pd.read_parquet(comparison_path)
    n_variants = len(comparison)

    # === Top: verdict callout ===
    promote_count = int((comparison["verdict"] == "PROMOTE").sum())
    methodology_count = int((comparison["verdict"] == "METHODOLOGY FINDING").sum())
    not_promoted_count = int((comparison["verdict"] == "NOT PROMOTED").sum())

    if promote_count > 0:
        st.success(
            f"**{promote_count} of {n_variants} variants promoted.** See per-variant detail below."
        )
    else:
        st.error(
            f"**No variant promoted; {methodology_count} of {n_variants} are "
            f"methodology findings** ({not_promoted_count} did not pass any "
            f"criterion). Per-variant `n_pass` summary below."
        )

    # Per-variant n_pass summary table — sorted by n_pass descending
    summary = comparison[["variant", "n_pass", "verdict"]].copy()
    summary = summary.sort_values("n_pass", ascending=False).reset_index(drop=True)
    summary["n_pass_display"] = summary["n_pass"].astype(str) + "/7"
    st.dataframe(
        summary[["variant", "n_pass_display", "verdict"]].rename(
            columns={"variant": "Variant", "n_pass_display": "Pass count",
                     "verdict": "Verdict"},
        ),
        use_container_width=True,
        hide_index=True,
    )

    # === Below verdict: cross-variant concentration overlap ===
    st.markdown("### Cross-variant concentration overlap")
    overlap_path = CONTRACT_V1_DIR / study_name / "comparison" / "concentration_overlap.parquet"
    corr_path = CONTRACT_V1_DIR / study_name / "comparison" / "concentration_corr_matrix.parquet"
    overlap_summary_path = CONTRACT_V1_DIR / study_name / "comparison" / "concentration_overlap_summary.json"

    if not (corr_path.exists() and overlap_summary_path.exists()):
        st.caption(
            "Concentration overlap artifacts not found "
            "(`concentration_corr_matrix.parquet` + "
            "`concentration_overlap_summary.json`). Run "
            "`scripts/research/phase5_analytics_v2.py --variants all` to "
            "produce them."
        )
    else:
        # --- 7x7 annotated heatmap (Spearman correlation of pct_of_total_alpha) ---
        corr_long = pd.read_parquet(corr_path)
        # Pivot to wide for the heatmap
        cm = corr_long.pivot(
            index="variant_a", columns="variant_b", values="spearman_corr",
        )
        # Order variants consistently for readability
        variant_order = sorted(cm.index)
        cm = cm.loc[variant_order, variant_order]

        # Color scale calibrated to the off-diagonal range (don't compress with 0-1 scale)
        off_diag = corr_long[corr_long["variant_a"] != corr_long["variant_b"]]["spearman_corr"]
        vmin = float(off_diag.min()) if not off_diag.empty else 0.8
        vmax = 1.0
        # Pad zmin slightly so the lowest values aren't at the edge of the scale
        zmin = max(0.0, vmin - 0.01)

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=cm.values,
            x=list(cm.columns),
            y=list(cm.index),
            colorscale="Blues",
            zmin=zmin,
            zmax=vmax,
            text=[[f"{v:.3f}" for v in row] for row in cm.values],
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b>: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Spearman corr"),
        ))
        heatmap_fig.update_layout(
            title=(
                "Cross-variant Spearman correlation on per-ticker "
                "pct_of_total_alpha"
            ),
            height=440,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

        # --- Appearance histogram (in-N-variants distribution) ---
        with open(overlap_summary_path) as f:
            overlap_summary = json.load(f)
        dist = overlap_summary.get("appearance_count_distribution", {}) or {}
        # Keys may be string in JSON; coerce to int
        try:
            keyed = {int(k): int(v) for k, v in dist.items()}
        except Exception:
            keyed = {}
        # Build a complete sequence 1..n_variants so empty bins render visibly
        xs = list(range(1, n_variants + 1))
        ys = [keyed.get(x, 0) for x in xs]
        hist_fig = go.Figure(data=go.Bar(
            x=xs, y=ys,
            text=[str(y) for y in ys],
            textposition="outside",
            marker_color="#1d4ed8",
        ))
        hist_fig.update_layout(
            title="Top-20 alpha contributors — appearance distribution",
            xaxis=dict(
                title="Appears in N variants' top-20",
                tickmode="linear", tick0=1, dtick=1,
            ),
            yaxis=dict(title="Count of unique tickers"),
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        # --- Caption stating the substantive finding ---
        mean_corr = overlap_summary.get("cross_variant_spearman_mean")
        n_in_all = keyed.get(n_variants, 0)
        n_union = overlap_summary.get("union_top_tickers_count", 0)
        mean_corr_str = f"{mean_corr:.3f}" if mean_corr is not None else "?"
        # Identify the outlier variant (lowest mean correlation with others)
        off_diag_by_a = (
            corr_long[corr_long["variant_a"] != corr_long["variant_b"]]
            .groupby("variant_a")["spearman_corr"].mean()
            .sort_values()
        )
        outlier_str = ""
        if len(off_diag_by_a) >= 2:
            outlier_variant = off_diag_by_a.index[0]
            outlier_corr = float(off_diag_by_a.iloc[0])
            outlier_str = (
                f" {outlier_variant} is the most divergent variant "
                f"with ~{outlier_corr:.2f} mean correlation to others."
            )
        st.caption(
            f"**The {mean_corr_str} mean cross-variant correlation and "
            f"{n_in_all} of {n_union} shared top contributors indicate that "
            f"concentration is model-determined — different construction "
            f"logics select largely the same names.**{outlier_str}"
        )

    # === Per-criterion comparison (collapsible) ===
    with st.expander(
        f"Full per-criterion comparison (all {n_variants} variants × 7 criteria)",
        expanded=False,
    ):
        st.caption(
            "Each criterion has a pre-committed threshold (see "
            "`docs/studies/larger_universe_v2/spec.md`). A variant must "
            "meet ALL seven to promote. Per-criterion value columns show "
            "magnitude of pass/fail (e.g. 'passed by 22%' vs 'failed by 1%')."
        )
        # Show value + pass columns for each criterion
        crit_cols = ["variant"]
        for i in range(1, 8):
            # Find the value column for this criterion (varies by criterion)
            value_col = next(
                (c for c in comparison.columns
                 if c.startswith(f"criterion_{i}_") and not c.endswith("_pass")),
                None,
            )
            pass_col = f"criterion_{i}_pass"
            if value_col is not None:
                crit_cols.append(value_col)
            if pass_col in comparison.columns:
                crit_cols.append(pass_col)
        existing = [c for c in crit_cols if c in comparison.columns]
        st.dataframe(
            comparison[existing], use_container_width=True, hide_index=True,
        )

    # === Walk-forward consistency stats per variant ===
    st.markdown("### Walk-forward consistency stats per variant")
    st.caption(
        "Per-window excess CAGR vs SPY across the 6 walk-forward retrains "
        "(3y-train / 1y-validation). Mean, std, positive count from each "
        "variant's `walk_forward.parquet`."
    )
    wf_cols = [
        "variant",
        "mean_excess_cagr_walkforward",
        "std_excess_cagr_walkforward",
        "median_excess_cagr_walkforward",
        "min_excess_cagr_walkforward",
        "max_excess_cagr_walkforward",
        "n_windows_positive",
    ]
    wf_present = [c for c in wf_cols if c in comparison.columns]
    if len(wf_present) > 1:
        wf_df = comparison[wf_present].copy()
        wf_df = wf_df.rename(columns={
            "variant": "Variant",
            "mean_excess_cagr_walkforward": "Mean excess",
            "std_excess_cagr_walkforward": "Std excess",
            "median_excess_cagr_walkforward": "Median excess",
            "min_excess_cagr_walkforward": "Min excess",
            "max_excess_cagr_walkforward": "Max excess",
            "n_windows_positive": "Pos. windows",
        })
        st.dataframe(wf_df, use_container_width=True, hide_index=True)

    # === Honest framing footer ===
    if promote_count == 0:
        st.info(
            "**v2's findings indicate the binding constraint for top-N "
            "equity strategies on this universe is signal extraction "
            "(Mechanism A), not portfolio construction (Mechanism B). "
            "See the v2 writeup at `docs/studies/larger_universe_v2/"
            "results.md` for full analysis, including the IC scope audit, "
            "decile structure under standard definitions, and per-variant "
            "supporting detail.**"
        )


def main_contract() -> None:
    """Renders contract-conformant studies (dashboard_contract_v1)."""
    st.sidebar.title("📊 Contract-conformant study")
    study_ref = sidebar_contract_picker()
    if study_ref is None:
        st.info(
            "No contract-conformant studies are available yet. The first "
            "such study lands as part of feat/larger-universe-v1-study. "
            "Switch to **Legacy studies** in the sidebar to explore promoted "
            "Optuna v1 studies."
        )
        return
    meta = load_contract_meta(study_ref)
    st.caption(
        f"Spec: `{meta.get('spec_doc', '—')}` • "
        f"Schema: `{meta.get('schema_version', '—')}` • "
        f"Created: {meta.get('created_at', '?')[:10]}"
    )

    # 7 universal contract tabs + optional 8th Variant Comparison tab for
    # multi-variant studies. The 7 universal tabs each render the
    # selected-variant's contract_v1/ artifacts (variant routing via the
    # composite study_ref "study/variant"; see _contract_dir).
    study_root, _ = _split_study_ref(study_ref)
    is_multi_variant = bool(list_contract_v1_variants(study_root))

    tab_labels = [
        "Overview", "Holdings", "Trades", "Alpha Attribution",
        "Diagnostics", "Walk-forward", "Tuning",
    ]
    if is_multi_variant:
        tab_labels.append("Variant Comparison")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        tab_contract_overview(study_ref)
    with tabs[1]:
        tab_contract_holdings(study_ref)
    with tabs[2]:
        tab_contract_trades(study_ref)
    with tabs[3]:
        tab_contract_alpha(study_ref)
    with tabs[4]:
        tab_contract_diagnostics(study_ref)
    with tabs[5]:
        tab_contract_walk_forward(study_ref)
    with tabs[6]:
        tab_contract_tuning(study_ref)
    if is_multi_variant:
        with tabs[7]:
            tab_contract_variant_comparison(study_root)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # Auth gate (cloud mode only). Local mode skips the gate so dev
    # iteration stays friction-free.
    if CLOUD_MODE:
        from dashboard_auth import gate
        gate()       # st.stop()s if not authenticated; renders sidebar logout

    # Asset-class selector goes BEFORE everything else in the sidebar so
    # it's the first control the user sees. Phase 2 (crypto) will swap
    # the body of the Crypto branch for a real renderer.
    asset_class_label = sidebar_asset_picker()

    st.title("📈 Paper Trader Dashboard")
    if CLOUD_MODE:
        st.caption("☁️ Cloud read-only build — data sourced from snapshot bucket.")

    if asset_class_label == "Crypto":
        st.info("Crypto dashboard is in development. Switch to **Stocks** "
                "to view current data.")
        return
    elif asset_class_label == "Options":
        st.info("Options module — Phase 2 in progress. "
                "See `docs/Options_Extension_Decisions.md`.")
        return

    # Phase 4.5 — sidebar separator between legacy (Optuna v1) and
    # contract-conformant studies. Legacy stays the default so existing
    # workflows are untouched. Contract-conformant studies render from
    # models/studies/<name>/contract_v1/ artifacts via main_contract().
    study_type = st.sidebar.radio(
        "Study type",
        options=["Legacy (Optuna v1)", "Contract-conformant (v1+)"],
        index=0,
        key="study_type_selector",
        help="Legacy renders promoted Optuna v1 studies via the live-fallback "
             "compute path. Contract-conformant reads pre-computed artifacts "
             "from models/studies/<name>/contract_v1/.",
    )
    if study_type == "Contract-conformant (v1+)":
        main_contract()
        return

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
