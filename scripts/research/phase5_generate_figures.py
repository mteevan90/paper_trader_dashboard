"""Generate static PNGs for the Larger Universe v1 writeup.

Reads from contract_v1/ artifacts; emits to
  docs/studies/larger_universe_v1/figures/

PNGs produced:
  - equity_curves.png        — NAV vs 4 benchmarks
  - year_by_year.png         — Year-by-year excess return bars per model
  - decile_returns.png       — Per-decile forward returns per model
  - alpha_attribution.png    — Top-15 contributors per model
  - walk_forward.png         — Per-window excess CAGR
  - ic_decomposition.png     — Full-IC vs top-quintile-IC bar chart
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "models" / "studies" / "larger_universe_v1" / "contract_v1"
FIG_DIR = ROOT / "docs" / "studies" / "larger_universe_v1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig_equity_curves():
    port = pd.read_parquet(CONTRACT / "portfolio.parquet")
    bench = pd.read_parquet(CONTRACT / "benchmarks.parquet")
    port["date"] = pd.to_datetime(port["date"])
    bench["date"] = pd.to_datetime(bench["date"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors_model = {"xgboost": "#1f4e79", "elasticnet": "#2e7d32"}
    colors_bench = {"SPY": "#888888", "RSP": "#a8a8a8", "IWM": "#c8c8c8", "EW-SP1500": "#777777"}
    for model in port["model"].unique():
        m = port[port["model"] == model]
        ax.plot(m["date"], m["nav"], label=f"{model}", linewidth=2,
                color=colors_model.get(model, "black"))
    for b in bench["benchmark"].unique():
        bb = bench[bench["benchmark"] == b]
        ax.plot(bb["date"], bb["nav"], label=b, linewidth=1.2, linestyle="--",
                color=colors_bench.get(b, "gray"))
    # Vertical line at OOS start
    ax.axvline(pd.Timestamp("2026-01-01"), color="red", linestyle=":", alpha=0.5)
    ax.text(pd.Timestamp("2026-01-01"), ax.get_ylim()[1] * 0.95, "  OOS start", color="red",
            fontsize=9, verticalalignment="top")

    ax.set_title("Larger Universe v1 — Strategy NAV vs Benchmarks\n"
                 "Test (2023-05-12 → 2025-12-31) + OOS slice (2026-01-01 → snapshot end)",
                 fontsize=11)
    ax.set_ylabel("NAV (starts at 1.0)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "equity_curves.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}")


def fig_year_by_year():
    """Year-by-year total return for each model and SPY in the test+OOS span."""
    port = pd.read_parquet(CONTRACT / "portfolio.parquet")
    bench = pd.read_parquet(CONTRACT / "benchmarks.parquet")
    port["date"] = pd.to_datetime(port["date"])
    bench["date"] = pd.to_datetime(bench["date"])
    spy = bench[bench["benchmark"] == "SPY"].set_index("date")["nav"]
    spy.index = pd.to_datetime(spy.index)
    rows = []
    for model in port["model"].unique():
        m = port[port["model"] == model].set_index("date")["nav"].sort_index()
        m.index = pd.to_datetime(m.index)
        # Yearly first/last
        df_y = pd.DataFrame({"nav": m.values, "year": m.index.year}, index=m.index)
        for y, g in df_y.groupby("year"):
            if len(g) < 2:
                continue
            port_ret = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
            spy_y = spy[spy.index.year == y]
            spy_ret = spy_y.iloc[-1] / spy_y.iloc[0] - 1 if len(spy_y) >= 2 else np.nan
            excess = port_ret - spy_ret
            rows.append({"model": model, "year": y, "port_ret": port_ret,
                         "spy_ret": spy_ret, "excess": excess})
    df = pd.DataFrame(rows)
    years = sorted(df["year"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(len(years))
    for i, model in enumerate(df["model"].unique()):
        sub = df[df["model"] == model].set_index("year").reindex(years)
        ax.bar(x + (i - 0.5) * width, sub["excess"].values * 100, width,
               label=model, color=("#1f4e79" if model == "xgboost" else "#2e7d32"),
               alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Excess return vs SPY (pp)")
    ax.set_title("Year-by-year excess return vs SPY")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "year_by_year.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}")


def fig_decile_returns():
    df = pd.read_parquet(CONTRACT / "decile_returns.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for i, model in enumerate(["xgboost", "elasticnet"]):
        m = df[df["model"] == model].sort_values("decile")
        ax = axes[i]
        bars = ax.bar(m["decile"].astype(int), m["mean_fwd_return"] * 100,
                       color=("#1f4e79" if model == "xgboost" else "#2e7d32"),
                       alpha=0.85)
        # Add error bars (std)
        ax.errorbar(m["decile"].astype(int), m["mean_fwd_return"] * 100,
                     yerr=m["std_fwd_return"] * 100, fmt="none", color="black",
                     alpha=0.4, linewidth=0.8)
        ax.set_title(f"{model} — mean forward 21d return per score decile")
        ax.set_xlabel("Decile (1 = lowest score, 10 = highest)")
        ax.set_ylabel("Mean fwd 21d return (%)")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = FIG_DIR / "decile_returns.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}")


def fig_alpha_attribution():
    df = pd.read_parquet(CONTRACT / "per_ticker_attribution.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    for i, model in enumerate(["xgboost", "elasticnet"]):
        m = df[df["model"] == model].nlargest(15, "pct_of_total_alpha")
        ax = axes[i]
        colors = ["#c0392b" if v > 25 else "#1f4e79" for v in m["pct_of_total_alpha"]]
        if model == "elasticnet":
            colors = ["#c0392b" if v > 25 else "#2e7d32" for v in m["pct_of_total_alpha"]]
        ax.barh(m["ticker"], m["pct_of_total_alpha"], color=colors, alpha=0.85)
        ax.axvline(25, color="red", linestyle="--", linewidth=1.2,
                    label="25% constraint")
        ax.set_title(f"{model} — top 15 alpha contributors")
        ax.set_xlabel("% of total excess return")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
        ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "alpha_attribution.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}")


def fig_walk_forward():
    df = pd.read_parquet(CONTRACT / "walk_forward.parquet")
    fig, ax = plt.subplots(figsize=(11, 5))
    windows = df["val_start"].unique()
    x = np.arange(len(windows))
    width = 0.35
    for i, model in enumerate(["xgboost", "elasticnet"]):
        m = df[df["model"] == model].set_index("val_start").reindex(windows)
        ax.bar(x + (i - 0.5) * width,
               m["excess_cagr_vs_spy"].values * 100, width,
               label=model, color=("#1f4e79" if model == "xgboost" else "#2e7d32"),
               alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([w[:7] for w in windows], rotation=45)
    ax.set_ylabel("Excess CAGR vs SPY (pp)")
    ax.set_xlabel("Walk-forward validation window start")
    ax.set_title("Walk-forward stability: per-window excess CAGR vs SPY\n"
                  "(6 rolling 3y train → 1y val windows, locked Phase 3 hyperparameters)",
                  fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "walk_forward.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}")


def fig_ic_decomposition():
    df = pd.read_parquet(CONTRACT / "ic_decomposition.parquet")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, df["full_ic_mean"], width,
           label="Full-cross-section IC", color="#888888", alpha=0.85)
    ax.bar(x + width/2, df["top_quintile_ic_mean"], width,
           label="Top-quintile IC", color="#1f4e79", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean IC (Spearman, held-out)")
    ax.set_title("IC decomposition — full-cross-section vs top-quintile\n"
                  "(held-out evaluation on Phase 4 test+OOS window)",
                  fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    # Annotate bars with values
    for i, model in enumerate(models):
        ax.annotate(f"{df['full_ic_mean'].iloc[i]:+.4f}",
                     xy=(i - width/2, df["full_ic_mean"].iloc[i]),
                     ha="center", va=("bottom" if df["full_ic_mean"].iloc[i] >= 0 else "top"),
                     fontsize=9)
        ax.annotate(f"{df['top_quintile_ic_mean'].iloc[i]:+.4f}",
                     xy=(i + width/2, df["top_quintile_ic_mean"].iloc[i]),
                     ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "ic_decomposition.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_equity_curves()
    fig_year_by_year()
    fig_decile_returns()
    fig_alpha_attribution()
    fig_walk_forward()
    fig_ic_decomposition()
    print("DONE.")
