"""Research script: single-name dependency analysis for Trial #325.

Re-runs #325's exact config 6 times: baseline + 5 blacklists ([NVDA],
[MSFT], [META], [AVGO], [AAPL]). Each variant uses the same snapshot
(pre_v2_20260505), the same macro version (v2), and the same architecture
(regime-dependent) — only the universe_blacklist field differs.

Outputs (research artifacts — NOT promoted, NOT synced to R2):
  models/cache/dashboard_results/325_blacklist_<TICKER>/
    meta.json, portfolio.parquet, trades.parquet, holdings.json, scores.json

Plus a summary CSV at:
  models/cache/dashboard_results/325_blacklist_summary.csv

Sanity gate: the baseline (empty blacklist) must reproduce #325's saved
portfolio_value series within float-noise tolerance. If it doesn't, the
blacklist field has a side effect somewhere — stop and investigate.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scistats

REPO_ROOT = Path(__file__).resolve().parent
SNAPSHOT = REPO_ROOT / "models" / "snapshots" / "pre_v2_20260505"
os.environ["PAPER_TRADER_DATA_ROOT"]    = str(SNAPSHOT)
os.environ["PAPER_TRADER_ARCHITECTURE"] = "regime-dependent"
os.environ["PAPER_TRADER_MACRO_VERSION"] = "v2"

sys.path.insert(0, str(REPO_ROOT / "src"))

from backtest import run_backtest                                    # noqa: E402
from backtest_config import BacktestConfig                           # noqa: E402
from objective import summarize_backtest, compute_objective          # noqa: E402
from optuna_runner import _load_full_shared_data                     # noqa: E402

LABEL_325   = "best_regime_dependent_v1_20260505_2240_325"
META_325    = REPO_ROOT / "models" / "cache" / "dashboard_results" / LABEL_325 / "meta.json"
SAVED_PV    = REPO_ROOT / "models" / "cache" / "dashboard_results" / LABEL_325 / "portfolio.parquet"
SAVED_TR    = REPO_ROOT / "models" / "cache" / "dashboard_results" / LABEL_325 / "trades.parquet"
OUT_BASE    = REPO_ROOT / "models" / "cache" / "dashboard_results"
SUMMARY_CSV = OUT_BASE / "325_blacklist_summary.csv"

BLACKLISTS = [
    ("baseline", []),
    ("NVDA",     ["NVDA"]),
    ("MSFT",     ["MSFT"]),
    ("META",     ["META"]),
    ("AVGO",     ["AVGO"]),
    ("AAPL",     ["AAPL"]),
]

_TRADING_DAYS_PER_YEAR = 252


def _config_from_325_meta(blacklist: list[str]) -> BacktestConfig:
    meta = json.load(META_325.open())
    cfg = dict(meta["config"])
    cfg["universe_blacklist"] = list(blacklist)
    # weight_alt fields are derived in __post_init__ — drop them so the
    # constructor recomputes (avoids stale cached values from the saved meta).
    cfg.pop("weight_alt", None)
    cfg.pop("weight_alt_offensive", None)
    return BacktestConfig(**{
        k: v for k, v in cfg.items()
        if k in BacktestConfig.__dataclass_fields__
    })


def _capm_alpha_beta(strat_pv: pd.Series, spy_close: pd.Series
                     ) -> tuple[float, float]:
    """Single-window CAPM alpha (annualized) + beta from daily returns.

    Mirrors rolling_metrics.compute_rolling_alpha's CAPM branch but on a
    single window covering the entire portfolio_df range. Used for the
    headline 'validation_alpha_capm_pp' summary column.
    """
    s = strat_pv.pct_change().dropna()
    spy = spy_close.loc[(spy_close.index >= s.index[0])
                         & (spy_close.index <= s.index[-1])]
    b = spy.pct_change().dropna()
    common = s.index.intersection(b.index)
    if len(common) < 20 or b.loc[common].std() == 0:
        return float("nan"), float("nan")
    slope, intercept, *_ = scistats.linregress(b.loc[common].values,
                                                s.loc[common].values)
    alpha_ann = float(intercept) * _TRADING_DAYS_PER_YEAR
    return alpha_ann, float(slope)


def _max_drawdown(values: pd.Series) -> float:
    return float(((values / values.cummax()) - 1).min())


def _run_one(label_suffix: str, blacklist: list[str],
             shared: dict, baseline_trades_df: pd.DataFrame | None
             ) -> dict:
    cfg = _config_from_325_meta(blacklist)
    out_dir = OUT_BASE / f"325_blacklist_{label_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running blacklist={blacklist!r} -> {out_dir.name} ===")
    t0 = time.perf_counter()
    portfolio_df, trades_df, scores, holdings = run_backtest(
        shared["featured_data"], shared["price_data"],
        split_date=cfg.validate_start,
        fund_data=shared["fund_data"],
        sector_map=shared["sector_map"],
        earnings_dates=shared["earnings_dates"],
        model=shared["model"],
        config=cfg,
        market_data=shared.get("market_data"),
        compute_rolling_metrics=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s — {len(portfolio_df)} days, "
          f"{len(trades_df)} trades")

    # Strip sizing_decisions from attrs before parquet write (same fix as
    # optuna_runner — pyarrow JSON-serializes attrs and Timestamps choke).
    portfolio_df.attrs.pop("sizing_decisions", None)

    # Save the same artifact set as save_hypothesis_result, minus the
    # benchmark close parquets (we don't promote these so the dashboard
    # never reads them; SPY/QQQ are already in shared if anyone needs).
    portfolio_df.to_parquet(out_dir / "portfolio.parquet")
    if not trades_df.empty:
        trades_df.to_parquet(out_dir / "trades.parquet")
    else:
        pd.DataFrame(columns=["date","ticker","action","shares","price","fee"]
                     ).to_parquet(out_dir / "trades.parquet")
    (out_dir / "holdings.json").write_text(json.dumps(holdings, indent=2,
                                                       default=str))
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2,
                                                     default=str))

    # Metrics ---------------------------------------------------------
    spy_close = shared["spy_close"]
    summary = summarize_backtest(portfolio_df, spy_close)
    objective_score = compute_objective(summary)  # the locked alpha-DD score
    arith_alpha_pp = (summary["strategy_annualized_return"]
                      - summary["spy_annualized_return"]) * 100
    capm_alpha, beta = _capm_alpha_beta(portfolio_df["portfolio_value"],
                                         spy_close)
    capm_alpha_pp = capm_alpha * 100 if not np.isnan(capm_alpha) else float("nan")
    total_return_pct = float(portfolio_df["portfolio_value"].iloc[-1]
                              / portfolio_df["portfolio_value"].iloc[0] - 1) * 100
    max_dd_pct = _max_drawdown(portfolio_df["portfolio_value"]) * 100
    n_trades = int(len(trades_df))

    rolling_12mo = (portfolio_df.attrs.get("rolling_metrics") or {}
                    ).get("rolling_12mo") or {}
    r12_obj = float(rolling_12mo.get("objective_score") or float("nan"))

    # How many of #325's ORIGINAL trades touched the blacklisted name?
    if blacklist and baseline_trades_df is not None:
        n_baseline_touching = int(
            baseline_trades_df["ticker"].isin(blacklist).sum())
    else:
        n_baseline_touching = 0

    # Top-5 holdings at validation end (by current_market_value).
    if isinstance(holdings, dict) and holdings:
        # holdings is dict[ticker -> {shares, entry_price, ...}]
        # We don't have current_market_value at the holdings level, but
        # the dict order is insertion-order. For "top 5 by value" we need
        # current price × shares. Use price_data for the last available date.
        last_dt = portfolio_df.index[-1]
        rows = []
        for tkr, h in holdings.items():
            shr = float(h.get("shares") or 0)
            pdf = shared["price_data"].get(tkr)
            if pdf is None or pdf.empty: continue
            # Closest <= last_dt
            avail = pdf.index[pdf.index <= last_dt]
            if len(avail) == 0: continue
            px = float(pdf.loc[avail[-1], "Close"])
            rows.append((tkr, shr * px))
        rows.sort(key=lambda r: r[1], reverse=True)
        top5 = ",".join(t for t, _ in rows[:5])
    else:
        top5 = ""

    # Write meta.json ------------------------------------------------
    meta_out = {
        "label":              out_dir.name,
        "research_artifact":  True,
        "promoted":           False,
        "based_on":           LABEL_325,
        "universe_blacklist": list(blacklist),
        "split_date":         cfg.validate_start,
        "n_days":             int(len(portfolio_df)),
        "n_trades":           n_trades,
        "n_holdings":         int(len(holdings)) if holdings else 0,
        "runtime_seconds":    round(elapsed, 2),
        "config":             cfg.to_dict(),
        "summary": {
            "strategy_total_return":      total_return_pct / 100,
            "strategy_annualized_return": summary["strategy_annualized_return"],
            "spy_annualized_return":      summary["spy_annualized_return"],
            "alpha_annualized":           arith_alpha_pp / 100,
            "alpha_capm_annualized":      capm_alpha,
            "beta":                       beta,
            "max_drawdown":               max_dd_pct / 100,
            "rolling_12mo_objective":     r12_obj,
            "objective_score":            objective_score,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2,
                                                    default=str))

    print(f"  alpha (arith) = {arith_alpha_pp:+.2f}pp/yr   "
          f"alpha (CAPM) = {capm_alpha_pp:+.2f}pp/yr   "
          f"beta = {beta:.3f}   "
          f"r12_obj = {r12_obj:.4f}")

    return {
        "blacklisted_ticker":            label_suffix,
        "validation_rolling_12mo_objective": r12_obj,
        "validation_alpha_annualized_pp":    arith_alpha_pp,
        "validation_alpha_capm_pp":          capm_alpha_pp,
        "validation_total_return_pct":       total_return_pct,
        "validation_max_drawdown_pct":       max_dd_pct,
        "validation_n_trades":               n_trades,
        "n_trades_involving_blacklisted_ticker_in_baseline": n_baseline_touching,
        "top_5_holdings_at_validation_end":  top5,
        "validation_beta":                   beta,
        "validation_objective_score":        objective_score,
        "_portfolio_df":                     portfolio_df,
        "_trades_df":                        trades_df,
    }


def _verify_baseline_matches_325(portfolio_df: pd.DataFrame,
                                  trades_df: pd.DataFrame) -> None:
    """Sanity gate — empty blacklist must reproduce #325's saved artifacts
    within float-noise. If max drift > $1, the blacklist filter has a
    side effect (e.g. accidental re-ordering) and we should stop."""
    if not SAVED_PV.exists():
        print("  WARN — saved #325 portfolio.parquet not found; skipping gate.")
        return
    saved = pd.read_parquet(SAVED_PV)["portfolio_value"]
    fresh = portfolio_df["portfolio_value"]
    n_diff = int((saved.round(2) != fresh.round(2)).sum())
    max_diff = float((saved - fresh).abs().max())
    print(f"  baseline gate: portfolio_value diffs (2dp): {n_diff} / "
          f"{len(saved)};  max abs drift: ${max_diff:,.4f}")
    if max_diff > 1.0:
        raise SystemExit(
            f"  STOP — baseline run drifts ${max_diff:.2f} from saved #325. "
            f"The empty-blacklist code path has a side effect; investigate "
            f"before running blacklist variants.")
    if SAVED_TR.exists():
        saved_n = len(pd.read_parquet(SAVED_TR))
        fresh_n = len(trades_df)
        print(f"  trade count: saved={saved_n}, fresh={fresh_n}")


def main() -> int:
    print("[BLACKLIST] Loading shared data once...")
    shared = _load_full_shared_data()

    results: list[dict] = []
    baseline_trades_df: pd.DataFrame | None = None

    for suffix, bl in BLACKLISTS:
        r = _run_one(suffix, bl, shared, baseline_trades_df)
        if suffix == "baseline":
            _verify_baseline_matches_325(r["_portfolio_df"], r["_trades_df"])
            baseline_trades_df = r["_trades_df"]
        results.append(r)

    # Summary CSV (drop the in-memory df handles before writing)
    df_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        df_rows.append(row)
    summary_df = pd.DataFrame(df_rows, columns=[
        "blacklisted_ticker",
        "validation_rolling_12mo_objective",
        "validation_alpha_annualized_pp",
        "validation_alpha_capm_pp",
        "validation_total_return_pct",
        "validation_max_drawdown_pct",
        "validation_n_trades",
        "validation_beta",
        "validation_objective_score",
        "n_trades_involving_blacklisted_ticker_in_baseline",
        "top_5_holdings_at_validation_end",
    ])
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[BLACKLIST] Wrote summary CSV: {SUMMARY_CSV}")

    # ---- Side-by-side comparison table ----
    base = next(r for r in results if r["blacklisted_ticker"] == "baseline")
    print("\n" + "=" * 130)
    print(f"{'Variant':<10} {'r12_obj':>9} {'α arith pp':>11} {'α CAPM pp':>10} "
          f"{'β':>6} {'tot ret%':>9} {'max DD%':>8} {'#trades':>7} "
          f"{'#trades→BL in baseline':>23}")
    print("-" * 130)
    for r in results:
        v = r["blacklisted_ticker"]
        flag = ""
        if v != "baseline":
            d_alpha = (r["validation_alpha_annualized_pp"]
                       - base["validation_alpha_annualized_pp"])
            if d_alpha < -20:
                flag = "  <<< α drops >20pp"
        print(f"{v:<10} "
              f"{r['validation_rolling_12mo_objective']:>9.4f} "
              f"{r['validation_alpha_annualized_pp']:>+11.2f} "
              f"{r['validation_alpha_capm_pp']:>+10.2f} "
              f"{r['validation_beta']:>6.3f} "
              f"{r['validation_total_return_pct']:>+9.2f} "
              f"{r['validation_max_drawdown_pct']:>+8.2f} "
              f"{r['validation_n_trades']:>7d} "
              f"{r['n_trades_involving_blacklisted_ticker_in_baseline']:>23d}"
              f"{flag}")
    print("=" * 130)
    print()
    print("Top-5 holdings at validation end:")
    for r in results:
        print(f"  {r['blacklisted_ticker']:<10} {r['top_5_holdings_at_validation_end']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
