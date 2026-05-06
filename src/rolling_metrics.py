"""rolling_metrics.py — rolling-window evaluation framework.

Replaces endpoint-sensitive single-period alpha with a distribution of
rolling-window alphas. Used by both the Optuna training objective and
the validation graduation gate, eliminating training/validation
metric misalignment.

Core metric: CAPM alpha via per-window OLS regression of strategy
returns vs benchmark returns, annualized.

Objective function (locked):
    score = p75(rolling_12mo_alpha) - 0.5 * max(0, -p25(rolling_12mo_alpha))

The penalty term only fires when p25 is negative (strategy's worst-
quartile windows lose vs benchmark). p75 rewards consistency at the top.

Window construction:
    - Primary: 12-month window
    - Secondary: 6-month window (diagnostic)
    - Step: monthly (windows start on first business day of each month)
    - "First complete window only" — no partial-window backfill at start

Diagnostics:
    - up_capture / down_capture / trending_capture (>+2% benchmark months)
    - Recovery capture + time-to-recovery at -5%, -10%, -15% drawdown
    - Failed-recovery counter
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


_TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Rolling alpha
# ---------------------------------------------------------------------------

def _month_starts(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return the first index entry of each calendar month within `index`.

    Used to step monthly through the data: the first business day of each
    month becomes a candidate window start.
    """
    if len(index) == 0:
        return []
    months = pd.date_range(start=index[0].normalize().replace(day=1),
                           end=index[-1].normalize(), freq="MS")
    out: list[pd.Timestamp] = []
    for m in months:
        candidates = index[index >= m]
        if len(candidates) > 0:
            out.append(candidates[0])
    return out


def compute_rolling_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window_months: int = 12,
    step: str = "monthly",
    method: str = "capm",
) -> pd.DataFrame:
    """Compute rolling-window alpha across the joint return series.

    Parameters
    ----------
    strategy_returns : daily returns indexed by date
    benchmark_returns : daily returns indexed by date
    window_months : 12 (primary) or 6 (secondary diagnostic)
    step : currently only "monthly" supported
    method : "capm" (regression intercept) or "simple" (mean diff)

    Returns
    -------
    DataFrame with one row per complete window, columns:
        window_start, window_end, alpha (annualized), beta (capm only,
        else None), n_days
    Empty DataFrame if not enough data for at least one full window.
    """
    if step != "monthly":
        raise ValueError(f"Only monthly step supported; got {step!r}")
    if method not in ("capm", "simple"):
        raise ValueError(f"method must be 'capm' or 'simple'; got {method!r}")

    common = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common) == 0:
        return pd.DataFrame(columns=["window_start", "window_end",
                                     "alpha", "beta", "n_days"])
    s = strategy_returns.loc[common].dropna()
    b = benchmark_returns.loc[common].dropna()
    common = s.index.intersection(b.index)
    s = s.loc[common]
    b = b.loc[common]

    starts = _month_starts(s.index)
    rows: list[dict] = []
    for ws in starts:
        we = ws + pd.DateOffset(months=window_months) - pd.Timedelta(days=1)
        if we > s.index[-1]:
            break  # incomplete window — stop (first complete window only)
        s_win = s.loc[ws:we]
        b_win = b.loc[ws:we]
        common_win = s_win.index.intersection(b_win.index)
        if len(common_win) < 20:
            continue  # too sparse to compute reliably
        s_arr = s_win.loc[common_win].values
        b_arr = b_win.loc[common_win].values

        if method == "capm":
            if b_arr.std() == 0 or np.isnan(b_arr.std()):
                continue
            slope, intercept, _r, _p, _se = stats.linregress(b_arr, s_arr)
            alpha = float(intercept) * _TRADING_DAYS_PER_YEAR
            beta = float(slope)
        else:  # simple
            alpha = (float(s_arr.mean()) - float(b_arr.mean())) * _TRADING_DAYS_PER_YEAR
            beta = None

        rows.append({
            "window_start": ws,
            "window_end":   we,
            "alpha":        alpha,
            "beta":         beta,
            "n_days":       int(len(common_win)),
        })

    return pd.DataFrame(rows)


def compute_alpha_distribution_stats(rolling_df: pd.DataFrame) -> dict:
    """Distribution stats over the rolling-window alpha series."""
    if rolling_df.empty or "alpha" not in rolling_df.columns:
        return {
            "p10": None, "p25": None, "median": None, "p75": None, "p90": None,
            "mean": None, "std": None,
            "count_positive": 0, "count_total": 0,
        }
    a = rolling_df["alpha"].dropna()
    if len(a) == 0:
        return {
            "p10": None, "p25": None, "median": None, "p75": None, "p90": None,
            "mean": None, "std": None,
            "count_positive": 0, "count_total": 0,
        }
    return {
        "p10":            float(np.percentile(a, 10)),
        "p25":            float(np.percentile(a, 25)),
        "median":         float(np.percentile(a, 50)),
        "p75":            float(np.percentile(a, 75)),
        "p90":            float(np.percentile(a, 90)),
        "mean":           float(a.mean()),
        "std":            float(a.std()),
        "count_positive": int((a > 0).sum()),
        "count_total":    int(len(a)),
    }


def compute_objective_score(rolling_df: pd.DataFrame) -> float:
    """Locked objective: p75(alpha) − 0.5 * max(0, −p25(alpha)).

    Returns sentinel −1e6 if the rolling DataFrame is empty (no complete
    windows). Otherwise the unbounded score (in annualized-alpha units).
    """
    if rolling_df.empty or "alpha" not in rolling_df.columns:
        return -1e6
    a = rolling_df["alpha"].dropna()
    if len(a) == 0:
        return -1e6
    p75 = float(np.percentile(a, 75))
    p25 = float(np.percentile(a, 25))
    penalty = 0.5 * max(0.0, -p25)
    return p75 - penalty


# ---------------------------------------------------------------------------
# Capture diagnostics
# ---------------------------------------------------------------------------

def _aggregate_monthly(daily_returns: pd.Series) -> pd.Series:
    """Compound daily returns into month-end totals."""
    if daily_returns.empty:
        return daily_returns
    return (1 + daily_returns).resample("ME").prod() - 1


def compute_capture_ratios(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """Standard up/down capture ratios plus trending-month capture.

    Aggregates daily returns to monthly first (industry convention).
    Returns percentages (e.g. up_capture=110 means 10% above benchmark in
    up months). None if the corresponding regime never appeared.
    """
    common = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common) == 0:
        return {"up_capture": None, "down_capture": None,
                "trending_capture": None,
                "n_up_months": 0, "n_down_months": 0,
                "n_trending_months": 0}
    s_m = _aggregate_monthly(strategy_returns.loc[common])
    b_m = _aggregate_monthly(benchmark_returns.loc[common])
    common_m = s_m.index.intersection(b_m.index)
    s_m = s_m.loc[common_m]
    b_m = b_m.loc[common_m]

    def _ratio(mask: pd.Series) -> float | None:
        if mask.sum() == 0:
            return None
        s_total = (1 + s_m[mask]).prod() - 1
        b_total = (1 + b_m[mask]).prod() - 1
        if abs(b_total) < 1e-12:
            return None
        return float(s_total / b_total * 100.0)

    return {
        "up_capture":        _ratio(b_m > 0),
        "down_capture":      _ratio(b_m < 0),
        "trending_capture":  _ratio(b_m > 0.02),
        "n_up_months":       int((b_m > 0).sum()),
        "n_down_months":     int((b_m < 0).sum()),
        "n_trending_months": int((b_m > 0.02).sum()),
    }


# ---------------------------------------------------------------------------
# Drawdown / recovery diagnostics
# ---------------------------------------------------------------------------

def _identify_drawdown_events(
    cum_levels: pd.Series, threshold: float
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]]:
    """Identify events where benchmark cum-level fell below `threshold`
    relative to a recent high, then either recovered (returned to that
    high) or never recovered by series end.

    Returns list of (peak_date, bottom_date, recovery_date_or_None).
    """
    if cum_levels.empty:
        return []
    peak = cum_levels.cummax()
    dd = cum_levels / peak - 1.0  # always <= 0
    events = []
    in_drawdown = False
    peak_value = None
    peak_date = None
    bottom_date = None

    for date in cum_levels.index:
        cur = cum_levels.loc[date]
        cur_dd = dd.loc[date]
        if not in_drawdown:
            if cur_dd <= threshold:
                # Drawdown event opens here
                peak_value = peak.loc[date]
                prior = cum_levels.loc[:date]
                # Last date the running max was achieved
                eq = prior[prior >= peak_value * (1 - 1e-12)]
                peak_date = eq.index[-1] if len(eq) else date
                bottom_date = date
                in_drawdown = True
        else:
            # Update bottom while still down
            if cur < cum_levels.loc[bottom_date]:
                bottom_date = date
            # Recovery: cum returns to peak_value
            if cur >= peak_value:
                events.append((peak_date, bottom_date, date))
                in_drawdown = False
                peak_value = None
                peak_date = None
                bottom_date = None

    if in_drawdown:
        events.append((peak_date, bottom_date, None))
    return events


def compute_recovery_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    drawdown_thresholds: list[float] | None = None,
) -> dict:
    """For each drawdown threshold, identify benchmark drawdown events and
    measure strategy's behavior across them.

    For each threshold returns:
        events_count: number of drawdown events identified
        recovery_capture_avg: mean strategy-return / benchmark-return ratio
            from drawdown bottom to recovery date (percentage; 100 = matched)
        time_to_recovery_ratio_avg: mean (strategy_TTR / benchmark_TTR);
            < 1.0 means strategy recovered faster than benchmark
        failed_recoveries: events where benchmark didn't recover by end
    """
    if drawdown_thresholds is None:
        drawdown_thresholds = [-0.05, -0.10, -0.15]

    common = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common) == 0:
        return {float(t): {"events_count": 0, "recovery_capture_avg": None,
                           "time_to_recovery_ratio_avg": None,
                           "failed_recoveries": 0}
                for t in drawdown_thresholds}

    s = strategy_returns.loc[common]
    b = benchmark_returns.loc[common]
    s_cum = (1 + s).cumprod()
    b_cum = (1 + b).cumprod()

    out: dict = {}
    for threshold in drawdown_thresholds:
        events = _identify_drawdown_events(b_cum, threshold)
        recovery_caps: list[float] = []
        ttr_ratios: list[float] = []
        failed = 0

        for peak_d, bottom_d, rec_d in events:
            if rec_d is None:
                failed += 1
                continue
            try:
                s_ret = float(s_cum.loc[rec_d]) / float(s_cum.loc[bottom_d]) - 1.0
                b_ret = float(b_cum.loc[rec_d]) / float(b_cum.loc[bottom_d]) - 1.0
            except KeyError:
                continue
            if abs(b_ret) < 1e-12:
                continue
            recovery_caps.append(s_ret / b_ret * 100.0)

            # Strategy's own time-to-recovery: from bottom_d to first date
            # strategy returns to its OWN pre-drawdown peak.
            try:
                s_peak_pre = float(s_cum.loc[:bottom_d].max())
                after = s_cum.loc[bottom_d:]
                eq = after[after >= s_peak_pre * (1 - 1e-12)]
                if len(eq) > 0:
                    s_rec_d = eq.index[0]
                    bench_ttr = (rec_d - bottom_d).days
                    strat_ttr = (s_rec_d - bottom_d).days
                    if bench_ttr > 0:
                        ttr_ratios.append(strat_ttr / bench_ttr)
            except Exception:
                continue

        out[float(threshold)] = {
            "events_count":               len(events),
            "recovery_capture_avg":       (float(np.mean(recovery_caps))
                                           if recovery_caps else None),
            "time_to_recovery_ratio_avg": (float(np.mean(ttr_ratios))
                                           if ttr_ratios else None),
            "failed_recoveries":          int(failed),
        }
    return out


# ---------------------------------------------------------------------------
# Top-level convenience: full rolling-metrics bundle
# ---------------------------------------------------------------------------

def compute_full_rolling_bundle(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """Compute the full bundle that gets stored under meta['rolling_metrics'].

    Bundle shape:
        rolling_12mo:  {alpha_distribution_stats, objective_score,
                        capm_windows: [...], simple_windows: [...]}
        rolling_6mo:   same shape
        capture:       {up_capture, down_capture, trending_capture, ...}
        recovery:      {-0.05: {...}, -0.10: {...}, -0.15: {...}}
    """
    bundle: dict = {}

    for w_months, key in [(12, "rolling_12mo"), (6, "rolling_6mo")]:
        capm = compute_rolling_alpha(strategy_returns, benchmark_returns,
                                     window_months=w_months, method="capm")
        simple = compute_rolling_alpha(strategy_returns, benchmark_returns,
                                       window_months=w_months, method="simple")
        bundle[key] = {
            "alpha_distribution_stats": compute_alpha_distribution_stats(capm),
            "objective_score":          compute_objective_score(capm),
            "capm_windows":             capm.assign(
                window_start=capm["window_start"].astype(str) if not capm.empty else [],
                window_end=capm["window_end"].astype(str) if not capm.empty else [],
            ).to_dict(orient="records") if not capm.empty else [],
            "simple_windows":           simple.assign(
                window_start=simple["window_start"].astype(str) if not simple.empty else [],
                window_end=simple["window_end"].astype(str) if not simple.empty else [],
            ).to_dict(orient="records") if not simple.empty else [],
        }

    bundle["capture"]  = compute_capture_ratios(strategy_returns, benchmark_returns)
    bundle["recovery"] = compute_recovery_metrics(strategy_returns, benchmark_returns)
    return bundle
