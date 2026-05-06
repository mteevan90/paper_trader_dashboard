"""benchmarks.py — benchmark return loaders for the rolling-metrics
evaluation framework.

Phase 1: only "SPY" is implemented. The function signature and module
docstring document the design considerations for "prior_model" so the
extension is well-scoped when phase 5+ wires it up.

prior_model design considerations (Section 2.3 of evaluation_framework_design):
    - "prior model" = the previously-locked production model (e.g., the
      Phase 0 trial #706 config running on the same training-window data).
    - Benchmark returns = the prior model's daily portfolio returns.
    - Coverage gaps: if the prior model has no portfolio_value at a given
      date (e.g., universe expansion before that model existed), the
      benchmark's daily return is NaN; rolling-window code skips windows
      where >50% of days are NaN (or some similar coverage threshold).
    - Versioning: pin which prior model is being benchmarked against via
      a snapshot reference in meta.json. Re-baselining happens explicitly
      (rescore_baseline.py against a new snapshot).
    - Cold-start: when there is no prior model (initial development),
      "prior_model" benchmark falls back to SPY.
"""

import pandas as pd


def get_benchmark_returns(
    benchmark_name: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    market_data: dict | None = None,
) -> pd.Series:
    """Return daily returns for the named benchmark over [start, end].

    Parameters
    ----------
    benchmark_name : "SPY" (phase 1). "prior_model" raises NotImplementedError.
    start_date, end_date : inclusive bounds (str ISO date or Timestamp).
    market_data : optional dict[str, pd.DataFrame] with "Close" column,
        as produced by fetch_data.get_stock_data_cached. If supplied and
        the benchmark is a ticker we already loaded, reuse the in-memory
        data instead of going to disk.

    Returns
    -------
    pd.Series of daily returns indexed by date, sliced to [start, end].
    Dropped NaN at the leading edge (pct_change leaves first row NaN).
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if benchmark_name == "SPY":
        if market_data is not None and "SPY" in market_data:
            close = market_data["SPY"]["Close"]
        else:
            # Fallback for callers that don't already have SPY in memory.
            # Routes through the same cache the backtest uses, so behavior
            # is consistent with the rest of the system.
            from fetch_data import get_stock_data_cached
            from feature_cache import PRICE_CACHE_DIR
            md = get_stock_data_cached(["SPY"], str(start_ts.date()),
                                       str(end_ts.date()),
                                       cache_dir=PRICE_CACHE_DIR)
            if "SPY" not in md or md["SPY"].empty:
                raise RuntimeError(
                    f"benchmarks.get_benchmark_returns: SPY data unavailable "
                    f"for [{start_ts.date()}, {end_ts.date()}]")
            close = md["SPY"]["Close"]

        returns = close.pct_change().dropna()
        mask = (returns.index >= start_ts) & (returns.index <= end_ts)
        return returns[mask]

    if benchmark_name == "prior_model":
        raise NotImplementedError(
            "benchmark_name='prior_model' is reserved for phase 5+. "
            "See benchmarks.py module docstring for the design "
            "considerations (Section 2.3 of evaluation_framework_design).")

    raise ValueError(f"Unknown benchmark: {benchmark_name!r}. "
                     f"Supported: 'SPY' (phase 1), 'prior_model' "
                     f"(phase 5+ reserved).")
