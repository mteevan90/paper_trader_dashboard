"""alt_signals.py — Alt bucket aggregation slot.

The composite score is currently
    composite = 0.35 * fund + 0.25 * tech + 0.25 * model + 0.15 * alt
              (+ 0.05 * analyst tiebreaker, applied post-composite)

The ``alt`` bucket is an aggregation slot: an equal-weight average of
all alt signals registered below. Each alt signal must produce a
``[0, 1]`` score per ticker per date, matching the existing scoring
convention (1 = bullish, 0 = bearish, 0.5 = neutral).

Adding a new alt signal requires exactly two edits, both in this file:
    1. Define a function ``score_<signal>(tickers, date) -> dict[str, float]``
    2. Append ``("<signal>", score_<signal>)`` to ``ALT_SIGNAL_REGISTRY``

Nothing else downstream changes — backtest.py, optuna_runner.py, the
dashboard, and the Optuna search space all consume the bucket via
``score_alt_bucket()`` and the existing ``weight_alt`` config field.

TODO(Group D): planned alt signals
  * Finnhub EPS revisions (analyst estimate revisions, surprise rates)
  * OpenInsider cluster buys (insider purchase clustering scoring)
  * Quiver congressional trades + government contract awards
"""

from typing import Callable

from finnhub_insider_signals import score_finnhub_insider_clusters

# Registry of (name, score_function) tuples. Group D signals append here.
# When empty, score_alt_bucket() returns 0.5 for every ticker (neutral)
# so the 0.15 alt weight contributes a constant offset to every composite
# (doesn't affect ranking).
#
# Segment 14: tried OpenInsider HTML scraping; openinsider.com's
# screener silently ignores the date-range parameter (CDN-cached recent-
# activity view returned for every query). Pivoted to Finnhub's
# /stock/insider-transactions bulk endpoint, which has multi-year free-
# tier history. Trade-off: Finnhub doesn't return insider role/title,
# so this is "all-insider clustering" not "senior-only clustering".
# See models/cache/finnhub_insider_signal_limitations.md for the
# expected academic-strength impact.
ALT_SIGNAL_REGISTRY: list[tuple[str, Callable[[list[str], object], dict[str, float]]]] = [
    ("finnhub_insider_clusters", score_finnhub_insider_clusters),
]


def score_alt_bucket(tickers: list[str], date) -> dict[str, float]:
    """Aggregate every registered alt signal into a single per-ticker score.

    Equal-weighted mean across all signals that produce a value for the
    ticker. A ticker missing from one signal but present in another gets
    the mean of the present ones; a ticker missing from every signal
    gets 0.5 (neutral). When the registry is empty, every ticker gets
    0.5 — deterministic and bit-identical across runs.
    """
    if not ALT_SIGNAL_REGISTRY:
        return {t: 0.5 for t in tickers}

    accumulator: dict[str, list[float]] = {t: [] for t in tickers}
    for name, fn in ALT_SIGNAL_REGISTRY:
        try:
            scores = fn(tickers, date)
        except Exception as e:
            # An alt signal that crashes shouldn't take down the
            # composite — note it and move on. Other signals + neutral
            # default still produce a valid score.
            print(f"  [ALT] signal {name!r} failed at {date}: {e}")
            continue
        for t in tickers:
            v = scores.get(t)
            if v is None:
                continue
            try:
                accumulator[t].append(float(v))
            except (TypeError, ValueError):
                continue

    out: dict[str, float] = {}
    for t in tickers:
        vs = accumulator[t]
        out[t] = (sum(vs) / len(vs)) if vs else 0.5
    return out
