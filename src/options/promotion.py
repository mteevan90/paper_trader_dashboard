"""Automated promotion gate (Phase 2 Section 8).

Five hardcoded checks codify the v1 promotion discipline:

1. **overfit_check** — val Calmar ≥ 0.5 × train Calmar
2. **beats_spy** — val Calmar > SPY Calmar over the val window
3. **beats_bxm** — val Calmar > BXM Calmar over the val window (CC only;
   auto-pass for CSP)
4. **no_underlying_concentration** — no underlying ablation has
   ``pct_alpha_attribution > 0.5``
5. **regime_independence** — high-IV-excluded ablation Calmar within 2x
   of low-IV-excluded (i.e., ratio ≥ 0.5)

Aggregation rule: 5/5 passes → ``promote``, 4/5 → ``borderline``,
≤3/5 → ``do_not_promote``. The CLI prompts for a human override and
re-writes ``promotion_decision.json`` with both the automated
recommendation and the human's final decision + reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.options.concentration import ConcentrationResult
from src.options.engine import StudyResults
from src.options.optuna_runner import (
    OptunaStudyResults,
    ZERO_DD_POSITIVE_RETURN_SENTINEL,
)


__all__ = [
    "PromotionCheck",
    "PromotionRecommendation",
    "evaluate_promotion",
    "write_promotion_decision",
    "calmar_from_series",
    "OVERFIT_RATIO_THRESHOLD",
    "MAX_UNDERLYING_ATTRIBUTION",
    "REGIME_RATIO_THRESHOLD",
]


OVERFIT_RATIO_THRESHOLD: float = 0.5
MAX_UNDERLYING_ATTRIBUTION: float = 0.5
REGIME_RATIO_THRESHOLD: float = 0.5  # within 2x ↔ ratio ≥ 0.5


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    """One criterion's pass/fail outcome with explanation."""

    criterion_name: str
    passed: bool
    expected: str
    actual: str
    explanation: str


@dataclass(frozen=True, slots=True)
class PromotionRecommendation:
    """Aggregated recommendation across all checks."""

    automated_recommendation: str
    checks: tuple[PromotionCheck, ...]
    train_calmar: float
    val_calmar: float
    spy_calmar_on_val: float
    bxm_calmar_on_val: Optional[float]
    summary: str

    def to_dict(self) -> dict:
        return {
            "automated_recommendation": self.automated_recommendation,
            "checks": [
                {
                    "criterion_name": c.criterion_name,
                    "passed": c.passed,
                    "expected": c.expected,
                    "actual": c.actual,
                    "explanation": c.explanation,
                }
                for c in self.checks
            ],
            "train_calmar": self.train_calmar,
            "val_calmar": self.val_calmar,
            "spy_calmar_on_val": self.spy_calmar_on_val,
            "bxm_calmar_on_val": self.bxm_calmar_on_val,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PromotionRecommendation":
        return cls(
            automated_recommendation=str(data["automated_recommendation"]),
            checks=tuple(
                PromotionCheck(
                    criterion_name=str(c["criterion_name"]),
                    passed=bool(c["passed"]),
                    expected=str(c["expected"]),
                    actual=str(c["actual"]),
                    explanation=str(c["explanation"]),
                )
                for c in data["checks"]
            ),
            train_calmar=float(data["train_calmar"]),
            val_calmar=float(data["val_calmar"]),
            spy_calmar_on_val=float(data["spy_calmar_on_val"]),
            bxm_calmar_on_val=(
                float(data["bxm_calmar_on_val"])
                if data.get("bxm_calmar_on_val") is not None
                else None
            ),
            summary=str(data["summary"]),
        )


# ----------------- Calmar over arbitrary value series -----------------


def calmar_from_series(values: pd.Series) -> float:
    """Calmar ratio for an arbitrary value series indexed by date.

    Mirrors :func:`calmar_objective` but operates on a generic
    pandas Series. Used for benchmark Calmar (SPY total_return_index,
    BXM close).

    Same edge-case treatment as :func:`calmar_objective`: <30 points
    → 0.0, initial ≤ 0 → 0.0, zero drawdown with positive return →
    ``ZERO_DD_POSITIVE_RETURN_SENTINEL``, complete wipeout → -1.0.
    """
    if values is None or len(values) < 30:
        return 0.0
    initial = float(values.iloc[0])
    final = float(values.iloc[-1])
    if initial <= 0:
        return 0.0
    days_idx = values.index
    try:
        first_date = days_idx[0]
        last_date = days_idx[-1]
        if hasattr(first_date, "date"):
            first_date = first_date.date()
        if hasattr(last_date, "date"):
            last_date = last_date.date()
        days = (last_date - first_date).days
    except Exception:
        days = len(values)
    years = days / 365.25 if days > 0 else 0.0
    if years <= 0:
        return 0.0
    if final <= 0:
        compound_return = -1.0
    else:
        compound_return = (final / initial) ** (1.0 / years) - 1.0

    peak = initial
    max_dd = 0.0
    for v in values:
        v = float(v)
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    if max_dd == 0.0:
        if compound_return > 0:
            return ZERO_DD_POSITIVE_RETURN_SENTINEL
        return compound_return
    return compound_return / max_dd


def _calmar_from_snapshots(snapshots) -> float:
    """Calmar from a list of DailySnapshot — uses portfolio_total."""
    if not snapshots:
        return 0.0
    series = pd.Series(
        [s.portfolio_total for s in snapshots],
        index=[s.sim_date for s in snapshots],
    )
    return calmar_from_series(series)


def _filter_by_date(
    series: pd.Series, start: date, end: date,
) -> pd.Series:
    """Return ``series`` restricted to dates in ``[start, end]``."""
    out = []
    out_index = []
    for idx, value in series.items():
        d = idx.date() if hasattr(idx, "date") else idx
        if start <= d <= end:
            out.append(value)
            out_index.append(d)
    return pd.Series(out, index=out_index)


# ----------------- check evaluators -----------------


def _check_regime_ratio(
    concentration_results: tuple[ConcentrationResult, ...],
) -> tuple[bool, str, str]:
    """Returns ``(passed, expected, actual)`` for the regime-independence
    check."""
    high = [
        r for r in concentration_results
        if r.ablation_dimension == "iv_regime"
        and r.ablation_value == "high"
    ]
    low = [
        r for r in concentration_results
        if r.ablation_dimension == "iv_regime"
        and r.ablation_value == "low"
    ]
    if not high or not low:
        return (
            False,
            "high and low IV-regime ablations both present",
            f"high={len(high)}, low={len(low)}",
        )
    high_calmar = high[0].ablated_calmar
    low_calmar = low[0].ablated_calmar
    abs_high = abs(high_calmar)
    abs_low = abs(low_calmar)
    max_abs = max(abs_high, abs_low)
    if max_abs == 0.0:
        return (
            True,
            "ratio undefined when both Calmars are 0 — auto-pass",
            "high=0, low=0",
        )
    ratio = min(abs_high, abs_low) / max_abs
    passed = ratio >= REGIME_RATIO_THRESHOLD
    expected = f"min/max ratio >= {REGIME_RATIO_THRESHOLD}"
    actual = (
        f"high_excluded_calmar={high_calmar:.4f}, "
        f"low_excluded_calmar={low_calmar:.4f}, ratio={ratio:.4f}"
    )
    return passed, expected, actual


def _check_overfit(train_calmar: float, val_calmar: float) -> tuple[bool, str, str]:
    expected = f"val_calmar >= {OVERFIT_RATIO_THRESHOLD} × train_calmar"
    threshold = OVERFIT_RATIO_THRESHOLD * train_calmar
    actual = (
        f"train={train_calmar:.4f}, val={val_calmar:.4f}, "
        f"threshold={threshold:.4f}"
    )
    return val_calmar >= threshold, expected, actual


def _check_beats_spy(
    val_calmar: float, spy_calmar: float,
) -> tuple[bool, str, str]:
    expected = "val_calmar > SPY_calmar (val window)"
    actual = f"val={val_calmar:.4f}, spy={spy_calmar:.4f}"
    return val_calmar > spy_calmar, expected, actual


def _check_beats_bxm(
    strategy_class: str,
    val_calmar: float,
    bxm_calmar: Optional[float],
) -> tuple[bool, str, str]:
    if strategy_class != "covered_call":
        return (
            True,
            "skipped (CSP studies don't compare to BXM)",
            "auto-pass",
        )
    if bxm_calmar is None:
        # No BXM data — defer to "borderline" via aggregation rather
        # than auto-fail. Spec: "no_data does not auto-fail".
        return (
            False,
            "BXM Calmar must be available for CC promotion",
            "BXM data unavailable (Tradier and yfinance both empty)",
        )
    expected = "val_calmar > BXM_calmar (val window)"
    actual = f"val={val_calmar:.4f}, bxm={bxm_calmar:.4f}"
    return val_calmar > bxm_calmar, expected, actual


def _check_no_underlying_concentration(
    concentration_results: tuple[ConcentrationResult, ...],
) -> tuple[bool, str, str]:
    underlying_results = [
        r for r in concentration_results
        if r.ablation_dimension == "underlying"
    ]
    if not underlying_results:
        return (
            False,
            "underlying ablations must exist",
            "no per-underlying ablations found",
        )
    max_attribution = max(
        r.pct_alpha_attribution for r in underlying_results
    )
    worst_ticker = max(
        underlying_results, key=lambda r: r.pct_alpha_attribution,
    ).ablation_value
    expected = (
        f"max underlying pct_alpha_attribution <= "
        f"{MAX_UNDERLYING_ATTRIBUTION}"
    )
    actual = (
        f"worst={worst_ticker} at {max_attribution:.4f}"
    )
    return max_attribution <= MAX_UNDERLYING_ATTRIBUTION, expected, actual


# ----------------- main entry point -----------------


def evaluate_promotion(
    *,
    strategy_class: str,
    primary_study: OptunaStudyResults,
    primary_results: StudyResults,
    spy_total_return: pd.DataFrame,
    bxm: Optional[pd.DataFrame],
    concentration_results: tuple[ConcentrationResult, ...],
) -> PromotionRecommendation:
    """Run the five hardcoded promotion checks and aggregate the
    recommendation.

    ``primary_study`` is the OptunaStudyResults summary; ``primary_results``
    is the best trial's full StudyResults (used to compute train and val
    Calmar). ``spy_total_return`` is a DataFrame with at minimum a
    ``total_return_index`` column. ``bxm`` is None when CSP-only or
    when both Tradier and yfinance returned empty.
    """
    train_snaps = [
        s for s in primary_results.daily_snapshots
        if s.train_val_label == "train"
    ]
    val_snaps = [
        s for s in primary_results.daily_snapshots
        if s.train_val_label == "val"
    ]
    train_calmar = _calmar_from_snapshots(train_snaps)
    val_calmar = _calmar_from_snapshots(val_snaps)

    val_window_start = (
        val_snaps[0].sim_date if val_snaps else date.min
    )
    val_window_end = val_snaps[-1].sim_date if val_snaps else date.max

    spy_series = spy_total_return.get(
        "total_return_index",
        pd.Series(dtype="float64"),
    )
    spy_val_series = _filter_by_date(
        spy_series, val_window_start, val_window_end,
    )
    spy_calmar_on_val = calmar_from_series(spy_val_series)

    bxm_calmar_on_val: Optional[float] = None
    if bxm is not None and "close" in bxm.columns and len(bxm) > 0:
        bxm_val_series = _filter_by_date(
            bxm["close"], val_window_start, val_window_end,
        )
        bxm_calmar_on_val = calmar_from_series(bxm_val_series)

    checks: list[PromotionCheck] = []

    passed, expected, actual = _check_overfit(train_calmar, val_calmar)
    checks.append(PromotionCheck(
        criterion_name="overfit_check",
        passed=passed,
        expected=expected,
        actual=actual,
        explanation=(
            "Val Calmar should be at least 50% of train Calmar to "
            "rule out overfitting on the training window."
        ),
    ))

    passed, expected, actual = _check_beats_spy(
        val_calmar, spy_calmar_on_val,
    )
    checks.append(PromotionCheck(
        criterion_name="beats_spy",
        passed=passed,
        expected=expected,
        actual=actual,
        explanation=(
            "Promoted study must outperform the SPY total-return "
            "benchmark on a Calmar basis over the val window."
        ),
    ))

    passed, expected, actual = _check_beats_bxm(
        strategy_class, val_calmar, bxm_calmar_on_val,
    )
    checks.append(PromotionCheck(
        criterion_name="beats_bxm",
        passed=passed,
        expected=expected,
        actual=actual,
        explanation=(
            "CC studies must outperform the CBOE BuyWrite Index "
            "(BXM) — covered calls are a beta-repackaging risk; "
            "BXM is the strategy-class-specific honesty check."
        ),
    ))

    passed, expected, actual = _check_no_underlying_concentration(
        concentration_results,
    )
    checks.append(PromotionCheck(
        criterion_name="no_underlying_concentration",
        passed=passed,
        expected=expected,
        actual=actual,
        explanation=(
            "No single underlying may contribute more than 50% of "
            "the training Calmar (the NVDA/META concentration check "
            "from §7 of the design memo)."
        ),
    ))

    passed, expected, actual = _check_regime_ratio(
        concentration_results,
    )
    checks.append(PromotionCheck(
        criterion_name="regime_independence",
        passed=passed,
        expected=expected,
        actual=actual,
        explanation=(
            "High-IV and low-IV regime ablations should produce "
            "Calmar within 2x of each other — the strategy's edge "
            "should not depend on a single vol regime."
        ),
    ))

    n_passed = sum(1 for c in checks if c.passed)
    if n_passed == 5:
        recommendation = "promote"
    elif n_passed == 4:
        recommendation = "borderline"
    else:
        recommendation = "do_not_promote"

    summary = (
        f"{recommendation} ({n_passed}/5 checks passed; "
        f"train_calmar={train_calmar:.4f}, val_calmar={val_calmar:.4f})"
    )

    return PromotionRecommendation(
        automated_recommendation=recommendation,
        checks=tuple(checks),
        train_calmar=train_calmar,
        val_calmar=val_calmar,
        spy_calmar_on_val=spy_calmar_on_val,
        bxm_calmar_on_val=bxm_calmar_on_val,
        summary=summary,
    )


def write_promotion_decision(
    output_dir: Path,
    recommendation: PromotionRecommendation,
    human_override: Optional[dict] = None,
) -> Path:
    """Persist ``promotion_decision.json``.

    If ``human_override`` is None, the file captures only the
    automated recommendation. Pass a dict with ``decision`` and
    ``reasoning`` keys to overlay the human's final call. The CLI
    calls this twice — once after the automated check, once after
    the prompt.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "promotion_decision.json"
    data = {
        "automated": recommendation.to_dict(),
        "human_override": human_override,
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    return path
