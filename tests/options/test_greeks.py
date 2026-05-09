"""Tests for src/options/greeks.py.

Pure math. No fixtures, no network. Each behavior covered exactly once.

Validation strategy:
- Hull textbook reference values (with pytest.approx tolerance that
  absorbs textbook rounding to 2-4 decimals).
- Put-call parity for randomized inputs (mathematical identity).
- Finite-difference Greeks against closed-form (1e-4 absolute, looser
  for theta which has the largest natural magnitude).
- Edge cases per the module docstring.
- Implied-vol round-trip + bracket-message check.
"""

from __future__ import annotations

import math
import random
from datetime import date

import pytest

from src.options.greeks import (
    GreeksResult,
    compute_all,
    delta,
    gamma,
    implied_vol,
    price,
    rho_per_bp,
    theta_per_day,
    time_to_expiration,
    vega_per_pct,
)


# ---- Hull reference values --------------------------------------------------
#
# Hull, "Options, Futures, and Other Derivatives", chapter on the Greek
# letters. Standard reference example: S=49, K=50, r=5%, q=0, vol=20%,
# T=20 weeks = 0.3846 years. Hull rounds his published values to 2-4
# decimals so use pytest.approx(abs=0.005, rel=0.001) per the spec.
_HULL_S = 49.0
_HULL_K = 50.0
_HULL_T = 20.0 / 52.0
_HULL_R = 0.05
_HULL_Q = 0.0
_HULL_VOL = 0.20
_HULL_TOL = pytest.approx  # alias for readability


def test_hull_call_price_matches_textbook():
    # Hull, Example 17.1: call price ≈ 2.40
    assert price(_HULL_S, _HULL_K, _HULL_T, _HULL_R, _HULL_Q, _HULL_VOL, "C") == \
        _HULL_TOL(2.40, abs=0.005, rel=0.001)


def test_hull_call_delta_matches_textbook():
    # Hull, Section 17.4 reference: delta ≈ 0.522
    assert delta(_HULL_S, _HULL_K, _HULL_T, _HULL_R, _HULL_Q, _HULL_VOL, "C") == \
        _HULL_TOL(0.522, abs=0.005, rel=0.001)


def test_hull_call_gamma_matches_textbook():
    # Hull, Section 17.5 reference: gamma ≈ 0.066
    assert gamma(_HULL_S, _HULL_K, _HULL_T, _HULL_R, _HULL_Q, _HULL_VOL, "C") == \
        _HULL_TOL(0.066, abs=0.005, rel=0.001)


def test_hull_call_vega_matches_textbook():
    # Hull, Section 17.6 reference: vega ≈ 12.10 (per 1.0 vol move).
    # Our vega_per_pct is per 0.01 vol move → 12.10 / 100 = 0.121.
    assert vega_per_pct(_HULL_S, _HULL_K, _HULL_T, _HULL_R, _HULL_Q, _HULL_VOL, "C") == \
        _HULL_TOL(0.121, abs=0.005, rel=0.001)


# ---- Put-call parity (mathematical identity) --------------------------------


def test_put_call_parity_holds_across_random_inputs():
    """C - P == S * e^(-qT) - K * e^(-rT) for any inputs."""
    rng = random.Random(20260509)
    for _ in range(50):
        s = rng.uniform(10.0, 500.0)
        k = rng.uniform(0.5 * s, 1.5 * s)
        t = rng.uniform(0.01, 2.0)
        r = rng.uniform(0.0, 0.08)
        q = rng.uniform(0.0, 0.05)
        vol = rng.uniform(0.05, 1.0)
        c = price(s, k, t, r, q, vol, "C")
        p = price(s, k, t, r, q, vol, "P")
        expected = s * math.exp(-q * t) - k * math.exp(-r * t)
        assert c - p == pytest.approx(expected, abs=1e-9)


# ---- Finite-difference Greeks vs closed-form -------------------------------
#
# For a small bump ε on the relevant input, the central-difference of
# price() should match the analytical Greek to ~1e-4. Theta has the
# largest natural magnitude (and is annualized in the analytical form
# vs per-day in our exposed function) so use a looser bound.


_FD_S = 100.0
_FD_K = 100.0
_FD_T = 0.5
_FD_R = 0.04
_FD_Q = 0.02
_FD_VOL = 0.25


def _fd(f, x, eps):
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


@pytest.mark.parametrize("ot", ["C", "P"])
def test_finite_diff_delta(ot):
    eps = 1e-4 * _FD_S
    fd = _fd(
        lambda s: price(s, _FD_K, _FD_T, _FD_R, _FD_Q, _FD_VOL, ot),
        _FD_S, eps,
    )
    assert delta(_FD_S, _FD_K, _FD_T, _FD_R, _FD_Q, _FD_VOL, ot) == \
        pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("ot", ["C", "P"])
def test_finite_diff_gamma(ot):
    eps = 1e-3 * _FD_S
    delta_fd = (
        delta(_FD_S + eps, _FD_K, _FD_T, _FD_R, _FD_Q, _FD_VOL, ot)
        - delta(_FD_S - eps, _FD_K, _FD_T, _FD_R, _FD_Q, _FD_VOL, ot)
    ) / (2.0 * eps)
    assert gamma(_FD_S, _FD_K, _FD_T, _FD_R, _FD_Q, _FD_VOL, ot) == \
        pytest.approx(delta_fd, abs=1e-4)


@pytest.mark.parametrize("ot", ["C", "P"])
def test_finite_diff_vega(ot):
    """vega_per_pct = analytical_vega / 100; FD bumps vol by ε then
    divides by 100 to match per-1-IV-point units."""
    eps = 1e-4
    fd_per_unit_vol = _fd(
        lambda v: price(_FD_S, _FD_K, _FD_T, _FD_R, _FD_Q, v, ot),
        _FD_VOL, eps,
    )
    assert vega_per_pct(_FD_S, _FD_K, _FD_T, _FD_R, _FD_Q, _FD_VOL, ot) == \
        pytest.approx(fd_per_unit_vol / 100.0, abs=1e-4)


@pytest.mark.parametrize("ot", ["C", "P"])
def test_finite_diff_theta(ot):
    """theta_per_day = -dprice/dt / 365 (theta is decay → negative
    sign on dt). Looser tolerance for theta's larger magnitude."""
    eps = 1e-5
    fd_per_year = -_fd(
        lambda t: price(_FD_S, _FD_K, t, _FD_R, _FD_Q, _FD_VOL, ot),
        _FD_T, eps,
    )
    assert theta_per_day(_FD_S, _FD_K, _FD_T, _FD_R, _FD_Q, _FD_VOL, ot) == \
        pytest.approx(fd_per_year / 365.0, abs=1e-3)


@pytest.mark.parametrize("ot", ["C", "P"])
def test_finite_diff_rho(ot):
    """rho_per_bp = analytical_rho / 10_000."""
    eps = 1e-5
    fd_per_unit_r = _fd(
        lambda r: price(_FD_S, _FD_K, _FD_T, r, _FD_Q, _FD_VOL, ot),
        _FD_R, eps,
    )
    assert rho_per_bp(_FD_S, _FD_K, _FD_T, _FD_R, _FD_Q, _FD_VOL, ot) == \
        pytest.approx(fd_per_unit_r / 10_000.0, abs=1e-4)


# ---- ATM / OTM / ITM at typical 30 DTE -------------------------------------


def test_atm_30dte_call_no_blowup():
    g = compute_all(100.0, 100.0, 30.0 / 365.0, 0.04, 0.02, 0.25, "C")
    assert all(math.isfinite(v) for v in (g.price, g.delta, g.gamma,
                                          g.theta_per_day, g.vega_per_pct,
                                          g.rho_per_bp))
    # ATM call delta ~ 0.5 ish (slightly off due to drift/dividends)
    assert 0.4 < g.delta < 0.65


def test_deep_otm_30dte_call_delta_near_zero():
    # 5σ OTM at 30 DTE
    sigma = 0.25 * math.sqrt(30 / 365.0)
    s, k = 100.0, 100.0 * math.exp(5 * sigma)
    g = compute_all(s, k, 30.0 / 365.0, 0.04, 0.02, 0.25, "C")
    assert 0.0 < g.delta < 0.05
    assert g.gamma >= 0.0


def test_deep_itm_30dte_put_delta_near_neg_one():
    sigma = 0.25 * math.sqrt(30 / 365.0)
    s, k = 100.0, 100.0 * math.exp(5 * sigma)
    g = compute_all(s, k, 30.0 / 365.0, 0.04, 0.02, 0.25, "P")
    assert -1.0 < g.delta < -0.95


# ---- Near-expiration numerical stability -----------------------------------


def test_one_day_to_expiry_no_blowup():
    g = compute_all(100.0, 100.0, 1.0 / 365.0, 0.04, 0.02, 0.25, "C")
    assert all(math.isfinite(v) for v in (g.price, g.delta, g.gamma,
                                          g.theta_per_day, g.vega_per_pct,
                                          g.rho_per_bp))


# ---- T == 0 edge case -------------------------------------------------------


def test_t_zero_call_returns_intrinsic_and_delta_indicator():
    # ITM call
    g = compute_all(105.0, 100.0, 0.0, 0.04, 0.02, 0.25, "C")
    assert g.price == 5.0
    assert g.delta == 1.0
    assert g.gamma == 0.0
    assert g.theta_per_day == 0.0
    assert g.vega_per_pct == 0.0
    assert g.rho_per_bp == 0.0


def test_t_zero_put_returns_intrinsic_and_delta_indicator():
    # ITM put
    g = compute_all(95.0, 100.0, 0.0, 0.04, 0.02, 0.25, "P")
    assert g.price == 5.0
    assert g.delta == -1.0


def test_t_zero_otm_call_zero_price_zero_delta():
    g = compute_all(95.0, 100.0, 0.0, 0.04, 0.02, 0.25, "C")
    assert g.price == 0.0
    assert g.delta == 0.0


# ---- vol == 0 edge case (separate from vol < 0) -----------------------------


def test_vol_zero_call_returns_intrinsic_and_delta_indicator():
    g = compute_all(105.0, 100.0, 0.5, 0.04, 0.02, 0.0, "C")
    assert g.price == 5.0
    assert g.delta == 1.0
    assert g.gamma == 0.0


def test_vol_negative_raises():
    """vol < 0 is a programming error and raises (not a degenerate case)."""
    with pytest.raises(ValueError, match="vol"):
        price(100.0, 100.0, 0.5, 0.04, 0.02, -0.01, "C")


# ---- Other validation errors ------------------------------------------------


def test_t_negative_raises():
    with pytest.raises(ValueError, match="t"):
        price(100.0, 100.0, -0.1, 0.04, 0.02, 0.25, "C")


def test_s_non_positive_raises():
    with pytest.raises(ValueError, match="s"):
        price(0.0, 100.0, 0.5, 0.04, 0.02, 0.25, "C")
    with pytest.raises(ValueError, match="s"):
        price(-1.0, 100.0, 0.5, 0.04, 0.02, 0.25, "C")


def test_k_non_positive_raises():
    with pytest.raises(ValueError, match="k"):
        price(100.0, 0.0, 0.5, 0.04, 0.02, 0.25, "C")
    with pytest.raises(ValueError, match="k"):
        price(100.0, -10.0, 0.5, 0.04, 0.02, 0.25, "C")


def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        price(100.0, 100.0, 0.5, 0.04, 0.02, 0.25, "X")


# ---- Implied vol ------------------------------------------------------------


def test_implied_vol_round_trips():
    rng = random.Random(20260510)
    for _ in range(20):
        s = rng.uniform(50.0, 300.0)
        k = rng.uniform(0.7 * s, 1.3 * s)
        t = rng.uniform(0.05, 1.5)
        r = rng.uniform(0.0, 0.06)
        q = rng.uniform(0.0, 0.04)
        true_vol = rng.uniform(0.10, 0.80)
        for ot in ("C", "P"):
            premium = price(s, k, t, r, q, true_vol, ot)
            recovered = implied_vol(premium, s, k, t, r, q, ot)
            assert recovered == pytest.approx(true_vol, abs=1e-6)


def test_implied_vol_below_lower_bound_raises():
    """A premium below the European no-arbitrage lower bound
    (S*e^(-qT) - K*e^(-rT) for calls) raises. This is stricter than
    the American intrinsic max(S-K, 0) when q < r."""
    with pytest.raises(ValueError, match="lower bound"):
        implied_vol(0.0, 105.0, 100.0, 0.5, 0.04, 0.02, "C")


def test_implied_vol_t_zero_raises():
    with pytest.raises(ValueError, match="t"):
        implied_vol(5.0, 105.0, 100.0, 0.0, 0.04, 0.02, "C")


def test_implied_vol_outside_bracket_raises_with_bracket_message():
    """A premium higher than what 500% vol can produce must raise
    RuntimeError naming the [1e-6, 5.0] bracket per the spec."""
    # An absurdly large premium for an OTM call → no vol in [1e-6, 5.0]
    # can match. (Cap is the upper bound 5.0; even at vol=5.0 the price
    # can't reach S * e^(-qT) which is the upper bound.)
    s, k = 100.0, 100.0
    upper_bound_price = s  # call price < S asymptotically
    with pytest.raises(RuntimeError, match=r"\[1e-06, 5.0\] bracket"):
        implied_vol(upper_bound_price + 1.0, s, k, 0.5, 0.04, 0.02, "C")


# ---- compute_all consistency with per-Greek functions ----------------------


def test_compute_all_matches_individual_functions():
    s, k, t, r, q, vol = 100.0, 105.0, 0.4, 0.04, 0.02, 0.30
    for ot in ("C", "P"):
        g = compute_all(s, k, t, r, q, vol, ot)
        assert g.price == pytest.approx(price(s, k, t, r, q, vol, ot), rel=1e-12)
        assert g.delta == pytest.approx(delta(s, k, t, r, q, vol, ot), rel=1e-12)
        assert g.gamma == pytest.approx(gamma(s, k, t, r, q, vol, ot), rel=1e-12)
        assert g.theta_per_day == pytest.approx(
            theta_per_day(s, k, t, r, q, vol, ot), rel=1e-12)
        assert g.vega_per_pct == pytest.approx(
            vega_per_pct(s, k, t, r, q, vol, ot), rel=1e-12)
        assert g.rho_per_bp == pytest.approx(
            rho_per_bp(s, k, t, r, q, vol, ot), rel=1e-12)


def test_compute_all_returns_frozen_dataclass():
    g = compute_all(100.0, 100.0, 0.5, 0.04, 0.02, 0.25, "C")
    assert isinstance(g, GreeksResult)
    with pytest.raises(Exception):
        g.delta = 0.0  # type: ignore[misc]


# ---- time_to_expiration ----------------------------------------------------


def test_time_to_expiration_30_days():
    assert time_to_expiration(date(2026, 1, 1), date(2026, 1, 31)) == \
        pytest.approx(30.0 / 365.0)


def test_time_to_expiration_90_days():
    assert time_to_expiration(date(2026, 1, 1), date(2026, 4, 1)) == \
        pytest.approx(90.0 / 365.0)


def test_time_to_expiration_same_day_returns_zero():
    assert time_to_expiration(date(2026, 1, 1), date(2026, 1, 1)) == 0.0


def test_time_to_expiration_past_raises():
    with pytest.raises(ValueError, match="before today"):
        time_to_expiration(date(2026, 5, 1), date(2026, 4, 1))
