"""Black-Scholes-Merton Greeks module (Phase 2 Section 3).

Closed-form, pure-function. No I/O, no network. Section 6 (engine)
consumes ``compute_all`` per (contract, day); Section 6's IV-regime
analysis consumes ``implied_vol`` for backtest IV reconstruction
(Tradier per-contract history returns OHLCV without IV).

Conventions
-----------
All values use trader-convention units, encoded in field names so
consumers don't have to remember:

- ``theta_per_day`` is the analytical theta divided by 365 (per
  calendar day).
- ``vega_per_pct`` is the analytical vega divided by 100 (per 1 IV
  point of move, i.e., per 0.01 vol).
- ``rho_per_bp`` is the analytical rho divided by 10_000 (per 1 bp
  of rate move, i.e., per 0.0001).
- ``delta`` and ``gamma`` are dimensionless and stand alone.

Inputs ``vol``, ``r``, ``q`` are decimal fractions: 0.20 for 20% vol,
0.04 for 4% rate, 0.0285 for 2.85% yield. ``option_type`` is ``'C'``
or ``'P'`` matching ``ContractSpec.option_type``.

Day count
---------
``time_to_expiration(today, expiration)`` returns
``(expiration - today).days / 365``. ACT/365 is hardcoded per §3 of
the design memo; v1.1+ adds basis selection if a study needs ACT/360
or business-day basis.

Edge cases
----------
- ``T < 0``                 → ``ValueError``
- ``S <= 0`` or ``K <= 0``  → ``ValueError``
- ``vol < 0``               → ``ValueError``
- ``option_type`` invalid  → ``ValueError``
- ``T == 0`` (or below the float-noise threshold) → returns intrinsic
  value for ``price``; delta = +1 (ITM call) / -1 (ITM put) / 0 (OTM);
  all other Greeks = 0.
- ``vol == 0`` → same as ``T == 0``; deterministic intrinsic + zero
  Greeks. Kept separate from ``vol < 0`` (which raises) on purpose —
  ``vol == 0`` is a degenerate-but-well-defined input, ``vol < 0`` is
  a programming error.

American-style options
----------------------
Treated as European. Early-exercise premium is ignored. Documented in
§8 of the design memo; v1.1+ adds Barone-Adesi-Whaley if Section 8
surfaces a meaningful gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from scipy.optimize import brentq
from scipy.stats import norm

_VALID_OPTION_TYPES = ("C", "P")
_T_EPSILON = 1e-9              # below this is "effectively zero years"
_VOL_EPSILON = 1e-12           # below this is "effectively zero vol"
_IV_LOWER_BRACKET = 1e-6
_IV_UPPER_BRACKET = 5.0


def time_to_expiration(today: date, expiration: date) -> float:
    """Years between ``today`` and ``expiration`` under ACT/365."""
    if expiration < today:
        raise ValueError(
            f"expiration {expiration} is before today {today}"
        )
    return (expiration - today).days / 365.0


@dataclass(frozen=True, slots=True)
class GreeksResult:
    price: float
    delta: float
    gamma: float
    theta_per_day: float
    vega_per_pct: float
    rho_per_bp: float


def _validate_inputs(s: float, k: float, t: float, vol: float,
                     option_type: str) -> None:
    if s <= 0:
        raise ValueError(f"s must be > 0; got {s!r}")
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k!r}")
    if t < 0:
        raise ValueError(f"t must be >= 0; got {t!r}")
    if vol < 0:
        raise ValueError(f"vol must be >= 0; got {vol!r}")
    if option_type not in _VALID_OPTION_TYPES:
        raise ValueError(
            f"option_type must be one of {_VALID_OPTION_TYPES}; "
            f"got {option_type!r}"
        )


def _intrinsic_price(s: float, k: float, option_type: str) -> float:
    if option_type == "C":
        return max(s - k, 0.0)
    return max(k - s, 0.0)


def _intrinsic_delta(s: float, k: float, option_type: str) -> float:
    if option_type == "C":
        if s > k:
            return 1.0
        return 0.0
    if s < k:
        return -1.0
    return 0.0


def _is_degenerate(t: float, vol: float) -> bool:
    return t < _T_EPSILON or vol < _VOL_EPSILON


def _d1_d2(s: float, k: float, t: float, r: float, q: float,
           vol: float) -> tuple[float, float]:
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def price(s: float, k: float, t: float, r: float, q: float,
          vol: float, option_type: str) -> float:
    """Black-Scholes-Merton price."""
    _validate_inputs(s, k, t, vol, option_type)
    if _is_degenerate(t, vol):
        return _intrinsic_price(s, k, option_type)
    d1, d2 = _d1_d2(s, k, t, r, q, vol)
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    if option_type == "C":
        return s * disc_q * norm.cdf(d1) - k * disc_r * norm.cdf(d2)
    return k * disc_r * norm.cdf(-d2) - s * disc_q * norm.cdf(-d1)


def delta(s: float, k: float, t: float, r: float, q: float,
          vol: float, option_type: str) -> float:
    _validate_inputs(s, k, t, vol, option_type)
    if _is_degenerate(t, vol):
        return _intrinsic_delta(s, k, option_type)
    d1, _ = _d1_d2(s, k, t, r, q, vol)
    disc_q = math.exp(-q * t)
    if option_type == "C":
        return disc_q * norm.cdf(d1)
    return disc_q * (norm.cdf(d1) - 1.0)


def gamma(s: float, k: float, t: float, r: float, q: float,
          vol: float, option_type: str) -> float:
    _validate_inputs(s, k, t, vol, option_type)
    if _is_degenerate(t, vol):
        return 0.0
    d1, _ = _d1_d2(s, k, t, r, q, vol)
    disc_q = math.exp(-q * t)
    return disc_q * norm.pdf(d1) / (s * vol * math.sqrt(t))


def theta_per_day(s: float, k: float, t: float, r: float, q: float,
                  vol: float, option_type: str) -> float:
    """Theta scaled to per-calendar-day (analytical theta / 365)."""
    _validate_inputs(s, k, t, vol, option_type)
    if _is_degenerate(t, vol):
        return 0.0
    d1, d2 = _d1_d2(s, k, t, r, q, vol)
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    sqrt_t = math.sqrt(t)
    common = -disc_q * s * norm.pdf(d1) * vol / (2.0 * sqrt_t)
    if option_type == "C":
        analytical = (
            common
            - r * k * disc_r * norm.cdf(d2)
            + q * s * disc_q * norm.cdf(d1)
        )
    else:
        analytical = (
            common
            + r * k * disc_r * norm.cdf(-d2)
            - q * s * disc_q * norm.cdf(-d1)
        )
    return analytical / 365.0


def vega_per_pct(s: float, k: float, t: float, r: float, q: float,
                 vol: float, option_type: str) -> float:
    """Vega per 1 IV point (analytical vega / 100)."""
    _validate_inputs(s, k, t, vol, option_type)
    if _is_degenerate(t, vol):
        return 0.0
    d1, _ = _d1_d2(s, k, t, r, q, vol)
    disc_q = math.exp(-q * t)
    analytical = s * disc_q * norm.pdf(d1) * math.sqrt(t)
    return analytical / 100.0


def rho_per_bp(s: float, k: float, t: float, r: float, q: float,
               vol: float, option_type: str) -> float:
    """Rho per 1 bp of rate move (analytical rho / 10_000)."""
    _validate_inputs(s, k, t, vol, option_type)
    if _is_degenerate(t, vol):
        return 0.0
    _, d2 = _d1_d2(s, k, t, r, q, vol)
    disc_r = math.exp(-r * t)
    if option_type == "C":
        analytical = k * t * disc_r * norm.cdf(d2)
    else:
        analytical = -k * t * disc_r * norm.cdf(-d2)
    return analytical / 10_000.0


def compute_all(s: float, k: float, t: float, r: float, q: float,
                vol: float, option_type: str) -> GreeksResult:
    """Compute price + all five Greeks in one shot, sharing intermediates.

    Functionally equivalent to calling each per-Greek function but
    computes ``d1``, ``d2``, ``N(±d1)``, ``N(±d2)``, ``n(d1)``, and
    discount factors once instead of five times.
    """
    _validate_inputs(s, k, t, vol, option_type)
    if _is_degenerate(t, vol):
        return GreeksResult(
            price=_intrinsic_price(s, k, option_type),
            delta=_intrinsic_delta(s, k, option_type),
            gamma=0.0,
            theta_per_day=0.0,
            vega_per_pct=0.0,
            rho_per_bp=0.0,
        )
    d1, d2 = _d1_d2(s, k, t, r, q, vol)
    sqrt_t = math.sqrt(t)
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    n_d1 = norm.pdf(d1)
    if option_type == "C":
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        price_val = s * disc_q * N_d1 - k * disc_r * N_d2
        delta_val = disc_q * N_d1
        analytical_theta = (
            -disc_q * s * n_d1 * vol / (2.0 * sqrt_t)
            - r * k * disc_r * N_d2
            + q * s * disc_q * N_d1
        )
        analytical_rho = k * t * disc_r * N_d2
    else:
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        price_val = k * disc_r * N_neg_d2 - s * disc_q * N_neg_d1
        delta_val = disc_q * (norm.cdf(d1) - 1.0)
        analytical_theta = (
            -disc_q * s * n_d1 * vol / (2.0 * sqrt_t)
            + r * k * disc_r * N_neg_d2
            - q * s * disc_q * N_neg_d1
        )
        analytical_rho = -k * t * disc_r * N_neg_d2
    gamma_val = disc_q * n_d1 / (s * vol * sqrt_t)
    analytical_vega = s * disc_q * n_d1 * sqrt_t
    return GreeksResult(
        price=price_val,
        delta=delta_val,
        gamma=gamma_val,
        theta_per_day=analytical_theta / 365.0,
        vega_per_pct=analytical_vega / 100.0,
        rho_per_bp=analytical_rho / 10_000.0,
    )


def implied_vol(option_price: float, s: float, k: float, t: float,
                r: float, q: float, option_type: str,
                *, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Implied volatility via Brent's method on ``price - option_price``.

    Brackets the search at ``[1e-6, 5.0]``. A vol outside that bracket
    raises ``RuntimeError`` with a message naming the bracket so
    Section 8 diagnostics are clear when a real edge case hits.

    ``option_price < intrinsic`` raises ``ValueError`` (no-arbitrage
    violation). ``T <= 0`` also raises since IV is undefined at expiry.
    """
    _validate_inputs(s, k, t, 0.0, option_type)
    if t <= 0:
        raise ValueError("t must be > 0 for implied_vol")
    # European lower bound — NOT the American intrinsic. A deep-ITM
    # European put with r > q can legitimately price below max(K-S, 0)
    # because the holder forgoes interest on the strike.
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    if option_type == "C":
        lower_bound = max(s * disc_q - k * disc_r, 0.0)
    else:
        lower_bound = max(k * disc_r - s * disc_q, 0.0)
    if option_price < lower_bound:
        raise ValueError(
            f"option_price {option_price} below no-arbitrage lower bound "
            f"{lower_bound} — no-arbitrage violation"
        )

    def objective(vol: float) -> float:
        return price(s, k, t, r, q, vol, option_type) - option_price

    f_lower = objective(_IV_LOWER_BRACKET)
    f_upper = objective(_IV_UPPER_BRACKET)
    if f_lower * f_upper > 0:
        raise RuntimeError(
            f"vol outside [{_IV_LOWER_BRACKET}, {_IV_UPPER_BRACKET}] bracket; "
            f"price({_IV_LOWER_BRACKET})={f_lower + option_price:.6f}, "
            f"price({_IV_UPPER_BRACKET})={f_upper + option_price:.6f}, "
            f"target={option_price:.6f}"
        )
    return brentq(
        objective, _IV_LOWER_BRACKET, _IV_UPPER_BRACKET,
        xtol=tol, maxiter=max_iter,
    )
