"""Position + lifecycle model (Phase 2 Section 4).

A multi-leg options position with first-class active-management exit
rules. The hybrid representation: canonical ``legs: tuple[Leg, ...]``
shape used by all engine code, plus per-strategy classmethod
constructors (``Position.covered_call``, ``Position.cash_secured_put``)
for self-documenting construction with validation.

A ``Leg`` carries ``(contract, sign, quantity)`` where ``contract`` is
one of ``ContractSpec`` (option), ``StockContract`` (stock), or
``CashContract`` (cash collateral) — discriminated by type. Explicit
cash legs on CSPs and explicit stock legs on CCs make the portfolio
accounting honest.

Market mapping convention
-------------------------
``mark_to_market`` and friends accept a ``market: Mapping[str, float]``
keyed as follows:

- Option legs: OCC string (``generate_occ_symbol(contract)``)
- Stock legs: ticker string (``StockContract.ticker``)
- Cash legs: ``"CASH"`` (optional — cash legs do not contribute to P&L
  in v1 per memo §8, so the value is unused)

entry_credit and mark_to_market math (locked convention)
--------------------------------------------------------
``entry_credit`` follows trader convention: net cash received at open,
positive for credits, negative for debits.

- CSP: ``+put_premium * 100`` per contract (e.g., +200 for $2.00 put)
- CC:  ``-stock_basis * 100 + call_premium * 100`` per contract
  (e.g., -9800 = -10000 + 200 for stock at $100, call at $2.00)
- Long premium (future): ``-premium_paid * 100`` (debit)

``mark_to_market(market)`` returns P&L in dollars:

::

    P&L = sum(leg.sign * leg.quantity * multiplier(leg) * mark(leg, market)
              for leg in legs
              if not isinstance(leg.contract, CashContract)) + entry_credit

Cash legs contribute zero to P&L because in v1 they are held at par
with zero yield (memo §8: "Cash legs in v1 are treated as zero-yield";
v1.1+ adds risk-free rate accrual). Stock legs **do** contribute (a CC's
stock leg moves with the underlying).

This deviates from the literal Section 4 spec ("sum over legs ... minus
entry_credit") in favor of the trader-view docstring and the test-name
semantics (P&L=0 at open, ``realized_pnl == entry_credit`` when option
expires worthless). The decision is recorded in the design memo §9
row 4 footnote.

``should_exit`` thresholds use ``abs(entry_credit)`` for symmetry across
credit and debit positions:

- profit_target triggers when ``P&L >= profit_target_pct * abs(entry_credit)``
- stop_loss triggers when ``P&L <= -stop_loss_pct * abs(entry_credit)``

Multipliers
-----------
- ``ContractSpec``  → 100 (one contract = 100 shares)
- ``StockContract`` → 1
- ``CashContract``  → 1 (irrelevant — cash legs excluded from P&L sum)

Settlement at expiration
------------------------
Per memo §8:

- European underlyings (SPX) cash-settle to intrinsic. Resolved state is
  ``EXPIRED_ITM`` (with ``closure_reason="expired_itm_cash_settled"``)
  if any leg is ITM at expiration spot, else ``EXPIRED_OTM``.
- American underlyings (SPY/QQQ/equity) share-settle short ITM legs.
  Resolved state is ``ASSIGNED`` (``closure_reason="assigned_call"`` or
  ``"assigned_put"``) if any short leg is ITM at expiration, else
  ``EXPIRED_OTM``. The spawned equity position from ``ASSIGNED`` is
  Section 6's concern, not handled here.
- Long ITM legs cash-settle to intrinsic in v1 (memo §10: real-broker
  auto-exercise modeling deferred to v1.1+).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Callable, Mapping

from src.options.greeks import GreeksResult, compute_all, time_to_expiration
from src.options.occ import generate_occ_symbol
from src.options.types import ContractSpec
from src.options.universe import get_underlying_metadata


__all__ = [
    "Leg",
    "CashContract",
    "StockContract",
    "ExitRules",
    "PositionState",
    "Position",
]


_CONTRACT_MULTIPLIER = 100  # one option contract = 100 shares


@dataclass(frozen=True, slots=True)
class CashContract:
    """Sentinel for cash collateral legs. No fields — all instances equal."""


@dataclass(frozen=True, slots=True)
class StockContract:
    """Sentinel for stock legs. Single field: the underlying ticker."""

    ticker: str


@dataclass(frozen=True, slots=True)
class Leg:
    """A single leg of a multi-leg position.

    ``sign``: ``+1`` (long) or ``-1`` (short).
    ``quantity``: positive integer (shares for ``StockContract``,
    contracts for ``ContractSpec``, dollars for ``CashContract``).
    ``CashContract`` legs must have ``sign=+1`` (no concept of borrowing
    cash from the position in v1).
    """

    contract: ContractSpec | StockContract | CashContract
    sign: int
    quantity: int

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError(f"sign must be -1 or +1; got {self.sign!r}")
        if self.quantity <= 0:
            raise ValueError(
                f"quantity must be > 0; got {self.quantity!r}"
            )
        if isinstance(self.contract, CashContract) and self.sign != 1:
            raise ValueError(
                "CashContract leg must have sign=+1; "
                f"got {self.sign!r}"
            )


@dataclass(frozen=True, slots=True)
class ExitRules:
    """Active-management exit thresholds.

    At least one of the three fields must be non-None — a position
    without any exit rule has no managed exit and would only resolve
    via expiration. Section 5 BacktestConfig enforces strategy-level
    constraints; ExitRules itself just validates per-field shape.

    - ``profit_target_pct`` ∈ (0, 1] — fraction of ``abs(entry_credit)``
    - ``time_stop_dte`` ≥ 0 — DTE threshold (e.g., 21 for "exit at 21 DTE")
    - ``stop_loss_pct`` > 0 — multiple of ``abs(entry_credit)`` (can be
      > 1; e.g., 2.0 for "stop at 200% of credit lost")
    """

    profit_target_pct: float | None
    time_stop_dte: int | None
    stop_loss_pct: float | None

    def __post_init__(self) -> None:
        if (
            self.profit_target_pct is None
            and self.time_stop_dte is None
            and self.stop_loss_pct is None
        ):
            raise ValueError(
                "ExitRules requires at least one of profit_target_pct, "
                "time_stop_dte, stop_loss_pct"
            )
        if self.profit_target_pct is not None:
            if not (0 < self.profit_target_pct <= 1):
                raise ValueError(
                    "profit_target_pct must be in (0, 1]; "
                    f"got {self.profit_target_pct!r}"
                )
        if self.time_stop_dte is not None and self.time_stop_dte < 0:
            raise ValueError(
                f"time_stop_dte must be >= 0; got {self.time_stop_dte!r}"
            )
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0:
            raise ValueError(
                f"stop_loss_pct must be > 0; got {self.stop_loss_pct!r}"
            )


class PositionState(str, Enum):
    OPEN = "OPEN"
    CLOSED_MANAGED = "CLOSED_MANAGED"
    EXPIRED_ITM = "EXPIRED_ITM"
    EXPIRED_OTM = "EXPIRED_OTM"
    ASSIGNED = "ASSIGNED"


def _validate_covered_call(legs: tuple[Leg, ...]) -> None:
    if len(legs) != 2:
        raise ValueError(
            f"covered_call requires exactly 2 legs; got {len(legs)}"
        )
    stock_leg, call_leg = legs
    if not isinstance(stock_leg.contract, StockContract):
        raise ValueError(
            "covered_call leg 0 must be a StockContract; "
            f"got {type(stock_leg.contract).__name__}"
        )
    if stock_leg.sign != 1:
        raise ValueError(
            f"covered_call stock leg must be long (sign=+1); "
            f"got sign={stock_leg.sign}"
        )
    if not isinstance(call_leg.contract, ContractSpec):
        raise ValueError(
            "covered_call leg 1 must be a ContractSpec (option); "
            f"got {type(call_leg.contract).__name__}"
        )
    if call_leg.contract.option_type != "C":
        raise ValueError(
            "covered_call option leg must be a call; "
            f"got option_type={call_leg.contract.option_type!r}"
        )
    if call_leg.sign != -1:
        raise ValueError(
            f"covered_call option leg must be short (sign=-1); "
            f"got sign={call_leg.sign}"
        )
    if stock_leg.contract.ticker != call_leg.contract.underlying:
        raise ValueError(
            "covered_call stock ticker must match call underlying; "
            f"got stock={stock_leg.contract.ticker!r}, "
            f"call underlying={call_leg.contract.underlying!r}"
        )
    if stock_leg.quantity != call_leg.quantity * _CONTRACT_MULTIPLIER:
        raise ValueError(
            "covered_call stock leg must hold 100 shares per call "
            f"contract; got stock_qty={stock_leg.quantity}, "
            f"call_qty={call_leg.quantity}"
        )


def _validate_cash_secured_put(legs: tuple[Leg, ...]) -> None:
    if len(legs) != 2:
        raise ValueError(
            f"cash_secured_put requires exactly 2 legs; got {len(legs)}"
        )
    cash_leg, put_leg = legs
    if not isinstance(cash_leg.contract, CashContract):
        raise ValueError(
            "cash_secured_put leg 0 must be a CashContract; "
            f"got {type(cash_leg.contract).__name__}"
        )
    if cash_leg.sign != 1:
        raise ValueError(
            f"cash_secured_put cash leg must be long (sign=+1); "
            f"got sign={cash_leg.sign}"
        )
    if not isinstance(put_leg.contract, ContractSpec):
        raise ValueError(
            "cash_secured_put leg 1 must be a ContractSpec (option); "
            f"got {type(put_leg.contract).__name__}"
        )
    if put_leg.contract.option_type != "P":
        raise ValueError(
            "cash_secured_put option leg must be a put; "
            f"got option_type={put_leg.contract.option_type!r}"
        )
    if put_leg.sign != -1:
        raise ValueError(
            f"cash_secured_put option leg must be short (sign=-1); "
            f"got sign={put_leg.sign}"
        )
    expected_collateral = int(
        put_leg.contract.strike * _CONTRACT_MULTIPLIER * put_leg.quantity
    )
    if cash_leg.quantity != expected_collateral:
        raise ValueError(
            "cash_secured_put cash leg must hold strike*100*contracts "
            f"in dollars; expected {expected_collateral}, "
            f"got {cash_leg.quantity}"
        )


_VALIDATORS: dict[str, Callable[[tuple[Leg, ...]], None]] = {
    "covered_call": _validate_covered_call,
    "cash_secured_put": _validate_cash_secured_put,
}


def _multiplier(contract: ContractSpec | StockContract | CashContract) -> int:
    if isinstance(contract, ContractSpec):
        return _CONTRACT_MULTIPLIER
    return 1


def _market_key(contract: ContractSpec | StockContract | CashContract) -> str:
    if isinstance(contract, ContractSpec):
        return generate_occ_symbol(contract)
    if isinstance(contract, StockContract):
        return contract.ticker
    return "CASH"


@dataclass(frozen=True, slots=True)
class Position:
    """A multi-leg options position with active-management lifecycle.

    See module docstring for the entry_credit / mark_to_market math
    convention.

    Construct via classmethod constructors (``covered_call``,
    ``cash_secured_put``) for v1 strategies; raw construction is
    supported for engine code that builds legs explicitly.
    """

    strategy_class: str
    legs: tuple[Leg, ...]
    entry_date: date
    entry_credit: float
    exit_rules: ExitRules
    state: PositionState
    closure_reason: str | None = None
    closure_date: date | None = None
    realized_pnl: float | None = None

    def __post_init__(self) -> None:
        validator = _VALIDATORS.get(self.strategy_class)
        if validator is None:
            raise ValueError(
                f"unknown strategy_class {self.strategy_class!r}; "
                f"known: {sorted(_VALIDATORS)}"
            )
        validator(self.legs)
        if self.state == PositionState.OPEN:
            if (
                self.closure_reason is not None
                or self.closure_date is not None
                or self.realized_pnl is not None
            ):
                raise ValueError(
                    "OPEN position must have closure_reason, closure_date, "
                    "and realized_pnl all None; got "
                    f"closure_reason={self.closure_reason!r}, "
                    f"closure_date={self.closure_date!r}, "
                    f"realized_pnl={self.realized_pnl!r}"
                )
        else:
            if self.closure_reason is None or self.closure_date is None:
                raise ValueError(
                    f"non-OPEN position (state={self.state.value}) must "
                    "have closure_reason and closure_date set; got "
                    f"closure_reason={self.closure_reason!r}, "
                    f"closure_date={self.closure_date!r}"
                )

    # ----- classmethod constructors -----

    @classmethod
    def covered_call(
        cls,
        underlying: str,
        call_contract: ContractSpec,
        entry_date: date,
        entry_credit: float,
        exit_rules: ExitRules,
        contracts: int = 1,
    ) -> "Position":
        """Build a covered call position (long stock + short call)."""
        legs = (
            Leg(
                contract=StockContract(ticker=underlying),
                sign=+1,
                quantity=contracts * _CONTRACT_MULTIPLIER,
            ),
            Leg(contract=call_contract, sign=-1, quantity=contracts),
        )
        return cls(
            strategy_class="covered_call",
            legs=legs,
            entry_date=entry_date,
            entry_credit=entry_credit,
            exit_rules=exit_rules,
            state=PositionState.OPEN,
        )

    @classmethod
    def cash_secured_put(
        cls,
        put_contract: ContractSpec,
        entry_date: date,
        entry_credit: float,
        exit_rules: ExitRules,
        contracts: int = 1,
    ) -> "Position":
        """Build a cash-secured put position (cash collateral + short put)."""
        cash_qty = int(
            put_contract.strike * _CONTRACT_MULTIPLIER * contracts
        )
        legs = (
            Leg(contract=CashContract(), sign=+1, quantity=cash_qty),
            Leg(contract=put_contract, sign=-1, quantity=contracts),
        )
        return cls(
            strategy_class="cash_secured_put",
            legs=legs,
            entry_date=entry_date,
            entry_credit=entry_credit,
            exit_rules=exit_rules,
            state=PositionState.OPEN,
        )

    # ----- lifecycle -----

    def evolve(self, **changes) -> "Position":
        """Return a new Position with the given fields replaced."""
        return dataclasses.replace(self, **changes)

    def mark_to_market(self, market: Mapping[str, float]) -> float:
        """Return P&L in dollars per the convention in the module docstring."""
        total = 0.0
        for leg in self.legs:
            if isinstance(leg.contract, CashContract):
                continue
            mark = market[_market_key(leg.contract)]
            total += (
                leg.sign * leg.quantity * _multiplier(leg.contract) * mark
            )
        return total + self.entry_credit

    def is_expired(self, today: date) -> bool:
        """True if the nearest-expiration option leg has expired by ``today``."""
        opt_legs = [
            leg for leg in self.legs
            if isinstance(leg.contract, ContractSpec)
        ]
        if not opt_legs:
            return False
        nearest = min(leg.contract.expiration_date for leg in opt_legs)
        return nearest <= today

    def days_to_expiration(self, today: date) -> int:
        """DTE of the nearest-expiration option leg, never negative."""
        opt_legs = [
            leg for leg in self.legs
            if isinstance(leg.contract, ContractSpec)
        ]
        if not opt_legs:
            raise ValueError(
                "days_to_expiration requires at least one option leg"
            )
        nearest = min(leg.contract.expiration_date for leg in opt_legs)
        return max((nearest - today).days, 0)

    def should_exit(
        self, market: Mapping[str, float], today: date
    ) -> tuple[bool, str | None]:
        """Evaluate exit rules in priority order: stop_loss, profit_target, time_stop.

        First trigger wins. Returns ``(False, None)`` if no rule fires.
        Closure reason format: ``profit_target_50pct``, ``time_stop_21dte``,
        ``stop_loss_200pct``.
        """
        rules = self.exit_rules
        pnl = self.mark_to_market(market)
        ref = abs(self.entry_credit)
        if rules.stop_loss_pct is not None:
            if pnl <= -rules.stop_loss_pct * ref:
                pct = int(round(rules.stop_loss_pct * 100))
                return True, f"stop_loss_{pct}pct"
        if rules.profit_target_pct is not None:
            if pnl >= rules.profit_target_pct * ref:
                pct = int(round(rules.profit_target_pct * 100))
                return True, f"profit_target_{pct}pct"
        if rules.time_stop_dte is not None:
            if self.days_to_expiration(today) <= rules.time_stop_dte:
                return True, f"time_stop_{rules.time_stop_dte}dte"
        return False, None

    def resolve_expiration(
        self, market: Mapping[str, float], today: date
    ) -> "Position":
        """Resolve the nearest-expiration option legs at ``today``.

        Cash-settles intrinsic value for European underlyings (SPX) and
        share-settles short ITM legs for American underlyings (per
        memo §8). Long ITM legs cash-settle to intrinsic in v1. The
        equity position spawned by ``state=ASSIGNED`` is Section 6's
        concern; this method only marks state and realized P&L.

        Returns a new ``Position`` with ``state``, ``closure_reason``,
        ``closure_date``, and ``realized_pnl`` set.
        """
        opt_legs = [
            leg for leg in self.legs
            if isinstance(leg.contract, ContractSpec)
        ]
        if not opt_legs:
            raise ValueError(
                "resolve_expiration requires at least one option leg"
            )
        nearest_exp = min(leg.contract.expiration_date for leg in opt_legs)
        underlying = opt_legs[0].contract.underlying
        spot = market[underlying]

        any_short_itm = False
        any_long_itm = False
        short_itm_call = False
        short_itm_put = False
        intrinsic_market: dict[str, float] = dict(market)
        for leg in opt_legs:
            contract = leg.contract
            if contract.expiration_date != nearest_exp:
                continue
            if contract.option_type == "C":
                intrinsic = max(spot - contract.strike, 0.0)
                is_itm = spot > contract.strike
            else:
                intrinsic = max(contract.strike - spot, 0.0)
                is_itm = spot < contract.strike
            intrinsic_market[generate_occ_symbol(contract)] = intrinsic
            if is_itm:
                if leg.sign == -1:
                    any_short_itm = True
                    if contract.option_type == "C":
                        short_itm_call = True
                    else:
                        short_itm_put = True
                else:
                    any_long_itm = True

        meta = get_underlying_metadata(underlying)
        if meta.exercise_style == "european":
            if any_short_itm or any_long_itm:
                new_state = PositionState.EXPIRED_ITM
                reason = "expired_itm_cash_settled"
            else:
                new_state = PositionState.EXPIRED_OTM
                reason = "expired_otm"
        else:  # american
            if any_short_itm:
                new_state = PositionState.ASSIGNED
                # Convention: name the assignment by the short leg type
                # that triggered. Short call ITM dominates if both
                # somehow occur (would only happen in a multi-leg
                # variant Section 4 doesn't construct).
                if short_itm_call:
                    reason = "assigned_call"
                else:
                    reason = "assigned_put"
            else:
                new_state = PositionState.EXPIRED_OTM
                reason = "expired_otm"

        realized = self.mark_to_market(intrinsic_market)
        return self.evolve(
            state=new_state,
            closure_reason=reason,
            closure_date=today,
            realized_pnl=realized,
        )

    def aggregate_greeks(
        self,
        market: Mapping[str, float],
        vol_lookup: Mapping[str, float],
        r: float,
        today: date,
    ) -> GreeksResult:
        """Sum signed leg-level Greeks into a portfolio-level GreeksResult.

        Cash legs contribute zero. Stock legs contribute
        ``delta = sign * quantity / 100`` (per-option-equivalent units —
        a CC's 100 shares = 1.0 contract delta) and zero for the other
        Greeks. Option legs use Section 3's ``compute_all`` with vol
        from ``vol_lookup`` keyed by OCC symbol, weighted by
        ``sign * quantity``.

        ``today`` parameter extends the spec — ``compute_all`` requires
        time-to-expiration, which needs a reference date.
        """
        agg_price = 0.0
        agg_delta = 0.0
        agg_gamma = 0.0
        agg_theta = 0.0
        agg_vega = 0.0
        agg_rho = 0.0
        for leg in self.legs:
            contract = leg.contract
            if isinstance(contract, CashContract):
                continue
            if isinstance(contract, StockContract):
                agg_delta += leg.sign * leg.quantity / _CONTRACT_MULTIPLIER
                continue
            # ContractSpec
            occ = generate_occ_symbol(contract)
            spot = market[contract.underlying]
            vol = vol_lookup[occ]
            meta = get_underlying_metadata(contract.underlying)
            t = time_to_expiration(today, contract.expiration_date)
            leg_greeks = compute_all(
                s=spot,
                k=contract.strike,
                t=t,
                r=r,
                q=meta.dividend_yield,
                vol=vol,
                option_type=contract.option_type,
            )
            weight = leg.sign * leg.quantity
            agg_price += weight * leg_greeks.price
            agg_delta += weight * leg_greeks.delta
            agg_gamma += weight * leg_greeks.gamma
            agg_theta += weight * leg_greeks.theta_per_day
            agg_vega += weight * leg_greeks.vega_per_pct
            agg_rho += weight * leg_greeks.rho_per_bp
        return GreeksResult(
            price=agg_price,
            delta=agg_delta,
            gamma=agg_gamma,
            theta_per_day=agg_theta,
            vega_per_pct=agg_vega,
            rho_per_bp=agg_rho,
        )
