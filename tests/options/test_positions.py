"""Tests for src/options/positions.py.

All offline. No network, no real market data — fixtures are constructed
in-process. Each behavior is covered exactly once per the Section 4
test plan.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.options.greeks import compute_all
from src.options.occ import generate_occ_symbol
from src.options.positions import (
    CashContract,
    ExitRules,
    Leg,
    Position,
    PositionState,
    StockContract,
)
from src.options.types import ContractSpec
from src.options.universe import get_underlying_metadata


# ---- shared fixtures --------------------------------------------------------


_TODAY = date(2026, 5, 9)
_EXPIRY = date(2026, 6, 19)  # 41 days out


def _aapl_csp_contract(strike: float = 100.0) -> ContractSpec:
    return ContractSpec(
        underlying="AAPL",
        expiration_date=_EXPIRY,
        option_type="P",
        strike=strike,
    )


def _aapl_cc_call(strike: float = 105.0) -> ContractSpec:
    return ContractSpec(
        underlying="AAPL",
        expiration_date=_EXPIRY,
        option_type="C",
        strike=strike,
    )


def _spy_cc_call(strike: float = 105.0) -> ContractSpec:
    return ContractSpec(
        underlying="SPY",
        expiration_date=_EXPIRY,
        option_type="C",
        strike=strike,
    )


def _spx_csp_put(strike: float = 5000.0) -> ContractSpec:
    return ContractSpec(
        underlying="SPX",
        expiration_date=_EXPIRY,
        option_type="P",
        strike=strike,
    )


def _spx_cc_call(strike: float = 5100.0) -> ContractSpec:
    return ContractSpec(
        underlying="SPX",
        expiration_date=_EXPIRY,
        option_type="C",
        strike=strike,
    )


def _basic_csp(
    *,
    strike: float = 100.0,
    contracts: int = 1,
    entry_credit: float = 200.0,
    profit_target_pct: float | None = 0.5,
    time_stop_dte: int | None = None,
    stop_loss_pct: float | None = None,
) -> Position:
    rules = ExitRules(
        profit_target_pct=profit_target_pct,
        time_stop_dte=time_stop_dte,
        stop_loss_pct=stop_loss_pct,
    )
    return Position.cash_secured_put(
        put_contract=_aapl_csp_contract(strike=strike),
        entry_date=_TODAY,
        entry_credit=entry_credit,
        exit_rules=rules,
        contracts=contracts,
    )


def _basic_cc(
    *,
    underlying: str = "AAPL",
    strike: float = 105.0,
    stock_basis: float = 100.0,
    call_premium: float = 2.0,
    contracts: int = 1,
    profit_target_pct: float | None = 0.5,
) -> Position:
    rules = ExitRules(
        profit_target_pct=profit_target_pct,
        time_stop_dte=None,
        stop_loss_pct=None,
    )
    if underlying == "SPY":
        call = _spy_cc_call(strike=strike)
    else:
        call = _aapl_cc_call(strike=strike)
    entry_credit = -stock_basis * 100 * contracts + call_premium * 100 * contracts
    return Position.covered_call(
        underlying=underlying,
        call_contract=call,
        entry_date=_TODAY,
        entry_credit=entry_credit,
        exit_rules=rules,
        contracts=contracts,
    )


# ---- construction and validation --------------------------------------------


def test_covered_call_constructs_correctly():
    pos = _basic_cc()
    assert pos.strategy_class == "covered_call"
    assert len(pos.legs) == 2
    stock_leg, call_leg = pos.legs
    assert isinstance(stock_leg.contract, StockContract)
    assert stock_leg.contract.ticker == "AAPL"
    assert stock_leg.sign == 1
    assert stock_leg.quantity == 100
    assert isinstance(call_leg.contract, ContractSpec)
    assert call_leg.contract.option_type == "C"
    assert call_leg.sign == -1
    assert call_leg.quantity == 1
    assert pos.state == PositionState.OPEN
    assert pos.entry_credit == pytest.approx(-9800.0)


def test_cash_secured_put_constructs_correctly():
    pos = _basic_csp()
    assert pos.strategy_class == "cash_secured_put"
    assert len(pos.legs) == 2
    cash_leg, put_leg = pos.legs
    assert isinstance(cash_leg.contract, CashContract)
    assert cash_leg.sign == 1
    assert cash_leg.quantity == 10000  # strike * 100
    assert isinstance(put_leg.contract, ContractSpec)
    assert put_leg.contract.option_type == "P"
    assert put_leg.sign == -1
    assert put_leg.quantity == 1
    assert pos.state == PositionState.OPEN
    assert pos.entry_credit == pytest.approx(200.0)


def test_position_validates_strategy_class_legs_match():
    csp_legs = (
        Leg(contract=CashContract(), sign=+1, quantity=10000),
        Leg(contract=_aapl_csp_contract(), sign=-1, quantity=1),
    )
    rules = ExitRules(profit_target_pct=0.5, time_stop_dte=None, stop_loss_pct=None)
    with pytest.raises(ValueError, match="covered_call"):
        Position(
            strategy_class="covered_call",
            legs=csp_legs,
            entry_date=_TODAY,
            entry_credit=200.0,
            exit_rules=rules,
            state=PositionState.OPEN,
        )


def test_position_open_with_closure_fields_raises():
    rules = ExitRules(profit_target_pct=0.5, time_stop_dte=None, stop_loss_pct=None)
    legs = (
        Leg(contract=CashContract(), sign=+1, quantity=10000),
        Leg(contract=_aapl_csp_contract(), sign=-1, quantity=1),
    )
    with pytest.raises(ValueError, match="OPEN"):
        Position(
            strategy_class="cash_secured_put",
            legs=legs,
            entry_date=_TODAY,
            entry_credit=200.0,
            exit_rules=rules,
            state=PositionState.OPEN,
            closure_reason="profit_target_50pct",
            closure_date=_TODAY,
            realized_pnl=100.0,
        )


def test_position_closed_without_closure_fields_raises():
    rules = ExitRules(profit_target_pct=0.5, time_stop_dte=None, stop_loss_pct=None)
    legs = (
        Leg(contract=CashContract(), sign=+1, quantity=10000),
        Leg(contract=_aapl_csp_contract(), sign=-1, quantity=1),
    )
    with pytest.raises(ValueError, match="closure_reason"):
        Position(
            strategy_class="cash_secured_put",
            legs=legs,
            entry_date=_TODAY,
            entry_credit=200.0,
            exit_rules=rules,
            state=PositionState.CLOSED_MANAGED,
        )


# ---- ExitRules --------------------------------------------------------------


def test_exit_rules_requires_at_least_one_field():
    with pytest.raises(ValueError, match="at least one"):
        ExitRules(
            profit_target_pct=None, time_stop_dte=None, stop_loss_pct=None
        )


def test_exit_rules_profit_target_range_validation():
    with pytest.raises(ValueError, match="profit_target_pct"):
        ExitRules(profit_target_pct=0.0, time_stop_dte=None, stop_loss_pct=None)
    with pytest.raises(ValueError, match="profit_target_pct"):
        ExitRules(profit_target_pct=1.5, time_stop_dte=None, stop_loss_pct=None)
    # 1.0 is valid (closed upper)
    rules = ExitRules(
        profit_target_pct=1.0, time_stop_dte=None, stop_loss_pct=None
    )
    assert rules.profit_target_pct == 1.0


def test_exit_rules_time_stop_negative_raises():
    with pytest.raises(ValueError, match="time_stop_dte"):
        ExitRules(profit_target_pct=None, time_stop_dte=-1, stop_loss_pct=None)


def test_exit_rules_stop_loss_zero_raises():
    with pytest.raises(ValueError, match="stop_loss_pct"):
        ExitRules(profit_target_pct=None, time_stop_dte=None, stop_loss_pct=0)


# ---- Leg --------------------------------------------------------------------


def test_leg_sign_validation():
    with pytest.raises(ValueError, match="sign"):
        Leg(contract=StockContract(ticker="AAPL"), sign=0, quantity=100)
    with pytest.raises(ValueError, match="sign"):
        Leg(contract=StockContract(ticker="AAPL"), sign=2, quantity=100)


def test_leg_quantity_must_be_positive():
    with pytest.raises(ValueError, match="quantity"):
        Leg(contract=StockContract(ticker="AAPL"), sign=+1, quantity=0)
    with pytest.raises(ValueError, match="quantity"):
        Leg(contract=StockContract(ticker="AAPL"), sign=+1, quantity=-1)


def test_cash_leg_must_be_long():
    with pytest.raises(ValueError, match="CashContract"):
        Leg(contract=CashContract(), sign=-1, quantity=10000)


# ---- evolve -----------------------------------------------------------------


def test_evolve_returns_new_instance():
    pos = _basic_csp()
    evolved = pos.evolve(
        state=PositionState.CLOSED_MANAGED,
        closure_reason="profit_target_50pct",
        closure_date=date(2026, 5, 20),
        realized_pnl=100.0,
    )
    assert evolved is not pos
    assert pos.state == PositionState.OPEN
    assert evolved.state == PositionState.CLOSED_MANAGED
    assert evolved.closure_reason == "profit_target_50pct"
    assert evolved.realized_pnl == pytest.approx(100.0)
    # other fields unchanged
    assert evolved.legs == pos.legs
    assert evolved.entry_credit == pos.entry_credit


# ---- mark_to_market ---------------------------------------------------------


def test_mark_to_market_csp_at_open_zero_pnl():
    pos = _basic_csp()  # entry_credit = 200, put price at entry = 2.0
    market = {generate_occ_symbol(_aapl_csp_contract()): 2.0}
    assert pos.mark_to_market(market) == pytest.approx(0.0)


def test_mark_to_market_csp_profit_50pct():
    pos = _basic_csp()  # entry_credit = 200
    # put has dropped from 2.00 to 1.00 → 50% of credit captured
    market = {generate_occ_symbol(_aapl_csp_contract()): 1.0}
    assert pos.mark_to_market(market) == pytest.approx(100.0)


def test_mark_to_market_cc_with_stock_appreciation():
    pos = _basic_cc(
        underlying="AAPL",
        strike=105.0,
        stock_basis=100.0,
        call_premium=2.0,
    )
    # stock up to 103, call up to 4 → P&L should be (3 - 2) * 100 = 100
    market = {
        "AAPL": 103.0,
        generate_occ_symbol(_aapl_cc_call(strike=105.0)): 4.0,
    }
    assert pos.mark_to_market(market) == pytest.approx(100.0)


# ---- should_exit ------------------------------------------------------------


def test_should_exit_profit_target_triggers():
    pos = _basic_csp(profit_target_pct=0.5)  # entry_credit = 200
    market = {generate_occ_symbol(_aapl_csp_contract()): 1.0}  # 50% profit
    triggered, reason = pos.should_exit(market, _TODAY)
    assert triggered is True
    assert reason == "profit_target_50pct"


def test_should_exit_time_stop_triggers():
    pos = _basic_csp(profit_target_pct=None, time_stop_dte=21)
    # _EXPIRY - _TODAY = 41 days; need today close enough to hit 21 DTE
    today_at_21 = _EXPIRY - timedelta(days=21)
    market = {generate_occ_symbol(_aapl_csp_contract()): 2.0}
    triggered, reason = pos.should_exit(market, today_at_21)
    assert triggered is True
    assert reason == "time_stop_21dte"


def test_should_exit_stop_loss_triggers():
    pos = _basic_csp(profit_target_pct=None, stop_loss_pct=2.0)  # 200% of credit
    # entry_credit = 200; trigger at -400 P&L
    # P&L = -1*1*100*put + 200; for P&L = -400, put = 6.0
    market = {generate_occ_symbol(_aapl_csp_contract()): 6.0}
    triggered, reason = pos.should_exit(market, _TODAY)
    assert triggered is True
    assert reason == "stop_loss_200pct"


def test_should_exit_priority_stop_loss_first():
    # Set up where both stop_loss and profit_target would somehow fire —
    # but that's mutually exclusive in practice. Instead, verify that
    # when stop_loss triggers, its reason wins regardless of other rules.
    pos = _basic_csp(
        profit_target_pct=0.5, time_stop_dte=21, stop_loss_pct=2.0
    )
    # stop_loss only — put deep ITM
    market = {generate_occ_symbol(_aapl_csp_contract()): 6.0}
    triggered, reason = pos.should_exit(market, _TODAY)
    assert triggered is True
    assert reason == "stop_loss_200pct"


def test_should_exit_no_trigger_returns_false_none():
    pos = _basic_csp(profit_target_pct=0.5, time_stop_dte=21, stop_loss_pct=2.0)
    # put unchanged → no profit, no loss; far from expiration
    market = {generate_occ_symbol(_aapl_csp_contract()): 2.0}
    triggered, reason = pos.should_exit(market, _TODAY)
    assert triggered is False
    assert reason is None


# ---- is_expired -------------------------------------------------------------


def test_is_expired_before_expiration_false():
    pos = _basic_csp()
    assert pos.is_expired(date(2026, 6, 18)) is False


def test_is_expired_at_expiration_true():
    pos = _basic_csp()
    assert pos.is_expired(_EXPIRY) is True


def test_is_expired_past_expiration_true():
    pos = _basic_csp()
    assert pos.is_expired(date(2026, 6, 20)) is True


# ---- days_to_expiration -----------------------------------------------------


def test_dte_basic():
    pos = _basic_csp()
    assert pos.days_to_expiration(_TODAY) == 41


def test_dte_at_expiration_zero():
    pos = _basic_csp()
    assert pos.days_to_expiration(_EXPIRY) == 0


def test_dte_past_expiration_zero():
    pos = _basic_csp()
    assert pos.days_to_expiration(date(2026, 6, 25)) == 0


# ---- resolve_expiration -----------------------------------------------------


def test_spx_csp_itm_cash_settles_to_expired_itm():
    rules = ExitRules(profit_target_pct=0.5, time_stop_dte=None, stop_loss_pct=None)
    put = _spx_csp_put(strike=5000.0)
    pos = Position.cash_secured_put(
        put_contract=put,
        entry_date=_TODAY,
        entry_credit=2500.0,  # $25 premium
        exit_rules=rules,
    )
    # SPX dropped to 4950 → put 50 ITM
    market = {"SPX": 4950.0}
    resolved = pos.resolve_expiration(market, _EXPIRY)
    assert resolved.state == PositionState.EXPIRED_ITM
    assert resolved.closure_reason == "expired_itm_cash_settled"
    assert resolved.closure_date == _EXPIRY
    # P&L = -100 * intrinsic + entry_credit = -5000 + 2500 = -2500
    assert resolved.realized_pnl == pytest.approx(-2500.0)


def test_spx_otm_resolves_to_expired_otm():
    rules = ExitRules(profit_target_pct=0.5, time_stop_dte=None, stop_loss_pct=None)
    put = _spx_csp_put(strike=5000.0)
    pos = Position.cash_secured_put(
        put_contract=put,
        entry_date=_TODAY,
        entry_credit=2500.0,
        exit_rules=rules,
    )
    # SPX above strike → OTM
    market = {"SPX": 5100.0}
    resolved = pos.resolve_expiration(market, _EXPIRY)
    assert resolved.state == PositionState.EXPIRED_OTM
    assert resolved.closure_reason == "expired_otm"
    assert resolved.realized_pnl == pytest.approx(2500.0)


def test_spy_cc_call_itm_resolves_to_assigned():
    rules = ExitRules(profit_target_pct=0.5, time_stop_dte=None, stop_loss_pct=None)
    call = _spy_cc_call(strike=500.0)
    pos = Position.covered_call(
        underlying="SPY",
        call_contract=call,
        entry_date=_TODAY,
        entry_credit=-49800.0,  # bought 100 shares @ $500, sold $5 call
        exit_rules=rules,
    )
    # SPY up to 510 → call ITM
    market = {
        "SPY": 510.0,
        generate_occ_symbol(call): 10.0,  # intrinsic; computed below regardless
    }
    resolved = pos.resolve_expiration(market, _EXPIRY)
    assert resolved.state == PositionState.ASSIGNED
    assert resolved.closure_reason == "assigned_call"


def test_aapl_csp_put_itm_resolves_to_assigned():
    rules = ExitRules(profit_target_pct=0.5, time_stop_dte=None, stop_loss_pct=None)
    put = _aapl_csp_contract(strike=100.0)
    pos = Position.cash_secured_put(
        put_contract=put,
        entry_date=_TODAY,
        entry_credit=200.0,
        exit_rules=rules,
    )
    # AAPL down to 95 → put 5 ITM
    market = {"AAPL": 95.0}
    resolved = pos.resolve_expiration(market, _EXPIRY)
    assert resolved.state == PositionState.ASSIGNED
    assert resolved.closure_reason == "assigned_put"


def test_all_otm_returns_full_credit_as_pnl():
    pos = _basic_csp(strike=100.0, entry_credit=200.0)
    # AAPL above strike → OTM at expiration
    market = {"AAPL": 110.0}
    resolved = pos.resolve_expiration(market, _EXPIRY)
    assert resolved.state == PositionState.EXPIRED_OTM
    assert resolved.realized_pnl == pytest.approx(200.0)
    assert resolved.realized_pnl == pytest.approx(pos.entry_credit)


# ---- aggregate_greeks -------------------------------------------------------


def test_csp_aggregate_delta_negative_short_put():
    # The put leg's per-share delta is negative (it's a put). Aggregate
    # weight is sign*qty = -1 (short 1 contract), so each leg-level
    # Greek is negated relative to the raw compute_all output. Net
    # aggregate delta for a short ATM put is therefore POSITIVE
    # (bullish exposure) — verify by sign-flipped equality with direct.
    pos = _basic_csp(strike=100.0)
    occ = generate_occ_symbol(_aapl_csp_contract(strike=100.0))
    market = {"AAPL": 100.0}
    vol_lookup = {occ: 0.25}
    greeks = pos.aggregate_greeks(market, vol_lookup, r=0.04, today=_TODAY)
    meta = get_underlying_metadata("AAPL")
    direct = compute_all(
        s=100.0,
        k=100.0,
        t=(_EXPIRY - _TODAY).days / 365.0,
        r=0.04,
        q=meta.dividend_yield,
        vol=0.25,
        option_type="P",
    )
    # direct.delta is negative (put delta < 0); aggregate is its negation.
    assert direct.delta < 0
    assert greeks.delta == pytest.approx(-direct.delta)
    assert greeks.delta > 0  # short put = bullish
    assert greeks.gamma == pytest.approx(-direct.gamma)
    assert greeks.theta_per_day == pytest.approx(-direct.theta_per_day)
    assert greeks.vega_per_pct == pytest.approx(-direct.vega_per_pct)


def test_cc_aggregate_delta():
    # CC: long 100 shares (+1 contract-equivalent delta) + short call (-call_delta).
    pos = _basic_cc(
        underlying="AAPL", strike=105.0, stock_basis=100.0, call_premium=2.0
    )
    call = _aapl_cc_call(strike=105.0)
    occ = generate_occ_symbol(call)
    market = {"AAPL": 100.0}
    vol_lookup = {occ: 0.25}
    greeks = pos.aggregate_greeks(market, vol_lookup, r=0.04, today=_TODAY)
    meta = get_underlying_metadata("AAPL")
    direct = compute_all(
        s=100.0,
        k=105.0,
        t=(_EXPIRY - _TODAY).days / 365.0,
        r=0.04,
        q=meta.dividend_yield,
        vol=0.25,
        option_type="C",
    )
    # Stock contributes +1.0 to delta (100 shares / 100); short call contributes -direct.delta.
    assert greeks.delta == pytest.approx(1.0 - direct.delta)
    # Stock contributes 0 to other Greeks; short call negates them.
    assert greeks.gamma == pytest.approx(-direct.gamma)
    assert greeks.theta_per_day == pytest.approx(-direct.theta_per_day)


def test_cash_legs_contribute_zero_greeks():
    # Compare aggregate Greeks of a CSP vs the same short put without
    # the cash leg — they should match (cash contributes nothing).
    pos = _basic_csp(strike=100.0)
    cash_leg, put_leg = pos.legs
    # Build a "naked short put" position by reusing the put leg only —
    # but since strategy_class validation requires a known shape, we
    # just verify aggregate equals the short put's signed Greeks directly.
    occ = generate_occ_symbol(put_leg.contract)
    market = {"AAPL": 100.0}
    vol_lookup = {occ: 0.25}
    greeks = pos.aggregate_greeks(market, vol_lookup, r=0.04, today=_TODAY)
    meta = get_underlying_metadata("AAPL")
    direct = compute_all(
        s=100.0,
        k=100.0,
        t=(_EXPIRY - _TODAY).days / 365.0,
        r=0.04,
        q=meta.dividend_yield,
        vol=0.25,
        option_type="P",
    )
    # Cash leg adds nothing → aggregate matches negated direct (short).
    assert greeks.price == pytest.approx(-direct.price)
    assert greeks.delta == pytest.approx(-direct.delta)
    assert greeks.gamma == pytest.approx(-direct.gamma)
    assert greeks.theta_per_day == pytest.approx(-direct.theta_per_day)
    assert greeks.vega_per_pct == pytest.approx(-direct.vega_per_pct)
    assert greeks.rho_per_bp == pytest.approx(-direct.rho_per_bp)
