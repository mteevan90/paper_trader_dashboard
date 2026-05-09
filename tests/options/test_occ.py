"""Tests for the OCC option symbol parser/generator (Phase 2 Section 1).

No network. Validates the strict 21-char in/out contract and the
round-trip property.
"""

from datetime import date

import pytest

from src.options.occ import generate_occ_symbol, parse_occ_symbol
from src.options.types import ContractSpec


def test_parse_aapl_call_270():
    spec = parse_occ_symbol("AAPL  220617C00270000")
    assert spec == ContractSpec(
        underlying="AAPL",
        expiration_date=date(2022, 6, 17),
        option_type="C",
        strike=270.0,
    )


def test_parse_spx_put():
    spec = parse_occ_symbol("SPX   220617P04500000")
    assert spec == ContractSpec(
        underlying="SPX",
        expiration_date=date(2022, 6, 17),
        option_type="P",
        strike=4500.0,
    )


def test_generate_aapl_call_270():
    spec = ContractSpec(
        underlying="AAPL",
        expiration_date=date(2022, 6, 17),
        option_type="C",
        strike=270.0,
    )
    assert generate_occ_symbol(spec) == "AAPL  220617C00270000"


@pytest.mark.parametrize("symbol", [
    "AAPL  220617C00270000",
    "SPX   220617P04500000",
    "NVDA  240119C00500000",
])
def test_round_trip_parse_then_generate(symbol):
    assert generate_occ_symbol(parse_occ_symbol(symbol)) == symbol


@pytest.mark.parametrize("spec", [
    ContractSpec("AAPL", date(2022, 6, 17), "C", 270.0),
    ContractSpec("SPX", date(2022, 6, 17), "P", 4500.0),
    ContractSpec("NVDA", date(2024, 1, 19), "C", 500.0),
])
def test_round_trip_generate_then_parse(spec):
    assert parse_occ_symbol(generate_occ_symbol(spec)) == spec


def test_round_trip_spx_strike_4500():
    """Round-trip a $4500 strike to confirm float64 precision holds at
    SPX-scale strikes, not just AAPL-scale."""
    spec = ContractSpec(
        underlying="SPX",
        expiration_date=date(2022, 6, 17),
        option_type="C",
        strike=4500.0,
    )
    symbol = generate_occ_symbol(spec)
    assert symbol == "SPX   220617C04500000"
    assert parse_occ_symbol(symbol) == spec


def test_parse_rejects_short_input():
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL  220617C0027000")  # 20 chars


def test_parse_rejects_long_input():
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL  220617C002700000")  # 22 chars


def test_parse_rejects_empty_underlying():
    with pytest.raises(ValueError):
        parse_occ_symbol("      220617C00270000")


def test_parse_rejects_bad_option_type():
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL  220617X00270000")


def test_parse_rejects_invalid_date():
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL  999999C00270000")


def test_parse_rejects_non_numeric_strike():
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL  220617C0027000A")


def test_generate_rejects_oversized_underlying():
    spec = ContractSpec(
        underlying="LONGSYM",  # 7 chars
        expiration_date=date(2022, 6, 17),
        option_type="C",
        strike=270.0,
    )
    with pytest.raises(ValueError):
        generate_occ_symbol(spec)


def test_generate_rejects_oversized_strike():
    spec = ContractSpec(
        underlying="AAPL",
        expiration_date=date(2022, 6, 17),
        option_type="C",
        strike=100000.0,
    )
    with pytest.raises(ValueError):
        generate_occ_symbol(spec)


def test_contract_spec_rejects_invalid_option_type():
    with pytest.raises(ValueError):
        ContractSpec(
            underlying="AAPL",
            expiration_date=date(2022, 6, 17),
            option_type="X",
            strike=270.0,
        )


def test_contract_spec_rejects_zero_strike():
    with pytest.raises(ValueError):
        ContractSpec(
            underlying="AAPL",
            expiration_date=date(2022, 6, 17),
            option_type="C",
            strike=0.0,
        )


def test_contract_spec_rejects_negative_strike():
    with pytest.raises(ValueError):
        ContractSpec(
            underlying="AAPL",
            expiration_date=date(2022, 6, 17),
            option_type="C",
            strike=-1.0,
        )
