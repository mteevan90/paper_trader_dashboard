"""OCC option symbol parser and generator.

OCC standard symbol format (21 characters):

    <UNDERLYING (6 chars, right-padded with spaces)>
    <YYMMDD (6 chars)>
    <C|P (1 char)>
    <STRIKE (8 digits, integer thousandths of a dollar, zero-padded)>

Example:
    ``AAPL  220617C00270000`` ↔ ContractSpec(
        underlying="AAPL", expiration_date=date(2022, 6, 17),
        option_type="C", strike=270.0)
"""

from datetime import date

from src.options.types import ContractSpec, _VALID_OPTION_TYPES


_OCC_LEN = 21
_UNDERLYING_LEN = 6
_DATE_LEN = 6
_TYPE_LEN = 1
_STRIKE_LEN = 8
_STRIKE_DIVISOR = 1000
_STRIKE_MAX_VALUE = 99999.999  # 8 digits ÷ 1000 = max representable strike


def parse_occ_symbol(s: str) -> ContractSpec:
    """Parse a 21-character OCC option symbol into a ``ContractSpec``.

    Strict: rejects any input that is not exactly 21 characters or that
    fails any field-level validation. Raises ``ValueError`` on
    malformed input.
    """
    if len(s) != _OCC_LEN:
        raise ValueError(
            f"OCC symbol must be {_OCC_LEN} chars; got {len(s)}: {s!r}")

    underlying = s[0:_UNDERLYING_LEN].rstrip()
    if not underlying:
        raise ValueError(f"OCC symbol has empty underlying: {s!r}")

    date_str = s[_UNDERLYING_LEN:_UNDERLYING_LEN + _DATE_LEN]
    try:
        # OCC YYMMDD: the 2000-prefix assumption is valid through 2099.
        # The OCC symbology standard was finalized for post-2010 use, so
        # there is no historical YY < 70 ambiguity to handle. Revisit if
        # this codebase is still alive in 2099.
        year = 2000 + int(date_str[0:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        expiration_date = date(year, month, day)
    except ValueError as e:
        raise ValueError(
            f"OCC symbol has invalid expiration date {date_str!r}: {s!r}"
        ) from e

    option_type = s[_UNDERLYING_LEN + _DATE_LEN]
    if option_type not in _VALID_OPTION_TYPES:
        raise ValueError(
            f"OCC symbol has invalid option type {option_type!r}; "
            f"expected one of {_VALID_OPTION_TYPES}: {s!r}")

    strike_str = s[_UNDERLYING_LEN + _DATE_LEN + _TYPE_LEN:]
    try:
        strike_thousandths = int(strike_str)
    except ValueError as e:
        raise ValueError(
            f"OCC symbol has non-numeric strike {strike_str!r}: {s!r}"
        ) from e
    strike = strike_thousandths / _STRIKE_DIVISOR

    return ContractSpec(
        underlying=underlying,
        expiration_date=expiration_date,
        option_type=option_type,
        strike=strike,
    )


def generate_occ_symbol(spec: ContractSpec) -> str:
    """Generate a 21-character OCC option symbol from a ``ContractSpec``.

    Round-trips with ``parse_occ_symbol``: for any valid spec,
    ``parse_occ_symbol(generate_occ_symbol(spec)) == spec``.

    Raises ``ValueError`` if the underlying is too long or the strike
    overflows the 8-digit integer-thousandths field.
    """
    if len(spec.underlying) > _UNDERLYING_LEN:
        raise ValueError(
            f"underlying {spec.underlying!r} exceeds {_UNDERLYING_LEN} chars")
    if spec.strike > _STRIKE_MAX_VALUE:
        raise ValueError(
            f"strike {spec.strike!r} exceeds OCC max {_STRIKE_MAX_VALUE}")

    underlying_field = spec.underlying.ljust(_UNDERLYING_LEN)
    date_field = spec.expiration_date.strftime("%y%m%d")
    strike_thousandths = int(round(spec.strike * _STRIKE_DIVISOR))
    strike_field = f"{strike_thousandths:0{_STRIKE_LEN}d}"

    return (underlying_field + date_field + spec.option_type + strike_field)
