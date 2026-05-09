"""Public API for the crypto universe module.

The stub implementations here are backed by ``STATIC_UNIVERSE``. The
survivorship-aware sibling, ``get_universe_at_date_v2``, has its
signature locked here so Sections 2/4/5 can build against it; the
implementation lands in Section 3.
"""

from datetime import date

from src.crypto.static_universe import STATIC_UNIVERSE
from src.crypto.types import TokenMeta


_BY_ID: dict[str, TokenMeta] = {t.coingecko_id: t for t in STATIC_UNIVERSE}


def get_universe_at_date(at_date: date, top_n: int) -> list[TokenMeta]:
    """Return the top-N tokens active on ``at_date``, ranked by market cap.

    STUB: Returns the static universe, optionally truncated to ``top_n``.
    The survivorship-aware implementation lives in
    ``get_universe_at_date_v2`` (Section 3). Callers should not pass
    ``at_date`` earlier than the earliest ``listing_date`` in the static
    universe.
    """
    active = [t for t in STATIC_UNIVERSE if is_token_active(t.coingecko_id, at_date)]
    return active[:top_n]


def is_token_active(coingecko_id: str, at_date: date) -> bool:
    """True if the token was listed and not yet delisted on ``at_date``."""
    token = _BY_ID.get(coingecko_id)
    if token is None:
        return False
    if at_date < token.listing_date:
        return False
    if token.delisting_date is not None and at_date >= token.delisting_date:
        return False
    return True


def get_token_metadata(coingecko_id: str) -> TokenMeta:
    """Lookup by CoinGecko ID. Raises KeyError if not in the universe."""
    return _BY_ID[coingecko_id]


def get_universe_at_date_v2(at_date: date, top_n: int) -> list[TokenMeta]:
    """Survivorship-aware version backed by point-in-time CoinGecko data.

    Will be implemented in Section 3 of the crypto extension. The
    function signature and return shape are locked here so Sections
    2/4/5 can build against it without waiting on the multi-day
    historical fetch.
    """
    raise NotImplementedError(
        "Survivorship-aware universe construction is Section 3 of the "
        "crypto extension. Until it lands, callers should use "
        "get_universe_at_date() with the static stub universe and "
        "treat any alpha numbers as plumbing-verification only."
    )
