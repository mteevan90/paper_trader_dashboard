"""Build the Larger Universe v1 ticker membership map.

The Larger Universe v1 study aims for best-effort survivorship-bias
mitigation over the last 10 years. The universe is constructed as the
UNION of:

  - Current SP500 + SP400 + SP600 constituents (~1,500 tickers)
  - Tickers that were in SP500 / SP400 / SP600 at any point in the last
    10 years but have since been removed (delisted, acquired, renamed,
    or relegated). Expected ~500-1000 additional historical names.

Sources (free, no auth):

  - Wikipedia "List of S&P 500 companies" — current constituents + a
    "Selected changes to the list of S&P 500 components" table that
    lists additions/removals with dates over the last ~10 years.
  - Same for SP400 and SP600 pages.
  - SEC ``company_tickers.json`` for CIK disambiguation of
    currently-listed entities. Tickers whose company name in the
    Wikipedia change history differs from the current SEC-mapped name
    are flagged as potential ticker-reuse cases.

Output: ``docs/larger_universe_v1_universe.json``

Schema (list of records):
    {
      "symbol": "AAPL",
      "company": "Apple Inc.",
      "cik": "0000320193" | null,
      "tier": "SP500" | "SP400" | "SP600",
      "status": "active" | "removed",
      "added_at": "YYYY-MM-DD" | null,
      "removed_at": "YYYY-MM-DD" | null,
      "reason": "free-text reason from Wikipedia" | null,
      "reuse_flag": bool,  // true if Wikipedia history shows >1
                            // distinct company name for this ticker
      "alt_company_names": ["..."]  // populated when reuse_flag is true
    }

Notes:
- Tickers like FB (Facebook → Meta rename 2022-06) will appear once as
  "removed" with company=Facebook,Inc. and tier=SP500. META appears
  separately as the active SP500 entry. The FB record's reuse_flag
  is set if FB's symbol has reappeared on the active list under a
  different company name.
- Delisting dates from Wikipedia are gold-standard for the Phase-3
  OTC-tail truncation: when Finnhub returns post-bankruptcy pink-sheet
  candles for SIVB/FRC/BBBY/etc., the price series gets clipped at
  ``removed_at`` from this map.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_OUT = ROOT / "docs" / "larger_universe_v1_universe.json"

logger = logging.getLogger(__name__)

WIKI = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "SP400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "SP600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_HEADERS = {
    # SEC requires a User-Agent identifying the caller per their access guide.
    "User-Agent": "paper_trader/larger-universe-v1 (research; mteevan90@gmail.com)",
}

# Window for "historical" universe: today - 10 years
WINDOW_YEARS = 10


@dataclass
class TickerRecord:
    symbol: str
    company: Optional[str] = None
    cik: Optional[str] = None
    tier: Optional[str] = None
    status: str = "active"  # "active" or "removed"
    added_at: Optional[str] = None
    removed_at: Optional[str] = None
    reason: Optional[str] = None
    reuse_flag: bool = False
    alt_company_names: list[str] = field(default_factory=list)


# ---------------- SEC CIK map ----------------


def fetch_sec_cik_map(session: requests.Session) -> dict[str, dict]:
    """Return ticker -> {cik, title} from SEC company_tickers.json."""
    r = session.get(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    raw = r.json()
    out: dict[str, dict] = {}
    for _, row in raw.items():
        sym = str(row["ticker"]).upper()
        out[sym] = {
            "cik": f"{int(row['cik_str']):010d}",
            "title": row["title"],
        }
    return out


# ---------------- Wikipedia parsing ----------------


def _read_wiki_html(url: str, session: requests.Session) -> str:
    r = session.get(
        url,
        headers={"User-Agent": SEC_HEADERS["User-Agent"]},
        timeout=30,
    )
    r.raise_for_status()
    return r.text


def parse_current_constituents(html: str, tier: str) -> dict[str, TickerRecord]:
    """Parse the "current constituents" table from a Wikipedia SP500/400/600 page.

    All three pages share the convention that the first wikitable is the
    current constituent list. Column names vary across pages but always
    include something like 'Symbol' / 'Ticker' and 'Security' / 'Company'.
    """
    tables = pd.read_html(io.StringIO(html))
    # Heuristic: the constituent table is large (>50 rows) and has at
    # least a ticker-shaped column and a company column.
    chosen = None
    for tbl in tables:
        if len(tbl) < 50:
            continue
        cols = [str(c).lower() for c in tbl.columns]
        has_ticker = any(("symbol" in c or "ticker" in c) for c in cols)
        has_company = any(("security" in c or "compan" in c or "name" in c) for c in cols)
        if has_ticker and has_company:
            chosen = tbl
            break
    if chosen is None:
        raise RuntimeError(f"could not locate current-constituents table for {tier}")

    cols_lower = {str(c).lower(): c for c in chosen.columns}
    # Identify the ticker and company columns
    tcol = next(c for k, c in cols_lower.items() if "symbol" in k or "ticker" in k)
    ccol = next(
        c for k, c in cols_lower.items()
        if "security" in k or "compan" in k or "name" in k
    )

    out: dict[str, TickerRecord] = {}
    for _, row in chosen.iterrows():
        sym = str(row[tcol]).strip().upper().replace(".", "-")
        if not sym or sym == "NAN":
            continue
        company = str(row[ccol]).strip() if pd.notna(row[ccol]) else None
        out[sym] = TickerRecord(symbol=sym, company=company, tier=tier, status="active")
    return out


# Match dates like 'July 1, 2022' (common Wikipedia format) and 'YYYY-MM-DD'
_DATE_PATTERNS = [
    re.compile(r"^([A-Z][a-z]+\s+\d{1,2},\s+\d{4})$"),
    re.compile(r"^(\d{4}-\d{2}-\d{2})$"),
]


def _to_iso_date(s: object) -> Optional[str]:
    if pd.isna(s):
        return None
    txt = str(s).strip()
    if not txt:
        return None
    try:
        return pd.to_datetime(txt).date().isoformat()
    except Exception:
        return None


def parse_changes_table(html: str, tier: str) -> list[dict]:
    """Extract addition/removal events from the changes table.

    The changes tables on SP500/400/600 Wikipedia pages have a multi-level
    header: top row has 'Date', 'Added', 'Removed', 'Reason'; second row
    splits 'Added' into 'Ticker' / 'Security' and same for 'Removed'.
    pandas.read_html will flatten the MultiIndex; the resulting column
    names are like 'Date Date', 'Added Ticker', 'Added Security',
    'Removed Ticker', 'Removed Security', 'Reason Reason'.
    """
    tables = pd.read_html(io.StringIO(html))
    candidate = None
    for tbl in tables:
        cols = [" ".join([str(c) for c in (col if isinstance(col, tuple) else (col,))]).lower() for col in tbl.columns]
        if any("added" in c for c in cols) and any("removed" in c for c in cols):
            candidate = tbl
            break
    if candidate is None:
        logger.warning("no changes table found for %s", tier)
        return []

    # Flatten columns; build a name lookup
    def flatten(col):
        if isinstance(col, tuple):
            return " ".join(str(c) for c in col if str(c).lower() != "nan").strip()
        return str(col).strip()

    candidate.columns = [flatten(c) for c in candidate.columns]
    cols_lower = {c.lower(): c for c in candidate.columns}

    def find_col(*keys):
        for c_lower, c_orig in cols_lower.items():
            if all(k in c_lower for k in keys):
                return c_orig
        return None

    date_col = find_col("date")
    added_tk = find_col("added", "ticker")
    added_co = find_col("added", "security") or find_col("added", "compan") or find_col("added", "name")
    removed_tk = find_col("removed", "ticker")
    removed_co = find_col("removed", "security") or find_col("removed", "compan") or find_col("removed", "name")
    reason_col = find_col("reason")

    out: list[dict] = []
    if not (date_col and (added_tk or removed_tk)):
        logger.warning("expected columns missing in changes table for %s", tier)
        return out

    for _, row in candidate.iterrows():
        iso = _to_iso_date(row[date_col]) if date_col else None
        if not iso:
            continue
        reason = str(row[reason_col]).strip() if reason_col and pd.notna(row[reason_col]) else None
        if added_tk and pd.notna(row[added_tk]):
            sym = str(row[added_tk]).strip().upper().replace(".", "-")
            co = str(row[added_co]).strip() if added_co and pd.notna(row[added_co]) else None
            out.append({"event": "added", "tier": tier, "date": iso,
                        "symbol": sym, "company": co, "reason": reason})
        if removed_tk and pd.notna(row[removed_tk]):
            sym = str(row[removed_tk]).strip().upper().replace(".", "-")
            co = str(row[removed_co]).strip() if removed_co and pd.notna(row[removed_co]) else None
            out.append({"event": "removed", "tier": tier, "date": iso,
                        "symbol": sym, "company": co, "reason": reason})
    return out


# ---------------- assembly ----------------


def build_universe() -> list[TickerRecord]:
    cutoff = (datetime.utcnow().date() - pd.DateOffset(years=WINDOW_YEARS)).date() \
        if hasattr(pd.DateOffset(years=WINDOW_YEARS), "date") \
        else (datetime.utcnow() - pd.DateOffset(years=WINDOW_YEARS)).date()
    # Above is defensive; simpler form:
    cutoff = (pd.Timestamp.utcnow().normalize() - pd.DateOffset(years=WINDOW_YEARS)).date()

    sess = requests.Session()
    logger.info("fetching SEC company_tickers.json...")
    sec_map = fetch_sec_cik_map(sess)
    logger.info("SEC map: %d tickers", len(sec_map))

    active: dict[str, TickerRecord] = {}
    all_events: list[dict] = []

    for tier, url in WIKI.items():
        logger.info("fetching wikipedia for %s...", tier)
        html = _read_wiki_html(url, sess)
        time.sleep(0.5)
        cur = parse_current_constituents(html, tier)
        logger.info("  %s current constituents: %d", tier, len(cur))
        for sym, rec in cur.items():
            # Tier precedence: SP500 > SP400 > SP600 (a ticker shouldn't
            # appear in multiple, but defensively pick the larger-cap tier)
            if sym in active:
                # don't overwrite a higher-tier classification
                tier_rank = {"SP500": 3, "SP400": 2, "SP600": 1}
                if tier_rank[tier] > tier_rank.get(active[sym].tier, 0):
                    active[sym].tier = tier
            else:
                active[sym] = rec
        events = parse_changes_table(html, tier)
        logger.info("  %s changes events: %d", tier, len(events))
        all_events.extend(events)

    # Annotate active with SEC CIK
    for sym, rec in active.items():
        sec = sec_map.get(sym)
        if sec is not None:
            rec.cik = sec["cik"]
            # If SEC's title disagrees with Wikipedia's, prefer SEC (more authoritative)
            if not rec.company:
                rec.company = sec["title"]

    # Process events. We want:
    #   - tickers added & later removed in the last 10y are "removed"
    #   - tickers removed in the last 10y that are NOT currently active = "removed"
    #   - tickers removed but now reappear under a different company = ticker reuse
    removed: dict[str, TickerRecord] = {}
    for ev in sorted(all_events, key=lambda e: e["date"]):
        ev_date = pd.to_datetime(ev["date"]).date()
        if ev_date < cutoff:
            continue
        sym = ev["symbol"]
        if ev["event"] == "removed":
            # Only include if not currently active under the same company name
            cur_active = active.get(sym)
            if cur_active is not None:
                # Possible ticker reuse: check company name
                if (ev.get("company") and cur_active.company and
                        _looks_like_different_company(ev["company"], cur_active.company)):
                    cur_active.reuse_flag = True
                    if ev["company"] not in cur_active.alt_company_names:
                        cur_active.alt_company_names.append(ev["company"])
                    # And create a "removed" record for the old entity
                    key = f"{sym}::{ev['company']}"
                    removed[key] = TickerRecord(
                        symbol=sym, company=ev.get("company"),
                        tier=ev["tier"], status="removed",
                        removed_at=ev["date"], reason=ev.get("reason"),
                        reuse_flag=True,
                    )
                else:
                    # The symbol is currently active under the same company.
                    # Maybe it was removed and re-added; don't double-count
                    # as removed.
                    pass
            else:
                key = f"{sym}::{ev.get('company') or ''}"
                if key not in removed:
                    removed[key] = TickerRecord(
                        symbol=sym, company=ev.get("company"),
                        tier=ev["tier"], status="removed",
                        removed_at=ev["date"], reason=ev.get("reason"),
                    )
                else:
                    # If we already saw an add+remove for this entity,
                    # update removed_at to the latest event.
                    removed[key].removed_at = ev["date"]
                    if not removed[key].reason and ev.get("reason"):
                        removed[key].reason = ev["reason"]
        elif ev["event"] == "added":
            cur_active = active.get(sym)
            if cur_active is not None:
                if not cur_active.added_at or cur_active.added_at < ev["date"]:
                    cur_active.added_at = ev["date"]
            else:
                # An "added" without subsequent remove and not currently active
                # is unusual (could be a same-day rename event); record it.
                key = f"{sym}::{ev.get('company') or ''}"
                if key in removed:
                    removed[key].added_at = ev["date"]
                else:
                    # Treat as a transient entry; create a record we can amend
                    removed[key] = TickerRecord(
                        symbol=sym, company=ev.get("company"),
                        tier=ev["tier"], status="removed",
                        added_at=ev["date"], reason=ev.get("reason"),
                    )

    all_records = list(active.values()) + list(removed.values())
    return all_records


def _looks_like_different_company(name_a: str, name_b: str) -> bool:
    """Heuristic: True if two company-name strings refer to different entities.

    Normalizes by lowercasing, stripping common suffixes (Inc., Corp.,
    Common Stock, Class A, etc.) and punctuation, then compares tokens.
    """
    def norm(s: str) -> set[str]:
        s2 = s.lower()
        for stop in ["inc.", "inc", "corp.", "corp", "corporation",
                     "company", "co.", "co", "common stock", "class a",
                     "class b", "ltd.", "ltd", "plc", ",", ".", "&", "the "]:
            s2 = s2.replace(stop, " ")
        return {tok for tok in s2.split() if tok}
    a = norm(name_a)
    b = norm(name_b)
    if not a or not b:
        return False
    overlap = a & b
    # If the names share at least one significant token, treat as same entity
    return len(overlap) == 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    records = build_universe()
    UNIVERSE_OUT.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_OUT.write_text(
        json.dumps([asdict(r) for r in records], indent=2),
        encoding="utf-8",
    )

    # Summary stats for the chat report
    active = [r for r in records if r.status == "active"]
    removed = [r for r in records if r.status == "removed"]
    by_tier_active = {}
    for r in active:
        by_tier_active.setdefault(r.tier, 0)
        by_tier_active[r.tier] += 1
    by_tier_removed = {}
    for r in removed:
        by_tier_removed.setdefault(r.tier, 0)
        by_tier_removed[r.tier] += 1
    reuse_count = sum(1 for r in active if r.reuse_flag)
    cik_known = sum(1 for r in active if r.cik)

    print()
    print(f"=== Larger Universe v1 — Wikipedia + SEC build summary ===")
    print(f"Total records:                 {len(records)}")
    print(f"Active (currently in index):   {len(active)}")
    for t in ("SP500", "SP400", "SP600", None):
        print(f"  {str(t):8s}                     {by_tier_active.get(t, 0)}")
    print(f"  with SEC CIK matched:        {cik_known}")
    print(f"  ticker-reuse flagged:        {reuse_count}")
    print(f"Removed (last 10y):            {len(removed)}")
    for t in ("SP500", "SP400", "SP600", None):
        print(f"  {str(t):8s}                     {by_tier_removed.get(t, 0)}")
    print(f"\nWrote: {UNIVERSE_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
