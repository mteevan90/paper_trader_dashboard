"""openinsider_signals.py — Insider cluster-buying signal.

First real alt signal in the bucket created by segment 12. Scrapes
openinsider.com's screener for purchase transactions, filters to senior
insiders (CEO / CFO / Director / President), and produces a 0-1 score
per ticker per date based on cluster intensity (number of unique senior
insiders × log purchase value, then universe rank-normalized).

Cache (under ``models/cache/openinsider/``) is append-only and stores
ALL transactions globally (not pre-filtered to UNIVERSE_TICKERS) so the
universe can be extended later without re-scraping.

Score function: ``score_openinsider_clusters(tickers, date)`` returns
``dict[ticker, float in [0.5, 1.0]]`` for tickers with qualifying
cluster activity, or ``0.5`` (neutral, matches alt-bucket convention)
for tickers with no qualifying activity.
"""

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE_URL = "http://openinsider.com/screener"
USER_AGENT = "paper-trader-research/0.1 (personal-use)"

# Trade-date range query — verified working in segment 14 probe. fd=0 / fdr
# does NOT honor the date filter; td=13 / tdr does.
SOURCE_PARAMS = {
    "td": "13",                        # custom trade-date range
    "xp": "1",                         # purchases only
    "xs": "0",                         # exclude sales
    "excludeDerivRelated": "1",        # exclude options/RSU mechanical events
    "xt": "",
    "xs1": "0", "xs2": "0", "xs3": "0", "xs4": "0", "xs5": "0", "xs6": "0",
    "cnt": "5000",                     # raise page-size cap
}
# Source URL pattern recorded in sidecar for provenance:
SOURCE_URL_PATTERN = (
    "http://openinsider.com/screener?td=13&tdr={start} - {end}"
    "&xp=1&xs=0&excludeDerivRelated=1&cnt=5000"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(
    BASE_DIR, "..", "models", "cache", "openinsider"))
CACHE_PARQUET = os.path.join(CACHE_DIR, "insider_transactions.parquet")
CACHE_META    = os.path.join(CACHE_DIR, "insider_transactions.meta.json")

CACHE_VERSION = "v1"

# Throttle + backoff per scope spec
THROTTLE_SECONDS  = 0.5             # 2 req/s ceiling
BACKOFF_DELAYS    = (5, 10, 20)     # seconds
REQUEST_TIMEOUT   = 30

# Score function parameters
LOOKBACK_DAYS         = 90
MIN_SENIOR_INSIDERS   = 2
MIN_TOTAL_VALUE_USD   = 10_000


# ---------------------------------------------------------------------------
# Senior-insider regex (segment 14: corrected for OpenInsider's abbreviations)
# ---------------------------------------------------------------------------

_DIR_OF_PATTERN          = re.compile(r"\bDir(?:ector)?\s+of\b", re.I)
_MANAGING_DIR_PATTERN    = re.compile(r"\bManaging\s+Director\b", re.I)
_VP_PATTERN              = re.compile(r"\bVice\s+President\b|\bV\.?P\.?\b", re.I)
_CHIEF_LEVEL_PATTERN     = re.compile(
    r"\b(CEO|CFO|COO|Chairman)\b|"
    r"\bChief\s+(Executive|Financial|Operating)\b", re.I)


def is_senior_insider(title: str) -> bool:
    """Return True when ``title`` represents a senior board/C-suite role
    in the sense the cluster-buying signal cares about: CEO, CFO,
    Director, President. Designed for OpenInsider's abbreviation-heavy
    title format (``Dir``, ``Pres``, ``Pres, CEO``, ``Dir, 10%``, etc.)."""
    t = (title or "").strip()
    if not t:
        return False

    # Reject operational-role false positives that contain senior keywords:
    #   "EVP, Dir of Trust Services" — operational, not board
    #   "Managing Director" — investment-banking title, not board
    if _DIR_OF_PATTERN.search(t):
        return False
    if _MANAGING_DIR_PATTERN.search(t):
        return False
    # "Vice President" alone isn't senior in this signal context, but
    # "VP & CEO" is. So reject VP only if no chief-level keyword is also
    # present.
    if _VP_PATTERN.search(t) and not _CHIEF_LEVEL_PATTERN.search(t):
        return False

    # Tokenize on commas / slashes / ampersands and exact-match abbreviations.
    tokens = [tok.strip() for tok in re.split(r"[,/&]", t) if tok.strip()]
    for tok in tokens:
        tok_l = tok.lower()
        if tok_l in ("ceo", "cfo", "coo", "pres", "president", "chairman"):
            return True
        if tok_l in ("dir", "director"):
            return True
        if tok_l.startswith("chief executive") \
           or tok_l.startswith("chief financial") \
           or tok_l.startswith("chief operating"):
            return True
    return False


# ---------------------------------------------------------------------------
# HTTP fetch + parse
# ---------------------------------------------------------------------------

def _parse_value(v) -> float | None:
    """Convert OpenInsider value string ('+$12,549') to float USD."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).replace("$", "").replace(",", "").replace("+", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fetch_chunk(start: str, end: str) -> pd.DataFrame:
    """Fetch + parse one trade-date-range chunk. Throttled + backoff on
    429/5xx. Returns a clean per-row DataFrame; raises after 3 backoffs."""
    params = {**SOURCE_PARAMS, "tdr": f"{start} - {end}"}
    headers = {"User-Agent": USER_AGENT}

    last_err: Exception | None = None
    for attempt, delay in enumerate([0] + list(BACKOFF_DELAYS)):
        if delay:
            print(f"  [OPENINSIDER] backoff {delay}s before attempt {attempt + 1}")
            time.sleep(delay)
        try:
            r = requests.get(SOURCE_URL, params=params, headers=headers,
                             timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}")
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"chunk {start}->{end} failed after retries: {last_err}")

    # Parse HTML — main data table is always the largest by cell count.
    tables = pd.read_html(StringIO(r.text))
    main = max(tables, key=lambda t: t.shape[0] * t.shape[1])
    main.columns = [str(c).replace("\xa0", " ").strip() for c in main.columns]

    needed = ["Trade Date", "Ticker", "Insider Name", "Title",
              "Trade Type", "Qty", "Value"]
    missing = [c for c in needed if c not in main.columns]
    if missing:
        raise RuntimeError(f"chunk {start}->{end}: missing columns {missing}; "
                           f"got {list(main.columns)}")

    df = pd.DataFrame({
        "ticker":           main["Ticker"].astype(str).str.upper().str.strip(),
        "transaction_date": pd.to_datetime(main["Trade Date"], errors="coerce"),
        "insider_name":     main["Insider Name"].astype(str).str.strip(),
        "insider_title":    main["Title"].astype(str).str.strip(),
        "transaction_type": main["Trade Type"].astype(str).str.strip(),
        "shares":           pd.to_numeric(main["Qty"].astype(str)
                                          .str.replace(",", "")
                                          .str.replace("+", ""),
                                          errors="coerce"),
        "value":            main["Value"].apply(_parse_value),
    })
    df = df[df["transaction_date"].notna()
            & df["ticker"].ne("")
            & df["ticker"].ne("nan")]
    return df


# ---------------------------------------------------------------------------
# Cache build (append-only)
# ---------------------------------------------------------------------------

def _load_meta() -> dict | None:
    if not os.path.exists(CACHE_META):
        return None
    try:
        with open(CACHE_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_meta(meta: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_META + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp, CACHE_META)


def _save_parquet(df: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_PARQUET + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, CACHE_PARQUET)


def _chunk_ranges(start: pd.Timestamp, end: pd.Timestamp,
                  chunk_days: int = 90) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end)
        out.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(days=1)
    return out


def build_cache(start: str = "2018-01-01", end: str | None = None,
                chunk_days: int = 90) -> pd.DataFrame:
    """Append-only cache build.

    On empty cache: full fetch from ``start`` to ``end``.
    On existing cache: only fetches the gap between ``meta.date_range_covered``
    and ``[start, end]``, then merges + dedups + saves.

    Returns the full cached DataFrame (sorted by ticker + transaction_date).
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    req_start = pd.Timestamp(start)
    req_end   = pd.Timestamp(end)

    existing: pd.DataFrame | None = None
    meta = _load_meta()
    if meta and meta.get("version") == CACHE_VERSION \
       and os.path.exists(CACHE_PARQUET):
        existing = pd.read_parquet(CACHE_PARQUET)
        cov = meta.get("date_range_covered", {})
        cov_start = pd.Timestamp(cov.get("start", req_start))
        cov_end   = pd.Timestamp(cov.get("end",   req_start))
        # Compute the gap to fetch on each side
        chunks_to_fetch: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if req_start < cov_start:
            chunks_to_fetch += _chunk_ranges(
                req_start, cov_start - pd.Timedelta(days=1), chunk_days)
        if req_end > cov_end:
            chunks_to_fetch += _chunk_ranges(
                cov_end + pd.Timedelta(days=1), req_end, chunk_days)
        if not chunks_to_fetch:
            print(f"  [OPENINSIDER] cache covers {cov_start.date()} -> "
                  f"{cov_end.date()}, request {req_start.date()} -> "
                  f"{req_end.date()} satisfied; no fetch")
            return existing
        print(f"  [OPENINSIDER] cache covers {cov_start.date()} -> "
              f"{cov_end.date()}; fetching {len(chunks_to_fetch)} new "
              f"chunk(s) for the gap")
    else:
        chunks_to_fetch = _chunk_ranges(req_start, req_end, chunk_days)
        if meta:
            print(f"  [OPENINSIDER] cache version mismatch "
                  f"({meta.get('version')!r} vs {CACHE_VERSION!r}) — "
                  f"rebuilding from scratch")
        else:
            print(f"  [OPENINSIDER] no cache found — fetching "
                  f"{len(chunks_to_fetch)} chunk(s) "
                  f"({req_start.date()} -> {req_end.date()})")

    pieces: list[pd.DataFrame] = []
    if existing is not None and not existing.empty:
        pieces.append(existing)

    t0 = time.perf_counter()
    for i, (cs, ce) in enumerate(chunks_to_fetch, 1):
        cs_str = cs.strftime("%Y-%m-%d")
        ce_str = ce.strftime("%Y-%m-%d")
        print(f"  [OPENINSIDER] {i:>3}/{len(chunks_to_fetch)}  "
              f"{cs_str} -> {ce_str}", end="  ", flush=True)
        sub = _fetch_chunk(cs_str, ce_str)
        print(f"rows={len(sub):>5}")
        pieces.append(sub)
        time.sleep(THROTTLE_SECONDS)

    df = pd.concat(pieces, ignore_index=True)
    # Deduplicate on the natural key
    df = df.drop_duplicates(
        subset=["ticker", "transaction_date", "insider_name", "value"],
        keep="last",
    )
    df = df.sort_values(["ticker", "transaction_date"]).reset_index(drop=True)

    _save_parquet(df)
    new_meta = {
        "version": CACHE_VERSION,
        "date_range_covered": {
            "start": str(min(pd.Timestamp(start),
                             pd.Timestamp(meta["date_range_covered"]["start"])
                             if meta else req_start).date()),
            "end":   str(max(pd.Timestamp(end),
                             pd.Timestamp(meta["date_range_covered"]["end"])
                             if meta else req_end).date()),
        },
        "fetched_at":   datetime.now(timezone.utc).isoformat(),
        "n_rows":       int(len(df)),
        "chunk_size_days":   chunk_days,
        "source_url_pattern": SOURCE_URL_PATTERN,
        "filters":      {"purchases_only": True,
                          "exclude_deriv_related": True},
    }
    _save_meta(new_meta)

    print(f"  [OPENINSIDER] cache saved: {len(df):,} rows, "
          f"{time.perf_counter() - t0:.1f}s wall")
    return df


# ---------------------------------------------------------------------------
# In-memory access (lazily loaded; per-ticker pre-grouped for fast scoring)
# ---------------------------------------------------------------------------

class _Loaded:
    df: pd.DataFrame | None = None
    by_ticker: dict[str, pd.DataFrame] = {}


def _ensure_loaded() -> None:
    if _Loaded.df is not None:
        return
    if not os.path.exists(CACHE_PARQUET):
        # Empty state — score function will return 0.5 neutral for all
        _Loaded.df = pd.DataFrame(columns=[
            "ticker", "transaction_date", "insider_name",
            "insider_title", "transaction_type", "shares", "value",
        ])
        _Loaded.by_ticker = {}
        return
    df = pd.read_parquet(CACHE_PARQUET)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    _Loaded.df = df
    # Pre-group by ticker for fast per-ticker window queries
    _Loaded.by_ticker = {
        tkr: sub.sort_values("transaction_date")
        for tkr, sub in df.groupby("ticker", sort=False)
    }


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def _ticker_cluster_score(tkr: str, date_ts: pd.Timestamp,
                          window_start: pd.Timestamp) -> float:
    """Raw cluster intensity for one ticker. 0.0 = no qualifying activity."""
    sub = _Loaded.by_ticker.get(tkr)
    if sub is None or sub.empty:
        return 0.0
    mask = ((sub["transaction_date"] >= window_start)
            & (sub["transaction_date"] <= date_ts))
    win = sub[mask]
    if win.empty:
        return 0.0
    senior = win[win["insider_title"].apply(is_senior_insider)]
    if senior.empty:
        return 0.0
    n_senior = senior["insider_name"].nunique()
    total_value = senior["value"].sum(skipna=True)
    if pd.isna(total_value):
        total_value = 0.0
    if n_senior < MIN_SENIOR_INSIDERS or total_value < MIN_TOTAL_VALUE_USD:
        return 0.0
    return float(n_senior) * math.log1p(float(total_value))


def score_openinsider_clusters(tickers: list[str], date) -> dict[str, float]:
    """Public: score the requested universe at ``date``.

    For each ticker, looks at insider purchases over the last 90 days,
    counts unique senior insiders, sums purchase value, and combines as
    ``n_senior × log(1 + total_value)``. Tickers below the qualifying
    floor (≥2 unique senior insiders AND ≥$10K combined value) get the
    neutral 0.5; qualifying tickers are universe-rank-normalized to
    [0.5, 1.0] within ``tickers`` at this date.
    """
    _ensure_loaded()
    date_ts = pd.Timestamp(date)
    window_start = date_ts - pd.Timedelta(days=LOOKBACK_DAYS)

    raw: dict[str, float] = {}
    for t in tickers:
        try:
            raw[t] = _ticker_cluster_score(t, date_ts, window_start)
        except Exception as e:
            # A bad row in the cache shouldn't kill the whole bucket.
            print(f"  [OPENINSIDER] {t} score error at {date_ts.date()}: {e}")
            raw[t] = 0.0

    # Rank-normalize active tickers to [0.5, 1.0]; inactive stay at 0.5.
    active_items = [(t, v) for t, v in raw.items() if v > 0.0]
    out: dict[str, float] = {t: 0.5 for t in tickers}
    if not active_items:
        return out

    active_items.sort(key=lambda x: x[1])
    n_active = len(active_items)
    for i, (t, _) in enumerate(active_items):
        rank_norm = i / max(n_active - 1, 1)   # [0, 1]
        out[t] = 0.5 + 0.5 * rank_norm         # [0.5, 1.0]
    return out


# ---------------------------------------------------------------------------
# CLI for cache build / refresh
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenInsider cache builder")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end",   default=None,
                        help="default = today")
    parser.add_argument("--chunk-days", type=int, default=90)
    args = parser.parse_args()
    df = build_cache(args.start, args.end, args.chunk_days)
    print(f"\nCache rows: {len(df):,}")
    if not df.empty:
        print(f"Date range: {df['transaction_date'].min().date()} -> "
              f"{df['transaction_date'].max().date()}")
        print(f"Unique tickers: {df['ticker'].nunique():,}")
