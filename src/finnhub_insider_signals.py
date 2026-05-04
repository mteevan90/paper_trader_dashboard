"""finnhub_insider_signals.py — Insider cluster-buying signal (all-insider).

STATUS: SOLE ALT SIGNAL pending future bucket population.

Segment 14 shipped this end-to-end. Segment 15 attempted to add a
second alt signal across four data sources (Finnhub
recommendation_trends, OpenInsider screener, FINRA RegSHO,
yfinance institutional_holders) — all four hit hard limitations
documented in ``models/cache/alt_signals_phase1_summary.md``.

The locked finding from Phase 1 (this signal alone): bit-identical
to Phase 0 baseline, because 15% bucket weight × ~0.2% qualification
rate on a 491-ticker large-cap universe produces a max 0.075-point
composite differential — not enough to re-rank top-N selection.
Whether 15% × dense multi-signal coverage would help is an open
architecture question, deferred until paid signals (Quiver ~$30/mo)
or SEC EDGAR 13F parsing get dedicated segment time. See
``models/cache/alt_signals_phase1_summary.md`` for the full
diagnosis and the two future-work paths.

----

Free-tier alt signal sourced from Finnhub's
``/stock/insider-transactions`` endpoint. Uses BULK queries (no symbol
parameter) to fetch all insider transactions globally for a date range,
then filters to ``source == "sec"`` (US listings) at parse time and
filters to the requested universe at score time. Cache is append-only,
so extending the date range only fetches the gap.

Important deviation from segment 14 plan: Finnhub's free-tier insider
endpoint does NOT return insider role/title. We can't filter to senior
insiders (CEO/CFO/Director/President) — the all-insider clustering
proxy is the best the free tier supports. See
``models/cache/finnhub_insider_signal_limitations.md`` for the full
rationale and the academic-strength impact.

Score: per (ticker, date), look at last 90 days of P-coded
(open-market purchase) transactions for that ticker. Count unique
``name`` values, sum ``change × transactionPrice`` as USD purchase
value. Filter ≥2 unique insiders + ≥$10K combined. Universe rank-
normalize qualifying tickers to [0.5, 1.0]; non-qualifying = 0.5
(neutral, matches alt-bucket convention from segment 12).
"""

import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))
_FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

API_URL = "https://finnhub.io/api/v1/stock/insider-transactions"
USER_AGENT = "paper-trader-research/0.1 (personal-use)"

CACHE_DIR = os.path.abspath(os.path.join(
    BASE_DIR, "..", "models", "cache", "finnhub_insider"))
CACHE_PARQUET = os.path.join(CACHE_DIR, "transactions.parquet")
CACHE_META    = os.path.join(CACHE_DIR, "transactions.meta.json")

CACHE_VERSION = "v1"

# Throttle: 60/min free-tier limit. 1.1s sleep is comfortably under.
THROTTLE_SECONDS  = 1.1
BACKOFF_DELAYS    = (5, 10, 20)
REQUEST_TIMEOUT   = 30

# Score parameters
LOOKBACK_DAYS         = 90
MIN_UNIQUE_INSIDERS   = 2
MIN_TOTAL_VALUE_USD   = 10_000

# Schema for the cached parquet
CACHE_COLUMNS = [
    "ticker", "transaction_date", "name", "change",
    "price", "value", "tx_code", "is_derivative", "source",
]


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

def _fetch_chunk(start: str, end: str) -> pd.DataFrame:
    """One bulk insider-transactions query for [start, end].

    Bulk = no ``symbol`` param → returns global activity. We filter to
    ``source == "sec"`` (US listings only) before returning. Throttled
    by the caller; backoff on 429/5xx retry up to len(BACKOFF_DELAYS)
    times before raising.
    """
    if not _FINNHUB_KEY:
        raise RuntimeError("FINNHUB_API_KEY missing from .env")
    params = {"from": start, "to": end, "token": _FINNHUB_KEY}
    headers = {"User-Agent": USER_AGENT}

    last_err: Exception | None = None
    for attempt, delay in enumerate([0] + list(BACKOFF_DELAYS)):
        if delay:
            print(f"  [FINNHUB] backoff {delay}s before attempt {attempt + 1}")
            time.sleep(delay)
        try:
            r = requests.get(API_URL, params=params, headers=headers,
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

    payload = r.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame(columns=CACHE_COLUMNS)

    df = pd.DataFrame(rows)
    # Filter to US SEC source. Other sources (e.g., "sedi" Canadian) get
    # dropped at fetch time so the cache stays focused.
    df = df[df.get("source", "") == "sec"].copy()
    if df.empty:
        return pd.DataFrame(columns=CACHE_COLUMNS)

    df["ticker"]           = df["symbol"].astype(str).str.upper().str.strip()
    df["transaction_date"] = pd.to_datetime(df["transactionDate"],
                                            errors="coerce")
    df["name"]             = df["name"].astype(str).str.strip()
    df["change"]           = pd.to_numeric(df["change"], errors="coerce")
    df["price"]            = pd.to_numeric(df["transactionPrice"],
                                            errors="coerce")
    df["tx_code"]          = df["transactionCode"].astype(str).str.strip()
    df["is_derivative"]    = df["isDerivative"].fillna(False).astype(bool)
    df["source"]           = df["source"].astype(str)
    df["value"]            = df["change"] * df["price"]

    df = df[df["transaction_date"].notna() & df["ticker"].ne("")]
    return df[CACHE_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cache build (append-only, monthly bulk chunks)
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


def _month_chunks(start: pd.Timestamp,
                  end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield month-aligned (start, end) inclusive chunks."""
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start.replace(day=1)
    while cur <= end:
        # Last day of this month
        next_month = (cur + pd.offsets.MonthBegin(1))
        month_end = min(next_month - pd.Timedelta(days=1), end)
        chunk_start = max(cur, start)
        chunks.append((chunk_start, month_end))
        cur = next_month
    return chunks


def build_cache(start: str = "2018-01-01",
                end: str | None = None) -> pd.DataFrame:
    """Append-only build / refresh of the global insider-transactions cache.

    On empty cache: full fetch from ``start`` to ``end`` in monthly bulk chunks.
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
        chunks_to_fetch: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if req_start < cov_start:
            chunks_to_fetch += _month_chunks(
                req_start, cov_start - pd.Timedelta(days=1))
        if req_end > cov_end:
            chunks_to_fetch += _month_chunks(
                cov_end + pd.Timedelta(days=1), req_end)
        if not chunks_to_fetch:
            print(f"  [FINNHUB] cache covers {cov_start.date()} -> "
                  f"{cov_end.date()}, request satisfied; no fetch")
            return existing
        print(f"  [FINNHUB] cache covers {cov_start.date()} -> "
              f"{cov_end.date()}; fetching {len(chunks_to_fetch)} new "
              f"month(s) for the gap")
    else:
        chunks_to_fetch = _month_chunks(req_start, req_end)
        if meta:
            print(f"  [FINNHUB] cache version mismatch "
                  f"({meta.get('version')!r} vs {CACHE_VERSION!r}) — "
                  f"rebuilding from scratch")
        else:
            print(f"  [FINNHUB] no cache — fetching {len(chunks_to_fetch)} "
                  f"month(s) ({req_start.date()} -> {req_end.date()})")

    pieces: list[pd.DataFrame] = []
    if existing is not None and not existing.empty:
        pieces.append(existing)

    t0 = time.perf_counter()
    for i, (cs, ce) in enumerate(chunks_to_fetch, 1):
        cs_str = cs.strftime("%Y-%m-%d")
        ce_str = ce.strftime("%Y-%m-%d")
        print(f"  [FINNHUB] {i:>3}/{len(chunks_to_fetch)}  "
              f"{cs_str} -> {ce_str}", end="  ", flush=True)
        sub = _fetch_chunk(cs_str, ce_str)
        print(f"rows={len(sub):>5}")
        pieces.append(sub)
        time.sleep(THROTTLE_SECONDS)

    df = pd.concat(pieces, ignore_index=True)
    df = df.drop_duplicates(
        subset=["ticker", "transaction_date", "name", "change",
                "price", "tx_code"],
        keep="last",
    )
    df = df.sort_values(["ticker", "transaction_date"]).reset_index(drop=True)

    _save_parquet(df)
    new_cov_start = min(req_start,
                        pd.Timestamp(meta["date_range_covered"]["start"])
                        if meta else req_start)
    new_cov_end = max(req_end,
                      pd.Timestamp(meta["date_range_covered"]["end"])
                      if meta else req_end)
    _save_meta({
        "version":       CACHE_VERSION,
        "date_range_covered": {
            "start": str(new_cov_start.date()),
            "end":   str(new_cov_end.date()),
        },
        "fetched_at":    datetime.now(timezone.utc).isoformat(),
        "n_rows":        int(len(df)),
        "n_tickers":     int(df["ticker"].nunique()),
        "endpoint":      API_URL,
        "chunk_strategy": "monthly bulk (no symbol param)",
        "filters":       {"source": "sec only at fetch time"},
    })

    print(f"  [FINNHUB] cache saved: {len(df):,} rows, "
          f"{df['ticker'].nunique():,} tickers, "
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
        _Loaded.df = pd.DataFrame(columns=CACHE_COLUMNS)
        _Loaded.by_ticker = {}
        return
    df = pd.read_parquet(CACHE_PARQUET)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    _Loaded.df = df
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
    mask = (
        (sub["transaction_date"] >= window_start)
        & (sub["transaction_date"] <= date_ts)
        & (sub["tx_code"] == "P")
        & (sub["change"] > 0)
    )
    win = sub[mask]
    if win.empty:
        return 0.0
    n_unique = win["name"].nunique()
    total_value = win["value"].sum(skipna=True)
    if pd.isna(total_value):
        total_value = 0.0
    if n_unique < MIN_UNIQUE_INSIDERS or total_value < MIN_TOTAL_VALUE_USD:
        return 0.0
    return float(n_unique) * math.log1p(float(total_value))


def score_finnhub_insider_clusters(tickers: list[str],
                                   date) -> dict[str, float]:
    """Public: score the requested universe at ``date``.

    All-insider proxy (no senior-only filter — Finnhub free tier doesn't
    return titles). Per ticker: count unique buyers in last 90 days of
    open-market purchases (``transactionCode == 'P'``, ``change > 0``),
    sum dollar value, apply ≥2-insider / ≥$10K filter, universe rank-
    normalize qualifying tickers to [0.5, 1.0]. Non-qualifying → 0.5.
    """
    _ensure_loaded()
    date_ts = pd.Timestamp(date)
    window_start = date_ts - pd.Timedelta(days=LOOKBACK_DAYS)

    raw: dict[str, float] = {}
    for t in tickers:
        try:
            raw[t] = _ticker_cluster_score(t, date_ts, window_start)
        except Exception as e:
            print(f"  [FINNHUB] {t} score error at {date_ts.date()}: {e}")
            raw[t] = 0.0

    active = [(t, v) for t, v in raw.items() if v > 0.0]
    out: dict[str, float] = {t: 0.5 for t in tickers}
    if not active:
        return out

    active.sort(key=lambda x: x[1])
    n = len(active)
    for i, (t, _) in enumerate(active):
        rank_norm = i / max(n - 1, 1)
        out[t] = 0.5 + 0.5 * rank_norm
    return out


# ---------------------------------------------------------------------------
# CLI for cache build / refresh
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finnhub insider cache builder")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end",   default=None,
                        help="default = today")
    args = parser.parse_args()
    df = build_cache(args.start, args.end)
    print(f"\nCache rows: {len(df):,}")
    if not df.empty:
        print(f"Date range: {df['transaction_date'].min().date()} -> "
              f"{df['transaction_date'].max().date()}")
        print(f"Unique tickers: {df['ticker'].nunique():,}")
        print(f"Code distribution (top 10):")
        for code, n in df["tx_code"].value_counts().head(10).items():
            print(f"  {code:<4}  {n:>7,}")
