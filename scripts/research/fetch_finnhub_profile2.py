"""Fetch /stock/profile2 for every symbol in the Larger Universe v1 universe.

Output: models/cache/equities/finnhub/profile/<SYM>.json per ticker.
Aggregated mapping: models/features/larger_universe_v1/sector_map.json

Rate limit: 60/min. ~2,122 calls → ~36 min wall-clock.

Resumable: skips tickers whose profile/<SYM>.json already exists.
"""
from __future__ import annotations

import json, logging, os, sys, time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.options.tradier import RateLimiter

KEY = os.environ["FINNHUB_API_KEY"]
BASE = "https://finnhub.io/api/v1"
PROFILE_DIR = ROOT / "models" / "cache" / "equities" / "finnhub" / "profile"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
UNIVERSE_PATH = ROOT / "docs" / "larger_universe_v1_universe.json"
SECTOR_OUT = ROOT / "models" / "features" / "larger_universe_v1" / "sector_map.json"

logger = logging.getLogger("profile2_fetch")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    stream=sys.stdout)

def _safe_filename(sym: str) -> str:
    """Replace path separators in symbol so it works as a filename.

    Dual-class tickers like 'UA/UAA' confuse the filesystem (the / gets
    interpreted as a directory separator). Normalize to a dash here.
    """
    return sym.replace("/", "-").replace("\\", "-")


def fetch_one(sym: str, limiter: RateLimiter) -> dict:
    cache = PROFILE_DIR / f"{_safe_filename(sym)}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    limiter.wait()
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/stock/profile2",
                             params={"symbol": sym, "token": KEY}, timeout=20)
            if r.status_code == 429:
                logger.warning("  %s: 429 attempt %d/3", sym, attempt+1)
                time.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            body = r.json() if r.text else {}
            tmp = cache.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(body), encoding="utf-8")
            tmp.replace(cache)
            return body
        except requests.RequestException as e:
            logger.warning("  %s: %s attempt %d/3", sym, type(e).__name__, attempt+1)
            time.sleep(2 ** attempt)
    return {}


def main() -> int:
    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    # Dedupe by symbol (mirror the fetcher's prefer-active dedup)
    by_sym = {}
    for r in universe:
        s = r["symbol"]
        if s not in by_sym or (by_sym[s]["status"] == "removed" and r["status"] == "active"):
            by_sym[s] = r
    symbols = sorted(by_sym.keys())
    logger.info("universe: %d unique symbols", len(symbols))

    limiter = RateLimiter(60)
    n_ok = n_empty = n_err = 0
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        body = fetch_one(sym, limiter)
        if body:
            n_ok += 1
        else:
            n_empty += 1
        if i % 50 == 0 or i == len(symbols):
            logger.info("  %d/%d  ok=%d empty=%d  (%.1fs)",
                        i, len(symbols), n_ok, n_empty, time.time() - t0)

    # Aggregate sector_map
    SECTOR_OUT.parent.mkdir(parents=True, exist_ok=True)
    sector_map: dict[str, dict] = {}
    for sym in symbols:
        p = PROFILE_DIR / f"{_safe_filename(sym)}.json"
        if not p.exists():
            continue
        body = json.loads(p.read_text(encoding="utf-8"))
        if not body:
            continue
        sector_map[sym] = {
            "sector": body.get("finnhubIndustry"),
            "ipo": body.get("ipo"),
            "share_outstanding_millions": body.get("shareOutstanding"),
            "market_cap_millions": body.get("marketCapitalization"),
            "name": body.get("name"),
        }
    SECTOR_OUT.write_text(json.dumps(sector_map, indent=2), encoding="utf-8")
    logger.info("sector_map: %d entries -> %s", len(sector_map), SECTOR_OUT)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
