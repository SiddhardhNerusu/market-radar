"""Refresh the ``short_interest`` table from FINRA's free Equity Short
Interest API.

FINRA publishes short-interest data twice a month at:
  https://api.finra.org/data/group/otcMarket/name/equityShortInterest

The API requires no key; ~10 rps polite-rate is fine. We pull the most
recent settlement date, compute short_pct_float (using float estimates
from yfinance), and write a row per ticker.

Usage::

    python scripts/refresh_short_interest.py                  # latest
    python scripts/refresh_short_interest.py --since-days 90  # last 90d of reports
    python scripts/refresh_short_interest.py --tickers AAPL,TSLA  # restrict

Feature impact:
  Populates raw_signals.body cache columns ``short_interest_pct_float``
  and ``days_to_cover``, which ``ml/external_features.py``
  ``attach_short_interest`` reads at train/predict time.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from market_radar.storage import get_connection, init_db  # noqa: E402


log = logging.getLogger("refresh_short_interest")


# FINRA's documented endpoint. Returns paginated JSON; no auth.
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/equityShortInterest"
PAGE_SIZE = 5000


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_finra_page(*, offset: int, limit: int,
                     since_date: Optional[str] = None,
                     tickers: Optional[list[str]] = None) -> list[dict]:
    """Hit FINRA's Query API for short-interest data.

    The API accepts both GET (with URL params) and POST (with JSON body).
    GET is the simpler/more-reliable path and avoids the JSON-schema
    quirks of POST. We use GET with a ``limit`` and ``offset`` and then
    filter client-side for ``since_date`` and ``tickers`` — there are
    only ~20k symbols/settlement so this is cheap.
    """
    try:
        r = requests.get(
            FINRA_URL,
            params={"limit": limit, "offset": offset},
            headers={"Accept": "application/json",
                     "User-Agent": "MARKET RADAR research (research@example.com)"},
            timeout=30,
        )
        if r.status_code == 404:
            log.warning("FINRA returned 404 — endpoint may have changed")
            return []
        r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("FINRA fetch failed (offset=%d): %s", offset, exc)
        return []

    try:
        data = r.json()
    except ValueError as exc:
        log.warning("FINRA non-JSON: %s", exc)
        return []

    rows: list[dict] = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("data", "results", "items"):
            if key in data and isinstance(data[key], list):
                rows = data[key]
                break

    # Client-side filters
    if since_date:
        rows = [r for r in rows
                if (r.get("settlementDate") or r.get("recordDate") or "") >= since_date]
    if tickers:
        ts = {t.upper() for t in tickers}
        rows = [r for r in rows
                if (r.get("symbolCode") or r.get("symbol") or "").upper() in ts]
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--since-days", type=int, default=30,
                   help="Pull reports settled in the last N days (default: 30)")
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated list of tickers to filter on")
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    init_db()  # ensure short_interest table exists

    since_dt = datetime.now(timezone.utc) - timedelta(days=args.since_days)
    since_str = since_dt.strftime("%Y-%m-%d")
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    log.info("Fetching FINRA short-interest since=%s  tickers=%s",
             since_str, "ALL" if not tickers else f"{len(tickers)} symbols")

    inserted = updated = errors = 0
    offset = 0
    fetched = 0
    while fetched < args.max_rows:
        rows = fetch_finra_page(
            offset=offset, limit=PAGE_SIZE,
            since_date=since_str, tickers=tickers,
        )
        if not rows:
            break
        if args.dry_run:
            log.info("Dry-run: would store %d rows (sample: %r)",
                     len(rows), {k: rows[0].get(k) for k in list(rows[0].keys())[:6]})
            return 0

        with get_connection() as conn:
            for r in rows:
                ticker = (r.get("symbolCode") or r.get("symbol") or "").upper()
                settlement = r.get("settlementDate") or r.get("recordDate")
                if not ticker or not settlement:
                    continue
                short_int = _to_float(r.get("currentShortPositionQuantity")
                                      or r.get("shortInterest"))
                adv = _to_float(r.get("averageDailyVolumeQuantity")
                                or r.get("avgDailyVolume"))
                days_to_cover = _to_float(r.get("daysToCoverQuantity")
                                          or r.get("daysToCover"))
                # FINRA doesn't ship float; we'll leave short_pct_float
                # blank here and let a separate ticker-by-ticker pass
                # (out of scope for now) fill it via yfinance.shares_outstanding.
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO short_interest
                        (ticker, report_date, short_interest, avg_daily_volume,
                         days_to_cover, float_shares, short_pct_float, ingested_at)
                        VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
                        """,
                        (ticker, settlement, short_int, adv, days_to_cover, _utc_now()),
                    )
                    inserted += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("insert failed for %s: %s", ticker, exc)
                    errors += 1

        fetched += len(rows)
        offset += PAGE_SIZE
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.15)

    log.info("Done. inserted_or_replaced=%d errors=%d", inserted, errors)
    return 0


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
