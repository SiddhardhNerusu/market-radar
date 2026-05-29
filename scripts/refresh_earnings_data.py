"""Refresh the ``earnings_data`` table with EPS surprise + revenue
surprise data from Finnhub's free tier.

Requires ``FINNHUB_API_KEY`` in ``.env`` (free key from finnhub.io).
Without the key, the script logs a warning and exits 0. The PEAD
feature simply defaults to "no surprise" when the table is empty.

Endpoints used (Finnhub free tier):
  - GET /api/v1/calendar/earnings  — earnings calendar (from/to dates)
  - GET /api/v1/stock/earnings     — historical surprises by symbol

We pull the calendar over a configurable window AND backfill the
trailing 12 quarters of surprises for our most-active tickers, so the
PEAD feature has lookups for both upcoming and historical events.

Usage::

    python scripts/refresh_earnings_data.py --since-days 30 --to-days 30
    python scripts/refresh_earnings_data.py --tickers AAPL,TSLA --backfill-quarters 8
    python scripts/refresh_earnings_data.py --dry-run

Feature impact:
  ``ml/external_features.attach_pead_features`` reads from this table
  and emits ``eps_surprise_pct`` + ``days_since_earnings``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from market_radar.config import CONFIG  # noqa: E402
from market_radar.storage import get_connection, init_db  # noqa: E402


log = logging.getLogger("refresh_earnings_data")


FINNHUB_BASE = "https://finnhub.io/api/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fnh_get(path: str, **params) -> Optional[dict]:
    if not CONFIG.finnhub_api_key:
        return None
    params["token"] = CONFIG.finnhub_api_key
    try:
        r = requests.get(f"{FINNHUB_BASE}{path}",
                         params=params, timeout=20,
                         headers={"User-Agent": "MARKET RADAR research"})
        if r.status_code == 429:
            log.warning("Finnhub rate-limited — pausing 10s")
            time.sleep(10)
            r = requests.get(f"{FINNHUB_BASE}{path}",
                             params=params, timeout=20,
                             headers={"User-Agent": "MARKET RADAR research"})
        if r.status_code == 403:
            log.error("Finnhub returned 403 — key invalid or endpoint paid-tier only")
            return None
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as exc:
        log.debug("Finnhub %s failed: %s", path, exc)
        return None


def fetch_calendar(*, from_d: str, to_d: str) -> list[dict]:
    payload = _fnh_get("/calendar/earnings", **{"from": from_d, "to": to_d})
    if not payload:
        return []
    return payload.get("earningsCalendar") or []


def fetch_historical_surprises(ticker: str, limit: int = 12) -> list[dict]:
    payload = _fnh_get("/stock/earnings", symbol=ticker, limit=limit)
    if isinstance(payload, list):
        return payload
    return []


def pick_tickers(top_n: int) -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ticker, COUNT(*) AS n FROM signal_scores
            WHERE scored_at >= datetime('now', '-30 days')
            GROUP BY ticker
            ORDER BY n DESC
            LIMIT ?
            """,
            (top_n,),
        ).fetchall()
    return [r["ticker"] for r in rows]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--since-days", type=int, default=30)
    p.add_argument("--to-days",    type=int, default=14)
    p.add_argument("--tickers",    type=str, default=None)
    p.add_argument("--top-tickers", type=int, default=100)
    p.add_argument("--backfill-quarters", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not CONFIG.finnhub_api_key:
        log.warning("FINNHUB_API_KEY not set in .env — skipping PEAD refresh. "
                    "Grab a free key at https://finnhub.io and add it.")
        return 0
    init_db()

    today = datetime.now(timezone.utc)
    from_d = (today - timedelta(days=args.since_days)).strftime("%Y-%m-%d")
    to_d = (today + timedelta(days=args.to_days)).strftime("%Y-%m-%d")

    log.info("Calendar window: %s → %s", from_d, to_d)
    cal = fetch_calendar(from_d=from_d, to_d=to_d)
    log.info("Calendar entries: %d", len(cal))

    inserted = 0
    with get_connection() as conn:
        for e in cal:
            ticker = (e.get("symbol") or "").upper()
            report_date = e.get("date")
            eps_est = e.get("epsEstimate")
            eps_act = e.get("epsActual")
            rev_est = e.get("revenueEstimate")
            rev_act = e.get("revenueActual")
            eps_surprise = (
                ((float(eps_act) - float(eps_est)) / abs(float(eps_est)) * 100.0)
                if eps_act not in (None, "") and eps_est not in (None, 0, "0", "")
                else None
            )
            rev_surprise = (
                ((float(rev_act) - float(rev_est)) / abs(float(rev_est)) * 100.0)
                if rev_act not in (None, "") and rev_est not in (None, 0, "0", "")
                else None
            )
            if not ticker or not report_date:
                continue
            if args.dry_run:
                log.info("  CAL %s %s eps=%s/%s rev=%s/%s surprise_eps=%s",
                         ticker, report_date, eps_act, eps_est,
                         rev_act, rev_est, eps_surprise)
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO earnings_data
                (ticker, report_date, eps_actual, eps_estimate, eps_surprise_pct,
                 revenue_actual, revenue_estimate, revenue_surprise_pct, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, report_date, eps_act, eps_est, eps_surprise,
                 rev_act, rev_est, rev_surprise, _utc_now()),
            )
            inserted += 1
        log.info("Calendar inserts: %d", inserted)

    # Backfill historical surprises for our most-active tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = pick_tickers(args.top_tickers)
    log.info("Backfilling %d-quarter history for %d tickers", args.backfill_quarters, len(tickers))

    hist_inserts = 0
    with get_connection() as conn:
        for ticker in tickers:
            hist = fetch_historical_surprises(ticker, limit=args.backfill_quarters)
            for h in hist:
                period = h.get("period") or h.get("date")
                eps_est = h.get("estimate")
                eps_act = h.get("actual")
                if not period or eps_act is None or eps_est in (None, 0):
                    continue
                try:
                    surprise = (float(eps_act) - float(eps_est)) / abs(float(eps_est)) * 100.0
                except (TypeError, ValueError):
                    continue
                if args.dry_run:
                    log.info("  HIST %s %s actual=%s est=%s surprise=%.2f",
                             ticker, period, eps_act, eps_est, surprise)
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO earnings_data
                    (ticker, report_date, eps_actual, eps_estimate, eps_surprise_pct,
                     revenue_actual, revenue_estimate, revenue_surprise_pct, ingested_at)
                    VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
                    """,
                    (ticker, period, eps_act, eps_est, surprise, _utc_now()),
                )
                hist_inserts += 1
            time.sleep(0.4)  # polite to Finnhub free tier

    log.info("Done. calendar_inserts=%d hist_inserts=%d", inserted, hist_inserts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
