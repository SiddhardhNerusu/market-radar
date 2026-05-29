"""Earnings calendar ingestor — fetches upcoming earnings dates from
Finnhub's free /calendar/earnings endpoint.

Why this matters: pre-earnings drift is a documented effect. The day or
two BEFORE an earnings report, implied volatility expands AND the
underlying often has directional drift. The bot's sizing pipeline reads
this table to boost positions on tickers with imminent earnings AND
fade positions in the day after (gap-risk avoidance).

Polled once per day from the daemon. Free Finnhub tier returns
~30 days forward of US earnings dates.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from ..config import CONFIG
from ..storage import get_connection

log = logging.getLogger(__name__)


def _ensure_table() -> None:
    """Create the earnings_calendar table if it doesn't exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS earnings_calendar (
                ticker          TEXT NOT NULL,
                report_date     TEXT NOT NULL,       -- YYYY-MM-DD
                report_time     TEXT,                -- 'bmo' / 'amc' / 'dmh' / null
                eps_estimate    REAL,
                eps_actual      REAL,
                revenue_estimate REAL,
                revenue_actual  REAL,
                fetched_at      TEXT NOT NULL,
                PRIMARY KEY (ticker, report_date)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_earnings_date "
            "ON earnings_calendar(report_date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_earnings_ticker "
            "ON earnings_calendar(ticker)"
        )


class EarningsCalendarIngestor:
    """Pulls the next 30 days of earnings from Finnhub."""

    name = "earnings_calendar"

    def poll(self) -> int:
        """Fetch + upsert next 30 days of earnings. Returns count written.

        Finnhub's free tier appears to choke on multi-week range queries
        (request hangs indefinitely). Workaround: walk day-by-day, single
        Finnhub call per date. With 60 calls/min limit and ~30 days,
        a full sweep takes ~30 sec total.
        """
        if not CONFIG.finnhub_api_key:
            log.debug("No FINNHUB_API_KEY set; skipping earnings_calendar")
            return 0
        _ensure_table()

        today = datetime.now(timezone.utc).date()
        url = "https://finnhub.io/api/v1/calendar/earnings"
        items: list[dict] = []
        for offset in range(30):
            day = today + timedelta(days=offset)
            params = {
                "from": day.isoformat(),
                "to": day.isoformat(),
                "token": CONFIG.finnhub_api_key,
            }
            try:
                r = requests.get(url, params=params, timeout=20)
                r.raise_for_status()
                payload = r.json()
                day_items = payload.get("earningsCalendar") or []
                items.extend(day_items)
            except Exception as exc:  # noqa: BLE001
                # Single-day failure — log + continue. Don't lose the
                # other 29 days because one date had a transient issue.
                log.debug("earnings_calendar %s fetch failed: %s",
                          day.isoformat(), exc)

        if not items:
            log.warning("earnings_calendar: 0 items across 30 days "
                        "(Finnhub free tier may be rate-limiting)")
            return 0

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        wrote = 0
        with get_connection() as conn:
            for it in items:
                symbol = (it.get("symbol") or "").upper().strip()
                date = (it.get("date") or "").strip()
                if not symbol or not date:
                    continue
                # Normalize Finnhub's "hour" field: 'bmo' = before market open,
                # 'amc' = after market close, 'dmh' = during market hours.
                hour = (it.get("hour") or "").lower().strip() or None
                try:
                    conn.execute(
                        """
                        INSERT INTO earnings_calendar
                          (ticker, report_date, report_time, eps_estimate,
                           eps_actual, revenue_estimate, revenue_actual,
                           fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ticker, report_date) DO UPDATE SET
                          report_time = excluded.report_time,
                          eps_estimate = COALESCE(excluded.eps_estimate, earnings_calendar.eps_estimate),
                          eps_actual = COALESCE(excluded.eps_actual, earnings_calendar.eps_actual),
                          revenue_estimate = COALESCE(excluded.revenue_estimate, earnings_calendar.revenue_estimate),
                          revenue_actual = COALESCE(excluded.revenue_actual, earnings_calendar.revenue_actual),
                          fetched_at = excluded.fetched_at
                        """,
                        (symbol, date, hour,
                         it.get("epsEstimate"), it.get("epsActual"),
                         it.get("revenueEstimate"), it.get("revenueActual"),
                         now_iso),
                    )
                    wrote += 1
                except Exception as exc:  # noqa: BLE001
                    log.debug("earnings upsert %s/%s failed: %s", symbol, date, exc)
        log.info("earnings_calendar: %d rows upserted (Finnhub returned %d)",
                 wrote, len(items))
        return wrote


def days_until_earnings(ticker: str) -> Optional[int]:
    """Return days until the next earnings report for ``ticker``, or
    None if not on the calendar. Same-day = 0, never-negative.
    """
    try:
        with get_connection() as conn:
            today_iso = datetime.now(timezone.utc).date().isoformat()
            row = conn.execute(
                """
                SELECT report_date FROM earnings_calendar
                WHERE UPPER(ticker) = ? AND report_date >= ?
                ORDER BY report_date ASC LIMIT 1
                """,
                (ticker.upper(), today_iso),
            ).fetchone()
        if not row:
            return None
        report_date = datetime.strptime(row[0], "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        return max((report_date - today).days, 0)
    except Exception:  # noqa: BLE001
        return None
