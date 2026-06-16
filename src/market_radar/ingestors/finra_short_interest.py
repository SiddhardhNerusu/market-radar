"""FINRA consolidated short-interest ingestor (coverage blueprint #5).

Short interest is a slow-moving but high-signal feature: a high short-interest /
high days-to-cover name is loaded with fuel for a squeeze, and a sharp jump in
short interest into a known catalyst is exactly the setup behind the violent
micro-cap "bangs" we hunt. This fills the ``short_interest`` table so the scoring
+ ML feature pipeline can read per-ticker short metrics.

Source — FINRA's free, public, no-auth OTC Transparency Data API. We use the
``consolidatedShortInterest`` dataset, which is the *consolidated* view across ALL
exchanges (NYSE / Nasdaq(NNM) / ARCA / AMEX / BZX / OTC ...), not just OTC. Short
interest is reported twice a month under FINRA Rule 4560 — mid-month (the 15th,
or the prior business day) and end-of-month settlement dates.

    POST https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest
    Content-Type: application/json   Accept: application/json
    body: {"limit": 5000, "offset": N,
           "compareFilters":[{"compareType":"EQUAL",
                              "fieldName":"settlementDate",
                              "fieldValue":"YYYY-MM-DD"}]}

The API requires ``settlementDate`` to be pinned with an EQUAL filter (it is a
partition key) before it will return / sort a slice, so we first DISCOVER the most
recent published settlement date by probing the canonical FINRA settlement dates
walking backward, then page the full slice (``record-max-limit`` is 5000; the
``record-total`` response header tells us how many rows exist so we page via
``offset``).

Response field names (consolidated dataset):
    symbolCode                    -> ticker   (OTC-only dataset uses
                                     securitiesInformationProcessorSymbolIdentifier;
                                     we accept either so the module is robust)
    settlementDate                -> report_date (YYYY-MM-DD)
    currentShortPositionQuantity  -> short_interest
    averageDailyVolumeQuantity    -> avg_daily_volume
    daysToCoverQuantity           -> days_to_cover (FINRA caps this at 999.99 for
                                     thin names; we pass it through but null the
                                     sentinel so it doesn't poison ML features)

FINRA does NOT publish float / short-%-of-float in this feed, so
``float_shares`` and ``short_pct_float`` are left NULL (other ingestors own those).

This module NEVER raises: any network / parse / DB failure is logged and the poll
returns the count written so far (0 on total failure). Polled twice a day from the
daemon — the data only changes ~twice a month but a cheap idempotent re-poll keeps
us latched onto the newest settlement date the moment FINRA publishes it.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import requests

from ..storage import get_connection

log = logging.getLogger(__name__)

DATASET_URL = (
    "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
)

# FINRA caps a single response slice at 5000 rows (record-max-limit header).
PAGE_LIMIT = 5000

# FINRA's sentinel for "average daily volume is ~0, so days-to-cover is
# effectively infinite". Passing 999.99 into an ML feature would be a fat
# outlier masquerading as a real value, so we null it.
DAYS_TO_COVER_SENTINEL = 999.99

_HEADERS = {
    "User-Agent": "market-radar/1.0 (short-interest monitor)",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def _candidate_settlement_dates(today: date, *, lookback: int = 6) -> list[str]:
    """Canonical FINRA settlement dates, most-recent-first.

    Short interest settles on the 15th and the last calendar day of each month
    (rolled back to the prior business day when that lands on a weekend). We
    don't know *exactly* which has been published yet, so we generate the last
    several candidates and let the caller probe newest-first for one that
    actually returns rows.
    """
    def _prev_business_day(d: date) -> date:
        # Sat -> Fri, Sun -> Fri. (Holidays are handled by the probe falling
        # through to the next-older candidate, so we don't hardcode a calendar.)
        while d.weekday() >= 5:  # 5=Sat, 6=Sun
            d -= timedelta(days=1)
        return d

    cands: list[date] = []
    # Walk back month by month, emitting end-of-month then mid-month for each.
    y, m = today.year, today.month
    for _ in range(lookback):
        # last day of month (y, m)
        if m == 12:
            first_next = date(y + 1, 1, 1)
        else:
            first_next = date(y, m + 1, 1)
        eom = first_next - timedelta(days=1)
        mid = date(y, m, 15)
        cands.append(_prev_business_day(eom))
        cands.append(_prev_business_day(mid))
        # step to previous month
        m -= 1
        if m == 0:
            m = 12
            y -= 1

    # Keep only dates that are not in the future, dedup, newest-first.
    out: list[str] = []
    seen: set[str] = set()
    for d in sorted((c for c in cands if c <= today), reverse=True):
        iso = d.isoformat()
        if iso not in seen:
            seen.add(iso)
            out.append(iso)
    return out


def _fetch_slice(
    session: requests.Session,
    *,
    settlement_date: str,
    offset: int,
    limit: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], Optional[int]]:
    """Fetch one page for ``settlement_date``. Returns (rows, record_total).

    ``record_total`` (total rows available for this date) comes from the
    response header when present, else None.
    """
    body = {
        "limit": limit,
        "offset": offset,
        "compareFilters": [
            {
                "compareType": "EQUAL",
                "fieldName": "settlementDate",
                "fieldValue": settlement_date,
            }
        ],
    }
    resp = session.post(DATASET_URL, json=body, timeout=timeout)
    if resp.status_code >= 400:
        log.warning("[finra_short_interest] HTTP %d for %s (offset=%d)",
                    resp.status_code, settlement_date, offset)
        return [], None

    try:
        payload = resp.json()
    except ValueError:
        log.warning("[finra_short_interest] non-JSON body for %s", settlement_date)
        return [], None

    # The API returns a bare JSON array of row objects. Be defensive about a
    # dict-wrapped shape too.
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("results") or []
    if not isinstance(payload, list):
        return [], None

    record_total: Optional[int] = None
    rt = resp.headers.get("record-total")
    if rt is not None:
        try:
            record_total = int(rt)
        except (TypeError, ValueError):
            record_total = None

    return [r for r in payload if isinstance(r, dict)], record_total


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Map one FINRA response row to a ``short_interest`` upsert dict.

    Returns None for rows we can't key (no ticker or no settlement date).
    Accepts either the consolidated dataset's ``symbolCode`` or the OTC-only
    dataset's ``securitiesInformationProcessorSymbolIdentifier`` for the ticker.
    """
    ticker = (
        row.get("symbolCode")
        or row.get("securitiesInformationProcessorSymbolIdentifier")
        or ""
    )
    ticker = str(ticker).strip().upper()
    report_date = str(row.get("settlementDate") or "").strip()
    if not ticker or not report_date:
        return None

    days_to_cover = _to_float(row.get("daysToCoverQuantity"))
    if days_to_cover is not None and days_to_cover >= DAYS_TO_COVER_SENTINEL:
        days_to_cover = None  # FINRA "effectively infinite" sentinel — drop it

    return {
        "ticker": ticker,
        "report_date": report_date,
        "short_interest": _to_float(row.get("currentShortPositionQuantity")),
        "avg_daily_volume": _to_float(row.get("averageDailyVolumeQuantity")),
        "days_to_cover": days_to_cover,
        # FINRA does not publish float / short-%-of-float in this feed.
        "float_shares": None,
        "short_pct_float": None,
    }


def upsert_rows(conn, rows: list[dict[str, Any]]) -> int:
    """UPSERT parsed rows into ``short_interest``. Returns count written.

    COALESCE on UPDATE preserves any ``float_shares`` / ``short_pct_float`` that a
    different ingestor may have populated for the same (ticker, report_date) —
    FINRA never supplies those, so an UPSERT here must not clobber them.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wrote = 0
    for r in rows:
        try:
            conn.execute(
                """
                INSERT INTO short_interest
                  (ticker, report_date, short_interest, avg_daily_volume,
                   days_to_cover, float_shares, short_pct_float, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, report_date) DO UPDATE SET
                  short_interest   = excluded.short_interest,
                  avg_daily_volume = excluded.avg_daily_volume,
                  days_to_cover    = excluded.days_to_cover,
                  float_shares     = COALESCE(excluded.float_shares, short_interest.float_shares),
                  short_pct_float  = COALESCE(excluded.short_pct_float, short_interest.short_pct_float),
                  ingested_at      = excluded.ingested_at
                """,
                (
                    r["ticker"], r["report_date"], r["short_interest"],
                    r["avg_daily_volume"], r["days_to_cover"],
                    r["float_shares"], r["short_pct_float"], now_iso,
                ),
            )
            wrote += 1
        except Exception as exc:  # noqa: BLE001
            log.debug("[finra_short_interest] upsert %s/%s failed: %s",
                      r.get("ticker"), r.get("report_date"), exc)
    return wrote


class FinraShortInterestIngestor:
    """FINRA consolidated short interest -> ``short_interest`` table.

    Standalone table-writer ingestor (mirrors EarningsCalendarIngestor): exposes
    ``poll() -> int`` for the daemon. NEVER raises.
    """

    name = "finra_short_interest"

    def __init__(self, *, timeout: float = 30.0, max_pages: int = 20) -> None:
        self.timeout = timeout
        # Safety cap on paging (5000 rows/page * 20 = 100k rows >> the ~22k
        # consolidated universe) so a misbehaving total can't loop forever.
        self.max_pages = max_pages
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------

    def _latest_settlement_date(self) -> Optional[str]:
        """Probe canonical settlement dates newest-first; return the first one
        that actually returns at least one row, else None."""
        today = datetime.now(timezone.utc).date()
        for iso in _candidate_settlement_dates(today):
            try:
                rows, _ = _fetch_slice(
                    self._session, settlement_date=iso, offset=0,
                    limit=1, timeout=self.timeout,
                )
            except requests.RequestException as exc:
                log.warning("[finra_short_interest] probe %s failed: %s", iso, exc)
                continue
            if rows:
                log.info("[finra_short_interest] latest settlement date: %s", iso)
                return iso
        return None

    def _fetch_all(self, settlement_date: str) -> list[dict[str, Any]]:
        """Page the full slice for ``settlement_date`` via offset."""
        all_rows: list[dict[str, Any]] = []
        record_total: Optional[int] = None
        for page in range(self.max_pages):
            offset = page * PAGE_LIMIT
            try:
                rows, rt = _fetch_slice(
                    self._session, settlement_date=settlement_date,
                    offset=offset, limit=PAGE_LIMIT, timeout=self.timeout,
                )
            except requests.RequestException as exc:
                log.warning("[finra_short_interest] page %d fetch failed: %s",
                            page, exc)
                break
            if rt is not None:
                record_total = rt
            if not rows:
                break
            all_rows.extend(rows)
            # Stop when we've pulled everything (by header) or got a short page.
            if record_total is not None and len(all_rows) >= record_total:
                break
            if len(rows) < PAGE_LIMIT:
                break
        return all_rows

    # ------------------------------------------------------------------

    def poll(self) -> int:
        """One ingest cycle: find latest settlement date, fetch + upsert.
        Returns number of rows written. Never raises (returns 0 on any failure
        or empty source)."""
        try:
            settlement_date = self._latest_settlement_date()
            if not settlement_date:
                log.warning("[finra_short_interest] no settlement date returned "
                            "rows (endpoint empty/unreachable); skipping")
                return 0

            raw_rows = self._fetch_all(settlement_date)
            if not raw_rows:
                log.warning("[finra_short_interest] settlement %s returned 0 rows",
                            settlement_date)
                return 0

            parsed: list[dict[str, Any]] = []
            for r in raw_rows:
                try:
                    p = parse_row(r)
                except Exception as exc:  # noqa: BLE001
                    log.debug("[finra_short_interest] parse failed: %s", exc)
                    continue
                if p is not None:
                    parsed.append(p)

            if not parsed:
                log.warning("[finra_short_interest] %d raw rows but 0 parsed "
                            "(field-name drift?)", len(raw_rows))
                return 0

            with get_connection() as conn:
                wrote = upsert_rows(conn, parsed)

            log.info("[finra_short_interest] settlement %s: %d rows upserted "
                     "(fetched %d)", settlement_date, wrote, len(raw_rows))
            return wrote
        except Exception as exc:  # noqa: BLE001 — top-level safety net
            log.exception("[finra_short_interest] poll failed: %s", exc)
            return 0
