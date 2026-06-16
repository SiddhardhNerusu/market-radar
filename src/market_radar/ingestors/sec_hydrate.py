"""Out-of-band SEC body hydration (ingestion blueprint #2 — the latency fix).

Live sec_edgar ingestion now runs ``fetch_bodies=False``: it writes a fast STUB
row (body = RSS summary, ``raw_payload.body_hydrated=False``) with NO HTTP in the
poll loop, so the single SQLite writer is never held across the network. This job
runs on a tight cadence, finds recent un-hydrated stubs, fetches the real filing
body via the shared rate-limited ``SecBodyFetcher`` OUTSIDE any write connection,
and UPDATEs the body in a short per-row connection. The classifier skips
un-hydrated stubs (until a grace window) so it only ever classifies real text.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ..storage import get_connection
from .sec_body_fetcher import SecBodyFetcher

log = logging.getLogger("marketradar.ingestors.sec_hydrate")

_FETCHER: Optional[SecBodyFetcher] = None


def _fetcher() -> SecBodyFetcher:
    global _FETCHER
    if _FETCHER is None:
        _FETCHER = SecBodyFetcher()
    return _FETCHER


def hydrate_sec_bodies(*, batch_size: int = 15, max_age_minutes: int = 120,
                       fetcher: Optional[SecBodyFetcher] = None) -> int:
    """Hydrate up to ``batch_size`` recent un-hydrated sec_edgar stubs.

    Returns the number of bodies successfully filled. Network I/O happens with
    NO db connection held; each successful fetch is written in its own short
    connection. ``fetcher`` is injectable for tests.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, raw_payload
            FROM raw_signals
            WHERE source = 'sec_edgar'
              AND json_extract(raw_payload, '$.body_hydrated') = 0
              AND ingested_at >= strftime('%Y-%m-%dT%H:%M:%SZ',
                                          datetime('now', ?))
            ORDER BY id DESC
            LIMIT ?
            """,
            (f"-{int(max_age_minutes)} minutes", int(batch_size)),
        ).fetchall()

    fx = fetcher or _fetcher()
    hydrated = 0
    for r in rows:
        try:
            payload = json.loads(r["raw_payload"]) if r["raw_payload"] else {}
        except (TypeError, ValueError):
            continue
        link = payload.get("link")
        if not link:
            continue
        try:
            body = fx.fetch_body(link, form_type=payload.get("form"))  # NETWORK — no db held
        except Exception as exc:  # noqa: BLE001 — never let a fetch crash the job
            log.debug("[sec_hydrate] fetch failed id=%s: %s", r["id"], exc)
            body = None
        if not body:
            continue
        payload["body_hydrated"] = True
        try:
            with get_connection() as conn:  # short per-row write
                conn.execute(
                    "UPDATE raw_signals SET body = ?, raw_payload = ? WHERE id = ?",
                    (body, json.dumps(payload), r["id"]),
                )
            hydrated += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("[sec_hydrate] update failed id=%s: %s", r["id"], exc)
    if rows:
        log.info("[sec_hydrate] %d/%d stubs hydrated", hydrated, len(rows))
    return hydrated
