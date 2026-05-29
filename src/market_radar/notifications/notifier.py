"""Notification dispatcher.

Finds newly scored signals at or above ``NOTIFICATION_THRESHOLD`` that
haven't been notified yet, dedupes against ``notifications_sent``, and
fires a macOS notification via ``send_notification``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import CONFIG
from ..storage import get_connection
from .macos import send_notification

log = logging.getLogger(__name__)


@dataclass
class NotificationStats:
    candidates: int = 0
    sent: int = 0
    skipped_dup: int = 0
    failed: int = 0


def dispatch_pending(
    *,
    threshold: float | None = None,
    channel: str = "macos_notification",
    batch_limit: int = 10,
) -> NotificationStats:
    """Send notifications for any unsent, above-threshold signal_scores."""
    stats = NotificationStats()
    thresh = threshold if threshold is not None else CONFIG.notification_threshold

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ss.id AS score_id, ss.ticker, ss.composite_score,
                   ss.event_type, ss.sentiment, ss.signal_class,
                   rs.title, rs.url, rs.source
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            LEFT JOIN notifications_sent ns
                   ON ns.score_id = ss.id AND ns.channel = ?
            WHERE ss.composite_score >= ?
              AND ns.id IS NULL
            ORDER BY ss.composite_score DESC, ss.id DESC
            LIMIT ?
            """,
            (channel, thresh, batch_limit),
        ).fetchall()
        stats.candidates = len(rows)
        if not rows:
            return stats

        for row in rows:
            title = f"📈 {row['ticker']}  ({row['composite_score']:.1f}/10)"
            subtitle_bits = [
                row["event_type"] or "signal",
                row["source"] or "",
            ]
            subtitle = " · ".join(b for b in subtitle_bits if b)
            message = (row["title"] or row["url"] or "Signal recorded")[:240]

            ok = send_notification(title=title, message=message, subtitle=subtitle)
            try:
                conn.execute(
                    """
                    INSERT INTO notifications_sent (score_id, channel, sent_at)
                    VALUES (?, ?, ?)
                    """,
                    (row["score_id"], channel, _utc_now_iso()),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "notifications_sent insert failed score_id=%s: %s",
                    row["score_id"], exc,
                )
                stats.failed += 1
                continue

            if ok:
                stats.sent += 1
            else:
                stats.failed += 1

    if stats.candidates:
        log.info(
            "notifications: candidates=%d sent=%d failed=%d (threshold=%.1f)",
            stats.candidates, stats.sent, stats.failed, thresh,
        )
    return stats


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
