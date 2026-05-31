"""Outcome tracker — the edge-measurement engine.

Two operations run on a schedule:

1. ``snapshot_pending_outcomes`` — for every signal_scores row that does
   not yet have a signal_outcomes row, fetch the current price and create
   the outcome row with ``price_at_flag``. Runs frequently (every 1–5 min)
   so signals are anchored within seconds of being scored.

2. ``update_due_outcomes`` — for every signal_outcomes row missing 1d/5d/20d
   prices but whose anchor is at least that old, look up the close on the
   first trading day on/after the target date. Runs less often (every hour).

The hit-rate aggregation per signal_class is a read-only SQL query the
dashboard runs — it doesn't live here.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Iterable, Optional

from ..storage import get_connection
from .price_fetcher import PriceFetcher, PriceSnapshot

log = logging.getLogger(__name__)


@dataclass
class SnapshotStats:
    pending: int = 0
    snapped: int = 0
    failed: int = 0


@dataclass
class UpdateStats:
    candidates_1d: int = 0
    candidates_5d: int = 0
    candidates_20d: int = 0
    updated: int = 0
    failed: int = 0
    fully_resolved: int = 0


# ---------------------------------------------------------------------------
# 1. Anchor snapshots (price at flag time)
# ---------------------------------------------------------------------------


def snapshot_pending_outcomes(
    *,
    batch_size: int = 200,
    fetcher: Optional[PriceFetcher] = None,
) -> SnapshotStats:
    """Snapshot ``price_at_flag`` for any scored signals missing an outcome row."""
    stats = SnapshotStats()
    fetcher = fetcher or PriceFetcher()

    with get_connection() as conn:
        # CRITICAL: skip backfilled signals — the backfill script handles
        # their pricing using historical data at filing time. If the daemon
        # snapshots them, it'll grab TODAY'S price as the anchor for a
        # 2024 filing, which produces meaningless outcomes.
        rows = conn.execute(
            """
            SELECT ss.id AS score_id, ss.ticker, ss.scored_at
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            LEFT JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE so.id IS NULL
              AND rs.source NOT LIKE 'sec_edgar_backfill_%'
            ORDER BY ss.id ASC
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
        stats.pending = len(rows)
        if not rows:
            return stats

        # Batch-fetch unique tickers
        unique_tickers = sorted({row["ticker"] for row in rows})
        log.info("snapshot_pending_outcomes: %d signals, %d unique tickers",
                 len(rows), len(unique_tickers))
        snapshots = fetcher.fetch_latest(unique_tickers)

        now = _utc_now_iso()
        for row in rows:
            snap: Optional[PriceSnapshot] = snapshots.get(row["ticker"])
            price = snap.price if snap else None
            price_ts = snap.timestamp_iso if snap else now

            try:
                conn.execute(
                    """
                    INSERT INTO signal_outcomes (
                        score_id, ticker, price_at_flag, price_at_flag_ts,
                        fully_resolved
                    ) VALUES (?, ?, ?, ?, 0)
                    """,
                    (row["score_id"], row["ticker"], price, price_ts),
                )
                if price is not None:
                    stats.snapped += 1
                else:
                    stats.failed += 1
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "snapshot insert failed score_id=%s ticker=%s: %s",
                    row["score_id"], row["ticker"], exc,
                )
                stats.failed += 1

    log.info("snapshot_pending_outcomes: pending=%d snapped=%d failed=%d",
             stats.pending, stats.snapped, stats.failed)
    return stats


# ---------------------------------------------------------------------------
# 2. Checkpoint updates (1d / 5d / 20d)
# ---------------------------------------------------------------------------


_CHECKPOINTS: list[tuple[int, str, str]] = [
    (1,  "price_1d",  "price_1d_ts"),
    (5,  "price_5d",  "price_5d_ts"),
    (20, "price_20d", "price_20d_ts"),
]

# Poison-pill guards (2026-05-31). The resolver used to ORDER BY anchor ASC
# with no give-up, so it retried the same ~200 oldest rows forever — delisted
# backfill tickers no source can price — and never reached resolvable recent
# signals, leaving the model with zero fresh labels.
MAX_RESOLVE_ATTEMPTS = 8       # quarantine after this many failures WITH NO success
                               # in between (the counter resets to 0 on any
                               # successful checkpoint write — so only genuinely
                               # dead rows, which never resolve, ever quarantine).
MAX_RESOLVE_AGE_DAYS = 90      # anchors older than this, still unresolved = dead


def _bump_attempts(conn, outcome_id) -> None:
    """Increment a row's failed-resolve counter so persistently-unresolvable
    rows get quarantined (see MAX_RESOLVE_ATTEMPTS) instead of retried forever."""
    try:
        conn.execute(
            "UPDATE signal_outcomes "
            "SET resolve_attempts = COALESCE(resolve_attempts, 0) + 1 WHERE id = ?",
            (outcome_id,),
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("resolve_attempts bump failed for %s: %s", outcome_id, exc)


def update_due_outcomes(
    *,
    batch_size: int = 200,
    fetcher: Optional[PriceFetcher] = None,
) -> UpdateStats:
    """Fill 1d / 5d / 20d prices for outcomes whose checkpoints have arrived."""
    stats = UpdateStats()
    fetcher = fetcher or PriceFetcher()

    now = datetime.now(timezone.utc)
    # Anchors older than this that are STILL unresolved are dead (delisted / no
    # forward bars anywhere) — skip so the loop reaches the resolvable backlog.
    floor_iso = (now - timedelta(days=MAX_RESOLVE_AGE_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    for window_days, price_col, ts_col in _CHECKPOINTS:
        target = now - timedelta(days=window_days)
        target_iso = target.strftime("%Y-%m-%dT%H:%M:%SZ")
        with get_connection() as conn:
            # Candidates: outcomes whose anchor is N+ days old, lack this
            # checkpoint, haven't exhausted resolve attempts, and aren't
            # ancient-dead. The attempt/age guards break the poison-pill that
            # starved the model of fresh labels.
            rows = conn.execute(
                f"""
                SELECT id AS outcome_id, score_id, ticker, price_at_flag,
                       price_at_flag_ts
                FROM signal_outcomes
                WHERE {price_col} IS NULL
                  AND price_at_flag_ts IS NOT NULL
                  AND price_at_flag_ts <= ?
                  AND price_at_flag_ts >= ?
                  AND COALESCE(resolve_attempts, 0) < ?
                ORDER BY price_at_flag_ts ASC
                LIMIT ?
                """,
                (target_iso, floor_iso, MAX_RESOLVE_ATTEMPTS, batch_size),
            ).fetchall()

            attr = f"candidates_{window_days}d"
            setattr(stats, attr, len(rows))
            if not rows:
                continue

            for row in rows:
                anchor_ts = row["price_at_flag_ts"]
                try:
                    anchor_dt = datetime.strptime(
                        anchor_ts, "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    log.warning("bad anchor ts on outcome %s: %r", row["outcome_id"], anchor_ts)
                    stats.failed += 1
                    _bump_attempts(conn, row["outcome_id"])
                    continue

                target_date = anchor_dt + timedelta(days=window_days)
                lookup = fetcher.fetch_closing_on_or_after(row["ticker"], target_date)
                if lookup is None:
                    stats.failed += 1
                    _bump_attempts(conn, row["outcome_id"])
                    continue
                close, close_ts = lookup

                ret_pct: Optional[float] = None
                if row["price_at_flag"] is not None and close:
                    try:
                        ret_pct = (close - float(row["price_at_flag"])) / float(row["price_at_flag"]) * 100.0
                    except (TypeError, ValueError, ZeroDivisionError):
                        ret_pct = None

                try:
                    conn.execute(
                        f"""
                        UPDATE signal_outcomes
                           SET {price_col} = ?,
                               {ts_col}    = ?,
                               return_{window_days}d_pct = ?,
                               resolve_attempts = 0,
                               fully_resolved = CASE
                                   WHEN ? = 'price_20d' THEN 1
                                   ELSE fully_resolved
                               END
                         WHERE id = ?
                        """,
                        (close, close_ts, ret_pct, price_col, row["outcome_id"]),
                    )
                    stats.updated += 1
                    if price_col == "price_20d":
                        stats.fully_resolved += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("outcome update failed id=%s: %s", row["outcome_id"], exc)
                    stats.failed += 1

    log.info(
        "update_due_outcomes: candidates 1d/5d/20d = %d/%d/%d updated=%d failed=%d resolved=%d",
        stats.candidates_1d, stats.candidates_5d, stats.candidates_20d,
        stats.updated, stats.failed, stats.fully_resolved,
    )
    return stats


# ---------------------------------------------------------------------------
# 3. Hit-rate aggregation (read-side helper, used by the dashboard)
# ---------------------------------------------------------------------------


def hit_rates_by_signal_class(
    *,
    min_samples: int = 20,
) -> list[dict]:
    """Return per signal_class aggregate stats for the dashboard.

    Output rows have ``signal_class, samples, avg_return_5d_pct, hit_rate_5d``
    and similar for 1d / 20d. signal_classes with fewer than ``min_samples``
    are excluded since their stats aren't yet meaningful.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                ss.signal_class,
                COUNT(*) AS samples,
                ROUND(AVG(so.return_1d_pct), 3) AS avg_return_1d,
                ROUND(AVG(so.return_5d_pct), 3) AS avg_return_5d,
                ROUND(AVG(so.return_20d_pct), 3) AS avg_return_20d,
                ROUND(AVG(CASE WHEN so.return_1d_pct > 0 THEN 1.0 ELSE 0.0 END), 3) AS hit_rate_1d,
                ROUND(AVG(CASE WHEN so.return_5d_pct > 0 THEN 1.0 ELSE 0.0 END), 3) AS hit_rate_5d,
                ROUND(AVG(CASE WHEN so.return_20d_pct > 0 THEN 1.0 ELSE 0.0 END), 3) AS hit_rate_20d
            FROM signal_scores ss
            JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE ss.signal_class IS NOT NULL
            GROUP BY ss.signal_class
            HAVING COUNT(*) >= ?
            ORDER BY samples DESC
            """,
            (min_samples,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
