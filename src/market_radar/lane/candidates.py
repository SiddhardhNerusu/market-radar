"""Form 4 code-P purchases -> deduped InsiderCandidate list.

``aggregate_purchases`` is pure and unit-tested. ``fetch_recent_insider_rows``
is the one SQL read. The strict one-bet-per-(ticker, event_day) rule lives
here AND in the ``lane_decisions`` UNIQUE constraint, so it cannot be
bypassed by a restart or a second process.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from .sessions import event_day_for


@dataclass
class InsiderCandidate:
    ticker: str
    event_day: str
    published_at: str            # earliest filing in the aggregate
    notional_usd: float
    role_score: int
    n_purchases: int
    signal_ids: list[int] = field(default_factory=list)


def _f(v, d=0.0) -> float:
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


def aggregate_purchases(
    rows: Iterable[dict],
    *,
    min_notional_usd: float,
    min_price: float,
    min_role_score: int = 0,
    exclude_10b5_1: bool = True,
) -> list[InsiderCandidate]:
    """Collapse insider_transactions rows to one candidate per (ticker,
    event_day). Only code 'P' + acquired rows count; plan (10b5-1) trades are
    excluded by default; the per-share price must clear ``min_price`` (the
    tier where the tilt was not a cost artifact). Same-day purchases by one or
    several insiders are SUMMED into a single bet — never counted twice."""
    agg: dict[tuple[str, str], InsiderCandidate] = {}
    for r in rows:
        if str(r.get("transaction_code") or "").upper() != "P":
            continue
        if int(r.get("is_acquired") or 0) != 1:
            continue
        if exclude_10b5_1 and int(r.get("is_10b5_1") or 0) == 1:
            continue
        ticker = str(r.get("ticker") or "").strip().upper()
        if not ticker or "/" in ticker or len(ticker) > 6:
            continue
        px = _f(r.get("price"))
        sh = _f(r.get("shares"))
        if px < min_price or sh <= 0:
            continue
        day = event_day_for(r.get("published_at"))
        if not day:
            continue
        key = (ticker, day)
        role = int(r.get("role_score") or 0)
        sid = r.get("signal_id")
        c = agg.get(key)
        if c is None:
            agg[key] = InsiderCandidate(
                ticker=ticker, event_day=day, published_at=str(r.get("published_at")),
                notional_usd=px * sh, role_score=role, n_purchases=1,
                signal_ids=[int(sid)] if sid is not None else [],
            )
        else:
            c.notional_usd += px * sh
            c.role_score = max(c.role_score, role)
            c.n_purchases += 1
            if sid is not None and int(sid) not in c.signal_ids:
                c.signal_ids.append(int(sid))
            if str(r.get("published_at")) < c.published_at:
                c.published_at = str(r.get("published_at"))
    out = [c for c in agg.values()
           if c.notional_usd >= min_notional_usd and c.role_score >= min_role_score]
    out.sort(key=lambda c: (-c.notional_usd, c.ticker))
    return out


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    try:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except sqlite3.Error:
        return False
    return col in cols


def fetch_recent_insider_rows(conn: sqlite3.Connection, *, since_utc: datetime) -> list[dict]:
    """Join insider_transactions to raw_signals for filings published since
    ``since_utc``. Tolerates an older DB without the is_10b5_1 column and a
    DB where the table does not exist yet (returns [])."""
    if not _has_column(conn, "insider_transactions", "ticker"):
        return []
    tenb = "COALESCE(ix.is_10b5_1, 0)" if _has_column(conn, "insider_transactions", "is_10b5_1") else "0"
    sql = f"""
        SELECT ix.signal_id, ix.ticker, ix.transaction_code, ix.shares, ix.price,
               ix.is_acquired, ix.role_score, {tenb} AS is_10b5_1,
               rs.published_at
        FROM insider_transactions ix
        JOIN raw_signals rs ON rs.id = ix.signal_id
        WHERE rs.published_at >= ?
          AND ix.transaction_code = 'P'
    """
    cur = conn.execute(sql, (since_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),))
    return [dict(r) for r in cur.fetchall()]


def recent_candidates(conn: sqlite3.Connection, *, now_utc: datetime, freshness_hours: int,
                      min_notional_usd: float, min_price: float, min_role_score: int,
                      exclude_10b5_1: bool) -> list[InsiderCandidate]:
    since = now_utc - timedelta(hours=freshness_hours)
    rows = fetch_recent_insider_rows(conn, since_utc=since)
    return aggregate_purchases(
        rows, min_notional_usd=min_notional_usd, min_price=min_price,
        min_role_score=min_role_score, exclude_10b5_1=exclude_10b5_1,
    )


def already_decided(conn: sqlite3.Connection, ticker: str, event_day: str) -> bool:
    r = conn.execute("SELECT 1 FROM lane_decisions WHERE ticker=? AND event_day=?",
                     (ticker, event_day)).fetchone()
    return r is not None


def in_cooldown(conn: sqlite3.Connection, ticker: str, *, now_utc: datetime,
                cooldown_days: int) -> Optional[str]:
    """Return the decided_at of a recent PLACED bet on this ticker inside the
    cooldown window, else None. Persisted, so it survives restarts."""
    since = (now_utc - timedelta(days=cooldown_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = conn.execute(
        """SELECT decided_at FROM lane_decisions
           WHERE ticker=? AND outcome IN ('placed','pending_submit') AND decided_at >= ?
           ORDER BY decided_at DESC LIMIT 1""", (ticker, since)).fetchone()
    return r[0] if r else None
