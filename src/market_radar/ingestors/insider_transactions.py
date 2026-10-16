"""Form 4 bodies -> structured ``insider_transactions`` rows.

This is the insider lane's input. It used to live only in a manual script;
now the daemon runs it every minute right behind SEC body hydration, so a
code-P purchase is a lane candidate within ~2 minutes of hitting EDGAR.
Idempotent: a Form 4 already parsed is never re-inserted.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from ..storage import get_connection
from .sec_form4_parser import parse_form4_xml

log = logging.getLogger("marketradar.ingestors.insider")

_DDL = """
CREATE TABLE IF NOT EXISTS insider_transactions (
    signal_id        INTEGER NOT NULL,
    ticker           TEXT,
    insider_name     TEXT,
    transaction_code TEXT,
    shares           REAL,
    price            REAL,
    is_acquired      INTEGER,
    officer_title    TEXT,
    is_officer       INTEGER,
    is_director      INTEGER,
    is_10pct         INTEGER,
    role_score       INTEGER,
    report_date      TEXT,
    ingested_at      TEXT NOT NULL,
    is_10b5_1        INTEGER DEFAULT 0,
    PRIMARY KEY (signal_id, transaction_code, shares)
)
"""


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(insider_transactions)").fetchall()]
    if "is_10b5_1" not in cols:
        conn.execute("ALTER TABLE insider_transactions ADD COLUMN is_10b5_1 INTEGER DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ins_tx_ticker ON insider_transactions(ticker, report_date)")


def refresh_insider_transactions(*, conn: Optional[sqlite3.Connection] = None, limit: Optional[int] = None,
                                 dry_run: bool = False) -> dict:
    """Parse every hydrated, not-yet-parsed Form 4 body. Returns counts."""
    own = conn is None
    cm = get_connection() if own else None
    c = cm.__enter__() if own else conn
    try:
        ensure_table(c)
        sql = """
            SELECT rs.id AS signal_id, rs.body, rs.published_at
            FROM raw_signals rs
            WHERE rs.source LIKE 'sec_edgar%'
              AND rs.title LIKE '4 - %'
              AND rs.body IS NOT NULL
              AND length(rs.body) >= 200
              AND NOT EXISTS (SELECT 1 FROM insider_transactions ix WHERE ix.signal_id = rs.id)
            ORDER BY rs.id DESC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = c.execute(sql).fetchall()
        inserted = skipped = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for r in rows:
            parsed = parse_form4_xml(r["body"] or "")
            if not parsed or not parsed.transactions:
                skipped += 1
                continue
            pub = r["published_at"] or ""
            for tx in parsed.transactions:
                if dry_run:
                    continue
                c.execute(
                    """INSERT OR REPLACE INTO insider_transactions
                       (signal_id, ticker, insider_name, transaction_code, shares, price, is_acquired,
                        officer_title, is_officer, is_director, is_10pct, role_score, report_date,
                        ingested_at, is_10b5_1)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (r["signal_id"], parsed.issuer_ticker, parsed.insider_name, tx.transaction_code,
                     tx.shares, tx.price_per_share, 1 if tx.is_acquired else 0, parsed.officer_title,
                     int(parsed.is_officer), int(parsed.is_director), int(parsed.is_10pct),
                     parsed.insider_role_score, parsed.period_of_report or (pub[:10] if pub else None),
                     now, int(parsed.is_10b5_1)),
                )
                inserted += 1
        if rows:
            log.info("insider refresh: bodies=%d inserted=%d unparseable=%d", len(rows), inserted, skipped)
        return {"bodies": len(rows), "inserted": inserted, "skipped": skipped}
    finally:
        if own:
            cm.__exit__(None, None, None)
