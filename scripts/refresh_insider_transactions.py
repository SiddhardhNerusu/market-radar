"""Parse stored Form 4 bodies → structured insider_transactions table.

Runs over raw_signals where source LIKE 'sec_edgar%' AND title LIKE '4 - %'
AND body IS NOT NULL. Idempotent. Adds:
  - insider_transactions  (one row per (signal_id, transaction))
  - insider_summary_30d   (rolling aggregate per ticker, for fast features)

The model's ``insider_recent_buys_30d`` / ``insider_recent_sells_30d`` /
``insider_role_score`` features read from these.
"""
from __future__ import annotations

import argparse, logging, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.ingestors.sec_form4_parser import parse_form4_xml  # noqa: E402
from market_radar.storage import get_connection, init_db  # noqa: E402

log = logging.getLogger("refresh_insider_transactions")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    init_db()
    with get_connection() as conn:
        conn.execute("""
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
                PRIMARY KEY (signal_id, transaction_code, shares)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ins_tx_ticker ON insider_transactions(ticker, report_date)")

        # Pull body-fetched Form 4s that we haven't parsed yet
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
        if args.limit:
            sql += f" LIMIT {int(args.limit)}"
        rows = conn.execute(sql).fetchall()
        log.info("Form 4 bodies to parse: %d", len(rows))

        inserted = 0
        skipped = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for r in rows:
            parsed = parse_form4_xml(r["body"] or "")
            if not parsed or not parsed.transactions:
                skipped += 1
                continue
            for tx in parsed.transactions:
                if args.dry_run:
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO insider_transactions
                    (signal_id, ticker, insider_name, transaction_code, shares,
                     price, is_acquired, officer_title, is_officer, is_director,
                     is_10pct, role_score, report_date, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r["signal_id"], parsed.issuer_ticker, parsed.insider_name,
                     tx.transaction_code, tx.shares, tx.price_per_share,
                     1 if tx.is_acquired else 0, parsed.officer_title,
                     int(parsed.is_officer), int(parsed.is_director),
                     int(parsed.is_10pct), parsed.insider_role_score,
                     parsed.period_of_report or r["published_at"][:10] if r["published_at"] else None,
                     now),
                )
                inserted += 1
    log.info("Done. transactions_inserted=%d skipped=%d", inserted, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
