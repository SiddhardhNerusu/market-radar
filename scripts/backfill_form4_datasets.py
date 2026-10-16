#!/usr/bin/env python3
"""Rebuild Form 4 purchase history from the SEC Insider Transactions Data Sets.

    python scripts/backfill_form4_datasets.py --since 2024-01-01            # all quarters to now
    python scripts/backfill_form4_datasets.py --since 2025-07-01 --until 2025-09-30
    python scripts/backfill_form4_datasets.py --since 2024-01-01 --accept-times  # + exact EDGAR timestamps

Writes raw_signals (one stub per filing, source 'sec_edgar_form4_ds') and
insider_transactions (code-P rows by default; --codes P,S for sales too).
Idempotent. Then run scripts/event_study_form4.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", required=True, help="YYYY-MM-DD; quarters from this date are loaded")
    p.add_argument("--until", default=None, help="YYYY-MM-DD (default today)")
    p.add_argument("--codes", default="P", help="transaction codes to keep, comma-separated")
    p.add_argument("--accept-times", action="store_true",
                   help="fetch exact EDGAR acceptance timestamps for rows that qualify for the lane")
    p.add_argument("--min-notional", type=float, default=25_000.0)
    p.add_argument("--min-price", type=float, default=10.0)
    p.add_argument("--max-accept-requests", type=int, default=2000)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
    log = logging.getLogger("backfill_form4_datasets")

    from market_radar.storage import get_connection, init_db
    from market_radar.backfill.form4_datasets import (
        download_quarter, enrich_accept_times, parse_quarter, quarters_between, write_rows)

    init_db()
    start = date.fromisoformat(args.since)
    end = date.fromisoformat(args.until) if args.until else date.today()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    totals = {"quarters": 0, "rows": 0, "signals_inserted": 0, "transactions_inserted": 0}
    for (y, q) in quarters_between(start, end):
        zp = download_quarter(y, q)
        if zp is None:
            continue
        rows = parse_quarter(zp, codes=codes)
        rows = [r for r in rows if start.isoformat() <= r.filing_date <= end.isoformat()]
        with get_connection() as conn:
            res = write_rows(conn, rows)
        totals["quarters"] += 1
        totals["rows"] += len(rows)
        totals["signals_inserted"] += res["signals_inserted"]
        totals["transactions_inserted"] += res["transactions_inserted"]
        log.info("%dq%d: %d rows -> +%d filings, +%d transactions", y, q, len(rows),
                 res["signals_inserted"], res["transactions_inserted"])
    log.info("done: %s", totals)
    if args.accept_times:
        with get_connection() as conn:
            n = enrich_accept_times(conn, min_notional=args.min_notional, min_price=args.min_price,
                                    max_requests=args.max_accept_requests)
        log.info("acceptance timestamps filled for %d qualifying filings", n)
    with get_connection() as conn:
        n_q = conn.execute(
            """SELECT COUNT(*) FROM insider_transactions ix JOIN raw_signals rs ON rs.id=ix.signal_id
               WHERE rs.source=? AND ix.transaction_code='P' AND ix.is_acquired=1
                 AND ix.price >= ? AND ix.shares*ix.price >= ?""",
            ("sec_edgar_form4_ds", args.min_price, args.min_notional)).fetchone()[0]
    log.info("qualifying purchases (>= $%.0f, px >= $%.0f) in the data-set rows: %d",
             args.min_notional, args.min_price, n_q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
