#!/usr/bin/env python3
"""Resolve edge_ledger rows (5d / 20d raw + SPY-adjusted returns) and print
the promotion gate verdict for each cohort.

    python scripts/edge_ledger_update.py            # resolve + report
    python scripts/edge_ledger_update.py --report   # report only
    python scripts/edge_ledger_update.py --horizon 20d --families 2

Run daily after the close (a launchd job is provided). The gate reads only
this table, so this is the single honest answer to "is there an edge yet?".
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", action="store_true", help="skip resolution, just evaluate")
    p.add_argument("--horizon", default="5d", choices=["5d", "20d"])
    p.add_argument("--families", type=int, default=1, help="Bonferroni divisor: cohorts you are testing")
    p.add_argument("--min-days", type=int, default=15)
    p.add_argument("--exclude-day", action="append", default=[], help="event_day to drop (e.g. a backfill batch)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    log = logging.getLogger("edge_ledger_update")

    from market_radar.config import CONFIG
    from market_radar.storage import get_connection, init_db
    from market_radar.lane import promotion
    from market_radar.lane.ledger import resolve_ledger

    if not CONFIG.db_path.exists():
        log.error("no database at %s", CONFIG.db_path)
        return 1
    init_db()
    today = date.today().isoformat()

    if not args.report:
        try:
            from market_radar.execution.alpaca_client import AlpacaClient
            client = AlpacaClient()

            def bars_fn(symbol: str, start_day: str, end_day: str) -> list[dict]:
                end = (date.fromisoformat(end_day) + timedelta(days=1)).isoformat()
                return client.get_daily_bars(symbol, start_day, end, limit=60)
        except Exception as exc:  # noqa: BLE001
            log.error("Alpaca unavailable (%s); run with --report or set ALPACA keys", exc)
            return 1
        with get_connection() as conn:
            n = resolve_ledger(conn, bars_fn=bars_fn, today=today)
        log.info("resolved %d ledger row(s)", n)

    with get_connection() as conn:
        cohorts = [r[0] for r in conn.execute("SELECT DISTINCT candidate FROM edge_ledger").fetchall()]
        if not cohorts:
            log.info("edge_ledger is empty — the lane has not seen a qualifying candidate yet")
            return 0
        for c in cohorts:
            rows = [dict(r) for r in conn.execute("SELECT * FROM edge_ledger WHERE candidate=?", (c,)).fetchall()]
            res = promotion.evaluate(rows, horizon=args.horizon, min_days=args.min_days,
                                     exclude_days=args.exclude_day, n_families=args.families)
            traded = sum(1 for r in rows if r.get("traded"))
            print(f"\n== {c}  (rows={len(rows)}, traded={traded}, horizon={args.horizon})")
            print(res.summary())
            if res.n_days < args.min_days:
                print(f"  underpowered: {res.n_days}/{args.min_days} distinct days — keep accruing, do not size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
