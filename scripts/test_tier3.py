"""Run Tier 3 polls and audit the results."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.ingestors import (  # noqa: E402
    RedditPublicIngestor,
    StockTwitsTrendingIngestor,
)
from market_radar.storage import get_connection, init_db  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    init_db()

    print("Polling Reddit (public JSON) …")
    rr = RedditPublicIngestor().poll()
    print(f"  fetched={rr.fetched} inserted={rr.inserted} dup={rr.duplicates} err={rr.errors}")

    print("Polling StockTwits trending …")
    rs = StockTwitsTrendingIngestor().poll()
    print(f"  fetched={rs.fetched} inserted={rs.inserted} dup={rs.duplicates} err={rs.errors}")
    print()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT source, COUNT(*) AS n
            FROM raw_signals
            WHERE source_tier = 3
            GROUP BY source
            ORDER BY n DESC
            """
        ).fetchall()
        print("Tier 3 per-source counts:")
        for row in rows:
            print(f"  {row['source']:<30} {row['n']}")
        print()

        rows = conn.execute(
            """
            SELECT st.ticker, COUNT(DISTINCT rs.id) AS mentions,
                   AVG(st.confidence) AS avg_conf
            FROM signal_tickers st
            JOIN raw_signals rs ON rs.id = st.signal_id
            WHERE rs.source_tier = 3
            GROUP BY st.ticker
            ORDER BY mentions DESC
            LIMIT 15
            """
        ).fetchall()
        print("Top Tier-3 tickers:")
        for row in rows:
            print(f"  {row['ticker']:<10} mentions={row['mentions']:<4} avg_conf={row['avg_conf']:.2f}")
        print()

        rows = conn.execute(
            """
            SELECT rs.title, rs.body, rs.source, rs.author,
                   GROUP_CONCAT(st.ticker, ',') AS tickers
            FROM raw_signals rs
            LEFT JOIN signal_tickers st ON st.signal_id = rs.id
            WHERE rs.source_tier = 3
            GROUP BY rs.id
            ORDER BY rs.id DESC
            LIMIT 12
            """
        ).fetchall()
        print("Recent Tier-3 signals (sample):")
        for row in rows:
            tickers = row["tickers"] or "—"
            text = (row["title"] or row["body"] or "")[:80]
            print(f"  [{tickers:<20}] {row['source']:<28} @{row['author'] or '?':<15} {text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
