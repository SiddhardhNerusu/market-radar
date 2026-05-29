"""Poll Tier 1 + Tier 2 ingestors, then audit the resulting signals.

Audit means:
  - per-source counts (how many entries each feed produced)
  - top tickers (which symbols are getting the most mentions)
  - ticker-extraction quality spot-check (sample of titles + extracted tickers)
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.ingestors import RssNewsIngestor, SecEdgarIngestor  # noqa: E402
from market_radar.storage import get_connection, init_db  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    init_db()

    sec = SecEdgarIngestor()
    rss = RssNewsIngestor()

    print("Polling SEC EDGAR …")
    r1 = sec.poll()
    print(f"  fetched={r1.fetched} inserted={r1.inserted} dup={r1.duplicates} err={r1.errors}")

    print("Polling RSS news …")
    r2 = rss.poll()
    print(f"  fetched={r2.fetched} inserted={r2.inserted} dup={r2.duplicates} err={r2.errors}")
    print()

    with get_connection() as conn:
        # Per-source counts
        rows = conn.execute(
            """
            SELECT source, source_tier, COUNT(*) AS n
            FROM raw_signals
            GROUP BY source
            ORDER BY n DESC
            """
        ).fetchall()
        print("Per-source counts in raw_signals:")
        for row in rows:
            print(f"  T{row['source_tier']}  {row['source']:<28} {row['n']}")
        print()

        # Top tickers
        rows = conn.execute(
            """
            SELECT st.ticker, COUNT(DISTINCT rs.id) AS mentions,
                   AVG(st.confidence) AS avg_conf
            FROM signal_tickers st
            JOIN raw_signals rs ON rs.id = st.signal_id
            GROUP BY st.ticker
            ORDER BY mentions DESC
            LIMIT 20
            """
        ).fetchall()
        print("Top 20 most-mentioned tickers:")
        for row in rows:
            print(f"  {row['ticker']:<10} mentions={row['mentions']:<4} avg_conf={row['avg_conf']:.2f}")
        print()

        # Spot-check ticker extraction quality on RSS signals (Tier 2)
        rows = conn.execute(
            """
            SELECT rs.id, rs.title, rs.source,
                   GROUP_CONCAT(st.ticker || ':' || ROUND(st.confidence, 2), ',') AS tickers
            FROM raw_signals rs
            JOIN signal_tickers st ON st.signal_id = rs.id
            WHERE rs.source_tier = 2
            GROUP BY rs.id
            ORDER BY RANDOM()
            LIMIT 15
            """
        ).fetchall()
        print("Random sample — RSS headlines + extracted tickers (sanity check):")
        for row in rows:
            title = (row["title"] or "")[:90]
            print(f"  [{row['tickers']:<25}] {row['source']:<22} {title}")
        print()

        # Untagged Tier-2 rows
        untagged = conn.execute(
            """
            SELECT COUNT(*) AS n FROM raw_signals rs
            LEFT JOIN signal_tickers st ON st.signal_id = rs.id
            WHERE rs.source_tier = 2 AND st.signal_id IS NULL
            """
        ).fetchone()
        tier2_total = conn.execute(
            "SELECT COUNT(*) AS n FROM raw_signals WHERE source_tier = 2"
        ).fetchone()
        print(
            f"Tier 2 ticker-match rate: "
            f"{tier2_total['n'] - untagged['n']} / {tier2_total['n']} matched "
            f"({100 * (1 - untagged['n'] / max(tier2_total['n'], 1)):.1f}%)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
