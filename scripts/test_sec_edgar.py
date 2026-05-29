"""Run one SEC EDGAR poll cycle and show what landed in the DB."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import CONFIG  # noqa: E402
from market_radar.ingestors import SecEdgarIngestor  # noqa: E402
from market_radar.storage import get_connection, init_db  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    init_db()
    ingestor = SecEdgarIngestor()
    print(f"Polling SEC EDGAR for forms: {ingestor.forms}")
    result = ingestor.poll()
    print(f"Result: fetched={result.fetched} inserted={result.inserted} "
          f"duplicates={result.duplicates} errors={result.errors}")
    print()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT rs.id, rs.published_at, rs.title, rs.url,
                   GROUP_CONCAT(st.ticker, ',') AS tickers,
                   rs.raw_payload
            FROM raw_signals rs
            LEFT JOIN signal_tickers st ON st.signal_id = rs.id
            WHERE rs.source = 'sec_edgar'
            GROUP BY rs.id
            ORDER BY rs.id DESC
            LIMIT 15
            """
        ).fetchall()

        if not rows:
            print("No signals stored.")
            return 0

        print(f"Latest {len(rows)} SEC filings stored:\n")
        for row in rows:
            ticker = row["tickers"] or "—"
            title = (row["title"] or "")[:80]
            print(f"  [{ticker:<10}] {row['published_at'] or '?':<22} {title}")

        # Per-form-type breakdown
        print()
        breakdown = conn.execute(
            """
            SELECT json_extract(raw_payload, '$.form') AS form,
                   COUNT(*) AS n,
                   SUM(CASE WHEN st.ticker IS NOT NULL THEN 1 ELSE 0 END) AS matched
            FROM raw_signals rs
            LEFT JOIN signal_tickers st ON st.signal_id = rs.id
            WHERE rs.source = 'sec_edgar'
            GROUP BY form
            ORDER BY n DESC
            """
        ).fetchall()
        print("Per-form-type breakdown:")
        for row in breakdown:
            print(f"  {row['form']:<12} count={row['n']:<4} ticker-matched={row['matched']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
