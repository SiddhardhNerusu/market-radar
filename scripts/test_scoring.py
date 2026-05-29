"""Run the scorer over whatever's in the DB and audit the top signals."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.scoring import score_pending  # noqa: E402
from market_radar.storage import get_connection, init_db  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    init_db()
    stats = score_pending()
    print(f"Scoring: candidates={stats.candidates} scored={stats.scored} errors={stats.errors}")
    print()

    with get_connection() as conn:
        print("Score distribution buckets:")
        rows = conn.execute(
            """
            SELECT
                CASE
                    WHEN composite_score >= 8.5 THEN 'A. 8.5–10  (very strong)'
                    WHEN composite_score >= 7.5 THEN 'B. 7.5–8.5 (strong)'
                    WHEN composite_score >= 6.0 THEN 'C. 6.0–7.5 (medium)'
                    WHEN composite_score >= 4.0 THEN 'D. 4.0–6.0 (weak)'
                    ELSE 'E. <4.0    (noise)'
                END AS bucket,
                COUNT(*) AS n
            FROM signal_scores
            GROUP BY bucket
            ORDER BY bucket
            """
        ).fetchall()
        for row in rows:
            print(f"  {row['bucket']:<30} {row['n']}")
        print()

        print("Top 20 signals by composite score:")
        rows = conn.execute(
            """
            SELECT ss.composite_score, ss.ticker, ss.event_type, ss.sentiment,
                   ss.factual, ss.corroboration_count, rs.source, rs.title
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            ORDER BY ss.composite_score DESC
            LIMIT 20
            """
        ).fetchall()
        for row in rows:
            sent_sign = "+" if (row["sentiment"] or 0) > 0 else "-" if (row["sentiment"] or 0) < 0 else "·"
            fact = "F" if row["factual"] == 1 else ("S" if row["factual"] == 0 else "?")
            title = (row["title"] or "")[:70]
            print(
                f"  {row['composite_score']:>4.1f}  "
                f"{row['ticker']:<10} "
                f"{row['event_type']:<22} "
                f"{sent_sign}{abs(row['sentiment'] or 0):.2f} {fact} "
                f"corr={row['corroboration_count']:<2} "
                f"[{row['source']:<22}] {title}"
            )
        print()

        print("Event-type breakdown:")
        rows = conn.execute(
            """
            SELECT event_type, COUNT(*) AS n,
                   ROUND(AVG(composite_score), 2) AS avg_score,
                   ROUND(MAX(composite_score), 2) AS max_score
            FROM signal_scores
            GROUP BY event_type
            ORDER BY n DESC
            """
        ).fetchall()
        for row in rows:
            print(
                f"  {row['event_type']:<24} n={row['n']:<4} "
                f"avg_score={row['avg_score']:<6} max={row['max_score']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
