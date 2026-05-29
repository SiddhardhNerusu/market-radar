"""Snapshot pending outcomes against the test DB and show what we recorded."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.outcomes import snapshot_pending_outcomes  # noqa: E402
from market_radar.storage import get_connection  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = snapshot_pending_outcomes(batch_size=300)
    print(f"\nSnapshot: pending={stats.pending} snapped={stats.snapped} failed={stats.failed}")
    print()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT so.ticker, so.price_at_flag, so.price_at_flag_ts,
                   ss.composite_score, ss.event_type, rs.title
            FROM signal_outcomes so
            JOIN signal_scores ss ON ss.id = so.score_id
            JOIN raw_signals rs ON rs.id = ss.signal_id
            ORDER BY ss.composite_score DESC, so.id ASC
            LIMIT 20
            """
        ).fetchall()

        print("Top 20 outcomes by composite score (anchor prices captured):")
        for row in rows:
            price = f"${row['price_at_flag']:>10.2f}" if row["price_at_flag"] is not None else f"{'(unpriced)':>12}"
            title = (row["title"] or "")[:60]
            print(
                f"  comp={row['composite_score']:>4.1f}  "
                f"{row['ticker']:<10} "
                f"{price}  "
                f"{row['event_type']:<22} "
                f"{title}"
            )
        print()

        # Coverage report
        priced = conn.execute(
            "SELECT COUNT(*) AS n FROM signal_outcomes WHERE price_at_flag IS NOT NULL"
        ).fetchone()
        unpriced = conn.execute(
            "SELECT COUNT(*) AS n FROM signal_outcomes WHERE price_at_flag IS NULL"
        ).fetchone()
        total = (priced["n"] or 0) + (unpriced["n"] or 0)
        print(f"Coverage: {priced['n']}/{total} outcomes priced "
              f"({100 * (priced['n'] or 0) / max(total, 1):.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
