"""Idempotent quarantine of corrupt outcome labels (P1 rebuild).

Flags signal_outcomes rows whose return is a data artifact — a sub-$1 anchor
division blowup or |return| > _MAX_ABS_RETURN_PCT (split / ticker-reuse) — as
``data_corrupt=1`` AND NULLs their ``return_*_pct`` columns, so NO unfiltered
``AVG(return)`` read (dashboard tiles, hit-rate stats) can resurrect the
+3,002,400% artifacts. Safe to run repeatedly; mirrors
``market_radar.outcomes.tracker.classify_return`` thresholds.

Usage:  PYTHONPATH=src .venv/bin/python scripts/quarantine_corrupt_outcomes.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.outcomes.tracker import _MAX_ABS_RETURN_PCT, _MIN_ANCHOR_USD
from market_radar.storage import get_connection


def main() -> int:
    with get_connection() as conn:
        conn.execute(
            f"""
            UPDATE signal_outcomes SET data_corrupt = 1
             WHERE (price_at_flag IS NOT NULL AND price_at_flag < {_MIN_ANCHOR_USD})
                OR ABS(COALESCE(return_1d_pct, 0))  > {_MAX_ABS_RETURN_PCT}
                OR ABS(COALESCE(return_5d_pct, 0))  > {_MAX_ABS_RETURN_PCT}
                OR ABS(COALESCE(return_20d_pct, 0)) > {_MAX_ABS_RETURN_PCT}
            """
        )
        flagged = conn.execute(
            "SELECT COUNT(*) FROM signal_outcomes WHERE data_corrupt = 1"
        ).fetchone()[0]
        # NULL the artifact returns so even an unfiltered AVG() read is clean.
        conn.execute(
            """
            UPDATE signal_outcomes
               SET return_1d_pct = NULL, return_5d_pct = NULL, return_20d_pct = NULL
             WHERE data_corrupt = 1
               AND (return_1d_pct IS NOT NULL
                    OR return_5d_pct IS NOT NULL
                    OR return_20d_pct IS NOT NULL)
            """
        )
        leaked = conn.execute(
            "SELECT COUNT(*) FROM signal_outcomes "
            "WHERE data_corrupt = 1 AND (return_1d_pct IS NOT NULL "
            "OR return_5d_pct IS NOT NULL OR return_20d_pct IS NOT NULL)"
        ).fetchone()[0]
    print(f"corrupt rows flagged: {flagged}; corrupt rows still carrying a return (must be 0): {leaked}")
    return 0 if leaked == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
