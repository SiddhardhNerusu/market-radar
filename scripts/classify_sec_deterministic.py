"""Deterministic SEC classification (ingestion blueprint #3).

Routes 8-K Item codes + SEC forms to real event_types (sec_item_codes.sec_event_type)
so the SEC bulk stops falling into 'other' — NO LLM. Only OVERWRITES vague labels
('other' / 'material_event' / a form-tag-bug 'insider_transaction'); genuine
specific labels (LLM or heuristic) are preserved. Reports the 8-K 'other' rate
before/after. Idempotent.

Usage:  PYTHONPATH=src .venv/bin/python scripts/classify_sec_deterministic.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.ingestors.sec_item_codes import sec_event_type
from market_radar.storage import get_connection

_VAGUE = "('other','material_event','insider_transaction')"


def _eightk_other(conn):
    row = conn.execute(
        "SELECT COUNT(*) n, "
        "SUM(CASE WHEN COALESCE(ss.event_type,'other') IN ('other','material_event') "
        "         THEN 1 ELSE 0 END) o "
        "FROM signal_scores ss JOIN raw_signals rs ON rs.id=ss.signal_id "
        "WHERE rs.source='sec_edgar' "
        "  AND json_extract(rs.raw_payload,'$.form') IN ('8-K','8-K/A','6-K','6-K/A')"
    ).fetchone()
    n, o = (row["n"] or 0), (row["o"] or 0)
    return n, o, (o / n * 100 if n else 0.0)


def main() -> int:
    with get_connection() as conn:
        n0, o0, r0 = _eightk_other(conn)
        print(f"8-K 'other' BEFORE: {o0}/{n0} = {r0:.1f}%")
        rows = conn.execute(
            "SELECT rs.id sid, rs.body body, "
            "       json_extract(rs.raw_payload,'$.form') form "
            "FROM raw_signals rs "
            "WHERE rs.source='sec_edgar' AND rs.body IS NOT NULL"
        ).fetchall()

    up_ss = up_lc = 0
    with get_connection() as conn:
        for r in rows:
            et = sec_event_type(r["form"], r["body"])
            if not et:
                continue  # genuinely ambiguous (or a Form-4) — leave for LLM
            up_ss += conn.execute(
                f"UPDATE signal_scores SET event_type=? WHERE signal_id=? "
                f"AND COALESCE(event_type,'other') IN {_VAGUE}",
                (et, r["sid"])).rowcount
            up_lc += conn.execute(
                f"UPDATE llm_classifications SET event_type=? WHERE signal_id=? "
                f"AND COALESCE(event_type,'other') IN {_VAGUE}",
                (et, r["sid"])).rowcount

    with get_connection() as conn:
        n1, o1, r1 = _eightk_other(conn)
    print(f"updated signal_scores={up_ss}, llm_classifications={up_lc}")
    print(f"8-K 'other' AFTER:  {o1}/{n1} = {r1:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
