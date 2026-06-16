"""Backfill the SEC form-tag corruption (ingestion blueprint #1 — GATES EVERYTHING).

EDGAR's &type= feed filter prefix-matches, so the live ingestor tagged ~7,000
424B* debt prospectuses (and other prefix collisions) with the QUERY form (e.g.
'4' => insider_transaction), poisoning classification, scoring AND measurement.
This recomputes each sec_edgar row's real form from its title leading token,
fixes raw_payload.form / form_event, and deterministically corrects the
downstream event_type for the prospectus rows mislabeled as insider_transaction.
Idempotent; no LLM.

Usage:  PYTHONPATH=src .venv/bin/python scripts/backfill_form_tags.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.ingestors.sec_edgar import FORM_TYPES
from market_radar.storage import get_connection


def real_form_from_title(title, fallback):
    if title and " - " in title:
        head = title.split(" - ", 1)[0].strip()
        if head:
            return head
    return fallback


def main() -> int:
    fixed = 0
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, raw_payload FROM raw_signals "
            "WHERE source='sec_edgar' AND raw_payload IS NOT NULL"
        ).fetchall()
        for r in rows:
            try:
                payload = json.loads(r["raw_payload"])
            except (TypeError, ValueError):
                continue
            stored = payload.get("form", "")
            real = real_form_from_title(r["title"], stored)
            if real == stored:
                continue
            payload["form"] = real
            payload["form_event"] = FORM_TYPES.get(real, "other")
            conn.execute("UPDATE raw_signals SET raw_payload=? WHERE id=?",
                         (json.dumps(payload), r["id"]))
            fixed += 1

    # Deterministically correct the downstream event_type for 424B* prospectuses
    # that the form-tag bug mislabeled as insider_transaction (a 424B IS a
    # prospectus, never an insider buy). Only touches the clearly-bad rows.
    with get_connection() as conn:
        sub = ("(SELECT id FROM raw_signals WHERE source='sec_edgar' "
               "AND title LIKE '424B%')")
        ss = conn.execute(
            f"UPDATE signal_scores SET event_type='routine_prospectus' "
            f"WHERE event_type='insider_transaction' AND signal_id IN {sub}").rowcount
        lc = conn.execute(
            f"UPDATE llm_classifications SET event_type='routine_prospectus' "
            f"WHERE event_type='insider_transaction' AND signal_id IN {sub}").rowcount

    print(f"raw_payload form fixed: {fixed}; signal_scores re-tagged: {ss}; "
          f"llm_classifications re-tagged: {lc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
