"""Smart LLM backfill — classify the most-impactful historical signals.

Strategy:
  1. Select backfilled SEC signals where the form is high-impact AND the
     issuer is a real operating company (not a routine-filer trust/bank).
  2. Order by recency (most recent 6 months prioritised — most relevant
     to the next weekly retrain).
  3. Apply the smart filter from llm/filter.py.
  4. Run LLM classification with daily-spend cap enforced.
  5. Stop when budget for the run is exhausted OR all selected signals done.

Usage:
    python scripts/run_llm_backfill.py
    python scripts/run_llm_backfill.py --budget-usd 12 --months 6
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import CONFIG  # noqa: E402
from market_radar.llm import LLMClassifier  # noqa: E402
from market_radar.llm.filter import should_classify  # noqa: E402
from market_radar.llm.spend import today_spend_usd, DAILY_SPEND_CAP_USD  # noqa: E402
from market_radar.storage import get_connection, init_db  # noqa: E402


HIGH_IMPACT_FORMS = (
    "8-K", "8-K/A",
    "4",                          # insider transactions
    "SC 13D", "SC 13D/A",         # activist stake
    "425",                        # M&A communication
    "S-1",                        # IPO
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-usd", type=float, default=12.0,
                   help="Stop classification once this much $ has been spent in this run")
    p.add_argument("--months", type=int, default=6,
                   help="How many trailing months of history to consider")
    p.add_argument("--max-signals", type=int, default=20000)
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be classified without spending")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("llm_backfill")

    if not CONFIG.has_anthropic:
        log.error("ANTHROPIC_API_KEY not set in .env — cannot run")
        return 1

    # Ensure migrations have applied — creates llm_classifications + llm_spend_daily
    # if they don't exist yet. Idempotent.
    init_db()

    # Select candidate rows
    forms_in = ",".join(f"'sec_edgar_backfill_{f.lower().replace(' ', '_').replace('/', '_')}'"
                        for f in HIGH_IMPACT_FORMS)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT ss.signal_id, ss.ticker, ss.composite_score, ss.event_type,
                   ss.sentiment, ss.signal_class,
                   rs.title, rs.body, rs.source, rs.source_tier,
                   rs.raw_payload, rs.content_hash, rs.published_at
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            LEFT JOIN llm_classifications lc
                   ON lc.signal_id = ss.signal_id AND lc.ticker = ss.ticker
            WHERE lc.id IS NULL
              AND rs.source IN ({forms_in})
              AND rs.published_at >= datetime('now', '-{int(args.months)} months')
            ORDER BY rs.published_at DESC
            LIMIT ?
            """,
            (args.max_signals,),
        ).fetchall()

    log.info("Candidate pool: %d signals (last %d months, forms %s)",
             len(rows), args.months, HIGH_IMPACT_FORMS)
    if not rows:
        log.info("Nothing to classify.")
        return 0

    if args.dry_run:
        # Filter + report without calling LLM
        by_reason: dict[str, int] = {}
        for row in rows:
            r = dict(row)
            ok, reason = should_classify(r)
            key = ("WILL: " + reason) if ok else ("SKIP: " + reason)
            by_reason[key] = by_reason.get(key, 0) + 1
        for k in sorted(by_reason, key=lambda x: -by_reason[x]):
            log.info("  %s × %d", k, by_reason[k])
        return 0

    # Live classification
    classifier = LLMClassifier()
    start_today_spent = today_spend_usd()
    run_budget_remaining = args.budget_usd
    classified = 0
    skipped_filter = 0
    skipped_dup = 0
    errors = 0
    seen_hashes: set[str] = set()

    with get_connection() as conn:
        for row in rows:
            r = dict(row)
            ok, reason = should_classify(r)
            if not ok:
                skipped_filter += 1
                continue

            chash = r.get("content_hash")
            if chash and chash in seen_hashes:
                skipped_dup += 1
                continue
            if chash:
                # Check if we've classified this hash already (any signal)
                existing = conn.execute(
                    """SELECT lc.event_type, lc.event_subtype, lc.sentiment,
                              lc.sentiment_magnitude, lc.factual, lc.extracted_fields,
                              lc.confidence, lc.model
                       FROM llm_classifications lc
                       JOIN raw_signals rs ON rs.id = lc.signal_id
                       WHERE rs.content_hash = ?
                       LIMIT 1""",
                    (chash,),
                ).fetchone()
                if existing:
                    classifier.store(conn, signal_id=r["signal_id"],
                                     ticker=r["ticker"], result={
                        "event_type": existing["event_type"],
                        "event_subtype": existing["event_subtype"],
                        "sentiment": existing["sentiment"],
                        "sentiment_magnitude": existing["sentiment_magnitude"],
                        "factual": existing["factual"],
                        "extracted_fields": json.loads(existing["extracted_fields"] or "{}"),
                        "confidence": existing["confidence"],
                        "model": existing["model"],
                        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                    })
                    seen_hashes.add(chash)
                    skipped_dup += 1
                    continue
                seen_hashes.add(chash)

            try:
                result = classifier.classify_one(r)
            except RuntimeError as exc:
                log.error("LLM auth/critical error — stopping: %s", exc)
                break
            if result is None:
                # cap hit or persistent failure
                spent_total = today_spend_usd()
                if spent_total >= DAILY_SPEND_CAP_USD:
                    log.info("Hit daily cap $%.2f — stopping. Re-run tomorrow to continue.",
                             DAILY_SPEND_CAP_USD)
                    break
                errors += 1
                continue

            classifier.store(conn, signal_id=r["signal_id"],
                             ticker=r["ticker"], result=result)
            classified += 1
            run_budget_remaining -= result.get("cost_usd") or 0
            if run_budget_remaining <= 0:
                log.info("Hit run budget $%.2f — stopping (rest will pick up tomorrow)",
                         args.budget_usd)
                break

            # Progress log every 50 classifications
            if classified % 50 == 0:
                spent = today_spend_usd() - start_today_spent
                log.info("  classified=%d  cost-this-run=$%.4f  remaining=$%.2f",
                         classified, spent, run_budget_remaining)

    spent_total = today_spend_usd() - start_today_spent
    log.info("Backfill done: classified=%d skipped-filter=%d skipped-dup=%d errors=%d "
             "spend-this-run=$%.4f",
             classified, skipped_filter, skipped_dup, errors, spent_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
