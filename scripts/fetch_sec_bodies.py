"""Backfill SEC filing bodies into ``raw_signals.body``.

Before this script: the SEC ingestor (live + backfill) stores only filing
metadata — title, accession, URL — never the actual filing document.
That made every LLM classification a no-op (the model saw "Body: (no body)").

This script walks the rows where the body is missing (or stuck on the
useless ``<b>Filed:</b> ...`` RSS-summary stub), resolves each filing's
primary document via :mod:`market_radar.ingestors.sec_body_fetcher`,
strips markup, and writes the cleaned text back into
``raw_signals.body``.

The script is idempotent (rows already populated are skipped), respects
SEC's 10 req/sec cap (we run ~6 rps in practice), and is interruptible
— results are committed every ``--commit-every`` rows.

Typical usage::

    # Dry run — show what would be fetched, no HTTP traffic
    python scripts/fetch_sec_bodies.py --months 6 --dry-run

    # Backfill the high-impact forms from the last 6 months (default)
    python scripts/fetch_sec_bodies.py --months 6

    # Just refill the rows we previously LLM-classified
    python scripts/fetch_sec_bodies.py --only-already-classified

The body fetcher caches every successful download in
``data/sec_body_cache/`` so re-runs don't refetch.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.ingestors.sec_body_fetcher import SecBodyFetcher  # noqa: E402
from market_radar.storage import get_connection  # noqa: E402


log = logging.getLogger("fetch_sec_bodies")


# Source-name prefixes that should be included. We use the prefix-match form
# because backfill sources are named per-form (``sec_edgar_backfill_8-k``,
# ``sec_edgar_backfill_4``, ...) and the live RSS source is just
# ``sec_edgar``.
DEFAULT_SOURCE_PREFIXES = ("sec_edgar",)

# Forms with the highest signal density — the ones the LLM filter actually
# classifies. Used to scope the default selection. We pattern-match against
# the title prefix because that's how every SEC row encodes its form.
HIGH_IMPACT_TITLE_PATTERNS = (
    "8-K - %",      "8-K/A - %",
    "4 - %",
    "SC 13D - %",   "SC 13D/A - %",
    "425 - %",
    "S-1 - %",      "S-1/A - %",
)


# RSS-summary "stub" pattern. The live ingestor stores something like
#   "<b>Filed:</b> 2026-05-13 <b>AccNo:</b> 0001104659-26-059734 ..."
# which is metadata, not content. We treat those as "missing body too".
STUB_BODY_PATTERN = "<b>Filed:</b>%"


def select_target_signal_ids(
    conn,
    *,
    months: int,
    forms: tuple[str, ...],
    only_already_classified: bool,
    include_stub_bodies: bool,
    limit: Optional[int],
) -> list[dict]:
    """Return rows that still need a body, ordered most-recent first."""
    where_clauses: list[str] = []
    params: list = []

    where_clauses.append("(" + " OR ".join(
        ["rs.source LIKE ?"] * len(DEFAULT_SOURCE_PREFIXES)
    ) + ")")
    params.extend([p + "%" for p in DEFAULT_SOURCE_PREFIXES])

    where_clauses.append("rs.url IS NOT NULL")

    if include_stub_bodies:
        where_clauses.append(
            "(rs.body IS NULL OR rs.body LIKE ?)"
        )
        params.append(STUB_BODY_PATTERN)
    else:
        where_clauses.append("rs.body IS NULL")

    if forms:
        title_or = " OR ".join(["rs.title LIKE ?"] * len(forms))
        where_clauses.append(f"({title_or})")
        params.extend(forms)

    if months and months > 0:
        where_clauses.append(
            f"(rs.published_at IS NULL "
            f"OR rs.published_at >= datetime('now', '-{int(months)} months'))"
        )

    if only_already_classified:
        where_clauses.append(
            "EXISTS (SELECT 1 FROM llm_classifications lc WHERE lc.signal_id = rs.id)"
        )

    sql = f"""
        SELECT rs.id, rs.source, rs.url, rs.title, rs.raw_payload,
               rs.published_at
        FROM raw_signals rs
        WHERE {' AND '.join(where_clauses)}
        ORDER BY rs.published_at DESC NULLS LAST, rs.id DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def extract_form(raw_payload: Optional[str]) -> Optional[str]:
    if not raw_payload:
        return None
    import json
    try:
        return json.loads(raw_payload).get("form")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


_STOP = False


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _STOP
        _STOP = True
        log.warning("Caught signal %d — finishing current row then exiting", signum)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Backfill SEC filing bodies into raw_signals.body"
    )
    p.add_argument("--months", type=int, default=6,
                   help="Trailing months of history to consider (default: 6). "
                        "0 = no date restriction.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on rows to process this run (default: unlimited).")
    p.add_argument("--commit-every", type=int, default=25,
                   help="Commit DB writes every N rows (default: 25).")
    p.add_argument("--min-interval", type=float, default=0.17,
                   help="Min seconds between SEC HTTP calls (default: 0.17).")
    p.add_argument("--high-impact-only", action="store_true", default=True,
                   help="Restrict to 8-K, 4, SC 13D, 425, S-1 (default: True).")
    p.add_argument("--all-forms", dest="high_impact_only", action="store_false",
                   help="Don't restrict to high-impact forms.")
    p.add_argument("--include-stub-bodies", action="store_true", default=True,
                   help="Also re-fetch rows that have the RSS-summary stub "
                        "instead of NULL (default: True).")
    p.add_argument("--null-only", dest="include_stub_bodies", action="store_false",
                   help="Only refill rows whose body IS NULL (skip stubs).")
    p.add_argument("--only-already-classified", action="store_true",
                   help="Restrict to rows referenced by llm_classifications "
                        "(the 6,837 previously-classified rows).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be fetched without doing any HTTP.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    forms = HIGH_IMPACT_TITLE_PATTERNS if args.high_impact_only else ()

    _install_signal_handlers()

    with get_connection() as conn:
        rows = select_target_signal_ids(
            conn,
            months=args.months,
            forms=forms,
            only_already_classified=args.only_already_classified,
            include_stub_bodies=args.include_stub_bodies,
            limit=args.limit,
        )

    log.info(
        "Selection: %d candidate rows (months=%s high_impact=%s "
        "stubs=%s only_classified=%s)",
        len(rows), args.months, args.high_impact_only,
        args.include_stub_bodies, args.only_already_classified,
    )
    if not rows:
        log.info("Nothing to do.")
        return 0

    # Report a per-source breakdown so the user can sanity-check scope
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    for src in sorted(by_source, key=lambda s: -by_source[s]):
        log.info("  %-32s  %d rows", src, by_source[src])

    if args.dry_run:
        log.info("Dry-run — exiting without fetching.")
        return 0

    fetcher = SecBodyFetcher(min_interval_s=args.min_interval)

    successes = 0
    misses = 0
    skipped_unparseable = 0
    started_at = time.monotonic()

    with get_connection() as conn:
        pending_commit = 0
        for i, row in enumerate(rows, 1):
            if _STOP:
                log.warning("Stop requested — bailing after %d rows", i - 1)
                break

            form = extract_form(row.get("raw_payload"))
            result = fetcher.fetch(row["url"], form_type=form)
            if result.body:
                conn.execute(
                    "UPDATE raw_signals SET body = ? WHERE id = ?",
                    (result.body, row["id"]),
                )
                successes += 1
                pending_commit += 1
            else:
                if result.source_strategy == "miss" and result.primary_doc is None \
                        and not row.get("url"):
                    skipped_unparseable += 1
                else:
                    misses += 1

            if pending_commit >= args.commit_every:
                conn.commit()
                pending_commit = 0

            if i % 50 == 0 or i == len(rows):
                elapsed = time.monotonic() - started_at
                rps = i / elapsed if elapsed > 0 else 0
                eta_s = (len(rows) - i) / rps if rps > 0 else 0
                log.info(
                    "  processed=%d/%d  ok=%d  miss=%d  rate=%.1f rows/s  eta=%ds",
                    i, len(rows), successes, misses, rps, int(eta_s),
                )

        conn.commit()

    log.info(
        "Done. processed=%d  bodies_written=%d  misses=%d  unparseable=%d  "
        "elapsed=%.1fs",
        len(rows), successes, misses, skipped_unparseable,
        time.monotonic() - started_at,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
