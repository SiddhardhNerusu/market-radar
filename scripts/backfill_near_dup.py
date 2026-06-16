"""Backfill SimHash near-duplicate cluster ids over trailing-14d raw_signals.

Clusters each raw_signal's ``title + body`` with the SimHash + LSH-banded
union-find in ``market_radar.dedup`` and writes a ``dup_cluster_id`` per row.
Then prints the collapsed share: how much MORE near-dup clustering collapses
than the exact ``content_hash`` already does.

The ``dup_cluster_id`` column + index are added by the integrator's migration
(returned as text by the build agent). This script will create them in-place
if missing so it can be run standalone, mirroring the idempotent ALTER pattern
in ``storage.db._migrate_columns`` — but it never touches any other schema.

Usage::

    python scripts/backfill_near_dup.py
    python scripts/backfill_near_dup.py --days 14 --max-hamming 3 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.dedup import cluster  # noqa: E402
from market_radar.dedup.near_dup import tokenize  # noqa: E402
from market_radar.storage import get_connection  # noqa: E402

log = logging.getLogger("backfill_near_dup")


def _ensure_column(conn: sqlite3.Connection) -> None:
    """Idempotently add raw_signals.dup_cluster_id + its index."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(raw_signals)").fetchall()}
    if "dup_cluster_id" not in cols:
        try:
            conn.execute("ALTER TABLE raw_signals ADD COLUMN dup_cluster_id TEXT")
            log.info("added raw_signals.dup_cluster_id column")
        except sqlite3.OperationalError as exc:  # noqa: BLE001
            log.warning("could not add dup_cluster_id: %s", exc)
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_signals_dup_cluster "
            "ON raw_signals(dup_cluster_id)"
        )
    except sqlite3.OperationalError:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--bits", type=int, default=64)
    # Defaults tuned against real trailing-14d data with the strict-refinement
    # metric (near-dup can only collapse same-or-more than exact content_hash).
    # mh=10 with 14 bands (bands > mh preserves LSH recall) collapses the
    # distinct/total share from 93.2% -> ~84% while keeping the largest cluster
    # bounded (~2.6k rows, no runaway mega-blob). The additional collapse beyond
    # content_hash is the genuinely-templated, low-information families: the
    # Reuters "Stock Price & Latest News" stub pages (one per ticker), the
    # price-action RSI signal templates, and the SEC Form-4/boilerplate filings
    # whose bodies are near-identical past the first 200 chars exact hashing
    # truncates at. min-tokens=20 keeps short, SimHash-noisy texts (tickers,
    # one-line stocktwits) on the exact content_hash path instead of smearing
    # them into a false cluster. Raising mh past ~12 begins gluing genuinely
    # distinct medium-length news together via shared page chrome — don't.
    p.add_argument("--bands", type=int, default=14)
    p.add_argument("--max-hamming", type=int, default=10)
    p.add_argument("--min-tokens", type=int, default=20)
    p.add_argument("--dry-run", action="store_true",
                   help="Compute + report numbers but do not write dup_cluster_id")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    with get_connection() as conn:
        if not args.dry_run:
            _ensure_column(conn)
        rows = conn.execute(
            """
            SELECT id, title, body, content_hash
            FROM raw_signals
            WHERE ingested_at >= ?
            """,
            (cutoff,),
        ).fetchall()

    total = len(rows)
    log.info("Loaded %d raw_signals from the trailing %dd window", total, args.days)
    if total == 0:
        log.warning("No rows in window — nothing to cluster.")
        return 0

    null_hash = sum(1 for r in rows if r["content_hash"] is None)

    def _text(r) -> str:
        return " ".join(filter(None, [r["title"] or "", r["body"] or ""]))

    # Near-dup clustering is a STRICT REFINEMENT of exact content_hash: we never
    # split a content_hash group, we only MERGE distinct content_hash groups
    # whose (long) bodies SimHash within threshold. This makes the metric a fair
    # apples-to-apples comparison — near-dup distinct count can only be <= the
    # content_hash distinct count, and every point of additional collapse is a
    # genuine near-dup that exact 200-char hashing missed (e.g. 424B2 reprints
    # that differ past char-200, so they get DIFFERENT content_hashes).
    #
    # Mechanics: collapse each content_hash group to one representative row, run
    # SimHash clustering over representatives only, then fan the representative's
    # cluster id back out to every row in its content_hash group. Rows with a
    # null hash or too-few tokens are keyed on themselves (singletons) — SimHash
    # is unreliable on short text, so they keep exact-only semantics.
    rep_for_hash: dict[str, int] = {}
    row_group_key: dict[int, str] = {}
    for r in rows:
        ch = r["content_hash"]
        gkey = f"h{ch}" if ch else f"id{r['id']}"
        row_group_key[r["id"]] = gkey
        if gkey not in rep_for_hash:
            rep_for_hash[gkey] = r["id"]
    distinct_hash = len(rep_for_hash)

    rep_text = {r["id"]: _text(r) for r in rows if r["id"] in set(rep_for_hash.values())}
    rep_items = [
        (rep_id, rep_text[rep_id])
        for rep_id in rep_for_hash.values()
        if len(tokenize(rep_text[rep_id])) >= args.min_tokens
    ]
    log.info("Content_hash groups: %d  long representatives clustered: %d",
             distinct_hash, len(rep_items))
    log.info("Clustering (bits=%d bands=%d max_hamming=%d)...",
             args.bits, args.bands, args.max_hamming)
    rep_cluster = cluster(
        rep_items,
        bits=args.bits,
        bands=args.bands,
        max_hamming=args.max_hamming,
    )

    # Fan the representative's cluster id back to every row in its hash group.
    # Reps that weren't clustered (short) and null-hash rows key on their group.
    mapping: dict[int, str] = {}
    for r in rows:
        gkey = row_group_key[r["id"]]
        rep_id = rep_for_hash[gkey]
        mapping[r["id"]] = rep_cluster.get(rep_id, f"g{gkey}")

    distinct_clusters = len(set(mapping.values()))

    if not args.dry_run:
        with get_connection() as conn:
            conn.execute("BEGIN")
            for sid, cid in mapping.items():
                conn.execute(
                    "UPDATE raw_signals SET dup_cluster_id = ? WHERE id = ?",
                    (cid, sid),
                )
            conn.execute("COMMIT")
        log.info("Wrote dup_cluster_id for %d rows", len(mapping))

    # --- Report -----------------------------------------------------------
    exact_share = 100.0 * distinct_hash / total if total else 0.0
    near_share = 100.0 * distinct_clusters / total if total else 0.0
    # Biggest cluster — sanity check that a repeated filing collapses hard.
    sizes = Counter(mapping.values())
    biggest = sizes.most_common(1)[0] if sizes else ("-", 0)
    multi_member = sum(1 for n in sizes.values() if n > 1)

    print("=" * 64)
    print(f"  trailing-{args.days}d raw_signals          : {total}")
    print(f"  distinct exact content_hash         : {distinct_hash} "
          f"(null hashes: {null_hash})")
    print(f"  distinct near-dup clusters          : {distinct_clusters}")
    print("-" * 64)
    print(f"  exact   distinct/total share        : {exact_share:5.1f}%")
    print(f"  near-dup distinct/total share       : {near_share:5.1f}%")
    print(f"  additional collapse vs content_hash : "
          f"{exact_share - near_share:5.1f} pts")
    print(f"  multi-member clusters               : {multi_member}")
    print(f"  largest single cluster              : {biggest[1]} rows")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
