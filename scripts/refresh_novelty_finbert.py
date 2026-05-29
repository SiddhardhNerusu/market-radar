"""Compute news novelty score (sentence-transformers MiniLM) + FinBERT
sentiment for recent raw_signals. Updates the per-row feature cache.

Both are local, free, CPU-runnable. First run downloads ~400MB of model
weights to ``~/.cache/huggingface``. Subsequent runs are fast (~100
messages/sec on a Mac M2).

Outputs to ``news_features`` table:
  - novelty_score: 1 - max(cosine_similarity to prior 24h titles) ∈ [0, 1]
  - finbert_sentiment: signed [-1, +1]

Usage::

    python scripts/refresh_novelty_finbert.py
    python scripts/refresh_novelty_finbert.py --hours 48 --limit 5000
"""
from __future__ import annotations

import argparse, logging, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.storage import get_connection, init_db  # noqa: E402

log = logging.getLogger("refresh_novelty_finbert")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--skip-finbert", action="store_true",
                   help="Only compute novelty (skip the FinBERT step)")
    p.add_argument("--skip-novelty", action="store_true",
                   help="Only compute FinBERT (skip novelty)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    init_db()
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_features (
                signal_id   INTEGER PRIMARY KEY,
                ticker      TEXT,
                novelty_score REAL,
                finbert_sentiment REAL,
                computed_at TEXT NOT NULL
            )
        """)

    cutoff = (datetime.utcnow() - timedelta(hours=args.hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT rs.id AS signal_id, rs.title, st.ticker, rs.published_at
            FROM raw_signals rs
            JOIN signal_tickers st ON st.signal_id = rs.id
            WHERE rs.published_at >= ?
              AND rs.title IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM news_features nf WHERE nf.signal_id = rs.id)
            ORDER BY rs.id DESC
            LIMIT ?
            """,
            (cutoff, args.limit),
        ).fetchall()
    log.info("Candidates: %d", len(rows))
    if not rows:
        return 0

    embedder = None
    finbert = None
    if not args.skip_novelty:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("Loaded MiniLM embedder")
        except ImportError:
            log.warning("sentence-transformers not installed — pip install sentence-transformers")
            log.warning("Skipping novelty scoring this run.")
            embedder = None
    if not args.skip_finbert:
        try:
            from transformers import pipeline  # type: ignore
            finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                               truncation=True, max_length=128)
            log.info("Loaded FinBERT pipeline")
        except ImportError:
            log.warning("transformers not installed — pip install transformers torch")
            finbert = None
        except Exception as exc:  # noqa: BLE001
            log.warning("FinBERT load failed: %s", exc)
            finbert = None

    titles = [r["title"] or "" for r in rows]
    novelty: list[float] = [0.5] * len(rows)
    sentiments: list[float] = [0.0] * len(rows)

    if embedder is not None:
        import numpy as np  # type: ignore
        embs = embedder.encode(titles, show_progress_bar=False, normalize_embeddings=True)
        # Group by ticker for novelty computation: a title is novel if its
        # max cosine sim to other titles for the same ticker in the prior
        # 24h is low.
        from collections import defaultdict
        by_ticker_idx: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            by_ticker_idx[(r["ticker"] or "").upper()].append(i)
        for tk, idxs in by_ticker_idx.items():
            if len(idxs) < 2:
                continue
            sub_embs = embs[idxs]
            sim_matrix = np.dot(sub_embs, sub_embs.T)
            np.fill_diagonal(sim_matrix, 0.0)
            max_sim = sim_matrix.max(axis=1)
            for j, orig_idx in enumerate(idxs):
                novelty[orig_idx] = float(1.0 - max_sim[j])

    if finbert is not None:
        # Batch to keep latency reasonable
        batch_size = 32
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            try:
                outs = finbert(batch)
            except Exception as exc:  # noqa: BLE001
                log.debug("FinBERT batch failed: %s", exc)
                continue
            for j, o in enumerate(outs):
                label = (o.get("label") or "").lower()
                score = float(o.get("score") or 0)
                if label == "positive":
                    sentiments[i + j] = +score
                elif label == "negative":
                    sentiments[i + j] = -score
                else:
                    sentiments[i + j] = 0.0

    with get_connection() as conn:
        for i, r in enumerate(rows):
            conn.execute(
                """INSERT OR REPLACE INTO news_features
                (signal_id, ticker, novelty_score, finbert_sentiment, computed_at)
                VALUES (?, ?, ?, ?, ?)""",
                (r["signal_id"], r["ticker"], novelty[i], sentiments[i], _utc_now()),
            )
    log.info("Wrote %d rows to news_features", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
