"""Composite scorer.

Processes recent ``raw_signals`` that don't yet have a ``signal_scores``
row, joins to ``signal_tickers``, computes a composite score per
(signal, ticker) pair, and inserts into ``signal_scores``.

Composite formula (impact-weighted, additive)
---------------------------------------------
For each (signal, ticker) pair:

    source_credibility   = source_weight / 3.33     # 0..3
    factual_bonus        = 1.0 if factual==1 else (0 if factual==0 else 0.5)
    event_impact         = EVENT_IMPACT[event_type] # 0..3 (M&A/FDA = 3; routine = 0.2)
    breadth              = min(0.5 * corroboration_count, 2.0)
    sentiment_directional = sentiment_magnitude     # 0..1 when |sent|>=0.3
    recency              = max(0, 0.5 * (1 - age_hours/24)) # 0..0.5
    megacap_bonus        = 0.5 if ticker in MEGACAPS else 0

    pump_penalty         = 2.0 if anti_pump_flag else 0
    routine_penalty      = 2.0 if routine_filing else 0

    composite = clamp(
        source_credibility + factual_bonus + event_impact
        + breadth + sentiment_directional + recency + megacap_bonus
        - pump_penalty - routine_penalty,
        0.0, 10.0,
    )

This re-anchors ranking on IMPACT instead of just source authority. A
routine Teucrium 424B3 prospectus loses 2 points; an M&A announcement on
a megacap gains the event_impact + megacap bonus. Top of the feed is now
"what could actually move money", not "what's the most credible filing".

Signal-class bucket (used by the outcome-tracking aggregator) is a
joined string like ``tier1_factual_bullish_high_corroboration`` so the
dashboard can group historical hit rates per class.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Optional

from datetime import datetime, timezone

from ..storage import get_connection
from .heuristics import classify_heuristic
from .impact import impact_for
from .megacaps import MEGACAPS
from .routine_patterns import is_routine_filing, routine_event_type
from .source_weights import weight_for

log = logging.getLogger(__name__)


@dataclass
class ScoringStats:
    candidates: int = 0
    scored: int = 0
    errors: int = 0


def score_pending(
    *,
    corroboration_window_hours: int = 4,
    batch_limit: int = 5000,
) -> ScoringStats:
    """Score every (raw_signal, ticker) pair not yet present in signal_scores."""
    stats = ScoringStats()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT rs.id            AS signal_id,
                   rs.source        AS source,
                   rs.source_tier   AS source_tier,
                   rs.title         AS title,
                   rs.body          AS body,
                   rs.author        AS author,
                   rs.author_metadata AS author_metadata_json,
                   rs.raw_payload   AS raw_payload_json,
                   rs.ingested_at   AS ingested_at,
                   rs.published_at  AS published_at,
                   st.ticker        AS ticker,
                   st.confidence    AS ticker_confidence,
                   st.asset_class   AS asset_class
            FROM raw_signals rs
            JOIN signal_tickers st ON st.signal_id = rs.id
            LEFT JOIN signal_scores ss ON ss.signal_id = rs.id AND ss.ticker = st.ticker
            WHERE ss.id IS NULL
            -- PRIORITY: score Tier 1/2 catalysts (SEC, wires, halts, movers) BEFORE
            -- Tier 3 social, so a fresh catalyst is never stuck behind a social-chatter
            -- backlog. FIFO within a tier (rs.id ASC) so nothing starves.
            ORDER BY rs.source_tier ASC, rs.id ASC
            LIMIT ?
            """,
            (batch_limit,),
        ).fetchall()
        stats.candidates = len(rows)
        if not rows:
            return stats

        for row in rows:
            try:
                composite, fields = _score_one(conn, row, corroboration_window_hours)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Scoring failed for signal_id=%s ticker=%s: %s",
                    row["signal_id"], row["ticker"], exc,
                )
                stats.errors += 1
                continue

            try:
                conn.execute(
                    """
                    INSERT INTO signal_scores (
                        signal_id, ticker, event_type, sentiment, sentiment_magnitude,
                        factual, source_weight, corroboration_count, author_quality,
                        anti_pump_flag, composite_score, signal_class, scored_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["signal_id"],
                        row["ticker"],
                        fields["event_type"],
                        fields["sentiment"],
                        fields["sentiment_magnitude"],
                        fields["factual"],
                        fields["source_weight"],
                        fields["corroboration_count"],
                        fields["author_quality"],
                        fields["anti_pump_flag"],
                        composite,
                        fields["signal_class"],
                        _utc_now_iso(),
                    ),
                )
                stats.scored += 1
            except sqlite3.IntegrityError as exc:
                log.warning(
                    "Score insert IntegrityError for signal_id=%s ticker=%s: %s",
                    row["signal_id"], row["ticker"], exc,
                )
                stats.errors += 1

    log.info(
        "score_pending: candidates=%d scored=%d errors=%d",
        stats.candidates, stats.scored, stats.errors,
    )
    return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _score_one(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    corroboration_window_hours: int,
    classification=None,
) -> tuple[float, dict]:
    source: str = row["source"]
    ticker: str = row["ticker"]

    # Decode JSON columns
    raw_payload = _safe_json(row["raw_payload_json"]) or {}
    author_metadata = _safe_json(row["author_metadata_json"]) or {}

    # Routine filing detection — short-circuit the event_type for known
    # boring administrative patterns (e.g. Teucrium 424B3)
    form = raw_payload.get("form")
    issuer = raw_payload.get("company_name") or row["author"]
    is_routine = is_routine_filing(form=form, issuer_name=issuer)

    # Classification: heuristic by default, OR a caller-supplied one. The LLM
    # pass uses the override (via rescore_with_classification) so its corrected
    # event_type/sentiment actually reaches the trade gate, instead of sitting
    # in a side table the gate never reads.
    if classification is None:
        classification = classify_heuristic(
            title=row["title"],
            body=row["body"],
            sec_form_event=raw_payload.get("form_event"),
            source=source,
        )
    if is_routine:
        # Override event type — we don't want "M&A" or "material event"
        # tags on routine prospectus filings.
        event_type = routine_event_type(form)
    else:
        event_type = classification.event_type

    # 1. Source credibility (0..3)
    source_weight = weight_for(source)
    ticker_confidence = float(row["ticker_confidence"] or 1.0)
    source_credibility = (source_weight / 3.33) * ticker_confidence

    # 2. Factual bonus (0..1)
    if classification.factual == 1:
        factual_bonus = 1.0
    elif classification.factual == 0:
        factual_bonus = 0.0
    else:
        factual_bonus = 0.5

    # 3. Event impact (0..3) — the big new factor
    event_impact = impact_for(event_type)

    # 4. Coverage breadth (0..2)
    corroboration_count = _corroboration_count(
        conn, ticker=ticker, exclude_signal_id=row["signal_id"],
        window_hours=corroboration_window_hours,
    )
    breadth = min(0.5 * corroboration_count, 2.0)

    # 5. Sentiment directional bonus (0..1)
    abs_sent = abs(classification.sentiment or 0.0)
    if abs_sent >= 0.3 and classification.sentiment_magnitude >= 0.3:
        sentiment_directional = classification.sentiment_magnitude
    else:
        sentiment_directional = 0.0

    # 6. Recency bonus (0..0.5) — fresher signals matter more
    recency = 0.0
    published_at = row["published_at"] or row["ingested_at"]
    if published_at:
        try:
            pub_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            age_hours = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
            recency = max(0.0, 0.5 * (1 - age_hours / 24))
        except (TypeError, ValueError):
            pass

    # 7. Megacap relevance (0..0.5)
    megacap_bonus = 0.5 if ticker.upper() in MEGACAPS else 0.0

    # Penalties
    anti_pump_flag = 1 if raw_payload.get("anti_pump_flag") else 0
    pump_penalty = 2.0 if anti_pump_flag else 0.0
    routine_penalty = 2.0 if is_routine else 0.0

    # Generic, direction-unknown insider filings (Form 4 with no clear buy
    # signal) are credibility-high but information-poor — mostly routine sells,
    # scheduled sales and option exercises. On SEC source-weight + corroboration
    # alone they were scoring ~9 and flooding "strong" (e.g. dozens of C / BAC
    # filings a day all at 9.3). The real edge is insider BUY clusters with clear
    # bullish sentiment, not the Form-4 firehose. Penalise the low-conviction
    # ones so "strong" stays meaningful. They already fail the |sentiment|>=0.5
    # trade-bypass gate, so this de-noises scoring/alerts without changing what
    # actually trades; genuine insider buys (event_type insider_buy, or an
    # insider_transaction with |sentiment|>=0.5) are spared.
    # Generic direction-unknown Form 4 (insider_transaction) is the low-alpha
    # firehose — routine insider trades carry ~zero documented alpha. The real
    # edge is identified opportunistic BUYS / clusters, which classify as
    # insider_buy (untouched here). Penalise the generic bucket regardless of
    # its (unreliable, direction-unknown) sentiment so the C/BAC/GS firehose
    # stays out of 'strong' and the trade gate.
    # [P2 follow-up: routine-vs-opportunistic split + buys>>sells + clusters.]
    low_conviction_insider = (event_type == "insider_transaction")
    insider_noise_penalty = 3.5 if low_conviction_insider else 0.0

    composite = (
        source_credibility
        + factual_bonus
        + event_impact
        + breadth
        + sentiment_directional
        + recency
        + megacap_bonus
        - pump_penalty
        - routine_penalty
        - insider_noise_penalty
    )
    composite = max(0.0, min(10.0, composite))

    # Compute author quality just for storage (kept for backward compat)
    author_quality_raw = _derive_author_quality(source, author_metadata, raw_payload)
    if author_quality_raw is None:
        author_quality = 0.9 if row["source_tier"] in (1, 2) else 0.6
    else:
        author_quality = author_quality_raw

    # Override classification.event_type with our possibly-routine-corrected version
    classification_event_type = event_type

    # Bucket key for outcome aggregation — uses the corrected event_type
    sentiment_dir = (
        "bullish" if classification.sentiment > 0.2
        else "bearish" if classification.sentiment < -0.2
        else "neutral"
    )
    corr_band = (
        "high_corr" if corroboration_count >= 3
        else "med_corr" if corroboration_count >= 1
        else "low_corr"
    )
    signal_class = "|".join([
        f"t{row['source_tier']}",
        classification_event_type,
        sentiment_dir,
        "factual" if classification.factual == 1 else "speculative",
        corr_band,
        "megacap" if megacap_bonus > 0 else "smallcap",
    ])

    return composite, {
        "event_type": classification_event_type,
        "sentiment": classification.sentiment,
        "sentiment_magnitude": classification.sentiment_magnitude,
        "factual": classification.factual,
        "source_weight": source_weight,
        "corroboration_count": corroboration_count,
        "author_quality": author_quality,
        "anti_pump_flag": anti_pump_flag or (1 if is_routine else 0),
        "signal_class": signal_class,
    }


def rescore_with_classification(
    conn: sqlite3.Connection,
    *,
    signal_id: int,
    ticker: str,
    event_type,
    sentiment,
    sentiment_magnitude,
    factual,
    corroboration_window_hours: int = 4,
) -> bool:
    """Re-score one signal with a (usually LLM-produced) classification and
    write it back to signal_scores.

    THE BRIDGE: without this, the LLM classifies into the separate
    ``llm_classifications`` table that the trade gate never reads, so a real
    earnings/M&A catalyst the heuristic mislabelled 'other' stays untradeable.
    Here we recompute the composite with the corrected label (re-applying source
    credibility, breadth, recency, megacap + penalties consistently via
    _score_one) and overwrite the gated columns. Returns True if it updated.
    """
    from .heuristics import HeuristicClassification

    row = conn.execute(
        """
        SELECT rs.id AS signal_id, rs.source AS source, rs.source_tier AS source_tier,
               rs.title AS title, rs.body AS body, rs.author AS author,
               rs.author_metadata AS author_metadata_json,
               rs.raw_payload AS raw_payload_json,
               rs.ingested_at AS ingested_at, rs.published_at AS published_at,
               st.ticker AS ticker, st.confidence AS ticker_confidence
        FROM raw_signals rs
        JOIN signal_tickers st ON st.signal_id = rs.id
        WHERE rs.id = ? AND st.ticker = ?
        """,
        (signal_id, ticker),
    ).fetchone()
    if row is None:
        return False

    sent = float(sentiment) if sentiment is not None else 0.0
    mag = float(sentiment_magnitude) if sentiment_magnitude is not None else abs(sent)
    fact = int(factual) if factual is not None else 1
    cls = HeuristicClassification(
        event_type=event_type or "other",
        sentiment=round(sent, 3),
        sentiment_magnitude=round(min(max(mag, 0.0), 1.0), 3),
        factual=fact,
    )

    composite, fields = _score_one(
        conn, row, corroboration_window_hours, classification=cls
    )

    conn.execute(
        """
        UPDATE signal_scores
           SET event_type = ?, sentiment = ?, sentiment_magnitude = ?,
               factual = ?, composite_score = ?, signal_class = ?
         WHERE signal_id = ? AND ticker = ?
        """,
        (
            fields["event_type"], fields["sentiment"], fields["sentiment_magnitude"],
            fields["factual"], composite, fields["signal_class"],
            signal_id, ticker,
        ),
    )
    return True


def _corroboration_count(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    exclude_signal_id: int,
    window_hours: int,
) -> int:
    """Count distinct STORIES (by content_hash) from Tier 1/2 signals on the same
    ticker in the trailing window — NOT distinct feeds. Counting feeds let one
    wire story syndicated across ~8 outlets (or ~30 Google-News topical feeds)
    inflate "corroboration" 8-30x with zero added information (ingestion deep-dive
    2026-06-15). Distinct content_hash de-syndicates. Excludes the signal scored."""
    sql = f"""
        SELECT COUNT(DISTINCT COALESCE(rs.content_hash, 'id:' || rs.id)) AS n
        FROM raw_signals rs
        JOIN signal_tickers st ON st.signal_id = rs.id
        WHERE st.ticker = ?
          AND rs.id != ?
          AND rs.source_tier IN (1, 2)
          AND COALESCE(rs.published_at, rs.ingested_at) >=
              strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now', '-{int(window_hours)} hours'))
    """
    row = conn.execute(sql, (ticker, exclude_signal_id)).fetchone()
    return int((row["n"] if row else 0) or 0)


def _derive_author_quality(
    source: str,
    author_metadata: dict,
    raw_payload: dict,
) -> Optional[float]:
    """Best-effort author-quality estimate.

    For institutional sources (SEC EDGAR, mainstream news) we don't gate
    on author quality — return None (caller defaults to 0.7).

    For social, look for karma/age proxies in the metadata. Until PRAW
    is wired in, Reddit public-JSON entries get score+num_comments as
    weak proxies. StockTwits has follower count.
    """
    if source.startswith("reddit_"):
        score = author_metadata.get("score")
        comments = author_metadata.get("num_comments")
        upvote_ratio = author_metadata.get("upvote_ratio")
        if score is None and comments is None:
            return None
        # Crude blend: positive engagement raises quality, negative drops it
        engagement = (score or 0) + 2 * (comments or 0)
        ratio = upvote_ratio if isinstance(upvote_ratio, (int, float)) else 0.5
        # Normalize: low engagement → ~0.5, high engagement (100+) → up to ~0.9
        eng_factor = min(engagement / 200.0, 0.4)  # 0..0.4
        return round(0.5 + eng_factor + (ratio - 0.5) * 0.2, 3)

    if source == "stocktwits_trending":
        followers = author_metadata.get("followers")
        ideas = author_metadata.get("ideas")
        if not followers and not ideas:
            return 0.45
        followers = followers or 0
        # 1000+ followers → 0.7, 10k+ → 0.85
        if followers >= 10_000:
            return 0.85
        if followers >= 1000:
            return 0.7
        if followers >= 100:
            return 0.55
        return 0.4

    # Institutional / wire sources — don't gate
    return None


def _safe_json(value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
