"""LLM classifier with hard spend cap + robust error handling.

Designed to be called periodically by the daemon (classify_pending) AND by
the bulk backfill script (classify_backfill_subset). Both paths share the
same cost-control machinery.

Behavior on errors:
  - Auth error → log + raise (config problem, user needs to fix)
  - Rate limit → exponential backoff, retry up to 3x
  - Malformed JSON → log + skip the row, don't store
  - Daily-cap hit → return early, no further calls until midnight UTC
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ..config import CONFIG
from ..storage import get_connection
from .filter import should_classify
from .prompt import (
    CLASSIFY_TOOL,
    LOW_CONFIDENCE_EVENT,
    LOW_CONFIDENCE_THRESHOLD,
    SYSTEM_PROMPT,
    approx_input_tokens,
    approx_output_tokens,
    build_user_prompt,
)
from .spend import LLMSpendTracker, estimate_cost, today_spend_usd

log = logging.getLogger(__name__)


@dataclass
class ClassifyStats:
    candidates: int = 0
    filtered_out: int = 0
    skipped_already_done: int = 0
    classified: int = 0
    errors: int = 0
    cap_hit: bool = False
    total_cost_usd: float = 0.0
    filter_reasons: dict[str, int] = field(default_factory=dict)


class LLMClassifier:
    """Wraps the Anthropic client + spend tracker + DB writes."""

    def __init__(self, *, model: Optional[str] = None,
                 max_retries: int = 3, retry_backoff: float = 2.0):
        if not CONFIG.has_anthropic:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not configured in .env — cannot classify"
            )
        self.model = model or CONFIG.anthropic_model
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.tracker = LLMSpendTracker()
        # Lazy import so daemon survives when anthropic isn't installed
        import anthropic  # type: ignore
        self._client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)

    # ------------------------------------------------------------------

    def classify_one(self, row: dict[str, Any]) -> Optional[dict]:
        """Classify one signal. Returns parsed result dict, or None on
        failure / cap-hit / filter-skip."""
        # Cost pre-check
        in_toks = approx_input_tokens(title=row.get("title"), body=row.get("body"))
        out_toks = approx_output_tokens()
        est = estimate_cost(input_tokens=in_toks, output_tokens=out_toks)
        allowed, today_spent, remaining = self.tracker.can_spend(est)
        if not allowed:
            log.info(
                "LLM daily cap reached ($%.2f / $%.2f) — skipping classification",
                today_spent, self.tracker.daily_cap_usd,
            )
            return None

        user_msg = build_user_prompt(
            title=row.get("title"),
            body=row.get("body"),
            source=row.get("source"),
            primary_ticker=row.get("ticker"),
        )

        # Anthropic call with retry
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            attempt += 1
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                    tools=[CLASSIFY_TOOL],
                    tool_choice={"type": "tool", "name": "classify_signal"},
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # Distinguish rate-limit vs auth vs other
                exc_name = type(exc).__name__
                if "RateLimit" in exc_name or "rate" in str(exc).lower():
                    sleep_s = min(self.retry_backoff ** attempt, 30)
                    log.warning("LLM rate-limit attempt %d/%d — sleeping %.1fs",
                                attempt, self.max_retries + 1, sleep_s)
                    time.sleep(sleep_s)
                    continue
                if "Auth" in exc_name or "401" in str(exc) or "API key" in str(exc):
                    log.error("LLM auth error — check ANTHROPIC_API_KEY in .env: %s", exc)
                    raise
                log.warning("LLM call error attempt %d/%d: %s", attempt, self.max_retries + 1, exc)
                time.sleep(min(self.retry_backoff ** attempt, 30))
        else:
            log.warning("LLM call exhausted retries: %s", last_exc)
            return None
        if msg is None:
            return None

        # Parse the FORCED tool call — guaranteed schema-valid input, so there is
        # NO free-text json.loads / code-fence path that can fail (blueprint #3
        # L137: zero parse-failure log lines over a full run).
        parsed: Optional[dict] = None
        for block in msg.content:
            if (getattr(block, "type", None) == "tool_use"
                    and getattr(block, "name", None) == "classify_signal"):
                parsed = dict(block.input or {})
                break
        if parsed is None:
            # No tool call (e.g. stop_reason='refusal') — skip, re-queue later.
            log.warning("LLM returned no classify_signal tool call for signal_id=%s",
                        row.get("signal_id"))
            return None

        # Low-confidence gate: bucket untrusted labels so they don't reach the
        # trade gate, but remain re-queueable.
        try:
            if float(parsed.get("confidence") or 0.0) < LOW_CONFIDENCE_THRESHOLD:
                parsed["event_type"] = LOW_CONFIDENCE_EVENT
        except (TypeError, ValueError):
            parsed["event_type"] = LOW_CONFIDENCE_EVENT
        parsed.setdefault("extracted_fields", {})

        # Compute actual cost from response usage
        actual_in = getattr(msg.usage, "input_tokens", in_toks)
        actual_out = getattr(msg.usage, "output_tokens", out_toks)
        actual_cost = estimate_cost(
            input_tokens=actual_in, output_tokens=actual_out
        )
        self.tracker.record(cost_usd=actual_cost)

        return {
            **parsed,
            "model": self.model,
            "input_tokens": actual_in,
            "output_tokens": actual_out,
            "cost_usd": actual_cost,
        }

    # ------------------------------------------------------------------

    def store(self, conn: sqlite3.Connection, *, signal_id: int,
              ticker: str, result: dict) -> None:
        """Persist one classification result to ``llm_classifications``."""
        extracted = result.get("extracted_fields") or {}
        conn.execute(
            """
            INSERT INTO llm_classifications (
                signal_id, ticker, event_type, event_subtype,
                sentiment, sentiment_magnitude, factual,
                extracted_fields, confidence,
                model, input_tokens, output_tokens, cost_usd,
                classified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id, ticker) DO UPDATE SET
                event_type = excluded.event_type,
                event_subtype = excluded.event_subtype,
                sentiment = excluded.sentiment,
                sentiment_magnitude = excluded.sentiment_magnitude,
                factual = excluded.factual,
                extracted_fields = excluded.extracted_fields,
                confidence = excluded.confidence,
                model = excluded.model,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                cost_usd = excluded.cost_usd,
                classified_at = excluded.classified_at
            """,
            (
                signal_id,
                ticker,
                result.get("event_type"),
                result.get("event_subtype"),
                result.get("sentiment"),
                result.get("sentiment_magnitude"),
                int(result.get("factual") or 0) if result.get("factual") is not None else None,
                json.dumps(extracted) if extracted else None,
                result.get("confidence"),
                result.get("model"),
                result.get("input_tokens"),
                result.get("output_tokens"),
                result.get("cost_usd"),
                _utc_now_iso(),
            ),
        )


# ----------------------------------------------------------------------
# Pipeline entry points
# ----------------------------------------------------------------------


def classify_pending(*, batch_size: int = 50,
                     classifier: Optional[LLMClassifier] = None) -> ClassifyStats:
    """Classify recent live signals that haven't been LLM-classified yet.

    Designed to be called every 1-5 min by the daemon. Uses the smart filter
    to skip noise + the hard daily-cap to stop when budget exhausted.
    """
    stats = ClassifyStats()
    if not CONFIG.has_anthropic:
        return stats

    classifier = classifier or LLMClassifier()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ss.signal_id, ss.ticker, ss.composite_score, ss.event_type,
                   ss.sentiment, ss.signal_class,
                   rs.title, rs.body, rs.source, rs.source_tier,
                   rs.raw_payload, rs.content_hash
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            LEFT JOIN llm_classifications lc
                   ON lc.signal_id = ss.signal_id AND lc.ticker = ss.ticker
            WHERE lc.id IS NULL
              AND rs.source NOT LIKE 'sec_edgar_backfill_%'
              AND rs.ingested_at >= strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now', '-2 days'))
              -- Don't classify an un-hydrated SEC stub (RSS metadata only) while
              -- it's young — give the out-of-band hydrate job time to fill the
              -- real body. After a 15-min grace, classify anyway so a stub whose
              -- body never resolves is never permanently lost.
              AND NOT (rs.source = 'sec_edgar'
                       AND COALESCE(json_extract(rs.raw_payload, '$.body_hydrated'), 1) = 0
                       AND rs.ingested_at >= strftime('%Y-%m-%dT%H:%M:%SZ',
                                                      datetime('now', '-15 minutes')))
            ORDER BY ss.composite_score DESC, ss.id DESC
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
        stats.candidates = len(rows)

        # Track content hashes already classified to skip dups within this batch
        seen_hashes: set[str] = set()

        for row in rows:
            r = dict(row)
            ok, reason = should_classify(r)
            stats.filter_reasons[reason] = stats.filter_reasons.get(reason, 0) + 1
            if not ok:
                stats.filtered_out += 1
                continue

            # Content-hash dedup: if we've classified this exact story already
            # (via another source), reuse the existing classification
            chash = r.get("content_hash")
            if chash:
                if chash in seen_hashes:
                    stats.skipped_already_done += 1
                    continue
                existing = conn.execute(
                    """
                    SELECT lc.* FROM llm_classifications lc
                    JOIN raw_signals rs ON rs.id = lc.signal_id
                    WHERE rs.content_hash = ?
                    LIMIT 1
                    """,
                    (chash,),
                ).fetchone()
                if existing:
                    # Copy the existing classification over to this signal
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
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    })
                    try:
                        from ..scoring.composite import rescore_with_classification
                        rescore_with_classification(
                            conn, signal_id=r["signal_id"], ticker=r["ticker"],
                            event_type=existing["event_type"],
                            sentiment=existing["sentiment"],
                            sentiment_magnitude=existing["sentiment_magnitude"],
                            factual=existing["factual"],
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("rescore after dedup-copy failed (%s/%s): %s",
                                    r["signal_id"], r["ticker"], exc)
                    stats.skipped_already_done += 1
                    seen_hashes.add(chash)
                    continue
                seen_hashes.add(chash)

            # Deterministic SEC routing (blueprint #3): an 8-K's Item codes / a
            # form type ARE its event taxonomy — classify without the LLM (and
            # skip the API cost) for sec_edgar rows with a real (hydrated) body.
            if r.get("source") == "sec_edgar":
                from ..ingestors.sec_item_codes import (
                    BEARISH_8K_EVENTS, sec_event_type, subtype_for_text,
                )
                try:
                    _form = json.loads(r.get("raw_payload") or "{}").get("form")
                except (TypeError, ValueError):
                    _form = None
                det_event = sec_event_type(_form, r.get("body"))
                if det_event:
                    _bearish = det_event in BEARISH_8K_EVENTS
                    det = {
                        "event_type": det_event,
                        "event_subtype": subtype_for_text(r.get("body")),
                        "sentiment": -0.6 if _bearish else 0.0,
                        "sentiment_magnitude": 0.6 if _bearish else 0.0,
                        "factual": 1, "extracted_fields": {}, "confidence": 0.9,
                        "model": "deterministic_sec",
                        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                    }
                    classifier.store(conn, signal_id=r["signal_id"],
                                     ticker=r["ticker"], result=det)
                    try:
                        from ..scoring.composite import rescore_with_classification
                        rescore_with_classification(
                            conn, signal_id=r["signal_id"], ticker=r["ticker"],
                            event_type=det_event, sentiment=det["sentiment"],
                            sentiment_magnitude=det["sentiment_magnitude"], factual=1)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("rescore after deterministic SEC classify failed "
                                    "(%s/%s): %s", r["signal_id"], r["ticker"], exc)
                    stats.classified += 1
                    continue

            # Actual LLM call
            try:
                result = classifier.classify_one(r)
            except RuntimeError as exc:
                log.error("LLM unrecoverable: %s", exc)
                stats.errors += 1
                break

            if result is None:
                # cap hit OR call failed — check which
                today_spent = today_spend_usd()
                if today_spent >= classifier.tracker.daily_cap_usd:
                    stats.cap_hit = True
                    break
                stats.errors += 1
                continue

            classifier.store(conn, signal_id=r["signal_id"],
                             ticker=r["ticker"], result=result)
            # Bridge the LLM label into the trade gate (recompute composite
            # from the corrected event_type/sentiment + write to signal_scores).
            try:
                from ..scoring.composite import rescore_with_classification
                rescore_with_classification(
                    conn, signal_id=r["signal_id"], ticker=r["ticker"],
                    event_type=result.get("event_type"),
                    sentiment=result.get("sentiment"),
                    sentiment_magnitude=result.get("sentiment_magnitude"),
                    factual=result.get("factual"),
                )
            except Exception as exc:  # noqa: BLE001 — rescore must not break classify
                log.warning("rescore after LLM classify failed (%s/%s): %s",
                            r["signal_id"], r["ticker"], exc)
            stats.classified += 1
            stats.total_cost_usd += result.get("cost_usd") or 0

    log.info(
        "LLM classify_pending: candidates=%d filtered=%d classified=%d "
        "dedup=%d errors=%d cap_hit=%s cost=$%.4f",
        stats.candidates, stats.filtered_out, stats.classified,
        stats.skipped_already_done, stats.errors,
        stats.cap_hit, stats.total_cost_usd,
    )
    return stats


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
