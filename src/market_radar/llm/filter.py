"""Filter rules for deciding which signals are worth LLM-classifying.

Goal: spend LLM budget only on signals that could actually move money.
Skip routine paperwork, low-tier social noise, and content we've already
classified (via content_hash deduplication).

Rough policy:
  - Always classify: Tier 1 SEC filings (8-K, Form 4 insider, 13D activist,
    425 M&A, S-1 IPO). These are where the real signal lives.
  - Always classify: Tier 2 news with composite_score >= 6.0
  - Classify Tier 3 (Reddit/StockTwits) only if composite_score >= 7.0
    AND not on a known pump-prone subreddit
  - Skip: routine_prospectus, routine_proxy, speculation
  - Skip: content_hash already classified anywhere in the DB
  - Skip: rows whose body is too short to be meaningfully classified, or
    big enough that the LLM call would be unaffordable (see body-length
    bounds below).
"""
from __future__ import annotations

from typing import Any, Optional


ALWAYS_CLASSIFY_FORMS = {
    "8-K", "8-K/A", "4", "SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A",
    "425", "S-1", "S-1/A", "DEF 14A",
}

SKIP_EVENT_TYPES = {
    "routine_prospectus",
    "routine_proxy",
    "speculation",
    "passive_5pct_stake",
}

PUMP_SUBREDDITS = {
    "wallstreetbets", "pennystocks", "Shortsqueeze", "SPACs",
    "Daytrading", "biotechplays",
}


# Body length bounds for SEC filings. Rows with bodies smaller than
# MIN_BODY_CHARS_SEC almost always failed body-fetching and would feed
# the LLM essentially just the title — exactly the failure mode that
# produced the 96% "other" debacle on 2026-05-13. Rows above
# MAX_BODY_CHARS_SEC would blow up our per-call cost; the body fetcher
# already caps stored bodies at 8 000 chars so this is mostly a safety
# net for any rows that slip past the fetcher.
MIN_BODY_CHARS_SEC = 500
MAX_BODY_CHARS_SEC = 50_000


def should_classify(row: dict[str, Any]) -> tuple[bool, str]:
    """Return (should_run_llm, reason).

    ``row`` is a dict joined from raw_signals + signal_scores. Expected keys:
    source, source_tier, event_type, composite_score, raw_payload (with form),
    sentiment, content_hash, signal_class.
    """
    source = (row.get("source") or "")
    tier = int(row.get("source_tier") or 0)
    event = row.get("event_type") or ""
    score = float(row.get("composite_score") or 0)

    # Skip backfill-only when we're processing live; but for the bulk
    # backfill enrichment, the caller explicitly opts in via a different
    # path.  Here we just include all sources.

    # 1. Hard skips
    if event in SKIP_EVENT_TYPES:
        return False, f"skip-event:{event}"
    if score < 4.0:
        return False, f"low-score:{score:.1f}"

    # 2. Tier 3 social: require higher bar, skip pump-prone subs
    if tier == 3:
        if any(p in source.lower() for p in {f"reddit_{s.lower()}" for s in PUMP_SUBREDDITS}):
            if score < 7.5:
                return False, f"pump-sub-low-score:{source}"
        if score < 6.0:
            return False, f"tier3-low-score:{score:.1f}"
        return True, "tier3-high-score"

    # 3. Tier 2 mainstream news: classify if score >= 6.0
    if tier == 2:
        if score >= 6.0:
            return True, "tier2-meaningful"
        return False, f"tier2-low-score:{score:.1f}"

    # 4. Tier 1 SEC: classify if form is in the always list
    if tier == 1:
        # Look at form from raw_payload if available
        # raw_payload comes from db as JSON string — caller should parse first
        form = (row.get("form") or "").upper()
        if not form and row.get("raw_payload"):
            import json
            try:
                rp = json.loads(row["raw_payload"]) if isinstance(row["raw_payload"], str) else row["raw_payload"]
                form = (rp.get("form") or "").upper()
            except Exception:
                pass
        if form not in {f.upper() for f in ALWAYS_CLASSIFY_FORMS}:
            return False, f"tier1-uninteresting-form:{form}"

        # Body length guard — prevents the "title-only" failure mode from
        # the 2026-05-13 LLM round, where the SEC ingestor stored only
        # the RSS metadata stub and the LLM saw an empty body. Without
        # a substantive body the LLM has nothing useful to classify.
        body = row.get("body") or ""
        body_len = len(body)
        if body_len < MIN_BODY_CHARS_SEC:
            return False, f"tier1-body-too-short:{body_len}"
        if body_len > MAX_BODY_CHARS_SEC:
            return False, f"tier1-body-too-long:{body_len}"
        return True, f"tier1-form:{form}"

    return False, "unknown-tier"
