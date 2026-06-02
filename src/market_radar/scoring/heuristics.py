"""Rule-based classification heuristics.

Used when no Anthropic API key is configured (free fallback) and as the
initial pass that the LLM later refines. Cheap, deterministic, transparent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class HeuristicClassification:
    event_type: str
    sentiment: float          # -1..+1
    sentiment_magnitude: float  # 0..1
    factual: int              # 0/1


# ------------------------------------------------------------------
# Event type keyword map. Patterns are tried in priority order; first
# match wins. The form_event column on SEC filings (already populated
# by ingestion) is used as a stronger hint when present.
# ------------------------------------------------------------------

EVENT_PATTERNS: list[tuple[str, list[str]]] = [
    # M&A — confirmed vs rumored handled below in factual flag
    ("m_a_announcement", [
        r"\b(?:agrees to acquire|to acquire|will acquire|has acquired|"
        r"acquires|to buy|will buy|merges? with|merger with|"
        r"merger of equals|definitive agreement to acquire|"
        r"all-cash deal|cash and stock|stock-for-stock|going private|"
        r"take[- ]?private|leveraged buyout|LBO|"
        r"combine[sd]? with|completes (?:the )?acquisition|"
        r"deal (?:closed|completed)|to merge|merger agreement)\b",
    ]),
    ("m_a_rumor", [
        r"\b(?:in talks to acquire|in talks to buy|considering acquisition|"
        r"weighing (?:a )?(?:bid|deal|sale|acquisition)|reportedly (?:in talks|considering)|"
        r"could acquire|may acquire|approached by|exploring (?:a )?sale|"
        r"strategic alternatives|rumored to|rumours? of)\b",
    ]),
    ("earnings_beat", [
        r"\b(?:beats|topped|exceed(?:s|ed)) (?:earnings|EPS|revenue|estimates|expectations|forecast)\b",
        r"\b(?:earnings|EPS|revenue) beat\b",
    ]),
    ("earnings_miss", [
        r"\b(?:misses|missed|fell short of) (?:earnings|EPS|revenue|estimates|expectations|forecast)\b",
        r"\b(?:earnings|EPS|revenue) miss\b",
    ]),
    ("guidance_raise", [
        r"\b(?:raises|hikes|boost(?:s|ed)|lifts) (?:full[- ]year |FY |its |the )?(?:outlook|guidance|forecast)\b",
        r"\bsees .{0,20}\babove\b",
        r"\bups .{0,15}(?:outlook|guidance|forecast|target)\b",
        r"\braises target\b",
    ]),
    ("guidance_cut", [
        r"\b(?:cuts|lowers|trims|reduces|slashes) (?:full[- ]year |FY |its |the )?(?:outlook|guidance|forecast)\b",
        r"\bsees .{0,20}\bbelow\b",
    ]),
    ("fda_approval", [
        r"\b(?:FDA approves|FDA approval|approved by the FDA|granted (?:FDA )?approval|"
        r"received (?:FDA )?approval|wins (?:FDA )?approval)\b",
    ]),
    ("fda_rejection", [
        r"\b(?:FDA rejects|FDA rejection|rejected by the FDA|complete response letter|"
        r"CRL from FDA|FDA declines|FDA denies)\b",
    ]),
    ("clinical_trial_result", [
        r"\b(?:Phase 3|Phase 2|Phase III|Phase II|clinical trial|primary endpoint|"
        r"topline|met its (?:primary )?endpoint|missed its (?:primary )?endpoint)\b",
    ]),
    ("analyst_upgrade", [
        r"\b(?:upgraded? to (?:buy|outperform|overweight)|raised price target|"
        r"price target (?:raised|increased|hiked)|reiterates? buy)\b",
    ]),
    ("analyst_downgrade", [
        r"\b(?:downgraded? to (?:sell|underperform|underweight)|cut price target|"
        r"price target (?:cut|lowered|trimmed))\b",
    ]),
    ("insider_buy", [
        r"\b(?:insider buying|director (?:buys|purchases)|CEO (?:buys|purchases)|"
        r"officer (?:bought|purchased)|10% owner reports purchase)\b",
    ]),
    ("insider_sell", [
        r"\b(?:insider selling|director (?:sells|disposes)|CEO (?:sells|disposes)|"
        r"officer (?:sold|disposed))\b",
    ]),
    ("activist_position", [
        r"\b(?:activist|13D filing|files 13D|takes (?:a )?stake|reveals stake|"
        r"discloses (?:a )?stake|building a position)\b",
    ]),
    ("ipo_registration", [
        r"\b(?:files for IPO|S-1 filing|IPO registration|set to go public|"
        r"plans to go public|prepares for IPO)\b",
    ]),
    ("macro", [
        r"\b(?:CPI|inflation|Fed (?:rate|hike|cut|decision)|FOMC|jobs report|"
        r"nonfarm payrolls|GDP|PMI|ISM)\b",
    ]),
    ("short_seller_report", [
        r"\b(?:short[- ]seller|Hindenburg|Muddy Waters|alleges fraud|short report)\b",
    ]),
    ("lawsuit", [
        r"\b(?:lawsuit|sued|class[- ]action|investigation|subpoena|SEC charges|"
        r"DOJ probe|antitrust)\b",
    ]),
    ("leadership_change", [
        r"\b(?:CEO (?:steps down|resigns|departs|appointed|named)|new CEO|"
        r"interim CEO|CFO (?:steps down|resigns|appointed)|board (?:appoints|elects))\b",
    ]),
    ("buyback", [
        r"\b(?:share repurchase|buyback (?:program|plan)|authoriz(?:es|ed) buyback|"
        r"announce[sd]? .{0,30} buyback)\b",
    ]),
    ("contract_award", [
        r"\b(?:wins .{0,20}contract|awarded .{0,20}contract|contract worth|"
        r"secures .{0,20}deal)\b",
    ]),
    ("dividend", [
        r"\b(?:dividend (?:increase|hike|raise|boost)|special dividend|"
        r"initiates? dividend|dividend cut|suspends? dividend)\b",
    ]),
    ("speculation", [
        r"\b(?:could|might|may|reportedly|rumored|allegedly|speculation|"
        r"hot stock|moonshot|squeeze|to the moon|YOLO|diamond hands)\b",
    ]),
]


# Words that signal a bullish or bearish slant. We weight bullish/bearish
# matches the same and use the difference to estimate sentiment direction.
BULLISH_WORDS = {
    "beats", "beat", "tops", "topped", "exceeds", "exceeded", "surges", "surged",
    "soars", "soared", "rallies", "rallied", "rallying", "jumps", "jumped",
    "spikes", "spiked", "climbs", "climbed", "rises", "rose", "rising", "gains",
    "gained", "gainer", "outperforms", "outperformed", "boosts", "boosted",
    "raises", "raised", "approve", "approves", "approved", "approval",
    "buy", "buys", "bullish", "upgrade", "upgrades", "upgraded", "outperform",
    "overweight", "beat", "blowout", "strong", "stronger", "bullish",
    "record", "all-time-high", "all-time high", "ath", "breakout", "moon",
    "rockets", "skyrockets", "vaults", "pops", "explodes",
}
BEARISH_WORDS = {
    "misses", "missed", "tumbles", "tumbled", "plunges", "plunged", "plummets",
    "plummeted", "drops", "dropped", "falls", "fell", "falling", "slumps",
    "slumped", "slides", "slid", "sliding", "loses", "lost", "loser",
    "underperforms", "underperformed", "cuts", "cut", "lowers", "lowered",
    "reject", "rejects", "rejected", "rejection", "sell", "sells", "bearish",
    "downgrade", "downgrades", "downgraded", "underperform", "underweight",
    "warning", "weak", "weakness", "disappoint", "disappointing", "disappointed",
    "bankruptcy", "delisting", "fraud", "scandal", "probe", "lawsuit",
    "crash", "collapse", "tank", "tanked",
    "craters", "sinks", "sags", "nosedives", "dives",
}

# Words signalling speculation / non-factual content
SPECULATION_WORDS = {
    "could", "might", "may", "reportedly", "rumored", "rumour", "rumor",
    "allegedly", "speculation", "rumored", "potential", "potentially",
    "consider", "considering", "explores", "weighing", "in talks",
    "I think", "i think", "imo", "IMO", "imho", "i bet", "I bet", "yolo", "YOLO",
}

# Words signalling factual / confirmed content
FACTUAL_WORDS = {
    "announced", "announces", "reports", "reported", "filed", "files",
    "filed", "approved", "rejected", "completed", "closed", "issued",
    "8-K", "S-1", "10-K", "10-Q", "Form 4", "13D", "13G", "DEF 14A",
}


def classify_heuristic(
    title: Optional[str],
    body: Optional[str],
    *,
    sec_form_event: Optional[str] = None,
    source: Optional[str] = None,
) -> HeuristicClassification:
    """Return a coarse rule-based classification."""
    text = " ".join(filter(None, [title, body]))
    text_lower = text.lower()

    # --- factual flag ---
    if (source or "") == "sec_edgar":
        factual = 1
    else:
        has_speculation = any(w in text_lower for w in SPECULATION_WORDS)
        has_factual_marker = any(w in text_lower for w in FACTUAL_WORDS)
        if has_factual_marker and not has_speculation:
            factual = 1
        elif has_speculation and not has_factual_marker:
            factual = 0
        else:
            factual = 1  # default to factual for news; LLM can refine later

    # --- event type ---
    event_type = None
    if sec_form_event:
        # Translate SEC form labels to our taxonomy
        if sec_form_event == "insider_transaction":
            event_type = "insider_transaction"
        elif sec_form_event == "activist_position":
            event_type = "activist_position"
        elif sec_form_event == "passive_5pct_stake":
            event_type = "passive_5pct_stake"
        elif sec_form_event == "ipo_registration":
            event_type = "ipo_registration"
        elif sec_form_event == "proxy_statement":
            event_type = "proxy_statement"
        elif sec_form_event == "material_event":
            # 8-Ks vary — fall through to keyword detection
            pass

    # Backfill-specific form_event values map to our taxonomy
    if not event_type and sec_form_event:
        backfill_map = {
            "m_a_communication": "m_a_announcement",
            "material_event_amend": "material_event_amend",
            "activist_position_amend": "activist_position",
            "passive_5pct_stake_amend": "passive_5pct_stake",
            "ipo_registration_amend": "ipo_registration",
        }
        event_type = backfill_map.get(sec_form_event)

    if not event_type:
        for et, patterns in EVENT_PATTERNS:
            if any(re.search(p, text, flags=re.IGNORECASE) for p in patterns):
                event_type = et
                break

    # Generic earnings fallback: precise patterns above win first, but a real
    # catalyst phrased loosely (e.g. "Q3 revenue jumps to a record") still
    # deserves an earnings label rather than falling through to "other".
    if not event_type:
        earnings_ctx = re.search(
            r"\b(?:Q[1-4]|quarterly|fiscal|quarter|earnings|revenue|EPS|profit|results)\b",
            text, flags=re.IGNORECASE,
        )
        if earnings_ctx:
            bullish_move = re.search(
                r"\b(?:jumps|surges|soars|rockets|tops|record|beats?|"
                r"above estimates|blow past)\b",
                text, flags=re.IGNORECASE,
            )
            bearish_move = re.search(
                r"\b(?:plunges|misses|falls short|below estimates|drops|"
                r"tumbles|sinks)\b",
                text, flags=re.IGNORECASE,
            )
            if bullish_move:
                event_type = "earnings_beat"
            elif bearish_move:
                event_type = "earnings_miss"

    if not event_type:
        event_type = "other"

    # --- sentiment direction + magnitude ---
    bull_hits = sum(1 for w in BULLISH_WORDS if re.search(rf"\b{re.escape(w)}\b", text_lower))
    bear_hits = sum(1 for w in BEARISH_WORDS if re.search(rf"\b{re.escape(w)}\b", text_lower))

    # Numeric-move sentiment: a big percentage move is the single strongest
    # directional signal in a headline (e.g. a "+47%" is far more informative
    # than any adjective) yet was previously ignored entirely. Count any
    # move >= 10% as a strong hit so it dominates magnitude.
    for m in re.finditer(
        r"(?:(?P<sign>[+-])|\b(?P<dir>up|down|soars|gains?|loses?|drops?)\s+)"
        r"(?P<num>\d{1,3}(?:\.\d+)?)\s*%",
        text, flags=re.IGNORECASE,
    ):
        try:
            num = float(m.group("num"))
        except (TypeError, ValueError):
            continue
        if num < 10:
            continue
        sign = m.group("sign")
        direction = (m.group("dir") or "").lower()
        if sign == "-" or direction in {"down", "loses", "lose", "drops", "drop"}:
            bear_hits += 1
        else:
            # "+", "up", "soars", "gains", or a bare percentage default bullish
            bull_hits += 1

    total = bull_hits + bear_hits
    if total == 0:
        sentiment = 0.0
        sentiment_magnitude = 0.0
    else:
        sentiment = (bull_hits - bear_hits) / total
        # Magnitude floor: a single decisive word used to map to 0.2, which the
        # downstream gate (needs >= 0.3) discarded. Floor at 0.6 for one hit and
        # grow from there, capped at 1.0.
        sentiment_magnitude = min(0.4 + 0.2 * total, 1.0)

    # Event-type bias for sentiment if keywords were too neutral. Only fires for
    # events with a known directional bias; for event_type == "other" we keep
    # the bull/bear difference computed above rather than forcing 0.0, so a
    # directional word still produces real sentiment on a generic bucket.
    if abs(sentiment) < 0.2:
        bullish_events = {
            "earnings_beat", "guidance_raise", "fda_approval", "analyst_upgrade",
            "insider_buy", "buyback", "dividend",
        }
        bearish_events = {
            "earnings_miss", "guidance_cut", "fda_rejection", "analyst_downgrade",
            "insider_sell", "lawsuit",
        }
        if event_type in bullish_events:
            sentiment = 0.6
            sentiment_magnitude = max(sentiment_magnitude, 0.5)
        elif event_type in bearish_events:
            sentiment = -0.6
            sentiment_magnitude = max(sentiment_magnitude, 0.5)

    return HeuristicClassification(
        event_type=event_type,
        sentiment=round(sentiment, 3),
        sentiment_magnitude=round(sentiment_magnitude, 3),
        factual=factual,
    )
