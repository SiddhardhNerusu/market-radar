"""Prompt + JSON schema for LLM signal classification.

We use Anthropic's structured-output via JSON. Returns a strict shape the
classifier validates before storing. Failed JSON is rejected; we don't
attempt to "recover" malformed responses — better to skip and re-classify
later than store bad labels.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a precise financial-event classifier for an algorithmic trading research system.

You will be shown ONE piece of financial content (SEC filing, news article, or social post). Your job is to extract a structured classification of what's actually happening, NOT to predict the price move.

Return ONE JSON object with this exact schema. Use null for unknown fields. Be conservative — when in doubt, choose "other" or null rather than guess.

{
  "event_type": one of [
    "m_a_announcement",         // confirmed merger/acquisition (target/acquirer named, deal terms)
    "m_a_rumor",                // unconfirmed talks/exploration
    "earnings_beat",            // company reported and beat estimates
    "earnings_miss",            // company reported and missed estimates
    "guidance_raise",           // company raised forward guidance
    "guidance_cut",             // company lowered forward guidance
    "fda_approval",             // regulatory approval (drug, device, etc.)
    "fda_rejection",            // CRL, declined, rejected
    "analyst_upgrade",          // analyst rating raised / price target up
    "analyst_downgrade",        // analyst rating cut / price target down
    "insider_buy",              // company insider purchased stock
    "insider_sell",             // company insider sold stock
    "activist_position",        // 13D filing, activist taking stake
    "buyback",                  // share repurchase announced
    "dividend",                 // dividend initiated/raised/cut
    "leadership_change",        // CEO/CFO appointed/resigned
    "lawsuit",                  // legal action, regulatory probe
    "ipo_registration",         // S-1 filed
    "macro",                    // Fed, CPI, tariffs, broad market news
    "stock_split",              // stock split, reverse split, spinoff
    "short_seller_report",      // Hindenburg, Muddy Waters, etc.
    "clinical_trial_result",    // Phase 2/3 trial results
    "contract_award",           // major contract win/loss
    "layoffs",                  // job cuts, restructuring
    "product_launch",           // new product, partnership, service
    "earnings_announcement",    // announces upcoming earnings date
    "other"                     // none of the above
  ],

  "sentiment": float in [-1, +1],     // -1 strongly bearish for the named ticker, +1 strongly bullish
  "sentiment_magnitude": float in [0, 1],  // how confident is the directional sentiment (0=neutral, 1=very strong)
  "factual": 0 or 1,                  // 1=confirmed/factual (filing, official press release), 0=speculation/rumor

  "tickers_mentioned": [<ticker symbol>, ...],  // tickers this signal is genuinely ABOUT (not just mentioned in passing)

  "extracted_fields": {                // event-type-specific structured fields; null any not applicable
    "deal_size_usd": float or null,
    "acquirer_ticker": string or null,
    "target_ticker": string or null,
    "eps_surprise_pct": float or null,
    "revenue_surprise_pct": float or null,
    "guidance_change_pct": float or null,
    "insider_role": string or null,        // "CEO", "CFO", "Director", "10% Owner"
    "insider_size_usd": float or null,
    "drug_name": string or null,
    "indication": string or null,
    "fda_decision": string or null,         // "approved", "rejected", "delayed", "fast-track"
    "trial_phase": string or null,          // "Phase 1/2/3"
    "trial_result": string or null,         // "positive", "negative", "mixed", "primary endpoint met"
    "analyst_firm": string or null,
    "price_target_old": float or null,
    "price_target_new": float or null,
    "buyback_size_usd": float or null,
    "dividend_change_pct": float or null,
    "lawsuit_party": string or null,        // "SEC", "DOJ", "shareholder class action"
    "macro_topic": string or null           // "rates", "inflation", "tariffs", "fed_decision"
  },

  "confidence": float in [0, 1]       // your confidence in this classification overall
}

Call the classify_signal tool with your structured classification. Be conservative — when a genuine event class doesn't clearly apply, set a LOW confidence rather than forcing a label.
"""


# Forced tool-use schema (blueprint #3 L137): forcing this tool call guarantees a
# schema-VALID JSON object — no free-text parsing, no json.loads reject path, no
# code-fence stripping. The classifier reads block.input directly.
_EVENT_TYPES = [
    "m_a_announcement", "m_a_rumor", "earnings_beat", "earnings_miss",
    "guidance_raise", "guidance_cut", "fda_approval", "fda_rejection",
    "analyst_upgrade", "analyst_downgrade", "insider_buy", "insider_sell",
    "activist_position", "buyback", "dividend", "leadership_change", "lawsuit",
    "ipo_registration", "macro", "stock_split", "short_seller_report",
    "clinical_trial_result", "contract_award", "layoffs", "product_launch",
    "earnings_announcement", "other",
]

CLASSIFY_TOOL = {
    "name": "classify_signal",
    "description": "Return the structured classification of the financial content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": _EVENT_TYPES},
            "sentiment": {"type": "number", "minimum": -1, "maximum": 1},
            "sentiment_magnitude": {"type": "number", "minimum": 0, "maximum": 1},
            "factual": {"type": "integer", "enum": [0, 1]},
            "tickers_mentioned": {"type": "array", "items": {"type": "string"}},
            "extracted_fields": {"type": "object"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["event_type", "sentiment", "sentiment_magnitude", "factual",
                     "tickers_mentioned", "extracted_fields", "confidence"],
    },
}

# Below this confidence the label is untrusted — bucketed so it does NOT pass the
# trade gate but is re-queueable.
LOW_CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_EVENT = "unclassified_low_confidence"


def build_user_prompt(*, title: str | None, body: str | None,
                      source: str | None, primary_ticker: str | None) -> str:
    """Format a single signal into the user message."""
    body_trimmed = (body or "")[:3000]  # cap to control input tokens
    return (
        f"Source: {source or 'unknown'}\n"
        f"Primary ticker hint: {primary_ticker or 'unknown'}\n\n"
        f"Title: {title or '(no title)'}\n\n"
        f"Body:\n{body_trimmed if body_trimmed else '(no body)'}"
    )


# Approximate token counts for cost prediction (Haiku tokenizer estimate)
def approx_input_tokens(*, title: str | None, body: str | None) -> int:
    """Rough token estimate: ~4 chars per token + system-prompt overhead."""
    SYS_OVERHEAD = 900
    text = (title or "") + (body or "")[:3000]
    return SYS_OVERHEAD + len(text) // 4 + 25  # +25 for user-msg framing


def approx_output_tokens() -> int:
    """Typical JSON response: ~200 tokens."""
    return 250
