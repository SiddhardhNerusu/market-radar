"""Event-type impact + action-label mapping.

Two roles in one module:
  1. ``EVENT_IMPACT`` — how much each event type *typically* moves prices.
     Used by the composite scorer to weight signals by market relevance.
  2. ``direction_for`` — maps (event_type, sentiment) to a directional bias
     used to choose between BUY / SHORT / WATCH / EXIT-style labels.
"""
from __future__ import annotations

from typing import Literal, Optional


# 0–3 scale: how much price action does this event usually drive?
# Calibrated qualitatively now; the ML model will refine empirically.
EVENT_IMPACT: dict[str, float] = {
    # Highest impact — frequently 5–20% moves
    "m_a_announcement":   3.0,
    "m_a_rumor":          2.4,
    "fda_approval":       3.0,
    "fda_rejection":      3.0,
    "earnings_beat":      2.7,
    "earnings_miss":      2.7,
    "guidance_raise":     2.5,
    "guidance_cut":       2.5,
    "macro":              2.8,
    "activist_position":  2.4,
    "clinical_trial_result": 3.0,   # binary biotech catalyst — large moves
    "short_seller_report": 2.4,     # activist short report — sharp downside

    # Medium impact — usually 1–5% moves
    "analyst_upgrade":    1.8,
    "analyst_downgrade":  1.8,
    "insider_buy":        2.0,
    "insider_sell":       1.4,   # often scheduled / less informative
    "insider_transaction": 1.8,  # generic; need filing read to know direction
    "buyback":            1.8,
    "dividend":           1.3,
    "leadership_change":  1.8,
    "lawsuit":            1.6,
    "contract_award":     1.9,      # major contract / deal win

    # Lower impact — usually <1% moves
    "ipo_registration":   1.2,
    "ipo_registration_amend": 0.9,
    "material_event":     1.5,   # generic 8-K; LLM would refine
    "material_event_amend": 1.0,
    "speculation":        0.7,
    "other":              1.0,

    # Routine / administrative — almost no impact
    "proxy_statement":    0.4,
    "passive_5pct_stake": 0.6,
    "routine_prospectus": 0.2,
    "routine_proxy":      0.2,
}


# Event direction bias. None = direction depends on filing content (e.g. Form 4
# is buy or sell — title alone can't tell).
EVENT_BIAS: dict[str, Optional[Literal["long", "short", "neutral"]]] = {
    "m_a_announcement":   "long",     # target usually pops on announcement
    "m_a_rumor":          "long",
    "fda_approval":       "long",
    "fda_rejection":      "short",
    "earnings_beat":      "long",
    "earnings_miss":      "short",
    "guidance_raise":     "long",
    "guidance_cut":       "short",
    "macro":              None,
    "activist_position":  "long",
    "analyst_upgrade":    "long",
    "analyst_downgrade":  "short",
    "insider_buy":        "long",
    "insider_sell":       "short",
    "insider_transaction": None,      # need filing direction
    "buyback":            "long",
    "dividend":           "long",
    "leadership_change":  None,
    "lawsuit":            "short",
    "clinical_trial_result": None,    # depends on met vs missed endpoint
    "short_seller_report": "short",
    "contract_award":     "long",
    "ipo_registration":   "neutral",
    "ipo_registration_amend": "neutral",
    "material_event":     None,
    "material_event_amend": None,
    "speculation":        None,
    "proxy_statement":    "neutral",
    "passive_5pct_stake": "neutral",
    "routine_prospectus": "neutral",
    "routine_proxy":      "neutral",
    "other":              None,
}


def impact_for(event_type: Optional[str]) -> float:
    if not event_type:
        return EVENT_IMPACT["other"]
    return EVENT_IMPACT.get(event_type, EVENT_IMPACT["other"])


def direction_for(event_type: Optional[str], sentiment: Optional[float]) -> Literal["long", "short", "neutral", "unknown"]:
    bias = EVENT_BIAS.get(event_type or "")
    if bias is not None:
        return bias  # type: ignore[return-value]
    if sentiment is None:
        return "unknown"
    if sentiment > 0.2:
        return "long"
    if sentiment < -0.2:
        return "short"
    return "neutral"
