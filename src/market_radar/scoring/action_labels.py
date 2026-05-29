"""Map (event_type, sentiment, score, measured_edge, holdings) → action label.

Action vocabulary the user agreed to:
    STRONG BUY  STRONG_BUY
    BUY         BUY
    WATCH       WATCH
    TRIM/EXIT   TRIM (only when user holds the ticker)
    SHORT       SHORT       (CFD-actionable; T212 Invest/ISA is long-only)
    STRONG SHORT STRONG_SHORT
    AVOID       AVOID
    SKIP        SKIP

The label is purely a *categorical description of the data*, not a trade
recommendation. STRONG variants only fire when measured edge crosses a
defined threshold over a defined sample size — never invented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .impact import direction_for

Action = Literal[
    "STRONG_BUY", "BUY", "WATCH", "TRIM", "SHORT", "STRONG_SHORT", "AVOID", "SKIP"
]


# Score thresholds for "strong" qualifier on top of direction
SCORE_STRONG = 8.5
SCORE_BUY = 6.0
SCORE_WATCH_FLOOR = 4.0

# Hit-rate thresholds. STRONG label requires BOTH a strong score AND
# measured edge that backs it up.
EDGE_HIT_RATE_LONG = 0.60      # ≥60% positive 5d return
EDGE_HIT_RATE_SHORT = 0.40     # ≤40% (i.e. ≥60% negative)
MIN_SAMPLES = 50

# ML model probability thresholds. The model is independent of the
# class-aggregate hit rate, so we accept either evidence path for STRONG.
MODEL_P_LONG = 0.62
MODEL_P_SHORT = 0.38


@dataclass
class ActionLabel:
    action: Action               # one of the Action literals
    display: str                 # pretty text for the UI
    direction: str               # 'long' / 'short' / 'neutral' / 'unknown'
    edge_built: bool             # True if STRONG variant is justified by data
    reason: str                  # one-line explanation
    rationale_short: str         # 5-10 word summary


def label_signal(
    *,
    event_type: Optional[str],
    sentiment: Optional[float],
    composite_score: float,
    user_holds_ticker: bool = False,
    measured_hit_rate_5d: Optional[float] = None,
    measured_samples: int = 0,
    model_p_5d: Optional[float] = None,
) -> ActionLabel:
    """Compute an action label for one signal.

    STRONG variants fire when ANY of these is true:
      a) Class-aggregate measured hit rate clears EDGE_HIT_RATE_LONG/SHORT
         over MIN_SAMPLES historical samples, OR
      b) ML model probability clears MODEL_P_LONG/SHORT.

    Both are independent evidence paths.
    """
    direction = direction_for(event_type, sentiment)

    # SKIP for noise / routine
    if composite_score < SCORE_WATCH_FLOOR or event_type in {"routine_prospectus", "routine_proxy"}:
        return ActionLabel(
            action="SKIP",
            display="SKIP",
            direction=direction,
            edge_built=False,
            reason="Routine or low-impact signal — not actionable",
            rationale_short="Routine / low impact",
        )

    has_strong_score = composite_score >= SCORE_STRONG
    has_min_data = measured_samples >= MIN_SAMPLES
    edge_long = (
        has_min_data
        and measured_hit_rate_5d is not None
        and measured_hit_rate_5d >= EDGE_HIT_RATE_LONG
    ) or (
        model_p_5d is not None and model_p_5d >= MODEL_P_LONG
    )
    edge_short = (
        has_min_data
        and measured_hit_rate_5d is not None
        and measured_hit_rate_5d <= EDGE_HIT_RATE_SHORT
    ) or (
        model_p_5d is not None and model_p_5d <= MODEL_P_SHORT
    )

    # Direction-aware branches
    if direction == "long":
        if has_strong_score and edge_long:
            return ActionLabel(
                action="STRONG_BUY",
                display="STRONG BUY",
                direction="long",
                edge_built=True,
                reason=f"Bullish signal + measured 5d hit rate {measured_hit_rate_5d:.0%} (n={measured_samples})",
                rationale_short=f"Bullish + {int(measured_hit_rate_5d*100)}% edge (n={measured_samples})",
            )
        if composite_score >= SCORE_BUY:
            edge_note = (
                f"edge {measured_hit_rate_5d:.0%} n={measured_samples}"
                if has_min_data and measured_hit_rate_5d is not None
                else f"edge building (n={measured_samples}/{MIN_SAMPLES})"
            )
            return ActionLabel(
                action="BUY",
                display="BUY",
                direction="long",
                edge_built=has_min_data,
                reason=f"Bullish event signal — {edge_note}",
                rationale_short=f"Bullish · {edge_note}",
            )

    if direction == "short":
        # User holds the ticker → TRIM/EXIT regardless of edge data
        if user_holds_ticker:
            return ActionLabel(
                action="TRIM",
                display="TRIM / EXIT",
                direction="short",
                edge_built=has_min_data,
                reason="Bearish signal on a position you hold",
                rationale_short="Bearish · you own it",
            )
        if has_strong_score and edge_short:
            return ActionLabel(
                action="STRONG_SHORT",
                display="STRONG SHORT",
                direction="short",
                edge_built=True,
                reason=f"Bearish signal + measured 5d hit rate {measured_hit_rate_5d:.0%} (n={measured_samples})",
                rationale_short=f"Bearish + {int(measured_hit_rate_5d*100)}% edge (n={measured_samples})",
            )
        if composite_score >= SCORE_BUY:
            edge_note = (
                f"edge {measured_hit_rate_5d:.0%} n={measured_samples}"
                if has_min_data and measured_hit_rate_5d is not None
                else f"edge building (n={measured_samples}/{MIN_SAMPLES})"
            )
            return ActionLabel(
                action="SHORT",
                display="SHORT",
                direction="short",
                edge_built=has_min_data,
                reason=f"Bearish event signal — {edge_note}",
                rationale_short=f"Bearish · {edge_note}",
            )
        return ActionLabel(
            action="AVOID",
            display="AVOID",
            direction="short",
            edge_built=False,
            reason="Bearish signal — avoid new long entry",
            rationale_short="Avoid entry",
        )

    # Neutral / unknown direction
    if composite_score >= SCORE_BUY:
        return ActionLabel(
            action="WATCH",
            display="WATCH",
            direction=direction,
            edge_built=False,
            reason="Significant signal but direction unclear — read filing",
            rationale_short="Watch · direction TBD",
        )

    return ActionLabel(
        action="SKIP",
        display="SKIP",
        direction=direction,
        edge_built=False,
        reason="Low-impact or unclear signal",
        rationale_short="Low impact",
    )
