"""Selective conformal trade gate.

A trade-vs-skip decision per signal. Combines:
  - Calibrated probability ≥ ``min_p`` (default 0.62)
  - Conformal interval width ≤ ``max_width`` (default 0.15)
  - Optionally: ensemble agreement (variance across base models ≤ threshold)

Returns ``True`` only when the model is confidently above the trade-able
threshold. Designed to be used by the dashboard's ranking and by future
real-money execution code.

The thresholds should be calibrated against measured outcomes once the
system has ≥ 2 weeks of resolved live signals — defaults here are the
literature-suggested starting points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_MIN_P = 0.62
DEFAULT_MAX_INTERVAL_WIDTH = 0.15
DEFAULT_MAX_BASE_VARIANCE = 0.02


@dataclass(frozen=True)
class GateDecision:
    trade: bool
    reason: str
    p_calibrated: float
    interval_width: Optional[float]
    base_disagreement: Optional[float]


def decide(
    *,
    p_calibrated: float,
    interval_width: Optional[float] = None,
    base_probabilities: Optional[list[float]] = None,
    min_p: float = DEFAULT_MIN_P,
    max_interval_width: float = DEFAULT_MAX_INTERVAL_WIDTH,
    max_base_variance: float = DEFAULT_MAX_BASE_VARIANCE,
) -> GateDecision:
    """Return a trade/skip decision with reason."""
    p = float(p_calibrated)
    disagreement: Optional[float] = None
    if base_probabilities:
        n = len(base_probabilities)
        mean = sum(base_probabilities) / n
        var = sum((x - mean) ** 2 for x in base_probabilities) / n
        disagreement = var

    if p < min_p:
        return GateDecision(False, f"p={p:.3f} < min {min_p:.2f}",
                            p, interval_width, disagreement)
    if interval_width is not None and interval_width > max_interval_width:
        return GateDecision(False,
                            f"conformal interval {interval_width:.3f} > max {max_interval_width:.2f}",
                            p, interval_width, disagreement)
    if disagreement is not None and disagreement > max_base_variance:
        return GateDecision(False,
                            f"base-model disagreement var={disagreement:.4f} > {max_base_variance:.3f}",
                            p, interval_width, disagreement)
    return GateDecision(True, "all gates passed", p, interval_width, disagreement)
