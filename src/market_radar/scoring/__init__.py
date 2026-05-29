"""Composite signal scoring + LLM classification."""
from .composite import ScoringStats, score_pending
from .heuristics import HeuristicClassification, classify_heuristic
from .source_weights import SOURCE_WEIGHTS, weight_for

__all__ = [
    "ScoringStats",
    "score_pending",
    "HeuristicClassification",
    "classify_heuristic",
    "SOURCE_WEIGHTS",
    "weight_for",
]
