"""LLM-based signal classification via Anthropic Haiku.

Reads SEC filing bodies + news article bodies + Reddit/social bodies and
extracts:
  - Refined event_type (more accurate than title-only heuristic)
  - Sentiment (-1..+1) + magnitude (0..1)
  - Factual flag (1=confirmed, 0=speculation)
  - Structured fields per event type (deal size, exec role, drug name, etc.)
  - Classification confidence

Cost-controlled: hard $2/day cap enforced before every API call. Daemon
classifies in batches with smart prefiltering (skip routine/low-score/dups).
"""
from .classifier import LLMClassifier, classify_pending
from .filter import should_classify
from .spend import LLMSpendTracker, today_spend_usd

__all__ = [
    "LLMClassifier",
    "classify_pending",
    "should_classify",
    "LLMSpendTracker",
    "today_spend_usd",
]
