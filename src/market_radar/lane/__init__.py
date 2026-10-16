"""The insider lane: a single long-only, multi-day equity strategy.

What it trades
--------------
Large open-market insider purchases (SEC Form 4, transaction code ``P``,
acquired, not under a 10b5-1 plan) on names priced >= $10. One bet per
(ticker, event_day). Entry is a marketable-limit BRACKET order so the stop
and target live at the broker from the moment of fill; the lane adds only a
time stop and a daily-loss flatten.

Why this and nothing else
-------------------------
The June 2026 research (docs/EDGE_HUNT_2026-06-16.md) found no edge in any
catalyst family on the data available. The only candidate that was
theory-backed, retail-defensible and positive at the median was this one,
and it was *fragile*: outlier-carried, p ~ 0.075, 13 distinct days. This
lane exists to settle that question honestly: every qualifying event is
written to ``edge_ledger`` whether or not it is traded, and the promotion
gate (``lane/promotion.py``) decides from that forward record alone.

Modules
-------
- ``config``      LaneConfig from LANE_* env vars (sizing, gates, caps).
- ``sessions``    US/Eastern session helpers: event_day, open/close windows.
- ``candidates``  Form 4 -> deduped InsiderCandidate list (pure + SQL).
- ``sizing``      fixed-fractional risk sizing against a real ATR stop (pure).
- ``ledger``      edge_ledger writes and 5d/20d resolution (pure core).
- ``promotion``   the fragile -> real promotion gate (pure).
- ``trader``      InsiderLaneTrader: the loop.
"""
from .config import LaneConfig
from .trader import InsiderLaneTrader

__all__ = ["InsiderLaneTrader", "LaneConfig"]
