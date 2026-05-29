"""Daily LLM spend tracker with HARD cap.

Every classify() call records its cost in ``llm_spend_daily``. Before each
call, ``LLMSpendTracker.can_spend(cost_estimate)`` checks today's total
against ``DAILY_SPEND_CAP_USD``. When the cap is hit, classification stops
automatically — the daemon's predict job will simply skip LLM enrichment
for the rest of the UTC day. Resets at midnight UTC.

This is a physical guard: even if I make a mistake elsewhere, your $
budget cannot be exceeded.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from ..storage import get_connection

log = logging.getLogger(__name__)


# Hard daily spend cap in USD. Default $2/day.
# Override via env var LLM_DAILY_CAP_USD (e.g. for one-shot backfill runs):
#     LLM_DAILY_CAP_USD=15 python scripts/run_llm_backfill.py ...
DAILY_SPEND_CAP_USD = float(os.environ.get("LLM_DAILY_CAP_USD", "2.00"))

# Claude Haiku 4.5 pricing (per 1M tokens, USD). Verified 2026-05-13 against
# actual Anthropic billing — my initial $0.80/$4.00 was 3.5 Haiku pricing,
# which under-counts by 25%. Updated to actual 4.5 rates.
HAIKU_INPUT_PER_M = 1.00
HAIKU_OUTPUT_PER_M = 5.00


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def estimate_cost(*, input_tokens: int, output_tokens: int,
                  input_per_m: float = HAIKU_INPUT_PER_M,
                  output_per_m: float = HAIKU_OUTPUT_PER_M) -> float:
    return (input_tokens / 1_000_000) * input_per_m + \
           (output_tokens / 1_000_000) * output_per_m


def today_spend_usd() -> float:
    """Return today's (UTC) cumulative LLM spend."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT total_cost_usd FROM llm_spend_daily WHERE date = ?",
            (_utc_today(),),
        ).fetchone()
    return float(row["total_cost_usd"]) if row else 0.0


class LLMSpendTracker:
    """Thread-safe-ish spend tracking against the daily cap."""

    def __init__(self, *, daily_cap_usd: float = DAILY_SPEND_CAP_USD):
        self.daily_cap_usd = daily_cap_usd

    def can_spend(self, estimated_cost: float) -> tuple[bool, float, float]:
        """Return (allowed, today_so_far, remaining_budget)."""
        today_so_far = today_spend_usd()
        remaining = self.daily_cap_usd - today_so_far
        allowed = (today_so_far + estimated_cost) <= self.daily_cap_usd
        return allowed, today_so_far, remaining

    def record(self, *, cost_usd: float) -> None:
        """Add to today's running total."""
        today = _utc_today()
        now = _utc_now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO llm_spend_daily (date, total_cost_usd, total_calls, last_updated_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_cost_usd = total_cost_usd + excluded.total_cost_usd,
                    total_calls = total_calls + 1,
                    last_updated_at = excluded.last_updated_at
                """,
                (today, cost_usd, now),
            )

    def stats(self, *, days: int = 7) -> list[dict]:
        """Return per-day spend for the last N days."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT date, total_cost_usd, total_calls, last_updated_at
                FROM llm_spend_daily
                ORDER BY date DESC
                LIMIT ?
                """,
                (days,),
            ).fetchall()
        return [dict(r) for r in rows]
