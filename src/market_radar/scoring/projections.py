"""Comparable-setup projections.

For a fresh signal, look up *historical* signals of similar class with
resolved outcomes. Return the empirical distribution of 1d/5d/20d returns:
median, P25/P75, P10/P90, hit rate, worst case.

This is the honest version of the "projection" feature the user asked for.
We never invent numbers — we report what actually happened to similar
setups in the past. The user gets:

    Past 87 comparables (M&A announcement on megacap):
      5-day return: median +1.3% (P25 -1.2%, P75 +4.8%, worst -8.7%)
      Hit rate (>0):  62%

…and can map that distribution to a price range from the current price.
"""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReturnDistribution:
    window_days: int
    samples: int
    median: Optional[float]
    p25: Optional[float]
    p75: Optional[float]
    p10: Optional[float]
    p90: Optional[float]
    worst: Optional[float]
    best: Optional[float]
    hit_rate: Optional[float]   # fraction with return > 0


@dataclass
class ProjectionResult:
    signal_class: str
    used_relaxed: bool          # True if we fell back to event_type-only match
    one_day: ReturnDistribution
    five_day: ReturnDistribution
    twenty_day: ReturnDistribution


def _pct(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _distribution(returns: list[float], window_days: int) -> ReturnDistribution:
    clean = [r for r in returns if r is not None]
    if not clean:
        return ReturnDistribution(
            window_days=window_days, samples=0,
            median=None, p25=None, p75=None, p10=None, p90=None,
            worst=None, best=None, hit_rate=None,
        )
    return ReturnDistribution(
        window_days=window_days,
        samples=len(clean),
        median=statistics.median(clean),
        p25=_pct(clean, 0.25),
        p75=_pct(clean, 0.75),
        p10=_pct(clean, 0.10),
        p90=_pct(clean, 0.90),
        worst=min(clean),
        best=max(clean),
        hit_rate=sum(1 for r in clean if r > 0) / len(clean),
    )


def project_for_class(
    conn: sqlite3.Connection,
    signal_class: str,
    *,
    min_exact_samples: int = 30,
) -> Optional[ProjectionResult]:
    """Build a return-distribution projection from historical comparables.

    Returns None if no comparable data exists at any level of relaxation.
    """
    if not signal_class:
        return None

    # 1. Exact-class match
    rows = conn.execute(
        """
        SELECT so.return_1d_pct, so.return_5d_pct, so.return_20d_pct
        FROM signal_scores ss
        JOIN signal_outcomes so ON so.score_id = ss.id
        WHERE ss.signal_class = ?
          AND so.return_5d_pct IS NOT NULL
        """,
        (signal_class,),
    ).fetchall()

    used_relaxed = False
    if len(rows) < min_exact_samples:
        # 2. Relaxed: event_type + sentiment_dir only
        parts = signal_class.split("|")
        if len(parts) >= 3:
            event_type = parts[1]
            sentiment_dir = parts[2]
            rows = conn.execute(
                """
                SELECT so.return_1d_pct, so.return_5d_pct, so.return_20d_pct
                FROM signal_scores ss
                JOIN signal_outcomes so ON so.score_id = ss.id
                WHERE ss.event_type = ?
                  AND (
                      (? = 'bullish' AND ss.sentiment > 0.2) OR
                      (? = 'bearish' AND ss.sentiment < -0.2) OR
                      (? = 'neutral' AND ss.sentiment BETWEEN -0.2 AND 0.2)
                  )
                  AND so.return_5d_pct IS NOT NULL
                """,
                (event_type, sentiment_dir, sentiment_dir, sentiment_dir),
            ).fetchall()
            used_relaxed = True

    if not rows:
        return None

    one_d = _distribution([r["return_1d_pct"] for r in rows], 1)
    five_d = _distribution([r["return_5d_pct"] for r in rows], 5)
    twenty_d = _distribution([r["return_20d_pct"] for r in rows], 20)

    return ProjectionResult(
        signal_class=signal_class,
        used_relaxed=used_relaxed,
        one_day=one_d,
        five_day=five_d,
        twenty_day=twenty_d,
    )


def projected_price_range(
    current_price: float,
    dist: ReturnDistribution,
) -> dict[str, Optional[float]]:
    """Convert a return distribution into projected price levels."""
    def _to_price(pct: Optional[float]) -> Optional[float]:
        if pct is None:
            return None
        return current_price * (1.0 + pct / 100.0)

    return {
        "window_days": dist.window_days,
        "samples": dist.samples,
        "median_price": _to_price(dist.median),
        "p25_price": _to_price(dist.p25),
        "p75_price": _to_price(dist.p75),
        "p10_price": _to_price(dist.p10),
        "p90_price": _to_price(dist.p90),
        "worst_price": _to_price(dist.worst),
        "best_price": _to_price(dist.best),
        "median_return_pct": dist.median,
        "hit_rate": dist.hit_rate,
    }
