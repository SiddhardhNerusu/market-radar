"""Calendar features — FOMC / CPI / NFP / turn-of-month.

Free; hardcoded schedules (FOMC meetings 2024-2028, CPI/NFP recurring
monthly schedules). Updated yearly when the Fed publishes the next year's
meeting calendar.

For each signal date we emit:
  - ``days_until_fomc``  : signed days to next FOMC (clipped to ±30)
  - ``days_until_cpi``   : signed days to next CPI release (±15)
  - ``days_until_nfp``   : signed days to next NFP release (±15)
  - ``is_turn_of_month`` : 1 if within ±3 calendar days of month boundary
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional


# FOMC meeting dates from federalreserve.gov. Update annually as new
# years are published. Each entry is the *second* day of the 2-day meeting,
# which is when the rate decision + presser happen.
FOMC_DATES: list[date] = [
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
    date(2027, 1, 27), date(2027, 3, 17), date(2027, 4, 28), date(2027, 6, 16),
    date(2027, 7, 28), date(2027, 9, 15), date(2027, 10, 27), date(2027, 12, 8),
]


def _cpi_release_date(year: int, month: int) -> date:
    """CPI releases mid-month, typically the 10th-15th. Best approximation:
    the 2nd Wednesday of the month for January-June, the 2nd Tuesday for
    July-December. Treating it as the 12th of the month is accurate to ±3
    days, which is fine for our `±15-day-window` feature.
    """
    return date(year, month, 12)


def _nfp_release_date(year: int, month: int) -> date:
    """NFP releases the 1st Friday of the month (or first business Friday)."""
    d = date(year, month, 1)
    # weekday(): Monday=0, Friday=4
    offset = (4 - d.weekday()) % 7
    return d + timedelta(days=offset)


def _signed_days_to_nearest(target: date, events: list[date],
                            cap: int = 30) -> float:
    """Return signed days to the nearest event in ``events``. Positive
    means the event is in the future; negative means it has just passed.
    Clipped to ±cap. Returns +cap if no event within range.
    """
    best: Optional[int] = None
    for e in events:
        delta = (e - target).days
        if abs(delta) > cap:
            continue
        if best is None or abs(delta) < abs(best):
            best = delta
    if best is None:
        return float(cap)
    return float(best)


def calendar_features_for(when: date) -> dict[str, float]:
    """Compute the calendar features for one date."""
    # Build CPI / NFP candidate dates for the current + adjacent month
    candidates_cpi: list[date] = []
    candidates_nfp: list[date] = []
    for delta_month in (-1, 0, 1, 2):
        m = when.month + delta_month
        y = when.year
        while m < 1:
            m += 12
            y -= 1
        while m > 12:
            m -= 12
            y += 1
        candidates_cpi.append(_cpi_release_date(y, m))
        candidates_nfp.append(_nfp_release_date(y, m))

    # Turn-of-month: within ±3 days of last/first of month
    # Last day of current month:
    if when.month == 12:
        next_first = date(when.year + 1, 1, 1)
    else:
        next_first = date(when.year, when.month + 1, 1)
    last_of_month = next_first - timedelta(days=1)
    first_of_month = date(when.year, when.month, 1)
    is_tom = 1.0 if (
        (last_of_month - when).days <= 3 or
        (when - first_of_month).days <= 3
    ) else 0.0

    return {
        "days_until_fomc": _signed_days_to_nearest(when, FOMC_DATES, cap=30),
        "days_until_cpi":  _signed_days_to_nearest(when, candidates_cpi, cap=15),
        "days_until_nfp":  _signed_days_to_nearest(when, candidates_nfp, cap=15),
        "is_turn_of_month": is_tom,
    }


def attach_calendar_features(rows: list[dict]) -> None:
    """In-place attach the 4 calendar features."""
    for r in rows:
        ts = r.get("published_at") or r.get("scored_at") or r.get("price_at_flag_ts")
        if not ts:
            continue
        try:
            d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        cf = calendar_features_for(d)
        r.update(cf)
