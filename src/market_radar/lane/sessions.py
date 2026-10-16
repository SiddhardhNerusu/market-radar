"""US/Eastern session helpers. Pure functions, no I/O.

The event_day rule matters: a Form 4 filed at 17:30 ET is actionable at the
NEXT open, so its bet belongs to the next trading day. Anchoring on the
filing's ``published_at`` (not the batch ``scored_at`` the old research
used) is what makes the forward record honest. Exchange holidays are not
modelled here; the trader also consults the broker clock before acting.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """ISO-8601 (with Z, an offset, or naive-as-UTC) -> aware UTC datetime."""
    if not ts:
        return None
    s = str(ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def to_et(d: datetime) -> datetime:
    return d.astimezone(ET)


def next_weekday(day: date) -> date:
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def event_day_for(published_at: str) -> Optional[str]:
    """The ET trading date on which a filing published at ``published_at``
    first becomes tradeable: same day if before the close, otherwise the
    next weekday. Returns 'YYYY-MM-DD' or None if unparseable."""
    d = parse_iso(published_at)
    if d is None:
        return None
    et = to_et(d)
    day = et.date()
    if et.time() >= RTH_CLOSE or day.weekday() >= 5:
        day = next_weekday(day)
    return day.isoformat()


def et_date(now_utc: datetime) -> str:
    return to_et(now_utc).date().isoformat()


def minutes_since_open(now_utc: datetime) -> float:
    et = to_et(now_utc)
    o = datetime.combine(et.date(), RTH_OPEN, tzinfo=ET)
    return (et - o).total_seconds() / 60.0


def minutes_to_close(now_utc: datetime) -> float:
    et = to_et(now_utc)
    c = datetime.combine(et.date(), RTH_CLOSE, tzinfo=ET)
    return (c - et).total_seconds() / 60.0


def in_entry_window(now_utc: datetime, *, first: int, last: int) -> bool:
    """Regular hours, past the first ``first`` minutes, with more than ``last``
    minutes left. The broker clock is checked separately for holidays."""
    if to_et(now_utc).weekday() >= 5:
        return False
    return minutes_since_open(now_utc) >= first and minutes_to_close(now_utc) > last
