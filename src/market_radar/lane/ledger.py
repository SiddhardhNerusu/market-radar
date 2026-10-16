"""edge_ledger: the forward out-of-sample record, and its resolution.

Every candidate that QUALIFIES is written here at decision time with the
live ask (an actionable price) and SPY at the same instant, traded or not.
``nth_close_after`` and ``resolve_row`` are pure; ``resolve_ledger`` does the
I/O with an injectable bars function so it is testable without a broker.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Callable, Optional

from ..backtest.replay import _round_trip_cost_frac
from .candidates import InsiderCandidate

HORIZONS = (5, 20)


def _now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_candidate(conn: sqlite3.Connection, *, candidate: str, cand: InsiderCandidate,
                     price_at_flag: Optional[float], bench_at_flag: Optional[float],
                     traded: bool, now: Optional[datetime] = None) -> None:
    """Append-only insert; a duplicate (candidate, ticker, event_day) is ignored
    except that ``traded`` may be upgraded 0 -> 1."""
    ts = _now_iso(now)
    conn.execute(
        """INSERT INTO edge_ledger
             (candidate, ticker, event_day, published_at, notional_usd, role_score,
              price_at_flag, price_at_flag_ts, bench_at_flag, cost_frac, traded, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(candidate, ticker, event_day) DO UPDATE SET
             traded = MAX(edge_ledger.traded, excluded.traded)""",
        (candidate, cand.ticker, cand.event_day, cand.published_at, cand.notional_usd,
         cand.role_score, price_at_flag, ts if price_at_flag else None, bench_at_flag,
         _round_trip_cost_frac(price_at_flag or 0.0), 1 if traded else 0, ts),
    )


def nth_close_after(bars: list[dict], event_day: str, n: int) -> Optional[tuple[str, float]]:
    """Close of the n-th trading bar strictly AFTER ``event_day`` (bars are
    dicts with 't' RFC3339 and 'c'). Returns (day, close) or None if not yet
    available. Day 0 is the event day itself; n=5 is the fifth session after."""
    days = sorted({(str(b["t"])[:10], float(b["c"])) for b in bars if b.get("c") is not None})
    after = [(d, c) for (d, c) in days if d > event_day]
    if len(after) < n:
        return None
    return after[n - 1]


def resolve_row(row: dict, bars: list[dict], bench_bars: list[dict]) -> dict:
    """Pure: given a ledger row and daily bars for the ticker and the
    benchmark, return the columns to update (only those now resolvable)."""
    upd: dict = {}
    p0 = row.get("price_at_flag")
    b0 = row.get("bench_at_flag")
    if not p0 or p0 <= 0:
        return upd
    for n in HORIZONS:
        if row.get(f"ret_{n}d_pct") is not None:
            continue
        hit = nth_close_after(bars, row["event_day"], n)
        if hit is None:
            continue
        _d, close = hit
        ret = (close / p0 - 1.0) * 100.0
        upd[f"price_{n}d"] = close
        upd[f"ret_{n}d_pct"] = round(ret, 4)
        bh = nth_close_after(bench_bars, row["event_day"], n) if b0 else None
        if bh is not None and b0:
            bret = (bh[1] / b0 - 1.0) * 100.0
            upd[f"bench_{n}d"] = bh[1]
            upd[f"abn_{n}d_pct"] = round(ret - bret, 4)
    return upd


def resolve_ledger(conn: sqlite3.Connection, *, bars_fn: Callable[[str, str, str], list[dict]],
                   today: str, benchmark: str = "SPY") -> int:
    """Fill ret/abn columns for unresolved rows whose horizon has elapsed.
    ``bars_fn(symbol, start_day, end_day)`` returns daily bars. Returns the
    number of rows updated."""
    rows = [dict(r) for r in conn.execute(
        """SELECT * FROM edge_ledger
           WHERE price_at_flag IS NOT NULL AND (ret_5d_pct IS NULL OR ret_20d_pct IS NULL)
             AND event_day < ?""", (today,)).fetchall()]
    if not rows:
        return 0
    bench_cache: dict[str, list[dict]] = {}
    n_upd = 0
    for r in rows:
        try:
            bars = bars_fn(r["ticker"], r["event_day"], today)
        except Exception:  # noqa: BLE001 — one bad symbol never blocks the rest
            continue
        key = r["event_day"]
        if key not in bench_cache:
            try:
                bench_cache[key] = bars_fn(benchmark, key, today)
            except Exception:  # noqa: BLE001
                bench_cache[key] = []
        upd = resolve_row(r, bars, bench_cache[key])
        if not upd:
            continue
        sets = ", ".join(f"{k} = ?" for k in upd)
        vals = list(upd.values())
        fully = (r.get("ret_20d_pct") is not None) or ("ret_20d_pct" in upd)
        conn.execute(
            f"UPDATE edge_ledger SET {sets}, resolved_at = ? WHERE id = ?",
            vals + [_now_iso() if fully else None, r["id"]],
        )
        n_upd += 1
    return n_upd
