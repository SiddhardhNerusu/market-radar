"""Published_at-anchored event study for large insider open-market purchases.

Everything in this module is a PURE function on in-memory rows / bars: no DB,
no network, no files. ``scripts/event_study_form4.py`` does the I/O.

Why this exists
---------------
The 2026-06-16 edge hunt (docs/EDGE_HUNT_2026-06-16.md) found a "fragile"
Form-4 code-P tilt, but the harness it used had flaws that each inflate a
result. This module is built so that none of them can recur:

* **Anchor on ``published_at``**, the moment the filing became public — never
  on ``scored_at`` (a batch time that collapses hundreds of event days into a
  handful of fake batch days).
* **Benchmark against SPY over the identical window** (market-adjusted,
  beta=1 by default; a ``beta`` override exists). Demeaning against the bot's
  own signal universe compared insider buys to cheap catalyst names that
  drift -3%/week and flattered everything.
* **Ticker-matched placebo permutation**: each event is compared with a random
  *non-event* day of the *same ticker*. Drawing with replacement across all
  days / all tickers (what the old "within-day permutation" actually did)
  manufactures significance because it ignores per-name volatility.
* **Strict dedup**: one event per ``(ticker, event_day)``. A single Form 4 is
  one bet, never seven (the old 30-day sticky window pseudo-replicated).
* **Bonferroni in code**: ``gauntlet(..., n_families_tested=N)`` divides alpha.
* **Drop-top-k scales with n** (k = max(1, round(0.1 n))), not a fixed 5.

Limitations (stated so nobody over-reads a pass)
------------------------------------------------
* Horizon returns are entry -> close on the Nth trading day. They ignore the
  intraday path (no stop-loss simulation), so any TP/SL overlay is untestable
  here.
* With minute bars we fill at the first regular-hours minute close at/after
  ``event_ts + latency``. IEX is a thin feed: minute closes there can differ
  from the consolidated tape by a few bps, especially in illiquid names.
* Without minute bars we fall back to the event-day close, which is
  **optimistic** for same-day filings (a zero-latency fill at the official
  close that is unexecutable for filings landing in the last minutes).
* The placebo pool is drawn from the same tickers over the same calendar
  window, so it shares the regime. It controls for name-specific drift and
  volatility, not for "this whole period was a bull tape".
* The NYSE calendar here is weekends + the regular holiday rule set; ad-hoc
  closures (e.g. national days of mourning) are not modelled. Pass
  ``trading_days`` (e.g. the set of dates with SPY bars) to override.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from ..backtest.replay import _round_trip_cost_frac

ET = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Event:
    """One deduplicated insider-purchase event: all code-P open-market buys of
    ``ticker`` that became actionable on ``event_day``."""

    ticker: str
    event_ts: datetime          # UTC; the LAST filing of the group (full info set known)
    event_day: date             # US/Eastern trading date on which it is actionable
    notional: float             # sum of shares * price over the group, USD
    role_score: int             # max over the group (3=C-suite, 2=director, 1=officer/10%)
    n_filings: int = 1
    avg_price: float = 0.0      # notional-weighted purchase price (for cost tiering)
    time_known: bool = True     # False when published_at carried no clock time
    n_insiders: int = 1         # distinct reporting owners buying in the group (cluster size)


@dataclass(frozen=True)
class EventReturn:
    """One event's return at one horizon.

    ``abnormal`` is market-adjusted and PRE-cost (comparable with the placebo
    pool, which is also pre-cost). ``net`` subtracts the round-trip cost tier
    for ``price``. Statistics on edge use ``net``; the permutation test uses
    ``abnormal``.
    """

    ticker: str
    event_day: date
    horizon: int
    raw: float
    abnormal: float
    net: float
    entry_px: float = 0.0


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def parse_ts(value: str | datetime | date) -> tuple[datetime, bool]:
    """Parse ``published_at`` (or a bar 't') to an aware UTC datetime.

    Returns ``(ts, time_known)``. Naive datetimes are assumed UTC (that is
    how the ingestors store them). A bare ``YYYY-MM-DD`` yields
    ``time_known=False`` and the timestamp 16:00 ET of that date, i.e. the
    conservative "we only learned about it at the close" reading.
    """
    if isinstance(value, datetime):
        ts = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc), True
    if isinstance(value, date):
        d = value
        return datetime.combine(d, REGULAR_CLOSE, tzinfo=ET).astimezone(timezone.utc), False
    s = str(value).strip()
    if len(s) == 10:
        d = date.fromisoformat(s)
        return datetime.combine(d, REGULAR_CLOSE, tzinfo=ET).astimezone(timezone.utc), False
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = s.replace(" ", "T", 1)
    # Trim sub-second precision beyond 6 digits (RFC3339 nanos from Alpaca).
    if "." in s:
        head, _, tail = s.partition(".")
        digits = ""
        rest = ""
        for i, ch in enumerate(tail):
            if ch.isdigit():
                digits += ch
            else:
                rest = tail[i:]
                break
        s = f"{head}.{digits[:6]}{rest}" if digits else f"{head}{rest}"
    ts = datetime.fromisoformat(s)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc), True


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> Optional[date]:
    if d.weekday() == 5:
        obs = d - timedelta(days=1)
        # NYSE rule: a Saturday Jan 1 is NOT observed on Dec 31.
        return None if obs.month == 12 and obs.day == 31 else obs
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[date]:
    """Regular NYSE holiday rules (no ad-hoc closures)."""
    fixed = [date(year, 1, 1), date(year, 7, 4), date(year, 12, 25)]
    if year >= 2022:
        fixed.append(date(year, 6, 19))
    out = {o for o in (_observed(d) for d in fixed) if o is not None}
    out.add(_nth_weekday(year, 1, 0, 3))      # MLK
    out.add(_nth_weekday(year, 2, 0, 3))      # Presidents
    out.add(_easter(year) - timedelta(days=2))  # Good Friday
    out.add(_last_weekday(year, 5, 0))        # Memorial
    out.add(_nth_weekday(year, 9, 0, 1))      # Labor
    out.add(_nth_weekday(year, 11, 3, 4))     # Thanksgiving
    return out


def is_trading_day(d: date, trading_days: Optional[set[date]] = None) -> bool:
    if trading_days is not None:
        return d in trading_days
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def next_trading_day(d: date, trading_days: Optional[set[date]] = None) -> date:
    """First trading day strictly after ``d``."""
    cur = d + timedelta(days=1)
    for _ in range(366):
        if is_trading_day(cur, trading_days):
            return cur
        cur += timedelta(days=1)
    raise ValueError(f"no trading day within a year of {d}")


def actionable_day(ts_utc: datetime, trading_days: Optional[set[date]] = None) -> date:
    """Trading date on which information published at ``ts_utc`` can be
    traded: same ET date if published before 16:00 ET on a trading day, else
    the next trading day. Pre-market filings are same-day (entry at 9:30)."""
    local = ts_utc.astimezone(ET)
    d = local.date()
    if local.time() >= REGULAR_CLOSE or not is_trading_day(d, trading_days):
        return next_trading_day(d, trading_days)
    return d


def bar_day(bar: Mapping) -> date:
    """ET calendar date of a bar. Accepts 't' as RFC3339 string, bare date
    string, datetime or date."""
    t = bar["t"]
    if isinstance(t, datetime):
        return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(ET).date()
    if isinstance(t, date):
        return t
    s = str(t)
    if len(s) == 10:
        return date.fromisoformat(s)
    return parse_ts(s)[0].astimezone(ET).date()


def trading_days_from_bars(bars: Iterable[Mapping]) -> set[date]:
    return {bar_day(b) for b in bars}


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------
def _truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(v)


def select_insider_events(
    rows: Iterable[Mapping],
    *,
    min_notional_usd: float = 25_000,
    min_price: float = 10.0,
    exclude_10b5_1: bool = True,
    senior_only: bool = False,
    min_insiders: int = 1,
    trading_days: Optional[set[date]] = None,
) -> list[Event]:
    """Turn insider_transactions rows into strictly deduplicated events.

    Input rows need: ticker, transaction_code, is_acquired, shares, price,
    published_at, role_score, is_10b5_1 (may be missing/None in old DBs; a
    missing flag is treated as "unknown" and the row is kept).

    Rules
    -----
    * Keep only code ``P`` AND acquired, with price >= ``min_price`` and
      positive shares (zero-price/zero-share rows are parser failures).
    * ``exclude_10b5_1``: drop rows flagged as plan trades (routine).
    * ``senior_only``: keep role_score >= 2 (director or C-suite).
    * Each transaction gets its own ``actionable_day`` from ``published_at``;
      rows are then grouped by ``(ticker, event_day)``: notional summed,
      role_score max'd, ``event_ts`` = the LAST filing in the group so that
      the aggregated notional was knowable at entry (no look-ahead through
      a later same-day filing).
    * ``min_notional_usd`` is applied to the AGGREGATED notional, so three
      $10k buys on one day form one $30k event. One event per
      (ticker, event_day) — never more.
    """
    groups: dict[tuple[str, date], dict] = {}
    for r in rows:
        ticker = (r.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        if (r.get("transaction_code") or "").strip().upper() != "P":
            continue
        if not _truthy(r.get("is_acquired")):
            continue
        try:
            shares = float(r.get("shares") or 0)
            price = float(r.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if shares <= 0 or price <= 0 or price < min_price:
            continue
        if exclude_10b5_1 and _truthy(r.get("is_10b5_1")):
            continue
        role = int(r.get("role_score") or 0)
        if senior_only and role < 2:
            continue
        published = r.get("published_at")
        if not published:
            continue
        try:
            ts, time_known = parse_ts(published)
        except ValueError:
            continue
        day = actionable_day(ts, trading_days)
        g = groups.setdefault((ticker, day), {
            "ts": ts, "notional": 0.0, "shares": 0.0, "role": 0, "n": 0,
            "time_known": True, "insiders": set(),
        })
        g["insiders"].add((r.get("insider_name") or f"sig{r.get('signal_id')}").strip().upper())
        g["ts"] = max(g["ts"], ts)
        g["notional"] += shares * price
        g["shares"] += shares
        g["role"] = max(g["role"], role)
        g["n"] += 1
        g["time_known"] = g["time_known"] and time_known

    events: list[Event] = []
    for (ticker, day), g in groups.items():
        if g["notional"] < min_notional_usd:
            continue
        if len(g["insiders"]) < min_insiders:
            continue
        events.append(Event(
            ticker=ticker,
            event_ts=g["ts"],
            event_day=day,
            notional=g["notional"],
            role_score=g["role"],
            n_filings=g["n"],
            avg_price=g["notional"] / g["shares"] if g["shares"] else 0.0,
            time_known=g["time_known"],
            n_insiders=len(g["insiders"]),
        ))
    events.sort(key=lambda e: (e.event_day, e.ticker))
    return events


# ---------------------------------------------------------------------------
# Entry and forward returns
# ---------------------------------------------------------------------------
def entry_price_for(
    event: Event,
    bars_minute: Optional[Sequence[Mapping]],
    bars_daily: Sequence[Mapping],
    latency_minutes: int = 2,
) -> Optional[tuple[datetime, float]]:
    """Entry timestamp and price for ``event``.

    With minute bars: the close of the first regular-hours (09:30-16:00 ET)
    minute bar on ``event_day`` whose start time is >= ``event_ts +
    latency_minutes`` (or >= 09:30 for pre-market / previous-evening
    filings). Alpaca bar timestamps are bar *starts*, so the fill is the
    close of a minute that began after we could have reacted.

    Without minute bars (None/empty): the ``event_day`` daily close. This is
    OPTIMISTIC for same-day filings: it assumes a zero-latency fill at the
    official close, which is unexecutable when the filing lands in the last
    minutes of the session, and it ignores the drift between the filing and
    the close in either direction. Prefer minute bars.
    """
    day = event.event_day
    if bars_minute:
        target = event.event_ts + timedelta(minutes=latency_minutes)
        open_ts = datetime.combine(day, REGULAR_OPEN, tzinfo=ET).astimezone(timezone.utc)
        close_ts = datetime.combine(day, REGULAR_CLOSE, tzinfo=ET).astimezone(timezone.utc)
        target = max(target, open_ts)
        candidates = []
        for b in bars_minute:
            ts = parse_ts(b["t"])[0]
            if ts.astimezone(ET).date() != day:
                continue
            if ts < open_ts or ts >= close_ts:
                continue
            if ts >= target:
                candidates.append((ts, float(b["c"])))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0]
        # Minute bars were provided but none at/after target during regular
        # hours on event_day (filing in the final minutes, or a data gap).
        # Return None rather than silently falling back to the optimistic
        # close rule; the caller decides whether to drop the event.
        return None
    for b in bars_daily:
        if bar_day(b) == day:
            entry_ts = datetime.combine(day, REGULAR_CLOSE, tzinfo=ET).astimezone(timezone.utc)
            return entry_ts, float(b["c"])
    return None


def forward_returns(
    entry_px: float,
    entry_ts: datetime,
    bars_daily: Sequence[Mapping],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[int, Optional[float]]:
    """Simple % return from ``entry_px`` to the close of the Nth trading day
    AFTER the entry's ET date, for each N in ``horizons``.

    "Trading days after" are counted on the ticker's own bar series: the Nth
    bar whose date is strictly later than the entry date. Missing horizons
    (series too short) map to ``None``. Close-to-close at the horizon end;
    the intraday path in between is ignored.
    """
    if entry_px <= 0:
        return {h: None for h in horizons}
    entry_day = entry_ts.astimezone(ET).date()
    future = sorted(
        ((bar_day(b), float(b["c"])) for b in bars_daily if bar_day(b) > entry_day),
        key=lambda x: x[0],
    )
    out: dict[int, Optional[float]] = {}
    for h in horizons:
        if h <= 0 or h > len(future):
            out[h] = None
        else:
            out[h] = (future[h - 1][1] / entry_px - 1.0) * 100.0
    return out


def abnormal_returns(
    raw_returns: Mapping[int, Optional[float]],
    bench_returns: Mapping[int, Optional[float]],
    beta: float = 1.0,
) -> dict[int, Optional[float]]:
    """Market-adjusted return: raw minus ``beta`` x benchmark over the same
    window. beta=1 (default) is the plain market-adjusted model; pass an
    estimated beta for a market-model adjustment. Missing on either side ->
    None."""
    out: dict[int, Optional[float]] = {}
    for h, r in raw_returns.items():
        b = bench_returns.get(h)
        out[h] = None if (r is None or b is None) else r - beta * b
    return out


def net_of_cost(abnormal_pct: float, price: float) -> float:
    """Abnormal return (%) minus the round-trip cost tier for ``price`` from
    ``market_radar.backtest.replay._round_trip_cost_frac`` (20 bps >= $50 ...
    400 bps sub-$1), expressed in %."""
    return abnormal_pct - 100.0 * _round_trip_cost_frac(price)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def drop_top_k_for(n: int) -> int:
    """k = max(1, round(0.1 n)). Scales with the sample instead of a fixed 5
    (which is 16% of n=32 but 1% of n=500)."""
    return max(1, int(round(0.1 * n))) if n > 0 else 0


def summarize(values: Sequence[float]) -> dict:
    """Robust summary. ``trim10_mean`` drops floor(0.1 n) from EACH tail;
    ``drop_top_k_mean`` drops the k largest values (k from
    ``drop_top_k_for``) and is the outlier-dependence check. ``t_stat`` is
    mean / (sd / sqrt n) with ddof=1 (NaN for n < 2)."""
    x = np.asarray([float(v) for v in values], dtype=float)
    n = int(x.size)
    nan = float("nan")
    if n == 0:
        return {"n": 0, "mean": nan, "median": nan, "std": nan, "trim10_mean": nan,
                "win_rate": nan, "drop_top_k": 0, "drop_top_k_mean": nan, "t_stat": nan}
    xs = np.sort(x)
    cut = int(math.floor(0.1 * n))
    trimmed = xs[cut:n - cut] if n - 2 * cut > 0 else xs
    k = drop_top_k_for(n)
    dropped = xs[: n - k]
    std = float(np.std(x, ddof=1)) if n > 1 else nan
    t = float(x.mean() / (std / math.sqrt(n))) if n > 1 and std > 0 else nan
    return {
        "n": n,
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": std,
        "trim10_mean": float(trimmed.mean()),
        "win_rate": float((x > 0).mean()),
        "drop_top_k": k,
        "drop_top_k_mean": float(dropped.mean()) if dropped.size else nan,
        "t_stat": t,
    }


def early_late_split(
    events_with_returns: Sequence[EventReturn],
) -> tuple[list[EventReturn], list[EventReturn]]:
    """Split on DISTINCT event days: the earliest ~70% of days are "early",
    the rest "late". Splitting on rows would let one busy day straddle the
    boundary or dominate one side."""
    days = sorted({er.event_day for er in events_with_returns})
    n_days = len(days)
    if n_days < 2:
        return list(events_with_returns), []
    n_early = min(max(1, int(round(0.7 * n_days))), n_days - 1)
    early_days = set(days[:n_early])
    early = [er for er in events_with_returns if er.event_day in early_days]
    late = [er for er in events_with_returns if er.event_day not in early_days]
    return early, late


def build_placebo_pool(
    ticker_bars: Mapping[str, Sequence[Mapping]],
    bench_bars: Sequence[Mapping],
    horizon: int,
    event_days: Mapping[str, Iterable[date]],
    *,
    beta: float = 1.0,
    window_days: Optional[int] = None,
) -> dict[str, list[tuple[date, float]]]:
    """Pre-cost abnormal close-to-close ``horizon``-day returns for every
    NON-event day of each ticker: the ticker-matched placebo pool.

    A day ``d`` is excluded if its forward window overlaps any event window
    of that ticker (|idx(d) - idx(event_day)| <= horizon). If ``window_days``
    is given, only days within that many trading days of the ticker's
    first/last event are used (keeps the pool in the same regime and bounds
    the data pull). Placebo entries are at the close of ``d``; events enter
    intraday, so the placebo window is a close-to-close approximation of
    the event window."""
    bench = {bar_day(b): float(b["c"]) for b in bench_bars}
    pool: dict[str, list[tuple[date, float]]] = {}
    for ticker, bars in ticker_bars.items():
        series = sorted(((bar_day(b), float(b["c"])) for b in bars), key=lambda x: x[0])
        if len(series) <= horizon:
            continue
        days = [d for d, _ in series]
        idx_of = {d: i for i, d in enumerate(days)}
        ev_idx = sorted(idx_of[d] for d in event_days.get(ticker, ()) if d in idx_of)
        lo, hi = 0, len(series) - horizon
        if window_days is not None and ev_idx:
            lo = max(0, ev_idx[0] - window_days)
            hi = min(hi, ev_idx[-1] + window_days + 1)
        out: list[tuple[date, float]] = []
        for i in range(lo, hi):
            if any(abs(i - e) <= horizon for e in ev_idx):
                continue
            d0, c0 = series[i]
            d1, c1 = series[i + horizon]
            if d0 not in bench or d1 not in bench or c0 <= 0 or bench[d0] <= 0:
                continue
            raw = (c1 / c0 - 1.0) * 100.0
            b = (bench[d1] / bench[d0] - 1.0) * 100.0
            out.append((d0, raw - beta * b))
        if out:
            pool[ticker] = out
    return pool


def permutation_pvalue(
    event_returns: Sequence[EventReturn],
    placebo_pool: Mapping[str, Sequence[tuple[date, float]]],
    n_iter: int = 5000,
    seed: int = 0,
) -> dict:
    """Ticker-matched placebo test.

    For each iteration, every event is replaced by a random NON-event day of
    the SAME ticker drawn from ``placebo_pool`` (the event's own day is
    excluded even if present). The statistic is the mean pre-cost abnormal
    return across the pseudo-events; ``p_value`` is the fraction of
    iterations whose mean is >= the observed mean (with the +1 correction).

    Why ticker-matched: the null question is "would buying this *name* on a
    random day have done as well?". Drawing across all tickers and all days
    (the old harness) mixes in low-vol large caps and high-vol microcaps, so
    the null's spread reflects the universe mix instead of the events'
    names, and a few volatile winners look significant. Matching on ticker
    holds name-specific volatility and drift fixed.

    Events whose ticker has no placebo entries are dropped from BOTH the
    observed mean and the null (``n_excluded`` reports how many). The pool
    shares the calendar window with the events, so this does not control
    for the whole period's regime.
    """
    rng = np.random.default_rng(seed)
    obs: list[float] = []
    pools: list[np.ndarray] = []
    excluded = 0
    for er in event_returns:
        entries = [v for (d, v) in placebo_pool.get(er.ticker, ()) if d != er.event_day]
        if not entries:
            excluded += 1
            continue
        obs.append(er.abnormal)
        pools.append(np.asarray(entries, dtype=float))
    nan = float("nan")
    if not obs:
        return {"observed_mean": nan, "null_mean": nan, "null_sd": nan, "p_value": nan,
                "n_events": 0, "n_excluded": excluded, "n_iter": n_iter}
    observed = float(np.mean(obs))
    sims = np.zeros(n_iter, dtype=float)
    for pool in pools:
        sims += pool[rng.integers(0, pool.size, size=n_iter)]
    sims /= len(pools)
    hits = int(np.sum(sims >= observed))
    return {
        "observed_mean": observed,
        "null_mean": float(sims.mean()),
        "null_sd": float(sims.std(ddof=1)) if n_iter > 1 else nan,
        "p_value": (hits + 1) / (n_iter + 1),
        "n_events": len(obs),
        "n_excluded": excluded,
        "n_iter": n_iter,
    }


def cluster_bootstrap_ci(
    event_returns: Sequence[EventReturn],
    n_iter: int = 5000,
    seed: int = 0,
    value: str = "net",
) -> dict:
    """Bootstrap the mean by resampling event DAYS (clusters) with
    replacement. Events on the same day share a market shock, so an i.i.d.
    bootstrap over events understates the variance. Returns the 2.5 / 97.5
    percentiles of the resampled mean of ``value`` (``net`` by default)."""
    rng = np.random.default_rng(seed)
    by_day: dict[date, list[float]] = {}
    for er in event_returns:
        by_day.setdefault(er.event_day, []).append(float(getattr(er, value)))
    clusters = list(by_day.values())
    nan = float("nan")
    if not clusters:
        return {"mean": nan, "ci_low": nan, "ci_high": nan, "n_clusters": 0, "n_iter": n_iter}
    sums = np.asarray([sum(c) for c in clusters], dtype=float)
    counts = np.asarray([len(c) for c in clusters], dtype=float)
    k = len(clusters)
    idx = rng.integers(0, k, size=(n_iter, k))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    all_vals = [v for c in clusters for v in c]
    return {
        "mean": float(np.mean(all_vals)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n_clusters": k,
        "n_iter": n_iter,
    }


def gauntlet(
    event_returns: Sequence[EventReturn],
    placebo_pool: Mapping[str, Sequence[tuple[date, float]]],
    n_families_tested: int = 1,
    *,
    min_days: int = 15,
    alpha: float = 0.05,
    n_iter: int = 5000,
    seed: int = 0,
) -> dict:
    """Promotion gate from docs/EDGE_HUNT_2026-06-16.md, with every rule in
    code. All criteria are on NET-of-cost market-adjusted returns except the
    permutation test (pre-cost, against the pre-cost placebo pool).

    Criteria (all must pass):
      n_days >= min_days; median > 0; trim10 > 0; drop_top_k > 0;
      early > 0 AND late > 0; permutation p < alpha / n_families_tested.

    ``n_families_tested`` is the Bonferroni divisor: the number of signal
    families / cuts looked at before settling on this one. Testing the
    insider cut, the senior cut and the >=$100k cut and reporting the best
    is three families, so pass 3.
    """
    n_families_tested = max(1, int(n_families_tested))
    alpha_adj = alpha / n_families_tested
    nets = [er.net for er in event_returns]
    stats = summarize(nets)
    early, late = early_late_split(event_returns)
    early_mean = float(np.mean([e.net for e in early])) if early else float("nan")
    late_mean = float(np.mean([e.net for e in late])) if late else float("nan")
    perm = permutation_pvalue(event_returns, placebo_pool, n_iter=n_iter, seed=seed)
    boot = cluster_bootstrap_ci(event_returns, n_iter=n_iter, seed=seed)
    n_days = len({er.event_day for er in event_returns})

    def _gt0(v: float) -> bool:
        return bool(v == v) and v > 0  # NaN-safe

    criteria = {
        "min_days": {"pass": n_days >= min_days, "value": n_days, "threshold": min_days},
        "median_positive": {"pass": _gt0(stats["median"]), "value": stats["median"], "threshold": 0.0},
        "trim10_positive": {"pass": _gt0(stats["trim10_mean"]), "value": stats["trim10_mean"], "threshold": 0.0},
        "drop_top_k_positive": {"pass": _gt0(stats["drop_top_k_mean"]), "value": stats["drop_top_k_mean"],
                                "threshold": 0.0, "k": stats["drop_top_k"]},
        "early_positive": {"pass": _gt0(early_mean), "value": early_mean, "threshold": 0.0,
                           "n": len(early)},
        "late_positive": {"pass": _gt0(late_mean), "value": late_mean, "threshold": 0.0,
                          "n": len(late)},
        "permutation_significant": {
            "pass": bool(perm["p_value"] == perm["p_value"]) and perm["p_value"] < alpha_adj,
            "value": perm["p_value"], "threshold": alpha_adj,
        },
    }
    return {
        "passed": all(c["pass"] for c in criteria.values()),
        "underpowered": n_days < min_days,
        "n_events": len(event_returns),
        "n_days": n_days,
        "alpha": alpha,
        "n_families_tested": n_families_tested,
        "alpha_bonferroni": alpha_adj,
        "criteria": criteria,
        "summary": stats,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "permutation": perm,
        "bootstrap": boot,
    }
