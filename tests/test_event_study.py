"""Synthetic-data tests for market_radar.research.event_study.

No network, no DB. Each test locks one of the methodological fixes listed in
the module docstring so the old mistakes (scored_at anchoring, sticky-window
pseudo-replication, fixed drop-5, Bonferroni-in-prose, cross-ticker
permutation) cannot silently return.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.research.event_study import (
    ET,
    Event,
    EventReturn,
    abnormal_returns,
    actionable_day,
    build_placebo_pool,
    cluster_bootstrap_ci,
    drop_top_k_for,
    early_late_split,
    entry_price_for,
    forward_returns,
    gauntlet,
    net_of_cost,
    next_trading_day,
    parse_ts,
    permutation_pvalue,
    select_insider_events,
    summarize,
)

# 2026-05-12 is a Tuesday; 2026-05-15 Friday; 2026-05-25 Memorial Day.
TUE = date(2026, 5, 12)


def _row(ticker="ACME", code="P", acquired=1, shares=1000, price=30.0,
         published="2026-05-12T14:30:00Z", role=1, plan=0):
    return {"ticker": ticker, "transaction_code": code, "is_acquired": acquired,
            "shares": shares, "price": price, "published_at": published,
            "role_score": role, "is_10b5_1": plan}


def _daily(ticker_start: date, closes: list[float]) -> list[dict]:
    """Daily bars on consecutive weekdays starting at ticker_start."""
    bars, d = [], ticker_start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        bars.append({"t": f"{d.isoformat()}T04:00:00Z", "o": c, "h": c, "l": c, "c": c, "v": 1})
        d += timedelta(days=1)
    return bars


# ---------------------------------------------------------------- selection
def test_dedup_two_same_day_purchases_become_one_event_with_summed_notional():
    rows = [
        _row(shares=1000, price=30.0, published="2026-05-12T14:30:00Z", role=1),
        _row(shares=500, price=32.0, published="2026-05-12T15:10:00Z", role=3),
    ]
    events = select_insider_events(rows, min_notional_usd=25_000)
    assert len(events) == 1
    ev = events[0]
    assert ev.ticker == "ACME"
    assert ev.event_day == TUE
    assert ev.notional == pytest.approx(1000 * 30.0 + 500 * 32.0)
    assert ev.role_score == 3
    assert ev.n_filings == 2
    # event_ts is the LAST filing so the aggregated notional was knowable.
    assert ev.event_ts == datetime(2026, 5, 12, 15, 10, tzinfo=timezone.utc)


def test_after_close_filing_rolls_to_next_trading_day():
    # 20:30Z = 16:30 ET on Tue -> Wed. Friday 21:00Z = 17:00 ET -> Monday.
    tue_evening = select_insider_events([_row(published="2026-05-12T20:30:00Z")])
    assert tue_evening[0].event_day == date(2026, 5, 13)
    fri_evening = select_insider_events([_row(published="2026-05-15T21:00:00Z")])
    assert fri_evening[0].event_day == date(2026, 5, 18)
    # Friday 2026-05-22 after close -> Memorial Day skipped -> Tue 05-26.
    pre_holiday = select_insider_events([_row(published="2026-05-22T21:00:00Z")])
    assert pre_holiday[0].event_day == date(2026, 5, 26)
    # Pre-market filing (12:00Z = 08:00 ET) is actionable the same day.
    premarket = select_insider_events([_row(published="2026-05-12T12:00:00Z")])
    assert premarket[0].event_day == TUE
    # Two filings straddling the close are two events (different event_day).
    both = select_insider_events([_row(published="2026-05-12T14:30:00Z"),
                                  _row(published="2026-05-12T20:30:00Z")])
    assert [e.event_day for e in both] == [TUE, date(2026, 5, 13)]


def test_date_only_published_at_is_treated_as_after_close():
    ev = select_insider_events([_row(published="2026-05-12")])
    assert ev[0].event_day == date(2026, 5, 13)
    assert ev[0].time_known is False


def test_selection_filters():
    base = [
        _row(code="S", acquired=0),                       # sale
        _row(code="P", acquired=0),                       # P but disposed (parser edge)
        _row(price=5.0),                                  # below min_price
        _row(shares=100, price=30.0),                     # $3k < min notional
        _row(plan=1, ticker="PLAN"),                      # 10b5-1
        _row(ticker="JUNIOR", role=1, shares=5000),       # junior insider
        _row(ticker="SENIOR", role=2, shares=5000),
    ]
    tickers = {e.ticker for e in select_insider_events(base)}
    assert tickers == {"JUNIOR", "SENIOR"}
    assert {e.ticker for e in select_insider_events(base, exclude_10b5_1=False)} == {"JUNIOR", "SENIOR", "PLAN"}
    assert {e.ticker for e in select_insider_events(base, senior_only=True)} == {"SENIOR"}
    # Missing is_10b5_1 column (old DB) is treated as unknown -> kept.
    r = _row(ticker="OLD", shares=5000)
    del r["is_10b5_1"]
    assert [e.ticker for e in select_insider_events([r])] == ["OLD"]


def test_trading_days_override_is_respected():
    # Pretend Wed 05-13 is closed: a Tue-evening filing goes to Thu.
    cal = {date(2026, 5, 12), date(2026, 5, 14), date(2026, 5, 15)}
    ts = parse_ts("2026-05-12T20:30:00Z")[0]
    assert actionable_day(ts, cal) == date(2026, 5, 14)
    assert next_trading_day(date(2026, 5, 12), cal) == date(2026, 5, 14)


# ---------------------------------------------------------------- entry
def _minute_bars(day: date, start_hm=(9, 30), n=390, px0=100.0):
    bars = []
    t0 = datetime.combine(day, datetime.min.time(), tzinfo=ET).replace(hour=start_hm[0], minute=start_hm[1])
    for i in range(n):
        ts = (t0 + timedelta(minutes=i)).astimezone(timezone.utc)
        bars.append({"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "c": px0 + i * 0.01})
    return bars


def test_entry_uses_first_minute_close_after_latency():
    ev = Event(ticker="ACME", event_ts=parse_ts("2026-05-12T14:30:00Z")[0],  # 10:30 ET
               event_day=TUE, notional=1e5, role_score=1)
    bars = _minute_bars(TUE)
    entry = entry_price_for(ev, bars, [], latency_minutes=2)
    assert entry is not None
    entry_ts, entry_px = entry
    assert entry_ts.astimezone(ET).strftime("%H:%M") == "10:32"
    assert entry_px == pytest.approx(100.0 + 62 * 0.01)   # 62 minutes after 09:30
    # Pre-market event clamps to the 09:30 bar.
    ev_pre = Event(ticker="ACME", event_ts=parse_ts("2026-05-12T11:00:00Z")[0],
                   event_day=TUE, notional=1e5, role_score=1)
    ts, px = entry_price_for(ev_pre, bars, [])
    assert ts.astimezone(ET).strftime("%H:%M") == "09:30" and px == pytest.approx(100.0)


def test_entry_falls_back_to_event_day_close_without_minute_bars():
    ev = Event(ticker="ACME", event_ts=parse_ts("2026-05-12T14:30:00Z")[0],
               event_day=TUE, notional=1e5, role_score=1)
    daily = _daily(date(2026, 5, 11), [10.0, 11.0, 12.0])
    ts, px = entry_price_for(ev, None, daily)
    assert px == 11.0
    assert ts.astimezone(ET).date() == TUE and ts.astimezone(ET).hour == 16
    # Minute bars present but filing after the last regular bar -> None, not
    # a silent fallback to the close.
    late = Event(ticker="ACME", event_ts=parse_ts("2026-05-12T19:59:30Z")[0],
                 event_day=TUE, notional=1e5, role_score=1)
    assert entry_price_for(late, _minute_bars(TUE), daily) is None


# ---------------------------------------------------------------- returns
def test_forward_returns_pick_nth_trading_day_after_entry_day():
    entry_ts = datetime.combine(TUE, datetime.min.time(), tzinfo=ET).replace(hour=10)
    # Mon 11 .. Fri 15, Mon 18 .. (weekends skipped by _daily)
    daily = _daily(date(2026, 5, 11), [100, 100, 110, 120, 130, 140, 150, 160])
    out = forward_returns(100.0, entry_ts, daily, horizons=(1, 2, 5, 20))
    assert out[1] == pytest.approx(10.0)    # Wed close 110
    assert out[2] == pytest.approx(20.0)    # Thu 120
    assert out[5] == pytest.approx(50.0)    # Tue 19 close 150
    assert out[20] is None                  # series too short -> explicit None


def test_abnormal_subtracts_benchmark_with_beta():
    raw = {1: 2.0, 5: 5.0, 20: None}
    bench = {1: 0.5, 5: 1.0, 20: 3.0}
    out = abnormal_returns(raw, bench)
    assert out == {1: pytest.approx(1.5), 5: pytest.approx(4.0), 20: None}
    assert abnormal_returns(raw, bench, beta=2.0)[5] == pytest.approx(3.0)


def test_net_of_cost_uses_replay_tier_table():
    assert net_of_cost(1.0, 100.0) == pytest.approx(1.0 - 100 * _round_trip_cost_frac(100.0))
    assert net_of_cost(1.0, 100.0) == pytest.approx(0.8)      # 20 bps tier
    assert net_of_cost(1.0, 12.0) == pytest.approx(0.5)       # 50 bps tier


# ---------------------------------------------------------------- summarize
def test_summarize_drop_top_k_scales_with_n_not_fixed_five():
    assert drop_top_k_for(8) == 1
    assert drop_top_k_for(32) == 3
    assert drop_top_k_for(50) == 5
    assert drop_top_k_for(500) == 50
    vals = [1.0] * 29 + [50.0, 60.0, 70.0]   # n=32 -> k=3 removes the three outliers
    s = summarize(vals)
    assert s["n"] == 32 and s["drop_top_k"] == 3
    assert s["drop_top_k_mean"] == pytest.approx(1.0)
    assert s["median"] == 1.0
    assert s["win_rate"] == 1.0
    # fixed drop-5 would have given a different number
    assert np.mean(sorted(vals)[:-5]) == pytest.approx(1.0) and s["mean"] > 1.0
    # trim10 drops floor(0.1n)=3 from each tail
    assert s["trim10_mean"] == pytest.approx(1.0)
    assert summarize([])["n"] == 0


# ---------------------------------------------------------------- early/late
def _er(ticker, day, val, horizon=5):
    return EventReturn(ticker=ticker, event_day=day, horizon=horizon, raw=val,
                       abnormal=val, net=val, entry_px=50.0)


def test_early_late_split_is_on_days_not_rows():
    d0 = date(2026, 3, 2)
    days = [d0 + timedelta(days=i) for i in range(10)]
    rows = [_er("A", days[0], 1.0), _er("B", days[0], 1.0), _er("C", days[0], 1.0)]
    rows += [_er("X", d, 1.0) for d in days[1:]]            # 12 rows over 10 days
    early, late = early_late_split(rows)
    early_days = {r.event_day for r in early}
    late_days = {r.event_day for r in late}
    assert early_days == set(days[:7]) and late_days == set(days[7:])
    assert len(early) == 9 and len(late) == 3
    # A row-based 70% split would have put 8 rows (through days[5]) early.
    assert early_days.isdisjoint(late_days)


# ---------------------------------------------------------------- permutation
def _pool(tickers, n_days=120, seed=1):
    rng = np.random.default_rng(seed)
    pool = {}
    d0 = date(2025, 11, 3)
    for i, t in enumerate(tickers):
        vol = 1.0 + i * 0.5
        pool[t] = [(d0 + timedelta(days=k), float(rng.normal(0.0, vol))) for k in range(n_days)]
    return pool


def test_permutation_p_is_near_half_under_null_and_small_under_shift():
    tickers = [f"T{i}" for i in range(20)]
    pool = _pool(tickers)
    rng = np.random.default_rng(7)
    event_day = date(2026, 5, 12)
    # Null: events are themselves draws from each ticker's placebo pool. Run
    # several independent null samples; p should scatter around 0.5.
    ps = []
    for rep in range(10):
        events = []
        for t in tickers:
            vals = [v for _, v in pool[t]]
            events.append(_er(t, event_day + timedelta(days=rep), float(rng.choice(vals))))
        ps.append(permutation_pvalue(events, pool, n_iter=2000, seed=rep)["p_value"])
    assert 0.2 < float(np.mean(ps)) < 0.8
    assert min(ps) < 0.5 < max(ps)
    # Obvious positive shift: +5 sigma-ish for every event.
    shifted = [_er(t, event_day, 6.0) for t in tickers]
    res = permutation_pvalue(shifted, pool, n_iter=2000, seed=0)
    assert res["p_value"] < 0.01
    assert res["n_events"] == 20 and res["n_excluded"] == 0


def test_permutation_is_ticker_matched_and_excludes_own_day():
    pool = {"LOWVOL": [(date(2026, 1, 1) + timedelta(days=i), 0.0) for i in range(30)],
            "HIVOL": [(date(2026, 1, 1) + timedelta(days=i), (-1) ** i * 10.0) for i in range(30)]}
    # The HIVOL event's own day carries +100 in the pool; it must be ignored.
    pool["HIVOL"].append((date(2026, 5, 12), 100.0))
    events = [_er("LOWVOL", date(2026, 5, 12), 0.5), _er("HIVOL", date(2026, 5, 12), 0.5)]
    res = permutation_pvalue(events, pool, n_iter=500, seed=0)
    assert res["null_mean"] == pytest.approx(0.0, abs=0.5)
    assert abs(res["null_sd"]) < 6  # only HIVOL contributes variance
    # Ticker with no pool is excluded from both sides, not silently zeroed.
    res2 = permutation_pvalue(events + [_er("NOPOOL", date(2026, 5, 12), 99.0)], pool, n_iter=100)
    assert res2["n_excluded"] == 1 and res2["observed_mean"] == pytest.approx(0.5)


def test_build_placebo_pool_excludes_event_windows_and_matches_benchmark():
    d0 = date(2026, 3, 2)
    closes = [100.0 * (1.01 ** i) for i in range(30)]      # +1%/day
    bench = [100.0 * (1.005 ** i) for i in range(30)]      # +0.5%/day
    tb = {"ACME": _daily(d0, closes)}
    bb = _daily(d0, bench)
    days = sorted({bar_day for bar_day in (b["t"][:10] for b in bb)})
    event_day = date.fromisoformat(days[15])
    pool = build_placebo_pool(tb, bb, horizon=5, event_days={"ACME": [event_day]})
    pool_days = [d for d, _ in pool["ACME"]]
    assert event_day not in pool_days
    # |idx - 15| <= 5 excluded -> days index 10..20 absent
    excluded = {date.fromisoformat(days[i]) for i in range(10, 21)}
    assert not (set(pool_days) & excluded)
    for _, v in pool["ACME"]:
        assert v == pytest.approx(((1.01 ** 5) - (1.005 ** 5)) * 100, abs=1e-6)


# ---------------------------------------------------------------- bootstrap
def test_cluster_bootstrap_resamples_days():
    d = date(2026, 5, 12)
    ers = [_er("A", d, 1.0), _er("B", d, 1.0), _er("C", d + timedelta(days=1), 3.0)]
    res = cluster_bootstrap_ci(ers, n_iter=2000, seed=0)
    assert res["n_clusters"] == 2
    assert res["mean"] == pytest.approx(5.0 / 3)
    # Resampling two clusters: possible means are {1, 3, 5/3}; CI spans them.
    assert res["ci_low"] == pytest.approx(1.0) and res["ci_high"] == pytest.approx(3.0)


# ---------------------------------------------------------------- gauntlet
def _good_sample(n_days: int, tickers_per_day: int = 2, val: float = 2.0):
    d0 = date(2026, 2, 2)
    days = []
    d = d0
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    ers, pool = [], {}
    for i, day in enumerate(days):
        for j in range(tickers_per_day):
            t = f"T{i}_{j}"
            ers.append(_er(t, day, val + 0.1 * j))
            pool[t] = [(date(2025, 6, 1) + timedelta(days=k), 0.0) for k in range(40)]
    return ers, pool


def test_gauntlet_fails_on_fewer_than_15_days_even_if_everything_else_passes():
    ers, pool = _good_sample(n_days=10)
    g = gauntlet(ers, pool, n_families_tested=1, n_iter=500)
    assert g["underpowered"] is True
    assert g["criteria"]["min_days"]["pass"] is False
    assert g["criteria"]["median_positive"]["pass"] is True
    assert g["criteria"]["early_positive"]["pass"] and g["criteria"]["late_positive"]["pass"]
    assert g["passed"] is False
    ers15, pool15 = _good_sample(n_days=15)
    g15 = gauntlet(ers15, pool15, n_families_tested=1, n_iter=500)
    assert g15["criteria"]["min_days"]["pass"] is True and g15["passed"] is True


def test_gauntlet_applies_bonferroni_divisor_in_code():
    # Build a sample whose permutation p lands between 0.05/8 and 0.05.
    rng = np.random.default_rng(3)
    d0 = date(2026, 2, 2)
    ers, pool = [], {}
    for i in range(20):
        t = f"T{i}"
        day = d0 + timedelta(days=i)
        pool[t] = [(date(2025, 6, 1) + timedelta(days=k), float(rng.normal(0, 1))) for k in range(60)]
        ers.append(_er(t, day, 0.45))
    g1 = gauntlet(ers, pool, n_families_tested=1, n_iter=4000, seed=0)
    g8 = gauntlet(ers, pool, n_families_tested=8, n_iter=4000, seed=0)
    p = g1["permutation"]["p_value"]
    assert g1["alpha_bonferroni"] == pytest.approx(0.05)
    assert g8["alpha_bonferroni"] == pytest.approx(0.05 / 8)
    assert g1["criteria"]["permutation_significant"]["threshold"] == pytest.approx(0.05)
    assert g8["criteria"]["permutation_significant"]["threshold"] == pytest.approx(0.00625)
    assert 0.00625 < p < 0.05, f"synthetic p={p} not in the discriminating band"
    assert g1["criteria"]["permutation_significant"]["pass"] is True
    assert g8["criteria"]["permutation_significant"]["pass"] is False
    assert g1["passed"] is True and g8["passed"] is False


def test_gauntlet_fails_when_outlier_carried():
    ers, pool = _good_sample(n_days=16, tickers_per_day=2, val=-0.2)
    # Two monster winners lift the mean but median/trim/drop-k stay negative.
    ers[0] = _er(ers[0].ticker, ers[0].event_day, 40.0)
    ers[1] = _er(ers[1].ticker, ers[1].event_day, 40.0)
    g = gauntlet(ers, pool, n_iter=500)
    assert g["summary"]["mean"] > 0
    assert g["criteria"]["median_positive"]["pass"] is False
    assert g["criteria"]["drop_top_k_positive"]["pass"] is False
    assert g["passed"] is False
