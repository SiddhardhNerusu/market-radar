"""Insider lane invariants. These lock the behaviours the June audits found
missing in the old trader: a real stop that exists at the broker, strict
one-bet-per-ticker-day dedup that survives restarts, a daily-loss switch that
FLATTENS, a time stop, idempotent submits, and a ledger row for every
qualifying candidate whether or not it was traded.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from market_radar.lane import candidates as cands
from market_radar.lane import promotion
from market_radar.lane.config import LaneConfig
from market_radar.lane.ledger import nth_close_after, resolve_row
from market_radar.lane.sessions import event_day_for, in_entry_window
from market_radar.lane.sizing import plan_entry
from market_radar.lane.trader import InsiderLaneTrader
from market_radar.storage.db import get_connection, init_db

UTC = timezone.utc


# ----------------------------------------------------------------- sessions
def test_event_day_rolls_after_close_and_over_weekend():
    assert event_day_for("2026-06-10T14:00:00Z") == "2026-06-10"      # 10:00 ET Wed
    assert event_day_for("2026-06-10T21:30:00Z") == "2026-06-11"      # 17:30 ET Wed -> Thu
    assert event_day_for("2026-06-12T22:00:00Z") == "2026-06-15"      # Fri evening -> Mon
    assert event_day_for("2026-06-13T15:00:00Z") == "2026-06-15"      # Saturday -> Mon
    assert event_day_for("garbage") is None


def test_entry_window_excludes_open_and_close():
    # 2026-06-10 is a Wednesday. 13:35Z = 09:35 ET; 15:00Z = 11:00 ET; 19:45Z = 15:45 ET
    assert not in_entry_window(datetime(2026, 6, 10, 13, 35, tzinfo=UTC), first=15, last=30)
    assert in_entry_window(datetime(2026, 6, 10, 15, 0, tzinfo=UTC), first=15, last=30)
    assert not in_entry_window(datetime(2026, 6, 10, 19, 45, tzinfo=UTC), first=15, last=30)
    assert not in_entry_window(datetime(2026, 6, 13, 15, 0, tzinfo=UTC), first=15, last=30)  # Saturday


# --------------------------------------------------------------- candidates
def _row(ticker="ACME", code="P", acquired=1, shares=1000, price=30.0, pub="2026-06-10T14:00:00Z",
         role=1, tenb=0, sid=1):
    return dict(ticker=ticker, transaction_code=code, is_acquired=acquired, shares=shares, price=price,
                published_at=pub, role_score=role, is_10b5_1=tenb, signal_id=sid)


def test_aggregate_dedups_same_day_and_filters():
    rows = [
        _row(sid=1, shares=1000, price=30.0, role=1),                 # $30k
        _row(sid=2, shares=500, price=30.0, role=3, pub="2026-06-10T15:00:00Z"),  # same day, $15k, CEO
        _row(sid=3, code="S"),                                          # sale: ignored
        _row(sid=4, tenb=1, shares=5000),                               # plan trade: excluded
        _row(sid=5, ticker="PENNY", price=4.0, shares=100000),          # below price floor
        _row(sid=6, ticker="SMALL", shares=100, price=30.0),            # $3k < min notional
        _row(sid=7, ticker="ACME", pub="2026-06-10T21:00:00Z", shares=2000, price=30.0),  # after close -> next day
    ]
    out = cands.aggregate_purchases(rows, min_notional_usd=25_000, min_price=10.0)
    keys = {(c.ticker, c.event_day): c for c in out}
    assert set(keys) == {("ACME", "2026-06-10"), ("ACME", "2026-06-11")}
    c = keys[("ACME", "2026-06-10")]
    assert c.notional_usd == pytest.approx(45_000)
    assert c.role_score == 3 and c.n_purchases == 2 and sorted(c.signal_ids) == [1, 2]
    assert c.published_at == "2026-06-10T14:00:00Z"


def test_aggregate_min_role_score():
    out = cands.aggregate_purchases([_row(role=1)], min_notional_usd=1, min_price=1, min_role_score=2)
    assert out == []


# ------------------------------------------------------------------- sizing
def _plan(**kw):
    base = dict(bid=49.95, ask=50.05, atr=1.0, book_equity=10_000, risk_pct=0.5, stop_atr_mult=2.0,
                tp_atr_mult=4.0, max_position_pct=20.0, limit_bps=20.0, max_spread_bps=50.0,
                min_price=10.0, open_heat_usd=0.0, max_heat_pct=5.0, hard_order_cap_usd=2_500)
    base.update(kw)
    return plan_entry(**base)


def test_sizing_is_risk_based_with_broker_stop():
    p = _plan()
    assert p.ok
    assert p.limit_price == pytest.approx(50.15, abs=0.01)
    assert p.stop_loss == pytest.approx(48.15, abs=0.01)
    assert p.take_profit == pytest.approx(54.15, abs=0.01)
    # $50 at risk / $2 per share = 25 shares; notional $1,254 under all caps
    assert p.qty == 25 and p.risk_usd == pytest.approx(50.0, abs=0.5)
    assert p.stop_loss < p.limit_price < p.take_profit


def test_sizing_rejects_wide_spread_low_price_and_no_atr():
    assert not _plan(bid=49.0, ask=50.0).ok            # ~200 bps spread
    assert not _plan(bid=8.0, ask=8.02).ok              # below floor
    assert not _plan(atr=0.0).ok


def test_sizing_sizes_down_to_portfolio_heat_and_hard_cap():
    tight = _plan(open_heat_usd=480.0)                  # $20 heat room -> 10 shares
    assert tight.ok and tight.qty == 10
    assert not _plan(open_heat_usd=500.0).ok            # heat at cap
    capped = _plan(hard_order_cap_usd=600.0)            # 600/50.15 -> 11 shares
    assert capped.ok and capped.qty == 11


# ------------------------------------------------------------------- ledger
def _bars(start="2026-06-10", n=25, px=100.0, step=1.0):
    d0 = datetime.fromisoformat(start)
    out = []
    for i in range(n):
        d = d0 + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        out.append({"t": d.strftime("%Y-%m-%dT04:00:00Z"), "c": px + step * len(out)})
    return out


def test_nth_close_after_skips_event_day():
    bars = _bars()
    day, close = nth_close_after(bars, "2026-06-10", 1)
    assert day == "2026-06-11" and close == 101.0
    assert nth_close_after(bars, "2026-06-10", 5)[0] == "2026-06-17"
    assert nth_close_after(bars[:3], "2026-06-10", 5) is None


def test_resolve_row_market_adjusts():
    row = {"event_day": "2026-06-10", "price_at_flag": 100.0, "bench_at_flag": 500.0,
           "ret_5d_pct": None, "ret_20d_pct": None}
    upd = resolve_row(row, _bars(px=100.0, step=1.0), _bars(px=500.0, step=0.0))
    assert upd["ret_5d_pct"] == pytest.approx(5.0)
    assert upd["abn_5d_pct"] == pytest.approx(5.0)   # SPY flat
    assert "ret_20d_pct" not in upd                     # not enough bars yet


# ---------------------------------------------------------------- promotion
def _ledger_rows(days, values, cost=0.002):
    return [{"event_day": d, "abn_5d_pct": v, "cost_frac": cost} for d, v in zip(days, values)]


def _days(n, start="2026-07-01"):
    d0 = datetime.fromisoformat(start)
    out, i = [], 0
    while len(out) < n:
        d = d0 + timedelta(days=i)
        i += 1
        if d.weekday() < 5:
            out.append(d.date().isoformat())
    return out


def test_gate_fails_on_too_few_days_even_if_positive():
    rows = _ledger_rows(_days(10), [2.0] * 10)
    res = promotion.evaluate(rows, n_iter=500)
    assert not res.passed and not res.checks["sample_size"][0]


def test_gate_fails_when_carried_by_outliers():
    days = _days(20)
    vals = [-0.3] * 18 + [40.0, 35.0]          # mean positive, median negative
    res = promotion.evaluate(_ledger_rows(days, vals), n_iter=500)
    assert not res.passed
    assert not res.checks["median_gt_0"][0]
    assert res.stats["mean"] > 0


def test_gate_passes_on_a_robust_cohort_and_bonferroni_bites():
    import random
    rng = random.Random(1)
    days = _days(30)
    vals = [1.5 + rng.gauss(0, 0.5) for _ in days]
    res = promotion.evaluate(_ledger_rows(days, vals), n_iter=1000)
    assert res.passed, res.summary()
    # with a tiny effect the Bonferroni divisor must be able to flip the verdict
    weak = [0.25 + rng.gauss(0, 0.9) for _ in days]
    r1 = promotion.evaluate(_ledger_rows(days, weak), n_iter=2000, n_families=1)
    r8 = promotion.evaluate(_ledger_rows(days, weak), n_iter=2000, n_families=8)
    assert r8.stats["alpha_adj"] == pytest.approx(0.05 / 8)
    assert (not r8.checks["cluster_bootstrap"][0]) or r1.checks["cluster_bootstrap"][0]


def test_drop_top_k_scales_with_n():
    xs = list(range(30))
    _, k = promotion.drop_top_k_mean(xs)
    assert k == 3
    _, k2 = promotion.drop_top_k_mean(list(range(500)))
    assert k2 == 50


# ------------------------------------------------------------------- trader
class _Order:
    def __init__(self, oid, coid, symbol, qty, status="new", limit_price=None, legs=None,
                 filled_qty=0.0, filled_avg_price=None, filled_at=None, submitted_at=None):
        self.id, self.client_order_id, self.symbol, self.qty = oid, coid, symbol, qty
        self.status, self.limit_price, self.legs = status, limit_price, legs or []
        self.filled_qty, self.filled_avg_price, self.filled_at = filled_qty, filled_avg_price, filled_at
        self.submitted_at = submitted_at


class FakeAlpaca:
    def __init__(self, *, equity=100_000.0, positions=None, quotes=None, is_open=True):
        self.equity = equity
        self.positions = list(positions or [])
        self.quotes = dict(quotes or {})
        self.is_open = is_open
        self.orders: dict[str, _Order] = {}
        self.brackets = []
        self.simple = []
        self.canceled_all = 0
        self.canceled_for = []

    def get_account(self):
        return NS(equity=self.equity, cash=self.equity, buying_power=self.equity * 2,
                  long_market_value=sum(p.market_value for p in self.positions))

    def get_positions(self):
        return list(self.positions)

    def get_market_clock(self):
        return {"is_open": self.is_open}

    def get_latest_quote(self, symbol):
        return self.quotes.get(symbol)

    def submit_bracket_order(self, *, symbol, side, qty, take_profit, stop_loss, limit_price=None,
                             time_in_force="day", client_order_id=None, **_):
        assert side == "buy" and stop_loss < limit_price < take_profit
        oid = f"o{len(self.orders) + 1}"
        o = _Order(oid, client_order_id, symbol, qty, limit_price=limit_price,
                   legs=[{"type": "limit", "status": "new"}, {"type": "stop", "status": "new"}])
        self.orders[oid] = o
        self.brackets.append(dict(symbol=symbol, qty=qty, tp=take_profit, sl=stop_loss, limit=limit_price,
                                  tif=time_in_force, coid=client_order_id))
        return NS(parent=o, take_profit_leg=None, stop_loss_leg=None)

    def submit_simple_order(self, **kw):
        self.simple.append(kw)
        return _Order(f"s{len(self.simple)}", kw.get("client_order_id"), kw["symbol"], kw["qty"])

    def get_order(self, oid):
        return self.orders[oid]

    def get_order_by_client_order_id(self, coid):
        return next((o for o in self.orders.values() if o.client_order_id == coid), None)

    def cancel_order(self, oid):
        self.orders[oid].status = "canceled"

    def cancel_all_orders(self):
        self.canceled_all += 1
        return 0

    def cancel_open_orders_for(self, symbol):
        self.canceled_for.append(symbol)
        return 1

    # helpers
    def fill(self, oid, price):
        o = self.orders[oid]
        o.status, o.filled_qty, o.filled_avg_price = "filled", o.qty, price
        o.filled_at = "2026-06-10T15:01:00Z"
        self.positions.append(NS(symbol=o.symbol, qty=o.qty, market_value=o.qty * price,
                                 avg_entry_price=price, unrealized_pl=0.0))

    def stop_out(self, oid):
        o = self.orders[oid]
        o.legs[1]["status"] = "filled"
        self.positions = [p for p in self.positions if p.symbol != o.symbol]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "lane.db"
    init_db(path)
    with get_connection(path) as c:
        c.execute("""CREATE TABLE insider_transactions (signal_id INTEGER, ticker TEXT, insider_name TEXT,
            transaction_code TEXT, shares REAL, price REAL, is_acquired INTEGER, officer_title TEXT,
            is_officer INTEGER, is_director INTEGER, is_10pct INTEGER, role_score INTEGER, report_date TEXT,
            ingested_at TEXT, is_10b5_1 INTEGER)""")
    return path


def _factory(path):
    @contextmanager
    def f():
        with get_connection(path) as c:
            yield c
    return f


def _seed_form4(path, *, ticker="ACME", shares=2000, price=50.0, pub="2026-06-10T14:00:00Z", sid=101, tenb=0):
    with get_connection(path) as c:
        c.execute("INSERT INTO raw_signals (id, source, source_tier, title, published_at, ingested_at) VALUES (?,?,?,?,?,?)",
                  (sid, "sec_edgar", 1, "4 - ACME", pub, pub))
        c.execute("""INSERT INTO insider_transactions (signal_id, ticker, transaction_code, shares, price, is_acquired,
                     role_score, report_date, ingested_at, is_10b5_1) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                  (sid, ticker, "P", shares, price, 1, 2, pub[:10], pub, tenb))


NOW = datetime(2026, 6, 10, 15, 0, tzinfo=UTC)   # 11:00 ET Wednesday


def _trader(path, alpaca, cfg=None, now=NOW):
    cfg = cfg or LaneConfig(book_equity_usd=10_000)
    return InsiderLaneTrader(cfg, alpaca=alpaca, conn_factory=_factory(path), atr_fn=lambda s: 1.0,
                             now_fn=lambda: now, notify=lambda m: None)


def test_places_bracket_once_and_writes_ledger(db):
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "SPY": (500.0, 500.1)})
    t = _trader(db, a)
    s1 = t.run_once()
    assert s1["placed"] == 1 and len(a.brackets) == 1
    b = a.brackets[0]
    assert b["symbol"] == "ACME" and b["tif"] == "gtc" and b["sl"] < b["limit"] < b["tp"]
    assert b["qty"] == 25                                  # $50 risk / $2 stop distance
    assert b["coid"] == "mr-ins-ACME-20260610"
    # second loop: strict dedup, nothing new
    s2 = t.run_once()
    assert s2["placed"] == 0 and len(a.brackets) == 1
    with get_connection(db) as c:
        dec = c.execute("SELECT outcome, parent_order_id FROM lane_decisions").fetchall()
        assert len(dec) == 1 and dec[0][0] == "placed" and dec[0][1] == "o1"
        led = c.execute("SELECT candidate, traded, price_at_flag, bench_at_flag FROM edge_ledger").fetchone()
        assert led[1] == 1 and led[2] == pytest.approx(50.05) and led[3] == pytest.approx(500.1)
        assert led[0].startswith("insider_P_px10_notional25k")


def test_ledger_row_written_even_when_not_traded(db):
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.0, 50.0), "SPY": (500.0, 500.1)})   # 200bps spread -> blocked
    t = _trader(db, a)
    s = t.run_once()
    assert s["placed"] == 0 and len(a.brackets) == 0
    with get_connection(db) as c:
        assert c.execute("SELECT outcome FROM lane_decisions").fetchone()[0] == "blocked"
        assert c.execute("SELECT traded FROM edge_ledger").fetchone()[0] == 0


def test_no_entries_outside_window_or_when_closed(db):
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "SPY": (500.0, 500.1)})
    assert _trader(db, a, now=datetime(2026, 6, 10, 13, 35, tzinfo=UTC)).run_once()["placed"] == 0
    a.is_open = False
    assert _trader(db, a).run_once()["placed"] == 0
    assert a.brackets == []


def test_dry_run_submits_nothing_but_records(db):
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "SPY": (500.0, 500.1)})
    t = _trader(db, a, cfg=LaneConfig(book_equity_usd=10_000, dry_run=True))
    t.run_once()
    assert a.brackets == []
    with get_connection(db) as c:
        out, det = c.execute("SELECT outcome, outcome_detail FROM lane_decisions").fetchone()
        assert out == "skipped" and det.startswith("dry-run: would buy 25")


def test_daily_loss_flattens_by_live_sign_and_halts(db):
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "SPY": (500.0, 500.1)})
    t = _trader(db, a)
    t.run_once()
    a.fill("o1", 50.1)
    t.run_once()                                   # reconcile -> open
    a.equity = 100_000 - 301.0                     # > 3% of the $10k book
    s = t.run_once()
    assert s["halted"] and a.canceled_all == 1
    assert len(a.simple) == 1
    close = a.simple[0]
    assert close["symbol"] == "ACME" and close["side"] == "sell" and close["qty"] == 25
    assert close["position_intent"] == "sell_to_close"
    # a new candidate the same day is not entered while halted
    _seed_form4(db, ticker="BETA", sid=202)
    a.quotes["BETA"] = (29.95, 30.05)
    s2 = t.run_once()
    assert s2["halted"] and len(a.brackets) == 1
    with get_connection(db) as c:
        assert tuple(c.execute("SELECT status, exit_reason FROM lane_positions").fetchone()) == ("closing", "daily_loss")


def test_reconcile_marks_stop_out_closed(db):
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "SPY": (500.0, 500.1)})
    t = _trader(db, a)
    t.run_once()
    a.fill("o1", 50.1)
    t.run_once()
    with get_connection(db) as c:
        assert tuple(c.execute("SELECT status, filled_avg_price FROM lane_positions").fetchone()) == ("open", 50.1)
    a.stop_out("o1")
    s = t.run_once()
    assert s["closed"] == 1
    with get_connection(db) as c:
        assert tuple(c.execute("SELECT status, exit_reason FROM lane_positions").fetchone()) == ("closed", "stop_loss")


def test_time_stop_cancels_legs_then_closes(db):
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "SPY": (500.0, 500.1)})
    t = _trader(db, a)
    t.run_once()
    a.fill("o1", 50.1)
    t.run_once()
    later = NOW + timedelta(days=15)
    t2 = _trader(db, a, now=later)
    s = t2.run_once()
    assert s["closed"] == 1 and a.canceled_for == ["ACME"]
    assert a.simple[-1]["side"] == "sell" and a.simple[-1]["position_intent"] == "sell_to_close"


def test_unfilled_parent_is_canceled_after_ttl(db):
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "SPY": (500.0, 500.1)})
    _trader(db, a).run_once()
    _trader(db, a, now=NOW + timedelta(minutes=61)).run_once()
    assert a.orders["o1"].status == "canceled"
    _trader(db, a, now=NOW + timedelta(minutes=62)).run_once()
    with get_connection(db) as c:
        assert tuple(c.execute("SELECT status, exit_reason FROM lane_positions").fetchone()) == ("canceled", "unfilled")


def test_pending_submit_is_recovered_not_resubmitted(db):
    """Crash between the broker accepting the order and our position insert."""
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "SPY": (500.0, 500.1)})
    coid = "mr-ins-ACME-20260610"
    a.submit_bracket_order(symbol="ACME", side="buy", qty=25, take_profit=54.15, stop_loss=48.15,
                           limit_price=50.15, client_order_id=coid)
    with get_connection(db) as c:
        c.execute("""INSERT INTO lane_decisions (ticker, event_day, published_at, notional_usd, role_score, outcome,
                     client_order_id, decided_at) VALUES ('ACME','2026-06-10','2026-06-10T14:00:00Z',100000,2,
                     'pending_submit',?, '2026-06-10T14:59:00Z')""", (coid,))
    t = _trader(db, a)
    s = t.run_once()
    assert s["placed"] == 0 and len(a.brackets) == 1          # adopted, not duplicated
    with get_connection(db) as c:
        assert tuple(c.execute("SELECT outcome, parent_order_id FROM lane_decisions").fetchone()) == ("placed", "o1")
        assert c.execute("SELECT status FROM lane_positions").fetchone()[0] == "submitted"


def test_max_positions_and_cooldown(db):
    cfg = LaneConfig(book_equity_usd=10_000, max_positions=1)
    _seed_form4(db, ticker="ACME", sid=1)
    _seed_form4(db, ticker="BETA", sid=2)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "BETA": (29.95, 30.05), "SPY": (500.0, 500.1)})
    t = _trader(db, a, cfg=cfg)
    t.run_once()
    assert len(a.brackets) == 1
    with get_connection(db) as c:
        rows = dict(c.execute("SELECT ticker, outcome FROM lane_decisions").fetchall())
        assert rows == {"ACME": "placed", "BETA": "skipped"}
    # a second filing on ACME two days later is inside the cooldown
    _seed_form4(db, ticker="ACME", sid=3, pub="2026-06-12T14:00:00Z")
    t3 = _trader(db, a, cfg=LaneConfig(book_equity_usd=10_000, max_positions=5),
                 now=datetime(2026, 6, 12, 15, 0, tzinfo=UTC))
    t3.run_once()
    with get_connection(db) as c:
        det = c.execute("SELECT outcome_detail FROM lane_decisions WHERE event_day='2026-06-12'").fetchone()[0]
        assert "already held" in det or "cooldown" in det


def test_snapshot_failure_is_fail_closed(db):
    _seed_form4(db)
    a = FakeAlpaca(quotes={"ACME": (49.95, 50.05), "SPY": (500.0, 500.1)})
    a.get_positions = lambda: (_ for _ in ()).throw(RuntimeError("down"))
    s = _trader(db, a).run_once()
    assert s == {"placed": 0, "closed": 0, "halted": False, "skipped": 0} and a.brackets == []


def test_config_validation_refuses_incoherent():
    with pytest.raises(ValueError):
        InsiderLaneTrader(LaneConfig(min_price=2.0), alpaca=FakeAlpaca(), conn_factory=lambda: None)
    assert LaneConfig().validate() == []
