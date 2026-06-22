"""Lock the fills-based realized-P&L FIFO (full-audit rebuild ch2). These pin the
source-of-truth booking so the ledger can never again silently drop exits."""
from market_radar.execution.pnl_reconcile import (
    realized_from_fills, _is_equity_symbol,
)


def _f(sym, side, qty, price, t):
    return {"symbol": sym, "side": side, "qty": qty, "price": price,
            "transaction_time": t}


def test_simple_round_trip_win():
    by_date, unc = realized_from_fills([
        _f("AAPL", "buy", 100, 10.0, "2026-06-01T14:00:00Z"),
        _f("AAPL", "sell", 100, 12.0, "2026-06-01T15:00:00Z"),
    ])
    assert unc == []
    r = by_date["2026-06-01"]
    assert r["realized"] == 200.0 and r["trades"] == 1
    assert r["wins"] == 1 and r["losses"] == 0


def test_loss_round_trip():
    by_date, _ = realized_from_fills([
        _f("MSFT", "buy", 50, 20.0, "2026-06-02T10:00:00Z"),
        _f("MSFT", "sell", 50, 18.0, "2026-06-02T11:00:00Z"),
    ])
    r = by_date["2026-06-02"]
    assert r["realized"] == -100.0 and r["losses"] == 1 and r["wins"] == 0


def test_fifo_partial_across_two_lots():
    # 100@5 (+200) + 50@6 (+50) = +250 on a single 150-share sell
    by_date, unc = realized_from_fills([
        _f("X", "buy", 100, 5.0, "2026-06-03T10:00:00Z"),
        _f("X", "buy", 100, 6.0, "2026-06-03T10:05:00Z"),
        _f("X", "sell", 150, 7.0, "2026-06-03T11:00:00Z"),
    ])
    assert unc == []
    r = by_date["2026-06-03"]
    assert r["realized"] == 250.0 and r["trades"] == 1


def test_uncovered_sell_flagged_not_counted():
    by_date, unc = realized_from_fills([
        _f("Y", "sell", 10, 5.0, "2026-06-04T10:00:00Z"),
    ])
    assert by_date == {}
    assert len(unc) == 1 and unc[0]["symbol"] == "Y" and unc[0]["qty"] == 10


def test_input_order_does_not_matter():
    # sell appears before buy in input; transaction_time still orders them
    by_date, unc = realized_from_fills([
        _f("Z", "sell", 50, 11.0, "2026-06-05T15:00:00Z"),
        _f("Z", "buy", 50, 10.0, "2026-06-05T14:00:00Z"),
    ])
    assert unc == []
    assert by_date["2026-06-05"]["realized"] == 50.0


def test_partial_close_leaves_remainder_open():
    by_date, unc = realized_from_fills([
        _f("Q", "buy", 100, 10.0, "2026-06-06T10:00:00Z"),
        _f("Q", "sell", 40, 12.0, "2026-06-06T11:00:00Z"),  # +80; 60 still open
    ])
    assert unc == []
    r = by_date["2026-06-06"]
    assert r["realized"] == 80.0 and r["trades"] == 1


def test_equity_symbol_filter():
    assert _is_equity_symbol("AAPL")
    assert _is_equity_symbol("VNCE")
    assert not _is_equity_symbol("BTC/USD")           # crypto pair
    assert not _is_equity_symbol("NVDA260605C00225000")  # OCC option
    assert not _is_equity_symbol("")


# --- reconcile-to-DB (the idempotent ledger rebuild) ---
from contextlib import contextmanager  # noqa: E402

from market_radar.execution.pnl_reconcile import reconcile_daily_pnl_from_fills  # noqa: E402
from market_radar.storage.db import get_connection as _gc, init_db  # noqa: E402


class _FakeAlpaca:
    """One page of fills, then empty (mimics Alpaca activities paging)."""
    def __init__(self, fills):
        self._fills = fills

    def _request(self, method, path, params=None):
        return self._fills if (params or {}).get("page_token") is None else []


def _factory(db):
    @contextmanager
    def cf(path=None):
        with _gc(db) as c:
            yield c
    return cf


def test_reconcile_writes_daily_pnl_excluding_crypto(tmp_path):
    db = tmp_path / "t.db"; init_db(db)
    fills = [
        _f("AAPL", "buy", 100, 10.0, "2026-06-20T14:00:00Z") | {"id": "1"},
        _f("AAPL", "sell", 100, 11.0, "2026-06-20T15:00:00Z") | {"id": "2"},
        _f("BTC/USD", "buy", 1, 100.0, "2026-06-20T16:00:00Z") | {"id": "3"},  # crypto: ignored
    ]
    summ = reconcile_daily_pnl_from_fills(
        _FakeAlpaca(fills), _factory(db), now_iso="2026-06-21T00:00:00Z")
    assert summ["total_realized"] == 100.0  # crypto excluded
    with _gc(db) as c:
        row = c.execute(
            "SELECT realized_pnl_usd, trades_count, wins, losses "
            "FROM bot_daily_pnl WHERE trading_date='2026-06-20'").fetchone()
    assert round(row[0], 2) == 100.0 and row[1] == 1 and row[2] == 1 and row[3] == 0


def test_reconcile_since_date_leaves_old_history_untouched(tmp_path):
    db = tmp_path / "t.db"; init_db(db)
    fills = [
        _f("AAPL", "buy", 10, 10.0, "2026-05-01T14:00:00Z") | {"id": "1"},
        _f("AAPL", "sell", 10, 12.0, "2026-05-01T15:00:00Z") | {"id": "2"},  # old: skipped
        _f("MSFT", "buy", 10, 10.0, "2026-06-20T14:00:00Z") | {"id": "3"},
        _f("MSFT", "sell", 10, 9.0, "2026-06-20T15:00:00Z") | {"id": "4"},   # -10: kept
    ]
    summ = reconcile_daily_pnl_from_fills(
        _FakeAlpaca(fills), _factory(db), since_date="2026-06-01",
        now_iso="2026-06-21T00:00:00Z")
    assert summ["dates_written"] == 1
    with _gc(db) as c:
        assert c.execute("SELECT COUNT(*) FROM bot_daily_pnl "
                         "WHERE trading_date='2026-05-01'").fetchone()[0] == 0
        assert round(c.execute("SELECT realized_pnl_usd FROM bot_daily_pnl "
                               "WHERE trading_date='2026-06-20'").fetchone()[0], 2) == -10.0


def test_reconcile_is_idempotent(tmp_path):
    db = tmp_path / "t.db"; init_db(db)
    fills = [
        _f("AAPL", "buy", 100, 10.0, "2026-06-20T14:00:00Z") | {"id": "1"},
        _f("AAPL", "sell", 100, 11.0, "2026-06-20T15:00:00Z") | {"id": "2"},
    ]
    fac = _factory(db)
    reconcile_daily_pnl_from_fills(_FakeAlpaca(fills), fac, now_iso="2026-06-21T00:00:00Z")
    reconcile_daily_pnl_from_fills(_FakeAlpaca(fills), fac, now_iso="2026-06-21T00:00:00Z")
    with _gc(db) as c:  # running twice must NOT double-count
        rows = c.execute("SELECT realized_pnl_usd FROM bot_daily_pnl "
                         "WHERE trading_date='2026-06-20'").fetchall()
    assert len(rows) == 1 and round(rows[0][0], 2) == 100.0
