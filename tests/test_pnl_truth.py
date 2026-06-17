"""Lock the P/L-accounting truth invariants (P0 rebuild). Before the fix,
_reconcile_realized_pnl overwrote realized_pnl_usd with an equity delta every
loop, clobbering the closed-trade ledger and hiding the ~$6,112 TRDA loss. These
tests fail if that clobber is ever reintroduced.

We build a LiveTrader without __init__ (no network) and exercise the two P/L
writers against a temp DB.
"""
from contextlib import contextmanager

import pytest

from market_radar.execution.live_trader import LiveTrader
from market_radar.storage.db import get_connection as real_get_connection
from market_radar.storage.db import init_db


class _FakeAcct:
    def __init__(self, equity, last_equity):
        self.equity = equity
        self.last_equity = last_equity


class _FakePos:
    def __init__(self, unrealized_pl):
        self.unrealized_pl = unrealized_pl


class _FakeAlpaca:
    def __init__(self, positions=None, fills=None):
        self._positions = positions or []
        self._fills = fills or []

    def get_positions(self):
        return list(self._positions)

    def list_account_activities(self, *, activity_type=None, page_size=100, after=None):
        return list(self._fills)


@pytest.fixture
def temp_db(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    return db


def _latest(db):
    with real_get_connection(db) as c:
        return c.execute(
            "SELECT realized_pnl_usd, trades_count, wins, losses, "
            "equity_delta_intraday_usd FROM bot_daily_pnl ORDER BY rowid DESC LIMIT 1"
        ).fetchone()


def test_update_daily_pnl_is_incremental_and_signed(temp_db):
    lt = LiveTrader.__new__(LiveTrader)
    with real_get_connection(temp_db) as conn:
        lt._update_daily_pnl(conn, 10.0)
        lt._update_daily_pnl(conn, -4.0)
    realized, trades, wins, losses, _ = _latest(temp_db)
    assert round(realized, 2) == 6.0, "realized must ACCUMULATE (10 + -4), not overwrite"
    assert trades == 2
    assert wins == 1 and losses == 1


def test_reconcile_writes_equity_delta_only_never_clobbers_realized(temp_db, monkeypatch):
    lt = LiveTrader.__new__(LiveTrader)
    lt.alpaca = _FakeAlpaca(positions=[])

    # Seed a genuine closed-trade realized value via the sole authorized writer.
    with real_get_connection(temp_db) as conn:
        lt._update_daily_pnl(conn, 25.0)

    # Redirect the method's internal get_connection() to the temp DB.
    @contextmanager
    def _temp_conn(path=None):
        with real_get_connection(temp_db) as c:
            yield c
    monkeypatch.setattr(
        "market_radar.execution.live_trader.get_connection", _temp_conn)

    # Account up +$100 vs prior close. The OLD bug would set realized = 100.
    lt._reconcile_realized_pnl(_FakeAcct(equity=100_100.0, last_equity=100_000.0))

    realized, _, _, _, equity_delta = _latest(temp_db)
    assert round(realized, 2) == 25.0, (
        "realized_pnl_usd is the closed-trade ledger and must NOT be clobbered "
        "by the equity delta (this is the exact F-grade bug)")
    assert round(equity_delta, 2) == 100.0, "equity delta belongs in its own column"


def test_reconcile_gap_alert_ignores_phantom_from_multiday_unrealized(temp_db, monkeypatch):
    """A multi-day open winner makes realized_implied = (today's equity delta) minus
    (unrealized SINCE ENTRY) swing sharply negative with ZERO trades. The unbooked-exit
    alert must NOT fire on that phantom — only when a real closing FILL occurred today
    (a genuine unbooked exit, like the TRDA loss, always leaves a fill)."""
    @contextmanager
    def _temp_conn(path=None):
        with real_get_connection(temp_db) as c:
            yield c
    monkeypatch.setattr(
        "market_radar.execution.live_trader.get_connection", _temp_conn)

    pos = _FakePos(unrealized_pl=121.0)            # winner sitting on +$121 since entry
    acct = _FakeAcct(equity=100_014.0, last_equity=100_000.0)  # only +$14 *today*
    # => realized_implied = 14 - 121 = -107 (phantom), ledger = 0, gap ~ -107.

    # No fills today -> phantom gap must be SUPPRESSED.
    lt = LiveTrader.__new__(LiveTrader)
    lt.alpaca = _FakeAlpaca(positions=[pos], fills=[])
    lt._reconcile_realized_pnl(acct)
    assert getattr(lt, "_reconcile_gap_alerted_on", None) is None, (
        "phantom gap from multi-day unrealized must not trigger the unbooked-exit alert")

    # A real closing fill today -> the alert MUST still fire (TRDA protection holds).
    lt2 = LiveTrader.__new__(LiveTrader)
    lt2.alpaca = _FakeAlpaca(positions=[pos], fills=[{"side": "sell", "symbol": "X", "qty": "1"}])
    lt2._reconcile_realized_pnl(acct)
    assert getattr(lt2, "_reconcile_gap_alerted_on", None) is not None, (
        "a real unbooked exit (closing fill present + ledger gap) must still alert")
