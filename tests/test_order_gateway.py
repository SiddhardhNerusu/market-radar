"""Regression tests that LOCK the capital-preservation invariants of the order
submission choke-point. These exist because of the 2026-06-10 TRDA death spiral.
If any of these fail, the bot must NOT trade.
"""
import pytest

from market_radar.execution.order_gateway import (
    CLOSE,
    OPEN,
    GateCaps,
    OrderGateway,
    plan_order,
)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _Pos:
    def __init__(self, symbol, qty, market_value):
        self.symbol = symbol
        self.qty = qty
        self.market_value = market_value


class _Order:
    def __init__(self, oid="x"):
        self.id = oid


class FakeAlpaca:
    """Records every submitted order and serves a mutable positions snapshot."""

    def __init__(self, positions=None, raise_on_positions=False):
        self._positions = list(positions or [])
        self.raise_on_positions = raise_on_positions
        self.submitted = []  # list of kwargs dicts

    def get_positions(self):
        if self.raise_on_positions:
            raise RuntimeError("alpaca down")
        return list(self._positions)

    def submit_simple_order(self, **kwargs):
        self.submitted.append(kwargs)
        return _Order(f"o{len(self.submitted)}")

    # test helper: apply a fill to the live snapshot
    def _apply_fill(self, symbol, signed_qty, price):
        for p in self._positions:
            if p.symbol == symbol:
                p.qty += signed_qty
                p.market_value = p.qty * price
                if abs(p.qty) < 1e-9:
                    self._positions.remove(p)
                return
        if abs(signed_qty) > 1e-9:
            self._positions.append(_Pos(symbol, signed_qty, signed_qty * price))


CAPS = GateCaps(
    gross_cap_usd=6000.0,
    per_symbol_cap_usd=1000.0,
    hard_notional_cap_usd=2000.0,
    allow_stock_shorts=False,
)


# --------------------------------------------------------------------------- #
# CLOSE invariants (the TRDA core)
# --------------------------------------------------------------------------- #
def test_close_long_sells_to_flat():
    d = plan_order(intent=CLOSE, symbol="AAA", requested_side="sell", requested_qty=100,
                   positions=[_Pos("AAA", 100, 1000)], caps=CAPS)
    assert d.allowed and d.side == "sell" and d.qty == 100
    assert d.position_intent == "sell_to_close"


def test_close_short_BUYS_to_flat_even_if_caller_says_sell():
    """The exact TRDA bug: caller's stale direction would SELL a position that
    has flipped short. The gateway MUST derive BUY from the live sign."""
    d = plan_order(intent=CLOSE, symbol="TRDA", requested_side="sell", requested_qty=63,
                   positions=[_Pos("TRDA", -63, -350)], caps=CAPS)
    assert d.allowed
    assert d.side == "buy", "closing a SHORT must BUY, never sell (would grow the short)"
    assert d.qty == 63
    assert d.position_intent == "buy_to_close"


def test_close_clamps_qty_to_live_position():
    """A caller asking to close more than exists can never over-shoot into a flip."""
    d = plan_order(intent=CLOSE, symbol="AAA", requested_side="sell", requested_qty=99999,
                   positions=[_Pos("AAA", 100, 1000)], caps=CAPS)
    assert d.allowed and d.qty == 100


def test_close_flat_is_noop():
    d = plan_order(intent=CLOSE, symbol="AAA", requested_side="sell", requested_qty=100,
                   positions=[], caps=CAPS)
    assert not d.allowed and "flat" in d.reason


def test_trda_death_spiral_cannot_recur():
    """End-to-end: a short that the buggy code doubled 63->126->...->16128.
    Drive repeated 'close' calls (with the OLD stale side='sell') through the
    gateway and prove the short only ever SHRINKS, with total BUYs == 63."""
    fake = FakeAlpaca(positions=[_Pos("TRDA", -63, -63 * 5.6)])
    gw = OrderGateway(fake, CAPS)
    bought = 0.0
    for _ in range(10):  # the buggy loop ran ~17 times, doubling each time
        order = gw.submit(intent=CLOSE, symbol="TRDA", side="sell", qty=63, ref_price=5.6)
        if order is None:
            break  # flat -> no-op, loop ends cleanly
        last = fake.submitted[-1]
        assert last["side"] == "buy", "every close of a short must BUY"
        bought += last["qty"]
        fake._apply_fill("TRDA", +last["qty"], 5.6)  # simulate the cover fill
    assert bought == 63, f"covered exactly the short, never doubled (bought={bought})"
    # And the position is flat, not a 16,128-share monster.
    assert fake.get_positions() == []
    # No SELL was ever sent on TRDA.
    assert all(o["side"] == "buy" for o in fake.submitted)


# --------------------------------------------------------------------------- #
# OPEN invariants
# --------------------------------------------------------------------------- #
def test_open_long_within_caps_ok():
    d = plan_order(intent=OPEN, symbol="AAA", requested_side="buy", requested_qty=50,
                   ref_price=10, positions=[], caps=CAPS)  # $500 notional
    assert d.allowed and d.side == "buy" and d.position_intent == "buy_to_open"


def test_open_stock_short_blocked_long_only():
    d = plan_order(intent=OPEN, symbol="AAA", requested_side="sell", requested_qty=10,
                   ref_price=10, positions=[], caps=CAPS)
    assert not d.allowed and "SHORT" in d.reason


def test_open_stock_short_allowed_when_flag_set():
    caps = GateCaps(6000, 1000, 2000, allow_stock_shorts=True)
    d = plan_order(intent=OPEN, symbol="AAA", requested_side="sell", requested_qty=10,
                   ref_price=10, positions=[], caps=caps)
    assert d.allowed and d.side == "sell" and d.position_intent == "sell_to_open"


def test_open_blocked_by_hard_notional_cap():
    # 16,128 sh * $5.60 ~= $90k — the TRDA order. Hard cap = $2k.
    d = plan_order(intent=OPEN, symbol="TRDA", requested_side="buy", requested_qty=16128,
                   ref_price=5.60, positions=[], caps=CAPS)
    assert not d.allowed and "hard cap" in d.reason


def test_open_blocked_by_per_symbol_cap():
    d = plan_order(intent=OPEN, symbol="AAA", requested_side="buy", requested_qty=150,
                   ref_price=10, positions=[], caps=CAPS)  # $1,500 > $1,000 per-symbol
    assert not d.allowed and "per-symbol" in d.reason


def test_open_blocked_by_aggregate_gross_cap():
    # Already $5,800 gross; a $500 add would breach the $6,000 gross cap.
    existing = [_Pos("X", 580, 5800)]
    d = plan_order(intent=OPEN, symbol="AAA", requested_side="buy", requested_qty=50,
                   ref_price=10, positions=existing, caps=CAPS)
    assert not d.allowed and "gross" in d.reason


def test_crypto_short_not_blocked_by_long_only():
    # Crypto pairs contain '/'; spot-only, so the long-only guard doesn't apply.
    d = plan_order(intent=OPEN, symbol="BTC/USD", requested_side="sell", requested_qty=1,
                   ref_price=100, positions=[], caps=CAPS)
    # blocked by something else? no — within caps and not a stock short.
    assert d.allowed


# --------------------------------------------------------------------------- #
# Gateway I/O behaviour
# --------------------------------------------------------------------------- #
def test_gateway_fails_closed_when_positions_unreadable():
    fake = FakeAlpaca(raise_on_positions=True)
    gw = OrderGateway(fake, CAPS)
    order = gw.submit(intent=CLOSE, symbol="AAA", side="sell", qty=10, ref_price=10)
    assert order is None
    assert fake.submitted == [], "nothing may be submitted when positions can't be read"


def test_gateway_blocked_open_submits_nothing():
    fake = FakeAlpaca(positions=[])
    gw = OrderGateway(fake, CAPS)
    order = gw.submit(intent=OPEN, symbol="TRDA", side="buy", qty=16128, ref_price=5.6)
    assert order is None and fake.submitted == []


def test_gateway_allowed_open_submits_with_intent():
    fake = FakeAlpaca(positions=[])
    gw = OrderGateway(fake, CAPS)
    order = gw.submit(intent=OPEN, symbol="AAA", side="buy", qty=50, ref_price=10)
    assert order is not None
    assert fake.submitted[0]["position_intent"] == "buy_to_open"
    assert fake.submitted[0]["side"] == "buy" and fake.submitted[0]["qty"] == 50


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
