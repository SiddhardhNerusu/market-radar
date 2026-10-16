"""Fixed-fractional risk sizing against a REAL stop. Pure, unit-tested.

The old sizer called itself quarter-Kelly, fed it a hard-coded p=0.70 and
always hit the 5% notional cap; every multiplier on top was a no-op. This
one asks the only question that matters for survival: how many shares can
I buy so that if the broker-side stop fires I lose ``risk_pct`` of the book?
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EntryPlan:
    ok: bool
    reason: str
    qty: int = 0
    limit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_usd: float = 0.0
    notional_usd: float = 0.0
    spread_bps: float = 0.0


def plan_entry(
    *,
    bid: float,
    ask: float,
    atr: float,
    book_equity: float,
    risk_pct: float,
    stop_atr_mult: float,
    tp_atr_mult: float,
    max_position_pct: float,
    limit_bps: float,
    max_spread_bps: float,
    min_price: float,
    open_heat_usd: float,
    max_heat_pct: float,
    hard_order_cap_usd: float,
) -> EntryPlan:
    if bid <= 0 or ask <= 0 or ask < bid:
        return EntryPlan(False, f"bad quote bid={bid} ask={ask}")
    mid = (bid + ask) / 2.0
    spread_bps = (ask - bid) / mid * 1e4
    if spread_bps > max_spread_bps:
        return EntryPlan(False, f"spread {spread_bps:.0f}bps > cap {max_spread_bps:.0f}bps",
                         spread_bps=spread_bps)
    if ask < min_price:
        return EntryPlan(False, f"price ${ask:.2f} < floor ${min_price:.2f}", spread_bps=spread_bps)
    if atr <= 0:
        return EntryPlan(False, "no ATR", spread_bps=spread_bps)
    if book_equity <= 0:
        return EntryPlan(False, "book equity <= 0", spread_bps=spread_bps)

    limit = round(ask * (1.0 + limit_bps / 1e4), 2)
    stop = round(limit - stop_atr_mult * atr, 2)
    target = round(limit + tp_atr_mult * atr, 2)
    if stop <= 0 or not (stop < limit < target):
        return EntryPlan(False, f"degenerate bracket stop={stop} limit={limit} tp={target}",
                         spread_bps=spread_bps)
    risk_per_share = limit - stop
    if risk_per_share / limit > 0.25:
        return EntryPlan(False, f"stop {risk_per_share / limit:.0%} away — too volatile for a 2xATR stop",
                         spread_bps=spread_bps)

    risk_budget = book_equity * risk_pct / 100.0
    qty = math.floor(risk_budget / risk_per_share)
    qty = min(qty, math.floor(book_equity * max_position_pct / 100.0 / limit))
    qty = min(qty, math.floor(hard_order_cap_usd / limit))
    heat_room = book_equity * max_heat_pct / 100.0 - open_heat_usd
    if heat_room <= 0:
        return EntryPlan(False, f"portfolio heat ${open_heat_usd:.0f} at cap", spread_bps=spread_bps)
    qty = min(qty, math.floor(heat_room / risk_per_share))
    if qty < 1:
        return EntryPlan(False, "qty < 1 after risk / notional / heat caps", spread_bps=spread_bps)
    return EntryPlan(
        True, "sized", qty=int(qty), limit_price=limit, stop_loss=stop, take_profit=target,
        risk_usd=round(qty * risk_per_share, 2), notional_usd=round(qty * limit, 2),
        spread_bps=spread_bps,
    )
