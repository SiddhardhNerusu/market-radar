"""Typed response models for the T212 client.

Field names mirror what the T212 API actually returns (verified live, not
guessed from docs). Unknown fields are preserved in ``raw`` so we can
re-process historical snapshots later without losing data when T212
adds new fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


@dataclass
class AccountCashBreakdown:
    available_to_trade: Optional[float]
    reserved_for_orders: Optional[float]
    in_pies: Optional[float]


@dataclass
class AccountInvestments:
    current_value: Optional[float]
    total_cost: Optional[float]
    realized_profit_loss: Optional[float]
    unrealized_profit_loss: Optional[float]


@dataclass
class AccountInfo:
    """Response of GET /api/v0/equity/account/summary."""
    id: Optional[int]
    currency: Optional[str]
    total_value: Optional[float]
    cash: Optional[AccountCashBreakdown]
    investments: Optional[AccountInvestments]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountCash:
    """Response of GET /api/v0/equity/account/cash."""
    free: Optional[float]
    total: Optional[float]
    invested: Optional[float]
    ppl: Optional[float]              # unrealized P/L on open positions
    result: Optional[float]           # realized profit/loss to date
    pie_cash: Optional[float]
    blocked: Optional[float]          # margin / pending order reserve
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@dataclass
class Instrument:
    ticker: str
    name: Optional[str]
    isin: Optional[str]
    currency: Optional[str]


@dataclass
class WalletImpact:
    currency: Optional[str]
    total_cost: Optional[float]
    current_value: Optional[float]
    unrealized_profit_loss: Optional[float]
    fx_impact: Optional[float]


@dataclass
class Position:
    """Single open position from GET /api/v0/equity/positions."""
    ticker: str                                # convenience copy of instrument.ticker
    name: Optional[str]
    isin: Optional[str]
    currency: Optional[str]
    quantity: float
    quantity_available_for_trading: Optional[float]
    quantity_in_pies: Optional[float]
    average_price_paid: float
    current_price: Optional[float]
    created_at: Optional[str]
    wallet_impact: Optional[WalletImpact]
    instrument: Optional[Instrument]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> Optional[float]:
        if self.current_price is None:
            return None
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> Optional[float]:
        if self.wallet_impact is not None:
            return self.wallet_impact.unrealized_profit_loss
        return None


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """Open order from GET /api/v0/equity/orders.

    Field names left in T212's camelCase for any keys we haven't verified live
    yet (no open orders during initial inspection). Will be tightened as we
    observe real responses.
    """
    id: Optional[int]
    ticker: str
    quantity: float
    type: Optional[str]              # MARKET / LIMIT / STOP / STOP_LIMIT
    status: Optional[str]
    creation_time: Optional[str]
    filled_quantity: Optional[float]
    filled_value: Optional[float]
    limit_price: Optional[float]
    stop_price: Optional[float]
    strategy: Optional[str]           # QUANTITY / VALUE
    value: Optional[float]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoricalOrder:
    """Historical order from GET /api/v0/equity/history/orders."""
    id: Optional[int]
    ticker: str
    type: Optional[str]
    status: Optional[str]
    ordered_quantity: Optional[float]
    filled_quantity: Optional[float]
    limit_price: Optional[float]
    stop_price: Optional[float]
    fill_price: Optional[float]
    date_created: Optional[str]
    date_executed: Optional[str]
    date_modified: Optional[str]
    fill_type: Optional[str]
    taxes: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dividend:
    ticker: str
    reference: Optional[str]
    amount: Optional[float]
    quantity: Optional[float]
    gross_amount_per_share: Optional[float]
    amount_in_euro: Optional[float]
    paid_on: Optional[str]
    type: Optional[str]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CashTransaction:
    reference: Optional[str]
    type: Optional[str]
    amount: Optional[float]
    date_time: Optional[str]
    raw: dict[str, Any] = field(default_factory=dict)
