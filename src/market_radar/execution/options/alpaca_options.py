"""Alpaca options REST client — extends the existing stock AlpacaClient.

API surface used:
  - GET  /v2/options/contracts        — list active option contracts
  - GET  /v1beta1/options/snapshots/{symbols}  — latest quote + greeks
  - POST /v2/orders (order_class='mleg', legs=[...])  — submit multi-leg

OCC contract symbol format (used directly as Alpaca order symbol):
    <ROOT><YY><MM><DD><C|P><STRIKE_PADDED_TO_8>
e.g.  AAPL241220C00190000  = AAPL, 2024-12-20, Call, strike 190.00

Multi-leg payload example (bull call debit spread, 1 contract):

    {
      "order_class": "mleg",
      "qty": "1",
      "type": "limit",
      "time_in_force": "day",
      "limit_price": "0.45",
      "legs": [
        {"symbol": "AAPL241220C00190000", "side": "buy",
         "position_intent": "buy_to_open",  "ratio_qty": "1"},
        {"symbol": "AAPL241220C00195000", "side": "sell",
         "position_intent": "sell_to_open", "ratio_qty": "1"}
      ]
    }
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional

from ..alpaca_client import AlpacaClient, AlpacaError

log = logging.getLogger("marketradar.execution.options")

OptionType = Literal["call", "put"]
OrderSide = Literal["buy", "sell"]
PositionIntent = Literal["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"]


# ---------------------------------------------------------------------------
# Typed responses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionContract:
    symbol: str                       # OCC symbol
    name: str
    underlying_symbol: str
    expiration_date: str              # 'YYYY-MM-DD'
    type: OptionType
    strike_price: float
    open_interest: int = 0
    close_price: Optional[float] = None
    tradable: bool = True
    multiplier: int = 100


@dataclass(frozen=True)
class OptionQuote:
    symbol: str
    bid: float
    ask: float
    bid_size: int = 0
    ask_size: int = 0
    last_price: float = 0.0
    last_trade_ts: Optional[str] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    underlying_price: Optional[float] = None
    timestamp: Optional[str] = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if (self.bid > 0 and self.ask > 0) else 0.0

    @property
    def effective_mid(self) -> float:
        """Real two-sided mid if available, else the last trade price. The
        'indicative' options feed frequently returns no resting quote even on
        liquid strikes (this was ~44% of failed builds). Spread orders are
        LIMIT at the net debit, so falling back to last trade can only cause a
        no-fill or a fill at our limit-or-better — never a worse-than-quoted
        fill."""
        m = self.mid
        if m > 0:
            return m
        return self.last_price if self.last_price > 0 else 0.0

    @property
    def spread_pct(self) -> float:
        m = self.mid
        if m <= 0:
            return float("inf")
        return (self.ask - self.bid) / m


@dataclass(frozen=True)
class OptionLeg:
    """One leg of a multi-leg order."""
    symbol: str
    side: OrderSide
    position_intent: PositionIntent
    ratio_qty: int = 1


@dataclass(frozen=True)
class MultiLegOrder:
    """A submitted multi-leg spread."""
    id: str
    client_order_id: str
    status: str
    order_class: str
    legs_submitted: list[OptionLeg]
    qty: float
    limit_price: Optional[float]
    submitted_at: str
    filled_at: Optional[str] = None
    filled_avg_price: Optional[float] = None
    legs_response: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OCC symbol math
# ---------------------------------------------------------------------------

def build_occ_symbol(root: str, exp: str, kind: OptionType, strike: float) -> str:
    """Build an OCC-style option symbol.

    >>> build_occ_symbol('AAPL', '2024-12-20', 'call', 190.0)
    'AAPL241220C00190000'
    """
    d = datetime.strptime(exp, "%Y-%m-%d").date()
    yy = f"{d.year % 100:02d}"
    mm = f"{d.month:02d}"
    dd = f"{d.day:02d}"
    cp = "C" if kind == "call" else "P"
    strike_thousandths = int(round(strike * 1000))
    strike_str = f"{strike_thousandths:08d}"
    return f"{root.upper()}{yy}{mm}{dd}{cp}{strike_str}"


def parse_occ_symbol(symbol: str) -> tuple[str, str, OptionType, float]:
    """Inverse of build_occ_symbol. Returns (root, expiration_iso, type, strike)."""
    # Strike is the last 8 digits; type is char before that; date is 6 digits before that.
    strike = int(symbol[-8:]) / 1000.0
    kind: OptionType = "call" if symbol[-9] == "C" else "put"
    yy = symbol[-15:-13]
    mm = symbol[-13:-11]
    dd = symbol[-11:-9]
    year = 2000 + int(yy)
    exp = f"{year:04d}-{mm}-{dd}"
    root = symbol[:-15]
    return root, exp, kind, strike


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AlpacaOptionsClient:
    """Options API wrapper that piggybacks on the stock AlpacaClient session."""

    def __init__(self, base_client: Optional[AlpacaClient] = None):
        self.base = base_client or AlpacaClient()

    # ------------------------------------------------------------------
    # Chain fetch
    # ------------------------------------------------------------------
    def list_contracts(
        self,
        underlying: str,
        *,
        expiration_gte: Optional[str] = None,
        expiration_lte: Optional[str] = None,
        type_: Optional[OptionType] = None,
        strike_gte: Optional[float] = None,
        strike_lte: Optional[float] = None,
        status: str = "active",
        limit: int = 500,
    ) -> list[OptionContract]:
        params: dict = {
            "underlying_symbols": underlying,
            "status": status,
            "limit": limit,
        }
        if expiration_gte:
            params["expiration_date_gte"] = expiration_gte
        if expiration_lte:
            params["expiration_date_lte"] = expiration_lte
        if type_:
            params["type"] = type_
        if strike_gte is not None:
            params["strike_price_gte"] = str(strike_gte)
        if strike_lte is not None:
            params["strike_price_lte"] = str(strike_lte)

        try:
            d = self.base._request("GET", "/v2/options/contracts", params=params)
        except AlpacaError as exc:
            log.warning("list_contracts(%s) failed: %s", underlying, exc)
            return []

        items = (d or {}).get("option_contracts", []) or []
        return [
            OptionContract(
                symbol=it["symbol"],
                name=it.get("name", it["symbol"]),
                underlying_symbol=it["underlying_symbol"],
                expiration_date=it["expiration_date"],
                type=it["type"],
                strike_price=float(it["strike_price"]),
                open_interest=int(it.get("open_interest", 0) or 0),
                close_price=(
                    float(it["close_price"]) if it.get("close_price") else None
                ),
                tradable=bool(it.get("tradable", True)),
                multiplier=int(it.get("size", 100) or 100),
            )
            for it in items
        ]

    # ------------------------------------------------------------------
    # Snapshots (latest quote + greeks)
    # ------------------------------------------------------------------
    def get_snapshots(self, symbols: list[str]) -> dict[str, OptionQuote]:
        if not symbols:
            return {}
        # Alpaca snapshot endpoint accepts up to 50 symbols at a time.
        # Symbols go in a query param, not the path.
        out: dict[str, OptionQuote] = {}
        for chunk in _chunked(symbols, 50):
            try:
                d = self.base._request(
                    "GET",
                    "/v1beta1/options/snapshots",
                    params={"symbols": ",".join(chunk), "feed": "indicative"},
                    base=self.base.data_base_url,
                )
            except AlpacaError as exc:
                log.warning("get_snapshots(%s…) failed: %s", chunk[0], exc)
                continue
            snaps = (d or {}).get("snapshots", {}) or {}
            for sym, snap in snaps.items():
                q = snap.get("latestQuote") or {}
                lt = snap.get("latestTrade") or {}
                greeks = snap.get("greeks") or {}
                under = snap.get("underlyingPrice")
                out[sym] = OptionQuote(
                    symbol=sym,
                    bid=float(q.get("bp", 0) or 0),
                    ask=float(q.get("ap", 0) or 0),
                    bid_size=int(q.get("bs", 0) or 0),
                    ask_size=int(q.get("as", 0) or 0),
                    last_price=float(lt.get("p", 0) or 0),
                    last_trade_ts=lt.get("t"),
                    implied_volatility=snap.get("impliedVolatility"),
                    delta=greeks.get("delta"),
                    gamma=greeks.get("gamma"),
                    theta=greeks.get("theta"),
                    vega=greeks.get("vega"),
                    underlying_price=float(under) if under else None,
                    timestamp=q.get("t"),
                )
        return out

    # ------------------------------------------------------------------
    # Multi-leg submit
    # ------------------------------------------------------------------
    def submit_multi_leg(
        self,
        *,
        legs: list[OptionLeg],
        qty: int,
        limit_price: float,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ) -> MultiLegOrder:
        """Submit a multi-leg ('mleg') options order.

        ``limit_price`` is the **net** debit (positive for spreads we're paying
        to open) or net credit (positive number, set ``negative`` not needed
        — Alpaca infers from leg sides).
        """
        if not legs:
            raise AlpacaError("legs[] empty")
        if qty <= 0:
            raise AlpacaError(f"qty must be > 0, got {qty}")
        payload = {
            "order_class": "mleg",
            "qty": str(qty),
            "type": "limit",
            "time_in_force": time_in_force,
            "limit_price": _round_price(limit_price),
            "extended_hours": extended_hours,
            "client_order_id": client_order_id or f"mr-opt-{uuid.uuid4().hex[:14]}",
            "legs": [
                {
                    "symbol": leg.symbol,
                    "side": leg.side,
                    "position_intent": leg.position_intent,
                    "ratio_qty": str(leg.ratio_qty),
                }
                for leg in legs
            ],
        }
        d = self.base._request("POST", "/v2/orders", json=payload)
        return MultiLegOrder(
            id=d.get("id", ""),
            client_order_id=d.get("client_order_id", ""),
            status=d.get("status", "unknown"),
            order_class=d.get("order_class", "mleg"),
            legs_submitted=legs,
            qty=float(d.get("qty", qty)),
            limit_price=float(d["limit_price"]) if d.get("limit_price") else None,
            submitted_at=d.get("submitted_at", ""),
            filled_at=d.get("filled_at"),
            filled_avg_price=(
                float(d["filled_avg_price"]) if d.get("filled_avg_price") else None
            ),
            legs_response=d.get("legs", []) or [],
        )

    def close_spread(
        self,
        *,
        legs: list[OptionLeg],
        qty: int,
        limit_price: float,
        client_order_id: Optional[str] = None,
    ) -> MultiLegOrder:
        """Submit the closing multi-leg order. Flips each leg's side + position_intent."""
        closing_legs = []
        for leg in legs:
            new_side: OrderSide = "sell" if leg.side == "buy" else "buy"
            new_intent: PositionIntent = (
                "sell_to_close" if leg.side == "buy" else "buy_to_close"
            )
            closing_legs.append(OptionLeg(
                symbol=leg.symbol, side=new_side,
                position_intent=new_intent, ratio_qty=leg.ratio_qty,
            ))
        return self.submit_multi_leg(
            legs=closing_legs, qty=qty, limit_price=limit_price,
            client_order_id=client_order_id or f"mr-opt-close-{uuid.uuid4().hex[:10]}",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_price(p: float) -> str:
    """Options prices use $0.01 ticks for premiums >= $3, $0.05 below.

    Alpaca will reject non-conforming ticks. We round conservatively to
    the wider tick to ensure acceptance."""
    p = float(p)
    if abs(p) >= 3.0:
        return f"{p:.2f}"
    # 0.05 tick
    return f"{round(p * 20) / 20:.2f}"


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
