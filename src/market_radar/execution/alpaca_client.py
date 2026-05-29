"""Alpaca REST trading API client.

Raw HTTP via ``requests`` — no SDK dependency. Targets the v2 Trading API,
which is identical between paper and live (only the base URL differs).

Docs:
    https://docs.alpaca.markets/reference/getaccount-1
    https://docs.alpaca.markets/reference/postorder

The base URL is read from ``ALPACA_BASE_URL`` (defaults to paper). The
client is read/write — placing a bracket order is one round-trip that
creates the parent + stop_loss + take_profit legs server-side, so even
if this process dies, Alpaca enforces the stops.

All errors raise ``AlpacaError`` (or a subclass) with the response body
attached so callers can log-and-continue or abort cleanly.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import requests

from ..config import CONFIG

log = logging.getLogger("marketradar.execution.alpaca")

OrderSide = Literal["buy", "sell"]
TimeInForce = Literal["day", "gtc", "opg", "cls", "ioc", "fok"]
OrderClass = Literal["simple", "bracket", "oco", "oto"]


class AlpacaError(Exception):
    """Raised on any non-2xx response or transport error."""

    def __init__(self, message: str, status: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class AlpacaAuthError(AlpacaError):
    """Raised on 401/403 — bad or missing credentials."""


class AlpacaRateLimitError(AlpacaError):
    """Raised after exhausting retries on a 429."""


# ---------------------------------------------------------------------------
# Typed responses (minimal — only the fields we actually use)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Account:
    id: str
    equity: float
    last_equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    long_market_value: float
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool
    daytrade_count: int
    currency: str
    status: str

    @property
    def is_tradeable(self) -> bool:
        return (
            self.status == "ACTIVE"
            and not self.trading_blocked
            and not self.account_blocked
        )

    @property
    def daily_pnl_pct(self) -> float:
        """(equity - last_equity) / last_equity, where last_equity is yesterday's
        close. Approximate intraday P&L %. Negative when down on the day."""
        if self.last_equity <= 0:
            return 0.0
        return (self.equity - self.last_equity) / self.last_equity


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    side: str                  # 'long' or 'short'
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float


@dataclass(frozen=True)
class Order:
    id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    qty: float
    filled_qty: float
    status: str                # 'new','filled','partially_filled','canceled',…
    order_class: str
    order_type: str
    time_in_force: str
    submitted_at: str
    filled_at: Optional[str]
    canceled_at: Optional[str]
    filled_avg_price: Optional[float]
    limit_price: Optional[float]
    stop_price: Optional[float]
    legs: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class BracketOrder:
    """A submitted bracket order with both protective legs."""

    parent: Order
    take_profit_leg: Optional[Order]
    stop_loss_leg: Optional[Order]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AlpacaClient:
    """HTTP client for the Alpaca v2 Trading API."""

    DEFAULT_TIMEOUT = 15  # seconds

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        data_base_url: str = "https://data.alpaca.markets",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ):
        key = (api_key or CONFIG.alpaca_api_key).strip()
        secret = (api_secret or CONFIG.alpaca_api_secret).strip()
        if not key or not secret:
            raise AlpacaAuthError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY + "
                "ALPACA_API_SECRET in .env."
            )
        self._key = key
        self._secret = secret
        self.base_url = (base_url or CONFIG.alpaca_base_url).rstrip("/")
        self.data_base_url = data_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
                "Accept": "application/json",
            }
        )
        log.info(
            "AlpacaClient initialised: base=%s mode=%s",
            self.base_url,
            "PAPER" if "paper" in self.base_url else "LIVE",
        )

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
        base: Optional[str] = None,
    ) -> Any:
        url = f"{base or self.base_url}{path}"
        backoff = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                log.warning("Alpaca %s %s transport error: %s (attempt %d)",
                            method, path, exc, attempt + 1)
                time.sleep(backoff)
                backoff *= 2
                continue

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra else backoff
                log.warning("Alpaca rate-limited; sleeping %.1fs", wait)
                time.sleep(wait)
                backoff *= 2
                continue

            if r.status_code in (401, 403):
                raise AlpacaAuthError(
                    f"Alpaca auth failed ({r.status_code}) for {method} {path}",
                    status=r.status_code, body=_safe_body(r),
                )

            if 200 <= r.status_code < 300:
                if r.status_code == 204 or not r.content:
                    return None
                try:
                    return r.json()
                except ValueError:
                    return r.text

            # 4xx other than auth / 429 — non-retryable.
            if 400 <= r.status_code < 500:
                raise AlpacaError(
                    f"Alpaca {method} {path} failed: {r.status_code} {_safe_body(r)}",
                    status=r.status_code, body=_safe_body(r),
                )

            # 5xx — retry with backoff.
            log.warning("Alpaca %s %s returned %d (attempt %d)",
                        method, path, r.status_code, attempt + 1)
            last_exc = AlpacaError(
                f"Alpaca server error {r.status_code}",
                status=r.status_code, body=_safe_body(r),
            )
            time.sleep(backoff)
            backoff *= 2

        if isinstance(last_exc, AlpacaError):
            raise last_exc
        raise AlpacaRateLimitError(
            f"Alpaca {method} {path} failed after {self.max_retries + 1} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Account + portfolio
    # ------------------------------------------------------------------

    def get_account(self) -> Account:
        d = self._request("GET", "/v2/account")
        return Account(
            id=d["id"],
            equity=float(d["equity"]),
            last_equity=float(d["last_equity"]),
            cash=float(d["cash"]),
            buying_power=float(d["buying_power"]),
            portfolio_value=float(d["portfolio_value"]),
            long_market_value=float(d.get("long_market_value", 0) or 0),
            pattern_day_trader=bool(d.get("pattern_day_trader", False)),
            trading_blocked=bool(d.get("trading_blocked", False)),
            account_blocked=bool(d.get("account_blocked", False)),
            daytrade_count=int(d.get("daytrade_count", 0) or 0),
            currency=d.get("currency", "USD"),
            status=d.get("status", "UNKNOWN"),
        )

    def get_positions(self) -> list[Position]:
        items = self._request("GET", "/v2/positions") or []
        return [
            Position(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                side=p["side"],
                avg_entry_price=float(p["avg_entry_price"]),
                market_value=float(p["market_value"]),
                unrealized_pl=float(p["unrealized_pl"]),
                unrealized_plpc=float(p["unrealized_plpc"]),
                current_price=float(p.get("current_price", 0) or 0),
            )
            for p in items
        ]

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            p = self._request("GET", f"/v2/positions/{symbol}")
        except AlpacaError as exc:
            if exc.status == 404:
                return None
            raise
        return Position(
            symbol=p["symbol"],
            qty=float(p["qty"]),
            side=p["side"],
            avg_entry_price=float(p["avg_entry_price"]),
            market_value=float(p["market_value"]),
            unrealized_pl=float(p["unrealized_pl"]),
            unrealized_plpc=float(p["unrealized_plpc"]),
            current_price=float(p.get("current_price", 0) or 0),
        )

    # ------------------------------------------------------------------
    # Market data (IEX feed is free with any account; SIP requires sub).
    # ------------------------------------------------------------------

    def get_latest_quote(self, symbol: str) -> Optional[tuple[float, float]]:
        """Return ``(bid, ask)`` for ``symbol``, or None if unavailable.
        Routes to the crypto endpoint when symbol contains '/'."""
        if "/" in symbol:
            return self.get_latest_crypto_quote(symbol)
        try:
            d = self._request(
                "GET",
                f"/v2/stocks/{symbol}/quotes/latest",
                params={"feed": "iex"},
                base=self.data_base_url,
            )
        except AlpacaError as exc:
            log.warning("get_latest_quote(%s) failed: %s", symbol, exc)
            return None
        q = (d or {}).get("quote") or {}
        bid = float(q.get("bp", 0) or 0)
        ask = float(q.get("ap", 0) or 0)
        if bid <= 0 or ask <= 0:
            return None
        return bid, ask

    def get_latest_trade(self, symbol: str) -> Optional[float]:
        """Return last trade price for ``symbol``, or None.
        Routes to the crypto endpoint when symbol contains '/'."""
        if "/" in symbol:
            return self.get_latest_crypto_trade(symbol)
        try:
            d = self._request(
                "GET",
                f"/v2/stocks/{symbol}/trades/latest",
                params={"feed": "iex"},
                base=self.data_base_url,
            )
        except AlpacaError as exc:
            log.warning("get_latest_trade(%s) failed: %s", symbol, exc)
            return None
        t = (d or {}).get("trade") or {}
        p = float(t.get("p", 0) or 0)
        return p if p > 0 else None

    def get_latest_crypto_quote(self, symbol: str) -> Optional[tuple[float, float]]:
        """Crypto quote (bid, ask) from Alpaca's /v1beta3 crypto endpoint."""
        try:
            d = self._request(
                "GET", "/v1beta3/crypto/us/latest/quotes",
                params={"symbols": symbol}, base=self.data_base_url,
            )
        except AlpacaError as exc:
            log.warning("get_latest_crypto_quote(%s) failed: %s", symbol, exc)
            return None
        q = (d or {}).get("quotes", {}).get(symbol) or {}
        bid = float(q.get("bp", 0) or 0)
        ask = float(q.get("ap", 0) or 0)
        if bid <= 0 or ask <= 0:
            return None
        return bid, ask

    def get_latest_crypto_trade(self, symbol: str) -> Optional[float]:
        """Crypto last-trade price from Alpaca's /v1beta3 crypto endpoint."""
        try:
            d = self._request(
                "GET", "/v1beta3/crypto/us/latest/trades",
                params={"symbols": symbol}, base=self.data_base_url,
            )
        except AlpacaError as exc:
            log.warning("get_latest_crypto_trade(%s) failed: %s", symbol, exc)
            return None
        t = (d or {}).get("trades", {}).get(symbol) or {}
        p = float(t.get("p", 0) or 0)
        return p if p > 0 else None

    def get_market_clock(self) -> dict:
        """Return Alpaca's clock dict: ``is_open``, ``next_open``, ``next_close``."""
        return self._request("GET", "/v2/clock")

    def is_market_open(self) -> bool:
        try:
            return bool(self.get_market_clock().get("is_open", False))
        except AlpacaError as exc:
            log.warning("Could not fetch market clock: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def list_orders(
        self,
        *,
        status: str = "all",
        limit: int = 100,
        after: Optional[str] = None,
        nested: bool = True,
    ) -> list[Order]:
        params = {"status": status, "limit": limit, "nested": str(nested).lower()}
        if after:
            params["after"] = after
        items = self._request("GET", "/v2/orders", params=params) or []
        return [_to_order(d) for d in items]

    def get_order(self, order_id: str) -> Order:
        return _to_order(self._request("GET", f"/v2/orders/{order_id}",
                                       params={"nested": "true"}))

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"/v2/orders/{order_id}")

    def cancel_all_orders(self) -> int:
        items = self._request("DELETE", "/v2/orders") or []
        return len(items)

    def close_position(self, symbol: str, *, percentage: float = 100.0) -> Optional[Order]:
        try:
            d = self._request(
                "DELETE",
                f"/v2/positions/{symbol}",
                params={"percentage": percentage},
            )
        except AlpacaError as exc:
            if exc.status == 404:
                return None
            raise
        return _to_order(d) if d else None

    def submit_simple_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = "gtc",
        client_order_id: Optional[str] = None,
    ) -> Order:
        """Submit a simple non-bracket order. Used for crypto (Alpaca does
        NOT support bracket orders on crypto) and any other case where the
        bot manages stop/TP itself via polling."""
        if qty <= 0:
            raise AlpacaError(f"qty must be > 0, got {qty}")
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": client_order_id or f"mr-s-{uuid.uuid4().hex[:18]}",
        }
        if limit_price is not None:
            payload["limit_price"] = _round_price(limit_price)
        d = self._request("POST", "/v2/orders", json=payload)
        return _to_order(d)

    def submit_bracket_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        take_profit: float,
        stop_loss: float,
        stop_loss_limit: Optional[float] = None,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = "day",
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ) -> BracketOrder:
        """Submit a one-shot bracket order.

        The parent leg is a marketable limit (or plain market if ``limit_price``
        is None). Once filled, Alpaca auto-arms the two child legs:
          * take_profit: limit @ ``take_profit``
          * stop_loss:   stop @ ``stop_loss`` (optionally stop-limit at
                         ``stop_loss_limit``)

        Either leg filling cancels the other (OCO).
        """
        if qty <= 0:
            raise AlpacaError(f"qty must be > 0, got {qty}")
        # Sanity-check the brackets against direction so we never submit an
        # order where the stop would trigger immediately.
        if side == "buy":
            if take_profit <= stop_loss:
                raise AlpacaError(
                    f"buy bracket invalid: TP {take_profit} <= SL {stop_loss}"
                )
        else:
            if take_profit >= stop_loss:
                raise AlpacaError(
                    f"sell bracket invalid: TP {take_profit} >= SL {stop_loss}"
                )

        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "type": "limit" if limit_price else "market",
            "time_in_force": time_in_force,
            "order_class": "bracket",
            "extended_hours": extended_hours,
            "client_order_id": client_order_id or f"mr-{uuid.uuid4().hex[:18]}",
            "take_profit": {"limit_price": _round_price(take_profit)},
            "stop_loss": {"stop_price": _round_price(stop_loss)},
        }
        if limit_price is not None:
            payload["limit_price"] = _round_price(limit_price)
        if stop_loss_limit is not None:
            payload["stop_loss"]["limit_price"] = _round_price(stop_loss_limit)

        d = self._request("POST", "/v2/orders", json=payload)
        parent = _to_order(d)
        legs = parent.legs or []
        tp_leg = next(
            (_to_order(L) for L in legs if L.get("order_type") == "limit"
             or (L.get("type") == "limit" and L.get("side") != side)),
            None,
        )
        sl_leg = next(
            (_to_order(L) for L in legs if L.get("order_type", "").startswith("stop")
             or (L.get("type", "").startswith("stop"))),
            None,
        )
        return BracketOrder(parent=parent, take_profit_leg=tp_leg, stop_loss_leg=sl_leg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_body(r: requests.Response) -> Any:
    try:
        return r.json()
    except ValueError:
        return (r.text or "")[:500]


def _round_price(p: float) -> str:
    """Alpaca rejects prices with > 4 decimals (or > 2 for prices >= $1)."""
    p = float(p)
    return f"{p:.2f}" if p >= 1.0 else f"{p:.4f}"


def _to_order(d: dict) -> Order:
    if d is None:
        return None  # type: ignore[return-value]
    return Order(
        id=d.get("id", ""),
        client_order_id=d.get("client_order_id", ""),
        symbol=d.get("symbol", ""),
        side=d.get("side", "buy"),  # type: ignore[arg-type]
        qty=float(d.get("qty", 0) or 0),
        filled_qty=float(d.get("filled_qty", 0) or 0),
        status=d.get("status", "unknown"),
        order_class=d.get("order_class", "simple"),
        order_type=d.get("type") or d.get("order_type", "market"),
        time_in_force=d.get("time_in_force", "day"),
        submitted_at=d.get("submitted_at", ""),
        filled_at=d.get("filled_at"),
        canceled_at=d.get("canceled_at"),
        filled_avg_price=(
            float(d["filled_avg_price"]) if d.get("filled_avg_price") else None
        ),
        limit_price=float(d["limit_price"]) if d.get("limit_price") else None,
        stop_price=float(d["stop_price"]) if d.get("stop_price") else None,
        legs=d.get("legs") or [],
    )
