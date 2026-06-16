"""The single order-submission choke-point for the paths that can short/flip.

The 2026-06-10 TRDA death spiral (a long stock that flipped short, then every
"sell to close" *added* to the short, doubling 63 -> 16,128 shares ~$90k on a
~$6.3k book) happened because the stock exit/refill paths called
``alpaca.submit_simple_order`` directly and bypassed the risk gate. A
symptom-site patch is not enough; the invariant must live in ONE place.

ROUTED THROUGH THIS GATEWAY (where a short/flip/oversize is possible):
  - stock entry, stock refill, stock RTH/extended-hours exit, crypto entry.

NOT routed (each reduce-only BY CONSTRUCTION — documented exceptions, not gaps):
  - crypto SL/TP exit & flip, the weekend crypto trim: Alpaca spot crypto cannot
    be shorted (CRYPTO-NO-SHORT entry guard), so a held position is always long
    and a close is always a sell-to-close that reduces toward flat.
  - EOD flatten / daily-flatten / profit-take: close via Alpaca ``close_position``
    (inherently reduce-only) or an explicit live-sign ``position_intent``.
  - the server-side trailing-stop arm: a broker-managed reducing stop.
A future refactor SHOULD centralise these too; until then they are safe but the
"one choke-point" guarantee holds only for the routed paths above.

Invariants (locked by tests/test_order_gateway.py):

  CLOSE (buy_to_close / sell_to_close)
    - Side and quantity are DERIVED FROM THE LIVE POSITION SIGN and clamped to
      ``|position|``. A close can NEVER flip a position or grow it. To reduce a
      LONG you SELL; to reduce a SHORT you BUY. If the symbol is already flat,
      the close is a no-op (refused, nothing submitted).

  OPEN (buy_to_open / sell_to_open)
    - Long-only: a stock SELL (short) is refused unless ``allow_stock_shorts``.
    - Aggregate gross: refuse if (current gross |market_value| + new notional)
      would exceed ``gross_cap_usd``.
    - Per-symbol notional ceiling.
    - Absolute hard-dollar ceiling on any single order, independent of any
      equity override/multiplier — the final backstop that would have capped
      the TRDA order regardless of every other check.

Fail-closed everywhere: if live positions cannot be read, NOTHING is submitted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger("marketradar.execution.gateway")

# Intents
OPEN = "open"
CLOSE = "close"

# $1 rounding slack so float noise doesn't trip a cap.
_SLACK = 1.0


@dataclass(frozen=True)
class GateCaps:
    """Capital-preservation limits enforced on every submit.

    gross_cap_usd:        max aggregate |market_value| across ALL positions.
    per_symbol_cap_usd:   max notional for a single new position.
    hard_notional_cap_usd: absolute ceiling on ANY single order, env-independent.
    allow_stock_shorts:   if False, a stock SELL-to-open is refused (long-only).
    """
    gross_cap_usd: float
    per_symbol_cap_usd: float
    hard_notional_cap_usd: float
    allow_stock_shorts: bool = False


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    side: Optional[str] = None
    qty: float = 0.0
    position_intent: Optional[str] = None


def is_stock(symbol: str) -> bool:
    """Alpaca crypto symbols are pairs containing '/'. Everything else is a
    US equity (or single-leg option, which we also treat as non-shortable here
    via its own explicit position_intent paths)."""
    return "/" not in str(symbol)


def _norm_positions(positions: Any) -> list[tuple[str, float, float]]:
    """Normalise positions (objects or dicts) to (SYMBOL, qty, market_value)."""
    out: list[tuple[str, float, float]] = []
    for p in positions or []:
        if isinstance(p, dict):
            sym, qty, mv = p.get("symbol"), p.get("qty"), p.get("market_value")
        else:
            sym = getattr(p, "symbol", None)
            qty = getattr(p, "qty", None)
            mv = getattr(p, "market_value", None)
        if sym is None:
            continue
        try:
            qty = float(qty) if qty is not None else 0.0
        except (TypeError, ValueError):
            qty = 0.0
        try:
            mv = float(mv) if mv is not None else 0.0
        except (TypeError, ValueError):
            mv = 0.0
        out.append((str(sym).upper(), qty, mv))
    return out


def plan_order(
    *,
    intent: str,
    symbol: str,
    requested_side: Optional[str] = None,
    requested_qty: float = 0.0,
    ref_price: Optional[float] = None,
    positions: Any = None,
    caps: GateCaps,
) -> GateDecision:
    """Pure decision function — no I/O. Given the LIVE positions snapshot and
    the caps, decide whether/how the order may be submitted. This is the heart
    of the safety guarantee and is unit-tested exhaustively."""
    symbol = str(symbol).upper()
    poss = _norm_positions(positions)

    # ----------------------------------------------------------------- CLOSE
    if intent == CLOSE:
        pos_qty = next((q for (s, q, _mv) in poss if s == symbol), 0.0)
        if abs(pos_qty) < 1e-9:
            return GateDecision(False, f"close refused: {symbol} is already flat")
        # Side & qty from the LIVE sign — the TRDA invariant. Reduce toward flat.
        side = "sell" if pos_qty > 0 else "buy"
        qty = abs(pos_qty)
        if requested_qty and float(requested_qty) > 0:
            qty = min(qty, float(requested_qty))  # never exceed |live position|
        if qty <= 0:
            return GateDecision(False, f"close refused: computed qty {qty} <= 0")
        pintent = "sell_to_close" if side == "sell" else "buy_to_close"
        return GateDecision(True, "ok", side=side, qty=qty, position_intent=pintent)

    # ------------------------------------------------------------------ OPEN
    if intent != OPEN:
        return GateDecision(False, f"open refused: unknown intent {intent!r}")

    side = (requested_side or "").lower()
    if side not in ("buy", "sell"):
        return GateDecision(False, f"open refused: invalid side {requested_side!r}")
    if requested_qty is None or float(requested_qty) <= 0:
        return GateDecision(False, f"open refused: qty {requested_qty} must be > 0")
    qty = float(requested_qty)

    # Long-only assertion (the TRDA vehicle was a short). Crypto is spot-only
    # on Alpaca and cannot short, so this only constrains equities.
    if is_stock(symbol) and side == "sell" and not caps.allow_stock_shorts:
        return GateDecision(
            False,
            f"open refused: stock SHORT on {symbol} blocked (long-only; "
            f"set LIVE_ALLOW_STOCK_SHORTS=1 to permit shorts)",
        )

    if ref_price is None or float(ref_price) <= 0:
        return GateDecision(
            False, f"open refused: no positive ref_price for notional check ({ref_price})")
    notional = qty * float(ref_price)

    # Absolute hard cap FIRST — the env-independent backstop.
    if notional > caps.hard_notional_cap_usd + _SLACK:
        return GateDecision(
            False,
            f"open refused: order notional ${notional:,.0f} exceeds absolute "
            f"hard cap ${caps.hard_notional_cap_usd:,.0f}",
        )
    if notional > caps.per_symbol_cap_usd + _SLACK:
        return GateDecision(
            False,
            f"open refused: order notional ${notional:,.0f} exceeds per-symbol "
            f"cap ${caps.per_symbol_cap_usd:,.0f}",
        )
    gross_now = sum(abs(mv) for (_s, _q, mv) in poss)
    if gross_now + notional > caps.gross_cap_usd + _SLACK:
        return GateDecision(
            False,
            f"open refused: gross ${gross_now + notional:,.0f} would exceed cap "
            f"${caps.gross_cap_usd:,.0f} (current gross ${gross_now:,.0f})",
        )

    pintent = "buy_to_open" if side == "buy" else "sell_to_open"
    return GateDecision(True, "ok", side=side, qty=qty, position_intent=pintent)


class OrderGateway:
    """Wraps the Alpaca client. The ONLY object permitted to call
    ``alpaca.submit_simple_order``. Construct once on the LiveTrader and route
    every order through :meth:`submit`."""

    def __init__(self, alpaca: Any, caps: GateCaps, *, alert=None) -> None:
        self.alpaca = alpaca
        self.caps = caps
        self._alert = alert  # optional callable(str) for operator notification

    def _live_positions(self, positions: Any) -> Optional[list]:
        if positions is not None:
            return positions
        try:
            return self.alpaca.get_positions()
        except Exception as exc:  # noqa: BLE001 — fail closed on any read error
            log.error("gateway: get_positions() failed — failing CLOSED: %s", exc)
            return None

    def submit(
        self,
        *,
        intent: str,
        symbol: str,
        side: Optional[str] = None,
        qty: float = 0.0,
        ref_price: Optional[float] = None,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
        positions: Any = None,
    ):
        """Validate against the live position snapshot, then submit. Returns the
        broker Order on success, or ``None`` if the guard refused (logged +
        alerted). NEVER raises on a guard refusal — callers treat None as
        'not submitted'."""
        poss = self._live_positions(positions)
        if poss is None:
            msg = (f"gateway REFUSED {intent} {symbol}: no live positions "
                   f"snapshot (fail-closed)")
            log.error(msg)
            self._notify(msg)
            return None

        plan = plan_order(
            intent=intent, symbol=symbol, requested_side=side,
            requested_qty=qty, ref_price=ref_price, positions=poss, caps=self.caps,
        )
        if not plan.allowed:
            log.error("gateway BLOCKED order: %s", plan.reason)
            self._notify(f"gateway BLOCKED: {plan.reason}")
            return None

        return self.alpaca.submit_simple_order(
            symbol=symbol,
            side=plan.side,
            qty=plan.qty,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            position_intent=plan.position_intent,
            extended_hours=extended_hours,
        )

    def _notify(self, msg: str) -> None:
        if self._alert is None:
            return
        try:
            self._alert(msg)
        except Exception:  # noqa: BLE001 — never let alerting break execution
            log.debug("gateway alert callback failed", exc_info=True)
