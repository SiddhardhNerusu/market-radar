"""Fills-based realized-P&L reconciliation (full-audit rebuild, ch2).

The per-trade ledger (bot_daily_pnl) historically captured only ~13% of the real
loss: exits were attributed onto entry rows and silently dropped for a whole
class of closes (bracket stops, extended-hours, closes-while-the-bot-was-down).
The audit's verdict was that the equity curve is the only honest P&L.

The honest forward fix: the broker's ACTUAL fills are the source of truth. Now
that the bot is a single LONG-ONLY EQUITY lane (crypto + options decommissioned),
realized P&L is a clean FIFO match of buy->sell fills. This module computes it
deterministically — same fills in, same numbers out — so the ledger can be
rebuilt from fills and can never silently diverge again.

`realized_from_fills` is pure (no I/O) and unit-tested. `pull_equity_fills`
pages the Alpaca activities API oldest-first and keeps only equity fills.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Iterable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_equity_symbol(sym: str) -> bool:
    """True for a plain equity ticker. Excludes crypto pairs ('BTC/USD') and OCC
    option symbols (e.g. 'NVDA260605C00225000' — long, with embedded digits)."""
    if not sym or "/" in sym:
        return False
    return not (len(sym) > 6 and any(c.isdigit() for c in sym))


def realized_from_fills(fills: Iterable[dict]) -> tuple[dict[str, dict], list[dict]]:
    """FIFO-match long-only equity fills into realized P&L per trading date.

    Each fill is a dict with: symbol, side ('buy'/'sell'), qty, price,
    transaction_time (ISO-8601 string). Order of the input does not matter — fills
    are sorted by transaction_time first.

    Returns ``(by_date, uncovered)``:
      • by_date: {'YYYY-MM-DD': {'realized', 'trades', 'wins', 'losses'}}, where
        one 'trade' is counted per CLOSING sell fill (a round-trip close), win/loss
        by that sell's net realized P&L.
      • uncovered: sells with no matching open lot (a data gap or a position opened
        before the fill window) — surfaced, never silently counted as P&L.
    """
    ordered = sorted(fills, key=lambda f: str(f.get("transaction_time", "")))
    lots: dict[str, deque] = defaultdict(deque)   # symbol -> deque([ [qty, price], ... ])
    by_date: dict[str, dict] = {}
    uncovered: list[dict] = []

    def _rec(day: str) -> dict:
        return by_date.setdefault(
            day, {"realized": 0.0, "trades": 0, "wins": 0, "losses": 0})

    for f in ordered:
        side = str(f.get("side", "")).lower()
        try:
            qty = abs(float(f["qty"]))
            price = float(f["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        sym = f.get("symbol", "")
        day = str(f.get("transaction_time", ""))[:10]

        if side == "buy":
            lots[sym].append([qty, price])
        elif side == "sell":
            rem = qty
            pnl = 0.0
            matched = False
            while rem > 1e-9 and lots[sym]:
                lot = lots[sym][0]
                m = min(rem, lot[0])
                pnl += (price - lot[1]) * m   # long: sell - buy cost basis
                matched = True
                lot[0] -= m
                rem -= m
                if lot[0] <= 1e-9:
                    lots[sym].popleft()
            if matched:
                rec = _rec(day)
                rec["realized"] += pnl
                rec["trades"] += 1
                if pnl > 0:
                    rec["wins"] += 1
                else:
                    rec["losses"] += 1
            if rem > 1e-9:
                uncovered.append({"symbol": sym, "qty": round(rem, 6), "date": day})

    for rec in by_date.values():
        rec["realized"] = round(rec["realized"], 2)
    return by_date, uncovered


def pull_equity_fills(alpaca, *, max_pages: int = 50) -> list[dict]:
    """Page the Alpaca FILL activities API OLDEST-first (direction=asc + page_token)
    so no early buys are missed, and keep only equity fills. The wrapper's
    list_account_activities exposes a single page, so we drive _request directly.
    """
    out: list[dict] = []
    token = None
    for _ in range(max_pages):
        params = {"page_size": 100, "direction": "asc"}
        if token:
            params["page_token"] = token
        page = alpaca._request(
            "GET", "/v2/account/activities/FILL", params=params) or []
        if not page:
            break
        for f in page:
            if _is_equity_symbol(f.get("symbol", "")):
                out.append(f)
        token = page[-1].get("id")
        if len(page) < 100:
            break
    return out


def reconcile_daily_pnl_from_fills(alpaca, conn_factory, *, since_date=None,
                                   now_iso=None) -> dict:
    """Rebuild bot_daily_pnl realized columns from Alpaca equity fills — idempotent
    and the source of truth. Only writes dates >= ``since_date`` (so the pre-rebuild
    mixed-asset history, which only the equity curve can honestly value, is left
    untouched). Returns a summary. Pass ``now_iso`` in tests for determinism.
    """
    fills = pull_equity_fills(alpaca)
    by_date, uncovered = realized_from_fills(fills)
    stamp = now_iso or _utc_now_iso()
    written = 0
    with conn_factory() as conn:
        for day, rec in sorted(by_date.items()):
            if since_date and day < since_date:
                continue
            conn.execute(
                """INSERT INTO bot_daily_pnl
                       (trading_date, realized_pnl_usd, trades_count, wins, losses, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(trading_date) DO UPDATE SET
                       realized_pnl_usd = excluded.realized_pnl_usd,
                       trades_count     = excluded.trades_count,
                       wins             = excluded.wins,
                       losses           = excluded.losses,
                       updated_at       = excluded.updated_at""",
                (day, rec["realized"], rec["trades"], rec["wins"], rec["losses"], stamp),
            )
            written += 1
    return {
        "dates_written": written,
        "uncovered": len(uncovered),
        "total_realized": round(sum(r["realized"] for r in by_date.values()), 2),
    }
