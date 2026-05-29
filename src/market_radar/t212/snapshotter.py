"""Periodically snapshot T212 account+positions to SQLite.

This drives the dashboard's Portfolio view and the equity curve. We snapshot
every configured T212 account (Invest, ISA) — read-only, never modifies
anything in T212.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ..storage import get_connection
from . import available_clients
from .client import T212Client, T212Error
from .fx import detect_quote_currency, to_usd

log = logging.getLogger(__name__)


@dataclass
class SnapshotStats:
    accounts_snapped: int = 0
    positions_snapped: int = 0
    orders_snapped: int = 0
    errors: int = 0


def snapshot_all() -> SnapshotStats:
    """Snapshot all configured T212 accounts. Safe to call repeatedly."""
    stats = SnapshotStats()
    clients = available_clients()
    if not clients:
        log.debug("No T212 clients configured — skipping snapshot")
        return stats

    snap_ts = _utc_now_iso()
    for client in clients:
        try:
            _snapshot_one(client, snap_ts, stats)
        except Exception as exc:  # noqa: BLE001
            log.warning("T212 snapshot for %s failed: %s", client.account_type, exc)
            stats.errors += 1
    return stats


def _snapshot_one(client: T212Client, snap_ts: str, stats: SnapshotStats) -> None:
    account_type = client.account_type

    # Account summary + cash
    try:
        info = client.account_summary()
        cash = client.account_cash()
    except T212Error as exc:
        log.warning("T212 %s account fetch failed: %s", account_type, exc)
        stats.errors += 1
        return

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO t212_account_snapshots (
                snapshot_at, account_type, cash, total_value, invested, pnl, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snap_ts,
                account_type,
                cash.free,
                info.total_value if info.total_value is not None else cash.total,
                cash.invested,
                cash.ppl,
                json.dumps({
                    "summary": info.raw,
                    "cash": cash.raw,
                }),
            ),
        )
        stats.accounts_snapped += 1

    # Positions
    try:
        positions = client.positions()
    except T212Error as exc:
        log.warning("T212 %s positions fetch failed: %s", account_type, exc)
        stats.errors += 1
        positions = []

    if positions:
        with get_connection() as conn:
            for p in positions:
                pnl_pct = None
                if (
                    p.current_price is not None
                    and p.average_price_paid
                ):
                    try:
                        pnl_pct = (p.current_price - p.average_price_paid) / p.average_price_paid * 100.0
                    except ZeroDivisionError:
                        pnl_pct = None

                # Currency normalization — T212 quotes some instruments in
                # pence/EUR/etc.  We store BOTH the native values and
                # USD-converted values so downstream consumers (risk
                # manager, dashboard, anything that does quantity * price)
                # get true USD without manual conversion.  Fail closed:
                # unknown ccy or FX miss → USD columns are 0 so we never
                # over-report dollar exposure.  Native values + raw_payload
                # preserve the original quote for traceability.
                quote_ccy = detect_quote_currency(p.ticker or "")
                native_current = float(p.current_price) if p.current_price is not None else None
                native_avg = float(p.average_price_paid) if p.average_price_paid is not None else None
                usd_current = to_usd(native_current, p.ticker or "") if native_current is not None else None
                usd_avg = to_usd(native_avg, p.ticker or "") if native_avg is not None else None
                if usd_current is None and native_current is not None:
                    log.warning(
                        "FX conversion failed for %s (ccy=%s) — storing 0 USD",
                        p.ticker, quote_ccy,
                    )
                native_mv = None
                if p.quantity is not None and native_current is not None:
                    try:
                        native_mv = float(p.quantity) * native_current
                    except (TypeError, ValueError):
                        native_mv = None
                mv_usd = 0.0
                if p.quantity is not None and usd_current is not None:
                    try:
                        mv_usd = float(p.quantity) * usd_current
                    except (TypeError, ValueError):
                        mv_usd = 0.0

                # The legacy `current_price` and `average_price` columns
                # now carry USD-converted values so existing consumers
                # (risk manager: quantity * current_price = USD exposure;
                # dashboard portfolio sort by quantity * current_price)
                # see a coherent USD-denominated portfolio.  Native values
                # remain available via raw_payload + native_market_value.
                # Fail-closed: when FX is unavailable we store 0 rather
                # than a misleading native number.
                stored_current = usd_current if usd_current is not None else 0.0
                stored_avg = usd_avg if usd_avg is not None else 0.0

                conn.execute(
                    """
                    INSERT INTO t212_positions (
                        snapshot_at, account_type, ticker, quantity, average_price,
                        current_price, pnl_pct, pnl_value, raw_payload,
                        quote_currency, native_market_value, market_value_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snap_ts,
                        account_type,
                        p.ticker,
                        p.quantity,
                        stored_avg,
                        stored_current,
                        pnl_pct,
                        p.unrealized_pnl,
                        json.dumps(p.raw),
                        quote_ccy,
                        native_mv,
                        mv_usd,
                    ),
                )
                stats.positions_snapped += 1

    # Open orders (idempotent upsert by order_id)
    try:
        orders = client.open_orders()
    except T212Error as exc:
        log.warning("T212 %s orders fetch failed: %s", account_type, exc)
        stats.errors += 1
        orders = []

    if orders:
        with get_connection() as conn:
            for o in orders:
                if o.id is None:
                    continue
                try:
                    conn.execute(
                        """
                        INSERT INTO t212_orders (
                            order_id, account_type, ticker, side, quantity, price,
                            order_type, status, placed_at, filled_at, raw_payload
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(order_id, account_type) DO UPDATE SET
                            status = excluded.status,
                            filled_at = COALESCE(t212_orders.filled_at, excluded.filled_at),
                            raw_payload = excluded.raw_payload
                        """,
                        (
                            str(o.id),
                            account_type,
                            o.ticker,
                            "buy" if (o.quantity or 0) > 0 else "sell",
                            abs(o.quantity or 0),
                            o.limit_price or o.stop_price,
                            o.type,
                            o.status,
                            o.creation_time or snap_ts,
                            None,
                            json.dumps(o.raw),
                        ),
                    )
                    stats.orders_snapped += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "t212_orders upsert failed order_id=%s: %s", o.id, exc
                    )
                    stats.errors += 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
