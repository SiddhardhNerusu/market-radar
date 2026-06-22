"""Post-close ledger reconcile — rebuild bot_daily_pnl realized columns from
Alpaca's ACTUAL equity fills (full-audit rebuild ch2, the source of truth).

Why this exists: bot_daily_pnl was historically written ONLY by
LiveTrader._update_daily_pnl — a running SUM of ESTIMATED per-exit P&L that
silently dropped a whole class of closes (bracket stops, extended-hours, closes
while the bot was down) and accumulated float drift (e.g. -47.7672). The honest
forward fix lives in market_radar.execution.pnl_reconcile: FIFO-match the broker's
real buy->sell fills so the ledger is deterministic and can never silently
diverge again. This script is the runtime path that makes that capability active.

It is idempotent (ON CONFLICT upsert, full rebuild every run) and only rewrites
trading dates >= the equity-only-lane cutover, so the pre-rebuild mixed-asset
history — which only the equity curve can honestly value — is left untouched.
daily_assessment.py keeps its "equity curve authoritative / ledger partial"
framing; this just makes the partial ledger fills-precise on the equity lane.

Scheduled via com.marketradar.reconcile.plist (after the US extended-hours close).
Idempotent, so the exact run time only affects how same-day-complete the ledger is;
a late after-hours fill is picked up by the next run.

Usage:  PYTHONPATH=src .venv/bin/python scripts/reconcile_pnl.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Equity-only-lane cutover: the 2026-06-21 rebuild date. Dates before this are
# the old crypto+options+equity mix that the per-trade ledger cannot honestly
# value (the equity curve is authoritative there), so the reconcile leaves them
# alone. Overridable via env for a one-off backfill from a different date.
CUTOVER_DATE = os.environ.get("LIVE_PNL_RECONCILE_SINCE", "2026-06-21")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    from market_radar.execution.alpaca_client import AlpacaClient
    from market_radar.execution.pnl_reconcile import reconcile_daily_pnl_from_fills
    from market_radar.storage import get_connection

    try:
        alpaca = AlpacaClient()
        # get_connection is a no-arg context manager defaulting to the prod DB,
        # so it IS the conn_factory the reconcile expects.
        summary = reconcile_daily_pnl_from_fills(
            alpaca, get_connection, since_date=CUTOVER_DATE)
    except Exception:  # noqa: BLE001 — launchd err log captures the trace
        logging.exception("reconcile_pnl FAILED")
        return 1

    logging.info(
        "reconcile_pnl ok since=%s: dates_written=%d total_realized=$%.2f uncovered=%d",
        CUTOVER_DATE, summary["dates_written"],
        summary["total_realized"], summary["uncovered"],
    )
    if summary["uncovered"]:
        # Sells with no matching open lot — a data gap (or a position opened before
        # the fill window). Surfaced, never silently counted. Grep-able in the err log.
        logging.warning(
            "reconcile_pnl: %d UNCOVERED sell(s) — possible fill-history gap",
            summary["uncovered"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
