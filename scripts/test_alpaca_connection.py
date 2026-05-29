#!/usr/bin/env python3
"""Smoke test the Alpaca connection.

Prints account info, open positions, today's order count, market clock,
and a quote for SPY. No orders are placed.

Run:
    python scripts/test_alpaca_connection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.execution import AlpacaClient
from market_radar.execution.alpaca_client import AlpacaError


GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main() -> int:
    print(f"{BOLD}=== Alpaca connection test ==={RESET}")
    try:
        client = AlpacaClient()
    except AlpacaError as exc:
        print(f"{RED}[FAIL]{RESET} construct: {exc}")
        return 1

    mode = "PAPER" if "paper" in client.base_url else "LIVE"
    print(f"  base_url: {client.base_url}  ({mode})")

    try:
        acc = client.get_account()
    except AlpacaError as exc:
        print(f"{RED}[FAIL]{RESET} get_account: {exc}")
        return 1
    print(f"{GREEN}[OK]{RESET}   account: id={acc.id[:8]}… status={acc.status} "
          f"equity=${acc.equity:,.2f} cash=${acc.cash:,.2f} "
          f"buying_power=${acc.buying_power:,.2f}")
    print(f"  intraday P&L: ${acc.equity - acc.last_equity:+,.2f} "
          f"({acc.daily_pnl_pct * 100:+.2f}%)")
    print(f"  tradeable: {acc.is_tradeable}  daytrades: {acc.daytrade_count}")

    try:
        positions = client.get_positions()
    except AlpacaError as exc:
        print(f"{RED}[FAIL]{RESET} get_positions: {exc}")
        return 1
    print(f"{GREEN}[OK]{RESET}   positions: {len(positions)} open")
    for p in positions[:10]:
        print(f"    {p.symbol:6s} qty={p.qty:>8.2f} mv=${p.market_value:>10,.2f} "
              f"u_pnl=${p.unrealized_pl:>+8.2f} ({p.unrealized_plpc*100:+.2f}%)")

    try:
        clock = client.get_market_clock()
    except AlpacaError as exc:
        print(f"{RED}[FAIL]{RESET} get_clock: {exc}")
        return 1
    print(f"{GREEN}[OK]{RESET}   clock: is_open={clock.get('is_open')}  "
          f"next_open={clock.get('next_open')}  next_close={clock.get('next_close')}")

    try:
        orders = client.list_orders(status="open", limit=10)
    except AlpacaError as exc:
        print(f"{RED}[FAIL]{RESET} list_orders: {exc}")
        return 1
    print(f"{GREEN}[OK]{RESET}   open orders: {len(orders)}")

    print(f"{DIM}  Fetching quote for SPY (free IEX feed)…{RESET}")
    try:
        q = client.get_latest_quote("SPY")
    except AlpacaError as exc:
        print(f"{RED}[WARN]{RESET} get_latest_quote(SPY): {exc}  (free-tier may not allow data)")
        q = None
    if q:
        bid, ask = q
        print(f"{GREEN}[OK]{RESET}   SPY  bid=${bid:.2f}  ask=${ask:.2f}  spread=${ask-bid:.4f}")
    else:
        print("       (no quote — data feed may not be enabled on this key)")

    print(f"\n{GREEN}{BOLD}All connection tests passed.{RESET}")
    print(f"{DIM}Next: python scripts/run_live_trader.py --dry-run --once{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
