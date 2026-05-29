"""Verify Trading 212 API key(s) work.

Calls a few read-only endpoints per configured account (Invest and/or ISA),
prints a summary, and exits 0 on success / 1 on any failure.

The API key value is NEVER printed — only a short SHA-256 fingerprint, so
this script is safe to run while screen-sharing or pasting output back in
chat.

    python scripts/test_t212_connection.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import CONFIG  # noqa: E402
from market_radar.t212 import (  # noqa: E402
    T212AuthError,
    T212Client,
    T212Error,
    available_clients,
)
from market_radar.t212.client import _fingerprint  # noqa: E402


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def fmt_money(value: float | None, currency: str | None) -> str:
    if value is None:
        return "n/a"
    cur = currency or ""
    return f"{value:,.2f} {cur}".strip()


def test_account(client: T212Client) -> bool:
    label = client.account_type.upper()
    print(f"\n=== {label} account ===")
    print(f"  Base URL: {client.base_url}")
    print(f"  Key fingerprint:    {_fingerprint(client.api_key)}")
    print(f"  Secret fingerprint: {_fingerprint(client.api_secret)}")
    print(f"  Auth header:        {client.basic_auth_preview()}")

    # 1. account_summary — proves the credentials authenticate
    try:
        info = client.account_summary()
    except T212AuthError as exc:
        print(f"  [FAIL] account_summary AUTH ERROR ({exc.status}): {exc}")
        return False
    except T212Error as exc:
        print(f"  [FAIL] account_summary error ({exc.status}): {exc}")
        return False

    cur = info.currency
    print(f"  [OK] account_summary → id={info.id} currency={cur} total_value={fmt_money(info.total_value, cur)}")
    if info.investments:
        print(
            f"         realized_pnl={fmt_money(info.investments.realized_profit_loss, cur)} "
            f"unrealized_pnl={fmt_money(info.investments.unrealized_profit_loss, cur)} "
            f"total_cost={fmt_money(info.investments.total_cost, cur)}"
        )

    # 2. account_cash — proves the Account-data scope is enabled
    try:
        cash = client.account_cash()
        print(
            f"  [OK] account_cash → total={fmt_money(cash.total, cur)} "
            f"free={fmt_money(cash.free, cur)} "
            f"invested={fmt_money(cash.invested, cur)} "
            f"ppl={fmt_money(cash.ppl, cur)} "
            f"pie_cash={fmt_money(cash.pie_cash, cur)}"
        )
    except T212Error as exc:
        print(f"  [WARN] account_cash failed ({exc.status}): {exc}")
        print("         → likely the 'Account data' scope was not enabled when you generated the key")

    # 3. positions — proves the Portfolio scope is enabled
    try:
        positions = client.positions()
        print(f"  [OK] positions → {len(positions)} open position(s)")
        for p in positions[:8]:
            current = p.current_price if p.current_price is not None else 0.0
            mkt_val = p.market_value or 0.0
            pnl = p.unrealized_pnl
            pnl_str = f"{pnl:+.2f}" if pnl is not None else "n/a"
            name = (p.name or "")[:24]
            print(
                f"        - {p.ticker:<14} {name:<24} "
                f"qty={p.quantity:<10.4f} "
                f"avg={p.average_price_paid:<8.2f} "
                f"cur={current:<8.2f} "
                f"mkt_val={mkt_val:<10.2f} "
                f"pnl={pnl_str}"
            )
        if len(positions) > 8:
            print(f"        ... and {len(positions) - 8} more")
    except T212Error as exc:
        print(f"  [WARN] positions failed ({exc.status}): {exc}")
        print("         → likely the 'Portfolio' scope was not enabled")

    # 4. open_orders — proves Orders - Read is enabled
    try:
        orders = client.open_orders()
        print(f"  [OK] open_orders → {len(orders)} pending order(s)")
    except T212Error as exc:
        print(f"  [WARN] open_orders failed ({exc.status}): {exc}")
        print("         → likely the 'Orders - Read' scope was not enabled (non-fatal)")

    return True


def main() -> int:
    configure_logging()
    clients = available_clients()
    if not clients:
        print("No T212 API keys found in .env.")
        print(
            "Set T212_INVEST_API_KEY and/or T212_ISA_API_KEY in "
            f"{CONFIG.project_root / '.env'} and rerun."
        )
        return 1

    print(f"Configured accounts: {[c.account_type for c in clients]}")
    print(f"Base URL: {CONFIG.t212_base_url}")

    all_ok = True
    for client in clients:
        ok = test_account(client)
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("All configured T212 accounts authenticated successfully.")
        return 0
    else:
        print("One or more T212 accounts failed to authenticate. Fix the keys/scopes and rerun.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
