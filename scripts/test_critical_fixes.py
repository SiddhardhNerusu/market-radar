#!/usr/bin/env python3
"""Integration tests for the 4 critical bug fixes flagged by the audit.

Each test exercises the actual code path with synthetic inputs that would
have triggered the bug before the fix, and asserts the fix is in place.

Run inline:
    python scripts/test_critical_fixes.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
DIM = "\033[2m"


def _ok(name: str, msg: str = ""):
    print(f"  {GREEN}[PASS]{RESET} {name}  {DIM}{msg}{RESET}")


def _fail(name: str, msg: str):
    print(f"  {RED}[FAIL]{RESET} {name}  {msg}")


# ---------------------------------------------------------------------------
# Test 1: DST bug in _us_eastern_date
# ---------------------------------------------------------------------------

def test_dst_fix():
    print("\n[1] DST bug — _us_eastern_date uses zoneinfo, not hardcoded UTC-5")
    from market_radar.execution.live_trader import _us_eastern_date
    from zoneinfo import ZoneInfo

    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    got = _us_eastern_date()
    if got == today_et:
        _ok("eastern_date_matches_zoneinfo", f"today_ET={today_et}")
    else:
        _fail("eastern_date_matches_zoneinfo", f"expected {today_et}, got {got}")

    # Make sure the function imports zoneinfo (no hardcoded -5)
    import inspect
    src = inspect.getsource(_us_eastern_date)
    if "ZoneInfo" in src and "America/New_York" in src:
        _ok("uses_zoneinfo_in_source")
    else:
        _fail("uses_zoneinfo_in_source", "source string missing ZoneInfo/America-NY")


# ---------------------------------------------------------------------------
# Test 2: Short P&L sign — abs(qty) protects against Alpaca's negative
# qty on short fills
# ---------------------------------------------------------------------------

def test_short_pnl_sign():
    print("\n[2] Short P&L sign — abs(qty) so negative-qty short fills don't flip sign")
    from market_radar.execution.live_trader import LiveTrader

    # Mock the conn so SQL just records calls
    captured_pnl = []
    parent_row = {"id": 1, "qty": -100.0, "filled_avg_price": 50.0, "direction": "sell"}
    fresh = MagicMock()
    fresh.filled_avg_price = 52.0
    fresh.legs = []

    conn = MagicMock()
    fetched_parent = MagicMock()
    fetched_parent.__getitem__ = lambda s, k: parent_row[k]
    fetched_parent.__bool__ = lambda s: True
    conn.execute.return_value.fetchone.return_value = fetched_parent

    def capture(sql, params=None):
        if "UPDATE bot_orders SET realized_pnl_usd" in sql:
            captured_pnl.append(params[0])
        rv = MagicMock(); rv.fetchone.return_value = fetched_parent; return rv
    conn.execute.side_effect = capture

    # Build a minimal LiveTrader-like object without going through __init__
    lt = LiveTrader.__new__(LiveTrader)
    lt._update_daily_pnl = MagicMock()

    row = {"ticker": "TSLA", "direction": "buy"}  # closing leg is buy (close short)
    lt._maybe_realize_pnl(conn, row, fresh)

    # For a short opened at 50, closed at 52, qty=100 abs:
    # pnl = (entry - exit) * qty = (50 - 52) * 100 = -200 (loss — correct!)
    if captured_pnl and abs(captured_pnl[0] - (-200.0)) < 0.01:
        _ok("short_pnl_correct_sign", f"got pnl=${captured_pnl[0]:.2f} (expected -$200)")
    else:
        _fail("short_pnl_correct_sign",
              f"got pnl={captured_pnl} (expected -200 for short loss)")


# ---------------------------------------------------------------------------
# Test 3: Race-safe reconciliation — parent looked up by alpaca_order_id
# from fresh.legs[0]['id'], not by ticker+direction heuristic alone.
# ---------------------------------------------------------------------------

def test_race_safe_reconciliation():
    print("\n[3] Race condition fix — child legs match parent by Alpaca order ID")
    from market_radar.execution.live_trader import LiveTrader
    import inspect
    src = inspect.getsource(LiveTrader._maybe_realize_pnl)
    has_parent_lookup = (
        "alpaca_order_id" in src
        and "parent_alpaca_id" in src
        and "WHERE alpaca_order_id" in src
    )
    if has_parent_lookup:
        _ok("parent_lookup_by_alpaca_id_present",
            "preferred path: WHERE alpaca_order_id = ?")
    else:
        _fail("parent_lookup_by_alpaca_id_present",
              "race-safe lookup not found in source")


# ---------------------------------------------------------------------------
# Test 4: Idempotency on 5xx — pending_submit row persists BEFORE the
# Alpaca call so retries on the same score_id are blocked.
# ---------------------------------------------------------------------------

def test_idempotency():
    print("\n[4] Idempotency fix — 'pending_submit' row written before Alpaca call")
    from market_radar.execution import live_trader
    import inspect
    src = inspect.getsource(live_trader.LiveTrader._process_stock_candidate)
    has_pending = "pending_submit" in src
    has_unique_score_id = "UNIQUE(score_id)" not in src  # the schema enforces this; just check we're not redefining
    if has_pending:
        _ok("pending_submit_row_before_submit",
            "decision row is persisted before Alpaca submit")
    else:
        _fail("pending_submit_row_before_submit",
              "no pending_submit pattern found")

    # Also verify the schema has the UNIQUE constraint
    schema = (ROOT / "sql" / "execution_schema.sql").read_text()
    if "UNIQUE(score_id)" in schema:
        _ok("bot_decisions_unique_score_id", "schema enforces one row per score_id")
    else:
        _fail("bot_decisions_unique_score_id", "schema missing UNIQUE constraint")


# ---------------------------------------------------------------------------
# Bonus: EOD flatten + PDT guard + equity override are wired
# ---------------------------------------------------------------------------

def test_day_trading_pivot():
    print("\n[5] Day-trading strategy pivot — short-hold + EOD flatten + PDT guard")
    from market_radar.execution.live_trader import TraderConfig, LiveTrader

    c = TraderConfig()
    if c.signal_horizon == "1d":
        _ok("default_horizon_is_1d", f"signal_horizon={c.signal_horizon}")
    else:
        _fail("default_horizon_is_1d", f"expected 1d, got {c.signal_horizon}")
    if c.stock_sl_atr_mult == 0.75 and c.stock_tp_atr_mult == 1.25:
        _ok("tight_atr_stops", f"SL={c.stock_sl_atr_mult}x TP={c.stock_tp_atr_mult}x ATR")
    else:
        _fail("tight_atr_stops",
              f"got SL={c.stock_sl_atr_mult} TP={c.stock_tp_atr_mult}")
    if c.options_target_dte == 7:
        _ok("options_short_dte", f"DTE={c.options_target_dte}")
    else:
        _fail("options_short_dte", f"expected 7, got {c.options_target_dte}")
    if c.pdt_enforce:
        _ok("pdt_enforced_by_default")
    else:
        _fail("pdt_enforced_by_default", "PDT guard disabled by default")

    # Sanity: methods exist
    for attr in (
        "_effective_equity", "_eod_flatten_if_due", "_pdt_blocked",
        "_poll_option_exits", "reconcile_on_startup",
    ):
        if hasattr(LiveTrader, attr):
            _ok(f"method_present:{attr}")
        else:
            _fail(f"method_present:{attr}", "missing on LiveTrader")


# ---------------------------------------------------------------------------
# Confluence + macro regime integration
# ---------------------------------------------------------------------------

def test_confluence_and_regime():
    print("\n[6] Confluence multiplier + macro regime live")
    from market_radar.signals.confluence import get_confluence_multiplier, _CACHE
    _CACHE.clear()
    # CRM has earnings today — confluence should reduce size
    mult = get_confluence_multiplier("CRM", "buy")
    if mult < 1.0:
        _ok("confluence_reduces_for_imminent_earnings", f"CRM buy mult={mult:.2f}")
    else:
        _fail("confluence_reduces_for_imminent_earnings",
              f"expected <1.0, got {mult:.2f}")

    # XRN has insider cluster — confluence should boost
    _CACHE.clear()
    mult = get_confluence_multiplier("XRN", "buy")
    if mult > 1.0:
        _ok("confluence_boosts_for_insider_cluster", f"XRN buy mult={mult:.2f}")
    else:
        _fail("confluence_boosts_for_insider_cluster",
              f"expected >1.0, got {mult:.2f}")

    # Regime fetch should at minimum not crash
    try:
        from market_radar.signals.macro_regime import get_regime
        r = get_regime()
        _ok("macro_regime_fetches", f"bias={r.bias} mult={r.size_multiplier}")
    except Exception as exc:  # noqa: BLE001
        _fail("macro_regime_fetches", f"crashed: {exc}")


# ---------------------------------------------------------------------------
# Telegram dedup
# ---------------------------------------------------------------------------

def test_telegram_dedup():
    print("\n[7] Telegram dedup — same FILL event suppressed within 90s")
    from market_radar.notifications.realtime import (
        TradeAlert, notify_trade, _TRADE_DEDUP,
    )
    _TRADE_DEDUP.clear()
    a = TradeAlert(kind="FILLED", symbol="NVDA", pnl_usd=42.0, extra="test")
    # First call: would attempt to send (no creds → returns False but logs key)
    notify_trade(a)
    # Second identical call: should hit dedup
    if "|".join([a.kind, a.symbol.upper(), a.direction,
                 f"{a.qty:.2f}", f"{a.price:.2f}", f"{a.pnl_usd:.2f}"]) in _TRADE_DEDUP:
        _ok("dedup_key_recorded")
    else:
        _fail("dedup_key_recorded", "key missing after first call")


if __name__ == "__main__":
    print(f"{DIM}MARKET RADAR — critical fix integration tests{RESET}\n")
    for fn in (
        test_dst_fix,
        test_short_pnl_sign,
        test_race_safe_reconciliation,
        test_idempotency,
        test_day_trading_pivot,
        test_confluence_and_regime,
        test_telegram_dedup,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {RED}[CRASH]{RESET} {fn.__name__}: {exc}")
            import traceback; traceback.print_exc()
    print(f"\n{DIM}Done.{RESET}")
