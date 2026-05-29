#!/usr/bin/env python3
"""Run the live trader loop.

Usage:
    # Default — paper trading, dry-run first (logs decisions, no orders)
    python scripts/run_live_trader.py --dry-run

    # Paper trading, real bracket orders against Alpaca's paper account
    python scripts/run_live_trader.py --paper

    # Live trading (REAL MONEY) — only do this after weeks of paper success
    python scripts/run_live_trader.py --live

Exit codes:
    0 — clean shutdown via SIGINT / SIGTERM
    1 — fatal startup error (bad credentials, DB unreachable, etc.)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true",
                      help="Force Alpaca paper endpoint (default).")
    mode.add_argument("--live", action="store_true",
                      help="Use Alpaca LIVE endpoint. Real money. Requires --i-understand.")
    parser.add_argument("--i-understand", action="store_true",
                        help="Required confirmation flag for --live.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log decisions, never call submit_bracket_order.")
    parser.add_argument("--interval", type=int, default=None,
                        help="Override LIVE_INTERVAL_SECONDS (seconds between loops).")
    parser.add_argument("--composite", type=float, default=None,
                        help="Override composite-score threshold (default 7.5).")
    parser.add_argument("--once", action="store_true",
                        help="Run a single loop iteration and exit (smoke test).")
    parser.add_argument("--options", action="store_true",
                        help="Enable Alpaca options spread routing for whitelisted underlyings "
                             "(SPY/QQQ/IWM/NVDA/TSLA/AAPL/AMD/META/MSFT/GOOGL/AMZN/COIN/PLTR). "
                             "Off by default; opt-in.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    log = logging.getLogger("run_live_trader")

    # Resolve mode → ALPACA_BASE_URL
    if args.live:
        if not args.i_understand:
            log.error(
                "Refusing to run --live without --i-understand. Live trading "
                "uses REAL money. Confirm with both flags after >= 2 weeks "
                "of profitable paper trading.")
            return 1
        os.environ["ALPACA_BASE_URL"] = "https://api.alpaca.markets"
        log.warning("⚠️  LIVE TRADING MODE — real money at risk.")
    else:
        os.environ["ALPACA_BASE_URL"] = "https://paper-api.alpaca.markets"

    if args.dry_run:
        os.environ["LIVE_DRY_RUN"] = "1"
    if args.options:
        os.environ["LIVE_OPTIONS_ENABLED"] = "1"

    if args.interval is not None:
        os.environ["LIVE_INTERVAL_SECONDS"] = str(args.interval)
    if args.composite is not None:
        os.environ["LIVE_COMPOSITE_THRESHOLD"] = str(args.composite)

    # Ensure tables exist
    from market_radar.storage.db import init_db
    init_db()

    # Import after env vars are set so config picks them up.
    from market_radar.execution import LiveTrader, TraderConfig
    try:
        trader = LiveTrader(config=TraderConfig.from_env())
    except Exception as exc:  # noqa: BLE001
        log.exception("Failed to construct LiveTrader: %s", exc)
        return 1

    if args.once:
        trader.run_once()
        return 0

    trader.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
