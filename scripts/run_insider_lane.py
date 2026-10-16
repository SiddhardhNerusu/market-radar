#!/usr/bin/env python3
"""Run the long-only insider lane (paper by default).

    python scripts/run_insider_lane.py --paper --dry-run --once   # what would it do
    python scripts/run_insider_lane.py --paper                    # paper trade
    python scripts/run_insider_lane.py --live --i-understand      # real money (don't, yet)

The lane sizes against LANE_BOOK_EQUITY_USD (default $10k), not Alpaca's
$100k paper balance. Set it to the capital you actually have.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true", help="Alpaca paper endpoint (default).")
    mode.add_argument("--live", action="store_true", help="Alpaca LIVE endpoint. Requires --i-understand.")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Log every decision; submit nothing.")
    p.add_argument("--once", action="store_true", help="One loop, then exit.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    log = logging.getLogger("run_insider_lane")

    if args.live:
        if not args.i_understand:
            log.error("Refusing --live without --i-understand. The promotion gate has not passed; stay on paper.")
            return 1
        os.environ["ALPACA_BASE_URL"] = "https://api.alpaca.markets"
        log.warning("LIVE TRADING MODE — real money at risk.")
    else:
        os.environ["ALPACA_BASE_URL"] = "https://paper-api.alpaca.markets"
    if args.dry_run:
        os.environ["LANE_DRY_RUN"] = "1"

    from market_radar.storage.db import init_db
    init_db()
    from market_radar.lane import InsiderLaneTrader, LaneConfig
    cfg = LaneConfig.from_env()
    problems = cfg.validate()
    if problems:
        for pr in problems:
            log.error("config: %s", pr)
        return 2
    log.info("cohort=%s book=$%.0f risk/trade=%.2f%% stop=%.1fxATR tp=%.1fxATR max_pos=%d heat<=%.1f%% daily_loss=%.1f%%",
             cfg.candidate_name, cfg.book_equity_usd, cfg.risk_per_trade_pct, cfg.stop_atr_mult,
             cfg.tp_atr_mult, cfg.max_positions, cfg.max_portfolio_heat_pct, cfg.daily_loss_pct)
    try:
        trader = InsiderLaneTrader(cfg)
    except Exception as exc:  # noqa: BLE001
        log.exception("failed to construct trader: %s", exc)
        return 1
    if args.once:
        log.info("run_once -> %s", trader.run_once())
        return 0
    trader.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
