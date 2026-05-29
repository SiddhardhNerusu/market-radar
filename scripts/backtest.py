#!/usr/bin/env python3
"""Backtest the strategy on the 156k historical signal_outcomes.

Usage:
    # Compare all presets side-by-side (the most useful command)
    python scripts/backtest.py --compare

    # Run one preset with detail
    python scripts/backtest.py --preset new_gate_options

    # Backtest a specific date range
    python scripts/backtest.py --preset new_gate --from 2026-01-01 --to 2026-05-15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.backtest import BacktestConfig, PRESETS, run_backtest
from market_radar.backtest.replay import format_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="new_gate")
    parser.add_argument("--compare", action="store_true",
                        help="Run all presets and print side-by-side summary")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (default: all history)")
    parser.add_argument("--to", dest="to_date", default=None,
                        help="End date YYYY-MM-DD")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if args.compare:
        results = []
        for name, preset in PRESETS.items():
            cfg = BacktestConfig(preset=preset, from_date=args.from_date,
                                 to_date=args.to_date)
            r = run_backtest(cfg)
            results.append(r)
        _print_compare(results)
        return 0

    preset = PRESETS[args.preset]
    cfg = BacktestConfig(preset=preset, from_date=args.from_date,
                         to_date=args.to_date)
    r = run_backtest(cfg)
    print(format_result(r))
    return 0


def _print_compare(results) -> None:
    print()
    print("=" * 110)
    print(f"  {'PRESET':22s}  {'TRADES':>7s}  {'HIT%':>5s}  {'AVG/TRADE':>10s}  "
          f"{'AVG/DAY':>9s}  {'TOTAL':>10s}  {'DD%':>6s}  {'SHARPE':>7s}")
    print("=" * 110)
    for r in results:
        print(f"  {r.preset_name:22s}  {r.n_trades:>7d}  {r.hit_rate:>5.1f}  "
              f"${r.avg_pnl_usd:>+9.2f}  ${r.avg_daily_pnl_usd:>+8.2f}  "
              f"${r.total_pnl_usd:>+9.0f}  {r.max_drawdown_pct:>5.1f}  "
              f"{r.sharpe_annualized:>7.2f}")
    print("=" * 110)
    print()
    print("For full detail on any preset:")
    print(f"  python scripts/backtest.py --preset <NAME>")


if __name__ == "__main__":
    raise SystemExit(main())
