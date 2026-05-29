"""Run the historical SEC backfill.

One-time (or occasional) process that seeds the database with the past 2
years of SEC EDGAR filings paired with yfinance historical prices and
computed 1d/5d/20d returns. Idempotent — safe to re-run; already-ingested
filings are skipped.

    python scripts/run_backfill.py                   # full 8-quarter backfill
    python scripts/run_backfill.py --quarters 4      # last year only
    python scripts/run_backfill.py --cap 1000        # test run, 1000 filings max
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.backfill import run_backfill  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quarters", type=int, default=8, help="Trailing quarters to backfill")
    p.add_argument("--cap", type=int, default=None, help="Limit total filings (for testing)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-process even filings already in the DB")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = run_backfill(
        quarters=args.quarters,
        cap_filings=args.cap,
        skip_existing=not args.no_skip_existing,
    )
    print()
    print("Backfill summary:")
    print(f"  quarters processed:       {stats.quarters}")
    print(f"  raw filings considered:   {stats.raw_filings}")
    print(f"  CIK→ticker matched:       {stats.ticker_matched}")
    print(f"  inserted signals:         {stats.inserted_signals}")
    print(f"  duplicates (skipped):     {stats.dup_signals}")
    print(f"  scored:                   {stats.scored}")
    print(f"  outcomes priced:          {stats.outcomes_priced}")
    print(f"  outcomes unpriced:        {stats.outcomes_unpriced}")
    print(f"  errors:                   {stats.errors}")
    return 0 if stats.errors < stats.raw_filings * 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
