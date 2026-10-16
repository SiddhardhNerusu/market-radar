"""Parse stored Form 4 bodies -> structured insider_transactions table.

The daemon now runs this every minute (job ``insider_refresh``); this script is
the manual / backfill entry point. Idempotent.

    python scripts/refresh_insider_transactions.py [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse, logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.ingestors.insider_transactions import refresh_insider_transactions  # noqa: E402
from market_radar.storage import init_db  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    init_db()
    out = refresh_insider_transactions(limit=args.limit, dry_run=args.dry_run)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
