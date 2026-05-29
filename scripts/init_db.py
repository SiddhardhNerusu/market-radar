"""Initialize the MARKET RADAR SQLite database from sql/schema.sql.

Idempotent — safe to run repeatedly.

    python scripts/init_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import CONFIG  # noqa: E402
from market_radar.storage import init_db  # noqa: E402


def main() -> None:
    print(f"Initializing database at: {CONFIG.db_path}")
    init_db()
    print("Done. Tables created:")
    import sqlite3
    with sqlite3.connect(str(CONFIG.db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (name,) in rows:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
