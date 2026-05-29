"""Daily DB backup with rotation. Called by launchd at 3am."""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import CONFIG  # noqa: E402

KEEP = int(os.getenv("BACKUP_KEEP_LAST", "14"))
BACKUP_DIR = ROOT / "data" / "backups"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    dest = BACKUP_DIR / f"market_radar.daily-{ts}.db"
    logging.info("Backing up %s → %s", CONFIG.db_path, dest)
    try:
        src = sqlite3.connect(str(CONFIG.db_path))
        dst = sqlite3.connect(str(dest))
        src.backup(dst)
        dst.close()
        src.close()
    except Exception as exc:  # noqa: BLE001
        logging.error("Backup failed: %s", exc)
        return 1
    sz = dest.stat().st_size
    logging.info("Backup OK (%.1f MB)", sz / 1e6)

    # Rotation — keep the most recent KEEP daily backups; prune older ones.
    dailies = sorted(
        BACKUP_DIR.glob("market_radar.daily-*.db"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(dailies) > KEEP:
        for old in dailies[:-KEEP]:
            try:
                old.unlink()
                logging.info("Rotated out: %s", old.name)
            except OSError as exc:
                logging.warning("Rotate failed for %s: %s", old.name, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
