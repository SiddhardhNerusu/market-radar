"""Daemon liveness check. Run via launchd every 5 min. If the daemon
PID isn't running, fire a Telegram alert (with cooldown so we don't
spam the owner)."""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

PID_FILE = ROOT / "logs" / "daemon.pid"
COOLDOWN_MIN = int(os.getenv("DAEMON_HEALTH_COOLDOWN_MIN", "30"))


def _running(pid: int) -> bool:
    """True iff PID exists and we have permission to signal it."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _launchctl_daemon_pid() -> Optional[int]:
    """Truth source: ask launchd directly. Returns the running PID for
    com.marketradar.daemon, or None if not running.

    Avoids false 'daemon not running' alerts when logs/daemon.pid carries
    a stale value (e.g., from a different machine or a prior session
    that didn't update the file).
    """
    try:
        import subprocess
        out = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=5,
        )
        for line in out.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 3 and parts[-1] == "com.marketradar.daemon":
                pid_str = parts[0]
                if pid_str == "-":
                    return None
                try:
                    return int(pid_str)
                except ValueError:
                    return None
        return None
    except Exception:
        return None


def _send_telegram(msg: str) -> bool:
    try:
        from market_radar.config import CONFIG
        if not (CONFIG.telegram_bot_token and CONFIG.telegram_chat_id):
            return False
        import requests
        r = requests.post(
            f"https://api.telegram.org/bot{CONFIG.telegram_bot_token}/sendMessage",
            json={
                "chat_id": CONFIG.telegram_chat_id,
                "text": msg,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        return r.status_code == 200
    except Exception as exc:  # noqa: BLE001
        logging.warning("Telegram send failed: %s", exc)
        return False


# Route every DB access through the central helper so we pick up the
# pinned pragmas (WAL + synchronous=NORMAL + busy_timeout=10s +
# foreign_keys=ON).  This launchd job runs every 5 minutes and used to
# open raw connections that raced with the daemon's writer.
from market_radar.storage import get_connection  # noqa: E402


def _ensure_table(db: str) -> None:
    try:
        with get_connection() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS daemon_health_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    sent_at TEXT NOT NULL
                )
                """
            )
    except Exception as exc:  # noqa: BLE001
        logging.warning("daemon_health_alerts table ensure failed: %s", exc)


def _last_alert_minutes_ago(db: str) -> float:
    """Read last daemon-down alert. Returns large number if none."""
    try:
        with get_connection() as con:
            row = con.execute(
                "SELECT sent_at FROM daemon_health_alerts ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return 999_999.0
        ts = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() / 60.0
    except Exception:
        return 999_999.0


def _record(db: str, kind: str) -> None:
    try:
        with get_connection() as con:
            con.execute(
                "INSERT INTO daemon_health_alerts (kind, sent_at) VALUES (?, ?)",
                (kind, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
            )
    except Exception as exc:  # noqa: BLE001
        logging.warning("record alert failed: %s", exc)


def main() -> int:
    from market_radar.config import CONFIG
    db = str(CONFIG.db_path)
    _ensure_table(db)

    # TRUTH SOURCE: ask launchctl directly. The PID file can be stale
    # (carried over from a different machine, race with daemon restart,
    # crash before flush). launchctl knows the true PID of the running
    # com.marketradar.daemon job. Fall back to the PID file only if
    # launchd doesn't report the job (e.g., running manually via nohup).
    launchd_pid = _launchctl_daemon_pid()
    if launchd_pid is not None and _running(launchd_pid):
        # Heal the PID file so manual scripts / dashboard still find truth.
        try:
            PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            PID_FILE.write_text(f"{launchd_pid}\n")
        except OSError:
            pass
        logging.info("Daemon PID %d is healthy (via launchctl)", launchd_pid)
        return 0

    # Fallback: check PID file (for manually-launched daemons).
    if not PID_FILE.exists():
        logging.warning("daemon.pid missing AND launchctl reports no daemon")
        if _last_alert_minutes_ago(db) > COOLDOWN_MIN:
            if _send_telegram(
                "⚠️ <b>MARKET RADAR — daemon not running</b>\n"
                "launchctl shows no com.marketradar.daemon AND "
                "logs/daemon.pid is missing"
            ):
                _record(db, "missing_pid")
        return 1

    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError) as exc:
        logging.warning("daemon.pid unreadable: %s", exc)
        if _last_alert_minutes_ago(db) > COOLDOWN_MIN:
            if _send_telegram(
                "⚠️ <b>MARKET RADAR — daemon not running</b>\n"
                f"PID file unreadable: {exc}"
            ):
                _record(db, "bad_pid_file")
        return 1

    if not _running(pid):
        logging.warning("Daemon PID %d is not running (also not in launchctl)", pid)
        if _last_alert_minutes_ago(db) > COOLDOWN_MIN:
            restart_cmd = (
                "launchctl kickstart -k gui/$(id -u)/com.marketradar.daemon"
            )
            if _send_telegram(
                f"⚠️ <b>MARKET RADAR — daemon not running</b>\n"
                f"PID {pid} dead, launchctl reports no daemon either. "
                f"Restart with:\n<code>{restart_cmd}</code>"
            ):
                _record(db, "pid_dead")
        return 1

    logging.info("Daemon PID %d is healthy (via PID file fallback)", pid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
