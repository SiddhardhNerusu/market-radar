"""Run the MARKET RADAR daemon in the foreground.

For background / always-on operation, install as a launchd job:

    bash scripts/install_launchd.sh

This script is what the launchd plist invokes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Write OUR PID to logs/daemon.pid on startup so the health-check job
# (scripts/check_daemon_health.py) tracks the right process. Previously
# this file could carry a stale PID from a different machine and the
# health check would spam "daemon not running" Telegram alerts every
# 30 minutes for a process that was never on this Mac.
def _write_pid_file() -> None:
    pid_file = ROOT / "logs" / "daemon.pid"
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{os.getpid()}\n")
    except OSError as exc:
        print(f"WARN: could not write {pid_file}: {exc}", file=sys.stderr)


from market_radar.daemon import main  # noqa: E402

if __name__ == "__main__":
    _write_pid_file()
    sys.exit(main())
