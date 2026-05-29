"""Weekly auto-retrain wrapper.

Runs ``train_ml.py --all``, captures result, sends Telegram on
completion (success OR failure), restarts the daemon if a new model
deployed.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOGS = ROOT / "logs"


def _notify(text: str) -> None:
    """Send a Telegram message — never raise.  Silent no-op when not configured."""
    try:
        from market_radar.config import CONFIG
        import requests
        if not (CONFIG.telegram_bot_token and CONFIG.telegram_chat_id):
            return
        requests.post(
            f"https://api.telegram.org/bot{CONFIG.telegram_bot_token}/sendMessage",
            json={
                "chat_id": CONFIG.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
    except Exception:
        pass


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"retrain-{datetime.now().strftime('%Y%m%d-%H%M')}.log"

    _notify("🔧 MARKET RADAR — weekly retrain starting")
    logging.info("Running train_ml.py --all → %s", log_path)

    py = str(ROOT / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable

    with open(log_path, "w") as f:
        r = subprocess.run(
            [py, "scripts/train_ml.py", "--all"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
        )

    success = (r.returncode == 0)

    # Parse newest val_auc from current.json (if available).
    val_auc = "?"
    try:
        ptr_path = ROOT / "data" / "models" / "current.json"
        ptr = json.loads(ptr_path.read_text())
        meta_path = Path(ptr.get("meta_path", ""))
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else ptr
        v = meta.get("val_auc")
        if isinstance(v, (int, float)):
            val_auc = f"{v:.4f}"
        elif v is not None:
            val_auc = str(v)
    except Exception:
        pass

    if success:
        _notify(
            f"✅ MARKET RADAR — retrain succeeded. "
            f"New main val_auc = {val_auc}. Daemon restarting."
        )
        # Restart the daemon so it picks up new model pointers.
        try:
            subprocess.run(["pkill", "-f", "run_daemon.py"], check=False)
            time.sleep(3)
            restart_log = LOGS / f"daemon-restart-{datetime.now().strftime('%H%M')}.log"
            subprocess.Popen(
                [py, str(ROOT / "scripts" / "run_daemon.py")],
                stdout=open(restart_log, "w"),
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Daemon restart failed: %s", exc)
            _notify(f"⚠️ Daemon failed to auto-restart after retrain: {exc}")
    else:
        _notify(
            f"❌ MARKET RADAR — retrain FAILED (exit {r.returncode}). "
            f"Check {log_path.name}."
        )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
