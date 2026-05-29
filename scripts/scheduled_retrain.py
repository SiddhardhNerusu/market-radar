"""Daily auto-retrain wrapper.

Runs ``train_ml.py --multi-horizon``, parses the result, appends an
audit entry to ``model_history.md``, sends Telegram, and commits +
pushes the audit log to GitHub for permanent record.

The trained model files themselves are gitignored (binary, large).
The audit log captures the metadata needed to reconstruct what changed:
version, val_auc, horizon, train rows, deployment decision, reasons.

Daemon picks up new models via the ``is_stale()`` pointer-watch in
``predict.py`` — no restart needed.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOGS = ROOT / "logs"
MODEL_HISTORY = ROOT / "model_history.md"


def _notify(text: str) -> None:
    """Send a Telegram message — never raise. Silent no-op when not configured."""
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


def _git_commit_push(commit_msg: str) -> tuple[bool, str]:
    """Stage model_history.md, commit, push to origin. Returns (success, info)."""
    env = os.environ.copy()
    # Make sure git can find user identity even when invoked by launchd.
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("PATH",
                   "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
                   + (":" + env["PATH"] if env.get("PATH") else ""))
    try:
        # Stage just the audit log (model files are gitignored anyway).
        subprocess.run(["git", "add", "model_history.md"],
                       cwd=str(ROOT), env=env, check=True,
                       timeout=10, capture_output=True)
        # Commit. If nothing to commit, that's fine — return success silently.
        r = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            if "nothing to commit" in (r.stdout + r.stderr).lower():
                return True, "nothing-to-commit"
            return False, f"commit failed: {r.stderr.strip()[:150]}"
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        # Push to origin.
        r2 = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=30,
        )
        if r2.returncode != 0:
            return False, f"committed {sha} but push failed: {r2.stderr.strip()[:150]}"
        return True, sha
    except subprocess.TimeoutExpired:
        return False, "git operation timed out"
    except Exception as exc:  # noqa: BLE001
        return False, f"git error: {exc}"


def _parse_current_model() -> dict:
    """Read data/models/current.json and merge with its meta file."""
    out: dict = {}
    try:
        ptr_path = ROOT / "data" / "models" / "current.json"
        if not ptr_path.exists():
            return out
        ptr = json.loads(ptr_path.read_text())
        out.update(ptr)
        meta_path_str = ptr.get("meta_path")
        if meta_path_str and Path(meta_path_str).exists():
            meta = json.loads(Path(meta_path_str).read_text())
            # Pull a few high-signal fields from meta.
            for k in ("train_rows", "val_rows", "val_auc", "val_accuracy",
                      "val_positive_class_rate", "val_auc_std", "horizon",
                      "label_col", "model_type", "calibrated"):
                if k in meta and k not in out:
                    out[k] = meta[k]
    except Exception:  # noqa: BLE001
        pass
    return out


def _append_history(entry_lines: list[str]) -> None:
    """Append a dated entry to model_history.md."""
    if not MODEL_HISTORY.exists():
        MODEL_HISTORY.write_text(
            "# MARKET RADAR — Model retrain history\n\n"
            "Auto-appended by `scripts/scheduled_retrain.py`. Each "
            "entry records what changed and whether the new model deployed.\n\n"
            "---\n\n"
        )
    with MODEL_HISTORY.open("a") as f:
        f.write("\n".join(entry_lines) + "\n\n---\n\n")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    log_path = LOGS / f"retrain-{now.strftime('%Y%m%d-%H%M')}.log"

    # Snapshot the OLD model state before retrain so we can diff.
    old_model = _parse_current_model()
    old_version = old_model.get("version", "(none)")
    old_auc = old_model.get("val_auc", "?")

    logging.info("Daily retrain starting → log: %s", log_path)

    py = str(ROOT / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable

    with open(log_path, "w") as f:
        r = subprocess.run(
            [py, "scripts/train_ml.py", "--multi-horizon"],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
            timeout=20 * 60,  # 20 min hard cap
        )

    success = (r.returncode == 0)
    new_model = _parse_current_model()
    new_version = new_model.get("version", "(unknown)")
    new_auc = new_model.get("val_auc", "?")
    train_rows = new_model.get("train_rows", "?")
    val_rows = new_model.get("val_rows", "?")
    horizon = new_model.get("horizon", "?")
    deployed = (new_version != old_version)

    # Format AUC for display
    def _fmt_auc(v) -> str:
        if isinstance(v, (int, float)):
            return f"{v:.4f}"
        return str(v) if v is not None else "?"

    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    summary_lines = [
        f"## {timestamp}",
        "",
        f"- **Exit code**: {r.returncode} ({'success' if success else 'FAILED'})",
        f"- **Old model**: `{old_version}` (val_auc={_fmt_auc(old_auc)})",
        f"- **New model**: `{new_version}` (val_auc={_fmt_auc(new_auc)})",
        f"- **Deployed**: {'YES' if deployed else 'no — AUC gate rejected new model'}",
        f"- **Horizon**: {horizon} day",
        f"- **Train rows**: {train_rows} / **Val rows**: {val_rows}",
        f"- **Log file**: `{log_path.name}`",
    ]

    if not success:
        summary_lines.append(f"- **Failure reason**: exit code {r.returncode} — see log")

    _append_history(summary_lines)

    # Auto-commit + push the audit log (model binary itself is gitignored).
    if deployed:
        commit_msg = (
            f"Retrain {now.strftime('%Y-%m-%d')}: model {new_version} deployed "
            f"(val_auc {_fmt_auc(old_auc)} → {_fmt_auc(new_auc)})\n\n"
            f"Auto-committed by scripts/scheduled_retrain.py. The model file "
            f"itself is gitignored — this commit records the retrain event in "
            f"model_history.md so the audit trail is permanent.\n\n"
            f"Horizon: {horizon}d. Train rows: {train_rows}. Val rows: {val_rows}.\n\n"
            f"Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
        )
    else:
        commit_msg = (
            f"Retrain {now.strftime('%Y-%m-%d')}: no deployment "
            f"(new val_auc {_fmt_auc(new_auc)} did not beat old {_fmt_auc(old_auc)})\n\n"
            f"AUC gate refused the new model — keeping `{old_version}`. "
            f"Audit entry written for traceability.\n\n"
            f"Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
        )

    ok, info = _git_commit_push(commit_msg)
    if ok and info != "nothing-to-commit":
        logging.info("model_history.md committed + pushed: %s", info)
    elif not ok:
        logging.warning("auto-commit failed: %s", info)
        _notify(f"⚠️ Retrain audit auto-commit failed: {info}")

    # Telegram summary
    if success and deployed:
        _notify(
            f"✅ <b>MARKET RADAR retrain</b>\n"
            f"Old val_auc: {_fmt_auc(old_auc)}\n"
            f"New val_auc: {_fmt_auc(new_auc)} ({new_version})\n"
            f"Daemon auto-loads new model within 30s."
        )
    elif success and not deployed:
        _notify(
            f"ℹ️ <b>MARKET RADAR retrain</b>\n"
            f"Trained OK but new val_auc {_fmt_auc(new_auc)} "
            f"did not beat current {_fmt_auc(old_auc)}. Keeping current model."
        )
    else:
        _notify(
            f"❌ <b>MARKET RADAR retrain FAILED</b>\n"
            f"Exit {r.returncode}. Check <code>{log_path.name}</code>."
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
