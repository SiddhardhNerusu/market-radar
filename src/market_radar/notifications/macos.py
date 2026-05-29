"""macOS native notifications via osascript.

No external dependency required. Falls back silently if osascript is missing
(e.g., running on Linux for tests).
"""
from __future__ import annotations

import shlex
import shutil
import subprocess


def send_notification(title: str, message: str, subtitle: str | None = None) -> bool:
    """Show a macOS notification. Returns True on success, False if unavailable."""
    if shutil.which("osascript") is None:
        return False

    # AppleScript display notification
    script_parts = [
        f'display notification {shlex.quote(message)}',
        f'with title {shlex.quote(title)}',
    ]
    if subtitle:
        script_parts.append(f'subtitle {shlex.quote(subtitle)}')
    script = " ".join(script_parts)

    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            timeout=10,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
