"""Run the MARKET RADAR daemon in the foreground.

For background / always-on operation, install as a launchd job:

    bash scripts/install_launchd.sh

This script is what the launchd plist invokes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.daemon import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
