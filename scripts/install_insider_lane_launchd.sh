#!/usr/bin/env bash
# Install the insider lane (paper) + the nightly edge-ledger job as LaunchAgents.
#
#   bash scripts/install_insider_lane_launchd.sh
#   launchctl unload ~/Library/LaunchAgents/com.marketradar.insiderlane.plist   # stop
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$HOME/Library/Logs/MarketRadar"

[ -f "$PY" ] || { echo "no venv at $PY — run scripts/setup.sh first"; exit 1; }
[ -f "$ROOT/.env" ] || { echo "no .env in $ROOT — refusing to install"; exit 1; }
grep -q "^LANE_BOOK_EQUITY_USD=" "$ROOT/.env" || \
  echo "warning: LANE_BOOK_EQUITY_USD not set in .env — the lane will size against \$10,000"

mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"
for LABEL in com.marketradar.insiderlane com.marketradar.edgeledger; do
  TARGET="$HOME/Library/LaunchAgents/${LABEL}.plist"
  sed -e "s|__PYTHON__|${PY}|g" -e "s|__PROJECT_ROOT__|${ROOT}|g" -e "s|__HOME__|${HOME}|g" \
      "$ROOT/launchd/${LABEL}.plist.template" > "$TARGET"
  launchctl unload "$TARGET" 2>/dev/null || true
  launchctl load -w "$TARGET"
  echo "loaded $LABEL"
done
echo "logs: tail -f $LOG_DIR/insiderlane.out.log"
