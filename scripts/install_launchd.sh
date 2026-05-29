#!/usr/bin/env bash
# Install MARKET RADAR as a per-user launchd job (LaunchAgent).
# Auto-starts at login. Restarts on crash (throttled). Logs to
# ~/Library/Logs/MarketRadar/.
#
# Usage:
#   bash scripts/install_launchd.sh
#
# To uninstall:
#   bash scripts/uninstall_launchd.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.marketradar.daemon"
PLIST_TARGET="$HOME/Library/LaunchAgents/${LABEL}.plist"
TEMPLATE="$ROOT/launchd/com.marketradar.daemon.plist.template"
LOG_DIR="$HOME/Library/Logs/MarketRadar"
PY="$ROOT/.venv/bin/python"

if [ ! -f "$PY" ]; then
    echo "Python venv not found at $PY"
    echo "Run scripts/setup.sh first to create the venv + install deps."
    exit 1
fi

if [ ! -f "$ROOT/.env" ]; then
    echo "No .env found in $ROOT — refusing to install without it."
    echo "Run scripts/setup.sh (creates .env from .env.example) and fill in your keys."
    exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$PLIST_TARGET")"

# Substitute placeholders in the template
sed \
    -e "s|__PYTHON__|${PY}|g" \
    -e "s|__PROJECT_ROOT__|${ROOT}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "$TEMPLATE" > "$PLIST_TARGET"

# Unload first in case an older version is loaded
if launchctl list | grep -q "${LABEL}"; then
    echo "==> Unloading existing job"
    launchctl unload "$PLIST_TARGET" 2>/dev/null || true
fi

echo "==> Loading $PLIST_TARGET"
launchctl load -w "$PLIST_TARGET"

echo ""
echo "MARKET RADAR daemon installed."
echo ""
echo "Status:"
launchctl list | grep "${LABEL}" || echo "  (not listed yet — wait a moment and check again)"
echo ""
echo "Logs:"
echo "  tail -f ${LOG_DIR}/daemon.out.log"
echo "  tail -f ${LOG_DIR}/daemon.err.log"
echo "  tail -f ${ROOT}/logs/daemon.log"
echo ""
echo "To restart: launchctl unload \"${PLIST_TARGET}\" && launchctl load \"${PLIST_TARGET}\""
echo "To uninstall: bash scripts/uninstall_launchd.sh"
