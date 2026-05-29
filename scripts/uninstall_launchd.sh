#!/usr/bin/env bash
# Remove the MARKET RADAR launchd job.
set -euo pipefail

LABEL="com.marketradar.daemon"
PLIST_TARGET="$HOME/Library/LaunchAgents/${LABEL}.plist"

if [ -f "$PLIST_TARGET" ]; then
    echo "==> Unloading $PLIST_TARGET"
    launchctl unload "$PLIST_TARGET" 2>/dev/null || true
    rm -f "$PLIST_TARGET"
    echo "Removed $PLIST_TARGET"
else
    echo "No plist found at $PLIST_TARGET — nothing to remove."
fi

echo ""
echo "MARKET RADAR daemon uninstalled. Logs in ~/Library/Logs/MarketRadar/ are preserved."
