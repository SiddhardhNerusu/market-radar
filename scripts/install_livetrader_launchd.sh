#!/usr/bin/env bash
# Install MARKET RADAR live trader as a per-user launchd job.
# Auto-starts at login, restarts on crash. Logs to ~/Library/Logs/MarketRadar/.
#
#   bash scripts/install_livetrader_launchd.sh
#   bash scripts/install_livetrader_launchd.sh --options    # also enable options spreads
#
# To uninstall:
#   launchctl unload ~/Library/LaunchAgents/com.marketradar.livetrader.plist
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.marketradar.livetrader"
PLIST_TARGET="$HOME/Library/LaunchAgents/${LABEL}.plist"
TEMPLATE="$ROOT/launchd/com.marketradar.livetrader.plist.template"
LOG_DIR="$HOME/Library/Logs/MarketRadar"
PY="$ROOT/.venv/bin/python"

WITH_OPTIONS=0
for arg in "$@"; do
    [ "$arg" = "--options" ] && WITH_OPTIONS=1
done

if [ ! -f "$PY" ]; then
    echo "Python venv not found at $PY"
    echo "Run scripts/setup.sh first."
    exit 1
fi

if [ ! -f "$ROOT/.env" ]; then
    echo "No .env found in $ROOT — add Alpaca keys first."
    exit 1
fi

# Sanity check: are Alpaca keys present?
if ! grep -qE "^ALPACA_API_KEY=.+" "$ROOT/.env" || ! grep -qE "^ALPACA_API_SECRET=.+" "$ROOT/.env"; then
    echo "ALPACA_API_KEY / ALPACA_API_SECRET not set in .env. Refusing to install live trader."
    exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$PLIST_TARGET")"

sed \
    -e "s|__PYTHON__|${PY}|g" \
    -e "s|__PROJECT_ROOT__|${ROOT}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "$TEMPLATE" > "$PLIST_TARGET"

# If --options requested, add the flag to ProgramArguments
if [ $WITH_OPTIONS -eq 1 ]; then
    python3 - <<PYEOF
import plistlib, pathlib, sys
p = pathlib.Path("$PLIST_TARGET")
data = plistlib.loads(p.read_bytes())
if '--options' not in data['ProgramArguments']:
    data['ProgramArguments'].append('--options')
p.write_bytes(plistlib.dumps(data))
print('  + Enabled --options flag')
PYEOF
fi

if launchctl list | grep -q "${LABEL}"; then
    echo "==> Unloading existing live trader job"
    launchctl unload "$PLIST_TARGET" 2>/dev/null || true
fi

echo "==> Loading $PLIST_TARGET"
launchctl load -w "$PLIST_TARGET"

echo ""
echo "Live trader installed. Status:"
launchctl list | grep "${LABEL}" || echo "  (not listed yet — wait a moment)"
echo ""
echo "Logs:"
echo "  tail -f ${LOG_DIR}/livetrader.out.log"
echo "  tail -f ${LOG_DIR}/livetrader.err.log"
echo ""
echo "Dashboard: http://127.0.0.1:8765/bot/status"
echo ""
echo "To stop:    launchctl unload \"${PLIST_TARGET}\""
echo "To enable emergency stop without unloading:"
echo "  echo 'RISK_EMERGENCY_STOP=1' >> .env  (bot picks it up next iteration)"
