#!/usr/bin/env bash
# One-shot setup: create venv, install deps, init DB.
# Run from project root:
#     bash scripts/setup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Creating virtualenv at .venv"
# Remove any leftover venv (could be from sandboxed scaffolding — won't run on macOS)
if [ -d .venv ]; then
    echo "    (removing existing .venv first)"
    rm -rf .venv
fi
python3 -m venv .venv

echo "==> Installing dependencies"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
    echo "==> Creating .env from template (fill in your keys)"
    cp .env.example .env
fi

echo "==> Initializing database"
python scripts/init_db.py

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. Edit .env and fill in your API keys (T212, Anthropic, Reddit, Finnhub, etc.)"
echo "  2. Run the daemon manually:    python scripts/run_daemon.py"
echo "  3. Install as launchd job:     bash scripts/install_launchd.sh"
