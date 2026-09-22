#!/usr/bin/env bash
# Double-click target: opens the ptq wizard in this terminal, keeps the window open at the end.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
export PATH="$HOME/.local/bin:$PATH"
source scripts/env.sh
if [ ! -x .venv/bin/ptq ]; then
  echo "Environment not set up yet; running: uv sync --extra cu130 --extra hqq --extra dev"
  uv sync --extra cu130 --extra hqq --extra dev || { echo "sync failed"; read -rp "press Enter to close"; exit 1; }
fi
uv run --no-sync ptq "$@"
echo
read -rp "Done. Press Enter to close this window."
