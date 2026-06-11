#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user uv
fi

uv venv .venv --python python3
source .venv/bin/activate
uv pip install -e ".[dev]"

echo "Core environment ready. Try:"
echo "  source .venv/bin/activate"
echo "  local3dai doctor"
echo "  local3dai run --prompt 'cyberpunk fox figurine' --backend mock"
