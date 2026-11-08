#!/usr/bin/env bash
set -euo pipefail

echo "Starting LLM Tile Visualizer Demo..."

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python3 -m tilevis demo "$@"
