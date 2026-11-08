#!/usr/bin/env bash
set -euo pipefail

echo "======================================================="
echo "  LLM Tile Visualizer (tilevis) - Demo Launcher"
echo "======================================================="

# 1. Activate local virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

TRACE_PATH="${1:-traces/samples/tiny_demo.trace.npz}"

# 2. Run preflight environment & trace verification
echo "[1/2] Running preflight diagnostics..."
python3 -m tilevis check "${TRACE_PATH}"

# 3. Launch interactive playback
echo ""
echo "[2/2] Starting interactive presentation playback..."
shift || true
python3 -m tilevis play "${TRACE_PATH}" "$@"
