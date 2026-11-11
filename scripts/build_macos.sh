#!/usr/bin/env bash
# Convenience script to build the portable macOS application bundle for USB deployment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Building macOS standalone player bundle..."
cd "${REPO_ROOT}"
python3 scripts/build_usb_dist.py
