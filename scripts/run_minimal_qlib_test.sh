#!/usr/bin/env bash
set -euo pipefail

# Minimal RD-Agent + Qlib quant loop test
# Usage (in WSL):
#   cd /mnt/f/Dev/RD-Agent-main
#   bash scripts/run_minimal_qlib_test.sh

RDAGENT_ROOT="/mnt/f/Dev/RD-Agent-main"
REGISTRY_DB="$RDAGENT_ROOT/git_ignore_folder/registry/registry.sqlite"

cd "$RDAGENT_ROOT"

echo "[INFO] Running minimal fin_quant loop (loop_n=1)..."

# Use the unified CLI entrance so that .env is loaded correctly
python -m rdagent.app.cli fin_quant --loop-n 1

echo "[INFO] Minimal fin_quant loop finished."
echo "[INFO] Registry DB should be at: $REGISTRY_DB"
