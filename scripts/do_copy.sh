#!/bin/bash
set -e
T="/home/lc999/data"
mkdir -p "$T/qlib_bin" "$T/factor_data" "$T/factor_data_debug" "$T/aistock_factors" "$T/data_governance" "$T/knowledge_base" "$T/qlib_snapshots"

echo "[1/7] Copying qlib_bin (~15GB)..."
rsync -a /mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209/ "$T/qlib_bin/" && echo "[1/7] DONE" || echo "[1/7] FAILED"

echo "[2/7] Copying factor_data..."
rsync -a /mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data/ "$T/factor_data/" && echo "[2/7] DONE" || echo "[2/7] FAILED"

echo "[3/7] Copying factor_data_debug..."
rsync -a /mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data_debug/ "$T/factor_data_debug/" && echo "[3/7] DONE" || echo "[3/7] FAILED"

echo "[4/7] Copying aistock_factors..."
rsync -a /mnt/f/Dev/AIstock/factors/ "$T/aistock_factors/" && echo "[4/7] DONE" || echo "[4/7] FAILED"

echo "[5/7] Copying data_governance..."
rsync -a /mnt/f/Dev/AIstock/data_governance/ "$T/data_governance/" && echo "[5/7] DONE" || echo "[5/7] FAILED"

echo "[6/7] Copying knowledge_base..."
rsync -a /mnt/f/Dev/RD-Agent-main/git_ignore_folder/knowledge_base/ "$T/knowledge_base/" && echo "[6/7] DONE" || echo "[6/7] FAILED"

echo "[7/7] Copying qlib_snapshots..."
rsync -a /mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209/ "$T/qlib_snapshots/" && echo "[7/7] DONE" || echo "[7/7] FAILED"

echo "=== ALL COPY COMPLETE ==="
du -sh "$T"/*
