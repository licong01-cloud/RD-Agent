#!/bin/bash
# Update hardcoded NTFS paths to ext4 paths in all relevant files

echo "=== Updating utils.py files ==="
find /mnt/f/Dev/RD-Agent-main -path '*/scenarios/qlib/experiment/utils.py' | while read f; do
    echo "  Updating: $f"
    # 1) default_gov_unix = Path("/mnt/f/Dev/AIstock/data_governance")
    sed -i 's|/mnt/f/Dev/AIstock/data_governance|/home/lc999/data/data_governance|g' "$f"
    # 2) default_gov_win = Path("F:/Dev/AIstock/data_governance")
    sed -i 's|F:/Dev/AIstock/data_governance|/home/lc999/data/data_governance|g' "$f"
    # 3) factors_root_p = Path("/mnt/f/Dev/AIstock/factors")
    sed -i 's|/mnt/f/Dev/AIstock/factors|/home/lc999/data/aistock_factors|g' "$f"
done

echo "=== Updating generate.py files ==="
find /mnt/f/Dev/RD-Agent-main -path '*/factor_data_template/generate.py' | while read f; do
    echo "  Updating: $f"
    sed -i 's|/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209|/home/lc999/data/qlib_bin|g' "$f"
done

echo "=== Updating results_api_server.py ==="
sed -i 's|/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209|/home/lc999/data/qlib_bin|g' /mnt/f/Dev/RD-Agent-main/rdagent/app/results_api_server.py

echo "=== Updating qlib_data_reader.py ==="
sed -i 's|F:/Dev/AIstock/qlib_bin/qlib_bin_20251209|/home/lc999/data/qlib_bin|g' /mnt/f/Dev/RD-Agent-main/rdagent/app/factor_metrics/qlib_data_reader.py
sed -i 's|/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209|/home/lc999/data/qlib_bin|g' /mnt/f/Dev/RD-Agent-main/rdagent/app/factor_metrics/qlib_data_reader.py

echo "=== Updating prompts.yaml files ==="
find /mnt/f/Dev/RD-Agent-main -name 'prompts.yaml' -path '*/scenarios/qlib/experiment/*' | while read f; do
    echo "  Updating: $f"
    sed -i 's|F:/Dev/RD-Agent-main/git_ignore_folder/sota_factors|/home/lc999/data/factor_data/sota_factors|g' "$f"
done

echo "=== Updating tools/*.py ==="
# dataset_governance_qa.py
sed -i 's|/mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209|/home/lc999/data/qlib_snapshots|g' /mnt/f/Dev/RD-Agent-main/tools/dataset_governance_qa.py
sed -i 's|/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209|/home/lc999/data/qlib_bin|g' /mnt/f/Dev/RD-Agent-main/tools/dataset_governance_qa.py
sed -i 's|/mnt/f/Dev/AIstock/data_governance|/home/lc999/data/data_governance|g' /mnt/f/Dev/RD-Agent-main/tools/dataset_governance_qa.py

# export_sh000300_benchmark.py
sed -i 's|/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209|/home/lc999/data/qlib_bin|g' /mnt/f/Dev/RD-Agent-main/tools/export_sh000300_benchmark.py

# compare_qlib_bins_688981.py
sed -i 's|/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209|/home/lc999/data/qlib_bin|g' /mnt/f/Dev/RD-Agent-main/tools/compare_qlib_bins_688981.py

# precompute_moneyflow_factors.py
sed -i 's|F:/Dev/AIstock/qlib_snapshots/qlib_export_20251209|/home/lc999/data/qlib_snapshots|g' /mnt/f/Dev/RD-Agent-main/tools/precompute_moneyflow_factors.py
sed -i 's|F:/Dev/AIstock/factors|/home/lc999/data/aistock_factors|g' /mnt/f/Dev/RD-Agent-main/tools/precompute_moneyflow_factors.py

# generate_static_factors_bundle.py
sed -i 's|F:/Dev/AIstock/qlib_snapshots/qlib_export_20251209|/home/lc999/data/qlib_snapshots|g' /mnt/f/Dev/RD-Agent-main/tools/generate_static_factors_bundle.py
sed -i 's|F:/Dev/AIstock/factors|/home/lc999/data/aistock_factors|g' /mnt/f/Dev/RD-Agent-main/tools/generate_static_factors_bundle.py

echo "=== Done ==="
