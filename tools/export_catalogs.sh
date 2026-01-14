#!/bin/bash
# 导出 Catalog（修复版）

set -e

echo "=== 导出 Catalog ==="
echo ""

cd /mnt/f/Dev/RD-Agent-main

# 确保输出目录存在
mkdir -p rdagent_data

# 导出 Factor Catalog
echo "1. 导出 Factor Catalog..."
python3 tools/export_aistock_factor_catalog.py \
    --registry-sqlite RDagentDB/registry.sqlite \
    --output rdagent_data/factor_catalog.json

# 导出 Model Catalog
echo "2. 导出 Model Catalog..."
python3 tools/export_aistock_model_catalog.py \
    --registry-sqlite RDagentDB/registry.sqlite \
    --output rdagent_data/model_catalog.json

# 导出 Strategy Catalog
echo "3. 导出 Strategy Catalog..."
python3 tools/export_aistock_strategy_catalog.py \
    --registry-sqlite RDagentDB/registry.sqlite \
    --output rdagent_data/strategy_catalog.json

# 导出 Loop Catalog
echo "4. 导出 Loop Catalog..."
python3 tools/export_aistock_loop_catalog.py \
    --registry-sqlite RDagentDB/registry.sqlite \
    --output rdagent_data/loop_catalog.json

echo ""
echo "✅ Catalog 导出完成"
echo ""

# 验证 Catalog
echo "验证 Catalog："
python3 -c "
import json
from pathlib import Path

catalogs = {
    'factor_catalog.json': 'factors',
    'model_catalog.json': 'models',
    'strategy_catalog.json': 'strategies',
    'loop_catalog.json': 'loops'
}

for filename, key in catalogs.items():
    path = Path('rdagent_data') / filename
    if path.exists():
        with open(path) as f:
            catalog = json.load(f)
            count = len(catalog.get(key, []))
            print(f'  {filename}: {count} 条记录')
    else:
        print(f'  {filename}: 文件不存在')
"
