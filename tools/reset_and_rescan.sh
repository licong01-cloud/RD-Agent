#!/bin/bash
# 重置 loop 固化状态并重新扫描

set -e

echo "=== 重置 loop 固化状态并重新扫描 ==="
echo ""

cd /mnt/f/Dev/RD-Agent-main

# 1. 备份数据库
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
echo "1. 备份数据库..."
cp RDagentDB/registry.sqlite "RDagentDB/registry.sqlite.backup.$BACKUP_DATE"
echo "✅ 备份完成: RDagentDB/registry.sqlite.backup.$BACKUP_DATE"
echo ""

# 2. 创建 production_bundles 表（如果不存在）
echo "2. 创建 production_bundles 表..."
sqlite3 RDagentDB/registry.sqlite << 'EOF'
CREATE TABLE IF NOT EXISTS production_bundles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_run_id TEXT NOT NULL,
    loop_id INTEGER NOT NULL,
    bundle_id TEXT NOT NULL UNIQUE,
    bundle_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(task_run_id, loop_id)
);

CREATE INDEX IF NOT EXISTS idx_production_bundles_task_run_loop ON production_bundles(task_run_id, loop_id);
CREATE INDEX IF NOT EXISTS idx_production_bundles_bundle_id ON production_bundles(bundle_id);
EOF
echo "✅ production_bundles 表创建完成"
echo ""

# 3. 重置 loop 固化状态
echo "3. 重置 loop 固化状态..."
sqlite3 RDagentDB/registry.sqlite << 'EOF'
UPDATE loops
SET is_solidified = 0,
    materialization_status = NULL,
    materialization_error = NULL,
    materialization_updated_at_utc = NULL
WHERE has_result = 1 OR has_result = '1';

SELECT 
    COUNT(*) as total_loops,
    SUM(CASE WHEN has_result = 1 OR has_result = '1' THEN 1 ELSE 0 END) as has_result_count,
    SUM(CASE WHEN is_solidified = 0 OR is_solidified = '0' THEN 1 ELSE 0 END) as unsolidified_count
FROM loops;
EOF
echo "✅ Loop 固化状态已重置"
echo ""

# 4. 运行 backfill 模式生成元数据
echo "4. 运行 backfill 模式生成元数据..."
python3 tools/backfill_registry_artifacts.py --mode backfill --all-task-runs --overwrite-json
echo "✅ Backfill 模式执行完成"
echo ""

# 5. 运行 solidify-all 创建资产包
echo "5. 运行 solidify-all 创建资产包..."
python3 tools/backfill_registry_artifacts.py --mode solidify-all --all-task-runs
echo "✅ Solidify-all 模式执行完成"
echo ""

# 6. 检查结果
echo "6. 检查结果..."
sqlite3 RDagentDB/registry.sqlite << 'EOF'
-- 检查 loops 表状态
SELECT 'Loops 表状态:' as info;
SELECT 
    COUNT(*) as total_loops,
    SUM(CASE WHEN has_result = 1 OR has_result = '1' THEN 1 ELSE 0 END) as has_result_count,
    SUM(CASE WHEN is_solidified = 1 OR is_solidified = '1' THEN 1 ELSE 0 END) as solidified_count,
    SUM(CASE WHEN materialization_status = 'done' THEN 1 ELSE 0 END) as materialization_done_count
FROM loops;

-- 检查 production_bundles 表
SELECT 'Production Bundles 表状态:' as info;
SELECT 
    COUNT(*) as total_bundles,
    COUNT(DISTINCT task_run_id) as unique_task_runs,
    COUNT(DISTINCT loop_id) as unique_loops
FROM production_bundles;

-- 检查 artifacts 表
SELECT 'Artifacts 表状态（元数据文件）:' as info;
SELECT 
    artifact_type,
    COUNT(*) as count
FROM artifacts
WHERE artifact_type IN ('model_meta', 'strategy_meta', 'factor_meta', 'factor_perf', 'feedback')
GROUP BY artifact_type
ORDER BY artifact_type;
EOF
echo ""

# 7. 导出 Catalog
echo "7. 导出 Catalog..."
mkdir -p rdagent_data

echo "  导出 Factor Catalog..."
python3 tools/export_aistock_factor_catalog.py

echo "  导出 Model Catalog..."
python3 tools/export_aistock_model_catalog.py

echo "  导出 Strategy Catalog..."
python3 tools/export_aistock_strategy_catalog.py

echo "  导出 Loop Catalog..."
python3 tools/export_aistock_loop_catalog.py

echo "✅ Catalog 导出完成"
echo ""

# 8. 验证 Catalog
echo "8. 验证 Catalog..."
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
echo ""

echo "=== 全量扫描完成 ==="
echo ""
echo "下一步：同步到 AIstock"
echo "  1. 拷贝 Catalog 文件到 AIstock"
echo "  2. 拷贝资产包到 AIstock"
echo "  3. 导入到 AIstock 数据库"
