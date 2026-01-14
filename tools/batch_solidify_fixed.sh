#!/bin/bash
# 分批运行 solidify-all 避免 OOM（修复版）

set -e

echo "=== 分批运行 solidify-all 避免 OOM ==="
echo ""

cd /mnt/f/Dev/RD-Agent-main

# 配置
BATCH_SIZE=10  # 每批处理 10 个 loop
DB_PATH="RDagentDB/registry.sqlite"

# 1. 查询需要处理的 loop
echo "1. 查询需要处理的 loop..."
sqlite3 "$DB_PATH" << 'EOF' > /tmp/needed_loops.txt
SELECT l.task_run_id, l.loop_id
FROM loops l
JOIN workspaces w ON l.task_run_id = w.task_run_id
WHERE (l.has_result = 1 OR has_result = '1')
  AND (l.materialization_status IS NULL OR l.materialization_status != 'done')
ORDER BY l.task_run_id, l.loop_id
LIMIT 1000;
EOF

TOTAL_COUNT=$(wc -l < /tmp/needed_loops.txt)
echo "找到 $TOTAL_COUNT 个需要处理的 loop"
echo ""

# 2. 分批处理
BATCH_NUM=1
PROCESSED=0

while IFS='|' read -r task_run_id loop_id; do
    if [ -z "$task_run_id" ] || [ -z "$loop_id" ]; then
        continue
    fi

    BATCH_INDEX=$((PROCESSED % BATCH_SIZE))
    
    if [ $BATCH_INDEX -eq 0 ]; then
        echo ""
        echo "=== 批次 $BATCH_NUM ==="
        BATCH_NUM=$((BATCH_NUM + 1))
    fi

    echo "  处理 loop: $task_run_id/$loop_id (进度: $((PROCESSED + 1))/$TOTAL_COUNT)"

    # 运行单个 loop 的 solidify
    python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from rdagent.utils.solidification import solidify_loop_assets
import sqlite3

db_path = Path('RDagentDB/registry.sqlite')
task_run_id = '$task_run_id'
loop_id = int('$loop_id')

try:
    bundle_id = solidify_loop_assets(task_run_id, loop_id, db_path=db_path)
    print(f'    ✅ 成功: bundle_id={bundle_id}')
    
    # 更新数据库
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        UPDATE loops
        SET materialization_status='done',
            materialization_error=NULL,
            materialization_updated_at_utc=datetime('now')
        WHERE task_run_id=? AND loop_id=?
    ''', (task_run_id, loop_id))
    conn.commit()
    conn.close()
except Exception as e:
    print(f'    ❌ 失败: {e}')
    
    # 记录错误
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        UPDATE loops
        SET materialization_status='failed',
            materialization_error=?,
            materialization_updated_at_utc=datetime('now')
        WHERE task_run_id=? AND loop_id=?
    ''', (str(e)[:512], task_run_id, loop_id))
    conn.commit()
    conn.close()
    sys.exit(1)
" || {
        echo "    ⚠️  Loop $task_run_id/$loop_id 处理失败，继续下一个"
    }

    PROCESSED=$((PROCESSED + 1))

    # 每批处理后暂停，释放内存
    if [ $BATCH_INDEX -eq $((BATCH_SIZE - 1)) ]; then
        echo "  批次完成，暂停 5 秒释放内存..."
        sleep 5
    fi
done < /tmp/needed_loops.txt

# 清理临时文件
rm -f /tmp/needed_loops.txt

echo ""
echo "=== 分批处理完成 ==="
echo ""

# 3. 检查结果
echo "3. 检查结果..."
sqlite3 "$DB_PATH" << 'EOF'
SELECT 
    COUNT(*) as total_loops,
    SUM(CASE WHEN has_result = 1 OR has_result = '1' THEN 1 ELSE 0 END) as has_result_count,
    SUM(CASE WHEN materialization_status = 'done' THEN 1 ELSE 0 END) as materialization_done_count,
    SUM(CASE WHEN materialization_status = 'failed' THEN 1 ELSE 0 END) as materialization_failed_count
FROM loops;

SELECT 'Production Bundles 表状态:' as info;
SELECT 
    COUNT(*) as total_bundles,
    COUNT(DISTINCT task_run_id) as unique_task_runs,
    COUNT(DISTINCT loop_id) as unique_loops
FROM production_bundles;
EOF

echo ""
echo "=== 完成 ==="
