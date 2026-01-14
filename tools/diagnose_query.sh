#!/bin/bash
# 诊断为何只有 1 个需要处理

echo "=== 诊断查询条件问题 ==="
echo ""

cd /mnt/f/Dev/RD-Agent-main

DB_PATH="RDagentDB/registry.sqlite"

# 1. 检查 loops 表状态
echo "1. Loops 表状态："
sqlite3 "$DB_PATH" << 'EOF'
SELECT 
    COUNT(*) as total_loops,
    SUM(CASE WHEN has_result = 1 OR has_result = '1' THEN 1 ELSE 0 END) as has_result_count,
    SUM(CASE WHEN is_solidified = 0 OR is_solidified = '0' THEN 1 ELSE 0 END) as not_solidified_count,
    SUM(CASE WHEN materialization_status IS NULL THEN 1 ELSE 0 END) as materialization_null_count,
    SUM(CASE WHEN materialization_status = 'done' THEN 1 ELSE 0 END) as materialization_done_count
FROM loops;
EOF

echo ""
echo "2. 检查有结果的 loop 的 is_solidified 分布："
sqlite3 "$DB_PATH" << 'EOF'
SELECT 
    is_solidified,
    COUNT(*) as count
FROM loops
WHERE has_result = 1 OR has_result = '1'
GROUP BY is_solidified;
EOF

echo ""
echo "3. 检查有结果的 loop 的 materialization_status 分布："
sqlite3 "$DB_PATH" << 'EOF'
SELECT 
    materialization_status,
    COUNT(*) as count
FROM loops
WHERE has_result = 1 OR has_result = '1'
GROUP BY materialization_status;
EOF

echo ""
echo "4. 检查 batch_solidify.sh 的查询结果："
sqlite3 "$DB_PATH" << 'EOF'
SELECT l.task_run_id, l.loop_id, l.has_result, l.is_solidified, l.materialization_status
FROM loops l
JOIN workspaces w ON l.task_run_id = w.task_run_id
WHERE (l.has_result = 1 OR l.has_result = '1')
  AND (l.materialization_status IS NULL OR l.materialization_status != 'done')
  AND (l.is_solidified = 0 OR l.is_solidified = '0')
ORDER BY l.task_run_id, l.loop_id
LIMIT 10;
EOF

echo ""
echo "5. 检查未处理的有结果 loop（去掉 is_solidified 条件）："
sqlite3 "$DB_PATH" << 'EOF'
SELECT COUNT(*) as count
FROM loops l
JOIN workspaces w ON l.task_run_id = w.task_run_id
WHERE (l.has_result = 1 OR has_result = '1')
  AND (l.materialization_status IS NULL OR l.materialization_status != 'done')
ORDER BY l.task_run_id, l.loop_id;
EOF

echo ""
echo "6. 检查未处理的有结果 loop（去掉所有条件）："
sqlite3 "$DB_PATH" << 'EOF'
SELECT 
    l.task_run_id,
    l.loop_id,
    l.has_result,
    l.is_solidified,
    l.materialization_status
FROM loops l
WHERE (l.has_result = 1 OR has_result = '1')
  AND (l.materialization_status IS NULL OR l.materialization_status != 'done')
ORDER BY l.task_run_id, l.loop_id
LIMIT 10;
EOF

echo ""
echo "=== 诊断完成 ==="
