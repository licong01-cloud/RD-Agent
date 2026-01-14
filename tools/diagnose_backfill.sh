#!/bin/bash
# 诊断 backfill 脚本执行问题

echo "=== 诊断 backfill 脚本执行问题 ==="
echo ""

cd /mnt/f/Dev/RD-Agent-main

# 1. 检查数据库中的 loop 状态
echo "1. 检查 loops 表状态："
sqlite3 RDagentDB/registry.sqlite << 'EOF'
SELECT 
    COUNT(*) as total_loops,
    SUM(CASE WHEN has_result = 1 OR has_result = '1' THEN 1 ELSE 0 END) as has_result_count,
    SUM(CASE WHEN is_solidified = 1 OR is_solidified = '1' THEN 1 ELSE 0 END) as solidified_count,
    SUM(CASE WHEN (has_result = 1 OR has_result = '1') AND (is_solidified = 1 OR is_solidified = '1') THEN 1 ELSE 0 END) as solidified_with_result,
    SUM(CASE WHEN (has_result = 1 OR has_result = '1') AND (is_solidified = 0 OR is_solidified = '0' OR is_solidified IS NULL) THEN 1 ELSE 0 END) as need_solidify
FROM loops;
EOF

echo ""
echo "2. 检查需要固化的 loop（前 10 个）："
sqlite3 RDagentDB/registry.sqlite << 'EOF'
SELECT 
    task_run_id,
    loop_id,
    has_result,
    is_solidified,
    materialization_status
FROM loops
WHERE (has_result = 1 OR has_result = '1') 
  AND (is_solidified = 0 OR is_solidified = '0' OR is_solidified IS NULL)
LIMIT 10;
EOF

echo ""
echo "3. 检查 workspace 路径（前 5 个需要固化的 loop）："
sqlite3 RDagentDB/registry.sqlite << 'EOF'
SELECT 
    l.task_run_id,
    l.loop_id,
    w.workspace_id,
    w.workspace_path,
    w.workspace_role
FROM loops l
JOIN workspaces w ON l.task_run_id = w.task_run_id
WHERE (l.has_result = 1 OR l.has_result = '1') 
  AND (l.is_solidified = 0 OR l.is_solidified = '0' OR l.is_solidified IS NULL)
LIMIT 5;
EOF

echo ""
echo "4. 检查 workspace 目录是否存在（前 5 个）："
sqlite3 RDagentDB/registry.sqlite << 'EOF'
SELECT 
    l.task_run_id,
    l.loop_id,
    w.workspace_path,
    CASE 
        WHEN w.workspace_path IS NULL THEN 'NULL'
        ELSE 'EXISTS'
    END as path_status
FROM loops l
JOIN workspaces w ON l.task_run_id = w.task_run_id
WHERE (l.has_result = 1 OR l.has_result = '1') 
  AND (l.is_solidified = 0 OR l.is_solidified = '0' OR l.is_solidified IS NULL)
LIMIT 5;
EOF

echo ""
echo "5. 检查 production_bundles 表："
sqlite3 RDagentDB/registry.sqlite << 'EOF'
SELECT 
    COUNT(*) as total_bundles,
    COUNT(DISTINCT task_run_id) as unique_task_runs,
    COUNT(DISTINCT loop_id) as unique_loops
FROM production_bundles;
EOF

echo ""
echo "=== 诊断完成 ==="
