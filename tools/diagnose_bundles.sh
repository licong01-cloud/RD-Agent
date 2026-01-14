#!/bin/bash
# 诊断 production_bundles 为 0 的问题

echo "=== 诊断 production_bundles 为 0 的问题 ==="
echo ""

cd /mnt/f/Dev/RD-Agent-main

# 1. 检查 loops 表状态
echo "1. Loops 表状态："
sqlite3 RDagentDB/registry.sqlite << 'EOF'
SELECT 
    COUNT(*) as total_loops,
    SUM(CASE WHEN has_result = 1 OR has_result = '1' THEN 1 ELSE 0 END) as has_result_count,
    SUM(CASE WHEN is_solidified = 1 OR is_solidified = '1' THEN 1 ELSE 0 END) as solidified_count,
    SUM(CASE WHEN materialization_status = 'done' THEN 1 ELSE 0 END) as materialization_done_count,
    SUM(CASE WHEN materialization_status = 'failed' THEN 1 ELSE 0 END) as materialization_failed_count
FROM loops;
EOF

echo ""
echo "2. 检查 materialization_status 分布："
sqlite3 RDagentDB/registry.sqlite << 'EOF'
SELECT 
    materialization_status,
    COUNT(*) as count
FROM loops
WHERE has_result = 1 OR has_result = '1'
GROUP BY materialization_status;
EOF

echo ""
echo "3. 检查失败的 loop（如果有）："
sqlite3 RDagentDB/registry.sqlite << 'EOF'
SELECT 
    task_run_id,
    loop_id,
    materialization_status,
    materialization_error
FROM loops
WHERE materialization_status = 'failed'
LIMIT 10;
EOF

echo ""
echo "4. 检查未处理的有结果 loop（前 10 个）："
sqlite3 RDagentDB/registry.sqlite << 'EOF'
SELECT 
    l.task_run_id,
    l.loop_id,
    l.has_result,
    l.is_solidified,
    l.materialization_status,
    w.workspace_path
FROM loops l
JOIN workspaces w ON l.task_run_id = w.task_run_id
WHERE (l.has_result = 1 OR l.has_result = '1')
  AND (l.materialization_status IS NULL OR l.materialization_status != 'done')
LIMIT 10;
EOF

echo ""
echo "5. 检查 workspace 是否存在（前 5 个未处理的 loop）："
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
WHERE (l.has_result = 1 OR has_result = '1')
  AND (l.materialization_status IS NULL OR l.materialization_status != 'done')
LIMIT 5;
EOF

echo ""
echo "=== 诊断完成 ==="
