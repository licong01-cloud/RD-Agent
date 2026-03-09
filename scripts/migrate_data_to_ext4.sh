#!/bin/bash
# ============================================================================
# 数据迁移脚本：NTFS (/mnt/f/) -> WSL ext4 (/home/lc999/data/)
# 
# 用途：将 RD-Agent 和 AIstock 的大数据文件从 NTFS 迁移到 ext4，
#       提升 Qlib 训练和因子计算的 I/O 性能。
#
# 注意：
#   - 使用 rsync 复制，保留原始文件不删除
#   - 首次运行需要较长时间（~22GB 数据）
#   - 后续运行为增量同步，速度很快
#
# 执行方式：在 WSL 中运行
#   chmod +x scripts/migrate_data_to_ext4.sh
#   bash scripts/migrate_data_to_ext4.sh
# ============================================================================

set -e

TARGET_ROOT="/home/lc999/data"
echo "=========================================="
echo "数据迁移: NTFS -> ext4"
echo "目标根目录: $TARGET_ROOT"
echo "=========================================="

# 创建目标目录结构
echo "[1/7] 创建目标目录..."
mkdir -p "$TARGET_ROOT/qlib_bin"
mkdir -p "$TARGET_ROOT/factor_data"
mkdir -p "$TARGET_ROOT/factor_data_debug"
mkdir -p "$TARGET_ROOT/aistock_factors"
mkdir -p "$TARGET_ROOT/data_governance"
mkdir -p "$TARGET_ROOT/knowledge_base"
mkdir -p "$TARGET_ROOT/qlib_snapshots"

# 1. Qlib bin 数据（最大，~15GB）
echo ""
echo "[2/7] 迁移 Qlib bin 数据..."
echo "  源: /mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209"
echo "  目标: $TARGET_ROOT/qlib_bin"
if [ -d "/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209" ]; then
    rsync -ah --progress /mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209/ "$TARGET_ROOT/qlib_bin/"
    echo "  ✓ Qlib bin 数据迁移完成"
else
    echo "  ✗ 源目录不存在，跳过"
fi

# 2. 因子实现源数据（~5GB）
echo ""
echo "[3/7] 迁移因子实现源数据..."
echo "  源: /mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data"
echo "  目标: $TARGET_ROOT/factor_data"
if [ -d "/mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data" ]; then
    rsync -ah --progress /mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data/ "$TARGET_ROOT/factor_data/"
    echo "  ✓ 因子源数据迁移完成"
else
    echo "  ✗ 源目录不存在，跳过"
fi

# 2b. 因子实现源数据 debug 版
echo ""
echo "[3b/7] 迁移因子实现源数据(debug)..."
echo "  源: /mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data_debug"
echo "  目标: $TARGET_ROOT/factor_data_debug"
if [ -d "/mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data_debug" ]; then
    rsync -ah --progress /mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data_debug/ "$TARGET_ROOT/factor_data_debug/"
    echo "  ✓ 因子源数据(debug)迁移完成"
else
    echo "  ✗ 源目录不存在，跳过"
fi

# 3. AIstock 因子产物
echo ""
echo "[4/7] 迁移 AIstock 因子产物..."
echo "  源: /mnt/f/Dev/AIstock/factors"
echo "  目标: $TARGET_ROOT/aistock_factors"
if [ -d "/mnt/f/Dev/AIstock/factors" ]; then
    rsync -ah --progress /mnt/f/Dev/AIstock/factors/ "$TARGET_ROOT/aistock_factors/"
    echo "  ✓ AIstock 因子产物迁移完成"
else
    echo "  ✗ 源目录不存在，跳过"
fi

# 4. 数据治理目录
echo ""
echo "[5/7] 迁移数据治理目录..."
echo "  源: /mnt/f/Dev/AIstock/data_governance"
echo "  目标: $TARGET_ROOT/data_governance"
if [ -d "/mnt/f/Dev/AIstock/data_governance" ]; then
    rsync -ah --progress /mnt/f/Dev/AIstock/data_governance/ "$TARGET_ROOT/data_governance/"
    echo "  ✓ 数据治理目录迁移完成"
else
    echo "  ✗ 源目录不存在，跳过"
fi

# 5. CoSTEER 知识库
echo ""
echo "[6/7] 迁移 CoSTEER 知识库..."
echo "  源: /mnt/f/Dev/RD-Agent-main/git_ignore_folder/knowledge_base"
echo "  目标: $TARGET_ROOT/knowledge_base"
if [ -d "/mnt/f/Dev/RD-Agent-main/git_ignore_folder/knowledge_base" ]; then
    rsync -ah --progress /mnt/f/Dev/RD-Agent-main/git_ignore_folder/knowledge_base/ "$TARGET_ROOT/knowledge_base/"
    echo "  ✓ CoSTEER 知识库迁移完成"
else
    echo "  ✗ 源目录不存在，跳过"
fi

# 6. Qlib snapshots
echo ""
echo "[7/7] 迁移 Qlib snapshots..."
echo "  源: /mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209"
echo "  目标: $TARGET_ROOT/qlib_snapshots"
if [ -d "/mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209" ]; then
    rsync -ah --progress /mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209/ "$TARGET_ROOT/qlib_snapshots/"
    echo "  ✓ Qlib snapshots 迁移完成"
else
    echo "  ✗ 源目录不存在，跳过"
fi

echo ""
echo "=========================================="
echo "数据迁移完成！"
echo ""
echo "目标目录结构:"
du -sh "$TARGET_ROOT"/* 2>/dev/null || echo "(目录为空或不存在)"
echo ""
echo "下一步：更新配置文件中的路径"
echo "  - RD-Agent-main/.env"
echo "  - AIstock/.env"
echo "  - 所有 YAML 配置文件"
echo "  - Python 文件中的硬编码路径"
echo "=========================================="
