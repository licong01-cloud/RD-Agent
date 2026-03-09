#!/usr/bin/env bash
# wait_and_run.sh — 等待当前 rdagent 任务完成后自动启动新一轮
# 用法: bash /mnt/f/Dev/RD-Agent-main/scripts/wait_and_run.sh

set -euo pipefail

PROJECT_DIR="/mnt/f/Dev/RD-Agent-main"
CONDA_ENV="rdagent-gpu"
CHECK_INTERVAL=600  # 10 分钟
LOOP_N=8

# 检测是否有 rdagent/qrun 进程在运行（排除自身）
count_running() {
    local count=0
    # 检测 qrun 进程
    count=$((count + $(pgrep -f "qrun conf_" 2>/dev/null | grep -v "^$$\$" | wc -l)))
    # 检测 rdagent 主进程
    count=$((count + $(pgrep -f "rdagent.app.cli fin_quant" 2>/dev/null | grep -v "^$$\$" | wc -l)))
    echo "$count"
}

echo "=========================================="
echo "  rdagent 任务队列监控"
echo "=========================================="
echo "  项目目录: $PROJECT_DIR"
echo "  Conda 环境: $CONDA_ENV"
echo "  新任务参数: --loop-n $LOOP_N"
echo "  检查间隔: $((CHECK_INTERVAL / 60)) 分钟"
echo "=========================================="

start_time=$(date +%s)
initial_count=$(count_running)

if [ "$initial_count" -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] 未检测到运行中的 rdagent/qrun 进程，直接启动新任务"
else
    echo "[$(date '+%H:%M:%S')] 检测到 $initial_count 个运行中的进程，开始等待..."

    while true; do
        sleep "$CHECK_INTERVAL"

        running=$(count_running)
        elapsed=$(( $(date +%s) - start_time ))
        hours=$((elapsed / 3600))
        mins=$(( (elapsed % 3600) / 60 ))

        if [ "$running" -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] 所有进程已结束（等待了 ${hours}h${mins}m）"
            break
        fi

        echo "[$(date '+%H:%M:%S')] 仍有 $running 个进程运行中（已等待 ${hours}h${mins}m）"
    done
fi

echo ""
echo "=========================================="
echo "  启动新任务: python -m rdagent.app.cli fin_quant --loop-n $LOOP_N"
echo "=========================================="
echo ""

# 激活 conda 环境并启动
cd "$PROJECT_DIR"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# 禁用 Python 输出缓冲，确保 stdout/stderr 实时显示
export PYTHONUNBUFFERED=1

# 用 exec 替换当前 shell，rdagent 的所有输出直接接管终端
exec python -m rdagent.app.cli fin_quant --loop-n "$LOOP_N"
