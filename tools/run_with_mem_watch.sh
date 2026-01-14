#!/usr/bin/env bash
# 用法:
#   ./run_with_mem_watch.sh "<command>" [interval_sec]
# 示例:
#   ./run_with_mem_watch.sh "python -m rdagent.app.cli fin_quant --loop-n 6" 1
#   ./run_with_mem_watch.sh "python -m rdagent.app.cli results_api --host 127.0.0.1 --port 9000" 5

CMD="$1"
# 为了更精细地捕获 OOM 前的内存变化，默认采样间隔改为 1 秒
INTERVAL="${2:-1}"  # 默认 1 秒

if [ -z "$CMD" ]; then
  echo "Usage: $0 \"<command>\" [interval_sec]"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="mem_watch_${TS}.log"

echo "Running: $CMD"
echo "Memory log: $LOG_FILE (interval=${INTERVAL}s)"

# 启动真正的命令
bash -c "$CMD" &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"

# 在后台监控这个 PID 及其直接子进程的内存使用，并定期记录全局内存 Top 进程
(
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    {
      echo "==== $(date '+%F %T') PID=$MAIN_PID ===="
      # 主进程
      ps -p "$MAIN_PID" -o pid,ppid,%mem,%cpu,rss,vsz,cmd --sort=-%mem 2>/dev/null
      echo "--- children ---"
      # 子进程（进程池/子 python 等）
      ps --ppid "$MAIN_PID" -o pid,ppid,%mem,%cpu,rss,vsz,cmd --sort=-%mem 2>/dev/null | head -n 15
      echo "--- top memory processes snapshot ---"
      # 全局内存占用 Top N 进程，帮助定位非直接子进程的高内存占用
      ps aux --sort=-%mem | head -n 15
      echo
    } >> "$LOG_FILE"
    sleep "$INTERVAL"
  done
  echo "Process $MAIN_PID exited, stop logging main loop." >> "$LOG_FILE"
  {
    echo "==== $(date '+%F %T') PID=$MAIN_PID final children check ===="
    echo "--- remaining children with PPID=$MAIN_PID (if any) ---"
    ps --ppid "$MAIN_PID" -o pid,ppid,%mem,%cpu,rss,vsz,cmd --sort=-%mem 2>/dev/null | head -n 20
    echo "--- final top memory processes snapshot ---"
    ps aux --sort=-%mem | head -n 15
    echo
  } >> "$LOG_FILE"
) &

# 等待主命令结束
wait "$MAIN_PID"
RET=$?
echo "Command exited with code $RET, see $LOG_FILE for memory trace."
exit $RET
