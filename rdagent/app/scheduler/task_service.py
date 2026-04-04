"""
Local JSONL-based persistence for tasks and datasets (interim implementation).

Notes:
- Intended as a placeholder before integrating a real DB/ORM.
- Provides basic CRUD-like helpers for TaskRecord and DatasetRecord.
- Worker/API layers can call these helpers to manage scheduler state.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path
from typing import Any

from .config_service import PROJECT_ROOT
from .models import DatasetRecord, TaskRecord

DATA_DIR = PROJECT_ROOT / "scheduler_data"
TASK_FILE = DATA_DIR / "tasks.jsonl"
DATASET_FILE = DATA_DIR / "datasets.jsonl"
# 使用git_ignore_folder避免污染项目根目录
LOG_DIR = PROJECT_ROOT / "git_ignore_folder" / "logs" / "scheduler_tasks"
RESULT_FILE = DATA_DIR / "results.jsonl"
LOCAL_TZ = timezone(timedelta(hours=8))

# rdagent CLI 输出的日志根目录
RDAGENT_LOG_ROOT = PROJECT_ROOT / "log"

# 日志目录名的时间戳格式
_LOG_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+$")


def _ensure_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (TASK_FILE, DATASET_FILE, RESULT_FILE):
        if not f.exists():
            f.write_text("", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    _ensure_files()
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    _ensure_files()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


# Dataset operations
def list_datasets() -> list[DatasetRecord]:
    return [DatasetRecord(**d) for d in _load_jsonl(DATASET_FILE)]


def create_dataset(rec: DatasetRecord) -> DatasetRecord:
    rec.created_at = datetime.now(LOCAL_TZ)
    _append_jsonl(DATASET_FILE, asdict(rec))
    return rec


# Task operations
def list_tasks() -> list[TaskRecord]:
    return [TaskRecord(**t) for t in _load_jsonl(TASK_FILE)]


def create_task(rec: TaskRecord) -> TaskRecord:
    # 自动分配自增 ID
    if rec.id is None:
        existing = _load_jsonl(TASK_FILE)
        max_id = max((t.get("id") or 0 for t in existing), default=0)
        rec.id = max_id + 1
    rec.created_at = datetime.now(LOCAL_TZ)
    rec.updated_at = rec.created_at
    _append_jsonl(TASK_FILE, asdict(rec))
    return rec


def update_task_status(task_id: str, status: str) -> TaskRecord | None:
    tasks = _load_jsonl(TASK_FILE)
    updated = None
    for t in tasks:
        if str(t.get("id")) == str(task_id) or t.get("name") == task_id:
            t["status"] = status
            t["updated_at"] = datetime.now(LOCAL_TZ).isoformat()
            updated = TaskRecord(**t)
    # rewrite file
    TASK_FILE.write_text("", encoding="utf-8")
    for t in tasks:
        _append_jsonl(TASK_FILE, t)
    return updated


def _update_task_field(task_id: str, field: str, value: Any) -> TaskRecord | None:
    """更新任务的单个字段（通用版）。"""
    tasks = _load_jsonl(TASK_FILE)
    updated = None
    for t in tasks:
        if str(t.get("id")) == str(task_id) or t.get("name") == task_id:
            t[field] = value
            t["updated_at"] = datetime.now(LOCAL_TZ).isoformat()
            updated = TaskRecord(**t)
    # rewrite file
    TASK_FILE.write_text("", encoding="utf-8")
    for t in tasks:
        _append_jsonl(TASK_FILE, t)
    return updated


def get_task(task_id: str) -> TaskRecord | None:
    for t in _load_jsonl(TASK_FILE):
        if str(t.get("id")) == str(task_id) or t.get("name") == task_id:
            return TaskRecord(**t)
    return None


def append_task_log(task_id: str, content: str) -> Path:
    _ensure_files()
    log_path = LOG_DIR / f"{task_id}.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
    return log_path


def read_task_log(task_id: str) -> str:
    _ensure_files()
    log_path = LOG_DIR / f"{task_id}.log"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8")


# Result operations
def record_result(task_id: str, result: dict) -> None:
    payload = {"task_id": task_id, **result}
    _append_jsonl(RESULT_FILE, payload)


def list_results(task_id: str | None = None) -> list[dict[str, Any]]:
    items = _load_jsonl(RESULT_FILE)
    if task_id:
        items = [i for i in items if str(i.get("task_id")) == str(task_id)]
    return items


def _extract_pid_from_log_dir(log_dir: Path) -> int | None:
    """从日志目录中提取最近活跃的 PID。

    快速策略：只扫描最后一个 Loop 目录的第一层子目录中的 PID 目录，
    避免 rglob 全量扫描大量 pkl 文件。

    目录结构: Loop_N/step/substep/PID/*.pkl
    """
    # 找最后一个 Loop 目录
    loop_dirs = sorted(log_dir.glob("Loop_*"), key=lambda p: p.name)
    if not loop_dirs:
        return None

    last_loop = loop_dirs[-1]
    # 在 last_loop 下找 PID 目录 (数字命名，深度 2-3 层)
    latest_pid = None
    latest_mtime = 0.0
    for step_dir in last_loop.iterdir():
        if not step_dir.is_dir():
            continue
        for sub_dir in step_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            # sub_dir 可能是 PID 目录或再深一层
            if sub_dir.name.isdigit():
                try:
                    mt = sub_dir.stat().st_mtime
                except OSError:
                    continue
                if mt > latest_mtime:
                    latest_mtime = mt
                    latest_pid = int(sub_dir.name)
            else:
                # 再看一层
                for pid_dir in sub_dir.iterdir():
                    if pid_dir.is_dir() and pid_dir.name.isdigit():
                        try:
                            mt = pid_dir.stat().st_mtime
                        except OSError:
                            continue
                        if mt > latest_mtime:
                            latest_mtime = mt
                            latest_pid = int(pid_dir.name)
    return latest_pid


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _read_cmdline(pid: int) -> list[str]:
    """读取 /proc/PID/cmdline。"""
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
        return data.decode("utf-8", errors="replace").split("\x00")
    except Exception:
        return []


def _infer_loop_n_from_cmdline(pid: int) -> int | None:
    """从进程命令行读取 --loop-n 参数。"""
    args = _read_cmdline(pid)
    for i, arg in enumerate(args):
        if arg == "--loop-n" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                pass
    return None


def _infer_task_status(log_dir: Path, pid: int | None = None) -> str:
    """根据日志目录内容推断任务状态。

    pid 参数由调用方传入（只对近期活跃的目录提取），避免此函数做昂贵的 PID 扫描。
    """
    # 进程存活 → running
    if pid and _is_pid_alive(pid):
        return "running"

    # 进程不存在 — 检查是否正常完成
    loop_dirs = sorted(log_dir.glob("Loop_*"), key=lambda p: p.name)
    if loop_dirs:
        last_loop = loop_dirs[-1]
        if (last_loop / "feedback").exists():
            return "success"
        # 有 Loop 但最后一个没完成 → fail
        return "fail"

    # 无 Loop 目录（空或刚创建）
    return "pending"


def _parse_log_dir_time(dir_name: str) -> datetime:
    """解析日志目录名中的创建时间。格式: 2026-03-23_17-01-00-775867

    rdagent 用 UTC 生成目录名，这里解析为 UTC 后转本地时区。
    """
    try:
        parts = dir_name.split("_")
        date_part = parts[0]  # 2026-03-23
        time_part = parts[1].replace("-", ":")[:8]  # 17:01:00
        utc_dt = datetime.fromisoformat(f"{date_part}T{time_part}+00:00")
        return utc_dt.astimezone(LOCAL_TZ)
    except Exception:
        return datetime.now(LOCAL_TZ)


def discover_unregistered_tasks() -> int:
    """扫描 rdagent log 目录，发现未注册到 Scheduler 的手动任务。

    性能优化：
    - 只处理时间戳格式的目录
    - 对非运行状态的任务只记录目录结构，不做深度 PID 扫描
    - 一次性读取已注册任务名，避免重复 I/O

    返回新发现的任务数量。
    """
    if not RDAGENT_LOG_ROOT.exists():
        return 0

    # 收集已注册任务的所有标识（一次性读取）
    existing = _load_jsonl(TASK_FILE)
    registered_names = {t.get("name") for t in existing}
    registered_log_dirs = {t.get("rdagent_log_dir") for t in existing if t.get("rdagent_log_dir")}
    # 也排除 scheduler 创建的任务已关联的 workspace_path（可能包含 log dir name）
    for t in existing:
        wp = t.get("workspace_path") or ""
        if wp:
            # workspace_path 可能是完整路径或仅目录名
            registered_log_dirs.add(Path(wp).name)
            registered_log_dirs.add(wp)

    count = 0
    for entry in sorted(RDAGENT_LOG_ROOT.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        # 只处理时间戳格式的目录
        if not _LOG_DIR_RE.match(entry.name):
            continue
        # 跳过已注册
        if entry.name in registered_names or entry.name in registered_log_dirs:
            continue

        # 快速检查：有 Loop_* 子目录才是真正的任务
        loop_dirs = sorted(entry.glob("Loop_*"))
        if not loop_dirs and not list(entry.glob("__session__")):
            continue  # 空目录或非任务目录，跳过

        loop_n_observed = len(loop_dirs)

        # 根据目录名时间戳判断是否近期任务（7 天内才做 PID 提取）
        created_ts = _parse_log_dir_time(entry.name)
        age_days = (datetime.now(LOCAL_TZ) - created_ts).total_seconds() / 86400

        pid = None
        loop_n_target = None
        if age_days < 7:
            # 近期任务：做 PID 提取
            pid = _extract_pid_from_log_dir(entry)
            if pid and _is_pid_alive(pid):
                loop_n_target = _infer_loop_n_from_cmdline(pid)

        status = _infer_task_status(entry, pid=pid if (pid and _is_pid_alive(pid)) else None)
        created = _parse_log_dir_time(entry.name)

        rec = TaskRecord(
            name=entry.name,
            status=status,
            loop_n=loop_n_target or loop_n_observed or 1,
            source="manual",
            rdagent_log_dir=entry.name,
            created_at=created,
            updated_at=created,
        )
        rec = create_task(rec)
        count += 1
        print(f"[discover] registered manual task: {entry.name} (status={status}, pid={pid})")

    return count


__all__ = [
    "DATA_DIR",
    "LOG_DIR",
    "RDAGENT_LOG_ROOT",
    "_LOG_DIR_RE",
    "_update_task_field",
    "_infer_task_status",
    "_extract_pid_from_log_dir",
    "append_task_log",
    "create_dataset",
    "create_task",
    "discover_unregistered_tasks",
    "get_task",
    "list_datasets",
    "list_results",
    "list_tasks",
    "read_task_log",
    "update_task_status",
]
