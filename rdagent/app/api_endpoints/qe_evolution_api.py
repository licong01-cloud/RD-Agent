"""
QE (QuantEvolver) 演进 API 端点

提供以下接口（双参数嵌套路由）：
1. POST /api/v1/qe_workspace/tasks/{task_id}/loops - 触发新 LOOP 的回测执行
2. GET /api/v1/qe_workspace/tasks/{task_id}/loops/{loop_id}/status - 查询 LOOP 状态
3. GET /api/v1/qe_workspace/tasks/{task_id}/loops/{loop_id}/metrics - 获取 LOOP 回测指标
4. GET /api/v1/qe_workspace/tasks/{task_id}/loops/{loop_id}/assets/download - 打包下载模型资产
5. DELETE /api/v1/qe_workspace/tasks/{task_id} - 清理任务工作区
6. GET /api/v1/qe_workspace/tasks/{task_id}/logs - 任务日志流（SSE）
7. GET /api/v1/qe_workspace/config - 工作区配置信息
"""

import asyncio
import base64
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from rdagent.app.api_endpoints.qe_dataset_identity import read_dataset_identity
from rdagent.app.api_endpoints.qe_environment_identity import (
    ExecutionEnvironmentIdentityError,
    get_execution_environment_identity,
)
from rdagent.app.api_endpoints.qe_kill_receipt import (
    KillReceiptConflictError,
    KillReceiptError,
    KillReceiptValidationError,
    execute_typed_kill_intent,
    public_kill_receipt_payload,
)
from rdagent.app.api_endpoints.qe_submission_receipt import (
    SubmissionReceiptConflictError,
    SubmissionReceiptError,
    SubmissionReceiptValidationError,
    canonical_request_digest,
    capture_process_identity,
    get_submission_receipt,
    get_submission_receipt_locked,
    loop_lifecycle_lock,
    observe_result_artifact,
    promote_submission_receipt_to_completed_from_verified_result_locked,
    public_receipt_payload,
    reserve_submission,
    transition_submission_receipt,
    transition_submission_receipt_locked,
    validate_submission_intent_hash,
    write_loop_status_locked,
    write_process_identity_locked,
)
from rdagent.app.api_endpoints.qe_workspace_catalog import (
    build_workspace_catalog,
    resolve_loop_dir,
    resolve_task_dir,
)
from rdagent.app.api_endpoints.qe_long_trend_evaluation import build_long_trend_router

logger = logging.getLogger(__name__)


_QE_BASH_PATH = "/bin/bash"
_QE_BASH_STARTUP_ENV_KEYS = frozenset({"BASH_ENV", "ENV", "BASHOPTS", "SHELLOPTS"})
_QE_DYNAMIC_LOADER_INJECTION_KEYS = frozenset({"LD_AUDIT", "LD_PRELOAD"})
_QE_DB_CREDENTIAL_PREFIXES = ("TDX_DB_", "POSTGRES_")
_QE_DB_CREDENTIAL_KEYS = frozenset(
    {
        "DATABASE_URL",
        "DB_HOST",
        "DB_NAME",
        "DB_PASSWORD",
        "DB_PORT",
        "DB_USER",
        "PGDATABASE",
        "PGHOST",
        "PGPASSFILE",
        "PGPASSWORD",
        "PGPORT",
        "PGSERVICE",
        "PGUSER",
        "SQLALCHEMY_DATABASE_URI",
        "SQLALCHEMY_DATABASE_URL",
    },
)
_QE_SECRET_CREDENTIAL_KEYS = frozenset(
    {
        "ACCESS_TOKEN",
        "API_KEY",
        "AUTH_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "PASSWORD",
        "PRIVATE_KEY",
        "SECRET",
        "TOKEN",
        "TUSHARE_TOKEN",
    },
)
_QE_SECRET_CREDENTIAL_SUFFIXES = (
    "_ACCESS_KEY_ID",
    "_ACCESS_TOKEN",
    "_API_KEY",
    "_AUTH_TOKEN",
    "_CLIENT_SECRET",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_SECRET_ACCESS_KEY",
    "_SECRET_KEY",
    "_TOKEN",
)
_QE_RESOURCE_SESSION_SECRET_FILE = "qe_resource_session_secret.json"  # noqa: S105 - filename, not a credential.


def _raise_runtime_error(message: str) -> NoReturn:
    raise RuntimeError(message)


def _raise_submission_receipt_error(message: str) -> NoReturn:
    raise SubmissionReceiptError(message)


def _raise_execution_environment_mismatch(
    *,
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> NoReturn:
    raise HTTPException(
        status_code=409,
        detail={
            "reason_code": "qe_execution_environment_identity_mismatch",
            "message": (
                "submitted durable execution identity belongs to a different "
                "QE deployment environment"
            ),
            "expected": expected,
            "actual": actual,
        },
    )


def _is_forbidden_qe_child_env_key(key: str) -> bool:
    """Return whether an environment key is forbidden at the QE exec boundary.

    QE compute is a file-only data plane.  Database and conventional external
    service credentials are carried by neither argv nor environment; the one
    scoped resource-session credential is materialized as a private workspace
    file.  Loader and Bash startup injection variables are also excluded before
    Bash itself starts, so command-level unsets are defense in depth rather
    than the first point at which a credential becomes unavailable.
    """

    name = str(key or "").upper()
    return (
        name in _QE_BASH_STARTUP_ENV_KEYS
        or name in _QE_DYNAMIC_LOADER_INJECTION_KEYS
        or name.startswith("BASH_FUNC_")
        or name.startswith(_QE_DB_CREDENTIAL_PREFIXES)
        or name in _QE_DB_CREDENTIAL_KEYS
        or name in _QE_SECRET_CREDENTIAL_KEYS
        or name.endswith(_QE_SECRET_CREDENTIAL_SUFFIXES)
    )


def _bind_spawned_process_identity(
    *,
    loop_dir: Path,
    loop_id: str,
    submission_intent_hash: str,
    process: subprocess.Popen[Any],
) -> dict[str, int]:
    """Persist exact process identity or kill the otherwise untracked process."""

    bound = False
    try:
        identity = capture_process_identity(process.pid)
        write_process_identity_locked(loop_dir, identity=identity)
        transition_submission_receipt_locked(
            loop_dir,
            loop_id=loop_id,
            submission_intent_hash=submission_intent_hash,
            status="running",
            process_identity=identity,
        )
        write_loop_status_locked(loop_dir, status="running", expected_current={None})
        bound = True
        return identity
    finally:
        if not bound:
            _terminate_untracked_process(process)


def _spawn_qe_process(
    *,
    command: str,
    stdout_fd: int,
    env: dict[str, str],
    cwd: Path,
) -> subprocess.Popen[Any]:
    # QE frozen commands are authored as Bash (for example they use ``source``,
    # ``compgen`` and ``pipefail``).  Execute that contract explicitly rather
    # than relying on /bin/sh, which is dash on the supported Ubuntu workers.
    # Startup files, exported shell functions, loader injection and credential
    # material are excluded before exec.  Errexit plus pipefail makes any
    # unhandled command/pipeline failure the process result.
    child_env = {
        key: value
        for key, value in env.items()
        if not _is_forbidden_qe_child_env_key(key)
    }
    return subprocess.Popen(  # noqa: S603 - audited QE command execution boundary.
        [
            _QE_BASH_PATH,
            "--noprofile",
            "--norc",
            "-o",
            "errexit",
            "-o",
            "pipefail",
            "-c",
            command,
        ],
        stdout=stdout_fd,
        stderr=subprocess.STDOUT,
        env=child_env,
        cwd=str(cwd),
        start_new_session=True,
        close_fds=True,
    )


def _atomic_write_private_text(target: Path, content: str) -> None:
    """Atomically replace *target* with a mode-0600 UTF-8 text file.

    ``mkstemp`` creates the inode private from its first observable moment.
    Replacing only after a flushed write prevents readers from observing a
    partial secret.  Every failure path removes the temporary file while
    leaving any previous target untouched.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    fd = -1
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        temporary = Path(temporary_name)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)  # noqa: PTH105 - explicit atomic replace is the security boundary.
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                logger.exception(
                    "Failed to close temporary QE resource-session secret: %s",
                    temporary,
                )
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Failed to remove temporary QE resource-session secret: %s",
                    temporary,
                )
                raise
        raise


def _write_experiment_file(*, target: Path, relative_path: str, content: str) -> bool:
    """Write one validated experiment file and return whether it was base64.

    Ordinary text and binary files deliberately retain their existing write
    semantics.  Only the resource-session secret uses the private atomic path.
    """

    output_relative_path = relative_path.removesuffix(".b64")
    if output_relative_path == _QE_RESOURCE_SESSION_SECRET_FILE:
        if relative_path.endswith(".b64"):
            message = (
                "QE_RESOURCE_SESSION_SECRET_ENCODING_INVALID: "
                "qe_resource_session_secret.json must be submitted as UTF-8 text"
            )
            raise RuntimeError(message)
        _atomic_write_private_text(target, content)
        return False
    if relative_path.endswith(".b64"):
        target.write_bytes(base64.b64decode(content, validate=True))
        return True
    target.write_text(content, encoding="utf-8")
    return False


def _terminate_untracked_process(process: subprocess.Popen[Any]) -> None:
    try:
        killpg = getattr(os, "killpg", None)
        if killpg is None:
            os.kill(process.pid, signal.SIGKILL)
        else:
            killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        return


def _prepare_created_loop_workspace(
    *,
    loop_dir: Path,
    task_id: str,
    loop_id: str,
    submission_intent_hash: str,
) -> tuple[dict[str, Any], bool]:
    try:
        with loop_lifecycle_lock(loop_dir, loop_id):
            latest = get_submission_receipt_locked(
                loop_dir,
                loop_id=loop_id,
                submission_intent_hash=submission_intent_hash,
            )
            if latest is None:
                _raise_runtime_error(
                    "QE loop reservation disappeared before workspace preparation",
                )
            if latest.get("status") == "cancelled":
                return latest, False
            if latest.get("status") != "reserved":
                _raise_runtime_error(
                    "QE workspace preparation requires a reserved receipt; "
                    f"actual={latest.get('status')!r}",
                )
            if loop_dir.exists():
                if not loop_dir.is_dir():
                    _raise_runtime_error(
                        f"Loop workspace is not a directory: {task_id}/{loop_id}",
                    )
                shutil.rmtree(loop_dir)
            return latest, True
    except (OSError, RuntimeError, SubmissionReceiptError) as exc:
        with loop_lifecycle_lock(loop_dir, loop_id):
            latest = get_submission_receipt_locked(
                loop_dir,
                loop_id=loop_id,
                submission_intent_hash=submission_intent_hash,
            )
            if latest is not None and latest.get("status") not in {
                "completed",
                "failed",
                "cancelled",
            }:
                transition_submission_receipt_locked(
                    loop_dir,
                    loop_id=loop_id,
                    submission_intent_hash=submission_intent_hash,
                    status="failed",
                )
                write_loop_status_locked(
                    loop_dir,
                    status="failed",
                    expected_current={None},
                )
        _raise_runtime_error(
            f"failed to prepare clean retry workspace for {task_id}/{loop_id}: {exc}",
        )

# 确保 .env 文件中的环境变量在模块加载时可用（热重载安全）
# override=False：不覆盖已存在的 shell 环境变量
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    if load_dotenv is not None:
        load_dotenv(_env_path, override=False)
    else:
        # python-dotenv 未安装时，手动解析 .env（仅补充缺失的环境变量）
        with open(_env_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip("'\"")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

router = APIRouter(prefix="/api/v1/qe_workspace", tags=["qe_evolution"])

# ── QE 专属配置 ──
# QE 与 RDAgent 主程序完全隔离，路径配置通过 RDAgent .env 环境变量管理。
# 未来部署到独立服务器时，通过启动命令或 .env 设置环境变量即可。
_WORKSPACE_BASE_RAW = str(os.environ.get("QE_WORKSPACE_WSL") or "").strip()
WORKSPACE_BASE = Path(_WORKSPACE_BASE_RAW) if _WORKSPACE_BASE_RAW else Path()
WORKSPACE_CONFIGURED = bool(_WORKSPACE_BASE_RAW)
if not WORKSPACE_CONFIGURED:
    logger.warning("QE_WORKSPACE_WSL 环境变量未设置，QE API 的 workspace 功能将不可用")

router.include_router(build_long_trend_router(WORKSPACE_BASE if WORKSPACE_CONFIGURED else None))

class LoopRunRequest(BaseModel):
    loop_index: int
    config: dict[str, Any]
    experiment_files: dict[str, str] | None = None
    wsl_command: str | None = None
    callback_url: str | None = None
    model_source: dict[str, Any] | None = None
    submission_intent_hash: str
    execution_identity_hash: str | None = None
    execution_environment_snapshot_id: str | None = None
    execution_environment_manifest_sha256: str | None = None
    postprocess_descriptor: dict[str, Any] | None = None

class LoopRunResponse(BaseModel):
    loop_id: str
    status: str
    message: str
    submission_intent_hash: str
    request_digest: str
    receipt_status: str
    duplicate_replay: bool
    execution_identity_hash: str | None = None
    execution_environment_snapshot_id: str | None = None
    execution_environment_manifest_sha256: str | None = None


class TypedKillIntentRequest(BaseModel):
    command_id: str
    kill_intent_generation: int
    kill_intent_hash: str
    expected_submission_intent_hash: str
    expected_process_identity: dict[str, int] | None = None
    expected_phase: str | None = None

def _get_task_dir(task_id: str) -> Path:
    return resolve_task_dir(WORKSPACE_BASE, task_id)

def _get_loop_dir(task_id: str, loop_id: str) -> Path:
    return resolve_loop_dir(WORKSPACE_BASE, task_id, loop_id)


def _append_log(loop_dir: Path, message: str):
    os.makedirs(loop_dir, exist_ok=True)
    log_file = loop_dir / "run.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def _resolve_loop_write_target(loop_dir: Path, relative_path: str) -> Path:
    raw_path = str(relative_path or "")
    normalized = raw_path.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(raw_path)
    if (
        not raw_path
        or "\x00" in raw_path
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise RuntimeError(
            f"QE_WORKSPACE_PATH_ESCAPE: invalid loop-relative path: {relative_path!r}",
        )
    loop_root = loop_dir.resolve()
    target = (loop_root / Path(*posix_path.parts)).resolve()
    try:
        target.relative_to(loop_root)
    except ValueError as exc:
        raise RuntimeError(
            f"QE_WORKSPACE_PATH_ESCAPE: path resolves outside Loop workspace: {relative_path!r}",
        ) from exc
    if target == loop_root:
        raise RuntimeError(
            f"QE_WORKSPACE_PATH_ESCAPE: file path resolves to Loop directory: {relative_path!r}",
        )
    return target


def _safe_extract_tar_to_loop(tar: Any, loop_dir: Path) -> None:
    planned: list[tuple[Any, Path]] = []
    seen_targets: set[Path] = set()
    for member in tar.getmembers():
        target = _resolve_loop_write_target(loop_dir, member.name)
        if target in seen_targets:
            raise RuntimeError(
                f"QE_WORKSPACE_TAR_UNSAFE: duplicate archive target: {member.name!r}",
            )
        seen_targets.add(target)
        if not (member.isdir() or member.isfile()):
            raise RuntimeError(
                "QE_WORKSPACE_TAR_UNSAFE: links, devices, and special entries are "
                f"not allowed: {member.name!r}",
            )
        planned.append((member, target))

    for member, target in planned:
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError(
                f"QE_WORKSPACE_TAR_UNSAFE: regular file has no readable payload: {member.name!r}",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def _receipt_http_error(exc: SubmissionReceiptError) -> HTTPException:
    if isinstance(exc, SubmissionReceiptConflictError):
        return HTTPException(
            status_code=409,
            detail={
                "reason_code": "qe_workspace_submission_identity_conflict",
                "message": str(exc),
            },
        )
    if isinstance(exc, SubmissionReceiptValidationError):
        return HTTPException(
            status_code=400,
            detail={
                "reason_code": "qe_workspace_submission_intent_invalid",
                "message": str(exc),
            },
        )
    return HTTPException(
        status_code=500,
        detail={
            "reason_code": "qe_workspace_submission_receipt_error",
            "message": str(exc),
        },
    )


def _kill_receipt_http_error(exc: KillReceiptError) -> HTTPException:
    if isinstance(exc, KillReceiptConflictError):
        return HTTPException(
            status_code=409,
            detail={
                "reason_code": "qe_workspace_typed_kill_identity_conflict",
                "message": str(exc),
            },
        )
    if isinstance(exc, KillReceiptValidationError):
        return HTTPException(
            status_code=400,
            detail={
                "reason_code": "qe_workspace_typed_kill_invalid",
                "message": str(exc),
            },
        )
    return HTTPException(
        status_code=500,
        detail={
            "reason_code": "qe_workspace_typed_kill_receipt_error",
            "message": str(exc),
        },
    )


def _status_with_receipt(status: str, receipt: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status}
    if receipt is not None:
        public = public_receipt_payload(receipt)
        payload.update(
            {
                "receipt_status": public["status"],
                "submission_intent_hash": public["submission_intent_hash"],
                "request_digest": public["request_digest"],
            },
        )
    return payload

async def _run_qlib_backtest(
    task_id: str,
    loop_id: str,
    config: dict[str, Any],
    experiment_files: dict[str, str] | None,
    wsl_command: str | None = None,
    callback_url: str | None = None,
    model_source: dict[str, Any] | None = None,
    *,
    submission_intent_hash: str,
):
    """
    后台任务：执行 QLib 回测。

    关键设计：
    - start_new_session=True: 子进程在独立会话中运行，uvicorn --reload 不会杀死它
    - stdout 重定向到日志文件（非 PIPE）：父进程死后子进程不会因管道断裂收到 SIGPIPE
    - pid.txt 记录子进程真实 PID（非 FastAPI worker PID）：健康检查准确判断
    - read_exp_res.py 集成到命令链：父进程死后仍会自动执行生成结果文件
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    os.makedirs(loop_dir, exist_ok=True)
    status_file = loop_dir / "status.txt"

    try:
        with loop_lifecycle_lock(loop_dir, loop_id):
            receipt = get_submission_receipt_locked(
                loop_dir,
                loop_id=loop_id,
                submission_intent_hash=submission_intent_hash,
            )
            if receipt is None:
                _raise_runtime_error(
                    "QE background execution has no matching durable submission receipt",
                )
            if receipt.get("status") == "cancelled":
                _append_log(loop_dir, "[START] submission already cancelled before setup")
                return
            if receipt.get("status") != "reserved":
                _raise_runtime_error(
                    "QE background execution requires a reserved submission receipt; "
                    f"actual={receipt.get('status')!r}",
                )

        # 保存配置
        config_file = loop_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f)

        logger.info(f"Starting QLib backtest for {loop_id} with config: {config}")
        _append_log(loop_dir, f"[INFO] Starting QLib backtest with config={json.dumps(config, ensure_ascii=False)}")

        # 写入实验文件
        if experiment_files:
            written_targets: set[Path] = set()
            for rel_path, content in experiment_files.items():
                output_rel_path = rel_path.removesuffix(".b64")
                validated_target = _resolve_loop_write_target(loop_dir, output_rel_path)
                if validated_target in written_targets:
                    raise RuntimeError(
                        f"QE_WORKSPACE_PATH_CONFLICT: duplicate output target: {rel_path!r}",
                    )
                written_targets.add(validated_target)
                validated_target.parent.mkdir(parents=True, exist_ok=True)
                decoded_from_base64 = _write_experiment_file(
                    target=validated_target,
                    relative_path=rel_path,
                    content=content,
                )
                if decoded_from_base64:
                    # base64 编码的二进制文件（如 benchmark_sh000300.parquet.b64）
                    _append_log(loop_dir, f"[INFO] Wrote binary file: {rel_path[:-4]} (decoded from b64)")
                else:
                    _append_log(loop_dir, f"[INFO] Wrote experiment file: {rel_path}")

        # 将环境变量注入，确保 model.py 等模块可被 qrun 导入
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{loop_dir}:{env.get('PYTHONPATH', '')}"
        env.setdefault("PYTHONUNBUFFERED", "1")

        # model_source mlruns 符号链接（策略演进复复用源任务训练的模型）
        if model_source:
            cross_node = model_source.get("cross_node", False)
            dst_mlruns = loop_dir / "mlruns"

            if cross_node:
                # 跨节点：从 experiment_files 中解压 mlruns_params.tar.gz
                tar_b64_file = loop_dir / "mlruns_params.tar.gz"
                if tar_b64_file.exists():
                    import io
                    import tarfile
                    tar_data = tar_b64_file.read_bytes()
                    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
                        _safe_extract_tar_to_loop(tar, loop_dir)
                    tar_b64_file.unlink()
                    _append_log(loop_dir, f"[INFO] Cross-node mlruns extracted to {dst_mlruns}")
                else:
                    msg = "Cross-node mode requires mlruns_params.tar.gz, but it was not provided"
                    _append_log(loop_dir, f"[ERROR] {msg}")
                    raise RuntimeError(msg)
            else:
                # 同节点：符号链接
                # qlib task_train 将 params.pkl 保存在 workspace 根级 mlruns/ 下，
                # 而 Loop*/mlruns/ 仅包含 QE 自身记录的元数据（无模型权重）。
                # 因此优先链接根级 mlruns，fallback 到 loop 级。
                src_task = model_source.get("source_task_id", "")
                src_loop = model_source.get("source_loop", "")
                if src_task and src_loop:
                    source_loop_dir = resolve_loop_dir(WORKSPACE_BASE, src_task, src_loop)
                    src_mlruns_root = source_loop_dir.parent / "mlruns"
                    src_mlruns_loop = source_loop_dir / "mlruns"
                    src_mlruns = src_mlruns_root if src_mlruns_root.exists() else src_mlruns_loop
                    if src_mlruns.exists() and not dst_mlruns.exists():
                        os.symlink(str(src_mlruns), str(dst_mlruns))
                        _append_log(loop_dir, f"[INFO] Symlink mlruns: {src_mlruns} → {dst_mlruns}")

        # 构造执行命令
        if wsl_command:
            # AIstock 传入的自定义 WSL 命令（auto 模式已包含 read_exp_res.py）
            final_cmd = wsl_command
            _append_log(loop_dir, f"[INFO] Using wsl_command: {final_cmd}")
        else:
            # 默认命令链：cd → prepare_factors → qrun → read_exp_res
            cmd_parts = [f"cd {shlex.quote(str(loop_dir))}"]
            if (loop_dir / "prepare_factors.py").exists():
                cmd_parts.append("python prepare_factors.py")
            cmd_parts.append("qrun conf.yaml")
            # 将 read_exp_res.py 集成到命令链，确保即使父进程重启也能生成结果文件
            if (loop_dir / "read_exp_res.py").exists():
                cmd_parts.append("python read_exp_res.py")
            final_cmd = " && ".join(cmd_parts)
            _append_log(loop_dir, f"[INFO] Executing command: {final_cmd}")

        # Bind cancellation-before-start, subprocess creation, and full process
        # incarnation persistence to one shared cross-process lifecycle lock.
        # A typed pre-start cancellation either wins before Popen or sees a fully
        # recorded PID/PGID/start-tick identity; it can never race in between.
        with loop_lifecycle_lock(loop_dir, loop_id):
            receipt = get_submission_receipt_locked(
                loop_dir,
                loop_id=loop_id,
                submission_intent_hash=submission_intent_hash,
            )
            if receipt is None:
                _raise_runtime_error(
                    "QE background execution lost its durable submission receipt before Popen",
                )
            if receipt.get("status") == "cancelled":
                _append_log(loop_dir, "[START] submission cancelled before subprocess creation")
                return
            if receipt.get("status") != "reserved":
                _raise_runtime_error(
                    "QE Popen requires a reserved submission receipt; "
                    f"actual={receipt.get('status')!r}",
                )
            existing_status = (
                status_file.read_text(encoding="utf-8").strip()
                if status_file.exists()
                else None
            )
            if existing_status is not None:
                _raise_runtime_error(
                    "QE Popen requires no existing status sidecar for this reserved attempt; "
                    f"actual={existing_status!r}",
                )
            transition_submission_receipt_locked(
                loop_dir,
                loop_id=loop_id,
                submission_intent_hash=submission_intent_hash,
                status="started",
            )
            log_file_path = loop_dir / "run.log"
            log_fd = os.open(str(log_file_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
            try:
                _proc = _spawn_qe_process(
                    command=final_cmd,
                    stdout_fd=log_fd,
                    env=env,
                    cwd=loop_dir,
                )
            finally:
                os.close(log_fd)
            process_identity = _bind_spawned_process_identity(
                loop_dir=loop_dir,
                loop_id=loop_id,
                submission_intent_hash=submission_intent_hash,
                process=_proc,
            )
        _append_log(
            loop_dir,
            "[INFO] Subprocess started, "
            f"pid={process_identity['pid']} pgid={process_identity['pgid']} "
            f"start_time_ticks={process_identity['start_time_ticks']}",
        )

        # 在线程池中等待子进程完成（不阻塞事件循环）
        # 如果 uvicorn reload 导致父进程重启，此 await 会被取消，但子进程继续运行
        loop = asyncio.get_event_loop()
        returncode = await loop.run_in_executor(None, _proc.wait)

        if returncode != 0:
            _raise_runtime_error(f"QLib backtest failed with return code {returncode}")

        # Persist the terminal sidecar and receipt under the same lifecycle lock.
        with loop_lifecycle_lock(loop_dir, loop_id):
            receipt = get_submission_receipt_locked(
                loop_dir,
                loop_id=loop_id,
                submission_intent_hash=submission_intent_hash,
            )
            if receipt is None:
                _raise_runtime_error("QE completion lost its durable submission receipt")
            if receipt.get("status") != "running":
                _raise_runtime_error(
                    "QE completion cannot overwrite a concurrent terminal/cancellation state; "
                    f"actual={receipt.get('status')!r}",
                )
            transition_submission_receipt_locked(
                loop_dir,
                loop_id=loop_id,
                submission_intent_hash=submission_intent_hash,
                status="completed",
            )
            write_loop_status_locked(loop_dir, status="completed", expected_current={"running"})
        _append_log(loop_dir, f"[DONE] loop={loop_id} status=completed")
        logger.info(f"Completed QLib backtest for {loop_id}")

        # 通知 AIstock backend（Loop 完成回调）
        if callback_url:
            try:
                import httpx as _httpx
                _cb_payload = {
                    "loop_id": f"{task_id}_{loop_id}",
                    "task_id": task_id,
                    "status": "completed",
                }
                async with _httpx.AsyncClient(timeout=10) as _cb:
                    await _cb.post(callback_url, json=_cb_payload)
                logger.info(f"Callback sent to {callback_url} for {loop_id}")
            except Exception as cb_err:
                logger.warning(f"Callback failed for {loop_id}: {cb_err}")

    except Exception as e:
        logger.error(f"Backtest failed for {loop_id}: {e}")
        try:
            with loop_lifecycle_lock(loop_dir, loop_id):
                receipt = get_submission_receipt_locked(
                    loop_dir,
                    loop_id=loop_id,
                    submission_intent_hash=submission_intent_hash,
                )
                current = status_file.read_text(encoding="utf-8").strip() if status_file.exists() else ""
                if receipt is None:
                    _raise_runtime_error("QE failure has no durable submission receipt")
                receipt_status = str(receipt.get("status") or "")
                result_observation = observe_result_artifact(loop_dir)
                if bool(result_observation.get("valid")):
                    # A result artifact is authoritative even when a late
                    # exception (for example callback/log cleanup) reaches this
                    # failure path.  The shared locked repair primitive prevents
                    # a completed result from being downgraded to failed/cancelled.
                    receipt = promote_submission_receipt_to_completed_from_verified_result_locked(
                        loop_dir,
                        loop_id=loop_id,
                        submission_intent_hash=submission_intent_hash,
                    )
                    if current != "completed":
                        write_loop_status_locked(
                            loop_dir,
                            status="completed",
                            expected_current={current or None},
                        )
                elif receipt_status not in {"completed", "failed", "cancelled"}:
                    terminal_status = (
                        current if current in {"completed", "failed", "cancelled"} else "failed"
                    )
                    transition_submission_receipt_locked(
                        loop_dir,
                        loop_id=loop_id,
                        submission_intent_hash=submission_intent_hash,
                        status=terminal_status,
                    )
                    if current not in {"completed", "failed", "cancelled"}:
                        write_loop_status_locked(
                            loop_dir,
                            status=terminal_status,
                            expected_current={None, "running"},
                        )
                elif current not in {"completed", "failed", "cancelled"}:
                    write_loop_status_locked(
                        loop_dir,
                        status=receipt_status,
                        expected_current={None, "running"},
                    )
                (loop_dir / "error.log").write_text(str(e), encoding="utf-8")
        except SubmissionReceiptError as receipt_error:
            logger.error(
                "Failed to persist terminal submission receipt for %s/%s: %s",
                task_id,
                loop_id,
                receipt_error,
            )
            _append_log(loop_dir, f"[ERROR] submission receipt terminal update failed: {receipt_error}")
        except (OSError, RuntimeError, TypeError, ValueError) as receipt_error:
            logger.exception(
                "Failed to persist terminal QE loop state for %s/%s",
                task_id,
                loop_id,
            )
            _append_log(loop_dir, f"[ERROR] terminal lifecycle update failed: {receipt_error}")
        _append_log(loop_dir, f"[ERROR] loop={loop_id} error={e!s}")

@router.get("/tasks/{task_id}/loops/{loop_id}/mlruns-params")
async def download_mlruns_params(task_id: str, loop_id: str):
    """打包 loop 的 mlruns 中训练好的模型 run 完整目录，返回 tar.gz。

    背景：仅打包 params.pkl 时，下游 mlflow 找不到 experiment 目录的 meta.yaml，
    会触发 MissingConfigException 并 fallback 到 loose params.pkl 加载，
    后者依赖 PYTHONPATH 中存在自定义 model 模块，跨节点常因模块缺失而 pickle
    反序列化失败 (No module named 'model')，导致 backtest-only 复用模型彻底失败。

    修复：定位每个 params.pkl 后，沿目录上溯打包整个 run 目录（含 meta.yaml /
    metrics / params / tags / artifacts）以及对应 experiment 目录的 meta.yaml，
    确保下游 R.get_exp(...) + recorder.load_object(...) 能走主路径。
    """
    import glob as _glob
    import io
    import tarfile
    loop_dir = _get_loop_dir(task_id, loop_id)
    mlruns_dir = loop_dir / "mlruns"
    if not mlruns_dir.exists():
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "mlruns not found"})
    params_files = _glob.glob(str(mlruns_dir / "**" / "params.pkl"), recursive=True)
    if not params_files:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "params.pkl not found"})

    # 收集需要打包的路径集合（去重，保持目录结构）。
    # mlruns 标准布局：mlruns/<exp_id>/<run_id>/artifacts/params.pkl
    # 即 params.pkl 的祖父是 run_dir，曾祖父是 exp_dir。
    files_to_pack: set[Path] = set()
    seen_exp_meta: set[Path] = set()
    for pf in params_files:
        pf_path = Path(pf)
        artifacts_dir = pf_path.parent
        run_dir = artifacts_dir.parent if artifacts_dir.name == "artifacts" else pf_path.parent
        exp_dir = run_dir.parent
        # 整个 run 目录（含 meta.yaml/metrics/params/tags/artifacts/...）
        if run_dir.is_dir():
            for p in run_dir.rglob("*"):
                if p.is_file():
                    files_to_pack.add(p)
        # experiment 级别 meta.yaml（mlflow R.get_exp 必读）
        exp_meta = exp_dir / "meta.yaml"
        if exp_meta not in seen_exp_meta and exp_meta.is_file():
            files_to_pack.add(exp_meta)
            seen_exp_meta.add(exp_meta)

    if not files_to_pack:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": "params.pkl found but run/exp dirs missing"})

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for fp in sorted(files_to_pack):
            arcname = os.path.relpath(str(fp), str(loop_dir))
            tar.add(str(fp), arcname=arcname)
    buf.seek(0)
    from fastapi.responses import Response
    return Response(content=buf.read(), media_type="application/gzip")

@router.get("/tasks/{task_id}/logs")
async def stream_task_logs(task_id: str):
    """
    输出任务日志流（SSE），供 AIstock 侧转发展示。
    """
    task_dir = _get_task_dir(task_id)

    async def event_generator():
        seen_offsets: dict[str, int] = {}
        idle_count = 0
        _MAX_IDLE = 300  # 300秒无新日志则终止 SSE
        while True:
            if not task_dir.exists():
                payload = {"status": "waiting", "logs": [f"Task directory not found yet: {task_id}"]}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(1)
                idle_count += 1
                if idle_count >= _MAX_IDLE:
                    payload = {"status": "timeout", "logs": ["SSE stream timeout: task directory not found"]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return
                continue

            had_new_lines = False
            loop_dirs = sorted([p for p in task_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
            for loop_dir in loop_dirs:
                log_file = loop_dir / "run.log"
                if not log_file.exists():
                    continue

                file_key = str(log_file)
                offset = seen_offsets.get(file_key, 0)
                with open(log_file, encoding="utf-8") as f:
                    f.seek(offset)
                    new_lines = [line.rstrip("\n") for line in f]
                    seen_offsets[file_key] = f.tell()

                if new_lines:
                    had_new_lines = True
                    payload = {"status": "running", "logs": [f"[{loop_dir.name}] {line}" for line in new_lines]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # 检查是否所有 loop 都到达终态（支持并行策略演进场景）
            if loop_dirs:
                all_terminal = True
                any_loop_has_status = False
                for ld in loop_dirs:
                    sf = ld / "status.txt"
                    if sf.exists():
                        any_loop_has_status = True
                        st = sf.read_text().strip()
                        if st not in ("completed", "failed", "cancelled"):
                            all_terminal = False
                            break
                    else:
                        all_terminal = False
                        break
                if any_loop_has_status and all_terminal:
                    final_status = "completed"
                    for ld in loop_dirs:
                        sf = ld / "status.txt"
                        if sf.exists():
                            loop_status = sf.read_text().strip()
                            if loop_status == "failed":
                                final_status = "failed"
                                break
                            if loop_status == "cancelled":
                                final_status = "cancelled"
                    payload = {"status": final_status, "logs": [f"All loops finished with status: {final_status}"]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return

            if had_new_lines:
                idle_count = 0
            else:
                idle_count += 1
                if idle_count >= _MAX_IDLE:
                    payload = {"status": "timeout", "logs": ["SSE stream timeout: no new logs"]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return

            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/execution-environment")
async def get_execution_environment() -> dict[str, Any]:
    """Return the cached, content-addressed QE deployment manifest.

    This endpoint is intentionally independent of a loop request.  It does not
    run `nvidia-smi`, inspect GPU/VRAM, or poll process state; the owning service
    constructs the manifest once per deployment and returns the same snapshot on
    every call.
    """

    try:
        return get_execution_environment_identity()
    except ExecutionEnvironmentIdentityError as exc:
        logger.exception("QE execution environment identity is unavailable")
        raise HTTPException(
            status_code=503,
            detail={
                "reason_code": "qe_execution_environment_identity_unavailable",
                "message": str(exc),
            },
        ) from exc


@router.get("/dataset-identity")
async def get_dataset_identity(
    node_id: str,
    data_root_uri: str | None = None,
) -> dict[str, Any]:
    """Return a verified QE dataset deployment manifest or visible evidence.

    This route only reads a pre-published manifest under an explicitly
    configured node-local root.  It never scans arbitrary directories and an
    incomplete result is evidence for follow-up data acquisition, not an
    approval or research-direction rejection.
    """

    return read_dataset_identity(data_root_uri=data_root_uri, node_id=node_id)


@router.post("/tasks/{task_id}/loops", response_model=LoopRunResponse)
async def create_and_run_loop(task_id: str, request: LoopRunRequest, background_tasks: BackgroundTasks):
    """
    接收演进配置并触发 QLib 回测
    """
    loop_id = f"Loop{request.loop_index}"
    loop_dir = _get_loop_dir(task_id, loop_id)

    try:
        intent_hash = validate_submission_intent_hash(request.submission_intent_hash)
        if any(
            value is not None
            for value in (
                request.execution_identity_hash,
                request.execution_environment_snapshot_id,
                request.execution_environment_manifest_sha256,
            )
        ):
            current_environment = get_execution_environment_identity()
            expected_binding = {
                "execution_environment_snapshot_id": current_environment[
                    "execution_environment_snapshot_id"
                ],
                "execution_environment_manifest_sha256": current_environment[
                    "execution_environment_manifest_sha256"
                ],
            }
            actual_binding = {
                "execution_environment_snapshot_id": (
                    request.execution_environment_snapshot_id
                ),
                "execution_environment_manifest_sha256": (
                    request.execution_environment_manifest_sha256
                ),
            }
            if actual_binding != expected_binding:
                _raise_execution_environment_mismatch(
                    expected=expected_binding,
                    actual=actual_binding,
                )
        request_digest = canonical_request_digest(
            loop_index=request.loop_index,
            config=request.config,
            experiment_files=request.experiment_files,
            wsl_command=request.wsl_command,
            model_source=request.model_source,
            execution_identity_hash=request.execution_identity_hash,
            execution_environment_snapshot_id=request.execution_environment_snapshot_id,
            execution_environment_manifest_sha256=request.execution_environment_manifest_sha256,
            postprocess_descriptor=request.postprocess_descriptor,
        )
        receipt, created = reserve_submission(
            loop_dir,
            task_id=task_id,
            loop_id=loop_id,
            submission_intent_hash=intent_hash,
            request_digest=request_digest,
            execution_identity_hash=request.execution_identity_hash,
            execution_environment_snapshot_id=request.execution_environment_snapshot_id,
            execution_environment_manifest_sha256=request.execution_environment_manifest_sha256,
        )
        if created:
            receipt, created = _prepare_created_loop_workspace(
                loop_dir=loop_dir,
                task_id=task_id,
                loop_id=loop_id,
                submission_intent_hash=intent_hash,
            )
            if created:
                background_tasks.add_task(
                    _run_qlib_backtest,
                    task_id,
                    loop_id,
                    request.config,
                    request.experiment_files,
                    request.wsl_command,
                    request.callback_url,
                    request.model_source,
                    submission_intent_hash=intent_hash,
                )

        return LoopRunResponse(
            loop_id=loop_id,
            status="accepted",
            message=(
                f"Loop {loop_id} accepted and reserved for background execution"
                if created
                else f"Loop {loop_id} already has the same durable submission receipt"
            ),
            submission_intent_hash=intent_hash,
            request_digest=request_digest,
            receipt_status=str(receipt["status"]),
            duplicate_replay=not created,
            execution_identity_hash=receipt.get("execution_identity_hash"),
            execution_environment_snapshot_id=receipt.get("execution_environment_snapshot_id"),
            execution_environment_manifest_sha256=receipt.get("execution_environment_manifest_sha256"),
        )
    except SubmissionReceiptError as exc:
        logger.error("Failed to reserve loop %s for task %s: %s", loop_id, task_id, exc)
        raise _receipt_http_error(exc) from exc
    except ExecutionEnvironmentIdentityError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "reason_code": "qe_execution_environment_identity_unavailable",
                "message": str(exc),
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger loop {loop_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/loops/{loop_id}/submission")
async def get_loop_submission(
    task_id: str,
    loop_id: str,
    submission_intent_hash: str | None = None,
):
    """Return the durable create receipt even before a Loop status file exists."""

    loop_dir = _get_loop_dir(task_id, loop_id)
    try:
        receipt = get_submission_receipt(
            loop_dir,
            loop_id=loop_id,
            submission_intent_hash=submission_intent_hash,
        )
    except SubmissionReceiptError as exc:
        raise _receipt_http_error(exc) from exc
    if receipt is None:
        return {
            "schema_version": "qe_submission_receipt_v1",
            "task_id": task_id,
            "loop_id": loop_id,
            "status": "not_reserved",
        }
    return public_receipt_payload(receipt)


@router.get("/tasks/{task_id}/loops/{loop_id}/status")
async def get_loop_status(task_id: str, loop_id: str):
    """
    查询 LOOP 状态。

    状态判断逻辑（准确判断，不做推测）：
    1. status.txt 存在且为 completed/failed → 直接返回（终态）
    2. status.txt 为 running → 检查 PID 是否存活：
       - PID 存活 → running
       - PID 不存活 + 结果文件存在 → completed（进程正常退出但 status.txt 未更新）
       - PID 不存活 + 无结果文件 → failed（进程异常终止）
    3. status.txt 不存在 → not_found
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    task_dir = _get_task_dir(task_id)
    status_file = loop_dir / "status.txt"

    try:
        receipt = get_submission_receipt(loop_dir, loop_id=loop_id)
    except SubmissionReceiptError as exc:
        raise _receipt_http_error(exc) from exc

    if not status_file.exists():
        if receipt is None:
            return {"status": "not_found"}
        receipt_status = str(receipt.get("status") or "")
        effective_status = "reserved_not_started" if receipt_status == "reserved" else receipt_status
        return _status_with_receipt(effective_status, receipt)

    status = status_file.read_text().strip()

    # 终态直接返回
    if status in ("completed", "failed", "cancelled"):
        if status != "completed" and receipt is not None and bool(observe_result_artifact(loop_dir).get("valid")):
            try:
                with loop_lifecycle_lock(loop_dir, loop_id):
                    latest = get_submission_receipt_locked(
                        loop_dir,
                        loop_id=loop_id,
                        submission_intent_hash=str(receipt["submission_intent_hash"]),
                    )
                    if latest is None:
                        _raise_submission_receipt_error(
                            "QE terminal status has no matching durable submission receipt",
                        )
                    receipt = promote_submission_receipt_to_completed_from_verified_result_locked(
                        loop_dir,
                        loop_id=loop_id,
                        submission_intent_hash=str(latest["submission_intent_hash"]),
                    )
                    write_loop_status_locked(
                        loop_dir,
                        status="completed",
                        expected_current={status},
                    )
                    status = "completed"
            except SubmissionReceiptError as exc:
                raise _receipt_http_error(exc) from exc
        if receipt is not None and receipt.get("status") != status:
            try:
                receipt = transition_submission_receipt(
                    loop_dir,
                    loop_id=loop_id,
                    submission_intent_hash=str(receipt["submission_intent_hash"]),
                    status=status,
                )
            except SubmissionReceiptError as exc:
                raise _receipt_http_error(exc) from exc
        return _status_with_receipt(status, receipt)

    if status == "running":
        pid_file = loop_dir / "pid.txt"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)  # 不发信号，仅检查进程是否存在
                return _status_with_receipt("running", receipt)
            except (ProcessLookupError, OSError):
                # PID 不存活，检查结果文件确定最终状态（loop 目录优先，兼容 task 目录）
                result_file = loop_dir / "qlib_results_enhanced.json"
                if not result_file.exists():
                    result_file = task_dir / "qlib_results_enhanced.json"
                if result_file.exists():
                    status = "completed"
                else:
                    status = "failed"
                _append_log(loop_dir, f"[DETECT] pid={pid} no longer alive, result_file_exists={result_file.exists()}, marking as {status}")
                try:
                    with loop_lifecycle_lock(loop_dir, loop_id):
                        latest = get_submission_receipt_locked(loop_dir, loop_id=loop_id)
                        current = status_file.read_text(encoding="utf-8").strip() if status_file.exists() else None
                        if latest is not None and latest.get("status") not in {
                            "completed",
                            "failed",
                            "cancelled",
                        }:
                            receipt = transition_submission_receipt_locked(
                                loop_dir,
                                loop_id=loop_id,
                                submission_intent_hash=str(latest["submission_intent_hash"]),
                                status=status,
                            )
                        else:
                            receipt = latest
                            if latest is not None and latest.get("status") in {
                                "completed",
                                "failed",
                                "cancelled",
                            }:
                                status = str(latest["status"])
                        if current not in {"completed", "failed", "cancelled"}:
                            write_loop_status_locked(
                                loop_dir,
                                status=status,
                                expected_current={None, "running"},
                            )
                except SubmissionReceiptError as exc:
                    raise _receipt_http_error(exc) from exc
                return _status_with_receipt(status, receipt)
        else:
            # 无 pid.txt，无法判断进程状态，保持当前状态
            return _status_with_receipt(status, receipt)

    # 其他未知状态（如 interrupted），直接返回
    return _status_with_receipt(status, receipt)

@router.post("/tasks/{task_id}/loops/{loop_id}/kill-intents")
async def submit_typed_kill_intent(
    task_id: str,
    loop_id: str,
    request: TypedKillIntentRequest,
) -> dict[str, Any]:
    """Apply one PID-reuse-safe, durable cancellation intent for a QE loop.

    This endpoint deliberately returns a typed receipt for every observed state.
    A successful HTTP response means the command was durably observed; the
    receipt's status, terminal_reason, and observations remain authoritative.
    """

    loop_dir = _get_loop_dir(task_id, loop_id)
    try:
        receipt = execute_typed_kill_intent(
            loop_dir,
            task_id=task_id,
            loop_id=loop_id,
            command_id=request.command_id,
            kill_intent_generation=request.kill_intent_generation,
            kill_intent_hash=request.kill_intent_hash,
            expected_submission_intent_hash=request.expected_submission_intent_hash,
            expected_process_identity=request.expected_process_identity,
            expected_phase=request.expected_phase,
        )
    except KillReceiptError as exc:
        raise _kill_receipt_http_error(exc) from exc
    return public_kill_receipt_payload(receipt)


@router.post("/tasks/{task_id}/loops/{loop_id}/kill")
async def kill_loop(task_id: str, loop_id: str):
    """
    终止正在运行的 Loop 进程。

    通过 pid.txt 获取子进程 PID，发送 SIGTERM 给进程组（优雅终止）。
    如果进程未在 5 秒内退出，发送 SIGKILL（强制终止）。
    使用 asyncio.sleep 避免阻塞事件循环。
    """
    import signal
    loop_dir = _get_loop_dir(task_id, loop_id)
    pid_file = loop_dir / "pid.txt"
    status_file = loop_dir / "status.txt"

    try:
        with loop_lifecycle_lock(loop_dir, loop_id):
            receipt = get_submission_receipt_locked(loop_dir, loop_id=loop_id)
            if not pid_file.exists():
                if receipt is not None and receipt.get("status") in {"reserved", "started"}:
                    receipt = transition_submission_receipt_locked(
                        loop_dir,
                        loop_id=loop_id,
                        submission_intent_hash=str(receipt["submission_intent_hash"]),
                        status="cancelled",
                    )
                    current = status_file.read_text(encoding="utf-8").strip() if status_file.exists() else None
                    write_loop_status_locked(
                        loop_dir,
                        status="cancelled",
                        expected_current={current},
                    )
                    _append_log(loop_dir, "[KILL] Cancelled reserved submission before subprocess start")
                    return {
                        "killed": False,
                        "pid": None,
                        "status": "cancelled",
                        "receipt_status": receipt["status"],
                    }
                raise HTTPException(status_code=404, detail=f"No pid.txt found for {task_id}/{loop_id}")
    except SubmissionReceiptError as exc:
        raise _receipt_http_error(exc) from exc

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid pid.txt: {e}")

    if pid <= 0:
        raise HTTPException(status_code=400, detail=f"Invalid PID value: {pid}")

    killed = False
    error_detail = None
    try:
        os.kill(pid, 0)  # Check if alive

        # 尝试杀进程组（包括子进程），失败则回退到单进程
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            _append_log(loop_dir, f"[KILL] Sent SIGTERM to process group pgid={pgid} (pid={pid})")
        except (ProcessLookupError, OSError):
            os.kill(pid, signal.SIGTERM)
            _append_log(loop_dir, f"[KILL] Sent SIGTERM to pid={pid} (pgid fallback)")

        # 非阻塞等待进程退出
        for _ in range(10):
            await asyncio.sleep(0.5)
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, OSError):
                killed = True
                break

        if not killed:
            # Force kill
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            _append_log(loop_dir, f"[KILL] Sent SIGKILL to pid={pid}")
            killed = True

    except (ProcessLookupError, OSError):
        # Process already dead
        killed = True
        _append_log(loop_dir, f"[KILL] pid={pid} already dead")
    except Exception as e:
        error_detail = str(e)
        _append_log(loop_dir, f"[KILL] Unexpected error killing pid={pid}: {e}")
        logger.error(f"Kill loop {task_id}/{loop_id} pid={pid} error: {e}")

    # Preserve legacy response semantics while making its state writers share the
    # same cross-process lock/CAS as typed cancellation and background startup.
    try:
        with loop_lifecycle_lock(loop_dir, loop_id):
            receipt = get_submission_receipt_locked(loop_dir, loop_id=loop_id)
            current = status_file.read_text(encoding="utf-8").strip() if status_file.exists() else ""
            if killed and current not in {"completed", "failed", "cancelled"}:
                if receipt is not None and receipt.get("status") not in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    receipt = transition_submission_receipt_locked(
                        loop_dir,
                        loop_id=loop_id,
                        submission_intent_hash=str(receipt["submission_intent_hash"]),
                        status="cancelled",
                    )
                write_loop_status_locked(
                    loop_dir,
                    status="cancelled",
                    expected_current={None, "running", "started"},
                )
                current = "cancelled"
                _append_log(loop_dir, "[KILL] Marked status as cancelled")
    except SubmissionReceiptError as exc:
        raise _receipt_http_error(exc) from exc

    result_status = current or (str(receipt.get("status")) if receipt is not None else "unknown")
    result = {"killed": killed, "pid": pid, "status": result_status}
    if receipt is not None:
        result["receipt_status"] = receipt["status"]
    if error_detail:
        result["error"] = error_detail
    return result

@router.get("/tasks/{task_id}/loops/{loop_id}/metrics")
async def get_loop_metrics(task_id: str, loop_id: str):
    """
    获取 LOOP 回测指标。

    read_exp_res.py 输出文件固定为：
    - qlib_results_enhanced.json（完整增强指标，含 summary 字段）
    搜索顺序：loop 目录（wsl_command cd 目标） → task 目录（兼容旧路径）。
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    task_dir = _get_task_dir(task_id)

    enhanced_file = None
    for search_dir in [loop_dir, task_dir]:
        candidate = search_dir / "qlib_results_enhanced.json"
        if candidate.exists():
            enhanced_file = candidate
            break

    if enhanced_file is None:
        raise HTTPException(status_code=404, detail="Metrics not ready: qlib_results_enhanced.json not found")

    with open(enhanced_file, encoding="utf-8") as f:
        data = json.load(f)

    return data.get("summary", data)

@router.get("/tasks/{task_id}/loops/{loop_id}/enhanced-metrics")
async def get_loop_enhanced_metrics(task_id: str, loop_id: str):
    """
    获取 LOOP 增强诊断指标（IC 时间序列、训练过程、收益曲线等）。
    从 qlib_results_enhanced.json 提取诊断子段，文件不存在则 404。
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    task_dir = _get_task_dir(task_id)

    enhanced_file = None
    for search_dir in [loop_dir, task_dir]:
        candidate = search_dir / "qlib_results_enhanced.json"
        if candidate.exists():
            enhanced_file = candidate
            break

    if enhanced_file is None:
        raise HTTPException(status_code=404, detail=f"qlib_results_enhanced.json not found in {loop_dir} or {task_dir}")

    with open(enhanced_file, encoding="utf-8") as f:
        full_data = json.load(f)

    _SECTION_KEYS = [
        "summary",
        "ic_diagnostics", "return_curves", "training_diagnostics",
        "trade_diagnostics", "prediction_diagnostics",
        "feature_importance", "top_stocks", "bottom_stocks", "stock_trades",
        "factor_analysis", "absolute_returns", "all_stocks", "stock_pnl_summary",
        "limit_analysis",
    ]
    result = {}
    for key in _SECTION_KEYS:
        if key in full_data:
            result[key] = full_data[key]

    # IC key 标准化: ic_dates → dates (前端期望 dates)
    ic = result.get("ic_diagnostics")
    if ic and "ic_dates" in ic and "dates" not in ic:
        ic["dates"] = ic.pop("ic_dates")

    if not result:
        raise HTTPException(status_code=500, detail="qlib_results_enhanced.json exists but contains no diagnostic sections")

    return result


@router.get("/tasks/{task_id}/loops/{loop_id}/assets/download")
async def download_loop_assets(task_id: str, loop_id: str):
    """
    模型资产打包(ZIP)下载
    """
    loop_dir = _get_loop_dir(task_id, loop_id)

    if not loop_dir.exists():
        raise HTTPException(status_code=404, detail="Loop workspace not found")

    zip_path = loop_dir / f"{loop_id}_assets.zip"

    # Create zip file containing models/ and features_order.txt
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            models_dir = loop_dir / "models"
            if models_dir.exists():
                for root, _, files in os.walk(models_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, loop_dir)
                        zipf.write(file_path, arcname)

            features_file = loop_dir / "features_order.txt"
            if features_file.exists():
                zipf.write(features_file, "features_order.txt")

    except Exception as e:
        logger.error(f"Failed to create assets zip for {loop_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create zip package")

    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="No assets found to download")

    return FileResponse(
        path=zip_path,
        filename=f"{loop_id}_assets.zip",
        media_type="application/zip",
    )


@router.get("/tasks/{task_id}/loops/{loop_id}/files")
async def list_workspace_files(task_id: str, loop_id: str) -> dict[str, Any]:
    """Return a complete, read-only loop-relative asset inventory.

    The endpoint never hashes or copies files and never follows symbolic links.
    Completed loops declare a complete namespace catalog; non-terminal loops
    remain explicitly partial so callers cannot mistake an in-flight snapshot
    for reproducible evidence.
    """

    loop_dir = resolve_loop_dir(WORKSPACE_BASE, task_id, loop_id)
    if not loop_dir.exists() or not loop_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Loop workspace not found: {task_id}/{loop_id}",
        )
    return build_workspace_catalog(loop_dir, task_id=task_id, loop_id=loop_id)


@router.get("/tasks/{task_id}/loops/{loop_id}/files/{file_path:path}")
async def get_workspace_file(task_id: str, loop_id: str, file_path: str):
    """读取 workspace 中的指定文件（用于多Alpha跨节点收集 multi_alpha_results.json 等）。

    安全：限制在 loop_dir 内，防止路径穿越攻击。
    """
    loop_dir = resolve_loop_dir(WORKSPACE_BASE, task_id, loop_id)
    if not loop_dir.exists():
        raise HTTPException(status_code=404, detail=f"Loop workspace not found: {task_id}/{loop_id}")

    target = (loop_dir / file_path).resolve()
    loop_dir_resolved = loop_dir.resolve()

    # 防止路径穿越
    try:
        target.relative_to(loop_dir_resolved)
    except ValueError:
        raise HTTPException(status_code=403, detail="路径越界")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # 根据扩展名决定 content-type
    suffix = target.suffix.lower()
    media_type = "application/octet-stream"
    if suffix == ".json":
        media_type = "application/json"
    elif suffix in (".txt", ".log"):
        media_type = "text/plain"
    elif suffix == ".pkl":
        media_type = "application/octet-stream"

    return FileResponse(path=target, media_type=media_type, filename=target.name)


@router.get("/tasks/{task_id}/loops/{loop_id}/groups/{group_name}/predictions")
async def get_group_predictions(task_id: str, loop_id: str, group_name: str):
    """下载指定组的 pred.pkl 文件（用于跨节点收集预测进行 meta 合并）。

    查找顺序：
      1. group_{group_name}/output/pred.pkl
      2. group_{group_name}/mlruns/**/artifacts/pred.pkl
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    if not loop_dir.exists():
        raise HTTPException(status_code=404, detail=f"Loop workspace not found: {task_id}/{loop_id}")

    group_dir = loop_dir / f"group_{group_name}"
    if not group_dir.exists():
        raise HTTPException(status_code=404, detail=f"Group not found: {group_name}")

    # 查找 pred.pkl
    candidates = [
        group_dir / "output" / "pred.pkl",
        group_dir / "pred.pkl",
    ]
    # 兜底：在 mlruns/artifacts 中搜索
    if not any(p.exists() for p in candidates):
        for p in group_dir.rglob("pred.pkl"):
            candidates.append(p)

    for pred_path in candidates:
        if pred_path.exists() and pred_path.is_file():
            return FileResponse(
                path=pred_path,
                media_type="application/octet-stream",
                filename=f"group_{group_name}_pred.pkl",
            )

    raise HTTPException(status_code=404, detail=f"pred.pkl not found for group {group_name}")


@router.delete("/tasks/{task_id}")
async def cleanup_task_workspace(task_id: str):
    """
    彻底删除任务工作区
    """
    task_dir = _get_task_dir(task_id)
    if task_dir.exists():
        try:
            shutil.rmtree(task_dir)
            return {"ok": True, "task_id": task_id}
        except Exception as e:
            logger.error(f"Failed to clean up workspace {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "task_id": task_id}


@router.delete("/tasks/{task_id}/loops/{loop_id}")
async def cleanup_loop_workspace(task_id: str, loop_id: str):
    """
    Delete a single Loop workspace under one task.

    This endpoint is intentionally scoped to task_id/loop_id so AIstock rerun
    can remove stale Loop artifacts without deleting sibling Loop results.
    """
    task_dir = _get_task_dir(task_id).resolve()
    loop_dir = (_get_loop_dir(task_id, loop_id)).resolve()
    try:
        loop_dir.relative_to(task_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="loop_id resolves outside task workspace") from exc

    if not loop_dir.exists():
        return {"ok": True, "task_id": task_id, "loop_id": loop_id, "existed": False}
    if not loop_dir.is_dir():
        raise HTTPException(status_code=500, detail=f"Loop workspace is not a directory: {loop_id}")

    try:
        shutil.rmtree(loop_dir)
        return {"ok": True, "task_id": task_id, "loop_id": loop_id, "existed": True}
    except Exception as e:
        logger.error(f"Failed to clean up loop workspace {task_id}/{loop_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_workspace_config():
    """
    返回 QE 工作区配置信息，供 AIstock 动态获取路径。
    所有路径直接从环境变量读取（RDAgent .env）。
    """
    return {
        "workspace_base": str(WORKSPACE_BASE),
        "factor_data_dir": os.environ.get("RDAGENT_FACTOR_DATA_WSL", ""),
        "qlib_data_path": os.environ.get("QLIB_DATA_PATH_WSL", ""),
    }
