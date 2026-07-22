"""Loop-owned durable API for QE long-trend postprocess jobs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from rdagent.app.api_endpoints.qe_environment_identity import get_execution_environment_identity
from rdagent.app.api_endpoints.qe_workspace_catalog import resolve_loop_dir

JOB_SCHEMA = "qe_long_trend_job_v1"
RECEIPT_SCHEMA = "qe_long_trend_job_receipt_v1"
ARTIFACT_CATALOG_SCHEMA = "qe_long_trend_node_artifact_catalog_v1"
BUNDLE_SCHEMA = "qe_long_trend_bundle_v1"
TERMINAL_STATUSES = frozenset({"succeeded", "partial", "failed", "cancelled"})
ACTIVE_STATUSES = frozenset({"queued", "starting", "running"})
DISPATCHER_IDENTITY_FILE = ".qe_long_trend_dispatcher_identity.json"
DISPATCHER_SPAWN_LOCK = ".qe_long_trend_dispatcher_spawn.lock"
_EVALUATION_RE = re.compile(r"^qelt_[0-9a-f]{64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_BUNDLE_PATHS = frozenset(
    {
        "backend/__init__.py",
        "backend/services/__init__.py",
        "backend/services/quantevolver/__init__.py",
        "backend/services/quantevolver/long_trend_evaluation_contract.py",
        "backend/services/quantevolver/long_trend_data_reader.py",
        "backend/services/quantevolver/long_trend_evaluation.py",
        "backend/services/quantevolver/qe_dataset_contract.py",
        "backend/services/quantevolver/long_trend_pickle_parser_entry.py",
        "backend/services/quantevolver/long_trend_worker_entry.py",
        "bundle_manifest.json",
    },
)


class QELongTrendJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "qe_long_trend_job_request_v1"
    evaluation_id: str
    run_id: str
    node_id: str
    profile_id: str
    profile_sha256: str
    evaluator_version: str
    evaluator_source_sha256: str
    execution_environment_snapshot_id: str
    execution_environment_manifest_sha256: str
    bundle_sha256: str
    qe_dataset_contract_id: str
    feature_snapshot: dict[str, Any]
    outcome_snapshot: dict[str, Any]
    feature_data_root_uri: str
    outcome_data_root_uri: str
    input_manifest_sha256: str
    input_artifact_hashes: dict[str, Any]
    artifact_paths: dict[str, str | None]
    artifact_hashes: dict[str, str]
    recorder_ref: dict[str, Any]
    catalog_digest: str
    catalog_completeness: str
    backtest_freq: str
    evaluation_asof: str
    label_horizon: int | None = None
    strategy_topk: int | None = None
    bundle: dict[str, Any]
    resource_session: dict[str, Any]
    resource_session_token: SecretStr
    resource_callback_url: str
    parser_timeout_seconds: int = Field(default=300, ge=30, le=1800)


class QELongTrendCancelIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_attempt_id: str
    expected_process_identity: dict[str, Any]
    expected_request_sha: str


def build_long_trend_router(workspace_base: Path | None) -> APIRouter:
    router = APIRouter()

    def require_workspace() -> Path:
        if workspace_base is None:
            raise HTTPException(
                status_code=503,
                detail={"reason_code": "QELT_WORKSPACE_NOT_CONFIGURED"},
            )
        return workspace_base

    @router.post("/tasks/{task_id}/loops/{loop_id}/long-trend-evaluations")
    async def create_job(
        task_id: str,
        loop_id: str,
        request: QELongTrendJobRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        configured_workspace = require_workspace()
        loop_dir = resolve_loop_dir(configured_workspace, task_id, loop_id)
        try:
            receipt = reserve_long_trend_job(
                loop_dir=loop_dir,
                task_id=task_id,
                loop_id=loop_id,
                request=request,
            )
        except QELongTrendNodeError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"reason_code": exc.reason_code, "message": str(exc), "context": exc.context},
            ) from exc
        background_tasks.add_task(spawn_long_trend_dispatcher, configured_workspace)
        return receipt

    @router.get("/tasks/{task_id}/loops/{loop_id}/long-trend-evaluations/{evaluation_id}")
    async def inspect_job(task_id: str, loop_id: str, evaluation_id: str) -> dict[str, Any]:
        job_dir = _evaluation_dir(resolve_loop_dir(require_workspace(), task_id, loop_id), evaluation_id)
        try:
            return inspect_long_trend_job(job_dir, task_id=task_id, loop_id=loop_id)
        except QELongTrendNodeError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"reason_code": exc.reason_code, "message": str(exc), "context": exc.context},
            ) from exc

    @router.get("/tasks/{task_id}/loops/{loop_id}/long-trend-evaluations/{evaluation_id}/artifacts")
    async def artifact_catalog(task_id: str, loop_id: str, evaluation_id: str) -> dict[str, Any]:
        job_dir = _evaluation_dir(resolve_loop_dir(require_workspace(), task_id, loop_id), evaluation_id)
        try:
            return build_long_trend_artifact_catalog(job_dir)
        except QELongTrendNodeError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"reason_code": exc.reason_code, "message": str(exc), "context": exc.context},
            ) from exc

    @router.get("/tasks/{task_id}/loops/{loop_id}/long-trend-evaluations/{evaluation_id}/artifacts/{artifact_path:path}")
    async def download_artifact(task_id: str, loop_id: str, evaluation_id: str, artifact_path: str):
        job_dir = _evaluation_dir(resolve_loop_dir(require_workspace(), task_id, loop_id), evaluation_id)
        target = _artifact_download_path(job_dir, artifact_path)
        if not target.is_file() or target.is_symlink():
            raise HTTPException(status_code=404, detail={"reason_code": "QELT_ARTIFACT_NOT_FOUND"})
        return FileResponse(target)

    @router.post("/tasks/{task_id}/loops/{loop_id}/long-trend-evaluations/{evaluation_id}/cancel-intents")
    async def cancel_job(
        task_id: str,
        loop_id: str,
        evaluation_id: str,
        intent: QELongTrendCancelIntent,
    ) -> dict[str, Any]:
        configured_workspace = require_workspace()
        job_dir = _evaluation_dir(resolve_loop_dir(configured_workspace, task_id, loop_id), evaluation_id)
        try:
            result = cancel_long_trend_attempt(job_dir, intent=intent)
        except QELongTrendNodeError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"reason_code": exc.reason_code, "message": str(exc), "context": exc.context},
            ) from exc
        spawn_long_trend_dispatcher(configured_workspace)
        return result

    return router


class QELongTrendNodeError(RuntimeError):
    def __init__(self, message: str, *, reason_code: str, status_code: int = 409, context: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.reason_code = reason_code
        self.status_code = int(status_code)
        self.context = dict(context or {})


def reserve_long_trend_job(
    *,
    loop_dir: Path,
    task_id: str,
    loop_id: str,
    request: QELongTrendJobRequest,
) -> dict[str, Any]:
    _validate_request_identity(request)
    environment = get_execution_environment_identity()
    expected_environment = {
        "execution_environment_snapshot_id": environment["execution_environment_snapshot_id"],
        "execution_environment_manifest_sha256": environment["execution_environment_manifest_sha256"],
    }
    actual_environment = {
        "execution_environment_snapshot_id": request.execution_environment_snapshot_id,
        "execution_environment_manifest_sha256": request.execution_environment_manifest_sha256,
    }
    if actual_environment != expected_environment:
        raise QELongTrendNodeError(
            "long-trend request belongs to a different execution environment",
            reason_code="QELT_EXECUTION_ENVIRONMENT_MISMATCH",
            context={"expected": expected_environment, "actual": actual_environment},
        )
    job_dir = _evaluation_dir(loop_dir, request.evaluation_id)
    public_request = request.model_dump(mode="json", exclude={"resource_session_token"})
    public_request["resource_session"] = dict(request.resource_session)
    request_sha = _sha256_json(public_request)
    job_id = f"qelt_job_{hashlib.sha256((request.evaluation_id + request.node_id).encode()).hexdigest()}"
    lock_path = job_dir.parent / ".reservation_locks" / f"{request.evaluation_id}.lock"
    with _exclusive_file_lock(lock_path):
        if job_dir.exists() and (job_dir / "job.json").is_file():
            existing = _read_json(job_dir / "job.json")
            if existing.get("request_sha") != request_sha or existing.get("job_id") != job_id:
                raise QELongTrendNodeError(
                    "evaluation identity already has different request content",
                    reason_code="QELT_NODE_JOB_IDENTITY_CONFLICT",
                )
            return _receipt(existing, task_id=task_id, loop_id=loop_id, duplicate=True)
        if job_dir.exists():
            import shutil

            shutil.rmtree(job_dir, ignore_errors=False)
        job_dir.mkdir(parents=True, exist_ok=False)
        runtime_dir = job_dir / "runtime"
        runtime_dir.mkdir()
        try:
            _write_verified_bundle(runtime_dir, request.bundle, request.bundle_sha256)
            _atomic_json(job_dir / "request.json", public_request)
            secret = {
                "resource_session_token": request.resource_session_token.get_secret_value(),
                "resource_callback_url": request.resource_callback_url,
            }
            _atomic_json(job_dir / "secret.json", secret, mode=0o600)
            job = {
                "schema_version": JOB_SCHEMA,
                "task_id": task_id,
                "loop_id": loop_id,
                "evaluation_id": request.evaluation_id,
                "job_id": job_id,
                "request_sha": request_sha,
                "status": "queued",
                "current_attempt_id": None,
                "execution_environment_snapshot_id": request.execution_environment_snapshot_id,
                "execution_environment_manifest_sha256": request.execution_environment_manifest_sha256,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
            }
            _atomic_json(job_dir / "job.json", job)
            return _receipt(job, task_id=task_id, loop_id=loop_id, duplicate=False)
        except Exception:
            # No job.json means this exact directory never became a durable queued job.
            if not (job_dir / "job.json").exists():
                import shutil

                shutil.rmtree(job_dir, ignore_errors=False)
            raise


def inspect_long_trend_job(job_dir: Path, *, task_id: str, loop_id: str) -> dict[str, Any]:
    job = _read_json_required(job_dir / "job.json", reason_code="QELT_NODE_JOB_NOT_FOUND", status_code=404)
    attempt_dir = _current_attempt_dir(job_dir, job)
    process_identity = _read_json(attempt_dir / "process_identity.json") if attempt_dir and (attempt_dir / "process_identity.json").is_file() else None
    terminal = _read_json(attempt_dir / "terminal_receipt.json") if attempt_dir and (attempt_dir / "terminal_receipt.json").is_file() else None
    return {
        **job,
        "task_id": task_id,
        "loop_id": loop_id,
        "process_identity": process_identity,
        "terminal_receipt": terminal,
    }


def build_long_trend_artifact_catalog(job_dir: Path) -> dict[str, Any]:
    job = _read_json_required(job_dir / "job.json", reason_code="QELT_NODE_JOB_NOT_FOUND", status_code=404)
    attempt_dir = _current_attempt_dir(job_dir, job)
    if attempt_dir is None:
        return {"schema_version": ARTIFACT_CATALOG_SCHEMA, "evaluation_id": job["evaluation_id"], "status": job["status"], "artifacts": []}
    artifacts_dir = attempt_dir / "artifacts"
    terminal_path = attempt_dir / "terminal_receipt.json"
    paths = []
    if artifacts_dir.is_dir():
        paths.extend(path for path in artifacts_dir.iterdir() if path.is_file() and not path.is_symlink())
    if terminal_path.is_file() and not terminal_path.is_symlink():
        paths.append(terminal_path)
    rows = []
    for path in sorted(paths, key=lambda item: item.name):
        digest, size = _sha256_file(path)
        relative = path.relative_to(job_dir).as_posix()
        rows.append({"relative_path": relative, "sha256": digest, "size_bytes": size})
    return {
        "schema_version": ARTIFACT_CATALOG_SCHEMA,
        "evaluation_id": job["evaluation_id"],
        "status": job["status"],
        "artifacts": rows,
    }


def cancel_long_trend_attempt(job_dir: Path, *, intent: QELongTrendCancelIntent) -> dict[str, Any]:
    job = _read_json_required(job_dir / "job.json", reason_code="QELT_NODE_JOB_NOT_FOUND", status_code=404)
    if job["status"] in TERMINAL_STATUSES:
        return {"schema_version": "qe_long_trend_cancel_receipt_v1", "status": "already_terminal", "evaluation_id": job["evaluation_id"]}
    if job.get("request_sha") != intent.expected_request_sha or job.get("current_attempt_id") != intent.expected_attempt_id:
        raise QELongTrendNodeError(
            "cancel intent does not match current evaluation attempt",
            reason_code="QELT_NODE_PROCESS_IDENTITY_CONFLICT",
        )
    attempt_dir = _current_attempt_dir(job_dir, job)
    actual = _read_json_required(
        attempt_dir / "process_identity.json" if attempt_dir else Path("missing"),
        reason_code="QELT_NODE_PROCESS_IDENTITY_CONFLICT",
    )
    if actual != intent.expected_process_identity or not _process_identity_alive(actual):
        raise QELongTrendNodeError(
            "cancel process identity is stale or no longer alive",
            reason_code="QELT_NODE_PROCESS_IDENTITY_CONFLICT",
        )
    pid = int(actual["pid"])
    try:
        process_group_id = os.getpgid(pid)
    except (OSError, ProcessLookupError) as exc:
        raise QELongTrendNodeError(
            "cancel process group is no longer available",
            reason_code="QELT_NODE_PROCESS_IDENTITY_CONFLICT",
        ) from exc
    if process_group_id != pid:
        raise QELongTrendNodeError(
            "long-trend worker is not the expected process-group leader",
            reason_code="QELT_NODE_PROCESS_IDENTITY_CONFLICT",
            context={"pid": pid, "process_group_id": process_group_id},
        )
    os.killpg(process_group_id, signal.SIGTERM)
    cancel = {
        "schema_version": "qe_long_trend_cancel_receipt_v1",
        "status": "signal_sent",
        "evaluation_id": job["evaluation_id"],
        "attempt_id": intent.expected_attempt_id,
        "process_identity": actual,
        "process_group_id": process_group_id,
        "created_at": _utc_now(),
    }
    _atomic_json(attempt_dir / "cancel_intent.json", cancel)
    return cancel


def spawn_long_trend_dispatcher(workspace_base: Path) -> bool:
    workspace = Path(workspace_base).resolve()
    if not workspace.is_dir():
        return False
    identity_path = workspace / DISPATCHER_IDENTITY_FILE
    with _exclusive_file_lock(workspace / DISPATCHER_SPAWN_LOCK):
        if identity_path.is_file():
            identity = _read_json(identity_path)
            if _process_identity_alive(identity):
                return False
            identity_path.unlink()
        env = dict(os.environ)
        env["QE_WORKSPACE_WSL"] = str(workspace)
        process = subprocess.Popen(  # noqa: S603 - fixed module invocation, no shell.
            [sys.executable, "-m", "rdagent.app.api_endpoints.qe_long_trend_worker", "--workspace-root", str(workspace)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=str(Path(__file__).resolve().parents[3]),
            start_new_session=True,
            close_fds=True,
        )
        try:
            identity = _capture_process_identity(process.pid)
        except Exception as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError as kill_exc:
                raise QELongTrendNodeError(
                    "dispatcher exited before its durable process identity could be captured",
                    reason_code="QELT_NODE_PROCESS_IDENTITY_CONFLICT",
                    context={
                        "pid": process.pid,
                        "capture_error_type": type(exc).__name__,
                        "capture_error": str(exc),
                        "cleanup_error_type": type(kill_exc).__name__,
                    },
                ) from exc
            raise QELongTrendNodeError(
                "dispatcher process identity capture failed",
                reason_code="QELT_NODE_PROCESS_IDENTITY_CONFLICT",
                context={
                    "pid": process.pid,
                    "capture_error_type": type(exc).__name__,
                    "capture_error": str(exc),
                },
            ) from exc
        _atomic_json(identity_path, identity)
        return True


def spawn_long_trend_registration_replayer(workspace_base: Path) -> None:
    if not Path(workspace_base).is_dir():
        return
    subprocess.Popen(  # noqa: S603 - fixed module invocation, no shell.
        [
            sys.executable,
            "-m",
            "rdagent.app.api_endpoints.qe_long_trend_registration_replayer",
            "--workspace-root",
            str(Path(workspace_base).resolve()),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=dict(os.environ),
        cwd=str(Path(__file__).resolve().parents[3]),
        start_new_session=True,
        close_fds=True,
    )


def _write_verified_bundle(runtime_dir: Path, bundle: Mapping[str, Any], expected_bundle_sha: str) -> None:
    if bundle.get("schema_version") != BUNDLE_SCHEMA or bundle.get("bundle_sha256") != expected_bundle_sha:
        raise QELongTrendNodeError("bundle identity is invalid", reason_code="QELT_BUNDLE_INVALID")
    manifest = bundle.get("manifest")
    files = bundle.get("files")
    if not isinstance(manifest, Mapping) or not isinstance(files, Mapping):
        raise QELongTrendNodeError("bundle manifest/files are invalid", reason_code="QELT_BUNDLE_INVALID")
    manifest_core = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    if _sha256_json(manifest_core) != expected_bundle_sha:
        raise QELongTrendNodeError("bundle manifest hash mismatch", reason_code="QELT_BUNDLE_INVALID")
    if set(files) != _ALLOWED_BUNDLE_PATHS:
        raise QELongTrendNodeError(
            "bundle file allowlist mismatch",
            reason_code="QELT_BUNDLE_INVALID",
            context={"extra": sorted(set(files) - _ALLOWED_BUNDLE_PATHS), "missing": sorted(_ALLOWED_BUNDLE_PATHS - set(files))},
        )
    rows = {str(row.get("relative_path")): row for row in manifest.get("files", []) if isinstance(row, Mapping)}
    for relative, source in files.items():
        _safe_relative(relative)
        if not isinstance(source, str):
            raise QELongTrendNodeError("bundle source must be UTF-8 text", reason_code="QELT_BUNDLE_INVALID")
        payload = source.encode("utf-8")
        if relative != "bundle_manifest.json":
            row = rows.get(relative)
            if not row or row.get("sha256") != hashlib.sha256(payload).hexdigest() or int(row.get("size_bytes") or -1) != len(payload):
                raise QELongTrendNodeError(f"bundle file hash mismatch: {relative}", reason_code="QELT_BUNDLE_INVALID")
        target = (runtime_dir / relative).resolve()
        target.relative_to(runtime_dir.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise QELongTrendNodeError(f"bundle target already exists: {relative}", reason_code="QELT_BUNDLE_INVALID")
        _atomic_bytes(target, payload)


def _validate_request_identity(request: QELongTrendJobRequest) -> None:
    if request.schema_version != "qe_long_trend_job_request_v1" or not _EVALUATION_RE.fullmatch(request.evaluation_id):
        raise QELongTrendNodeError("invalid long-trend request identity", reason_code="QELT_NODE_JOB_IDENTITY_CONFLICT")
    for field in (
        "profile_sha256", "evaluator_source_sha256", "execution_environment_manifest_sha256",
        "bundle_sha256", "input_manifest_sha256", "catalog_digest",
    ):
        if not _SHA_RE.fullmatch(str(getattr(request, field) or "")):
            raise QELongTrendNodeError(f"invalid {field}", reason_code="QELT_NODE_JOB_IDENTITY_CONFLICT")
    if request.catalog_completeness not in {"complete", "partial"}:
        raise QELongTrendNodeError("invalid catalog completeness", reason_code="QELT_NODE_JOB_IDENTITY_CONFLICT")


def _evaluation_dir(loop_dir: Path, evaluation_id: str) -> Path:
    if not _EVALUATION_RE.fullmatch(str(evaluation_id or "")):
        raise QELongTrendNodeError("invalid evaluation_id", reason_code="QELT_NODE_JOB_IDENTITY_CONFLICT", status_code=403)
    root = (loop_dir / "long_trend_evaluations").resolve()
    target = (root / evaluation_id).resolve()
    target.relative_to(root)
    return target


def _artifact_download_path(job_dir: Path, value: str) -> Path:
    _safe_relative(value)
    target = (job_dir / value).resolve()
    target.relative_to(job_dir.resolve())
    job = _read_json_required(job_dir / "job.json", reason_code="QELT_NODE_JOB_NOT_FOUND", status_code=404)
    attempt_dir = _current_attempt_dir(job_dir, job)
    allowed: set[Path] = set()
    if attempt_dir is not None:
        artifacts_dir = attempt_dir / "artifacts"
        if artifacts_dir.is_dir():
            allowed.update(
                path.resolve()
                for path in artifacts_dir.iterdir()
                if path.is_file() and not path.is_symlink()
            )
        terminal = attempt_dir / "terminal_receipt.json"
        if terminal.is_file() and not terminal.is_symlink():
            allowed.add(terminal.resolve())
    if target not in allowed:
        raise HTTPException(status_code=403, detail={"reason_code": "QELT_ARTIFACT_PATH_FORBIDDEN"})
    return target


def _safe_relative(value: str) -> None:
    path = PurePosixPath(str(value or ""))
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise QELongTrendNodeError("path escapes evaluation directory", reason_code="QELT_ARTIFACT_PATH_FORBIDDEN", status_code=403)


def _current_attempt_dir(job_dir: Path, job: Mapping[str, Any]) -> Path | None:
    attempt_id = str(job.get("current_attempt_id") or "")
    if not attempt_id:
        return None
    target = (job_dir / "attempts" / attempt_id).resolve()
    target.relative_to((job_dir / "attempts").resolve())
    return target


def _receipt(job: Mapping[str, Any], *, task_id: str, loop_id: str, duplicate: bool) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "task_id": task_id,
        "loop_id": loop_id,
        "evaluation_id": job["evaluation_id"],
        "job_id": job["job_id"],
        "request_sha": job["request_sha"],
        "status": job["status"],
        "duplicate_replay": bool(duplicate),
        "current_attempt_id": job.get("current_attempt_id"),
        "execution_environment_snapshot_id": job["execution_environment_snapshot_id"],
        "execution_environment_manifest_sha256": job["execution_environment_manifest_sha256"],
    }


def _process_identity_alive(identity: Mapping[str, Any]) -> bool:
    try:
        pid = int(identity["pid"])
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ")
        return (
            stat[2] != "Z"
            and int(stat[21]) == int(identity["start_ticks"])
            and hashlib.sha256(command).hexdigest() == identity["command_sha256"]
        )
    except (KeyError, ValueError, OSError, ProcessLookupError):
        return False


def _capture_process_identity(pid: int) -> dict[str, Any]:
    stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").split()
    command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\x00", b" ")
    return {
        "pid": int(pid),
        "start_ticks": int(stat[21]),
        "command_sha256": hashlib.sha256(command).hexdigest(),
    }


def _read_json_required(path: Path, *, reason_code: str, status_code: int = 409) -> dict[str, Any]:
    if not path.is_file():
        raise QELongTrendNodeError(f"required durable file is missing: {path.name}", reason_code=reason_code, status_code=status_code)
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QELongTrendNodeError(
            f"durable JSON is unreadable: {path}: {exc}",
            reason_code="QELT_NODE_JOB_IDENTITY_CONFLICT",
        ) from exc
    if not isinstance(payload, dict):
        raise QELongTrendNodeError("durable JSON must be an object", reason_code="QELT_NODE_JOB_IDENTITY_CONFLICT")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o644) -> None:
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _atomic_bytes(path, encoded, mode=mode)


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_replace(source: Path, target: Path) -> None:
    if os.name != "nt":
        os.replace(source, target)
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file_ex = kernel32.MoveFileExW
    move_file_ex.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
    move_file_ex.restype = wintypes.BOOL
    if not move_file_ex(str(source), str(target), 0x00000001 | 0x00000008):
        error = ctypes.get_last_error()
        raise OSError(error, f"MoveFileExW write-through replace failed: {source} -> {target}")


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
