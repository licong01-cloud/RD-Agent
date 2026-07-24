"""Detached single-slot FIFO supervisor for persisted QE long-trend jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rdagent.app.api_endpoints.qe_long_trend_evaluation import (
    DISPATCHER_IDENTITY_FILE,
    DISPATCHER_SPAWN_LOCK,
    TERMINAL_STATUSES,
    _atomic_json,
    _exclusive_file_lock,
    _read_json,
    _sha256_file,
    _utc_now,
)

OUTBOX_SCHEMA_VERSION = "qe_long_trend_resource_outbox_v2"
OUTBOX_IDENTITY_FIELDS = (
    "session_id",
    "source_run_key",
    "task_id",
    "loop_id",
    "loop_index",
    "node_id",
    "sequence_no",
    "phase",
)
CONFLICT_BACKOFF_BASE_SECONDS = 300
RETRYABLE_BACKOFF_BASE_SECONDS = 15
CONFLICT_BACKOFF_MAX_SECONDS = 3600
RETRYABLE_BACKOFF_MAX_SECONDS = 300
SHA256_HEX_LENGTH = 64
HTTP_SUCCESS_MIN = 200
HTTP_SUCCESS_MAX_EXCLUSIVE = 300
HTTP_CONFLICT_STATUS = 409
HTTP_SERVER_ERROR_MIN = 500
HTTP_SERVER_ERROR_MAX = 599
OUTBOX_IDENTITY_CONFLICT_REASON = "QELT_RESOURCE_OUTBOX_IDENTITY_CONFLICT"
OUTBOX_IDENTITY_INVALID_REASON = "QELT_RESOURCE_OUTBOX_IDENTITY_INVALID"
CONTROL_STATE_CONFLICT_REASON = "QELT_CONTROL_STATE_CONFLICT"


class QELTResourceOutboxError(RuntimeError):
    """Durable resource-outbox identity or schema conflict."""


def _outbox_error(reason_code: str, message: str) -> QELTResourceOutboxError:
    return QELTResourceOutboxError(f"{reason_code}: {message}")


def main(argv: list[str] | None = None) -> int:
    import fcntl  # noqa: PLC0415 - Linux-only worker entrypoint; Windows imports this module for API/tests.

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.workspace_root).resolve()
    if not root.is_dir():
        return 2
    identity = _capture_process_identity(os.getpid())
    if not _claim_dispatcher_identity(root, identity):
        return 0
    try:
        lock_path = root / ".qe_long_trend_eval_slot.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return _run_dispatcher_loop(root)
    finally:
        _release_dispatcher_identity(root, identity)


def _run_dispatcher_loop(root: Path) -> int:
    while True:
        queued = _queued_jobs(root)
        if not queued:
            _replay_outboxes(root)
            return 0
        try:
            _run_or_monitor_job(queued[0])
        except Exception as exc:  # noqa: BLE001 - supervisor must durably finalize every worker failure.
            _record_dispatcher_failure(root, queued[0], exc)


def _claim_dispatcher_identity(root: Path, identity: Mapping[str, Any]) -> bool:
    identity_path = root / DISPATCHER_IDENTITY_FILE
    with _exclusive_file_lock(root / DISPATCHER_SPAWN_LOCK):
        if identity_path.is_file():
            existing = _read_json(identity_path)
            if existing != dict(identity) and _process_alive(existing):
                return False
        _atomic_json(identity_path, dict(identity))
        return True


def _release_dispatcher_identity(root: Path, identity: Mapping[str, Any]) -> None:
    identity_path = root / DISPATCHER_IDENTITY_FILE
    with _exclusive_file_lock(root / DISPATCHER_SPAWN_LOCK):
        if identity_path.is_file() and _read_json(identity_path) == dict(identity):
            identity_path.unlink()


def _queued_jobs(root: Path) -> list[Path]:
    rows: list[tuple[str, str, Path]] = []
    for path in root.glob("*/Loop*/long_trend_evaluations/qelt_*/job.json"):
        try:
            job = _read_json(path)
        except Exception as exc:  # noqa: BLE001 - a corrupt job must be recorded without blocking other jobs.
            _append_dispatcher_error(
                root,
                {"stage": "scan_job", "job_path": str(path), "error_type": type(exc).__name__, "message": str(exc)},
            )
            continue
        if job.get("status") in {"queued", "starting", "running"}:
            rows.append((str(job.get("created_at") or ""), str(job.get("evaluation_id") or ""), path.parent))
    return [item[2] for item in sorted(rows)]


def _run_or_monitor_job(job_dir: Path) -> None:
    job = _read_json(job_dir / "job.json")
    if job.get("status") in TERMINAL_STATUSES:
        return
    attempt_id = str(job.get("current_attempt_id") or "")
    if attempt_id:
        attempt_dir = job_dir / "attempts" / attempt_id
        process_path = attempt_dir / "process_identity.json"
        if process_path.is_file():
            identity = _read_json(process_path)
            if _process_alive(identity):
                _monitor_existing(job_dir, attempt_dir, identity)
                return
        terminal = attempt_dir / "terminal_receipt.json"
        if terminal.is_file():
            _promote_terminal(job_dir, attempt_dir)
            return

    attempts_root = job_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_no = 1 + len([path for path in attempts_root.iterdir() if path.is_dir()])
    attempt_id = f"qelt_attempt_{str(job['evaluation_id'])[5:21]}_{attempt_no}"
    attempt_dir = attempts_root / attempt_id
    attempt_dir.mkdir()
    claim_path = attempt_dir / "claim.json"
    fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "attempt_id": attempt_id,
                "attempt_no": attempt_no,
                "claimed_at": _utc_now(),
                "supervisor_pid": os.getpid(),
            },
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.flush()
        os.fsync(handle.fileno())
    job.update({"status": "starting", "current_attempt_id": attempt_id, "updated_at": _utc_now()})
    _atomic_json(job_dir / "job.json", job)

    request = _read_json(job_dir / "request.json")
    worker_request = _build_worker_request(job_dir, attempt_dir, job, request)
    request_path = attempt_dir / "worker_request.json"
    _atomic_json(request_path, worker_request)
    artifacts_dir = attempt_dir / "artifacts"
    artifacts_dir.mkdir()
    runtime_dir = job_dir / "runtime"
    env = _worker_environment(runtime_dir)
    stdout = (attempt_dir / "stdout.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed hashed bundle entry.
            [
                sys.executable,
                "-m",
                "backend.services.quantevolver.long_trend_worker_entry",
                "--request",
                str(request_path),
                "--output-dir",
                str(artifacts_dir),
            ],
            cwd=str(attempt_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        identity = _capture_process_identity(process.pid)
        _atomic_json(attempt_dir / "process_identity.json", identity)
        job.update({"status": "running", "updated_at": _utc_now()})
        _atomic_json(job_dir / "job.json", job)
        _queue_resource_event(job_dir, job, attempt_id, sequence_no=1, phase="long_trend_eval", phase_status="running")
        process.wait()
    finally:
        stdout.close()
    _finalize_attempt(job_dir, attempt_dir, returncode=process.returncode)


def _monitor_existing(job_dir: Path, attempt_dir: Path, identity: Mapping[str, Any]) -> None:
    while _process_alive(identity):
        time.sleep(1.0)
    terminal = attempt_dir / "artifacts" / "worker_terminal_receipt.json"
    if terminal.is_file():
        _finalize_attempt(job_dir, attempt_dir, returncode=None)
        return
    _finalize_attempt(job_dir, attempt_dir, returncode=None)


def _finalize_attempt(job_dir: Path, attempt_dir: Path, *, returncode: int | None) -> None:
    worker_terminal = attempt_dir / "artifacts" / "worker_terminal_receipt.json"
    if not worker_terminal.is_file():
        job = _read_json(job_dir / "job.json")
        cancel_path = attempt_dir / "cancel_intent.json"
        cancel = _read_json(cancel_path) if cancel_path.is_file() else {}
        cancelled = cancel.get("status") in {"requested", "signal_sent"}
        status = "cancelled" if cancelled else "failed"
        reason_code = "QELT_CANCELLED" if cancelled else "QELT_NODE_PROCESS_IDENTITY_CONFLICT"
        families = {
            name: {
                "status": "NOT_COMPUTABLE",
                "available_inputs": [],
                "missing_inputs": ["worker_terminal_receipt"],
                "coverage": {},
                "limitations": [f"worker exited returncode={returncode} without terminal receipt"],
                "supporting_artifacts": [],
                "reason_codes": [reason_code],
                "data_actions": [],
            }
            for name in (
                "signal_path",
                "position_episode",
                "portfolio_result",
                "order_fill",
                "execution_cause",
                "sector_regime",
            )
        }
        compact = {
            "schema_version": "qe_long_trend_worker_compact_v1",
            "receipt_stage": "worker_terminal",
            "evaluation_id": job["evaluation_id"],
            "worker_terminal_sha256": {"type": "explicit_null", "field": "worker_terminal_sha256"},
            "family_status": families,
            "headline_metrics": {},
            "data_action_plan": [],
            "platform_delivery_status": {"worker": status, "cas": "awaiting_collect"},
            "artifact_manifest_uri": {"type": "explicit_null", "field": "artifact_manifest_uri"},
            "artifact_manifest_sha256": {"type": "explicit_null", "field": "artifact_manifest_sha256"},
        }
        _atomic_json(attempt_dir / "artifacts" / "worker_compact_receipt.json", compact)
        receipt = {
            "schema_version": "qe_long_trend_worker_terminal_v1",
            "evaluation_id": job["evaluation_id"],
            "job_id": job["job_id"],
            "attempt_id": job["current_attempt_id"],
            "request_sha": job["request_sha"],
            "status": status,
            "reason_code": reason_code,
            "reason_json": {"returncode": returncode, "message": "worker ended without its own terminal receipt"},
            "family_status": families,
            "metrics": [],
            "data_action_plan": [],
            "platform_delivery_status": {"worker": status, "cas": "awaiting_collect"},
            "artifacts": {},
            "stats": {},
        }
        _atomic_json(worker_terminal, receipt)
    receipt = _read_json(worker_terminal)
    _atomic_json(attempt_dir / "terminal_receipt.json", receipt)
    _promote_terminal(job_dir, attempt_dir)


def _promote_terminal(job_dir: Path, attempt_dir: Path) -> None:
    job = _read_json(job_dir / "job.json")
    receipt = _read_json(attempt_dir / "terminal_receipt.json")
    status = str(receipt.get("status") or "failed")
    if status not in TERMINAL_STATUSES:
        status = "failed"
    job.update({"status": status, "updated_at": _utc_now(), "completed_at": _utc_now()})
    _atomic_json(job_dir / "job.json", job)
    phase = "completed" if status in {"succeeded", "partial"} else status
    _queue_resource_event(
        job_dir,
        job,
        str(job["current_attempt_id"]),
        sequence_no=2,
        phase=phase,
        phase_status=status,
        receipt=receipt,
    )


def _build_worker_request(
    job_dir: Path,
    attempt_dir: Path,
    job: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    loop_root = job_dir.parents[1]
    artifact_paths: dict[str, str | None] = {}
    for name, relative in dict(request.get("artifact_paths") or {}).items():
        if relative in (None, ""):
            artifact_paths[name] = None
            continue
        target = (loop_root / str(relative)).resolve()
        target.relative_to(loop_root.resolve())
        expected_hash = dict(request.get("artifact_hashes") or {}).get(name)
        if not isinstance(expected_hash, str) or len(expected_hash) != SHA256_HEX_LENGTH:
            message = f"QELT_ARTIFACT_HASH_MISMATCH: missing frozen hash for {name}"
            raise RuntimeError(message)
        actual_hash, _size = _sha256_file(target)
        if actual_hash != expected_hash:
            message = f"QELT_ARTIFACT_HASH_MISMATCH: {name}: expected={expected_hash} actual={actual_hash}"
            raise RuntimeError(message)
        artifact_paths[name] = str(target)
    feature_root = _resolve_registered_dataset_root(request["feature_data_root_uri"])
    outcome_root = _resolve_registered_dataset_root(request["outcome_data_root_uri"])
    feature_workspace = _bind_dataset_workspace(attempt_dir / "datasets" / "feature", feature_root)
    outcome_workspace = _bind_dataset_workspace(attempt_dir / "datasets" / "outcome", outcome_root)
    return {
        "schema_version": "qe_long_trend_worker_request_v1",
        **{
            key: request.get(key)
            for key in (
                "evaluation_id",
                "run_id",
                "node_id",
                "profile_id",
                "profile_sha256",
                "evaluator_version",
                "evaluator_source_sha256",
                "bundle_sha256",
                "qe_dataset_contract_id",
                "feature_snapshot",
                "outcome_snapshot",
                "input_manifest_sha256",
                "input_artifact_hashes",
                "evaluation_asof",
                "label_horizon",
                "strategy_topk",
                "parser_timeout_seconds",
                "execution_environment_snapshot_id",
                "execution_environment_manifest_sha256",
            )
        },
        "job_id": job["job_id"],
        "attempt_id": job["current_attempt_id"],
        "request_sha": job["request_sha"],
        "loop_root": str(loop_root),
        "feature_workspace_root": str(feature_workspace),
        "outcome_workspace_root": str(outcome_workspace),
        "feature_data_root": str(feature_root),
        "outcome_data_root": str(outcome_root),
        "artifact_paths": artifact_paths,
    }


def _resolve_registered_dataset_root(uri: str) -> Path:
    raw = str(uri or "").strip()
    raw = raw.removeprefix("file://")
    path = Path(raw).expanduser().resolve()
    configured: list[str] = []
    for key in ("QE_QLIB_DATA_PATH", "RDAGENT_FACTOR_DATA_WSL"):
        value = str(os.environ.get(key) or "").strip()
        if value:
            configured.append(value)
    for key in ("QE_DATASET_IDENTITY_ROOTS", "QE_REGISTERED_DATASET_ROOTS"):
        configured.extend(value.strip() for value in str(os.environ.get(key) or "").split(os.pathsep) if value.strip())
    allowed_roots = [Path(value).expanduser().resolve() for value in configured]
    if not allowed_roots or not any(path == root or root in path.parents for root in allowed_roots):
        message = f"QELT_EXECUTION_ENVIRONMENT_MISMATCH: dataset root is not registered: {path}"
        raise RuntimeError(message)
    return path


def _bind_dataset_workspace(workspace: Path, root: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=False)
    for name in ("meta.json", "daily_pv.h5", "sector_data.h5"):
        source = (root / name).resolve(strict=True)
        try:
            source.relative_to(root.resolve())
        except ValueError as exc:
            message = f"QELT_EXECUTION_ENVIRONMENT_MISMATCH: dataset file escapes registered root: {source}"
            raise RuntimeError(message) from exc
        if not source.is_file():
            message = f"QELT_EXECUTION_ENVIRONMENT_MISMATCH: dataset file is not regular: {source}"
            raise RuntimeError(message)
        target = workspace / name
        target.symlink_to(source)
    return workspace


def _worker_environment(runtime_dir: Path) -> dict[str, str]:
    blocked = ("PG", "DATABASE_", "POSTGRES_", "QE_RESOURCE_", "AISTOCK_PREDICTION_STORE_")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith(blocked)}
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(runtime_dir) + (os.pathsep + prior if prior else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _capture_process_identity(pid: int) -> dict[str, Any]:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ")
    return {"pid": pid, "start_ticks": int(stat[21]), "command_sha256": hashlib.sha256(command).hexdigest()}


def _process_alive(identity: Mapping[str, Any]) -> bool:
    try:
        pid = int(identity["pid"])
        os.kill(pid, 0)
        if Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2] == "Z":
            return False
        return _capture_process_identity(pid) == dict(identity)
    except (KeyError, ValueError, OSError, ProcessLookupError):
        return False


def _queue_resource_event(
    job_dir: Path,
    job: Mapping[str, Any],
    attempt_id: str,
    *,
    sequence_no: int,
    phase: str,
    phase_status: str,
    receipt: Mapping[str, Any] | None = None,
) -> None:
    request = _read_json(job_dir / "request.json")
    session = request.get("resource_session")
    if not isinstance(session, Mapping):
        raise _outbox_error(CONTROL_STATE_CONFLICT_REASON, "worker request has no resource_session identity")
    stats = receipt.get("stats") if isinstance(receipt, Mapping) and isinstance(receipt.get("stats"), Mapping) else {}
    payload = {
        "session_id": session.get("session_id"),
        "source_run_key": session.get("source_run_key"),
        "task_id": job["task_id"],
        "loop_id": job["loop_id"],
        "loop_index": int(str(job["loop_id"]).removeprefix("Loop")),
        "node_id": request["node_id"],
        "sequence_no": sequence_no,
        "phase": phase,
        "phase_status": phase_status,
        "duration_seconds": stats.get("duration_seconds"),
        "sample_count": stats.get("signal_rows"),
        "process_rss_peak_bytes": stats.get("process_rss_peak_bytes"),
        "reason_code": receipt.get("reason_code") if isinstance(receipt, Mapping) else None,
        "metadata": {
            "evaluation_id": job["evaluation_id"],
            "job_id": job["job_id"],
            "attempt_id": attempt_id,
            "output_rows": stats.get("signal_rows"),
            "artifact_bytes": _artifact_bytes(job_dir / "attempts" / attempt_id),
        },
    }
    outbox_dir = job_dir / "outbox"
    outbox_dir.mkdir(exist_ok=True)
    outbox = outbox_dir / f"{sequence_no:06d}.json"
    candidate = _new_outbox_row(payload)
    with _exclusive_file_lock(outbox.with_suffix(".lock")):
        if outbox.is_file():
            existing = _normalize_outbox_row(_read_json(outbox), outbox)
            if existing["event_sha256"] != candidate["event_sha256"]:
                raise _outbox_error(
                    OUTBOX_IDENTITY_CONFLICT_REASON,
                    f"sequence {sequence_no} already has a different durable event hash",
                )
            if existing["payload_identity"] != candidate["payload_identity"]:
                raise _outbox_error(
                    OUTBOX_IDENTITY_CONFLICT_REASON,
                    f"sequence {sequence_no} already has a different durable payload identity",
                )
            _atomic_json(outbox, existing)
        else:
            _atomic_json(outbox, candidate)
    _deliver_outbox(job_dir, outbox)


def _canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in OUTBOX_IDENTITY_FIELDS if payload.get(field) in (None, "")]
    if missing:
        raise _outbox_error(OUTBOX_IDENTITY_INVALID_REASON, f"missing fields {missing}")
    identity = {field: payload[field] for field in OUTBOX_IDENTITY_FIELDS}
    identity["loop_index"] = int(identity["loop_index"])
    identity["sequence_no"] = int(identity["sequence_no"])
    return identity


def _new_outbox_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    materialized = dict(payload)
    return {
        "schema_version": OUTBOX_SCHEMA_VERSION,
        "payload": materialized,
        "payload_identity": _payload_identity(materialized),
        "sequence_no": int(materialized["sequence_no"]),
        "event_sha256": hashlib.sha256(_canonical_payload_bytes(materialized)).hexdigest(),
        "delivery_state": "pending",
        "delivered": False,
        "delivery_attempt_count": 0,
        "next_attempt_at": None,
        "last_delivery_error": None,
    }


def _normalize_outbox_row(row: Mapping[str, Any], path: Path) -> dict[str, Any]:
    normalized = dict(row)
    payload = normalized.get("payload")
    if not isinstance(payload, Mapping):
        raise _outbox_error(OUTBOX_IDENTITY_INVALID_REASON, f"{path} has no payload object")
    identity = _payload_identity(payload)
    try:
        path_sequence = int(path.stem)
    except ValueError as exc:
        raise _outbox_error(OUTBOX_IDENTITY_INVALID_REASON, f"non-numeric outbox path {path}") from exc
    if path_sequence != identity["sequence_no"]:
        raise _outbox_error(
            OUTBOX_IDENTITY_CONFLICT_REASON,
            f"path sequence {path_sequence} != payload sequence {identity['sequence_no']}",
        )
    event_sha256 = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
    stored_hash = normalized.get("event_sha256")
    if stored_hash not in (None, event_sha256):
        raise _outbox_error(
            OUTBOX_IDENTITY_CONFLICT_REASON,
            "stored event hash does not match payload",
        )
    stored_identity = normalized.get("payload_identity")
    if stored_identity is not None and dict(stored_identity) != identity:
        raise _outbox_error(
            OUTBOX_IDENTITY_CONFLICT_REASON,
            "stored payload identity does not match payload",
        )
    normalized.update(
        {
            "schema_version": OUTBOX_SCHEMA_VERSION,
            "payload": dict(payload),
            "payload_identity": identity,
            "sequence_no": identity["sequence_no"],
            "event_sha256": event_sha256,
            "delivery_state": normalized.get("delivery_state")
            or ("delivered" if normalized.get("delivered") is True else "pending"),
            "delivered": normalized.get("delivered") is True,
            "delivery_attempt_count": int(normalized.get("delivery_attempt_count") or 0),
            "next_attempt_at": normalized.get("next_attempt_at"),
            "last_delivery_error": normalized.get("last_delivery_error"),
        },
    )
    return normalized


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _next_attempt_at(attempt_count: int, *, conflict: bool) -> str:
    base = CONFLICT_BACKOFF_BASE_SECONDS if conflict else RETRYABLE_BACKOFF_BASE_SECONDS
    maximum = CONFLICT_BACKOFF_MAX_SECONDS if conflict else RETRYABLE_BACKOFF_MAX_SECONDS
    delay = min(maximum, base * (2 ** min(max(attempt_count - 1, 0), 10)))
    return (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat().replace("+00:00", "Z")


def _response_detail(raw_body: bytes) -> tuple[str | None, str | None, Any]:
    if not raw_body:
        return None, None, None
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, raw_body.decode("utf-8", errors="replace")[:4096], None
    detail = decoded.get("detail") if isinstance(decoded, Mapping) else decoded
    if isinstance(detail, Mapping):
        reason_code = detail.get("reason_code")
        message = detail.get("message") or detail.get("detail")
        return str(reason_code) if reason_code else None, str(message) if message else None, dict(detail)
    return None, str(detail) if detail is not None else None, detail


def _record_delivery_failure(
    path: Path,
    row: dict[str, Any],
    *,
    delivery_state: str,
    reason_code: str,
    error_type: str,
    conflict: bool,
    http_status: int | None = None,
    aistock_reason_code: str | None = None,
    message: str | None = None,
    detail: Any = None,
) -> None:
    attempt_count = int(row.get("delivery_attempt_count") or 0) + 1
    attempted_at = _utc_now()
    row.update(
        {
            "delivered": False,
            "delivery_state": delivery_state,
            "delivery_attempt_count": attempt_count,
            "last_delivery_attempt_at": attempted_at,
            "next_attempt_at": _next_attempt_at(attempt_count, conflict=conflict),
            "last_delivery_error": {
                "reason_code": reason_code,
                "error_type": error_type,
                "http_status": http_status,
                "aistock_reason_code": aistock_reason_code,
                "message": message,
                "detail": detail,
                "payload_identity": row["payload_identity"],
                "sequence_no": row["sequence_no"],
                "event_sha256": row["event_sha256"],
            },
        },
    )
    _atomic_json(path, row)


def _http_failure_fields(
    status: int,
    *,
    aistock_reason: str | None,
    message: str | None,
    detail: Any,
) -> dict[str, Any]:
    conflict = status == HTTP_CONFLICT_STATUS
    server_error = HTTP_SERVER_ERROR_MIN <= status <= HTTP_SERVER_ERROR_MAX
    if conflict:
        delivery_state = "conflict_reconciliation_required"
        reason_code = "QELT_RESOURCE_CALLBACK_HTTP_CONFLICT"
        error_type = "http_conflict"
    elif server_error:
        delivery_state = "retryable_http"
        reason_code = "QELT_RESOURCE_CALLBACK_HTTP_5XX"
        error_type = "http_5xx"
    else:
        delivery_state = "http_rejected"
        reason_code = "QELT_RESOURCE_CALLBACK_HTTP_REJECTED"
        error_type = "http_rejected"
    return {
        "delivery_state": delivery_state,
        "reason_code": reason_code,
        "error_type": error_type,
        "conflict": conflict,
        "http_status": status,
        "aistock_reason_code": aistock_reason,
        "message": message,
        "detail": detail,
    }


def _deliver_outbox(job_dir: Path, path: Path) -> bool:
    with _exclusive_file_lock(path.with_suffix(".lock")):
        row = _normalize_outbox_row(_read_json(path), path)
        if row["delivered"] is True:
            return True
        next_attempt_at = _parse_utc(row.get("next_attempt_at"))
        if next_attempt_at is not None and next_attempt_at > datetime.now(timezone.utc):
            _atomic_json(path, row)
            return False
        secret = _read_json(job_dir / "secret.json")
        body = _canonical_payload_bytes(row["payload"])
        request = urllib.request.Request(  # noqa: S310 - callback URL is frozen by the signed request.
            str(secret["resource_callback_url"]),
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-QE-Resource-Token": str(secret["resource_session_token"]),
            },
            method="POST",
        )
        failure: dict[str, Any] | None = None
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                if response.status < HTTP_SUCCESS_MIN or response.status >= HTTP_SUCCESS_MAX_EXCLUSIVE:
                    raw_body = response.read()
                    aistock_reason, message, detail = _response_detail(raw_body)
                    failure = _http_failure_fields(
                        int(response.status),
                        aistock_reason=aistock_reason,
                        message=message,
                        detail=detail,
                    )
        except urllib.error.HTTPError as exc:
            raw_body = exc.read()
            aistock_reason, message, detail = _response_detail(raw_body)
            failure = _http_failure_fields(
                int(exc.code),
                aistock_reason=aistock_reason,
                message=message or str(exc),
                detail=detail,
            )
        except TimeoutError as exc:
            failure = {
                "delivery_state": "retryable_timeout",
                "reason_code": "QELT_RESOURCE_CALLBACK_TIMEOUT",
                "error_type": "timeout",
                "conflict": False,
                "message": str(exc),
            }
        except urllib.error.URLError as exc:
            is_timeout = isinstance(exc.reason, (TimeoutError, socket.timeout))
            failure = {
                "delivery_state": "retryable_timeout" if is_timeout else "retryable_network",
                "reason_code": (
                    "QELT_RESOURCE_CALLBACK_TIMEOUT"
                    if is_timeout
                    else "QELT_RESOURCE_CALLBACK_NETWORK_FAILED"
                ),
                "error_type": "timeout" if is_timeout else "network",
                "conflict": False,
                "message": str(exc),
            }
        except OSError as exc:
            failure = {
                "delivery_state": "retryable_network",
                "reason_code": "QELT_RESOURCE_CALLBACK_NETWORK_FAILED",
                "error_type": "os_error",
                "conflict": False,
                "message": str(exc),
            }
        if failure is not None:
            _record_delivery_failure(path, row, **failure)
            return False
        delivered_at = _utc_now()
        row.update(
            {
                "delivered": True,
                "delivery_state": "delivered",
                "delivered_at": delivered_at,
                "delivery_attempt_count": int(row.get("delivery_attempt_count") or 0) + 1,
                "last_delivery_attempt_at": delivered_at,
                "next_attempt_at": None,
                "last_delivery_error": None,
            },
        )
        _atomic_json(path, row)
        return True


def _replay_outboxes(root: Path) -> int:
    pending = 0
    for path in sorted(root.glob("*/Loop*/long_trend_evaluations/qelt_*/outbox/*.json")):
        try:
            if not _deliver_outbox(path.parents[1], path):
                pending += 1
        except Exception as exc:  # noqa: BLE001, PERF203 - persist corruption and continue other durable outboxes.
            pending += 1
            _append_dispatcher_error(
                root,
                {
                    "stage": "replay_outbox",
                    "outbox_path": str(path),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            continue
    return pending


def _record_dispatcher_failure(root: Path, job_dir: Path, exc: Exception) -> None:
    _append_dispatcher_error(
        root,
        {"stage": "run_job", "job_dir": str(job_dir), "error_type": type(exc).__name__, "message": str(exc)},
    )
    try:
        job = _read_json(job_dir / "job.json")
        if job.get("status") in TERMINAL_STATUSES:
            return
        attempt_id = str(job.get("current_attempt_id") or "")
        if attempt_id:
            attempt_dir = job_dir / "attempts" / attempt_id
            process_path = attempt_dir / "process_identity.json"
            if process_path.is_file() and _process_alive(_read_json(process_path)):
                return
        else:
            attempt_id = f"qelt_attempt_{str(job['evaluation_id'])[5:21]}_supervisor"
            attempt_dir = job_dir / "attempts" / attempt_id
        (attempt_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        job.update({"status": "starting", "current_attempt_id": attempt_id, "updated_at": _utc_now()})
        _atomic_json(job_dir / "job.json", job)
        _finalize_attempt(job_dir, attempt_dir, returncode=-1)
    except Exception as persist_exc:  # noqa: BLE001 - last-resort durable supervisor error receipt.
        _append_dispatcher_error(
            root,
            {
                "stage": "persist_dispatcher_failure",
                "job_dir": str(job_dir),
                "error_type": type(persist_exc).__name__,
                "message": str(persist_exc),
            },
        )


def _append_dispatcher_error(root: Path, payload: Mapping[str, Any]) -> None:
    path = root / ".qe_long_trend_dispatcher_errors.jsonl"
    row = {"schema_version": "qe_long_trend_dispatcher_error_v1", "created_at": _utc_now(), **dict(payload)}
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_bytes(attempt_dir: Path) -> int:
    total = 0
    artifacts = attempt_dir / "artifacts"
    if artifacts.is_dir():
        total = sum(path.stat().st_size for path in artifacts.iterdir() if path.is_file())
    return total


if __name__ == "__main__":
    raise SystemExit(main())
