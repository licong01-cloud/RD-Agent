"""Durable, cross-process submission receipts for QE Workspace loops."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on Windows.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - unavailable on POSIX.
    msvcrt = None

SCHEMA_VERSION = "qe_submission_receipt_v1"
RECEIPT_DIR_NAME = ".submission_receipts"
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOOP_ID_RE = re.compile(r"^Loop[1-9][0-9]*$")
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_ALLOWED_TRANSITIONS = {
    "reserved": frozenset({"reserved", "started", "running", "failed", "cancelled"}),
    "started": frozenset({"started", "running", "completed", "failed", "cancelled"}),
    "running": frozenset({"running", "completed", "failed", "cancelled"}),
    "completed": frozenset({"completed"}),
    "failed": frozenset({"failed"}),
    "cancelled": frozenset({"cancelled"}),
}
_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


class SubmissionReceiptError(RuntimeError):
    """Base error for receipt persistence or validation failures."""


class SubmissionReceiptConflictError(SubmissionReceiptError):
    """The same remote identity was reused with a different payload identity."""


class SubmissionReceiptValidationError(SubmissionReceiptError):
    """A caller supplied an invalid receipt identity or request field."""


class SubmissionReceiptTransitionError(SubmissionReceiptError):
    """A receipt state transition violates the durable state machine."""


def canonical_request_digest(
    *,
    loop_index: int,
    config: Mapping[str, Any],
    experiment_files: Mapping[str, str] | None,
    wsl_command: str | None,
    model_source: Mapping[str, Any] | None,
) -> str:
    """Hash execution-affecting request fields while excluding callback transport."""

    file_hashes = {
        str(path): hashlib.sha256(str(content).encode("utf-8")).hexdigest()
        for path, content in sorted((experiment_files or {}).items())
    }
    payload = {
        "loop_index": int(loop_index),
        "config": dict(config),
        "experiment_files_sha256": file_hashes,
        "wsl_command": wsl_command or "",
        "model_source": dict(model_source or {}),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_submission_intent_hash(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HEX_SHA256_RE.fullmatch(normalized):
        message = "submission_intent_hash must be a lowercase SHA-256 hex digest"
        raise SubmissionReceiptValidationError(message)
    return normalized


def reserve_submission(
    loop_dir: Path,
    *,
    task_id: str,
    loop_id: str,
    submission_intent_hash: str,
    request_digest: str,
) -> tuple[dict[str, Any], bool]:
    """Atomically reserve one task/loop identity before background execution."""

    intent_hash = validate_submission_intent_hash(submission_intent_hash)
    digest = validate_submission_intent_hash(request_digest)
    with _receipt_lock(loop_dir, loop_id):
        existing = _read_receipt_unlocked(
            loop_dir,
            loop_id,
            submission_intent_hash=intent_hash,
        )
        if existing is not None:
            expected = {
                "task_id": str(task_id),
                "loop_id": str(loop_id),
                "submission_intent_hash": intent_hash,
                "request_digest": digest,
            }
            actual = {key: existing.get(key) for key in expected}
            if actual != expected:
                message = (
                    "QE Workspace submission identity already exists with different "
                    "intent or request digest"
                )
                raise SubmissionReceiptConflictError(message)
            return existing, False

        latest = _read_latest_receipt_unlocked(loop_dir, loop_id)
        if latest is not None and str(latest.get("status") or "") not in _TERMINAL_STATUSES:
            message = (
                "QE Workspace loop already has a non-terminal submission attempt "
                "with a different intent"
            )
            raise SubmissionReceiptConflictError(message)

        now = _utc_now_iso()
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "task_id": str(task_id),
            "loop_id": str(loop_id),
            "submission_intent_hash": intent_hash,
            "request_digest": digest,
            "status": "reserved",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "running_at": None,
            "finished_at": None,
            "pid": None,
        }
        _atomic_write_receipt_unlocked(
            loop_dir,
            loop_id,
            intent_hash,
            receipt,
        )
        return receipt, True


def get_submission_receipt(
    loop_dir: Path,
    *,
    loop_id: str,
    submission_intent_hash: str | None = None,
) -> dict[str, Any] | None:
    with _receipt_lock(loop_dir, loop_id):
        if submission_intent_hash is None:
            return _read_latest_receipt_unlocked(loop_dir, loop_id)
        return _read_receipt_unlocked(
            loop_dir,
            loop_id,
            submission_intent_hash=validate_submission_intent_hash(
                submission_intent_hash,
            ),
        )


def transition_submission_receipt(
    loop_dir: Path,
    *,
    loop_id: str,
    submission_intent_hash: str,
    status: str,
    pid: int | None = None,
) -> dict[str, Any]:
    """Apply one validated receipt transition under the same cross-process lock."""

    intent_hash = validate_submission_intent_hash(submission_intent_hash)
    next_status = str(status or "").strip().lower()
    if next_status not in _ALLOWED_TRANSITIONS:
        message = f"unsupported submission receipt status: {next_status!r}"
        raise SubmissionReceiptTransitionError(message)

    with _receipt_lock(loop_dir, loop_id):
        receipt = _read_receipt_unlocked(
            loop_dir,
            loop_id,
            submission_intent_hash=intent_hash,
        )
        if receipt is None:
            message = (
                f"submission receipt is missing for {loop_id}; execution cannot advance "
                "without reservation"
            )
            raise SubmissionReceiptTransitionError(message)
        if receipt.get("submission_intent_hash") != intent_hash:
            message = "submission receipt intent hash does not match the executing background task"
            raise SubmissionReceiptConflictError(message)
        current_status = str(receipt.get("status") or "")
        if next_status not in _ALLOWED_TRANSITIONS.get(current_status, frozenset()):
            message = (
                "invalid submission receipt transition: "
                f"{current_status!r} -> {next_status!r}"
            )
            raise SubmissionReceiptTransitionError(message)

        now = _utc_now_iso()
        updated = dict(receipt)
        updated["status"] = next_status
        updated["updated_at"] = now
        if next_status == "started" and not updated.get("started_at"):
            updated["started_at"] = now
        if next_status == "running":
            updated["started_at"] = updated.get("started_at") or now
            updated["running_at"] = updated.get("running_at") or now
            if pid is not None:
                if int(pid) <= 0:
                    message = f"invalid execution pid: {pid}"
                    raise SubmissionReceiptTransitionError(message)
                updated["pid"] = int(pid)
        if next_status in _TERMINAL_STATUSES:
            updated["finished_at"] = updated.get("finished_at") or now
        _atomic_write_receipt_unlocked(
            loop_dir,
            loop_id,
            intent_hash,
            updated,
        )
        return updated


def public_receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable API payload without internal lock or filesystem paths."""

    return {
        "schema_version": receipt.get("schema_version"),
        "task_id": receipt.get("task_id"),
        "loop_id": receipt.get("loop_id"),
        "submission_intent_hash": receipt.get("submission_intent_hash"),
        "request_digest": receipt.get("request_digest"),
        "status": receipt.get("status"),
        "created_at": receipt.get("created_at"),
        "updated_at": receipt.get("updated_at"),
        "started_at": receipt.get("started_at"),
        "running_at": receipt.get("running_at"),
        "finished_at": receipt.get("finished_at"),
        "pid": receipt.get("pid"),
    }


def _receipt_root(loop_dir: Path) -> Path:
    return loop_dir.parent / RECEIPT_DIR_NAME


def _receipt_path(
    loop_dir: Path,
    loop_id: str,
    submission_intent_hash: str,
) -> Path:
    _validate_loop_id(loop_id)
    intent_hash = validate_submission_intent_hash(submission_intent_hash)
    return _receipt_root(loop_dir) / f"{loop_id}.{intent_hash}.json"


def _legacy_receipt_path(loop_dir: Path, loop_id: str) -> Path:
    _validate_loop_id(loop_id)
    return _receipt_root(loop_dir) / f"{loop_id}.json"


def _lock_path(loop_dir: Path, loop_id: str) -> Path:
    _validate_loop_id(loop_id)
    return _receipt_root(loop_dir) / f"{loop_id}.lock"


def _validate_loop_id(loop_id: str) -> None:
    if not _LOOP_ID_RE.fullmatch(str(loop_id or "")):
        message = f"invalid QE loop id for submission receipt: {loop_id!r}"
        raise SubmissionReceiptValidationError(message)


@contextmanager
def _receipt_lock(loop_dir: Path, loop_id: str) -> Iterator[None]:
    receipt_root = _receipt_root(loop_dir)
    receipt_root.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(loop_dir, loop_id)
    local_lock = _local_lock_for(lock_path)
    with local_lock, lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            if msvcrt is None:  # pragma: no cover - defensive platform guard.
                message = "QE submission receipts require msvcrt on Windows"
                raise SubmissionReceiptError(message)

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
                os.fsync(lock_file.fileno())
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            if fcntl is None:  # pragma: no cover - defensive platform guard.
                message = "QE submission receipts require fcntl on non-Windows runtimes"
                raise SubmissionReceiptError(message)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_receipt_unlocked(
    loop_dir: Path,
    loop_id: str,
    *,
    submission_intent_hash: str,
) -> dict[str, Any] | None:
    path = _receipt_path(loop_dir, loop_id, submission_intent_hash)
    if not path.exists():
        legacy_path = _legacy_receipt_path(loop_dir, loop_id)
        if not legacy_path.exists():
            return None
        legacy = _read_receipt_file(legacy_path, loop_id=loop_id)
        if legacy.get("submission_intent_hash") != submission_intent_hash:
            return None
        return legacy
    return _read_receipt_file(
        path,
        loop_id=loop_id,
        expected_submission_intent_hash=submission_intent_hash,
    )


def _read_receipt_file(
    path: Path,
    *,
    loop_id: str,
    expected_submission_intent_hash: str | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        message = f"submission receipt is unreadable: {path}: {exc}"
        raise SubmissionReceiptError(message) from exc
    if not isinstance(payload, dict):
        message = f"submission receipt must contain a JSON object: {path}"
        raise SubmissionReceiptError(message)
    required = {
        "schema_version",
        "task_id",
        "loop_id",
        "submission_intent_hash",
        "request_digest",
        "status",
        "created_at",
        "updated_at",
    }
    missing = sorted(required.difference(payload))
    if missing:
        message = f"submission receipt is missing required fields: {missing}"
        raise SubmissionReceiptError(message)
    if payload.get("schema_version") != SCHEMA_VERSION:
        message = f"unsupported submission receipt schema: {payload.get('schema_version')!r}"
        raise SubmissionReceiptError(message)
    if payload.get("loop_id") != loop_id:
        message = (
            "submission receipt loop identity mismatch: "
            f"expected={loop_id!r} actual={payload.get('loop_id')!r}"
        )
        raise SubmissionReceiptError(message)
    actual_intent_hash = validate_submission_intent_hash(
        str(payload.get("submission_intent_hash") or ""),
    )
    if (
        expected_submission_intent_hash is not None
        and actual_intent_hash != expected_submission_intent_hash
    ):
        message = (
            "submission receipt intent identity mismatch: "
            f"expected={expected_submission_intent_hash!r} actual={actual_intent_hash!r}"
        )
        raise SubmissionReceiptError(message)
    validate_submission_intent_hash(str(payload.get("request_digest") or ""))
    if payload.get("status") not in _ALLOWED_TRANSITIONS:
        message = f"submission receipt has invalid status: {payload.get('status')!r}"
        raise SubmissionReceiptError(message)
    return payload


def _read_latest_receipt_unlocked(
    loop_dir: Path,
    loop_id: str,
) -> dict[str, Any] | None:
    _validate_loop_id(loop_id)
    receipt_root = _receipt_root(loop_dir)
    if not receipt_root.exists():
        return None
    candidates: list[dict[str, Any]] = []
    for path in receipt_root.glob(f"{loop_id}.*.json"):
        intent_hash = path.name[len(loop_id) + 1 : -len(".json")]
        if not _HEX_SHA256_RE.fullmatch(intent_hash):
            continue
        receipt = _read_receipt_file(
            path,
            loop_id=loop_id,
            expected_submission_intent_hash=intent_hash,
        )
        candidates.append(receipt)
    legacy_path = _legacy_receipt_path(loop_dir, loop_id)
    if legacy_path.exists():
        candidates.append(_read_receipt_file(legacy_path, loop_id=loop_id))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("updated_at") or ""),
            str(item.get("submission_intent_hash") or ""),
        ),
    )


def _atomic_write_receipt_unlocked(
    loop_dir: Path,
    loop_id: str,
    submission_intent_hash: str,
    receipt: Mapping[str, Any],
) -> None:
    path = _receipt_path(loop_dir, loop_id, submission_intent_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(
        dict(receipt),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # QE Workspace production runs on Linux/WSL. Windows cannot fsync a
        # directory handle through os.open, while os.replace remains atomic.
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _local_lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCAL_LOCKS[key] = lock
        return lock
