# ruff: noqa: EM101, EM102, TRY003
"""Opaque, restart-safe cursor support for QE log SSE streams.

Domain-specific 410 messages intentionally stay at their validation sites so
each fail-closed cursor condition remains attributable and testable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

CURSOR_REASON_CODE = "qe_log_cursor_expired"
CURSOR_CONFLICT_REASON_CODE = "qe_log_cursor_conflict"
_CURSOR_VERSION = 1
_CURSOR_CONTEXT = b"rdagent-qe-log-cursor-v1\0"
_MAX_CURSOR_BYTES = 16 * 1024
_DEFAULT_INITIAL_TAIL_LINES = 200
_DEFAULT_TAIL_BYTES_PER_FILE = 64 * 1024
_EXPECTED_LOG_PATH_PARTS = 2
_MAX_APPEND_BYTES_PER_FILE = 1024 * 1024


class QELogCursorError(ValueError):
    def __init__(self, message: str, *, reason_code: str = CURSOR_REASON_CODE) -> None:
        super().__init__(message)
        self.reason_code = reason_code

    @classmethod
    def expired(cls, message: str) -> QELogCursorError:
        return cls(message)

    @classmethod
    def conflict(cls, message: str) -> QELogCursorError:
        return cls(message, reason_code=CURSOR_CONFLICT_REASON_CODE)


@dataclass(frozen=True)
class QELogCursorState:
    stream_uuid: str
    seq: int
    offsets: dict[str, int]
    identities: dict[str, str] = field(default_factory=dict)
    terminal: bool = False


def stream_uuid_for_task(task_id: str, task_dir: Path) -> str:
    identity = f"rdagent-qe-log:{task_id}:{task_dir.resolve().as_posix()}"
    return uuid.uuid5(uuid.NAMESPACE_URL, identity).hex


def _validated_relative_log_path(value: str) -> str:
    normalized = str(value or "").replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != _EXPECTED_LOG_PATH_PARTS
        or path.name != "run.log"
    ):
        raise QELogCursorError.expired(f"invalid QE log cursor path: {value!r}")
    return path.as_posix()


def encode_cursor(state: QELogCursorState) -> str:
    offsets = {
        _validated_relative_log_path(path): int(offset)
        for path, offset in sorted(state.offsets.items())
    }
    if state.seq < 0 or any(offset < 0 for offset in offsets.values()):
        raise QELogCursorError.expired("QE log cursor sequence and offsets must be non-negative")
    body = {
        "v": _CURSOR_VERSION,
        "stream_uuid": str(state.stream_uuid),
        "seq": int(state.seq),
        "offsets": offsets,
        "identities": {
            _validated_relative_log_path(path): str(identity)
            for path, identity in sorted(state.identities.items())
        },
        "terminal": bool(state.terminal),
    }
    raw_body = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    envelope = {
        "body": body,
        "sha256_16": hashlib.sha256(_CURSOR_CONTEXT + raw_body).hexdigest()[:16],
    }
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > _MAX_CURSOR_BYTES:
        raise QELogCursorError.expired("QE log cursor exceeds the bounded token size")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_cursor(token: str) -> QELogCursorState:
    value = str(token or "").strip()
    if not value or len(value) > _MAX_CURSOR_BYTES * 2:
        raise QELogCursorError.expired("QE log cursor is empty or oversized")
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        envelope = json.loads(raw.decode("utf-8"))
        body = envelope["body"]
        expected = str(envelope["sha256_16"])
        raw_body = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        actual = hashlib.sha256(_CURSOR_CONTEXT + raw_body).hexdigest()[:16]
        if actual != expected:
            raise QELogCursorError.expired("QE log cursor checksum mismatch")
        if int(body["v"]) != _CURSOR_VERSION:
            raise QELogCursorError.expired("QE log cursor version is unsupported")
        offsets_raw = body.get("offsets")
        if not isinstance(offsets_raw, dict):
            raise QELogCursorError.expired("QE log cursor offsets are invalid")
        offsets = {
            _validated_relative_log_path(path): int(offset)
            for path, offset in offsets_raw.items()
        }
        identities_raw = body.get("identities") or {}
        if not isinstance(identities_raw, dict):
            raise QELogCursorError.expired("QE log cursor identities are invalid")
        identities = {
            _validated_relative_log_path(path): str(identity)
            for path, identity in identities_raw.items()
        }
        state = QELogCursorState(
            stream_uuid=str(body["stream_uuid"]),
            seq=int(body["seq"]),
            offsets=offsets,
            identities=identities,
            terminal=bool(body.get("terminal", False)),
        )
    except QELogCursorError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise QELogCursorError.expired("QE log cursor is malformed") from exc
    if state.seq < 0 or any(offset < 0 for offset in state.offsets.values()):
        raise QELogCursorError.expired("QE log cursor contains a negative sequence or offset")
    return state


def resolve_resume_cursor(after_cursor: str | None, last_event_id: str | None) -> str | None:
    query_cursor = str(after_cursor or "").strip() or None
    header_cursor = str(last_event_id or "").strip() or None
    if query_cursor and header_cursor and query_cursor != header_cursor:
        raise QELogCursorError.conflict("after_cursor and Last-Event-ID disagree")
    return query_cursor or header_cursor


def list_run_logs(task_dir: Path) -> list[tuple[str, Path]]:
    if not task_dir.exists():
        return []
    files: list[tuple[str, Path]] = []
    for loop_dir in sorted((path for path in task_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
        log_file = loop_dir / "run.log"
        if log_file.is_file():
            files.append((f"{loop_dir.name}/run.log", log_file))
    return files


def _file_identity(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_dev}:{stat.st_ino}"


def validate_cursor_state(
    state: QELogCursorState,
    *,
    task_id: str,
    task_dir: Path,
    files: Iterable[tuple[str, Path]] | None = None,
) -> None:
    expected_stream_uuid = stream_uuid_for_task(task_id, task_dir)
    if state.stream_uuid != expected_stream_uuid:
        raise QELogCursorError.expired("QE log cursor belongs to another task stream")
    file_map = dict(files if files is not None else list_run_logs(task_dir))
    for relative_path, offset in state.offsets.items():
        path = file_map.get(relative_path)
        if path is None:
            if offset:
                raise QELogCursorError.expired(f"QE log cursor source disappeared: {relative_path}")
            continue
        expected_identity = state.identities.get(relative_path)
        if expected_identity and _file_identity(path) != expected_identity:
            raise QELogCursorError.expired(f"QE log cursor source was replaced: {relative_path}")
        if path.stat().st_size < offset:
            raise QELogCursorError.expired(f"QE log cursor source was truncated: {relative_path}")


def _read_tail(path: Path, *, max_lines: int, max_bytes: int) -> list[str]:
    size = path.stat().st_size
    if size <= 0 or max_lines <= 0:
        return []
    with path.open("rb") as handle:
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]
    return lines[-max_lines:]


def initial_cursor_state(
    *,
    task_id: str,
    task_dir: Path,
    max_lines: int = _DEFAULT_INITIAL_TAIL_LINES,
    max_bytes_per_file: int = _DEFAULT_TAIL_BYTES_PER_FILE,
) -> tuple[QELogCursorState, list[str]]:
    files = list_run_logs(task_dir)
    offsets: dict[str, int] = {}
    identities: dict[str, str] = {}
    lines: list[str] = []
    for relative_path, path in files:
        offsets[relative_path] = path.stat().st_size
        identities[relative_path] = _file_identity(path)
        loop_name = PurePosixPath(relative_path).parts[0]
        lines.extend(
            f"[{loop_name}] {line}"
            for line in _read_tail(path, max_lines=max_lines, max_bytes=max_bytes_per_file)
        )
    bounded_lines = lines[-max(0, max_lines):]
    return (
        QELogCursorState(
            stream_uuid=stream_uuid_for_task(task_id, task_dir),
            seq=1 if bounded_lines else 0,
            offsets=offsets,
            identities=identities,
            terminal=False,
        ),
        bounded_lines,
    )


def consume_new_lines(
    state: QELogCursorState,
    *,
    task_id: str,
    task_dir: Path,
) -> tuple[QELogCursorState, list[str]]:
    files = list_run_logs(task_dir)
    validate_cursor_state(state, task_id=task_id, task_dir=task_dir, files=files)
    offsets = dict(state.offsets)
    identities = dict(state.identities)
    lines: list[str] = []
    for relative_path, path in files:
        offset = offsets.get(relative_path, 0)
        identities.setdefault(relative_path, _file_identity(path))
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(_MAX_APPEND_BYTES_PER_FILE)
            offsets[relative_path] = handle.tell()
        if not data:
            continue
        loop_name = PurePosixPath(relative_path).parts[0]
        lines.extend(
            f"[{loop_name}] {line}"
            for line in data.decode("utf-8", errors="replace").splitlines()
        )
    return (
        QELogCursorState(
            stream_uuid=state.stream_uuid,
            seq=state.seq + (1 if lines else 0),
            offsets=offsets,
            identities=identities,
            terminal=state.terminal,
        ),
        lines,
    )
