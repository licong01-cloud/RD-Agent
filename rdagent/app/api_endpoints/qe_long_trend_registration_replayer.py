"""Durable recovery daemon for normal-Loop F-014 registrations pending before control-row creation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

PENDING_INDEX_SCHEMA = "qe_long_trend_registration_pending_index_v1"
PENDING_DIR = ".qe_long_trend_registration_pending"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    if not 1.0 <= float(args.poll_interval_seconds) <= 300.0:
        parser.error("--poll-interval-seconds must be between 1 and 300")
    workspace = Path(args.workspace_root).resolve(strict=True)
    _run_replay_loop(workspace, poll_interval_seconds=float(args.poll_interval_seconds))
    return 0


def _run_replay_loop(workspace: Path, *, poll_interval_seconds: float) -> None:
    while True:
        _replay_cycle(workspace)
        time.sleep(poll_interval_seconds)


async def run_replay_loop(workspace: Path, *, poll_interval_seconds: float = 15.0) -> None:
    """Continuously replay pending registrations for the owning API lifespan."""

    if not 1.0 <= float(poll_interval_seconds) <= 300.0:
        raise ValueError("poll_interval_seconds must be between 1 and 300")
    root = Path(workspace).resolve(strict=True)
    while True:
        await asyncio.to_thread(_replay_cycle, root)
        await asyncio.sleep(float(poll_interval_seconds))


def _replay_cycle(workspace: Path) -> bool:
    """Run one cross-process-singleton replay scan.

    The lock is held only for the scan, so no interpreter can retain ownership
    across an API deployment.  Multiple API workers may schedule the loop, but
    only one process performs a given scan.
    """

    import fcntl

    lock_path = workspace / ".qe_long_trend_registration_replay.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            _run_recovery_scan(workspace)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return True


def _run_recovery_scan(workspace: Path) -> None:
    try:
        _replay_once(workspace)
    except Exception as exc:
        _append_error(
            workspace,
            {
                "stage": "scan_pending_registrations",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
    # Resource events use the same API-lifecycle recovery heartbeat.  The
    # worker still performs an immediate delivery and one final replay, while
    # this scan guarantees eventual delivery after AIstock becomes reachable
    # without keeping a detached old-code dispatcher alive forever.
    from rdagent.app.api_endpoints.qe_long_trend_worker import _replay_outboxes

    _replay_outboxes(workspace)


def _replay_once(workspace: Path) -> None:
    pending_root = workspace / PENDING_DIR
    if not pending_root.is_dir():
        return
    for index_path in sorted(pending_root.glob("*.json"), key=lambda path: path.name):
        try:
            _replay_one(workspace, index_path)
        except Exception as exc:
            _append_error(
                workspace,
                {
                    "index_path": index_path.name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )


def _replay_one(workspace: Path, index_path: Path) -> None:
    if index_path.is_symlink() or not index_path.is_file():
        raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: index is not a regular file")
    payload = _read_json(index_path)
    if payload.get("schema_version") != PENDING_INDEX_SCHEMA:
        raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: schema mismatch")
    task_id = _safe_component(payload.get("task_id"), "task_id")
    loop_id = _safe_component(payload.get("loop_id"), "loop_id")
    if not loop_id.startswith("Loop") or not loop_id[4:].isdigit() or int(loop_id[4:]) < 1:
        raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: loop_id is invalid")
    expected_name = f"{task_id}__{loop_id}.json"
    if index_path.name != expected_name:
        raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: index filename identity mismatch")
    relative = PurePosixPath(str(payload.get("loop_relative_path") or ""))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: loop path is unsafe")
    loop_root = (workspace / Path(*relative.parts)).resolve(strict=True)
    loop_root.relative_to(workspace)
    adapter = (loop_root / "long_trend_postprocess_adapter.py").resolve(strict=True)
    descriptor = (loop_root / "qe_long_trend_postprocess_descriptor.json").resolve(strict=True)
    for path in (adapter, descriptor):
        path.relative_to(loop_root)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: replay input is not a regular file")
    if _sha256_file(descriptor) != str(payload.get("descriptor_sha256") or ""):
        raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: descriptor hash mismatch")
    if _sha256_file(adapter) != str(payload.get("adapter_sha256") or ""):
        raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: adapter hash mismatch")
    success_receipt = loop_root / "qe_long_trend_registration.json"
    if success_receipt.is_file() and not success_receipt.is_symlink():
        success = _read_json(success_receipt)
        evaluation_id = str(success.get("evaluation_id") or "")
        request_sha = str(success.get("request_sha") or "")
        if (
            success.get("schema_version") == "qe_long_trend_registration_v1"
            and evaluation_id.startswith("qelt_")
            and len(evaluation_id) == 69
            and len(request_sha) == 64
        ):
            index_path.unlink(missing_ok=True)
            return
    pending_receipt = loop_root / "postprocess_registration_pending.json"
    if (
        pending_receipt.is_symlink()
        or not pending_receipt.is_file()
        or _sha256_file(pending_receipt) != str(payload.get("pending_receipt_sha256") or "")
    ):
        raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: pending receipt hash mismatch")
    log_path = loop_root / "long_trend_registration_replay.log"
    with log_path.open("ab") as log_handle:
        completed = subprocess.run(  # noqa: S603 - fixed interpreter, fixed validated adapter path, no shell.
            [sys.executable, str(adapter)],
            cwd=str(loop_root),
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            timeout=45,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"QELT_REGISTRATION_REPLAY_FAILED: task_id={task_id} loop_id={loop_id} returncode={completed.returncode}",
        )
    if index_path.exists() and not (loop_root / "postprocess_registration_pending.json").is_file():
        raise RuntimeError("QELT_REGISTRATION_REPLAY_FAILED: adapter left an inconsistent pending index")


def _append_error(workspace: Path, payload: Mapping[str, Any]) -> None:
    target = workspace / ".qe_long_trend_registration_replay_errors.jsonl"
    line = json.dumps(
        {"created_at": datetime.now(timezone.utc).isoformat(), **dict(payload)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_component(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise RuntimeError(f"QELT_REGISTRATION_PENDING_INDEX_INVALID: invalid {field_name}")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("QELT_REGISTRATION_PENDING_INDEX_INVALID: JSON must be an object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
