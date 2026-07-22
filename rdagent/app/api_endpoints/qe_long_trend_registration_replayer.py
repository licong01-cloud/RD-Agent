"""Durable recovery daemon for normal-Loop F-014 registrations pending before control-row creation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

PENDING_INDEX_SCHEMA = "qe_long_trend_registration_pending_index_v1"
PENDING_DIR = ".qe_long_trend_registration_pending"


def main(argv: list[str] | None = None) -> int:
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    if not 1.0 <= float(args.poll_interval_seconds) <= 300.0:
        parser.error("--poll-interval-seconds must be between 1 and 300")
    workspace = Path(args.workspace_root).resolve(strict=True)
    lock_path = workspace / ".qe_long_trend_registration_replay.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        _run_replay_loop(workspace, poll_interval_seconds=float(args.poll_interval_seconds))
    return 0


def _run_replay_loop(workspace: Path, *, poll_interval_seconds: float) -> None:
    while True:
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
        time.sleep(poll_interval_seconds)


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
    fd, name = tempfile.mkstemp(prefix="qelt_replay_error_", dir=workspace)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            if target.is_file():
                handle.write(target.read_bytes())
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


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
