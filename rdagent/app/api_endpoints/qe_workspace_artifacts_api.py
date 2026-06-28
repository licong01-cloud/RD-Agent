"""Content-addressed workspace artifact store for QE execution nodes.

AIstock combine-backtest remote dispatch uses this API for large workspace
artifacts that do not fit the existing small experiment-file channel.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.requests import ClientDisconnect

router = APIRouter(prefix="/api/v1/qe_workspace/artifacts", tags=["qe-workspace-artifacts"])

_STREAM_CHUNK_SIZE = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
logger = logging.getLogger(__name__)


def _validate_sha256(value: str) -> str:
    sha256 = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(sha256):
        raise HTTPException(status_code=400, detail=f"invalid workspace artifact sha256: {value!r}")
    return sha256


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _artifact_store_root() -> Path:
    raw = (
        os.environ.get("QE_WORKSPACE_ARTIFACT_STORE")
        or os.environ.get("QE_WORKSPACE_ARTIFACT_STORE_ROOT")
        or os.environ.get("AISTOCK_QE_WORKSPACE_ARTIFACT_STORE")
    )
    root = Path(raw).expanduser() if raw and raw.strip() else _repo_root() / "qe_workspace_artifact_store"
    if not root.is_absolute():
        raise HTTPException(status_code=500, detail="QE workspace artifact store root must be absolute")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_path(sha256: str) -> Path:
    return _artifact_store_root() / _validate_sha256(sha256)


def _unlink_tmp_file(tmp_path: Path | None) -> None:
    if tmp_path is None:
        return
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError as cleanup_exc:
        logger.warning("failed to remove rejected workspace artifact temp file %s: %s", tmp_path, cleanup_exc)


def _raise_empty_upload(path: Path) -> None:
    raise HTTPException(status_code=400, detail=f"empty workspace artifact upload: {path.name}")


def _raise_sha_mismatch(*, expected_sha256: str, actual_sha256: str) -> None:
    raise HTTPException(
        status_code=400,
        detail=(
            "workspace artifact sha256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        ),
    )


async def _atomic_write_artifact_stream(path: Path, request: Request, *, expected_sha256: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    size = 0
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            async for chunk in request.stream():
                if not chunk:
                    continue
                size += len(chunk)
                digest.update(chunk)
                tmp.write(chunk)
    except (ClientDisconnect, OSError) as exc:
        _unlink_tmp_file(tmp_path)
        raise HTTPException(
            status_code=500,
            detail=f"workspace artifact stream write failed: {type(exc).__name__}: {exc}",
        ) from exc

    if size <= 0:
        _unlink_tmp_file(tmp_path)
        _raise_empty_upload(path)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        _unlink_tmp_file(tmp_path)
        _raise_sha_mismatch(expected_sha256=expected_sha256, actual_sha256=actual_sha256)
    try:
        tmp_path.replace(path)
    except OSError as exc:
        _unlink_tmp_file(tmp_path)
        raise HTTPException(
            status_code=500,
            detail=f"workspace artifact commit failed: {type(exc).__name__}: {exc}",
        ) from exc
    return {"exists": True, "size": size, "sha256": actual_sha256}


def _artifact_status_payload(sha256: str) -> dict[str, Any]:
    normalized = _validate_sha256(sha256)
    path = _artifact_path(normalized)
    exists = path.exists() and path.is_file()
    return {
        "exists": exists,
        "size": path.stat().st_size if exists else 0,
        "sha256": normalized,
        "artifact_store_root": str(path.parent),
    }


@router.head("/{sha256}")
def head_workspace_artifact(sha256: str) -> Response:
    """Return artifact status via headers for HEAD clients."""

    payload = _artifact_status_payload(sha256)
    return Response(
        status_code=200,
        headers={
            "X-Artifact-Exists": "1" if payload["exists"] else "0",
            "X-Artifact-Size": str(payload["size"]),
            "X-Artifact-Sha256": str(payload["sha256"]),
            "X-Artifact-Store-Root": str(payload["artifact_store_root"]),
        },
    )


@router.get("/{sha256}")
def get_workspace_artifact(sha256: str) -> JSONResponse:
    """Return whether a content-addressed artifact is cached on this node."""

    return JSONResponse(_artifact_status_payload(sha256))


@router.post("/{sha256}")
async def upload_workspace_artifact(sha256: str, request: Request) -> dict[str, Any]:
    """Stream-upload an artifact and verify its URL sha256 before accepting it."""

    normalized = _validate_sha256(sha256)
    path = _artifact_path(normalized)
    result = await _atomic_write_artifact_stream(path, request, expected_sha256=normalized)
    return {
        "ok": True,
        "artifact_store_root": str(path.parent),
        **result,
    }
