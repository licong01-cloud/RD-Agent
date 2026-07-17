"""Read-only, loop-relative QE workspace catalog construction."""

from __future__ import annotations

import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

SCHEMA_VERSION = "hmm_qe_asset_catalog_v1"
TERMINAL_COMPLETE_STATUSES = frozenset({"completed", "success", "succeeded"})


def resolve_loop_dir(workspace_base: Path, task_id: str, loop_id: str) -> Path:
    """Resolve a loop directory without allowing task/loop path traversal."""

    workspace_root = workspace_base.resolve()
    loop_dir = (workspace_root / task_id / loop_id).resolve()
    try:
        loop_dir.relative_to(workspace_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="workspace path escapes configured root",
        ) from exc
    return loop_dir


def _modified_at_iso(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _catalog_entry(
    relative_path: str,
    stat_result: os.stat_result,
    *,
    content_type: str | None,
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "size_bytes": stat_result.st_size,
        "sha256": None,
        "content_type": content_type,
        "modified_at": _modified_at_iso(stat_result.st_mtime),
        "source": "qe_workspace",
        "trust_level": "unverified_evidence",
        "access_mode": "inspection_only",
        "schema_version": SCHEMA_VERSION,
        "parser_contract": None,
    }


def _list_loop_entries(loop_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Enumerate the loop namespace without following links or reparse points."""

    files: list[dict[str, Any]] = []
    warnings: list[str] = []
    pending: list[tuple[Path, str]] = [(loop_dir, "")]
    while pending:
        directory, relative_prefix = pending.pop()
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            location = relative_prefix or "."
            raise HTTPException(
                status_code=500,
                detail=f"workspace catalog scan failed at {location}: {exc}",
            ) from exc
        for child in children:
            relative_path = (
                f"{relative_prefix}/{child.name}"
                if relative_prefix
                else child.name
            )
            try:
                stat_result = child.stat(follow_symlinks=False)
                if child.is_symlink():
                    files.append(
                        _catalog_entry(
                            relative_path,
                            stat_result,
                            content_type="inode/symlink",
                        ),
                    )
                    warnings.append(f"symlink_not_followed:{relative_path}")
                elif child.is_dir(follow_symlinks=False):
                    pending.append((Path(child.path), relative_path))
                elif child.is_file(follow_symlinks=False):
                    content_type, _encoding = mimetypes.guess_type(child.name)
                    files.append(
                        _catalog_entry(
                            relative_path,
                            stat_result,
                            content_type=content_type or "application/octet-stream",
                        ),
                    )
                else:
                    warnings.append(f"special_entry_not_catalogued:{relative_path}")
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"workspace catalog stat failed at {relative_path}: {exc}",
                ) from exc
    files.sort(key=lambda row: row["relative_path"])
    warnings.sort()
    return files, warnings


def build_workspace_catalog(
    loop_dir: Path,
    *,
    task_id: str,
    loop_id: str,
) -> dict[str, Any]:
    """Build a deterministic catalog without hashing, copying, or following links."""

    files, warnings = _list_loop_entries(loop_dir)
    status_path = loop_dir / "status.txt"
    loop_status = ""
    if status_path.is_file():
        try:
            loop_status = status_path.read_text(encoding="utf-8").strip().lower()
        except (OSError, UnicodeError) as exc:
            warnings.append(f"status_unreadable:{type(exc).__name__}")
    terminal_complete = loop_status in TERMINAL_COMPLETE_STATUSES
    if not terminal_complete:
        warnings.append(f"loop_not_terminal:{loop_status or 'unknown'}")

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "loop_name": loop_id,
        "loop_status": loop_status or None,
        "catalog_completeness": "complete" if terminal_complete else "partial",
        "files": files,
        "warnings": sorted(set(warnings)),
    }
