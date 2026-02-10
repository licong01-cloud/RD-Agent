"""Template publish/history/rollback service for app_tpl bundles."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, TemplateSyntaxError

from .models import TemplateHistoryRecord
from .storage import append_history, list_history

PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_TPL_ROOT = PROJECT_ROOT / "app_tpl"
HISTORY_ROOT = PROJECT_ROOT / "history"
TEMPLATE_HISTORY_DIR = HISTORY_ROOT / "template_bundles"
LOCAL_TZ = timezone(timedelta(hours=8))

ALLOWED_SUFFIXES = {".yaml", ".yml", ".json"}
_JINJA_ENV = Environment(autoescape=True)


def _timestamp() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_rel_path(rel_path: str) -> str:
    if not rel_path or rel_path.startswith(("/", "\\")):
        message = f"Invalid template path: {rel_path}"
        raise ValueError(message)
    norm = Path(rel_path)
    if ".." in norm.parts:
        message = f"Invalid template path: {rel_path}"
        raise ValueError(message)
    if not rel_path.startswith("rdagent/"):
        message = f"Template path must be under rdagent/: {rel_path}"
        raise ValueError(message)
    if norm.suffix not in ALLOWED_SUFFIXES:
        message = f"Unsupported template file type: {rel_path}"
        raise ValueError(message)
    return rel_path


def _validate_content(rel_path: str, content: str) -> None:
    suffix = Path(rel_path).suffix
    try:
        if suffix in {".yaml", ".yml"}:
            yaml.safe_load(content)
        elif suffix == ".json":
            json.loads(content)
    except Exception as exc:
        message = f"Invalid {suffix} content for template: {rel_path}"
        raise ValueError(message) from exc
    try:
        if "{{" in content or "{%" in content:
            _JINJA_ENV.parse(content)
    except TemplateSyntaxError as exc:
        message = f"Invalid Jinja template syntax: {rel_path}"
        raise ValueError(message) from exc


def _backup_existing(output_dir: Path, scenario: str, version: str) -> Path | None:
    if not output_dir.exists():
        return None
    if not any(output_dir.rglob("*")):
        return None
    _ensure_dir(TEMPLATE_HISTORY_DIR)
    backup_dir = TEMPLATE_HISTORY_DIR / f"{_timestamp()}_{scenario}_{version}"
    shutil.copytree(output_dir, backup_dir)
    return backup_dir


def _write_files(output_dir: Path, files: Iterable[dict[str, Any]]) -> None:
    for item in files:
        rel_path = _validate_rel_path(str(item.get("path", "")))
        content = item.get("content")
        if content is None:
            message = f"Missing content for template: {rel_path}"
            raise ValueError(message)
        _validate_content(rel_path, str(content))
        dst = output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(str(content), encoding="utf-8")


def _build_manifest(output_dir: Path, payload: dict[str, Any]) -> Path:
    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(output_dir).as_posix()
        if rel_path == "manifest.json":
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": rel_path, "sha256": sha, "size": path.stat().st_size})
    manifest = {
        "scenario": payload.get("scenario"),
        "version": payload.get("version"),
        "task_id": payload.get("task_id"),
        "created_at": datetime.now(LOCAL_TZ).isoformat(),
        "description": payload.get("description"),
        "base_version": payload.get("base_version"),
        "changed_files": payload.get("changed_files") or [],
        "files": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _record_history(
    *,
    action: str,
    scenario: str,
    version: str,
    task_id: str | None,
    backup_path: Path | None = None,
    manifest_path: Path | None = None,
) -> None:
    record = TemplateHistoryRecord(
        file_name="manifest.json",
        backup_path=str(backup_path) if backup_path else "",
        task_id=task_id,
        extra={
            "action": action,
            "scenario": scenario,
            "version": version,
            "manifest_path": str(manifest_path) if manifest_path else "",
        },
    )
    append_history(record)


def publish_templates(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = payload.get("scenario")
    version = payload.get("version")
    files = payload.get("files") or []
    if not scenario or not version:
        message = "Missing scenario or version"
        raise ValueError(message)
    if not isinstance(files, list) or not files:
        message = "files must be a non-empty list"
        raise ValueError(message)

    output_dir = APP_TPL_ROOT / scenario / version
    # 仅记录历史，不备份文件（app_tpl/不是运行时代码，发布不影响RDAgent运行）
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_files(output_dir, files)
    manifest_path = _build_manifest(output_dir, payload)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    _record_history(
        action="publish",
        scenario=scenario,
        version=version,
        task_id=payload.get("task_id"),
        backup_path=None,
        manifest_path=manifest_path,
    )

    return {
        "status": "ok",
        "scenario": scenario,
        "version": version,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_hash,
        "backup_path": "",
    }


def list_template_history(scenario: str | None = None, version: str | None = None) -> dict[str, Any]:
    records = []
    for record in list_history():
        data = asdict(record)
        extra = data.get("extra") or {}
        if scenario and extra.get("scenario") != scenario:
            continue
        if version and extra.get("version") != version:
            continue
        records.append(data)
    return {"items": records}


def rollback_template(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = payload.get("scenario")
    version = payload.get("version")
    backup_path = payload.get("backup_path")
    if not backup_path and (not scenario or not version):
        message = "Missing scenario/version or backup_path"
        raise ValueError(message)

    backup = Path(backup_path) if backup_path else None
    if backup is None:
        history = list_template_history(scenario=scenario, version=version)["items"]
        if not history:
            message = "No history records found for rollback"
            raise ValueError(message)
        backup_str = history[-1].get("backup_path")
        if not backup_str:
            message = "History record missing backup_path"
            raise ValueError(message)
        backup = Path(backup_str)

    output_dir = APP_TPL_ROOT / str(scenario) / str(version)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(backup, output_dir)

    _record_history(
        action="rollback",
        scenario=str(scenario),
        version=str(version),
        task_id=payload.get("task_id"),
        backup_path=backup,
    )

    return {
        "status": "ok",
        "scenario": scenario,
        "version": version,
        "output_dir": str(output_dir),
        "backup_path": str(backup),
    }


__all__ = [
    "list_template_history",
    "publish_templates",
    "rollback_template",
]
