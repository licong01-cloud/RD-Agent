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


# ---------------------------------------------------------------------------
# New template management APIs (P2)
# ---------------------------------------------------------------------------

BINARY_SUFFIXES = {".h5"}
BACKUPS_ROOT = PROJECT_ROOT / "git_ignore_folder" / "backups"


def _is_data_science_path(rel_path: str) -> bool:
    return "rdagent/scenarios/data_science/" in rel_path.replace("\\", "/")


def list_templates(scenario: str | None = None) -> dict[str, Any]:
    """List all template bundles under app_tpl/."""
    items: list[dict[str, Any]] = []
    if not APP_TPL_ROOT.exists():
        return {"items": items}
    for manifest in APP_TPL_ROOT.rglob("manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        sc = str(data.get("scenario") or manifest.parent.parent.name)
        if scenario and sc != scenario:
            continue
        version = str(data.get("version") or manifest.parent.name)
        files = data.get("files") if isinstance(data.get("files"), list) else []
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        items.append({
            "scenario": sc,
            "version": version,
            "created_at": data.get("created_at"),
            "description": data.get("description"),
            "base_version": data.get("base_version"),
            "changed_files": data.get("changed_files") or [],
            "files_count": len(files),
            "manifest_path": str(manifest),
            "manifest_hash": manifest_hash,
            "is_editable": version != "v0",
        })
    items.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    return {"items": items}


def list_template_files(scenario: str, version: str) -> dict[str, Any]:
    """List files in a template bundle (excluding binary and data_science)."""
    manifest_path = APP_TPL_ROOT / scenario / version / "manifest.json"
    if not manifest_path.exists():
        return {"error": "manifest not found", "items": []}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = data.get("files") if isinstance(data.get("files"), list) else []
    items = [f for f in files if isinstance(f, dict)]
    items = [item for item in items if Path(str(item.get("path") or "")).suffix not in BINARY_SUFFIXES]
    items = [item for item in items if not _is_data_science_path(str(item.get("path") or ""))]
    items.sort(key=lambda x: str(x.get("path") or ""))
    return {"items": items}


def get_template_file(scenario: str, version: str, rel_path: str) -> dict[str, Any]:
    """Read a single template file content."""
    rel_path = _validate_rel_path(rel_path)
    if Path(rel_path).suffix in BINARY_SUFFIXES:
        return {"error": "binary template file is not readable"}
    if _is_data_science_path(rel_path):
        return {"error": "data_science templates are hidden"}
    target = APP_TPL_ROOT / scenario / version / rel_path
    if not target.exists():
        return {"error": "file not found"}
    content = target.read_text(encoding="utf-8")
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"path": rel_path, "content": content, "sha256": sha}


def save_template_file(scenario: str, version: str, rel_path: str, content: str) -> dict[str, Any]:
    """Save content to a single template file."""
    if version == "v0":
        return {"error": "v0 template is read-only"}
    rel_path = _validate_rel_path(rel_path)
    if _is_data_science_path(rel_path):
        return {"error": "data_science templates are hidden"}
    _validate_content(rel_path, content)
    target = APP_TPL_ROOT / scenario / version / rel_path
    if not target.exists():
        return {"error": "file not found"}
    target.write_text(content, encoding="utf-8")
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    # update manifest entry
    manifest_path = APP_TPL_ROOT / scenario / version / "manifest.json"
    if manifest_path.exists():
        mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
        files_list = mdata.get("files") if isinstance(mdata.get("files"), list) else []
        updated = False
        for item in files_list:
            if isinstance(item, dict) and item.get("path") == rel_path:
                item["size"] = target.stat().st_size
                item["sha256"] = sha
                updated = True
                break
        if not updated:
            files_list.append({"path": rel_path, "size": target.stat().st_size, "sha256": sha})
        mdata["files"] = files_list
        manifest_path.write_text(json.dumps(mdata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "path": rel_path, "sha256": sha}


def delete_template(scenario: str, version: str) -> dict[str, Any]:
    """Delete a template bundle."""
    if version == "v0":
        return {"error": "v0 template is read-only"}
    target_root = APP_TPL_ROOT / scenario / version
    if not target_root.exists():
        return {"error": "template not found"}
    shutil.rmtree(target_root)
    _record_history(action="delete", scenario=scenario, version=version, task_id=None)
    return {"ok": True, "scenario": scenario, "version": version}


def _create_backup(scenario: str, version: str) -> str:
    """Create a backup of current rdagent/ runtime files listed in manifest."""
    backup_id = _timestamp()
    backup_root = BACKUPS_ROOT / backup_id
    backup_root.mkdir(parents=True, exist_ok=True)
    manifest_path = APP_TPL_ROOT / scenario / version / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    backed_up: list[dict[str, Any]] = []
    for file_info in files:
        rel_path = file_info.get("path")
        if not rel_path:
            continue
        source = PROJECT_ROOT / rel_path
        if source.exists():
            target = backup_root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            backed_up.append({
                "path": rel_path,
                "size": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            })
    meta = {
        "backup_id": backup_id,
        "created_at": datetime.now(LOCAL_TZ).isoformat(),
        "scenario": scenario,
        "version": version,
        "files": backed_up,
    }
    (backup_root / "backup_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _cleanup_old_backups()
    return backup_id


def _cleanup_old_backups(max_backups: int = 10) -> None:
    if not BACKUPS_ROOT.exists():
        return
    backups = sorted(
        [d for d in BACKUPS_ROOT.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True,
    )
    for old in backups[max_backups:]:
        shutil.rmtree(old)


def _apply_template_files(scenario: str, version: str) -> list[dict[str, Any]]:
    """Copy template files from app_tpl/ to rdagent/ runtime."""
    template_root = APP_TPL_ROOT / scenario / version
    manifest = json.loads((template_root / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    applied: list[dict[str, Any]] = []
    for file_info in files:
        rel_path = file_info.get("path")
        if not rel_path:
            continue
        src = template_root / rel_path
        dst = PROJECT_ROOT / rel_path
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        old_size = dst.stat().st_size if dst.exists() else 0
        shutil.copy2(src, dst)
        new_sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        applied.append({
            "path": rel_path,
            "size": dst.stat().st_size,
            "sha256": new_sha,
            "verified": new_sha == file_info.get("sha256"),
            "old_size": old_size,
            "size_changed": old_size != dst.stat().st_size,
        })
    return applied


def _verify_template(scenario: str, version: str) -> dict[str, Any]:
    """Verify that rdagent/ runtime files match the template."""
    template_root = APP_TPL_ROOT / scenario / version
    manifest = json.loads((template_root / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    result: dict[str, Any] = {
        "verified": True,
        "total_files": len(files),
        "verified_files": 0,
        "failed_files": [],
        "missing_files": [],
    }
    for file_info in files:
        rel_path = file_info.get("path")
        if not rel_path:
            continue
        rdagent_file = PROJECT_ROOT / rel_path
        if not rdagent_file.exists():
            result["verified"] = False
            result["missing_files"].append(rel_path)
            continue
        actual_sha = hashlib.sha256(rdagent_file.read_bytes()).hexdigest()
        if actual_sha == file_info.get("sha256"):
            result["verified_files"] += 1
        else:
            result["verified"] = False
            result["failed_files"].append({
                "path": rel_path,
                "expected": file_info.get("sha256"),
                "actual": actual_sha,
            })
    return result


def _rollback_from_backup(backup_id: str) -> dict[str, Any]:
    """Restore rdagent/ runtime files from a backup."""
    backup_root = BACKUPS_ROOT / backup_id
    if not backup_root.exists():
        raise ValueError(f"Backup not found: {backup_id}")
    meta_file = backup_root / "backup_meta.json"
    if not meta_file.exists():
        raise ValueError(f"Backup metadata not found: {backup_id}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    restored: list[dict[str, Any]] = []
    for file_info in meta.get("files", []):
        rel_path = file_info.get("path")
        if not rel_path:
            continue
        backup_file = backup_root / rel_path
        rdagent_file = PROJECT_ROOT / rel_path
        if not backup_file.exists():
            continue
        rdagent_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_file, rdagent_file)
        actual_sha = hashlib.sha256(rdagent_file.read_bytes()).hexdigest()
        restored.append({
            "path": rel_path,
            "verified": actual_sha == file_info.get("sha256"),
        })
    return {
        "backup_id": backup_id,
        "restored_files": restored,
        "scenario": meta.get("scenario"),
        "version": meta.get("version"),
    }


def apply_template(scenario: str, version: str, *, force: bool = False, backup: bool = True) -> dict[str, Any]:
    """Apply a template: backup current files, copy template files, verify."""
    manifest_path = APP_TPL_ROOT / scenario / version / "manifest.json"
    if not manifest_path.exists():
        return {"error": f"Template not found: {scenario}/{version}"}
    backup_id = None
    try:
        if backup:
            backup_id = _create_backup(scenario, version)
        applied = _apply_template_files(scenario, version)
        verification = _verify_template(scenario, version)
        if not verification["verified"] and not force:
            if backup_id:
                _rollback_from_backup(backup_id)
            return {
                "error": f"Verification failed: {verification['verified_files']}/{verification['total_files']}",
                "backup_id": backup_id,
                "verification": verification,
            }
        _record_history(
            action="apply",
            scenario=scenario,
            version=version,
            task_id=None,
            backup_path=BACKUPS_ROOT / backup_id if backup_id else None,
        )
        return {
            "ok": True,
            "scenario": scenario,
            "version": version,
            "backup_id": backup_id,
            "applied_files": applied,
            "verification": verification,
        }
    except Exception as exc:
        if backup_id:
            try:
                _rollback_from_backup(backup_id)
            except Exception:
                pass
        return {"error": str(exc), "backup_id": backup_id}


def get_sync_status() -> dict[str, Any]:
    """Check if rdagent/ runtime files match the active template."""
    # Try to detect active template from the most recent 'apply' history
    records = list_template_history()["items"]
    active: dict[str, str] | None = None
    for rec in reversed(records):
        extra = rec.get("extra") or {}
        if extra.get("action") == "apply":
            active = {"scenario": extra.get("scenario", ""), "version": extra.get("version", "")}
            break
    if not active or not active.get("scenario") or not active.get("version"):
        return {"synced": False, "active_template": None, "diff_files": []}
    try:
        verification = _verify_template(active["scenario"], active["version"])
        diff_files = verification.get("failed_files", []) + [
            {"path": p, "status": "missing"} for p in verification.get("missing_files", [])
        ]
        return {
            "synced": verification["verified"],
            "active_template": active,
            "diff_files": diff_files,
        }
    except Exception as e:
        return {"synced": False, "active_template": active, "diff_files": [], "error": str(e)}


def refresh_template_sha256(scenario: str, version: str) -> dict[str, Any]:
    """Recalculate SHA256 for all files in a template bundle."""
    template_root = APP_TPL_ROOT / scenario / version
    manifest_path = template_root / "manifest.json"
    if not manifest_path.exists():
        return {"error": f"Template not found: {scenario}/{version}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    updated_count = 0
    updated_files: list[dict[str, Any]] = []
    for file_info in files:
        rel_path = file_info.get("path")
        if not rel_path:
            continue
        fp = template_root / rel_path
        if not fp.exists():
            continue
        old_sha = file_info.get("sha256", "")
        new_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        new_size = fp.stat().st_size
        file_info["sha256"] = new_sha
        file_info["size"] = new_size
        if old_sha != new_sha:
            updated_count += 1
            updated_files.append({"path": rel_path, "old_sha256": old_sha[:16] + "...", "new_sha256": new_sha[:16] + "..."})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "ok": True,
        "scenario": scenario,
        "version": version,
        "total_files": len(files),
        "updated_files": updated_count,
        "updated_file_list": updated_files,
    }


def list_backups() -> dict[str, Any]:
    """List all backup snapshots."""
    if not BACKUPS_ROOT.exists():
        return {"items": []}
    items: list[dict[str, Any]] = []
    for backup_dir in sorted(BACKUPS_ROOT.iterdir(), key=lambda x: x.name, reverse=True):
        if not backup_dir.is_dir():
            continue
        meta_file = backup_dir / "backup_meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            items.append({
                "backup_id": meta.get("backup_id"),
                "created_at": meta.get("created_at"),
                "scenario": meta.get("scenario"),
                "version": meta.get("version"),
                "files_count": len(meta.get("files", [])),
            })
        except Exception:
            continue
    return {"items": items}


ENV_PATH = PROJECT_ROOT / ".env"


def _read_env_lines() -> list[str]:
    """Read .env file as lines."""
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(encoding="utf-8").splitlines()


def _write_env_lines(lines: list[str]) -> None:
    """Write lines back to .env file."""
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_app_tpl_value(scenario: str, version: str) -> str:
    """Compute the APP_TPL value for a given scenario/version.

    The value is relative to PROJ_PATH (rdagent/), e.g. ``../app_tpl/all/v4/rdagent``.
    """
    return f"../app_tpl/{scenario}/{version}/rdagent"


def activate_template_env(scenario: str, version: str) -> dict[str, Any]:
    """Activate a template by setting RD_AGENT_SETTINGS__APP_TPL in .env.

    This makes RDAgent's template loader (tpl.py) prioritise files under
    ``app_tpl/{scenario}/{version}/rdagent/`` without copying anything.

    Only yaml/txt templates loaded via ``T()`` are affected; .py files are
    NOT overridden by this mechanism.
    """
    # Validate template exists
    tpl_dir = APP_TPL_ROOT / scenario / version
    if not tpl_dir.exists():
        return {"error": f"Template not found: {scenario}/{version}"}
    rdagent_sub = tpl_dir / "rdagent"
    if not rdagent_sub.exists():
        return {"error": f"Template has no rdagent/ subdirectory: {scenario}/{version}"}

    app_tpl_value = _compute_app_tpl_value(scenario, version)
    env_key = "RD_AGENT_SETTINGS__APP_TPL"

    lines = _read_env_lines()
    found = False
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{env_key}=") or stripped.startswith(f"{env_key} ="):
            new_lines.append(f"{env_key}={app_tpl_value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{env_key}={app_tpl_value}")

    _write_env_lines(new_lines)

    # Classify which files are yaml-only vs py
    manifest_path = tpl_dir / "manifest.json"
    py_files: list[str] = []
    yaml_files: list[str] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for f in manifest.get("files", []):
            p = str(f.get("path", ""))
            if Path(p).suffix in {".py"}:
                py_files.append(p)
            else:
                yaml_files.append(p)

    return {
        "ok": True,
        "scenario": scenario,
        "version": version,
        "app_tpl": app_tpl_value,
        "yaml_files_covered": len(yaml_files),
        "py_files_need_copy": len(py_files),
        "py_files": py_files,
        "note": "yaml/txt templates are now loaded from app_tpl via APP_TPL env. "
                ".py files are NOT overridden; use apply_template (copy mode) if needed.",
    }


def get_active_env_template() -> dict[str, Any]:
    """Read the current RD_AGENT_SETTINGS__APP_TPL value from .env."""
    env_key = "RD_AGENT_SETTINGS__APP_TPL"
    lines = _read_env_lines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{env_key}="):
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            # Parse scenario/version from value like ../app_tpl/all/v4/rdagent
            parts = value.replace("\\", "/").split("/")
            # Expected: ../app_tpl/{scenario}/{version}/rdagent
            scenario = None
            version = None
            try:
                idx = parts.index("app_tpl")
                if idx + 2 < len(parts):
                    scenario = parts[idx + 1]
                    version = parts[idx + 2]
            except ValueError:
                pass
            return {
                "app_tpl": value,
                "scenario": scenario,
                "version": version,
                "source": ".env",
            }
    return {"app_tpl": None, "scenario": None, "version": None, "source": ".env"}


__all__ = [
    "activate_template_env",
    "apply_template",
    "delete_template",
    "get_active_env_template",
    "get_sync_status",
    "get_template_file",
    "list_backups",
    "list_template_files",
    "list_template_history",
    "list_templates",
    "publish_templates",
    "refresh_template_sha256",
    "rollback_template",
    "save_template_file",
]
