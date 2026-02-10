"""Build an app_tpl template bundle from TaskConfig."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).parent.parent
LOCAL_TZ = timezone(timedelta(hours=8))
REGISTRY_PATH = REPO_ROOT / "configs" / "template_registry.yaml"
DEFAULT_TEMPLATE_GLOBS = {
    "qlib": [
        "rdagent/scenarios/qlib/prompts.yaml",
        "rdagent/scenarios/qlib/experiment/prompts.yaml",
        "rdagent/scenarios/qlib/experiment/model_template/*.yaml",
        "rdagent/scenarios/qlib/experiment/factor_template/*.yaml",
    ],
}
APP_TPL_VERSION_PARTS = 1
APP_TPL_SCENARIO_PARTS = 2


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        message = "TaskConfig must be a JSON object"
        raise TypeError(message)
    return cast("dict[str, Any]", data)


def _ensure_relative_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() or ".." in path.parts:
        message = f"Invalid relative path: {path_str}"
        raise ValueError(message)
    return path


def _resolve_output_dir(task_cfg: dict[str, Any], output_root: Path) -> tuple[Path, str, str]:
    app_tpl = task_cfg.get("app_tpl")
    scenario = task_cfg.get("scenario") or "qlib"
    version = task_cfg.get("version")
    if app_tpl:
        app_tpl_path = _ensure_relative_path(app_tpl)
        output_dir = REPO_ROOT / app_tpl_path
        parts = app_tpl_path.parts
        if not version and len(parts) >= APP_TPL_VERSION_PARTS:
            version = parts[-1]
        if not scenario and len(parts) >= APP_TPL_SCENARIO_PARTS:
            scenario = parts[-2]
    else:
        if not version:
            message = "Missing 'version' in TaskConfig"
            raise ValueError(message)
        output_dir = output_root / scenario / version
    return output_dir, scenario, version or "unknown"


def _load_registry() -> dict[str, Any] | None:
    if not REGISTRY_PATH.exists():
        return None
    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        message = "Template registry must be a mapping"
        raise TypeError(message)
    return data


def _matches_exclude(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def _iter_registry_scope(scope: dict[str, Any], scenario: str) -> list[tuple[Path, list[str], list[str]]]:
    scenarios = scope.get("scenarios") or []
    if scenarios and scenario != "all" and scenario not in scenarios:
        return []
    roots = scope.get("roots") or []
    include = scope.get("include") or []
    exclude = scope.get("exclude") or []
    if not isinstance(roots, list) or not isinstance(include, list) or not isinstance(exclude, list):
        message = "Template registry roots/include/exclude must be lists"
        raise TypeError(message)
    resolved: list[tuple[Path, list[str], list[str]]] = []
    for root in roots:
        root_path = REPO_ROOT / str(root)
        if not root_path.exists():
            message = f"Template registry root not found: {root_path}"
            raise FileNotFoundError(message)
        resolved.append((root_path, [str(p) for p in include], [str(p) for p in exclude]))
    return resolved


def _collect_registry_matches(
    files: set[Path],
    root_path: Path,
    include: list[str],
    exclude: list[str],
) -> None:
    for pattern in include:
        for match in root_path.glob(pattern):
            if not match.is_file():
                continue
            if match.suffix == ".py":
                message = f"Registry matched core file: {match}"
                raise ValueError(message)
            if _matches_exclude(match, root_path, exclude):
                continue
            if _is_backup_file(match):
                continue
            files.add(match)


def _collect_registry_templates(scenario: str) -> list[Path]:
    registry = _load_registry()
    if not registry:
        return []
    scopes = registry.get("scopes") or []
    if not isinstance(scopes, list):
        message = "Template registry scopes must be a list"
        raise TypeError(message)
    files: set[Path] = set()
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        for root_path, include, exclude in _iter_registry_scope(scope, scenario):
            _collect_registry_matches(files, root_path, include, exclude)
    return sorted(files)


def _collect_default_templates(scenario: str) -> list[Path]:
    registry_files = _collect_registry_templates(scenario)
    if registry_files:
        return registry_files
    patterns = DEFAULT_TEMPLATE_GLOBS.get(scenario)
    if not patterns:
        message = f"Unsupported scenario: {scenario}"
        raise ValueError(message)
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(REPO_ROOT.glob(pattern))
        if not matches:
            message = f"No template files matched: {pattern}"
            raise FileNotFoundError(message)
        files.extend(match for match in matches if not _is_backup_file(match))
    return files


def _is_backup_file(path: Path) -> bool:
    name = path.name.lower()
    return "backup" in name or name.endswith(".bak")


def _copy_file(src: Path, output_dir: Path) -> Path:
    rel_path = src.relative_to(REPO_ROOT)
    dst = output_dir / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _apply_template_files(output_dir: Path, template_files: list[dict[str, Any]] | None) -> None:
    if not template_files:
        return
    for item in template_files:
        rel_path = _ensure_relative_path(item.get("path", ""))
        dst = output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        source = item.get("source", "default")
        content = item.get("content")
        if source == "inline" or content is not None:
            if content is None:
                message = f"Missing content for inline template: {rel_path}"
                raise ValueError(message)
            dst.write_text(content, encoding="utf-8")
            continue
        src = REPO_ROOT / rel_path
        if not src.exists():
            message = f"Template source not found: {src}"
            raise FileNotFoundError(message)
        shutil.copy2(src, dst)


def _patch_prompt_value(value: str, patch: dict[str, Any]) -> str:
    if not isinstance(value, str):
        value = ""
    if "replace" in patch and patch["replace"] is not None:
        value = str(patch["replace"])
    prepend = patch.get("prepend") or []
    append = patch.get("append") or []
    if prepend:
        value = "\n".join(prepend) + "\n" + value
    if append:
        value = value.rstrip() + "\n" + "\n".join(append)
    return value


def _apply_prompt_patch(output_dir: Path, task_cfg: dict[str, Any]) -> None:
    prompts_path = output_dir / "rdagent/scenarios/qlib/prompts.yaml"
    if not prompts_path.exists():
        return
    prompt_data = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or {}
    patch_cfg = task_cfg.get("prompt_patch") or {}
    for key, patch in patch_cfg.items():
        if key not in prompt_data or not isinstance(patch, dict):
            continue
        prompt_data[key] = _patch_prompt_value(str(prompt_data.get(key, "")), patch)

    model_allowlist = task_cfg.get("model_allowlist") or []
    if model_allowlist:
        allowlist_text = (
            "IMPORTANT: Only use models from allowlist: " + ", ".join(model_allowlist) + "."
        )
        prompt_data["model_hypothesis_specification"] = (
            str(prompt_data.get("model_hypothesis_specification", "")).rstrip()
            + "\n"
            + allowlist_text
        )

    factor_allowlist = task_cfg.get("factor_allowlist") or []
    if factor_allowlist:
        allowlist_text = (
            "IMPORTANT: Only design factors from allowlist: " + ", ".join(factor_allowlist) + "."
        )
        prompt_data["factor_hypothesis_specification"] = (
            str(prompt_data.get("factor_hypothesis_specification", "")).rstrip()
            + "\n"
            + allowlist_text
        )

    prompts_path.write_text(
        yaml.safe_dump(prompt_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _build_manifest(
    output_dir: Path,
    scenario: str,
    version: str,
    task_cfg: dict[str, Any],
) -> None:
    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(output_dir).as_posix()
        if rel_path == "manifest.json":
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": rel_path, "sha256": sha, "size": path.stat().st_size})
    description = task_cfg.get("template_description")
    base_version = task_cfg.get("base_version")
    changed_files = task_cfg.get("changed_files") or []
    if not isinstance(changed_files, list):
        changed_files = []

    manifest = {
        "scenario": scenario,
        "version": version,
        "task_id": task_cfg.get("task_id"),
        "created_at": datetime.now(LOCAL_TZ).isoformat(),
        "description": description,
        "base_version": base_version,
        "changed_files": changed_files,
        "files": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_bundle(
    task_cfg_path: Path,
    output_root: Path,
    *,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    task_cfg = _load_json(task_cfg_path)
    output_dir, scenario, version = _resolve_output_dir(task_cfg, output_root)

    if output_dir.exists() and any(output_dir.rglob("*")) and not overwrite:
        message = f"Output directory not empty: {output_dir}"
        raise FileExistsError(message)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    template_base = task_cfg.get("template_base", "default")
    if template_base != "none":
        template_paths = _collect_default_templates(scenario)
        if dry_run:
            _print_dry_run_summary("bundle", template_paths)
        for template_path in template_paths:
            if dry_run:
                continue
            _copy_file(template_path, output_dir)

    if not dry_run:
        _apply_template_files(output_dir, task_cfg.get("template_files"))

    if not dry_run:
        _apply_prompt_patch(output_dir, task_cfg)
        _build_manifest(output_dir, scenario, version, task_cfg)

    return output_dir


def _print_dry_run_summary(action: str, template_paths: list[Path]) -> None:
    print(f"[dry-run] {action} template files: {len(template_paths)}")
    for path in template_paths:
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        print(f"[dry-run] - {rel_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build app_tpl template bundle from TaskConfig")
    parser.add_argument("--task-config", required=True, help="Path to TaskConfig JSON")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "app_tpl"),
        help="Output root folder (default: app_tpl)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    output_dir = build_bundle(
        Path(args.task_config),
        Path(args.output_root),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"Template bundle output: {output_dir}")


if __name__ == "__main__":
    main()
