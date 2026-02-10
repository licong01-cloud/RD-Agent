"""Snapshot default templates into app_tpl bundle."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

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
DEFAULT_SCENARIO = "qlib"
DEFAULT_VERSION = "v0"


def _load_registry() -> dict[str, Any] | None:
    if not REGISTRY_PATH.exists():
        return None
    if yaml is None:
        message = "Missing PyYAML dependency for template registry"
        raise RuntimeError(message)
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


def _build_manifest(output_dir: Path, scenario: str, version: str) -> None:
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
        "scenario": scenario,
        "version": version,
        "created_at": datetime.now(LOCAL_TZ).isoformat(),
        "files": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def snapshot_templates(
    *,
    scenario: str,
    version: str,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    output_dir = output_root / scenario / version
    if output_dir.exists() and any(output_dir.rglob("*")) and not overwrite:
        message = f"Output directory not empty: {output_dir}"
        raise FileExistsError(message)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    template_paths = _collect_default_templates(scenario)
    if dry_run:
        _print_dry_run_summary("snapshot", template_paths)
    for template_path in template_paths:
        if dry_run:
            continue
        _copy_file(template_path, output_dir)

    if not dry_run:
        _build_manifest(output_dir, scenario, version)

    return output_dir


def _print_dry_run_summary(action: str, template_paths: list[Path]) -> None:
    print(f"[dry-run] {action} template files: {len(template_paths)}")
    for path in template_paths:
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        print(f"[dry-run] - {rel_path}")


def _resolve_output_root(raw: str) -> Path:
    root = Path(raw)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot default templates into app_tpl")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, help="Scenario name")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Template version (default: v0)")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "app_tpl"),
        help="Output root folder (default: app_tpl)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    output_dir = snapshot_templates(
        scenario=args.scenario,
        version=args.version,
        output_root=_resolve_output_root(args.output_root),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"Template snapshot output: {output_dir}")


if __name__ == "__main__":
    main()
