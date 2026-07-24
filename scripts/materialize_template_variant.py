"""Materialize a fixed-horizon template variant outside the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "aistock_fixed_horizon_template_variant_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MaterializationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_spec(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid variant spec: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise MaterializationError(f"unsupported variant schema: {payload.get('schema_version')}")
    for key in (
        "scenario",
        "version",
        "base_version",
        "label_horizon",
        "label_from",
        "label_to",
        "expected_replacement_count",
    ):
        if payload.get(key) in (None, ""):
            raise MaterializationError(f"variant spec missing {key}: {path}")
    return payload


def _manifest(output_dir: Path, spec: dict[str, Any], changed_files: list[str]) -> dict[str, Any]:
    entries = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
    content_identity = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": spec["scenario"],
        "version": spec["version"],
        "base_version": spec["base_version"],
        "label_horizon": int(spec["label_horizon"]),
        "label_expression": spec["label_to"],
        "changed_files": sorted(changed_files),
        "content_sha256": content_identity,
        "files": entries,
    }


def materialize(*, spec_path: Path, base_dir: Path, output_dir: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    base_dir = base_dir.resolve()
    output_dir = output_dir.resolve()
    project_root = PROJECT_ROOT.resolve()
    if _is_relative_to(output_dir, project_root):
        raise MaterializationError(f"output_dir must be outside repository: {output_dir}")
    if output_dir.exists():
        raise MaterializationError(f"output_dir already exists: {output_dir}")
    if not base_dir.is_dir() or not (base_dir / "manifest.json").is_file():
        raise MaterializationError(f"base template is incomplete: {base_dir}")
    required_inputs = (
        "rdagent/scenarios/qlib/experiment/factor_template/benchmark_sh000300.parquet",
        "rdagent/scenarios/qlib/experiment/model_template/benchmark_sh000300.parquet",
        "rdagent/scenarios/qlib/experiment/factor_template/minute_execution_contract.py",
        "rdagent/scenarios/qlib/experiment/model_template/minute_execution_contract.py",
        "rdagent/scenarios/qlib/experiment/factor_template/tail_twap_strategy.py",
        "rdagent/scenarios/qlib/experiment/model_template/tail_twap_strategy.py",
        "rdagent/scenarios/qlib/experiment/factor_template/tail_twap_v24_strategy.py",
        "rdagent/scenarios/qlib/experiment/model_template/tail_twap_v24_strategy.py",
        "rdagent/scenarios/qlib/experiment/factor_template/tail_twap_v25_strategy.py",
        "rdagent/scenarios/qlib/experiment/model_template/tail_twap_v25_strategy.py",
    )
    missing = [relative for relative in required_inputs if not (base_dir / relative).is_file()]
    if missing:
        raise MaterializationError(f"base template missing required runtime inputs: {missing}")

    spec = _load_spec(spec_path)
    expected_base = str(spec["base_version"])
    if base_dir.name != expected_base:
        raise MaterializationError(
            f"base template version mismatch: expected={expected_base} actual={base_dir.name}"
        )

    replacement_count = 0
    changed_relative_paths: list[str] = []
    for yaml_path in sorted(base_dir.rglob("*.yaml")):
        content = yaml_path.read_text(encoding="utf-8")
        count = content.count(str(spec["label_from"]))
        if count:
            replacement_count += count
            changed_relative_paths.append(yaml_path.relative_to(base_dir).as_posix())
    if replacement_count != int(spec["expected_replacement_count"]):
        raise MaterializationError(
            "fixed-horizon replacement count mismatch: "
            f"expected={spec['expected_replacement_count']} actual={replacement_count}"
        )

    shutil.copytree(base_dir, output_dir, copy_function=shutil.copy2)
    for relative in changed_relative_paths:
        yaml_path = output_dir / relative
        content = yaml_path.read_text(encoding="utf-8")
        yaml_path.write_text(
            content.replace(str(spec["label_from"]), str(spec["label_to"])),
            encoding="utf-8",
        )

    changed_files = changed_relative_paths
    manifest = _manifest(output_dir, spec, changed_files)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Variant JSON specification.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT / "app_tpl/all/v4",
        help="Physical base template directory (default: app_tpl/all/v4).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New artifact directory outside the repo.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        manifest = materialize(spec_path=args.spec, base_dir=args.base_dir, output_dir=args.output_dir)
    except MaterializationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "manifest": manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
