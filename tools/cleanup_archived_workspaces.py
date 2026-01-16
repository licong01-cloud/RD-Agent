import argparse
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ArchivedExperiment:
    task_run_id: str | None
    loop_id: int | None
    workspace_id: str | None
    workspace_path: Path | None
    archiving_level: int | None
    can_cleanup: bool | None


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load archived experiments manifest from JSON.

    支持两种结构：
    - 顶层就是 list[entry]
    - {"experiments": [entry, ...]}
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        items = data.get("experiments")
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
    raise SystemExit(f"Unsupported manifest structure in {path!s}")


def _build_workspace_index(conn: sqlite3.Connection) -> dict[str, Path]:
    """Build index: workspace_id -> workspace_path."""

    sql = """
    SELECT DISTINCT
      workspace_id,
      workspace_path
    FROM workspaces
    WHERE workspace_path IS NOT NULL AND workspace_path != ''
    """
    out: dict[str, Path] = {}
    for row in conn.execute(sql):
        ws_id = str(row["workspace_id"])
        ws_path = str(row["workspace_path"])
        if not ws_path:
            continue
        out[ws_id] = Path(ws_path)
    return out


def _iter_archived_experiments(
    manifest_entries: Iterable[dict[str, Any]],
    ws_index: dict[str, Path],
    *,
    min_level: int,
) -> list[ArchivedExperiment]:
    results: list[ArchivedExperiment] = []

    for item in manifest_entries:
        task_run_id = str(item.get("task_run_id")) if item.get("task_run_id") is not None else None
        loop_id_raw = item.get("loop_id")
        loop_id: int | None = None
        if isinstance(loop_id_raw, int):
            loop_id = loop_id_raw
        else:
            try:
                if loop_id_raw is not None:
                    loop_id = int(loop_id_raw)
            except Exception:
                loop_id = None

        workspace_id = str(item.get("workspace_id")) if item.get("workspace_id") is not None else None
        archiving_level_raw = item.get("archiving_level")
        archiving_level: int | None = None
        if isinstance(archiving_level_raw, int):
            archiving_level = archiving_level_raw
        else:
            try:
                if archiving_level_raw is not None:
                    archiving_level = int(archiving_level_raw)
            except Exception:
                archiving_level = None

        can_cleanup_val = item.get("can_cleanup")
        can_cleanup: bool | None
        if isinstance(can_cleanup_val, bool):
            can_cleanup = can_cleanup_val
        elif can_cleanup_val in ("true", "True", "1", 1):
            can_cleanup = True
        elif can_cleanup_val in ("false", "False", "0", 0):
            can_cleanup = False
        else:
            can_cleanup = None

        # 过滤条件：
        # - archiving_level >= min_level（如提供）；
        # - 若提供 can_cleanup，则要求为 True。
        if archiving_level is not None and archiving_level < min_level:
            continue
        if can_cleanup is False:
            continue

        ws_path: Path | None = None

        # 1) 优先用 workspace_path（若直接提供）。
        if isinstance(item.get("workspace_path"), str) and item["workspace_path"]:
            ws_path = Path(item["workspace_path"])
        # 2) 否则尝试通过 workspace_id 从 registry 中解析路径。
        elif workspace_id is not None and workspace_id in ws_index:
            ws_path = ws_index[workspace_id]

        results.append(
            ArchivedExperiment(
                task_run_id=task_run_id,
                loop_id=loop_id,
                workspace_id=workspace_id,
                workspace_path=ws_path,
                archiving_level=archiving_level,
                can_cleanup=can_cleanup,
            )
        )

    return results


def _print_plan(exps: list[ArchivedExperiment]) -> None:
    print("Planned workspace deletions (unique paths):")
    seen: set[Path] = set()
    for exp in exps:
        if exp.workspace_path is None:
            continue
        if exp.workspace_path in seen:
            continue
        seen.add(exp.workspace_path)
        meta = []
        if exp.task_run_id:
            meta.append(f"task_run_id={exp.task_run_id}")
        if exp.loop_id is not None:
            meta.append(f"loop_id={exp.loop_id}")
        if exp.workspace_id:
            meta.append(f"workspace_id={exp.workspace_id}")
        if exp.archiving_level is not None:
            meta.append(f"archiving_level={exp.archiving_level}")
        print(f"- {exp.workspace_path}  (" + ", ".join(meta) + ")")


def _delete_workspaces(exps: list[ArchivedExperiment]) -> None:
    seen: set[Path] = set()
    for exp in exps:
        ws_path = exp.workspace_path
        if ws_path is None:
            continue
        if ws_path in seen:
            continue
        seen.add(ws_path)
        if not ws_path.exists():
            print(f"[SKIP] {ws_path} (not found)")
            continue
        try:
            print(f"[DELETE] {ws_path}")
            shutil.rmtree(ws_path)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] failed to delete {ws_path}: {e!r}")


def run(registry_sqlite: Path, manifest_path: Path, *, min_level: int, confirm: bool) -> None:
    if not registry_sqlite.exists():
        raise SystemExit(f"registry.sqlite not found: {registry_sqlite}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")

    conn = sqlite3.connect(str(registry_sqlite))
    conn.row_factory = sqlite3.Row
    try:
        ws_index = _build_workspace_index(conn)
    finally:
        conn.close()

    manifest = _load_manifest(manifest_path)
    experiments = _iter_archived_experiments(manifest, ws_index, min_level=min_level)

    if not experiments:
        print("No experiments meet archiving_level / can_cleanup conditions; nothing to do.")
        return

    _print_plan(experiments)

    if not confirm:
        print("\nDry run only. Re-run with --confirm to actually delete these workspaces.")
        return

    print("\nProceeding to delete workspaces...\n")
    _delete_workspaces(experiments)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cleanup RD-Agent workspaces that have been fully archived on AIstock side. "
            "This tool relies on an AIstock-exported manifest JSON and registry.sqlite."
        )
    )
    parser.add_argument(
        "--registry-sqlite",
        required=True,
        help="Path to RD-Agent registry.sqlite (WSL/Linux path).",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "Path to AIstock-exported manifest JSON describing archived experiments. "
            "Structure: either a list[entry] or {\"experiments\": [entry, ...]} where each entry "
            "may contain task_run_id, loop_id, workspace_id, workspace_path, archiving_level, can_cleanup."
        ),
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=2,
        help="Minimum archiving_level to consider a workspace deletable (default: 2).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete workspaces. Without this flag, the script only prints a dry-run plan.",
    )

    args = parser.parse_args()

    registry_sqlite = Path(args.registry_sqlite)
    manifest_path = Path(args.manifest)

    run(registry_sqlite=registry_sqlite, manifest_path=manifest_path, min_level=args.min_level, confirm=args.confirm)


if __name__ == "__main__":
    main()
