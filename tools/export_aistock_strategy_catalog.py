import argparse
import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@dataclass
class StrategySourceRow:
    task_run_id: str
    loop_id: int
    scenario: str | None
    step_name: str | None
    action: str | None
    workspace_id: str
    workspace_path: str


_STRATEGY_NS = uuid.UUID("7b6b2c6c-0b4d-4e2c-9b7f-3b1b9b4f3a11")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_strategy_sources(conn: sqlite3.Connection) -> list[StrategySourceRow]:
    sql = """
    SELECT
      l.task_run_id,
      l.loop_id,
      l.action,
      w.step_name,
      w.workspace_id,
      w.workspace_path
    FROM loops l
    JOIN workspaces w
      ON l.task_run_id = w.task_run_id
     AND l.loop_id = w.loop_id
    WHERE
      (l.has_result = 1 OR l.has_result = '1')
    """
    rows = conn.execute(sql).fetchall()
    out: list[StrategySourceRow] = []
    for r in rows:
        out.append(
            StrategySourceRow(
                task_run_id=str(r["task_run_id"]),
                loop_id=int(r["loop_id"]),
                # loops 表没有 scenario 字段，仅保留为 None。
                scenario=None,
                # step_name 来自 workspaces 表。
                step_name=str(r["step_name"]) if r["step_name"] is not None else None,
                # action 使用 loops.action。
                action=str(r["action"]) if r["action"] is not None else None,
                workspace_id=str(r["workspace_id"]),
                workspace_path=str(r["workspace_path"]),
            )
        )
    return out


def _find_yaml_templates(ws_root: Path) -> list[str]:
    paths: list[str] = []
    try:
        for p in ws_root.rglob("*.yaml"):
            try:
                paths.append(str(p.relative_to(ws_root)))
            except Exception:
                paths.append(p.name)
        for p in ws_root.rglob("*.yml"):
            try:
                paths.append(str(p.relative_to(ws_root)))
            except Exception:
                paths.append(p.name)
    except Exception:
        return []
    # 去重并排序，保证稳定性
    return sorted(sorted(set(paths)))


def _stable_strategy_id(*, scenario: str | None, step_name: str | None, action: str | None, templates: list[str]) -> str:
    base = {
        "scenario": scenario or "",
        "step_name": step_name or "",
        "action": action or "",
        "templates": templates,
    }
    key = json.dumps(base, sort_keys=True, ensure_ascii=False)
    return uuid.uuid5(_STRATEGY_NS, key).hex


def _load_yaml_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_strategy_struct(ws_root: Path, templates: list[str]) -> dict[str, Any]:
    """Best-effort extraction of structured strategy config from YAML templates.

    不对字段做精简：
    - 直接透传 YAML 中的 data_handler_config / port_analysis_config / task 下的 model/dataset/record 等子树；
    - 仅做轻量重组，便于 AIstock 侧消费。
    """

    data_config: dict[str, Any] | None = None
    portfolio_config: dict[str, Any] | None = None
    backtest_config: dict[str, Any] | None = None
    model_config: dict[str, Any] | None = None
    dataset_config: dict[str, Any] | None = None

    for rel in templates:
        cfg_path = ws_root / rel
        cfg = _load_yaml_if_exists(cfg_path)
        if not cfg:
            continue

        # data_handler_config 直接透传
        dh = cfg.get("data_handler_config")
        if isinstance(dh, dict) and data_config is None:
            data_config = dh

        # port_analysis_config 下的 strategy/backtest
        pac = cfg.get("port_analysis_config")
        if isinstance(pac, dict):
            if portfolio_config is None and isinstance(pac.get("strategy"), dict):
                portfolio_config = pac["strategy"]
            if backtest_config is None and isinstance(pac.get("backtest"), dict):
                backtest_config = pac["backtest"]

        task = cfg.get("task")
        if isinstance(task, dict):
            if model_config is None and isinstance(task.get("model"), dict):
                model_config = task["model"]
            if dataset_config is None and isinstance(task.get("dataset"), dict):
                dataset_config = task["dataset"]

        # 如果主要字段都已经找到，可以提前结束
        if data_config and portfolio_config and backtest_config and model_config and dataset_config:
            break

    result: dict[str, Any] = {}
    if data_config is not None:
        result["data_config"] = data_config
    if dataset_config is not None:
        result["dataset_config"] = dataset_config
    if portfolio_config is not None:
        result["portfolio_config"] = portfolio_config
    if backtest_config is not None:
        result["backtest_config"] = backtest_config
    if model_config is not None:
        result["model_config"] = model_config

    # 衍生字段：feature_list / market / instruments / freq
    feature_list: list[str] = []
    # 1) 优先从 data_config.infer_processors 中抽取 FilterCol.kwargs.col_list
    try:
        dc = data_config or {}
        infer_procs = dc.get("infer_processors") or []
        if isinstance(infer_procs, list):
            for proc in infer_procs:
                if not isinstance(proc, dict):
                    continue
                if proc.get("class") != "FilterCol":
                    continue
                kwargs = proc.get("kwargs") or {}
                cols = kwargs.get("col_list") or []
                if isinstance(cols, list):
                    for c in cols:
                        if isinstance(c, str) and c not in feature_list:
                            feature_list.append(c)
    except Exception:
        pass

    # 若 data_config 未命中，再从 dataset_config.kwargs.handler.kwargs.infer_processors 中尝试
    if not feature_list and dataset_config is not None:
        try:
            ds_kwargs = dataset_config.get("kwargs") or {}
            handler = ds_kwargs.get("handler") or {}
            h_kwargs = handler.get("kwargs") or {}
            infer_procs = h_kwargs.get("infer_processors") or []
            if isinstance(infer_procs, list):
                for proc in infer_procs:
                    if not isinstance(proc, dict):
                        continue
                    if proc.get("class") != "FilterCol":
                        continue
                    kwargs = proc.get("kwargs") or {}
                    cols = kwargs.get("col_list") or []
                    if isinstance(cols, list):
                        for c in cols:
                            if isinstance(c, str) and c not in feature_list:
                                feature_list.append(c)
        except Exception:
            pass

    if feature_list:
        result["feature_list"] = feature_list

    # 衍生 market / instruments / freq（freq 暂按 Phase2 约定默认为 1d）
    market: str | None = None
    instruments: Any | None = None
    freq: str | None = "1d"

    # instruments 优先从 dataset_config.handler.kwargs.instruments 获取，其次 data_config.instruments
    try:
        if dataset_config is not None:
            ds_kwargs = dataset_config.get("kwargs") or {}
            handler = ds_kwargs.get("handler") or {}
            h_kwargs = handler.get("kwargs") or {}
            instruments = h_kwargs.get("instruments")
    except Exception:
        instruments = None

    if instruments is None and data_config is not None:
        instruments = data_config.get("instruments")

    # 简单派生 market：若 instruments 是字符串，例如 "all" 或 指数代码，则原样暴露给 AIstock，后续由 AIstock 侧做精细映射
    if isinstance(instruments, str):
        market = instruments

    if market is not None:
        result["market"] = market
    if instruments is not None:
        result["instruments"] = instruments
    if freq is not None:
        result["freq"] = freq

    return result


def _to_native_path(p_str: str) -> Path:
    """Convert path between WSL and Windows format based on current OS."""
    import os
    if not p_str:
        return Path()
    is_windows = os.name == "nt"
    if is_windows and p_str.startswith("/mnt/"):
        parts = p_str.split("/")
        if len(parts) < 3:
            return Path(p_str)
        drive = parts[2].upper()
        return Path(f"{drive}:\\") / Path(*parts[3:])
    elif not is_windows and len(p_str) > 1 and p_str[1] == ":" and p_str[2] == "\\":
        drive = p_str[0].lower()
        rel = p_str[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}") / rel
    return Path(p_str)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run(registry_sqlite: Path, output_path: Path) -> None:
    conn = sqlite3.connect(str(registry_sqlite))
    conn.row_factory = sqlite3.Row
    try:
        sources = _fetch_strategy_sources(conn)
    finally:
        conn.close()

    strategies_by_id: dict[str, dict[str, Any]] = {}

    for src in sources:
        ws_root = _to_native_path(src.workspace_path)
        if not ws_root.exists():
            continue

        templates = _find_yaml_templates(ws_root)
        strategy_id = _stable_strategy_id(
            scenario=src.scenario,
            step_name=src.step_name,
            action=src.action,
            templates=templates,
        )

        if strategy_id in strategies_by_id:
            continue

        struct = _extract_strategy_struct(ws_root, templates)

        # REQ-STRATEGY-P3-001: Load Python implementation info from strategy_meta.json
        strategy_meta = _load_json_if_exists(ws_root / "strategy_meta.json")
        impl_info = {}
        if strategy_meta:
            impl_info = {
                "impl_module": strategy_meta.get("impl_module"),
                "impl_func": strategy_meta.get("impl_func"),
                "strategy_name": strategy_meta.get("strategy_name")
            }

        entry: dict[str, Any] = {
            "strategy_id": strategy_id,
            "scenario": src.scenario,
            "step_name": src.step_name,
            "action": src.action,
            "workspace_example": {
                "task_run_id": src.task_run_id,
                "loop_id": src.loop_id,
                "workspace_id": src.workspace_id,
                "workspace_path": src.workspace_path,
            },
            "template_files": templates,
            "python_implementation": impl_info,
        }
        # 将解析出的结构化字段直接挂载，避免精简
        entry.update(struct)

        strategies_by_id[strategy_id] = entry

    strategies = [strategies_by_id[k] for k in sorted(strategies_by_id.keys())]

    payload = {
        "version": "v1",
        "generated_at_utc": _utc_now_iso(),
        "source": "rdagent_tools",
        "strategies": strategies,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AIstock-facing strategy catalog as JSON.")
    parser.add_argument(
        "--registry-sqlite",
        required=True,
        help="Path to registry.sqlite (WSL/Linux path).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for strategy_catalog.",
    )
    args = parser.parse_args()

    registry_sqlite = Path(args.registry_sqlite)
    if not registry_sqlite.exists():
        raise SystemExit(f"registry.sqlite not found: {registry_sqlite}")

    output_path = Path(args.output)

    run(registry_sqlite=registry_sqlite, output_path=output_path)


if __name__ == "__main__":
    main()
