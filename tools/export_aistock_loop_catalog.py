import argparse
import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@dataclass
class LoopRow:
    task_run_id: str
    loop_id: int
    scenario: str | None
    step_name: str | None
    action: str | None
    status: str | None
    has_result: bool
    workspace_id: str
    workspace_path: str
    log_trace_path: str | None
    log_dir: str | None
    materialization_status: str | None
    materialization_error: str | None
    asset_bundle_id: str | None
    is_solidified: int | None
    sync_status: str | None


_STRATEGY_NS = uuid.UUID("7b6b2c6c-0b4d-4e2c-9b7f-3b1b9b4f3a11")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fetch_loops(conn: sqlite3.Connection, limit: int | None = None) -> list[LoopRow]:
    sql = """
    SELECT
      l.task_run_id,
      l.loop_id,
      l.action,
      l.status,
      l.has_result,
      l.log_dir,
      l.materialization_status,
      l.materialization_error,
      l.asset_bundle_id,
      l.is_solidified,
      l.sync_status,
      w.step_name,
      w.workspace_id,
      w.workspace_path,
      tr.log_trace_path
    FROM loops l
    JOIN workspaces w
      ON l.task_run_id = w.task_run_id
     AND l.loop_id = w.loop_id
    LEFT JOIN task_runs tr
      ON l.task_run_id = tr.task_run_id
    WHERE
      (l.has_result = 1 OR l.has_result = '1')
      AND (l.is_solidified = 1 OR l.is_solidified = '1')
      AND l.asset_bundle_id IS NOT NULL
      AND l.asset_bundle_id != ''
    ORDER BY l.started_at_utc DESC
    """
    params: tuple[Any, ...] = ()
    if limit is not None and limit > 0:
        sql += "\n    LIMIT ?"
        params = (limit,)

    rows = conn.execute(sql, params).fetchall()
    out: list[LoopRow] = []
    for r in rows:
        out.append(
            LoopRow(
                task_run_id=str(r["task_run_id"]),
                loop_id=int(r["loop_id"]),
                # loops 表没有 scenario，保持为 None。
                scenario=None,
                # workspaces.step_name
                step_name=str(r["step_name"]) if r["step_name"] is not None else None,
                # loops.action
                action=str(r["action"]) if r["action"] is not None else None,
                status=str(r["status"]) if r["status"] is not None else None,
                has_result=bool(r["has_result"] in (1, "1", True)),
                workspace_id=str(r["workspace_id"]),
                workspace_path=str(r["workspace_path"]),
                log_trace_path=str(r["log_trace_path"]) if r["log_trace_path"] is not None else None,
                log_dir=str(r["log_dir"]) if r["log_dir"] is not None else None,
                materialization_status=str(r["materialization_status"]) if r["materialization_status"] is not None else None,
                materialization_error=str(r["materialization_error"]) if r["materialization_error"] is not None else None,
                asset_bundle_id=str(r["asset_bundle_id"]) if r["asset_bundle_id"] is not None else None,
                is_solidified=int(r["is_solidified"]) if r["is_solidified"] is not None else None,
                sync_status=str(r["sync_status"]) if r["sync_status"] is not None else None,
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


def _build_loop_entry(loop: LoopRow) -> dict[str, Any] | None:
    ws_root = _to_native_path(loop.workspace_path)
    if not ws_root.exists():
        return None

    # REQ-LOOP-VALIDATION: 验证Loop的核心资产完整性
    # Loop必须具备以下核心资产才能在AIstock侧执行选股和模拟盘运行
    # 1. 模型权重（必需）
    model_files: list[Path] = []
    for name in ("model.pkl", "model.joblib", "model.bin", "model.onnx", "model.pt", "model.pth"):
        p = ws_root / name
        if p.exists():
            model_files.append(p)
    
    # 检查mlruns目录中的params.pkl（对于LGBModel等qlib模型）
    mlruns_dir = ws_root / "mlruns"
    if not model_files and mlruns_dir.exists():
        params_pkl_files = list(mlruns_dir.rglob("params.pkl"))
        if params_pkl_files:
            model_files = params_pkl_files
    
    if not model_files:
        # 缺少模型权重，不导出该Loop
        return None
    
    # 2. 配置文件（必需）
    yaml_files = list(ws_root.glob("*.yaml")) + list(ws_root.glob("*.yml"))
    if not yaml_files:
        # 缺少配置文件，不导出该Loop
        return None
    
    # 3. Python代码（必需）
    py_files = list(ws_root.glob("*.py"))
    if not py_files:
        # 缺少Python代码，不导出该Loop
        return None
    
    # 4. 因子列表（必需）
    factor_names: list[str] = []
    factor_perf_path = ws_root / "factor_perf.json"
    factor_perf = _load_json_if_exists(factor_perf_path) or {}
    try:
        combos = factor_perf.get("combinations", [])
        if isinstance(combos, list):
            for combo in combos:
                names = combo.get("factor_names") or []
                if isinstance(names, list):
                    for n in names:
                        if isinstance(n, str) and n not in factor_names:
                            factor_names.append(n)
    except Exception:
        factor_names = []
    
    if not factor_names:
        # 缺少因子列表，不导出该Loop
        return None
    
    # 5. 策略ID（必需）
    templates = _find_yaml_templates(ws_root)
    if not templates:
        # 缺少策略ID，不导出该Loop
        return None
    
    factor_meta_path = ws_root / "factor_meta.json"
    feedback_path = ws_root / "feedback.json"
    ret_curve_path = ws_root / "ret_curve.png"
    dd_curve_path = ws_root / "dd_curve.png"
    # 常见模型与训练目录
    mlruns_dir = ws_root / "mlruns"

    feedback = _load_json_if_exists(feedback_path) or {}

    # Extract main window metrics best-effort，并将关键字段结构化挂在 loop 上。
    metrics: dict[str, Any] = {}
    annualized_return: Any | None = None
    max_drawdown: Any | None = None
    sharpe: Any | None = None
    ic: Any | None = None
    ic_ir: Any | None = None
    win_rate: Any | None = None

    try:
        windows = []
        combos = factor_perf.get("combinations", [])
        if isinstance(combos, list) and combos:
            first_combo = combos[0]
            windows = first_combo.get("windows", []) or []
        if isinstance(windows, list) and windows:
            main_w = windows[0]

            # 直接挂在 window 上的关键字段
            if "annual_return" in main_w:
                annualized_return = main_w.get("annual_return")
            if "max_drawdown" in main_w:
                max_drawdown = main_w.get("max_drawdown")
            if "sharpe" in main_w:
                sharpe = main_w.get("sharpe")

            m = main_w.get("metrics", {}) or {}
            if isinstance(m, dict):
                metrics = m

                # 兼容两类结构：
                # 1) {"1day...annualized_return": 0.87}
                # 2) {"Unnamed: 0": "1day...annualized_return", "0": 0.87}

                # annualized_return：优先查找 key 本身包含 "annualized_return" 的情形
                if annualized_return is None:
                    for k, v in m.items():
                        if isinstance(k, str) and "annualized_return" in k:
                            annualized_return = v
                            break

                # 若上述方式未命中，再尝试 pandas DataFrame to_json 的占位结构
                if annualized_return is None:
                    metric_name = m.get("Unnamed: 0")
                    metric_value = m.get("0")
                    if (
                        isinstance(metric_name, str)
                        and "annualized_return" in metric_name
                        and metric_value is not None
                    ):
                        annualized_return = metric_value

                # IC / IC IR / 胜率等：根据常见命名约定做 best-effort 抽取
                for k, v in m.items():
                    if not isinstance(k, str):
                        continue
                    lk = k.lower()
                    # 优先匹配大写IC或小写ic
                    if ic is None and (k == "IC" or k == "ic" or lk == "ic"):
                        ic = v
                    if ic_ir is None and (k == "ICIR" or k == "ic_ir" or lk == "icir" or lk == "ic ir"):
                        ic_ir = v
                    if win_rate is None and ("win_rate" in lk or "winrate" in lk or "hit_ratio" in lk):
                        win_rate = v
    except Exception:
        metrics = {}

    # Extract decision and summary texts from feedback.
    decision = feedback.get("decision")
    summary = feedback.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    execution = summary.get("execution")
    value_feedback = summary.get("value_feedback")
    shape_feedback = summary.get("shape_feedback")

    # Compute strategy_id using same rule as export_aistock_strategy_catalog.
    strategy_id: str | None = None
    try:
        templates = _find_yaml_templates(ws_root)
        if templates:
            strategy_id = _stable_strategy_id(
                scenario=loop.scenario,
                step_name=loop.step_name,
                action=loop.action,
                templates=templates,
            )
    except Exception:
        strategy_id = None

    paths: dict[str, str] = {}
    if factor_meta_path.exists():
        paths["factor_meta"] = factor_meta_path.name
    if factor_perf_path.exists():
        paths["factor_perf"] = factor_perf_path.name
    if feedback_path.exists():
        paths["feedback"] = feedback_path.name
    if ret_curve_path.exists():
        paths["ret_curve"] = ret_curve_path.name
    if dd_curve_path.exists():
        paths["dd_curve"] = dd_curve_path.name
    if mlruns_dir.exists():
        paths["mlruns"] = mlruns_dir.name
    if model_files:
        # 仅记录文件名列表，AIstock 可在 workspace 根目录下查找
        paths["model_files"] = [p.name for p in model_files]

    # log 目录根路径：优先使用 loops.log_dir，若缺失则回退到 task_runs.log_trace_path
    log_dir = loop.log_dir or loop.log_trace_path

    return {
        "task_run_id": loop.task_run_id,
        "loop_id": loop.loop_id,
        "workspace_id": loop.workspace_id,
        # 绝对 workspace 根路径，方便 AIstock 直接定位 YAML / 配置文件
        "workspace_path": loop.workspace_path,
        "scenario": loop.scenario,
        "step_name": loop.step_name,
        "action": loop.action,
        "status": loop.status,
        "has_result": loop.has_result,
        "log_dir": log_dir,
        "materialization_status": loop.materialization_status,
        "materialization_error": loop.materialization_error,
        "asset_bundle_id": loop.asset_bundle_id,
        "is_solidified": bool(loop.is_solidified) if loop.is_solidified is not None else False,
        "sync_status": loop.sync_status or "pending",
        "strategy_id": strategy_id,
        "factor_names": factor_names,
        # 回测关键绩效指标（仅在 loop 层结构化存储）
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "ic": ic,
        "ic_ir": ic_ir,
        "win_rate": win_rate,
        "metrics": metrics,
        "decision": decision,
        "summary_texts": {
            "execution": execution,
            "value_feedback": value_feedback,
            "shape_feedback": shape_feedback,
        },
        "paths": paths,
    }


def run(registry_sqlite: Path, output_path: Path, *, limit: int | None = None) -> None:
    conn = sqlite3.connect(str(registry_sqlite))
    conn.row_factory = sqlite3.Row
    try:
        loops = _fetch_loops(conn, limit=limit)
    finally:
        conn.close()

    items: list[dict[str, Any]] = []
    for loop in loops:
        entry = _build_loop_entry(loop)
        if entry is not None:
            items.append(entry)

    payload = {
        "version": "v1",
        "generated_at_utc": _utc_now_iso(),
        "source": "rdagent_tools",
        "loops": items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AIstock-facing loop/backtest catalog as JSON.")
    parser.add_argument(
        "--registry-sqlite",
        required=True,
        help="Path to registry.sqlite (WSL/Linux path).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for loop_catalog.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for number of latest loops; 0 or negative means no explicit limit.",
    )
    args = parser.parse_args()

    registry_sqlite = Path(args.registry_sqlite)
    if not registry_sqlite.exists():
        raise SystemExit(f"registry.sqlite not found: {registry_sqlite}")

    output_path = Path(args.output)

    limit: int | None = None
    if args.limit and args.limit > 0:
        limit = args.limit

    run(registry_sqlite=registry_sqlite, output_path=output_path, limit=limit)


if __name__ == "__main__":
    main()
