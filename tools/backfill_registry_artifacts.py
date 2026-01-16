import argparse
import json
import os
import sys
import sqlite3
import uuid
import yaml
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rdagent.core.proposal import HypothesisFeedback
from rdagent.log.storage import FileStorage
from rdagent.log.utils import extract_loopid_func_name
from rdagent.components.coder.factor_coder.factor import FactorTask

def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _extract_alpha158_factors_from_conf(conf: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract Alpha158 factor definitions from Qlib config."""
    dh_cfg = conf.get("data_handler_config") or {}
    if not isinstance(dh_cfg, dict):
        return []

    alpha_cfg: dict[str, Any] | None = None
    direct_alpha = dh_cfg.get("alpha158_config")
    if isinstance(direct_alpha, dict):
        alpha_cfg = direct_alpha
    else:
        data_loader = dh_cfg.get("data_loader") or {}
        if isinstance(data_loader, dict):
            kwargs = data_loader.get("kwargs") or {}
            if isinstance(kwargs, dict):
                nested_alpha = kwargs.get("alpha158_config")
                if isinstance(nested_alpha, dict):
                    alpha_cfg = nested_alpha

    if not isinstance(alpha_cfg, dict):
        return []

    feature = alpha_cfg.get("feature") or []
    if not isinstance(feature, list) or len(feature) < 2:
        return []

    expr_list = feature[0] or []
    name_list = feature[1] or []
    if not (isinstance(expr_list, list) and isinstance(name_list, list)):
        return []

    n = min(len(expr_list), len(name_list))
    factors: list[dict[str, Any]] = []
    region = conf.get("qlib_init", {}).get("region", "cn")
    for i in range(n):
        expr = expr_list[i]
        name = name_list[i]
        if not isinstance(expr, str) or not isinstance(name, str):
            continue
        factors.append({
            "name": name,
            "expression": expr,
            "source": "qlib_alpha158",
            "region": region,
            "tags": ["alpha158"],
        })
    return factors

from rdagent.utils.artifacts_writer import (
    _build_factor_meta_dict,
    _build_factor_perf_from_metrics,
    _build_feedback_dict,
    _extract_model_metadata_from_workspace,
)


@dataclass
class WorkspaceRow:
    workspace_id: str
    task_run_id: str
    loop_id: int
    workspace_role: str
    experiment_type: str
    step_name: str | None
    status: str | None
    workspace_path: str
    meta_path: str | None
    summary_path: str | None
    manifest_path: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_ARTIFACT_NS = uuid.UUID("3e36bb8a-2a9b-4c56-8f50-6312f0d89c42")


def _stable_artifact_id(*, task_run_id: str, workspace_id: str, artifact_type: str, name: str) -> str:
    key = f"{task_run_id}|{workspace_id}|{artifact_type}|{name}"
    return uuid.uuid5(_ARTIFACT_NS, key).hex


def _stable_file_id(*, artifact_id: str, workspace_id: str, path_rel: str, kind: str) -> str:
    key = f"{artifact_id}|{workspace_id}|{kind}|{path_rel}"
    return uuid.uuid5(_ARTIFACT_NS, key).hex


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _rel(ws_root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(ws_root))
    except Exception:
        return str(p)


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _best_effort_file_meta(p: Path) -> tuple[int | None, str | None]:
    try:
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        return int(st.st_size), mtime
    except Exception:
        return None, None


def _save_ret_plots(ret_pkl: Path, ws_root: Path) -> list[Path]:
    """Deprecated: AIstock no longer needs return plots."""
    return []


def _iter_yaml_confs(ws_root: Path) -> list[Path]:
    out = []
    try:
        out.extend(list(ws_root.glob("*.yaml")))
        out.extend(list(ws_root.glob("*.yml")))
    except Exception:
        return []
    return [p for p in out if p.exists()]


def _read_loop_action(cur: sqlite3.Cursor, task_run_id: str, loop_id: int) -> str | None:
    try:
        row = cur.execute(
            "select action from loops where task_run_id=? and loop_id=?",
            (task_run_id, loop_id),
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _read_materialization_status(cur: sqlite3.Cursor, task_run_id: str, loop_id: int) -> str | None:
    """Read materialization_status from loops table (best-effort)."""
    try:
        row = cur.execute(
            "select materialization_status from loops where task_run_id=? and loop_id=?",
            (task_run_id, loop_id),
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _update_loop_log_dir(cur: sqlite3.Cursor, task_run_id: str, loop_id: int, log_dir: str) -> None:
    try:
        cur.execute(
            "UPDATE loops SET log_dir=? WHERE task_run_id=? AND loop_id=? AND (log_dir IS NULL OR log_dir='')",
            (log_dir, task_run_id, loop_id),
        )
    except Exception:
        pass


def _update_workspace_paths(
    cur: sqlite3.Cursor,
    ws: WorkspaceRow,
    meta_rel: str,
    summary_rel: str,
    manifest_rel: str,
) -> None:
    cur.execute(
        """
        update workspaces
        set meta_path=?, summary_path=?, manifest_path=?, updated_at_utc=?
        where workspace_id=?
        """,
        (meta_rel, summary_rel, manifest_rel, _utc_now_iso(), ws.workspace_id),
    )


def _insert_artifact(
    cur: sqlite3.Cursor,
    *,
    artifact_id: str,
    ws: WorkspaceRow,
    artifact_type: str,
    name: str,
    status: str,
    primary_flag: bool,
    summary: dict[str, Any],
    entry_path: str,
    model_type: str | None = None,
    model_conf: dict[str, Any] | None = None,
    dataset_conf: dict[str, Any] | None = None,
    feature_schema: dict[str, Any] | None = None,
) -> None:
    cur.execute(
        """
        insert or replace into artifacts
        (artifact_id, task_run_id, loop_id, workspace_id, artifact_type, name, version, status, primary_flag,
         summary_json, entry_path, model_type, model_conf_json, dataset_conf_json, feature_schema_json,
         created_at_utc, updated_at_utc)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact_id,
            ws.task_run_id,
            ws.loop_id,
            ws.workspace_id,
            artifact_type,
            name,
            "v1",
            status,
            1 if primary_flag else 0,
            json.dumps(summary, ensure_ascii=False, default=str),
            entry_path,
            model_type,
            json.dumps(model_conf, ensure_ascii=False, default=str) if model_conf else None,
            json.dumps(dataset_conf, ensure_ascii=False, default=str) if dataset_conf else None,
            json.dumps(feature_schema, ensure_ascii=False, default=str) if feature_schema else None,
            _utc_now_iso(),
            _utc_now_iso(),
        ),
    )


def _insert_artifact_file(
    cur: sqlite3.Cursor,
    *,
    artifact_id: str,
    ws: WorkspaceRow,
    path_rel: str,
    kind: str,
    size_bytes: int | None,
    mtime_utc: str | None,
) -> None:
    cur.execute(
        """
        insert or replace into artifact_files
        (file_id, artifact_id, workspace_id, path, sha256, size_bytes, mtime_utc, kind)
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _stable_file_id(
                artifact_id=artifact_id,
                workspace_id=ws.workspace_id,
                path_rel=path_rel,
                kind=kind,
            ),
            artifact_id,
            ws.workspace_id,
            path_rel,
            None,
            size_bytes,
            mtime_utc,
            kind,
        ),
    )


def _cleanup_existing_for_workspace(
    cur: sqlite3.Cursor,
    ws: WorkspaceRow,
    *,
    artifact_types: Iterable[str],
) -> None:
    # Remove previously backfilled artifacts for this workspace to avoid duplicates from older runs.
    # We restrict by workspace_id + task_run_id + artifact_type.
    types = list(artifact_types)
    if not types:
        return
    q_marks = ",".join(["?"] * len(types))
    artifact_ids = [
        r[0]
        for r in cur.execute(
            f"select artifact_id from artifacts where task_run_id=? and workspace_id=? and artifact_type in ({q_marks})",
            (ws.task_run_id, ws.workspace_id, *types),
        ).fetchall()
    ]
    if not artifact_ids:
        return
    q_marks2 = ",".join(["?"] * len(artifact_ids))
    cur.execute(
        f"delete from artifact_files where artifact_id in ({q_marks2})",
        artifact_ids,
    )
    cur.execute(
        f"delete from artifacts where artifact_id in ({q_marks2})",
        artifact_ids,
    )


def _check_phase2_artifacts_for_workspace(cur: sqlite3.Cursor, ws: WorkspaceRow) -> dict[str, Any]:
    """只读检查：当前 workspace 是否具备 Phase 2 需要的 JSON / 图表及其 registry 记录。

    注意：
    - 只执行文件存在性检查与只读 SQL 查询；
    - 不写任何 JSON、不插入/更新/删除任何 DB 记录；
    - 返回结构用于在 CLI 中汇总展示缺失情况。
    """

    ws_root = Path(ws.workspace_path)
    if not ws_root.exists():
        return {
            "workspace_id": ws.workspace_id,
            "loop_id": ws.loop_id,
            "workspace_role": ws.workspace_role,
            "skipped": True,
            "reason": "workspace_path_not_found",
        }

    # 基础文件探测
    factor_meta = ws_root / "factor_meta.json"
    factor_perf = ws_root / "factor_perf.json"
    feedback = ws_root / "feedback.json"
    ret_curve = ws_root / "ret_curve.png"
    dd_curve = ws_root / "dd_curve.png"

    # 复用 action/has_result 判定逻辑
    qlib_res = ws_root / "qlib_res.csv"
    ret_pkl = ws_root / "ret.pkl"
    combined_factors = ws_root / "combined_factors_df.parquet"

    action = ws.experiment_type or _read_loop_action(cur, ws.task_run_id, ws.loop_id) or "unknown"
    if action == "model":
        has_result = bool(qlib_res.exists() and ret_pkl.exists())
    elif action == "factor":
        has_result = bool(combined_factors.exists())
    else:
        has_result = False

    def _count_artifact_files_by_path(rel_path: str) -> int:
        try:
            row = cur.execute(
                """
                select count(*)
                from artifact_files f
                join artifacts a on a.artifact_id = f.artifact_id
                where a.workspace_id=? and f.path=?
                """,
                (ws.workspace_id, rel_path),
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _file_info(p: Path) -> dict[str, Any]:
        exists = p.exists()
        rel = _rel(ws_root, p) if exists else None
        count = _count_artifact_files_by_path(rel) if rel is not None else 0
        return {
            "exists": bool(exists),
            "rel_path": rel,
            "registry_file_records": count,
        }

    files_info = {
        "factor_meta.json": _file_info(factor_meta),
        "factor_perf.json": _file_info(factor_perf),
        "feedback.json": _file_info(feedback),
        "ret_curve.png": _file_info(ret_curve),
        "dd_curve.png": _file_info(dd_curve),
    }

    return {
        "workspace_id": ws.workspace_id,
        "task_run_id": ws.task_run_id,
        "loop_id": ws.loop_id,
        "workspace_role": ws.workspace_role,
        "experiment_type": ws.experiment_type,
        "action": action,
        "has_result": has_result,
        "files": files_info,
    }


def _collect_log_session_loops(log_path: Path) -> dict[int, dict[str, Any]]:
    """从 log 目录中收集每个 loop 的关键信息（因子任务 + HypothesisFeedback 决策）。

    返回结构：
    {
      loop_id: {
        "factor_items": [ {name/description/formulation/variables...}, ... ],
        "feedback": {"decision": bool | None, ...},
      },
      ...
    }
    """

    loops: dict[int, dict[str, Any]] = {}

    if not log_path.exists():
        return loops

    for msg in FileStorage(log_path).iter_msg():
        li_str, _fn = extract_loopid_func_name(msg.tag)
        if li_str is None:
            continue
        try:
            li = int(li_str)
        except Exception:
            continue

        info = loops.setdefault(li, {"factor_items": [], "feedback": None, "workspace_paths": set()})

        # 因子任务列表：list[FactorTask]
        try:
            if isinstance(msg.content, list) and msg.content:
                first = msg.content[0]
                if isinstance(first, FactorTask):
                    items: list[dict[str, Any]] = []
                    for t in msg.content:  # type: ignore[assignment]
                        if not isinstance(t, FactorTask):
                            continue
                        items.append(
                            {
                                "name": t.factor_name,
                                "source": "rdagent_generated",
                                "description_cn": t.factor_description or "",
                                "formula_hint": t.factor_formulation or "",
                                "tags": [],
                                "variables": getattr(t, "variables", {}),
                            }
                        )
                    # 覆盖为该 loop 最新一批因子任务
                    if items:
                        info["factor_items"] = items
        except Exception:
            # 解析失败不影响其他 loop
            pass

        # HypothesisFeedback / ExperimentFeedback
        try:
            if isinstance(msg.content, HypothesisFeedback):
                hf: HypothesisFeedback = msg.content
                fb_dict = {
                    "decision": getattr(hf, "decision", None),
                    "observations": getattr(hf, "observations", "") or "",
                    "hypothesis_evaluation": getattr(hf, "hypothesis_evaluation", "") or "",
                    "new_hypothesis": getattr(hf, "new_hypothesis", "") or "",
                    "reason": getattr(hf, "reason", "") or "",
                    "exception": getattr(hf, "exception", None),
                }
                # 以最后一次 feedback 为准
                info["feedback"] = fb_dict
        except Exception:
            pass

        # 记录 workspace 路径：优先从 experiment.experiment_workspace.workspace_path 获取
        try:
            obj = msg.content
            ew = getattr(obj, "experiment_workspace", None)
            ws_path = None
            if ew is not None:
                ws_path = getattr(ew, "workspace_path", None)
            if ws_path is None:
                ws_path = getattr(obj, "workspace_path", None)
            if ws_path:
                try:
                    ws_str = str(ws_path)
                    if ws_str:
                        info["workspace_paths"].add(ws_str)
                except Exception:
                    pass
        except Exception:
            pass

    return loops


def _enrich_existing_factor_meta(
    ws_root: Path,
    *,
    factor_items_from_logs: list[dict[str, Any]] | None,
    dry_run: bool,
) -> None:
    """Enrich existing factor_meta.json with factor information parsed from logs.

    目的：
    - 在不整体重写 factor_meta.json 的前提下，利用 log 中解析出的因子任务信息，
      为已有的因子条目补全缺失的 `description_cn` / `formula_hint` / `variables` 等字段；
    - 保持幂等：只在原值为 None/空字符串/空 dict 时才写入新值；
    - dry_run=True 时仅执行差异计算，不写盘。
    """

    if not factor_items_from_logs:
        return

    factor_meta_path = ws_root / "factor_meta.json"
    if not factor_meta_path.exists():
        # 若文件不存在，则由正常的 factor_meta 生成逻辑负责创建
        return

    try:
        with factor_meta_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return

    factors = payload.get("factors") or []
    if not isinstance(factors, list):
        return

    # 将现有因子按 name 建索引，便于用 log 中信息补充
    by_name: dict[str, dict[str, Any]] = {}
    for item in factors:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            by_name[name] = item

    changed = False

    for log_item in factor_items_from_logs:
        if not isinstance(log_item, dict):
            continue
        name = log_item.get("name")
        if not isinstance(name, str) or not name:
            continue

        target = by_name.get(name)
        if target is None:
            # 若现有 factor_meta 中不存在该因子，则追加一条完整记录
            new_entry: dict[str, Any] = {
                "name": name,
            }
            for key in [
                "source",
                "description_cn",
                "formula_hint",
                "tags",
                "variables",
            ]:
                if key in log_item:
                    new_entry[key] = log_item[key]

            # 为新增条目设置默认频率/对齐/NaN 策略
            new_entry.setdefault("freq", "1d")
            new_entry.setdefault("align", "close")
            new_entry.setdefault("nan_policy", "dataservice_default")
            factors.append(new_entry)
            by_name[name] = new_entry
            changed = True
            continue

        # 仅在原值为空时，从 log 中补写描述/表达式/变量等信息
        for key in ["description_cn", "formula_hint", "variables"]:
            if key not in log_item:
                continue
            current_val = target.get(key)
            new_val = log_item.get(key)
            if current_val in (None, "", {}, []):
                if new_val not in (None, "", {}, []):
                    target[key] = new_val
                    changed = True

        # 在已有条目上补充默认频率/对齐/NaN 策略（仅在缺失时设置）
        if target.get("freq") in (None, ""):
            target["freq"] = "1d"
            changed = True
        if target.get("align") in (None, ""):
            target["align"] = "close"
            changed = True
        if target.get("nan_policy") in (None, ""):
            target["nan_policy"] = "dataservice_default"
            changed = True

    if not changed:
        return

    if dry_run:
        return

    try:
        with factor_meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        # 写失败时不影响主流程
        return


def _to_native_path(p_str: str) -> Path:
    """Convert path between WSL and Windows format based on current OS."""
    if not p_str:
        return Path()
    is_windows = os.name == "nt"
    # Normalize slashes first
    p_str = p_str.replace("/", os.sep).replace("\\", os.sep)
    
    if is_windows:
        # Handle WSL path: /mnt/f/... -> F:\...
        if p_str.lower().startswith(f"{os.sep}mnt{os.sep}"):
            parts = p_str.split(os.sep)
            if len(parts) >= 3:
                drive = parts[2].upper()
                return Path(f"{drive}:\\") / Path(*parts[3:])
        # Handle already Windows path
        if len(p_str) > 1 and p_str[1] == ":":
            return Path(p_str)
    else:
        # Handle Windows path in WSL: F:\... -> /mnt/f/...
        if len(p_str) > 1 and p_str[1] == ":":
            drive = p_str[0].lower()
            rel = p_str[3:].replace("\\", "/")
            return Path(f"/mnt/{drive}") / rel
            
    return Path(p_str)


def _iter_workspaces(cur: sqlite3.Cursor, task_run_id: str | None = None) -> Iterable[WorkspaceRow]:
    query = "SELECT workspace_id, task_run_id, loop_id, workspace_role, experiment_type, status, workspace_path, meta_path, summary_path, manifest_path FROM workspaces"
    params = []
    if task_run_id:
        query += " WHERE task_run_id = ?"
        params.append(task_run_id)

    rows = cur.execute(query, params).fetchall()
    for r in rows:
        # 路径转换
        ws_path = str(_to_native_path(r[6]))
        yield WorkspaceRow(
            workspace_id=r[0],
            task_run_id=r[1],
            loop_id=r[2],
            workspace_role=r[3],
            experiment_type=r[4],
            step_name=None,
            status=r[5],
            workspace_path=ws_path,
            meta_path=r[7],
            summary_path=r[8],
            manifest_path=r[9],
        )


def _maybe_backfill_phase2_for_workspace(
    cur: sqlite3.Cursor,
    ws: WorkspaceRow,
    *,
    overwrite_json: bool,
    dry_run: bool,
    factor_items_from_logs: list[dict[str, Any]] | None = None,
    feedback_from_logs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Best-effort Phase2 补录：为历史 workspace 生成 factor_meta/factor_perf/feedback/ret_curve 图表并登记。

    约定：
    - 仅在 has_result=True 的 workspace 上尝试补录；
    - 遵守 overwrite_json 标志：为 False 时，不覆盖已存在的 JSON 文件；
    - dry_run=True 时，仅返回计划执行的操作，不写任何文件/DB。
    """

    ws_root = Path(ws.workspace_path)
    if not ws_root.exists():
        return {
            "workspace_id": ws.workspace_id,
            "loop_id": ws.loop_id,
            "skipped": True,
            "reason": "workspace_path_not_found",
        }

    qlib_res = ws_root / "qlib_res.csv"
    ret_pkl = ws_root / "ret.pkl"
    combined_factors = ws_root / "combined_factors_df.parquet"

    action = ws.experiment_type or _read_loop_action(cur, ws.task_run_id, ws.loop_id) or "unknown"
    if action == "model":
        has_result = bool(qlib_res.exists() and ret_pkl.exists())
    elif action == "factor":
        has_result = bool(combined_factors.exists())
    else:
        has_result = False

    results: dict[str, Any] = {
        "workspace_id": ws.workspace_id,
        "loop_id": ws.loop_id,
        "action": action,
        "has_result": has_result,
        "generated_files": [],
        "registered_artifacts": [],
    }

    # ------------------------
    # Phase 3: Strategy / Model / Factor Interface (Even if has_result is False)
    # ------------------------
    
    # 1. Strategy pythonization
    if action in ("model", "factor", "unknown"):
        try:
            from rdagent.utils.artifacts_writer import _sync_strategy_impl_to_shared_lib
            strategy_meta = _sync_strategy_impl_to_shared_lib(ws_root=ws_root)
            if strategy_meta:
                strategy_meta_path = ws_root / "strategy_meta.json"
                _write_json(strategy_meta_path, strategy_meta, overwrite=overwrite_json)
                
                # Register as artifact
                artifact_id = _stable_artifact_id(
                    task_run_id=ws.task_run_id,
                    workspace_id=ws.workspace_id,
                    artifact_type="strategy_meta",
                    name="strategy_meta.json",
                )
                if not dry_run:
                    _insert_artifact(
                        cur,
                        artifact_id=artifact_id,
                        ws=ws,
                        artifact_type="strategy_meta",
                        name="strategy_meta.json",
                        status="present",
                        primary_flag=False,
                        summary={"file": "strategy_meta.json"},
                        entry_path=_rel(ws_root, strategy_meta_path),
                    )
                results["generated_files"].append("strategy_meta.json")
                results["registered_artifacts"].append("strategy_meta")
        except Exception:
            pass

    # 2. Model Metadata
    if action == "model":
        try:
            from rdagent.utils.artifacts_writer import _extract_model_metadata_from_workspace
            model_metadata = _extract_model_metadata_from_workspace(ws_root)
            if model_metadata:
                model_meta_path = ws_root / "model_meta.json"
                # Add basic info
                model_metadata["task_run_id"] = ws.task_run_id
                model_metadata["loop_id"] = ws.loop_id
                model_metadata["workspace_id"] = ws.workspace_id
                
                _write_json(model_meta_path, model_metadata, overwrite=overwrite_json)
                
                # Register as artifact
                artifact_id = _stable_artifact_id(
                    task_run_id=ws.task_run_id,
                    workspace_id=ws.workspace_id,
                    artifact_type="model_meta",
                    name="model_meta.json",
                )
                if not dry_run:
                    _insert_artifact(
                        cur,
                        artifact_id=artifact_id,
                        ws=ws,
                        artifact_type="model_meta",
                        name="model_meta.json",
                        status="present",
                        primary_flag=False,
                        summary={"file": "model_meta.json"},
                        entry_path=_rel(ws_root, model_meta_path),
                    )
                results["generated_files"].append("model_meta.json")
                results["registered_artifacts"].append("model_meta")
        except Exception:
            pass

    if not has_result:
        return results

    # ------------------------
    # Phase 2: Performance / Feedback (Only if has_result is True)
    # ------------------------

    # 尝试从 qlib_res.csv 中读取一行聚合指标，供 factor_perf / feedback 使用
    metrics: dict[str, Any] | None = None
    if qlib_res.exists():
        try:
            import csv
            with qlib_res.open("r", encoding="utf-8") as csv_f:
                reader = csv.DictReader(csv_f)
                rows = list(reader)
                if rows:
                    # Convert string values to float where possible
                    metrics = {}
                    for k, v in rows[0].items():
                        try:
                            metrics[k] = float(v)
                        except (ValueError, TypeError):
                            metrics[k] = v
        except Exception:
            metrics = None

    # ------------------------
    # ------------------------
    # factor_meta.json
    # ------------------------
    factor_meta_path = ws_root / "factor_meta.json"
    factor_items: list[dict[str, Any]] = []
    
    if factor_meta_path.exists():
        # 即使不重写，也先加载现有的以供后续性能关联使用
        existing_meta = _load_json_if_exists(factor_meta_path)
        if existing_meta and isinstance(existing_meta.get("factors"), list):
            factor_items = existing_meta["factors"]

    # REQ-FACTOR-VALIDATION: 验证rdagent_generated因子的数据完整性
    # rdagent_generated因子必须有factor_perf.json才能被补录
    factor_perf_path = ws_root / "factor_perf.json"
    factor_perf = _load_json_if_exists(factor_perf_path)
    
    # 检查是否有rdagent_generated因子
    has_rdagent_generated = False
    for f in factor_items:
        if isinstance(f, dict) and f.get("source") == "rdagent_generated":
            has_rdagent_generated = True
            break
    
    # 如果有rdagent_generated因子但没有factor_perf.json，跳过该workspace
    if has_rdagent_generated and not factor_perf:
        return {
            "workspace_id": ws.workspace_id,
            "loop_id": ws.loop_id,
            "skipped": True,
            "reason": "rdagent_generated_factors_missing_performance_data",
        }
    
    need_write_factor_meta = overwrite_json or (not factor_meta_path.exists())
    if need_write_factor_meta:
        # 1. 尝试从 Qlib YAML 配置中提取全量 Alpha158 因子
        if not factor_items:
            yaml_files = list(ws_root.glob("*.yaml")) + list(ws_root.glob("*.yml"))
            for yf in yaml_files:
                conf = _load_yaml(yf)
                if conf:
                    extracted = _extract_alpha158_factors_from_conf(conf)
                    if extracted:
                        factor_items = extracted
                        break

        # 2. 兜底逻辑：从 combined_factors_df.parquet 提取（不再是优先推荐方式）
        if not factor_items and combined_factors.exists():
            # 只要 parquet 存在，就尝试提取因子。
            # AIstock 需要全量因子（包括 Alpha158/360 等基础因子），因此不再根据回测结果过滤。
            try:
                import pyarrow.parquet as pq
                meta = pq.read_metadata(combined_factors)
                cols = meta.schema.names
                # 排除索引列（通常是 datetime, instrument 等，由具体的 parquet 结构决定）
                # 在 RD-Agent 的 combined_factors 中，通常列就是因子名。
                factor_items = []
                for name in cols:
                    if name in ("datetime", "instrument", "index", "level_0", "level_1"):
                        continue
                    is_alpha = any(x in name.lower() for x in ["alpha158", "alpha360"])
                    factor_items.append(
                        {
                            "name": name,
                            "source": "sota" if is_alpha else "rdagent_generated",
                            "region": "cn",
                            "description_cn": "",
                            "formula_hint": name if is_alpha else "",
                            "tags": ["alpha158"] if "alpha158" in name.lower() else (["alpha360"] if "alpha360" in name.lower() else []),
                            "freq": "1d",
                            "align": "close",
                            "nan_policy": "dataservice_default",
                        }
                    )
            except Exception:
                pass
        
        if factor_items:
            try:
                created_at_utc = _utc_now_iso()
                factor_meta_payload = _build_factor_meta_dict(
                    factors=factor_items,
                    task_run_id=ws.task_run_id,
                    loop_id=ws.loop_id,
                    created_at_utc=created_at_utc,
                )
            except Exception:
                factor_meta_payload = None

            if factor_meta_payload is not None:
                # 因子代码同步到共享库 (Phase 3 准备)
                try:
                    from rdagent.utils.artifacts_writer import _sync_factor_impl_to_shared_lib
                    # 补录时也可以尝试同步代码实现
                    factor_meta_payload = _sync_factor_impl_to_shared_lib(
                        ws_root=ws_root,
                        factor_meta_payload=factor_meta_payload,
                    ) or factor_meta_payload
                except Exception:
                    pass

                if not dry_run:
                    _write_json(factor_meta_path, factor_meta_payload, overwrite=overwrite_json)

                    artifact_id = _stable_artifact_id(
                        task_run_id=ws.task_run_id,
                        workspace_id=ws.workspace_id,
                        artifact_type="factor_meta",
                        name="factor_meta.json",
                    )
                    status = "present" if factor_meta_path.exists() else "missing"
                    _insert_artifact(
                        cur,
                        artifact_id=artifact_id,
                        ws=ws,
                        artifact_type="factor_meta",
                        name="factor_meta.json",
                        status=status,
                        primary_flag=False,
                        summary={"file": "factor_meta.json"},
                        entry_path=_rel(ws_root, factor_meta_path),
                    )
                    if factor_meta_path.exists():
                        size_bytes, mtime_utc = _best_effort_file_meta(factor_meta_path)
                        _insert_artifact_file(
                            cur,
                            artifact_id=artifact_id,
                            ws=ws,
                            path_rel=_rel(ws_root, factor_meta_path),
                            kind="factor_meta",
                            size_bytes=size_bytes,
                            mtime_utc=mtime_utc,
                        )

                results["generated_files"].append("factor_meta.json")
                results["registered_artifacts"].append("factor_meta")

    else:
        # 文件已存在且不允许整体覆盖时，尝试基于 log 信息做增量补齐
        _enrich_existing_factor_meta(
            ws_root,
            factor_items_from_logs=factor_items_from_logs,
            dry_run=dry_run,
        )

    # ------------------------
    # factor_perf.json
    # ------------------------
    factor_perf_path = ws_root / "factor_perf.json"
    need_write_factor_perf = overwrite_json or (not factor_perf_path.exists())
    if need_write_factor_perf and metrics is not None:
        try:
            # 对因子实验，尽量复用在线逻辑：基于 combined_factors_df.parquet 计算单因子描述统计
            per_factor_stats: list[dict[str, Any]] | None = None
            # Skip heavy per-factor descriptive stats computation to avoid OOM and save time.

            from rdagent.utils.artifacts_writer import _enrich_metrics_with_qlib_res, _build_factor_perf_from_metrics

            enriched_metrics = _enrich_metrics_with_qlib_res(ws_root, metrics or {})

            # 关联因子名称，以便 AIstock 侧能将指标与因子对应起来
            factor_names_list: list[str] = []
            if isinstance(factor_items, list):
                for f_item in factor_items:
                    if isinstance(f_item, dict):
                        f_name = f_item.get("name")
                        if isinstance(f_name, str) and f_name:
                            factor_names_list.append(f_name)

            factor_perf_payload = _build_factor_perf_from_metrics(
                metrics=enriched_metrics,
                task_run_id=ws.task_run_id,
                loop_id=ws.loop_id,
                factor_names=factor_names_list,
                window_name="main_window",
                factors_perf=per_factor_stats,
            )
        except Exception:
            factor_perf_payload = None

        if factor_perf_payload is not None:
            if not dry_run:
                _write_json(factor_perf_path, factor_perf_payload, overwrite=overwrite_json)

                artifact_id = _stable_artifact_id(
                    task_run_id=ws.task_run_id,
                    workspace_id=ws.workspace_id,
                    artifact_type="factor_perf",
                    name="factor_perf.json",
                )
                status = "present" if factor_perf_path.exists() else "missing"
                _insert_artifact(
                    cur,
                    artifact_id=artifact_id,
                    ws=ws,
                    artifact_type="factor_perf",
                    name="factor_perf.json",
                    status=status,
                    primary_flag=False,
                    summary={"file": "factor_perf.json"},
                    entry_path=_rel(ws_root, factor_perf_path) if factor_perf_path.exists() else _rel(ws_root, factor_perf_path),
                )
                if factor_perf_path.exists():
                    size_bytes, mtime_utc = _best_effort_file_meta(factor_perf_path)
                    _insert_artifact_file(
                        cur,
                        artifact_id=artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, factor_perf_path),
                        kind="factor_perf",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )

            results["generated_files"].append("factor_perf.json")
            results["registered_artifacts"].append("factor_perf")

    # ------------------------
    # feedback.json（基于指标的最小可用 summary）
    # ------------------------
    feedback_path = ws_root / "feedback.json"
    need_write_feedback = overwrite_json or (not feedback_path.exists())
    if need_write_feedback:
        feedback_payload = None
        try:
            summary: dict[str, Any] = {
                "execution": "",
                "value_feedback": "",
                "shape_feedback": "",
                "limitations": [],
                "code_critic": [],
            }

            decision_val: bool | None = None
            hypothesis_text: str | None = None

            if feedback_from_logs is not None:
                decision_val = feedback_from_logs.get("decision")
                hypothesis_text = feedback_from_logs.get("new_hypothesis") or None
                summary["execution"] = feedback_from_logs.get("observations", "") or ""
                summary["value_feedback"] = feedback_from_logs.get("hypothesis_evaluation", "") or ""
                # 形状反馈先简单复用 reason
                summary["shape_feedback"] = feedback_from_logs.get("reason", "") or ""
            elif metrics is not None:
                parts: list[str] = []
                for k in ["annualized_return", "ann_return", "max_drawdown", "mdd", "sharpe", "multi_score"]:
                    if k in metrics:
                        parts.append(f"{k}={metrics[k]}")
                summary["value_feedback"] = "; ".join(parts)

            feedback_payload = _build_feedback_dict(
                decision=decision_val,
                hypothesis=hypothesis_text,
                summary=summary,
                task_run_id=ws.task_run_id,
                loop_id=ws.loop_id,
                generated_at_utc=_utc_now_iso(),
            )
        except Exception:
            feedback_payload = None

        if feedback_payload is not None:
            if not dry_run:
                _write_json(feedback_path, feedback_payload, overwrite=overwrite_json)

                artifact_id = _stable_artifact_id(
                    task_run_id=ws.task_run_id,
                    workspace_id=ws.workspace_id,
                    artifact_type="feedback",
                    name="feedback.json",
                )
                status = "present" if feedback_path.exists() else "missing"
                _insert_artifact(
                    cur,
                    artifact_id=artifact_id,
                    ws=ws,
                    artifact_type="feedback",
                    name="feedback.json",
                    status=status,
                    primary_flag=False,
                    summary={"file": "feedback.json"},
                    entry_path=_rel(ws_root, feedback_path) if feedback_path.exists() else _rel(ws_root, feedback_path),
                )
                if feedback_path.exists():
                    size_bytes, mtime_utc = _best_effort_file_meta(feedback_path)
                    _insert_artifact_file(
                        cur,
                        artifact_id=artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, feedback_path),
                        kind="feedback",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )

            results["generated_files"].append("feedback.json")
            results["registered_artifacts"].append("feedback")

    # ------------------------
    # ret_curve.png (Deprecated: AIstock no longer needs return plots)
    # ------------------------
    pass

    # ------------------------
    # strategy pythonization (Phase 3)
    # ------------------------
    if action in ("model", "factor", "unknown"):
        try:
            from rdagent.utils.artifacts_writer import _sync_strategy_impl_to_shared_lib
            strategy_meta = _sync_strategy_impl_to_shared_lib(ws_root=ws_root)
            if strategy_meta:
                strategy_meta_path = ws_root / "strategy_meta.json"
                _write_json(strategy_meta_path, strategy_meta, overwrite=overwrite_json)
                
                # Register as artifact
                artifact_id = _stable_artifact_id(
                    task_run_id=ws.task_run_id,
                    workspace_id=ws.workspace_id,
                    artifact_type="strategy_meta",
                    name="strategy_meta.json",
                )
                _insert_artifact(
                    cur,
                    artifact_id=artifact_id,
                    ws=ws,
                    artifact_type="strategy_meta",
                    name="strategy_meta.json",
                    status="present",
                    primary_flag=False,
                    summary={"file": "strategy_meta.json"},
                    entry_path=_rel(ws_root, strategy_meta_path),
                )
                results["generated_files"].append("strategy_meta.json")
                results["registered_artifacts"].append("strategy_meta")
        except Exception:
            pass

    return results


def _maybe_backfill_one_workspace(
    cur: sqlite3.Cursor,
    ws: WorkspaceRow,
    *,
    overwrite_json: bool,
    dry_run: bool,
    cleanup_existing: bool,
) -> dict[str, Any]:
    # 路径转换：必须在检测文件前完成，兼容 Windows/WSL
    ws_root = _to_native_path(ws.workspace_path)
    if not ws_root.exists():
        return {"workspace_id": ws.workspace_id, "skipped": True, "reason": f"workspace_path_not_found: {ws_root}"}

    meta_path = ws_root / "workspace_meta.json"
    summary_path = ws_root / "experiment_summary.json"
    manifest_path = ws_root / "manifest.json"

    # Detect files - 优先检查根目录，避免在大目录下 os.walk
    qlib_res = ws_root / "qlib_res.csv"
    ret_pkl = ws_root / "ret.pkl"
    ret_schema_parquet = ws_root / "ret_schema.parquet"
    ret_schema_json = ws_root / "ret_schema.json"
    signals_parquet = ws_root / "signals.parquet"
    signals_json = ws_root / "signals.json"
    mlruns = ws_root / "mlruns"
    combined_factors = ws_root / "combined_factors_df.parquet"
    factor_py = ws_root / "factor.py"
    model_pkl = ws_root / "model.pkl"
    
    # 仅当根目录缺失且业务类型匹配时，才深度搜索子目录以提升性能
    action = ws.experiment_type or _read_loop_action(cur, ws.task_run_id, ws.loop_id) or "unknown"
    
    # 路径转换：必须在检测文件前完成，兼容 Windows/WSL
    ws_root = _to_native_path(ws.workspace_path)
    if not ws_root.exists():
        return {"workspace_id": ws.workspace_id, "skipped": True, "reason": f"workspace_path_not_found: {ws_root}"}

    meta_path = ws_root / "workspace_meta.json"
    summary_path = ws_root / "experiment_summary.json"
    manifest_path = ws_root / "manifest.json"

    # Detect files - 优先检查根目录，避免在大目录下 os.walk
    qlib_res = ws_root / "qlib_res.csv"
    ret_pkl = ws_root / "ret.pkl"
    ret_schema_parquet = ws_root / "ret_schema.parquet"
    ret_schema_json = ws_root / "ret_schema.json"
    signals_parquet = ws_root / "signals.parquet"
    signals_json = ws_root / "signals.json"
    mlruns = ws_root / "mlruns"
    combined_factors = ws_root / "combined_factors_df.parquet"
    factor_py = ws_root / "factor.py"
    model_pkl = ws_root / "model.pkl"
    
    if not (factor_py.exists() or model_pkl.exists() or qlib_res.exists() or combined_factors.exists()):
        # 深度探测逻辑：仅扫描前两层目录，避免深入大型子目录
        for root, dirs, files in os.walk(str(ws_root)):
            # 排除大型无关目录
            dirs[:] = [d for d in dirs if d not in ("mlruns", ".git", "__pycache__", "data", "result", "node_modules", ".venv", "site-packages")]
            
            # 限制扫描深度
            depth = root[len(str(ws_root)):].count(os.sep)
            if depth > 2:
                dirs[:] = []
                continue

            for f in files:
                if f == "factor.py":
                    factor_py = Path(root) / f
                elif f == "model.pkl":
                    model_pkl = Path(root) / f
            
            if factor_py.exists() and model_pkl.exists():
                break

    yaml_confs = _iter_yaml_confs(ws_root)

    # REQ-LOOP-VALIDATION: 验证Loop的核心资产完整性
    # Loop必须具备以下核心资产才能在AIstock侧执行选股和模拟盘运行
    # 1. 模型权重（必需）
    model_files: list[Path] = []
    for name in ("model.pkl", "model.joblib", "model.bin", "model.onnx", "model.pt", "model.pth"):
        p = ws_root / name
        if p.exists():
            model_files.append(p)
    
    if not model_files and action == "model":
        # 缺少模型权重，不补录该Loop
        return {
            "workspace_id": ws.workspace_id,
            "loop_id": ws.loop_id,
            "skipped": True,
            "reason": "missing_model_weights",
        }
    
    # 2. 配置文件（必需）
    yaml_confs = _iter_yaml_confs(ws_root)
    if not yaml_confs:
        # 缺少配置文件，不补录该Loop
        return {
            "workspace_id": ws.workspace_id,
            "loop_id": ws.loop_id,
            "skipped": True,
            "reason": "missing_yaml_configs",
        }
    
    # 3. Python代码（必需）
    py_files = list(ws_root.glob("*.py"))
    if not py_files:
        # 缺少Python代码，不补录该Loop
        return {
            "workspace_id": ws.workspace_id,
            "loop_id": ws.loop_id,
            "skipped": True,
            "reason": "missing_python_code",
        }
    
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
    
    if not factor_names and action == "model":
        # 缺少因子列表，不补录该Loop
        return {
            "workspace_id": ws.workspace_id,
            "loop_id": ws.loop_id,
            "skipped": True,
            "reason": "missing_factor_names",
        }

    # 增强 results 检测逻辑：只要有核心物理资产（代码或权重）即视为 has_result=1
    if action == "model":
        has_result = bool(qlib_res.exists() or (model_pkl and model_pkl.exists()))
        required_files = ["qlib_res.csv", "ret.pkl", "model.pkl"]
    elif action == "factor":
        has_result = bool(combined_factors.exists() or (factor_py and factor_py.exists()))
        required_files = ["combined_factors_df.parquet", "factor.py"]
    else:
        # Fallback detection for unknown actions
        has_result = bool(qlib_res.exists() or combined_factors.exists() or (factor_py and factor_py.exists()) or (model_pkl and model_pkl.exists()))
        required_files = []

    # 如果当前 workspace 判定已有结果，并且不是 dry_run，则将对应 loop 的 has_result 标记为 1。
    # 这样 Loop Catalog 等只依赖 loops.has_result 的消费方也能看到这些补录出来的结果。
    if has_result and (not dry_run):
        try:
            cur.execute(
                """
                UPDATE loops
                SET has_result=1
                WHERE task_run_id=? AND loop_id=? AND (has_result IS NULL OR has_result=0)
                """,
                (ws.task_run_id, ws.loop_id),
            )
        except Exception:
            # 更新失败不影响后续文件与 artifacts 补录
            pass

    now_utc = _utc_now_iso()

    task_run_snapshot = {"task_run_id": ws.task_run_id}
    loop_snapshot = {"task_run_id": ws.task_run_id, "loop_id": ws.loop_id, "action": action}

    pointers = {
        "meta_path": _rel(ws_root, meta_path),
        "summary_path": _rel(ws_root, summary_path),
        "manifest_path": _rel(ws_root, manifest_path),
    }

    result_criteria = {"required_files": required_files, "has_result": has_result}

    meta_payload = {
        "task_run_id": ws.task_run_id,
        "loop_id": ws.loop_id,
        "action": action,
        "workspace_id": ws.workspace_id,
        "workspace_role": ws.workspace_role,
        "experiment_type": ws.experiment_type,
        "workspace_path": str(ws_root),
        "status": ws.status,
        "has_result": has_result,
        "generated_at_utc": now_utc,
        "result_criteria": result_criteria,
        "task_run": task_run_snapshot,
        "loop": loop_snapshot,
        "pointers": pointers,
    }

    summary_payload: dict[str, Any] = {
        "task_run_id": ws.task_run_id,
        "loop_id": ws.loop_id,
        "action": action,
        "workspace_id": ws.workspace_id,
        "workspace_role": ws.workspace_role,
        "experiment_type": ws.experiment_type,
        "workspace_path": str(ws_root),
        "status": ws.status,
        "has_result": has_result,
        "generated_at_utc": now_utc,
        "result_criteria": result_criteria,
        "task_run": task_run_snapshot,
        "loop": loop_snapshot,
        "pointers": pointers,
        "files": {
            "qlib_res.csv": _rel(ws_root, qlib_res) if qlib_res.exists() else None,
            "ret.pkl": _rel(ws_root, ret_pkl) if ret_pkl.exists() else None,
            "ret_schema.parquet": _rel(ws_root, ret_schema_parquet) if ret_schema_parquet.exists() else None,
            "ret_schema.json": _rel(ws_root, ret_schema_json) if ret_schema_json.exists() else None,
            "signals.parquet": _rel(ws_root, signals_parquet) if signals_parquet.exists() else None,
            "signals.json": _rel(ws_root, signals_json) if signals_json.exists() else None,
            "combined_factors_df.parquet": _rel(ws_root, combined_factors) if combined_factors.exists() else None,
            "mlruns": _rel(ws_root, mlruns) if mlruns.exists() else None,
        },
        "artifacts": [],
    }

    manifest_payload: dict[str, Any] = {
        "task_run_id": ws.task_run_id,
        "loop_id": ws.loop_id,
        "action": action,
        "workspace_id": ws.workspace_id,
        "workspace_role": ws.workspace_role,
        "experiment_type": ws.experiment_type,
        "workspace_path": str(ws_root),
        "status": ws.status,
        "has_result": has_result,
        "generated_at_utc": now_utc,
        "result_criteria": result_criteria,
        "task_run": task_run_snapshot,
        "loop": loop_snapshot,
        "pointers": pointers,
        "artifacts": [],
    }

    if not dry_run:
        _write_json(meta_path, meta_payload, overwrite=overwrite_json)
        _write_json(summary_path, summary_payload, overwrite=overwrite_json)
        _write_json(manifest_path, manifest_payload, overwrite=overwrite_json)

        _update_workspace_paths(
            cur,
            ws,
            meta_rel=_rel(ws_root, meta_path),
            summary_rel=_rel(ws_root, summary_path),
            manifest_rel=_rel(ws_root, manifest_path),
        )

    # Backfill artifacts
    artifacts_created: list[str] = []

    if ws.workspace_role == "experiment_workspace":
        if cleanup_existing and (not dry_run):
            if action == "model":
                _cleanup_existing_for_workspace(
                    cur,
                    ws,
                    artifact_types=[
                        "report",
                        "model",
                        "config_snapshot",
                        "backtest_metrics",
                        "backtest_curve",
                        "signals",
                        "ret_schema",
                    ],
                )
            elif action == "factor":
                _cleanup_existing_for_workspace(
                    cur,
                    ws,
                    artifact_types=["factor_data"],
                )
        if action == "model":
            # REQ-MODEL-P3-010: Ensure model_meta.json is generated with authoritative metadata
            model_meta = _extract_model_metadata_from_workspace(ws_root)
            model_meta_path = ws_root / "model_meta.json"
            _write_json(model_meta_path, model_meta, overwrite=overwrite_json)

            # Register model_meta as a separate artifact for easier consumption
            mm_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="model_meta",
                name="model_meta.json",
            )
            _insert_artifact(
                cur,
                artifact_id=mm_artifact_id,
                ws=ws,
                artifact_type="model_meta",
                name="model_meta.json",
                status="present",
                primary_flag=False,
                summary={"file": "model_meta.json"},
                entry_path=_rel(ws_root, model_meta_path),
                model_type=model_meta.get("model_type"),
                model_conf=model_meta.get("model_conf"),
                dataset_conf=model_meta.get("dataset_conf"),
                feature_schema=model_meta.get("feature_schema"),
            )
            artifacts_created.append(mm_artifact_id)

            report_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="report",
                name="qlib_report",
            )
            _insert_artifact(
                cur,
                artifact_id=report_artifact_id,
                ws=ws,
                artifact_type="report",
                name="qlib_report",
                status="present" if (qlib_res.exists() and ret_pkl.exists()) else "missing",
                primary_flag=True,
                summary={
                    "files": [
                        "qlib_res.csv",
                        "ret.pkl",
                        "ret_schema.parquet",
                        "ret_schema.json",
                        "signals.parquet",
                        "signals.json",
                    ]
                },
                entry_path=_rel(ws_root, qlib_res) if qlib_res.exists() else ".",
                model_type=model_meta.get("model_type"),
                model_conf=model_meta.get("model_conf"),
                dataset_conf=model_meta.get("dataset_conf"),
                feature_schema=model_meta.get("feature_schema"),
            )

            report_files = [
                qlib_res,
                ret_pkl,
                ret_schema_parquet,
                ret_schema_json,
                signals_parquet,
                signals_json,
            ]
            for p in report_files:
                if not p.exists():
                    continue
                size_bytes, mtime_utc = _best_effort_file_meta(p)
                _insert_artifact_file(
                    cur,
                    artifact_id=report_artifact_id,
                    ws=ws,
                    path_rel=_rel(ws_root, p),
                    kind="report",
                    size_bytes=size_bytes,
                    mtime_utc=mtime_utc,
                )
            artifacts_created.append(report_artifact_id)

            # Backtest metrics artifact (qlib_res.csv)
            metrics_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="backtest_metrics",
                name="qlib_res.csv",
            )
            metrics_status = "present" if qlib_res.exists() else "missing"
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=metrics_artifact_id,
                    ws=ws,
                    artifact_type="backtest_metrics",
                    name="qlib_res.csv",
                    status=metrics_status,
                    primary_flag=True,
                    summary={"file": "qlib_res.csv"},
                    entry_path=_rel(ws_root, qlib_res) if qlib_res.exists() else ".",
                )
                if qlib_res.exists():
                    size_bytes, mtime_utc = _best_effort_file_meta(qlib_res)
                    _insert_artifact_file(
                        cur,
                        artifact_id=metrics_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, qlib_res),
                        kind="backtest_metrics",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(metrics_artifact_id)

            # Backtest curve artifact (ret.pkl)
            curve_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="backtest_curve",
                name="ret.pkl",
            )
            curve_status = "present" if ret_pkl.exists() else "missing"
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=curve_artifact_id,
                    ws=ws,
                    artifact_type="backtest_curve",
                    name="ret.pkl",
                    status=curve_status,
                    primary_flag=True,
                    summary={"file": "ret.pkl"},
                    entry_path=_rel(ws_root, ret_pkl) if ret_pkl.exists() else ".",
                )
                if ret_pkl.exists():
                    size_bytes, mtime_utc = _best_effort_file_meta(ret_pkl)
                    _insert_artifact_file(
                        cur,
                        artifact_id=curve_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, ret_pkl),
                        kind="backtest_curve",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(curve_artifact_id)

            # Signals artifact (signals.parquet / signals.json)
            signals_entry: Path | None = None
            if signals_parquet.exists():
                signals_entry = signals_parquet
            elif signals_json.exists():
                signals_entry = signals_json

            signals_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="signals",
                name="signals",
            )
            signals_status = "present" if signals_entry is not None else "missing"
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=signals_artifact_id,
                    ws=ws,
                    artifact_type="signals",
                    name="signals",
                    status=signals_status,
                    primary_flag=False,
                    summary={
                        "files": [
                            p.name
                            for p in [signals_parquet, signals_json]
                            if p.exists()
                        ]
                    },
                    entry_path=_rel(ws_root, signals_entry) if signals_entry is not None else ".",
                )
                if signals_entry is not None:
                    size_bytes, mtime_utc = _best_effort_file_meta(signals_entry)
                    _insert_artifact_file(
                        cur,
                        artifact_id=signals_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, signals_entry),
                        kind="signals",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(signals_artifact_id)

            # ret_schema artifact (ret_schema.parquet / ret_schema.json)
            schema_entry: Path | None = None
            if ret_schema_parquet.exists():
                schema_entry = ret_schema_parquet
            elif ret_schema_json.exists():
                schema_entry = ret_schema_json

            schema_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="ret_schema",
                name="ret_schema",
            )
            schema_status = "present" if schema_entry is not None else "missing"
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=schema_artifact_id,
                    ws=ws,
                    artifact_type="ret_schema",
                    name="ret_schema",
                    status=schema_status,
                    primary_flag=False,
                    summary={
                        "files": [
                            p.name
                            for p in [ret_schema_parquet, ret_schema_json]
                            if p.exists()
                        ]
                    },
                    entry_path=_rel(ws_root, schema_entry) if schema_entry is not None else ".",
                )
                if schema_entry is not None:
                    size_bytes, mtime_utc = _best_effort_file_meta(schema_entry)
                    _insert_artifact_file(
                        cur,
                        artifact_id=schema_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, schema_entry),
                        kind="ret_schema",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(schema_artifact_id)

            mlruns_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="model",
                name="mlruns",
            )
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=mlruns_artifact_id,
                    ws=ws,
                    artifact_type="model",
                    name="mlruns",
                    status="present" if mlruns.exists() else "missing",
                    primary_flag=False,
                    summary={"path": "mlruns"},
                    entry_path=_rel(ws_root, mlruns) if mlruns.exists() else ".",
                )
                if mlruns.exists():
                    size_bytes, mtime_utc = _best_effort_file_meta(mlruns)
                    _insert_artifact_file(
                        cur,
                        artifact_id=mlruns_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, mlruns),
                        kind="model",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(mlruns_artifact_id)

            cfg_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="config_snapshot",
                name="workspace_configs",
            )
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=cfg_artifact_id,
                    ws=ws,
                    artifact_type="config_snapshot",
                    name="workspace_configs",
                    status="present" if len(yaml_confs) > 0 else "missing",
                    primary_flag=False,
                    summary={"count": len(yaml_confs)},
                    entry_path=".",
                )
                for p in yaml_confs:
                    size_bytes, mtime_utc = _best_effort_file_meta(p)
                    _insert_artifact_file(
                        cur,
                        artifact_id=cfg_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, p),
                        kind="config_snapshot",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(cfg_artifact_id)

        elif action == "factor":
            factor_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="factor_data",
                name="combined_factors",
            )
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=factor_artifact_id,
                    ws=ws,
                    artifact_type="factor_data",
                    name="combined_factors",
                    status="present" if combined_factors.exists() else "missing",
                    primary_flag=True,
                    summary={"files": ["combined_factors_df.parquet"]},
                    entry_path=_rel(ws_root, combined_factors) if combined_factors.exists() else ".",
                )
                if combined_factors.exists():
                    size_bytes, mtime_utc = _best_effort_file_meta(combined_factors)
                    _insert_artifact_file(
                        cur,
                        artifact_id=factor_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, combined_factors),
                        kind="factor_data",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(factor_artifact_id)

    return {
        "workspace_id": ws.workspace_id,
        "loop_id": ws.loop_id,
        "action": action,
        "workspace_role": ws.workspace_role,
        "has_result": has_result,
        "artifacts_created": artifacts_created,
        "wrote_json": (not dry_run),
    }


def _load_workspaces(cur: sqlite3.Cursor, task_run_id: str | None = None) -> list[WorkspaceRow]:
    query = """
        select
          workspace_id, task_run_id, loop_id, workspace_role, experiment_type, step_name, status,
          workspace_path, meta_path, summary_path, manifest_path
        from workspaces
    """
    params = []
    if task_run_id:
        query += " where task_run_id = ?"
        params.append(task_run_id)

    query += """
        order by loop_id asc, case when workspace_role='experiment_workspace' then 0 else 1 end, workspace_id asc
    """
    rows = cur.execute(query, params).fetchall()

    out: list[WorkspaceRow] = []
    for r in rows:
        # 统一进行路径转换
        ws_path = str(_to_native_path(r[7]))
        out.append(
            WorkspaceRow(
                workspace_id=r[0],
                task_run_id=r[1],
                loop_id=_safe_int(r[2]),
                workspace_role=r[3],
                experiment_type=r[4],
                step_name=r[5],
                status=r[6],
                workspace_path=ws_path,
                meta_path=r[8],
                summary_path=r[9],
                manifest_path=r[10],
            )
        )
    return out

def _iter_task_runs(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    """Fetch all task_runs with their log_trace_path from the registry.

    说明：
    - 仅依赖 task_runs 表中的 task_run_id / log_trace_path / created_at_utc；
    - 不做任何写操作；
    - 调用方可据此按 task_run 粒度驱动 log-based backfill。
    """

    rows = cur.execute(
        "select task_run_id, log_trace_path, created_at_utc from task_runs order by created_at_utc asc"
    ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "task_run_id": str(r[0]),
                "log_trace_path": r[1],
                "created_at_utc": r[2],
            }
        )
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="RDagentDB/registry.sqlite")
    ap.add_argument("--task-run-id")
    ap.add_argument(
        "--all-task-runs",
        action="store_true",
        help="When set (and --log-path is not provided), process all task_runs in the registry instead of a single --task-run-id.",
    )
    ap.add_argument(
        "--mode",
        choices=["backfill", "materialize-pending", "materialize-all", "check", "solidify-all"],
        default="backfill",
        help="backfill: update task_runs/loops/workspaces, materialize-*: generate artifacts, solidify-all: copy assets to bundles",
    )
    ap.add_argument(
        "--since-date",
        help="Optional ISO date/time (e.g. 2025-01-01 or 2025-01-01T00:00:00) to only scan workspaces whose workspace_path mtime is on/after this value.",
    )
    ap.add_argument(
        "--max-loops",
        type=int,
        help="Optional upper bound on the number of workspaces to process (after filtering).",
    )
    ap.add_argument("--overwrite-json", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-experiment-workspace", action="store_true")
    ap.add_argument(
        "--solidify-heartbeat-seconds",
        type=int,
        default=60,
        help="When --mode solidify-all, print a heartbeat every N seconds to show progress.",
    )
    ap.add_argument(
        "--solidify-stall-seconds",
        type=int,
        default=300,
        help="When --mode solidify-all, abort if production_bundles has no file/mtime change for N seconds.",
    )
    ap.add_argument(
        "--cleanup-existing",
        action="store_true",
        help="Delete existing artifacts/artifact_files for the selected workspaces before backfilling (useful to remove duplicates from previous runs).",
    )
    ap.add_argument(
        "--log-path",
        help=(
            "RD-Agent 日志根目录。当在 backfill 模式下提供该参数时，工具会自动从 registry.sqlite 的 task_runs 表中"
            "读取每个 task_run 的 log_trace_path，拼接成完整日志目录，并按 task_run/loop 粒度执行基于决策的 Phase2 补录"
            "（仅对 HypothesisFeedback.decision 为 True 的 loop 生效），无需人工逐个指定 workspace 或日志路径。"
        ),
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")

    con = sqlite3.connect(str(db_path), timeout=60)
    try:
        cur = con.cursor()

        # Log 驱动模式：仅在 backfill 时生效。
        # 此时 --log-path 被视为 RD-Agent 日志根目录，具体 task_run 的日志目录从 task_runs.log_trace_path 中获取。
        if args.log_path and args.mode == "backfill":
            log_root = Path(args.log_path)

            print(f"db={db_path}")
            print(f"log_root={log_root}")
            print(f"mode={args.mode}")
            print(f"dry_run={bool(args.dry_run)} overwrite_json={bool(args.overwrite_json)}")

            task_runs = _iter_task_runs(cur)
            log_drive_results: list[dict[str, Any]] = []

            for tr in task_runs:
                tr_id = tr["task_run_id"]
                log_trace_path = tr.get("log_trace_path")
                if not log_trace_path:
                    continue

                # 若 log_trace_path 为相对路径，则视为相对于 log_root；若为绝对路径则直接使用
                candidate = Path(str(log_trace_path))
                if not candidate.is_absolute():
                    log_path = log_root / candidate
                else:
                    log_path = candidate

                if not log_path.exists():
                    continue

                loops_info = _collect_log_session_loops(log_path)

                print(f"task_run_id={tr_id} log_path={log_path} loops_in_log={len(loops_info)}")

                # 调试：打印每个 loop 的决策和解析出的 workspace_paths
                try:
                    debug_view = {
                        str(loop_id): {
                            "decision": (info.get("feedback") or {}).get("decision"),
                            "workspace_paths": sorted(list(info.get("workspace_paths") or [])),
                            "factor_items_count": len(info.get("factor_items") or []),
                        }
                        for loop_id, info in loops_info.items()
                    }
                    print("loops_info_debug=", json.dumps(debug_view, ensure_ascii=False, indent=2))
                except Exception:
                    pass

                for loop_id, info in loops_info.items():
                    fb = info.get("feedback") or {}
                    decision = fb.get("decision")
                    # 仅对决策为 True 的 loop 做因子/反馈补录
                    if not decision:
                        continue

                    # 按 task_run_id + loop_id 精确定位 experiment_workspace，避免 loop_id 跨 task_run 混淆
                    rows = cur.execute(
                        """
                        select
                          workspace_id, task_run_id, loop_id, workspace_role, experiment_type, step_name, status,
                          workspace_path, meta_path, summary_path, manifest_path
                        from workspaces
                        where task_run_id=? and loop_id=? and workspace_role='experiment_workspace'
                        """,
                        (tr_id, loop_id),
                    ).fetchall()

                    for row in rows:
                        ws = WorkspaceRow(
                            workspace_id=row[0],
                            task_run_id=row[1],
                            loop_id=_safe_int(row[2]),
                            workspace_role=row[3],
                            experiment_type=row[4],
                            step_name=row[5],
                            status=row[6],
                            workspace_path=row[7],
                            meta_path=row[8],
                            summary_path=row[9],
                            manifest_path=row[10],
                        )

                        base_res = _maybe_backfill_one_workspace(
                            cur,
                            ws,
                            overwrite_json=bool(args.overwrite_json),
                            dry_run=bool(args.dry_run),
                            cleanup_existing=bool(args.cleanup_existing),
                        )

                        phase2_res = _maybe_backfill_phase2_for_workspace(
                            cur,
                            ws,
                            overwrite_json=bool(args.overwrite_json),
                            dry_run=bool(args.dry_run),
                            factor_items_from_logs=info.get("factor_items") or None,
                            feedback_from_logs=fb,
                        )

                        # Update log_dir for the loop
                        if not args.dry_run:
                            _update_loop_log_dir(cur, ws.task_run_id, ws.loop_id, str(log_path))

                        log_drive_results.append(
                            {
                                "workspace_id": ws.workspace_id,
                                "task_run_id": ws.task_run_id,
                                "loop_id": ws.loop_id,
                                "workspace_role": ws.workspace_role,
                                "action": base_res.get("action"),
                                "has_result": base_res.get("has_result"),
                                "phase1": base_res,
                                "phase2": phase2_res,
                            }
                        )

            if not args.dry_run:
                con.commit()

            print(json.dumps({"results": log_drive_results}, ensure_ascii=False, indent=2))
            return

        # 任务 ID 选择逻辑：
        # - 若提供了 --task-run-id，则仅处理该任务；
        # - 若未提供，但在 materialize-pending 或 check 模式下，或显式指定了 --all-task-runs，则处理所有任务。
        task_run_ids: list[str]
        if args.task_run_id:
            task_run_ids = [str(args.task_run_id)]
        elif args.all_task_runs or args.mode in ["materialize-pending", "check"]:
            rows = cur.execute("select distinct task_run_id from task_runs order by created_at_utc asc").fetchall()
            task_run_ids = [str(r[0]) for r in rows]
        else:
            if not args.log_path:
                raise SystemExit("--task-run-id is required when --log-path is not provided and --all-task-runs is not set")
            task_run_ids = []  # Log-path mode handled separately above

        wss: list[WorkspaceRow] = []
        for tr_id in task_run_ids:
            wss.extend(_load_workspaces(cur, tr_id))

        if args.only_experiment_workspace:
            wss = [w for w in wss if w.workspace_role == "experiment_workspace"]

        # since-date: 基于 workspace_path 的文件 mtime 做时间过滤，避免依赖 DB schema 细节
        if args.since_date:
            try:
                try:
                    since_dt = datetime.fromisoformat(args.since_date)
                except ValueError:
                    # 仅日期时补零点
                    since_dt = datetime.fromisoformat(args.since_date + "T00:00:00")
                since_ts = since_dt.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                since_ts = None

            if since_ts is not None:
                filtered: list[WorkspaceRow] = []
                for w in wss:
                    try:
                        st = os.stat(w.workspace_path)
                        if st.st_mtime >= since_ts:
                            filtered.append(w)
                    except Exception:
                        # 如果无法 stat，则仍然保留，避免误删
                        filtered.append(w)
                wss = filtered

        if args.max_loops is not None and args.max_loops > 0:
            wss = wss[: args.max_loops]

        print(f"db={db_path}")
        if args.all_task_runs:
            print("task_run_id=<ALL>")
        else:
            print(f"task_run_id={args.task_run_id}")
        print(f"mode={args.mode}")
        print(f"workspaces={len(wss)}")
        print(f"dry_run={bool(args.dry_run)} overwrite_json={bool(args.overwrite_json)}")

        results: list[dict[str, Any]] = []

        if args.mode == "backfill":
            for ws in wss:
                # Phase1：meta/summary/manifest + 基础 artifacts（保持原有行为）
                base_res = _maybe_backfill_one_workspace(
                    cur,
                    ws,
                    overwrite_json=bool(args.overwrite_json),
                    dry_run=bool(args.dry_run),
                    cleanup_existing=bool(args.cleanup_existing),
                )

                # Phase2：在已有结果的基础上补录 factor_meta/factor_perf/feedback/ret_curve 图表
                phase2_res = _maybe_backfill_phase2_for_workspace(
                    cur,
                    ws,
                    overwrite_json=bool(args.overwrite_json),
                    dry_run=bool(args.dry_run),
                )

                # Update log_dir for the loop if log_trace_path is available in task_runs
                if not args.dry_run:
                    tr_row = cur.execute(
                        "SELECT log_trace_path FROM task_runs WHERE task_run_id=?", (ws.task_run_id,)
                    ).fetchone()
                    if tr_row and tr_row[0]:
                        _update_loop_log_dir(cur, ws.task_run_id, ws.loop_id, tr_row[0])

                results.append({
                    "workspace_id": ws.workspace_id,
                    "loop_id": ws.loop_id,
                    "action": base_res.get("action"),
                    "workspace_role": ws.workspace_role,
                    "has_result": base_res.get("has_result"),
                    "phase1": base_res,
                    "phase2": phase2_res,
                })

            if not args.dry_run:
                con.commit()

        elif args.mode in ["materialize-pending", "materialize-all"]:
            # Phase2 离线补录模式
            # materialize-pending: 仅针对 materialization_status 为 pending/failed/NULL 的 loop 执行
            # materialize-all: 针对所有匹配的 loop 执行
            # 逻辑变更：不再仅处理 experiment_workspace，因为部分资产（如因子代码）可能在 'other' 角色目录中
            wss_to_process = wss # [w for w in wss if w.workspace_role == "experiment_workspace"]

            total = len(wss_to_process)
            processed = 0
            print(f"Mode: {args.mode}, Total workspaces: {total}")

            for ws in wss_to_process:
                processed += 1
                if processed % 10 == 0 or processed == total:
                    print(f"Progress: {processed}/{total} ({(processed/total)*100:.1f}%)")

                if args.mode == "materialize-pending":
                    mat_status = _read_materialization_status(cur, ws.task_run_id, ws.loop_id)
                    if mat_status not in (None, "pending", "failed"):
                        continue
                else:
                    mat_status = _read_materialization_status(cur, ws.task_run_id, ws.loop_id)

                loop_key = {
                    "task_run_id": ws.task_run_id,
                    "loop_id": ws.loop_id,
                }

                loop_result: dict[str, Any] = {
                    "workspace_id": ws.workspace_id,
                    "workspace_role": ws.workspace_role,
                    "task_run_id": ws.task_run_id,
                    "loop_id": ws.loop_id,
                    "previous_materialization_status": mat_status,
                }

                try:
                    if not args.dry_run:
                        # 标记为 running
                        cur.execute(
                            """
                            UPDATE loops
                            SET materialization_status=?, materialization_error=NULL, materialization_updated_at_utc=?
                            WHERE task_run_id=? AND loop_id=?
                            """,
                            ("running", _utc_now_iso(), ws.task_run_id, ws.loop_id),
                        )

                    # 复用现有 Phase1 + Phase2 补录逻辑
                    base_res = _maybe_backfill_one_workspace(
                        cur,
                        ws,
                        overwrite_json=bool(args.overwrite_json),
                        dry_run=bool(args.dry_run),
                        cleanup_existing=bool(args.cleanup_existing),
                    )

                    phase2_res = _maybe_backfill_phase2_for_workspace(
                        cur,
                        ws,
                        overwrite_json=bool(args.overwrite_json),
                        dry_run=bool(args.dry_run),
                    )

                    # 补充更新 loop 的 log_dir（如果 task_runs 中有记录）
                    if not args.dry_run:
                        tr_row = cur.execute(
                            "SELECT log_trace_path FROM task_runs WHERE task_run_id=?", (ws.task_run_id,)
                        ).fetchone()
                        if tr_row and tr_row[0]:
                            _update_loop_log_dir(cur, ws.task_run_id, ws.loop_id, tr_row[0])

                    loop_result["phase1"] = base_res
                    loop_result["phase2"] = phase2_res

                    if not args.dry_run:
                        # 标记为 done
                        cur.execute(
                            """
                            UPDATE loops
                            SET materialization_status=?, materialization_error=NULL, materialization_updated_at_utc=?
                            WHERE task_run_id=? AND loop_id=?
                            """,
                            ("done", _utc_now_iso(), ws.task_run_id, ws.loop_id),
                        )

                    loop_result["materialization_status"] = "done"
                    loop_result["materialization_error"] = None

                except Exception as e:  # noqa: BLE001
                    err_msg = f"{type(e).__name__}: {e}"
                    loop_result["materialization_status"] = "failed"
                    loop_result["materialization_error"] = err_msg

                    if not args.dry_run:
                        try:
                            cur.execute(
                                """
                                UPDATE loops
                                SET materialization_status=?, materialization_error=?, materialization_updated_at_utc=?
                                WHERE task_run_id=? AND loop_id=?
                                """,
                                ("failed", err_msg[:512], _utc_now_iso(), ws.task_run_id, ws.loop_id),
                            )
                        except Exception:
                            # 状态更新失败不影响本次物化结果的记录输出
                            pass

                results.append({**loop_key, **loop_result})

            if not args.dry_run:
                con.commit()

        elif args.mode == "solidify-all":
            # Phase3 固化模式：将所有有结果但未固化的 loop 进行资产打包与元数据持久化
            # 注意：这里的观测逻辑是为了避免“闷跑数小时无输出”。

            # 1) 按 loop 粒度去重（每个 loop 选一个代表性 workspace）
            unique_loops: dict[tuple[str, int], WorkspaceRow] = {}
            for ws in wss:
                loop_key = (ws.task_run_id, ws.loop_id)
                if loop_key not in unique_loops:
                    unique_loops[loop_key] = ws
                elif ws.workspace_role == "experiment_workspace":
                    unique_loops[loop_key] = ws

            from rdagent.utils.solidification import solidify_loop_assets
            import time

            def _bundle_state() -> tuple[int, int, float]:
                repo_root = Path(db_path).parent.parent
                bundle_root = repo_root / "RDagentDB" / "production_bundles"
                if not bundle_root.exists():
                    return (0, 0, 0.0)

                nonempty_dirs = 0
                file_count = 0
                latest_mtime = 0.0

                for d in bundle_root.iterdir():
                    if not d.is_dir():
                        continue
                    has_file = False
                    for p in d.rglob("*"):
                        if not p.is_file():
                            continue
                        has_file = True
                        file_count += 1
                        try:
                            mt = p.stat().st_mtime
                            if mt > latest_mtime:
                                latest_mtime = mt
                        except Exception:
                            pass
                    if has_file:
                        nonempty_dirs += 1

                return (nonempty_dirs, file_count, latest_mtime)

            # 2) 预统计：到底有多少 loop 会进入 solidify（has_result=1 且 is_solidified!=1）
            total = len(unique_loops)
            eligible = 0
            already_done = 0
            no_result = 0
            legacy_skipped = 0
            for (tr_id, l_id), _ws in unique_loops.items():
                # 跳过历史遗留的 unknown_* task_run，避免把脏数据聚合进一个 bundle
                if isinstance(tr_id, str) and tr_id.startswith("unknown_"):
                    legacy_skipped += 1
                    continue
                row = cur.execute(
                    "SELECT has_result, is_solidified FROM loops WHERE task_run_id=? AND loop_id=?",
                    (tr_id, l_id),
                ).fetchone()
                if not row or not row[0]:
                    no_result += 1
                    continue
                if row[1]:
                    already_done += 1
                    continue
                eligible += 1

            pre_state = _bundle_state()
            print(
                f"Mode: {args.mode}, Unique loops: {total}, Eligible(to solidify): {eligible}, "
                f"NoResult: {no_result}, AlreadySolidified: {already_done}, LegacySkipped: {legacy_skipped}"
            )
            print(
                f"production_bundles state: nonempty_dirs={pre_state[0]}, files={pre_state[1]}, latest_mtime={pre_state[2]}"
            )

            processed = 0
            last_change_mono = time.monotonic()
            last_heartbeat_mono = time.monotonic()
            last_state = pre_state
            last_loop: str | None = None

            for (tr_id, l_id), _ws in unique_loops.items():
                processed += 1

                if isinstance(tr_id, str) and tr_id.startswith("unknown_"):
                    results.append(
                        {
                            "task_run_id": tr_id,
                            "loop_id": l_id,
                            "status": "skipped",
                            "reason": "legacy_task_run",
                        }
                    )
                    continue

                row = cur.execute(
                    "SELECT has_result, is_solidified FROM loops WHERE task_run_id=? AND loop_id=?",
                    (tr_id, l_id),
                ).fetchone()
                if not row or not row[0] or row[1]:
                    continue

                now_mono = time.monotonic()
                hb_sec = max(1, int(args.solidify_heartbeat_seconds))
                stall_sec = int(args.solidify_stall_seconds)
                if now_mono - last_heartbeat_mono >= hb_sec:
                    cur_state = _bundle_state()
                    since_change = int(now_mono - last_change_mono)
                    print(
                        f"[Heartbeat] processed={processed}/{total}, eligible={eligible}, "
                        f"bundle_nonempty={cur_state[0]}, files={cur_state[1]}, "
                        f"since_bundle_change={since_change}s, last_loop={last_loop}"
                    )
                    last_heartbeat_mono = now_mono
                    if stall_sec > 0 and since_change >= stall_sec:
                        raise RuntimeError(
                            f"Abort: production_bundles no change for {since_change}s (>= {stall_sec}s). "
                            f"last_loop={last_loop}, last_state={last_state}, current_state={cur_state}"
                        )

                loop_start_mono = time.monotonic()
                last_loop = f"{tr_id}/{l_id}"
                print(f"[{processed}/{total}] Solidifying loop {tr_id}/{l_id}...")

                try:
                    bundle_id = solidify_loop_assets(tr_id, l_id, db_path=db_path)
                    if bundle_id:
                        results.append(
                            {
                                "task_run_id": tr_id,
                                "loop_id": l_id,
                                "status": "solidified",
                                "asset_bundle_id": bundle_id,
                            }
                        )
                    else:
                        results.append(
                            {
                                "task_run_id": tr_id,
                                "loop_id": l_id,
                                "status": "skipped",
                                "reason": "missing_primary_assets",
                            }
                        )
                except Exception as e:  # noqa: BLE001
                    print(f"Failed to solidify loop {tr_id}/{l_id}: {e}")
                    results.append(
                        {
                            "task_run_id": tr_id,
                            "loop_id": l_id,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

                loop_elapsed = time.monotonic() - loop_start_mono
                cur_state = _bundle_state()
                if cur_state != last_state:
                    last_state = cur_state
                    last_change_mono = time.monotonic()
                    print(
                        f"[BundleChanged] loop={tr_id}/{l_id}, elapsed={loop_elapsed:.1f}s, "
                        f"nonempty_dirs={cur_state[0]}, files={cur_state[1]}, latest_mtime={cur_state[2]}"
                    )
                else:
                    print(
                        f"[NoBundleChange] loop={tr_id}/{l_id}, elapsed={loop_elapsed:.1f}s, "
                        f"nonempty_dirs={cur_state[0]}, files={cur_state[1]}"
                    )

            if not args.dry_run:
                con.commit()

        else:  # check 模式：严格只读
            for ws in wss:
                res = _check_phase2_artifacts_for_workspace(cur, ws)
                results.append(res)

        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    finally:
        con.close()


if __name__ == "__main__":
    main()
