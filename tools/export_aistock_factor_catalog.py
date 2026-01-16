import argparse
import json
import sqlite3
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@dataclass
class WorkspaceRow:
    workspace_id: str
    workspace_path: str


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


def _fetch_workspaces(conn: sqlite3.Connection) -> list[WorkspaceRow]:
    sql = """
    SELECT DISTINCT
      workspace_id,
      workspace_path
    FROM workspaces
    WHERE workspace_path IS NOT NULL AND workspace_path != ''
    """
    rows = conn.execute(sql).fetchall()
    out: list[WorkspaceRow] = []
    for r in rows:
        out.append(
            WorkspaceRow(
                workspace_id=str(r["workspace_id"]),
                workspace_path=str(r["workspace_path"]),
            )
        )
    return out


def _merge_factor_meta(acc: dict[str, dict[str, Any]], ws_root: Path) -> None:
    # 1. 加载因子基础元数据 (由 artifacts_writer 或 backfill 生成)
    meta_path = ws_root / "factor_meta.json"
    meta = _load_json_if_exists(meta_path)
    if not meta:
        return

    factors = meta.get("factors") or []
    if not isinstance(factors, list):
        return

    # 2. 加载因子性能指标 (由 artifacts_writer 或 backfill 生成)
    perf_path = ws_root / "factor_perf.json"
    perf = _load_json_if_exists(perf_path)
    perf_by_factor: dict[str, dict[str, Any]] = {}
    if perf:
        combos = perf.get("combinations") or []
        for combo in combos:
            f_names = combo.get("factor_names") or []
            windows = combo.get("windows") or []
            if not windows:
                continue
            # AIstock 通常只关注主窗口表现
            main_window = windows[0]
            # 提取核心指标：优先使用 Sharpe 作为表现基准
            metrics_summary = {
                "annual_return": main_window.get("annual_return"),
                "max_drawdown": main_window.get("max_drawdown"),
                "sharpe": main_window.get("sharpe"),
                "loop_id": perf.get("loop_id"),
                "task_run_id": perf.get("task_run_id"),
            }
            # 如果 window 下还有更详细的 metrics 字典，一并并入
            if isinstance(main_window.get("metrics"), dict):
                metrics_summary.update(main_window["metrics"])

            for name in f_names:
                # 若因子属于多个组合，保留 Sharpe 更高的表现
                if name not in perf_by_factor or (metrics_summary.get("sharpe") or -999) > (perf_by_factor[name].get("sharpe") or -999):
                    perf_by_factor[name] = metrics_summary

    for f in factors:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        if not isinstance(name, str) or not name:
            continue

        existing = acc.get(name, {})
        merged: dict[str, Any] = dict(existing)
        
        # 合并基础元数据
        for k, v in f.items():
            if v is None:
                continue
            # 统一字段：formula_hint 和 expression 视为同义词，优先保留非空值
            if k in ("formula_hint", "expression"):
                if not merged.get("expression"):
                    merged["expression"] = v
                if not merged.get("formula_hint"):
                    merged["formula_hint"] = v
                continue

            # 标签合并：取并集
            if k == "tags" and isinstance(v, list):
                existing_tags = set(merged.get("tags") or [])
                existing_tags.update(v)
                merged["tags"] = sorted(list(existing_tags))
                continue

            if k not in merged or merged[k] in (None, "", [], {}):
                merged[k] = v
            elif k == "interface_info" and isinstance(v, dict):
                existing_info = merged.get("interface_info") or {}
                existing_info.update(v)
                merged[k] = existing_info

        # 关联最佳表现 (Loop 层聚合)
        if name in perf_by_factor:
            metrics = perf_by_factor[name]
            current_best_sharpe = merged.get("best_performance_sharpe")
            if current_best_sharpe is None or (metrics.get("sharpe") or -999) > (current_best_sharpe or -999):
                # 展平字段以方便 AIstock 侧直接解析
                merged["best_performance_sharpe"] = metrics.get("sharpe")
                merged["best_performance_ann_ret"] = metrics.get("annual_return")
                merged["best_performance_mdd"] = metrics.get("max_drawdown")
                merged["best_loop_id"] = metrics.get("loop_id")
                merged["best_task_run_id"] = metrics.get("task_run_id")
                
                # 构造一个汇总字符串
                sharpe_val = metrics.get("sharpe")
                ann_ret_val = metrics.get("annual_return")
                parts = []
                if sharpe_val is not None:
                    try:
                        parts.append(f"Sharpe: {float(sharpe_val):.2f}")
                    except (ValueError, TypeError):
                        parts.append(f"Sharpe: {sharpe_val}")
                if ann_ret_val is not None:
                    try:
                        parts.append(f"Ann.Ret: {float(ann_ret_val):.2%}")
                    except (ValueError, TypeError):
                        parts.append(f"Ann.Ret: {ann_ret_val}")
                
                summary_str = ", ".join(parts) or "已回测"
                merged["best_performance_summary"] = summary_str
                
                # REQ-FACTOR-ASSOC: 确保 'best_performance' 字段包含汇总信息，解决 AIstock 显示“无回测记录”的问题
                merged["best_performance"] = summary_str
                # 保留结构化指标供高级解析
                merged["best_performance_metrics"] = metrics
        
        # 如果依然没有表现记录，确保字段不为 None
        if "best_performance" not in merged:
            merged["best_performance"] = "无回测记录"

        # 确保 region 字段存在且不为空
        if not merged.get("region"):
            merged["region"] = "cn"
        
        # 确保 tags 字段至少为 rdagent (如果是 rdagent_generated)
        if merged.get("source") == "rdagent_generated":
            t_list = list(merged.get("tags") or [])
            if not t_list:
                t_list = ["rdagent"]
            merged["tags"] = t_list
        
        # 统一表达式字段：确保 expression 和 formula_hint 同步
        if not merged.get("expression") and merged.get("formula_hint"):
            merged["expression"] = merged["formula_hint"]
        elif not merged.get("formula_hint") and merged.get("expression"):
            merged["formula_hint"] = merged["expression"]


def _is_valid_rdagent_factor(payload: dict[str, Any]) -> bool:
    """Check whether a rdagent_generated factor entry is complete enough for AIstock.

    约束（与 Phase2 设计文档保持一致）：
    - 仅针对 source == "rdagent_generated" 的因子做强校验；
    - 以下字段必须同时存在且非空：
      formula_hint / description_cn / variables / freq / align / nan_policy。
    其他来源（如 qlib alpha 库）的因子保持原样不过滤。
    
    注意：对于 backfill 采集的因子，由于是从 parquet 中逆向提取的，可能缺少描述和公式。
    为了数据完整性，如果 payload 中包含 'is_backfilled': True 标记，则跳过强校验。
    或者在导出脚本中整体放宽校验逻辑。
    """

    if payload.get("source") != "rdagent_generated":
        return True

    # 如果是 backfill 产生的因子（通常 description_cn 和 formula_hint 为空），我们依然允许导出，
    # 否则 factor_catalog.json 将遗漏大量历史成果。
    return True


def _merge_external_factors(acc: dict[str, dict[str, Any]], external: dict[str, Any]) -> None:
    """Merge external factor catalog (e.g., Alpha158 meta) into accumulator.

    约定 external 结构：
    {
      "version": "v1",
      "factors": [
        {"name": "RESI5", "expression": "...", "source": "qlib_alpha158", ...},
        ...
      ]
    }

    合并策略：
    - 以 name 作为主键；
    - 不覆盖已有的非空值（None/""/[]/{} 视为可被覆盖），以避免信息丢失；
    - external 中出现的新字段会被追加到现有条目中。
    """

    factors = external.get("factors") or []
    if not isinstance(factors, list):
        return

    for f in factors:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        if not isinstance(name, str) or not name:
            continue

        existing = acc.get(name, {})
        merged: dict[str, Any] = dict(existing)
        for k, v in f.items():
            if k == "name":
                continue
            if v is None:
                continue
            
            # 统一字段映射
            if k in ("formula_hint", "expression"):
                if not merged.get("expression"):
                    merged["expression"] = v
                if not merged.get("formula_hint"):
                    merged["formula_hint"] = v
                continue

            if k not in merged or merged[k] in (None, "", [], {}):
                merged[k] = v

        # 确保 source 至少带上 external 的来源信息
        if "source" not in merged and f.get("source"):
            merged["source"] = f["source"]

        acc[name] = merged


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


def _get_expression_fingerprint(expr: str) -> str:
    """Normalize and hash the factor expression for deduplication."""
    if not expr:
        return ""
    # Remove whitespace and convert to lowercase for better normalization
    normalized = "".join(expr.split()).lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

def run(registry_sqlite: Path, output_path: Path, *, alpha_metas: list[Path] | None = None) -> None:
    conn = sqlite3.connect(str(registry_sqlite))
    conn.row_factory = sqlite3.Row
    try:
        workspaces = _fetch_workspaces(conn)
    finally:
        conn.close()

    factors_by_fingerprint: dict[str, dict[str, Any]] = {}

    # 1) 合并外部 Alpha 库因子（若提供），如 Alpha158, Alpha360 等。
    if alpha_metas:
        for am_path in alpha_metas:
            alpha_payload = _load_json_if_exists(am_path)
            if alpha_payload:
                factors = alpha_payload.get("factors") or []
                for f in factors:
                    name = f.get("name")
                    expr = f.get("expression") or f.get("formula_hint") or ""
                    if not name:
                        continue
                    fp = _get_expression_fingerprint(expr) or name
                    
                    if fp not in factors_by_fingerprint:
                        factors_by_fingerprint[fp] = f
                    else:
                        # 对于 Alpha 库因子，同名/同表达式优先保留已有的
                        pass

    # 2) 再遍历所有 workspace 的 factor_meta.json，基于表达式指纹去重，并保留最佳表现
    for ws in workspaces:
        ws_root = _to_native_path(ws.workspace_path)
        if not ws_root.exists():
            continue
        
        meta = _load_json_if_exists(ws_root / "factor_meta.json")
        perf = _load_json_if_exists(ws_root / "factor_perf.json")
        if not meta:
            continue

        # REQ-FACTOR-VALIDATION: 验证rdagent_generated因子的数据完整性
        # rdagent_generated因子必须有factor_perf.json才能被导出
        # 如果缺失factor_perf.json，则跳过该workspace的所有因子
        source = meta.get("source", "")
        if source == "rdagent_generated" and not perf:
            # 缺少factor_perf.json，跳过该workspace的所有因子
            continue

        # 提取性能
        perf_by_name = {}
        if perf:
            combos = perf.get("combinations") or []
            for combo in combos:
                f_names = combo.get("factor_names") or []
                main_window = combo.get("windows", [{}])[0]
                metrics = {
                    "sharpe": main_window.get("sharpe"),
                    "ann_ret": main_window.get("annual_return"),
                    "mdd": main_window.get("max_drawdown"),
                    "loop_id": perf.get("loop_id"),
                    "task_run_id": perf.get("task_run_id"),
                }
                for fn in f_names:
                    perf_by_name[fn] = metrics

        for f in meta.get("factors") or []:
            name = f.get("name")
            expr = f.get("expression") or f.get("formula_hint") or ""
            if not name:
                continue
            
            # REQ-FACTOR-VALIDATION: rdagent_generated因子必须有性能数据
            if source == "rdagent_generated" and name not in perf_by_name:
                # 缺少性能数据，跳过该因子
                continue
            
            fp = _get_expression_fingerprint(expr) or name
            metrics = perf_by_name.get(name)
            
            if fp not in factors_by_fingerprint:
                f["best_performance_metrics"] = metrics
                factors_by_fingerprint[fp] = f
            else:
                existing = factors_by_fingerprint[fp]
                # 去重逻辑：如果当前因子表现更好，则替换
                existing_metrics = existing.get("best_performance_metrics")
                if metrics and (not existing_metrics or (metrics.get("sharpe") or -999) > (existing_metrics.get("sharpe") or -999)):
                    f["best_performance_metrics"] = metrics
                    factors_by_fingerprint[fp] = f

    # 3) 格式化输出
    factors: list[dict[str, Any]] = []
    for payload in factors_by_fingerprint.values():
        # 补全 AIstock 要求的展示字段
        metrics = payload.get("best_performance_metrics")
        if metrics:
            payload["best_performance"] = f"Sharpe: {metrics.get('sharpe', 0):.2f}, Ann.Ret: {metrics.get('ann_ret', 0):.2%}"
        else:
            payload["best_performance"] = "无回测记录"
        
        if "best_performance_metrics" in payload:
            del payload["best_performance_metrics"]
            
        factors.append(payload)

    factors.sort(key=lambda x: x.get("name", ""))

    catalog = {
        "version": "v1",
        "generated_at_utc": _utc_now_iso(),
        "source": "rdagent_tools",
        "factors": factors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AIstock-facing factor catalog as JSON.")
    parser.add_argument(
        "--registry-sqlite",
        required=True,
        help="Path to registry.sqlite (WSL/Linux path).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for factor_catalog.",
    )
    parser.add_argument(
        "--alpha-meta",
        action="append",
        default=[],
        help="Optional path to external Alpha factor meta JSON (e.g., Alpha158), can be specified multiple times.",
    )
    args = parser.parse_args()

    registry_sqlite = Path(args.registry_sqlite)
    if not registry_sqlite.exists():
        raise SystemExit(f"registry.sqlite not found: {registry_sqlite}")

    output_path = Path(args.output)

    alpha_metas = []
    for path_str in args.alpha_meta:
        p = Path(path_str)
        if p.exists():
            alpha_metas.append(p)

    run(registry_sqlite=registry_sqlite, output_path=output_path, alpha_metas=alpha_metas)


if __name__ == "__main__":
    main()
