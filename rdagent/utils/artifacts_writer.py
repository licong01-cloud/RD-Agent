import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _metrics_summary(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Helper to extract key metrics summary for loop metadata."""
    if not metrics:
        return {}
    summary = {}
    for k in ["annualized_return", "ann_return", "max_drawdown", "mdd", "sharpe", "multi_score"]:
        if k in metrics:
            summary[k] = metrics[k]
    return summary


def _rel(p: Path, ws_root: Path) -> str:
    """Return path relative to workspace root when possible."""
    try:
        return str(p.relative_to(ws_root))
    except Exception:
        return str(p)


def _sync_factor_impl_to_shared_lib(
    ws_root: Path,
    factor_meta_payload: dict[str, Any],
    shared_lib_root: str | None = None,
) -> dict[str, Any] | None:
    """Synchronize factor implementations to a shared library and update metadata."""
    try:
        import os
        if shared_lib_root is None:
            shared_lib_root = os.getenv("RD_FACTORS_LIB_ROOT", "F:\\Dev\\rd-factors-lib")
            
        shared_root = Path(shared_lib_root)
        can_sync = shared_root.exists()
        gen_py = shared_root / "rd_factors_lib" / "generated.py" if can_sync else None
        if can_sync and gen_py and not gen_py.parent.exists():
            can_sync = False

        factors = factor_meta_payload.get("factors") or []
        if not factors:
            return factor_meta_payload

        code_map:dict[str, str] = {}
        if ws_root.exists():
            for py_file in ws_root.rglob("*.py"):
                try:
                    # Skip irrelevant directories to speed up scanning
                    if any(part in ("mlruns", ".git", "__pycache__", "data", "result", "node_modules", ".venv", "site-packages") for part in py_file.parts):
                        continue

                    # Skip common non-source files
                    if py_file.name in ("read_exp_res.py", "runtime_info.py", "setup.py", "generated.py", "__init__.py"):
                        continue
                        
                    content = py_file.read_text(encoding="utf-8", errors="replace")
                    for f in factors:
                        name = f.get("name")
                        if not name:
                            continue
                        
                        # Try to find class or function definition
                        # Pattern 1: class Name...
                        # Pattern 2: def Name...
                        # Pattern 3: def calculate_Name... (common in some templates)
                        patterns = [
                            rf"(class|def)\s+{re.escape(name)}\b",
                            rf"(class|def)\s+calculate_{re.escape(name)}\b"
                        ]
                        for pattern in patterns:
                            match_obj = re.search(pattern, content)
                            if match_obj:
                                code_map[name] = content
                                f["_matched_name"] = match_obj.group(0).split()[-1]
                                break
                except Exception:
                    continue

        existing_content = ""
        if can_sync and gen_py and gen_py.exists():
            try:
                existing_content = gen_py.read_text(encoding="utf-8")
            except Exception:
                existing_content = ""
        
        new_content_lines: list[str] = []
        updated_factors = []

        for f in factors:
            name = f.get("name")
            if not name:
                updated_factors.append(f)
                continue

            f["impl_module"] = "rd_factors_lib.generated"
            f["impl_func"] = name
            f["impl_version"] = datetime.now(timezone.utc).strftime("%Y%m%d")

            matched_name = f.pop("_matched_name", name)
            if name in code_map:
                code = code_map[name]
                # Better block extraction: find the definition and all subsequent indented lines
                # OR until next class/def at same indentation level
                pattern = rf"(?:^|\n)((?:class|def)\s+{re.escape(matched_name)}[\s\S]*?)(?=\n(?:class|def|if __name__ ==|\s*#|\s*$)|\Z)"
                match = re.search(pattern, code, re.MULTILINE)
                
                if not match:
                    # Fallback to a simpler search if multiline match fails
                    match = re.search(rf"((class|def)\s+{re.escape(matched_name)}.*)", code)

                if match:
                    body = match.group(1).strip()
                    is_class = "class " in match.group(0) # Simple check for class vs def

                    if is_class:
                        wrapper_func_name = f"factor_{name}"
                        f["interface_info"] = {
                            "type": "class",
                            "standard_wrapper": wrapper_func_name
                        }
                        
                        if can_sync and gen_py:
                            # Check if already exists in generated.py
                            if f"class {matched_name}" not in existing_content and f"def {matched_name}" not in existing_content:
                                if not existing_content.endswith("\n\n") and existing_content:
                                    if not new_content_lines:
                                        new_content_lines.append("\n")
                                 
                                new_content_lines.append(body + "\n")
                                 
                                if f"def {wrapper_func_name}" not in existing_content:
                                    wrapper_code = (
                                        f"\ndef {wrapper_func_name}(df: pd.DataFrame) -> pd.Series:\n"
                                        f"    \"\"\"Standard wrapper for {matched_name} class.\"\"\"\n"
                                        f"    return {matched_name}().fit_transform(df)\n"
                                    )
                                    new_content_lines.append(wrapper_code)
                    else:
                        f["interface_info"] = {"type": "function"}
                        if can_sync and gen_py:
                            if f"def {matched_name}" not in existing_content:
                                if not existing_content.endswith("\n\n") and existing_content:
                                    if not new_content_lines:
                                        new_content_lines.append("\n")
                                new_content_lines.append(body + "\n")
                else:
                    f["interface_info"] = {"type": "function"}
            else:
                f["interface_info"] = {"type": "function"}
            
            updated_factors.append(f)


        if can_sync and gen_py and new_content_lines:
            with gen_py.open("a", encoding="utf-8") as f_out:
                f_out.write("\n" + "\n".join(new_content_lines))

        factor_meta_payload["factors"] = updated_factors
        return factor_meta_payload
    except Exception:
        return factor_meta_payload


def _sync_strategy_impl_to_shared_lib(
    ws_root: Path,
    shared_lib_root: str | None = None,
) -> dict[str, Any] | None:
    """REQ-STRATEGY-P3-001: Synchronize strategy template to a shared library as Python code.
    REQ-STRATEGY-P3-020: Extract portfolio_config for strategy configuration schema.
    """
    try:
        import os
        import yaml
        if shared_lib_root is None:
            shared_lib_root = os.getenv("RD_STRATEGIES_LIB_ROOT", "F:\\Dev\\rd-strategies-lib")

        shared_root = Path(shared_lib_root)
        if not shared_root.exists():
            return None

        gen_py = shared_root / "rd_strategies_lib" / "generated.py"
        if not gen_py.parent.exists():
            gen_py.parent.mkdir(parents=True, exist_ok=True)

        yaml_files = list(ws_root.glob("*.yaml")) + list(ws_root.glob("*.yml"))
        strategy_config = None
        portfolio_config = None

        for yf in yaml_files:
            try:
                with yf.open("r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if not isinstance(content, dict):
                        continue
                    # 增强策略检测逻辑
                    if "strategy" in content:
                        strategy_config = content["strategy"]
                        # Extract portfolio_config (REQ-STRATEGY-P3-020)
                        if "portfolio" in content:
                            portfolio_config = content["portfolio"]
                        break
                    elif "port_analysis_config" in content and "strategy" in content["port_analysis_config"]:
                        strategy_config = content["port_analysis_config"]["strategy"]
                        if "portfolio" in content["port_analysis_config"]:
                            portfolio_config = content["port_analysis_config"]["portfolio"]
                        break
                    elif "task" in content and "record" in content["task"]:
                        for record in content["task"]["record"]:
                            if record.get("class") == "PortAnaRecord" and "config" in record.get("kwargs", {}):
                                strategy_config = record["kwargs"]["config"].get("strategy")
                                if strategy_config:
                                    # Try to extract portfolio from config
                                    if "portfolio" in record["kwargs"]["config"]:
                                        portfolio_config = record["kwargs"]["config"]["portfolio"]
                                    break
                        if strategy_config:
                            break
            except Exception:
                continue

        # Generate default portfolio_config if not found (REQ-STRATEGY-P3-020)
        if portfolio_config is None:
            portfolio_config = {
                "signal_config": {
                    "top_k": 50,
                    "min_score": 0.5,
                    "max_positions": 50,
                    "score_field": "score"
                },
                "weight_config": {
                    "method": "equal_weight",
                    "max_single_weight": 0.05
                },
                "rebalance_config": {
                    "freq": "1d",
                    "rebalance_threshold": 0.1
                },
                "risk_config": {
                    "max_drawdown": 0.2,
                    "max_single_loss": 0.05
                }
            }

        if strategy_config:
            # 生成 Python 包装代码
            strategy_name = f"strategy_{ws_root.name.replace('-', '_')}"
            config_str = json.dumps(strategy_config, ensure_ascii=False, indent=4)
            py_code = (
                f"\ndef get_{strategy_name}_config():\n"
                f"    \"\"\"Generated from {ws_root.name} strategy template.\"\"\"\n"
                f"    return {config_str}\n"
            )

            with gen_py.open("a", encoding="utf-8") as f_out:
                f_out.write(py_code)

            return {
                "strategy_name": strategy_name,
                "impl_module": "rd_strategies_lib.generated",
                "impl_func": f"get_{strategy_name}_config",
                "python_implementation": {
                    "module": "rd_strategies_lib.generated",
                    "func": f"get_{strategy_name}_config"
                },
                "portfolio_config": portfolio_config
            }

        # 即使没找到 YAML 也要返回一个基础结构以支持补录生成 strategy_meta.json
        return {
            "strategy_name": f"strategy_{ws_root.name.replace('-', '_')}",
            "impl_module": "rd_strategies_lib.generated",
            "impl_func": f"get_placeholder_config",
            "python_implementation": {
                "module": "rd_strategies_lib.generated",
                "func": "get_placeholder_config"
            },
            "portfolio_config": portfolio_config
        }
    except Exception:
        return None

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Best-effort JSON writer used by artifact generation code."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        # JSON 写盘失败不应影响主流程
        pass

def _build_factor_meta_dict(
    *,
    factors: list[dict[str, Any]] | None,
    task_run_id: str,
    loop_id: int,
    created_at_utc: str,
) -> dict[str, Any]:
    """Construct the payload of factor_meta.json (schema v1).

    This helper focuses on enforcing a stable envelope and basic fields.
    The actual factor list can be composed by callers based on available
    experiment context.

    REQ-FACTOR-P3-020: Add input_schema for factor input data validation.
    """

    factors = factors or []
    exp_id = f"{task_run_id}/{loop_id}"
    normalized: list[dict[str, Any]] = []
    for f in factors:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        if not name:
            continue

        # variables 字段：直接透传调用方提供的结构（通常来自 FactorTask.variables 或日志）；
        # 若不存在则使用空 dict，保持 schema 稳定且不丢信息。
        raw_vars = f.get("variables")
        if raw_vars is None:
            variables: dict[str, Any] = {}
        elif isinstance(raw_vars, dict):
            variables = raw_vars
        else:
            # 兼容历史上传入字符串/列表等情况，统一封装到 value 字段，避免信息丢失。
            variables = {"value": raw_vars}

        # 频率/对齐/NaN 策略：从调用方透传，若缺失则按 Phase2 约定使用默认值。
        freq = f.get("freq") or "1d"
        align = f.get("align") or "close"
        nan_policy = f.get("nan_policy") or "dataservice_default"

        # 衍生标签：基于因子名做启发式分类 (如 Alpha158 / Alpha360)
        tags = list(f.get("tags", []) or [])
        if "alpha158" in name.lower() and "alpha158" not in tags:
            tags.append("alpha158")
        if "alpha360" in name.lower() and "alpha360" not in tags:
            tags.append("alpha360")
        if f.get("source", "rdagent_generated") == "rdagent_generated" and not tags:
            tags.append("rdagent")

        formula_hint = f.get("formula_hint") or f.get("expression") or ""

        # Generate input_schema (REQ-FACTOR-P3-020)
        input_schema = _generate_input_schema(name, tags)

        item = {
            "name": name,
            "source": f.get("source", "rdagent_generated"),
            "region": f.get("region", "cn"),  # 默认 A 股 cn
            "description_cn": f.get("description_cn", ""),
            "formula_hint": formula_hint,
            "expression": formula_hint,  # 统一双向对齐
            "created_at_utc": f.get("created_at_utc", created_at_utc),
            "experiment_id": f.get("experiment_id", exp_id),
            "tags": tags,
            "variables": variables,
            "freq": freq,
            "align": align,
            "nan_policy": nan_policy,
            "input_schema": input_schema,
        }
        normalized.append(item)

    return {
        "version": "v1",
        "task_run_id": task_run_id,
        "loop_id": loop_id,
        "created_at_utc": created_at_utc,
        "factors": normalized,
    }


def _generate_input_schema(factor_name: str, tags: list[str]) -> dict[str, Any]:
    """Generate input_schema for factor based on factor name and tags.

    REQ-FACTOR-P3-020: Provide data structure definition for factor calculation.
    """
    import re

    # Determine factor type
    is_technical = any(tag in tags for tag in ["alpha158", "alpha360", "rdagent"])
    is_fundamental = any(kw in factor_name.lower() for kw in ["pe", "pb", "ps", "pcf", "market_cap", "turnover"])
    is_moneyflow = any(kw in factor_name.lower() for kw in ["moneyflow", "net_inflow", "net_outflow"])

    # Set required fields based on factor type
    if is_fundamental:
        required_fields = ["pe", "pb", "market_cap", "turnover_rate"]
        optional_fields = ["ps", "pcf", "roe", "roa"]
    elif is_moneyflow:
        required_fields = ["net_inflow", "net_outflow", "money_flow"]
        optional_fields = ["net_inflow_rate", "net_outflow_rate"]
    else:
        # Default: technical factor
        required_fields = ["open", "high", "low", "close", "volume"]
        optional_fields = ["amount", "pct_chg", "turnover_rate"]

    # Infer lookback_days from factor name (e.g., "10D" -> 10, "20D" -> 20)
    lookback_days = 10  # Default
    match = re.search(r"(\d+)D", factor_name)
    if match:
        lookback_days = int(match.group(1))

    return {
        "required_fields": required_fields,
        "optional_fields": optional_fields,
        "lookback_days": lookback_days,
        "index_type": "MultiIndex(datetime, instrument)",
        "description": f"因子计算所需的输入数据结构定义 (lookback: {lookback_days}天)"
    }


def _build_factor_perf_dict(
    *,
    factors_perf: list[dict[str, Any]] | None,
    combinations: list[dict[str, Any]] | None,
    task_run_id: str,
    loop_id: int,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Construct the payload of factor_perf.json (schema v1).

    The caller is responsible for preparing per-factor metrics and
    combination stats; this helper only normalizes envelope fields and
    ensures basic structure consistency with the design document.
    """

    def _norm_list(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not items:
            return out
        for it in items:
            if isinstance(it, dict):
                out.append(it)
        return out

    return {
        "version": "v1",
        "task_run_id": task_run_id,
        "loop_id": loop_id,
        "generated_at_utc": generated_at_utc,
        "factors": _norm_list(factors_perf),
        "combinations": _norm_list(combinations),
    }


def _build_feedback_dict(
    *,
    decision: bool | None,
    hypothesis: str | None,
    summary: dict[str, Any] | None,
    task_run_id: str,
    loop_id: int,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Construct the payload of feedback.json (schema v1).

    The summary dict can already be aggregated from execution/value/shape
    feedback; this helper just applies defaults and adds envelope fields.
    """

    summary = summary or {}
    limitations = summary.get("limitations") or []
    code_critic = summary.get("code_critic") or []

    return {
        "version": "v1",
        "task_run_id": task_run_id,
        "loop_id": loop_id,
        "generated_at_utc": generated_at_utc,
        "decision": bool(decision) if decision is not None else None,
        "hypothesis": hypothesis or "",
        "summary": {
            "execution": summary.get("execution", ""),
            "value_feedback": summary.get("value_feedback", ""),
            "shape_feedback": summary.get("shape_feedback", ""),
            "code_critic": list(code_critic),
            "limitations": list(limitations),
        },
    }


def _build_feedback_from_hypothesis_feedback(
    *,
    feedback_obj: Any | None,
    exp_obj: Any | None,
    task_run_id: str,
    loop_id: int,
) -> dict[str, Any] | None:
    """Helper to convert a HypothesisFeedback-like object into feedback.json payload.

    This follows the contract specified in the Phase 2 design doc (section 3.3.1):

    - feedback_obj.decision -> decision
    - feedback_obj.new_hypothesis (or exp_obj.hypothesis.hypothesis) -> hypothesis
    - feedback_obj.observations -> summary.execution
    - feedback_obj.hypothesis_evaluation -> summary.value_feedback
    - feedback_obj.reason -> appended to shape/value feedback (callers may decide)

    For now this helper only constructs an in-memory dict and does not write any
    files or touch the registry. Callers may choose to further enrich the
    `summary` dict with `limitations` / `code_critic` if such fields are
    available on the feedback object.
    """

    if feedback_obj is None:
        return None

    generated_at_utc = datetime.now(timezone.utc).isoformat()

    decision = getattr(feedback_obj, "decision", None)
    new_hypothesis = getattr(feedback_obj, "new_hypothesis", None)
    base_hypothesis = None
    try:
        if exp_obj is not None and getattr(exp_obj, "hypothesis", None) is not None:
            base_hypothesis = getattr(exp_obj.hypothesis, "hypothesis", None)
    except Exception:
        base_hypothesis = None

    hypothesis_text = new_hypothesis or base_hypothesis or ""

    observations = getattr(feedback_obj, "observations", "") or ""
    hypothesis_eval = getattr(feedback_obj, "hypothesis_evaluation", "") or ""
    reason = getattr(feedback_obj, "reason", "") or ""

    # 鑻ュ弽棣堝璞′笂宸茬粡鎸傝浇浜嗙粨鏋勫寲鐨?limitations / code_critic锛屽垯鐩存帴閫忎紶锛屼繚璇佷俊鎭笉琚簿绠€銆?
    raw_limitations = getattr(feedback_obj, "limitations", None)
    raw_code_critic = getattr(feedback_obj, "code_critic", None)

    limitations = list(raw_limitations or [])
    code_critic = list(raw_code_critic or [])

    summary: dict[str, Any] = {
        "execution": observations,
        "value_feedback": hypothesis_eval,
        # shape_feedback 鍏堢畝鍗曞鐢?reason锛屽悗缁彲鍦ㄥ叿浣撳疄鐜颁腑缁嗗寲鎷嗗垎
        "shape_feedback": reason,
        "limitations": limitations,
        "code_critic": code_critic,
    }

    return _build_feedback_dict(
        decision=decision,
        hypothesis=hypothesis_text,
        summary=summary,
        task_run_id=task_run_id,
        loop_id=loop_id,
        generated_at_utc=generated_at_utc,
    )


def _build_factor_meta_from_experiment(
    *,
    exp_obj: Any | None,
    task_run_id: str,
    loop_id: int,
) -> dict[str, Any] | None:
    """Best-effort helper to extract factor_meta from a FactorExperiment-like object.

    璇存槑锛?
    - 浠呬緷璧栦簬 `exp_obj.sub_tasks` 涓殑鍥犲瓙浠诲姟淇℃伅锛?
    - 涓嶈闂?workspace 鎴栨枃浠剁郴缁燂紱
    - 浠呮瀯閫犲唴瀛樹腑鐨?payload dict锛屼笉鍐欐枃浠躲€佷笉瑙﹁揪 registry銆?

    鐩爣锛?
    - 涓哄悗缁?`factor_meta.json` 钀界洏鍑嗗绋冲畾鐨勬彁鍙栭€昏緫锛?
    - 瀛楁鍚箟涓?Phase 2 鏂囨。 3.1 淇濇寔涓€鑷达細
      - name / source / description_cn / formula_hint / tags / created_at_utc / experiment_id銆?
    """

    if exp_obj is None:
        return None

    sub_tasks = getattr(exp_obj, "sub_tasks", None)
    if not sub_tasks:
        return None

    factors: list[dict[str, Any]] = []
    for t in sub_tasks:
        # 浠呭鐞?FactorTask 椋庢牸鐨勪换鍔★細鍏峰 factor_name / factor_description / factor_formulation
        name = getattr(t, "factor_name", None) or getattr(t, "name", None)
        if not name:
            continue

        desc = getattr(t, "factor_description", None) or getattr(t, "description", "")
        formulation = getattr(t, "factor_formulation", "")
        # variables 鐩墠浠呯敤浜?tags 鐨勫惎鍙戝紡琛ュ厖锛屽悗缁彲鍦ㄥ洜瀛愬簱涓繘涓€姝ヨВ鏋?
        variables = getattr(t, "variables", None)

        factor_item: dict[str, Any] = {
            "name": name,
            "description_cn": desc,
            "formula_hint": formulation,
            # Phase 2 鍏堢粺涓€瑙嗕负 rdagent_generated锛屾湭鏉ュ彲鏍规嵁鏉ユ簮鍖哄垎 alpha/external
            "source": "rdagent_generated",
            "region": "cn",
            # tags 鏆傜暀绌哄垪琛紝鍚庣画鍙粨鍚堝彉閲忓悕/妯℃澘鑷姩濉厖
            "tags": [],
            # 鍥犲瓙璁＄畻棰戠巼涓庡榻愯鍒欙紙Phase 2 榛樿锛欰 鑲℃棩绾裤€佹敹鐩樺榻愶級
            "freq": "1d",
            "align": "close",
            # 鏁板€肩ǔ瀹氭€?NaN 澶勭悊绛栫暐锛氶伒寰暟鎹湇鍔″眰缁熶竴缂哄け鍊艰鑼?
            "nan_policy": "dataservice_default",
        }

        if variables is not None:
            factor_item["variables"] = variables

        # created_at_utc / experiment_id 鐢?_build_factor_meta_dict 缁熶竴琛ュ厖
        factors.append(factor_item)

    if not factors:
        return None

    created_at_utc = datetime.now(timezone.utc).isoformat()
    return _build_factor_meta_dict(
        factors=factors,
        task_run_id=task_run_id,
        loop_id=loop_id,
        created_at_utc=created_at_utc,
    )


def _build_factor_perf_from_metrics(
    *,
    metrics: dict[str, Any] | None,
    task_run_id: str,
    loop_id: int,
    factor_names: list[str] | None = None,
    window_name: str | None = None,
    factors_perf: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build a factor_perf-style payload from aggregated metrics.

    说明：
    - 保留 metrics 字典中的 *全部* 指标，不做字段级精简；
    - 同时在 window 层提供若干常用 summary 字段（annual_return / max_drawdown / sharpe），
      以兼容现有消费方；
    - `factor_names` 用于关联该回测指标对应的因子列表。
    """

    if not metrics:
        return None

    generated_at_utc = datetime.now(timezone.utc).isoformat()

    # 从已有 metrics 字典中提取基础指标，缺失时置 None，同时保留完整 metrics 以避免信息丢失。
    ann_ret = metrics.get("annualized_return") or metrics.get("ann_return")
    mdd = metrics.get("max_drawdown") or metrics.get("mdd")
    sharpe = metrics.get("sharpe") or metrics.get("multi_score")

    window_payload: dict[str, Any] = {
        "name": window_name or "main_window",
        "annual_return": ann_ret,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        # 将完整 metrics 字典嵌入 window 下，保证所有聚合指标比特均可被消费。
        "metrics": dict(metrics),
    }

    combo = {
        "name": "main",  # 单一主组合，后续可根据具体实验命名
        "factor_names": factor_names or [],
        "windows": [window_payload],
    }

    return _build_factor_perf_dict(
        factors_perf=factors_perf or [],
        combinations=[combo],
        task_run_id=task_run_id,
        loop_id=loop_id,
        generated_at_utc=generated_at_utc,
    )


def _load_qlib_res_metrics(csv_path: Path) -> dict[str, Any] | None:
    """Parse qlib_res.csv into a flat metrics dict.

    璇存槑锛?
    - 灏濊瘯瑙ｆ瀽浜岀淮 CSV锛坢etric_name,value锛夛紝蹇界暐绌鸿鍜岄潪娉曡锛?
    - 瀵规暟鍊煎瓧娈靛仛 float 杞崲锛屽け璐ユ椂鍘熸牱淇濈暀涓哄瓧绗︿覆锛?
    - 浠讳綍寮傚父瑙嗕负鏃犳寚鏍囷紝杩斿洖 None锛岄伩鍏嶅奖鍝嶄富娴佺▼銆?
    """

    if not csv_path.exists():
        return None

    try:
        import csv
    except Exception:
        return None

    metrics: dict[str, Any] = {}
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                key = (row[0] or "").strip()
                if not key:
                    continue
                raw_val = row[1]
                val: Any
                try:
                    # 灏濊瘯瑙ｆ瀽涓?float锛岃В鏋愬け璐ュ垯淇濈暀鍘熷瀛楃涓?
                    val = float(raw_val)
                except Exception:
                    val = raw_val
                metrics[key] = val
    except Exception:
        return None

    return metrics or None


def _enrich_metrics_with_qlib_res(ws_root: Path, base_metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """Merge in-memory metrics with qlib_res.csv metrics and derive canonical KPIs.

    鐩爣锛?
    - 鍙 workspace 涓瓨鍦?qlib_res.csv锛屽氨浠庝腑鎶藉彇 annualized_return / max_drawdown / IC 绛夛紱
    - 浼樺厛淇濈暀璋冪敤鏂瑰凡鏈夌殑瀛楁鍊硷紝浠呭湪缂哄け鏃朵娇鐢?qlib_res 涓殑鏁版嵁锛?
    - 杩斿洖鍚堝苟鍚庣殑 metrics 瀛楀吀锛岀敤浜庡啓鍏?factor_perf.json / summary / registry銆?
    """

    merged: dict[str, Any] = {}
    if base_metrics:
        merged.update(base_metrics)

    # 浠?qlib_res.csv 涓鍙栧畬鏁存寚鏍?
    qlib_res = ws_root / "qlib_res.csv"
    qlib_metrics = _load_qlib_res_metrics(qlib_res)
    if qlib_metrics:
        # 灏?qlib_res 涓殑鍘熷鎸囨爣鍚嶇О鏁翠綋骞跺叆 metrics锛岄伩鍏嶄俊鎭涪澶?
        for k, v in qlib_metrics.items():
            merged.setdefault(k, v)

        # 鏄犲皠鍑犱釜鍏抽敭涓氬姟鎸囨爣鍒拌鑼冨瓧娈靛悕锛堜粎鍦ㄥ皻鏈～鍏呮椂锛?
        if "annualized_return" not in merged:
            merged["annualized_return"] = (
                qlib_metrics.get("1day.excess_return_without_cost.annualized_return")
                or qlib_metrics.get("1day.excess_return_with_cost.annualized_return")
            )

        if "max_drawdown" not in merged:
            merged["max_drawdown"] = (
                qlib_metrics.get("1day.excess_return_without_cost.max_drawdown")
                or qlib_metrics.get("1day.excess_return_with_cost.max_drawdown")
            )

        # qlib 鐨?information_ratio 鍦ㄦ澶勮浣?Sharpe 鍨嬫寚鏍囦娇鐢?
        if "sharpe" not in merged:
            merged["sharpe"] = (
                qlib_metrics.get("1day.excess_return_without_cost.information_ratio")
                or qlib_metrics.get("1day.excess_return_with_cost.information_ratio")
            )

        if "ic" not in merged:
            merged["ic"] = qlib_metrics.get("IC")

        if "ic_ir" not in merged:
            merged["ic_ir"] = qlib_metrics.get("ICIR")

    return merged or None


def _extract_model_metadata_from_workspace(ws_root: Path) -> dict[str, Any]:
    """Extract detailed model metadata (conf, dataset, feature schema) from workspace.

    REQ-MODEL-P3-010: Ensure authoritative metadata for model reconstruction and inference.
    REQ-MODEL-P3-020: Extract preprocess_config for data preprocessing consistency.
    REQ-MODEL-P3-030: Extract actual model type from model.py (highest priority for accuracy).
    """
    import yaml
    model_metadata = {
        "model_type": None,
        "model_conf": None,
        "dataset_conf": None,
        "feature_schema": None,
        "preprocess_config": None,
    }

    # 1. Extract actual model type from model.py (highest priority for accuracy)
    model_py = ws_root / "model.py"
    if model_py.exists():
        try:
            with open(model_py, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找模型类定义
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('class ') and '(' in line and 'Model' in line:
                    # 提取类名
                    class_start = line.find('class ') + 6
                    class_end = line.find('(')
                    if class_start > 5 and class_end > class_start:
                        model_metadata["model_type"] = line[class_start:class_end].strip()
                        break
                elif line.startswith('model_cls = '):
                    # 提取model_cls变量
                    model_metadata["model_type"] = line.split('=')[1].strip()
                    break
        except Exception:
            pass

    yaml_files = list(ws_root.glob("*.yaml")) + list(ws_root.glob("*.yml"))
    for yf in yaml_files:
        try:
            with yf.open("r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if not isinstance(content, dict):
                    continue

                # Support scenarios and top-level task structures
                target_configs = [content]
                if "scenario" in content and isinstance(content["scenario"], dict):
                    target_configs.append(content["scenario"])

                for base in target_configs:
                    # 1. Extract model_conf
                    task = base.get("task")
                    if isinstance(task, dict):
                        m_conf = task.get("model")
                        if isinstance(m_conf, dict):
                            if model_metadata["model_conf"] is None:
                                model_metadata["model_conf"] = m_conf
                                # Only set model_type from YAML if not already set from model.py
                                if model_metadata["model_type"] is None and "class" in m_conf:
                                    model_metadata["model_type"] = m_conf["class"]

                        # 2. Extract dataset_conf
                        d_conf = task.get("dataset")
                        if isinstance(d_conf, dict):
                            if model_metadata["dataset_conf"] is None:
                                model_metadata["dataset_conf"] = d_conf

                            # 4. Extract preprocess_config (REQ-MODEL-P3-020)
                            if model_metadata["preprocess_config"] is None:
                                try:
                                    # Try to extract from dataset.kwargs.handler.kwargs.preprocess
                                    preprocess = d_conf.get("kwargs", {}).get("handler", {}).get("kwargs", {}).get("preprocess")
                                    if isinstance(preprocess, dict):
                                        model_metadata["preprocess_config"] = preprocess
                                except Exception:
                                    pass

                    # 3. Extract feature_schema (multiple locations)
                    if model_metadata["feature_schema"] is None:
                        # Case A: data_handler_config in base
                        dh_conf = base.get("data_handler_config")
                        if isinstance(dh_conf, dict):
                            model_metadata["feature_schema"] = dh_conf.get("feature")

                        # Case B: nested in task.dataset.kwargs.handler.kwargs
                        if model_metadata["feature_schema"] is None and isinstance(task, dict):
                            try:
                                dh_inner = task.get("dataset", {}).get("kwargs", {}).get("handler", {}).get("kwargs", {}).get("data_handler_config")
                                if isinstance(dh_inner, dict):
                                    model_metadata["feature_schema"] = dh_inner.get("feature")
                            except Exception:
                                pass

                # If we found enough info, we can stop early
                if (model_metadata["model_conf"] and model_metadata["dataset_conf"]):
                    break
        except Exception:
            continue

    # If preprocess_config is not found, use default config
    if model_metadata["preprocess_config"] is None:
        model_metadata["preprocess_config"] = {
            "normalize": "zscore",
            "fillna": "forward_fill",
            "clip": None,
            "standardize_features": True
        }

    return model_metadata


def write_loop_artifacts(
    reg: Any,
    *,
    task_run_id: str,
    scenario: str,
    log_trace_path: str,
    loop_id: int,
    step_name: str,
    action: str | None,
    status: str,
    metrics: dict[str, Any] | None,
    exp_obj: Any,
) -> None:
    """Lightweight writer for loop/workspace registry state.

    Optimized for AIstock:
    1. Skip coding stage processing.
    2. No heavy Parquet or historical data reading.
    3. Only perform sync/metadata extraction if results exist.
    """
    try:
        if step_name == "coding" or exp_obj is None:
            return

        ew = getattr(exp_obj, "experiment_workspace", None)
        ew_path = getattr(ew, "workspace_path", None) if ew is not None else None
        if ew_path is None:
            return

        ws_root = Path(str(ew_path))
        workspace_id = ws_root.name

        # 1. 检测成果是否存在 (仅做轻量级 existence check)
        has_result = False
        combined_factors = ws_root / "combined_factors_df.parquet"
        qlib_res = ws_root / "qlib_res.csv"
        ret_pkl = ws_root / "ret.pkl"

        if action == "factor":
            has_result = combined_factors.exists()
        elif action == "model":
            has_result = qlib_res.exists() and ret_pkl.exists()

        # 2. 如果有结果，进行轻量级元数据提取和同步
        enriched_metrics: dict[str, Any] = metrics or {}
        if has_result:
            # 提取现有指标，不进行重计算
            qlib_metrics = _enrich_metrics_with_qlib_res(ws_root, enriched_metrics)
            if qlib_metrics:
                enriched_metrics = qlib_metrics
            
            # 同步源码到共享库 (AIstock 运行必需)
            # 1. 首先尝试从实验对象中构建元数据 (主要包含当前 loop 产生的因子描述和源码)
            factor_meta_payload = _build_factor_meta_from_experiment(
                exp_obj=exp_obj,
                task_run_id=task_run_id,
                loop_id=loop_id,
            )
            
            # 2. 增强逻辑：从 combined_factors_df.parquet 中提取全量因子名 (确保 Alpha158/360 不缺失)
            if combined_factors.exists():
                try:
                    import pyarrow.parquet as pq
                    meta = pq.read_metadata(combined_factors)
                    all_cols = meta.schema.names
                    
                    factor_meta_payload = factor_meta_payload or {
                        "version": "v1",
                        "task_run_id": task_run_id,
                        "loop_id": loop_id,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "factors": []
                    }
                    
                    existing_names = {f.get("name") for f in factor_meta_payload.get("factors", [])}
                    for name in all_cols:
                        if name in ("datetime", "instrument", "index", "level_0", "level_1"):
                            continue
                        if name not in existing_names:
                            is_alpha = any(x in name.lower() for x in ["alpha158", "alpha360"])
                            factor_meta_payload["factors"].append({
                                "name": name,
                                "source": "sota" if is_alpha else "rdagent_generated",
                                "region": "cn",
                                "description_cn": "",
                                "formula_hint": name if is_alpha else "",  # 对于 SOTA 因子，通常名字本身就是表达式或 key
                                "tags": ["alpha158"] if "alpha158" in name.lower() else (["alpha360"] if "alpha360" in name.lower() else []),
                                "freq": "1d",
                                "align": "close",
                                "nan_policy": "dataservice_default",
                            })
                except Exception:
                    pass

            if factor_meta_payload:
                factor_meta_payload = _sync_factor_impl_to_shared_lib(
                    ws_root=ws_root,
                    factor_meta_payload=factor_meta_payload,
                ) or factor_meta_payload
                _write_json(ws_root / "factor_meta.json", factor_meta_payload)

            # 写入性能元数据 (不包含 parquet 统计)
            factor_names = [f.get("name") for f in factor_meta_payload.get("factors", [])] if factor_meta_payload else []
            factor_perf_payload = _build_factor_perf_from_metrics(
                metrics=enriched_metrics,
                task_run_id=task_run_id,
                loop_id=loop_id,
                factor_names=factor_names,
            )
            if factor_perf_payload:
                _write_json(ws_root / "factor_perf.json", factor_perf_payload)

        # 3. 更新 Registry 状态
        try:
            reg.upsert_loop(
                task_run_id=task_run_id,
                loop_id=loop_id,
                action=action,
                status=status,
                metrics=enriched_metrics,
                best_workspace_id=workspace_id if has_result else None,
                has_result=has_result,
            )
            
            reg.upsert_workspace(
                workspace_id=workspace_id,
                task_run_id=task_run_id,
                loop_id=loop_id,
                workspace_role="experiment_workspace",
                experiment_type=action,
                step_name=step_name,
                status=status,
                workspace_path=str(ws_root),
            )
        except Exception:
            pass

    except Exception:
        pass
