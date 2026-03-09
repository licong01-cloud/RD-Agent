#!/usr/bin/env python3
"""
rdagent_task_analyzer.py - Standalone diagnostic script for RDAgent task runs.

Reads pickle log files from an rdagent task and produces a structured diagnostic
report covering hypothesis evolution, metrics, timing, feedback quality, and
optionally live resource utilization.

Usage:
    python scripts/rdagent_task_analyzer.py <task_id> [--log-dir PATH] [--live-check] [--json]

No rdagent imports required.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Section 1: Pickle Loading
# ---------------------------------------------------------------------------

DEFAULT_LOG_DIR = Path(r"F:\Dev\RD-Agent-main\log")

IMPORTANT_METRICS = [
    "IC",
    "ICIR",
    "Rank IC",
    "Rank ICIR",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
    "1day.excess_return_with_cost.information_ratio",
]


class _CompatUnpickler(pickle.Unpickler):
    """Handle cross-platform Path deserialization."""

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
            return Path
        if module == "pathlib" and name in {"PurePosixPath", "PureWindowsPath"}:
            from pathlib import PurePath

            return PurePath
        return super().find_class(module, name)


def safe_pickle_load(path: Path) -> Any | None:
    """Load a pickle file with compatible unpickler; return None on failure."""
    try:
        with path.open("rb") as f:
            return _CompatUnpickler(f).load()
    except Exception:
        return None


def load_latest_session(log_dir: Path) -> tuple[Any | None, str | None, int]:
    """
    Load the session snapshot with the longest trace.hist from __session__/.

    Returns (loop_obj, session_dir_name, hist_len).
    """
    session_root = log_dir / "__session__"
    if not session_root.exists():
        return None, None, 0

    best_obj: Any | None = None
    best_dir: str | None = None
    best_hist_len = 0

    for snap in sorted(session_root.iterdir(), key=lambda p: p.name):
        if not snap.is_dir():
            continue
        # Prefer 1_coding as it tends to have the longest hist
        candidates = []
        pref = snap / "1_coding"
        if pref.exists() and pref.is_file():
            candidates.append(pref)
        for fp in snap.iterdir():
            if fp.is_file() and fp not in candidates:
                candidates.append(fp)

        for fp in candidates:
            obj = safe_pickle_load(fp)
            if obj is None:
                continue
            trace = getattr(obj, "trace", None)
            hist = getattr(trace, "hist", None) if trace else None
            n = len(hist) if isinstance(hist, list) else 0
            if n > best_hist_len:
                best_hist_len = n
                best_obj = obj
                best_dir = snap.name
    return best_obj, best_dir, best_hist_len


# ---------------------------------------------------------------------------
# Section 2: Data Extraction (per loop)
# ---------------------------------------------------------------------------

def _find_pkl_files(base: Path, sub_path: str) -> list[Path]:
    """Find all .pkl files under base/sub_path/*/*.pkl, sorted by name."""
    target = base / sub_path
    if not target.exists():
        return []
    pkls = sorted(target.glob("*/*.pkl"))
    return pkls


def _extract_loop_dirs(log_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (loop_id, loop_path) for all Loop_N dirs."""
    loops = []
    for d in log_dir.iterdir():
        m = re.match(r"^Loop_(\d+)$", d.name)
        if m and d.is_dir():
            loops.append((int(m.group(1)), d))
    loops.sort(key=lambda x: x[0])
    return loops


def extract_hypothesis(loop_dir: Path) -> dict[str, Any]:
    """Extract hypothesis info from hypothesis generation pkl."""
    pkls = _find_pkl_files(loop_dir, "direct_exp_gen/hypothesis generation")
    if not pkls:
        return {}
    obj = safe_pickle_load(pkls[-1])
    if obj is None:
        return {}
    result: dict[str, Any] = {"type": type(obj).__name__}
    result["hypothesis"] = getattr(obj, "hypothesis", None)
    result["reason"] = getattr(obj, "reason", None)
    result["concise_reason"] = getattr(obj, "concise_reason", None)
    result["action"] = getattr(obj, "action", None)
    return result


def extract_experiment(loop_dir: Path) -> dict[str, Any]:
    """Extract experiment/task info from experiment generation pkl."""
    pkls = _find_pkl_files(loop_dir, "direct_exp_gen/experiment generation")
    if not pkls:
        return {}
    obj = safe_pickle_load(pkls[-1])
    if obj is None:
        return {}
    result: dict[str, Any] = {"type": type(obj).__name__}

    # Extract sub_tasks info
    sub_tasks = getattr(obj, "sub_tasks", None) or []
    tasks_info = []
    for t in sub_tasks[:5]:
        ti: dict[str, Any] = {"type": type(t).__name__}
        ti["name"] = getattr(t, "name", None) or getattr(t, "factor_name", None) or getattr(t, "model_name", None)
        ti["description"] = getattr(t, "description", None) or getattr(t, "factor_description", None)
        # Extract model hyperparameters
        for attr in ("hyperparameters", "training_hyperparameters", "architecture"):
            val = getattr(t, attr, None)
            if val is not None:
                ti[attr] = val
        # Extract factor-specific attributes (variables, formulation)
        for attr in ("variables", "formulation", "factor_formulation"):
            val = getattr(t, attr, None)
            if val is not None:
                ti[attr] = val
        tasks_info.append(ti)
    result["tasks"] = tasks_info

    # based_experiments count
    based = getattr(obj, "based_experiments", None) or []
    result["based_experiments_count"] = len(based)

    return result


def extract_runner_result(loop_dir: Path) -> dict[str, Any]:
    """Extract metrics from runner result pkl (Experiment.result is a pd.Series)."""
    pkls = _find_pkl_files(loop_dir, "running/runner result")
    if not pkls:
        return {}
    obj = safe_pickle_load(pkls[-1])
    if obj is None:
        return {}
    result_data: dict[str, Any] = {"type": type(obj).__name__}

    exp_result = getattr(obj, "result", None)
    if exp_result is None:
        ri = getattr(obj, "running_info", None)
        if ri:
            exp_result = getattr(ri, "result", None)

    if isinstance(exp_result, pd.Series):
        metrics = {}
        for m in IMPORTANT_METRICS:
            if m in exp_result.index:
                val = exp_result[m]
                metrics[m] = float(val) if pd.notna(val) else None
        result_data["metrics"] = metrics
        # Also store all available metric keys
        result_data["all_metric_keys"] = list(exp_result.index[:20])
    elif isinstance(exp_result, dict):
        result_data["metrics"] = {k: v for k, v in exp_result.items() if k in IMPORTANT_METRICS}
    else:
        result_data["metrics"] = None
        result_data["result_type"] = type(exp_result).__name__ if exp_result is not None else "None"

    return result_data


def extract_feedback(loop_dir: Path) -> dict[str, Any]:
    """Extract feedback info from feedback pkl."""
    pkls = _find_pkl_files(loop_dir, "feedback/feedback")
    if not pkls:
        return {}
    obj = safe_pickle_load(pkls[-1])
    if obj is None:
        return {}
    result: dict[str, Any] = {"type": type(obj).__name__}
    result["decision"] = getattr(obj, "decision", None)
    result["observations"] = _truncate(getattr(obj, "observations", None), 300)
    result["hypothesis_evaluation"] = _truncate(getattr(obj, "hypothesis_evaluation", None), 200)
    result["new_hypothesis"] = _truncate(getattr(obj, "new_hypothesis", None), 200)
    result["reason"] = _truncate(getattr(obj, "reason", None), 200)
    return result


def extract_timing(loop_dir: Path) -> dict[str, dict[str, Any]]:
    """Extract timing info for each step from time_info pkl files."""
    times: dict[str, dict[str, Any]] = {}
    for step_dir in loop_dir.iterdir():
        if not step_dir.is_dir():
            continue
        ti_dir = step_dir / "time_info"
        if not ti_dir.exists():
            continue
        for pid_dir in ti_dir.iterdir():
            if not pid_dir.is_dir():
                continue
            for pkl_file in sorted(pid_dir.glob("*.pkl")):
                obj = safe_pickle_load(pkl_file)
                if not isinstance(obj, dict):
                    continue
                st = obj.get("start_time")
                et = obj.get("end_time")
                if st and et:
                    duration = (et - st).total_seconds()
                    times[step_dir.name] = {
                        "start_time": st.isoformat() if hasattr(st, "isoformat") else str(st),
                        "end_time": et.isoformat() if hasattr(et, "isoformat") else str(et),
                        "duration_seconds": round(duration, 1),
                        "duration_human": _fmt_duration(et - st),
                    }
    return times


def count_evo_loops(loop_dir: Path) -> int:
    """Count CoSTEER evo_loop_* directories in coding/."""
    coding_dir = loop_dir / "coding"
    if not coding_dir.exists():
        return 0
    return len([d for d in coding_dir.iterdir() if d.is_dir() and d.name.startswith("evo_loop_")])


def analyze_costeer_detail(loop_dir: Path) -> dict[str, Any]:
    """Analyze CoSTEER evolution detail: per-iteration factor count, LLM calls, debug retries."""
    coding_dir = loop_dir / "coding"
    if not coding_dir.exists():
        return {}
    evo_dirs = sorted(
        [d for d in coding_dir.iterdir() if d.is_dir() and d.name.startswith("evo_loop_")],
        key=lambda d: d.name,
    )
    if not evo_dirs:
        return {}

    result: dict[str, Any] = {"iterations": [], "total_evo_iters": len(evo_dirs)}
    total_llm_calls = 0

    for evo_dir in evo_dirs:
        debug_llm_dir = evo_dir / "debug_llm"
        if not debug_llm_dir.exists():
            result["iterations"].append({
                "name": evo_dir.name, "factors": 0, "llm_calls": 0, "needed_debug": 0,
            })
            continue

        by_factor: dict[str, int] = {}
        for factor_dir in debug_llm_dir.iterdir():
            if not factor_dir.is_dir():
                continue
            pid_pair = factor_dir.name
            parts = pid_pair.split("-")
            factor_pid = parts[-1] if len(parts) >= 2 else pid_pair
            call_count = len(list(factor_dir.glob("*.pkl")))
            by_factor[factor_pid] = call_count

        n_factors = len(by_factor)
        n_calls = sum(by_factor.values())
        n_needed_debug = sum(1 for v in by_factor.values() if v > 1)
        total_llm_calls += n_calls

        result["iterations"].append({
            "name": evo_dir.name,
            "factors": n_factors,
            "llm_calls": n_calls,
            "needed_debug": n_needed_debug,
        })

    result["total_llm_calls"] = total_llm_calls

    # Extract final factor implementation status from coder result
    coder_pkls = sorted(
        (coding_dir / "coder result").rglob("*.pkl") if (coding_dir / "coder result").exists() else []
    )
    if coder_pkls:
        obj = safe_pickle_load(coder_pkls[-1])
        if isinstance(obj, list):
            factors_info = []
            for item in obj:
                t = getattr(item, "target_task", None)
                if t:
                    name = getattr(t, "factor_name", "?")
                    impl = getattr(t, "factor_implementation", None)
                    factors_info.append({"name": name, "implemented": impl})
            result["final_factors"] = factors_info
            result["implemented_count"] = sum(1 for f in factors_info if f.get("implemented") is True)
            result["failed_count"] = sum(1 for f in factors_info if f.get("implemented") is False)

    return result


def extract_model_code(loop_dir: Path) -> dict[str, Any]:
    """Extract generated model.py or factor code from coder result pkl."""
    pkls = _find_pkl_files(loop_dir, "coding/coder result")
    if not pkls:
        return {}
    obj = safe_pickle_load(pkls[-1])
    if obj is None:
        return {}
    result: dict[str, Any] = {}

    # Handle list format (factor tasks store coder result as list[FactorFBWorkspace])
    if isinstance(obj, list):
        factor_files: dict[str, str] = {}
        for i, item in enumerate(obj):
            fd = getattr(item, "file_dict", {}) or {}
            t = getattr(item, "target_task", None)
            fname = (getattr(t, "factor_name", None) or getattr(t, "name", None) or f"factor_{i}") if t else f"factor_{i}"
            for key, code in fd.items():
                if isinstance(code, str) and key.endswith(".py"):
                    factor_files[f"workspace_{i}/{key}"] = code
            # Also extract variables from target_task
            if t:
                variables = getattr(t, "variables", None)
                if isinstance(variables, dict):
                    result.setdefault("task_variables", {})[fname] = variables
        if factor_files:
            result["factor_files"] = factor_files
            result["file_keys"] = list(factor_files.keys())
        return result

    sw = getattr(obj, "sub_workspace_list", None) or []
    if sw and sw[0]:
        fd = getattr(sw[0], "file_dict", {})
        result["model_py"] = fd.get("model.py")
        # Also try factor code files
        for key in fd:
            if key.endswith(".py") and key != "model.py":
                result.setdefault("factor_files", {})[key] = fd[key]
        result["file_keys"] = list(fd.keys())
    # Also check other sub_workspaces for factor tasks with multiple factors
    if len(sw) > 1:
        factor_codes = {}
        for i, w in enumerate(sw[1:], 1):
            fd_i = getattr(w, "file_dict", {})
            for key in fd_i:
                if key.endswith(".py"):
                    factor_codes[f"workspace_{i}/{key}"] = fd_i[key]
        if factor_codes:
            result.setdefault("factor_files", {}).update(factor_codes)
    # Also extract task details from the experiment obj in coder result
    tasks = getattr(obj, "sub_tasks", None) or []
    if tasks:
        t = tasks[0]
        result["model_type"] = getattr(t, "model_type", None)
        result["architecture"] = getattr(t, "architecture", None)
        hp = getattr(t, "hyperparameters", None)
        thp = getattr(t, "training_hyperparameters", None)
        if hp:
            result["hyperparameters"] = hp
        if thp:
            result["training_hyperparameters"] = thp
        # Factor-specific attributes
        result["factor_name"] = getattr(t, "factor_name", None) or getattr(t, "name", None)
        result["factor_description"] = getattr(t, "factor_description", None) or getattr(t, "description", None)
    return result


def extract_training_log(loop_dir: Path) -> dict[str, Any]:
    """Extract and parse Qlib training log from Qlib_execute_log pkl."""
    pkls = _find_pkl_files(loop_dir, "running/Qlib_execute_log")
    if not pkls:
        return {}
    obj = safe_pickle_load(pkls[-1])
    if obj is None:
        return {}
    log_text = str(obj)
    result: dict[str, Any] = {"log_length": len(log_text)}

    # Extract template rendering context (actual hyperparameters used)
    ctx_match = re.search(r"Render the template with the context:\s*(\{[^}]+\})", log_text)
    if ctx_match:
        try:
            # Parse the context dict string
            ctx_str = ctx_match.group(1).replace("'", '"')
            result["render_context"] = json.loads(ctx_str)
        except Exception:
            result["render_context_raw"] = ctx_match.group(1)

    # Extract model parameters summary
    param_match = re.search(r"GeneralPTNN parameters setting:\n(.*?)(?=\n\[\d|$)", log_text, re.DOTALL)
    if param_match:
        params = {}
        for line in param_match.group(1).split("\n"):
            line = line.strip()
            if " : " in line:
                k, v = line.split(" : ", 1)
                params[k.strip()] = v.strip()
        result["actual_params"] = params

    # Extract model size
    size_match = re.search(r"model size:\s*([\d.]+)\s*MB", log_text)
    if size_match:
        result["model_size_mb"] = float(size_match.group(1))

    # Extract train/valid sample counts
    train_match = re.search(r"Train samples:\s*(\d+)", log_text)
    valid_match = re.search(r"Valid samples:\s*(\d+)", log_text)
    if train_match:
        result["train_samples"] = int(train_match.group(1))
    if valid_match:
        result["valid_samples"] = int(valid_match.group(1))

    # Parse epoch-by-epoch loss
    epoch_pattern = re.compile(r"Epoch(\d+):\s+train\s+([\d.]+),\s+valid\s+([\d.]+)")
    epochs = []
    for m in epoch_pattern.finditer(log_text):
        epochs.append({
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "valid_loss": float(m.group(3)),
        })
    result["epochs"] = epochs
    result["total_epochs_trained"] = len(epochs)

    # Extract early stop info
    early_match = re.search(r"early stop", log_text)
    result["early_stopped"] = early_match is not None

    best_match = re.search(r"best score:\s*([\d.]+)\s*@\s*(\d+)\s*epoch", log_text)
    if best_match:
        result["best_valid_loss"] = float(best_match.group(1))
        result["best_epoch"] = int(best_match.group(2))

    # Extract epoch timestamps for timing analysis
    epoch_time_pattern = re.compile(
        r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\).*Epoch(\d+):\s+train"
    )
    epoch_times = []
    for m in epoch_time_pattern.finditer(log_text):
        epoch_times.append({
            "epoch": int(m.group(2)),
            "timestamp": m.group(1),
        })
    if len(epoch_times) >= 2:
        try:
            t0 = datetime.strptime(epoch_times[0]["timestamp"], "%Y-%m-%d %H:%M:%S,%f")
            t1 = datetime.strptime(epoch_times[1]["timestamp"], "%Y-%m-%d %H:%M:%S,%f")
            result["seconds_per_epoch"] = round((t1 - t0).total_seconds(), 1)
        except Exception:
            pass

    # Extract IC/ICIR from log
    ic_match = re.search(r"'IC':\s*[\w.]*\(([\d.e+-]+)\)", log_text)
    if ic_match:
        result["log_ic"] = float(ic_match.group(1))

    # --- LGBModel (LightGBM/GBDT) training log parsing ---
    if "LGBModel" in log_text or "train's l2:" in log_text:
        result["model_type"] = "LGBModel"

        # Parse early_stop patience: "Training until validation scores don't improve for N rounds"
        patience_match = re.search(r"don't improve for (\d+) rounds", log_text)
        if patience_match:
            result["lgb_patience"] = int(patience_match.group(1))

        # Parse iteration logs: [N]\ttrain's l2: xxx\tvalid's l2: xxx
        lgb_iter_pattern = re.compile(r"\[(\d+)\]\s+train's l2:\s*([\d.e+-]+)\s+valid's l2:\s*([\d.e+-]+)")
        lgb_iterations = []
        for m in lgb_iter_pattern.finditer(log_text):
            lgb_iterations.append({
                "iteration": int(m.group(1)),
                "train_l2": float(m.group(2)),
                "valid_l2": float(m.group(3)),
            })
        result["lgb_iterations"] = lgb_iterations

        # Parse early stopping best iteration
        best_iter_match = re.search(
            r"Early stopping, best iteration is:\s*\[(\d+)\]\s+train's l2:\s*([\d.e+-]+)\s+valid's l2:\s*([\d.e+-]+)",
            log_text,
        )
        if best_iter_match:
            result["lgb_early_stopped"] = True
            result["lgb_best_iteration"] = int(best_iter_match.group(1))
            result["lgb_best_train_l2"] = float(best_iter_match.group(2))
            result["lgb_best_valid_l2"] = float(best_iter_match.group(3))
        else:
            result["lgb_early_stopped"] = False
            # If no early stop, best is the last iteration with lowest valid
            if lgb_iterations:
                best = min(lgb_iterations, key=lambda x: x["valid_l2"])
                result["lgb_best_iteration"] = best["iteration"]
                result["lgb_best_train_l2"] = best["train_l2"]
                result["lgb_best_valid_l2"] = best["valid_l2"]

        # Total iterations trained
        if lgb_iterations:
            result["lgb_total_iterations"] = lgb_iterations[-1]["iteration"]

    return result


def extract_costeer_config() -> dict[str, Any]:
    """Extract CoSTEER configuration from config file and environment."""
    result: dict[str, Any] = {}
    # Read config.py defaults
    config_path = Path(r"F:\Dev\RD-Agent-main\rdagent\components\coder\CoSTEER\config.py")
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        max_loop_match = re.search(r"max_loop:\s*int\s*=\s*(\d+)", text)
        if max_loop_match:
            result["default_max_loop"] = int(max_loop_match.group(1))
    # Check env overrides
    for key in ("CoSTEER_max_loop", "COSTEER_MAX_LOOP", "CODER_MAX_ITER"):
        val = os.environ.get(key)
        if val:
            result[f"env_{key}"] = val
    # Check .env file
    env_path = Path(r"F:\Dev\RD-Agent-main\.env")
    if env_path.exists():
        env_text = env_path.read_text(encoding="utf-8")
        for line in env_text.split("\n"):
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            if "max_loop" in line.lower() or "max_iter" in line.lower():
                result["env_file_" + line.split("=")[0].strip()] = line.split("=", 1)[1].strip()
    return result


def extract_prompt_config() -> dict[str, Any]:
    """Extract key hyperparameter recommendations from prompts.yaml files."""
    result: dict[str, Any] = {}
    prompt_files = [
        Path(r"F:\Dev\RD-Agent-main\rdagent\scenarios\qlib\prompts.yaml"),
        Path(r"F:\Dev\RD-Agent-main\rdagent\scenarios\qlib\experiment\prompts.yaml"),
    ]
    for pf in prompt_files:
        if not pf.exists():
            continue
        text = pf.read_text(encoding="utf-8")
        key = pf.parent.name + "/" + pf.name
        info: dict[str, Any] = {}
        # Extract batch_size recommendations
        bs_tabular = re.search(r"Tabular.*?batch_size[:\s]*([\d,-]+)", text)
        bs_ts = re.search(r"TimeSeries.*?batch_size[:\s]*([\d,-]+)", text)
        if bs_tabular:
            info["tabular_batch_size_range"] = bs_tabular.group(1)
        if bs_ts:
            info["timeseries_batch_size_range"] = bs_ts.group(1)
        # Extract lr recommendations
        lr_gru = re.search(r"GRU/LSTM.*?(\d+e-\d+)\s*[~～]\s*(\d+e-\d+)", text)
        lr_tf = re.search(r"Transformer.*?(\d+e-\d+)\s*[~～]\s*(\d+e-\d+)", text)
        if lr_gru:
            info["gru_lr_range"] = f"{lr_gru.group(1)} ~ {lr_gru.group(2)}"
        if lr_tf:
            info["transformer_lr_range"] = f"{lr_tf.group(1)} ~ {lr_tf.group(2)}"
        result[key] = info

    # Check conf template default batch_size
    conf_path = Path(r"F:\Dev\RD-Agent-main\rdagent\scenarios\qlib\experiment\model_template\conf_sota_factors_model.yaml")
    if conf_path.exists():
        text = conf_path.read_text(encoding="utf-8")
        bs_match = re.search(r"batch_size:\s*\{\{.*?default\((\d+)\)", text)
        if bs_match:
            result["conf_default_batch_size"] = int(bs_match.group(1))
    return result


# ---------------------------------------------------------------------------
# Section 3: Analysis Functions
# ---------------------------------------------------------------------------

def analyze_evolution(loop_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare metrics across loops, compute deltas and trends."""
    analysis: dict[str, Any] = {"loops_with_metrics": 0, "metric_trends": {}}

    # Collect metric values per loop
    metric_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for ld in loop_data:
        metrics = ld.get("runner_result", {}).get("metrics")
        if not metrics:
            continue
        analysis["loops_with_metrics"] += 1
        loop_id = ld["loop_id"]
        for k, v in metrics.items():
            if v is not None:
                metric_series[k].append((loop_id, v))

    for metric_name, values in metric_series.items():
        if len(values) < 2:
            analysis["metric_trends"][metric_name] = {
                "values": values,
                "trend": "insufficient_data",
            }
            continue
        first_val = values[0][1]
        last_val = values[-1][1]
        delta = last_val - first_val
        # For max_drawdown, lower (less negative) is better
        if "drawdown" in metric_name.lower():
            improving = delta > 0  # less negative = improving
        else:
            improving = delta > 0
        analysis["metric_trends"][metric_name] = {
            "values": values,
            "first": first_val,
            "last": last_val,
            "delta": round(delta, 6),
            "trend": "improving" if improving else "degrading",
        }

    return analysis


def analyze_parallelism(loop_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect if coding steps across loops overlap in time (parallelism)."""
    coding_intervals: list[tuple[int, datetime, datetime]] = []
    for ld in loop_data:
        timing = ld.get("timing", {})
        coding_t = timing.get("coding", {})
        st_str = coding_t.get("start_time")
        et_str = coding_t.get("end_time")
        if st_str and et_str:
            try:
                st = datetime.fromisoformat(st_str)
                et = datetime.fromisoformat(et_str)
                coding_intervals.append((ld["loop_id"], st, et))
            except (ValueError, TypeError):
                pass

    overlaps: list[str] = []
    for i in range(len(coding_intervals)):
        for j in range(i + 1, len(coding_intervals)):
            li, si, ei = coding_intervals[i]
            lj, sj, ej = coding_intervals[j]
            if si < ej and sj < ei:
                overlaps.append(f"Loop_{li} coding overlaps with Loop_{lj} coding")

    return {
        "coding_intervals_found": len(coding_intervals),
        "overlaps_detected": len(overlaps),
        "overlaps": overlaps,
        "conclusion": "Parallel coding detected" if overlaps else "Sequential execution (no parallelism)",
    }


def analyze_propagation(session_obj: Any, loop_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify trace.hist growth and based_experiments chain."""
    result: dict[str, Any] = {}

    if session_obj is not None:
        trace = getattr(session_obj, "trace", None)
        hist = getattr(trace, "hist", None) if trace else None
        result["trace_hist_len"] = len(hist) if isinstance(hist, list) else 0
    else:
        result["trace_hist_len"] = None

    # Check based_experiments chain growth
    based_counts = []
    for ld in loop_data:
        exp = ld.get("experiment", {})
        based_counts.append(exp.get("based_experiments_count", 0))
    result["based_experiments_per_loop"] = based_counts

    if len(based_counts) >= 2:
        growing = all(based_counts[i] <= based_counts[i + 1] for i in range(len(based_counts) - 1))
        result["chain_growing"] = growing
    else:
        result["chain_growing"] = None

    return result


def analyze_feedback_quality(loop_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Check if feedback contains quantitative training data."""
    results: list[dict[str, Any]] = []
    for ld in loop_data:
        fb = ld.get("feedback", {})
        if not fb:
            results.append({"loop_id": ld["loop_id"], "has_feedback": False})
            continue

        obs = fb.get("observations", "") or ""
        eval_text = fb.get("hypothesis_evaluation", "") or ""
        combined = obs + " " + eval_text

        has_numbers = bool(re.search(r"\d+\.\d+", combined))
        has_metric_names = any(m.lower() in combined.lower() for m in ["IC", "ICIR", "annualized_return", "drawdown", "return", "sharpe"])
        has_comparison = any(w in combined.lower() for w in ["improve", "worse", "better", "degrad", "increase", "decrease"])

        results.append({
            "loop_id": ld["loop_id"],
            "has_feedback": True,
            "decision": fb.get("decision"),
            "has_quantitative_data": has_numbers,
            "has_metric_references": has_metric_names,
            "has_comparison_language": has_comparison,
        })

    return {"per_loop": results}


def analyze_hyperparameters(loop_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Track hyperparameter changes across model loops."""
    hp_history: list[dict[str, Any]] = []
    for ld in loop_data:
        exp = ld.get("experiment", {})
        tasks = exp.get("tasks", [])
        for t in tasks:
            entry: dict[str, Any] = {"loop_id": ld["loop_id"], "task_name": t.get("name")}
            for attr in ("hyperparameters", "training_hyperparameters", "architecture"):
                if attr in t:
                    entry[attr] = t[attr]
            hp_history.append(entry)

    return {"hyperparameter_history": hp_history}


def analyze_training_convergence(loop_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep analysis of training convergence per loop with model code."""
    results = []
    for ld in loop_data:
        tlog = ld.get("training_log", {})
        mcode = ld.get("model_code", {})
        has_pytorch = bool(tlog.get("epochs"))
        has_lgb = bool(tlog.get("lgb_iterations"))
        if not has_pytorch and not has_lgb:
            continue

        lid = ld["loop_id"]

        # ----- LGBModel (LightGBM/GBDT) path -----
        if has_lgb and not has_pytorch:
            info: dict[str, Any] = {"loop_id": lid}
            info["model_type"] = "LGBModel"
            iters = tlog["lgb_iterations"]
            info["lgb_total_iterations"] = tlog.get("lgb_total_iterations", iters[-1]["iteration"] if iters else 0)
            info["lgb_patience"] = tlog.get("lgb_patience")
            info["lgb_early_stopped"] = tlog.get("lgb_early_stopped", False)
            info["lgb_best_iteration"] = tlog.get("lgb_best_iteration")
            info["lgb_best_train_l2"] = tlog.get("lgb_best_train_l2")
            info["lgb_best_valid_l2"] = tlog.get("lgb_best_valid_l2")
            info["lgb_iterations"] = iters

            # First/last loss
            info["first_train_loss"] = iters[0]["train_l2"] if iters else None
            info["last_train_loss"] = iters[-1]["train_l2"] if iters else None
            info["first_valid_loss"] = iters[0]["valid_l2"] if iters else None
            info["last_valid_loss"] = iters[-1]["valid_l2"] if iters else None

            # Loss reduction
            if iters:
                info["train_loss_reduction_pct"] = round(
                    (iters[0]["train_l2"] - iters[-1]["train_l2"]) / max(iters[0]["train_l2"], 1e-10) * 100, 3
                )
                info["valid_loss_reduction_pct"] = round(
                    (iters[0]["valid_l2"] - iters[-1]["valid_l2"]) / max(iters[0]["valid_l2"], 1e-10) * 100, 3
                )

            # LGBModel diagnoses
            diagnoses = []
            if iters and iters[0]["valid_l2"] > 0.95:
                diagnoses.append(
                    "LGB_LOSS_NEAR_BASELINE: valid L2 ~1.0 indicates GBDT barely predicts "
                    "beyond mean (for CSZScoreNorm targets)"
                )
            best_iter = tlog.get("lgb_best_iteration", 0)
            total_iter = info["lgb_total_iterations"]
            if best_iter and total_iter and best_iter < total_iter * 0.1:
                diagnoses.append(
                    f"LGB_EARLY_PEAK: best iteration {best_iter} is within first 10% of "
                    f"{total_iter} total iterations — model barely learns beyond initial splits"
                )
            if iters:
                valid_range = max(i["valid_l2"] for i in iters) - min(i["valid_l2"] for i in iters)
                if valid_range < 0.0005:
                    diagnoses.append(
                        f"LGB_VALID_FLAT: valid L2 range is only {valid_range:.6f} across "
                        f"all iterations — factors provide minimal signal for GBDT"
                    )
            info["diagnoses"] = diagnoses
            results.append(info)
            continue

        # ----- PyTorch (GeneralPTNN) path -----
        epochs = tlog["epochs"]
        info: dict[str, Any] = {"loop_id": lid}

        # Basic training info
        info["model_type"] = mcode.get("model_type", "unknown")
        info["model_size_mb"] = tlog.get("model_size_mb")
        info["train_samples"] = tlog.get("train_samples")
        info["valid_samples"] = tlog.get("valid_samples")
        info["total_epochs"] = tlog.get("total_epochs_trained", 0)
        info["early_stopped"] = tlog.get("early_stopped", False)
        info["best_epoch"] = tlog.get("best_epoch")
        info["best_valid_loss"] = tlog.get("best_valid_loss")
        info["seconds_per_epoch"] = tlog.get("seconds_per_epoch")

        # Actual hyperparameters used (from template rendering)
        ctx = tlog.get("render_context", {})
        actual_params = tlog.get("actual_params", {})
        info["actual_batch_size"] = int(ctx.get("batch_size", actual_params.get("batch_size", 0)))
        info["actual_lr"] = ctx.get("lr", actual_params.get("lr"))
        info["actual_n_epochs"] = int(ctx.get("n_epochs", actual_params.get("n_epochs", 0)))
        info["actual_early_stop"] = int(ctx.get("early_stop", actual_params.get("early_stop", 0)))
        info["actual_weight_decay"] = ctx.get("weight_decay", actual_params.get("weight_decay"))
        info["dataset_cls"] = ctx.get("dataset_cls", "DatasetH")

        # Training loss analysis
        train_losses = [e["train_loss"] for e in epochs]
        valid_losses = [e["valid_loss"] for e in epochs]

        info["first_train_loss"] = train_losses[0] if train_losses else None
        info["last_train_loss"] = train_losses[-1] if train_losses else None
        info["first_valid_loss"] = valid_losses[0] if valid_losses else None
        info["last_valid_loss"] = valid_losses[-1] if valid_losses else None

        # Compute total loss reduction
        if train_losses:
            info["train_loss_reduction_pct"] = round(
                (train_losses[0] - train_losses[-1]) / max(train_losses[0], 1e-10) * 100, 3
            )
        if valid_losses:
            info["valid_loss_reduction_pct"] = round(
                (valid_losses[0] - valid_losses[-1]) / max(valid_losses[0], 1e-10) * 100, 3
            )

        # Convergence diagnosis
        diagnoses = []
        # 1. Check if model barely learns (loss ~1.0 means predicting zeros for normalized targets)
        if train_losses and train_losses[0] > 0.95:
            diagnoses.append("LOSS_NEAR_BASELINE: train loss ~1.0 indicates model barely predicts beyond mean (for CSZScoreNorm targets, MSE=1.0 equals predicting all zeros)")

        # 2. Check if best epoch is epoch 0 (model never improved)
        if tlog.get("best_epoch") == 0:
            diagnoses.append("BEST_EPOCH_ZERO: best validation score at epoch 0 means the model NEVER improved during training - likely learning rate, architecture, or batch_size issue")

        # 3. Check for validation loss not improving
        if valid_losses and len(valid_losses) >= 5:
            v_first = valid_losses[0]
            v_best = min(valid_losses)
            if abs(v_first - v_best) / max(v_first, 1e-10) < 0.001:
                diagnoses.append("VALID_LOSS_FLAT: validation loss essentially flat throughout training, model fails to generalize")

        # 4. Check for overfitting (train much better than valid)
        if train_losses and valid_losses:
            gap = valid_losses[-1] - train_losses[-1]
            if gap > 0.01 and len(epochs) > 10:
                diagnoses.append(f"OVERFITTING: train-valid gap = {gap:.4f}, model memorizes training data")

        # 5. Steps per epoch and GPU efficiency
        bs = info["actual_batch_size"]
        n_train = info.get("train_samples", 0)
        if bs and n_train:
            steps_per_epoch = n_train // bs
            info["steps_per_epoch"] = steps_per_epoch
            sec_per_epoch = info.get("seconds_per_epoch")
            if sec_per_epoch and steps_per_epoch:
                info["ms_per_step"] = round(sec_per_epoch / steps_per_epoch * 1000, 1)
            if bs < 1024 and n_train > 1000000:
                diagnoses.append(
                    f"SMALL_BATCH_LARGE_DATA: batch_size={bs} with {n_train:,} samples = {steps_per_epoch:,} steps/epoch. "
                    f"GPU is starved - majority of time is data loading overhead, not GPU compute. "
                    f"For a {info.get('model_size_mb', '?')}MB model on 16GB GPU, batch_size could be 2048-8192."
                )

        # 6. Excessive training time
        sec = info.get("seconds_per_epoch")
        total_epochs = info.get("total_epochs", 0)
        if sec and total_epochs:
            total_hours = sec * total_epochs / 3600
            info["total_training_hours"] = round(total_hours, 2)
            if total_hours > 2 and info.get("best_epoch") == 0:
                diagnoses.append(
                    f"WASTED_COMPUTE: trained for {total_hours:.1f}h but best epoch was 0. "
                    f"All training time after early_stop patience was wasted."
                )

        info["diagnoses"] = diagnoses
        results.append(info)
    return results


def analyze_prompt_config_consistency(loop_data: list[dict[str, Any]], prompt_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Check if actual hyperparameters follow prompt recommendations."""
    issues = []
    conf_default_bs = prompt_config.get("conf_default_batch_size", 16384)

    for ld in loop_data:
        tlog = ld.get("training_log", {})
        mcode = ld.get("model_code", {})
        ctx = tlog.get("render_context", {})
        if not ctx:
            continue

        lid = ld["loop_id"]
        model_type = mcode.get("model_type", "unknown")
        dataset_cls = ctx.get("dataset_cls", "DatasetH")
        bs = int(ctx.get("batch_size", conf_default_bs))

        # Check batch_size vs model_type
        if dataset_cls == "TSDatasetH":
            if bs < 2048:
                issues.append({
                    "loop_id": lid,
                    "issue": "BATCH_SIZE_TOO_SMALL_FOR_TS",
                    "detail": f"TSDatasetH with batch_size={bs}. Prompt recommends 2048-8192 for TimeSeries models (<1MB) on 16GB GPU.",
                })
            elif bs > 8192:
                issues.append({
                    "loop_id": lid,
                    "issue": "BATCH_SIZE_TOO_LARGE_FOR_TS",
                    "detail": f"TSDatasetH with batch_size={bs}. Prompt recommends max 8192 for TimeSeries.",
                })
        elif dataset_cls == "DatasetH":
            if bs < 2048:
                issues.append({
                    "loop_id": lid,
                    "issue": "BATCH_SIZE_TOO_SMALL_TABULAR",
                    "detail": f"DatasetH (Tabular) with batch_size={bs}. Prompt recommends 4096-16384.",
                })

        # Check lr vs model type
        lr_str = ctx.get("lr", "0")
        try:
            lr = float(lr_str)
        except (ValueError, TypeError):
            lr = 0
        hyp = ld.get("hypothesis", {})
        action = hyp.get("action", "")
        if action == "model" and lr > 0:
            if "gru" in str(mcode.get("model_py", "")).lower() and lr > 5e-4:
                issues.append({
                    "loop_id": lid,
                    "issue": "LR_HIGH_FOR_GRU",
                    "detail": f"GRU model with lr={lr}. Prompt recommends 2e-4 ~ 5e-4 for GRU/LSTM.",
                })
            if "transformer" in str(mcode.get("model_py", "")).lower() and lr > 5e-4:
                issues.append({
                    "loop_id": lid,
                    "issue": "LR_HIGH_FOR_TRANSFORMER",
                    "detail": f"Transformer model with lr={lr}. Prompt recommends 1e-4 ~ 5e-4.",
                })

    return issues


def analyze_model_code_quality(loop_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect common model code bugs via static pattern analysis."""
    issues = []
    for ld in loop_data:
        mc = ld.get("model_code", {})
        code = mc.get("model_py", "")
        if not code:
            continue
        lid = ld["loop_id"]

        # Bug: nn.Module subclasses created inside forward()
        # Pattern: find forward() method body, then check for nn.Linear/nn.Conv etc.
        fwd_match = re.search(r"def forward\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)", code, re.DOTALL)
        if fwd_match:
            fwd_body = fwd_match.group(1)
            # Check for nn.Linear, nn.Conv1d, nn.Conv2d, nn.GRU, nn.LSTM, nn.LayerNorm, nn.BatchNorm
            nn_in_forward = re.findall(
                r"(nn\.(?:Linear|Conv[12]d|GRU|LSTM|LayerNorm|BatchNorm\d*d?|Embedding|MultiheadAttention|TransformerEncoder\w*))\s*\(",
                fwd_body,
            )
            if nn_in_forward:
                unique = list(set(nn_in_forward))
                issues.append({
                    "loop_id": lid,
                    "issue": "NN_MODULE_IN_FORWARD",
                    "detail": (
                        f"Found {', '.join(unique)} created inside forward(). "
                        f"These modules are NOT registered as parameters and will NOT be trained by the optimizer. "
                        f"All nn.Module instances must be created in __init__()."
                    ),
                })

        # Bug: .to(device) inside forward on newly created modules (sign of dynamic creation)
        if fwd_match:
            fwd_body = fwd_match.group(1)
            to_device_pattern = re.findall(r"nn\.\w+\([^)]*\)\.to\(", fwd_body)
            if to_device_pattern and not nn_in_forward:
                issues.append({
                    "loop_id": lid,
                    "issue": "DYNAMIC_MODULE_TO_DEVICE",
                    "detail": "Found nn.Module(...).to(device) pattern in forward() - likely creating untrained modules dynamically.",
                })

        # Warning: Very small model (may not have enough capacity)
        model_size = ld.get("training_log", {}).get("model_size_mb")
        if model_size is not None and model_size < 0.05:
            issues.append({
                "loop_id": lid,
                "issue": "MODEL_TOO_SMALL",
                "detail": f"Model is only {model_size}MB. May lack capacity to learn complex patterns. Consider hidden_dim >= 64.",
            })

        # Warning: No dropout in model (potential overfitting)
        if code and "dropout" not in code.lower() and "Dropout" not in code:
            train_samples = ld.get("training_log", {}).get("train_samples", 0)
            if train_samples and train_samples > 500000:
                # Only flag if it's a complex model (has attention or multiple layers)
                if any(kw in code for kw in ["Attention", "Transformer", "MultiheadAttention", "num_layers"]):
                    issues.append({
                        "loop_id": lid,
                        "issue": "NO_DROPOUT",
                        "detail": "Complex model without dropout. Risk of overfitting on large dataset.",
                    })

    return issues


# -- Field whitelist for coverage analysis (from actual static_factors.parquet schema) --
FIELD_WHITELIST: dict[str, list[str]] = {
    "db": [
        "db_circ_mv", "db_close", "db_dv_ratio", "db_dv_ttm",
        "db_float_share", "db_free_share", "db_pb", "db_pe", "db_pe_ttm",
        "db_ps", "db_ps_ttm", "db_total_mv", "db_total_share",
        "db_turnover_rate", "db_turnover_rate_f", "db_volume_ratio",
    ],
    "mf": [
        "mf_elg_buy_amt", "mf_elg_buy_vol", "mf_elg_net_amt",
        "mf_elg_net_amt_20d", "mf_elg_net_amt_5d", "mf_elg_net_amt_ratio",
        "mf_elg_net_amt_ratio_20d", "mf_elg_net_amt_ratio_5d",
        "mf_elg_net_vol", "mf_elg_net_vol_ratio",
        "mf_elg_sell_amt", "mf_elg_sell_vol",
        "mf_elg_share_in_main_amt", "mf_elg_share_in_main_vol",
        "mf_lg_buy_amt", "mf_lg_buy_vol", "mf_lg_sell_amt", "mf_lg_sell_vol",
        "mf_main_net_amt", "mf_main_net_amt_20d", "mf_main_net_amt_5d",
        "mf_main_net_amt_ratio", "mf_main_net_amt_ratio_20d", "mf_main_net_amt_ratio_5d",
        "mf_main_net_vol", "mf_main_net_vol_ratio",
        "mf_md_buy_amt", "mf_md_buy_vol", "mf_md_sell_amt", "mf_md_sell_vol",
        "mf_net_amt", "mf_net_vol",
        "mf_sm_buy_amt", "mf_sm_buy_vol", "mf_sm_sell_amt", "mf_sm_sell_vol",
        "mf_total_net_amt", "mf_total_net_amt_20d", "mf_total_net_amt_5d",
        "mf_total_net_amt_ratio", "mf_total_net_amt_ratio_20d", "mf_total_net_amt_ratio_5d",
        "mf_total_net_vol", "mf_total_net_vol_ratio",
    ],
    "bb": [
        "bb_bvps", "bb_eps", "bb_fixed_assets", "bb_gpr", "bb_holder_num",
        "bb_liquid_assets", "bb_npr", "bb_pe_dyn", "bb_per_undp",
        "bb_profit_yoy", "bb_reserved", "bb_reserved_pershare",
        "bb_rev_yoy", "bb_total_assets", "bb_undp",
    ],
    "cp": [
        "cp_cost_5pct", "cp_cost_15pct", "cp_cost_50pct", "cp_cost_85pct", "cp_cost_95pct",
        "cp_his_high", "cp_his_low", "cp_weight_avg", "cp_winner_rate",
    ],
    "precomputed": [
        "PriceStrength_10D", "value_pe_inv", "value_pb_inv",
        "size_log_mv", "liquidity_turnover", "liquidity_vol_ratio",
    ],
    "daily_pv": ["open", "close", "high", "low", "volume", "amount"],
}

ALL_WHITELIST_FIELDS: set[str] = set()
for _grp in FIELD_WHITELIST.values():
    ALL_WHITELIST_FIELDS.update(_grp)


def _extract_fields_from_code(code: str) -> set[str]:
    """Extract referenced *input* data fields from factor Python code via regex.

    Excludes factor output column names (result_df["factor_name"] = ...) since
    those are outputs, not data dependencies.
    """
    fields: set[str] = set()
    if not code:
        return fields

    # Collect output column names to exclude
    output_cols: set[str] = set()
    for m in re.finditer(r'''result_df\[["'](\w+)["']\]''', code):
        output_cols.add(m.group(1))

    # Match df["field"], static_df["field"], data["field"]
    for m in re.finditer(r'''(?:df|static_df|data)\[["'](\w+)["']\]''', code):
        f = m.group(1)
        if f not in output_cols:
            fields.add(f)
    # Match column references in read_parquet columns=[...]
    for m in re.finditer(r'''columns\s*=\s*\[([^\]]+)\]''', code):
        for cm in re.finditer(r'''["'](\w+)["']''', m.group(1)):
            fields.add(cm.group(1))
    # Match references in required_cols / required_static_cols lists
    for m in re.finditer(r'''required\w*_?cols\s*=\s*\[([^\]]+)\]''', code):
        for cm in re.finditer(r'''["'](\w+)["']''', m.group(1)):
            fields.add(cm.group(1))
    return fields


def _extract_math_ops(code: str) -> set[str]:
    """Extract key mathematical operations from factor code."""
    ops: set[str] = set()
    if not code:
        return ops
    if ".shift(" in code:
        for m in re.finditer(r'\.shift\((\d+)\)', code):
            ops.add(f"shift({m.group(1)})")
    if ".rolling(" in code:
        for m in re.finditer(r'\.rolling\((?:window=)?(\d+)', code):
            ops.add(f"rolling({m.group(1)})")
    if ".pct_change(" in code:
        ops.add("pct_change")
    if "np.log" in code:
        ops.add("log")
    if "np.sign" in code:
        ops.add("sign")
    if "np.where" in code:
        ops.add("conditional")
    if ".rank(" in code:
        ops.add("rank")
    if ".std(" in code or ".var(" in code:
        ops.add("volatility")
    if ".corr(" in code:
        ops.add("correlation")
    for op in [" * ", " / ", " + ", " - "]:
        if op in code:
            ops.add({"*": "multiply", "/": "divide", "+": "add", "-": "subtract"}[op.strip()])
    return ops


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def analyze_factor_homogeneity(loop_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze factor homogeneity across loops: field usage, similarity, coverage."""
    result: dict[str, Any] = {
        "factors": [],
        "similarity_matrix": [],
        "field_coverage": {},
        "issues": [],
    }

    # 1. Extract per-factor info (fields + ops) from all loops
    all_factors: list[dict[str, Any]] = []
    for ld in loop_data:
        lid = ld["loop_id"]
        hyp = ld.get("hypothesis", {})
        action = hyp.get("action", "unknown")
        if action == "model":
            continue  # Skip model loops

        mc = ld.get("model_code", {})
        factor_files = mc.get("factor_files", {})
        # Also include the primary factor.py
        primary_code = None
        for key in (mc.get("file_keys") or []):
            if key == "factor.py" and key not in factor_files:
                # Try to get from model_py fallback or factor_files
                pass
        # Check factor_files for factor.py
        if "factor.py" in factor_files:
            primary_code = factor_files["factor.py"]

        # Collect all factor codes for this loop
        codes: dict[str, str] = {}
        if primary_code:
            fname = mc.get("factor_name", "factor_0")
            codes[fname] = primary_code
        for fkey, fcode in factor_files.items():
            if fkey == "factor.py":
                continue
            # workspace_N/factor.py -> factor_N
            ws_match = re.match(r"workspace_(\d+)/", fkey)
            idx = int(ws_match.group(1)) if ws_match else len(codes)
            # Try to get factor name from experiment tasks
            exp_tasks = ld.get("experiment", {}).get("tasks", [])
            if idx < len(exp_tasks):
                fname = exp_tasks[idx].get("name", f"factor_{idx}")
            else:
                fname = f"factor_{idx}"
            codes[fname] = fcode

        # If no factor_files found but experiment has factor tasks, still record them
        # Try to extract fields from variables dict (experiment generation or coder result)
        if not codes:
            exp_tasks = ld.get("experiment", {}).get("tasks", [])
            task_variables = mc.get("task_variables", {})
            field_pattern = re.compile(r'^(db_|mf_|bb_|cp_)')
            if exp_tasks and action == "factor":
                for i, t in enumerate(exp_tasks):
                    fname = t.get("name", f"factor_{i}")
                    # Try variables from coder result task_variables first
                    vars_dict = task_variables.get(fname, {})
                    if not vars_dict:
                        # Fallback: variables from experiment generation
                        vars_dict = t.get("variables", {})
                    fields = set()
                    if isinstance(vars_dict, dict):
                        for var_name in vars_dict.keys():
                            if field_pattern.match(var_name) or var_name in ALL_WHITELIST_FIELDS:
                                fields.add(var_name)
                    all_factors.append({
                        "loop_id": lid,
                        "name": fname,
                        "description": t.get("description", ""),
                        "fields": fields,
                        "ops": set(),
                        "source": "variables",
                    })
            continue

        for fname, fcode in codes.items():
            fields = _extract_fields_from_code(fcode)
            ops = _extract_math_ops(fcode)
            # Fallback: if code extraction found no fields, try variables dict
            if not fields:
                task_variables = mc.get("task_variables", {})
                vars_dict = task_variables.get(fname, {})
                if not vars_dict:
                    exp_tasks = ld.get("experiment", {}).get("tasks", [])
                    for t in exp_tasks:
                        if t.get("name") == fname:
                            vars_dict = t.get("variables", {})
                            break
                if isinstance(vars_dict, dict):
                    field_pattern_fb = re.compile(r'^(db_|mf_|bb_|cp_)')
                    for var_name in vars_dict.keys():
                        if field_pattern_fb.match(var_name) or var_name in ALL_WHITELIST_FIELDS:
                            fields.add(var_name)
            # Find matching description from experiment tasks
            desc = ""
            exp_tasks = ld.get("experiment", {}).get("tasks", [])
            for t in exp_tasks:
                if t.get("name") == fname:
                    desc = t.get("description", "")
                    break
            all_factors.append({
                "loop_id": lid,
                "name": fname,
                "description": desc,
                "fields": fields,
                "ops": ops,
            })

    result["factors"] = [
        {
            "loop_id": f["loop_id"],
            "name": f["name"],
            "description": f["description"],
            "fields": sorted(f["fields"]),
            "ops": sorted(f["ops"]),
        }
        for f in all_factors
    ]

    # 2. Cross-factor similarity matrix (Jaccard on field sets)
    n = len(all_factors)
    sim_pairs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = all_factors[i], all_factors[j]
            field_sim = _jaccard(fi["fields"], fj["fields"])
            op_sim = _jaccard(fi["ops"], fj["ops"])
            combined = 0.7 * field_sim + 0.3 * op_sim
            if combined > 0.3:  # Only report notable similarities
                sim_pairs.append({
                    "factor_a": f"L{fi['loop_id']}:{fi['name']}",
                    "factor_b": f"L{fj['loop_id']}:{fj['name']}",
                    "field_similarity": round(field_sim, 3),
                    "op_similarity": round(op_sim, 3),
                    "combined_similarity": round(combined, 3),
                    "shared_fields": sorted(fi["fields"] & fj["fields"]),
                    "cross_loop": fi["loop_id"] != fj["loop_id"],
                })
    sim_pairs.sort(key=lambda x: x["combined_similarity"], reverse=True)
    result["similarity_matrix"] = sim_pairs

    # 3. Field coverage analysis
    all_used: set[str] = set()
    for f in all_factors:
        all_used.update(f["fields"])

    coverage: dict[str, Any] = {}
    for grp_name, grp_fields in FIELD_WHITELIST.items():
        grp_set = set(grp_fields)
        used = grp_set & all_used
        coverage[grp_name] = {
            "total": len(grp_set),
            "used": len(used),
            "coverage_pct": round(100 * len(used) / len(grp_set), 1) if grp_set else 0,
            "used_fields": sorted(used),
            "unused_fields": sorted(grp_set - used),
        }
    total_available = len(ALL_WHITELIST_FIELDS)
    total_used = len(all_used & ALL_WHITELIST_FIELDS)
    coverage["_total"] = {
        "total": total_available,
        "used": total_used,
        "coverage_pct": round(100 * total_used / total_available, 1) if total_available else 0,
    }
    result["field_coverage"] = coverage

    # 4. Generate issues
    issues: list[dict[str, Any]] = []

    # High cross-loop similarity
    cross_loop_high = [s for s in sim_pairs if s["cross_loop"] and s["combined_similarity"] >= 0.5]
    if cross_loop_high:
        issues.append({
            "issue": "CROSS_LOOP_HOMOGENEITY",
            "severity": "HIGH",
            "detail": (
                f"{len(cross_loop_high)} factor pair(s) across different loops have "
                f"combined similarity >= 0.5. Top: {cross_loop_high[0]['factor_a']} vs "
                f"{cross_loop_high[0]['factor_b']} (sim={cross_loop_high[0]['combined_similarity']}). "
                f"Shared fields: {cross_loop_high[0]['shared_fields']}"
            ),
            "pairs": cross_loop_high,
        })

    # Low field coverage
    total_cov = coverage.get("_total", {})
    if total_cov.get("coverage_pct", 100) < 20:
        issues.append({
            "issue": "LOW_FIELD_COVERAGE",
            "severity": "HIGH",
            "detail": (
                f"Only {total_cov['used']}/{total_cov['total']} whitelist fields used "
                f"({total_cov['coverage_pct']}%). Large unexplored data space."
            ),
        })

    # Group-specific coverage gaps
    for grp_name, grp_info in coverage.items():
        if grp_name.startswith("_"):
            continue
        if grp_info["total"] > 5 and grp_info["coverage_pct"] < 10:
            issues.append({
                "issue": f"UNUSED_GROUP_{grp_name.upper()}",
                "severity": "MEDIUM",
                "detail": (
                    f"Field group '{grp_name}' has {grp_info['total']} fields but only "
                    f"{grp_info['used']} used ({grp_info['coverage_pct']}%). "
                    f"Consider exploring: {', '.join(grp_info['unused_fields'][:5])}"
                ),
            })

    # Hallucinated fields (used but not in whitelist)
    hallucinated = all_used - ALL_WHITELIST_FIELDS - {"factor"}  # 'factor' is a valid daily_pv column
    if hallucinated:
        issues.append({
            "issue": "HALLUCINATED_FIELDS",
            "severity": "HIGH",
            "detail": (
                f"Fields used but NOT in whitelist (may not exist): {sorted(hallucinated)}"
            ),
        })

    result["issues"] = issues

    # 5. Per-loop innovation metrics (with performance data)
    # Build a quick lookup: loop_id -> metrics + decision
    _loop_metrics_map: dict[int, dict[str, Any]] = {}
    for ld in loop_data:
        lid_tmp = ld["loop_id"]
        rr = ld.get("runner_result", {})
        metrics_tmp = rr.get("metrics", {}) or {}
        fb_tmp = ld.get("feedback", {})
        _loop_metrics_map[lid_tmp] = {
            "annualized_return": metrics_tmp.get("1day.excess_return_with_cost.annualized_return"),
            "max_drawdown": metrics_tmp.get("1day.excess_return_with_cost.max_drawdown"),
            "IC": metrics_tmp.get("IC"),
            "decision": fb_tmp.get("decision"),
        }

    per_loop_innovation: list[dict[str, Any]] = []
    # Group factors by loop_id
    loop_ids = sorted(set(f["loop_id"] for f in all_factors))
    cumulative_fields: set[str] = set()  # Fields seen in all previous loops
    for lid in loop_ids:
        loop_factors = [f for f in all_factors if f["loop_id"] == lid]
        loop_fields: set[str] = set()
        for f in loop_factors:
            loop_fields.update(f["fields"])
        # New fields: fields in this loop that haven't appeared in any previous loop
        new_fields = loop_fields - cumulative_fields
        # Overlap with previous loops
        if cumulative_fields:
            overlap_rate = len(loop_fields & cumulative_fields) / len(loop_fields) if loop_fields else 0.0
        else:
            overlap_rate = 0.0  # First loop has no overlap
        # Prefix distribution for this loop
        prefix_dist: dict[str, int] = {}
        for fld in loop_fields:
            for px in ("db_", "mf_", "bb_", "cp_"):
                if fld.startswith(px):
                    prefix_dist[px + "*"] = prefix_dist.get(px + "*", 0) + 1
                    break
            else:
                if fld in ALL_WHITELIST_FIELDS:
                    prefix_dist["other"] = prefix_dist.get("other", 0) + 1
        # Math ops diversity for this loop
        loop_ops: set[str] = set()
        for f in loop_factors:
            loop_ops.update(f["ops"])

        # Performance metrics for this loop
        lm = _loop_metrics_map.get(lid, {})

        per_loop_innovation.append({
            "loop_id": lid,
            "factor_count": len(loop_factors),
            "total_fields": len(loop_fields),
            "new_fields": sorted(new_fields),
            "new_field_count": len(new_fields),
            "overlap_rate_with_prev": round(overlap_rate, 3),
            "prefix_distribution": prefix_dist,
            "math_ops": sorted(loop_ops),
            "math_ops_count": len(loop_ops),
            "annualized_return": lm.get("annualized_return"),
            "max_drawdown": lm.get("max_drawdown"),
            "IC": lm.get("IC"),
            "decision": lm.get("decision"),
        })
        # Add this loop's fields to cumulative set
        cumulative_fields.update(loop_fields)

    result["per_loop_innovation"] = per_loop_innovation

    # Innovation issues
    for pli in per_loop_innovation:
        if pli["loop_id"] == loop_ids[0]:
            continue  # Skip first loop (baseline)
        if pli["new_field_count"] == 0:
            issues.append({
                "issue": "NO_NEW_FIELDS",
                "severity": "HIGH",
                "detail": (
                    f"Loop {pli['loop_id']} introduced 0 new fields. "
                    f"All {pli['total_fields']} fields were already used in previous loops."
                ),
            })
        elif pli["new_field_count"] < 2:
            issues.append({
                "issue": "FEW_NEW_FIELDS",
                "severity": "MEDIUM",
                "detail": (
                    f"Loop {pli['loop_id']} introduced only {pli['new_field_count']} new field(s): "
                    f"{pli['new_fields']}. Recommend ≥2 per loop."
                ),
            })
        if pli["overlap_rate_with_prev"] > 0.6:
            issues.append({
                "issue": "HIGH_FIELD_OVERLAP",
                "severity": "HIGH",
                "detail": (
                    f"Loop {pli['loop_id']} has {pli['overlap_rate_with_prev']:.0%} field overlap "
                    f"with previous loops. Factors are rehashing known fields."
                ),
            })

    # Update issues in result
    result["issues"] = issues

    return result


# ---------------------------------------------------------------------------
# Section 4: Live Resource Check
# ---------------------------------------------------------------------------

def check_live_resources() -> dict[str, Any]:
    """Check GPU, processes, and memory via WSL commands."""
    result: dict[str, Any] = {}

    # GPU check via nvidia-smi
    try:
        gpu_out = subprocess.run(
            ["wsl", "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if gpu_out.returncode == 0 and gpu_out.stdout.strip():
            lines = gpu_out.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "name": parts[0],
                        "utilization_pct": int(parts[1]),
                        "memory_used_mb": int(parts[2]),
                        "memory_total_mb": int(parts[3]),
                        "temperature_c": int(parts[4]),
                        "memory_pct": round(int(parts[2]) / max(int(parts[3]), 1) * 100, 1),
                    })
            result["gpu"] = gpus
        else:
            result["gpu"] = None
            result["gpu_error"] = gpu_out.stderr.strip() if gpu_out.stderr else "no output"
    except Exception as e:
        result["gpu"] = None
        result["gpu_error"] = str(e)

    # Process check for qrun/python processes
    try:
        ps_out = subprocess.run(
            ["wsl", "ps", "aux"],
            capture_output=True, text=True, timeout=10,
        )
        if ps_out.returncode == 0:
            qlib_procs = []
            for line in ps_out.stdout.split("\n"):
                if any(kw in line.lower() for kw in ["qrun", "qlib", "train.py", "workflow"]):
                    qlib_procs.append(line.strip())
            result["qlib_processes"] = qlib_procs if qlib_procs else ["none found"]
        else:
            result["qlib_processes"] = None
    except Exception as e:
        result["qlib_processes"] = None
        result["process_error"] = str(e)

    # Memory check
    try:
        mem_out = subprocess.run(
            ["wsl", "free", "-m"],
            capture_output=True, text=True, timeout=10,
        )
        if mem_out.returncode == 0:
            for line in mem_out.stdout.split("\n"):
                if line.startswith("Mem:"):
                    parts = line.split()
                    if len(parts) >= 3:
                        result["memory"] = {
                            "total_mb": int(parts[1]),
                            "used_mb": int(parts[2]),
                            "used_pct": round(int(parts[2]) / max(int(parts[1]), 1) * 100, 1),
                        }
    except Exception as e:
        result["memory_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Section 5: Report Generation
# ---------------------------------------------------------------------------

def _truncate(s: Any, max_len: int = 200) -> str | None:
    if s is None:
        return None
    s = str(s)
    return s[:max_len] + "..." if len(s) > max_len else s


def _fmt_duration(td: timedelta) -> str:
    s = int(td.total_seconds())
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def generate_text_report(
    task_id: str,
    log_dir: Path,
    loop_data: list[dict[str, Any]],
    session_obj: Any,
    session_dir: str | None,
    session_hist_len: int,
    evolution_analysis: dict[str, Any],
    parallelism_analysis: dict[str, Any],
    propagation_analysis: dict[str, Any],
    feedback_analysis: dict[str, Any],
    hyperparam_analysis: dict[str, Any],
    convergence_analysis: list[dict[str, Any]],
    costeer_config: dict[str, Any],
    prompt_config: dict[str, Any],
    prompt_issues: list[dict[str, Any]],
    code_quality_issues: list[dict[str, Any]],
    live_resources: dict[str, Any] | None,
    homogeneity_analysis: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    sep = "=" * 80

    # --- Section 1: Task Overview ---
    lines.append(sep)
    lines.append(f"  RDAGENT TASK DIAGNOSTIC REPORT")
    lines.append(sep)
    lines.append(f"Task ID:          {task_id}")
    lines.append(f"Log Directory:    {log_dir}")
    lines.append(f"Loops Completed:  {len([ld for ld in loop_data if ld.get('feedback')])}")
    lines.append(f"Loops Total:      {len(loop_data)}")
    lines.append(f"Session Snapshot: {session_dir} (trace.hist length: {session_hist_len})")

    # Determine current step
    last_loop = loop_data[-1] if loop_data else None
    if last_loop:
        timing = last_loop.get("timing", {})
        steps_with_time = list(timing.keys())
        fb = last_loop.get("feedback")
        if fb:
            lines.append(f"Current Status:   Loop_{last_loop['loop_id']} COMPLETED (all steps done)")
        elif steps_with_time:
            lines.append(f"Current Status:   Loop_{last_loop['loop_id']} IN PROGRESS (last step: {steps_with_time[-1]})")
        else:
            lines.append(f"Current Status:   Loop_{last_loop['loop_id']} STARTING")

    lines.append("")

    # --- Section 2: Loop-by-Loop Summary ---
    lines.append(sep)
    lines.append("  LOOP-BY-LOOP SUMMARY")
    lines.append(sep)
    for ld in loop_data:
        lid = ld["loop_id"]
        lines.append(f"\n--- Loop {lid} ---")

        # Hypothesis
        hyp = ld.get("hypothesis", {})
        if hyp:
            action = hyp.get("action", "unknown")
            lines.append(f"  Action:      {action}")
            lines.append(f"  Hypothesis:  {_truncate(hyp.get('hypothesis'), 120)}")
            lines.append(f"  Reason:      {_truncate(hyp.get('concise_reason') or hyp.get('reason'), 120)}")

        # Tasks
        exp = ld.get("experiment", {})
        tasks = exp.get("tasks", [])
        if tasks:
            for i, t in enumerate(tasks):
                lines.append(f"  Task[{i}]:     {t.get('name', 'N/A')} ({t.get('type', 'unknown')})")
                for attr in ("hyperparameters", "training_hyperparameters", "architecture"):
                    if attr in t:
                        lines.append(f"    {attr}: {json.dumps(t[attr], default=str)[:200]}")

        # Evo loops
        evo = ld.get("evo_loop_count", 0)
        lines.append(f"  CoSTEER Evo Iterations: {evo}")

        # Timing
        timing = ld.get("timing", {})
        if timing:
            total = sum(v.get("duration_seconds", 0) for v in timing.values())
            lines.append(f"  Timing (total {_fmt_duration(timedelta(seconds=total))}):")
            for step_name, t in timing.items():
                lines.append(f"    {step_name:20s} {t['duration_human']:>12s}")

        # Metrics
        rr = ld.get("runner_result", {})
        metrics = rr.get("metrics")
        if metrics:
            lines.append("  Metrics:")
            for k, v in metrics.items():
                # Short name for display
                short = k.split(".")[-1] if "." in k else k
                lines.append(f"    {short:30s} {v:>12.6f}" if v is not None else f"    {short:30s} {'N/A':>12s}")

        # Feedback
        fb = ld.get("feedback", {})
        if fb:
            dec = fb.get("decision")
            dec_str = "ACCEPTED (replace SOTA)" if dec else "REJECTED"
            lines.append(f"  Feedback Decision: {dec_str}")
            lines.append(f"    Observations: {_truncate(fb.get('observations'), 150)}")

    lines.append("")

    # --- Section 3: Performance Evolution Table ---
    lines.append(sep)
    lines.append("  PERFORMANCE EVOLUTION")
    lines.append(sep)

    evo_a = evolution_analysis
    if evo_a["loops_with_metrics"] == 0:
        lines.append("  No metrics data available yet.")
    else:
        # Build table header
        metric_names = list(evo_a["metric_trends"].keys())
        if metric_names:
            # Header row
            header = f"  {'Loop':>6s}"
            for mn in metric_names:
                short = mn.split(".")[-1] if "." in mn else mn
                header += f"  {short:>15s}"
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))

            # Collect all loop ids that have any metrics
            all_loop_ids = set()
            for mn in metric_names:
                for lid, _ in evo_a["metric_trends"][mn].get("values", []):
                    all_loop_ids.add(lid)

            for lid in sorted(all_loop_ids):
                row = f"  {lid:>6d}"
                for mn in metric_names:
                    vals = {v[0]: v[1] for v in evo_a["metric_trends"][mn].get("values", [])}
                    if lid in vals:
                        row += f"  {vals[lid]:>15.6f}"
                    else:
                        row += f"  {'N/A':>15s}"
                lines.append(row)

            lines.append("")
            lines.append("  Trends:")
            for mn in metric_names:
                info = evo_a["metric_trends"][mn]
                short = mn.split(".")[-1] if "." in mn else mn
                if info["trend"] == "insufficient_data":
                    lines.append(f"    {short}: insufficient data")
                else:
                    arrow = "^" if info["trend"] == "improving" else "v"
                    lines.append(f"    {short}: {info['first']:.6f} -> {info['last']:.6f} (delta: {info['delta']:+.6f}) [{arrow} {info['trend']}]")

    lines.append("")

    # --- Section 4: Hyperparameter Evolution ---
    lines.append(sep)
    lines.append("  HYPERPARAMETER EVOLUTION")
    lines.append(sep)
    hp_hist = hyperparam_analysis.get("hyperparameter_history", [])
    if not hp_hist:
        lines.append("  No hyperparameter data found.")
    else:
        for entry in hp_hist:
            lid = entry.get("loop_id")
            name = entry.get("task_name", "unknown")
            lines.append(f"  Loop {lid} - {name}:")
            for attr in ("hyperparameters", "training_hyperparameters", "architecture"):
                if attr in entry:
                    lines.append(f"    {attr}: {json.dumps(entry[attr], default=str)[:300]}")

    lines.append("")

    # --- Section 5: Parallelism Analysis ---
    lines.append(sep)
    lines.append("  PARALLELISM ANALYSIS")
    lines.append(sep)
    lines.append(f"  Coding intervals found: {parallelism_analysis['coding_intervals_found']}")
    lines.append(f"  Overlaps detected:      {parallelism_analysis['overlaps_detected']}")
    lines.append(f"  Conclusion:             {parallelism_analysis['conclusion']}")
    for o in parallelism_analysis.get("overlaps", []):
        lines.append(f"    - {o}")

    lines.append("")

    # --- Section 6: Loop Result Propagation ---
    lines.append(sep)
    lines.append("  LOOP RESULT PROPAGATION")
    lines.append(sep)
    prop = propagation_analysis
    lines.append(f"  trace.hist length:           {prop.get('trace_hist_len', 'N/A')}")
    lines.append(f"  based_experiments per loop:   {prop.get('based_experiments_per_loop', [])}")
    chain = prop.get("chain_growing")
    if chain is True:
        lines.append("  Chain status:                Growing (each loop builds on previous)")
    elif chain is False:
        lines.append("  Chain status:                NOT growing - possible propagation issue")
    else:
        lines.append("  Chain status:                Insufficient data")

    lines.append("")

    # --- Section 7: Training Validation Info in Feedback ---
    lines.append(sep)
    lines.append("  FEEDBACK QUALITY ANALYSIS")
    lines.append(sep)
    for fb_info in feedback_analysis.get("per_loop", []):
        lid = fb_info["loop_id"]
        if not fb_info["has_feedback"]:
            lines.append(f"  Loop {lid}: No feedback yet")
            continue
        dec = fb_info.get("decision")
        quant = fb_info.get("has_quantitative_data", False)
        metric_ref = fb_info.get("has_metric_references", False)
        comparison = fb_info.get("has_comparison_language", False)
        quality = "GOOD" if (quant and metric_ref) else "FAIR" if quant else "POOR"
        lines.append(
            f"  Loop {lid}: decision={'ACCEPT' if dec else 'REJECT'}, "
            f"quantitative={quant}, metric_refs={metric_ref}, comparisons={comparison} "
            f"-> Quality: {quality}"
        )

    lines.append("")

    # --- Section 8: Training Convergence Deep-Dive ---
    lines.append(sep)
    lines.append("  TRAINING CONVERGENCE ANALYSIS")
    lines.append(sep)
    if not convergence_analysis:
        lines.append("  No training data available.")
    else:
        for ca in convergence_analysis:
            lid = ca["loop_id"]
            model_type = ca.get("model_type", "N/A")
            lines.append(f"\n  --- Loop {lid} Training ({model_type}) ---")

            if model_type == "LGBModel":
                # LGBModel rendering
                lines.append(f"    Model Type:         LGBModel (LightGBM GBDT)")
                lines.append(f"    Early Stop Patience:{ca.get('lgb_patience', 'N/A')} rounds")
                lines.append(f"    Total Iterations:   {ca.get('lgb_total_iterations', 'N/A')}")
                lines.append(f"    Best Iteration:     {ca.get('lgb_best_iteration', 'N/A')}")
                lines.append(f"    Early Stopped:      {ca.get('lgb_early_stopped', False)}")
                lines.append(f"    Best Train L2:      {ca.get('lgb_best_train_l2', 'N/A')}")
                lines.append(f"    Best Valid L2:      {ca.get('lgb_best_valid_l2', 'N/A')}")

                # Iteration progression
                lgb_iters = ca.get("lgb_iterations", [])
                if lgb_iters:
                    lines.append(f"    Train L2:           {ca.get('first_train_loss', 'N/A')} -> {ca.get('last_train_loss', 'N/A')} ({ca.get('train_loss_reduction_pct', 'N/A')}% reduction)")
                    lines.append(f"    Valid L2:           {ca.get('first_valid_loss', 'N/A')} -> {ca.get('last_valid_loss', 'N/A')} ({ca.get('valid_loss_reduction_pct', 'N/A')}% reduction)")
                    lines.append(f"    Iteration Log:")
                    for it in lgb_iters:
                        lines.append(f"      [{it['iteration']:>4}]  train L2: {it['train_l2']:.6f}  valid L2: {it['valid_l2']:.6f}")
            else:
                # PyTorch / GeneralPTNN rendering (existing)
                lines.append(f"    Model Type:       {model_type}")
                lines.append(f"    Model Size:       {ca.get('model_size_mb', 'N/A')} MB")
                lines.append(f"    Dataset:          {ca.get('dataset_cls', 'N/A')}")
                lines.append(f"    Train Samples:    {ca.get('train_samples', 'N/A'):,}" if ca.get('train_samples') else "    Train Samples:    N/A")
                lines.append(f"    Batch Size:       {ca.get('actual_batch_size', 'N/A')}")
                steps = ca.get("steps_per_epoch")
                if steps:
                    lines.append(f"    Steps/Epoch:      {steps:,}")
                lines.append(f"    Learning Rate:    {ca.get('actual_lr', 'N/A')}")
                lines.append(f"    Weight Decay:     {ca.get('actual_weight_decay', 'N/A')}")
                lines.append(f"    N_Epochs:         {ca.get('actual_n_epochs', 'N/A')}")
                lines.append(f"    Early Stop:       {ca.get('actual_early_stop', 'N/A')}")
                lines.append(f"    Epochs Trained:   {ca.get('total_epochs', 'N/A')}")
                lines.append(f"    Best Epoch:       {ca.get('best_epoch', 'N/A')}")
                lines.append(f"    Sec/Epoch:        {ca.get('seconds_per_epoch', 'N/A')}s")
                ms_step = ca.get("ms_per_step")
                if ms_step:
                    lines.append(f"    Ms/Step:          {ms_step}ms")
                total_h = ca.get("total_training_hours")
                if total_h:
                    lines.append(f"    Total Train Time: {total_h}h")
                lines.append(f"    Train Loss:       {ca.get('first_train_loss', 'N/A')} -> {ca.get('last_train_loss', 'N/A')} ({ca.get('train_loss_reduction_pct', 'N/A')}% reduction)")
                lines.append(f"    Valid Loss:       {ca.get('first_valid_loss', 'N/A')} -> {ca.get('last_valid_loss', 'N/A')} ({ca.get('valid_loss_reduction_pct', 'N/A')}% reduction)")
                lines.append(f"    Best Valid Loss:  {ca.get('best_valid_loss', 'N/A')}")
                lines.append(f"    Early Stopped:    {ca.get('early_stopped', False)}")

            # Diagnoses (shared)
            diag = ca.get("diagnoses", [])
            if diag:
                lines.append("    ** DIAGNOSES **")
                for d in diag:
                    lines.append(f"      ! {d}")

    lines.append("")

    # --- Section 9: CoSTEER & Prompt Configuration ---
    lines.append(sep)
    lines.append("  COSTEER & PROMPT CONFIGURATION")
    lines.append(sep)
    lines.append(f"  CoSTEER max_loop (default):  {costeer_config.get('default_max_loop', 'N/A')}")
    for k, v in costeer_config.items():
        if k.startswith("env"):
            lines.append(f"  {k}: {v}")
    lines.append(f"  Conf default batch_size:     {prompt_config.get('conf_default_batch_size', 'N/A')}")
    for key, info in prompt_config.items():
        if isinstance(info, dict):
            lines.append(f"  [{key}]")
            for k, v in info.items():
                lines.append(f"    {k}: {v}")

    if prompt_issues:
        lines.append("")
        lines.append("  ** PROMPT/CONFIG ISSUES **")
        for pi in prompt_issues:
            lines.append(f"    Loop {pi['loop_id']}: [{pi['issue']}] {pi['detail']}")

    lines.append("")

    # --- Section 10: Model/Factor Code Summary ---
    lines.append(sep)
    lines.append("  MODEL/FACTOR CODE SUMMARY")
    lines.append(sep)
    for ld in loop_data:
        mc = ld.get("model_code", {})
        lid = ld["loop_id"]
        # Model code
        if mc.get("model_py"):
            code = mc["model_py"]
            lines.append(f"\n  --- Loop {lid} model.py ({len(code)} chars) ---")
            code_lines = code.split("\n")
            for cl in code_lines[:30]:
                lines.append(f"    {cl}")
            if len(code_lines) > 30:
                lines.append(f"    ... ({len(code_lines) - 30} more lines)")
        # Factor code
        factor_files = mc.get("factor_files", {})
        if factor_files:
            for fname, fcode in list(factor_files.items())[:3]:  # Show up to 3 factor files
                lines.append(f"\n  --- Loop {lid} {fname} ({len(fcode)} chars) ---")
                fcode_lines = fcode.split("\n")
                for cl in fcode_lines[:20]:
                    lines.append(f"    {cl}")
                if len(fcode_lines) > 20:
                    lines.append(f"    ... ({len(fcode_lines) - 20} more lines)")
        # Show file keys if we have code
        if mc.get("file_keys") and not mc.get("model_py") and not factor_files:
            lines.append(f"\n  --- Loop {lid} files: {mc['file_keys']} (no code extracted) ---")

    lines.append("")

    # --- Section 10.5: Model Code Quality Issues ---
    if code_quality_issues:
        lines.append(sep)
        lines.append("  MODEL CODE QUALITY ISSUES")
        lines.append(sep)
        for cqi in code_quality_issues:
            lines.append(f"  Loop {cqi['loop_id']}: [{cqi['issue']}] {cqi['detail']}")
        lines.append("")

    # --- Section 10.6: Factor Homogeneity Analysis ---
    if homogeneity_analysis and homogeneity_analysis.get("factors"):
        ha = homogeneity_analysis
        lines.append(sep)
        lines.append("  FACTOR HOMOGENEITY ANALYSIS")
        lines.append(sep)

        # Per-factor field usage
        lines.append("\n  Factor Field Usage:")
        for f in ha["factors"]:
            wl_fields = [x for x in f["fields"] if x in ALL_WHITELIST_FIELDS]
            other_fields = [x for x in f["fields"] if x not in ALL_WHITELIST_FIELDS and x != "factor"]
            lines.append(f"    L{f['loop_id']}:{f['name']}")
            lines.append(f"      Fields ({len(f['fields'])}): {', '.join(f['fields'][:15])}")
            lines.append(f"      Ops:    {', '.join(f['ops']) if f['ops'] else '(none extracted)'}")
            if other_fields:
                lines.append(f"      [!] Non-whitelist fields: {', '.join(other_fields)}")

        # Cross-loop similarity
        sim = ha.get("similarity_matrix", [])
        cross_loop_sims = [s for s in sim if s["cross_loop"]]
        if cross_loop_sims:
            lines.append("\n  Cross-Loop Factor Similarity (Jaccard, descending):")
            for s in cross_loop_sims[:10]:
                tag = " *** HIGH" if s["combined_similarity"] >= 0.5 else ""
                lines.append(
                    f"    {s['factor_a']:40s} vs {s['factor_b']:40s}  "
                    f"fields={s['field_similarity']:.2f}  ops={s['op_similarity']:.2f}  "
                    f"combined={s['combined_similarity']:.2f}{tag}"
                )
                if s["shared_fields"]:
                    lines.append(f"      Shared: {', '.join(s['shared_fields'])}")

        # Same-loop similarity (within-loop homogeneity)
        same_loop_sims = [s for s in sim if not s["cross_loop"] and s["combined_similarity"] >= 0.4]
        if same_loop_sims:
            lines.append("\n  Same-Loop Factor Similarity (potential redundancy):")
            for s in same_loop_sims[:5]:
                lines.append(
                    f"    {s['factor_a']:40s} vs {s['factor_b']:40s}  "
                    f"combined={s['combined_similarity']:.2f}"
                )

        # Field coverage
        cov = ha.get("field_coverage", {})
        total_cov = cov.get("_total", {})
        lines.append(f"\n  Field Coverage: {total_cov.get('used', 0)}/{total_cov.get('total', 0)} "
                      f"({total_cov.get('coverage_pct', 0)}%)")
        lines.append(f"    {'Group':<15s} {'Used':>5s} / {'Total':>5s}  {'Coverage':>8s}  Unused (sample)")
        for grp_name in ["db", "mf", "bb", "cp", "precomputed", "daily_pv"]:
            gi = cov.get(grp_name, {})
            unused_sample = ", ".join(gi.get("unused_fields", [])[:4])
            if gi.get("unused_fields", []):
                if len(gi["unused_fields"]) > 4:
                    unused_sample += f" (+{len(gi['unused_fields'])-4} more)"
            lines.append(
                f"    {grp_name:<15s} {gi.get('used', 0):>5d} / {gi.get('total', 0):>5d}  "
                f"{gi.get('coverage_pct', 0):>7.1f}%  {unused_sample}"
            )

        # Issues
        hi = ha.get("issues", [])
        if hi:
            lines.append("\n  Homogeneity Issues:")
            for issue in hi:
                sev = issue.get("severity", "")
                lines.append(f"    [{sev}] {issue['issue']}: {issue['detail']}")

        # Per-loop innovation analysis
        pli_list = ha.get("per_loop_innovation", [])
        if pli_list:
            lines.append("\n  Per-Loop Innovation Analysis:")
            lines.append(f"    {'Loop':>4s}  {'Factors':>7s}  {'Fields':>6s}  {'New':>5s}  {'Overlap%':>8s}  "
                         f"{'AnnRet':>8s}  {'MaxDD':>8s}  {'IC':>8s}  {'Decision':>8s}  Prefix Distribution")
            lines.append("    " + "-" * 120)
            # Track SOTA for relative comparison
            sota_ret: float | None = None
            sota_dd: float | None = None
            for pli in pli_list:
                pd_str = "  ".join(f"{k}:{v}" for k, v in sorted(pli["prefix_distribution"].items()))
                # Format performance columns
                ann_ret = pli.get("annualized_return")
                ann_ret_str = f"{ann_ret:>7.1%}" if ann_ret is not None else f"{'N/A':>7s}"
                mdd = pli.get("max_drawdown")
                mdd_str = f"{mdd:>7.1%}" if mdd is not None else f"{'N/A':>7s}"
                ic_val = pli.get("IC")
                ic_str = f"{ic_val:>7.4f}" if ic_val is not None else f"{'N/A':>7s}"
                dec = pli.get("decision")
                if dec is True:
                    dec_str = f"{'ACC':>8s}"
                    sota_ret = ann_ret
                    sota_dd = mdd
                elif dec is False:
                    dec_str = f"{'REJ':>8s}"
                elif dec is None:
                    dec_str = f"{'---':>8s}"
                else:
                    dec_str = f"{'???':>8s}"
                lines.append(
                    f"    {pli['loop_id']:>4d}  {pli['factor_count']:>7d}  "
                    f"{pli['total_fields']:>6d}  {pli['new_field_count']:>5d}  "
                    f"{pli['overlap_rate_with_prev']:>7.0%}  "
                    f"{ann_ret_str}  {mdd_str}  {ic_str}  {dec_str}  {pd_str}"
                )
                if pli["new_fields"]:
                    lines.append(f"           New fields: {', '.join(pli['new_fields'][:10])}"
                                 + (f" (+{len(pli['new_fields'])-10} more)" if len(pli['new_fields']) > 10 else ""))

            # Summary stats
            if len(pli_list) >= 2:
                avg_new = sum(p["new_field_count"] for p in pli_list[1:]) / max(len(pli_list) - 1, 1)
                avg_overlap = sum(p["overlap_rate_with_prev"] for p in pli_list[1:]) / max(len(pli_list) - 1, 1)
                total_unique = len(set(f for p in pli_list for f in p["new_fields"]) |
                                   set(f for fa in ha["factors"] if fa["loop_id"] == pli_list[0]["loop_id"] for f in fa["fields"]))
                lines.append(f"\n    Innovation Summary (excl. first loop):")
                lines.append(f"      Avg new fields/loop:    {avg_new:.1f}  (target: ≥2)")
                lines.append(f"      Avg overlap with prev:  {avg_overlap:.0%}  (target: <40%)")
                lines.append(f"      Total unique fields:    {total_unique}")
                # Diagnosis
                if avg_new < 2:
                    lines.append(f"      [!] INSUFFICIENT_INNOVATION: avg new fields ({avg_new:.1f}) < 2 per loop")
                if avg_overlap > 0.6:
                    lines.append(f"      [!] HIGH_OVERLAP: avg field overlap ({avg_overlap:.0%}) > 60%")

                # Performance summary: annualized_return and max_drawdown trends
                rets = [(p["loop_id"], p["annualized_return"]) for p in pli_list if p.get("annualized_return") is not None]
                dds = [(p["loop_id"], p["max_drawdown"]) for p in pli_list if p.get("max_drawdown") is not None]
                if rets:
                    best_ret_lid, best_ret = max(rets, key=lambda x: x[1])
                    worst_ret_lid, worst_ret = min(rets, key=lambda x: x[1])
                    first_ret = rets[0][1]
                    last_ret = rets[-1][1]
                    ret_trend = "improving" if last_ret > first_ret else "degrading"
                    lines.append(f"\n    Performance Summary:")
                    lines.append(f"      Annualized Return:  {first_ret:.1%} -> {last_ret:.1%}  "
                                 f"[{'↑' if ret_trend == 'improving' else '↓'} {ret_trend}]")
                    lines.append(f"        Best:  Loop {best_ret_lid} = {best_ret:.1%}  |  "
                                 f"Worst: Loop {worst_ret_lid} = {worst_ret:.1%}  |  "
                                 f"Range: {best_ret - worst_ret:.1%}")
                    if sota_ret is not None:
                        # Show how many loops beat SOTA
                        beat_count = sum(1 for _, r in rets if r > sota_ret)
                        lines.append(f"        SOTA ({sota_ret:.1%}):  {beat_count}/{len(rets)} loops exceeded SOTA return")
                if dds:
                    # For max_drawdown, less negative = better
                    best_dd_lid, best_dd = max(dds, key=lambda x: x[1])  # least negative
                    worst_dd_lid, worst_dd = min(dds, key=lambda x: x[1])  # most negative
                    first_dd = dds[0][1]
                    last_dd = dds[-1][1]
                    dd_trend = "improving" if last_dd > first_dd else "degrading"
                    lines.append(f"      Max Drawdown:       {first_dd:.1%} -> {last_dd:.1%}  "
                                 f"[{'↑' if dd_trend == 'improving' else '↓'} {dd_trend}]")
                    lines.append(f"        Best:  Loop {best_dd_lid} = {best_dd:.1%}  |  "
                                 f"Worst: Loop {worst_dd_lid} = {worst_dd:.1%}  |  "
                                 f"Range: {best_dd - worst_dd:.1%}")
                    if sota_dd is not None:
                        beat_dd_count = sum(1 for _, d in dds if d > sota_dd)  # less negative = better
                        lines.append(f"        SOTA ({sota_dd:.1%}):  {beat_dd_count}/{len(dds)} loops had less drawdown than SOTA")

                # Near-miss analysis: loops that improved but didn't pass 10% threshold
                if sota_ret is not None and sota_dd is not None:
                    near_misses = []
                    for p in pli_list:
                        if p.get("decision") is not False:
                            continue
                        ar = p.get("annualized_return")
                        dd = p.get("max_drawdown")
                        if ar is None or dd is None:
                            continue
                        ret_pct = (ar - sota_ret) / abs(sota_ret) * 100 if sota_ret else 0
                        dd_pct = (dd - sota_dd) / abs(sota_dd) * 100 if sota_dd else 0  # positive = improved
                        # Count as near-miss if return or IC improved but < 10%
                        if ret_pct > 0 or dd_pct > 0:
                            near_misses.append({
                                "loop_id": p["loop_id"],
                                "ret_pct": ret_pct,
                                "dd_pct": dd_pct,
                            })
                    if near_misses:
                        lines.append(f"\n      Near-Miss Analysis (REJECTED but improved vs SOTA):")
                        for nm in near_misses:
                            ret_s = f"Return {nm['ret_pct']:+.1f}%"
                            dd_s = f"MaxDD {nm['dd_pct']:+.1f}%"
                            lines.append(f"        Loop {nm['loop_id']:>2d}: {ret_s}, {dd_s}"
                                         + ("  ← closest to ACCEPT" if nm["ret_pct"] > 5 else ""))

        lines.append("")

    # --- Section 11: Live Resource Check ---
    if live_resources is not None:
        lines.append(sep)
        lines.append("  LIVE RESOURCE CHECK")
        lines.append(sep)

        gpus = live_resources.get("gpu")
        if gpus:
            for g in gpus:
                lines.append(f"  GPU:   {g['name']}")
                lines.append(f"    Utilization:  {g['utilization_pct']}%")
                lines.append(f"    VRAM:         {g['memory_used_mb']}MB / {g['memory_total_mb']}MB ({g['memory_pct']}%)")
                lines.append(f"    Temperature:  {g['temperature_c']}C")
        elif "gpu_error" in live_resources:
            lines.append(f"  GPU:   Error - {live_resources['gpu_error']}")

        procs = live_resources.get("qlib_processes")
        if procs:
            lines.append(f"  Qlib Processes: {len(procs)}")
            for p in procs[:5]:
                lines.append(f"    {p[:120]}")

        mem = live_resources.get("memory")
        if mem:
            lines.append(f"  Memory: {mem['used_mb']}MB / {mem['total_mb']}MB ({mem['used_pct']}%)")

        lines.append("")

    # --- Section 12: Timing Breakdown Summary ---
    lines.append(sep)
    lines.append("  TIMING BREAKDOWN SUMMARY")
    lines.append(sep)
    step_names = ["direct_exp_gen", "coding", "running", "feedback"]
    # Build per-loop timing table
    has_timing = any(ld.get("timing") for ld in loop_data)
    if has_timing:
        # Header
        lines.append(f"  {'Loop':>4s}  {'Factors':>7s}  {'ExpGen':>8s}  {'Coding':>8s}  {'Running':>8s}  {'Feedback':>8s}  {'Total':>8s}  {'Coding%':>7s}  {'CoSTEER':>7s}")
        lines.append("  " + "-" * 85)

        grand_total = {s: 0.0 for s in step_names}
        grand_total_all = 0.0
        for ld in loop_data:
            lid = ld["loop_id"]
            timing = ld.get("timing", {})
            if not timing:
                continue
            durations = {}
            for s in step_names:
                durations[s] = timing.get(s, {}).get("duration_seconds", 0)
                grand_total[s] += durations[s]
            loop_total = sum(durations.values())
            grand_total_all += loop_total
            coding_pct = durations["coding"] / loop_total * 100 if loop_total > 0 else 0
            factor_count = len(ld.get("factor_code", {}).get("factors", []))
            if factor_count == 0:
                # Try from coder result
                factor_count = ld.get("coder_result_count", "?")
            evo_count = ld.get("evo_loop_count", 0)
            lines.append(
                f"  {lid:>4d}  {str(factor_count):>7s}  "
                f"{_fmt_duration(timedelta(seconds=durations['direct_exp_gen'])):>8s}  "
                f"{_fmt_duration(timedelta(seconds=durations['coding'])):>8s}  "
                f"{_fmt_duration(timedelta(seconds=durations['running'])):>8s}  "
                f"{_fmt_duration(timedelta(seconds=durations['feedback'])):>8s}  "
                f"{_fmt_duration(timedelta(seconds=loop_total)):>8s}  "
                f"{coding_pct:>6.1f}%  "
                f"{evo_count:>7d}"
            )

        # Grand total row
        if grand_total_all > 0:
            lines.append("  " + "-" * 85)
            lines.append(
                f"  {'SUM':>4s}  {'':>7s}  "
                f"{_fmt_duration(timedelta(seconds=grand_total['direct_exp_gen'])):>8s}  "
                f"{_fmt_duration(timedelta(seconds=grand_total['coding'])):>8s}  "
                f"{_fmt_duration(timedelta(seconds=grand_total['running'])):>8s}  "
                f"{_fmt_duration(timedelta(seconds=grand_total['feedback'])):>8s}  "
                f"{_fmt_duration(timedelta(seconds=grand_total_all)):>8s}  "
                f"{grand_total['coding'] / grand_total_all * 100:>6.1f}%  "
                f"{'':>7s}"
            )
            lines.append("")
            # Per-step percentage breakdown
            lines.append("  Time distribution:")
            for s in step_names:
                pct = grand_total[s] / grand_total_all * 100
                bar = "#" * int(pct / 2)
                lines.append(f"    {s:20s}: {_fmt_duration(timedelta(seconds=grand_total[s])):>8s} ({pct:5.1f}%) {bar}")
    else:
        lines.append("  No timing data available.")
    lines.append("")

    # --- Section 13: CoSTEER Iteration Detail ---
    lines.append(sep)
    lines.append("  CoSTEER ITERATION DETAIL")
    lines.append(sep)
    has_costeer = any(ld.get("costeer_detail") for ld in loop_data)
    if has_costeer:
        for ld in loop_data:
            cd = ld.get("costeer_detail")
            if not cd:
                continue
            lid = ld["loop_id"]
            total_iters = cd.get("total_evo_iters", 0)
            total_llm = cd.get("total_llm_calls", 0)
            impl_count = cd.get("implemented_count", "?")
            fail_count = cd.get("failed_count", 0)
            final_factors = cd.get("final_factors", [])
            total_factors = len(final_factors) if final_factors else "?"
            hit_max = total_iters >= 5  # CoSTEER max_loop default = 5

            lines.append(f"\n  Loop {lid}: {total_iters} evo iterations, "
                         f"{impl_count}/{total_factors} factors passed"
                         f"{' (HIT MAX_LOOP!)' if hit_max else ''}")

            # Per-iteration detail table
            iters = cd.get("iterations", [])
            if iters:
                lines.append(f"    {'Iteration':>16s}  {'Factors':>7s}  {'LLM Calls':>9s}  {'Need Debug':>10s}")
                lines.append("    " + "-" * 50)
                for it in iters:
                    lines.append(
                        f"    {it['name']:>16s}  {it['factors']:>7d}  "
                        f"{it['llm_calls']:>9d}  {it['needed_debug']:>10d}"
                    )
                lines.append(f"    {'TOTAL':>16s}  {'':>7s}  {total_llm:>9d}")

            # Final factor status
            if final_factors:
                lines.append(f"\n    Final factor status:")
                for f in final_factors:
                    status = "PASS" if f.get("implemented") is True else "FAIL"
                    mark = "+" if status == "PASS" else "X"
                    lines.append(f"      [{mark}] {f.get('name', '?')} -> {status}")

            if fail_count > 0:
                lines.append(f"    WARNING: {fail_count} factor(s) failed after {total_iters} iterations")

        # Summary table
        lines.append(f"\n  CoSTEER Summary:")
        lines.append(f"    {'Loop':>4s}  {'Iters':>5s}  {'Pass':>4s}  {'Fail':>4s}  {'LLM Calls':>9s}  {'Status':>12s}")
        lines.append("    " + "-" * 50)
        for ld in loop_data:
            cd = ld.get("costeer_detail")
            if not cd:
                continue
            lid = ld["loop_id"]
            total_iters = cd.get("total_evo_iters", 0)
            impl = cd.get("implemented_count", 0)
            fail = cd.get("failed_count", 0)
            llm = cd.get("total_llm_calls", 0)
            if total_iters >= 5 and fail > 0:
                status = "HIT MAX+FAIL"
            elif total_iters >= 5:
                status = "HIT MAX"
            elif fail > 0:
                status = "PARTIAL"
            else:
                status = "ALL PASS"
            lines.append(f"    {lid:>4d}  {total_iters:>5d}  {impl:>4d}  {fail:>4d}  {llm:>9d}  {status:>12s}")
    else:
        lines.append("  No CoSTEER iteration data available.")
    lines.append("")

    # --- Section 14: Analytical Conclusions & Recommendations ---
    lines.append(sep)
    lines.append("  ANALYTICAL CONCLUSIONS & RECOMMENDATIONS")
    lines.append(sep)

    conclusions = generate_conclusions(
        loop_data, evolution_analysis, parallelism_analysis,
        propagation_analysis, feedback_analysis, live_resources,
        convergence_analysis, prompt_issues, code_quality_issues,
        homogeneity_analysis,
    )
    for i, c in enumerate(conclusions, 1):
        lines.append(f"  {i}. [{c['category']}] {c['finding']}")
        if c.get("recommendation"):
            lines.append(f"     -> {c['recommendation']}")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def generate_conclusions(
    loop_data: list[dict[str, Any]],
    evolution_analysis: dict[str, Any],
    parallelism_analysis: dict[str, Any],
    propagation_analysis: dict[str, Any],
    feedback_analysis: dict[str, Any],
    live_resources: dict[str, Any] | None,
    convergence_analysis: list[dict[str, Any]] | None = None,
    prompt_issues: list[dict[str, Any]] | None = None,
    code_quality_issues: list[dict[str, Any]] | None = None,
    homogeneity_analysis: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Generate analytical conclusions and recommendations."""
    conclusions: list[dict[str, str]] = []

    # 0. Code quality bugs (critical - these cause silent training failure)
    if code_quality_issues:
        for cqi in code_quality_issues:
            conclusions.append({
                "category": f"CODE_BUG_L{cqi['loop_id']}",
                "finding": f"[{cqi['issue']}] {cqi['detail']}",
                "recommendation": "Fix model code: all nn.Module subclasses must be created in __init__(), not forward().",
            })

    # 0.5 Factor homogeneity issues
    if homogeneity_analysis:
        for hi in homogeneity_analysis.get("issues", []):
            rec = ""
            if hi["issue"] == "CROSS_LOOP_HOMOGENEITY":
                rec = (
                    "New factors reuse fields from previous loops. "
                    "Explore unused field groups (mf_sm/md/lg/elg, bb_*, cp_concentration) "
                    "for genuinely orthogonal signals."
                )
            elif hi["issue"] == "LOW_FIELD_COVERAGE":
                rec = (
                    "Most whitelist fields are unexplored. "
                    "Prompt should encourage exploring new field groups rather than recombining known ones."
                )
            elif hi["issue"] == "HALLUCINATED_FIELDS":
                rec = "Factor code references non-existent fields. Check whitelist constraint in prompts."
            elif hi["issue"] == "NO_NEW_FIELDS":
                rec = (
                    "This loop added zero new fields — pure rehash of previous work. "
                    "Check that hypothesis generation and feedback prompts enforce field diversity."
                )
            elif hi["issue"] == "FEW_NEW_FIELDS":
                rec = (
                    "This loop introduced very few new fields. Target is ≥2 new fields per loop. "
                    "Prompt should enforce 'field usage audit' and prioritize unexplored prefixes."
                )
            elif hi["issue"] == "HIGH_FIELD_OVERLAP":
                rec = (
                    "High overlap with previous loops indicates factors are minor variants, not genuinely new. "
                    "Feedback 'New Hypothesis' should guide toward different data categories."
                )
            elif "UNUSED_GROUP" in hi["issue"]:
                rec = f"Consider adding prompt guidance to explore this field group."
            conclusions.append({
                "category": "HOMOGENEITY",
                "finding": f"[{hi['issue']}] {hi['detail']}",
                "recommendation": rec,
            })

        # Overall innovation summary conclusion
        pli_list = homogeneity_analysis.get("per_loop_innovation", [])
        if len(pli_list) >= 2:
            avg_new = sum(p["new_field_count"] for p in pli_list[1:]) / max(len(pli_list) - 1, 1)
            avg_overlap = sum(p["overlap_rate_with_prev"] for p in pli_list[1:]) / max(len(pli_list) - 1, 1)
            if avg_new < 2 or avg_overlap > 0.5:
                conclusions.append({
                    "category": "INNOVATION",
                    "finding": (
                        f"Factor research lacks innovation: avg {avg_new:.1f} new fields/loop "
                        f"(target ≥2), avg {avg_overlap:.0%} overlap with previous (target <40%)."
                    ),
                    "recommendation": (
                        "Prompts have been updated with field usage audit and anti-repetition constraints. "
                        "Verify next task run shows improved diversity. Consider Layer 2 code injection "
                        "if prompt-only changes are insufficient."
                    ),
                })

            # Performance-based conclusions
            rets = [(p["loop_id"], p["annualized_return"]) for p in pli_list if p.get("annualized_return") is not None]
            dds = [(p["loop_id"], p["max_drawdown"]) for p in pli_list if p.get("max_drawdown") is not None]
            decisions = [p.get("decision") for p in pli_list if p.get("decision") is not None]

            if rets and len(rets) >= 2:
                best_ret_lid, best_ret = max(rets, key=lambda x: x[1])
                first_ret = rets[0][1]
                last_ret = rets[-1][1]
                accept_count = sum(1 for d in decisions if d is True)
                reject_count = sum(1 for d in decisions if d is False)

                # Find SOTA baseline (first ACCEPTED loop)
                sota_ret_val = None
                for p in pli_list:
                    if p.get("decision") is True:
                        sota_ret_val = p.get("annualized_return")
                        break

                if sota_ret_val is not None and reject_count > 0:
                    # Count near-misses (improved but < 10% threshold)
                    near_miss_count = 0
                    for p in pli_list:
                        if p.get("decision") is not False:
                            continue
                        ar = p.get("annualized_return")
                        if ar is not None and ar > sota_ret_val:
                            pct_improve = (ar - sota_ret_val) / abs(sota_ret_val) * 100
                            if 0 < pct_improve < 10:
                                near_miss_count += 1
                    if near_miss_count > 0:
                        conclusions.append({
                            "category": "PERFORMANCE",
                            "finding": (
                                f"{near_miss_count} loop(s) improved annualized return vs SOTA "
                                f"but were REJECTED (improvement < 10% threshold). "
                                f"Best return: Loop {best_ret_lid} = {best_ret:.1%}."
                            ),
                            "recommendation": (
                                "Consider lowering SOTA acceptance threshold from 10% to 5%, "
                                "or using composite criteria (e.g., ACCEPT when ALL core metrics improve)."
                            ),
                        })

                # Overall return trend
                if last_ret > first_ret:
                    conclusions.append({
                        "category": "PERFORMANCE",
                        "finding": f"Annualized return trending up: {first_ret:.1%} -> {last_ret:.1%} (best: Loop {best_ret_lid} = {best_ret:.1%}).",
                        "recommendation": "",
                    })
                elif last_ret < first_ret * 0.9:
                    conclusions.append({
                        "category": "PERFORMANCE",
                        "finding": f"Annualized return degrading: {first_ret:.1%} -> {last_ret:.1%}.",
                        "recommendation": "Factor additions may be introducing noise. Consider factor orthogonalization or selection.",
                    })

            if dds and len(dds) >= 2:
                best_dd_lid, best_dd = max(dds, key=lambda x: x[1])
                worst_dd_lid, worst_dd = min(dds, key=lambda x: x[1])
                if abs(worst_dd) > 0.25:
                    conclusions.append({
                        "category": "PERFORMANCE",
                        "finding": f"Max drawdown exceeded -25% in Loop {worst_dd_lid} ({worst_dd:.1%}). Risk control concern.",
                        "recommendation": "Large drawdown suggests factor combination amplifies tail risk. Consider risk-aware factor selection.",
                    })

    # 1. Training convergence issues (highest priority)
    if convergence_analysis:
        for ca in convergence_analysis:
            lid = ca["loop_id"]
            diag = ca.get("diagnoses", [])
            for d in diag:
                cat = d.split(":")[0] if ":" in d else "TRAINING"
                conclusions.append({
                    "category": f"TRAINING_L{lid}",
                    "finding": d,
                    "recommendation": "",
                })

    # 2. Prompt/config issues
    if prompt_issues:
        for pi in prompt_issues:
            conclusions.append({
                "category": f"CONFIG_L{pi['loop_id']}",
                "finding": f"[{pi['issue']}] {pi['detail']}",
                "recommendation": "",
            })

    # 1. Evolution effectiveness
    trends = evolution_analysis.get("metric_trends", {})
    ic_trend = trends.get("IC", {})
    if ic_trend.get("trend") == "improving":
        conclusions.append({
            "category": "EVOLUTION",
            "finding": f"IC is improving: {ic_trend.get('first', 0):.4f} -> {ic_trend.get('last', 0):.4f}",
            "recommendation": "Hypothesis-driven evolution is working. Continue current strategy.",
        })
    elif ic_trend.get("trend") == "degrading":
        conclusions.append({
            "category": "EVOLUTION",
            "finding": f"IC is degrading: {ic_trend.get('first', 0):.4f} -> {ic_trend.get('last', 0):.4f}",
            "recommendation": "Consider adjusting hypothesis generation prompt or reverting to best-performing config.",
        })

    # Check IC absolute level
    ic_vals = ic_trend.get("values", [])
    if ic_vals:
        last_ic = ic_vals[-1][1]
        if last_ic < 0.03:
            conclusions.append({
                "category": "METRICS",
                "finding": f"IC={last_ic:.4f} is below useful threshold (0.03-0.05).",
                "recommendation": "Focus on factor quality: ensure alpha factors have signal, not just noise. Consider feature engineering prompt improvements.",
            })
        elif last_ic >= 0.05:
            conclusions.append({
                "category": "METRICS",
                "finding": f"IC={last_ic:.4f} is in a healthy range (>0.05).",
                "recommendation": "",
            })

    # 2. Timing analysis
    total_running = 0
    total_coding = 0
    total_exp_gen = 0
    for ld in loop_data:
        timing = ld.get("timing", {})
        total_running += timing.get("running", {}).get("duration_seconds", 0)
        total_coding += timing.get("coding", {}).get("duration_seconds", 0)
        total_exp_gen += timing.get("direct_exp_gen", {}).get("duration_seconds", 0)
    grand_total = total_running + total_coding + total_exp_gen

    if grand_total > 0:
        running_pct = total_running / grand_total * 100
        coding_pct = total_coding / grand_total * 100
        conclusions.append({
            "category": "TIMING",
            "finding": f"Running (backtest) takes {running_pct:.0f}% of total time, coding {coding_pct:.0f}%.",
            "recommendation": (
                "Running step dominates. Consider reducing backtest date range or using lighter model for faster iteration."
                if running_pct > 60
                else "Coding step dominates. Consider reducing CoSTEER evo iterations or simplifying prompts."
                if coding_pct > 60
                else "Time distribution is balanced."
            ),
        })

    # 3. GPU utilization (if live check)
    if live_resources:
        gpus = live_resources.get("gpu", [])
        for g in (gpus or []):
            util = g.get("utilization_pct", 0)
            vram_pct = g.get("memory_pct", 0)
            if util < 20:
                conclusions.append({
                    "category": "GPU",
                    "finding": f"GPU utilization is very low ({util}%). VRAM: {vram_pct:.0f}%.",
                    "recommendation": (
                        "Low GPU usage suggests the model training is CPU-bound or uses a very small model. "
                        "Consider: (1) Increase batch_size to better utilize GPU memory, "
                        "(2) Use larger/deeper models (more layers/hidden units), "
                        "(3) Ensure CUDA is properly configured in WSL, "
                        "(4) Check if DataLoader num_workers is adequate."
                    ),
                })
            elif util < 50:
                conclusions.append({
                    "category": "GPU",
                    "finding": f"GPU utilization is moderate ({util}%). VRAM: {vram_pct:.0f}%.",
                    "recommendation": "Consider increasing batch size or model complexity to better utilize available GPU.",
                })

    # 4. Feedback quality
    poor_count = sum(
        1 for fb in feedback_analysis.get("per_loop", [])
        if fb.get("has_feedback") and not fb.get("has_quantitative_data")
    )
    if poor_count > 0:
        conclusions.append({
            "category": "FEEDBACK",
            "finding": f"{poor_count} loop(s) have feedback without quantitative metrics data.",
            "recommendation": "Ensure the feedback prompt includes actual metric values for better hypothesis guidance.",
        })

    # 5. Propagation check
    chain = propagation_analysis.get("chain_growing")
    if chain is False:
        conclusions.append({
            "category": "PROPAGATION",
            "finding": "based_experiments chain is NOT growing across loops.",
            "recommendation": "Check that accepted experiments are being correctly propagated to the next loop's based_experiments.",
        })

    # 6. Evo loop count
    evo_counts = [ld.get("evo_loop_count", 0) for ld in loop_data]
    if evo_counts and max(evo_counts) <= 1:
        conclusions.append({
            "category": "COSTEER",
            "finding": f"CoSTEER only runs {max(evo_counts)} evolution iteration(s) per loop.",
            "recommendation": "Consider increasing CoSTEER max iterations (e.g., CODER_MAX_ITER=5) to give code more chances to fix errors.",
        })

    # 7. Decision pattern
    decisions = [ld.get("feedback", {}).get("decision") for ld in loop_data if ld.get("feedback")]
    accept_rate = sum(1 for d in decisions if d is True) / max(len(decisions), 1) * 100
    if decisions:
        conclusions.append({
            "category": "DECISIONS",
            "finding": f"Accept rate: {accept_rate:.0f}% ({sum(1 for d in decisions if d)}/{len(decisions)} loops accepted).",
            "recommendation": (
                "All loops rejected. The system may be stuck. Check if the SOTA baseline is too strong or if generated hypotheses are too conservative."
                if accept_rate == 0 and len(decisions) >= 2
                else ""
            ),
        })

    if not conclusions:
        conclusions.append({
            "category": "INFO",
            "finding": "Insufficient data for detailed analysis. Task may still be in early stages.",
            "recommendation": "Wait for more loops to complete before drawing conclusions.",
        })

    return conclusions



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RDAgent Task Diagnostic Analyzer - Standalone tool for analyzing rdagent task runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/rdagent_task_analyzer.py 2026-03-04_00-30-54-801262
  python scripts/rdagent_task_analyzer.py 2026-03-04_00-30-54-801262 --live-check
  python scripts/rdagent_task_analyzer.py 2026-03-04_00-30-54-801262 --json
  python scripts/rdagent_task_analyzer.py 2026-03-04_00-30-54-801262 --log-dir /path/to/logs
        """,
    )
    parser.add_argument("task_id", help="Task ID (directory name under log/)")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help=f"Log root directory (default: {DEFAULT_LOG_DIR})")
    parser.add_argument("--live-check", action="store_true", help="Include live GPU/process/memory check via WSL")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON instead of text report")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) / args.task_id
    if not log_dir.exists():
        print(f"ERROR: Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    # Load session
    session_obj, session_dir, session_hist_len = load_latest_session(log_dir)

    # If no session obj found, try to build trace from hist
    if session_obj is None:
        # Fallback: just scan loop dirs
        pass

    # Extract per-loop data
    loop_dirs = _extract_loop_dirs(log_dir)
    loop_data: list[dict[str, Any]] = []
    for loop_id, loop_path in loop_dirs:
        ld: dict[str, Any] = {"loop_id": loop_id}
        ld["hypothesis"] = extract_hypothesis(loop_path)
        ld["experiment"] = extract_experiment(loop_path)
        ld["runner_result"] = extract_runner_result(loop_path)
        ld["feedback"] = extract_feedback(loop_path)
        ld["timing"] = extract_timing(loop_path)
        ld["evo_loop_count"] = count_evo_loops(loop_path)
        ld["costeer_detail"] = analyze_costeer_detail(loop_path)
        ld["model_code"] = extract_model_code(loop_path)
        ld["training_log"] = extract_training_log(loop_path)

        # Count factor workspaces from coder result (list of FactorFBWorkspace)
        coder_pkls = _find_pkl_files(loop_path, "coding/coder result")
        if coder_pkls:
            coder_obj = safe_pickle_load(coder_pkls[-1])
            if isinstance(coder_obj, list):
                ld["coder_result_count"] = len(coder_obj)

        loop_data.append(ld)

    # Run analyses
    evolution_analysis = analyze_evolution(loop_data)
    parallelism_analysis = analyze_parallelism(loop_data)
    propagation_analysis = analyze_propagation(session_obj, loop_data)
    feedback_analysis = analyze_feedback_quality(loop_data)
    hyperparam_analysis = analyze_hyperparameters(loop_data)
    convergence_analysis = analyze_training_convergence(loop_data)
    costeer_config = extract_costeer_config()
    prompt_config = extract_prompt_config()
    prompt_issues = analyze_prompt_config_consistency(loop_data, prompt_config)
    code_quality_issues = analyze_model_code_quality(loop_data)
    homogeneity_analysis = analyze_factor_homogeneity(loop_data)

    # Live resource check
    live_resources = check_live_resources() if args.live_check else None

    # Generate conclusions
    conclusions = generate_conclusions(
        loop_data, evolution_analysis, parallelism_analysis,
        propagation_analysis, feedback_analysis, live_resources,
        convergence_analysis, prompt_issues, code_quality_issues,
        homogeneity_analysis,
    )

    # Output
    if args.json:
        report = {
            "task_id": args.task_id,
            "log_dir": str(log_dir),
            "loops_completed": len([ld for ld in loop_data if ld.get("feedback")]),
            "loops_total": len(loop_data),
            "session_dir": session_dir,
            "session_hist_len": session_hist_len,
            "loop_data": loop_data,
            "evolution_analysis": evolution_analysis,
            "parallelism_analysis": parallelism_analysis,
            "propagation_analysis": propagation_analysis,
            "feedback_quality": feedback_analysis,
            "hyperparameter_evolution": hyperparam_analysis,
            "convergence_analysis": convergence_analysis,
            "costeer_config": costeer_config,
            "prompt_config": prompt_config,
            "prompt_issues": prompt_issues,
            "code_quality_issues": code_quality_issues,
            "homogeneity_analysis": homogeneity_analysis,
            "live_resources": live_resources,
            "conclusions": conclusions,
        }
        print(json.dumps(report, indent=2, default=str))
    else:
        print(generate_text_report(
            args.task_id, log_dir, loop_data, session_obj, session_dir, session_hist_len,
            evolution_analysis, parallelism_analysis, propagation_analysis,
            feedback_analysis, hyperparam_analysis, convergence_analysis,
            costeer_config, prompt_config, prompt_issues, code_quality_issues,
            live_resources, homogeneity_analysis,
        ))


if __name__ == "__main__":
    main()
