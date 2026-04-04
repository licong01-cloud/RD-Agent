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

import numpy as np
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


def extract_backtest_portfolio(loop_dir: Path) -> dict[str, Any]:
    """Extract detailed backtest portfolio data: positions, indicators, report.

    Locates the experiment workspace from the runner result pkl, then loads
    positions_normal_1day.pkl, indicators_normal_1day.pkl, report_normal_1day.pkl
    from the mlruns artifacts directory.

    If mlruns is not under workspace_path (common for factor experiments), falls
    back to parsing the training log for mlflow experiment_id / run_id and searching
    all workspaces under RD-Agent_workspace.
    """
    pkls = _find_pkl_files(loop_dir, "running/runner result")
    if not pkls:
        return {}
    obj = safe_pickle_load(pkls[-1])
    if obj is None:
        return {}

    # Find workspace path
    ws = getattr(obj, "experiment_workspace", None)
    ws_path = getattr(ws, "workspace_path", None) if ws else None
    if ws_path is None:
        # Try sub_workspace_list
        sub_ws_list = getattr(obj, "sub_workspace_list", None)
        if sub_ws_list:
            ws_path = getattr(sub_ws_list[0], "workspace_path", None)
    if ws_path is None:
        return {}
    ws_path = Path(ws_path)

    # Strategy 1: Find portfolio_analysis directly under workspace mlruns
    pa_dirs = list(ws_path.glob("mlruns/**/portfolio_analysis"))

    # Strategy 2: If not found, parse training log for mlflow IDs and search globally
    if not pa_dirs:
        log_pkls = _find_pkl_files(loop_dir, "running/Qlib_execute_log")
        if log_pkls:
            log_text = safe_pickle_load(log_pkls[-1])
            if isinstance(log_text, str):
                exp_id = None
                run_id = None
                for line in log_text.split("\n"):
                    if "Experiment" in line and "starts running" in line:
                        m = re.search(r"Experiment\s+(\d+)", line)
                        if m:
                            exp_id = m.group(1)
                    if "Recorder" in line and "starts running" in line:
                        m = re.search(r"Recorder\s+([a-f0-9]+)", line)
                        if m:
                            run_id = m.group(1)
                if exp_id and run_id:
                    # Search all RD-Agent_workspace directories
                    ws_root = ws_path.parent  # typically RD-Agent_workspace/
                    if ws_root.exists():
                        found = list(ws_root.glob(f"*/mlruns/{exp_id}/{run_id}/artifacts/portfolio_analysis"))
                        if found:
                            pa_dirs = found

    if not pa_dirs:
        return {}
    pa_dir = pa_dirs[0]

    result: dict[str, Any] = {"workspace_path": str(ws_path), "pa_dir": str(pa_dir)}

    # Load positions
    pos_file = pa_dir / "positions_normal_1day.pkl"
    if pos_file.exists():
        positions = safe_pickle_load(pos_file)
        if positions and isinstance(positions, dict):
            result["positions"] = positions

    # Load indicators
    ind_file = pa_dir / "indicators_normal_1day.pkl"
    if ind_file.exists():
        indicators = safe_pickle_load(ind_file)
        if isinstance(indicators, pd.DataFrame):
            result["indicators"] = indicators

    # Load report
    rpt_file = pa_dir / "report_normal_1day.pkl"
    if rpt_file.exists():
        report = safe_pickle_load(rpt_file)
        if isinstance(report, pd.DataFrame):
            result["report"] = report

    return result


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
    result: dict[str, Any] = {"log_length": len(log_text), "raw_text": log_text}

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

    # Parse epoch-by-epoch loss (supports old and new formats)
    epoch_pattern = re.compile(r"Epoch(\d+):\s+train\s+([\d.]+),\s+valid\s+([\d.]+)")
    epochs = []
    for m in epoch_pattern.finditer(log_text):
        epochs.append({
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "valid_loss": float(m.group(3)),
        })
    # Fallback: valid-only format (per-epoch train eval skipped)
    if not epochs:
        epoch_validonly_pattern = re.compile(r"Epoch(\d+):\s+train\s+N/A.*?valid\s+([\d.eE+-]+)")
        for m in epoch_validonly_pattern.finditer(log_text):
            epochs.append({
                "epoch": int(m.group(1)),
                "train_loss": None,
                "valid_loss": float(m.group(2)),
            })
        if epochs:
            result["train_eval_skipped"] = True
    result["epochs"] = epochs
    result["total_epochs_trained"] = len(epochs)

    # Parse final one-shot train eval (after best model loaded)
    final_train_match = re.search(
        r"Final train eval.*?train\s+([\d.eE+-]+),?\s*valid\s+([\d.eE+-]+)",
        log_text,
    )
    if final_train_match:
        result["final_train_score"] = float(final_train_match.group(1))
        result["final_valid_score"] = float(final_train_match.group(2))

    # Extract early stop info
    early_match = re.search(r"early stop", log_text)
    result["early_stopped"] = early_match is not None

    best_match = re.search(r"best score:\s*([\d.]+)\s*@\s*(\d+)\s*epoch", log_text)
    if best_match:
        result["best_valid_loss"] = float(best_match.group(1))
        result["best_epoch"] = int(best_match.group(2))

    # Extract epoch timestamps for timing analysis
    epoch_time_pattern = re.compile(
        r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\).*Epoch(\d+):"
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
        train_losses = [e["train_loss"] for e in epochs if e.get("train_loss") is not None]
        valid_losses = [e["valid_loss"] for e in epochs]
        train_eval_skipped = tlog.get("train_eval_skipped", False)
        final_train_score = tlog.get("final_train_score")

        info["first_train_loss"] = train_losses[0] if train_losses else None
        info["last_train_loss"] = train_losses[-1] if train_losses else None
        info["first_valid_loss"] = valid_losses[0] if valid_losses else None
        info["last_valid_loss"] = valid_losses[-1] if valid_losses else None
        info["train_eval_skipped"] = train_eval_skipped
        info["final_train_score"] = final_train_score

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
        elif train_eval_skipped and final_train_score is not None and final_train_score > 0.95:
            diagnoses.append("LOSS_NEAR_BASELINE: final one-shot train loss ~1.0 indicates model barely predicts beyond mean")

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
        elif train_eval_skipped and final_train_score is not None and valid_losses:
            best_valid = info.get("best_valid_loss", valid_losses[-1])
            if best_valid is not None:
                gap = best_valid - final_train_score
                if gap > 0.01:
                    diagnoses.append(f"OVERFITTING: final train-valid gap = {gap:.4f} (one-shot eval of best model)")

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
    "sw2": [
        "sw2_open", "sw2_high", "sw2_low", "sw2_close",
        "sw2_pct_change", "sw2_vol", "sw2_amount",
        "sw2_pe", "sw2_pb", "sw2_total_mv",
        "sw2_mf_buy_sm_amt", "sw2_mf_sell_sm_amt",
        "sw2_mf_buy_md_amt", "sw2_mf_sell_md_amt",
        "sw2_mf_buy_lg_amt", "sw2_mf_sell_lg_amt",
        "sw2_mf_buy_elg_amt", "sw2_mf_sell_elg_amt",
        "sw2_mf_net_amt",
        "sw2_mf_buy_elg_vol", "sw2_mf_sell_elg_vol",
        "sw2_mf_net_vol",
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
# Section 4a: Memory & CoSTEER Intensity Profile
# ---------------------------------------------------------------------------

def analyze_memory_profile(
    session_obj: Any,
    loop_data: list[dict[str, Any]],
    log_dir: Path,
) -> dict[str, Any]:
    """Analyze session snapshot sizes, CoSTEER work intensity, and memory risk.

    This section examines:
    - Session snapshot size growth across loops
    - CoSTEER work intensity per loop (evo_iters × factors)
    - Knowledge base and evolving_trace accumulation
    - Estimated evaluation call counts
    - Memory risk assessment
    """
    result: dict[str, Any] = {
        "session_snapshots": [],
        "per_loop_intensity": [],
        "kb_info": None,
        "issues": [],
    }

    # 1. Session snapshot sizes and component breakdown
    session_dir_path = log_dir / "__session__"
    if session_dir_path.is_dir():
        for li_name in sorted(os.listdir(session_dir_path)):
            fb_path = session_dir_path / li_name / "3_feedback"
            if not fb_path.is_file():
                continue
            snap_info: dict[str, Any] = {
                "loop_id": int(li_name),
                "total_mb": fb_path.stat().st_size / 1024 / 1024,
            }
            try:
                sess = safe_pickle_load(fb_path)
                if sess is not None:
                    coder = getattr(sess, "coder", None)
                    if coder:
                        snap_info["coder_mb"] = len(pickle.dumps(coder)) / 1024 / 1024
                        # Knowledge base
                        rag = getattr(coder, "rag", None)
                        kb = getattr(rag, "knowledgebase", None) if rag else None
                        if kb:
                            snap_info["kb_mb"] = len(pickle.dumps(kb)) / 1024 / 1024
                            graph = getattr(kb, "graph", None)
                            if graph and hasattr(graph, "nodes"):
                                snap_info["kb_nodes"] = len(graph.nodes)
                        # Evolving trace
                        ea = getattr(coder, "evolve_agent", None)
                        if ea:
                            et = getattr(ea, "evolving_trace", None)
                            if et:
                                snap_info["evo_trace_mb"] = len(pickle.dumps(et)) / 1024 / 1024
                                snap_info["evo_trace_steps"] = len(et)
                    # Trace hist
                    trace = getattr(sess, "trace", None)
                    if trace:
                        hist = getattr(trace, "hist", [])
                        snap_info["hist_len"] = len(hist)
                        # Count total SOTA factors
                        sota_factors = 0
                        for exp, fb in hist:
                            if fb and getattr(fb, "decision", False):
                                if hasattr(exp, "sub_tasks"):
                                    sota_factors += len(exp.sub_tasks)
                                elif hasattr(exp, "sub_workspace_list"):
                                    sota_factors += len(exp.sub_workspace_list)
                        snap_info["sota_factors"] = sota_factors
                    del sess
            except Exception:
                pass
            result["session_snapshots"].append(snap_info)

    # 2. Per-loop CoSTEER work intensity
    total_work = 0
    total_eval_calls = 0
    max_single_loop_work = 0
    consecutive_max = 0
    max_consecutive_max = 0

    for ld in loop_data:
        loop_id = ld["loop_id"]
        evo_iters = ld.get("evo_loop_count", 0)
        factor_count = ld.get("coder_result_count", 5)
        work = evo_iters * factor_count
        eval_calls = work * 19  # ~19 evaluate calls per factor per iteration
        total_work += work
        total_eval_calls += eval_calls

        if work > max_single_loop_work:
            max_single_loop_work = work

        # Track consecutive MAX_LOOP hits
        if evo_iters >= 5:
            consecutive_max += 1
            max_consecutive_max = max(max_consecutive_max, consecutive_max)
        else:
            consecutive_max = 0

        # Count light loops (1 iteration = quick convergence)
        is_light = evo_iters <= 1

        intensity = {
            "loop_id": loop_id,
            "evo_iters": evo_iters,
            "factor_count": factor_count,
            "work_units": work,
            "eval_calls": eval_calls,
            "cumulative_work": total_work,
            "cumulative_eval_calls": total_eval_calls,
            "hit_max_loop": evo_iters >= 5,
            "is_light": is_light,
        }
        result["per_loop_intensity"].append(intensity)

    # 3. Summary statistics
    n_loops = len(loop_data)
    light_loops = sum(1 for p in result["per_loop_intensity"] if p["is_light"])
    max_loops_hit = sum(1 for p in result["per_loop_intensity"] if p["hit_max_loop"])

    result["summary"] = {
        "total_loops": n_loops,
        "total_work_units": total_work,
        "total_eval_calls": total_eval_calls,
        "max_single_loop_work": max_single_loop_work,
        "avg_work_per_loop": total_work / max(n_loops, 1),
        "light_loops": light_loops,
        "max_loops_hit": max_loops_hit,
        "max_consecutive_max_loop": max_consecutive_max,
    }

    # 4. KB info from last session snapshot
    if result["session_snapshots"]:
        last_snap = result["session_snapshots"][-1]
        result["kb_info"] = {
            "kb_mb": last_snap.get("kb_mb"),
            "kb_nodes": last_snap.get("kb_nodes"),
        }

    # 5. Snapshot growth analysis
    if len(result["session_snapshots"]) >= 2:
        first_total = result["session_snapshots"][0]["total_mb"]
        last_total = result["session_snapshots"][-1]["total_mb"]
        growth_mb = last_total - first_total
        growth_pct = (last_total / first_total - 1) * 100 if first_total > 0 else 0
        result["snapshot_growth"] = {
            "first_mb": first_total,
            "last_mb": last_total,
            "growth_mb": growth_mb,
            "growth_pct": growth_pct,
        }

    # 6. Issue detection
    issues = result["issues"]

    # High CoSTEER intensity
    if total_work > 100:
        issues.append({
            "issue": "HIGH_COSTEER_INTENSITY",
            "detail": (
                f"Total work units = {total_work} (evo_iters×factors), "
                f"total eval calls ≈ {total_eval_calls}. "
                f"High memory churn from pickle deserialization + deepcopy."
            ),
        })

    # Consecutive MAX_LOOP hits
    if max_consecutive_max >= 2:
        issues.append({
            "issue": "CONSECUTIVE_MAX_LOOP",
            "detail": (
                f"{max_consecutive_max} consecutive loops hit MAX_LOOP=5. "
                f"No 'light' loops to relieve memory pressure from glibc arena inflation."
            ),
        })

    # No light loops
    if n_loops >= 4 and light_loops == 0:
        issues.append({
            "issue": "NO_LIGHT_LOOPS",
            "detail": (
                f"Zero light loops (≤1 evo iteration) in {n_loops} loops. "
                f"Continuous high memory churn without GC breathing room."
            ),
        })

    # Single loop overload
    if max_single_loop_work >= 30:
        overload_loop = max(
            result["per_loop_intensity"], key=lambda x: x["work_units"]
        )
        issues.append({
            "issue": "SINGLE_LOOP_OVERLOAD",
            "detail": (
                f"Loop {overload_loop['loop_id']}: {overload_loop['evo_iters']} iters × "
                f"{overload_loop['factor_count']} factors = {overload_loop['work_units']} work units "
                f"(≈{overload_loop['eval_calls']} eval calls). This is exceptionally high."
            ),
        })

    # Snapshot growth
    sg = result.get("snapshot_growth")
    if sg and sg["growth_pct"] > 30:
        issues.append({
            "issue": "SESSION_SNAPSHOT_GROWTH",
            "detail": (
                f"Session snapshot grew from {sg['first_mb']:.1f} MB to {sg['last_mb']:.1f} MB "
                f"(+{sg['growth_pct']:.0f}%). Indicates accumulating state in coder/KB."
            ),
        })

    return result


# ---------------------------------------------------------------------------
# Section 4b: Backtest Portfolio Analysis
# ---------------------------------------------------------------------------


def _extract_cost_rates(workspace_path: str | None) -> dict[str, float]:
    """Extract transaction cost rates from workspace conf_*.yaml files.

    Returns dict with open_cost, close_cost, min_cost.
    Falls back to known defaults from MEMORY.md.
    """
    defaults = {"open_cost": 0.000095, "close_cost": 0.000595, "min_cost": 5.0}
    if workspace_path is None:
        return defaults
    ws = Path(workspace_path)
    conf_files = list(ws.glob("conf_*.yaml")) + list(ws.glob("conf*.yaml"))
    for cf in conf_files:
        try:
            text = cf.read_text(encoding="utf-8")
        except Exception:
            continue
        oc = re.search(r"open_cost\s*:\s*([0-9.eE\-]+)", text)
        cc = re.search(r"close_cost\s*:\s*([0-9.eE\-]+)", text)
        mc = re.search(r"min_cost\s*:\s*([0-9.eE\-]+)", text)
        result = dict(defaults)
        if oc:
            result["open_cost"] = float(oc.group(1))
        if cc:
            result["close_cost"] = float(cc.group(1))
        if mc:
            result["min_cost"] = float(mc.group(1))
        return result
    return defaults


def _get_position_dict(pos_obj: Any) -> dict[str, Any]:
    """Extract raw dict from Position object or plain dict."""
    if isinstance(pos_obj, dict):
        return pos_obj
    # qlib.backtest.position.Position has .position attribute
    p = getattr(pos_obj, "position", None)
    if isinstance(p, dict):
        return p
    return {}


def analyze_per_loop_execution_quality(loop_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract per-loop execution quality metrics: FFR, PA, deal_amount,
    with_cost vs without_cost, and execution strategy from each loop.

    This enables comparison of execution quality across loops to detect
    improvements from strategy changes (e.g., TWAPStrategy → TailTWAPWithLimitStrategy).
    """
    results: list[dict[str, Any]] = []
    for ld in loop_data:
        loop_id = ld["loop_id"]
        entry: dict[str, Any] = {"loop_id": loop_id}

        # --- Extract FFR, PA, deal_amount from indicators ---
        bp = ld.get("backtest_portfolio", {})
        indicators = bp.get("indicators")
        if isinstance(indicators, pd.DataFrame):
            if "ffr" in indicators.columns:
                ffr = indicators["ffr"].dropna()
                if len(ffr) > 0:
                    entry["ffr_mean"] = round(float(ffr.mean()), 4)
                    entry["ffr_min"] = round(float(ffr.min()), 4)
                    entry["days_ffr_below_085"] = int((ffr < 0.85).sum())
            if "pa" in indicators.columns:
                pa = indicators["pa"].dropna()
                if len(pa) > 0:
                    entry["pa_mean"] = round(float(pa.mean()), 6)
                    entry["pa_days_negative"] = int((pa < 0).sum())
            if "deal_amount" in indicators.columns:
                da = indicators["deal_amount"].dropna()
                if len(da) > 0:
                    entry["deal_amount_avg"] = round(float(da.mean()), 0)

        # --- Extract with_cost vs without_cost from runner result ---
        metrics = ld.get("runner_result", {}).get("metrics", {}) or {}
        ann_with = metrics.get("1day.excess_return_with_cost.annualized_return")
        ann_without = metrics.get("1day.excess_return_without_cost.annualized_return")
        if ann_with is not None:
            entry["ann_ret_with_cost"] = round(float(ann_with), 6)
        if ann_without is not None:
            entry["ann_ret_without_cost"] = round(float(ann_without), 6)
        if ann_with is not None and ann_without is not None:
            entry["cost_drag_pct"] = round((ann_without - ann_with) * 100, 2)

        # --- Fallback: FFR/PA from runner_result metrics (1day.ffr, 1day.pa) ---
        if "ffr_mean" not in entry:
            ffr_metric = metrics.get("1day.ffr")
            if ffr_metric is not None:
                entry["ffr_mean"] = round(float(ffr_metric), 4)
        if "pa_mean" not in entry:
            pa_metric = metrics.get("1day.pa")
            if pa_metric is not None:
                entry["pa_mean"] = round(float(pa_metric), 6)

        # --- Detect execution strategy from qrun log and workspace conf ---
        log_text = ld.get("training_log", {}).get("raw_text", "")
        ws_path_str = bp.get("workspace_path", "")
        strategy_detected = False
        if ws_path_str:
            ws_p = Path(ws_path_str)
            for cf in list(ws_p.glob("conf_*.yaml")) + list(ws_p.glob("conf*.yaml")):
                try:
                    ct = cf.read_text(encoding="utf-8")
                    if "TailTWAPWithLimitStrategy" in ct or "tail_twap_strategy" in ct:
                        entry["exec_strategy"] = "TailTWAP+Limit"
                        strategy_detected = True
                    elif "TWAPStrategy" in ct:
                        entry["exec_strategy"] = "TWAP"
                        strategy_detected = True
                    if strategy_detected:
                        break
                except Exception:
                    pass
        if not strategy_detected and log_text:
            if "TailTWAPWithLimitStrategy" in log_text:
                entry["exec_strategy"] = "TailTWAP+Limit"
            elif "TWAPStrategy" in log_text:
                entry["exec_strategy"] = "TWAP"
            elif "NestedExecutor" in log_text:
                entry["exec_strategy"] = "Nested(unknown)"

        # --- Detect minute-line mode ---
        if "Minute memory patch" in log_text or "freq: 1min" in log_text:
            entry["minute_mode"] = True
        elif ws_path_str:
            ws_p = Path(ws_path_str)
            for cf in list(ws_p.glob("conf_*.yaml")) + list(ws_p.glob("conf*.yaml")):
                try:
                    ct = cf.read_text(encoding="utf-8")
                    if "freq: 1min" in ct or "time_per_step: 1min" in ct:
                        entry["minute_mode"] = True
                        break
                except Exception:
                    pass

        results.append(entry)
    return results


def analyze_backtest_portfolio(loop_data: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Comprehensive backtest portfolio analysis: capital utilization, trading
    activity, turnover, per-stock P&L, and trade win rates.

    Uses the last loop that has backtest_portfolio data.
    """
    # Find the latest loop with portfolio data
    portfolio_data = None
    source_loop = -1
    for ld in reversed(loop_data):
        bp = ld.get("backtest_portfolio")
        if bp and "positions" in bp:
            portfolio_data = bp
            source_loop = ld["loop_id"]
            break
    if portfolio_data is None:
        return None

    positions = portfolio_data["positions"]
    indicators: pd.DataFrame | None = portfolio_data.get("indicators")

    # Extract transaction cost rates from workspace config
    cost_rates = _extract_cost_rates(portfolio_data.get("workspace_path"))
    report: pd.DataFrame | None = portfolio_data.get("report")

    dates = sorted(positions.keys())
    n_days = len(dates)
    if n_days < 2:
        return None

    result: dict[str, Any] = {"source_loop": source_loop, "n_trading_days": n_days}
    result["date_range"] = [str(dates[0])[:10], str(dates[-1])[:10]]

    # =========================================================
    # 1. Capital Utilization
    # =========================================================
    utils: list[float] = []
    stock_counts: list[int] = []
    cash_list: list[float] = []
    total_list: list[float] = []

    for d in dates:
        p = _get_position_dict(positions[d])
        cash_d = p.get("cash", 0)
        stocks_d = {k: v for k, v in p.items() if k not in ("cash", "now_account_value") and isinstance(v, dict)}
        sv = sum(v.get("amount", 0) * v.get("price", 0) for v in stocks_d.values())
        total = cash_d + sv
        utils.append(sv / total * 100 if total > 0 else 0)
        stock_counts.append(len(stocks_d))
        cash_list.append(cash_d)
        total_list.append(total)

    result["utilization"] = {
        "avg": round(float(np.mean(utils)), 1),
        "min": round(float(np.min(utils)), 1),
        "max": round(float(np.max(utils)), 1),
        "std": round(float(np.std(utils)), 1),
        "days_below_90": sum(1 for u in utils if u < 90),
        "days_below_80": sum(1 for u in utils if u < 80),
    }
    result["stock_counts"] = {
        "avg": round(float(np.mean(stock_counts)), 1),
        "min": int(np.min(stock_counts)),
        "max": int(np.max(stock_counts)),
    }
    # Stock count distribution (top buckets)
    from collections import Counter
    sc_dist = Counter(stock_counts)
    result["stock_count_distribution"] = dict(sorted(sc_dist.items()))

    # =========================================================
    # 2. Trading Activity (entries/exits)
    # =========================================================
    prev_stocks: set[str] = set()
    daily_entries: list[int] = []
    daily_exits: list[int] = []
    for i, d in enumerate(dates):
        p = _get_position_dict(positions[d])
        curr = set(k for k in p if k not in ("cash", "now_account_value") and isinstance(p[k], dict))
        if i > 0:
            daily_entries.append(len(curr - prev_stocks))
            daily_exits.append(len(prev_stocks - curr))
        prev_stocks = curr

    result["trading_activity"] = {
        "avg_daily_entries": round(float(np.mean(daily_entries)), 1) if daily_entries else 0,
        "avg_daily_exits": round(float(np.mean(daily_exits)), 1) if daily_exits else 0,
        "max_daily_entries": int(np.max(daily_entries)) if daily_entries else 0,
        "max_daily_exits": int(np.max(daily_exits)) if daily_exits else 0,
        "zero_trade_days": sum(1 for e, x in zip(daily_entries, daily_exits) if e == 0 and x == 0),
    }

    # =========================================================
    # 3. Turnover Estimation (from position changes)
    # =========================================================
    turnovers: list[float] = []
    for i in range(1, n_days):
        p_prev = _get_position_dict(positions[dates[i - 1]])
        p_curr = _get_position_dict(positions[dates[i]])

        prev_stk = {k: v.get("amount", 0) * v.get("price", 0)
                     for k, v in p_prev.items()
                     if k not in ("cash", "now_account_value") and isinstance(v, dict)}
        curr_stk = {k: v.get("amount", 0) * v.get("price", 0)
                     for k, v in p_curr.items()
                     if k not in ("cash", "now_account_value") and isinstance(v, dict)}

        traded_value = 0.0
        for s in set(prev_stk) | set(curr_stk):
            pv = prev_stk.get(s, 0)
            cv = curr_stk.get(s, 0)
            if pv == 0:
                traded_value += cv  # new buy
            elif cv == 0:
                traded_value += pv  # full sell

        total_val = sum(curr_stk.values())
        if total_val > 0:
            turnovers.append(traded_value / total_val)

    avg_turnover = float(np.mean(turnovers)) if turnovers else 0
    result["turnover"] = {
        "avg_daily": round(avg_turnover, 4),
        "avg_daily_pct": round(avg_turnover * 100, 2),
        "annualized": round(avg_turnover * 252, 2),
    }

    # =========================================================
    # 4. Return Analysis
    # =========================================================
    initial_val = total_list[0] if total_list[0] > 0 else 1e8
    final_val = total_list[-1]
    n_years = n_days / 252
    total_ret = final_val / initial_val - 1
    cagr = (final_val / initial_val) ** (1 / n_years) - 1 if n_years > 0 else 0

    result["returns"] = {
        "initial_capital": round(initial_val, 0),
        "final_value": round(final_val, 0),
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "n_years": round(n_years, 2),
    }

    # =========================================================
    # 4b. Monthly Returns
    # =========================================================
    if len(total_list) >= 2 and len(dates) == len(total_list):
        monthly_returns: dict[str, float] = {}
        prev_month_val = total_list[0]
        prev_month_key = str(dates[0])[:7]
        for i in range(1, len(dates)):
            month_key = str(dates[i])[:7]
            if month_key != prev_month_key:
                # End of previous month
                monthly_returns[prev_month_key] = round(
                    (total_list[i - 1] / prev_month_val - 1) * 100, 2
                ) if prev_month_val > 0 else 0.0
                prev_month_val = total_list[i - 1]
                prev_month_key = month_key
        # Final month
        monthly_returns[prev_month_key] = round(
            (total_list[-1] / prev_month_val - 1) * 100, 2
        ) if prev_month_val > 0 else 0.0
        result["monthly_returns"] = monthly_returns

    # =========================================================
    # 4c. Risk Metrics (Sharpe, Sortino, MaxDD Duration, Calmar)
    # =========================================================
    if len(total_list) >= 3:
        daily_returns = np.diff(total_list) / np.array(total_list[:-1])
        daily_returns = daily_returns[np.isfinite(daily_returns)]

        if len(daily_returns) > 1:
            avg_ret = float(np.mean(daily_returns))
            std_ret = float(np.std(daily_returns, ddof=1))
            sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

            downside = daily_returns[daily_returns < 0]
            downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
            sortino = (avg_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

            # Max drawdown duration
            cummax = np.maximum.accumulate(total_list)
            drawdowns = (np.array(total_list) - cummax) / cummax
            max_dd_pct = float(np.min(drawdowns)) * 100

            # Duration of max drawdown (days in drawdown state)
            in_dd = drawdowns < 0
            max_dd_duration = 0
            cur_dd_duration = 0
            for dd_flag in in_dd:
                if dd_flag:
                    cur_dd_duration += 1
                    max_dd_duration = max(max_dd_duration, cur_dd_duration)
                else:
                    cur_dd_duration = 0

            calmar = (cagr / abs(max_dd_pct / 100)) if max_dd_pct < 0 else 0.0

            result["risk_metrics"] = {
                "sharpe_ratio": round(sharpe, 3),
                "sortino_ratio": round(sortino, 3),
                "max_drawdown_pct": round(max_dd_pct, 2),
                "max_drawdown_duration_days": max_dd_duration,
                "calmar_ratio": round(calmar, 3),
                "daily_volatility_pct": round(std_ret * 100, 4),
                "annualized_volatility_pct": round(std_ret * np.sqrt(252) * 100, 2),
            }

    # =========================================================
    # 5. Indicators Summary (FFR, PA, etc.)
    # =========================================================
    if indicators is not None and "ffr" in indicators.columns:
        ffr = indicators["ffr"].dropna()
        result["ffr_analysis"] = {
            "mean": round(float(ffr.mean()), 4),
            "min": round(float(ffr.min()), 4),
            "std": round(float(ffr.std()), 4),
            "days_below_085": int((ffr < 0.85).sum()),
            "days_below_070": int((ffr < 0.70).sum()),
        }
        # FFR buckets
        bins = [0, 0.80, 0.90, 0.95, 1.01]
        labels = ["<0.80", "0.80-0.90", "0.90-0.95", "0.95-1.00"]
        ffr_bucket = pd.cut(ffr, bins=bins, labels=labels)
        result["ffr_buckets"] = {
            lab: int(cnt) for lab, cnt in ffr_bucket.value_counts().items()
        }
    if indicators is not None and "pa" in indicators.columns:
        pa = indicators["pa"].dropna()
        if len(pa) > 0:
            result["pa_analysis"] = {
                "mean": round(float(pa.mean()), 6),
                "median": round(float(pa.median()), 6),
                "min": round(float(pa.min()), 6),
                "max": round(float(pa.max()), 6),
                "std": round(float(pa.std()), 6),
                "days_negative": int((pa < 0).sum()),
                "pct_negative": round((pa < 0).sum() / len(pa) * 100, 1),
            }

    if indicators is not None and "deal_amount" in indicators.columns:
        da = indicators["deal_amount"]
        result["deal_amount"] = {
            "avg_daily": round(float(da.mean()), 0),
            "total": round(float(da.sum()), 0),
        }

    # =========================================================
    # 6. Per-Stock P&L and Win Rate Analysis
    #    Uses Qlib close price data for accurate exit prices
    #    (especially for 1-day holdings where snapshot only has entry close)
    # =========================================================

    # Load Qlib close prices for accurate sell-day pricing
    close_lookup: dict[tuple[str, str], float] = {}
    _qlib_loaded = False
    try:
        import qlib
        from qlib.data import D
        # Determine qlib data path
        qlib_bin_paths = [
            Path("/home/lc999/data/qlib_bin"),
            Path.home() / "data" / "qlib_bin",
            Path.home() / "data" / "qlib_data",
        ]
        qlib_path = None
        for p in qlib_bin_paths:
            if p.exists() and (p / "instruments").exists():
                qlib_path = p
                break
        if qlib_path:
            try:
                qlib.init(provider_uri={"day": str(qlib_path)}, expression_cache=None, dataset_cache=None)
            except Exception:
                pass  # already initialized
            date_range_start = str(dates[0])[:10]
            date_range_end = str(dates[-1])[:10]
            # Pad by a few days for next-day lookups
            from datetime import datetime as _dt, timedelta as _td
            start_dt = _dt.strptime(date_range_start, "%Y-%m-%d") - _td(days=5)
            end_dt = _dt.strptime(date_range_end, "%Y-%m-%d") + _td(days=10)
            close_df = D.features(
                instruments=D.instruments(market="all"),
                fields=["$close"],
                start_time=start_dt.strftime("%Y-%m-%d"),
                end_time=end_dt.strftime("%Y-%m-%d"),
                freq="day",
            )
            close_df.columns = ["close"]
            close_df = close_df.reset_index()
            close_df.columns = ["instrument", "datetime", "close"]
            for _, row in close_df.iterrows():
                key = (row["instrument"], str(row["datetime"])[:10])
                close_lookup[key] = float(row["close"])
            _qlib_loaded = True
    except Exception:
        pass  # Qlib not available; fall back to snapshot prices

    # Track each stock's complete trade history
    stock_history: dict[str, list[dict]] = defaultdict(list)
    for d in dates:
        p = _get_position_dict(positions[d])
        for sid, v in p.items():
            if sid in ("cash", "now_account_value") or not isinstance(v, dict):
                continue
            stock_history[sid].append({
                "date": str(d)[:10],
                "amount": float(v.get("amount", 0)),
                "price": float(v.get("price", 0)),
            })

    all_dates_str = [str(d)[:10] for d in dates]

    # Build per-stock trade records: identify entry/exit pairs
    trade_records: list[dict[str, Any]] = []
    open_positions: dict[str, dict] = {}

    for sid, hist in stock_history.items():
        date_set = set(h["date"] for h in hist)
        periods: list[list[dict]] = []
        current_period: list[dict] = []

        for ds in all_dates_str:
            if ds in date_set:
                matching = [h for h in hist if h["date"] == ds]
                if matching:
                    if not current_period:
                        current_period = [matching[0]]
                    else:
                        current_period.append(matching[0])
            else:
                if current_period:
                    periods.append(current_period)
                    current_period = []
        if current_period:
            periods.append(current_period)

        for period in periods:
            entry_price = period[0]["price"]
            entry_date = period[0]["date"]
            exit_date = period[-1]["date"]
            exit_price_snapshot = period[-1]["price"]
            holding_days = len(period)
            amount = period[0]["amount"]

            is_last_date = (exit_date == all_dates_str[-1])
            if is_last_date and exit_date in date_set:
                pnl_pct = (exit_price_snapshot / entry_price - 1) * 100 if entry_price > 0 else 0
                open_positions[sid] = {
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "current_price": exit_price_snapshot,
                    "holding_days": holding_days,
                    "amount": amount,
                    "unrealized_pnl_pct": round(pnl_pct, 2),
                }
            else:
                # Determine actual exit price:
                # Sell happens on the NEXT trading day after last holding day
                exit_date_idx = all_dates_str.index(exit_date)
                sell_date_idx = exit_date_idx + 1
                if sell_date_idx < len(all_dates_str):
                    sell_date = all_dates_str[sell_date_idx]
                else:
                    sell_date = exit_date

                # Try Qlib close price for sell_date (more accurate)
                exit_price_actual = close_lookup.get((sid, sell_date))
                if exit_price_actual is None or np.isnan(exit_price_actual):
                    # Fallback: use last holding day's snapshot close
                    exit_price_actual = exit_price_snapshot
                    price_source = "snapshot"
                else:
                    price_source = "qlib_next_day"

                pnl_pct = (exit_price_actual / entry_price - 1) * 100 if entry_price > 0 else 0
                pnl_value = (exit_price_actual - entry_price) * amount

                trade_records.append({
                    "stock": sid,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "sell_date": sell_date,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price_actual, 4),
                    "amount": amount,
                    "holding_days": holding_days,
                    "pnl_pct": round(pnl_pct, 2),
                    "pnl_value": round(pnl_value, 2),
                    "price_source": price_source,
                })

    result["qlib_price_loaded"] = _qlib_loaded
    result["close_lookup_size"] = len(close_lookup)

    # Per-trade cost computation
    open_cost = cost_rates["open_cost"]
    close_cost = cost_rates["close_cost"]
    min_cost = cost_rates["min_cost"]
    for t in trade_records:
        buy_value = t["entry_price"] * t["amount"]
        sell_value = t["exit_price"] * t["amount"]
        t["buy_cost"] = round(max(buy_value * open_cost, min_cost), 2)
        t["sell_cost"] = round(max(sell_value * close_cost, min_cost), 2)
        t["total_cost"] = round(t["buy_cost"] + t["sell_cost"], 2)
        t["net_pnl"] = round(t["pnl_value"] - t["total_cost"], 2)

    result["cost_rates"] = cost_rates

    # Win rate statistics
    if trade_records:
        one_day_trades = [t for t in trade_records if t["holding_days"] == 1]
        multi_day_trades = [t for t in trade_records if t["holding_days"] > 1]

        # When Qlib prices are loaded, 1-day trades have accurate P&L → use ALL trades
        # When Qlib is not available, 1-day trades have pnl=0 → use multi-day only
        if _qlib_loaded:
            effective_trades = trade_records
        else:
            effective_trades = multi_day_trades if multi_day_trades else trade_records
        wins = [t for t in effective_trades if t["pnl_pct"] > 0]
        losses = [t for t in effective_trades if t["pnl_pct"] < 0]
        flat = [t for t in effective_trades if t["pnl_pct"] == 0]
        pnl_pcts = [t["pnl_pct"] for t in effective_trades]
        win_pnls = [t["pnl_pct"] for t in wins]
        loss_pnls = [t["pnl_pct"] for t in losses]
        holding_days_list = [t["holding_days"] for t in effective_trades]

        win_rate = len(wins) / len(effective_trades) * 100 if effective_trades else 0

        # Profit factor = gross profit / gross loss
        gross_profit = sum(t["pnl_value"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl_value"] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # 1-day trade stats
        one_day_wins = [t for t in one_day_trades if t["pnl_pct"] > 0]
        one_day_losses = [t for t in one_day_trades if t["pnl_pct"] < 0]
        one_day_wr = len(one_day_wins) / (len(one_day_wins) + len(one_day_losses)) * 100 if (one_day_wins or one_day_losses) else 0
        one_day_net = sum(t["pnl_value"] for t in one_day_trades)

        # Multi-day trade stats
        multi_wins = [t for t in multi_day_trades if t["pnl_pct"] > 0]
        multi_losses = [t for t in multi_day_trades if t["pnl_pct"] < 0]
        multi_wr = len(multi_wins) / (len(multi_wins) + len(multi_losses)) * 100 if (multi_wins or multi_losses) else 0
        multi_net = sum(t["pnl_value"] for t in multi_day_trades)

        result["trade_stats"] = {
            "total_trades": len(trade_records),
            "one_day_trades": len(one_day_trades),
            "one_day_pct": round(len(one_day_trades) / len(trade_records) * 100, 1),
            "one_day_win_rate": round(one_day_wr, 1),
            "one_day_net_pnl": round(one_day_net, 0),
            "multi_day_trades": len(multi_day_trades),
            "multi_day_win_rate": round(multi_wr, 1),
            "multi_day_net_pnl": round(multi_net, 0),
            "effective_trades": len(effective_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "flat_trades": len(flat),
            "win_rate_pct": round(win_rate, 1),
            "avg_pnl_pct": round(float(np.mean(pnl_pcts)), 2) if pnl_pcts else 0,
            "avg_win_pct": round(float(np.mean(win_pnls)), 2) if win_pnls else 0,
            "avg_loss_pct": round(float(np.mean(loss_pnls)), 2) if loss_pnls else 0,
            "max_win_pct": round(float(np.max(win_pnls)), 2) if win_pnls else 0,
            "max_loss_pct": round(float(np.min(loss_pnls)), 2) if loss_pnls else 0,
            "median_pnl_pct": round(float(np.median(pnl_pcts)), 2) if pnl_pcts else 0,
            "profit_factor": round(profit_factor, 2),
            "gross_profit": round(gross_profit, 0),
            "gross_loss": round(gross_loss, 0),
            "net_pnl": round(gross_profit - gross_loss, 0),
            "avg_holding_days": round(float(np.mean(holding_days_list)), 1) if holding_days_list else 0,
            "median_holding_days": int(np.median(holding_days_list)) if holding_days_list else 0,
            "qlib_price_used": _qlib_loaded,
        }

        # Top 10 most profitable trades
        sorted_trades = sorted(trade_records, key=lambda t: t["pnl_value"], reverse=True)
        result["top_winners"] = sorted_trades[:10]
        result["top_losers"] = sorted_trades[-10:][::-1]  # worst first

        # Transaction cost summary
        total_commission = sum(t["total_cost"] for t in trade_records)
        avg_cost_per_trade = total_commission / len(trade_records) if trade_records else 0
        initial_val_for_cost = total_list[0] if total_list and total_list[0] > 0 else 1e8
        n_years_cost = n_days / 252 if n_days > 0 else 1
        cost_drag_annualized_pct = (total_commission / initial_val_for_cost / n_years_cost) * 100
        result["cost_analysis"] = {
            "open_cost_rate": open_cost,
            "close_cost_rate": close_cost,
            "min_cost": min_cost,
            "total_commission": round(total_commission, 2),
            "avg_cost_per_trade": round(avg_cost_per_trade, 2),
            "cost_drag_annualized_pct": round(cost_drag_annualized_pct, 2),
        }

        # Per-stock aggregated P&L (enhanced)
        stock_pnl: dict[str, dict] = defaultdict(lambda: {
            "total_pnl": 0.0, "total_pnl_net": 0.0, "total_cost": 0.0,
            "total_buy_value": 0.0, "total_sell_value": 0.0,
            "trades": 0, "wins": 0, "total_pnl_pct": 0.0,
            "holding_days_sum": 0, "max_win_pct": -999.0, "max_loss_pct": 999.0,
        })
        for t in trade_records:
            s = t["stock"]
            sp = stock_pnl[s]
            sp["total_pnl"] += t["pnl_value"]
            sp["total_pnl_net"] += t["net_pnl"]
            sp["total_cost"] += t["total_cost"]
            sp["total_buy_value"] += t["entry_price"] * t["amount"]
            sp["total_sell_value"] += t["exit_price"] * t["amount"]
            sp["total_pnl_pct"] += t["pnl_pct"]
            sp["trades"] += 1
            sp["holding_days_sum"] += t["holding_days"]
            if t["pnl_pct"] > 0:
                sp["wins"] += 1
            if t["pnl_pct"] > sp["max_win_pct"]:
                sp["max_win_pct"] = t["pnl_pct"]
            if t["pnl_pct"] < sp["max_loss_pct"]:
                sp["max_loss_pct"] = t["pnl_pct"]

        def _stock_summary(s: str, v: dict) -> dict:
            return {
                "stock": s,
                "total_pnl": round(v["total_pnl"], 0),
                "total_pnl_net": round(v["total_pnl_net"], 0),
                "total_cost": round(v["total_cost"], 2),
                "trades": v["trades"],
                "wins": v["wins"],
                "win_rate": round(v["wins"] / v["trades"] * 100, 1) if v["trades"] > 0 else 0,
                "avg_pnl_pct": round(v["total_pnl_pct"] / v["trades"], 2) if v["trades"] > 0 else 0,
                "avg_holding_days": round(v["holding_days_sum"] / v["trades"], 1) if v["trades"] > 0 else 0,
                "max_win_pct": round(v["max_win_pct"], 2) if v["max_win_pct"] > -999 else 0,
                "max_loss_pct": round(v["max_loss_pct"], 2) if v["max_loss_pct"] < 999 else 0,
            }

        sorted_stocks = sorted(stock_pnl.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
        result["all_stocks_by_pnl"] = [_stock_summary(s, v) for s, v in sorted_stocks]
        result["top_stocks_by_pnl"] = [_stock_summary(s, v) for s, v in sorted_stocks[:15]]
        result["bottom_stocks_by_pnl"] = [_stock_summary(s, v) for s, v in sorted_stocks[-15:][::-1]]

        # Consecutive win/loss streaks
        max_con_wins = 0
        max_con_losses = 0
        cur_wins = 0
        cur_losses = 0
        # Sort trades by exit date for streak analysis
        sorted_by_exit = sorted(trade_records, key=lambda t: t["sell_date"])
        for t in sorted_by_exit:
            if t["pnl_pct"] > 0:
                cur_wins += 1
                cur_losses = 0
            elif t["pnl_pct"] < 0:
                cur_losses += 1
                cur_wins = 0
            else:
                cur_wins = 0
                cur_losses = 0
            max_con_wins = max(max_con_wins, cur_wins)
            max_con_losses = max(max_con_losses, cur_losses)
        result["streaks"] = {
            "max_consecutive_wins": max_con_wins,
            "max_consecutive_losses": max_con_losses,
        }

        # Win rate by holding period bucket
        bucket_stats: dict[str, dict] = {}
        for label, lo, hi in [("1d", 1, 1), ("2-3d", 2, 3), ("4-7d", 4, 7), ("8-14d", 8, 14), ("15+d", 15, 9999)]:
            bucket_trades = [t for t in trade_records if lo <= t["holding_days"] <= hi]
            if bucket_trades:
                bw = sum(1 for t in bucket_trades if t["pnl_pct"] > 0)
                bucket_stats[label] = {
                    "count": len(bucket_trades),
                    "win_rate": round(bw / len(bucket_trades) * 100, 1),
                    "avg_pnl": round(float(np.mean([t["pnl_pct"] for t in bucket_trades])), 2),
                }
        result["win_rate_by_holding_period"] = bucket_stats

    result["open_positions_count"] = len(open_positions)
    result["trade_records"] = trade_records

    # =========================================================
    # 7. Diagnoses
    # =========================================================
    issues: list[dict[str, str]] = []

    avg_util = result["utilization"]["avg"]
    if avg_util < 85:
        issues.append({"code": "LOW_UTILIZATION", "detail": f"Average capital utilization {avg_util:.1f}% < 85%. Significant cash drag on returns."})

    ffr_info = result.get("ffr_analysis")
    if ffr_info and ffr_info["mean"] < 0.95:
        issues.append({"code": "LOW_FFR", "detail": f"Mean FFR {ffr_info['mean']:.3f} < 0.95. {ffr_info.get('days_below_085', 0)} days below 0.85. Check only_tradable config."})

    ts = result.get("trade_stats")
    if ts:
        if ts["win_rate_pct"] < 45:
            issues.append({"code": "LOW_WIN_RATE", "detail": f"Win rate {ts['win_rate_pct']:.1f}% < 45%. Strategy may lack edge."})
        if ts["profit_factor"] < 1.0:
            issues.append({"code": "NEGATIVE_EXPECTANCY", "detail": f"Profit factor {ts['profit_factor']:.2f} < 1.0. Strategy is net losing money on trades."})
        if ts["avg_holding_days"] < 2:
            issues.append({"code": "EXCESSIVE_CHURN", "detail": f"Avg holding {ts['avg_holding_days']:.1f} days. Extreme short-term turnover."})

    turnover_info = result.get("turnover")
    if turnover_info and turnover_info["annualized"] > 50:
        issues.append({"code": "HIGH_TURNOVER", "detail": f"Annualized turnover {turnover_info['annualized']:.1f}x. Transaction costs may erode alpha significantly."})

    ca = result.get("cost_analysis")
    if ca and ca["cost_drag_annualized_pct"] > 2.0:
        issues.append({"code": "HIGH_COST_DRAG", "detail": f"Annualized cost drag {ca['cost_drag_annualized_pct']:.2f}% > 2%. Transaction costs significantly eroding returns."})

    rm = result.get("risk_metrics")
    if rm:
        if rm["sharpe_ratio"] < 0.5:
            issues.append({"code": "LOW_SHARPE", "detail": f"Sharpe ratio {rm['sharpe_ratio']:.3f} < 0.5. Risk-adjusted returns are poor."})
        if rm["max_drawdown_duration_days"] > 60:
            issues.append({"code": "PROLONGED_DRAWDOWN", "detail": f"Max drawdown lasted {rm['max_drawdown_duration_days']} days (> 60). Strategy recovery is slow."})

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
    memory_profile: dict[str, Any] | None = None,
    portfolio_analysis: dict[str, Any] | None = None,
    execution_quality: list[dict[str, Any]] | None = None,
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
                if ca.get("train_eval_skipped"):
                    ft = ca.get("final_train_score")
                    if ft is not None:
                        lines.append(f"    Train Loss:       per-epoch skipped, final one-shot = {ft}")
                    else:
                        lines.append(f"    Train Loss:       per-epoch skipped (no final eval)")
                else:
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
        for grp_name in ["db", "mf", "bb", "cp", "sw2", "precomputed", "daily_pv"]:
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

    # --- Section 15: Memory & CoSTEER Intensity Profile ---
    if memory_profile:
        lines.append(sep)
        lines.append("  MEMORY & COSTEER INTENSITY PROFILE")
        lines.append(sep)

        # Session snapshot table
        snaps = memory_profile.get("session_snapshots", [])
        if snaps:
            lines.append("")
            lines.append("  Session Snapshot Sizes:")
            lines.append(f"    {'Session':>8s}  {'Total':>8s}  {'Coder':>8s}  {'KB':>8s}  {'KB Nodes':>9s}  {'EvoTrace':>9s}  {'EvoSteps':>9s}  {'Hist':>5s}  {'SOTA Fac':>9s}")
            lines.append(f"    {'-'*80}")
            for s in snaps:
                total = f"{s.get('total_mb', 0):.1f} MB"
                coder = f"{s.get('coder_mb', 0):.1f} MB" if s.get('coder_mb') else "N/A"
                kb = f"{s.get('kb_mb', 0):.1f} MB" if s.get('kb_mb') else "N/A"
                kb_n = f"{s.get('kb_nodes', 0):>5d}" if s.get('kb_nodes') else "N/A"
                et = f"{s.get('evo_trace_mb', 0):.1f} MB" if s.get('evo_trace_mb') else "N/A"
                et_s = f"{s.get('evo_trace_steps', 0):>5d}" if s.get('evo_trace_steps') is not None else "N/A"
                hist = f"{s.get('hist_len', 0):>3d}" if s.get('hist_len') is not None else "N/A"
                sf = f"{s.get('sota_factors', 0):>5d}" if s.get('sota_factors') is not None else "N/A"
                lines.append(f"    {s['loop_id']:>8d}  {total:>8s}  {coder:>8s}  {kb:>8s}  {kb_n:>9s}  {et:>9s}  {et_s:>9s}  {hist:>5s}  {sf:>9s}")

        # Snapshot growth
        sg = memory_profile.get("snapshot_growth")
        if sg:
            lines.append(f"\n  Snapshot Growth: {sg['first_mb']:.1f} MB -> {sg['last_mb']:.1f} MB "
                         f"(+{sg['growth_mb']:.1f} MB, +{sg['growth_pct']:.0f}%)")

        # CoSTEER work intensity table
        pli = memory_profile.get("per_loop_intensity", [])
        if pli:
            lines.append("")
            lines.append("  CoSTEER Work Intensity:")
            lines.append(f"    {'Loop':>5s}  {'Iters':>5s}  {'Factors':>7s}  {'Work':>5s}  {'EvalCalls':>10s}  {'Cumul.Work':>11s}  {'Cumul.Eval':>11s}  {'Status':>12s}")
            lines.append(f"    {'-'*80}")
            for p in pli:
                status = ""
                if p["hit_max_loop"]:
                    status = "HIT MAX"
                elif p["is_light"]:
                    status = "light"
                lines.append(
                    f"    {p['loop_id']:>5d}  {p['evo_iters']:>5d}  {p['factor_count']:>7d}  "
                    f"{p['work_units']:>5d}  {p['eval_calls']:>10d}  "
                    f"{p['cumulative_work']:>11d}  {p['cumulative_eval_calls']:>11d}  {status:>12s}"
                )

        # Summary
        summary = memory_profile.get("summary", {})
        if summary:
            lines.append(f"\n  Intensity Summary:")
            lines.append(f"    Total work units:         {summary['total_work_units']}")
            lines.append(f"    Total eval calls:         ~{summary['total_eval_calls']}")
            lines.append(f"    Max single-loop work:     {summary['max_single_loop_work']}")
            lines.append(f"    Avg work/loop:            {summary['avg_work_per_loop']:.1f}")
            lines.append(f"    Light loops (≤1 iter):    {summary['light_loops']}/{summary['total_loops']}")
            lines.append(f"    MAX_LOOP hits:            {summary['max_loops_hit']}")
            lines.append(f"    Consecutive MAX_LOOP:     {summary['max_consecutive_max_loop']}")

        # Issues
        issues = memory_profile.get("issues", [])
        if issues:
            lines.append(f"\n  Memory Risk Issues:")
            for mi in issues:
                lines.append(f"    [!] {mi['issue']}: {mi['detail']}")

        lines.append("")

    # --- Section 17: Backtest Portfolio & Trade Analysis ---
    if portfolio_analysis:
        lines.append(sep)
        lines.append("  BACKTEST PORTFOLIO & TRADE ANALYSIS")
        lines.append(sep)
        lines.append(f"  Source: Loop {portfolio_analysis['source_loop']}, "
                      f"{portfolio_analysis['n_trading_days']} trading days "
                      f"({portfolio_analysis['date_range'][0]} ~ {portfolio_analysis['date_range'][1]})")

        # Returns
        ret = portfolio_analysis.get("returns", {})
        lines.append(f"\n  --- Returns ---")
        lines.append(f"    Initial Capital:   {ret.get('initial_capital', 0):>15,.0f}")
        lines.append(f"    Final Value:       {ret.get('final_value', 0):>15,.0f}")
        lines.append(f"    Total Return:      {ret.get('total_return_pct', 0):>14.2f}%")
        lines.append(f"    CAGR:              {ret.get('cagr_pct', 0):>14.2f}%")

        # Capital utilization
        util = portfolio_analysis.get("utilization", {})
        lines.append(f"\n  --- Capital Utilization ---")
        lines.append(f"    Average:           {util.get('avg', 0):>14.1f}%")
        lines.append(f"    Min / Max:         {util.get('min', 0):.1f}% / {util.get('max', 0):.1f}%")
        lines.append(f"    Days < 90%:        {util.get('days_below_90', 0):>14d}")
        lines.append(f"    Days < 80%:        {util.get('days_below_80', 0):>14d}")

        # Stock counts
        sc = portfolio_analysis.get("stock_counts", {})
        lines.append(f"\n  --- Stock Holdings ---")
        lines.append(f"    Avg / Min / Max:   {sc.get('avg', 0):.1f} / {sc.get('min', 0)} / {sc.get('max', 0)}")
        # Distribution
        dist = portfolio_analysis.get("stock_count_distribution", {})
        if dist:
            lines.append(f"    Distribution:")
            for cnt, days in sorted(dist.items()):
                bar = "#" * min(days, 60)
                lines.append(f"      {cnt:>3d} stocks: {days:>4d} days {bar}")

        # Turnover
        to = portfolio_analysis.get("turnover", {})
        lines.append(f"\n  --- Turnover ---")
        lines.append(f"    Avg Daily:         {to.get('avg_daily_pct', 0):>14.2f}%")
        lines.append(f"    Annualized:        {to.get('annualized', 0):>14.2f}x")

        # Trading activity
        ta = portfolio_analysis.get("trading_activity", {})
        lines.append(f"\n  --- Daily Trading Activity ---")
        lines.append(f"    Avg Entries/Exits:  {ta.get('avg_daily_entries', 0):.1f} / {ta.get('avg_daily_exits', 0):.1f}")
        lines.append(f"    Max Entries/Exits:  {ta.get('max_daily_entries', 0)} / {ta.get('max_daily_exits', 0)}")
        lines.append(f"    Zero-trade Days:   {ta.get('zero_trade_days', 0):>14d}")

        # FFR
        ffr = portfolio_analysis.get("ffr_analysis")
        if ffr:
            lines.append(f"\n  --- Fill Fulfillment Rate (FFR) ---")
            lines.append(f"    Mean FFR:          {ffr['mean']:>14.4f}")
            lines.append(f"    Min FFR:           {ffr['min']:>14.4f}")
            lines.append(f"    Days FFR < 0.85:   {ffr['days_below_085']:>14d}")
            buckets = portfolio_analysis.get("ffr_buckets", {})
            if buckets:
                lines.append(f"    FFR Buckets:")
                for lab in ["<0.80", "0.80-0.90", "0.90-0.95", "0.95-1.00"]:
                    lines.append(f"      {lab:>12s}: {buckets.get(lab, 0):>4d} days")

        # PA (Price Advantage)
        pa_info = portfolio_analysis.get("pa_analysis")
        if pa_info:
            lines.append(f"\n  --- Price Advantage (PA) ---")
            lines.append(f"    Mean PA:           {pa_info['mean']:>14.6f}")
            lines.append(f"    Median PA:         {pa_info['median']:>14.6f}")
            lines.append(f"    Min / Max:         {pa_info['min']:.6f} / {pa_info['max']:.6f}")
            lines.append(f"    Std:               {pa_info['std']:>14.6f}")
            lines.append(f"    Days PA < 0:       {pa_info['days_negative']:>14d} ({pa_info['pct_negative']:.1f}%)")
            lines.append(f"    (PA>0 = bought cheaper than close; PA<0 = bought more expensive)")

        # Transaction cost analysis
        ca = portfolio_analysis.get("cost_analysis")
        if ca:
            lines.append(f"\n  --- Transaction Cost Analysis ---")
            lines.append(f"    Open Cost Rate:    {ca['open_cost_rate']*10000:>14.2f} bps")
            lines.append(f"    Close Cost Rate:   {ca['close_cost_rate']*10000:>14.2f} bps")
            lines.append(f"    Min Cost/Trade:    {ca['min_cost']:>14.1f}")
            lines.append(f"    Total Commission:  {ca['total_commission']:>14,.2f}")
            lines.append(f"    Avg Cost/Trade:    {ca['avg_cost_per_trade']:>14,.2f}")
            lines.append(f"    Ann. Cost Drag:    {ca['cost_drag_annualized_pct']:>14.2f}%")

        # Risk metrics
        rm = portfolio_analysis.get("risk_metrics")
        if rm:
            lines.append(f"\n  --- Risk Metrics ---")
            lines.append(f"    Sharpe Ratio:      {rm['sharpe_ratio']:>14.3f}")
            lines.append(f"    Sortino Ratio:     {rm['sortino_ratio']:>14.3f}")
            lines.append(f"    Max Drawdown:      {rm['max_drawdown_pct']:>14.2f}%")
            lines.append(f"    MaxDD Duration:    {rm['max_drawdown_duration_days']:>14d} days")
            lines.append(f"    Calmar Ratio:      {rm['calmar_ratio']:>14.3f}")
            lines.append(f"    Ann. Volatility:   {rm['annualized_volatility_pct']:>14.2f}%")

        # Monthly returns
        monthly = portfolio_analysis.get("monthly_returns", {})
        if monthly:
            lines.append(f"\n  --- Monthly Returns ---")
            # Group by year for compact display
            years: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for ym, ret_pct in monthly.items():
                yr = ym[:4]
                mon = ym[5:7]
                years[yr].append((mon, ret_pct))
            lines.append(f"    {'Year':>6s}  " + "  ".join(f"{'M'+str(i):>6s}" for i in range(1, 13)))
            lines.append(f"    {'─'*6}  " + "  ".join(f"{'─'*6}" for _ in range(12)))
            for yr in sorted(years.keys()):
                mon_map = {m: r for m, r in years[yr]}
                row_parts = []
                for i in range(1, 13):
                    m_key = f"{i:02d}"
                    if m_key in mon_map:
                        row_parts.append(f"{mon_map[m_key]:>6.1f}")
                    else:
                        row_parts.append(f"{'':>6s}")
                lines.append(f"    {yr:>6s}  " + "  ".join(row_parts))

        # Trade stats (win rate)
        ts = portfolio_analysis.get("trade_stats")
        if ts:
            qlib_note = " (Qlib close prices)" if ts.get("qlib_price_used") else " (snapshot prices only)"
            lines.append(f"\n  --- Trade Win/Loss Analysis{qlib_note} ---")
            lines.append(f"    Total Trades:      {ts['total_trades']:>14d}")
            one_day = ts.get('one_day_trades', 0)
            one_day_pct = ts.get('one_day_pct', 0)
            if one_day > 0:
                one_day_wr = ts.get('one_day_win_rate', 0)
                one_day_net = ts.get('one_day_net_pnl', 0)
                lines.append(f"    1-Day Trades:      {one_day:>14d}  ({one_day_pct:.1f}%)")
                lines.append(f"    1-Day Win Rate:    {one_day_wr:>14.1f}%")
                lines.append(f"    1-Day Net P&L:     {one_day_net:>14,.0f}")
            multi_day_n = ts.get('multi_day_trades', 0)
            if multi_day_n > 0:
                multi_wr = ts.get('multi_day_win_rate', 0)
                multi_net = ts.get('multi_day_net_pnl', 0)
                lines.append(f"    Multi-Day Trades:  {multi_day_n:>14d}")
                lines.append(f"    Multi-Day WinRate: {multi_wr:>14.1f}%")
                lines.append(f"    Multi-Day Net P&L: {multi_net:>14,.0f}")
            lines.append(f"    --- Overall ---")
            lines.append(f"    Winning / Losing:  {ts['winning_trades']} / {ts['losing_trades']} / {ts.get('flat_trades', 0)} flat")
            lines.append(f"    Win Rate:          {ts['win_rate_pct']:>14.1f}%")
            lines.append(f"    Avg Win / Loss:    {ts['avg_win_pct']:.2f}% / {ts['avg_loss_pct']:.2f}%")
            lines.append(f"    Profit Factor:     {ts['profit_factor']:>14.2f}")
            lines.append(f"    Gross Profit:      {ts.get('gross_profit', 0):>14,.0f}")
            lines.append(f"    Gross Loss:        {ts.get('gross_loss', 0):>14,.0f}")
            lines.append(f"    Net P&L:           {ts.get('net_pnl', 0):>14,.0f}")
            lines.append(f"    Avg Holding Days:  {ts['avg_holding_days']:>14.1f}")
            lines.append(f"    Median Holding:    {ts['median_holding_days']:>14d} days")

        # Streaks
        streaks = portfolio_analysis.get("streaks")
        if streaks:
            lines.append(f"\n  --- Streak Analysis ---")
            lines.append(f"    Max Consec. Wins:  {streaks['max_consecutive_wins']:>14d}")
            lines.append(f"    Max Consec. Losses:{streaks['max_consecutive_losses']:>14d}")

        # Win rate by holding period
        wrp = portfolio_analysis.get("win_rate_by_holding_period", {})
        if wrp:
            lines.append(f"\n  --- Win Rate by Holding Period ---")
            lines.append(f"    {'Period':>10s} {'Trades':>8s} {'WinRate':>8s} {'AvgP&L':>8s}")
            lines.append(f"    {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
            for label in ["1d", "2-3d", "4-7d", "8-14d", "15+d"]:
                if label in wrp:
                    b = wrp[label]
                    lines.append(f"    {label:>10s} {b['count']:>8d} {b['win_rate']:>7.1f}% {b['avg_pnl']:>7.2f}%")

        # Top winners/losers
        tw = portfolio_analysis.get("top_winners", [])
        if tw:
            lines.append(f"\n  --- Top 10 Winning Trades ---")
            lines.append(f"    {'Stock':>12s} {'Entry':>12s} {'Exit':>12s} {'Days':>5s} {'P&L%':>8s} {'P&L Value':>12s} {'Cost':>8s} {'NetP&L':>12s}")
            for t in tw[:10]:
                lines.append(f"    {t['stock']:>12s} {t['entry_date']:>12s} {t['exit_date']:>12s} "
                              f"{t['holding_days']:>5d} {t['pnl_pct']:>7.2f}% {t['pnl_value']:>12,.0f} "
                              f"{t.get('total_cost', 0):>8,.0f} {t.get('net_pnl', t['pnl_value']):>12,.0f}")

        tl = portfolio_analysis.get("top_losers", [])
        if tl:
            lines.append(f"\n  --- Top 10 Losing Trades ---")
            lines.append(f"    {'Stock':>12s} {'Entry':>12s} {'Exit':>12s} {'Days':>5s} {'P&L%':>8s} {'P&L Value':>12s} {'Cost':>8s} {'NetP&L':>12s}")
            for t in tl[:10]:
                lines.append(f"    {t['stock']:>12s} {t['entry_date']:>12s} {t['exit_date']:>12s} "
                              f"{t['holding_days']:>5d} {t['pnl_pct']:>7.2f}% {t['pnl_value']:>12,.0f} "
                              f"{t.get('total_cost', 0):>8,.0f} {t.get('net_pnl', t['pnl_value']):>12,.0f}")

        # Top/Bottom 15 stocks by total P&L (enhanced)
        ts_pnl = portfolio_analysis.get("top_stocks_by_pnl", [])
        if ts_pnl:
            n_top = len(ts_pnl)
            lines.append(f"\n  --- Top {n_top} Stocks by Cumulative P&L ---")
            lines.append(f"    {'Stock':>12s} {'Trades':>6s} {'WR%':>6s} {'GrossP&L':>10s} {'Costs':>8s} {'NetP&L':>10s} {'AvgHold':>7s} {'MaxWin%':>8s} {'MaxLoss%':>8s}")
            for s in ts_pnl:
                lines.append(f"    {s['stock']:>12s} {s['trades']:>6d} {s['win_rate']:>5.1f}% "
                              f"{s['total_pnl']:>10,.0f} {s['total_cost']:>8,.0f} {s['total_pnl_net']:>10,.0f} "
                              f"{s['avg_holding_days']:>7.1f} {s['max_win_pct']:>7.2f}% {s['max_loss_pct']:>7.2f}%")

        bs_pnl = portfolio_analysis.get("bottom_stocks_by_pnl", [])
        if bs_pnl:
            n_bot = len(bs_pnl)
            lines.append(f"\n  --- Bottom {n_bot} Stocks by Cumulative P&L ---")
            lines.append(f"    {'Stock':>12s} {'Trades':>6s} {'WR%':>6s} {'GrossP&L':>10s} {'Costs':>8s} {'NetP&L':>10s} {'AvgHold':>7s} {'MaxWin%':>8s} {'MaxLoss%':>8s}")
            for s in bs_pnl:
                lines.append(f"    {s['stock']:>12s} {s['trades']:>6d} {s['win_rate']:>5.1f}% "
                              f"{s['total_pnl']:>10,.0f} {s['total_cost']:>8,.0f} {s['total_pnl_net']:>10,.0f} "
                              f"{s['avg_holding_days']:>7.1f} {s['max_win_pct']:>7.2f}% {s['max_loss_pct']:>7.2f}%")

        total_unique_stocks = len(portfolio_analysis.get("all_stocks_by_pnl", []))
        if total_unique_stocks > 30:
            lines.append(f"\n    ({total_unique_stocks} total unique stocks traded. Use --export-trades for full list.)")

        # Portfolio issues
        p_issues = portfolio_analysis.get("issues", [])
        if p_issues:
            lines.append(f"\n  ** PORTFOLIO DIAGNOSES **")
            for pi in p_issues:
                lines.append(f"    ! {pi['code']}: {pi['detail']}")

        lines.append("")

    # --- Section 18: Execution Quality Evolution ---
    if execution_quality and any(eq.get("ffr_mean") is not None or eq.get("pa_mean") is not None for eq in execution_quality):
        lines.append(sep)
        lines.append("  EXECUTION QUALITY EVOLUTION (Per-Loop)")
        lines.append(sep)

        # Build header and rows
        has_ffr = any(eq.get("ffr_mean") is not None for eq in execution_quality)
        has_pa = any(eq.get("pa_mean") is not None for eq in execution_quality)
        has_da = any(eq.get("deal_amount_avg") is not None for eq in execution_quality)
        has_cost = any(eq.get("cost_drag_pct") is not None for eq in execution_quality)
        has_strat = any(eq.get("exec_strategy") is not None for eq in execution_quality)

        hdr = f"  {'Loop':>5s}"
        div_len = 8
        if has_ffr:
            hdr += f"  {'FFR':>8s}  {'FFR<.85':>7s}"
            div_len += 19
        if has_pa:
            hdr += f"  {'PA(mean)':>10s}  {'PA<0%':>5s}"
            div_len += 19
        if has_da:
            hdr += f"  {'DealAmt':>12s}"
            div_len += 14
        if has_cost:
            hdr += f"  {'RetW/Cost':>9s}  {'RetNoCost':>9s}  {'CostDrag':>8s}"
            div_len += 32
        if has_strat:
            hdr += f"  {'Strategy':>16s}"
            div_len += 18
        lines.append(hdr)
        lines.append("  " + "-" * div_len)

        for eq in execution_quality:
            row = f"  {eq['loop_id']:>5d}"
            if has_ffr:
                ffr_val = f"{eq['ffr_mean']:.4f}" if eq.get("ffr_mean") is not None else "  N/A"
                ffr_lo = f"{eq.get('days_ffr_below_085', 0):>5d}d" if eq.get("ffr_mean") is not None else "  N/A"
                row += f"  {ffr_val:>8s}  {ffr_lo:>7s}"
            if has_pa:
                pa_val = f"{eq['pa_mean']:.6f}" if eq.get("pa_mean") is not None else "     N/A"
                pa_neg = f"{eq.get('pa_days_negative', 0):>3d}d" if eq.get("pa_mean") is not None else " N/A"
                row += f"  {pa_val:>10s}  {pa_neg:>5s}"
            if has_da:
                da_val = f"{eq['deal_amount_avg']:,.0f}" if eq.get("deal_amount_avg") is not None else "N/A"
                row += f"  {da_val:>12s}"
            if has_cost:
                rw = f"{eq['ann_ret_with_cost']*100:.1f}%" if eq.get("ann_ret_with_cost") is not None else "N/A"
                rn = f"{eq['ann_ret_without_cost']*100:.1f}%" if eq.get("ann_ret_without_cost") is not None else "N/A"
                cd = f"{eq['cost_drag_pct']:.2f}%" if eq.get("cost_drag_pct") is not None else "N/A"
                row += f"  {rw:>9s}  {rn:>9s}  {cd:>8s}"
            if has_strat:
                sv = eq.get("exec_strategy", "N/A")
                row += f"  {sv:>16s}"
            lines.append(row)

        # Trend analysis
        ffr_vals = [eq["ffr_mean"] for eq in execution_quality if eq.get("ffr_mean") is not None]
        pa_vals = [eq["pa_mean"] for eq in execution_quality if eq.get("pa_mean") is not None]
        if len(ffr_vals) >= 2:
            delta = ffr_vals[-1] - ffr_vals[0]
            trend = "improving" if delta > 0 else "degrading"
            lines.append(f"\n  FFR Trend: {ffr_vals[0]:.4f} -> {ffr_vals[-1]:.4f} (delta: {delta:+.4f}) [{'^ ' + trend if delta > 0 else 'v ' + trend}]")
        if len(pa_vals) >= 2:
            delta = pa_vals[-1] - pa_vals[0]
            trend = "improving" if delta > 0 else "degrading"
            lines.append(f"  PA Trend:  {pa_vals[0]:.6f} -> {pa_vals[-1]:.6f} (delta: {delta:+.6f}) [{'^ ' + trend if delta > 0 else 'v ' + trend}]")

        lines.append("")

    # --- Section 16: Analytical Conclusions & Recommendations ---
    lines.append(sep)
    lines.append("  ANALYTICAL CONCLUSIONS & RECOMMENDATIONS")
    lines.append(sep)

    conclusions = generate_conclusions(
        loop_data, evolution_analysis, parallelism_analysis,
        propagation_analysis, feedback_analysis, live_resources,
        convergence_analysis, prompt_issues, code_quality_issues,
        homogeneity_analysis, memory_profile, portfolio_analysis,
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
    memory_profile: dict[str, Any] | None = None,
    portfolio_analysis: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Generate analytical conclusions and recommendations."""
    conclusions: list[dict[str, str]] = []

    # 0. Memory profile issues (critical - OOM risk)
    if memory_profile:
        for mi in memory_profile.get("issues", []):
            rec = ""
            if mi["issue"] == "HIGH_COSTEER_INTENSITY":
                rec = (
                    "High eval call count causes glibc malloc arena inflation. "
                    "Consider adding gc.collect() between CoSTEER iterations or "
                    "setting MALLOC_ARENA_MAX=2 to limit arena count."
                )
            elif mi["issue"] == "CONSECUTIVE_MAX_LOOP":
                rec = (
                    "Consecutive MAX_LOOP hits = continuous heavy memory churn. "
                    "Review factor prompt clarity to reduce CoSTEER retries. "
                    "Consider clearing loop_prev_out for completed loops."
                )
            elif mi["issue"] == "NO_LIGHT_LOOPS":
                rec = (
                    "All loops require multiple CoSTEER iterations. "
                    "Factor code templates or field validation should be improved "
                    "to reduce compile failures."
                )
            elif mi["issue"] == "SINGLE_LOOP_OVERLOAD":
                rec = (
                    "One loop has exceptionally high work intensity. "
                    "6+ factors with 5 iterations is an OOM risk factor."
                )
            elif mi["issue"] == "SESSION_SNAPSHOT_GROWTH":
                rec = (
                    "Session state is growing across loops. "
                    "Knowledge base embedding accumulation is the likely cause."
                )
            conclusions.append({
                "category": "MEMORY",
                "finding": f"[{mi['issue']}] {mi['detail']}",
                "recommendation": rec,
            })

    # 0.5 Code quality bugs (critical - these cause silent training failure)
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

    # Portfolio analysis issues
    if portfolio_analysis:
        for pi in portfolio_analysis.get("issues", []):
            rec = ""
            if pi["code"] == "LOW_UTILIZATION":
                rec = "Check if only_tradable=true is set. Low utilization often caused by limit-up stocks wasting TopK slots."
            elif pi["code"] == "LOW_FFR":
                rec = "Enable only_tradable=true + forbid_all_trade_at_limit=false in strategy kwargs to skip limit-up stocks."
            elif pi["code"] == "LOW_WIN_RATE":
                rec = "Win rate below 45% suggests weak signal. Consider concentrating alpha with smaller TopK or improving factor quality."
            elif pi["code"] == "NEGATIVE_EXPECTANCY":
                rec = "Profit factor < 1.0 means strategy loses money on average. Review signal quality and transaction costs."
            elif pi["code"] == "EXCESSIVE_CHURN":
                rec = "Very short holding periods increase cost drag. Consider hold_thresh or inertia_bonus to reduce noise-driven turnover."
            elif pi["code"] == "HIGH_TURNOVER":
                rec = "High annualized turnover erodes alpha via transaction costs. Consider reducing n_drop or adding turnover inertia."
            elif pi["code"] == "HIGH_COST_DRAG":
                rec = "Transaction costs > 2% annualized drag. Reduce turnover or negotiate lower commission rates."
            elif pi["code"] == "LOW_SHARPE":
                rec = "Sharpe < 0.5 indicates poor risk-adjusted returns. Improve signal quality or reduce drawdowns."
            elif pi["code"] == "PROLONGED_DRAWDOWN":
                rec = "Drawdown > 60 days suggests strategy struggles to recover. Consider diversification or stop-loss mechanisms."
            conclusions.append({
                "category": "PORTFOLIO",
                "finding": f"[{pi['code']}] {pi['detail']}",
                "recommendation": rec,
            })

        ts = portfolio_analysis.get("trade_stats")
        if ts:
            conclusions.append({
                "category": "PORTFOLIO",
                "finding": (
                    f"Trade stats: {ts['total_trades']} trades, {ts['win_rate_pct']:.1f}% win rate, "
                    f"profit factor {ts['profit_factor']:.2f}, avg holding {ts['avg_holding_days']:.1f}d."
                ),
                "recommendation": "",
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
    parser.add_argument("--export-trades", metavar="PATH", help="Export trade records and stock summaries to CSV files ({PATH}_trades.csv, {PATH}_stocks.csv)")
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
        ld["backtest_portfolio"] = extract_backtest_portfolio(loop_path)

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
    memory_profile = analyze_memory_profile(session_obj, loop_data, log_dir)
    portfolio_analysis = analyze_backtest_portfolio(loop_data)
    execution_quality = analyze_per_loop_execution_quality(loop_data)

    # Live resource check
    live_resources = check_live_resources() if args.live_check else None

    # Generate conclusions
    conclusions = generate_conclusions(
        loop_data, evolution_analysis, parallelism_analysis,
        propagation_analysis, feedback_analysis, live_resources,
        convergence_analysis, prompt_issues, code_quality_issues,
        homogeneity_analysis, memory_profile, portfolio_analysis,
    )

    # CSV export
    if args.export_trades and portfolio_analysis:
        trades_path = args.export_trades + "_trades.csv"
        stocks_path = args.export_trades + "_stocks.csv"
        trade_records_list = portfolio_analysis.get("trade_records", [])
        if trade_records_list:
            trades_df = pd.DataFrame(trade_records_list)
            trades_df.to_csv(trades_path, index=False)
            print(f"Exported {len(trade_records_list)} trades to {trades_path}", file=sys.stderr)
        all_stocks = portfolio_analysis.get("all_stocks_by_pnl", [])
        if all_stocks:
            stock_df = pd.DataFrame(all_stocks)
            stock_df.to_csv(stocks_path, index=False)
            print(f"Exported {len(all_stocks)} stock summaries to {stocks_path}", file=sys.stderr)

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
            "memory_profile": memory_profile,
            "portfolio_analysis": portfolio_analysis,
            "execution_quality": execution_quality,
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
            live_resources, homogeneity_analysis, memory_profile,
            portfolio_analysis, execution_quality,
        ))


if __name__ == "__main__":
    main()
