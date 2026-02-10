from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from rdagent.log.storage import FileStorage
from rdagent.log.ui.utils import get_sota_exp_stat


TASK_LOG_DIR = Path(r"F:\Dev\RD-Agent-main\log\2026-01-06_06-00-53-321254")
OUTPUT_JSON = Path(r"F:\Dev\RD-Agent-main\tmp\sota_log_inspect_2026-01-06_06-00-53-321254.json")


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
            return Path
        return super().find_class(module, name)


def _is_factor_exp(exp: Any) -> bool:
    try:
        return "factor" in type(exp).__name__.lower()
    except Exception:
        return False


def _is_model_exp(exp: Any) -> bool:
    try:
        return "model" in type(exp).__name__.lower()
    except Exception:
        return False


def _safe_file_dict(exp_obj: Any) -> dict[str, Any] | None:
    try:
        sw_list = getattr(exp_obj, "sub_workspace_list", None) or []
        if not sw_list:
            return None
        fd = getattr(sw_list[0], "file_dict", None)
        return fd if isinstance(fd, dict) else None
    except Exception:
        return None


def _extract_loop_id(tag: str) -> int | None:
    try:
        if tag.startswith("Loop_"):
            return int(tag.split(".")[0].replace("Loop_", ""))
    except Exception:
        return None
    return None


def _scan_obj(obj: Any, depth: int = 0, max_depth: int = 4) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if obj is None or depth > max_depth:
        return hits
    try:
        if hasattr(obj, "hist"):
            hits.append({"kind": "trace_like", "type": type(obj).__name__})
        if hasattr(obj, "sota_exp_to_submit"):
            hits.append({"kind": "sota_holder", "type": type(obj).__name__})
        if hasattr(obj, "decision"):
            hits.append(
                {"kind": "decision_holder", "type": type(obj).__name__, "decision": getattr(obj, "decision", None)}
            )
        if hasattr(obj, "sub_tasks"):
            sub_tasks = getattr(obj, "sub_tasks", None) or []
            if sub_tasks:
                st0 = sub_tasks[0]
                hits.append(
                    {
                        "kind": "factor_task",
                        "type": type(obj).__name__,
                        "factor_name": getattr(st0, "factor_name", None),
                        "factor_formulation": getattr(st0, "factor_formulation", None),
                    }
                )
    except Exception:
        pass

    try:
        if isinstance(obj, dict):
            for v in obj.values():
                hits.extend(_scan_obj(v, depth + 1, max_depth))
        elif isinstance(obj, (list, tuple)):
            for v in obj[:20]:
                hits.extend(_scan_obj(v, depth + 1, max_depth))
    except Exception:
        pass
    return hits


def _load_session_obj(session_dir: Path) -> tuple[object | None, str | None, int | None, str | None]:
    if not session_dir.exists() or not session_dir.is_dir():
        return None, None, None, "session_dir_missing"

    candidates: list[Path] = []
    pref = session_dir / "1_coding"
    if pref.exists() and pref.is_file():
        candidates.append(pref)
    for fp in session_dir.iterdir():
        if fp.is_file() and fp not in candidates:
            candidates.append(fp)
    if not candidates:
        return None, None, None, f"no_session_files: {session_dir}"

    best_obj: object | None = None
    best_len = -1
    best_file: str | None = None
    last_err: str | None = None
    for fp in candidates:
        try:
            with fp.open("rb") as f:
                obj = _CompatUnpickler(f).load()
            trace = getattr(obj, "trace", None)
            hist = getattr(trace, "hist", None) if trace is not None else None
            n = int(len(hist) if hist else 0)
            if n > best_len:
                best_len = n
                best_obj = obj
                best_file = fp.name
        except Exception as e:
            last_err = str(e)
            continue
    if best_obj is None:
        return None, None, None, f"pickle_load_failed: {last_err}" if last_err else "pickle_load_failed"
    return best_obj, best_file, int(best_len), None


def main() -> None:
    if not TASK_LOG_DIR.exists():
        raise FileNotFoundError(f"log_dir_not_found: {TASK_LOG_DIR}")

    storage = FileStorage(TASK_LOG_DIR)

    trace_msgs = list(storage.iter_msg(tag="trace"))
    trace_obj = trace_msgs[-1].content if trace_msgs else None

    tag_stats: dict[str, int] = {}
    trace_candidates: list[dict[str, Any]] = []
    sota_tag_hits: list[str] = []
    for msg in storage.iter_msg():
        tag_stats[msg.tag] = tag_stats.get(msg.tag, 0) + 1
        tag_lower = msg.tag.lower()
        if "sota" in tag_lower:
            sota_tag_hits.append(msg.tag)
        content = msg.content
        try:
            if hasattr(content, "hist"):
                trace_candidates.append({"tag": msg.tag, "type": type(content).__name__})
            elif hasattr(content, "sota_exp_to_submit"):
                trace_candidates.append({"tag": msg.tag, "type": type(content).__name__})
        except Exception:
            continue

    last_factor_exp = None
    last_factor_index = None
    last_model_exp = None
    last_model_index = None

    session_root = TASK_LOG_DIR / "__session__"
    best_session_obj: object | None = None
    best_session_dir: str | None = None
    best_session_file: str | None = None
    best_session_hist_len: int | None = None
    best_session_err: str | None = None
    if session_root.exists() and session_root.is_dir():
        for snap in sorted(session_root.iterdir(), key=lambda p: p.name):
            if not snap.is_dir():
                continue
            obj, fname, hist_len, err = _load_session_obj(snap)
            if err:
                best_session_err = err
                continue
            if hist_len is not None and (best_session_hist_len is None or hist_len > best_session_hist_len):
                best_session_obj = obj
                best_session_dir = snap.name
                best_session_file = fname
                best_session_hist_len = hist_len

    sota_factors: list[dict[str, Any]] = []

    if trace_obj is None and best_session_obj is not None:
        trace_obj = getattr(best_session_obj, "trace", None)

    if trace_obj is not None:
        hist = getattr(trace_obj, "hist", None)
        if isinstance(hist, list):
            for i in range(len(hist) - 1, -1, -1):
                try:
                    exp, feedback = hist[i]
                except Exception:
                    continue
                decision = getattr(feedback, "decision", None) if feedback is not None else None
                if decision is True and _is_factor_exp(exp):
                    last_factor_exp = exp
                    last_factor_index = i
                    break

            if last_factor_index is not None:
                for j in range(int(last_factor_index) + 1, len(hist)):
                    try:
                        exp, feedback = hist[j]
                    except Exception:
                        continue
                    decision = getattr(feedback, "decision", None) if feedback is not None else None
                    if decision is True and _is_model_exp(exp):
                        last_model_exp = exp
                        last_model_index = j
                        break

    if last_factor_exp is not None:
        based_exps = getattr(last_factor_exp, "based_experiments", None) or []
        for based_exp in based_exps:
            try:
                sub_tasks = getattr(based_exp, "sub_tasks", None) or []
                if not sub_tasks:
                    continue
                fd = _safe_file_dict(based_exp)
                sota_factors.append(
                    {
                        "name": getattr(sub_tasks[0], "factor_name", None),
                        "formulation": getattr(sub_tasks[0], "factor_formulation", None),
                        "code_len": len(fd.get("factor.py", "")) if isinstance(fd, dict) and fd.get("factor.py") else 0,
                    }
                )
            except Exception:
                continue

        try:
            sub_tasks = getattr(last_factor_exp, "sub_tasks", None) or []
            fd = _safe_file_dict(last_factor_exp)
            if sub_tasks:
                sota_factors.append(
                    {
                        "name": getattr(sub_tasks[0], "factor_name", None),
                        "formulation": getattr(sub_tasks[0], "factor_formulation", None),
                        "code_len": len(fd.get("factor.py", "")) if isinstance(fd, dict) and fd.get("factor.py") else 0,
                    }
                )
        except Exception:
            pass

    sota_exp, sota_loop_id, sota_score, sota_stat = get_sota_exp_stat(TASK_LOG_DIR, selector="auto")

    model_workspace = None
    combined_factors_exists = None
    combined_factors_cols = None
    if last_model_exp is not None:
        try:
            workspace_path = getattr(getattr(last_model_exp, "experiment_workspace", None), "workspace_path", None)
            if workspace_path:
                model_workspace = str(workspace_path)
                combined_path = Path(str(workspace_path)) / "combined_factors_df.parquet"
                combined_factors_exists = combined_path.exists()
                if combined_factors_exists:
                    try:
                        import pandas as pd

                        cols = list(pd.read_parquet(combined_path, engine="pyarrow").columns)
                        combined_factors_cols = cols
                    except Exception:
                        combined_factors_cols = None
        except Exception:
            pass

    output = {
        "task_log_dir": str(TASK_LOG_DIR),
        "trace_found": trace_obj is not None,
        "session_snapshot_dir": best_session_dir,
        "session_snapshot_file": best_session_file,
        "session_hist_len": best_session_hist_len,
        "session_load_error": best_session_err,
        "trace_candidates": trace_candidates[:10],
        "tag_stats_total": len(tag_stats),
        "tag_stats_sample": sorted(tag_stats.items(), key=lambda x: (-x[1], x[0]))[:30],
        "sota_tag_hits": list(dict.fromkeys(sota_tag_hits))[:20],
        "last_sota_factor_index": last_factor_index,
        "last_sota_factor_type": type(last_factor_exp).__name__ if last_factor_exp is not None else None,
        "sota_factors_count": len(sota_factors),
        "sota_factors_sample": sota_factors[:10],
        "sota_exp_to_submit_found": sota_exp is not None,
        "sota_loop_id": sota_loop_id,
        "sota_score": sota_score,
        "sota_stat": sota_stat,
        "model_exp_index": last_model_index,
        "model_exp_type": type(last_model_exp).__name__ if last_model_exp is not None else None,
        "model_workspace": model_workspace,
        "combined_factors_exists": combined_factors_exists,
        "combined_factors_cols_len": len(combined_factors_cols) if combined_factors_cols else None,
        "combined_factors_cols_sample": combined_factors_cols[:20] if combined_factors_cols else None,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
