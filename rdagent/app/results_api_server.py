from __future__ import annotations

import io
import json
import logging
import os
import pickle
import re
import subprocess
import time
import zipfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Dict

_logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Query
from rdagent.app.scheduler.server import create_app as create_scheduler_app
from rdagent.app.llm_config import router as llm_config_router
from rdagent.app.api_endpoints import sota_factors_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from rdagent.scenarios.qlib.experiment.quant_experiment import (
    QlibFactorExperiment,
    QlibModelExperiment,
)


def _load_json(path: Path) -> Any | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_app() -> FastAPI:
    app = FastAPI(title="RD-Agent Results API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 复用 Scheduler API, 统一挂载到 /scheduler
    app.mount("/scheduler", create_scheduler_app())

    # LLM Configuration Management API
    app.include_router(llm_config_router)
    
    # Health Check API
    from rdagent.app.api_endpoints import health_router
    app.include_router(health_router)
    
    # SOTA Factors Extractor API
    app.include_router(sota_factors_router)
    
    # Alpha Baseline Factors API
    from rdagent.app.api_endpoints import alpha_baseline_router
    app.include_router(alpha_baseline_router)

    # Results API：本期作为 AIstock 初始化/增量同步的主接口。
    # 权威来源：log/<task_id>/ 与 log/<task_id>/__session__/ （不依赖 registry.sqlite / loop / SQLite）。

    def _repo_root() -> Path:
        # rdagent/app/results_api_server.py -> repo_root
        return Path(__file__).resolve().parents[2]

    def _log_root() -> Path:
        return _repo_root() / "log"

    def _workspace_root() -> Path:
        return _repo_root() / "git_ignore_folder" / "RD-Agent_workspace"

    def _aistock_catalog_root() -> Path:
        # 兼容：RD-Agent 侧将 AIstock catalog 输出到 repo_root/RDagentDB/aistock
        return _repo_root() / "RDagentDB" / "aistock"

    def _to_native_path(p_str: str | Path) -> Path:
        """将WSL路径与Windows路径互转为当前系统可读路径。"""
        if isinstance(p_str, Path):
            p_str = str(p_str)
        if not p_str:
            return Path()
        is_windows = os.name == "nt"
        p_norm = str(p_str).replace("/", os.sep).replace("\\", os.sep)
        if is_windows:
            lower_p = p_norm.lower()
            prefix = f"{os.sep}mnt{os.sep}"
            if lower_p.startswith(prefix):
                parts = Path(p_norm).parts
                if len(parts) >= 3:
                    drive = parts[2].upper()
                    return Path(f"{drive}:\\") / Path(*parts[3:])
            if len(p_norm) > 1 and p_norm[1] == ":":
                return Path(p_norm)
        elif len(p_norm) > 1 and p_norm[1] == ":":
            drive = p_norm[0].lower()
            rel = p_norm[3:].replace("\\", "/")
            return Path(f"/mnt/{drive}") / rel
        return Path(p_norm)

    @app.get("/_debug/paths")
    def debug_paths() -> dict[str, Any]:
        return {
            "ok": True,
            "repo_root": str(_repo_root()),
            "log_root": str(_log_root()),
            "workspace_root": str(_workspace_root()),
            "aistock_catalog_root": str(_aistock_catalog_root()),
        }

    def _try_get_registry():
        try:
            from rdagent.utils.registry.sqlite_registry import get_registry  # type: ignore

            return get_registry()
        except Exception:
            return None

    _hex32_re = re.compile(r"\b[0-9a-f]{32}\b", flags=re.IGNORECASE)
    _ws_re = re.compile(r"RD-Agent_workspace/([0-9a-f]{32})", flags=re.IGNORECASE)

    # pickle 序列化时的旧 (module, class_name) → 新 module 映射
    # 只有确实已迁移到 quant_experiment 的类才做重映射；
    # QlibFactorScenario / QlibModelScenario 仍留在各自原模块中，不能重映射。
    _CLASS_REMAP: dict[tuple[str, str], str] = {
        ("rdagent.scenarios.qlib.experiment.factor_experiment", "QlibFactorExperiment"): "rdagent.scenarios.qlib.experiment.quant_experiment",
        ("rdagent.scenarios.qlib.experiment.model_experiment", "QlibModelExperiment"): "rdagent.scenarios.qlib.experiment.quant_experiment",
    }

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):  # type: ignore[override]
            # 兼容：部分环境用 pathlib.PosixPath / WindowsPath 序列化
            if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
                return Path
            # 兼容：旧版 pickle 中实验类的模块路径已迁移（按类名精确匹配）
            module = _CLASS_REMAP.get((module, name), module)
            return super().find_class(module, name)

    def _pickle_load_compat(p: Path) -> Any:
        with p.open("rb") as f:
            return _CompatUnpickler(f).load()

    def _pickle_dump_compat(obj: Any, p: Path) -> None:
        with p.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _iter_log_tasks(*, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        root = _log_root()
        if not root.exists() or not root.is_dir():
            return []
        items: list[tuple[float, str]] = []
        for entry in root.iterdir():
            if not entry.is_dir() or entry.name.startswith("__"):
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            items.append((mtime, entry.name))
        items.sort(key=lambda x: x[0], reverse=True)
        sliced = items[int(offset) : int(offset) + int(limit)]
        out: list[dict[str, Any]] = []
        for mtime, name in sliced:
            out.append({
                "task_id": name, 
                "updated_at_utc": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
            })
        return out

    def _check_sota_exists(task_id: str) -> bool:
        """检查 task 是否有 SOTA 因子产生。"""
        try:
            log_dir = (_log_root() / str(task_id)).resolve()
            if not log_dir.exists() or not log_dir.is_dir():
                return False
            
            # 方案 1: 检查是否存在 sota_exp_to_submit 消息
            from rdagent.log.storage import FileStorage
            storage = FileStorage(log_dir)
            try:
                # 只要能迭代出任何一个 sota_exp_to_submit 标签的消息，就认为有 SOTA
                for _ in storage.iter_msg(tag="sota_exp_to_submit"):
                    return True
            except Exception:
                pass
            
            # 方案 2: 检查 session 中是否有 decision=True 的因子实验
            session_obj, _, _, _, _ = _load_best_session_for_task(task_id)
            if session_obj:
                trace = getattr(session_obj, "trace", None)
                hist = getattr(trace, "hist", None) if trace is not None else None
                if hist:
                    for exp, feedback in hist:
                        decision = getattr(feedback, "decision", None)
                        if decision is True and isinstance(exp, QlibFactorExperiment):
                            return True
            return False
        except Exception:
            return False

    async def _get_task_info_parallel(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=min(len(items), 20)) as executor:
            tasks = []
            for item in items:
                tasks.append(loop.run_in_executor(executor, _check_sota_exists, item["task_id"]))
            
            sota_results = await asyncio.gather(*tasks)
            
            for item, has_sota in zip(items, sota_results):
                item["has_sota"] = has_sota
        return items

    @app.get("/tasks")
    async def tasks_list(
        limit: int = Query(200, ge=1, le=2000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        items = _iter_log_tasks(limit=int(limit), offset=int(offset))
        if items:
            items = await _get_task_info_parallel(items)
        return {"ok": True, "count": len(items), "items": items, "limit": int(limit), "offset": int(offset)}

    def _find_latest_session_snapshot_dir(session_root: Path) -> tuple[Path | None, str | None]:
        """确定性选择最新且包含 3_feedback 的 snapshot 目录。

        规则：
        - snapshot 目录按 loop 编号递增，最大编号即最新 loop。
        - 但最新 loop 可能正在进行中（尚未生成 3_feedback），此时需回退到次新目录。
        - 从最大编号开始向下查找，取第一个包含 3_feedback 文件的目录。
        """
        try:
            if not session_root.exists() or not session_root.is_dir():
                return None, f"session_root_not_found: {session_root}"
            loop_dirs = [p for p in session_root.iterdir() if p.is_dir() and p.name.isdigit()]
            if not loop_dirs:
                return None, f"no_session_loop_dirs: {session_root}"
            loop_dirs.sort(key=lambda p: int(p.name), reverse=True)
            # 从最大编号开始向下查找，取第一个包含 3_feedback 的目录
            for d in loop_dirs:
                if (d / "3_feedback").exists():
                    return d, None
            return None, f"no_3_feedback_in_any_snapshot: {session_root}"
        except Exception as e:
            return None, str(e)

    def _try_read_asset_from_exp_obj(exp_obj: object, basename: str) -> tuple[bytes | None, str | None]:
        """从 experiment/sub_workspace 中读取指定 basename 资产。

        说明：
        - 仅用于 Results API 侧兜底导出（AIstock 侧禁止遍历 workspace）。
        - 优先使用 file_dict（若存在），其次尝试 workspace_path 文件。
        - 仅按固定文件名读取，不递归遍历。
        """
        try:
            errors: list[str] = []

            def _resolve_ws_path(ws_path: str | Path) -> Path:
                return _to_native_path(ws_path)

            def _try_weight_candidates(ws_root: Path) -> tuple[bytes | None, str | None]:
                if basename not in {"model.pkl", "params.pkl"}:
                    return None, None
                # 1) workspace 根目录下查找固定文件名
                direct = ws_root / basename
                if direct.exists() and direct.is_file():
                    try:
                        return direct.read_bytes(), None
                    except Exception as e:
                        return None, f"model_weight_read_failed:{e}"
                # 尝试另一个固定文件名 (model.pkl <-> params.pkl)
                alt = "params.pkl" if basename == "model.pkl" else "model.pkl"
                alt_path = ws_root / alt
                if alt_path.exists() and alt_path.is_file():
                    try:
                        return alt_path.read_bytes(), None
                    except Exception as e:
                        return None, f"model_weight_read_failed:{e}"
                # 2) mlruns 目录递归查找（与 task_assets_extractor 逻辑一致）
                mlruns_dir = ws_root / "mlruns"
                if mlruns_dir.exists() and mlruns_dir.is_dir():
                    candidates: list[Path] = []
                    for pkl_file in mlruns_dir.rglob("params.pkl"):
                        if pkl_file.is_file():
                            candidates.append(pkl_file)
                    if not candidates:
                        for pkl_file in mlruns_dir.rglob("model.pkl"):
                            if pkl_file.is_file():
                                candidates.append(pkl_file)
                    if candidates:
                        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        try:
                            return candidates[0].read_bytes(), None
                        except Exception as e:
                            return None, f"mlruns_weight_read_failed:{e}"
                return None, "model_weight_not_found_in_workspace_or_mlruns"

            def _coerce_bytes(v: Any) -> bytes | None:
                if isinstance(v, (bytes, bytearray)):
                    return bytes(v)
                if isinstance(v, str):
                    return v.encode("utf-8")
                return None

            def _try_path(p: Path, tag: str) -> tuple[bytes | None, str | None]:
                if not p.exists() or not p.is_file():
                    return None, f"{tag}_asset_not_found:{basename}"
                try:
                    return p.read_bytes(), None
                except Exception as e:
                    return None, f"{tag}_read_failed:{e}"

            # 0) exp_obj 自身可能就是 workspace（例如 QlibFBWorkspace）
            try:
                fd_direct = getattr(exp_obj, "file_dict", None)
                if isinstance(fd_direct, dict) and basename in fd_direct:
                    bs = _coerce_bytes(fd_direct.get(basename))
                    if bs is not None:
                        return bs, None
                    errors.append("workspace_file_dict_nonbytes")
                ws_direct = getattr(exp_obj, "workspace_path", None)
                if isinstance(ws_direct, (str, Path)) and str(ws_direct).strip():
                    ws_root = _resolve_ws_path(str(ws_direct).strip())
                    p = (ws_root / basename).resolve()
                    bs, err = _try_path(p, "workspace")
                    if bs is not None and not err:
                        return bs, None
                    if err:
                        errors.append(err)
                    bs, err = _try_weight_candidates(ws_root)
                    if bs is not None and not err:
                        return bs, None
                    if err:
                        errors.append(err)
            except Exception:
                pass

            # 1) sub_workspace_list: file_dict / workspace_path
            sw_list = getattr(exp_obj, "sub_workspace_list", None) or []
            for sw in sw_list:
                if sw is None:
                    continue
                fd = getattr(sw, "file_dict", None)
                if isinstance(fd, dict) and basename in fd:
                    bs = _coerce_bytes(fd.get(basename))
                    if bs is not None:
                        return bs, None
                    errors.append("sub_workspace_file_dict_nonbytes")

                ws_path = getattr(sw, "workspace_path", None)
                if isinstance(ws_path, (str, Path)) and str(ws_path).strip():
                    ws_root = _resolve_ws_path(str(ws_path).strip())
                    p = (ws_root / basename).resolve()
                    bs, err = _try_path(p, "sub_workspace")
                    if bs is not None and not err:
                        return bs, None
                    if err:
                        errors.append(err)
                    bs, err = _try_weight_candidates(ws_root)
                    if bs is not None and not err:
                        return bs, None
                    if err:
                        errors.append(err)
                else:
                    errors.append("sub_workspace_path_missing")

            # 2) experiment_workspace: file_dict / workspace_path
            exp_ws = getattr(exp_obj, "experiment_workspace", None)
            if exp_ws is not None:
                fd = getattr(exp_ws, "file_dict", None)
                if isinstance(fd, dict) and basename in fd:
                    bs = _coerce_bytes(fd.get(basename))
                    if bs is not None:
                        return bs, None
                    errors.append("experiment_workspace_file_dict_nonbytes")

                ws_path = getattr(exp_ws, "workspace_path", None)
                if isinstance(ws_path, (str, Path)) and str(ws_path).strip():
                    ws_root = _resolve_ws_path(str(ws_path).strip())
                    p = (ws_root / basename).resolve()
                    bs, err = _try_path(p, "experiment_workspace")
                    if bs is not None and not err:
                        return bs, None
                    if err:
                        errors.append(err)
                    bs, err = _try_weight_candidates(ws_root)
                    if bs is not None and not err:
                        return bs, None
                    if err:
                        errors.append(err)
                else:
                    errors.append("experiment_workspace_path_missing")

            return None, ";".join(dict.fromkeys(errors)) if errors else "workspace_path_missing"
        except Exception as e:
            return None, str(e)

    def _pick_main_sub_workspace_index(exp_obj: object) -> int | None:
        """选择主子工作区：优先含权重文件的 sub_workspace，其次含 factor.py 的 sub_workspace。"""
        try:
            sw_list = getattr(exp_obj, "sub_workspace_list", None) or []
            for i, sw in enumerate(sw_list):
                if sw is None:
                    continue
                fd = getattr(sw, "file_dict", None)
                if isinstance(fd, dict):
                    key, _ = _pick_weight_key_from_file_dict(fd)
                    if key:
                        return i
            for i, sw in enumerate(sw_list):
                if sw is None:
                    continue
                fd = getattr(sw, "file_dict", None)
                if isinstance(fd, dict):
                    key, _ = _pick_factor_entry_key_from_file_dict(fd)
                    if key:
                        return i
            return 0 if sw_list else None
        except Exception:
            return None

    def _collect_based_sub_workspaces(factor_exp: object) -> list[tuple[int, object]]:
        """当 based_experiments 为空时，用 factor_exp.sub_workspace_list 作为 based 因子来源。"""
        try:
            sw_list = getattr(factor_exp, "sub_workspace_list", None) or []
            main_idx = _pick_main_sub_workspace_index(factor_exp)
            out: list[tuple[int, object]] = []
            for idx, sw in enumerate(sw_list):
                if sw is None:
                    continue
                if main_idx is not None and idx == main_idx:
                    continue
                out.append((idx, sw))
            return out
        except Exception:
            return []

    def _pick_model_meta_key_from_file_dict(file_dict: dict[str, Any]) -> tuple[str | None, str | None]:
        """确定性选择训练特征合同文件 key。

        规则：仅认 model_meta.json（固定文件名），不猜测、不遍历。
        """
        try:
            keys = [str(k) for k in file_dict.keys()]
            keys_l = {k.lower(): k for k in keys}
            if "model_meta.json" in keys_l:
                return keys_l["model_meta.json"], None
            return None, "model_meta_not_found_in_file_dict"
        except Exception as e:
            return None, str(e)

    def _collect_based_factor_experiments(factor_exp: object) -> list[object]:
        try:
            based = getattr(factor_exp, "based_experiments", None) or []
            out: list[object] = []
            for exp in based:
                if exp is None:
                    continue
                try:
                    if not isinstance(exp, QlibFactorExperiment):
                        continue
                except Exception:
                    continue
                # 过滤空占位符（无 sub_tasks 的基线因子实验）
                sub_tasks = getattr(exp, "sub_tasks", None)
                if not sub_tasks:
                    continue
                out.append(exp)
            return out
        except Exception:
            return []

    # session 缓存：避免每次 API 调用都重新加载几十 MB 的 pickle 文件
    # key = task_id, value = (mtime_key, result_tuple)
    _session_cache: dict[str, tuple[str, tuple]] = {}
    # 基于时间的快速缓存跳过：{task_id: last_load_timestamp}
    # 在 TTL 内直接返回缓存结果，完全跳过文件系统操作（WSL 跨 NTFS 的 iterdir/stat 约 120 秒）
    _session_cache_ts: dict[str, float] = {}
    _SESSION_CACHE_TTL = 300.0  # 5 分钟内不重新检查文件系统

    def _load_best_session_for_task(
        task_id: str,
    ) -> tuple[object | None, str | None, str | None, int | None, str | None]:
        """返回 (session_obj, source_session_dir_id, chosen_session_file, hist_len, error)
        
        内置基于文件 mtime 的缓存：同一 task_id + 同一 snapshot_dir + 同一最新文件 mtime
        时直接返回缓存结果，避免重复加载大型 pickle 文件。
        
        额外优化：在 TTL 内直接返回缓存结果，跳过文件系统操作（WSL 跨 NTFS 极慢）。
        """
        try:
            tid = str(task_id).strip()
            if not tid:
                return None, None, None, None, "task_id 为空"

            # 快速路径：TTL 内直接返回缓存结果，完全跳过文件系统操作
            cached = _session_cache.get(tid)
            cached_ts = _session_cache_ts.get(tid, 0.0)
            if cached is not None and (time.time() - cached_ts) < _SESSION_CACHE_TTL:
                _logger.info(f"[{tid}] session fast-cache hit (age={time.time() - cached_ts:.0f}s)")
                return cached[1]

            log_dir = (_log_root() / tid).resolve()
            session_root = (log_dir / "__session__").resolve()
            snapshot_dir, err = _find_latest_session_snapshot_dir(session_root)
            if err or snapshot_dir is None:
                return None, None, None, None, err

            source_session_dir_id = snapshot_dir.name
            # 只加载 3_feedback（hist 最完整：feedback 步骤追加当前 loop 结果）
            # 不再遍历其他文件作为 fallback —— 确定性逻辑
            target_file = snapshot_dir / "3_feedback"
            if not target_file.exists() or not target_file.is_file():
                return None, source_session_dir_id, None, None, f"3_feedback_not_found: {snapshot_dir}"

            # 构建缓存 key：snapshot_dir + 文件 mtime
            try:
                file_mtime = target_file.stat().st_mtime
            except OSError:
                file_mtime = 0.0
            cache_key = f"{snapshot_dir}:{file_mtime}"

            if cached is not None and cached[0] == cache_key:
                _logger.info(f"[{tid}] session cache hit (snapshot={source_session_dir_id})")
                _session_cache_ts[tid] = time.time()  # 刷新 TTL
                return cached[1]

            _logger.info(f"[{tid}] loading session pickle (snapshot={source_session_dir_id}, file=3_feedback) ...")
            t0 = time.time()

            best_obj: object | None = None
            best_len = -1
            best_file: str | None = None
            last_err: str | None = None
            try:
                obj = _pickle_load_compat(target_file)
                trace = getattr(obj, "trace", None)
                hist = getattr(trace, "hist", None) if trace is not None else None
                best_len = int(len(hist) if hist else 0)
                best_obj = obj
                best_file = "3_feedback"
            except Exception as e:
                last_err = str(e)

            elapsed = time.time() - t0
            _logger.info(f"[{tid}] session loaded in {elapsed:.1f}s (file={best_file}, hist_len={best_len})")

            if best_obj is None:
                result = (None, source_session_dir_id, None, None, f"pickle_load_failed: {last_err}" if last_err else "pickle_load_failed")
            else:
                result = (best_obj, source_session_dir_id, best_file, int(best_len), None)

            # 缓存结果（只保留最近 8 个 task 的缓存，避免内存无限增长）
            if len(_session_cache) >= 8:
                oldest_key = next(iter(_session_cache))
                del _session_cache[oldest_key]
                _session_cache_ts.pop(oldest_key, None)
            _session_cache[tid] = (cache_key, result)
            _session_cache_ts[tid] = time.time()  # 记录加载时间，启用快速缓存

            return result
        except Exception as e:
            return None, None, None, None, str(e)

    def _find_last_sota_factor_and_following_model(session_obj: object) -> tuple[dict[str, Any], str | None]:
        """按《模型权重文件定位方案_v2》：

        阶段1（Task-only strict）权重定位以“最后一个 decision=True 的因子实验”为唯一 anchor：
        - 因子实验本身在回测时使用/产出模型权重（如 model.pkl/params.pkl），应随 session 持久化到 sub_workspace_list.file_dict。
        - 禁止使用“模型 loop/模型演进实验”的权重作为一期选股依据。

        返回：
        - last_sota_factor_index
        - model_exp_index: 兼容字段；允许为空（一期不依赖“模型实验进入回测”）
        - _last_sota_factor_exp
        - _model_exp: 兼容字段；允许为空
        """
        try:
            trace = getattr(session_obj, "trace", None)
            hist = getattr(trace, "hist", None) if trace is not None else None
            if not hist:
                return {}, "trace_hist_empty"

            def _is_factor_exp(x: object) -> bool:
                return isinstance(x, QlibFactorExperiment)

            def _is_model_exp(x: object) -> bool:
                return isinstance(x, QlibModelExperiment)

            last_factor_exp = None
            last_factor_index: int | None = None
            for i in range(len(hist) - 1, -1, -1):
                exp, feedback = hist[i]
                decision = getattr(feedback, "decision", None) if feedback is not None else None
                if decision is True and _is_factor_exp(exp):
                    last_factor_exp = exp
                    last_factor_index = i
                    break
            if last_factor_exp is None or last_factor_index is None:
                return {}, "no_accepted_factor_experiment"

            # 注意：一期 strict 下，模型权重以“最后进入 SOTA 的因子实验所在 loop 的回测权重”为准，
            # 不要求存在后续模型实验节点；因此这里仅做兼容性探测，不存在也不报错。
            model_exp = None
            model_exp_index: int | None = None
            for j in range(int(last_factor_index) + 1, len(hist)):
                exp, feedback = hist[j]
                decision = getattr(feedback, "decision", None) if feedback is not None else None
                if decision is True and _is_model_exp(exp):
                    model_exp = exp
                    model_exp_index = j
                    break

            return {
                "last_sota_factor_index": last_factor_index,
                "model_exp_index": model_exp_index,
                "_last_sota_factor_exp": last_factor_exp,
                "_model_exp": model_exp,
            }, None
        except Exception as e:
            return {}, str(e)

    def _find_sota_factor_from_log(task_id: str) -> tuple[dict[str, Any], str | None]:
        try:
            log_dir = (_log_root() / str(task_id)).resolve()
            if not log_dir.exists() or not log_dir.is_dir():
                return {}, f"log_dir_not_found: {log_dir}"
            from rdagent.log.ui.utils import get_sota_exp_stat  # type: ignore

            sota_exp, sota_loop_id, sota_score, sota_stat = get_sota_exp_stat(log_dir, selector="auto")
            if sota_exp is None:
                return {}, "no_sota_exp_in_log"
            return {
                "last_sota_factor_index": None,
                "model_exp_index": None,
                "sota_loop_id": sota_loop_id,
                "sota_score": sota_score,
                "sota_stat": sota_stat,
                "anchor_source": "log_ui",
                "_last_sota_factor_exp": sota_exp,
                "_model_exp": None,
            }, None
        except Exception as e:
            return {}, str(e)

    def _resolve_sota_anchor(task_id: str, session_obj: object | None) -> tuple[dict[str, Any], str | None]:
        if session_obj is not None:
            anchor_info, anchor_err = _find_last_sota_factor_and_following_model(session_obj)
            if not anchor_err:
                anchor_info.setdefault("anchor_source", "session_trace")
                return anchor_info, None
        return _find_sota_factor_from_log(task_id)

    def _extract_file_dict_from_sub_workspaces(
        exp_obj: object,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """从 sub_workspace_list 合并得到一个 file_dict。

        注意：不少任务的权重/参数可能不在第一个 sub-workspace 中，若只取第一个非空 dict 会漏 key。
        这里按“union”合并所有 dict：遇到重复 key 时，优先保留第一个出现的值（避免后续覆盖）。
        """
        try:
            sw_list = getattr(exp_obj, "sub_workspace_list", None) or []
            merged: dict[str, Any] = {}
            any_dict = False
            # exp_obj 可能直接是 workspace：优先合并自身 file_dict
            direct_fd = getattr(exp_obj, "file_dict", None)
            if isinstance(direct_fd, dict) and direct_fd:
                any_dict = True
                for k, v in direct_fd.items():
                    ks = str(k)
                    if ks in merged:
                        continue
                    merged[ks] = v
            for sw in sw_list:
                if sw is None:
                    continue
                fd = getattr(sw, "file_dict", None)
                if not isinstance(fd, dict) or not fd:
                    continue
                any_dict = True
                for k, v in fd.items():
                    ks = str(k)
                    if ks in merged:
                        continue
                    merged[ks] = v
            if merged:
                return merged, None
            if any_dict:
                return None, "sub_workspace_list_file_dict_all_empty"
            return None, "sub_workspace_list_file_dict_empty"
        except Exception as e:
            return None, str(e)

    def _pick_factor_entry_key_from_file_dict(file_dict: dict[str, Any]) -> tuple[str | None, str | None]:
        """确定性选择因子入口代码 key。

        规则（不猜测，不遍历 workspace）：
        - 若存在 factor_entry.py 则选它；
        - 否则若存在 factor.py 则选它；
        - 否则返回错误（不推测）。
        """
        try:
            keys = [str(k) for k in file_dict.keys()]
            keys_l = {k.lower(): k for k in keys}
            for k in ("factor_entry.py", "factor.py"):
                if k in keys_l:
                    return keys_l[k], None
            # 不再推测：唯一 .py 文件 != 因子入口，直接报错
            return None, "no_factor_entry_in_file_dict"
        except Exception as e:
            return None, str(e)

    def _pick_weight_key_from_file_dict(file_dict: dict[str, Any]) -> tuple[str | None, str | None]:
        try:
            keys = list(file_dict.keys())
            keys_l = {str(k).lower(): k for k in keys}
            # 确定性逻辑: 只认固定文件名，不推测不遍历
            for k in ("model.pkl", "params.pkl"):
                if k in keys_l:
                    return keys_l[k], None
            return None, "no_weight_in_file_dict(expected model.pkl or params.pkl)"
        except Exception as e:
            return None, str(e)

    def _iter_recent_log_tasks(*, limit: int) -> list[dict[str, Any]]:
        root = _log_root()
        if not root.exists():
            return []

        items: list[tuple[float, str]] = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            items.append((mtime, entry.name))

        items.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for mtime, name in items[: int(limit)]:
            out.append({"task_id": name, "updated_at_utc": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()})
        return out

    def _extract_workspace_ids_from_log_dir(*, task_id: str) -> list[str]:
        log_dir = _log_root() / str(task_id)
        if not log_dir.exists() or not log_dir.is_dir():
            return []

        found: set[str] = set()
        max_files = 200
        max_read_bytes = 256 * 1024
        scanned = 0
        for p in log_dir.rglob("*"):
            if not p.is_file():
                continue
            scanned += 1
            if scanned > max_files:
                break
            try:
                bs = p.read_bytes()[:max_read_bytes]
            except OSError:
                continue
            try:
                text = bs.decode("utf-8", errors="ignore")
            except Exception:
                continue
            for m in _ws_re.finditer(text):
                found.add(m.group(1).lower())

        # 过滤掉不存在的 workspace
        ws_root = _workspace_root()
        out: list[str] = []
        for ws in sorted(found):
            if (ws_root / ws).exists():
                out.append(ws)
        return out

    def _ensure_task_log_dir(task_id: str) -> Path:
        tid = str(task_id).strip()
        if not tid:
            raise HTTPException(status_code=400, detail="task_id 为空")
        d = (_log_root() / tid).resolve()
        if not d.exists() or not d.is_dir():
            raise HTTPException(status_code=404, detail=f"task log dir not found: {tid}")
        return d

    def _list_workspace_assets(*, workspace_dir: Path) -> list[dict[str, Any]]:
        allow_ext = {
            ".py",
            ".yaml",
            ".yml",
            ".json",
            ".csv",
            ".pkl",
            ".pt",
            ".txt",
            ".md",
            ".png",
        }
        max_bytes = 200 * 1024 * 1024

        out: list[dict[str, Any]] = []
        for p in workspace_dir.iterdir():
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix not in allow_ext:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > max_bytes:
                continue
            out.append({"name": p.name, "size": int(size)})

        out.sort(key=lambda x: (x["name"] != "factor_entry.py", x["name"]))
        return out

    def _pick_primary_assets(*, ws_id: str, files: list[dict[str, Any]]) -> dict[str, str]:
        names = [f["name"] for f in files]

        def rel(n: str) -> str:
            return f"workspaces/{ws_id}/{n}"

        factor = "factor_entry.py" if "factor_entry.py" in names else ""

        model = ""
        for cand in ("model.pkl", "model.pt", "model.pth"):
            if cand in names:
                model = cand
                break

        config = ""
        for cand in ("conf_baseline.yaml", "conf_baseline.yml", "conf_baseline_factors_model.yaml", "conf_sota_factors_model.yaml"):
            if cand in names:
                config = cand
                break
        if not config:
            for n in names:
                if n.lower().endswith((".yaml", ".yml")):
                    config = n
                    break

        return {
            "factor_entry_relpath": rel(factor) if factor else "",
            "model_weight_relpath": rel(model) if model else "",
            "config_relpath": rel(config) if config else "",
        }

    def _build_task_manifest(task_id: str) -> dict[str, Any]:
        # 以 log/session 为权威：生成“锚点清单 + 可下载 file_dict keys +（可选）workspace 文件列表”。
        _ensure_task_log_dir(task_id)

        session_obj, source_session_dir_id, chosen_session_file, hist_len, session_err = _load_best_session_for_task(task_id)

        anchor_info, anchor_err = _resolve_sota_anchor(task_id, session_obj)
        if anchor_err and session_obj is None:
            anchor_err = session_err

        model_fd_keys: list[str] = []
        resolved_model_weight_key: str | None = None
        resolved_model_weight_err: str | None = None

        model_exp = anchor_info.get("_model_exp")
        if model_exp is not None:
            fd, fd_err = _extract_file_dict_from_sub_workspaces(model_exp)
            if fd is not None:
                model_fd_keys = [str(k) for k in fd.keys()]
                resolved_model_weight_key, resolved_model_weight_err = _pick_weight_key_from_file_dict(fd)
            else:
                resolved_model_weight_err = fd_err

        # best-effort workspace 视图：仅用于兼容旧下载接口与调试，不作为选股权威
        ws_ids = _extract_workspace_ids_from_log_dir(task_id=task_id)
        ws_id = ws_ids[0] if ws_ids else ""
        ws_dir = (_workspace_root() / ws_id).resolve() if ws_id else None
        assets: list[dict[str, Any]] = []
        if ws_dir is not None and ws_dir.exists() and ws_dir.is_dir():
            try:
                files = _list_workspace_assets(workspace_dir=ws_dir)
                assets = [{"relpath": f"workspaces/{ws_id}/{f['name']}", "size": f["size"]} for f in files]
            except Exception:
                assets = []

        return {
            "schema_version": 2,
            "task_id": str(task_id),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "log": {
                "log_dir": str((_log_root() / str(task_id)).resolve()),
            },
            "session_anchor": {
                "source_session_dir_id": source_session_dir_id,
                "chosen_session_file": chosen_session_file,
                "hist_len": hist_len,
                "error": anchor_err,
            },
            "sota_factor_anchor": {
                "last_sota_factor_index": anchor_info.get("last_sota_factor_index"),
                "model_exp_index": anchor_info.get("model_exp_index"),
                "model_file_dict_keys": model_fd_keys,
                "resolved_model_weight_key": resolved_model_weight_key,
                "resolved_model_weight_error": resolved_model_weight_err,
            },
            "assets": assets,
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tasks/latest")
    def tasks_latest(limit: int = 20) -> dict[str, Any]:
        items = _iter_log_tasks(limit=int(limit), offset=0)
        return {"ok": True, "count": len(items), "tasks": items}


    @app.get("/tasks/{task_id}/summary")
    def task_summary(task_id: str) -> dict[str, Any]:
        d = _ensure_task_log_dir(task_id)
        ws_ids = _extract_workspace_ids_from_log_dir(task_id=str(task_id))
        session_obj, source_session_dir_id, chosen_session_file, hist_len, session_err = _load_best_session_for_task(task_id)
        anchor_info, anchor_err = _resolve_sota_anchor(task_id, session_obj)
        if anchor_err and session_obj is None:
            anchor_err = session_err
        return {
            "ok": True,
            "task_id": str(task_id),
            "log_dir": str(d),
            "workspace_ids": ws_ids,
            "session_anchor": {
                "source_session_dir_id": source_session_dir_id,
                "chosen_session_file": chosen_session_file,
                "hist_len": hist_len,
                "error": anchor_err,
            },
            "sota_factor_anchor": {
                "last_sota_factor_index": anchor_info.get("last_sota_factor_index"),
                "model_exp_index": anchor_info.get("model_exp_index"),
            },
        }

    @app.get("/tasks/{task_id}/session_anchor")
    def task_session_anchor(task_id: str) -> dict[str, Any]:
        _ensure_task_log_dir(task_id)
        session_obj, source_session_dir_id, chosen_session_file, hist_len, session_err = _load_best_session_for_task(task_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail=session_err or "session_not_found")
        return {
            "ok": True,
            "task_id": str(task_id),
            "source_session_dir_id": source_session_dir_id,
            "chosen_session_file": chosen_session_file,
            "hist_len": hist_len,
        }

    @app.get("/tasks/{task_id}/debug_trace")
    def task_debug_trace(task_id: str) -> dict[str, Any]:
        """诊断端点：列出 session trace 中每个 hist 条目的实验类型和 decision 状态。"""
        _ensure_task_log_dir(task_id)
        session_obj, source_session_dir_id, chosen_session_file, hist_len, session_err = _load_best_session_for_task(task_id)
        if session_obj is None:
            return {"ok": False, "error": session_err or "session_not_found"}

        trace = getattr(session_obj, "trace", None)
        hist = getattr(trace, "hist", None) if trace is not None else None
        if not hist:
            return {"ok": False, "error": "trace_hist_empty", "source_session_dir_id": source_session_dir_id}

        entries = []
        for i, item in enumerate(hist):
            exp, feedback = item if isinstance(item, (list, tuple)) and len(item) >= 2 else (item, None)
            exp_type = type(exp).__name__
            exp_module = type(exp).__module__
            is_factor = isinstance(exp, QlibFactorExperiment)
            is_model = isinstance(exp, QlibModelExperiment)
            decision = getattr(feedback, "decision", None) if feedback is not None else None
            feedback_type = type(feedback).__name__ if feedback is not None else None

            entry = {
                "index": i,
                "exp_type": exp_type,
                "exp_module": exp_module,
                "is_QlibFactorExperiment": is_factor,
                "is_QlibModelExperiment": is_model,
                "feedback_type": feedback_type,
                "decision": decision,
            }

            # 检查 sub_tasks
            sub_tasks = getattr(exp, "sub_tasks", None)
            if sub_tasks:
                names = []
                for st in sub_tasks[:5]:
                    n = getattr(st, "factor_name", None) or getattr(st, "name", None)
                    names.append(str(n) if n else "?")
                entry["sub_task_names_sample"] = names
                entry["sub_tasks_count"] = len(sub_tasks)

            entries.append(entry)

        return {
            "ok": True,
            "task_id": str(task_id),
            "source_session_dir_id": source_session_dir_id,
            "hist_len": len(hist),
            "entries": entries,
        }

    @app.get("/tasks/{task_id}/sota_factor_anchor")
    def task_sota_factor_anchor(
        task_id: str,
        auto_backfill: bool = Query(True, description="当 SOTA 因子实验 file_dict 缺少权重时，是否自动回填"),
    ) -> dict[str, Any]:
        _ensure_task_log_dir(task_id)
        session_obj, source_session_dir_id, chosen_session_file, hist_len, session_err = _load_best_session_for_task(task_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail=session_err or "session_not_found")
        anchor_info, anchor_err = _resolve_sota_anchor(task_id, session_obj)
        if anchor_err:
            return {
                "ok": False,
                "task_id": str(task_id),
                "error": anchor_err,
                "source_session_dir_id": source_session_dir_id,
                "chosen_session_file": chosen_session_file,
                "hist_len": hist_len,
            }

        factor_exp = anchor_info.get("_last_sota_factor_exp")
        factor_fd: dict[str, Any] | None = None
        factor_fd_err: str | None = None
        if factor_exp is not None:
            factor_fd, factor_fd_err = _extract_file_dict_from_sub_workspaces(factor_exp)
        factor_fd_keys = [str(k) for k in factor_fd.keys()] if isinstance(factor_fd, dict) else []
        factor_entry_key, factor_entry_err = (None, None)
        if isinstance(factor_fd, dict):
            factor_entry_key, factor_entry_err = _pick_factor_entry_key_from_file_dict(factor_fd)

        based_factor_entries: list[dict[str, Any]] = []
        try:
            if factor_exp is not None:
                for i, bexp in enumerate(_collect_based_factor_experiments(factor_exp)):
                    # 对于 based factor，直接从 sub_workspace_list[0].file_dict 获取，避免合并时重复 key 被跳过
                    sw_list = getattr(bexp, "sub_workspace_list", None) or []
                    fd = None
                    derr = None
                    if sw_list and sw_list[0] is not None:
                        fd = getattr(sw_list[0], "file_dict", None)
                        if not isinstance(fd, dict):
                            fd = None
                            derr = "sub_workspace_0_file_dict_missing"
                    else:
                        derr = "sub_workspace_list_empty"
                    keys = [str(k) for k in fd.keys()] if isinstance(fd, dict) else []
                    b_entry_key = None
                    b_entry_err = None
                    ws_path = getattr(getattr(bexp, "experiment_workspace", None), "workspace_path", None)
                    ws_path = str(ws_path) if isinstance(ws_path, str) else None
                    sub_ws_summaries: list[dict[str, Any]] = []
                    for sw in getattr(bexp, "sub_workspace_list", None) or []:
                        if sw is None:
                            continue
                        entry: dict[str, Any] = {}
                        sw_path = getattr(sw, "workspace_path", None)
                        if isinstance(sw_path, (str, Path)) and str(sw_path).strip():
                            entry["workspace_path"] = str(sw_path)
                        sw_fd = getattr(sw, "file_dict", None)
                        if isinstance(sw_fd, dict):
                            entry["file_dict_keys"] = [str(k) for k in sw_fd.keys()]
                        if entry:
                            sub_ws_summaries.append(entry)
                    if isinstance(fd, dict):
                        b_entry_key, b_entry_err = _pick_factor_entry_key_from_file_dict(fd)

                    # 关键: based factor 与 sota factor 往往同名 (factor.py)。为避免 /asset_bytes key 冲突，
                    # 对 based factor 的 key 做命名空间: based_factor_{i}/<basename>
                    if b_entry_key:
                        b_entry_key = f"based_factor_{i}/{Path(str(b_entry_key)).name}"
                    based_factor_entries.append(
                        {
                            "based_index": i,
                            "type": type(bexp).__name__,
                            "workspace_path": ws_path,
                            "sub_workspaces": sub_ws_summaries,
                            "factor_file_dict_keys": keys,
                            "factor_file_dict_error": derr,
                            "resolved_factor_entry_key": b_entry_key,
                            "resolved_factor_entry_error": b_entry_err,
                        }
                    )
        except Exception as e:
            based_factor_entries = [{"error": str(e)}]

        # 严格按 v2 方案：仅从“最后一个被接受因子实验”的 file_dict 取回测权重。
        resolved_key, resolved_err = (None, None)
        resolved_model_meta_key, resolved_model_meta_err = (None, None)
        model_fd: dict[str, Any] | None = None
        model_fd_err: str | None = None
        if factor_exp is not None and isinstance(factor_fd, dict):
            model_fd = factor_fd
            model_fd_err = factor_fd_err
            resolved_key, resolved_err = _pick_weight_key_from_file_dict(factor_fd)
            resolved_model_meta_key, resolved_model_meta_err = _pick_model_meta_key_from_file_dict(factor_fd)
        else:
            resolved_err = "factor_exp_missing"
            resolved_model_meta_err = "factor_exp_missing"

        backfill_diag: dict[str, Any] | None = None
        if auto_backfill:
            backfill_diag = {"ok": False, "error": "backfill_disabled_by_v2_scheme"}
            if factor_exp is not None and not resolved_key:
                for candidate in ("model.pkl", "params.pkl"):
                    bs, err = _try_read_asset_from_exp_obj(factor_exp, candidate)
                    if bs is not None and not err:
                        resolved_key = candidate
                        resolved_err = None
                        backfill_diag = {"ok": True, "source": "workspace_fallback", "key": candidate}
                        break
            if factor_exp is not None and not resolved_model_meta_key:
                meta_bs, meta_err = _try_read_asset_from_exp_obj(factor_exp, "model_meta.json")
                if meta_bs is not None and not meta_err:
                    resolved_model_meta_key = "model_meta.json"
                    resolved_model_meta_err = None
                    if not backfill_diag or not backfill_diag.get("ok"):
                        backfill_diag = {"ok": True, "source": "workspace_fallback", "key": "model_meta.json"}

        model_fd_keys: list[str] = []
        if isinstance(model_fd, dict):
            for k in model_fd.keys():
                kl = str(k).lower()
                if kl.endswith((".pkl", ".pth", ".pt", ".ckpt", ".bin")) and kl not in {"ret.pkl", "pred.pkl"}:
                    model_fd_keys.append(str(k))
        return {
            "ok": True,
            "task_id": str(task_id),
            "source_session_dir_id": source_session_dir_id,
            "chosen_session_file": chosen_session_file,
            "hist_len": hist_len,
            "last_sota_factor_index": anchor_info.get("last_sota_factor_index"),
            "model_exp_index": anchor_info.get("model_exp_index"),
            "factor_file_dict_keys": factor_fd_keys,
            "resolved_factor_entry_key": factor_entry_key,
            "resolved_factor_entry_error": factor_entry_err,
            "factor_file_dict_error": factor_fd_err,
            "based_factor_entries": based_factor_entries,
            "model_file_dict_keys": model_fd_keys,
            "resolved_model_weight_key": resolved_key,
            "resolved_model_weight_error": resolved_err,
            "resolved_model_weight_source": "factor_exp" if resolved_key else None,
            "resolved_model_meta_key": resolved_model_meta_key,
            "resolved_model_meta_error": resolved_model_meta_err,
            "resolved_model_meta_source": "factor_exp" if resolved_model_meta_key else None,
            "model_file_dict_error": model_fd_err,
            "backfill": backfill_diag,
        }

    # v2_alignment_preview 结果级缓存: {task_id: (session_dir_id, result_dict)}
    _v2_preview_cache: dict[str, tuple[str | None, dict[str, Any]]] = {}

    @app.get("/tasks/{task_id}/v2_alignment_preview")
    def task_v2_alignment_preview(task_id: str) -> dict[str, Any]:
        """V2 对齐预览：返回 SOTA 因子数、Alpha 基线因子数、模型特征数及对齐验证结果。

        用于 AIstock 同步前判断数据是否完整、是否值得同步。
        不下载任何大文件，仅从 session trace + workspace 元数据推导。
        """
        _ensure_task_log_dir(task_id)
        session_obj, source_session_dir_id, chosen_session_file, hist_len, session_err = _load_best_session_for_task(task_id)
        if session_obj is None:
            return {"ok": False, "task_id": str(task_id), "error": session_err or "session_not_found"}

        # 结果级缓存：session未变化时直接返回缓存结果，避免重复IO
        cached = _v2_preview_cache.get(task_id)
        if cached is not None and cached[0] == source_session_dir_id:
            _logger.info(f"[{task_id}] v2_alignment_preview result cache hit")
            return cached[1]

        anchor_info, anchor_err = _resolve_sota_anchor(task_id, session_obj)
        if anchor_err:
            return {"ok": False, "task_id": str(task_id), "error": anchor_err}

        factor_exp = anchor_info.get("_last_sota_factor_exp")
        if factor_exp is None:
            return {"ok": False, "task_id": str(task_id), "error": "sota_factor_exp_missing"}

        last_sota_factor_index = anchor_info.get("last_sota_factor_index")

        # ---------- 1) SOTA 因子列表（从 workspace parquet 列推导） ----------
        sota_factors: list[str] = []
        sota_source: str | None = None
        ws_path_raw = getattr(getattr(factor_exp, "experiment_workspace", None), "workspace_path", None)
        ws_native: Path | None = None
        if ws_path_raw is not None:
            ws_native = _to_native_path(ws_path_raw)
        if ws_native is not None and ws_native.exists():
            pq = ws_native / "combined_factors_df.parquet"
            if pq.exists():
                try:
                    import pyarrow.parquet as _pq  # type: ignore
                    schema = _pq.read_schema(str(pq))
                    raw_names = schema.names
                    # 处理 MultiIndex 列（tuple 编码为 "('feature', 'name')" 字符串）
                    parsed: list[str] = []
                    for n in raw_names:
                        if n.startswith("(") and "," in n:
                            try:
                                parts = n.strip("()' ").split("', '")
                                parsed.append(parts[1] if len(parts) > 1 else parts[0])
                            except Exception:
                                parsed.append(n)
                        else:
                            if n.lower() not in ("datetime", "instrument"):
                                parsed.append(n)
                    sota_factors = parsed
                    sota_source = "parquet_schema"
                except Exception as e:
                    sota_source = f"parquet_read_error: {e}"

        if not sota_factors:
            # 回退：从 sub_tasks 中收集因子名
            sub_tasks = getattr(factor_exp, "sub_tasks", None) or []
            for st in sub_tasks:
                name = getattr(st, "factor_name", None) or getattr(st, "name", None)
                if name:
                    sota_factors.append(str(name))
            sota_source = "sub_tasks_fallback" if sota_factors else "no_sota_factors_found"

        # ---------- 2) Alpha 基线因子列表（从 workspace yaml 配置文件） ----------
        # Alpha 基线因子定义在 conf_combined_factors_dynamic.yaml 或
        # conf_combined_factors_sota_model.yaml 的 alpha158_config.feature 中，
        # 而不是 model_meta.json（model_meta.json 存在于模型实验 workspace，不在因子实验 workspace）。
        alpha_factors: list[str] = []
        alpha_source: str | None = None

        # 优先从 workspace yaml 配置文件读取 alpha158_config.feature 因子名
        _yaml_candidates = [
            "conf_combined_factors_dynamic.yaml",
            "conf_combined_factors_sota_model.yaml",
            "conf_combined_factors.yaml",
        ]
        if ws_native is not None and ws_native.exists():
            for _yf in _yaml_candidates:
                _yp = ws_native / _yf
                if _yp.exists():
                    try:
                        import yaml as _yaml
                        _yt = _yp.read_text(encoding="utf-8")
                        _yobj = _yaml.safe_load(_yt)
                        # 路径1: data_handler_config.data_loader.kwargs.alpha158_config.feature[2] (因子名列表)
                        _dhc = _yobj.get("data_handler_config") or {}
                        _dl = _dhc.get("data_loader") or {}
                        _dl_kw = _dl.get("kwargs") or {}
                        _a158 = _dl_kw.get("alpha158_config") or {}
                        _feat = _a158.get("feature") or []
                        # feature 是三元素列表: [表达式列表, 名称列表]，名称在 index=2
                        if isinstance(_feat, list) and len(_feat) >= 3:
                            names = _feat[2]
                            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                                alpha_factors = names
                                alpha_source = f"workspace/{_yf}:alpha158_config.feature[2]"
                                break
                        # 路径2: 如果 feature 只有2个元素 [表达式, 名称]
                        if isinstance(_feat, list) and len(_feat) == 2:
                            names = _feat[1]
                            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                                alpha_factors = names
                                alpha_source = f"workspace/{_yf}:alpha158_config.feature[1]"
                                break
                    except Exception as e:
                        alpha_source = f"yaml_parse_error({_yf}): {e}"

        # 回退：从 model_meta.json 读取（兼容旧版本）
        if not alpha_factors and ws_native is not None and (ws_native / "model_meta.json").exists():
            try:
                import json as _json
                meta_text = (ws_native / "model_meta.json").read_text(encoding="utf-8")
                meta_obj = _json.loads(meta_text)
                handler_kw = meta_obj.get("dataset_conf", {}).get("kwargs", {}).get("handler", {}).get("kwargs", {})
                for proc in handler_kw.get("infer_processors", []):
                    if proc.get("class") == "FilterCol":
                        alpha_factors = proc.get("kwargs", {}).get("col_list", [])
                        alpha_source = "workspace/model_meta.json"
                        break
            except Exception as e:
                if not alpha_source:
                    alpha_source = f"model_meta_parse_error: {e}"

        if not alpha_factors:
            # 回退：从 file_dict 读取 model_meta.json
            fd, _ = _extract_file_dict_from_sub_workspaces(factor_exp)
            if isinstance(fd, dict) and "model_meta.json" in fd:
                try:
                    import json as _json
                    raw = fd["model_meta.json"]
                    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    meta_obj = _json.loads(text)
                    handler_kw = meta_obj.get("dataset_conf", {}).get("kwargs", {}).get("handler", {}).get("kwargs", {})
                    for proc in handler_kw.get("infer_processors", []):
                        if proc.get("class") == "FilterCol":
                            alpha_factors = proc.get("kwargs", {}).get("col_list", [])
                            alpha_source = "file_dict/model_meta.json"
                            break
                except Exception as e:
                    if not alpha_source:
                        alpha_source = f"file_dict_meta_parse_error: {e}"

        # ---------- 3) 模型特征数（从 workspace mlruns params.pkl） ----------
        model_feature_count: int | None = None
        model_source: str | None = None

        if ws_native is not None and ws_native.exists():
            mlruns = ws_native / "mlruns"
            if mlruns.exists() and mlruns.is_dir():
                params_candidates = sorted(mlruns.rglob("params.pkl"), key=lambda p: p.stat().st_mtime)
                if params_candidates:
                    try:
                        import pickle as _pkl
                        with params_candidates[-1].open("rb") as _f:
                            model_obj = _pkl.load(_f)
                        inner = getattr(model_obj, "model", model_obj)
                        if hasattr(inner, "num_feature"):
                            model_feature_count = int(inner.num_feature())
                            model_source = f"mlruns/{params_candidates[-1].relative_to(ws_native)}"
                    except Exception as e:
                        model_source = f"model_load_error: {e}"
            # 也检查根目录 params.pkl
            if model_feature_count is None:
                root_params = ws_native / "params.pkl"
                if root_params.exists():
                    try:
                        import pickle as _pkl
                        with root_params.open("rb") as _f:
                            model_obj = _pkl.load(_f)
                        inner = getattr(model_obj, "model", model_obj)
                        if hasattr(inner, "num_feature"):
                            model_feature_count = int(inner.num_feature())
                            model_source = "workspace/params.pkl"
                    except Exception as e:
                        model_source = f"root_params_load_error: {e}"

        # ---------- 4) 对齐验证 ----------
        expected_total = len(alpha_factors) + len(sota_factors)
        is_aligned = (model_feature_count is not None and expected_total == model_feature_count)

        # 构建完整 factor_order
        factor_order = list(alpha_factors) + list(sota_factors)

        result = {
            "ok": True,
            "task_id": str(task_id),
            "hist_len": hist_len,
            "last_sota_factor_index": last_sota_factor_index,
            "sota_factors": sota_factors,
            "sota_factors_count": len(sota_factors),
            "sota_source": sota_source,
            "alpha_factors": alpha_factors,
            "alpha_factors_count": len(alpha_factors),
            "alpha_source": alpha_source,
            "model_feature_count": model_feature_count,
            "model_source": model_source,
            "expected_total_features": expected_total,
            "is_aligned": is_aligned,
            "factor_order": factor_order,
            "has_model_weight": model_feature_count is not None,
            "has_model_meta": len(alpha_factors) > 0,
            "workspace_path": str(ws_native) if ws_native else None,
        }
        # 保存到结果级缓存
        _v2_preview_cache[task_id] = (source_session_dir_id, result)
        return result

    @app.get("/tasks/{task_id}/asset_bytes")
    def task_asset_bytes(task_id: str, key: str = Query(..., description="sub_workspace_list.file_dict 的 key")) -> StreamingResponse:
        _ensure_task_log_dir(task_id)
        k = str(key or "").strip()
        if not k:
            raise HTTPException(status_code=400, detail="key 为空")

        # 支持命名空间 key：based_factor_{i}/<basename>
        based_index: int | None = None
        inner_key = k
        try:
            if k.startswith("based_factor_") and "/" in k:
                prefix, rest = k.split("/", 1)
                if prefix.startswith("based_factor_"):
                    idx_s = prefix.replace("based_factor_", "").strip()
                    if idx_s.isdigit():
                        based_index = int(idx_s)
                        inner_key = rest
        except Exception:
            based_index = None
            inner_key = k
        session_obj, _, _, _, session_err = _load_best_session_for_task(task_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail=session_err or "session_not_found")
        anchor_info, anchor_err = _resolve_sota_anchor(task_id, session_obj)
        if anchor_err:
            raise HTTPException(status_code=404, detail=anchor_err)

        # 先尝试因子实验 file_dict / 基线因子 file_dict，再尝试模型实验 file_dict（都来自 session，不遍历 workspace）
        cand_fds: list[tuple[str, dict[str, Any] | None, str | None, object | None]] = []

        factor_exp = anchor_info.get("_last_sota_factor_exp")
        if factor_exp is not None:
            fd, fd_err = _extract_file_dict_from_sub_workspaces(factor_exp)
            cand_fds.append(("factor", fd, fd_err, factor_exp))

            try:
                for i, bexp in enumerate(_collect_based_factor_experiments(factor_exp)):
                    if based_index is not None and i != based_index:
                        continue
                    # 对于 based factor，直接从 sub_workspace_list[0].file_dict 获取，避免合并时重复 key 被跳过
                    sw_list = getattr(bexp, "sub_workspace_list", None) or []
                    if sw_list and sw_list[0] is not None:
                        bfd = getattr(sw_list[0], "file_dict", None)
                        bfd_err = None
                        if not isinstance(bfd, dict):
                            bfd = None
                            bfd_err = "sub_workspace_0_file_dict_missing"
                        cand_fds.append((f"based_factor_{i}", bfd, bfd_err, bexp))
                    else:
                        cand_fds.append((f"based_factor_{i}", None, "sub_workspace_list_empty", bexp))
            except Exception:
                pass

        # 严格方案：不从模型实验读取 file_dict

        chosen_v: Any = None
        chosen_from: str | None = None
        last_fd_err: str | None = None
        for tag, fd, fd_err, _exp_obj in cand_fds:
            if fd is None:
                last_fd_err = fd_err
                continue
            if inner_key in fd:
                chosen_v = fd.get(inner_key)
                chosen_from = tag
                break

        if chosen_from is None:
            if inner_key in {"model.pkl", "params.pkl", "model_meta.json"} and factor_exp is not None:
                bs, err = _try_read_asset_from_exp_obj(factor_exp, inner_key)
                if bs is not None and not err:
                    chosen_v = bs
                    chosen_from = "workspace_fallback"
                else:
                    raise HTTPException(status_code=404, detail=err or last_fd_err or f"key_not_found_in_file_dict: {k}")
            else:
                raise HTTPException(status_code=404, detail=last_fd_err or f"key_not_found_in_file_dict: {k}")

        v = chosen_v
        if isinstance(v, str):
            bs = v.encode("utf-8")
        elif isinstance(v, (bytes, bytearray)):
            bs = bytes(v)
        else:
            raise HTTPException(status_code=422, detail=f"unsupported_value_type: {type(v)}")

        def iter_bytes():
            yield bs

        filename = Path(inner_key.replace("\\", "/")).name or "asset.bin"
        return StreamingResponse(
            iter_bytes(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}", "X-RDAgent-Source": str(chosen_from)},
        )

    @app.post("/ops/backfill_factor_weights")
    def ops_backfill_factor_weights(
        task_id: str = Query(..., description="task_id (log dir name)"),
        dry_run: bool = Query(False, description="只诊断不落盘"),
    ) -> dict[str, Any]:
        """对历史 task 回填模型权重到 session file_dict。

        目的：让旧任务满足 AIstock Task-only strict：权重必须从 SOTA 因子实验的 sub_workspace_list.file_dict 获取。
        约束：
        - 不使用模型 loop 的任何信息；
        - 不遍历 workspace 根目录猜测；
        - 仅使用 session 内 SOTA 因子实验对象 + 其 experiment_workspace.workspace_path（若存在）；
        - 若 workspace 内不存在权重文件，则尝试从该 workspace 的 mlflow/qlib recorder 中 load_object 导出。
        """

        _ensure_task_log_dir(task_id)
        session_obj, source_session_dir_id, chosen_session_file, hist_len, session_err = _load_best_session_for_task(task_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail=session_err or "session_not_found")
        if not chosen_session_file:
            raise HTTPException(status_code=404, detail="chosen_session_file_missing")

        anchor_info, anchor_err = _resolve_sota_anchor(task_id, session_obj)
        if anchor_err:
            raise HTTPException(status_code=404, detail=anchor_err)

        factor_exp = anchor_info.get("_last_sota_factor_exp")
        if factor_exp is None:
            raise HTTPException(status_code=404, detail="sota_factor_exp_missing")

        factor_fd, factor_fd_err = _extract_file_dict_from_sub_workspaces(factor_exp)
        factor_fd_keys = [str(k) for k in factor_fd.keys()] if isinstance(factor_fd, dict) else []
        pre_key, pre_err = (None, None)
        if isinstance(factor_fd, dict):
            pre_key, pre_err = _pick_weight_key_from_file_dict(factor_fd)

        ws_path_raw = getattr(getattr(factor_exp, "experiment_workspace", None), "workspace_path", None)
        if isinstance(ws_path_raw, str) and ws_path_raw.strip():
            ws_path = Path(ws_path_raw.strip())
        else:
            ws_path = ws_path_raw if isinstance(ws_path_raw, Path) else None

        diag: dict[str, Any] = {
            "ok": False,
            "task_id": str(task_id),
            "source_session_dir_id": source_session_dir_id,
            "chosen_session_file": chosen_session_file,
            "hist_len": hist_len,
            "last_sota_factor_index": anchor_info.get("last_sota_factor_index"),
            "workspace_path": str(ws_path) if ws_path else None,
            "factor_file_dict_keys": factor_fd_keys,
            "pre_resolved_model_weight_key": pre_key,
            "pre_resolved_model_weight_error": pre_err,
            "dry_run": bool(dry_run),
            "actions": [],
        }

        # 找到一个可写入的 sub_workspace.file_dict
        sw_list = getattr(factor_exp, "sub_workspace_list", None) or []
        target_sw = None
        for sw in sw_list:
            if sw is None:
                continue
            fd = getattr(sw, "file_dict", None)
            if isinstance(fd, dict):
                target_sw = sw
                break
        if target_sw is None:
            raise HTTPException(status_code=422, detail=factor_fd_err or "no_sub_workspace_file_dict")

        def _inject_bytes(k: str, bs: bytes) -> None:
            if not isinstance(getattr(target_sw, "file_dict", None), dict):
                return
            target_sw.file_dict[k] = bs

        injected: list[str] = []
        exported: list[str] = []

        # 1) 优先从 workspace 现有文件注入（固定文件名，不做目录遍历/猜测）
        if ws_path is not None and ws_path.exists() and ws_path.is_dir():
            # 1.1 训练特征合同：model_meta.json
            pmm = (ws_path / "model_meta.json").resolve()
            if pmm.exists() and pmm.is_file():
                try:
                    bs = pmm.read_bytes()
                    if bs:
                        _inject_bytes("model_meta.json", bs)
                        injected.append("model_meta.json")
                except Exception as e:
                    diag["actions"].append({"stage": "inject_from_workspace", "key": "model_meta.json", "error": str(e)})

            for k in ["model.pkl", "params.pkl"]:
                p = (ws_path / k).resolve()
                if p.exists() and p.is_file():
                    try:
                        bs = p.read_bytes()
                        if bs:
                            _inject_bytes(k, bs)
                            injected.append(k)
                    except Exception as e:
                        diag["actions"].append({"stage": "inject_from_workspace", "key": k, "error": str(e)})

        # 2) 若 workspace 未命中，尝试从该 workspace 的 recorder 导出并注入
        if not injected and ws_path is not None and ws_path.exists() and ws_path.is_dir():
            try:
                import os as _os
                import qlib as _qlib
                from qlib.workflow import R as _R

                _provider_uri = (
                    _os.environ.get("QLIB_PROVIDER_URI", "").strip()
                    or _os.environ.get("AISTOCK_QLIB_PROVIDER_URI", "").strip()
                    or "/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209"
                )
                _provider_uri = _provider_uri.strip() or "~/.qlib/qlib_data/cn_data"
                try:
                    _qlib.init(provider_uri=_provider_uri)
                except Exception:
                    pass

                _tracking_uri = _os.environ.get("MLFLOW_TRACKING_URI")
                if not _tracking_uri:
                    _local_mlruns = ws_path / "mlruns"
                    if _local_mlruns.exists():
                        _os.environ["MLFLOW_TRACKING_URI"] = str(_local_mlruns)

                exps = _R.list_experiments()
                latest_rec = None
                for en in exps:
                    try:
                        recs = _R.list_recorders(experiment_name=en)
                    except Exception:
                        continue
                    for rid in recs:
                        if rid is None:
                            continue
                        try:
                            rec = _R.get_recorder(recorder_id=rid, experiment_name=en)
                            end_time = rec.info.get("end_time") if isinstance(rec.info, dict) else None
                            if end_time is None:
                                continue
                            if latest_rec is None or end_time > latest_rec.info.get("end_time"):
                                latest_rec = rec
                        except Exception:
                            continue

                if latest_rec is not None:
                    discovered: list[str] = []
                    try:
                        info = getattr(latest_rec, "info", None)
                        run_id = None
                        if isinstance(info, dict):
                            run_id = info.get("run_id") or info.get("id") or info.get("run")
                        run_id = run_id or getattr(latest_rec, "id", None) or getattr(latest_rec, "run_id", None)

                        try:
                            import mlflow  # type: ignore
                            from mlflow.tracking import MlflowClient  # type: ignore

                            _tracking_uri2 = _os.environ.get("MLFLOW_TRACKING_URI")
                            if _tracking_uri2:
                                mlflow.set_tracking_uri(_tracking_uri2)

                            if run_id:
                                client = MlflowClient()

                                def _walk_artifacts(prefix: str = "") -> None:
                                    try:
                                        items = client.list_artifacts(str(run_id), path=prefix)
                                    except Exception:
                                        return
                                    for it in items or []:
                                        p = getattr(it, "path", None)
                                        if not p:
                                            continue
                                        if getattr(it, "is_dir", False):
                                            _walk_artifacts(p)
                                        else:
                                            discovered.append(str(p))

                                _walk_artifacts("")
                        except Exception:
                            pass
                    except Exception:
                        pass

                    diag["actions"].append({"stage": "export_from_recorder", "discovered_artifacts": discovered})

                    if not discovered:
                        try:
                            mlruns_dir = (ws_path / "mlruns").resolve()
                            if mlruns_dir.exists() and mlruns_dir.is_dir():
                                found_files: list[str] = []
                                try:
                                    for p in mlruns_dir.rglob("*.pkl"):
                                        try:
                                            if p.is_file():
                                                found_files.append(str(p))
                                        except Exception:
                                            continue
                                except Exception:
                                    found_files = []

                                diag["actions"].append({"stage": "export_from_recorder", "mlruns_pkl_files": found_files[:200]})

                                def _score_path(s: str) -> tuple[int, int, str]:
                                    name = s.replace("\\", "/").split("/")[-1].lower()
                                    score = 0
                                    if "model" in name:
                                        score += 10
                                    if "param" in name:
                                        score += 8
                                    if name in {"model.pkl", "params.pkl"}:
                                        score += 100
                                    if "pred" in name:
                                        score -= 5
                                    return (-score, len(s), s)

                                for fp in sorted(found_files, key=_score_path)[:20]:
                                    if injected:
                                        break
                                    try:
                                        bs = Path(fp).read_bytes()
                                    except Exception as e:
                                        diag["actions"].append({"stage": "export_from_recorder", "key": fp, "error": str(e)})
                                        continue
                                    if not bs:
                                        continue
                                    name = fp.replace("\\", "/").split("/")[-1].lower()
                                    dst_key = "params.pkl" if name.startswith("param") or "param" in name else "model.pkl"
                                    try:
                                        if not dry_run:
                                            (ws_path / dst_key).write_bytes(bs)
                                        exported.append(str(fp))
                                        _inject_bytes(dst_key, bs)
                                        injected.append(dst_key)
                                    except Exception as e:
                                        diag["actions"].append({"stage": "export_from_recorder", "key": fp, "error": str(e)})
                        except Exception:
                            pass

                    candidate_src_keys: list[str] = []
                    for k in ["model.pkl", "params.pkl", "model", "params", "pred.pkl"]:
                        if k not in candidate_src_keys:
                            candidate_src_keys.append(k)
                    for p in discovered:
                        pn = str(p).split("/")[-1]
                        if not pn:
                            continue
                        if pn.lower().endswith(".pkl") and pn not in candidate_src_keys:
                            candidate_src_keys.append(pn)

                    for src_key in candidate_src_keys:
                        if injected:
                            break
                        try:
                            obj = latest_rec.load_object(src_key)
                        except Exception as e:
                            diag["actions"].append({"stage": "export_from_recorder", "key": src_key, "error": str(e)})
                            continue
                        try:
                            if isinstance(obj, (bytes, bytearray)):
                                bs = bytes(obj)
                            else:
                                bs = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                            if not bs:
                                continue

                            dst_key = "params.pkl" if str(src_key).lower().startswith("param") else "model.pkl"
                            if not dry_run:
                                (ws_path / dst_key).write_bytes(bs)
                            exported.append(str(src_key))
                            _inject_bytes(dst_key, bs)
                            injected.append(dst_key)
                        except Exception as e:
                            diag["actions"].append({"stage": "export_from_recorder", "key": src_key, "error": str(e)})
            except Exception as e:
                diag["actions"].append({"stage": "export_from_recorder", "error": str(e)})

        post_fd, _ = _extract_file_dict_from_sub_workspaces(factor_exp)
        post_key, post_err = (None, None)
        if isinstance(post_fd, dict):
            post_key, post_err = _pick_weight_key_from_file_dict(post_fd)

        diag["exported"] = exported
        diag["injected"] = injected
        diag["post_resolved_model_weight_key"] = post_key
        diag["post_resolved_model_weight_error"] = post_err

        if dry_run:
            diag["ok"] = bool(post_key)
            return diag

        # 写回 session 快照（先备份）
        snap_dir = _find_latest_session_snapshot_dir((_log_root() / str(task_id) / "__session__"))[0]
        if snap_dir is None:
            raise HTTPException(status_code=404, detail="session_snapshot_dir_not_found")
        snap_path = (snap_dir / str(chosen_session_file)).resolve()
        if not snap_path.exists() or not snap_path.is_file():
            raise HTTPException(status_code=404, detail=f"session_snapshot_file_not_found: {snap_path}")

        try:
            bak = snap_path.with_suffix(snap_path.suffix + ".bak")
            if not bak.exists():
                bak.write_bytes(snap_path.read_bytes())
        except Exception as e:
            diag["actions"].append({"stage": "backup", "error": str(e)})

        try:
            _pickle_dump_compat(session_obj, snap_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"session_rewrite_failed: {e}")

        diag["ok"] = bool(post_key)
        return diag

    def _extract_loop_details_from_log(task_id: str) -> tuple[list[dict[str, Any]], str | None]:
        """
        从log目录提取所有LOOP的详细信息
        基于analyze_all_loops_detail.py的简化版逻辑，确保数据准确性
        """
        try:
            import pickle
            import platform
            import importlib
            from pathlib import WindowsPath, PosixPath, Path
            
            # 跨平台路径兼容：根据当前系统自动转换
            class PathCompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # 处理路径类的跨平台兼容
                    if module == "pathlib":
                        # 在WSL/Linux环境下，将WindowsPath转换为PosixPath
                        if platform.system() != "Windows":
                            if name == "WindowsPath":
                                return PosixPath
                        # 在Windows环境下，将PosixPath转换为WindowsPath
                        else:
                            if name == "PosixPath":
                                return WindowsPath
                    
                    # 处理__mp_main__模块（multiprocessing导致的模块名变化）
                    if module == "__mp_main__":
                        # 尝试从rdagent包中导入
                        try:
                            # ModelRDLoop通常在rdagent.scenarios.qlib.experiment.model_experiment中
                            if name == "ModelRDLoop":
                                from rdagent.scenarios.qlib.experiment.model_experiment import ModelRDLoop
                                return ModelRDLoop
                            elif name == "FactorRDLoop":
                                from rdagent.scenarios.qlib.experiment.factor_experiment import FactorRDLoop
                                return FactorRDLoop
                            # 如果是其他类，尝试通用导入
                            else:
                                # 尝试从常见的RD-Agent模块导入
                                for mod_path in [
                                    "rdagent.scenarios.qlib.experiment.model_experiment",
                                    "rdagent.scenarios.qlib.experiment.factor_experiment",
                                    "rdagent.core.experiment",
                                ]:
                                    try:
                                        mod = importlib.import_module(mod_path)
                                        if hasattr(mod, name):
                                            return getattr(mod, name)
                                    except (ImportError, AttributeError):
                                        continue
                        except ImportError:
                            pass
                    
                    # 默认行为
                    return super().find_class(module, name)
            
            log_dir = (_log_root() / str(task_id)).resolve()
            if not log_dir.exists() or not log_dir.is_dir():
                return [], f"log_dir_not_found: {log_dir}"
            
            session_folder = log_dir / "__session__"
            if not session_folder.exists():
                return [], "session_folder_not_found"
            
            # 遍历所有LOOP文件夹
            loop_folders = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("Loop_")])
            all_loops_info = []
            
            for loop_folder in loop_folders:
                try:
                    loop_id = int(loop_folder.name.split("_")[1])
                except Exception:
                    continue
                
                loop_session_file = session_folder / str(loop_id) / "3_feedback"
                
                if not loop_session_file.exists():
                    all_loops_info.append({
                        "loop_id": loop_id,
                        "exp_type": "unknown",
                        "hypothesis": None,
                        "reason": None,
                        "valid_score": None,
                        "is_sota": False,
                        "feedback": None,
                    })
                    continue
                
                try:
                    with loop_session_file.open("rb") as f:
                        session = PathCompatUnpickler(f).load()
                    
                    hist = session.trace.hist
                    if len(hist) == 0:
                        all_loops_info.append({
                            "loop_id": loop_id,
                            "exp_type": "unknown",
                            "hypothesis": None,
                            "reason": None,
                            "valid_score": None,
                            "is_sota": False,
                            "feedback": None,
                        })
                        continue
                    
                    # 获取最后一个实验（当前LOOP的实验）
                    loop_exp, loop_feedback = hist[-1]
                    exp_type = type(loop_exp).__name__
                    
                    # 判断是factor还是model
                    is_factor = "Factor" in exp_type
                    is_model = "Model" in exp_type
                    
                    loop_info = {
                        "loop_id": loop_id,
                        "exp_type": exp_type,
                        "hypothesis": None,
                        "reason": None,
                        "valid_score": None,
                        "test_score": None,
                        "mle_score": None,
                        "annualized_return": None,
                        "max_drawdown": None,
                        "information_ratio": None,
                        "is_sota": bool(getattr(loop_feedback, "decision", False)),
                        "feedback": bool(getattr(loop_feedback, "decision", False)),
                    }
                    
                    # 提取因子或模型的基本信息
                    if is_factor and hasattr(loop_exp, 'sub_tasks') and len(loop_exp.sub_tasks) > 0:
                        # 获取每个因子的final_decision（通过prop_dev_feedback）
                        feedback_list = None
                        pdf = getattr(loop_exp, 'prop_dev_feedback', None)
                        if pdf is not None:
                            if hasattr(pdf, 'feedback_list'):
                                feedback_list = pdf.feedback_list
                            elif hasattr(pdf, '__iter__'):
                                feedback_list = list(pdf)
                        
                        # 收集所有因子和通过final_decision筛选的因子
                        all_factors = []
                        tested_factors = []
                        first_factor_description = None
                        
                        for idx, task in enumerate(loop_exp.sub_tasks):
                            fname = getattr(task, 'factor_name', None)
                            if not fname:
                                continue
                            all_factors.append(fname)
                            
                            # 检查final_decision
                            fd = None
                            if feedback_list and idx < len(feedback_list):
                                fb_item = feedback_list[idx]
                                if fb_item is not None and hasattr(fb_item, 'final_decision'):
                                    fd = fb_item.final_decision
                            
                            # 只有final_decision=True的因子才是真正参加回测的
                            if fd is True:
                                tested_factors.append(fname)
                                if first_factor_description is None:
                                    if hasattr(task, 'factor_description') and task.factor_description:
                                        first_factor_description = task.factor_description
                                    elif hasattr(task, 'description') and task.description:
                                        first_factor_description = task.description
                        
                        # 如果无法获取feedback_list（旧数据），回退到显示所有因子
                        if feedback_list is None:
                            tested_factors = all_factors
                        
                        # hypothesis字段存储通过筛选的因子名称（用分号连接）
                        if tested_factors:
                            loop_info['hypothesis'] = '; '.join(tested_factors)
                        elif all_factors:
                            loop_info['hypothesis'] = '; '.join(all_factors)
                        
                        # reason字段存储第一个通过筛选的因子描述
                        if first_factor_description:
                            loop_info['reason'] = first_factor_description
                        
                        # tested_factors: 真正参加回测的因子（final_decision=True）
                        # all_factors: 所有研发的因子（包括被筛掉的）
                        loop_info['tested_factors'] = tested_factors
                        loop_info['all_factors'] = all_factors
                        loop_info['tested_count'] = len(tested_factors)
                        loop_info['total_count'] = len(all_factors)
                        
                    elif is_model and hasattr(loop_exp, 'sub_tasks') and len(loop_exp.sub_tasks) > 0:
                        # 取第一个模型的信息作为代表
                        task = loop_exp.sub_tasks[0]
                        if hasattr(task, 'name'):
                            loop_info['hypothesis'] = task.name
                        if hasattr(task, 'description'):
                            loop_info['tested_factors'] = [task.name]
                        else:
                            loop_info['tested_factors'] = []
                    
                    # 提取回测结果
                    if hasattr(loop_exp, 'result') and loop_exp.result is not None:
                        try:
                            result = loop_exp.result
                            # 尝试提取常见指标
                            if hasattr(result, 'get'):
                                ic_val = result.get('IC')
                                if ic_val is not None:
                                    loop_info['valid_score'] = round(float(ic_val), 5)
                                
                                ann_ret = result.get('1day.excess_return_with_cost.annualized_return') or \
                                         result.get('1day.excess_return_without_cost.annualized_return')
                                if ann_ret is not None:
                                    loop_info['annualized_return'] = round(float(ann_ret), 5)
                                
                                max_dd = result.get('1day.excess_return_with_cost.max_drawdown') or \
                                        result.get('1day.excess_return_without_cost.max_drawdown')
                                if max_dd is not None:
                                    loop_info['max_drawdown'] = round(float(max_dd), 5)
                                
                                info_ratio = result.get('1day.excess_return_with_cost.information_ratio') or \
                                            result.get('1day.excess_return_without_cost.information_ratio')
                                if info_ratio is not None:
                                    loop_info['information_ratio'] = round(float(info_ratio), 5)
                            elif hasattr(result, 'index'):
                                # pandas Series格式
                                if 'IC' in result.index:
                                    loop_info['valid_score'] = round(float(result['IC']), 5)
                                if '1day.excess_return_without_cost.annualized_return' in result.index:
                                    loop_info['annualized_return'] = round(float(result['1day.excess_return_without_cost.annualized_return']), 5)
                                if '1day.excess_return_without_cost.max_drawdown' in result.index:
                                    loop_info['max_drawdown'] = round(float(result['1day.excess_return_without_cost.max_drawdown']), 5)
                                if '1day.excess_return_without_cost.information_ratio' in result.index:
                                    loop_info['information_ratio'] = round(float(result['1day.excess_return_without_cost.information_ratio']), 5)
                        except Exception:
                            pass
                    
                    all_loops_info.append(loop_info)
                
                except Exception as e:
                    all_loops_info.append({
                        "loop_id": loop_id,
                        "exp_type": "error",
                        "hypothesis": None,
                        "reason": f"读取失败: {str(e)}",
                        "valid_score": None,
                        "is_sota": False,
                        "feedback": None,
                    })
            
            return all_loops_info, None
        except Exception as e:
            return [], f"extract_error: {e}"
    
    @app.get("/tasks/{task_id}/loops", summary="获取Task所有LOOP的详细信息")
    def get_task_loops(task_id: str) -> dict[str, Any]:
        """获取指定task的所有LOOP详细信息，包括任务类型、回测结果、SOTA因子等。"""
        _ensure_task_log_dir(task_id)
        loops_data, error = _extract_loop_details_from_log(task_id)
        
        if error:
            return {
                "ok": False,
                "task_id": str(task_id),
                "error": error,
                "loops": [],
            }
        
        return {
            "ok": True,
            "task_id": str(task_id),
            "count": len(loops_data),
            "loops": loops_data,
        }

    @app.get("/tasks/{task_id}")
    def task_manifest(task_id: str) -> dict[str, Any]:
        return _build_task_manifest(task_id)

    @app.get("/tasks/{task_id}/model.pkl")
    def download_model_weight(task_id: str) -> StreamingResponse:
        """下载task的模型权重文件"""
        _ensure_task_log_dir(task_id)
        
        # 尝试从workspace查找model.pkl
        ws_ids = _extract_workspace_ids_from_log_dir(task_id=str(task_id))
        ws_id = ws_ids[0] if ws_ids else ""
        ws_dir = (_workspace_root() / ws_id).resolve() if ws_id else None
        
        if ws_dir and ws_dir.exists():
            # 尝试直接路径
            model_file = ws_dir / "model.pkl"
            if model_file.exists() and model_file.is_file():
                def iter_file():
                    with model_file.open("rb") as f:
                        while True:
                            chunk = f.read(1024 * 1024)
                            if not chunk:
                                break
                            yield chunk
                
                return StreamingResponse(
                    iter_file(),
                    media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=model.pkl"},
                )
            
            # 尝试从mlruns查找
            mlruns_dir = ws_dir / "mlruns"
            if mlruns_dir.exists():
                for model_path in mlruns_dir.rglob("model.pkl"):
                    if model_path.is_file():
                        def iter_file():
                            with model_path.open("rb") as f:
                                while True:
                                    chunk = f.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    yield chunk
                        
                        return StreamingResponse(
                            iter_file(),
                            media_type="application/octet-stream",
                            headers={"Content-Disposition": "attachment; filename=model.pkl"},
                        )
        
        raise HTTPException(status_code=404, detail="model.pkl not found")
    
    @app.get("/tasks/{task_id}/factor.py")
    def download_factor_entry(task_id: str) -> StreamingResponse:
        """下载task的因子入口文件"""
        _ensure_task_log_dir(task_id)
        
        # 尝试从workspace查找factor.py
        ws_ids = _extract_workspace_ids_from_log_dir(task_id=str(task_id))
        ws_id = ws_ids[0] if ws_ids else ""
        ws_dir = (_workspace_root() / ws_id).resolve() if ws_id else None
        
        if ws_dir and ws_dir.exists():
            factor_file = ws_dir / "factor.py"
            if factor_file.exists() and factor_file.is_file():
                def iter_file():
                    with factor_file.open("rb") as f:
                        while True:
                            chunk = f.read(1024 * 1024)
                            if not chunk:
                                break
                            yield chunk
                
                return StreamingResponse(
                    iter_file(),
                    media_type="text/plain",
                    headers={"Content-Disposition": "attachment; filename=factor.py"},
                )
        
        raise HTTPException(status_code=404, detail="factor.py not found")

    @app.get("/tasks/{task_id}/assets")
    def task_asset_download(task_id: str, relpath: str = Query(...)) -> StreamingResponse:
        # 兼容旧接口：优先从 workspace_dir 直接下载（best-effort）。
        # 注意：选股/同步主链路应使用 /asset_bytes（来自 session file_dict）。
        _ensure_task_log_dir(task_id)
        rp = str(relpath or "").strip().replace("\\", "/")
        while rp.startswith("/"):
            rp = rp[1:]
        if not rp:
            raise HTTPException(status_code=400, detail="relpath 为空")

        ws_ids = _extract_workspace_ids_from_log_dir(task_id=str(task_id))
        ws_id = ws_ids[0] if ws_ids else ""
        ws_dir = (_workspace_root() / ws_id).resolve() if ws_id else None
        base_dir: Path | None = ws_dir if (ws_dir is not None and ws_dir.exists()) else None
        if base_dir is None:
            raise HTTPException(status_code=404, detail="no workspace available for this task")

        # 兼容 relpath = workspaces/<ws_id>/xxx
        if rp.startswith("workspaces/"):
            parts = rp.split("/", 2)
            if len(parts) >= 3:
                rp = parts[2]

        target = (base_dir / rp).resolve()
        if not str(target).startswith(str(base_dir.resolve())):
            raise HTTPException(status_code=400, detail="invalid relpath")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"asset not found: {relpath}")

        def iter_file():
            with target.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            iter_file(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={target.name}"},
        )

    @app.get("/catalog/factors")
    def get_factor_catalog() -> Any:
        catalog_path = _aistock_catalog_root() / "factor_catalog.json"
        payload = _load_json(catalog_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="factor_catalog.json not found")
        return payload

    @app.get("/catalog/strategies")
    def get_strategy_catalog() -> Any:
        catalog_path = _aistock_catalog_root() / "strategy_catalog.json"
        payload = _load_json(catalog_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="strategy_catalog.json not found")
        return payload

    @app.get("/catalog/loops")
    def get_loop_catalog() -> Any:
        catalog_path = _aistock_catalog_root() / "loop_catalog.json"
        payload = _load_json(catalog_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="loop_catalog.json not found")
        return payload

    @app.get("/catalog/models")
    def get_model_catalog() -> Any:
        """Return model_catalog.json for AIstock, if present.

        行为与其它 /catalog/* 路由一致：
        - 从 registry.sqlite 所在目录旁的 RDagentDB/aistock/model_catalog.json 读取；
        - 若文件不存在或无法解析，则返回 404。
        """

        catalog_path = _aistock_catalog_root() / "model_catalog.json"
        payload = _load_json(catalog_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="model_catalog.json not found")
        return payload

    @app.get("/alpha158/meta")
    def get_alpha158_meta() -> Any:
        meta_path = _aistock_catalog_root() / "alpha158_meta.json"
        payload = _load_json(meta_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="alpha158_meta.json not found")
        return payload

    @app.get("/alpha360/meta")
    def get_alpha360_meta() -> Any:
        meta_path = _aistock_catalog_root() / "alpha360_meta.json"
        payload = _load_json(meta_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="alpha360_meta.json not found")
        return payload

    @app.get("/catalog/incremental")
    def get_incremental_catalog(
        last_sync_time: str | None = Query(None, description="ISO-8601 timestamp of last sync"),
        limit: int = 100,
    ) -> Any:
        """Fetch incremental loop and factor metadata since last_sync_time."""
        reg = _try_get_registry()
        if reg is None:
            raise HTTPException(status_code=501, detail="registry is not available in this Results API mode")
        db_path = reg.config.db_path
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            # 1. Fetch solidified loops
            query = "SELECT * FROM loops WHERE is_solidified = 1"
            params: list[Any] = []
            if last_sync_time:
                # Use updated_at_utc for reliable incremental sync
                query += " AND updated_at_utc > ?"
                params.append(last_sync_time)
            
            query += " ORDER BY updated_at_utc ASC LIMIT ?"
            params.append(int(limit))
            
            loops = conn.execute(query, params).fetchall()
            
            results = []
            for l in loops:
                l_dict = dict(l)
                # Fetch associated factors from factor_registry
                factors = conn.execute(
                    "SELECT * FROM factor_registry WHERE asset_bundle_id = ?",
                    (l["asset_bundle_id"],)
                ).fetchall()
                l_dict["factors"] = [dict(f) for f in factors]
                results.append(l_dict)
                
            return {
                "version": "v1",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "count": len(results),
                "loops": results
            }
        finally:
            conn.close()

    @app.get("/artifacts/bundle/{asset_bundle_id}")
    def download_bundle(asset_bundle_id: str) -> StreamingResponse:
        """Stream a zipped asset bundle for the given asset_bundle_id."""
        reg = _try_get_registry()
        if reg is None:
            raise HTTPException(status_code=501, detail="registry is not available in this Results API mode")
        db_path = Path(reg.config.db_path)
        repo_root = db_path.parent.parent
        bundle_dir = repo_root / "RDagentDB" / "production_bundles" / asset_bundle_id
        
        if not bundle_dir.exists():
            raise HTTPException(status_code=404, detail=f"Bundle {asset_bundle_id} not found")

        def iter_zip():
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(bundle_dir):
                    for file in files:
                        file_path = Path(root) / file
                        zf.write(file_path, file_path.relative_to(bundle_dir))
            
            buffer.seek(0)
            yield from buffer

        return StreamingResponse(
            iter_zip(),
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": f"attachment; filename=bundle_{asset_bundle_id}.zip"}
        )

    @app.post("/ops/sync-confirm")
    def sync_confirm(payload: dict[str, str]) -> dict[str, str]:
        """AIstock confirms successful sync of an asset bundle."""
        asset_bundle_id = payload.get("asset_bundle_id")
        status = payload.get("status")
        
        if not asset_bundle_id or status != "synced":
            raise HTTPException(status_code=400, detail="Invalid payload")
            
        reg = _try_get_registry()
        if reg is None:
            raise HTTPException(status_code=501, detail="registry is not available in this Results API mode")
        db_path = reg.config.db_path
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "UPDATE loops SET sync_status = 'synced' WHERE asset_bundle_id = ?",
                (asset_bundle_id,)
            )
            conn.commit()
            return {"status": "ok", "message": f"Bundle {asset_bundle_id} marked as synced"}
        finally:
            conn.close()

    @app.post("/ops/materialize-and-refresh")
    def materialize_and_refresh() -> dict[str, Any]:
        """Trigger offline materialization for pending loops and refresh AIstock catalogs.

        This Ops endpoint is intended to be called from AIstock backend running on the
        same machine or trusted network. It performs the following steps synchronously:

        1. Run tools/backfill_registry_artifacts.py --mode materialize-pending (提取指标与元数据)
        2. Run tools/backfill_registry_artifacts.py --mode solidify-all (固化物理资产，支持全量 Alpha158 与扁平化结构)
        3. Regenerate factor/strategy/model/loop catalogs under RDagentDB/aistock/ (指纹去重)
        """

        reg = _try_get_registry()
        if reg is None:
            raise HTTPException(status_code=501, detail="registry is not available in this Results API mode")
        db_path = Path(reg.config.db_path)
        root = db_path.parent
        project_root = Path(__file__).resolve().parents[2]
        tools_dir = project_root / "tools"

        def _run(cmd: list[str]) -> tuple[int, str]:
            try:
                # 使用 sys.executable 确保环境一致
                import sys
                
                # 如果是 Windows 环境，subprocess.run 默认支持
                proc = subprocess.run(
                    [sys.executable] + cmd if not cmd[0].startswith("python") else cmd,
                    cwd=str(project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                return proc.returncode, proc.stdout
            except Exception as e:
                return 1, f"failed to run {' '.join(cmd)}: {e}"

        results: list[dict[str, Any]] = []

        # 1) materialize-pending (Generate missing factor_meta.json/factor_perf.json)
        backfill_script = tools_dir / "backfill_registry_artifacts.py"
        code, out = _run([str(backfill_script), "--db", str(db_path), "--mode", "materialize-pending"])
        results.append({"step": "materialize_pending", "returncode": code, "output": out})

        # 2) solidify-all (Copy assets to flattened bundles and fill factor_registry)
        # 这里会触发最新的 solidification.py 逻辑，支持全量 Alpha158 提取和扁平化目录
        code, out = _run([str(backfill_script), "--db", str(db_path), "--mode", "solidify-all"])
        results.append({"step": "solidify_all", "returncode": code, "output": out})

        # 3) refresh catalogs (基于指纹去重的导出)
        catalog_scripts = [
            ("export_factor", tools_dir / "export_aistock_factor_catalog.py"),
            ("export_strategy", tools_dir / "export_aistock_strategy_catalog.py"),
            ("export_model", tools_dir / "export_aistock_model_catalog.py"),
            ("export_loop", tools_dir / "export_aistock_loop_catalog.py"),
        ]

        aistock_dir = root / "aistock"
        aistock_dir.mkdir(parents=True, exist_ok=True)
        alpha_metas = list(aistock_dir.glob("alpha*_meta.json"))

        for name, script in catalog_scripts:
            out_file = aistock_dir / f"{name.split('_')[1]}_catalog.json"
            cmd = [str(script), "--registry-sqlite", str(db_path), "--output", str(out_file)]
            if name == "export_factor":
                for am in alpha_metas:
                    cmd.extend(["--alpha-meta", str(am)])
            
            code, out = _run(cmd)
            results.append({"step": name, "returncode": code, "output": out})

        overall_ok = all(r.get("returncode") == 0 for r in results)
        if not overall_ok:
            raise HTTPException(status_code=500, detail={"steps": results})

        return {"status": "ok", "steps": results}

    @app.get("/factors/{name}")
    def get_factor(name: str) -> Any:
        # 简单实现：在 factor_catalog 中按 name 查一条
        catalog_path = _aistock_catalog_root() / "factor_catalog.json"
        payload = _load_json(catalog_path)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=404, detail="factor_catalog.json not found")
        factors = payload.get("factors") or []
        if not isinstance(factors, list):
            raise HTTPException(status_code=404, detail="invalid factor_catalog format")
        for f in factors:
            if isinstance(f, dict) and f.get("name") == name:
                return f
        raise HTTPException(status_code=404, detail="factor not found")

    @app.get("/loops/{task_run_id}/{loop_id}/artifacts")
    def get_loop_artifacts(task_run_id: str, loop_id: int) -> Any:
        # 直接查询 registry.artifacts / artifact_files 表，返回指定 loop 的汇总视图
        reg = _try_get_registry()
        if reg is None:
            raise HTTPException(status_code=501, detail="registry is not available in this Results API mode")
        conn = None
        try:
            import sqlite3

            db_path = reg.config.db_path
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            cur = conn.execute(
                """
                SELECT a.artifact_id, a.artifact_type, a.name, a.status, a.primary_flag,
                       a.entry_path, a.summary_json,
                       f.file_id, f.path AS file_path, f.sha256, f.size_bytes, f.mtime_utc, f.kind
                FROM artifacts AS a
                LEFT JOIN artifact_files AS f ON a.artifact_id = f.artifact_id
                WHERE a.task_run_id = ? AND a.loop_id = ?
                ORDER BY a.artifact_type, a.name
                """,
                (task_run_id, int(loop_id)),
            )
            rows = cur.fetchall()
        except Exception as e:  # pragma: no cover - best effort
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=f"registry query failed: {e}")

        artifacts: dict[str, dict[str, Any]] = {}
        for r in rows:
            aid = str(r["artifact_id"])
            art = artifacts.get(aid)
            if art is None:
                art = {
                    "artifact_id": aid,
                    "artifact_type": r["artifact_type"],
                    "name": r["name"],
                    "status": r["status"],
                    "primary_flag": bool(r["primary_flag"]),
                    "entry_path": r["entry_path"],
                    "summary": {},
                    "files": [],
                }
                try:
                    summary_json = r["summary_json"]
                    if isinstance(summary_json, str) and summary_json:
                        art["summary"] = json.loads(summary_json)
                except Exception:
                    pass
                artifacts[aid] = art

            file_id = r["file_id"]
            if file_id is None:
                continue
            art["files"].append(
                {
                    "file_id": file_id,
                    "path": r["file_path"],
                    "sha256": r["sha256"],
                    "size_bytes": r["size_bytes"],
                    "mtime_utc": r["mtime_utc"],
                    "kind": r["kind"],
                }
            )

        return {"task_run_id": task_run_id, "loop_id": loop_id, "artifacts": list(artifacts.values())}

    @app.get("/tasks/{task_id}/workspaces", summary="获取Task的workspace信息")
    def get_task_workspaces(task_id: str) -> dict[str, Any]:
        """获取指定task的workspace信息。
        
        双重发现策略：
        1. 文本扫描：通过正则匹配 log 目录下所有文件中的 workspace ID（覆盖面最广）
        2. Pickle解析：从 session pickle 中提取 workspace_path（兼容旧逻辑）
        两种策略结果取并集，确保覆盖所有 workspace。
        """
        import subprocess
        
        try:
            _ensure_task_log_dir(task_id)
            task_dir = (_log_root() / task_id).resolve()
            ws_root = _workspace_root()
            
            result = {
                "ok": True,
                "task_id": task_id,
                "task_dir": str(task_dir),
                "workspaces": [],
                "total_size_mb": 0
            }
            
            # ===== 收集 workspace ID（hex32）=====
            workspace_ids: set[str] = set()
            
            # 策略 A：文本扫描（主策略，覆盖面最广）
            # 复用 _extract_workspace_ids_from_log_dir，扫描 log 目录下所有文件内容
            # 匹配 RD-Agent_workspace/<hex32> 模式，不依赖 pickle 反序列化
            try:
                ids_from_scan = _extract_workspace_ids_from_log_dir(task_id=task_id)
                workspace_ids.update(ids_from_scan)
            except Exception:
                pass
            
            # 策略 B：Pickle 解析（补充策略，兼容旧逻辑）
            try:
                for loop_dir in task_dir.iterdir():
                    if not loop_dir.is_dir() or not loop_dir.name.startswith("Loop_"):
                        continue
                    runner_result_dir = loop_dir / "running" / "runner result"
                    if not runner_result_dir.exists():
                        continue
                    for pkl_file in runner_result_dir.rglob("*.pkl"):
                        try:
                            obj = _pickle_load_compat(pkl_file)
                            ws_path = getattr(getattr(obj, "experiment_workspace", None), "workspace_path", None)
                            if ws_path and isinstance(ws_path, (str, Path)):
                                for m in _ws_re.finditer(str(ws_path)):
                                    workspace_ids.add(m.group(1).lower())
                            sub_ws_list = getattr(obj, "sub_workspace_list", None) or []
                            for sub_ws in sub_ws_list:
                                if sub_ws is None:
                                    continue
                                sub_ws_path = getattr(sub_ws, "workspace_path", None)
                                if sub_ws_path and isinstance(sub_ws_path, (str, Path)):
                                    for m in _ws_re.finditer(str(sub_ws_path)):
                                        workspace_ids.add(m.group(1).lower())
                        except Exception:
                            continue
            except Exception:
                pass
            
            # ===== 合并结果，构建 workspace 信息列表 =====
            total_size_bytes = 0
            for ws_id in sorted(workspace_ids):
                try:
                    ws_path = ws_root / ws_id
                    if not ws_path.exists() or not ws_path.is_dir():
                        continue
                    
                    ws_size_mb = 0.0
                    try:
                        du_result = subprocess.run(
                            ["du", "-sb", str(ws_path)],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if du_result.returncode == 0:
                            ws_size_bytes = int(du_result.stdout.split()[0])
                            ws_size_mb = round(ws_size_bytes / (1024 * 1024), 2)
                            total_size_bytes += ws_size_bytes
                    except Exception:
                        pass
                    
                    result["workspaces"].append({
                        "name": ws_id,
                        "path": str(ws_path),
                        "size_mb": ws_size_mb
                    })
                except Exception:
                    continue
            
            result["total_size_mb"] = round(total_size_bytes / (1024 * 1024), 2)
            
            return result
            
        except Exception as e:
            return {
                "ok": False,
                "task_id": task_id,
                "error": str(e),
                "workspaces": []
            }

    @app.delete("/tasks/{task_id}", summary="删除Task及其所有数据")
    def delete_task(task_id: str) -> dict[str, Any]:
        """删除指定task的日志目录和所有workspace目录。
        
        双重发现策略（与 get_task_workspaces 一致）：
        1. 文本扫描：通过正则匹配 log 目录下所有文件中的 workspace ID
        2. Pickle解析：从 session pickle 中提取 workspace_path
        两种策略结果取并集，确保删除时不遗漏任何 workspace。
        """
        import shutil
        import subprocess
        
        try:
            task_dir = (_log_root() / task_id).resolve()
            ws_root = _workspace_root()
            
            if not task_dir.exists():
                return {
                    "ok": False,
                    "task_id": task_id,
                    "error": "Task目录不存在"
                }
            
            deleted_items = []
            total_size_mb = 0.0
            
            # ===== 双重发现策略收集 workspace ID =====
            workspace_ids: set[str] = set()
            
            # 策略 A：文本扫描（主策略）
            try:
                ids_from_scan = _extract_workspace_ids_from_log_dir(task_id=task_id)
                workspace_ids.update(ids_from_scan)
            except Exception:
                pass
            
            # 策略 B：Pickle 解析（补充策略）
            try:
                for loop_dir in task_dir.iterdir():
                    if not loop_dir.is_dir() or not loop_dir.name.startswith("Loop_"):
                        continue
                    runner_result_dir = loop_dir / "running" / "runner result"
                    if not runner_result_dir.exists():
                        continue
                    for pkl_file in runner_result_dir.rglob("*.pkl"):
                        try:
                            obj = _pickle_load_compat(pkl_file)
                            ws_path = getattr(getattr(obj, "experiment_workspace", None), "workspace_path", None)
                            if ws_path and isinstance(ws_path, (str, Path)):
                                for m in _ws_re.finditer(str(ws_path)):
                                    workspace_ids.add(m.group(1).lower())
                            sub_ws_list = getattr(obj, "sub_workspace_list", None) or []
                            for sub_ws in sub_ws_list:
                                if sub_ws is None:
                                    continue
                                sub_ws_path = getattr(sub_ws, "workspace_path", None)
                                if sub_ws_path and isinstance(sub_ws_path, (str, Path)):
                                    for m in _ws_re.finditer(str(sub_ws_path)):
                                        workspace_ids.add(m.group(1).lower())
                        except Exception:
                            continue
            except Exception:
                pass
            
            # ===== 计算 task 目录大小 =====
            task_size_mb = 0.0
            try:
                du_result = subprocess.run(
                    ["du", "-sb", str(task_dir)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if du_result.returncode == 0:
                    task_size_bytes = int(du_result.stdout.split()[0])
                    task_size_mb = round(task_size_bytes / (1024 * 1024), 2)
                    total_size_mb += task_size_mb
            except Exception:
                pass
            
            # ===== 删除 task 日志目录 =====
            shutil.rmtree(task_dir)
            deleted_items.append({
                "type": "task_log",
                "path": str(task_dir),
                "size_mb": task_size_mb
            })
            
            # ===== 删除所有发现的 workspace 目录 =====
            ws_deleted_count = 0
            for ws_id in sorted(workspace_ids):
                try:
                    ws_path = ws_root / ws_id
                    if ws_path.exists() and ws_path.is_dir():
                        ws_size_mb = 0.0
                        try:
                            du_result = subprocess.run(
                                ["du", "-sb", str(ws_path)],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            if du_result.returncode == 0:
                                ws_size_bytes = int(du_result.stdout.split()[0])
                                ws_size_mb = round(ws_size_bytes / (1024 * 1024), 2)
                                total_size_mb += ws_size_mb
                        except Exception:
                            pass
                        
                        shutil.rmtree(ws_path)
                        ws_deleted_count += 1
                        deleted_items.append({
                            "type": "workspace",
                            "path": str(ws_path),
                            "size_mb": ws_size_mb
                        })
                except Exception:
                    continue
            
            return {
                "ok": True,
                "task_id": task_id,
                "deleted_items": deleted_items,
                "total_size_mb": round(total_size_mb, 2),
                "message": f"成功删除task {task_id}及{ws_deleted_count}个workspace，释放空间 {total_size_mb:.2f} MB"
            }
            
        except Exception as e:
            return {
                "ok": False,
                "task_id": task_id,
                "error": f"删除失败: {str(e)}"
            }

    @app.get("/tasks/{task_id}/complete_assets")
    def get_task_complete_assets(task_id: str) -> Dict[str, Any]:
        """获取TASK的完整资产（SOTA因子、因子代码、模型权重、特征序列）
        
        这是新的统一API端点，复用验证脚本的成功逻辑，确保数据完整性和一致性。
        
        返回结构：
        {
            "ok": true/false,
            "task_id": "xxx",
            "session_info": {...},
            "sota_factors": {...},
            "factor_codes": [...],
            "model_weight": {...},
            "feature_sequence": {...},
            "validation": {...}
        }
        """
        try:
            from rdagent.app.task_assets_extractor import TaskAssetsExtractor
            
            log_root = _log_root()
            workspace_root = _workspace_root()
            
            extractor = TaskAssetsExtractor(log_root, workspace_root)
            result = extractor.extract_complete_assets(task_id)
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "ok": False,
                "task_id": task_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    return app
