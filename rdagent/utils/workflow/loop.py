"""
This is a class that try to store/resume/traceback the workflow session


Postscripts:
- Originally, I want to implement it in a more general way with python generator.
  However, Python generator is not picklable (dill does not support pickle as well)

"""

import asyncio
import concurrent.futures
import copy
import json
import os
import pickle
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import psutil
from tqdm.auto import tqdm

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.log.conf import LOG_SETTINGS
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer
from rdagent.utils.workflow.tracking import WorkflowTracker


def _write_abort_reason(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


class LoopMeta(type):

    @staticmethod
    def _get_steps(bases: tuple[type, ...]) -> list[str]:
        """
        Recursively get all the `steps` from the base classes and combine them into a single list.

        Args:
            bases (tuple): A tuple of base classes.

        Returns:
            List[Callable]: A list of steps combined from all base classes.
        """
        steps = []
        for base in bases:
            for step in LoopMeta._get_steps(base.__bases__) + getattr(base, "steps", []):
                if step not in steps and step not in ["load", "dump"]:  # incase user override the load/dump method
                    steps.append(step)
        return steps

    def __new__(mcs, clsname: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> Any:
        """
        Create a new class with combined steps from base classes and current class.

        Args:
            clsname (str): Name of the new class.
            bases (tuple): Base classes.
            attrs (dict): Attributes of the new class.

        Returns:
            LoopMeta: A new instance of LoopMeta.
        """
        steps = LoopMeta._get_steps(bases)  # all the base classes of parents
        for name, attr in attrs.items():
            if not name.startswith("_") and callable(attr) and not isinstance(attr, type):
                # NOTE: `not isinstance(attr, type)` is trying to exclude class type attribute
                if name not in steps and name not in ["load", "dump"]:  # incase user override the load/dump method
                    # NOTE: if we override the step in the subclass
                    # Then it is not the new step. So we skip it.
                    steps.append(name)
        attrs["steps"] = steps
        return super().__new__(mcs, clsname, bases, attrs)


@dataclass
class LoopTrace:
    start: datetime  # the start time of the trace
    end: datetime  # the end time of the trace
    step_idx: int
    # TODO: more information about the trace


class LoopBase:
    """
    Assumption:
    - The last step is responsible for recording information!!!!

    Unsolved problem:
    - Global variable synchronization when `force_subproc` is True
        - Timer
    """

    steps: list[str]  # a list of steps to work on
    loop_trace: dict[int, list[LoopTrace]]

    skip_loop_error: tuple[type[BaseException], ...] = ()  # you can define a list of error that will skip current loop
    withdraw_loop_error: tuple[
        type[BaseException], ...
    ] = ()  # you can define a list of error that will withdraw current loop

    EXCEPTION_KEY = "_EXCEPTION"
    LOOP_IDX_KEY = "_LOOP_IDX"
    SENTINEL = -1

    _pbar: tqdm  # progress bar instance

    class LoopTerminationError(Exception):
        """Exception raised when loop conditions indicate the loop should terminate"""

    class LoopResumeError(Exception):
        """Exception raised when loop conditions indicate the loop should stop all coroutines and resume"""

    def __init__(self) -> None:
        # progress control
        self.loop_idx: int = 0  # current loop index / next loop index to kickoff
        self.step_idx: defaultdict[int, int] = defaultdict(int)  # dict from loop index to next step index
        self.queue: asyncio.Queue[Any] = asyncio.Queue()

        # A stable id for the current task run. It will be pickled with the session and reused on resume.
        self.task_run_id: str = uuid.uuid4().hex

        # Guard to avoid repeatedly writing identical task_run metadata on every step.
        self._registry_task_run_written: bool = False

        # Store step results for all loops in a nested dictionary, following information will be stored:
        # - loop_prev_out[loop_index][step_name]: the output of the step function
        # - loop_prev_out[loop_index][<special keys like LOOP_IDX_KEY or EXCEPTION_KEY>]: the special keys
        self.loop_prev_out: dict[int, dict[str, Any]] = defaultdict(dict)
        self.loop_trace = defaultdict(list[LoopTrace])  # the key is the number of loop
        self.session_folder = Path(LOG_SETTINGS.trace_path) / "__session__"
        self.timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        self.tracker = WorkflowTracker(self)  # Initialize tracker with this LoopBase instance

        # progress control
        self.loop_n: Optional[int] = None  # remain loop count
        self.step_n: Optional[int] = None  # remain step count

        self.semaphores: dict[str, asyncio.Semaphore] = {}

    def get_unfinished_loop_cnt(self, next_loop: int) -> int:
        n = 0
        for li in range(next_loop):
            if self.step_idx[li] < len(self.steps):  # unfinished loop
                n += 1
        return n

    def get_semaphore(self, step_name: str) -> asyncio.Semaphore:
        if isinstance(limit := RD_AGENT_SETTINGS.step_semaphore, dict):
            limit = limit.get(step_name, 1)  # default to 1 if not specified

        # NOTE:
        # (1) we assume the record step is always the last step to modify the global environment,
        #     so we set the limit to 1 to avoid race condition
        # (2) Because we support (-1,) as local selection; So it is hard to align a) the comparision target in `feedbck`
        #     and b) parent node in `record`; So we prevent parallelism in `feedback` and `record` to avoid inconsistency
        if step_name in ("record", "feedback"):
            limit = 1

        if step_name not in self.semaphores:
            self.semaphores[step_name] = asyncio.Semaphore(limit)
        return self.semaphores[step_name]

    @property
    def pbar(self) -> tqdm:
        """Progress bar property that initializes itself if it doesn't exist."""
        if getattr(self, "_pbar", None) is None:
            self._pbar = tqdm(total=len(self.steps), desc="Workflow Progress", unit="step")
        return self._pbar

    def close_pbar(self) -> None:
        if getattr(self, "_pbar", None) is not None:
            self._pbar.close()
            del self._pbar

    def _check_exit_conditions_on_step(self, loop_id: Optional[int] = None, step_id: Optional[int] = None) -> None:
        """Check if the loop should continue or terminate.

        Raises
        ------
        LoopTerminationException
            When conditions indicate that the loop should terminate
        """
        # Check step count limitation
        if self.step_n is not None:
            if self.step_n <= 0:
                raise self.LoopTerminationError("Step count reached")
            self.step_n -= 1

        # Check timer timeout
        if self.timer.started:
            if self.timer.is_timeout():
                logger.warning("Timeout, exiting the loop.")
                raise self.LoopTerminationError("Timer timeout")
            else:
                logger.info(f"Timer remaining time: {self.timer.remain_time()}")

    async def _run_step(self, li: int, force_subproc: bool = False) -> None:
        """Execute a single step (next unrun step) in the workflow (async version with force_subproc option).

        Parameters
        ----------
        li : int
            Loop index

        force_subproc : bool
            Whether to force the step to run in a subprocess in asyncio

        Returns
        -------
        Any
            The result of the step function
        """
        si = self.step_idx[li]
        name = self.steps[si]

        async with self.get_semaphore(name):

            logger.info(f"Start Loop {li}, Step {si}: {name}")
            self.tracker.log_workflow_state()

            with logger.tag(f"Loop_{li}.{name}"):
                start = datetime.now(timezone.utc)
                func: Callable[..., Any] = cast(Callable[..., Any], getattr(self, name))

                def _safe_get_action(loop_prev_out: dict[str, Any]) -> str | None:
                    try:
                        deg = loop_prev_out.get("direct_exp_gen")
                        if isinstance(deg, dict):
                            propose = deg.get("propose")
                            return getattr(propose, "action", None)
                    except Exception:
                        return None

                def _safe_get_workspace_id_from_path(p: Any) -> str | None:
                    try:
                        pp = Path(str(p))
                        return pp.name or None
                    except Exception:
                        return None

                def _best_effort_upsert_workspaces(reg, *, exp: Any, status: str) -> None:
                    try:
                        ew = getattr(exp, "experiment_workspace", None)
                        if ew is not None:
                            ew_path = getattr(ew, "workspace_path", None)
                            ws_id = _safe_get_workspace_id_from_path(ew_path)
                            if ws_id and ew_path is not None:
                                reg.upsert_workspace(
                                    workspace_id=ws_id,
                                    task_run_id=getattr(self, "task_run_id", ""),
                                    loop_id=li,
                                    workspace_role="experiment_workspace",
                                    experiment_type=_safe_get_action(self.loop_prev_out.get(li, {})),
                                    step_name=name,
                                    status=status,
                                    workspace_path=str(ew_path),
                                )
                    except Exception:
                        pass

                    try:
                        sws = getattr(exp, "sub_workspace_list", None)
                        if isinstance(sws, list):
                            for sw in sws:
                                if sw is None:
                                    continue
                                sw_path = getattr(sw, "workspace_path", None)
                                ws_id = _safe_get_workspace_id_from_path(sw_path)
                                if ws_id and sw_path is not None:
                                    reg.upsert_workspace(
                                        workspace_id=ws_id,
                                        task_run_id=getattr(self, "task_run_id", ""),
                                        loop_id=li,
                                        workspace_role="sub_workspace",
                                        experiment_type=_safe_get_action(self.loop_prev_out.get(li, {})),
                                        step_name=name,
                                        status=status,
                                        workspace_path=str(sw_path),
                                    )
                    except Exception:
                        pass

                # Best-effort registry hook at step start (must never break main workflow).
                try:
                    from rdagent.utils.registry.sqlite_registry import get_registry, should_enable_registry
                    from rdagent.log.conf import LOG_SETTINGS

                    if should_enable_registry():
                        reg = get_registry()
                        if not getattr(self, "_registry_task_run_written", False):
                            reg.upsert_task_run(
                                task_run_id=getattr(self, "task_run_id", ""),
                                scenario=type(self).__name__,
                                status="running",
                                log_trace_path=str(getattr(LOG_SETTINGS, "trace_path", "")),
                                params={
                                    "step_semaphore": RD_AGENT_SETTINGS.step_semaphore,
                                    "workspace_path": str(getattr(RD_AGENT_SETTINGS, "workspace_path", "")),
                                    "max_parallel": RD_AGENT_SETTINGS.get_max_parallel(),
                                    "force_subproc": RD_AGENT_SETTINGS.is_force_subproc(),
                                },
                            )
                            self._registry_task_run_written = True
                        reg.upsert_loop(
                            task_run_id=getattr(self, "task_run_id", ""),
                            loop_id=li,
                            action=_safe_get_action(self.loop_prev_out.get(li, {})),
                            status="running",
                        )
                except Exception:
                    pass

                next_step_idx = si + 1
                step_forward = True
                # NOTE: each step are aware are of current loop index
                # It is very important to set it before calling the step function!
                self.loop_prev_out[li][self.LOOP_IDX_KEY] = li

                try:
                    # Call function with current loop's output, await if coroutine or use ProcessPoolExecutor for sync if required
                    if force_subproc:
                        curr_loop = asyncio.get_running_loop()
                        with concurrent.futures.ProcessPoolExecutor() as pool:
                            # Using deepcopy is to avoid triggering errors like "RuntimeError: dictionary changed size during iteration"
                            # GUESS: Some content in self.loop_prev_out[li] may be in the middle of being changed.
                            result = await curr_loop.run_in_executor(
                                pool, copy.deepcopy(func), copy.deepcopy(self.loop_prev_out[li])
                            )
                    else:
                        # auto determine whether to run async or sync
                        if asyncio.iscoroutinefunction(func):
                            result = await func(self.loop_prev_out[li])
                        else:
                            # Default: run sync function directly
                            result = func(self.loop_prev_out[li])
                    # Store result in the nested dictionary
                    self.loop_prev_out[li][name] = result
                except Exception as e:
                    if isinstance(e, self.skip_loop_error):
                        logger.warning(f"Skip loop {li} due to {e}")
                        # Jump to the last step (assuming last step is for recording)
                        next_step_idx = len(self.steps) - 1
                        self.loop_prev_out[li][name] = None
                        self.loop_prev_out[li][self.EXCEPTION_KEY] = e
                    elif isinstance(e, self.withdraw_loop_error):
                        logger.warning(f"Withdraw loop {li} due to {e}")
                        # Back to previous loop
                        self.withdraw_loop(li)
                        step_forward = False

                        msg = "We have reset the loop instance, stop all the routines and resume."
                        raise self.LoopResumeError(msg) from e
                    else:
                        raise  # re-raise unhandled exceptions
                finally:
                    # No matter the execution succeed or not, we have to finish the following steps

                    # Best-effort registry hook at step end.
                    try:
                        from rdagent.utils.registry.sqlite_registry import get_registry, should_enable_registry

                        if should_enable_registry():
                            reg = get_registry()
                            is_failed = self.EXCEPTION_KEY in self.loop_prev_out.get(li, {})
                            is_final_step = si == (len(self.steps) - 1)
                            status = "failed" if is_failed else ("success" if is_final_step else "running")
                            err = self.loop_prev_out.get(li, {}).get(self.EXCEPTION_KEY)
                            reg.upsert_loop(
                                task_run_id=getattr(self, "task_run_id", ""),
                                loop_id=li,
                                action=_safe_get_action(self.loop_prev_out.get(li, {})),
                                status=status,
                                error_type=type(err).__name__ if err is not None else None,
                                error_message=str(err) if err is not None else None,
                            )

                            exp_obj = self.loop_prev_out.get(li, {}).get(name)
                            if exp_obj is None:
                                try:
                                    li_state = self.loop_prev_out.get(li, {})
                                    if isinstance(li_state, dict):
                                        for _k in reversed(list(li_state.keys())):
                                            if _k in {self.EXCEPTION_KEY}:
                                                continue
                                            v = li_state.get(_k)
                                            if v is None:
                                                continue
                                            if hasattr(v, "experiment_workspace") or hasattr(v, "result"):
                                                exp_obj = v
                                                break
                                except Exception:
                                    exp_obj = exp_obj
                            if exp_obj is not None:
                                _best_effort_upsert_workspaces(reg, exp=exp_obj, status=status)

                            if is_final_step and (not is_failed):
                                try:
                                    exp_for_metrics = exp_obj
                                    metrics_series = getattr(exp_for_metrics, "result", None)
                                    metrics: dict[str, Any] | None = None
                                    if metrics_series is not None:
                                        try:
                                            metrics = dict(metrics_series)
                                        except Exception:
                                            metrics = None

                                    ew = getattr(exp_for_metrics, "experiment_workspace", None)
                                    ew_path = getattr(ew, "workspace_path", None) if ew is not None else None
                                    ws_id = _safe_get_workspace_id_from_path(ew_path)
                                    if ws_id and ew_path is not None:
                                        ws_root = Path(str(ew_path))
                                        qlib_res = ws_root / "qlib_res.csv"
                                        ret_pkl = ws_root / "ret.pkl"
                                        ret_schema_parquet = ws_root / "ret_schema.parquet"
                                        ret_schema_json = ws_root / "ret_schema.json"
                                        signals_parquet = ws_root / "signals.parquet"
                                        signals_json = ws_root / "signals.json"
                                        mlruns = ws_root / "mlruns"

                                        combined_factors = ws_root / "combined_factors_df.parquet"
                                        yaml_confs = list(ws_root.glob("*.yaml")) + list(ws_root.glob("*.yml"))

                                        action = _safe_get_action(self.loop_prev_out.get(li, {}))
                                        has_result = False
                                        if action == "model":
                                            has_result = qlib_res.exists() and ret_pkl.exists()
                                        elif action == "factor":
                                            has_result = combined_factors.exists()

                                        if has_result:
                                            reg.update_loop_metrics(
                                                task_run_id=getattr(self, "task_run_id", ""),
                                                loop_id=li,
                                                metrics=metrics,
                                                best_workspace_id=ws_id,
                                                has_result=True,
                                            )

                                        def _rel(p: Path) -> str:
                                            try:
                                                return str(p.relative_to(ws_root))
                                            except Exception:
                                                return str(p)

                                        def _write_json(path: Path, payload: dict[str, Any]) -> None:
                                            try:
                                                with path.open("w", encoding="utf-8") as f:
                                                    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
                                            except Exception:
                                                pass

                                        meta_path = ws_root / "workspace_meta.json"
                                        summary_path = ws_root / "experiment_summary.json"
                                        manifest_path = ws_root / "manifest.json"

                                        now_utc = datetime.now(timezone.utc).isoformat()

                                        def _metrics_summary(m: dict[str, Any] | None) -> dict[str, Any]:
                                            if not m:
                                                return {}
                                            out: dict[str, Any] = {}
                                            key_map = {
                                                "ic_mean": ["IC", "ic_mean", "ic"],
                                                "rank_ic_mean": ["Rank IC", "rank_ic_mean", "rank_ic"],
                                                "ann_return": ["annualized_return", "ann_return"],
                                                "mdd": ["max_drawdown", "mdd"],
                                                "turnover": ["turnover"],
                                                "multi_score": ["multi_score", "sharpe"],
                                            }
                                            for k, cands in key_map.items():
                                                for cand in cands:
                                                    if cand in m and m[cand] is not None:
                                                        out[k] = m[cand]
                                                        break
                                            return out

                                        if action == "model":
                                            result_criteria = {
                                                "required_files": ["qlib_res.csv", "ret.pkl"],
                                                "has_result": bool(qlib_res.exists() and ret_pkl.exists()),
                                            }
                                        elif action == "factor":
                                            result_criteria = {
                                                "required_files": ["combined_factors_df.parquet"],
                                                "has_result": bool(combined_factors.exists()),
                                            }
                                        else:
                                            result_criteria = {"required_files": [], "has_result": bool(has_result)}

                                        reg_db_path = None
                                        try:
                                            reg_db_path = str(getattr(reg, "config", None).db_path)
                                        except Exception:
                                            reg_db_path = None

                                        workspace_row = {
                                            "workspace_id": ws_id,
                                            "task_run_id": getattr(self, "task_run_id", ""),
                                            "loop_id": li,
                                            "workspace_role": "experiment_workspace",
                                            "experiment_type": action,
                                            "step_name": name,
                                            "status": status,
                                            "workspace_path": str(ws_root),
                                            "meta_path": _rel(meta_path),
                                            "summary_path": _rel(summary_path),
                                            "manifest_path": _rel(manifest_path),
                                        }

                                        task_run_snapshot = {
                                            "task_run_id": getattr(self, "task_run_id", ""),
                                            "scenario": type(self).__name__,
                                            "log_trace_path": str(getattr(LOG_SETTINGS, "trace_path", "")),
                                        }

                                        loop_snapshot = {
                                            "task_run_id": getattr(self, "task_run_id", ""),
                                            "loop_id": li,
                                            "action": action,
                                            "status": status,
                                            "has_result": has_result,
                                            "best_workspace_id": ws_id,
                                            "key_metrics": _metrics_summary(metrics),
                                        }

                                        meta_payload = {
                                            "task_run_id": getattr(self, "task_run_id", ""),
                                            "loop_id": li,
                                            "step_name": name,
                                            "action": action,
                                            "workspace_id": ws_id,
                                            "best_workspace_id": ws_id,
                                            "workspace_role": "experiment_workspace",
                                            "experiment_type": action,
                                            "workspace_path": str(ws_root),
                                            "status": status,
                                            "has_result": has_result,
                                            "generated_at_utc": now_utc,
                                            "result_criteria": result_criteria,
                                            "registry": {
                                                "db_path": reg_db_path,
                                            },
                                            "task_run": task_run_snapshot,
                                            "loop": loop_snapshot,
                                            "workspace_row": workspace_row,
                                            "pointers": {
                                                "meta_path": _rel(meta_path),
                                                "summary_path": _rel(summary_path),
                                                "manifest_path": _rel(manifest_path),
                                            },
                                        }
                                        _write_json(meta_path, meta_payload)

                                        summary_payload: dict[str, Any] = {
                                            "task_run_id": getattr(self, "task_run_id", ""),
                                            "loop_id": li,
                                            "step_name": name,
                                            "action": action,
                                            "workspace_id": ws_id,
                                            "best_workspace_id": ws_id,
                                            "workspace_role": "experiment_workspace",
                                            "experiment_type": action,
                                            "workspace_path": str(ws_root),
                                            "status": status,
                                            "has_result": has_result,
                                            "generated_at_utc": now_utc,
                                            "result_criteria": result_criteria,
                                            "key_metrics": _metrics_summary(metrics),
                                            "registry": {
                                                "db_path": reg_db_path,
                                            },
                                            "task_run": task_run_snapshot,
                                            "loop": loop_snapshot,
                                            "workspace_row": workspace_row,
                                            "pointers": {
                                                "meta_path": _rel(meta_path),
                                                "summary_path": _rel(summary_path),
                                                "manifest_path": _rel(manifest_path),
                                            },
                                            "metrics": metrics or {},
                                            "files": {
                                                "qlib_res.csv": _rel(qlib_res) if qlib_res.exists() else None,
                                                "ret.pkl": _rel(ret_pkl) if ret_pkl.exists() else None,
                                                "ret_schema.parquet": _rel(ret_schema_parquet)
                                                if ret_schema_parquet.exists()
                                                else None,
                                                "ret_schema.json": _rel(ret_schema_json)
                                                if ret_schema_json.exists()
                                                else None,
                                                "signals.parquet": _rel(signals_parquet)
                                                if signals_parquet.exists()
                                                else None,
                                                "signals.json": _rel(signals_json) if signals_json.exists() else None,
                                                "combined_factors_df.parquet": _rel(combined_factors)
                                                if combined_factors.exists()
                                                else None,
                                                "mlruns": _rel(mlruns) if mlruns.exists() else None,
                                            },
                                            "artifacts": [],
                                        }
                                        _write_json(summary_path, summary_payload)

                                        manifest_payload: dict[str, Any] = {
                                            "task_run_id": getattr(self, "task_run_id", ""),
                                            "loop_id": li,
                                            "action": action,
                                            "workspace_id": ws_id,
                                            "best_workspace_id": ws_id,
                                            "workspace_role": "experiment_workspace",
                                            "experiment_type": action,
                                            "workspace_path": str(ws_root),
                                            "status": status,
                                            "has_result": has_result,
                                            "generated_at_utc": now_utc,
                                            "result_criteria": result_criteria,
                                            "key_metrics": _metrics_summary(metrics),
                                            "registry": {
                                                "db_path": reg_db_path,
                                            },
                                            "task_run": task_run_snapshot,
                                            "loop": loop_snapshot,
                                            "workspace_row": workspace_row,
                                            "pointers": {
                                                "meta_path": _rel(meta_path),
                                                "summary_path": _rel(summary_path),
                                                "manifest_path": _rel(manifest_path),
                                            },
                                            "artifacts": [],
                                        }
                                        _write_json(manifest_path, manifest_payload)

                                        try:
                                            reg.upsert_workspace(
                                                workspace_id=ws_id,
                                                task_run_id=getattr(self, "task_run_id", ""),
                                                loop_id=li,
                                                workspace_role="experiment_workspace",
                                                experiment_type=action,
                                                step_name=name,
                                                status=status,
                                                workspace_path=str(ws_root),
                                                meta_path=_rel(meta_path),
                                                summary_path=_rel(summary_path),
                                                manifest_path=_rel(manifest_path),
                                            )
                                        except Exception:
                                            pass

                                        if action == "model":
                                            report_artifact_id = uuid.uuid4().hex
                                            reg.upsert_artifact(
                                                artifact_id=report_artifact_id,
                                                task_run_id=getattr(self, "task_run_id", ""),
                                                loop_id=li,
                                                workspace_id=ws_id,
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
                                                entry_path=_rel(qlib_res),
                                            )

                                            summary_payload["artifacts"].append(
                                                {
                                                    "artifact_id": report_artifact_id,
                                                    "artifact_type": "report",
                                                    "name": "qlib_report",
                                                    "status": "present"
                                                    if (qlib_res.exists() and ret_pkl.exists())
                                                    else "missing",
                                                    "entry_path": _rel(qlib_res),
                                                    "files": [
                                                        _rel(qlib_res) if qlib_res.exists() else None,
                                                        _rel(ret_pkl) if ret_pkl.exists() else None,
                                                        _rel(ret_schema_parquet)
                                                        if ret_schema_parquet.exists()
                                                        else None,
                                                        _rel(ret_schema_json) if ret_schema_json.exists() else None,
                                                        _rel(signals_parquet)
                                                        if signals_parquet.exists()
                                                        else None,
                                                        _rel(signals_json) if signals_json.exists() else None,
                                                    ],
                                                }
                                            )
                                            manifest_payload["artifacts"].append(
                                                {
                                                    "artifact_id": report_artifact_id,
                                                    "artifact_type": "report",
                                                    "name": "qlib_report",
                                                    "status": "present"
                                                    if (qlib_res.exists() and ret_pkl.exists())
                                                    else "missing",
                                                    "entry_path": _rel(qlib_res),
                                                    "files": [
                                                        _rel(qlib_res) if qlib_res.exists() else None,
                                                        _rel(ret_pkl) if ret_pkl.exists() else None,
                                                        _rel(ret_schema_parquet)
                                                        if ret_schema_parquet.exists()
                                                        else None,
                                                        _rel(ret_schema_json) if ret_schema_json.exists() else None,
                                                        _rel(signals_parquet)
                                                        if signals_parquet.exists()
                                                        else None,
                                                        _rel(signals_json) if signals_json.exists() else None,
                                                    ],
                                                }
                                            )

                                            for p in [
                                                qlib_res,
                                                ret_pkl,
                                                ret_schema_parquet,
                                                ret_schema_json,
                                                signals_parquet,
                                                signals_json,
                                            ]:
                                                try:
                                                    if not p.exists():
                                                        continue
                                                    size_bytes, mtime_utc = reg._best_effort_file_meta(p)
                                                    sha256 = reg._best_effort_sha256(p) if p.is_file() else None
                                                    reg.upsert_artifact_file(
                                                        file_id=uuid.uuid4().hex,
                                                        artifact_id=report_artifact_id,
                                                        workspace_id=ws_id,
                                                        path=_rel(p),
                                                        sha256=sha256,
                                                        size_bytes=size_bytes,
                                                        mtime_utc=mtime_utc,
                                                        kind="report",
                                                    )
                                                except Exception:
                                                    pass

                                            model_artifact_id = uuid.uuid4().hex
                                            reg.upsert_artifact(
                                                artifact_id=model_artifact_id,
                                                task_run_id=getattr(self, "task_run_id", ""),
                                                loop_id=li,
                                                workspace_id=ws_id,
                                                artifact_type="model",
                                                name="mlruns",
                                                status="present" if mlruns.exists() else "missing",
                                                primary_flag=False,
                                                summary={"path": "mlruns"},
                                                entry_path=_rel(mlruns),
                                            )

                                            summary_payload["artifacts"].append(
                                                {
                                                    "artifact_id": model_artifact_id,
                                                    "artifact_type": "model",
                                                    "name": "mlruns",
                                                    "status": "present" if mlruns.exists() else "missing",
                                                    "entry_path": _rel(mlruns),
                                                    "files": [_rel(mlruns) if mlruns.exists() else None],
                                                }
                                            )
                                            manifest_payload["artifacts"].append(
                                                {
                                                    "artifact_id": model_artifact_id,
                                                    "artifact_type": "model",
                                                    "name": "mlruns",
                                                    "status": "present" if mlruns.exists() else "missing",
                                                    "entry_path": _rel(mlruns),
                                                    "files": [_rel(mlruns) if mlruns.exists() else None],
                                                }
                                            )
                                            if mlruns.exists():
                                                try:
                                                    size_bytes, mtime_utc = reg._best_effort_file_meta(mlruns)
                                                    reg.upsert_artifact_file(
                                                        file_id=uuid.uuid4().hex,
                                                        artifact_id=model_artifact_id,
                                                        workspace_id=ws_id,
                                                        path=_rel(mlruns),
                                                        sha256=None,
                                                        size_bytes=size_bytes,
                                                        mtime_utc=mtime_utc,
                                                        kind="model",
                                                    )
                                                except Exception:
                                                    pass

                                                try:
                                                    root_parts_len = len(mlruns.resolve().parts)
                                                    max_depth = 6
                                                    max_files = 60
                                                    picked = 0
                                                    key_names = {
                                                        "mlmodel",
                                                        "conda.yaml",
                                                        "requirements.txt",
                                                        "python_env.yaml",
                                                        "model.pkl",
                                                    }
                                                    for fp in mlruns.rglob("*"):
                                                        try:
                                                            if picked >= max_files:
                                                                break
                                                            if not fp.is_file():
                                                                continue
                                                            depth = len(fp.resolve().parts) - root_parts_len
                                                            if depth > max_depth:
                                                                continue
                                                            nm = fp.name.lower()
                                                            if (nm not in key_names) and (not nm.endswith(".pkl")):
                                                                continue

                                                            size_bytes, mtime_utc = reg._best_effort_file_meta(fp)
                                                            sha256 = reg._best_effort_sha256(fp)
                                                            reg.upsert_artifact_file(
                                                                file_id=uuid.uuid4().hex,
                                                                artifact_id=model_artifact_id,
                                                                workspace_id=ws_id,
                                                                path=_rel(fp),
                                                                sha256=sha256,
                                                                size_bytes=size_bytes,
                                                                mtime_utc=mtime_utc,
                                                                kind="model",
                                                            )
                                                            picked += 1
                                                        except Exception:
                                                            continue
                                                except Exception:
                                                    pass

                                            cfg_artifact_id = uuid.uuid4().hex
                                            reg.upsert_artifact(
                                                artifact_id=cfg_artifact_id,
                                                task_run_id=getattr(self, "task_run_id", ""),
                                                loop_id=li,
                                                workspace_id=ws_id,
                                                artifact_type="config_snapshot",
                                                name="workspace_configs",
                                                status="present" if len(yaml_confs) > 0 else "missing",
                                                primary_flag=False,
                                                summary={"count": len(yaml_confs)},
                                                entry_path=str(ws_root),
                                            )

                                            summary_payload["artifacts"].append(
                                                {
                                                    "artifact_id": cfg_artifact_id,
                                                    "artifact_type": "config_snapshot",
                                                    "name": "workspace_configs",
                                                    "status": "present" if len(yaml_confs) > 0 else "missing",
                                                    "entry_path": ".",
                                                    "files": [
                                                        _rel(p)
                                                        for p in (yaml_confs[:50] if len(yaml_confs) > 0 else [])
                                                        if p.exists()
                                                    ],
                                                }
                                            )
                                            manifest_payload["artifacts"].append(
                                                {
                                                    "artifact_id": cfg_artifact_id,
                                                    "artifact_type": "config_snapshot",
                                                    "name": "workspace_configs",
                                                    "status": "present" if len(yaml_confs) > 0 else "missing",
                                                    "entry_path": ".",
                                                    "files": [
                                                        _rel(p)
                                                        for p in (yaml_confs[:50] if len(yaml_confs) > 0 else [])
                                                        if p.exists()
                                                    ],
                                                }
                                            )
                                            for yp in yaml_confs:
                                                try:
                                                    if not yp.exists() or not yp.is_file():
                                                        continue
                                                    size_bytes, mtime_utc = reg._best_effort_file_meta(yp)
                                                    sha256 = reg._best_effort_sha256(yp)
                                                    reg.upsert_artifact_file(
                                                        file_id=uuid.uuid4().hex,
                                                        artifact_id=cfg_artifact_id,
                                                        workspace_id=ws_id,
                                                        path=_rel(yp),
                                                        sha256=sha256,
                                                        size_bytes=size_bytes,
                                                        mtime_utc=mtime_utc,
                                                        kind="config",
                                                    )
                                                except Exception:
                                                    pass

                                            _write_json(summary_path, summary_payload)
                                            _write_json(manifest_path, manifest_payload)
                                        elif action == "factor":
                                            fs_artifact_id = uuid.uuid4().hex
                                            reg.upsert_artifact(
                                                artifact_id=fs_artifact_id,
                                                task_run_id=getattr(self, "task_run_id", ""),
                                                loop_id=li,
                                                workspace_id=ws_id,
                                                artifact_type="feature_set",
                                                name="combined_factors_df",
                                                status="present" if combined_factors.exists() else "missing",
                                                primary_flag=True,
                                                summary={"file": "combined_factors_df.parquet"},
                                                entry_path=_rel(combined_factors),
                                            )

                                            summary_payload["artifacts"].append(
                                                {
                                                    "artifact_id": fs_artifact_id,
                                                    "artifact_type": "feature_set",
                                                    "name": "combined_factors_df",
                                                    "status": "present" if combined_factors.exists() else "missing",
                                                    "entry_path": _rel(combined_factors),
                                                    "files": [_rel(combined_factors) if combined_factors.exists() else None],
                                                }
                                            )
                                            manifest_payload["artifacts"].append(
                                                {
                                                    "artifact_id": fs_artifact_id,
                                                    "artifact_type": "feature_set",
                                                    "name": "combined_factors_df",
                                                    "status": "present" if combined_factors.exists() else "missing",
                                                    "entry_path": _rel(combined_factors),
                                                    "files": [_rel(combined_factors) if combined_factors.exists() else None],
                                                }
                                            )
                                            if combined_factors.exists():
                                                try:
                                                    size_bytes, mtime_utc = reg._best_effort_file_meta(combined_factors)
                                                    sha256 = reg._best_effort_sha256(combined_factors)
                                                    reg.upsert_artifact_file(
                                                        file_id=uuid.uuid4().hex,
                                                        artifact_id=fs_artifact_id,
                                                        workspace_id=ws_id,
                                                        path=_rel(combined_factors),
                                                        sha256=sha256,
                                                        size_bytes=size_bytes,
                                                        mtime_utc=mtime_utc,
                                                        kind="data",
                                                    )
                                                except Exception:
                                                    pass

                                            cfg_artifact_id = uuid.uuid4().hex
                                            reg.upsert_artifact(
                                                artifact_id=cfg_artifact_id,
                                                task_run_id=getattr(self, "task_run_id", ""),
                                                loop_id=li,
                                                workspace_id=ws_id,
                                                artifact_type="config_snapshot",
                                                name="workspace_configs",
                                                status="present" if len(yaml_confs) > 0 else "missing",
                                                primary_flag=False,
                                                summary={"count": len(yaml_confs)},
                                                entry_path=str(ws_root),
                                            )

                                            summary_payload["artifacts"].append(
                                                {
                                                    "artifact_id": cfg_artifact_id,
                                                    "artifact_type": "config_snapshot",
                                                    "name": "workspace_configs",
                                                    "status": "present" if len(yaml_confs) > 0 else "missing",
                                                    "entry_path": ".",
                                                }
                                            )
                                            manifest_payload["artifacts"].append(
                                                {
                                                    "artifact_id": cfg_artifact_id,
                                                    "artifact_type": "config_snapshot",
                                                    "name": "workspace_configs",
                                                    "status": "present" if len(yaml_confs) > 0 else "missing",
                                                    "entry_path": ".",
                                                }
                                            )

                                            for yp in yaml_confs:
                                                try:
                                                    if not yp.exists() or not yp.is_file():
                                                        continue
                                                    size_bytes, mtime_utc = reg._best_effort_file_meta(yp)
                                                    sha256 = reg._best_effort_sha256(yp)
                                                    reg.upsert_artifact_file(
                                                        file_id=uuid.uuid4().hex,
                                                        artifact_id=cfg_artifact_id,
                                                        workspace_id=ws_id,
                                                        path=_rel(yp),
                                                        sha256=sha256,
                                                        size_bytes=size_bytes,
                                                        mtime_utc=mtime_utc,
                                                        kind="config",
                                                    )
                                                except Exception:
                                                    pass

                                            _write_json(summary_path, summary_payload)
                                            _write_json(manifest_path, manifest_payload)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Record the trace
                    end = datetime.now(timezone.utc)
                    self.loop_trace[li].append(LoopTrace(start, end, step_idx=si))
                    logger.log_object(
                        {
                            "start_time": start,
                            "end_time": end,
                        },
                        tag="time_info",
                    )
                    if step_forward:
                        # Increment step index
                        self.step_idx[li] = next_step_idx

                        # Update progress bar
                        current_step = self.step_idx[li]
                        self.pbar.n = current_step
                        next_step = self.step_idx[li] % len(self.steps)
                        self.pbar.set_postfix(
                            loop_index=li + next_step_idx // len(self.steps),
                            step_index=next_step,
                            step_name=self.steps[next_step],
                        )

                        # Save snapshot after completing the step;
                        # 1) It has to be after the step_idx is updated, so loading the snapshot will be on the right step.
                        # 2) Only save it when the step forward, withdraw does not worth saving.
                        if name in self.loop_prev_out[li]:
                            # 3) Only dump the step if (so we don't have to redo the step when we load the session again)
                            # it has been executed successfully
                            self.dump(self.session_folder / f"{li}" / f"{si}_{name}")

                        self._check_exit_conditions_on_step(loop_id=li, step_id=si)
                    else:
                        logger.warning(f"Step forward {si} of loop {li} is skipped.")

    async def kickoff_loop(self) -> None:
        while True:
            li = self.loop_idx

            # exit on loop limitation
            if self.loop_n is not None:
                if self.loop_n <= 0:
                    for _ in range(RD_AGENT_SETTINGS.get_max_parallel()):
                        self.queue.put_nowait(self.SENTINEL)
                    break
                self.loop_n -= 1

            # NOTE:
            # Try best to kick off the first step; the first step is always the ExpGen;
            # it have the right to decide when to stop yield new Experiment
            if self.step_idx[li] == 0:
                # Assume the first step is ExpGen
                # Only kick off ExpGen when it is never kicked off before
                await self._run_step(li)
            self.queue.put_nowait(li)  # the loop `li` has been kicked off, waiting for workers to pick it up
            self.loop_idx += 1
            await asyncio.sleep(0)

    async def execute_loop(self) -> None:
        while True:
            # 1) get the tasks to goon loop `li`
            li = await self.queue.get()
            if li == self.SENTINEL:
                break
            # 2) run the unfinished steps
            while self.step_idx[li] < len(self.steps):
                if self.step_idx[li] == len(self.steps) - 1:
                    # NOTE: assume the last step is record, it will be fast and affect the global environment
                    # if it is the last step, run it directly ()
                    await self._run_step(li)
                else:
                    # await the step; parallel running happens here!
                    # Only trigger subprocess if we have more than one process.
                    await self._run_step(li, force_subproc=RD_AGENT_SETTINGS.is_force_subproc())

    async def run(self, step_n: int | None = None, loop_n: int | None = None, all_duration: str | None = None) -> None:
        """Run the workflow loop.

        Parameters
        ----------
        loop_n: int | None
            How many loops to run; if current loop is incomplete, it will be counted as the first loop for completion
            `None` indicates to run forever until error or KeyboardInterrupt
        all_duration : str | None
            Maximum duration to run, in format accepted by the timer
        """
        # Initialize timer if duration is provided
        if all_duration is not None and not self.timer.started:
            self.timer.reset(all_duration=all_duration)

        if step_n is not None:
            self.step_n = step_n
        if loop_n is not None:
            self.loop_n = loop_n

        # empty the queue when restarting
        while not self.queue.empty():
            self.queue.get_nowait()
        self.loop_idx = (
            0  # if we rerun the loop, we should revert the loop index to 0 to make sure every loop is correctly kicked
        )

        tasks: list[asyncio.Task] = []
        while True:
            try:
                # run one kickoff_loop and execute_loop
                tasks = [
                    asyncio.create_task(t)
                    for t in [
                        self.kickoff_loop(),
                        *[self.execute_loop() for _ in range(RD_AGENT_SETTINGS.get_max_parallel())],
                    ]
                ]
                await asyncio.gather(*tasks)

                # Best-effort task_run final status update.
                try:
                    from rdagent.utils.registry.sqlite_registry import get_registry, should_enable_registry

                    if should_enable_registry():
                        get_registry().upsert_task_run(
                            task_run_id=getattr(self, "task_run_id", ""),
                            scenario=type(self).__name__,
                            status="success",
                            log_trace_path=str(getattr(LOG_SETTINGS, "trace_path", "")),
                        )
                except Exception:
                    pass
                break
            except self.LoopResumeError as e:
                logger.warning(f"Stop all the routines and resume loop: {e}")
                self.loop_idx = 0
            except self.LoopTerminationError as e:
                _write_abort_reason(
                    Path(LOG_SETTINGS.trace_path) / "abort_reason.json",
                    {
                        "time_utc": datetime.now(timezone.utc).isoformat(),
                        "reason": str(e),
                        "exception_type": type(e).__name__,
                        "loop_idx": self.loop_idx,
                        "step_idx": getattr(self, "step_idx", None),
                        "max_parallel": RD_AGENT_SETTINGS.get_max_parallel(),
                        "traceback": traceback.format_exc(),
                    },
                )
                logger.warning(f"Reach stop criterion and stop loop: {e}")
                kill_subprocesses()  # NOTE: coroutine-based workflow can't automatically stop subprocesses.

                # Best-effort task_run abort status update.
                try:
                    from rdagent.utils.registry.sqlite_registry import get_registry, should_enable_registry

                    if should_enable_registry():
                        get_registry().upsert_task_run(
                            task_run_id=getattr(self, "task_run_id", ""),
                            scenario=type(self).__name__,
                            status="aborted",
                            log_trace_path=str(getattr(LOG_SETTINGS, "trace_path", "")),
                        )
                except Exception:
                    pass
                break
            except (KeyboardInterrupt, asyncio.CancelledError) as e:
                _write_abort_reason(
                    Path(LOG_SETTINGS.trace_path) / "abort_reason.json",
                    {
                        "time_utc": datetime.now(timezone.utc).isoformat(),
                        "reason": "KeyboardInterrupt" if isinstance(e, KeyboardInterrupt) else "CancelledError",
                        "exception_type": type(e).__name__,
                        "loop_idx": self.loop_idx,
                        "step_idx": getattr(self, "step_idx", None),
                        "max_parallel": RD_AGENT_SETTINGS.get_max_parallel(),
                        "traceback": traceback.format_exc(),
                    },
                )

                # Best-effort task_run abort status update.
                try:
                    from rdagent.utils.registry.sqlite_registry import get_registry, should_enable_registry

                    if should_enable_registry():
                        get_registry().upsert_task_run(
                            task_run_id=getattr(self, "task_run_id", ""),
                            scenario=type(self).__name__,
                            status="aborted",
                            log_trace_path=str(getattr(LOG_SETTINGS, "trace_path", "")),
                        )
                except Exception:
                    pass
                raise
            except Exception:
                # Best-effort task_run failure status update.
                try:
                    from rdagent.utils.registry.sqlite_registry import get_registry, should_enable_registry

                    if should_enable_registry():
                        get_registry().upsert_task_run(
                            task_run_id=getattr(self, "task_run_id", ""),
                            scenario=type(self).__name__,
                            status="failed",
                            log_trace_path=str(getattr(LOG_SETTINGS, "trace_path", "")),
                        )
                except Exception:
                    pass
                raise
            finally:
                # cancel all previous tasks before resuming all loops or exit
                for t in tasks:
                    t.cancel()
                self.close_pbar()

    def withdraw_loop(self, loop_idx: int) -> None:
        prev_session_dir = self.session_folder / str(loop_idx - 1)
        prev_path = min(
            (p for p in prev_session_dir.glob("*_*") if p.is_file()),
            key=lambda item: int(item.name.split("_", 1)[0]),
            default=None,
        )
        if prev_path:
            loaded = type(self).load(
                prev_path,
                checkout=True,
                replace_timer=True,
            )
            logger.info(f"Load previous session from {prev_path}")
            # Overwrite current instance state
            self.__dict__ = loaded.__dict__
        else:
            logger.error(f"No previous dump found at {prev_session_dir}, cannot withdraw loop {loop_idx}")
            raise

    def dump(self, path: str | Path) -> None:
        if RD_Agent_TIMER_wrapper.timer.started:
            RD_Agent_TIMER_wrapper.timer.update_remain_time()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def truncate_session_folder(self, li: int, si: int) -> None:
        """
        Clear the session folder by removing all session objects after the given loop index (li) and step index (si).
        """
        # clear session folders after the li
        for sf in self.session_folder.iterdir():
            if sf.is_dir() and int(sf.name) > li:
                for file in sf.iterdir():
                    file.unlink()
                sf.rmdir()

        # clear step session objects in the li
        final_loop_session_folder = self.session_folder / str(li)
        for step_session in final_loop_session_folder.glob("*_*"):
            if step_session.is_file():
                step_id = int(step_session.name.split("_", 1)[0])
                if step_id > si:
                    step_session.unlink()

    @classmethod
    def load(
        cls,
        path: str | Path,
        checkout: bool | Path | str = False,
        replace_timer: bool = True,
    ) -> "LoopBase":
        """
        Load a session from a given path.
        Parameters
        ----------
        path : str | Path
            The path to the session file.
        checkout : bool | Path | str
            If True, the new loop will use the existing folder and clear logs for sessions after the one corresponding to the given path.
            If False, the new loop will use the existing folder but keep the logs for sessions after the one corresponding to the given path.
            If a path (or a str like Path) is provided, the new loop will be saved to that path, leaving the original path unchanged.
        replace_timer : bool
            If a session is loaded, determines whether to replace the timer with session.timer.
            Default is True, which means the session timer will be replaced with the current timer.
            If False, the session timer will not be replaced.
        Returns
        -------
        LoopBase
            An instance of LoopBase with the loaded session.
        """
        path = Path(path)
        # if the path is a directory, load the latest session
        if path.is_dir():
            if path.name != "__session__":
                path = path / "__session__"

            if not path.exists():
                raise FileNotFoundError(f"No session file found in {path}")

            # iterate the dump steps in increasing order
            files = sorted(path.glob("*/*_*"), key=lambda f: (int(f.parent.name), int(f.name.split("_")[0])))
            path = files[-1]
            logger.info(f"Loading latest session from {path}")
        with path.open("rb") as f:
            session = cast(LoopBase, pickle.load(f))

        # set session folder
        if checkout:
            if checkout is True:
                logger.set_storages_path(session.session_folder.parent)
                max_loop = max(session.loop_trace.keys())

                # truncate log storages after the max loop
                session.truncate_session_folder(max_loop, len(session.loop_trace[max_loop]) - 1)
                logger.truncate_storages(session.loop_trace[max_loop][-1].end)
            else:
                checkout = Path(checkout)
                checkout.mkdir(parents=True, exist_ok=True)
                session.session_folder = checkout / "__session__"
                logger.set_storages_path(checkout)

        if session.timer.started:
            if replace_timer:
                RD_Agent_TIMER_wrapper.replace_timer(session.timer)
                RD_Agent_TIMER_wrapper.timer.restart_by_remain_time()
            else:
                # Use the default timer to replace the session timer
                session.timer = RD_Agent_TIMER_wrapper.timer

        return session

    def __getstate__(self) -> dict[str, Any]:
        res = {}
        for k, v in self.__dict__.items():
            if k not in ["queue", "semaphores", "_pbar"]:
                res[k] = v
        return res

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.queue = asyncio.Queue()
        self.semaphores = {}


def kill_subprocesses() -> None:
    """
    Due to the coroutine-based nature of the workflow, the event loop of the main process can't
    stop all the subprocesses start by `curr_loop.run_in_executor`. So we need to kill them manually.
    Otherwise, the subprocesses will keep running in the background and the the main process keeps waiting.
    """
    current_proc = psutil.Process(os.getpid())
    for child in current_proc.children(recursive=True):
        try:
            print(f"Terminating subprocess PID {child.pid} ({child.name()})")
            child.terminate()
        except Exception as ex:
            print(f"Could not terminate subprocess {child.pid}: {ex}")
    print("Finished terminating subprocesses. Then force killing still alive subprocesses.")
    _, alive = psutil.wait_procs(current_proc.children(recursive=True), timeout=3)
    for p in alive:
        try:
            print(f"Killing still alive subprocess PID {p.pid} ({p.name()})")
            p.kill()
        except Exception as ex:
            print(f"Could not kill subprocess {p.pid}: {ex}")
    print("Finished killing subprocesses.")
