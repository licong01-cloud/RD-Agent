"""Run RD-Agent loops with TaskConfig and app_tpl overrides."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

REPO_ROOT = Path(__file__).parent.parent

MODE_TO_MODULE = {
    "model_evolve": "rdagent.app.qlib_rd_loop.model",
    "factor_evolve": "rdagent.app.qlib_rd_loop.factor",
    "quant_evolve": "rdagent.app.qlib_rd_loop.quant",
    "model_retrain": "rdagent.app.qlib_rd_loop.model",
    "model": "rdagent.app.qlib_rd_loop.model",
    "factor": "rdagent.app.qlib_rd_loop.factor",
    "quant": "rdagent.app.qlib_rd_loop.quant",
}


def _load_task_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_app_tpl(task_cfg: dict[str, Any]) -> str:
    app_tpl = task_cfg.get("app_tpl")
    if app_tpl:
        return app_tpl
    scenario = task_cfg.get("scenario", "all")
    version = task_cfg.get("version")
    if not version:
        raise ValueError("Missing 'version' in TaskConfig")
    # The value is relative to PROJ_PATH (rdagent/ directory).
    # load_content() computes: PROJ_PATH / app_tpl / relative_path
    # so we need "../app_tpl/{scenario}/{version}/rdagent" to resolve
    # to the correct path: app_tpl/{scenario}/{version}/rdagent/scenarios/...
    return f"../app_tpl/{scenario}/{version}/rdagent"


def _resolve_sota_path(task_cfg: dict[str, Any]) -> str | None:
    raw = task_cfg.get("sota_factor_task_path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def _apply_runtime_env(task_cfg: dict[str, Any], env: dict[str, str]) -> None:
    runtime_env = task_cfg.get("runtime_env") or {}
    for key, value in runtime_env.items():
        env[str(key)] = str(value)


def _build_command(task_cfg: dict[str, Any]) -> list[str]:
    mode = task_cfg.get("mode")
    if not mode:
        raise ValueError("Missing 'mode' in TaskConfig")
    module = MODE_TO_MODULE.get(mode)
    if not module:
        raise ValueError(f"Unsupported mode: {mode}")
    cmd = [sys.executable, "-m", module]
    if task_cfg.get("loop_n") is not None:
        cmd.extend(["--loop_n", str(task_cfg["loop_n"])])
    if task_cfg.get("all_duration"):
        cmd.extend(["--all_duration", str(task_cfg["all_duration"])])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RD-Agent with TaskConfig")
    parser.add_argument("--task-config", required=True, help="Path to TaskConfig JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    if load_dotenv:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    task_cfg = _load_task_config(Path(args.task_config))
    env = os.environ.copy()
    env["RD_AGENT_SETTINGS__APP_TPL"] = _resolve_app_tpl(task_cfg)

    sota_path = _resolve_sota_path(task_cfg)
    if sota_path:
        env["QLIB_SOTA_FACTOR_PATH"] = sota_path

    _apply_runtime_env(task_cfg, env)

    cmd = _build_command(task_cfg)
    print(f"Run command: {' '.join(cmd)}")
    print(f"RD_AGENT_SETTINGS__APP_TPL={env['RD_AGENT_SETTINGS__APP_TPL']}")
    if sota_path:
        print(f"QLIB_SOTA_FACTOR_PATH={sota_path}")

    if args.dry_run:
        return

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
