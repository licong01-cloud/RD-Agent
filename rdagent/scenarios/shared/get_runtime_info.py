from pathlib import Path

from rdagent.core.experiment import FBWorkspace
from rdagent.utils.env import Env

# 进程级缓存：相同 env 类型的运行时信息只查询一次，避免每次 LLM 调用都创建临时 workspace 目录
_runtime_env_cache: dict[str, str] = {}


def get_runtime_environment_by_env(env: Env) -> str:
    cache_key = type(env).__name__
    if cache_key in _runtime_env_cache:
        return _runtime_env_cache[cache_key]

    implementation = FBWorkspace()
    fname = "runtime_info.py"
    implementation.inject_files(**{fname: (Path(__file__).absolute().resolve().parent / "runtime_info.py").read_text()})
    try:
        stdout = implementation.execute(env=env, entry=f"python {fname}")
    finally:
        implementation.clear()  # 执行完毕立即清理临时 workspace 目录

    _runtime_env_cache[cache_key] = stdout
    return stdout


def check_runtime_environment(env: Env) -> str:
    implementation = FBWorkspace()
    try:
        # 1) Check if strace exists in env
        strace_check = implementation.execute(env=env, entry="which strace || echo MISSING").strip()
        if strace_check.endswith("MISSING"):
            raise RuntimeError("`strace` not found in the target environment.")

        # 2) Check if coverage module works in env
        coverage_check = implementation.execute(env=env, entry="python -m coverage --version || echo MISSING").strip()
        if coverage_check.endswith("MISSING"):
            raise RuntimeError("`coverage` module not found or not runnable in the target environment.")
    finally:
        implementation.clear()  # 无论成功失败都清理临时 workspace 目录
