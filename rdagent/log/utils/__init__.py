import inspect
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict, cast


def _should_colorize() -> bool:
    """检测是否应启用 ANSI 颜色输出。

    当 stdout/stderr 都不是 TTY（如子进程输出重定向到文件）时禁用颜色，
    避免日志文件中充斥大量 ANSI 转义码导致前端显示乱码。
    可通过环境变量 NO_COLOR=1 强制禁用，FORCE_COLOR=1 强制启用。
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


_COLORIZE = _should_colorize()


class LogColors:
    """
    ANSI color codes for use in console output.
    Auto-disabled when output is not a terminal (e.g., redirected to file).
    """

    RED = "\033[91m" if _COLORIZE else ""
    GREEN = "\033[92m" if _COLORIZE else ""
    YELLOW = "\033[93m" if _COLORIZE else ""
    BLUE = "\033[94m" if _COLORIZE else ""
    MAGENTA = "\033[95m" if _COLORIZE else ""
    CYAN = "\033[96m" if _COLORIZE else ""
    WHITE = "\033[97m" if _COLORIZE else ""
    GRAY = "\033[90m" if _COLORIZE else ""
    BLACK = "\033[30m" if _COLORIZE else ""

    BOLD = "\033[1m" if _COLORIZE else ""
    ITALIC = "\033[3m" if _COLORIZE else ""

    END = "\033[0m" if _COLORIZE else ""

    @classmethod
    def get_all_colors(cls: type["LogColors"]) -> list:
        names = dir(cls)
        names = [name for name in names if not name.startswith("__") and not callable(getattr(cls, name))]
        return [getattr(cls, name) for name in names]

    def render(self, text: str, color: str = "", style: str = "") -> str:
        """
        render text by input color and style.
        It's not recommend that input text is already rendered.
        """
        # This method is called too frequently, which is not good.
        colors = self.get_all_colors()
        # Perhaps color and font should be distinguished here.
        if color and color in colors:
            error_message = f"color should be in: {colors} but now is: {color}"
            raise ValueError(error_message)
        if style and style in colors:
            error_message = f"style should be in: {colors} but now is: {style}"
            raise ValueError(error_message)

        text = f"{color}{text}{self.END}"

        return f"{style}{text}{self.END}"

    @staticmethod
    def remove_ansi_codes(s: str) -> str:
        """
        It is for removing ansi ctrl characters in the string(e.g. colored text)
        """
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        return ansi_escape.sub("", s)


class CallerInfo(TypedDict):
    function: str
    line: int
    name: Optional[str]


def get_caller_info(level: int = 2) -> CallerInfo:
    # Get the current stack information
    stack = inspect.stack()
    # The second element is usually the caller's information
    caller_info = stack[level]
    frame = caller_info[0]
    info: CallerInfo = {
        "line": caller_info.lineno,
        "name": frame.f_globals["__name__"],  # Get the module name from the frame's globals
        "function": frame.f_code.co_name,  # Get the caller's function name
    }
    return info


def is_valid_session(log_path: Path) -> bool:
    return log_path.is_dir() and log_path.joinpath("__session__").exists()


def extract_loopid_func_name(tag: str) -> tuple[str, str] | tuple[None, None]:
    """extract loop id and function name from the tag in Message"""
    match = re.search(r"Loop_(\d+)\.([^.]+)", tag)
    return cast(tuple[str, str], match.groups()) if match else (None, None)


def extract_evoid(tag: str) -> str | None:
    """extract evo id from the tag in Message"""
    match = re.search(r"\.evo_loop_(\d+)\.", tag)
    return cast(str, match.group(1)) if match else None


def extract_json(log_content: str) -> dict | None:
    match = re.search(r"\{.*\}", log_content, re.DOTALL)
    if match:
        return cast(dict, json.loads(match.group(0)))
    return None


def gen_datetime(dt: datetime | None = None) -> datetime:
    """
    Generate a datetime object in UTC timezone.
    - If `dt` is None, it will return the current time in UTC.
    - If `dt` is provided, it will convert it to UTC timezone.
    """
    if dt is None:
        return datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc)


def dict_get_with_warning(d: dict, key: str, default: Any = None) -> Any:
    """
    Motivation:
    - When handling the repsonse from the LLM, we may use dict get to get the value.
    - the function prevent falling into default value **silently**.
    - Instead, it will log a warning message.
    """
    from rdagent.log import rdagent_logger as logger

    if key not in d:
        logger.warning(f"Key {key} not found in {d}")
        return default
    return d[key]
