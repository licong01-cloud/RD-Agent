"""Inspect where qlib is imported from and what version it reports.

Usage (in WSL):

    conda activate rdagent-gpu
    python inspect_qlib_env.py
"""

import inspect

import qlib


def main() -> None:
    module_file = inspect.getfile(qlib)
    version = getattr(qlib, "__version__", "<no __version__ attr>")

    print("qlib module file:", module_file)
    print("qlib __version__:", version)


if __name__ == "__main__":
    main()
