"""RD-Agent shared factor library (rd_factors_lib).

This package is intended to hold reference implementations of factors
that are produced and validated by RD-Agent during evolution.

In Phase 2 it serves as a shared, versioned code base that both
RD-Agent and AIstock can import for research and alignment tests.
Actual execution in production/simulation remains on the AIstock side
(Phase 3), which may re-implement factors on top of its own data
service while keeping numerical behavior aligned with this package.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - best effort version resolution
    __version__ = version("rd_factors_lib")
except PackageNotFoundError:  # local editable install or no package metadata
    try:
        from pathlib import Path

        _ver_file = Path(__file__).with_name("VERSION")
        __version__ = _ver_file.read_text(encoding="utf-8").strip()
    except Exception:
        __version__ = "0.0.0"

__all__: list[str] = []
