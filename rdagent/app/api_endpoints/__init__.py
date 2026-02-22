"""
RD-Agent API端点模块

包含以下子模块：
- sota_factors_api: SOTA因子API
- alpha_baseline_api: Alpha基线因子API
- health_check: 健康检查API
- qe_file_sync_api: QE文件同步API
- qe_experiment_setup_api: QE实验工作区设置API
"""

from .sota_factors_api import router as sota_factors_router
from .alpha_baseline_api import router as alpha_baseline_router
from .health_check import router as health_router
from .qe_file_sync_api import router as qe_file_sync_router
from .qe_experiment_setup_api import router as qe_experiment_setup_router

__all__ = [
    "sota_factors_router",
    "alpha_baseline_router",
    "health_router",
    "qe_file_sync_router",
    "qe_experiment_setup_router",
]
