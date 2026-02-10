"""
RD-Agent数据提取模块

包含以下子模块：
- sota_factors_extractor: SOTA因子提取
- alpha_baseline_extractor: Alpha基线因子提取
- model_weight_extractor: 模型权重提取
- feature_sequence_extractor: 特征序列提取
"""

from .sota_factors_extractor import SOTAFactorsExtractor, PathCompatUnpickler
from .alpha_baseline_extractor import AlphaBaselineExtractor

__all__ = [
    "SOTAFactorsExtractor",
    "PathCompatUnpickler",
    "AlphaBaselineExtractor",
]
