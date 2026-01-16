from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# NOTE:
# 这些 schema 仅作为 RD-Agent 内部配置与契约对象的“最小可用版本”，
# 当前阶段不会被外部（AIstock）直接依赖，也不会在 CLI / API 中强制使用。
# 后续在 Phase 3 冻结对外 JSON 协议时，可以在不破坏现有代码的前提下演进字段。


ExperimentKind = Literal["research", "benchmark", "ab_test"]


@dataclass
class FactorRef:
    """引用单个因子的最小结构。

    - name: 因子名（在 factor_catalog 中唯一）；
    - source: 来源标签（rdagent_generated / qlib_alpha158 / external_manual 等）；
    - version: 可选版本号，用于约束因子共享包版本。
    """

    name: str
    source: str | None = None
    version: str | None = None


@dataclass
class FactorSetConfig:
    """因子集合配置。

    Phase 3 顶层设计中仅要求：
    - 列出所用因子列表及来源标签；
    - 为后续扩展预留参数位（如聚合方式、打分规则等）。
    """

    factors: list[FactorRef] = field(default_factory=list)
    # 例如：alpha158 / rdagent_generated / external_manual 等
    tags: list[str] = field(default_factory=list)
    # 预留扩展字段（如加权方式等），不做强约束
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """策略配置（简化版）。

    对齐顶层与 Phase 3 设计中的描述：
    - strategy_type: 策略类型，例如 "topk" / "long_short" / "market_neutral" 等；
    - risk_model: 仓位与风险管理相关的参数；
    - trading_rule: 调仓频率、持仓周期等；
    - benchmark: 对标指数；
    - prompt_template_id: 关联的提示词模板（若有）。
    """

    strategy_type: str
    name: str | None = None
    description: str | None = None

    # 仓位与风险管理配置
    risk_model: dict[str, Any] = field(default_factory=dict)

    # 交易规则（调仓频率、持仓周期、手续费等）
    trading_rule: dict[str, Any] = field(default_factory=dict)

    # 回测相关配置
    benchmark: str | None = None

    # 与上层提示词体系的关联（可选）
    prompt_template_id: str | None = None

    # 预留：用于兼容现有 Qlib YAML 的原始或派生字段
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainTestConfig:
    """训练/验证/测试时间区间与数据配置。

    仅约定字符串形式的日期区间，具体含义由调用方解释：
    - train_start / train_end
    - valid_start / valid_end
    - test_start / test_end

    另预留 dataset_extra，用于映射到 Qlib 的 data_handler_config / dataset_config 等。
    """

    train_start: str | None = None
    train_end: str | None = None
    valid_start: str | None = None
    valid_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    # 与数据集/市场相关的附加配置（market、segments、provider_uri 等）
    dataset_extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveConfig:
    """实验目标与约束（简化版）。"""

    primary_metric: str = "ic"  # 例如 ic / sharpe / ann_ret 等
    direction: Literal["maximize", "minimize"] = "maximize"
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """模型配置。
    
    直接映射到 Qlib 的 model 部分。
    """
    model_type: str | None = None  # 例如 "LGBModel"
    model_class: str | None = None  # Qlib class 完整路径
    kwargs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """数据集配置。
    
    直接映射到 Qlib 的 dataset 部分。
    """
    class_name: str | None = "DatasetH"
    handler_class: str | None = "Alpha158"
    handler_kwargs: dict[str, Any] = field(default_factory=dict)
    segments: dict[str, list[str]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRequest:
    """AIstock → RD-Agent 的实验请求对象（内部 schema 草案）。
    
    当前仅在 RD-Agent 内部使用，用于统一组织：
    - 策略配置（StrategyConfig）；
    - 因子集合配置（FactorSetConfig）；
    - 训练/验证/测试区间与数据配置（TrainTestConfig）；
    - 目标与约束（ObjectiveConfig）；
    - 模型配置（ModelConfig）；
    - 数据集细节配置（DatasetConfig）。

    后续若要对外暴露 JSON 协议，可以对该 dataclass 做 `.to_dict()` 封装，
    并在不破坏现有字段的前提下增加版本号等信息。
    """

    kind: ExperimentKind = "research"
    strategy: StrategyConfig | None = None
    factor_set: FactorSetConfig | None = None
    train_test: TrainTestConfig | None = None
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    model: ModelConfig | None = None
    dataset: DatasetConfig | None = None

    # 元信息：便于在 registry 与 artifacts 中追踪
    name: str | None = None
    description: str | None = None
    user_tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
