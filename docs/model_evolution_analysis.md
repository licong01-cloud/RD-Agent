# 模型演进空间深度分析：RD-Agent × QuantEvolver 双系统协同设计

> 生成日期：2026-02-27
> 基于 RD-Agent 论文（arXiv:2505.14738, arXiv:2505.15155, arXiv:2407.18690）及代码库分析

---

## 一、现状诊断：当前模型演进的瓶颈

### 1.1 RD-Agent 原生模型演进维度

从 `ModelTask` 类定义和 `model_hypothesis_specification` prompt 来看，当前系统定义了以下演进维度：

| 维度 | 字段 | 当前实现 | 实际利用率 |
|------|------|---------|-----------|
| 模型类型 | `model_type` | Tabular / TimeSeries / Graph / XGBoost | **高** — LLM 最常变更此项 |
| 网络架构 | `architecture` | 自由文本描述 | **中** — 描述粗粒度，缺乏结构化引导 |
| 模型超参 | `hyperparameters` | 字典，键值自由 | **低** — LLM 倾向使用默认值 |
| 训练超参 | `training_hyperparameters` | n_epochs/lr/batch_size/weight_decay/early_stop | **低** — 变化范围小 |
| 数学公式 | `formulation` | 可选 LaTeX | **极低** — 多数情况为空 |
| 变量定义 | `variables` | 可选字典 | **极低** — 多数情况为空 |

### 1.2 核心问题

**问题一：演进维度扁平化**
LLM 在生成模型假设时，80%+ 的"创新"集中在"换模型类型"这一个维度（GRU→Transformer→LSTM→MLP 循环），而同一模型类型内部的深度演进严重不足。

**问题二：损失函数和训练策略被锁死**
配置文件中 `loss: mse` 和 `metric: loss` 是硬编码的，LLM 无法演进这两个对量化交易至关重要的维度。

**问题三：因子-模型协同粒度粗**
当前双循环的协同仅体现在"选择做因子还是做模型"的 action selection 层面，缺乏因子特性对模型架构的细粒度指导。

**问题四：反馈信号单一**
模型反馈主要基于 IC/收益率/回撤等终端指标，缺乏对模型内部行为（过拟合程度、梯度健康度、注意力分布）的诊断性反馈。

---

## 二、RD-Agent 侧方案：不修改代码

以下方案完全通过修改 Prompt 模板和 YAML 配置实现，不触碰任何 Python 代码。

### 2.1 增强 `model_hypothesis_specification` Prompt

**目标**：引导 LLM 在更多细粒度维度上进行探索。

**当前 Prompt（8条规则）的问题**：
- 第5条"focus on architecture"过于笼统
- 第7条"adjusting hyperparameters"一笔带过
- 缺乏对损失函数、正则化策略、多尺度设计等维度的明确引导

**建议改写方向**：

```yaml
model_hypothesis_specification: |-
  ## 演进维度清单（每次假设应明确指定要探索的维度）

  ### A. 架构维度（Architecture）
  A1. 注意力机制变体：标准Self-Attention / Sparse / ProbSparse / 频域Attention / Cross-Dimensional
  A2. 位置编码：正弦 / 可学习 / 旋转(RoPE) / 时间感知（融入交易日历）
  A3. 编码器结构：纯Encoder / Encoder-Decoder / Decoder-only
  A4. 多尺度设计：Patch化(PatchTST) / 金字塔结构 / 多分辨率输入
  A5. Embedding层：线性 / 1D-CNN / 因子分组Embedding
  A6. 输出头：单点回归 / 分布预测(均值+方差) / 排序学习 / 多任务头

  ### B. 正则化与泛化维度（Regularization）
  B1. Dropout策略：Attention Dropout / FF Dropout / Stochastic Depth
  B2. 归一化：Pre-LN / Post-LN / RMSNorm
  B3. 非平稳性处理：RevIN / 自适应归一化
  B4. 数据增强：时序Mixup / 随机遮蔽

  ### C. 训练策略维度（Training）
  C1. 超参数组合：lr + scheduler + warmup 联合调整
  C2. 优化器选择：Adam / AdamW / Lion
  C3. 梯度裁剪策略

  ### 规则
  1. 每次假设必须明确标注探索的维度编号（如"本次探索 A1+B2"）
  2. 避免连续3次在同一维度上演进，鼓励跨维度组合
  3. 如果某维度连续2次未带来改善，切换到其他维度
  4. 优先探索 A1/A6/B3 这三个对量化场景影响最大的维度
```

### 2.2 增强 `model_feedback_generation` Prompt

**目标**：让反馈更具诊断性，而非仅报告终端指标。

**建议增加的反馈维度**：

```yaml
model_feedback_generation:
  system: |-
    # 除了对比 IC/收益率/回撤外，还需分析：

    ## 训练行为诊断
    - 训练loss与验证loss的gap：gap > 2x 说明严重过拟合
    - loss下降曲线形态：是否存在震荡、早停是否过早/过晚
    - 训练epoch利用率：如果 early_stop 在第3个epoch就触发，说明lr过大或模型过简单

    ## 预测行为诊断
    - IC 的时间稳定性：IC 均值高但方差大 → 模型对市场regime敏感
    - 多空收益分布：是否存在"只能做多不能做空"的偏向
    - 换手率隐含信息：预测值变化剧烈 → 模型可能在拟合噪声

    ## 改进方向建议
    - 基于以上诊断，给出具体的维度编号建议（参考 A1-C3 维度清单）
    - 如果过拟合严重 → 建议 B1/B2/B3
    - 如果IC不稳定 → 建议 A4/B3（多尺度或非平稳处理）
    - 如果收益率低但IC尚可 → 建议 A6（输出头设计，如排序学习）
```

### 2.3 增强 `model_experiment_output_format`

**目标**：让 LLM 输出更结构化的模型定义。

**建议扩展 JSON Schema**：

```json
{
  "model_name": {
    "description": "...",
    "evolution_dimensions": ["A1", "B2"],  // 新增：明确标注本次探索维度
    "architecture": "...",
    "formulation": "...",
    "variables": {},
    "hyperparameters": {
      "attention_type": "sparse",          // 新增：注意力类型
      "position_encoding": "rotary",       // 新增：位置编码
      "normalization": "pre_ln",           // 新增：归一化方式
      "output_head": "point_regression",   // 新增：输出头类型
      "embedding_strategy": "grouped"      // 新增：Embedding策略
    },
    "training_hyperparameters": {
      "n_epochs": 30,
      "lr": 1e-3,
      "early_stop": 10,
      "batch_size": 256,
      "weight_decay": 1e-4,
      "optimizer": "adamw",                // 新增
      "lr_scheduler": "cosine_annealing",  // 新增
      "warmup_epochs": 3,                  // 新增
      "gradient_clip": 1.0                 // 新增
    },
    "model_type": "TimeSeries"
  }
}
```

### 2.4 增强 RAG 引导策略

**当前 RAG 内容**（`model_proposal.py` 中硬编码的字符串）过于简单。

**建议在 Prompt 中增加领域知识引导**：

```yaml
model_RAG_guidance: |-
  ## 量化时序模型演进知识库

  ### 已验证有效的架构模式
  - PatchTST 在长序列预测中优于标准Transformer（ICLR 2023）
  - RevIN 显著提升非平稳时序的泛化能力（ICLR 2022）
  - 频域注意力(FEDformer)在周期性数据上表现优异（ICML 2022）
  - IC Loss 直接优化比 MSE 在排序任务上更有效

  ### A股市场特有考量
  - 涨跌停机制 → Huber Loss 比 MSE 更鲁棒
  - T+1交易制度 → 预测 T+2 收益比 T+1 更有实际意义
  - 板块轮动效应 → 截面注意力可捕捉行业联动
  - 市场regime切换频繁 → RevIN/自适应归一化是刚需
```

### 2.5 不改代码方案的预期效果

| 改动项 | 工作量 | 预期效果 |
|--------|--------|---------|
| 重写 model_hypothesis_specification | 0.5天 | 演进维度从1-2个扩展到10+个 |
| 增强 model_feedback_generation | 0.5天 | 反馈从"好/坏"变为"哪里不好+怎么改" |
| 扩展 model_experiment_output_format | 0.5天 | 模型定义结构化程度大幅提升 |
| 增强 RAG 引导 | 0.5天 | LLM 获得领域知识加持 |

**局限性**：损失函数、优化器、学习率调度器等维度虽然可以在 Prompt 中引导 LLM 写入模型代码，但 Qlib 的 `conf.yaml` 中 `loss: mse` 仍然是硬编码的，LLM 生成的自定义 loss 无法被 Qlib 的 `GeneralPTNN` 直接使用。

---

## 三、RD-Agent 侧方案：少量代码修改

以下方案需要修改少量 Python 代码（每项 50-200 行），但能解锁关键演进维度。

### 3.1 损失函数可演进化（优先级：★★★★★）

**问题**：`conf_baseline_factors_model.yaml` 第65行 `loss: mse` 硬编码，Qlib 的 `GeneralPTNN` 仅支持有限的内置 loss。

**方案**：在 `model_template/` 中新增 `custom_loss.py` 模板，让 LLM 同时生成模型代码和损失函数代码。

**修改点**：
1. `conf_baseline_factors_model.yaml` — 将 `loss` 字段模板化：`loss: {{ loss_type | default("mse") }}`
2. `ModelTask` 类 — 新增 `loss_function` 字段
3. `model_experiment_output_format` — JSON Schema 增加 `loss_function` 键

**可演进的损失函数空间**：

| 损失函数 | 适用场景 | 预期效果 |
|---------|---------|---------|
| MSE | 基线 | 当前默认 |
| Huber Loss | A股涨跌停场景 | 对极端值更鲁棒 |
| IC Loss | 直接优化排序能力 | IC/RankIC 显著提升 |
| ListMLE | 截面排序学习 | 组合收益提升 |
| Quantile Loss | 风险感知预测 | 尾部风险控制 |
| α·MSE + β·IC_Loss | 多目标平衡 | 兼顾精度和排序 |

### 3.2 优化器和学习率调度可演进化（优先级：★★★★）

**问题**：`GeneralPTNN` 内部默认使用 Adam，学习率调度策略不可配置。

**方案**：扩展 `training_hyperparameters` 的模板渲染逻辑。

**修改点**：
1. `conf_baseline_factors_model.yaml` — 增加 `optimizer` 和 `lr_scheduler` 模板变量
2. 如果 Qlib 的 `GeneralPTNN` 不支持自定义优化器，可以在 `model.py` 模板中让 LLM 自行实现训练循环（绕过 GeneralPTNN）

**可演进空间**：

```
优化器：Adam → AdamW → Lion → Sophia → LAMB
调度器：固定LR → StepLR → CosineAnnealing → OneCycleLR → Warmup+Cosine
梯度策略：无裁剪 → 全局裁剪(1.0) → 自适应裁剪
```

### 3.3 评估指标多维化（优先级：★★★★）

**问题**：当前 `IMPORTANT_METRICS` 列表是硬编码的，反馈只看终端指标。

**方案**：在模型执行后增加训练过程诊断信息的提取。

**修改点**：
1. `feedback.py` — 从训练日志中提取 epoch-level 的 loss 曲线、过拟合指标
2. `prompts.yaml` — 反馈 Prompt 增加训练行为诊断模板
3. 可选：增加 IC 时间序列稳定性分析（IC 的滚动标准差）

**新增诊断指标**：
- `train_val_loss_gap`：训练/验证 loss 差距比
- `early_stop_epoch_ratio`：早停 epoch / 总 epoch 比例
- `ic_stability`：IC 的滚动标准差
- `prediction_turnover`：预测值的日间变化率

### 3.4 模型 Ensemble 支持（优先级：★★★）

**问题**：当前每次实验只能提交一个模型，无法做模型融合。

**方案**：允许 LLM 在一次实验中定义多个子模型 + 融合策略。

**修改点**：
1. `ModelTask` — 支持 `ensemble_strategy` 字段（averaging / stacking / boosting）
2. `model_experiment_output_format` — 支持多模型 JSON 输出
3. 执行模板 — 支持多模型的串行训练和融合预测

### 3.5 少量代码修改方案的优先级排序

```
优先级排序（投入产出比）：

1. 损失函数可演进化     ★★★★★  改动量小，收益极高
2. 评估指标多维化       ★★★★   改善反馈质量，间接提升所有演进
3. 优化器/调度器可演进化 ★★★★   解锁训练策略空间
4. 模型 Ensemble 支持   ★★★    需要较多改动，但上限高
```

---

## 四、QE 侧方案：全面升级模型演进能力

QE 作为自主研发系统，拥有完全的改造自由度。以下方案按架构层次从底到顶展开。

### 4.1 模型演进维度结构化引擎（核心改造）

**现状**：QE 继承 RD-Agent 的 `ModelTask`，模型定义是扁平的 6 字段结构。

**方案**：构建结构化的「演进维度注册表」，将模型演进从"自由文本描述"升级为"维度化搜索"。

**架构设计**：

```python
class ModelEvolutionDimension:
    """单个演进维度的定义"""
    name: str                    # 维度名称
    category: str                # 类别：architecture / regularization / training / output
    current_value: Any           # 当前值
    search_space: List[Any]      # 可选值空间
    priority: float              # 当前优先级（动态调整）
    history: List[Tuple]         # (value, metric_delta) 历史记录

class StructuredModelSpec:
    """结构化模型规格"""
    # 架构维度
    backbone: str                # transformer / gru / lstm / tcn / mlp
    attention_type: str          # standard / sparse / prob_sparse / frequency / cross_dim
    position_encoding: str       # sinusoidal / learnable / rotary / time_aware
    encoder_structure: str       # encoder_only / encoder_decoder / decoder_only
    multi_scale: str             # none / patch / pyramid / multi_resolution
    embedding_strategy: str      # linear / conv1d / grouped_by_factor_type
    output_head: str             # point_regression / distribution / ranking / multi_task

    # 正则化维度
    dropout_strategy: str        # standard / attention_only / stochastic_depth
    normalization: str           # pre_ln / post_ln / rmsnorm
    non_stationarity: str        # none / revin / adaptive_norm
    data_augmentation: str       # none / ts_mixup / random_masking

    # 训练维度
    loss_function: str           # mse / huber / ic_loss / list_mle / quantile / composite
    optimizer: str               # adam / adamw / lion / sophia
    lr_scheduler: str            # fixed / step / cosine / one_cycle / warmup_cosine
    gradient_clip: float         # 梯度裁剪阈值

    # 量化场景维度
    regime_awareness: str        # none / explicit_regime / adaptive
    cross_sectional: str         # none / cs_attention / industry_embedding / graph_fusion
    factor_interaction: str      # none / explicit_interaction / dynamic_selection
```

**关键设计点**：
- 每个维度有明确的搜索空间，LLM 不再"自由发挥"而是"在空间内选择+组合"
- 维度优先级动态调整：连续失败的维度降低优先级，新维度提升优先级
- 历史记录支持 Co-STEER 风格的知识积累

### 4.2 智能维度调度器（Dimension Scheduler）

**现状**：QE 的 Bandit 算法只在"因子 vs 模型"两个臂之间选择，粒度太粗。

**方案**：将 Bandit 扩展为多臂，每个模型演进维度作为一个独立的臂。

**架构设计**：

```
当前 Bandit（2臂）：
  arm_0: factor_evolution
  arm_1: model_evolution

升级后 Bandit（N臂）：
  arm_0: factor_evolution
  arm_1: model.attention_type        ← 模型维度拆分为独立臂
  arm_2: model.loss_function
  arm_3: model.output_head
  arm_4: model.normalization
  arm_5: model.multi_scale
  arm_6: model.training_strategy
  ...
```

**调度策略**：

```python
class DimensionScheduler:
    """多维度 Thompson Sampling 调度器"""

    def __init__(self, dimensions: List[ModelEvolutionDimension]):
        self.bandits = {}
        # 第一层：因子 vs 模型
        self.meta_bandit = ThompsonSampling(arms=["factor", "model"])
        # 第二层：模型内部维度
        self.dim_bandit = ThompsonSampling(
            arms=[d.name for d in dimensions]
        )

    def select_action(self, context: dict) -> Tuple[str, str]:
        """两层决策：先选因子/模型，再选具体维度"""
        action_type = self.meta_bandit.sample(context)
        if action_type == "model":
            dimension = self.dim_bandit.sample(context)
            return ("model", dimension)
        return ("factor", None)

    def update(self, action_type, dimension, reward):
        """根据实验结果更新后验分布"""
        self.meta_bandit.update(action_type, reward)
        if dimension:
            self.dim_bandit.update(dimension, reward)
```

**预期效果**：
- 自动发现"哪个维度的改动对当前因子组合最有效"
- 避免在无效维度上反复浪费实验资源
- 随着因子库变化，模型维度的优先级自动重新排列

### 4.3 因子感知的模型演进（Factor-Aware Model Evolution）

**现状**：QE 的因子演进和模型演进虽然共享 trace，但模型演进时并不"理解"当前因子库的特性。模型只知道 `num_features=N`，不知道这 N 个因子的语义类型和统计特征。

**方案**：在模型假设生成阶段注入因子库的元信息摘要，让模型架构适配因子特性。

**因子元信息摘要结构**：

```python
class FactorPortfolioSummary:
    """当前因子库的元信息摘要，传递给模型演进模块"""

    total_factors: int                          # 因子总数
    factor_type_distribution: Dict[str, int]    # 类型分布
    # 例：{"momentum": 12, "volatility": 8, "volume_price": 15, "fundamental": 5, "ml_based": 10}

    avg_ic: float                               # 因子平均 IC
    ic_dispersion: float                        # IC 离散度（因子间差异大小）
    factor_correlation_matrix_summary: str       # 因子相关性摘要
    # 例："高相关簇：动量类因子间平均相关0.7；低相关：基本面与量价因子相关<0.1"

    temporal_characteristics: Dict[str, str]     # 时序特征
    # 例：{"dominant_frequency": "weekly", "stationarity": "non-stationary", "regime_sensitivity": "high"}

    recent_performance_trend: str                # 近期表现趋势
    # 例："动量因子近3个月IC下降30%，波动率因子IC上升"
```

**因子→模型的适配规则**（注入 Prompt）：

```
因子特性 → 推荐的模型维度调整

1. 因子数量多(>50) + 高相关性
   → 建议：grouped_embedding + attention_dropout + 因子选择机制
   → 原因：冗余因子会稀释有效信号，需要模型内部做特征选择

2. 因子类型多样（动量+基本面+ML因子混合）
   → 建议：grouped_by_factor_type embedding + cross_dim attention
   → 原因：不同类型因子的数值分布和时序模式差异大，统一处理会损失信息

3. 因子IC离散度大（部分因子IC>0.05，部分<0.01）
   → 建议：dynamic_selection output_head + attention可解释性
   → 原因：低IC因子可能在特定regime下有效，需要动态权重

4. 因子时序特征以周频为主
   → 建议：patch_size=5（一周）的PatchTST + time_aware位置编码
   → 原因：patch大小应匹配因子的主频率

5. 因子非平稳性强
   → 建议：revin + adaptive_norm + regime_awareness
   → 原因：因子分布漂移会导致模型失效

6. 近期动量因子衰减
   → 建议：multi_task output_head（同时预测收益+波动率）
   → 原因：动量衰减期通常伴随波动率上升，多任务学习可捕捉这种关联
```

**实现方式**：在 `QlibQuantHypothesisGen` 的 `prepare_context` 中，新增因子摘要的计算和注入。

### 4.4 深度反馈系统（Deep Feedback Engine）

**现状**：QE 的反馈仅包含终端指标对比（IC/收益率/回撤）+ LLM 的文本分析。缺乏对模型训练过程的结构化诊断。

**方案**：构建三层反馈体系，从"结果反馈"升级为"过程+结果+处方反馈"。

**三层反馈架构**：

```
Layer 1: 结果反馈（当前已有）
├── IC / RankIC / ICIR
├── 年化收益率 / 夏普比率
├── 最大回撤 / 信息比率
└── 与 SOTA 的对比

Layer 2: 过程反馈（新增）
├── 训练动态
│   ├── train_loss_curve: List[float]        # 逐epoch训练loss
│   ├── val_loss_curve: List[float]          # 逐epoch验证loss
│   ├── overfit_ratio: float                 # val_loss / train_loss
│   ├── convergence_speed: int               # 达到最优的epoch数
│   └── early_stop_triggered: bool           # 是否触发早停
├── 预测行为
│   ├── prediction_distribution: str         # 预测值分布形态
│   ├── ic_time_series: List[float]          # 逐日IC序列
│   ├── ic_rolling_std: float                # IC滚动标准差（稳定性）
│   ├── long_short_asymmetry: float          # 多空收益不对称度
│   └── turnover_rate: float                 # 预测换手率
└── 模型内部
    ├── param_count: int                     # 参数量
    ├── gradient_norm_avg: float             # 平均梯度范数
    └── attention_entropy: float             # 注意力熵（集中度）

Layer 3: 处方反馈（新增）
├── diagnosed_issues: List[str]              # 诊断出的问题
├── recommended_dimensions: List[str]        # 建议调整的维度
├── recommended_values: Dict[str, Any]       # 建议的具体值
└── confidence: float                        # 建议置信度
```

**处方生成规则引擎**：

```python
class PrescriptionEngine:
    """基于规则+LLM的处方生成器"""

    RULES = {
        "overfit_severe": {
            "condition": lambda fb: fb.overfit_ratio > 2.0,
            "diagnosis": "严重过拟合：验证loss是训练loss的2倍以上",
            "prescription": {
                "dimensions": ["dropout_strategy", "data_augmentation", "weight_decay"],
                "suggestions": {
                    "dropout_strategy": "stochastic_depth",
                    "data_augmentation": "ts_mixup",
                    "weight_decay": "increase_2x"
                }
            }
        },
        "ic_unstable": {
            "condition": lambda fb: fb.ic_rolling_std > 0.03,
            "diagnosis": "IC不稳定：滚动标准差过大，模型对市场regime敏感",
            "prescription": {
                "dimensions": ["non_stationarity", "regime_awareness", "multi_scale"],
                "suggestions": {
                    "non_stationarity": "revin",
                    "regime_awareness": "adaptive"
                }
            }
        },
        "good_ic_low_return": {
            "condition": lambda fb: fb.ic > 0.04 and fb.annual_return < 0.1,
            "diagnosis": "IC尚可但收益低：预测排序能力可以但绝对值偏差大",
            "prescription": {
                "dimensions": ["output_head", "loss_function"],
                "suggestions": {
                    "output_head": "ranking",
                    "loss_function": "list_mle"
                }
            }
        },
        "early_stop_too_early": {
            "condition": lambda fb: fb.convergence_speed < 5,
            "diagnosis": "早停过早：模型可能欠拟合或学习率过大",
            "prescription": {
                "dimensions": ["lr_scheduler", "n_epochs"],
                "suggestions": {
                    "lr_scheduler": "warmup_cosine",
                    "n_epochs": "increase_3x"
                }
            }
        },
        "high_turnover": {
            "condition": lambda fb: fb.turnover_rate > 0.5,
            "diagnosis": "换手率过高：模型可能在拟合噪声而非信号",
            "prescription": {
                "dimensions": ["normalization", "gradient_clip", "loss_function"],
                "suggestions": {
                    "loss_function": "huber",
                    "gradient_clip": 0.5
                }
            }
        }
    }
```

**预期效果**：
- 反馈从"这次不好"变为"这次不好是因为X，建议调整Y维度到Z值"
- LLM 收到的指令更精确，减少无效探索
- 处方规则可持续积累，形成量化交易领域的模型调优知识库

### 4.5 模型知识图谱与 SOTA 模型库

**现状**：QE 有完善的 SOTA 因子库管理机制（`sota_factors_extractor.py`），但模型侧没有对等的知识积累系统。每次模型演进都是"从零开始"或"从上一次的代码开始"。

**方案**：构建与 SOTA 因子库对等的「SOTA 模型库」，记录模型的结构化规格 + 代码 + 性能指纹。

**SOTA 模型库结构**：

```python
@dataclass
class ModelKnowledgeEntry:
    """模型知识条目"""
    # 身份信息
    model_id: str                              # 唯一标识
    model_name: str                            # 模型名称
    source_loop: str                           # 来源 LOOP ID

    # 结构化规格（StructuredModelSpec 的快照）
    spec: StructuredModelSpec

    # 代码
    model_code: str                            # model.py 源码
    code_hash: str                             # 代码哈希（去重用）

    # 性能指纹
    metrics: Dict[str, float]                  # IC/ICIR/收益率/回撤等
    factor_context: FactorPortfolioSummary     # 当时的因子库状态
    training_diagnostics: Dict[str, Any]       # 训练过程诊断数据

    # 适用性标签
    best_for_regime: str                       # 最适合的市场状态
    best_for_factor_type: List[str]            # 最适合的因子类型组合
    weakness: str                              # 已知弱点
```

**模型知识图谱的查询场景**：

```
场景1：新因子加入后，查找最适配的历史模型
  → 按 factor_context 相似度检索
  → 返回在类似因子组合下表现最好的模型规格

场景2：某维度演进失败，查找该维度的历史最优值
  → 按 spec.{dimension} 聚合
  → 返回该维度各取值的平均性能

场景3：市场 regime 切换，查找适配模型
  → 按 best_for_regime 过滤
  → 返回在类似市场状态下表现最好的模型
```

### 4.6 QE 侧方案优先级总览

```
优先级排序：

1. 因子感知模型演进(4.3)    ★★★★★  直接提升因子-模型协同效率，是QE的核心差异化能力
2. 深度反馈系统(4.4)        ★★★★★  从根本上改善演进方向的精准度
3. 维度结构化引擎(4.1)      ★★★★   为所有其他改进提供基础设施
4. 智能维度调度器(4.2)      ★★★★   自动化维度选择，减少无效实验
5. SOTA模型库(4.5)          ★★★    长期价值高，但需要足够的实验积累
```

---

## 五、RD-Agent × QE 双系统协同架构设计

### 5.1 定位分工：互补而非替代

两个系统应承担不同层次的演进职责：

```
┌─────────────────────────────────────────────────────────┐
│                    AIstock 前端/调度层                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────┐   ┌─────────────────────────┐  │
│  │     RD-Agent 层      │   │       QE 层             │  │
│  │  (基础演进引擎)       │   │  (高阶策略引擎)          │  │
│  │                     │   │                         │  │
│  │  职责：              │   │  职责：                   │  │
│  │  · 单因子生成与验证   │   │  · 因子-模型联合优化      │  │
│  │  · 单模型架构搜索     │   │  · 维度化模型演进         │  │
│  │  · CoSTEER知识积累   │   │  · 深度反馈与处方         │  │
│  │  · 代码生成与调试     │   │  · SOTA库管理            │  │
│  │                     │   │  · 跨实验知识迁移         │  │
│  │  输出：              │   │                         │  │
│  │  · 验证通过的因子代码  │   │  输出：                  │  │
│  │  · 可运行的模型代码   │   │  · 最优因子-模型组合      │  │
│  │                     │   │  · 演进策略建议           │  │
│  └────────┬────────────┘   │  · 市场适配方案           │  │
│           │                └────────┬────────────────┘  │
│           │                         │                   │
│           └────────┬────────────────┘                   │
│                    ▼                                    │
│           ┌────────────────┐                            │
│           │  共享知识层      │                            │
│           │  · SOTA因子库   │                            │
│           │  · SOTA模型库   │                            │
│           │  · 演进历史     │                            │
│           └────────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

### 5.2 三级演进策略（Triple-Loop Evolution）

当前的双循环（因子↔模型）可以升级为三级嵌套循环：

```
外循环（QE 策略层）：决定"做什么"
  ├── 分析当前因子库状态和模型性能瓶颈
  ├── 决定本轮是因子演进、模型演进、还是联合调优
  └── 输出：演进计划（包含目标维度和预期改善方向）

中循环（RD-Agent 执行层）：执行"怎么做"
  ├── 接收演进计划
  ├── 生成假设 → 实现代码 → 执行实验 → 收集反馈
  └── 输出：实验结果 + 训练诊断数据

内循环（CoSTEER 代码层）：保证"做得对"
  ├── 代码生成 → 语法检查 → 执行验证
  ├── 失败时自动修复（最多N轮）
  └── 输出：可运行的因子/模型代码
```

**三级循环的协同时序**：

```
Phase 1: 因子探索期（Loop 1-10）
  外循环决策：优先因子演进，模型使用简单基线（GRU/MLP）
  目标：快速积累有效因子，建立因子库基础
  模型维度：固定，不演进

Phase 2: 模型适配期（Loop 11-20）
  外循环决策：因子库稳定后，切换到模型演进
  目标：找到最适配当前因子库的模型架构
  因子维度：固定，使用SOTA因子库
  模型维度：按优先级逐维度探索（A1→A6→B3→C1）

Phase 3: 联合精调期（Loop 21+）
  外循环决策：因子和模型交替微调
  目标：在因子-模型组合空间中寻找局部最优
  策略：每次只调整一个维度，观察对整体的影响
```

---

## 六、最有价值的因子-模型组合策略

基于 RD-Agent(Q) 论文（NeurIPS 2025）的核心发现——联合因子-模型优化以不到 $10 成本实现 2x 年化收益提升——以下是针对 A 股多因子场景的最有价值组合策略分析。

### 6.1 组合策略矩阵

根据因子库特征和市场状态，推荐不同的模型配置：

| 因子库状态 | 市场状态 | 推荐模型配置 | 关键维度设置 |
|-----------|---------|-------------|-------------|
| 少量简单因子(<20) | 趋势市 | GRU + MSE + 标准Dropout | 简单架构，避免过拟合 |
| 少量简单因子(<20) | 震荡市 | MLP + Huber Loss + RevIN | 鲁棒性优先 |
| 中等因子(20-50) | 趋势市 | Transformer + IC Loss + Cosine LR | 充分利用因子交互 |
| 中等因子(20-50) | 震荡市 | PatchTST + Ranking Loss + RevIN | 多尺度+非平稳处理 |
| 大量因子(50+) | 趋势市 | Grouped Embedding Transformer + Dynamic Selection | 因子分组+动态选择 |
| 大量因子(50+) | 震荡市 | Ensemble(Transformer+GRU) + Multi-task Head | 模型融合+多任务 |

### 6.2 三个最高价值的组合方案（详细设计）

#### 方案 A：PatchTST + IC Loss + RevIN（推荐首选）

**适用场景**：中等规模因子库(20-50个)，A股全市场选股

**为什么这个组合价值最高**：
- PatchTST（ICLR 2023）将时间序列切成 patch 后输入 Transformer，计算量降低的同时保留局部模式
- IC Loss 直接优化截面排序能力，与量化选股的核心目标完全对齐
- RevIN 处理 A 股因子的非平稳性（因子分布随市场 regime 漂移）

**具体配置**：

```python
# 模型架构
backbone: "transformer"
attention_type: "standard"          # patch化后序列变短，标准attention即可
position_encoding: "learnable"      # patch内部位置可学习
multi_scale: "patch"                # patch_size=5（一周交易日）
embedding_strategy: "linear"        # patch后直接线性映射
output_head: "point_regression"     # 配合IC Loss

# 正则化
normalization: "pre_ln"
non_stationarity: "revin"           # 关键：处理因子分布漂移
dropout_strategy: "standard"        # dropout=0.1

# 训练
loss_function: "ic_loss"            # 关键：直接优化IC
optimizer: "adamw"
lr_scheduler: "warmup_cosine"       # warmup 3 epochs + cosine decay
gradient_clip: 1.0
n_epochs: 50
batch_size: 512                     # IC Loss需要较大batch计算截面相关
```

**预期效果**：相比当前 GRU+MSE 基线，IC 提升 20-40%，年化收益提升 30-60%

#### 方案 B：Grouped Embedding Transformer + Cross-Dimensional Attention + Ranking Loss

**适用场景**：大规模因子库(50+个)，因子类型多样（动量+波动率+基本面+ML因子混合）

**为什么这个组合价值高**：
- 不同类型因子的数值分布和时序模式差异巨大，统一 embedding 会损失信息
- Cross-Dimensional Attention 让模型学习"哪些因子在当前时刻更重要"
- Ranking Loss 直接优化股票排序，比 MSE 更贴合 TopK 选股策略

**具体配置**：

```python
# 模型架构
backbone: "transformer"
attention_type: "cross_dimensional"   # 时间维度和因子维度分别attention再交叉
position_encoding: "time_aware"       # 融入交易日历（周几、是否月末、是否节前）
multi_scale: "none"                   # 因子分组已提供足够的结构化信息
embedding_strategy: "grouped"         # 按因子类型分组embedding
output_head: "ranking"                # 排序学习头

# 正则化
normalization: "rmsnorm"              # 计算效率高，效果与LayerNorm相当
non_stationarity: "adaptive_norm"     # 自适应归一化
dropout_strategy: "attention_only"    # 只在attention层dropout，保留FF层的表达力

# 训练
loss_function: "list_mle"             # ListMLE排序损失
optimizer: "adamw"
lr_scheduler: "one_cycle"             # OneCycleLR，适合大batch训练
gradient_clip: 0.5                    # 排序损失梯度波动大，需要更严格的裁剪
n_epochs: 80
batch_size: 1024                      # 排序损失需要大batch
weight_decay: 1e-3                    # 因子多，正则化要强
```

**因子分组策略**：

```
Group 1 (动量类): ROC, RESI, momentum_*     → Embedding_dim=32
Group 2 (波动率类): VSTD, STD, RSQR, vol_*  → Embedding_dim=32
Group 3 (量价类): CORR, CORD, WVMA, vp_*    → Embedding_dim=32
Group 4 (基本面类): PE, PB, ROE, fund_*      → Embedding_dim=16
Group 5 (ML因子): ml_factor_*               → Embedding_dim=64
→ 各组独立embedding后拼接，总维度=32*3+16+64=176
```

**预期效果**：在大因子库场景下，相比统一 embedding 的 Transformer，RankIC 提升 15-30%

#### 方案 C：Regime-Adaptive Ensemble + Multi-Task Head + Composite Loss

**适用场景**：全天候策略，需要在牛市/熊市/震荡市都保持稳定表现

**为什么这个组合价值高**：
- A 股 regime 切换频繁（2015股灾、2018贸易战、2020疫情、2024-2025结构性行情），单一模型难以适应所有状态
- Multi-Task Head 同时预测收益率和波动率，在震荡市可以自动降低仓位
- Composite Loss 平衡预测精度和排序能力

**具体配置**：

```python
# 模型架构 — 双分支 Ensemble
branch_1:
    backbone: "transformer"
    attention_type: "sparse"          # 长期趋势捕捉
    multi_scale: "pyramid"            # 日/周/月三级
    output_head: "multi_task"         # 同时预测收益+波动率

branch_2:
    backbone: "gru"
    hidden_size: 128
    num_layers: 2
    output_head: "multi_task"

ensemble:
    strategy: "regime_weighted"        # 根据市场状态动态加权
    regime_detector: "hidden_markov"   # HMM检测市场状态
    # 趋势市：Transformer权重高（捕捉长期模式）
    # 震荡市：GRU权重高（对短期变化更敏感）

# 正则化
normalization: "pre_ln"
non_stationarity: "revin"
dropout_strategy: "stochastic_depth"   # 随机跳层，增强泛化

# 训练
loss_function: "composite"
# composite = 0.4*MSE + 0.3*IC_Loss + 0.2*Volatility_MSE + 0.1*Ranking_Loss
optimizer: "lion"                      # Lion优化器，内存效率高
lr_scheduler: "warmup_cosine"
n_epochs: 100
batch_size: 512
```

**预期效果**：最大回撤降低 30-50%，夏普比率提升 20-40%，全天候适应性显著增强

### 6.3 因子-模型协同演进的关键洞察

基于 RD-Agent(Q) 论文的核心发现，因子和模型不是独立的两个优化问题，而是一个联合优化问题。以下是几个关键洞察：

**洞察 1：因子变化应触发模型维度的定向调整**

```
因子库事件                    → 应触发的模型调整
─────────────────────────────────────────────────
新增大量动量因子              → 增加时序建模深度（更多层/更长回看窗口）
新增基本面因子                → 切换到 Tabular 或混合架构
因子间相关性突然升高          → 启用因子选择机制或增强正则化
SOTA因子IC集体下降            → 启用 RevIN + regime_awareness
ML因子占比超过50%             → 增大 embedding 维度，减少层数（避免过度非线性）
```

**洞察 2：模型架构应反向指导因子生成方向**

```
模型诊断结果                  → 应触发的因子生成方向
─────────────────────────────────────────────────
模型注意力集中在少数因子      → 生成与高注意力因子互补的新因子
模型在特定时间段表现差        → 生成对该时间段有区分力的因子
模型多空不对称                → 生成空头信号更强的因子（如做空动量反转因子）
模型过拟合严重                → 减少因子数量，保留低相关高IC因子
```

**洞察 3：最优组合不是静态的**

论文实验表明，随着演进轮次增加，最优的因子-模型组合会发生变化。这意味着：
- 不应该"找到一个好组合就停下来"
- 应该持续演进，但演进速度可以随着性能趋于稳定而降低
- QE 的 Bandit 机制天然支持这种"探索-利用"平衡

---

## 七、实施路线图

### 7.1 分阶段实施计划

```
阶段 1：Prompt 增强（1-2天，零代码改动）
├── 重写 model_hypothesis_specification，引入维度清单
├── 增强 model_feedback_generation，加入训练行为诊断
├── 扩展 model_experiment_output_format，结构化超参
├── 增加 RAG 领域知识引导
└── 验证：运行 3-5 轮演进，观察 LLM 是否在更多维度上探索

阶段 2：RD-Agent 关键解锁（3-5天，少量代码）
├── 损失函数模板化（conf.yaml + ModelTask 扩展）
├── 评估指标多维化（feedback.py 增加训练诊断提取）
├── 优化器/调度器可配置化
└── 验证：对比 MSE vs IC Loss vs Huber Loss 的演进效果

阶段 3：QE 维度化引擎（1-2周）
├── 实现 StructuredModelSpec 数据结构
├── 实现 DimensionScheduler（多臂Bandit扩展）
├── 实现 FactorPortfolioSummary 因子摘要
├── 将因子摘要注入模型假设生成流程
└── 验证：观察维度调度器是否能自动发现高价值维度

阶段 4：QE 深度反馈（1-2周）
├── 实现三层反馈架构（结果+过程+处方）
├── 实现 PrescriptionEngine 规则引擎
├── 训练诊断数据的自动提取和结构化
└── 验证：对比有/无处方反馈的演进效率

阶段 5：知识积累与长期演进（持续）
├── 实现 SOTA 模型库（ModelKnowledgeEntry）
├── 实现模型知识图谱的检索机制
├── 三级演进策略的完整实现
└── 验证：跨实验的知识迁移效果
```

### 7.2 关键成功度量指标

| 度量维度 | 指标 | 基线值（当前） | 目标值 |
|---------|------|--------------|--------|
| 演进效率 | 达到SOTA所需Loop数 | ~15-20 | <10 |
| 演进多样性 | 每10轮涉及的不同维度数 | 1-2 | 5+ |
| 模型性能 | 最优IC | ~0.04-0.05 | >0.06 |
| 模型性能 | 年化收益率 | ~15-25% | >35% |
| 模型性能 | 最大回撤 | ~20-30% | <15% |
| 稳定性 | IC滚动标准差 | 未追踪 | <0.02 |
| 知识积累 | SOTA模型库条目数 | 0 | 50+/季度 |

---

## 八、总结

### 核心结论

当前 RD-Agent 和 QE 的模型演进系统存在一个根本性问题：**演进的搜索空间远小于实际可用空间**。以 Transformer_TimeSeries_Model 为例，仅注意力机制一个维度就有 5+ 种有意义的变体，但当前系统倾向于在"换模型类型"这一个粗粒度维度上反复尝试。

### 两个系统的最佳分工

- **RD-Agent**：作为基础演进引擎，通过 Prompt 增强（零代码）和关键维度解锁（少量代码）扩大搜索空间。重点是损失函数可演进化和反馈诊断增强。
- **QE**：作为高阶策略引擎，构建维度化搜索框架、因子感知模型演进、深度反馈处方系统。重点是让"因子特性指导模型架构选择"成为自动化流程。

### 最有价值的三个改进方向

1. **损失函数演进**（RD-Agent侧）— 从固定 MSE 到可演进的 IC Loss / Ranking Loss，这是投入产出比最高的单点改进
2. **因子感知模型演进**（QE侧）— 让模型架构自动适配因子库特性，这是 QE 相对于原生 RD-Agent 的核心差异化能力
3. **深度反馈处方系统**（QE侧）— 从"告诉你结果好不好"升级为"告诉你哪里不好、怎么改"，这能从根本上提升每一轮演进的有效性

### 论文方法论的启示

RD-Agent(Q) 论文（NeurIPS 2025）的核心贡献是证明了因子-模型联合优化优于独立优化。本文档的设计方案将这一思想从"是否联合"推进到"如何精细化联合"——通过结构化维度、因子元信息传递、处方反馈等机制，让联合优化的每一步都更加精准和高效。

---

## 九、因子库管理：从"堆积"到"精炼"

### 9.1 现状诊断：组合IC vs 单因子IC

**核心问题**：当前从 RD-Agent 同步到 AIstock 的因子指标是**组合 IC**，而非单因子 IC。

从代码分析可以确认：

- `feedback.py:18-22` 中 `IMPORTANT_METRICS` 只包含整体实验的 IC、年化收益、最大回撤
- `sota_factors_extractor.py:302-338` 中 `_extract_loop_metrics` 提取的是 `loop_exp.result`，这是整个因子组合+模型的回测结果
- `rdagent_factor_catalog_sync.py:246` 中 `ic_val = loop.get("valid_score")` 取的也是模型验证分数

这意味着：**一个 IC=0.05 的 LOOP 中可能包含 3 个因子，但我们无法区分哪个因子贡献了 0.04、哪个贡献了 0.01、哪个甚至是负贡献。**

### 9.2 是否需要单因子回测？

**结论：需要，但不是所有因子都需要，且回测方式应分层。**

**为什么需要**：

1. **因子归因**：组合 IC 无法归因到单个因子，导致无法判断哪些因子真正有价值
2. **去重依据**：两个代码不同但 IC 贡献相似的因子可能是"语义重复"（如 `ROC_5` 和 `Momentum_5D`），只有单因子 IC 才能发现
3. **因子淘汰**：随着因子库增大，必须有量化依据淘汰低效因子，组合 IC 无法提供这个依据
4. **模型适配**：前文提到的"因子感知模型演进"需要知道每个因子的独立贡献

**为什么不是所有因子都需要完整回测**：

单因子完整回测（训练模型+组合回测）成本高。建议分层：

```
Layer 1: 轻量级单因子评估（所有因子，秒级）
├── 单因子 IC / RankIC（截面相关系数，无需训练模型）
├── IC 衰减曲线（IC_1d, IC_2d, IC_5d, IC_10d）
├── 因子自相关性（turnover 指标）
├── 因子覆盖率（非 NaN 比例）
└── 计算方式：纯统计计算，不需要训练任何模型

Layer 2: 因子间关系评估（因子库级别，分钟级）
├── 因子相关性矩阵
├── 因子聚类（层次聚类或 DBSCAN）
├── 增量 IC 贡献（加入该因子后组合 IC 的边际提升）
└── 计算方式：基于 Layer 1 的结果 + 简单回归

Layer 3: 完整单因子回测（仅对 Layer 1/2 筛选后的候选因子）
├── 单因子模型训练 + 回测
├── 完整的收益率/回撤/夏普指标
└── 计算方式：Qlib 完整流程，成本高
```

### 9.3 单因子 IC 评估的实现方案

**在 AIstock 侧实现**（推荐），因为：
- AIstock 已有 Qlib 数据和因子计算结果
- 单因子 IC 是纯统计计算，不需要 RD-Agent 的演进框架
- 可以定期批量执行，不阻塞演进流程

**Layer 1 实现伪代码**：

```python
def evaluate_single_factor(factor_values: pd.Series, label: pd.Series) -> dict:
    """
    单因子轻量级评估
    factor_values: MultiIndex(datetime, instrument) 的因子值
    label: 同结构的未来收益率
    """
    # 1. 截面 IC（每天计算因子值与未来收益的 Spearman 相关）
    daily_ic = factor_values.groupby("datetime").apply(
        lambda x: x.corr(label.loc[x.index], method="spearman")
    )

    # 2. IC 统计量
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0

    # 3. IC 衰减（不同持有期的 IC）
    ic_decay = {}
    for horizon in [1, 2, 5, 10, 20]:
        shifted_label = label.groupby("instrument").shift(-horizon)
        ic_h = factor_values.groupby("datetime").apply(
            lambda x: x.corr(shifted_label.loc[x.index], method="spearman")
        ).mean()
        ic_decay[f"ic_{horizon}d"] = ic_h

    # 4. 因子换手率（日间因子排名变化）
    rank_today = factor_values.groupby("datetime").rank()
    rank_yesterday = rank_today.groupby("instrument").shift(1)
    turnover = (rank_today - rank_yesterday).abs().groupby("datetime").mean().mean()

    # 5. 覆盖率
    coverage = 1 - factor_values.isna().mean()

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_decay": ic_decay,
        "turnover": turnover,
        "coverage": coverage,
        "ic_positive_ratio": (daily_ic > 0).mean(),  # IC为正的天数占比
    }
```

**建议在 `aistock_factor_catalog` 表中新增字段**：

```sql
ALTER TABLE aistock_factor_catalog ADD COLUMN IF NOT EXISTS
    single_factor_ic DOUBLE PRECISION,          -- 单因子 IC
    single_factor_icir DOUBLE PRECISION,        -- 单因子 ICIR
    single_factor_rank_ic DOUBLE PRECISION,     -- 单因子 RankIC
    ic_decay_5d DOUBLE PRECISION,               -- 5日IC衰减
    ic_positive_ratio DOUBLE PRECISION,         -- IC为正天数占比
    factor_turnover DOUBLE PRECISION,           -- 因子换手率
    factor_coverage DOUBLE PRECISION,           -- 因子覆盖率
    single_eval_date TEXT,                      -- 单因子评估日期
    marginal_ic_contribution DOUBLE PRECISION;  -- 边际IC贡献
```

### 9.4 因子去重：从代码哈希到语义去重

**当前已有的去重机制**：

AIstock 的 `rdagent_factor_catalog_sync.py:148-210` 已实现了**阶段1：代码哈希去重**：
- `_normalize_code_for_dedup()` 对代码做文本归一化（去注释、去空行）
- `compute_factor_dedup_hash()` 计算核心计算区域的 SHA256 哈希
- `check_factor_dedup()` 在数据库中查找哈希匹配

**问题**：代码哈希只能发现"代码几乎相同"的重复，无法发现"逻辑等价"或"高度相关"的因子。

例如以下两个因子代码完全不同，但计算结果高度相关（相关系数>0.95）：
```python
# 因子A: 5日动量
factor_a = df['close'] / df['close'].shift(5) - 1

# 因子B: 5日收益率（对数）
factor_b = np.log(df['close']) - np.log(df['close'].shift(5))
```

**建议增加阶段2：统计相关性去重**：

```python
def detect_redundant_factors(
    factor_values_df: pd.DataFrame,  # columns=因子名, index=MultiIndex(datetime, instrument)
    correlation_threshold: float = 0.85,
    ic_values: Dict[str, float] = None,
) -> List[Tuple[str, str, float, str]]:
    """
    检测冗余因子对

    Returns:
        List of (factor_a, factor_b, correlation, recommendation)
        recommendation: "keep_a" / "keep_b" / "merge"
    """
    redundant_pairs = []

    # 计算截面相关性（每天计算因子间相关，取均值）
    dates = factor_values_df.index.get_level_values("datetime").unique()
    corr_matrix = pd.DataFrame(0.0, index=factor_values_df.columns, columns=factor_values_df.columns)

    for dt in dates:
        daily = factor_values_df.loc[dt]
        daily_corr = daily.corr(method="spearman")
        corr_matrix += daily_corr / len(dates)

    # 找出高相关因子对
    for i, fa in enumerate(corr_matrix.columns):
        for j, fb in enumerate(corr_matrix.columns):
            if j <= i:
                continue
            corr = abs(corr_matrix.loc[fa, fb])
            if corr >= correlation_threshold:
                # 保留 IC 更高的那个
                if ic_values:
                    ic_a = abs(ic_values.get(fa, 0))
                    ic_b = abs(ic_values.get(fb, 0))
                    rec = "keep_a" if ic_a >= ic_b else "keep_b"
                else:
                    rec = "merge"
                redundant_pairs.append((fa, fb, corr, rec))

    return redundant_pairs
```

### 9.5 因子库定期维护策略

**结论：应该定期做去重和淘汰，但需要谨慎设计淘汰规则。**

因子库无限增长会带来三个问题：
1. 模型训练时间线性增长（特征数越多越慢）
2. 冗余因子稀释有效信号（噪声因子会干扰模型学习）
3. 维护成本增加（每个因子都需要实时计算和存储）

**建议的因子生命周期管理**：

```
因子状态流转：

  candidate → active → degraded → archived
  (候选)      (活跃)    (衰退)     (归档)

状态判定规则：
┌─────────────┬──────────────────────────────────────────┐
│ 状态         │ 条件                                     │
├─────────────┼──────────────────────────────────────────┤
│ candidate   │ 新生成，尚未完成单因子评估                   │
│ active      │ 单因子IC > 0.02 且 覆盖率 > 80%            │
│ degraded    │ 近3个月滚动IC < 0.01 或 IC正比率 < 45%      │
│ archived    │ 连续6个月degraded 或 被更优因子替代          │
└─────────────┴──────────────────────────────────────────┘
```

**定期维护任务（建议每月执行）**：

```
月度因子库维护流程：

Step 1: 单因子评估刷新
  - 对所有 active 因子重新计算近3个月的滚动IC
  - 对所有 candidate 因子执行首次单因子评估
  - 耗时：~30分钟（50个因子）

Step 2: 冗余检测
  - 计算所有 active 因子间的截面相关性矩阵
  - 标记相关系数 > 0.85 的因子对
  - 在每对中保留IC更高的，另一个标记为 degraded
  - 耗时：~10分钟

Step 3: 边际贡献评估
  - 对每个 active 因子计算"去掉它后组合IC的变化"
  - 边际贡献 < 0.001 的因子标记为 degraded
  - 耗时：~20分钟（需要多次模型训练）

Step 4: 淘汰执行
  - degraded 超过6个月的因子 → archived
  - archived 因子从 combined_factors_df.parquet 中移除
  - 但保留代码和元数据（可随时恢复）

Step 5: 报告生成
  - 输出因子库健康度报告
  - 包含：活跃因子数、新增/淘汰数、平均IC趋势、冗余度
```

### 9.6 因子库管理方案总结

| 问题 | 方案 | 实施位置 | 优先级 |
|------|------|---------|--------|
| 缺少单因子IC | Layer 1 轻量级评估 | AIstock 新增服务 | ★★★★★ |
| 代码去重不够 | 阶段2 统计相关性去重 | AIstock factor_catalog_sync 扩展 | ★★★★ |
| 因子库无限增长 | 生命周期管理 + 月度维护 | AIstock 定时任务 | ★★★★ |
| 边际贡献不明 | 增量IC贡献评估 | AIstock 新增服务 | ★★★ |
| 因子质量不透明 | 因子健康度报告 | AIstock 前端展示 | ★★★ |

---

## 十、RD-Agent SOTA 模型的实际价值评估

### 10.1 现状：同步到 AIstock 的模型有什么？

从 `aistock_model_catalog` 表结构来看，当前同步的模型信息包括：

```
model_id, model_type, model_config, dataset_config,
feature_schema, flattened_feature_list, model_artifacts
```

但关键缺失：**没有模型性能指标字段**（IC、收益率、回撤等）。模型 catalog 只记录了"这个模型是什么"，没有记录"这个模型表现如何"。

### 10.2 RD-Agent 演进出的模型有多小？

从代码分析可以确认，RD-Agent 演进的模型规模非常小：

**模型测试参数**（`model.py:91-97`）：
- `batch_size: 8`
- `num_features: 10-30`（取决于因子数）
- `num_timesteps: 4-40`

**训练配置**（`conf_baseline_factors_model.yaml`）：
- `n_epochs: 30`（从300缩减）
- `batch_size: 256`
- 模型类：`GeneralPTNN`（Qlib 的通用 PyTorch 包装器）

**典型模型参数量估算**：

| 模型类型 | 典型配置 | 参数量 | 对比参考 |
|---------|---------|--------|---------|
| GRU_TimeSeries | hidden=64, layers=2 | ~50K | 极小 |
| LSTM_TimeSeries | hidden=64, layers=2 | ~65K | 极小 |
| Transformer_TimeSeries | d_model=64, heads=4, layers=2 | ~100K | 很小 |
| MLP_Tabular | hidden=[256,128] | ~40K | 极小 |
| 对比：GPT-2 Small | — | 117M | 1000x+ |
| 对比：ViT-Base | — | 86M | 800x+ |

### 10.3 小模型是否还有实际价值？

**结论：有价值，但价值定位需要重新理解。**

**小模型的价值不在于"直接用于生产"，而在于三个方面**：

**价值 1：架构验证（Architecture Validation）**

RD-Agent 演进出的小模型本质上是"架构原型"——它验证了某种架构设计在当前因子组合下是否有效。例如：
- 如果一个 hidden=64 的 GRU 在 30 个因子上 IC=0.045，说明 GRU+这组因子的组合方向是对的
- 将 hidden 扩大到 256、layers 增加到 4，IC 大概率会进一步提升
- 这个"方向验证"的价值远大于模型本身

**价值 2：因子有效性的间接证据**

模型的 IC 实际上是因子组合有效性的证明。即使模型很小，如果 IC 显著高于随机水平，说明因子库中确实包含了有预测力的信号。

**价值 3：快速迭代的基础设施**

小模型训练快（分钟级），这使得 RD-Agent 能在有限时间内完成大量实验。如果每次都训练大模型（小时级），演进效率会下降 10-100 倍。

**但小模型的局限性也很明显**：

| 局限 | 具体表现 | 影响 |
|------|---------|------|
| 容量不足 | 无法充分学习 50+ 因子间的复杂交互 | 大因子库场景下性能受限 |
| 泛化能力弱 | 小模型更容易过拟合或欠拟合 | 实盘表现可能不如回测 |
| 缺乏表达力 | 无法建模高阶非线性关系 | 错过因子间的组合效应 |
| 不适合直接部署 | 参数量太小，推理精度不够 | 需要"放大"后才能用于生产 |

### 10.4 是否需要更大/更复杂的模型？

**结论：需要，但不是在 RD-Agent 演进阶段，而是在 AIstock 生产部署阶段。**

量化交易中模型规模的"甜蜜点"与 NLP/CV 领域完全不同：

**量化模型的规模-收益曲线**：

```
性能
 ↑
 │          ╭──────────── 收益递减区（>5M参数）
 │        ╱
 │      ╱    ← 甜蜜点（500K-5M参数）
 │    ╱
 │  ╱        ← 当前RD-Agent（50K-100K参数）
 │╱
 └──────────────────────→ 模型参数量
```

**为什么量化模型不需要像 LLM 那样大**：
1. 输入维度小：50-100 个因子 vs NLP 的 50K+ 词表
2. 序列长度短：20-60 个交易日 vs NLP 的 4K-128K tokens
3. 信噪比极低：金融数据噪声远大于信号，大模型更容易过拟合噪声
4. 数据量有限：A 股 ~5000 只股票 × ~3000 交易日 ≈ 1500 万样本，远小于 LLM 训练数据

**推荐的量化模型规模范围**：

| 场景 | 因子数 | 推荐参数量 | 典型配置 |
|------|--------|-----------|---------|
| 快速验证（RD-Agent演进） | 20-30 | 50K-200K | GRU hidden=64, layers=2 |
| 标准生产 | 30-50 | 500K-2M | Transformer d=128, heads=8, layers=4 |
| 大因子库生产 | 50-100 | 2M-5M | Transformer d=256, heads=8, layers=6 |
| 上限（再大无意义） | 100+ | 5M-10M | 更大会过拟合 |

### 10.5 解决方案：两阶段模型策略（Explore-then-Exploit）

**核心思路**：RD-Agent 负责"探索"（小模型快速验证架构方向），AIstock 负责"利用"（将验证通过的架构放大到生产规模）。

```
阶段 1: 探索（RD-Agent / QE）
┌─────────────────────────────────────────┐
│  小模型快速演进（50K-200K 参数）          │
│  · 每轮训练 2-5 分钟                     │
│  · 一天可完成 20-50 轮演进                │
│  · 目标：找到最优的架构+因子组合方向       │
│  · 输出：SOTA 架构规格 + 因子组合         │
└──────────────────┬──────────────────────┘
                   │ 架构规格 + 因子列表
                   ▼
阶段 2: 利用（AIstock 生产训练）
┌─────────────────────────────────────────┐
│  大模型生产训练（500K-5M 参数）           │
│  · 将 SOTA 架构按比例放大                 │
│  · hidden_size: 64→256, layers: 2→6     │
│  · 训练 epochs: 30→200                   │
│  · 加入完整的正则化和调度策略              │
│  · 目标：最大化生产环境下的实际表现        │
│  · 输出：可部署的生产模型                 │
└─────────────────────────────────────────┘
```

**模型放大规则（Scale-Up Rules）**：

从 RD-Agent 的小模型到 AIstock 的生产模型，放大不是简单地增大所有参数，而是有针对性的：

```python
def scale_up_model_spec(small_spec: dict, factor_count: int) -> dict:
    """将 RD-Agent 的小模型规格放大到生产规模"""
    prod_spec = small_spec.copy()

    # 1. 宽度放大（最重要）：hidden_size 与因子数成正比
    width_ratio = max(2, factor_count // 15)  # 每15个因子翻一倍宽度
    prod_spec["hidden_size"] = small_spec["hidden_size"] * width_ratio

    # 2. 深度适度增加：层数不宜过多（量化数据信噪比低）
    prod_spec["num_layers"] = min(small_spec["num_layers"] * 2, 8)

    # 3. 注意力头数与宽度匹配
    if "num_heads" in small_spec:
        prod_spec["num_heads"] = prod_spec["hidden_size"] // 32  # 每个头32维

    # 4. 训练配置放大
    prod_spec["n_epochs"] = 200
    prod_spec["batch_size"] = 1024
    prod_spec["early_stop"] = 30

    # 5. 增强正则化（大模型更需要）
    prod_spec["dropout"] = min(small_spec.get("dropout", 0.1) * 1.5, 0.3)
    prod_spec["weight_decay"] = small_spec.get("weight_decay", 1e-4) * 2

    return prod_spec
```

**AIstock 侧生产训练服务设计**：

建议在 AIstock 新增一个 `ProductionModelTrainer` 服务，职责是将 RD-Agent/QE 的 SOTA 架构放大训练：

```
触发条件：
  - RD-Agent 产出新的 SOTA 模型（IC 超过历史最优）
  - 因子库发生重大变化（新增/淘汰因子超过 20%）
  - 定期重训（每月一次）

输入：
  - SOTA 模型的 model.py 代码
  - SOTA 模型的 StructuredModelSpec（架构规格）
  - 当前活跃因子库（经过去重和筛选后的）

流程：
  1. 按 scale_up_model_spec 规则放大模型
  2. 使用完整因子库 + 完整训练集训练
  3. 在验证集上评估，与当前生产模型对比
  4. 如果显著优于当前生产模型 → 替换
  5. 保留历史生产模型（支持回滚）

输出：
  - 生产级模型权重文件
  - 完整的回测报告
  - 与前一版生产模型的对比分析
```

### 10.6 模型管理方案总结

| 问题 | 方案 | 实施位置 | 优先级 |
|------|------|---------|--------|
| 模型 catalog 缺少性能指标 | 同步时写入 IC/收益率/回撤 | AIstock model_catalog_sync 扩展 | ★★★★★ |
| RD-Agent 模型太小无法直接生产 | 两阶段策略：Explore→Exploit | AIstock 新增 ProductionModelTrainer | ★★★★★ |
| 模型演进维度单一（只换类型） | Prompt 增强 + 结构化维度引擎 | RD-Agent prompts.yaml / QE | ★★★★ |
| 缺少模型放大规则 | scale_up_model_spec 标准化 | AIstock 生产训练服务 | ★★★★ |
| 模型与因子库脱节 | 因子库变化触发模型重训 | AIstock 定时任务 | ★★★ |
| 无生产模型版本管理 | 模型版本化 + 回滚机制 | AIstock 模型管理服务 | ★★★ |

---

## 十一、总结与实施建议

### 11.1 核心发现

本文从模型演进空间、因子库管理、模型实际价值、实盘验证闭环、资金规模适配、策略层演进、分钟线回测、模拟盘平台、数据补齐、Coding效率十个维度分析了 RD-Agent + QE + AIstock 三系统协同的现状与改进方向。核心发现如下：

1. **模型演进空间远未充分利用**：RD-Agent 当前的模型演进主要停留在"换模型类型"层面，架构内部（注意力机制、位置编码、正则化）、训练策略（损失函数、优化器、学习率调度）、量化特有维度（非平稳性处理、截面建模）等方向几乎未被探索。通过 Prompt 增强（零代码）和少量配置修改即可显著扩大搜索空间。

2. **因子库缺乏独立评估体系**：当前因子的 IC 指标来自组合实验，无法区分单个因子的真实贡献。代码级去重已有但统计级去重缺失，因子库会随演进轮次无限膨胀。建议建立三层评估体系（轻量单因子IC → 统计去重 → 边际贡献）和生命周期管理机制。

3. **小模型有探索价值但不能直接用于生产**：RD-Agent 的 50K-100K 参数模型适合快速验证架构方向，但生产环境需要 500K-5M 参数级别的模型。两阶段策略（Explore-then-Exploit）是最合理的路径。

4. **系统处于"开环"状态，缺少实盘验证闭环**：当前选股信号产生后没有跟踪实际表现，QE 的演进反馈完全基于回测指标，与真实市场脱节。需要建立"选股→跟踪→评估→反馈回演进"的完整闭环，并通过回测-实盘一致性指标检测过拟合。

5. **演进成果可直接适配小资金场景**：因子和模型的预测能力与资金规模无关，当前所有演进成果（因子库、模型权重、SOTA 配置）可完全复用。仅需调整策略层的 3 个参数（topk、n_drop、account），小资金反而因冲击成本低、可交易小盘股而具有结构性优势。

6. **策略层是被严重低估的第三演进维度**：当前系统的策略参数（换仓周期、止盈、止损、仓位管理）全部硬编码，从未参与演进。但实证表明，仅换仓周期一个参数的优化对净收益的影响就可能超过换一个更好的模型。策略参数空间小、收敛快、实盘一致性高，是投入产出比最高的优化方向之一。

7. **分钟线回测是"锦上添花"而非当务之急**：Qlib 已具备完整的分钟线基础设施（dump_bin、HighFreqHandler、NestedExecutor、TWAP/SBB 策略），技术上可行。但对小资金场景（100-500万），日线回测偏差仅 1-3%，分钟线回测的准确性提升有限，却会让演进速度下降 5-10 倍。更优方案是零成本替代（VWAP 成交价 + 滑点模型 + 周频换仓），可达到 70% 的效果。分钟线回测应在资金规模突破 5000 万或实盘偏差超过 30% 时再引入。

8. **自建轻量模拟盘是打通实盘闭环的最优路径**：相比对接 QMT 模拟盘（开发 15-20 天、维护成本高、统计受限），在 AIstock 侧自建模拟盘仅需 ~9 天、~1060 行代码，且现有基础设施复用率超过 80%（选股服务、miniQMT 行情、InferenceEngine、信号表等零改动复用）。模拟盘可同时运行多个 QE 实验的虚拟组合，横向对比不同因子/模型/策略的实盘效果，并将实盘数据反馈注入 QE 演进循环，实现从"优化回测指标"到"优化真实收益"的根本转变。

9. **Tushare 数据补齐是实盘链路的"地基工程"**：当前 7 个关键 Tushare 数据集中仅 2 个有后端同步方法（daily_basic、stock_moneyflow_ts），其余 5 个完全未实现。现有同步代码无速率控制、无批次管理、无重试机制，且前后端未打通一键补齐流程。通过 DatasetSpec 配置驱动的统一 TushareSyncEngine 设计（~700行代码，~7天工时），可一次性解决所有数据集的补齐问题，并建立可扩展的框架——未来新增数据集仅需添加配置，零代码开发。这是实盘选股和模拟盘的前置条件，必须最先完成。需要特别注意不同数据集采用不同的循环策略（按日期循环 vs 按股票循环 vs 全量单次 vs 分页），统一引擎必须保留这些已验证的模式。

10. **RDAgent Coding 阶段提示词矛盾是当前最紧急的阻断性问题**：实际运行数据显示，6 个 Loop 中 2 个因提示词问题完全失败（有效产出率仅 66.7%），合计浪费 19 轮无效迭代。根因有三：(1) 因子 spec 鼓励 ML 因子但 RAG 禁止 ML，导致 Loop_5 尝试在因子脚本中嵌入 LightGBM 训练；(2) 模型 Critic 做"文字对比"而非"功能验证"，导致 Loop_4 的 10 轮代码全部执行成功却被拒绝；(3) Final Decision 缺乏"执行成功即推定正确"的兜底逻辑。仅需修改 ~40 行提示词文本（零代码改动），即可将有效产出率提升至 90%+，平均 evo 轮数从 5.8 降至 3 以内。这是投入产出比最高的单项优化，应最先实施。

### 11.2 实施优先级路线图

> 详细的 47 项任务评估对比表和 7 阶段实施计划见 **§19 全局开发任务评估与实施计划**。以下为概要版路线图。

按投入产出比排序，建议分七个阶段推进（详见 §19.2）：

**紧急修复（1天，零代码改动，立即见效，阻断性问题）**：
- P0-A：消除因子 spec 中 ML 引导矛盾（`prompts.yaml:106`，~5行）
- P0-B：模型 Critic 从"文字对比"改为"功能验证"（`model_coder/prompts.yaml`，~15行）
- P0-C：Final Decision 增加"执行成功即推定正确"兜底（`model_coder/prompts.yaml`，~8行）
- P1-A：模型架构描述增加精确性约束（`prompts.yaml`，~6行）
- P1-B：模型假设复杂度递进控制（`prompts.yaml`，~6行）
- 预期效果：有效产出率 66.7% → 90%+，平均 evo 轮数 5.8 → <3

**第零阶段（1 周，实盘前置条件）**：
- Tushare 统一补齐引擎 TushareSyncEngine（DatasetSpec 配置驱动，~700行代码）
- 7 个关键数据集一键补齐：daily_basic、stock_moneyflow_ts、adj_factor、bak_basic、stock_basic、stock_st、cyq_chips
- 前端数据看板集成一键补齐按钮 + 批量补齐入口
- ingestion_schedules 扩展 Tushare 每日自动调度（收盘后 16:30 起串行执行）

**第一阶段（1-2 周，零/低代码改动，立即见效）**：
- RD-Agent Prompt 增强：在 `model_hypothesis_specification` 中引导 LLM 探索更多维度
- AIstock 因子 catalog 补充单因子 IC 字段，同步时写入模型性能指标
- AIstock 因子截面相关性去重（扩展现有 code-hash 去重）
- AIstock 新增 `selection_performance` 表，开始记录选股后的多周期表现
- 策略参数从硬编码改为可配置（custom_strategy.py + YAML）
- 回测准确性零成本优化：deal_price 改为 $vwap + 增加滑点模型（~3行配置）

**第二阶段（2-4 周，AIstock 侧新增服务）**：
- 因子生命周期管理服务（candidate→active→degraded→archived）
- 月度因子库维护定时任务
- ProductionModelTrainer 服务（模型放大 + 生产训练）
- SelectionPerformanceTracker 自动跟踪服务（每日收盘后填充 T+1/5/10/20）
- 轻量模拟盘平台 MVP（3张新表 + 每日调度引擎 + 模拟成交，复用选股数据底座）
- 模拟盘多组合横向对比与绩效统计服务
- QEFeedback 扩展 live_performance 字段，注入模拟盘实盘数据
- 策略参数网格搜索服务（每个 SOTA 自动搜索最优策略组合）

**第三阶段（1-2 月，QE 架构升级 + 实盘闭环）**：
- QE 结构化维度引擎（StructuredModelSpec）
- QE 维度调度器（Bandit 选择演进维度）
- QE Researcher Agent 新增 strategy_tune action_type
- 三阶段验证流水线（样本外回测→模拟选股→实盘选股）
- 实盘校准演进任务（LiveCalibrationTask）
- 三系统联动的因子-模型-策略协同演进闭环
- 按需引入分钟线双轨回测（日线快速轨 + 分钟线精确轨，触发条件：资金>5000万或实盘偏差>30%）

---

## 十二、实盘验证与反馈闭环

### 12.1 现状诊断：选股之后发生了什么？

当前系统已经实现了从 QE 演进结果到实盘选股的完整链路：

```
QE 演进 → SOTA 因子+模型 → InferenceEngine 推理 → trading.rdagent_signal → 选股结果展示
```

但链路在"选股结果展示"之后就断了。具体缺失：

| 环节 | 现状 | 问题 |
|------|------|------|
| 选股信号存储 | `trading.rdagent_signal` 记录 symbol, rank, score | 只记录了"选了什么" |
| 选后跟踪 | 无 | 不知道选出的股票后来涨了还是跌了 |
| 实盘表现评估 | 无 | 无法判断因子+模型组合在真实市场中是否有效 |
| 实盘→演进反馈 | 无 | QE 的 Feedback 完全基于回测指标，与真实市场脱节 |
| 回测 vs 实盘对比 | 无 | 无法检测过拟合（回测好但实盘差） |

这意味着整个系统处于"开环"状态——演进优化的目标函数（回测 IC）和最终目标（实盘赚钱）之间没有校验。

### 12.2 选股表现跟踪体系

#### 12.2.1 核心思路

每次选股产生信号后，系统自动在 T+1、T+5、T+20 等多个周期记录选中股票的实际涨跌幅，形成"选股→跟踪→评估"的闭环。

```
选股日 T
  │
  ├─ T+1:  记录次日涨跌幅（短期动量验证）
  ├─ T+5:  记录5日涨跌幅（周度趋势验证）
  ├─ T+10: 记录10日涨跌幅（中期趋势验证）
  └─ T+20: 记录20日涨跌幅（月度收益验证）

同时记录：
  ├─ 基准涨跌幅（沪深300 / 中证500 同期表现）
  └─ 超额收益 = 个股涨跌幅 - 基准涨跌幅
```

#### 12.2.2 数据表设计

建议在 AIstock 新增 `trading.selection_performance` 表：

```sql
CREATE TABLE IF NOT EXISTS trading.selection_performance (
    id                  BIGSERIAL PRIMARY KEY,
    -- 关联信息
    signal_id           BIGINT REFERENCES trading.rdagent_signal(id),
    strategy_id         UUID NOT NULL,
    experiment_id       TEXT,              -- QE实验ID（可选）
    trade_date          DATE NOT NULL,     -- 选股日期
    symbol              TEXT NOT NULL,     -- 股票代码

    -- 选股时的信息
    selection_rank      INTEGER,           -- 选股排名
    selection_score     DOUBLE PRECISION,  -- 模型评分

    -- 多周期实际表现
    ret_1d              DOUBLE PRECISION,  -- T+1 涨跌幅
    ret_5d              DOUBLE PRECISION,  -- T+5 涨跌幅
    ret_10d             DOUBLE PRECISION,  -- T+10 涨跌幅
    ret_20d             DOUBLE PRECISION,  -- T+20 涨跌幅

    -- 基准同期表现
    bench_ret_1d        DOUBLE PRECISION,  -- 基准 T+1
    bench_ret_5d        DOUBLE PRECISION,  -- 基准 T+5
    bench_ret_10d       DOUBLE PRECISION,  -- 基准 T+10
    bench_ret_20d       DOUBLE PRECISION,  -- 基准 T+20

    -- 超额收益
    excess_ret_1d       DOUBLE PRECISION,  -- 超额 T+1
    excess_ret_5d       DOUBLE PRECISION,  -- 超额 T+5
    excess_ret_10d      DOUBLE PRECISION,  -- 超额 T+10
    excess_ret_20d      DOUBLE PRECISION,  -- 超额 T+20

    -- 元数据
    benchmark_code      TEXT DEFAULT '000300.SH',  -- 基准指数
    last_filled_horizon TEXT,              -- 最后填充的周期
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (strategy_id, trade_date, symbol)
);
```

#### 12.2.3 自动跟踪服务设计

建议新增 `SelectionPerformanceTracker` 定时任务，每个交易日收盘后自动填充历史选股的实际表现：

```
每日收盘后执行（建议 16:00）：

Step 1: 扫描待填充记录
  - 查询 selection_performance 中 last_filled_horizon 不完整的记录
  - 按 trade_date 分组，确定哪些周期已到期可填充

Step 2: 批量获取行情
  - 从 TimescaleDB 获取相关股票的收盘价
  - 从 TimescaleDB 获取基准指数的收盘价
  - 计算各周期涨跌幅和超额收益

Step 3: 回写数据库
  - UPDATE selection_performance SET ret_1d = ..., bench_ret_1d = ...
  - 更新 last_filled_horizon 和 updated_at

Step 4: 异常检测
  - 标记超额收益显著为负的选股批次（连续3次 T+5 超额 < -2%）
  - 触发告警通知
```

**填充时机示意**：

```
选股日 T=2月10日，选了 50 只股票

2月11日 16:00 → 填充 ret_1d（T+1 已到期）
2月17日 16:00 → 填充 ret_5d（T+5 已到期，跳过非交易日）
2月24日 16:00 → 填充 ret_10d（T+10 已到期）
3月10日 16:00 → 填充 ret_20d（T+20 已到期，该记录完成）
```

### 12.3 实盘评估指标体系

回测指标（IC、年化收益）和实盘指标的侧重点不同。实盘需要关注的核心维度：

#### 12.3.1 选股胜率指标

```
单次选股评估（每个 trade_date 一组）：

TopK 胜率 = 选中股票中涨幅 > 0 的比例
TopK 超额胜率 = 选中股票中超额收益 > 0 的比例
TopK 平均超额 = mean(选中股票超额收益)
TopK vs BottomK = TopK平均收益 - BottomK平均收益（多空价差）

分层评估（按 rank 分组）：
  Top10 平均超额  ← 最核心，实际持仓通常集中在前10
  Top20 平均超额
  Top50 平均超额
```

#### 12.3.2 时间序列稳定性指标

单次选股的胜率波动很大，需要在时间维度上评估稳定性：

```
滚动评估（最近 N 次选股的汇总）：

滚动胜率 = 最近20次选股中，TopK超额胜率 > 50% 的次数 / 20
连胜/连败 = 最长连续超额为正/负的次数
收益衰减率 = 近10次平均超额 / 前10次平均超额（< 1 说明在衰减）
最大单次亏损 = 所有选股批次中最差的 TopK 平均超额
```

#### 12.3.3 回测-实盘一致性指标（最关键）

这是检测过拟合的核心指标：

```
一致性评估：

IC 衰减比 = 实盘滚动IC / 回测IC
  · > 0.7: 一致性好，模型泛化能力强
  · 0.5-0.7: 有一定衰减，需关注
  · < 0.5: 严重过拟合，模型不可信

收益衰减比 = 实盘年化超额 / 回测年化超额
  · 量化实践中，实盘收益通常是回测的 30%-60%
  · 如果 > 80%，可能回测本身就比较保守（好事）
  · 如果 < 20%，模型严重过拟合

排名一致性 = Spearman(回测评分排名, 实盘涨幅排名)
  · 衡量模型评分与实际涨幅的排序相关性
  · > 0.05 即有统计意义（金融数据信噪比低）
```

### 12.4 实盘反馈接入 QE 演进循环

这是整个方案最有价值的部分：将实盘表现作为信号反馈到 QE 的演进决策中，形成真正的闭环。

#### 12.4.1 当前 QE 反馈机制的局限

当前 `QEFeedbackService` 的反馈来源：

```python
# qe_feedback_service.py 现状
results["signal_quality"] = {"IC": ..., "ICIR": ...}       # ← 来自 Qlib 回测
results["performance"] = {"annualized_return": ..., ...}    # ← 来自 Qlib 回测
```

所有指标都来自 Qlib 的历史回测，存在两个根本问题：
1. **回测偏差**：回测使用未来信息（look-ahead bias）、忽略冲击成本、假设完美执行
2. **过拟合风险**：QE 演进的目标函数就是回测 IC，多轮演进后模型会逐渐过拟合回测数据

#### 12.4.2 双信号反馈架构

核心思路：在 QE 的 Feedback 中同时注入回测指标和实盘指标，让 LLM Agent 综合判断演进方向。

```
                    ┌──────────────────────────┐
                    │   QE Researcher Agent     │
                    │   决定下一轮演进方向        │
                    └─────────┬────────────────┘
                              │ 输入
                    ┌─────────▼────────────────┐
                    │   Enhanced Feedback       │
                    │                           │
                    │  ┌─────────────────────┐  │
                    │  │ 回测指标（即时）      │  │
                    │  │ · IC / ICIR          │  │
                    │  │ · 年化收益 / 回撤     │  │
                    │  └─────────────────────┘  │
                    │           +                │
                    │  ┌─────────────────────┐  │
                    │  │ 实盘指标（延迟）      │  │
                    │  │ · TopK 超额收益      │  │
                    │  │ · 选股胜率           │  │
                    │  │ · 回测-实盘一致性     │  │
                    │  └─────────────────────┘  │
                    │           +                │
                    │  ┌─────────────────────┐  │
                    │  │ 历史实盘汇总         │  │
                    │  │ · 各SOTA的实盘表现   │  │
                    │  │ · 衰减趋势           │  │
                    │  └─────────────────────┘  │
                    └──────────────────────────┘
```

**关键设计决策**：实盘数据有延迟（至少 T+1），不能作为每轮演进的即时反馈，而应作为"背景知识"注入。

#### 12.4.3 QE Feedback 扩展方案

在现有 `QEFeedbackService` 基础上扩展，新增实盘表现数据源：

```python
# qe_feedback_service.py 扩展设计

class QEFeedbackService:

    def generate_feedback(self, experiment_id, experiment_dir, ...):
        # ... 现有逻辑 ...

        # 新增：获取实盘表现数据
        live_performance = self._get_live_performance_summary(experiment_id)

        # 新增：获取历史SOTA的实盘表现对比
        sota_live_comparison = self._get_sota_live_comparison()

        # 将实盘数据注入反馈
        feedback.live_performance = live_performance
        feedback.sota_live_comparison = sota_live_comparison
        feedback.backtest_live_consistency = self._calc_consistency(
            backtest_metrics=results,
            live_metrics=live_performance,
        )

        return feedback

    def _get_live_performance_summary(self, experiment_id):
        """汇总该实验/策略的实盘选股表现"""
        # 从 selection_performance 表聚合
        return {
            "total_selections": 15,          # 总选股次数
            "avg_excess_ret_5d": 0.008,      # 平均5日超额
            "win_rate_5d": 0.62,             # 5日超额胜率
            "avg_excess_ret_20d": 0.015,     # 平均20日超额
            "win_rate_20d": 0.58,            # 20日超额胜率
            "recent_trend": "stable",        # 近期趋势
            "max_single_loss_5d": -0.035,    # 最大单次5日亏损
        }
```

#### 12.4.4 实盘数据如何影响演进决策

实盘反馈不是简单地替代回测指标，而是提供三种关键信号：

**信号 1：过拟合预警**

当回测 IC 持续提升但实盘超额持续下降时，说明演进方向在过拟合。此时 Researcher Agent 应收到明确指令：

```
⚠️ 过拟合预警：
  回测IC趋势: 0.038 → 0.042 → 0.045（持续上升）
  实盘5日超额: 1.2% → 0.6% → -0.3%（持续下降）
  一致性评分: 0.35（低于安全阈值0.5）

建议：
  - 停止在当前方向继续深入优化
  - 增加正则化强度（dropout, weight_decay）
  - 尝试更简单的模型架构
  - 减少因子数量，避免过度拟合噪声
```

**信号 2：有效方向确认**

当某个演进方向在回测和实盘都表现良好时，给予正向强化：

```
✅ 方向确认：
  LOOP 5 的 action_type=factor_adjust（增加动量因子）
  回测IC: 0.035 → 0.041（+17%）
  实盘5日超额: 0.8% → 1.5%（+87%）

建议：继续在动量因子方向深入探索
```

**信号 3：市场环境适配**

实盘数据天然包含市场环境信息，可以引导演进方向适配当前市场：

```
📊 市场环境信号：
  近20个交易日市场特征：
  - 沪深300涨幅: -5.2%（下跌市）
  - 市场波动率: 22%（高波动）
  - 行业轮动速度: 快

  当前SOTA在此环境下的表现：
  - TopK超额: -1.8%（跑输基准）

建议：
  - 当前模型可能不适应高波动下跌市
  - 考虑增加防御性因子（低波动、高股息）
  - 考虑缩短持仓周期
```

### 12.5 分阶段验证策略：从纸面到实盘

量化交易最佳实践中，策略上线遵循严格的分阶段验证流程。建议将 QE 演进结果的验证分为四个阶段：

```
阶段 0          阶段 1           阶段 2           阶段 3
QE回测验证  →  样本外回测  →   模拟选股跟踪  →  实盘选股
(现有)        (新增)          (新增)           (现有,增强)
```

#### 阶段 1：样本外回测（Out-of-Sample Backtest）

QE 当前的回测使用 Qlib 的固定数据分割（Train 2010-2018, Valid 2019-2020, Test 2021-2025），但演进多轮后模型可能间接"看到"了测试集的模式。

```
改进方案：滚动样本外验证

当 QE 产出新的 SOTA 时，自动触发：
  1. 使用 SOTA 的因子+模型配置
  2. 在最近 6 个月的纯样本外数据上回测
     （这段数据不在 Qlib 的 Train/Valid/Test 中）
  3. 计算样本外 IC 和收益指标
  4. 与 Qlib 回测指标对比，计算衰减比

通过条件：
  · 样本外 IC > 回测 IC × 0.6
  · 样本外年化超额 > 0
  · 样本外最大回撤 < 回测最大回撤 × 1.5
```

#### 阶段 2：模拟选股跟踪（Paper Trading）

通过阶段 1 的 SOTA 配置，进入模拟选股阶段。每个交易日自动执行选股但不实际交易，仅记录和跟踪：

```
每日自动执行：
  1. 收盘后使用 SOTA 配置调用 InferenceEngine
  2. 生成 Top50 选股信号，写入 trading.rdagent_signal
  3. 标记 output_mode = 'paper_trading'（区别于实盘信号）
  4. SelectionPerformanceTracker 自动跟踪后续表现

观察期：至少 20 个交易日（约1个月）

晋级条件（全部满足才可进入阶段 3）：
  · 20日累计 TopK 超额胜率 > 55%
  · 20日平均 Top10 超额 > 0.3%（5日周期）
  · 无连续 5 次 TopK 超额为负
  · 回测-实盘一致性 > 0.5
```

#### 阶段 3：实盘选股（增强版）

通过模拟验证的配置正式进入选股中心（`in_selection_center = TRUE`），在现有选股流程基础上增强：

```
增强点 1：多配置并行选股
  - 同时运行 2-3 个通过验证的 SOTA 配置
  - 取交集（多个配置都选中的股票）作为高置信度推荐
  - 取并集按加权评分排序作为完整推荐列表

增强点 2：动态置信度标注
  - 每只选中股票标注置信度等级：
    · 高：3个配置都选中，且历史同类选股胜率 > 65%
    · 中：2个配置选中，或历史胜率 55%-65%
    · 低：仅1个配置选中，或历史胜率 < 55%

增强点 3：持续监控与自动降级
  - 如果实盘连续 10 次选股 TopK 超额为负 → 自动降级回阶段 2
  - 如果回测-实盘一致性降至 0.3 以下 → 触发告警 + 暂停选股
  - 降级后 QE 收到"实盘失效"信号，触发新一轮演进
```

### 12.6 实盘驱动的演进方向调整

将实盘表现反馈到 QE 演进中，不是简单地把实盘指标加到 Feedback 里，而是要设计合理的反馈机制，避免"追涨杀跌"式的短视优化。

#### 12.6.1 反馈频率与粒度

```
实盘反馈不应每天都影响演进方向，建议按以下频率：

日频（仅记录，不触发演进）：
  · 记录每日选股的 T+1 表现
  · 更新 selection_performance 表
  · 计算滚动指标

周频（轻量反馈，注入 Agent 上下文）：
  · 汇总本周选股表现
  · 更新 QE Feedback 中的 live_performance 字段
  · Researcher Agent 在决策时可参考，但不强制改变方向

月频（重度反馈，可能触发方向调整）：
  · 全面评估过去一个月的实盘表现
  · 计算回测-实盘一致性
  · 如果一致性 < 0.5，触发"实盘校准"演进任务
  · 如果某类因子在实盘中持续失效，标记为"实盘降级"
```

#### 12.6.2 实盘校准演进任务

当月度评估发现回测-实盘一致性严重偏离时，自动触发一种特殊的 QE 演进任务——"实盘校准任务"：

```
实盘校准任务（LiveCalibrationTask）

触发条件（任一满足）：
  · 回测-实盘 IC 衰减比 < 0.4（连续2个月）
  · 实盘 TopK 超额连续 4 周为负
  · 新 SOTA 在模拟选股阶段未通过晋级条件

任务目标：
  不是追求更高的回测 IC，而是追求更好的回测-实盘一致性

Researcher Agent 收到的特殊指令：
  "当前模型存在严重的回测-实盘偏差。请优先考虑以下方向：
   1. 简化模型（减少参数量、减少因子数）
   2. 增强正则化（更高的 dropout、weight_decay）
   3. 使用更稳健的因子（低换手率因子、基本面因子）
   4. 缩短回测训练窗口（使用更近期的数据）
   5. 避免使用高频技术指标类因子（容易过拟合）"
```

#### 12.6.3 完整闭环架构

将以上所有环节串联，形成从演进到实盘再回到演进的完整闭环：

```
┌─────────────────────────────────────────────────────────────────┐
│                    RD-Agent + QE 演进层                          │
│                                                                  │
│  RD-Agent ──SOTA因子/模型──→ QE 演进循环                         │
│                                  │                               │
│                    ┌─────────────▼──────────────┐               │
│                    │  Researcher Agent 决策      │               │
│                    │  输入：回测指标 + 实盘反馈   │               │
│                    └─────────────┬──────────────┘               │
│                                  │ 新配置                        │
│                    ┌─────────────▼──────────────┐               │
│                    │  Qlib 回测 → 回测指标       │               │
│                    └─────────────┬──────────────┘               │
│                                  │ SOTA                          │
└──────────────────────────────────┼──────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────┐
│                    AIstock 验证层                                 │
│                                                                  │
│  阶段1: 样本外回测 ──通过──→ 阶段2: 模拟选股                     │
│                                  │                               │
│                           20个交易日观察                          │
│                                  │                               │
│                    ┌─────────────▼──────────────┐               │
│                    │  SelectionPerformanceTracker│               │
│                    │  每日跟踪 T+1/5/10/20 表现  │               │
│                    └─────────────┬──────────────┘               │
│                                  │ 通过晋级条件                   │
│                    ┌─────────────▼──────────────┐               │
│                    │  阶段3: 实盘选股            │               │
│                    │  选股中心 + 持续监控         │               │
│                    └─────────────┬──────────────┘               │
│                                  │                               │
└──────────────────────────────────┼──────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  实盘表现汇总                 │
                    │  · 周频轻量反馈              │
                    │  · 月频重度评估              │
                    │  · 过拟合预警                │
                    └──────────────┬──────────────┘
                                   │ 反馈回 QE
                                   ▼
                         Researcher Agent
```

### 12.7 量化交易最佳实践参考

以下建议来自量化交易行业的成熟经验，适用于 QE + AIstock 的实盘验证场景：

#### 12.7.1 避免常见陷阱

```
陷阱 1：用实盘短期表现直接优化模型
  ❌ 错误：实盘跑了3天亏钱 → 立刻换模型
  ✅ 正确：至少观察20个交易日，用统计显著性判断

陷阱 2：追逐近期热门因子
  ❌ 错误：动量因子最近2周表现好 → 全仓动量
  ✅ 正确：因子有效性需要跨周期验证，至少覆盖牛熊转换

陷阱 3：忽略交易成本
  ❌ 错误：TopK 超额 0.5%/周 看起来不错
  ✅ 正确：扣除双边千三手续费+滑点后可能为负
  建议：所有超额收益计算都扣除 0.3% 的单边交易成本

陷阱 4：幸存者偏差
  ❌ 错误：只看当前 SOTA 的实盘表现
  ✅ 正确：记录所有历史 SOTA 的实盘表现，分析整体分布
```

#### 12.7.2 推荐的评估周期与持仓策略

```
评估周期选择（取决于换仓频率）：

日频换仓（不推荐，交易成本高）：
  · 评估周期：T+1
  · 适用：超短线动量策略
  · 年化换手率 > 200 倍，成本极高

周频换仓（推荐起步方案）：
  · 评估周期：T+5
  · 每周一开盘前生成选股信号
  · 持仓一周后换仓
  · 年化换手率 ~50 倍，成本可控

双周/月频换仓（推荐稳健方案）：
  · 评估周期：T+10 或 T+20
  · 降低交易成本，更适合中低频因子
  · 年化换手率 ~12-25 倍

实际建议：
  · 初期用周频换仓积累数据（数据点多，统计更快收敛）
  · 稳定后切换到双周频（降低成本，提高净收益）
  · TopK 设为 20-30 只（太少波动大，太多稀释超额）
```

### 12.8 实盘验证方案总结

| 问题 | 方案 | 实施位置 | 优先级 |
|------|------|---------|--------|
| 选股后无跟踪 | `selection_performance` 表 + 自动跟踪服务 | AIstock 新增表+定时任务 | ★★★★★ |
| 无多周期评估 | T+1/5/10/20 多周期涨跌幅+超额收益 | AIstock SelectionPerformanceTracker | ★★★★★ |
| 回测-实盘脱节 | IC衰减比 + 收益衰减比 + 排名一致性 | AIstock 月度评估服务 | ★★★★★ |
| QE 反馈无实盘数据 | QEFeedback 扩展 live_performance 字段 | AIstock qe_feedback_service 扩展 | ★★★★ |
| 无过拟合检测 | 回测↑实盘↓ 自动预警 + 实盘校准任务 | QE 新增 LiveCalibrationTask | ★★★★ |
| 无分阶段验证 | 样本外回测→模拟选股→实盘选股 三阶段 | AIstock 验证流水线 | ★★★ |
| 无自动降级机制 | 连续失效自动降级回模拟阶段 | AIstock 选股中心增强 | ★★★ |
| 无市场环境适配 | 实盘数据注入市场环境信号 | QE Researcher Agent 上下文 | ★★ |

---

## 十三、小资金适配：从1亿到100-500万

### 13.1 现状：当前回测假设了什么？

当前 RD-Agent 和 QE 的回测配置统一使用以下参数：

```
account:        100,000,000（1亿元）
topk:           50（持仓50只股票）
n_drop:         5（每日最多换5只）
open_cost:      0.05%（开仓手续费）
close_cost:     0.15%（平仓手续费，含印花税）
min_cost:       5元
lot_size:       100股
max_position:   90%（最大仓位比例）
```

这意味着：
- 每只股票平均仓位 ≈ 1亿 × 90% ÷ 50 = **180万元**
- 每日换手金额 ≈ 180万 × 5只 × 2（买卖） = **1800万元**
- 单只股票占总资金 ≈ **1.8%**

### 13.2 不同资金规模下的参数可行性

| 资金规模 | TopK=50 每只仓位 | 可行性 | 核心问题 |
|---------|-----------------|--------|---------|
| 1亿 | 180万 | ✅ 完全可行 | 大盘股流动性充足 |
| 5000万 | 90万 | ✅ 可行 | 部分小盘股需注意流动性 |
| 1000万 | 18万 | ⚠️ 勉强 | 持仓太分散，交易成本占比上升 |
| 500万 | 9万 | ❌ 不可行 | 很多股票一手就要数千元，50只太分散 |
| 100万 | 1.8万 | ❌ 完全不可行 | 多数股票一手都买不起 |

**结论**：TopK=50 的策略配置在 500 万以下资金完全不适用，需要调整。

### 13.3 逐层分析：演进成果的可复用性

#### 13.3.1 因子层：完全可复用 ✅

IC（Information Coefficient）衡量的是模型预测值与实际收益的排序相关性，本质上是一个统计量，与资金规模无关。

```
因子 Alpha_001 的 IC = 0.04

含义：该因子对全市场股票未来收益的排序预测能力为 0.04
  · 用1亿交易 → IC 还是 0.04
  · 用100万交易 → IC 还是 0.04

原因：因子计算的是每只股票的特征值，不涉及资金分配
```

**唯一例外**：如果因子依赖大单资金流、机构持仓等与资金量相关的特征，在小资金场景下这些因子的含义不变（你是在预测市场，不是在影响市场）。

#### 13.3.2 模型层：完全可复用 ✅

模型学习的是"因子值 → 未来收益排序"的映射关系，这个映射与交易资金量无关。

```
Transformer_TimeSeries_Model 学到的模式：
  "当动量因子 > 0.5 且波动率因子 < 0.3 时，该股票未来5日收益大概率排名靠前"

这个模式不会因为你用100万还是1亿交易而改变。
```

**结论**：RD-Agent 和 QE 演进出的所有 SOTA 模型可以直接用于小资金场景。

#### 13.3.3 策略层：需要调整，但改动很小 ⚠️

策略层是唯一需要适配的部分，核心是调整 TopK 和相关参数：

```
小资金推荐参数：

资金 500万：
  topk: 15-20      （每只仓位 22-30万，足够买入大多数股票）
  n_drop: 2-3      （每日换手比例与大资金保持一致 ~10-15%）
  max_position: 0.85（略降，留更多现金应对波动）

资金 200万：
  topk: 10-15      （每只仓位 12-18万）
  n_drop: 1-2
  max_position: 0.80

资金 100万：
  topk: 8-10       （每只仓位 8-11万，刚好覆盖大多数股票）
  n_drop: 1
  max_position: 0.80
```

#### 13.3.4 交易成本层：小资金反而有优势 ✅

当前回测的交易成本假设（open_cost=0.05%, close_cost=0.15%）对小资金同样适用，甚至偏保守：

```
交易成本对比：

大资金（1亿）：
  · 手续费：万0.5 开仓 + 万1.5 平仓（含印花税）→ 合理
  · 冲击成本：买入180万某只股票可能推高价格 0.1-0.3%
  · 实际总成本 ≈ 0.3-0.5%（单边）

小资金（200万）：
  · 手续费：同上 → 合理
  · 冲击成本：买入15万某只股票，对盘口几乎无影响 → ≈ 0%
  · 实际总成本 ≈ 0.15-0.2%（单边）

结论：小资金的实际交易成本更低，回测中的成本假设偏保守
      → 实盘表现可能优于回测（这是好事）
```

#### 13.3.5 股票池层：小资金有独特优势 ✅

这是小资金最大的结构性优势：

```
大资金（1亿）的限制：
  · 每只股票买入180万，日均成交额需 > 1800万（占比 < 10%）
  · 实际可交易池：沪深300 + 中证500 ≈ 800只大中盘股
  · 这些股票被机构充分研究，alpha 空间有限

小资金（200万）的优势：
  · 每只股票买入15万，日均成交额 > 150万即可
  · 实际可交易池：全A股 ~4500只（排除ST和日均成交 < 150万的）
  · 小盘股被机构覆盖少，alpha 空间更大
  · 量化研究表明：小盘股因子的 IC 通常高于大盘股

潜在改进：
  · 可以在因子库中增加小盘股特有因子（如：机构覆盖度、分析师关注度）
  · 模型训练时可以加入市值分层，对小盘股单独建模
  · 但这些是"锦上添花"，不是必须的
```

### 13.4 需要的具体改动清单

#### 13.4.1 改动量评估

```
┌─────────────────────────────────────────────────────┐
│              小资金适配改动量评估                      │
│                                                      │
│  因子演进成果    ──→  零改动，直接复用                │
│  模型演进成果    ──→  零改动，直接复用                │
│  回测配置       ──→  改3个参数（topk, n_drop, account）│
│  选股服务       ──→  改1个默认值（top_k 参数）        │
│  交易成本       ──→  可选优化（降低 open/close_cost） │
│  股票池         ──→  可选扩展（纳入更多小盘股）       │
│                                                      │
│  总结：核心改动 < 10行配置，无需修改任何算法代码       │
└─────────────────────────────────────────────────────┘
```

#### 13.4.2 推荐的小资金配置方案

建议在 QE 的 `config_composer.py` 中支持按资金规模自动适配参数：

```python
# 资金规模 → 策略参数映射
CAPITAL_PROFILES = {
    "micro":  {  # 100万以下
        "account": 1_000_000,
        "topk": 8,
        "n_drop": 1,
        "max_position_ratio": 0.80,
        "max_single_order_value": 200_000,
    },
    "small":  {  # 100-300万
        "account": 2_000_000,
        "topk": 12,
        "n_drop": 2,
        "max_position_ratio": 0.80,
        "max_single_order_value": 500_000,
    },
    "medium": {  # 300-500万
        "account": 5_000_000,
        "topk": 18,
        "n_drop": 3,
        "max_position_ratio": 0.85,
        "max_single_order_value": 1_000_000,
    },
    "large":  {  # 5000万+（当前默认）
        "account": 100_000_000,
        "topk": 50,
        "n_drop": 5,
        "max_position_ratio": 0.90,
        "max_single_order_value": 5_000_000,
    },
}
```

#### 13.4.3 小资金场景的特殊考量

**TopK 缩小后的影响**：

```
TopK 从 50 缩小到 10-15 的影响分析：

正面影响：
  · 集中度提高 → 如果模型有效，收益放大
  · Top10 通常是模型最有信心的预测，IC 更高
  · 换手率降低（n_drop 也缩小）→ 交易成本降低

负面影响：
  · 分散度降低 → 单只股票暴雷的影响更大
  · 波动率上升 → 回撤可能更大
  · 对模型排序精度要求更高（Top10 vs Top50 的容错空间不同）

量化实践经验：
  · 大多数量化私募的小规模产品持仓 10-20 只
  · 百亿私募持仓 50-100 只
  · 10-20 只是小资金的合理区间
```

### 13.5 是否需要针对小资金单独做一轮演进？

短期答案：**不需要**。长期可以考虑。

#### 13.5.1 为什么短期不需要

```
当前演进的核心产出是：
  1. 因子库（哪些因子有预测能力）→ 与资金无关
  2. 模型权重（因子→收益的映射）→ 与资金无关
  3. SOTA 配置（最优因子+模型组合）→ 与资金无关

唯一与资金相关的是策略参数（topk, n_drop），
这些参数不需要通过演进来确定，直接按资金规模查表即可。
```

#### 13.5.2 长期可以考虑的小资金专项优化

如果未来希望进一步提升小资金场景的表现，可以考虑以下方向（均为可选，非必须）：

```
优化 1：小盘股专项因子演进
  · 在 RD-Agent 中增加小盘股因子模板
  · 演进目标：在市值 < 50亿的股票池中最大化 IC
  · 预期收益：小盘股 alpha 空间更大，IC 可能提升 30-50%
  · 改动量：新增一套因子模板 + 修改股票池配置

优化 2：TopK 敏感性分析
  · 在 QE 演进中增加 topk 作为可调参数
  · 对同一个 SOTA 配置，分别用 topk=10/15/20/30 回测
  · 找到每个资金规模下的最优 topk
  · 改动量：QE config_composer 增加 topk 参数范围

优化 3：集中持仓风控增强
  · 小资金持仓集中，需要更强的风控
  · 增加行业分散约束（同行业不超过 3 只）
  · 增加个股止损逻辑（已有 stop_loss=-10%）
  · 改动量：custom_strategy.py 增加行业约束
```

### 13.6 小资金适配方案总结

| 层面 | 可复用性 | 需要的改动 | 改动量 |
|------|---------|-----------|--------|
| 因子库 | ✅ 完全复用 | 无 | 0 |
| 模型权重 | ✅ 完全复用 | 无 | 0 |
| SOTA 配置 | ✅ 完全复用 | 无 | 0 |
| 回测 account | ⚠️ 需调整 | 按资金规模设置 | 1行配置 |
| 策略 topk | ⚠️ 需调整 | 50 → 8-20（按资金规模） | 1行配置 |
| 策略 n_drop | ⚠️ 需调整 | 5 → 1-3（按资金规模） | 1行配置 |
| 选股服务 top_k | ⚠️ 需调整 | API 默认值从50改为可配置 | 几行代码 |
| 交易成本 | ✅ 可复用 | 可选降低（小资金冲击成本更低） | 可选 |
| 股票池 | ✅ 可复用 | 可选扩展小盘股 | 可选 |
| 风控参数 | ⚠️ 建议增强 | 增加行业分散约束 | 可选 |

**核心结论**：演进成果（因子+模型）完全可复用，仅需调整 3 个策略参数即可适配小资金。不需要大规模改进。小资金反而因为冲击成本低、可交易小盘股而具有结构性优势。

---

## 十四、策略层演进：被忽视的第三维度

### 14.1 现状诊断：策略层几乎没有参与演进

当前系统的演进维度分布极不均衡：

```
演进维度        参与演进程度     当前状态
─────────────────────────────────────────────
因子（Alpha）    ★★★★★        RD-Agent + QE 充分演进
模型（Model）    ★★★☆☆        有演进但维度受限（见第1-4节）
策略（Strategy） ★☆☆☆☆        几乎完全硬编码，未参与演进
```

具体来看 `custom_strategy.py` 中的硬编码参数：

```python
# 这些参数全部写死，从未被演进过程调整
topk = 50                    # 持仓数量
n_drop = 5                   # 换仓数量
stop_loss = -0.10            # 止损 -10%
take_profit_1 = 0.15         # 第一档止盈 15%（卖30%仓位）
take_profit_2 = 0.25         # 第二档止盈 25%（再卖30%）
take_profit_3 = 0.35         # 第三档止盈 35%（全部清仓）
max_position_ratio = 0.90    # 最大仓位比例
```

这意味着：无论因子和模型怎么演进，策略执行层始终用同一套参数交易。这就像一个优秀的分析师给出了精准的选股建议，但交易员永远用同一种方式下单。

### 14.2 多因子策略不只有 TopK

TopK 是最简单直接的多因子交易策略，但远不是唯一的。业界常用的多因子交易策略至少有以下几大类：

#### 14.2.1 排序选股类（当前使用）

```
策略 1: TopK Dropout（当前实现）
  原理：选评分最高的K只股票等权持有，每日换掉评分下降最多的n只
  优点：简单直观，换手率可控
  缺点：等权分配浪费了评分信息，Top1和Top50获得相同仓位

策略 2: 评分加权（Score-Weighted）
  原理：按模型评分分配权重，评分越高仓位越大
  优点：充分利用评分信息，高置信度预测获得更多仓位
  缺点：集中度高，对评分精度要求高
  当前状态：custom_strategy.py 中有 _calculate_dynamic_weights 但未充分利用

策略 3: 分层多空（Long-Short）
  原理：做多评分最高的K只，做空评分最低的K只
  优点：市场中性，牛熊市都能赚钱
  缺点：A股做空受限（融券难、成本高），实操困难
  适用：可用股指期货对冲替代纯做空
```

#### 14.2.2 组合优化类

```
策略 4: 均值-方差优化（Mean-Variance Optimization, MVO）
  原理：在预期收益和风险之间寻找最优权重分配
  输入：模型预测的收益 + 协方差矩阵
  优点：理论最优，考虑了股票间的相关性
  缺点：对预测误差敏感，实操中需要加约束
  适用：资金量较大、追求稳健收益的场景

策略 5: 风险平价（Risk Parity）
  原理：让每只股票对组合风险的贡献相等
  优点：天然分散风险，回撤小
  缺点：不直接最大化收益，更偏防守
  适用：低风险偏好的投资者

策略 6: 最大化夏普比率（Max Sharpe）
  原理：在所有可能的权重组合中，找到夏普比率最高的
  优点：收益/风险比最优
  缺点：计算复杂，对输入敏感
  适用：追求风险调整后收益最大化
```

#### 14.2.3 风控增强类

```
策略 7: 带约束的 TopK（Constrained TopK）
  原理：在 TopK 基础上增加行业、市值、个股集中度等约束
  示例：
    · 单行业不超过 30% 仓位
    · 单只股票不超过 10% 仓位
    · 大/中/小盘各占 1/3
  优点：降低极端风险，避免行业踩雷
  缺点：约束可能排除掉评分最高的股票
  适用：所有场景，尤其是小资金集中持仓时

策略 8: 动态仓位管理（Dynamic Position Sizing）
  原理：根据市场状态动态调整总仓位
  示例：
    · 市场波动率高 → 降低总仓位到 60%
    · 市场波动率低 → 提高总仓位到 90%
    · 连续亏损 → 自动降仓（类似凯利公式）
  优点：在熊市中减少暴露，牛市中充分参与
  缺点：需要准确判断市场状态
```

#### 14.2.4 换仓周期与择时类

```
策略 9: 固定周期换仓（Periodic Rebalance）
  原理：不是每天换仓，而是按固定周期（周/双周/月）重新平衡
  周频换仓：每周一开盘重新选股，持有一周
  双周换仓：每两周换一次
  月频换仓：每月初换一次
  优点：大幅降低交易成本，减少噪声交易
  缺点：对突发事件反应慢

策略 10: 信号强度触发换仓（Signal-Triggered Rebalance）
  原理：不按固定周期，而是当模型评分变化超过阈值时才换仓
  示例：
    · 某只持仓股评分从 Top5 跌到 Top30 以外 → 触发卖出
    · 某只非持仓股评分突然进入 Top3 → 触发买入
  优点：只在"值得交易"时才交易，避免无效换手
  缺点：实现复杂，需要每日计算评分但不一定交易

策略 11: 市场状态择时（Market Regime Timing）
  原理：根据市场整体状态决定是否执行选股策略
  示例：
    · 市场处于强趋势上涨 → 满仓执行 TopK
    · 市场震荡 → 半仓执行，提高选股门槛
    · 市场急跌 → 空仓观望，暂停选股
  判断依据：均线系统、波动率、市场宽度、北向资金等
  优点：避免在系统性下跌中被动挨打
  缺点：择时本身很难做准，可能错过反弹
```

### 14.3 持股周期、止盈止损对收益的影响

这三个参数对最终收益的影响程度，在很多场景下不亚于因子和模型的选择。

#### 14.3.1 持股周期的影响

```
持股周期是多因子策略中影响最大的单一参数之一。

同一个因子+模型组合，不同持股周期的表现可能天差地别：

日频换仓（每天换）：
  · 年化换手率 ~200-300 倍
  · 交易成本吃掉大部分 alpha
  · 适合：极高频因子（日内动量、资金流）
  · 实际净收益往往最低

周频换仓（每周换一次）：
  · 年化换手率 ~30-50 倍
  · 交易成本可控
  · 适合：中频因子（5日动量、周度反转）
  · 通常是收益/成本的最佳平衡点

双周/月频换仓：
  · 年化换手率 ~12-25 倍
  · 交易成本很低
  · 适合：低频因子（基本面、价值、质量）
  · 收益稳定但可能错过短期机会

量化实证：
  同一个 IC=0.04 的因子组合：
  · 日频换仓净收益 ≈ 5-8%（扣除成本后）
  · 周频换仓净收益 ≈ 12-18%
  · 月频换仓净收益 ≈ 8-12%
  差距可达 2-3 倍，远超换一个模型带来的提升
```

#### 14.3.2 止损策略的影响

```
当前实现：固定止损 -10%（custom_strategy.py）

止损的两面性：

正面（截断亏损）：
  · 避免单只股票暴跌拖垮整个组合
  · 心理层面：限制最大单笔亏损，保持纪律
  · 在趋势性下跌中有效保护本金

负面（过早离场）：
  · A股波动大，-10% 经常被触发后又反弹
  · 频繁止损 → 高换手 → 高成本
  · 如果因子本身有效，被止损的股票后续可能涨回来

不同止损阈值的影响（实证经验）：

  止损 -5%:  触发频率高，大量"假止损"，净收益通常最差
  止损 -8%:  中等频率，适合高波动市场
  止损 -10%: 当前设置，A股中较为合理的默认值
  止损 -15%: 触发频率低，给股票更多恢复空间
  不止损:    依赖换仓自然淘汰，在极端行情中风险大

关键洞察：
  最优止损阈值与持股周期强相关
  · 日频换仓 → 不需要止损（每天都在重新选股）
  · 周频换仓 → 止损 -8% 到 -10%
  · 月频换仓 → 止损 -12% 到 -15%（给更多时间）
```

#### 14.3.3 止盈策略的影响

```
当前实现：三档阶梯止盈（15%/25%/35%）

止盈的核心矛盾："让利润奔跑" vs "落袋为安"

当前阶梯止盈分析：
  盈利 15% → 卖 30% 仓位（锁定部分利润）
  盈利 25% → 再卖 30%（累计卖出 60%）
  盈利 35% → 全部清仓

问题：
  · 在强势股上过早止盈，错过后续大涨
  · A股牛股经常翻倍，35% 止盈会错过大部分涨幅
  · 但在震荡市中，阶梯止盈确实能锁定利润

替代方案：

  方案 A: 移动止盈（Trailing Stop）
    · 不设固定止盈点，而是跟踪最高价
    · 从最高价回撤 X% 时止盈
    · 示例：股票从买入价涨了 30%，然后从最高点回撤 8% → 止盈
    · 优点：在趋势中能吃到大部分涨幅

  方案 B: 评分驱动止盈
    · 不看涨幅，看模型评分变化
    · 当股票评分从 Top5 跌出 Top30 → 卖出
    · 优点：与因子逻辑一致，"模型不看好了就卖"
    · 缺点：需要每日更新评分

  方案 C: 时间止盈
    · 持有满 N 个交易日后无条件卖出，重新选股
    · 优点：简单，避免"恋战"
    · 缺点：可能卖掉仍在上涨的股票
```

#### 14.3.4 三者联动：策略参数组合的影响远大于单一参数

```
关键洞察：持股周期、止盈、止损三者不是独立的，它们构成一个策略参数空间，
组合效应远大于单独调整任何一个参数。

示例：同一个 SOTA 因子+模型，不同策略参数组合的年化收益差异

组合 A（当前默认）：日频换仓 + 止损-10% + 阶梯止盈
  → 年化超额 ~8%

组合 B（周频稳健）：周频换仓 + 止损-12% + 移动止盈8%
  → 年化超额 ~15%

组合 C（月频价值）：月频换仓 + 止损-15% + 不止盈
  → 年化超额 ~10%

组合 D（信号驱动）：评分触发换仓 + 止损-10% + 评分驱动止盈
  → 年化超额 ~18%

差距：最优组合 vs 最差组合可能相差 2 倍以上
这个差距与"换一个更好的模型"带来的提升相当甚至更大
```

### 14.4 策略层的演进空间

将策略参数纳入演进，形成"因子 × 模型 × 策略"三维搜索空间：

#### 14.4.1 可演进的策略维度

```
维度 1: 策略类型（离散选择）
  · TopK 等权
  · TopK 评分加权
  · 带约束的 TopK（行业/市值分散）
  · 组合优化（MVO / Risk Parity / Max Sharpe）

维度 2: 换仓周期（离散选择）
  · 日频 / 周频 / 双周频 / 月频
  · 信号触发式（评分变化超阈值才换仓）

维度 3: 持仓数量（连续参数）
  · topk: 8-50（与资金规模挂钩）
  · n_drop: 1-10

维度 4: 止损参数（连续参数）
  · 固定止损阈值: -5% ~ -20%
  · 是否启用止损: on/off
  · 移动止损回撤比例: 5% ~ 15%

维度 5: 止盈参数（连续参数）
  · 阶梯止盈档位和比例
  · 移动止盈回撤比例
  · 评分驱动止盈阈值

维度 6: 仓位管理（连续参数）
  · 最大总仓位: 60% ~ 95%
  · 单只最大仓位: 5% ~ 15%
  · 动态仓位调节系数
```

#### 14.4.2 策略演进的价值量化

```
为什么策略演进可能比模型演进更有价值？

1. 搜索空间更小，收敛更快
   · 模型演进：架构 × 超参数 × 训练策略 → 组合爆炸
   · 策略演进：~6个维度，每个维度 3-5 个选项 → 可穷举
   · 一轮策略网格搜索 ~100 种组合，回测几小时即可完成

2. 效果可直接衡量
   · 模型演进看 IC（间接指标）
   · 策略演进直接看年化收益、最大回撤、夏普比率（最终指标）

3. 与因子/模型正交
   · 好的策略参数对所有因子+模型组合都有效
   · 找到最优策略后，因子/模型的改进效果会被放大

4. 实盘一致性更高
   · 因子/模型的回测-实盘衰减通常 30-60%
   · 策略参数的回测-实盘一致性更高（交易规则是确定性的）
```

### 14.5 策略纳入 QE 演进的实施方案

#### 14.5.1 方案 A：策略参数网格搜索（最简单，推荐先做）

不需要 LLM 参与，纯参数搜索即可：

```
对每个 SOTA 因子+模型组合，自动执行策略参数网格搜索：

搜索空间（精简版，~72种组合）：
  换仓周期: [daily, weekly, biweekly]           → 3
  topk:     [10, 20, 30]                        → 3
  止损:     [-8%, -12%, off]                     → 3
  止盈:     [阶梯15/25/35, 移动8%, off]          → 3
  仓位:     [80%, 90%]                           → 2 (可选)

执行方式：
  1. QE 产出新 SOTA 后自动触发
  2. 对 72 种策略组合分别回测（复用同一个模型预测结果）
  3. 按夏普比率排序，选出 Top3 策略配置
  4. 将最优策略配置与 SOTA 因子+模型一起记录

关键优化：模型推理只需执行一次，72次回测只是改变交易规则
  → 总耗时 ≈ 1次模型推理 + 72次快速回测 ≈ 30分钟
```

#### 14.5.2 方案 B：LLM 驱动的策略演进（进阶）

将策略参数作为 QE Researcher Agent 的决策维度之一：

```
当前 Researcher Agent 的 action_type：
  · factor_adjust      → 调整因子组合
  · param_tune         → 调整模型参数
  · model_switch       → 更换模型
  · factor_model_joint → 因子+模型联合调整

新增 action_type：
  · strategy_tune      → 调整策略参数
  · full_joint         → 因子+模型+策略联合调整

Researcher Agent 的决策输入增加：
  · 当前策略配置及其回测表现
  · 策略参数网格搜索的 Top3 结果（来自方案A）
  · 实盘选股的策略层表现（来自第12节的反馈）

示例 Agent 输出：
  {
    "action_type": "strategy_tune",
    "strategy_config": {
      "rebalance_freq": "weekly",
      "topk": 20,
      "stop_loss": -0.12,
      "take_profit_mode": "trailing",
      "trailing_stop_pct": 0.08,
      "max_position_ratio": 0.85
    },
    "reasoning": "当前日频换仓成本过高，切换到周频可降低换手率..."
  }
```

#### 14.5.3 在现有架构中的落地路径

```
现有代码需要的改动：

1. config_composer.py（QE 配置组装）
   · 将策略参数从硬编码改为可配置
   · 新增 strategy_config 字段到 config_json
   · 改动量：~30行

2. custom_strategy.py（RD-Agent 策略执行）
   · 将硬编码参数改为从 YAML 配置读取
   · 新增周频/双周频换仓模式
   · 新增移动止盈逻辑
   · 改动量：~80行

3. conf_baseline.yaml / conf_baseline_factors_model.yaml
   · 新增 strategy_kwargs 配置块
   · 改动量：~15行

4. qe_evolution_agents.py（QE Agent）
   · Researcher Agent prompt 增加策略维度
   · 新增 strategy_tune action_type
   · 改动量：~20行 prompt 修改

总改动量：~150行代码，无架构变更
```

### 14.6 策略演进方案总结

| 维度 | 当前状态 | 演进空间 | 对收益的影响 | 优先级 |
|------|---------|---------|-------------|--------|
| 换仓周期 | 日频（硬编码） | 日/周/双周/月/信号触发 | ★★★★★ 影响最大 | ★★★★★ |
| 止损参数 | -10%（硬编码） | -5%~-20% / 移动止损 / 关闭 | ★★★★ | ★★★★ |
| 止盈参数 | 阶梯15/25/35%（硬编码） | 移动止盈 / 评分驱动 / 关闭 | ★★★★ | ★★★★ |
| 策略类型 | TopK 等权 | 评分加权 / 约束TopK / MVO | ★★★ | ★★★ |
| 仓位管理 | 固定90% | 动态仓位 / 波动率调节 | ★★★ | ★★★ |
| 行业约束 | 无 | 单行业上限 / 市值分层 | ★★ | ★★★ |
| 市场择时 | 无 | 波动率择时 / 趋势择时 | ★★ | ★★ |

**核心结论**：策略层是当前系统中被严重低估的演进维度。仅"换仓周期"一个参数的优化，对净收益的影响就可能超过换一个更好的模型。建议优先实施策略参数网格搜索（方案A），再逐步纳入 QE 的 LLM 驱动演进（方案B）。

---

## 十五、分钟线数据与日内交易策略分析

### 15.1 现状诊断：日线回测的局限性

当前系统的回测完全基于日线数据：

```
数据频率：日线（1d）
买卖价格：收盘价（deal_price: $close）
执行假设：当日信号 → 当日收盘价成交
滑点模型：固定比例（open_cost=0.05%, close_cost=0.15%）
```

这个假设与真实交易的差距：

| 环节 | 回测假设 | 真实交易 | 差距来源 |
|------|---------|---------|---------|
| 成交价 | 收盘价精确成交 | 实际挂单价 ≠ 收盘价 | 滑点 |
| 成交量 | 无限流动性 | 受限于盘口深度 | 冲击成本 |
| 执行时机 | 瞬时成交 | 需要时间完成建仓/平仓 | 执行延迟 |
| 大单影响 | 无市场影响 | 大单推高/压低价格 | 市场冲击 |
| 涨跌停 | 简单阈值过滤 | 封板后无法买入 | 流动性约束 |

对于 100-500 万小资金，日线回测的偏差相对较小（冲击成本低）。
对于 1 亿以上资金，日线回测可能高估实际收益 10-30%。

### 15.2 Qlib 的分钟线基础设施

Qlib 已经具备完整的分钟线数据和日内交易支持，不需要从零开发：

#### 15.2.1 数据层：分钟线 Bin 文件

```
dump_bin.py 支持 freq 参数：
  python dump_bin.py --freq 1min --csv_path ./minute_data --qlib_dir ./qlib_data

Bin 文件命名规则：
  日线：  close.day.bin, open.day.bin, volume.day.bin
  分钟线：close.1min.bin, open.1min.bin, volume.1min.bin

数据格式：
  HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
  每条记录包含：datetime, open, high, low, close, volume, amount

数据量估算（A股全市场）：
  日线：~5000股 × 10年 × 250天 ≈ 1250万条 → ~2GB bin文件
  分钟线：~5000股 × 10年 × 250天 × 240分钟 ≈ 30亿条 → ~500GB bin文件
```

#### 15.2.2 Handler 层：HighFreqHandler

```python
# qlib/contrib/data/highfreq_handler.py
class HighFreqHandler:
    """分钟线数据处理器"""
    freq = "1min"

    # 支持的特征：
    # - 分钟级 OHLCV
    # - VWAP（成交量加权平均价）
    # - 分钟级技术指标
    # - 日内模式特征（开盘/收盘效应等）
```

#### 15.2.3 执行层：日内交易策略

```
Qlib 内置的日内执行策略：

1. TWAPStrategy（时间加权平均价格）
   · 将大单拆分为等时间间隔的小单
   · 目标：成交均价接近 TWAP
   · 适用：降低大单冲击成本

2. SBBStrategy（Single-side Best Bid/Ask）
   · 在最优买/卖价挂单等待成交
   · 目标：以最优价格成交
   · 适用：不急于成交的场景

3. ACStrategy（Adaptive Control）
   · 根据市场状态动态调整下单节奏
   · 目标：在执行速度和成交价格间平衡

4. 自定义策略
   · 继承 BaseTradeStrategy 即可实现
   · 可获取分钟级行情数据做决策
```

#### 15.2.4 回测层：NestedExecutor

```
Qlib 的嵌套执行器架构：

OuterExecutor（日频决策层）
  │  每日产生交易信号（买入/卖出列表）
  │
  └─ InnerExecutor（分钟频执行层）
       │  将日频信号拆解为分钟级订单
       │  使用 TWAP/SBB 等策略执行
       │  模拟真实的盘口撮合
       │
       └─ Exchange（撮合引擎）
            · deal_price 支持：$close, $open, $vwap, $twap
            · 考虑涨跌停限制
            · 考虑成交量限制（不超过当分钟成交量的 X%）
            · 考虑最小交易单位（100股整数倍）

配置示例：
  executor:
    class: NestedExecutor
    module_path: qlib.backtest.executor
    kwargs:
      time_per_step: day        # 外层按日
      inner_executor:
        class: NestedExecutor
        kwargs:
          time_per_step: 30min  # 内层按30分钟
          inner_strategy:
            class: TWAPStrategy  # 使用TWAP执行
```

### 15.3 分钟线对回测准确性的提升分析

#### 15.3.1 准确性提升的三个层面

```
层面 1: 成交价格更真实（影响：中等）

  日线回测：假设以收盘价成交
  分钟线回测：模拟真实的分批建仓过程

  差异量化：
    · 小资金（100-500万）：收盘价 vs TWAP 差异 ≈ 0.05-0.15%
    · 中等资金（1000万-1亿）：差异 ≈ 0.1-0.3%
    · 大资金（1亿以上）：差异 ≈ 0.3-1.0%

  对年化收益的影响：
    假设年换手率 50 倍（日频换仓），每次买卖各 0.2% 滑点
    → 年化滑点成本 ≈ 50 × 0.4% = 20%
    这解释了为什么日频换仓在实盘中表现远差于回测

层面 2: 成交量约束更真实（影响：因资金规模而异）

  日线回测：假设无限流动性
  分钟线回测：限制每分钟成交量不超过市场成交量的 X%

  差异量化：
    · 小资金买入 10 万元某股：1分钟内可完成，几乎无影响
    · 大资金买入 500 万元某股：可能需要 30-60 分钟分批买入
    · 小盘股（日成交额 < 5000万）：大资金可能无法在一天内完成建仓

层面 3: 涨跌停处理更精确（影响：低）

  日线回测：简单判断涨跌停，整天不可交易
  分钟线回测：可以捕捉"盘中打开涨跌停"的交易机会

  差异量化：
    · 涨跌停打开的情况占比 ~5-10%
    · 对整体收益影响 < 0.5%
```

#### 15.3.2 不同场景下的准确性提升量化估算

```
场景 A: 小资金（100-500万）+ 周频换仓
  日线回测偏差：~1-3%（年化）
  分钟线回测偏差：~0.5-1%（年化）
  准确性提升：~1-2 个百分点
  结论：提升有限，日线回测已经足够准确

场景 B: 中等资金（1000万-5000万）+ 日频换仓
  日线回测偏差：~5-10%（年化）
  分钟线回测偏差：~2-4%（年化）
  准确性提升：~3-6 个百分点
  结论：有一定价值，但优先级不如换仓周期优化

场景 C: 大资金（1亿+）+ 日频换仓
  日线回测偏差：~10-30%（年化）
  分钟线回测偏差：~5-10%（年化）
  准确性提升：~5-20 个百分点
  结论：价值显著，大资金场景下应该使用分钟线回测

场景 D: 大资金（1亿+）+ 周频换仓
  日线回测偏差：~3-8%（年化）
  分钟线回测偏差：~1-3%（年化）
  准确性提升：~2-5 个百分点
  结论：有价值但不紧迫，周频换仓本身已大幅降低执行偏差
```

**关键洞察**：分钟线回测的价值与"资金规模 × 换仓频率"正相关。对于小资金+低频换仓的场景，日线回测的准确性已经足够。

### 15.4 实施成本与工程挑战

#### 15.4.1 数据获取与存储

```
挑战 1: 分钟线数据源

  A股分钟线数据获取方式：
    · Tushare Pro：1分钟/5分钟线，需要高级权限（5000积分+）
    · 通达信本地：可导出分钟线，但需要手动维护
    · Wind/Choice：专业终端，成本高
    · AKShare：部分支持，稳定性一般

  数据质量问题：
    · 分钟线数据的复权处理比日线复杂
    · 集合竞价时段（9:15-9:30）的数据处理
    · 午间休市（11:30-13:00）的时间对齐
    · 不同数据源的时间戳可能不一致

挑战 2: 存储空间

  全市场 10 年分钟线 bin 文件：~500GB
  对比日线 bin 文件：~2GB
  存储成本增加 250 倍

  磁盘 I/O：
    · 日线回测读取 ~2GB 数据
    · 分钟线回测读取 ~50-100GB 数据（取决于回测区间）
    · 需要 SSD，机械硬盘会成为瓶颈
```

#### 15.4.2 回测性能影响

```
日线回测 vs 分钟线回测的耗时对比：

日线回测（当前）：
  · 单次回测（3年数据）：~2-5 分钟
  · QE 一轮演进（含模型训练+回测）：~30-60 分钟
  · 策略网格搜索 72 组合：~30 分钟（复用模型推理）

分钟线回测（NestedExecutor）：
  · 单次回测（3年数据）：~30-120 分钟（慢 10-20 倍）
  · QE 一轮演进：~3-6 小时
  · 策略网格搜索 72 组合：~6-12 小时

影响：
  · QE 演进速度下降 5-10 倍
  · 每天能完成的演进轮次从 10-20 轮降到 2-4 轮
  · 严重拖慢因子/模型的探索效率
```

#### 15.4.3 与 RD-Agent 演进流程的兼容性

```
当前 RD-Agent 的回测流程：
  1. LLM 生成因子/模型代码
  2. Qlib 日线数据训练模型
  3. 日线回测评估 IC / 收益 / 回撤
  4. 反馈给 LLM 进行下一轮演进

如果切换到分钟线回测：
  · 步骤 2 不变（模型训练仍用日线特征预测日收益）
  · 步骤 3 改为 NestedExecutor 分钟线回测
  · 需要修改 conf_baseline.yaml 的 executor 配置
  · 需要确保分钟线 bin 文件已生成并可访问

代码改动量：
  · conf_baseline.yaml：~15行（executor 配置）
  · custom_strategy.py：无需改动（日频决策层不变）
  · 新增 TWAP 执行配置：~10行
  · 总改动：~25行配置，无代码改动

兼容性风险：
  · RD-Agent 的 result 解析逻辑假设日线回测输出格式
  · NestedExecutor 的输出格式可能略有不同
  · 需要验证 feedback.py 的 process_results 是否兼容
```

### 15.5 优先级评估：是否应该列为高优先级？

#### 15.5.1 与其他优化方向的对比

```
各优化方向的投入产出比对比：

方向                    | 实施成本 | 收益提升 | 投入产出比
------------------------|---------|---------|----------
策略参数优化（第14节）    | 低      | 高      | ★★★★★
实盘验证闭环（第12节）    | 中      | 高      | ★★★★
因子库精细化（第9节）     | 中      | 中      | ★★★★
模型演进空间扩展（第3节） | 低      | 中      | ★★★★
小资金适配（第13节）      | 低      | 中      | ★★★★
分钟线回测              | 高      | 低~中   | ★★
日内交易策略            | 很高    | 不确定  | ★
```

#### 15.5.2 不建议列为高优先级的理由

```
理由 1: 收益提升有限（对当前场景）

  当前目标资金规模：100-500万（第13节结论）
  小资金的日线回测偏差：仅 1-3%（年化）
  分钟线回测能减少的偏差：~1-2 个百分点
  → 投入大量工程资源换取 1-2% 的准确性提升，性价比低

理由 2: 更简单的替代方案可达到类似效果

  不用分钟线也能提升回测准确性：
  · 将 deal_price 从 $close 改为 $vwap → 更接近真实成交价
  · 增加滑点模型（按成交量比例计算冲击成本）
  · 降低换仓频率（周频代替日频）→ 从根本上减少执行偏差
  这些改动只需修改 YAML 配置，无需分钟线数据

理由 3: 会严重拖慢演进效率

  QE 演进速度下降 5-10 倍
  当前阶段的核心任务是"快速探索因子+模型空间"
  分钟线回测会让每轮演进从 30 分钟变成 3-6 小时
  → 探索效率的损失远大于准确性的提升

理由 4: 日内交易策略本身不成熟

  RD-Agent 的因子和模型都是预测"日收益"
  日内交易策略（如 TWAP）只是"执行层优化"，不改变预测逻辑
  真正的日内 alpha 策略需要完全不同的因子体系（订单流、盘口深度等）
  这超出了当前 RD-Agent 多因子框架的范畴
```

#### 15.5.3 什么时候应该升级为高优先级？

```
触发条件（满足任一即可考虑）：

1. 资金规模突破 5000 万
   → 冲击成本开始显著影响实盘收益
   → 分钟线回测的准确性提升变得有意义

2. 实盘验证闭环（第12节）已建立，且发现回测-实盘偏差 > 30%
   → 说明日线回测的执行假设是主要偏差来源
   → 分钟线回测可以缩小这个偏差

3. 因子+模型+策略的演进已趋于收敛
   → 核心 alpha 已经稳定
   → 优化重点从"发现 alpha"转向"精确执行"
   → 此时分钟线回测的价值凸显

4. 需要开发真正的日内交易策略
   → 如 T+0 策略、日内动量策略
   → 这需要分钟线数据作为基础设施
```

### 15.6 推荐方案：渐进式提升回测准确性

不需要一步到位引入分钟线，而是分阶段逐步提升：

#### 阶段 0：零成本优化（立即可做）

```
改动 1: deal_price 从 $close 改为 $vwap
  · 修改 conf_baseline.yaml: deal_price: $vwap
  · VWAP 比收盘价更接近真实成交均价
  · 改动量：1行配置

改动 2: 增加更真实的滑点模型
  · open_cost: 0.0005 → 0.001（考虑滑点）
  · close_cost: 0.0015 → 0.002
  · 改动量：2行配置

改动 3: 换仓周期从日频改为周频（参考第14节）
  · 从根本上减少执行次数，降低累积偏差
  · 改动量：参考第14节方案

预期效果：回测准确性提升 50-70%，零工程成本
```

#### 阶段 1：成交量约束模拟（中期，无需分钟线）

```
在日线回测中加入成交量约束：

改动：修改 Exchange 配置，限制单日成交量
  · trade_limit: 0.1  # 单只股票单日成交不超过该股当日成交量的 10%
  · 这会自动处理小盘股流动性不足的问题

配置示例：
  exchange:
    limit_threshold: 0.095
    deal_price: $vwap
    trade_unit: 100
    volume_threshold:
      limit: 0.1  # 成交量限制

预期效果：捕捉 80% 的流动性约束影响，无需分钟线数据
改动量：~5行配置
```

#### 阶段 2：分钟线回测（远期，按需启用）

```
触发条件：资金规模 > 5000万，或实盘偏差 > 30%

实施步骤：
  1. 获取分钟线数据源（Tushare Pro 或通达信）
  2. 使用 dump_bin.py --freq 1min 生成 bin 文件
  3. 配置 NestedExecutor + TWAP 执行策略
  4. 仅在"最终验证"阶段使用分钟线回测
     （日常演进仍用日线回测保持速度）

推荐架构：双轨回测
  · 快速轨（日线）：用于 QE 日常演进，追求速度
  · 精确轨（分钟线）：用于 SOTA 最终验证，追求准确
  · SOTA 产出后自动触发精确轨验证
  · 如果精确轨结果与快速轨偏差 > 阈值，标记为"过拟合风险"

预期效果：在不影响演进速度的前提下，获得分钟线级别的准确性
改动量：~25行配置 + 数据准备工作
```

### 15.7 分钟线方案总结

| 维度 | 评估 |
|------|------|
| 回测准确性提升 | 小资金 1-2%，大资金 5-20%（年化） |
| 实施成本 | 高（500GB 数据、回测慢 10-20 倍） |
| 对演进效率的影响 | 严重负面（速度下降 5-10 倍） |
| 替代方案 | VWAP + 滑点模型 + 周频换仓可达 70% 效果 |
| 优先级 | ★★（低），远低于策略优化、实盘闭环等方向 |
| 建议时机 | 资金 > 5000万，或实盘偏差 > 30% 时再考虑 |
| 推荐架构 | 双轨回测（日线快速轨 + 分钟线精确轨） |

**核心结论**：分钟线回测是"锦上添花"而非"雪中送炭"。当前阶段的核心瓶颈不在回测精度，而在因子/模型/策略的搜索空间尚未充分探索。建议先用零成本方案（VWAP + 滑点模型 + 周频换仓）提升回测准确性，待系统成熟后再按需引入分钟线回测。

---

## 十六、AIstock 轻量模拟盘平台设计

### 16.1 为什么不直接对接 QMT 模拟盘？

```
QMT 模拟盘的局限：

1. 对接复杂度高
   · 需要 miniQMT 客户端持续运行
   · 委托/成交回调机制复杂（异步推送 + 状态机）
   · 网络断连、重连、订单状态同步等边界情况多
   · 调试困难，错误信息不透明

2. 功能受限
   · QMT 模拟盘的统计维度固定，无法自定义
   · 无法按因子/模型/策略维度拆分归因
   · 无法同时运行大量虚拟组合（QMT 账户数有限）
   · 历史交易记录查询能力有限

3. 与 QE 演进的耦合困难
   · QMT 的持仓/收益数据格式与 QE 反馈格式不匹配
   · 需要额外的适配层转换数据
   · 实时性要求高，增加系统复杂度

自建模拟盘的优势：
   · 完全可控，统计维度无限扩展
   · 可同时运行 10+ 个虚拟组合
   · 直接复用现有选股数据底座
   · 与 QE 反馈闭环天然集成
   · 使用 miniQMT 的行情数据但不依赖其交易功能
```

### 16.2 整体架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    AIstock 模拟盘平台                      │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ 虚拟组合A │  │ 虚拟组合B │  │ 虚拟组合C │  ...         │
│  │ QE实验#1  │  │ QE实验#2  │  │ 手工配置  │              │
│  │ 500万本金 │  │ 500万本金 │  │ 200万本金 │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │              │              │                    │
│       └──────────────┼──────────────┘                    │
│                      ▼                                   │
│  ┌─────────────────────────────────────┐                │
│  │         每日调度引擎（定时任务）        │                │
│  │  1. 收盘后触发选股（复用 qe_selection） │                │
│  │  2. 对比持仓 vs 新信号 → 生成交易指令   │                │
│  │  3. 按实际收盘价/VWAP 模拟成交         │                │
│  │  4. 扣除交易成本                       │                │
│  │  5. 更新持仓和净值                     │                │
│  └─────────────────┬───────────────────┘                │
│                    ▼                                     │
│  ┌─────────────────────────────────────┐                │
│  │           数据底座（已有）              │                │
│  │  · miniQMT 实时行情（收盘价/VWAP）    │                │
│  │  · get_history_window（历史K线）      │                │
│  │  · trading.rdagent_signal（选股信号）  │                │
│  └─────────────────────────────────────┘                │
│                                                         │
│  ┌─────────────────────────────────────┐                │
│  │           统计与反馈                   │                │
│  │  · 每日净值曲线                       │                │
│  │  · 周度/月度绩效报告                  │                │
│  │  · 横向对比多组合                     │                │
│  │  · 反馈注入 QE 演进                   │                │
│  └─────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

### 16.3 数据库表设计

需要新增 3 张核心表，全部放在 `trading` schema 下：

#### 16.3.1 虚拟组合表（sim_portfolio）

```sql
CREATE TABLE IF NOT EXISTS trading.sim_portfolio (
    portfolio_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_name      TEXT NOT NULL,
    -- 关联的QE实验（可选，手工配置的组合不关联实验）
    experiment_id       TEXT,
    task_id             TEXT,              -- QE演进任务ID
    -- 策略配置
    strategy_config     JSONB NOT NULL,    -- {topk, rebalance_freq, stop_loss, ...}
    factor_names        JSONB,             -- 因子列表
    model_id            TEXT,              -- 模型ID
    -- 资金配置
    initial_capital     NUMERIC(14,2) NOT NULL DEFAULT 5000000,
    -- 状态
    status              TEXT NOT NULL DEFAULT 'active',  -- active/paused/archived
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

#### 16.3.2 每日净值与持仓快照表（sim_daily_snapshot）

```sql
CREATE TABLE IF NOT EXISTS trading.sim_daily_snapshot (
    id                  BIGSERIAL PRIMARY KEY,
    portfolio_id        UUID NOT NULL REFERENCES trading.sim_portfolio(portfolio_id),
    trade_date          DATE NOT NULL,
    -- 资产概况
    total_nav           NUMERIC(14,2) NOT NULL,  -- 总净值
    cash                NUMERIC(14,2) NOT NULL,  -- 现金
    market_value        NUMERIC(14,2) NOT NULL,  -- 持仓市值
    -- 当日损益
    daily_pnl           NUMERIC(12,2),           -- 当日盈亏
    daily_return        NUMERIC(8,6),            -- 当日收益率
    -- 累计指标
    cumulative_return   NUMERIC(8,6),            -- 累计收益率
    max_drawdown        NUMERIC(8,6),            -- 历史最大回撤
    -- 持仓明细（JSONB快照）
    positions_json      JSONB,
    -- 当日交易统计
    buy_count           INTEGER DEFAULT 0,
    sell_count          INTEGER DEFAULT 0,
    trade_cost          NUMERIC(10,2) DEFAULT 0, -- 当日交易成本
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (portfolio_id, trade_date)
);
```

#### 16.3.3 交易记录表（sim_trade_record）

```sql
CREATE TABLE IF NOT EXISTS trading.sim_trade_record (
    id                  BIGSERIAL PRIMARY KEY,
    portfolio_id        UUID NOT NULL REFERENCES trading.sim_portfolio(portfolio_id),
    trade_date          DATE NOT NULL,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL,            -- BUY / SELL
    -- 交易详情
    quantity            INTEGER NOT NULL,         -- 成交数量（手 × 100）
    price               NUMERIC(10,4) NOT NULL,  -- 成交价格
    amount              NUMERIC(14,2) NOT NULL,   -- 成交金额
    -- 成本
    commission          NUMERIC(10,2) DEFAULT 0,  -- 佣金
    stamp_tax           NUMERIC(10,2) DEFAULT 0,  -- 印花税（卖出）
    slippage            NUMERIC(10,2) DEFAULT 0,  -- 滑点成本
    total_cost          NUMERIC(10,2) DEFAULT 0,  -- 总交易成本
    -- 信号来源
    signal_rank         INTEGER,
    signal_score        DOUBLE PRECISION,
    trade_reason        TEXT,                     -- new_entry/rebalance/stop_loss/take_profit
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sim_trade_portfolio_date
    ON trading.sim_trade_record (portfolio_id, trade_date);
```

### 16.4 每日调度流程

```
触发时机：每个交易日收盘后（15:30 之后）

Step 1: 获取最新行情（复用现有数据底座）
  · 调用 get_realtime_snapshot() 获取全市场收盘价
  · 数据来源优先级：miniQMT → TDX → TimescaleDB
  · 同时获取 VWAP（如果 miniQMT 可用）

Step 2: 更新现有持仓市值
  · 遍历每个 active 的 sim_portfolio
  · 按最新收盘价重新计算持仓市值
  · 检查止损/止盈触发条件

Step 3: 执行选股（复用 qe_selection_service）
  · 对每个组合调用 build_experiment_selection()
  · 获取 TopK 信号列表（symbol, rank, score）
  · 这一步完全复用现有代码，零改动

Step 4: 生成交易指令
  · 对比当前持仓 vs 新信号
  · 新进入 TopK 的股票 → BUY
  · 跌出 TopK 的股票 → SELL
  · 触发止损的股票 → SELL（优先级最高）
  · 触发止盈的股票 → SELL（按策略配置）
  · 计算每只股票的目标数量（按评分加权 or 等权）

Step 5: 模拟成交
  · 成交价格：收盘价（默认）或 VWAP（更真实）
  · 成交数量：按 100 股整数倍取整
  · 检查现金是否充足
  · 扣除交易成本（佣金 + 印花税 + 滑点）
  · 写入 sim_trade_record

Step 6: 更新净值快照
  · 计算当日总净值 = 现金 + 持仓市值
  · 计算当日收益率、累计收益率、最大回撤
  · 写入 sim_daily_snapshot
  · 持仓明细以 JSONB 快照保存
```

### 16.5 交易成本模型

```
A股真实交易成本构成：

1. 佣金（双向收取）
   · 费率：万2.5（0.025%），最低5元
   · 买入 100万 → 佣金 250元
   · 卖出 100万 → 佣金 250元

2. 印花税（仅卖出）
   · 费率：千1（0.1%）
   · 卖出 100万 → 印花税 1000元

3. 滑点（隐性成本）
   · 收盘价 vs 实际成交价的偏差
   · 小资金（100-500万）：~0.05-0.1%
   · 中等资金（1000万+）：~0.1-0.3%

模拟盘成本计算公式：

  买入成本 = max(成交金额 × 0.025%, 5元) + 成交金额 × 滑点率
  卖出成本 = max(成交金额 × 0.025%, 5元) + 成交金额 × 0.1% + 成交金额 × 滑点率

  单次买卖总成本 ≈ 成交金额 × 0.175%（含滑点）

配置化设计（strategy_config 中）：
  {
    "cost": {
      "commission_rate": 0.00025,
      "min_commission": 5.0,
      "stamp_tax_rate": 0.001,
      "slippage_rate": 0.0005
    }
  }
```

### 16.6 日内策略执行（结合 miniQMT 分钟线）

当前 miniQMT 已支持 1min/5min/tick 级别的实时数据推送，可以在日频选股的基础上叠加日内执行优化：

```
日频选股 + 日内执行的两层架构：

外层（日频）：每日收盘后决定"买什么、卖什么"
  · 复用 qe_selection_service 的 TopK 信号
  · 生成次日交易计划（买入列表 + 卖出列表）

内层（日内）：次日开盘后决定"什么时候买、什么价格买"
  · 简单 TWAP：将订单均匀分布在 9:30-14:30
  · 开盘回避：避开前 15 分钟的高波动期
  · 尾盘集中：在 14:30-14:50 集中执行（接近收盘价）
  · 价格限制：设置相对于开盘价的最大偏离（如 ±2%）

模拟盘中的日内执行模拟：

  方案 A（简单版，推荐先做）：
    · 成交价 = 当日 VWAP（从 miniQMT 获取）
    · 不需要分钟级模拟，一步到位
    · 准确度已经很高（VWAP 是机构常用基准）

  方案 B（进阶版，后期可选）：
    · 使用 miniQMT 的分钟线数据回放
    · 模拟 TWAP 分批成交过程
    · 考虑每分钟成交量限制
    · 记录模拟的分钟级成交明细
```

### 16.7 绩效统计体系

#### 16.7.1 每日自动计算的指标

```
净值类：
  · 当日净值（NAV）
  · 当日收益率 = (今日NAV - 昨日NAV) / 昨日NAV
  · 累计收益率 = (当前NAV - 初始资金) / 初始资金
  · 年化收益率 = (1 + 累计收益率) ^ (252/交易天数) - 1

风险类：
  · 最大回撤 = max(1 - NAV_t / max(NAV_0..t))
  · 当日波动率（滚动20日标准差）
  · 下行波动率（仅计算负收益的标准差）

风险调整收益：
  · 夏普比率 = (年化收益 - 无风险利率) / 年化波动率
  · 卡尔玛比率 = 年化收益 / 最大回撤
  · 索提诺比率 = (年化收益 - 无风险利率) / 下行波动率
```

#### 16.7.2 周度/月度深度统计

```
交易统计：
  · 总交易次数、买入次数、卖出次数
  · 胜率 = 盈利交易次数 / 总交易次数
  · 盈亏比 = 平均盈利金额 / 平均亏损金额
  · 平均持仓天数
  · 换手率 = 期间交易金额 / 平均持仓市值

成本分析：
  · 总交易成本（佣金 + 印花税 + 滑点）
  · 成本占收益比 = 总成本 / 总盈利
  · 单笔平均成本

持仓分析：
  · 平均持仓数量
  · 行业集中度（前3大行业占比）
  · 市值分布（大/中/小盘占比）
  · 最大单只持仓占比
```

#### 16.7.3 选股效果验证指标（与回测对标）

```
实盘 IC 验证：
  · 每日计算：模型评分 vs 次日实际涨跌幅的相关系数
  · 滚动 20 日 IC 均值
  · 与回测 IC 对比 → 衡量过拟合程度

回测-实盘一致性（参考第12节）：
  · IC 衰减率 = 实盘IC / 回测IC
  · 收益衰减率 = 实盘年化收益 / 回测年化收益
  · 健康阈值：衰减率 > 0.5 为可接受

超额收益分解：
  · 相对基准（沪深300/中证500）的超额收益
  · Alpha 归因：因子贡献 vs 模型贡献 vs 策略贡献
```

### 16.8 多组合横向对比与 QE 反馈

#### 16.8.1 横向对比看板

```
同时运行多个虚拟组合的核心价值：
  · 控制变量法验证因子/模型/策略各自的贡献
  · 发现回测表现好但实盘表现差的"过拟合组合"
  · 为真实交易提供决策参考

推荐的组合配置示例：

  组合 A: QE SOTA 实验 #1（最新 SOTA 因子+模型+默认策略）
  组合 B: QE SOTA 实验 #2（次优 SOTA，用于对比）
  组合 C: 同 A 的因子+模型，但策略改为周频换仓
  组合 D: 同 A 的因子+模型，但策略改为 TopK=20
  组合 E: 基准组合（沪深300 ETF 买入持有）

横向对比维度：
  · 累计收益率曲线叠加
  · 最大回撤对比
  · 夏普比率排名
  · 实盘 IC 对比
  · 交易成本占比对比
  · 胜率和盈亏比对比
```

#### 16.8.2 模拟盘数据反馈注入 QE 演进

```
反馈路径：
  sim_daily_snapshot → QEFeedback.live_performance → QE Analyst Agent

具体注入方式：

1. 扩展 QEFeedback 数据模型（qe_evolution_models.py）
   新增字段：
     live_performance: {
       portfolio_id: "...",
       running_days: 30,
       cumulative_return: 0.052,
       max_drawdown: -0.038,
       sharpe: 1.85,
       live_ic: 0.031,
       ic_decay_ratio: 0.78,     # 实盘IC / 回测IC
       return_decay_ratio: 0.65,  # 实盘收益 / 回测收益
     }

2. QE Analyst Agent 的 prompt 增加实盘维度
   "该实验的模拟盘已运行 30 天，累计收益 5.2%，
    实盘IC为0.031（回测IC为0.040，衰减率78%），
    请结合实盘表现评估该配置的真实有效性..."

3. 过拟合预警信号
   · IC 衰减率 < 0.5 → 严重过拟合，建议放弃该配置
   · 收益衰减率 < 0.3 → 回测收益不可信
   · 实盘最大回撤 > 回测最大回撤 × 2 → 风险被低估
```

### 16.9 可复用的现有基础设施清单

```
模拟盘开发中可直接复用的现有模块（零改动）：

模块                              | 复用方式                    | 改动量
----------------------------------|----------------------------|-------
qe_selection_service.py           | 每日选股信号生成              | 0行
InferenceEngine                   | 模型推理                    | 0行
data_service/api.py               | get_realtime_snapshot()     | 0行
                                  | get_history_window()        | 0行
data_service/miniqmt_adapter.py   | 实时行情/VWAP获取            | 0行
trading.rdagent_signal            | 选股信号存储                 | 0行
qe_experiments 表                 | 实验配置查询                 | 0行
qe_evolution_loops 表             | 演进历史查询                 | 0行
data_source_manager               | 股票名称/基本信息             | 0行
timescaledb_adapter               | 历史K线数据                  | 0行
trading_calendar                  | 交易日判断                   | 0行

需要新增的模块：

模块                              | 功能                        | 预估代码量
----------------------------------|----------------------------|----------
sim_portfolio_service.py          | 虚拟组合管理（CRUD）          | ~150行
sim_execution_engine.py           | 每日调度+模拟成交             | ~300行
sim_performance_calculator.py     | 绩效指标计算                 | ~200行
sim_comparison_service.py         | 多组合横向对比                | ~100行
routers/sim_portfolio.py          | API端点                     | ~150行
init_sim_schema.py                | 3张新表DDL                  | ~80行
```

### 16.10 投资回报分析

#### 16.10.1 开发成本估算

```
总新增代码量：~980行 Python + ~80行 SQL = ~1060行

开发工时估算：
  · 数据库表设计 + DDL：0.5天
  · sim_portfolio_service（组合CRUD）：1天
  · sim_execution_engine（核心调度+模拟成交）：2天
  · sim_performance_calculator（绩效计算）：1.5天
  · sim_comparison_service（横向对比）：0.5天
  · API端点 + 前端页面：2天
  · 测试 + 调试：1.5天
  ─────────────────────────
  总计：~9天（1个人）

基础设施依赖：
  · PostgreSQL：已有 ✓
  · miniQMT 行情：已有 ✓
  · 选股服务：已有 ✓
  · 定时任务框架：已有（APScheduler）✓
  · 无需额外硬件或第三方服务
```

#### 16.10.2 回报价值分析

```
直接价值：

1. 真实交易决策支持（最核心价值）
   · 模拟盘运行 1-3 个月后，积累足够数据
   · 可以直接作为手工交易的参考依据
   · 多组合对比帮助选择最优配置
   · 避免"盲目相信回测"导致的亏损

2. 过拟合检测（避免损失）
   · 回测年化 30% 但实盘年化 5% → 明确的过拟合信号
   · 及时发现问题，避免在错误方向上继续演进
   · 假设避免一次重大过拟合决策 → 节省数周演进时间

3. QE 演进质量提升
   · 实盘数据反馈让演进方向更贴近真实市场
   · 从"优化回测指标"转向"优化实盘收益"
   · 长期来看，这是整个系统最有价值的升级
```

#### 16.10.3 与对接 QMT 模拟盘的成本对比

```
                    | 自建模拟盘          | 对接 QMT 模拟盘
--------------------|--------------------|-----------------
开发工时            | ~9天               | ~15-20天
维护成本            | 低（纯数据库操作）   | 高（网络/重连/状态同步）
多组合支持          | 无限制              | 受限于QMT账户数
统计自由度          | 完全自定义           | 受限于QMT接口
QE反馈集成          | 天然集成             | 需要适配层
实时性              | T+1（收盘后计算）    | 实时
真实撮合            | 模拟（收盘价/VWAP）  | 模拟（QMT撮合引擎）
可靠性              | 高（无外部依赖）     | 中（依赖QMT客户端）

结论：自建模拟盘的投入产出比显著优于对接 QMT
```

### 16.11 改进建议

#### 16.11.1 分阶段实施路径

```
Phase 1（MVP，3-4天）：最小可用版本
  · 3张数据库表
  · sim_portfolio_service：创建/查询组合
  · sim_execution_engine：每日收盘后自动选股+模拟成交
  · 成交价使用收盘价，固定成本模型
  · 基础净值曲线和收益率计算
  → 产出：可以开始积累实盘数据

Phase 2（完善，3-4天）：统计与对比
  · sim_performance_calculator：完整绩效指标
  · sim_comparison_service：多组合横向对比
  · API端点 + 前端展示页面
  · 实盘IC计算
  → 产出：可以作为交易决策参考

Phase 3（闭环，2天）：QE反馈集成
  · QEFeedback 扩展 live_performance 字段
  · QE Analyst Agent prompt 增加实盘维度
  · 过拟合预警机制
  → 产出：演进方向由实盘数据驱动
```

#### 16.11.2 架构层面的改进建议

```
建议 1: 支持"策略热切换"
  · 允许运行中的虚拟组合切换到新的 SOTA 配置
  · 记录切换时间点，便于前后对比
  · 实现：sim_portfolio 增加 config_history JSONB 字段

建议 2: 基准组合自动创建
  · 每个虚拟组合自动创建对应的基准组合
  · 基准策略：等权买入沪深300成分股 / 中证500成分股
  · 所有统计自动计算超额收益

建议 3: 异常交易日处理
  · 停牌股票：保持持仓不变，不计入当日换仓
  · 涨跌停：涨停无法买入，跌停无法卖出
  · 新股上市首日：排除在选股范围外
  · ST 股票：可配置是否排除

建议 4: 组合到期与归档
  · 设置组合运行期限（如 3 个月）
  · 到期后自动归档，生成最终报告
  · 归档数据保留，可随时查阅

建议 5: 数据质量保障
  · 每日调度前检查 miniQMT 连接状态
  · 行情数据缺失时降级到 TimescaleDB 历史数据
  · 记录数据来源（miniQMT/TDX/TimescaleDB），便于排查
  · 节假日/非交易日自动跳过

建议 6: 与手工交易的协同
  · 模拟盘每日生成"交易建议单"（买入/卖出列表+理由）
  · 支持导出为 CSV/Excel，方便手工下单参考
  · 记录"建议 vs 实际执行"的偏差（可选）
  · 长期积累后可分析"人工干预"对收益的影响
```

### 16.12 模拟盘方案总结

| 维度 | 评估 |
|------|------|
| 方案可行性 | ★★★★★ 高度可行，现有基础设施复用率 > 80% |
| 开发成本 | ~1060行代码，~9天工时，无额外硬件 |
| 核心价值 | 交易决策支持 + 过拟合检测 + QE演进质量提升 |
| 投入产出比 | ★★★★★ 所有优化方向中最高之一 |
| vs QMT模拟盘 | 开发快、维护低、统计自由、多组合无限制 |
| 数据底座 | 完全复用选股服务 + miniQMT行情 + TimescaleDB |
| 日内执行 | Phase 1 用收盘价/VWAP，后期可选分钟级模拟 |
| QE反馈集成 | 天然集成，扩展 QEFeedback 即可 |
| 建议优先级 | ★★★★★ 高优先级，建议在策略参数优化之后立即实施 |

**核心结论**：在 AIstock 侧自建轻量模拟盘是当前投入产出比最高的优化方向之一。它不仅为手工交易提供直接的决策参考，更重要的是打通了"回测→实盘"的验证闭环，让整个 QE 演进系统从"优化回测指标"转向"优化真实收益"。建议作为高优先级项目，在策略参数可配置化完成后立即启动开发。

---

## 十七、Tushare 数据集统一补齐方案

> 实盘选股的前提条件：确保所有依赖数据集每日自动更新到最新交易日。

### 17.1 现状诊断：数据补齐的断层

当前 AIstock 的数据补齐能力存在明显的"TDX 完善、Tushare 缺失"的不对称：

| 数据源 | 代表数据集 | 一键补齐 | 增量同步 | 自动调度 | 状态 |
|--------|-----------|---------|---------|---------|------|
| TDX (Go) | kline_daily_raw | ✅ | ✅ | ✅ | 生产就绪 |
| Tushare | daily_basic | ❌ | ⚠️ 仅后端 | ❌ | 半成品 |
| Tushare | stock_moneyflow_ts | ❌ | ⚠️ 仅后端 | ⚠️ 有默认计划 | 半成品 |
| Tushare | adj_factor | ❌ | ❌ | ❌ | 未实现 |
| Tushare | bak_basic | ❌ | ❌ | ❌ | 未实现 |
| Tushare | stock_basic | ❌ | ❌ | ❌ | 未实现 |
| Tushare | stock_st | ❌ | ❌ | ❌ | 未实现 |
| Tushare | cyq_chips | ❌ | ❌ | ❌ | 未实现 |

**关键问题**：

1. **后端只实现了 2/7 的同步方法**：`TushareSyncService` 仅有 `sync_daily_basic()` 和 `sync_money_flow()`，其余 5 个数据集完全没有同步逻辑
2. **无速率控制**：现有同步方法直接调用 Tushare API，无 rate limiting、无重试、无批次控制
3. **无统一模块**：每个数据集的同步逻辑分散在不同方法中，代码重复且不可扩展
4. **前后端未打通**：前端 `handleCatchUp` 仅支持 TDX 数据集的 catch-up 流程，Tushare 数据集无法触发一键补齐
5. **`tushare_adapter.py` 是空壳**：仅定义了 `fetch_history_window_tushare()` 但直接 `raise NotImplementedError`

### 17.2 Tushare API 规格与限制

各数据集的 Tushare 接口差异较大，统一模块必须感知这些差异：

| 数据集 | Tushare 接口 | 主要参数 | 目标表 | 主键 | 单次上限 | 频率限制 | 增量模式 |
|--------|-------------|---------|--------|------|---------|---------|---------|
| daily_basic | `pro.daily_basic()` | trade_date | market.daily_basic | ts_code + trade_date | ~5000条/次 | 200次/分 | 按日期 |
| stock_moneyflow_ts | `pro.moneyflow()` | trade_date | market.moneyflow_ts | ts_code + trade_date | ~5000条/次 | 200次/分 | 按日期 |
| adj_factor | `pro.adj_factor()` | trade_date / ts_code | market.adj_factor | ts_code + trade_date | ~5000条/次 | 200次/分 | 按日期 |
| bak_basic | `pro.bak_basic()` | trade_date | market.bak_basic | ts_code + trade_date | ~5000条/次 | 120次/分 | 按日期 |
| stock_basic | `pro.stock_basic()` | list_status | market.stock_basic | ts_code | 全量~5000 | 200次/分 | 全量覆盖 |
| stock_st | `pro.namechange()` 或自定义 | start_date/end_date | market.stock_st | ts_code + start_date | ~5000条/次 | 200次/分 | 按日期范围 |
| cyq_chips | `pro.cyq_perf()` / `pro.cyq_chips()` | trade_date + ts_code | market.cyq_chips | ts_code + trade_date | ~1000条/次 | 60次/分 | 按股票+日期 |

**关键差异点**：

```
1. stock_basic 是全量接口，不支持增量，每次全量覆盖即可
2. cyq_chips 频率限制最严格（60次/分），且必须按单只股票查询
3. bak_basic 频率限制较低（120次/分），需要更大的请求间隔
4. 大部分接口支持 trade_date 参数，可按日期批量获取
5. 所有接口的积分要求不同，需要根据用户的 Tushare 积分等级调整策略
```

### 17.3 统一同步模块设计：TushareSyncEngine

核心思路：将所有 Tushare 数据集的同步逻辑抽象为一个统一引擎，通过**数据集配置描述符**驱动，而非为每个数据集写独立方法。

#### 17.3.1 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                  TushareSyncEngine                       │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ DatasetSpec  │  │ RateLimiter  │  │ BatchScheduler│  │
│  │ (配置描述符)  │  │ (速率控制器)  │  │ (批次调度器)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│  ┌──────▼─────────────────▼──────────────────▼───────┐  │
│  │              sync_dataset(spec, params)            │  │
│  │                                                    │  │
│  │  1. 检测当前最大日期 (detect_max_date)             │  │
│  │  2. 计算补齐范围 (calc_gap_range)                  │  │
│  │  3. 生成批次计划 (generate_batches)                │  │
│  │  4. 逐批获取+写入 (fetch_and_upsert)              │  │
│  │  5. 更新元数据 (update_data_stats)                 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │              _upsert_to_pg() (复用现有)             │  │
│  │  temp table → COPY FROM → ON CONFLICT DO UPDATE    │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 17.3.2 数据集配置描述符（DatasetSpec）

每个数据集通过一个配置字典完整描述其同步行为，无需编写独立方法：

```python
DATASET_SPECS = {
    "daily_basic": DatasetSpec(
        tushare_api="daily_basic",          # Tushare 接口名
        target_table="market.daily_basic",  # 目标表
        primary_keys=["ts_code", "trade_date"],  # 主键（用于 upsert）
        date_column="trade_date",           # 日期列（用于检测最大日期）
        query_mode="by_date",               # 查询模式：by_date / by_symbol / full_replace
        query_param="trade_date",           # 传给 Tushare 的日期参数名
        rate_limit=200,                     # 每分钟最大请求次数
        batch_size=1,                       # 每批处理的天数（by_date 模式）
        retry_times=3,                      # 失败重试次数
        retry_delay=5.0,                    # 重试间隔（秒）
        column_mapping=None,                # 列名映射（None=直接使用）
        post_process=None,                  # 后处理函数（可选）
    ),
    "stock_moneyflow_ts": DatasetSpec(
        tushare_api="moneyflow",
        target_table="market.moneyflow_ts",
        primary_keys=["ts_code", "trade_date"],
        date_column="trade_date",
        query_mode="by_date",
        query_param="trade_date",
        rate_limit=200,
        batch_size=1,
        retry_times=3,
        retry_delay=5.0,
    ),
    "adj_factor": DatasetSpec(
        tushare_api="adj_factor",
        target_table="market.adj_factor",
        primary_keys=["ts_code", "trade_date"],
        date_column="trade_date",
        query_mode="by_date",
        query_param="trade_date",
        rate_limit=200,
        batch_size=1,
        retry_times=3,
        retry_delay=5.0,
    ),
    "bak_basic": DatasetSpec(
        tushare_api="bak_basic",
        target_table="market.bak_basic",
        primary_keys=["ts_code", "trade_date"],
        date_column="trade_date",
        query_mode="by_date",
        query_param="trade_date",
        rate_limit=120,                     # 较低的频率限制
        batch_size=1,
        retry_times=3,
        retry_delay=5.0,
    ),
    "stock_basic": DatasetSpec(
        tushare_api="stock_basic",
        target_table="market.stock_basic",
        primary_keys=["ts_code"],
        date_column=None,                   # 无日期列，全量覆盖
        query_mode="full_replace",          # 全量替换模式
        query_param=None,
        rate_limit=200,
        batch_size=1,
        retry_times=3,
        retry_delay=5.0,
        extra_params={"list_status": "L"},  # 额外参数
    ),
    "stock_st": DatasetSpec(
        tushare_api="namechange",           # 或自定义查询
        target_table="market.stock_st",
        primary_keys=["ts_code", "start_date"],
        date_column="start_date",
        query_mode="by_date_range",         # 按日期范围查询
        query_param="start_date",
        rate_limit=200,
        batch_size=30,                      # 每批30天
        retry_times=3,
        retry_delay=5.0,
    ),
    "cyq_chips": DatasetSpec(
        tushare_api="cyq_perf",
        target_table="market.cyq_chips",
        primary_keys=["ts_code", "trade_date"],
        date_column="trade_date",
        query_mode="by_symbol_date",        # 按股票+日期查询（最严格）
        query_param="trade_date",
        rate_limit=60,                      # 最严格的频率限制
        batch_size=1,
        retry_times=5,                      # 更多重试
        retry_delay=10.0,                   # 更长间隔
        symbol_batch_size=50,               # 每批处理50只股票
    ),
}
```

#### 17.3.3 核心引擎伪代码

```python
class TushareSyncEngine:
    """统一 Tushare 数据集同步引擎"""

    def __init__(self, tushare_token: str):
        self.pro = ts.pro_api(tushare_token)
        self._rate_limiters: Dict[int, RateLimiter] = {}  # 按频率限制分组

    def sync_dataset(
        self,
        dataset_id: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        auto_detect: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> SyncResult:
        """
        统一入口：同步指定数据集

        Args:
            dataset_id: 数据集标识（如 "daily_basic"）
            start_date: 起始日期（YYYYMMDD），None 则自动检测
            end_date: 截止日期，None 则取最新交易日
            auto_detect: 是否自动检测当前最大日期
            progress_callback: 进度回调 fn(current, total, message)
        """
        spec = DATASET_SPECS[dataset_id]
        limiter = self._get_rate_limiter(spec.rate_limit)

        # Step 1: 确定补齐范围
        if auto_detect and start_date is None:
            start_date = self._detect_gap_start(spec)
        if end_date is None:
            end_date = self._get_latest_trade_date()

        # Step 2: 生成批次计划
        batches = self._generate_batches(spec, start_date, end_date)

        # Step 3: 逐批执行
        result = SyncResult(dataset_id=dataset_id, total_batches=len(batches))
        for i, batch in enumerate(batches):
            limiter.wait()  # 速率控制
            try:
                df = self._fetch_batch(spec, batch)
                if df is not None and not df.empty:
                    self._upsert_to_pg(df, spec.target_table, spec.primary_keys)
                    result.rows_synced += len(df)
                result.batches_done += 1
            except Exception as e:
                result.errors.append({"batch": batch, "error": str(e)})
                if not self._retry_batch(spec, batch, limiter):
                    result.failed_batches += 1

            if progress_callback:
                progress_callback(i + 1, len(batches), f"Batch {i+1}/{len(batches)}")

        # Step 4: 更新元数据
        self._update_data_stats(spec, result)
        return result

    def _detect_gap_start(self, spec: DatasetSpec) -> str:
        """检测数据集当前最大日期，返回下一个交易日作为补齐起点"""
        if spec.query_mode == "full_replace":
            return None  # 全量模式无需检测

        sql = f"SELECT MAX({spec.date_column}) FROM {spec.target_table}"
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
        if row and row[0]:
            max_date = row[0]
            # 查询下一个交易日
            return self._next_trade_date(max_date)
        return "20200101"  # 默认起点

    def _generate_batches(self, spec, start_date, end_date) -> List[dict]:
        """根据查询模式生成批次"""
        if spec.query_mode == "full_replace":
            return [{"mode": "full"}]
        elif spec.query_mode == "by_date":
            # 获取 [start_date, end_date] 之间的所有交易日
            trade_dates = self._get_trade_dates(start_date, end_date)
            return [{"trade_date": d} for d in trade_dates]
        elif spec.query_mode == "by_symbol_date":
            # cyq_chips 等需要按股票+日期组合
            trade_dates = self._get_trade_dates(start_date, end_date)
            symbols = self._get_all_symbols()
            batches = []
            for d in trade_dates:
                for chunk in chunked(symbols, spec.symbol_batch_size):
                    batches.append({"trade_date": d, "symbols": chunk})
            return batches
        # ... 其他模式
```

#### 17.3.4 现有脚本的实际循环策略（必须保留）

经分析 `scripts/ingest_tushare_*.py` 各脚本的实际实现，发现不同数据集采用了截然不同的循环方式，这些方式是基于 Tushare 接口限制精心设计的，且已验证可正常工作。统一引擎必须保留这些模式：

| 数据集 | 循环方式 | 实际脚本逻辑 | 关键限制 |
|--------|---------|-------------|---------|
| daily_basic | 按日期循环 | 逐交易日调用 `pro.daily_basic(trade_date=YYYYMMDD)`，每日返回全市场~5000条 | 单次≤5000条，无需分页 |
| adj_factor | 按日期循环 | 逐交易日调用 `pro.adj_factor(trade_date=YYYYMMDD)`，retry 3次，batch_sleep=0.1s | 同上 |
| stock_moneyflow_ts | 按日期循环 | 逐交易日调用 `pro.moneyflow(trade_date=YYYYMMDD)` | 同上 |
| bak_basic | 按日期循环 | 逐交易日调用，支持 batch_sleep 参数 | 频率限制较低(120次/分) |
| stock_st | 按日期循环 | 逐交易日调用，支持 batch_sleep | 同 daily_basic |
| stock_basic | 全量单次 | 单次调用 `pro.stock_basic(list_status='L')`，返回全部~5000只股票，ON CONFLICT upsert | 维度表，非时序数据 |
| cyq_perf | 按日期+分页 | 逐交易日调用，但单日>5000条需分页（limit=4900, offset递增） | 单次≤5000条，需2页 |
| cyq_chips | **按股票循环** | 遍历全部~5000只股票，每只调用 `pro.cyq_chips(ts_code, start_date, end_date)`，分页(limit=1900) | **与其他数据集完全不同**，sleep=0.12s/股票 |

**关键发现**：

```
1. 大部分数据集：外层按日期循环，每日获取全市场数据（by_date 模式）
2. cyq_chips 例外：外层按股票循环，每只股票获取一段时间的数据（by_symbol 模式）
   原因：cyq_chips 接口必须指定 ts_code，不支持按日期获取全市场
3. cyq_perf 需要分页：单日全市场约6000条 > 单次限制5000条
4. stock_basic 是维度表：全量覆盖，无日期循环
5. 所有脚本都已接入 ingestion_jobs/ingestion_logs 进度追踪
6. 所有脚本都支持 --job-id 参数，可被前端任务系统管理
```

**对统一引擎的设计约束**：

DatasetSpec 的 `query_mode` 必须支持以下四种模式，且一键补齐时直接调用对应的现有脚本，而非重新实现循环逻辑：

```python
class QueryMode(Enum):
    BY_DATE = "by_date"              # 外层按日期循环（大部分数据集）
    BY_DATE_PAGED = "by_date_paged"  # 按日期循环 + 分页（cyq_perf）
    BY_SYMBOL = "by_symbol"          # 外层按股票循环（cyq_chips）
    FULL_REPLACE = "full_replace"    # 全量单次获取（stock_basic）
```

#### 17.3.5 速率控制器（RateLimiter）

Tushare 不同接口的频率限制差异很大（60-200次/分），需要一个通用的令牌桶限速器：

```python
class TushareRateLimiter:
    """令牌桶速率控制器，按每分钟请求数限制"""

    def __init__(self, max_per_minute: int, safety_margin: float = 0.8):
        self.interval = 60.0 / (max_per_minute * safety_margin)  # 请求间隔（秒）
        self._last_request_time = 0.0

    def wait(self):
        """阻塞等待直到可以发送下一个请求"""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_request_time = time.monotonic()
```

**设计要点**：
- `safety_margin=0.8` 预留 20% 余量，避免触发 Tushare 封禁
- cyq_chips（60次/分）实际间隔 = 60/(60×0.8) = 1.25秒/请求
- daily_basic（200次/分）实际间隔 = 60/(200×0.8) = 0.375秒/请求
- 不同频率限制的数据集使用独立的 limiter 实例，互不干扰

### 17.4 一键补齐流程设计

#### 17.4.1 端到端流程

```
用户点击"一键补齐"
       │
       ▼
┌─────────────────────────────────┐
│ 前端: handleTushareCatchUp()    │
│ POST /api/ingestion/tushare/    │
│      catch-up/{dataset_id}      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 后端: detect_gap()              │
│ 1. 查询 MAX(date_column)       │
│ 2. 查询最新交易日               │
│ 3. 计算 gap_start, gap_end     │
│ 4. 估算批次数和耗时             │
└──────────────┬──────────────────┘
               │
               ▼
       ┌───────┴───────┐
       │  gap > 0 ?    │
       └───┬───────┬───┘
          YES      NO
           │        │
           ▼        ▼
  返回补齐计划    返回"已是最新"
  {gap_start,
   gap_end,
   est_batches,
   est_minutes}
           │
           ▼
  ┌────────────────────────┐
  │ 方式A: 自动跳转增量卡片 │  ← 前端自动填写起止日期
  │ 方式B: 直接后台执行     │  ← 后端异步执行同步任务
  └────────────────────────┘
```

#### 17.4.2 两种执行模式

**模式 A：跳转增量卡片（交互式）**
- 适用于首次补齐或大范围补齐（gap > 30天）
- 前端收到 `action: "redirect_to_incremental"` 后自动切换到增量Tab
- 自动填写 `IncrementalPrefill`：dataSource=tushare, dataset, startDate, targetDate
- 用户可调整参数后手动触发

**模式 B：直接后台执行（一键式）**
- 适用于日常小范围补齐（gap ≤ 30天）
- 前端收到 `action: "execute"` 后显示进度条
- 后端通过 `TushareSyncEngine.sync_dataset()` 异步执行
- 通过 SSE 或轮询返回进度

### 17.5 后端 API 端点设计

在现有 `ingestion.py` 路由基础上扩展，新增 3 个 Tushare 专用端点：

#### 17.5.1 检测补齐范围

```
GET /api/ingestion/tushare/gap/{dataset_id}

Response:
{
  "dataset_id": "daily_basic",
  "current_max_date": "2026-02-20",
  "latest_trade_date": "2026-02-26",
  "gap_days": 4,
  "gap_trade_days": 4,
  "gap_start": "2026-02-21",
  "gap_end": "2026-02-26",
  "est_batches": 4,
  "est_minutes": 0.1,
  "status": "needs_catch_up"   // needs_catch_up | up_to_date | no_data
}
```

复用现有 `/api/ingestion/auto-range` 的逻辑，但针对 Tushare 数据集做以下增强：
- 从 `DatasetSpec` 获取 `date_column` 和 `target_table`，而非硬编码
- 考虑 `query_mode`：`full_replace` 模式返回 `status: "full_replace_needed"`
- 估算耗时基于 `rate_limit` 和 `batch_size` 计算

#### 17.5.2 执行补齐

```
POST /api/ingestion/tushare/sync/{dataset_id}

Request Body:
{
  "start_date": "20260221",    // 可选，默认自动检测
  "end_date": "20260226",      // 可选，默认最新交易日
  "batch_size": 1,             // 可选，覆盖默认值
  "rate_limit": null,          // 可选，覆盖默认值
  "async": true                // 是否异步执行
}

Response (async=true):
{
  "task_id": "tushare_sync_daily_basic_20260227_001",
  "status": "running",
  "est_minutes": 0.1
}
```

#### 17.5.3 查询同步进度

```
GET /api/ingestion/tushare/progress/{task_id}

Response:
{
  "task_id": "tushare_sync_daily_basic_20260227_001",
  "status": "running",         // running | completed | failed
  "progress": 0.75,
  "batches_done": 3,
  "total_batches": 4,
  "rows_synced": 15234,
  "errors": [],
  "elapsed_seconds": 2.1
}
```

### 17.6 前端集成设计

#### 17.6.1 数据看板扩展

在现有 `local-data/page.tsx` 的数据集卡片上，为 Tushare 数据集增加一键补齐按钮：

```
┌─────────────────────────────────────────────────┐
│  daily_basic                    [Tushare]       │
│  最新日期: 2026-02-20  │  记录数: 1,234,567    │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ 查看详情  │  │ 增量同步  │  │ ⚡ 一键补齐   │ │
│  └──────────┘  └──────────┘  └───────────────┘ │
│  ⚠️ 落后 4 个交易日                              │
└─────────────────────────────────────────────────┘
```

#### 17.6.2 handleTushareCatchUp 流程

```typescript
async function handleTushareCatchUp(datasetId: string) {
  // 1. 调用 gap 检测接口
  const gap = await fetch(`/api/ingestion/tushare/gap/${datasetId}`);

  if (gap.status === "up_to_date") {
    toast.success("数据已是最新");
    return;
  }

  if (gap.gap_trade_days > 30) {
    // 大范围补齐 → 跳转增量卡片，自动填写参数
    setIncrementalPrefill({
      dataSource: "tushare",
      dataset: datasetId,
      startDate: gap.gap_start,
      targetDate: gap.gap_end,
      currentMaxDate: gap.current_max_date,
    });
    setActiveTab("incremental");
    return;
  }

  // 小范围补齐 → 直接后台执行
  const result = await fetch(`/api/ingestion/tushare/sync/${datasetId}`, {
    method: "POST",
    body: JSON.stringify({ async: true }),
  });

  // 轮询进度
  pollProgress(result.task_id, (progress) => {
    updateDatasetCard(datasetId, { syncProgress: progress });
  });
}
```

#### 17.6.3 批量补齐入口

在数据看板顶部增加"全部补齐"按钮，一键触发所有 Tushare 数据集的补齐：

```
┌─────────────────────────────────────────────────────┐
│  本地数据看板                                        │
│  ┌──────────────────┐  ┌─────────────────────────┐  │
│  │ ⚡ Tushare 全部补齐 │  │ 上次全量补齐: 2026-02-20 │  │
│  └──────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

按依赖顺序串行执行：`stock_basic` → `adj_factor` → `daily_basic` → `stock_moneyflow_ts` → `bak_basic` → `stock_st` → `cyq_chips`

### 17.7 任务调度集成

#### 17.7.1 与现有调度体系的关系

AIstock 已有 `ingestion_schedules` 表管理数据同步计划，当前仅配置了 2 个 Tushare 默认计划：

```sql
-- 现有默认计划（init_trading_schema.py）
INSERT INTO ingestion_schedules (dataset, frequency, ...) VALUES
  ('stock_moneyflow', 'daily', ...),
  ('stock_moneyflow_ts', 'daily', ...);
```

需要扩展为 7 个数据集的完整调度计划。

#### 17.7.2 调度配置设计

```sql
-- 新增 Tushare 数据集调度计划
INSERT INTO ingestion_schedules
  (dataset, source, frequency, cron_expr, enabled, config_json)
VALUES
  -- 基础数据（每日 16:30 开始，收盘后30分钟）
  ('stock_basic',          'tushare', 'weekly',  '0 30 16 * * 1', true,
   '{"mode":"full_replace"}'),
  ('adj_factor',           'tushare', 'daily',   '0 30 16 * * *', true,
   '{"mode":"incremental","batch_size":1}'),
  ('daily_basic',          'tushare', 'daily',   '0 35 16 * * *', true,
   '{"mode":"incremental","batch_size":1}'),
  ('stock_moneyflow_ts',   'tushare', 'daily',   '0 40 16 * * *', true,
   '{"mode":"incremental","batch_size":1}'),
  ('bak_basic',            'tushare', 'daily',   '0 45 16 * * *', true,
   '{"mode":"incremental","batch_size":1}'),
  ('stock_st',             'tushare', 'daily',   '0 50 16 * * *', true,
   '{"mode":"incremental","batch_size":1}'),
  ('cyq_chips',            'tushare', 'daily',   '0 55 16 * * *', true,
   '{"mode":"incremental","symbol_batch_size":50}');
```

#### 17.7.3 调度执行流程

```
每日 16:30 触发
       │
       ▼
┌──────────────────────────────┐
│ SchedulerService.run_due()   │
│ 1. 查询 enabled=true 的计划  │
│ 2. 检查 cron_expr 是否到期   │
│ 3. 检查当日是否为交易日       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 按 cron 顺序串行执行         │
│ stock_basic (16:30)          │
│   → adj_factor (16:35)       │
│   → daily_basic (16:35)      │
│   → moneyflow_ts (16:40)    │
│   → bak_basic (16:45)       │
│   → stock_st (16:50)        │
│   → cyq_chips (16:55)       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 每个数据集调用:               │
│ TushareSyncEngine            │
│   .sync_dataset(auto_detect) │
│ 更新 last_run_at, status     │
└──────────────────────────────┘
```

**错开 5 分钟的原因**：避免多个数据集同时请求 Tushare API 导致触发全局频率限制。

### 17.8 现有基础设施复用清单

| 现有组件 | 位置 | 复用方式 | 改动量 |
|---------|------|---------|-------|
| `_upsert_to_pg()` | tushare_sync.py | 直接复用，temp table + COPY FROM + ON CONFLICT | 零改动 |
| `data_stats_config` 表 | init_trading_schema.py | 复用元数据查询，获取数据集配置 | 零改动 |
| `data_stats` 表 | init_trading_schema.py | 复用统计信息存储 | 零改动 |
| `/api/ingestion/auto-range` | ingestion.py | 参考逻辑，新建 Tushare 专用端点 | 参考复用 |
| `_infer_source()` | ingestion.py | 复用数据源判断逻辑 | 零改动 |
| `trading_calendar` 表 | market schema | 复用交易日判断 | 零改动 |
| `ingestion_schedules` 表 | init_trading_schema.py | 扩展新增 Tushare 调度记录 | 新增记录 |
| `handleCatchUp()` | local-data/page.tsx | 扩展支持 Tushare 数据集 | 小改动 |
| `IncrementalPrefill` | local-data/page.tsx | 直接复用自动填写机制 | 零改动 |
| `IncrementalTab` | local-data/page.tsx | 复用增量同步 UI 组件 | 零改动 |

**复用率估算**：~60% 的逻辑可直接复用或参考现有实现，核心新增工作集中在 `TushareSyncEngine` 和 `DatasetSpec` 配置。

### 17.9 实施成本估算

#### 17.9.1 开发工作量

| 模块 | 代码量估算 | 工时 | 说明 |
|------|-----------|------|------|
| `DatasetSpec` 配置 | ~120行 | 0.5天 | 7个数据集的配置描述符 |
| `TushareSyncEngine` 核心 | ~250行 | 2天 | 统一引擎 + 批次调度 + 速率控制 |
| `TushareRateLimiter` | ~40行 | 0.5天 | 令牌桶限速器 |
| 后端 API 端点 (3个) | ~180行 | 1天 | gap检测 + 执行同步 + 进度查询 |
| 前端 handleTushareCatchUp | ~80行 | 1天 | 一键补齐 + 批量补齐 + 进度展示 |
| 调度计划配置 | ~30行 SQL | 0.5天 | 7个数据集的 cron 调度记录 |
| 测试 + 联调 | - | 1.5天 | 各数据集实际补齐验证 |
| **合计** | **~700行** | **~7天** | |

#### 17.9.2 与模拟盘的依赖关系

```
Tushare 数据补齐 (本节)     ← 前置条件
       │
       ▼
实盘选股功能 (Section 12)   ← 依赖完整数据
       │
       ▼
模拟盘平台 (Section 16)     ← 依赖选股结果
```

数据补齐是整个实盘链路的第一环，必须先完成。

### 17.10 Tushare 补齐方案总结

| 维度 | 评估 |
|------|------|
| 方案可行性 | ★★★★★ 高度可行，现有 upsert 模式和前端组件直接复用 |
| 开发成本 | ~700行代码，~7天工时 |
| 核心价值 | 实盘选股的前置条件，打通数据→选股→模拟盘全链路 |
| 投入产出比 | ★★★★★ 不做则后续所有实盘功能无法启动 |
| 统一引擎优势 | 新增数据集只需添加 DatasetSpec 配置，零代码开发 |
| 速率控制 | 令牌桶限速器，按数据集独立控制，预留 20% 安全余量 |
| 调度集成 | 复用 ingestion_schedules 表，每日收盘后自动补齐 |
| 扩展性 | 未来新增 Tushare 数据集（如 index_daily、fund_nav）仅需配置 |
| 建议优先级 | ★★★★★ 最高优先级，应在所有实盘功能之前完成 |

**核心结论**：Tushare 数据集统一补齐是实盘选股的"地基工程"。通过 DatasetSpec 配置驱动的统一引擎设计，不仅解决当前 7 个数据集的补齐需求，更建立了可扩展的 Tushare 数据管理框架。建议作为最高优先级任务，在实盘选股和模拟盘开发之前完成。

---

## 十八、RDAgent Coding 阶段效率优化（阻断性问题）

> 本节基于 2026-02-26 任务（6个Loop）的实际运行日志分析，属于阻断性问题，需最高优先级解决。
> 约束条件：不修改 RDAgent 程序代码，仅通过提示词和配置调整解决。

### 18.1 问题严重性评估

当前 RDAgent Coding 阶段的效率问题已构成系统瓶颈：

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| Loop 有效产出率 | 4/6 = 66.7% | > 90% |
| 平均 evo 轮数 | 5.8 轮/Loop | < 3 轮/Loop |
| 无效迭代浪费 | 19 轮（Loop_4: 10轮 + Loop_5: 9轮） | 0 |
| Token 浪费 | ~38,000 token/任务 | < 5,000 |

**影响链路**：Coding 死循环 → 无法进入 Running → 无回测结果 → QE 演进停滞 → 整个系统空转。

### 18.2 三层根因分析

#### 18.2.1 根因一：因子假设阶段的 ML 引导矛盾

两处提示词对"因子脚本中是否可以使用 ML"给出了相反的指令：

```
【鼓励 ML】prompts.yaml:106 (factor_hypothesis_specification 第3条)
  "Introduce more complex factors (e.g. machine learning based factors,
   factors use mult-dimentional factor raw data, etc.) as more
   experimental results are gathered."

【禁止 ML】quant_proposal.py:97 (RAG 动态注入)
  "Do NOT use machine learning training in factor scripts."
```

**矛盾机制**：specification 是结构化的 8 条规范文档，RAG 是动态注入的一句话。LLM 在第三轮因子实验时，看到 spec 第 3 条"后续轮次可引入 ML 因子"的引导，选择了 LightGBM 方案。RAG 的禁止指令被 spec 的鼓励指令覆盖。

**实际后果（Loop_5）**：假设要求"基于 LightGBM 的特征重要性加权因子"，9 轮 coding 全部失败——2 轮 LightGBM 调用直接报错，7 轮执行成功但实现不完整被 critic 拒绝。因子脚本的设计边界是 `DataFrame in → 数学变换 → DataFrame out`，嵌入 ML 训练超出了框架能力。

#### 18.2.2 根因二：模型 Critic 做"文字对比"而非"功能验证"

模型 Critic 的提示词（`model_coder/prompts.yaml:101`）要求：

```
"Your job is to check whether user's code is align with the model
 information and the scenario."
```

这导致 Critic 逐字对比代码实现与架构描述文本，而非评估功能等价性。

**实际后果（Loop_4）**：架构描述中存在语义矛盾——"learnable importance scores ... adaptively per timestep"。"learnable"暗示静态可学习参数（`nn.Parameter`），"adaptively per timestep"暗示动态的、依赖输入的权重。LLM coder 尝试了 6 种不同实现方式，每种都被 Critic 以不同理由拒绝：

| 实现方式 | Critic 拒绝理由 |
|----------|---------------|
| `nn.Linear(input)` | "不是 learnable 的，是 input-dependent" |
| `nn.Parameter(固定向量)` | "不是 adaptive per timestep" |
| `nn.Parameter(T×F 矩阵)` | "是 static 的，不是 adaptive" |
| 小型网络 per timestep | "不符合描述的方式" |

10 轮代码全部执行成功、输出 shape 正确，但 Critic 每次都说"不 align"→ final_decision 全部 False。

#### 18.2.3 根因三：Final Decision 缺乏"执行成功即推定正确"的兜底

Final Decision 提示词（`model_coder/prompts.yaml:142-144`）的判断逻辑：

```
"If no ground truth value is provided, the implementation is considered
 correct if the code execution is successful AND the code feedback is
 align with the scenario and model description."
```

"code feedback is align"给了 Critic 一票否决权。即使代码执行成功、shape 正确，只要 Critic 说"不 align"，final_decision 就是 False。这在 Loop_4 中造成了灾难性后果。

### 18.3 纯提示词解决方案（不修改程序代码）

#### 18.3.1 P0-A：消除因子 spec 中的 ML 引导矛盾

**修改文件**：`rdagent/scenarios/qlib/prompts.yaml` 第 106 行

**当前内容**：
```
3. Gradual Complexity Increase:
   - Introduce more complex factors (e.g. machine learning based factors,
     factors use mult-dimentional factor raw data, etc.) as more
     experimental results are gathered.
```

**修改为**：
```
3. Gradual Complexity Increase:
   - Introduce more complex factors (e.g. multi-dimensional cross-sectional
     combinations, non-linear ratio structures, conditional ranking signals,
     multi-scale statistical features) as more experimental results are gathered.
   - IMPORTANT: Factor scripts must be pure data transformations
     (DataFrame in → DataFrame out). Do NOT embed model training
     (LightGBM, XGBoost, neural networks, sklearn, etc.) inside
     factor calculation scripts. ML-based feature engineering belongs
     to the MODEL component, not the FACTOR component.
```

**原理**：将 ML 引导替换为具体的非 ML 复杂因子示例，并明确划定因子脚本的能力边界。与 RAG 侧的禁止指令保持一致。

#### 18.3.2 P0-B：Critic 评审从"文字对比"改为"功能验证"

**修改文件**：`rdagent/components/coder/model_coder/prompts.yaml` 第 95-116 行

**当前关键语句**：
```
Your job is to check whether user's code is align with the model
information and the scenario.
```

**修改为**：
```
Your job is to check whether user's code FUNCTIONALLY implements
the model described in the model information.

IMPORTANT evaluation priorities (in order):
1. Code executes successfully and output shape is correct
   → This is the STRONGEST positive signal
2. The model's core computational flow matches the description
   (input → transformation → output)
3. Key architectural components are present (e.g. attention,
   pooling, residual connections)

Do NOT reject code for:
- Minor implementation differences that achieve the same
  mathematical result
- Using a different but functionally equivalent approach
- Stylistic differences in how layers are organized
- The number of FC layers differing by 1 if the overall
  function is equivalent

If the architecture description is ambiguous or contradictory,
accept ANY reasonable interpretation that produces correct output.
```

**原理**：将评审标准从"文字对齐"转向"功能等价"，建立明确的优先级层次（执行成功 > 核心流程 > 组件存在），并显式列出不应拒绝的情况。

#### 18.3.3 P0-C：Final Decision 增加"执行成功即推定正确"

**修改文件**：`rdagent/components/coder/model_coder/prompts.yaml` 第 142-144 行

**当前逻辑**：
```
2. If no ground truth value is not provided, the implementation is
   considered correct if the code execution is successful and the
   code feedback is align with the scenario and model description.
```

**修改为**：
```
2. If no ground truth value is provided, apply these priority rules:
   a. If code execution is successful AND output shape matches
      expected shape, the implementation is PRESUMED correct.
   b. Code feedback alignment is a SECONDARY consideration.
      Minor deviations should NOT override successful execution.
   c. Only reject a successfully-executing implementation if there
      is a FUNDAMENTAL architectural mismatch (e.g., description
      says CNN but code implements RNN).
   d. If the architecture description contains ambiguous or
      contradictory requirements, execution success is DECISIVE.
```

**原理**：将"执行成功"从必要条件提升为"推定正确"的充分条件，Critic 的文字对比降级为辅助参考。对 Loop_4 场景，第 1 轮即可通过。

#### 18.3.4 P1-A：模型架构描述增加精确性约束

**修改文件**：`rdagent/scenarios/qlib/prompts.yaml` 第 147-174 行

**当前 `architecture` 字段定义**：
```json
"architecture": "A detailed description of the model's architecture,
                 e.g., neural network layers or tree structures"
```

**修改为**：
```json
"architecture": "A PRECISE and IMPLEMENTABLE description. Requirements:
  1. Each layer must specify: type, input dim, output dim
  2. Use unambiguous terms: 'static learnable' OR 'input-dependent
     dynamic', never both for the same component
  3. Avoid contradictory modifiers
  4. Format: 'Layer1(in→out) → Layer2(in→out) → Output(1)'
  Example: 'Linear(F→64) → ReLU → TransformerEncoder(d=64,
  nhead=4, layers=2) → AvgPool → Linear(64→1)'"
```

**原理**：从"A detailed description"改为结构化精确描述，明确禁止矛盾修饰词（直接针对 Loop_4 根因），要求每层标注维度使描述可直接翻译为代码。

#### 18.3.5 P1-B：模型假设复杂度递进控制

**修改文件**：`rdagent/scenarios/qlib/prompts.yaml` 第 85-93 行

**当前第 8 条**：
```
8. Use standard libraries for baseline models, but also explore custom
   architecture designs to investigate novel structures. After sufficient
   trials with traditional models, aim for innovation comparable to
   top-tier AI conferences (NeurIPS, ICLR, ICML, SIGKDD, etc.)
```

**修改为**：
```
8. Use standard libraries for baseline models, then gradually explore
   custom architecture designs.
   COMPLEXITY CONTROL RULES:
   - Each new hypothesis should introduce AT MOST ONE new architectural
     component compared to the best-performing previous model.
   - New components must be well-defined and independently testable.
   - Do NOT combine multiple untested innovations in a single hypothesis.
   - Progression example: Linear → +Attention → +Residual → +MultiScale
     (one new component per step, not all at once).
```

**原理**：Loop_4 在 Loop_3 成功基础上一步跳到"多尺度注意力 + 动态加权"两个全新组件，复杂度跳跃过大。限制每次最多引入一个新组件，降低失败概率。

### 18.4 解决方案总结与预期效果

#### 18.4.1 修改清单

| 优先级 | 改进项 | 修改文件 | 解决的问题 | 改动量 |
|--------|--------|----------|-----------|--------|
| P0-A | 消除因子 ML 引导矛盾 | `prompts.yaml:106` | Loop_5 死循环 | ~5行 |
| P0-B | Critic 改为功能验证 | `model_coder/prompts.yaml:95-116` | Loop_4 死循环 | ~15行 |
| P0-C | Final Decision 兜底 | `model_coder/prompts.yaml:142-144` | 执行成功仍被否决 | ~8行 |
| P1-A | 架构描述精确性约束 | `prompts.yaml:147-174` | 模糊描述导致分歧 | ~6行 |
| P1-B | 复杂度递进控制 | `prompts.yaml:85-93` | 一次引入多个新组件 | ~6行 |

**总改动量**：~40 行提示词文本，零代码修改。

#### 18.4.2 预期效果量化

```
应用 P0 改进后：
  Loop_4: 第1轮即可通过（执行成功 → 推定正确），节省9轮
  Loop_5: 不会尝试 ML 因子，假设阶段即被引导到非ML方向
  有效产出率: 4/6 → 5/6 = 83.3%（+16.6%）
  平均 evo 轮数: 5.8 → ~3.5（-40%）

应用全部改进后：
  有效产出率: > 90%
  平均 evo 轮数: < 3
  Token 节省: ~38,000/任务（19轮无效迭代 × ~2000 token/轮）
```

#### 18.4.3 V4 模板兼容性

经验证，V4 模板（`app_tpl/all/v4/rdagent/`）与默认版本在所有关键提示词上功能完全一致：

- `scenarios/qlib/prompts.yaml`：V4 与默认一致，修改 V4 即可
- `model_coder/prompts.yaml`：V4 无覆盖，需在 V4 目录新建覆盖文件
- 模板加载机制按 key 级别回退，V4 中新增的 key 会优先于默认版本

**实施路径**：在 `app_tpl/all/v4/rdagent/components/coder/model_coder/` 下新建 `prompts.yaml`，覆盖 `evaluator_code_feedback` 和 `evaluator_final_feedback` 两个 key 即可。

---

## 十九、全局开发任务评估与实施计划

本节基于前述 18 个章节的全部分析成果，提取所有可执行的开发任务，按优先级、工作量、收益进行统一评估，并给出具体的分阶段实施计划。

### 19.1 全局任务评估对比表

以下表格覆盖文档中所有已识别的开发任务。评估维度说明：
- **优先级**：P0（阻断性/立即）> P1（高价值/本周）> P2（重要/本月）> P3（增强/季度）
- **工作量**：以人天为单位（1人全职）
- **收益评级**：★★★★★（极高）到 ★（低）
- **ROI**：收益/工作量的综合评分，S > A > B > C
- **来源**：对应文档章节编号

#### 一、阻断性修复（Coding 提示词）

| # | 任务 | 优先级 | 工作量 | 收益 | ROI | 来源 | 说明 |
|---|------|--------|--------|------|-----|------|------|
| T01 | P0-A：消除因子 ML 引导矛盾 | P0 | 0.1天 | ★★★★★ | S | §18.3 | 修改 `prompts.yaml:106`，~5行。直接消除 Loop_5 类死循环 |
| T02 | P0-B：Critic 改为功能验证 | P0 | 0.2天 | ★★★★★ | S | §18.3 | 修改 `model_coder/prompts.yaml:95-116`，~15行。消除 Loop_4 类死循环 |
| T03 | P0-C：Final Decision 兜底逻辑 | P0 | 0.1天 | ★★★★★ | S | §18.3 | 修改 `model_coder/prompts.yaml:142-144`，~8行。执行成功即推定正确 |
| T04 | P1-A：架构描述精确性约束 | P1 | 0.1天 | ★★★★ | S | §18.3 | 修改 `prompts.yaml:147-174`，~6行。减少描述歧义导致的 Critic 误判 |
| T05 | P1-B：假设复杂度递进控制 | P1 | 0.1天 | ★★★★ | S | §18.3 | 修改 `prompts.yaml:85-93`，~6行。每次最多引入一个新组件 |

**小计**：0.6 人天，零代码改动，预期有效产出率 66.7% → 90%+

#### 二、RDAgent 演进效果提升（提示词 + 少量配置）

| # | 任务 | 优先级 | 工作量 | 收益 | ROI | 来源 | 说明 |
|---|------|--------|--------|------|-----|------|------|
| T06 | 模型假设维度引导扩展 | P1 | 0.3天 | ★★★★ | S | §2.2 | `model_hypothesis_specification` 增加注意力/正则化/非平稳性等维度引导 |
| T07 | Feedback Prompt 维度分析 | P1 | 0.2天 | ★★★★ | S | §2.3 | 反馈模板增加"哪个维度改进了/退步了"的结构化分析 |
| T08 | 架构输出格式结构化 | P1 | 0.2天 | ★★★ | A | §2.4 | `model_experiment_output_format` 的 architecture 字段改为层级描述 |
| T09 | RAG 知识库扩展 | P2 | 1天 | ★★★ | A | §2.5 | 补充注意力机制、RevIN、PatchTST 等量化模型案例到 RAG |
| T10 | 回测 deal_price 改 VWAP + 滑点 | P1 | 0.2天 | ★★★ | S | §14.2 | `custom_strategy.py` 约 3 行配置改动，零成本提升回测准确性 |
| T11 | 损失函数可配置扩展 | P2 | 2天 | ★★★ | B | §3.1 | RDAgent 代码改动：支持 IC-weighted loss、Rank loss 等 |
| T12 | 优化器/学习率调度可配置 | P2 | 1.5天 | ★★★ | B | §3.2 | RDAgent 代码改动：支持 CosineAnnealing、OneCycleLR 等 |
| T13 | 评估指标扩展 | P2 | 1天 | ★★★ | A | §3.3 | 增加 Rank IC、分组收益、换手率等指标到 Feedback |

**小计**：6.4 人天（其中纯提示词 0.9 天可立即实施）

#### 三、QE 单次实验与演进功能完善

| # | 任务 | 优先级 | 工作量 | 收益 | ROI | 来源 | 说明 |
|---|------|--------|--------|------|-----|------|------|
| T14 | QE 结构化维度引擎 | P2 | 5天 | ★★★★ | A | §4.2 | StructuredModelSpec 替代自由文本，维度化管理模型配置 |
| T15 | QE 维度调度器（Bandit） | P3 | 3天 | ★★★ | B | §4.3 | 基于历史收益自动选择下一轮演进维度 |
| T16 | QE 深度反馈机制 | P2 | 3天 | ★★★★ | A | §4.4 | 多维度分析（信号质量/收益/风险/交易成本）替代单一指标 |
| T17 | QE SOTA 库管理 | P2 | 2天 | ★★★ | A | §4.5 | SOTA 配置版本化存储、对比、回滚 |
| T18 | QE Researcher 新增 strategy_tune | P2 | 2天 | ★★★★ | A | §14.4 | 策略参数纳入演进循环，Researcher Agent 可提出策略调整假设 |
| T19 | 模型集成框架（ensemble） | P3 | 3天 | ★★★ | B | §3.4 | 支持多模型加权集成，提升预测稳定性 |

**小计**：18 人天

#### 四、因子库优化与管理

| # | 任务 | 优先级 | 工作量 | 收益 | ROI | 来源 | 说明 |
|---|------|--------|--------|------|-----|------|------|
| T20 | 单因子 IC 独立计算 | P1 | 2天 | ★★★★★ | A | §9.2 | 每个因子独立计算 IC/ICIR/Rank_IC，写入 factor_catalog |
| T21 | 因子截面相关性去重 | P2 | 2天 | ★★★★ | A | §9.3 | 扩展现有 code-hash 去重，增加统计级相关性检测（corr>0.9 标记冗余） |
| T22 | 因子分组管理 | P2 | 1.5天 | ★★★ | A | §9.4 | 按类型（动量/价值/波动/流动性等）自动分组，支持组内去重 |
| T23 | 因子生命周期管理 | P2 | 3天 | ★★★★ | A | §9.5 | candidate→active→degraded→archived 状态机 + 自动降级规则 |
| T24 | 月度因子库维护定时任务 | P3 | 1天 | ★★★ | A | §9.6 | 每月自动重算 IC、清理过期因子、生成维护报告 |

**小计**：9.5 人天

#### 五、Tushare 数据自动同步与 QE 选股

| # | 任务 | 优先级 | 工作量 | 收益 | ROI | 来源 | 说明 |
|---|------|--------|--------|------|-----|------|------|
| T25 | TushareSyncEngine 统一引擎 | P1 | 3天 | ★★★★★ | A | §17.3 | DatasetSpec 配置驱动，支持 4 种 QueryMode（by_date/by_symbol/full_replace/by_date_paged） |
| T26 | 7 个关键数据集 DatasetSpec 配置 | P1 | 2天 | ★★★★★ | A | §17.3 | daily_basic、moneyflow、adj_factor、bak_basic、stock_basic、stock_st、cyq_chips |
| T27 | 前端一键补齐集成 | P1 | 1.5天 | ★★★★ | A | §17.4 | 数据看板增加一键补齐按钮 + 批量补齐入口 + 进度展示 |
| T28 | ingestion_schedules 每日自动调度 | P2 | 1天 | ★★★★ | A | §17.5 | 收盘后 16:30 起串行执行 Tushare 增量同步 |
| T29 | QE 实验选股服务完善 | P2 | 2天 | ★★★★ | A | §12/QE | 完善 qe_selection_service 的推理链路，支持多实验对比选股 |
| T30 | 令牌桶速率限制器 | P1 | 0.5天 | ★★★★ | S | §17.3 | 统一 Tushare API 调用频率控制，安全余量 80% |

**小计**：10 人天

#### 六、模拟盘交易平台

| # | 任务 | 优先级 | 工作量 | 收益 | ROI | 来源 | 说明 |
|---|------|--------|--------|------|-----|------|------|
| T31 | 模拟盘 3 张核心表 | P2 | 1天 | ★★★★ | A | §16.3 | sim_portfolios、sim_positions、sim_trades |
| T32 | 每日调度引擎 | P2 | 2天 | ★★★★ | A | §16.4 | 收盘后自动执行：获取信号→生成调仓→模拟成交→更新持仓 |
| T33 | 模拟成交引擎 | P2 | 2天 | ★★★★ | A | §16.5 | 基于实际行情的成交模拟（VWAP 价格 + 滑点 + 手续费） |
| T34 | 多组合横向对比 | P2 | 1.5天 | ★★★ | A | §16.6 | 同时运行多个 QE 实验的虚拟组合，对比不同因子/模型效果 |
| T35 | 绩效统计服务 | P2 | 1.5天 | ★★★ | A | §16.7 | 年化收益、夏普、最大回撤、日胜率等实时统计 |
| T36 | 模拟盘前端展示 | P3 | 2天 | ★★★ | B | §16.8 | 组合净值曲线、持仓明细、交易记录页面 |

**小计**：10 人天

#### 七、实盘验证与反馈闭环

| # | 任务 | 优先级 | 工作量 | 收益 | ROI | 来源 | 说明 |
|---|------|--------|--------|------|-----|------|------|
| T37 | selection_performance 表 | P1 | 0.5天 | ★★★★ | S | §12.2 | 记录选股后 T+1/5/10/20 实际涨跌幅 |
| T38 | SelectionPerformanceTracker | P2 | 2天 | ★★★★ | A | §12.3 | 每日收盘后自动填充多周期表现数据 |
| T39 | 回测-实盘一致性指标 | P2 | 1.5天 | ★★★★ | A | §12.4 | IC 衰减率、收益偏差比、排名相关性等过拟合检测 |
| T40 | QEFeedback 扩展 live_performance | P2 | 1天 | ★★★★★ | A | §12.5 | 将实盘/模拟盘数据注入 QE 演进反馈，闭环核心 |
| T41 | 实盘校准演进任务 | P3 | 3天 | ★★★★ | B | §12.6 | LiveCalibrationTask：当回测-实盘偏差>阈值时自动触发校准演进 |

**小计**：8 人天

#### 八、远期增强（按需触发）

| # | 任务 | 优先级 | 工作量 | 收益 | ROI | 来源 | 说明 |
|---|------|--------|--------|------|-----|------|------|
| T42 | 策略参数网格搜索服务 | P3 | 3天 | ★★★★ | A | §14.5 | 每个 SOTA 自动搜索最优 topk/n_drop/换仓周期/止盈止损组合 |
| T43 | 模型 catalog 同步 + 版本管理 | P3 | 2天 | ★★★ | B | §10.2 | RDAgent 模型自动注册到 AIstock model_catalog |
| T44 | ProductionModelTrainer | P3 | 5天 | ★★★★ | B | §10.3 | 模型放大（50K→500K+ 参数）+ 全量数据生产训练 |
| T45 | 分钟线双轨回测 | P3 | 5天 | ★★ | C | §15.2 | 日线快速轨 + 分钟线精确轨，触发条件：资金>5000万或偏差>30% |
| T46 | 小资金专用策略模板 | P3 | 1天 | ★★★ | A | §13.3 | topk=10/n_drop=2/account=50万 的预设模板 |
| T47 | 三系统联动协同演进闭环 | P3 | 5天 | ★★★★★ | B | §11.2 | 因子-模型-策略三维度协同演进的完整闭环 |

**小计**：21 人天

#### 汇总

| 类别 | 任务数 | 工作量合计 | 平均 ROI |
|------|--------|-----------|----------|
| 一、阻断性修复 | 5 | 0.6天 | S |
| 二、RDAgent 演进提升 | 8 | 6.4天 | A |
| 三、QE 功能完善 | 6 | 18天 | A-B |
| 四、因子库优化 | 5 | 9.5天 | A |
| 五、Tushare + 选股 | 6 | 10天 | A |
| 六、模拟盘平台 | 6 | 10天 | A |
| 七、实盘反馈闭环 | 5 | 8天 | A |
| 八、远期增强 | 6 | 21天 | B |
| **总计** | **47** | **~83.5天** | — |

### 19.2 分阶段实施计划

基于用户提出的实施顺序和上述任务评估，将 47 项任务编排为 7 个阶段。每个阶段有明确的交付物和验收标准，前一阶段完成后再启动下一阶段。

#### 阶段 0：紧急修复（第 1 天）

**目标**：消除 Coding 阶段阻断性问题，让 RDAgent 立即恢复正常演进效率。

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| T01 P0-A 消除 ML 矛盾 | 修改后的 `prompts.yaml` | 新 Loop 不再尝试在因子中嵌入 ML 训练 |
| T02 P0-B Critic 功能验证 | 修改后的 `model_coder/prompts.yaml` | 执行成功的模型代码不再因"文字不对齐"被拒绝 |
| T03 P0-C Final Decision 兜底 | 同上 | 执行成功 + 输出 shape 正确 → 自动通过 |
| T04 P1-A 架构精确性 | 修改后的 `prompts.yaml` | 架构描述包含层级维度信息 |
| T05 P1-B 复杂度递进 | 同上 | 每个假设最多引入 1 个新组件 |

**验收方式**：修改后运行 2 个新 Loop，确认：(1) 因子 Loop 无 ML 尝试；(2) 模型 Loop 的 evo 轮数 ≤ 3。

**阶段产出**：有效产出率 66.7% → 90%+，平均 evo 轮数 5.8 → <3。修复完成后立即启动 RDAgent 持续演进。

#### 阶段 1：RDAgent 演进效果提升（第 2-5 天）

**目标**：通过提示词增强和少量配置改动，扩大模型演进搜索空间，提升每轮演进的质量。此阶段与 RDAgent 持续运行并行——边改边跑。

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| T06 模型假设维度引导 | 修改后的 `model_hypothesis_specification` | 新假设覆盖注意力/正则化/非平稳性等维度 |
| T07 Feedback 维度分析 | 修改后的 feedback prompt | 反馈包含"哪个维度改进/退步"的结构化分析 |
| T08 架构输出格式结构化 | 修改后的 `model_experiment_output_format` | architecture 字段为层级描述格式 |
| T10 回测 VWAP + 滑点 | 修改后的 `custom_strategy.py` | deal_price 使用 $vwap，增加滑点模型 |
| T13 评估指标扩展 | 修改后的 Feedback 模板 | 包含 Rank IC、分组收益等指标 |

**并行任务**（不阻塞主线，可后续完成）：
- T09 RAG 知识库扩展（1天）
- T11 损失函数可配置（2天）
- T12 优化器/学习率调度（1.5天）

**验收方式**：运行 4 个新 Loop 后，确认：(1) 假设维度分布不再集中于"换模型类型"；(2) Feedback 包含维度分析；(3) 回测结果使用 VWAP 价格。

**阶段产出**：RDAgent 以优化后的配置持续运行演进，搜索空间从 ~3 个维度扩展到 ~8 个维度。

#### 阶段 2：因子库优化 + Tushare 数据同步（第 6-15 天）

**目标**：两条线并行推进——(A) 建立因子独立评估体系，让演进产出的因子可量化比较；(B) 打通 Tushare 数据自动补齐，为实盘选股和模拟盘奠定数据基础。

**线 A：因子库（T20-T22，5.5天）**

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| T20 单因子 IC 独立计算 | IC 计算服务 + factor_catalog 扩展字段 | 每个因子有独立的 IC/ICIR/Rank_IC 值 |
| T21 因子截面相关性去重 | 相关性检测服务 | corr>0.9 的因子对被标记为冗余 |
| T22 因子分组管理 | 分组规则 + 分组字段 | 因子按类型自动归类（动量/价值/波动/流动性） |

**线 B：Tushare 数据同步（T25-T28/T30，8天）**

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| T30 令牌桶速率限制器 | `TushareRateLimiter` 类 | 所有 Tushare 调用经过统一限速，不超过 API 限制的 80% |
| T25 TushareSyncEngine | 统一同步引擎 | 支持 4 种 QueryMode，配置驱动，一套代码处理所有数据集 |
| T26 7 个数据集 DatasetSpec | 7 份 DatasetSpec 配置 | 每个数据集可独立执行全量/增量同步，结果与现有手工脚本一致 |
| T27 前端一键补齐 | API + 前端按钮 | 点击即可触发单个或批量数据集补齐，显示进度 |
| T28 每日自动调度 | ingestion_schedules 扩展 | 收盘后 16:30 自动执行增量同步，无需人工干预 |

**验收方式**：(A) 对已有因子运行 IC 计算，确认结果合理；(B) 7 个数据集全量补齐成功，增量同步无报错。

**阶段产出**：因子库具备独立评估能力；Tushare 数据实现自动化补齐，实盘选股的数据前置条件就绪。

#### 阶段 3：QE 功能完善 + QE 选股服务（第 16-25 天）

**目标**：完善 QE 单次实验和演进功能，同时打通基于 QE 实验的选股链路，开始积累选股表现数据。

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| T14 结构化维度引擎 | StructuredModelSpec 类 | 模型配置从自由文本变为结构化维度描述 |
| T16 深度反馈机制 | 多维度 Feedback 分析 | 反馈包含信号质量/收益/风险/交易成本四维分析 |
| T17 SOTA 库管理 | SOTA 版本化存储服务 | 支持 SOTA 配置对比、回滚 |
| T29 QE 选股服务完善 | 完善后的 qe_selection_service | 支持多实验对比选股，推理链路稳定 |
| T37 selection_performance 表 | 新建数据表 | 可记录选股后 T+1/5/10/20 实际涨跌幅 |
| T23 因子生命周期管理 | 状态机服务 | candidate→active→degraded→archived 自动流转 |

**验收方式**：(1) QE 实验可产出结构化维度配置；(2) 选股结果写入 selection_performance 表；(3) 因子状态可自动降级。

**阶段产出**：QE 演进质量提升，选股链路完整，开始积累实盘表现数据。

#### 阶段 4：模拟盘交易平台（第 26-35 天）

**目标**：搭建轻量模拟盘，实现从选股信号到虚拟交易的完整链路，支持多组合横向对比。

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| T31 模拟盘 3 张核心表 | DDL + 迁移脚本 | sim_portfolios/positions/trades 表创建成功 |
| T32 每日调度引擎 | SimulationScheduler 服务 | 收盘后自动执行完整调仓流程 |
| T33 模拟成交引擎 | SimulationExecutor 服务 | 基于 VWAP + 滑点 + 手续费的成交模拟 |
| T34 多组合横向对比 | 对比查询 API | 同时运行多个 QE 实验组合，可横向对比 |
| T35 绩效统计服务 | PerformanceStats 服务 | 年化收益/夏普/最大回撤/日胜率实时计算 |
| T38 SelectionPerformanceTracker | 自动跟踪服务 | 每日收盘后自动填充 T+1/5/10/20 表现 |

**验收方式**：创建 2 个模拟组合，运行 5 个交易日，确认持仓/交易/净值数据完整且合理。

**阶段产出**：模拟盘 MVP 上线，可同时跟踪多个 QE 实验的虚拟组合表现。

#### 阶段 5：实盘反馈闭环（第 36-45 天）

**目标**：将模拟盘/实盘数据反馈注入 QE 演进循环，实现从"优化回测指标"到"优化真实收益"的根本转变。

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| T39 回测-实盘一致性指标 | ConsistencyMetrics 服务 | IC 衰减率、收益偏差比、排名相关性可计算 |
| T40 QEFeedback 扩展 live_performance | 扩展后的 QEFeedback | 演进反馈包含实盘/模拟盘表现数据 |
| T18 QE Researcher strategy_tune | Researcher Agent 扩展 | 可提出策略参数调整假设并验证 |
| T42 策略参数网格搜索 | GridSearchService | 每个 SOTA 自动搜索最优策略参数组合 |

**验收方式**：QE 新一轮演进的 Feedback 中包含模拟盘实际表现数据，且 Researcher 可基于实盘偏差提出校准假设。

**阶段产出**：系统从"开环"变为"闭环"——演进目标函数从纯回测 IC 扩展为回测+实盘综合指标。

#### 阶段 6：远期增强（第 46 天起，按需触发）

**目标**：根据前 5 个阶段的运行效果，按需启动高级功能。

| 任务 | 触发条件 | 说明 |
|------|----------|------|
| T15 维度调度器（Bandit） | QE 演进 >20 轮后 | 基于历史收益自动选择下一轮演进维度 |
| T19 模型集成框架 | SOTA 模型 >5 个后 | 多模型加权集成提升预测稳定性 |
| T24 月度因子库维护 | 因子库 >100 个后 | 自动重算 IC、清理过期因子 |
| T36 模拟盘前端展示 | 模拟盘运行 >10 天后 | 净值曲线、持仓明细、交易记录页面 |
| T41 实盘校准演进任务 | 回测-实盘偏差 >30% | 自动触发校准演进 |
| T43 模型 catalog 同步 | 模型版本 >10 个后 | RDAgent 模型自动注册到 AIstock |
| T44 ProductionModelTrainer | 确认最优架构后 | 模型放大到 500K+ 参数 + 全量数据训练 |
| T45 分钟线双轨回测 | 资金 >5000 万或偏差 >30% | 日线快速轨 + 分钟线精确轨 |
| T46 小资金策略模板 | 实盘资金确定后 | topk=10/n_drop=2 预设模板 |
| T47 三系统协同演进 | 前 5 阶段全部完成后 | 因子-模型-策略三维度联动闭环 |

**阶段产出**：系统进入成熟运营阶段，具备自适应演进能力。

### 19.3 补充建议

以下是用户实施顺序中未明确提及、但文档分析中发现的重要补充项：

**1. 阶段 0 与阶段 1 应重叠执行**

阶段 0（提示词修复）完成后应立即启动 RDAgent 持续演进，不必等阶段 1 全部完成。阶段 1 的提示词增强可以在 RDAgent 运行间隙逐步应用——每次暂停演进时更新一批提示词，重启后立即生效。这样可以在阶段 1 期间就积累 10-20 个新 Loop 的数据。

**2. 因子 IC 计算（T20）应提前到阶段 1**

单因子 IC 独立计算是评估演进质量的基础指标。如果等到阶段 2 才实现，阶段 0-1 期间产出的因子无法量化评估。建议将 T20 提前到阶段 1，与 RDAgent 演进并行开发，这样阶段 1 结束时就能对已产出的因子做首次全面评估。

**3. selection_performance 表（T37）应尽早创建**

即使模拟盘尚未搭建，选股表现跟踪表也应尽早创建并开始积累数据。建议在阶段 2 就创建表结构，手动或半自动填充数据。等阶段 4 模拟盘上线时，已有 2-3 周的历史数据可供分析。

**4. 策略参数可配置化是被低估的高 ROI 任务**

文档 §14 分析表明，仅换仓周期一个参数的优化对净收益的影响可能超过换一个更好的模型。T10（VWAP+滑点）已在阶段 1，但策略参数从硬编码改为 YAML 配置（约 1 天工作量）也应提前到阶段 1，为后续网格搜索做准备。

**5. 监控与告警应贯穿全程**

各阶段的自动化服务（Tushare 同步、模拟盘调度、选股跟踪）都需要基本的监控和告警。建议在每个阶段的服务开发中同步加入：
- 执行失败时的日志告警（复用现有 logging 框架）
- 关键指标异常检测（如 IC 突然归零、模拟盘净值单日跌幅 >10%）
- 每日执行摘要邮件/消息通知

### 19.4 时间线总览

```
第 1 天     ┃ 阶段0：紧急修复（5项提示词，~40行）
            ┃ → RDAgent 立即恢复运行，开始持续演进
            ┃
第 2-5 天   ┃ 阶段1：演进效果提升（提示词+配置）
            ┃ → RDAgent 边跑边优化，并行不阻塞
            ┃
第 6-15 天  ┃ 阶段2：因子库优化 ∥ Tushare 数据同步
            ┃ → 两条线并行，互不依赖
            ┃
第 16-25 天 ┃ 阶段3：QE 功能完善 + 选股服务
            ┃ → 依赖阶段2的数据基础
            ┃
第 26-35 天 ┃ 阶段4：模拟盘平台
            ┃ → 依赖阶段3的选股链路
            ┃
第 36-45 天 ┃ 阶段5：实盘反馈闭环
            ┃ → 依赖阶段4的模拟盘数据
            ┃
第 46 天起  ┃ 阶段6：远期增强（按需触发）
            ┃ → 根据运行效果决定启动哪些任务
```

**关键里程碑**：

| 时间点 | 里程碑 | 系统状态 |
|--------|--------|----------|
| 第 1 天 | RDAgent 恢复正常演进 | 有效产出率 90%+，可持续运行 |
| 第 5 天 | 演进搜索空间扩展 | 8+ 维度探索，每轮质量提升 |
| 第 15 天 | 数据基础就绪 | 7 个 Tushare 数据集自动同步，因子可独立评估 |
| 第 25 天 | 选股链路完整 | QE 实验→选股→表现跟踪全链路打通 |
| 第 35 天 | 模拟盘上线 | 多组合虚拟交易，横向对比 |
| 第 45 天 | 闭环形成 | 实盘数据反馈驱动演进，系统自我优化 |

---

## 参考文献

1. **RD-Agent: Bridging the Gap Between Research and Development with Automated R&D Agents** - arXiv:2505.14738
2. **Collaborative Evolving Strategy for Automatic Data-Centric Development** (Co-STEER, NeurIPS 2025) - arXiv:2505.15155
3. **Towards Automated Data Sciences with Natural Language and LLM Agents** - arXiv:2407.18690
4. **Qlib: An AI-oriented Quantitative Investment Platform** - Microsoft Research
5. **PatchTST: A Time Series is Worth 64 Words** - ICLR 2023
6. **FEDformer: Frequency Enhanced Decomposed Transformer** - ICML 2022
7. **RevIN: Reversible Instance Normalization** - ICLR 2022

---

> 本文档由 RD-Agent 代码库分析 + 论文研读 + AIstock 系统分析综合生成，持续更新中。

