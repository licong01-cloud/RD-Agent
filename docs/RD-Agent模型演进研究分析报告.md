# RD-Agent模型演进研究分析报告

## 目录
1. [原始提示词文件分析](#1-原始提示词文件分析)
2. [XGBoost模型支持问题分析](#2-xgboost模型支持问题分析)
3. [论文中模型支持分析](#3-论文中模型支持分析)
4. [训练参数含义](#4-训练参数含义)
5. [策略配置分析](#5-策略配置分析)
6. [模型类型扩展建议](#6-模型类型扩展建议)
7. [不支持模型类型的潜力分析](#7-不支持模型类型的潜力分析)
8. [代码修改建议](#8-代码修改建议)
9. [依赖安装命令](#9-依赖安装命令)
10. [备份命令](#10-备份命令)

---

## 1. 原始提示词文件分析

### 1.1 原始文件位置

- **文件1**：`F:\Dev\RD-Agent-main-git\rdagent\scenarios\qlib\prompts.yaml`
- **文件2**：`F:\Dev\RD-Agent-main-git\rdagent\components\coder\model_coder\prompts.yaml`

### 1.2 原始文件关键内容

#### scenarios/qlib/prompts.yaml

**model_hypothesis_specification（第85-93行）**：
```yaml
5. Focus exclusively on the architecture of PyTorch models. Each hypothesis should specifically address architectural decisions, such as layer configurations, activation functions, regularization methods, and overall model structure. DO NOT do any feature-specific processing. Instead, you can propose innovative transformations on the input time-series data to enhance model training effectiveness.
```

**model_experiment_output_format（第161行）**：
```yaml
"model_type": "Tabular or TimeSeries"  # Should be one of "Tabular" or "TimeSeries"
```

**training_hyperparameters**：
```yaml
"n_epochs": "100",
"early_stop": 10,
```

**factor_hypothesis_specification（第95-112行）**：
```yaml
1. **1-5 Factors per Generation:**
    - Ensure each generation produces 1-5 factors.
```

#### components/coder/model_coder/prompts.yaml

**extract_model_formulation_system（第25行）**：
```yaml
"model_type": "Tabular or TimeSeries or Graph or XGBoost"  # Should be one of "Tabular", "TimeSeries", "Graph", or "XGBoost"
```

**evolving_strategy_model_coder（第32行）**：
```yaml
User is trying to implement some pytorch models in the following scenario:
```

### 1.3 原始文件与当前文件的主要区别

| 对比项 | 原始文件 | 当前使用文件 | 状态 |
|--------|---------|-------------|------|
| **模型相关** | | | |
| 模型架构限制 | PyTorch models | PyTorch models | ✅ 相同 |
| model_type（scenarios） | Tabular or TimeSeries | Tabular or TimeSeries | ✅ 相同 |
| model_type（model_coder） | Tabular or TimeSeries or Graph or XGBoost | Tabular or TimeSeries or Graph or XGBoost | ✅ 相同 |
| 训练轮数 | 100轮 | 20轮 | ⚠️ 不同 |
| 早停轮数 | 10轮 | 5轮 | ⚠️ 不同 |
| **因子相关** | | | |
| 因子数量限制 | 1-5个 | 不限制 | ⚠️ 不同 |
| 数据源约束 | 不存在 | 必须使用静态/资金流因子 | ⚠️ 新增 |
| **策略相关** | | | |
| 选股数量 | 30只（等权重） | 50只（动态权重） | ⚠️ 不同 |
| 止盈止损 | +10%止盈，-10%止损 | +15%/+25%/+35%分批止盈，-10%止损 | ⚠️ 不同 |

---

## 2. XGBoost模型支持问题分析

### 2.1 问题根源

**矛盾1：跨文件不一致**

- **scenarios/qlib/prompts.yaml**：
  - `model_hypothesis_specification`：要求"PyTorch models"
  - `model_experiment_output_format`：`model_type`只有"Tabular or TimeSeries"

- **model_coder/prompts.yaml**：
  - `extract_model_formulation_system`：`model_type`包含"Graph or XGBoost"
  - `evolving_strategy_model_coder`：说"implement some pytorch models"

**矛盾2：系统提示与model_type不匹配**

- 第32行："User is trying to implement some pytorch models"
- 第25行：`model_type`包含"Graph or XGBoost"

### 2.2 错误示例

```python
# 错误示例：XGBoost包装在PyTorch中
class XGBoost_TimeSeries_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.internal_model = xgb.XGBRegressor(...)
    
    def forward(self, x):
        return self.internal_model.predict(x)

# 错误原因：
# 1. 没有nn.Parameter，导致optimizer得到空参数列表
# 2. XGBoost不支持梯度反向传播
# 3. 训练失败：ValueError: optimizer got an empty parameter list
```

### 2.3 原始提示词是否有XGBoost支持

| 版本 | model_hypothesis_specification | model_experiment_output_format | 是否支持XGBoost |
|------|------------------------------|--------------------------------|----------------|
| Git HEAD | PyTorch models | Tabular or TimeSeries | ❌ |
| 所有backup | PyTorch models | Tabular or TimeSeries | ❌ |
| 当前版本 | PyTorch models | Tabular or TimeSeries | ❌（scenarios/qlib）<br>✅（model_coder） |

**结论**：
- ❌ 原始版本**不支持XGBoost**
- ⚠️ 当前版本存在**跨文件不一致**

---

## 3. 论文中模型支持分析

### 3.1 论文中提到的模型列表

**机器学习模型**：
- Linear, MLP, LightGBM, XGBoost, CatBoost, DoubleEnsemble

**深度学习模型**：
- GRU, LSTM, ALSTM, Transformer, PatchTST, iTransformer, Mamba

**图神经网络**：
- TRA, MASTER, GATs

### 3.2 原始文件的支持情况

| 模型类型 | 论文支持 | 原始文件支持 | 原因 |
|---------|---------|-------------|------|
| **PyTorch深度学习模型** | | | |
| GRU | ✅ | ✅ | 支持PyTorch模型 |
| LSTM | ✅ | ✅ | 支持PyTorch模型 |
| ALSTM | ✅ | ✅ | 支持PyTorch模型 |
| Transformer | ✅ | ✅ | 支持PyTorch模型 |
| PatchTST | ✅ | ✅ | 支持PyTorch模型 |
| iTransformer | ✅ | ✅ | 支持PyTorch模型 |
| Mamba | ✅ | ✅ | 支持PyTorch模型 |
| MLP | ✅ | ✅ | 支持PyTorch模型 |
| **机器学习模型** | | | |
| Linear | ✅ | ❌ | 不支持ML框架 |
| LightGBM | ✅ | ❌ | 不支持LightGBM |
| XGBoost | ✅ | ⚠️ | model_type有XGBoost，但系统提示要求PyTorch |
| CatBoost | ✅ | ❌ | 不支持CatBoost |
| DoubleEnsemble | ✅ | ❌ | 不支持ML框架 |
| **图神经网络** | | | |
| TRA | ✅ | ⚠️ | model_type有Graph，但系统提示要求PyTorch |
| MASTER | ✅ | ⚠️ | model_type有Graph，但系统提示要求PyTorch |
| GATs | ✅ | ⚠️ | model_type有Graph，但系统提示要求PyTorch |

### 3.3 原始文件是否满足论文要求

**结论**：❌ **不能满足**

**原因**：
1. **框架限制**：只支持PyTorch框架
2. **不支持ML模型**：Linear、LightGBM、CatBoost、DoubleEnsemble
3. **不支持图模型**：TRA、MASTER、GATs（虽然model_type有Graph，但系统提示要求PyTorch）
4. **存在矛盾**：导致生成错误的混合架构

---

## 4. 训练参数含义

### 4.1 训练轮数（n_epochs）

**含义**：模型在整个训练数据上训练的完整次数

**作用**：
- 控制训练的总时长
- 防止欠拟合或过拟合
- 影响模型性能

**对比**：
- 原始版本：100轮
- 当前版本：20轮
- 建议：40轮

### 4.2 早停轮数（early_stop）

**含义**：如果在验证集上连续这么多轮性能没有提升，就提前停止训练

**作用**：
- 防止过拟合
- 节省训练时间
- 提高泛化能力

**对比**：
- 原始版本：10轮
- 当前版本：5轮
- 建议：10轮

---

## 5. 策略配置分析

### 5.1 策略配置的位置

**YAML配置文件**（conf_sota_factors_model.yaml第54-73行）：
```yaml
port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
    backtest:
        start_time: 2021-01-01
        end_time: 2025-11-28
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5
```

### 5.2 策略参数说明

| 参数 | 含义 | 原始版本 | 当前版本 |
|------|------|---------|---------|
| topk | 选股数量 | 30只 | 50只 |
| n_drop | 每次调仓随机丢弃的股票数 | 未明确 | 5只 |
| limit_threshold | 涨跌停限制 | 9.5% | 9.5% |
| open_cost | 开仓成本 | 0.05% | 0.05% |
| close_cost | 平仓成本 | 0.15% | 0.15% |

### 5.3 止盈止损的设置

**原始版本**：
```
if a single position reaches +10% profit before 5 trading days, it can be taken profit and closed early
if the unrealized loss reaches -10%, a stop-loss should be triggered
```

**当前版本**：
```
if it reaches a profit of +15%, 30% of the position is sold
at +25% profit, another 30% is sold (cumulative 60%)
at +35% profit, the entire position is sold
If the unrealized loss reaches -10%, a stop-loss is triggered
```

### 5.4 策略是否固定

| 策略参数 | 是否由LLM生成 | 位置 |
|---------|--------------|------|
| topk（选股数量） | ❌ 固定 | YAML配置文件 |
| n_drop（丢弃数量） | ❌ 固定 | YAML配置文件 |
| 权重分配 | ❌ 固定 | TopkDropoutStrategy（等权重）或动态权重（score²） |
| 止盈止损 | ❌ 固定 | 提示词中描述 |
| 交易成本 | ❌ 固定 | YAML配置文件 |

**LLM的职责范围**：
- ✅ 因子设计（factor_hypothesis）
- ✅ 模型架构（model_hypothesis）
- ✅ 模型超参数（n_epochs, lr, early_stop等）

**LLM不负责演进的部分**：
- ❌ 策略参数（topk, n_drop）
- ❌ 权重分配方式
- ❌ 止盈止损规则
- ❌ 交易成本

### 5.5 原生策略是否买进30只股票

**答案**：是的，原生策略是买进30只股票。

**证据**：
1. 提示词中明确说明"select exactly 30 stocks"
2. YAML配置文件中`topk: 30`（原始版本）
3. 等权重分配，每只股票约3.33%

---

## 6. 模型类型扩展建议

### 6.1 修改方案总览

| 文件 | 修改内容 | 原值 | 新值 |
|------|---------|------|------|
| `rdagent/scenarios/qlib/prompts.yaml` | `model_hypothesis_specification`第5条 | PyTorch models | 支持多种模型类型 |
| `rdagent/scenarios/qlib/prompts.yaml` | `model_experiment_output_format`的`model_type` | Tabular or TimeSeries | PyTorch or ML or Graph |
| `rdagent/scenarios/qlib/prompts.yaml` | `model_experiment_output_format`的`n_epochs` | 20 | 40 |
| `rdagent/scenarios/qlib/prompts.yaml` | `model_experiment_output_format`的`early_stop` | 5 | 10 |
| `rdagent/scenarios/qlib/prompts.yaml` | `factor_hypothesis_specification`第1条 | 任意数量 | 2-5个 |
| `rdagent/components/coder/model_coder/prompts.yaml` | `extract_model_formulation_system`的`model_type` | Tabular or TimeSeries or Graph or XGBoost | PyTorch or ML or Graph |
| `rdagent/components/coder/model_coder/prompts.yaml` | `evolving_strategy_model_coder`的system提示 | implement some pytorch models | implement models（添加框架要求） |

### 6.2 详细修改方案

#### 修改1：rdagent/scenarios/qlib/prompts.yaml

**位置1：第90行（model_hypothesis_specification第5条）**

**当前内容**：
```yaml
5. Focus exclusively on the architecture of PyTorch models. Each hypothesis should specifically address architectural decisions, such as layer configurations, activation functions, regularization methods, and overall model structure. DO NOT do any feature-specific processing. Instead, you can propose innovative transformations on the input time-series data to enhance model training effectiveness.
```

**修改为**：
```yaml
5. Focus exclusively on the architecture of models. Each hypothesis should specifically address architectural decisions.
   - For PyTorch models (GRU, LSTM, ALSTM, Transformer, PatchTST, iTransformer, Mamba, MLP): layer configurations, activation functions, regularization methods, and overall neural network structure
   - For ML models (Linear, LightGBM, XGBoost, CatBoost, DoubleEnsemble): tree structure, boosting parameters, feature engineering approaches
   - For Graph models (TRA, MASTER, GATs): graph structure, node/edge features, aggregation methods
   - CRITICAL: Never mix different model frameworks. Do NOT wrap ML models inside torch.nn.Module.
```

**位置2：第170行（model_experiment_output_format的model_type）**

**当前内容**：
```yaml
"model_type": "Tabular or TimeSeries"  # Should be one of "Tabular" or "TimeSeries"
```

**修改为**：
```yaml
"model_type": "PyTorch or ML or Graph"  # Should be one of "PyTorch", "ML", "Graph"
```

**位置3：第164行（model_experiment_output_format的n_epochs）**

**当前内容**：
```yaml
"n_epochs": "20",
```

**修改为**：
```yaml
"n_epochs": "40",
```

**位置4：第166行（model_experiment_output_format的early_stop）**

**当前内容**：
```yaml
"early_stop": 5,
```

**修改为**：
```yaml
"early_stop": 10,
```

**位置5：第96行（factor_hypothesis_specification第1条）**

**当前内容**：
```yaml
1. **Factors per Generation（不再限制数量）：**
    - 每一轮可以给出 **任意数量** 的候选因子（按你的策略/预算决定），以支持整体量化演进，而不是只做少量因子试探。
```

**修改为**：
```yaml
1. **Factors per Generation（2-5个）：**
    - 每一轮可以给出 **2-5个** 候选因子，以支持整体量化演进，同时保持实验的可控性。
    - 在给出多个因子时，请在类型、数据来源和结构上保持一定多样性，避免高度相似的微调版本。
```

#### 修改2：rdagent/components/coder/model_coder/prompts.yaml

**位置1：第25行（extract_model_formulation_system的model_type）**

**当前内容**：
```yaml
"model_type": "Tabular or TimeSeries or Graph or XGBoost"  # Should be one of "Tabular", "TimeSeries", "Graph", or "XGBoost"
```

**修改为**：
```yaml
"model_type": "PyTorch or ML or Graph"  # Should be one of "PyTorch", "ML", "Graph"
```

**位置2：第32行（evolving_strategy_model_coder的system提示）**

**当前内容**：
```yaml
User is trying to implement some pytorch models in the following scenario:
{{ scenario }}
Your code is expected to align the scenario in any form which means The user needs to get the prediction of the model based on the input data.
```

**修改为**：
```yaml
User is trying to implement models in the following scenario:
{{ scenario }}
Your code is expected to align the scenario in any form which means The user needs to get the prediction of the model based on the input data.

CRITICAL FRAMEWORK REQUIREMENTS:
- If model_type is "PyTorch": Use torch.nn.Module with nn.Parameter layers (e.g., nn.Linear, nn.GRU, nn.LSTM, nn.Transformer). The model MUST have trainable parameters that can be optimized by PyTorch optimizers.
- If model_type is "ML": Use sklearn or similar ML frameworks directly (e.g., LinearRegression, LightGBM, XGBoost, CatBoost). DO NOT wrap in torch.nn.Module.
- If model_type is "Graph": Use torch_geometric or similar graph neural network frameworks. DO NOT wrap ML models inside torch.nn.Module.
- NEVER mix frameworks: Do not create a class that inherits from torch.nn.Module but uses ML models internally. This will cause "optimizer got an empty parameter list" errors.
```

---

## 7. 不支持模型类型的潜力分析

### 7.1 不支持的模型类型

| 模型类型 | 量化交易实践 | 潜力评分 | 实现难度 |
|---------|------------|---------|---------|
| 隐马尔可夫模型（HMM） | ✅ 有实践 | ★★★★☆ | ⚠️ 中 |
| 强化学习模型 | ✅ 有实践 | ★★★★★ | ⚠️ 高 |
| 贝叶斯模型 | ✅ 有实践 | ★★★☆☆ | ⚠️ 中高 |
| 高斯过程模型 | ✅ 有实践 | ★★★☆☆ | ⚠️ 中高 |
| 概率图模型（PGM） | ✅ 有实践 | ★★☆☆☆ | ⚠️ 高 |

### 7.2 各模型详细分析

#### 7.2.1 隐马尔可夫模型（HMM）

**量化交易实践**：
- ✅ 用于市场状态识别（牛市/熊市/震荡市）
- ✅ 用于识别价格序列的隐藏状态
- ✅ 结合技术指标，识别市场模式

**优势**：
- 能够捕捉价格序列的隐藏状态
- 适合处理时间序列的马尔可夫性质
- 计算效率较高

**劣势**：
- 假设状态转移是马尔可夫的，可能过于简化
- 难以处理高维特征
- 训练数据需求较大

**收益提升**：
- 识别市场状态，调整策略
- 捕捉价格序列模式
- 结合技术指标优化

**风险控制**：
- 识别市场风险状态
- 提前预警市场转换
- 动态调整风险暴露

---

#### 7.2.2 强化学习模型

**量化交易实践**：
- ✅ 用于智能交易策略
- ✅ 用于动态仓位管理
- ✅ 用于最优执行算法

**优势**：
- 能够学习复杂的策略
- 适应市场变化
- 可以处理连续动作空间

**劣势**：
- 训练复杂度高
- 需要大量数据
- 风险控制困难
- 可能过拟合历史数据

**收益提升**：
- 能够优化交易时机
- 动态调整仓位
- 适应不同市场状态

**风险控制**：
- 可以学习风险控制策略
- 动态调整止损止盈
- 多资产风险分散

---

#### 7.2.3 贝叶斯模型

**量化交易实践**：
- ✅ 用于不确定性量化
- ✅ 用于风险预测
- ✅ 用于模型组合

**优势**：
- 能够量化不确定性
- 适合小样本学习
- 可以结合先验知识

**劣势**：
- 计算复杂度高
- 推理速度慢
- 难以处理高维数据

**收益提升**：
- 量化预测不确定性
- 结合先验知识
- 模型组合优化

**风险控制**：
- 量化预测风险
- 动态调整仓位
- 避免过度自信

---

#### 7.2.4 高斯过程模型

**量化交易实践**：
- ✅ 用于价格预测
- ✅ 用于波动率预测
- ✅ 用于异常检测

**优势**：
- 能够量化预测不确定性
- 适合小样本学习
- 可以处理非线性关系

**劣势**：
- 计算复杂度高（O(n³)）
- 难以处理大规模数据
- 核函数选择困难

**收益提升**：
- 小样本预测准确
- 量化预测不确定性
- 超参数优化

**风险控制**：
- 量化预测风险
- 避免过度拟合
- 异常检测

---

#### 7.2.5 概率图模型（PGM）

**量化交易实践**：
- ✅ 用于因果推断
- ✅ 用于风险因子建模
- ✅ 用于资产相关性建模

**优势**：
- 能够建模变量间的因果关系
- 可解释性强
- 适合处理缺失数据

**劣势**：
- 结构学习困难
- 计算复杂度高
- 难以处理高维数据

**收益提升**：
- 因果推断优化策略
- 事件驱动策略
- 风险因子建模

**风险控制**：
- 因果关系分析
- 风险因子识别
- 相关性建模

---

### 7.3 优先级排序（从高到低）

#### 1. 强化学习模型（★★★★★）

**潜力**：
- ✅ 最高潜力，能够学习复杂的动态策略
- ✅ 可以自适应市场变化
- ✅ 适合动态仓位管理和风险控制

**实现难度**：⚠️ 高

---

#### 2. 隐马尔可夫模型（★★★★☆）

**潜力**：
- ✅ 高潜力，适合市场状态识别
- ✅ 计算效率高
- ✅ 可解释性强

**实现难度**：⚠️ 中

---

#### 3. 贝叶斯模型（★★★☆☆）

**潜力**：
- ✅ 中等潜力，适合不确定性量化
- ✅ 适合小样本学习
- ✅ 可解释性强

**实现难度**：⚠️ 中高

---

#### 4. 高斯过程模型（★★★☆☆）

**潜力**：
- ✅ 中等潜力，适合小样本预测
- ✅ 能够量化不确定性
- ✅ 适合超参数优化

**实现难度**：⚠️ 中高

---

#### 5. 概率图模型（★★☆☆☆）

**潜力**：
- ✅ 较低潜力，适合因果推断
- ✅ 可解释性强
- ✅ 适合事件驱动策略

**实现难度**：⚠️ 高

---

## 8. 代码修改建议

### 8.1 支持隐马尔可夫模型（HMM）

**需要修改的代码**：

| 修改位置 | 修改内容 | 难度 |
|---------|---------|------|
| **RD-Agent** | 添加HMM适配器 | ⚠️ 中 |
| **RD-Agent** | 修改模型选择逻辑 | ⚠️ 低 |
| **RD-Agent** | 修改提示词 | ⚠️ 低 |
| **Qlib** | ❌ 不需要修改 | - |

**修改建议**：
```python
# 在RD-Agent中添加HMM适配器
class HMMAdapter:
    def __init__(self, model_config):
        from hmmlearn import hmm
        self.model = hmm.GaussianHMM(**model_config)
    
    def fit(self, X, y):
        self.model.fit(X)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```

---

### 8.2 支持强化学习模型

**需要修改的代码**：

| 修改位置 | 修改内容 | 难度 |
|---------|---------|------|
| **RD-Agent** | 添加RL适配器 | ⚠️ 高 |
| **RD-Agent** | 修改训练流程 | ⚠️ 高 |
| **RD-Agent** | 修改提示词 | ⚠️ 中 |
| **Qlib** | ❌ 不需要修改 | - |

**修改建议**：
```python
# 在RD-Agent中添加RL适配器
class RLAdapter:
    def __init__(self, model_config):
        import gym
        import stable_baselines3
        self.env = TradingEnv()
        self.model = stable_baselines3.PPO(**model_config)
    
    def fit(self, X, y):
        self.model.learn(total_timesteps=10000)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```

---

### 8.3 支持贝叶斯模型

**需要修改的代码**：

| 修改位置 | 修改内容 | 难度 |
|---------|---------|------|
| **RD-Agent** | 添加贝叶斯适配器 | ⚠️ 中 |
| **RD-Agent** | 修改模型选择逻辑 | ⚠️ 低 |
| **RD-Agent** | 修改提示词 | ⚠️ 低 |
| **Qlib** | ❌ 不需要修改 | - |

**修改建议**：
```python
# 在RD-Agent中添加贝叶斯适配器
class BayesianAdapter:
    def __init__(self, model_config):
        import pymc3
        self.model = pymc3.Model()
        # 构建贝叶斯模型
    
    def fit(self, X, y):
        with self.model:
            self.trace = pymc3.sample(1000)
        return self
    
    def predict(self, X):
        with self.model:
            return pymc3.sample_posterior_predictive(self.trace, data=X)
```

---

### 8.4 支持高斯过程模型

**需要修改的代码**：

| 修改位置 | 修改内容 | 难度 |
|---------|---------|------|
| **RD-Agent** | 添加GP适配器 | ⚠️ 中 |
| **RD-Agent** | 修改模型选择逻辑 | ⚠️ 低 |
| **RD-Agent** | 修改提示词 | ⚠️ 低 |
| **Qlib** | ❌ 不需要修改 | - |

**修改建议**：
```python
# 在RD-Agent中添加GP适配器
class GPAdapter:
    def __init__(self, model_config):
        import gpytorch
        self.model = gpytorch.models.GPRegressionModel(**model_config)
    
    def fit(self, X, y):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        # 训练GP模型
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X)
```

---

### 8.5 支持概率图模型

**需要修改的代码**：

| 修改位置 | 修改内容 | 难度 |
|---------|---------|------|
| **RD-Agent** | 添加PGM适配器 | ⚠️ 高 |
| **RD-Agent** | 修改模型选择逻辑 | ⚠️ 中 |
| **RD-Agent** | 修改提示词 | ⚠️ 中 |
| **Qlib** | ❌ 不需要修改 | - |

**修改建议**：
```python
# 在RD-Agent中添加PGM适配器
class PGMAdapter:
    def __init__(self, model_config):
        import pgmpy
        self.model = pgmpy.models.BayesianModel()
        # 构建概率图模型
    
    def fit(self, X, y):
        self.model.fit(X)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```

---

## 9. 依赖安装命令

### 9.1 基础依赖安装

```bash
# 安装机器学习模型库
pip install lightgbm xgboost catboost

# 安装图神经网络库
pip install torch-geometric

# 安装隐马尔可夫模型库（可选，如果需要支持HMM）
pip install hmmlearn
```

### 9.2 详细说明

| 库 | 用途 | 版本建议 |
|----|------|---------|
| lightgbm | LightGBM模型 | >= 4.0.0 |
| xgboost | XGBoost模型 | >= 2.0.0 |
| catboost | CatBoost模型 | >= 1.2.0 |
| torch-geometric | 图神经网络 | >= 2.3.0 |
| hmmlearn | 隐马尔可夫模型（可选） | >= 0.2.8 |

### 9.3 安装顺序建议

1. **首先安装基础依赖**：
```bash
pip install lightgbm xgboost catboost
```

2. **然后安装图神经网络依赖**：
```bash
pip install torch-geometric
```

3. **最后安装可选依赖**：
```bash
pip install hmmlearn
```

### 9.4 验证安装

```bash
python -c "import lightgbm, xgboost, catboost, torch_geometric; print('All dependencies installed successfully!')"
```

---

## 10. 备份命令

### 10.1 WSL环境备份命令

```bash
# 获取当前日期和时间
DATE=$(date +%Y%m%d_%H%M%S)

# 备份scenarios/qlib/prompts.yaml
cp /mnt/f/Dev/RD-Agent-main/rdagent/scenarios/qlib/prompts.yaml /mnt/f/Dev/RD-Agent-main/rdagent/scenarios/qlib/prompts.yaml.backup_${DATE}

# 备份components/coder/model_coder/prompts.yaml
cp /mnt/f/Dev/RD-Agent-main/rdagent/components/coder/model_coder/prompts.yaml /mnt/f/Dev/RD-Agent-main/rdagent/components/coder/model_coder/prompts.yaml.backup_${DATE}

# 验证备份
ls -lh /mnt/f/Dev/RD-Agent-main/rdagent/scenarios/qlib/prompts.yaml.backup_${DATE}
ls -lh /mnt/f/Dev/RD-Agent-main/rdagent/components/coder/model_coder/prompts.yaml.backup_${DATE}
```

### 10.2 备份文件命名规则

**格式**：`prompts.yaml.backup_YYYYMMDD_HHMMSS`

**示例**：
- `prompts.yaml.backup_20260118_132500`（2026年1月18日13:25:00）
- `prompts.yaml.backup_20260118_133000`（2026年1月18日13:30:00）

**优势**：
- ✅ 包含日期和精确时间，避免冲突
- ✅ 按时间排序，便于查找
- ✅ 不会被其他备份脚本覆盖

---

## 总结

### 1. 原始提示词文件分析
- ✅ 原始文件严格限制为"PyTorch models"
- ✅ 存在跨文件不一致（scenarios/qlib vs model_coder）
- ✅ 存在系统提示与model_type不匹配

### 2. XGBoost模型支持问题
- ❌ 原始版本不支持XGBoost
- ⚠️ 当前版本存在跨文件不一致
- ⚠️ 导致生成错误的混合架构

### 3. 论文中模型支持
- ❌ 原始版本不能满足论文中所有模型的支持要求
- ⚠️ 只支持PyTorch深度学习模型
- ❌ 不支持ML模型和图模型

### 4. 训练参数调整
- ✅ n_epochs：20 → 40
- ✅ early_stop：5 → 10

### 5. 因子数量限制
- ✅ 因子数量：任意数量 → 2-5个

### 6. 模型类型扩展
- ✅ 修改`model_hypothesis_specification`，支持多种模型类型
- ✅ 扩展`model_type`字段
- ✅ 修改`model_coder`的系统提示

### 7. 不支持模型的潜力
- ✅ 强化学习模型（★★★★★）
- ✅ 隐马尔可夫模型（★★★★☆）
- ✅ 贝叶斯模型（★★★☆☆）
- ✅ 高斯过程模型（★★★☆☆）
- ✅ 概率图模型（★★☆☆☆）

### 8. 代码修改建议
- ✅ 只需要修改RD-Agent的代码
- ❌ 不需要修改Qlib的代码
- ⚠️ 添加适配器是关键

### 9. 依赖安装
```bash
pip install lightgbm xgboost catboost torch-geometric
```

### 10. 备份命令
```bash
DATE=$(date +%Y%m%d_%H%M%S)
cp /mnt/f/Dev/RD-Agent-main/rdagent/scenarios/qlib/prompts.yaml /mnt/f/Dev/RD-Agent-main/rdagent/scenarios/qlib/prompts.yaml.backup_${DATE}
cp /mnt/f/Dev/RD-Agent-main/rdagent/components/coder/model_coder/prompts.yaml /mnt/f/Dev/RD-Agent-main/rdagent/components/coder/model_coder/prompts.yaml.backup_${DATE}
```

---

**文档版本**：v1.0
**创建日期**：2026-01-18
**最后更新**：2026-01-18
