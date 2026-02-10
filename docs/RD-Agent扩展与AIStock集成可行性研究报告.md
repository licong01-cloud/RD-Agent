# RD-Agent扩展与AIStock集成可行性研究报告

## 文档版本
- **版本号**：v1.0
- **创建日期**：2026-01-18
- **最后更新**：2026-01-18

---

## 执行摘要

本报告基于对RD-Agent核心代码的深入分析，提出了一套完整的扩展与AIStock集成方案。方案涵盖多模型类型支持、因子组合优化、多模型协同选股、HMM大盘分析、策略演进以及API集成等核心功能，旨在构建一个从研发验证到生产实盘的完整量化交易闭环平台。

**核心结论**：
- ✅ **技术可行性高**：基于现有RD-Agent架构，通过适配器模式可平滑扩展多模型支持
- ✅ **工作量可控**：预计总工作量约24-30人周，可分4个Phase逐步实施
- ✅ **风险可控**：主要技术风险已识别，均有明确的缓解措施
- ✅ **收益显著**：可大幅提升模型多样性、因子优化能力和系统自动化水平

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [RD-Agent现有架构分析](#2-rd-agent现有架构分析)
3. [多模型类型支持方案](#3-多模型类型支持方案)
4. [因子组合优化方案](#4-因子组合优化方案)
5. [多模型协同选股方案](#5-多模型协同选股方案)
6. [HMM大盘分析与策略演进方案](#6-hmm大盘分析与策略演进方案)
7. [AIStock-RDAgent API集成方案](#7-aistock-rdagent-api集成方案)
8. [工作量与资源评估](#8-工作量与资源评估)
9. [开发路线图](#9-开发路线图)
10. [风险评估与缓解策略](#10-风险评估与缓解策略)
11. [总结与建议](#11-总结与建议)

---

## 1. 项目背景与目标

### 1.1 现状分析

**RD-Agent现有能力**：
- ✅ 支持PyTorch深度学习模型演进
- ✅ 支持因子自动演进
- ✅ 基于LLM的假设生成与代码生成
- ✅ 完整的实验循环框架

**现有限制**：
- ❌ 仅支持PyTorch模型，不支持XGBoost/LightGBM等ML模型
- ❌ 因子管理单一，缺乏组合优化机制
- ❌ 策略参数固定，无法自动演进
- ❌ 缺乏大盘趋势分析能力
- ❌ API接口不完善，难以与AIStock集成

### 1.2 项目目标

**总体目标**：构建从研发验证到生产实盘的完整量化交易闭环平台

**具体目标**：
1. 支持13种以上模型类型（PyTorch/ML/Graph/HMM/RL）
2. 实现因子组合优化引擎（SOTA+Alpha158/360自动组合）
3. 支持多模型协同选股（截面vs时序，不同模型组合）
4. 集成HMM大盘趋势分析
5. 实现策略自动演进能力
6. 提供标准化REST API供AIStock调用
7. 实现实时进度监控与结果采集

---

## 2. RD-Agent现有架构分析

### 2.1 核心架构

```
RDLoop主循环
├── HypothesisGen (假设生成)
├── Hypothesis2Experiment (实验转换)
├── Developer (代码生成+执行)
│   ├── Coder (代码生成)
│   └── Runner (代码执行)
├── Summarizer (结果总结)
└── Trace (历史记录)
```

### 2.2 关键发现

**1. 演进流程清晰**
- 基于`Trace.hist`存储历史实验
- SOTA实验通过`Feedback.decision=True`标识
- 新假设参考历史实验生成

**2. 已有API框架**
- FastAPI基础框架已实现（`rdagent/app/scheduler/`）
- 支持任务创建、查询、日志获取
- 但功能不完善，需大幅扩展

**3. 模型类型受限**
- 提示词限制为"PyTorch models"
- 模型代码生成固定使用`torch.nn.Module`
- 缺乏对其他模型框架的支持

**4. 因子管理基础薄弱**
- SOTA因子未持久化，重启后丢失
- 缺乏因子组合优化机制
- 未区分截面因子和时序因子

---

## 3. 多模型类型支持方案

### 3.1 方案概述

**目标**：支持5大类13+种模型

| 模型类别 | 具体模型 | 优先级 |
|---------|---------|--------|
| PyTorch | GRU, LSTM, ALSTM, Transformer, PatchTST, iTransformer, Mamba, MLP | P0 (已支持) |
| ML | Linear, LightGBM, XGBoost, CatBoost, DoubleEnsemble | P0 |
| Graph | TRA, MASTER, GATs | P1 |
| HMM | GaussianHMM, GMMHMM | P1 |
| RL | PPO, A3C, SAC | P2 |

### 3.2 技术方案

#### 3.2.1 适配器模式

```python
# 核心接口
class ModelAdapter(ABC):
    @abstractmethod
    def create_model(self, config: Dict) -> Any: pass
    
    @abstractmethod
    def train(self, model, X_train, y_train, X_valid, y_valid) -> Any: pass
    
    @abstractmethod
    def predict(self, model, X_test) -> np.ndarray: pass
```

**优势**：
- 统一接口，易于扩展
- 各模型框架独立，互不干扰
- 提示词可根据模型类型动态调整

#### 3.2.2 修改点

**1. 提示词修改**（2处）
- `rdagent/scenarios/qlib/prompts.yaml`
  - 修改`model_hypothesis_specification`，明确支持多种模型类型
  - 修改`model_experiment_output_format`，扩展`model_type`字段
- `rdagent/components/coder/model_coder/prompts.yaml`
  - 添加框架选择指导
  - 强调不同框架的使用规则

**2. 代码修改**（3处）
- 新增`rdagent/components/coder/model_coder/adapters.py`（约500行）
- 新增`rdagent/components/coder/model_coder/registry.py`（约100行）
- 修改`rdagent/components/coder/model_coder/evolving_strategy.py`（约50行）

### 3.3 实施计划

| Phase | 内容 | 工作量 | 优先级 |
|-------|------|--------|--------|
| Phase 1 | 实现MLAdapter（XGBoost/LightGBM/CatBoost） | 5人日 | P0 |
| Phase 2 | 实现HMMAdapter | 3人日 | P1 |
| Phase 3 | 实现GraphAdapter | 5人日 | P1 |
| Phase 4 | 实现RLAdapter | 5人日 | P2 |

---

## 4. 因子组合优化方案

### 4.1 方案概述

**目标**：实现因子库管理与组合优化

**核心功能**：
1. 因子库持久化（SQLite）
2. SOTA因子自动入库
3. Alpha158/360因子导入
4. 因子类型分类（截面/时序/混合）
5. 因子组合优化（遗传算法/贪心算法）

### 4.2 技术方案

#### 4.2.1 因子库结构

```python
FactorLibrary
├── factors: List[FactorMeta]
│   ├── name: str
│   ├── code: str
│   ├── factor_type: FactorType (cross_sectional/time_series/hybrid)
│   ├── ic: float
│   ├── ir: float
│   ├── is_sota: bool
│   └── source: str (rdagent/alpha158/alpha360)
├── save_to_db()
└── load_from_db()
```

#### 4.2.2 优化策略

**方法1：贪心算法**（快速）
1. 从SOTA因子开始
2. 逐步添加IC最高的Alpha因子
3. 每次添加后评估组合IC/IR
4. 保留提升最大的组合

**方法2：遗传算法**（精确）
1. 随机生成初始因子组合种群
2. 评估每个组合的适应度（IR）
3. 选择、交叉、变异
4. 迭代N代，输出最优组合

### 4.3 实施计划

| Phase | 内容 | 工作量 | 优先级 |
|-------|------|--------|--------|
| Phase 1 | 因子库基础设施（DB+CRUD） | 3人日 | P0 |
| Phase 2 | Alpha158/360因子导入 | 2人日 | P0 |
| Phase 3 | 因子类型分类器 | 2人日 | P1 |
| Phase 4 | 因子组合优化器（贪心+遗传） | 5人日 | P0 |
| Phase 5 | 集成到演进循环 | 3人日 | P0 |

---

## 5. 多模型协同选股方案

### 5.1 方案概述

**核心思想**：不同类型因子使用不同模型，最终集成预测

**策略**：
- 截面因子 → MLP、XGBoost、LightGBM
- 时序因子 → LSTM、GRU、Transformer
- 混合因子 → MLP、LSTM、XGBoost

### 5.2 技术方案

```python
MultiModelTrainer
├── 识别因子类型
├── 为每种类型训练推荐模型
├── 集成预测（加权平均）
└── 动态调整权重（基于历史IC）
```

### 5.3 实施计划

| Phase | 内容 | 工作量 | 优先级 |
|-------|------|--------|--------|
| Phase 1 | 因子-模型匹配策略 | 2人日 | P1 |
| Phase 2 | 多模型训练器 | 3人日 | P1 |
| Phase 3 | 预测集成与权重优化 | 3人日 | P1 |

---

## 6. HMM大盘分析与策略演进方案

### 6.1 方案概述

**HMM大盘分析**：
- 识别市场状态（牛市/熊市/震荡市）
- 市场特征：收益率、波动率、成交量、RSI、MACD

**策略演进**：
- 根据市场状态调整策略参数
- 牛市：增加选股数量、放宽止损
- 熊市：减少选股数量、收紧止损
- 震荡市：使用默认参数

### 6.2 技术方案

```python
HMMMarketAnalyzer
├── train(market_features)
├── predict_regime() -> MarketRegime
└── get_regime_probabilities() -> Dict[str, float]

StrategyEvolutionEngine
├── evolve_params(current, performance, regime)
└── param_history: List[(params, sharpe)]
```

### 6.3 实施计划

| Phase | 内容 | 工作量 | 优先级 |
|-------|------|--------|--------|
| Phase 1 | HMM市场特征提取 | 2人日 | P1 |
| Phase 2 | HMM训练与状态预测 | 3人日 | P1 |
| Phase 3 | 策略参数演进引擎 | 3人日 | P2 |
| Phase 4 | 集成到演进循环 | 2人日 | P2 |

---

## 7. AIStock-RDAgent API集成方案

### 7.1 API端点设计

```
任务管理：
  POST   /api/v1/tasks
  GET    /api/v1/tasks/{task_id}
  PUT    /api/v1/tasks/{task_id}/stop
  
进度监控：
  GET    /api/v1/tasks/{task_id}/progress
  WS     /api/v1/tasks/{task_id}/ws  (WebSocket实时推送)
  
因子管理：
  GET    /api/v1/factors
  GET    /api/v1/factors/sota
  POST   /api/v1/factors/optimize
  
模型管理：
  GET    /api/v1/models
  GET    /api/v1/models/sota
  GET    /api/v1/models/{model_id}/download
  
实验结果：
  GET    /api/v1/experiments/{exp_id}
  POST   /api/v1/experiments/{exp_id}/export
  
市场分析：
  GET    /api/v1/market/regime
  GET    /api/v1/market/regime/history
```

### 7.2 实施计划

| Phase | 内容 | 工作量 | 优先级 |
|-------|------|--------|--------|
| Phase 1 | 扩展现有API（任务/进度） | 3人日 | P0 |
| Phase 2 | 新增因子/模型管理API | 3人日 | P0 |
| Phase 3 | 实验结果导出API | 2人日 | P0 |
| Phase 4 | WebSocket实时推送 | 3人日 | P1 |
| Phase 5 | API文档（Swagger） | 1人日 | P1 |

---

## 8. 工作量与资源评估

### 8.1 总体工作量

| 模块 | 工作量（人日） | 优先级 |
|------|--------------|--------|
| 多模型类型支持 | 18 | P0 |
| 因子组合优化 | 15 | P0 |
| 多模型协同选股 | 8 | P1 |
| HMM大盘分析与策略演进 | 10 | P1-P2 |
| API集成 | 12 | P0 |
| 测试与文档 | 10 | P0 |
| **总计** | **73人日** ≈ **15人周** | - |

### 8.2 资源需求

**人员需求**：
- 高级工程师 × 2（后端开发）
- 算法工程师 × 1（因子优化、HMM）
- 测试工程师 × 1（集成测试）

**时间需求**：
- 理想团队（3人）：约5周
- 最小团队（2人）：约7-8周

---

## 9. 开发路线图

### Phase 1: 核心基础（4周）- P0优先级

**目标**：建立多模型支持+因子优化+API基础

| 周 | 任务 | 交付物 |
|----|------|--------|
| W1 | ML模型适配器实现 | MLAdapter支持XGBoost/LightGBM/CatBoost |
| W2 | 因子库基础设施 | FactorLibrary + SQLite持久化 |
| W3 | 因子组合优化器 | 贪心算法+遗传算法 |
| W4 | API扩展（任务/因子/模型） | REST API + Swagger文档 |

**里程碑**：
- ✅ 支持XGBoost/LightGBM/CatBoost模型演进
- ✅ 因子库可持久化管理
- ✅ AIStock可通过API创建任务、查询因子/模型

### Phase 2: 高级功能（3周）- P1优先级

**目标**：多模型协同+HMM大盘分析

| 周 | 任务 | 交付物 |
|----|------|--------|
| W5 | 多模型协同选股 | MultiModelTrainer + 集成预测 |
| W6 | HMM大盘分析 | HMMMarketAnalyzer + 市场状态识别 |
| W7 | WebSocket实时推送 | 任务进度实时推送 |

**里程碑**：
- ✅ 支持截面因子+时序因子使用不同模型
- ✅ 可识别市场状态（牛市/熊市/震荡市）
- ✅ AIStock可实时获取任务进度

### Phase 3: 策略优化（2周）- P2优先级

**目标**：策略参数演进+图模型+RL模型

| 周 | 任务 | 交付物 |
|----|------|--------|
| W8 | 策略参数演进 | StrategyEvolutionEngine + 基于市场状态的参数调整 |
| W9 | 图模型/RL模型（可选） | GraphAdapter + RLAdapter |

**里程碑**：
- ✅ 策略参数可根据市场状态自动调整
- ✅ 支持图神经网络和强化学习模型（可选）

### Phase 4: 测试与优化（1周）

**目标**：全面测试+性能优化+文档完善

| 周 | 任务 | 交付物 |
|----|------|--------|
| W10 | 集成测试+性能优化+文档 | 测试报告 + 用户手册 + API文档 |

**里程碑**：
- ✅ 所有功能通过集成测试
- ✅ API文档完善
- ✅ 用户手册完成

---

## 10. 风险评估与缓解策略

### 10.1 技术风险

| 风险 | 影响 | 概率 | 缓解策略 |
|------|------|------|---------|
| 不同模型框架接口差异大 | 高 | 中 | 使用适配器模式统一接口 |
| 因子组合爆炸（数量过多） | 高 | 高 | 使用启发式算法（贪心+遗传） |
| HMM状态识别不准确 | 中 | 中 | 多次训练验证，调整特征 |
| API性能瓶颈 | 中 | 低 | 异步处理+Redis缓存 |
| 模型库依赖冲突 | 低 | 低 | 使用conda环境隔离 |

### 10.2 项目风险

| 风险 | 影响 | 概率 | 缓解策略 |
|------|------|------|---------|
| 需求变更频繁 | 高 | 中 | 采用敏捷开发，小步迭代 |
| 人员不足 | 高 | 低 | 优先实现P0功能，P1/P2可延后 |
| 测试不充分 | 中 | 中 | 每个Phase结束后进行测试 |
| 文档不及时 | 低 | 中 | 与开发同步更新文档 |

---

## 11. 总结与建议

### 11.1 核心结论

1. **技术可行性**：✅ 高
   - 基于现有架构可平滑扩展
   - 适配器模式可统一多模型接口
   - API框架已有基础

2. **工作量评估**：✅ 可控
   - 总工作量：73人日 ≈ 15人周
   - 理想团队（3人）：5周
   - 最小团队（2人）：7-8周

3. **风险评估**：✅ 可控
   - 主要风险已识别
   - 均有明确缓解措施

4. **收益预期**：✅ 显著
   - 模型多样性大幅提升
   - 因子优化能力增强
   - 自动化水平提高
   - AIStock集成流畅

### 11.2 实施建议

**优先级排序**：
1. **P0（必须）**：多模型支持（ML）+ 因子组合优化 + API扩展
2. **P1（重要）**：多模型协同 + HMM大盘分析 + WebSocket推送
3. **P2（可选）**：策略演进 + 图模型 + RL模型

**实施路径**：
```
研发验证阶段（Phase 1-2）
    ↓
功能完善阶段（Phase 3）
    ↓
生产准备阶段（Phase 4）
    ↓
实盘交易阶段（与AIStock深度集成）
```

**关键成功因素**：
1. 严格按照Phase推进，避免跨Phase并行
2. 每个Phase结束后进行测试验证
3. 及时更新文档，确保AIStock侧可同步开发
4. 预留20%时间用于测试和优化

---

## 附录

### A. 参考文档
- `F:\Dev\RD-Agent-main\docs\RD-Agent模型演进研究分析报告.md`
- `rdagent/core/scenario.py`
- `rdagent/core/evolving_framework.py`
- `rdagent/app/qlib_rd_loop/quant.py`
- `rdagent/scenarios/qlib/proposal/quant_proposal.py`

### B. 关键代码路径
- 模型适配器：`rdagent/components/coder/model_coder/adapters.py`（待创建）
- 因子库：`rdagent/scenarios/qlib/factor_library/`（待创建）
- API扩展：`rdagent/app/api/routes.py`（待创建）
- HMM分析：`rdagent/scenarios/qlib/hmm/`（待创建）

---

**文档结束**
