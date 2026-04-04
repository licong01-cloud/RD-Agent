# 需求文档：P4/P1/P5 策略增强

## 简介

本文档定义 AIstock 量化交易系统三个优先级功能的需求：
- **P4** — 规则层止损/换手控制：新建 `TopkDropoutWithRiskControlStrategy` 继承 `TopkDropoutStrategy`，加入止损和换手率上限等确定性风控规则，降低最大回撤和换手成本。不修改原有策略，可随时切换回原策略。（本阶段不设置止盈和最大持仓天数，因实测盈利股票可达 300%+，长期持仓盈利显著）
- **P1** — 行业 HMM 热度跟踪：为每个申万一级行业建立 2 状态 HMM，动态调整个股信号权重以跟随行业轮动
- **P5** — Optuna 贝叶斯超参优化：用 Optuna TPE 替代 QE 演进中纯 LLM 猜测超参数的方式，提升 param_tune 成功率

三个功能独立验证，P4 和 P5 无依赖可并行实施，P1 依赖 P0（TopK=20）已完成。

## 术语表

- **TopkDropoutStrategy**: 位于 `topk_dropout.py` 的持仓再平衡策略，负责根据评分排名决定买卖信号（原始策略，不修改）
- **TopkDropoutWithRiskControlStrategy**: 位于 `topk_dropout_rc.py` 的风控增强策略，继承 `TopkDropoutStrategy`，新增止损和换手率上限规则，策略代码 `TOPK_DROPOUT_RC`
- **generate_orders**: `TopkDropoutStrategy` 的核心方法，输入评分列表和当前持仓，输出买卖信号列表
- **score_items**: 全量股票评分列表，已按 score 降序排列，每条包含 symbol、score、rank 等字段
- **current_positions**: 当前持仓字典，键为 symbol，值包含 quantity、avg_cost、market_value、entry_date
- **close_price_fn**: 可调用对象，接受 (symbol, date) 返回最近收盘价
- **signal**: 买卖信号字典，包含 portfolio_id、signal_date、trade_date、symbol、side、target_quantity、target_weight、score
- **HMM**: 隐马尔可夫模型（Hidden Markov Model），用于从观测序列推断隐藏状态
- **申万一级行业**: 申万宏源证券的行业分类标准一级分类，约 31 个行业
- **热态（trending）**: HMM 推断的行业活跃状态，表示该行业处于资金流入、价格上行趋势
- **冷态（fading）**: HMM 推断的行业低迷状态，表示该行业处于资金流出、价格下行趋势
- **冷却期**: 状态切换后的锁定窗口，防止频繁切换造成信号抖动
- **EvolutionModelAgent**: 位于 `qe_evolution_agents.py` 的 7 步模型编排器，Step 6 由 LLM 设计超参数
- **HYPERPARAM_RANGES**: `EvolutionModelAgent` 中定义的各模型类型合法超参数范围字典
- **Optuna**: 开源超参数优化框架，核心算法为 TPE（Tree-structured Parzen Estimator）
- **TPE**: 贝叶斯优化算法，用历史试验结果建模指导下一次搜索
- **QE_Evolution_Loop**: QE 自动演进的一轮实验，包含配置、训练、评估、决策完整流程
- **qe_evolution_loops**: 数据库表，记录每轮演进的配置、指标、分析结果
- **param_tune**: QE 演进方向之一，保持因子和模型架构不变，仅调整超参数
- **trial**: Optuna 中的一次超参数试验记录，包含参数组合和对应的目标值


## 需求

---

### 需求 1：个股止损规则

**用户故事：** 作为量化交易系统运营者，我希望持仓个股亏损达到阈值时自动卖出，以避免单股大亏拖累组合净值。

#### 验收标准

1. WHEN 持仓个股的未实现亏损百分比达到或超过 10%（即当前价格 ≤ avg_cost × 0.90），THE TopkDropoutWithRiskControlStrategy SHALL 在 generate_orders 返回的信号列表中生成该股票的全量 SELL 信号（target_quantity 等于当前持仓数量）
2. THE TopkDropoutWithRiskControlStrategy SHALL 使用 close_price_fn 获取的最新收盘价与 current_positions 中的 avg_cost 计算未实现亏损百分比
3. WHEN 止损卖出信号生成时，THE TopkDropoutWithRiskControlStrategy SHALL 在信号中标注 reason 字段为 "stop_loss"
4. THE TopkDropoutWithRiskControlStrategy SHALL 在执行 TopK dropout 逻辑之前优先处理止损卖出，止损卖出的股票不参与后续 dropout 排序

---

### 需求 2：每日换手率上限控制

**用户故事：** 作为量化交易系统运营者，我希望每日换仓比例受到上限约束，以控制交易成本。

#### 验收标准

1. THE TopkDropoutWithRiskControlStrategy SHALL 从 config 中读取 max_daily_turnover_pct 参数（默认值 0.30，即 30%）
2. THE TopkDropoutWithRiskControlStrategy SHALL 计算当日所有卖出信号的市值总和占组合总资产 portfolio_value 的比例作为换手率
3. WHEN 当日计划换手率超过 max_daily_turnover_pct 时，THE TopkDropoutWithRiskControlStrategy SHALL 按信号优先级保留卖出信号直到换手率不超过上限，优先级顺序为：止损 > force_sell（跌出 TopK）> dropout_sell
4. WHEN 卖出信号被换手率上限截断时，THE TopkDropoutWithRiskControlStrategy SHALL 相应减少买入信号数量，保持卖出数量与买入数量的平衡
5. THE TopkDropoutWithRiskControlStrategy SHALL 在日志中记录实际换手率和被截断的信号数量


---

### 需求 3：P4 风控规则参数可配置

**用户故事：** 作为量化交易系统运营者，我希望止损阈值和换手率上限均可通过 portfolio_config 配置，以便根据市场环境灵活调整。

#### 验收标准

1. THE TopkDropoutWithRiskControlStrategy SHALL 从 config 字典中读取以下参数：stop_loss_pct（默认 0.10）、max_daily_turnover_pct（默认 0.30）
2. WHEN config 中未提供上述参数时，THE TopkDropoutWithRiskControlStrategy SHALL 使用对应的默认值
3. THE TopkDropoutWithRiskControlStrategy SHALL 在 generate_orders 方法入口处验证参数合法性：stop_loss_pct 在 (0, 1) 范围内、max_daily_turnover_pct 在 (0, 1] 范围内
4. IF 参数值超出合法范围，THEN THE TopkDropoutWithRiskControlStrategy SHALL 记录警告日志并使用默认值

---

### 需求 4：行业 HMM 模型训练

**用户故事：** 作为量化策略研究员，我希望为每个申万一级行业训练独立的 2 状态 HMM 模型，以识别行业的热态和冷态。

#### 验收标准

1. THE SectorHMMTrainer SHALL 为每个申万一级行业（约 31 个）独立训练一个 2 状态 HMM 模型
2. THE SectorHMMTrainer SHALL 使用以下 4 个观测量构建观测矩阵（每个交易日一行，4 列）：行业指数日收益率、行业相对沪深 300 超额收益（20 日滚动均值）、行业成交量占全市场比例、行业内涨停家数占比
3. THE SectorHMMTrainer SHALL 使用 hmmlearn 库的 GaussianHMM 进行模型拟合，训练窗口为最近 3 年的交易日数据
4. WHEN 训练完成时，THE SectorHMMTrainer SHALL 将每个行业的 HMM 模型参数（转移矩阵、均值向量、协方差矩阵）持久化存储
5. IF 某个行业的训练数据不足 120 个交易日，THEN THE SectorHMMTrainer SHALL 跳过该行业并记录警告日志
6. THE SectorHMMTrainer SHALL 通过比较两个状态的均值向量中行业指数日收益率分量，将均值较高的状态标记为热态（trending），较低的标记为冷态（fading）

---

### 需求 5：行业 HMM 推断与热度系数输出

**用户故事：** 作为量化策略研究员，我希望每个交易日能获取各行业的热度系数，以便在选股时动态调整信号权重。

#### 验收标准

1. WHEN 给定一个交易日期时，THE SectorHMMInference SHALL 对每个已训练的申万一级行业 HMM 模型执行 Viterbi 解码，输出当日的隐状态（热态或冷态）
2. THE SectorHMMInference SHALL 根据隐状态映射热度系数：热态 → 1.5，冷态 → 0.5
3. THE SectorHMMInference SHALL 返回一个字典，键为申万一级行业代码，值为对应的热度系数（float）
4. WHILE 某行业处于状态切换冷却期内（状态切换后 3 个交易日），THE SectorHMMInference SHALL 保持该行业的热度系数为切换前的值，不响应新的状态变化
5. THE SectorHMMInference SHALL 维护每个行业的上次状态切换日期，用于冷却期判断
6. IF 某行业无已训练的 HMM 模型，THEN THE SectorHMMInference SHALL 返回该行业的热度系数为 1.0（中性值）

---

### 需求 6：行业热度系数集成到选股策略

**用户故事：** 作为量化策略研究员，我希望 TopkDropoutWithRiskControlStrategy 在选股时使用行业热度系数调整个股评分权重，使组合自然向热门行业集中。

#### 验收标准

1. WHEN generate_orders 被调用时，THE TopkDropoutWithRiskControlStrategy SHALL 调用 SectorHMMInference 获取当日各行业热度系数
2. THE TopkDropoutWithRiskControlStrategy SHALL 将 score_items 中每只股票的 score 乘以其所属申万一级行业的热度系数，得到调整后的评分
3. THE TopkDropoutWithRiskControlStrategy SHALL 使用调整后的评分进行 TopK 排序和 dropout 决策
4. WHEN config 中 enable_sector_hmm 参数为 False 或未设置时，THE TopkDropoutWithRiskControlStrategy SHALL 跳过行业热度调整，使用原始评分
5. THE TopkDropoutWithRiskControlStrategy SHALL 在日志中记录热态行业列表和冷态行业列表


---

### 需求 7：Optuna Study 管理与历史 Trial 注入

**用户故事：** 作为量化策略研究员，我希望 QE 演进系统能创建和管理 Optuna Study，并将历史演进中的超参数试验记录注入为先验知识，以加速贝叶斯搜索收敛。

#### 验收标准

1. THE OptunaHyperparamOptimizer SHALL 为每个 (task_id, model_type) 组合创建或加载一个持久化的 Optuna Study，使用 TPE 采样器
2. WHEN 创建新 Study 时，THE OptunaHyperparamOptimizer SHALL 从 qe_evolution_loops 表中查询该 task_id 下所有 action_type 为 "param_tune" 的已完成 Loop，提取其 model_params 和 IC 指标
3. THE OptunaHyperparamOptimizer SHALL 将查询到的历史 trial 通过 Optuna 的 study.add_trial API 注入为已完成的 trial，目标值为对应 Loop 的 IC 值
4. THE OptunaHyperparamOptimizer SHALL 支持跨 task_id 的先验迁移：当同一 model_type 在其他 task 中有已完成的 param_tune trial 时，将其作为低权重先验注入（通过设置 trial 的 intermediate_values 标记来源）
5. IF qe_evolution_loops 中无可用的历史 trial，THEN THE OptunaHyperparamOptimizer SHALL 使用 HYPERPARAM_RANGES 定义的范围作为均匀先验，正常启动 TPE 搜索

---

### 需求 8：Optuna 超参数建议生成

**用户故事：** 作为量化策略研究员，我希望 QE 演进的 param_tune 方向能使用 Optuna TPE 生成超参数建议，替代纯 LLM 随机猜测。

#### 验收标准

1. WHEN EvolutionModelAgent 的 Step 6（design_model_config）执行 param_tune 方向时，THE OptunaHyperparamOptimizer SHALL 使用 Optuna study.ask() 生成一组候选超参数
2. THE OptunaHyperparamOptimizer SHALL 根据 HYPERPARAM_RANGES 中对应 model_type 的定义，为每个超参数设置 suggest_float 或 suggest_int 的搜索范围
3. THE OptunaHyperparamOptimizer SHALL 将 Optuna 生成的候选超参数传递给 LLM 作为参考建议（而非完全替代 LLM），LLM 可在此基础上微调
4. WHEN 演进 Loop 完成并获得指标后，THE OptunaHyperparamOptimizer SHALL 调用 study.tell(trial, IC_value) 将结果反馈给 Optuna，更新贝叶斯模型
5. THE OptunaHyperparamOptimizer SHALL 对整数类型超参数（如 max_depth、num_leaves、n_epochs、batch_size、early_stop）使用 suggest_int，对浮点类型超参数（如 learning_rate、lr、weight_decay）使用 suggest_float 并设置 log=True 进行对数空间搜索

---

### 需求 9：Optuna 集成到 QE 演进流程

**用户故事：** 作为量化策略研究员，我希望 Optuna 优化器无缝集成到现有 QE 演进流程中，不破坏现有的 Agent 编排架构。

#### 验收标准

1. THE AutoEvolutionScheduler SHALL 在 process_completed_loop 中检测当前 Loop 的 action_type，当为 "param_tune" 时调用 OptunaHyperparamOptimizer.tell() 反馈结果
2. WHEN EvolutionModelAgent 执行 param_tune 时，THE EvolutionModelAgent SHALL 在 Step 6 的 LLM prompt 中注入 Optuna 建议的超参数作为 "optuna_suggestion" 字段
3. THE OptunaHyperparamOptimizer SHALL 将 Optuna Study 持久化到文件系统（使用 optuna.storages.RDBStorage 或 JournalFileStorage），路径为 `{QE_SOTA_ASSETS_DIR}/optuna_studies/{task_id}_{model_type}.db`
4. IF Optuna 库未安装或 Study 加载失败，THEN THE EvolutionModelAgent SHALL 回退到纯 LLM 模式并记录警告日志，不中断演进流程
5. THE OptunaHyperparamOptimizer SHALL 在每次 ask() 时记录日志，包含 trial_number、建议的超参数值和当前 Study 中已有的 trial 数量

---

### 需求 10：Optuna 跨模型类型先验迁移

**用户故事：** 作为量化策略研究员，我希望不同模型类型（如 LGB 和 PTNN）的最优超参数经验能在 QE 演进轮次间迁移，减少冷启动探索。

#### 验收标准

1. WHEN 创建新的 Optuna Study 时，THE OptunaHyperparamOptimizer SHALL 查询 qe_evolution_loops 中同一 model_type 但不同 task_id 的最优 trial（IC 最高的前 10 条）
2. THE OptunaHyperparamOptimizer SHALL 将跨 task 的最优 trial 注入新 Study，注入时在 trial 的 user_attrs 中标记 source_task_id 以区分来源
3. WHEN 同一 model_type 的跨 task trial 数量超过 50 条时，THE OptunaHyperparamOptimizer SHALL 仅注入 IC 排名前 20 的 trial，避免低质量先验污染搜索空间
4. THE OptunaHyperparamOptimizer SHALL 支持的 model_type 包括：LGB、XGB、CATBOOST、LINEAR、PTNN，与 HYPERPARAM_RANGES 的键保持一致
