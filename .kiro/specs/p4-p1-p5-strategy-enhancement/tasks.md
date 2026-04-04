# 实施计划：P4/P1/P5 策略增强

## 概述

将 P4（规则层风控）、P1（行业 HMM 热度跟踪）、P5（Optuna 贝叶斯超参优化）三个独立模块按增量方式实现。P4 通过新建策略子类 `TopkDropoutWithRiskControlStrategy` 实现，不修改原有 `TopkDropoutStrategy`。P4 和 P5 可并行，P1 在 P4 之后集成到新策略类中。

## 任务

- [x] 1. P4: 新建风控策略类并实现止损规则
  - [x] 1.1 创建 `AIstock/backend/rebalance_strategies/topk_dropout_rc.py`，实现 `TopkDropoutWithRiskControlStrategy` 类
    - 继承 `TopkDropoutStrategy`，设置 `STRATEGY_CODE = "TOPK_DROPOUT_RC"`，使用 `@register` 装饰器注册
    - 实现 `_validate_risk_params(config)` 静态方法：读取 `stop_loss_pct`（默认 0.10）和 `max_daily_turnover_pct`（默认 0.30），验证合法性，非法值记录警告并回退默认值
    - 实现 `_check_stop_loss()` 方法：遍历 `current_positions`，用 `close_price_fn` 获取当前价格，当 `price ≤ avg_cost × (1 - stop_loss_pct)` 时生成 SELL 信号，`target_quantity` 等于持仓数量，`reason="stop_loss"`
    - 重写 `generate_orders()` 方法：先执行止损检查，从 score_items 和 current_positions 中移除止损股票，再调用 `super().generate_orders()` 执行原有 TopK Dropout 逻辑
    - 在 `registry.py` 的 `_ensure_strategies_loaded()` 中添加 `from . import topk_dropout_rc` 确保新策略被加载
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3, 3.4_

  - [x] 1.2 编写 Property 1 属性测试：止损信号生成
    - **Property 1: 止损信号生成**
    - 使用 hypothesis 生成随机持仓（avg_cost、quantity）和随机价格，验证当 price ≤ avg_cost × (1 - stop_loss_pct) 时信号列表包含对应 SELL 信号，reason 为 "stop_loss"，target_quantity 等于持仓数量
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [x] 1.3 编写 Property 4 属性测试：非法参数回退到默认值
    - **Property 4: 非法参数回退到默认值**
    - 使用 hypothesis 生成超出合法范围的 stop_loss_pct 和 max_daily_turnover_pct，验证 generate_orders 行为等价于使用默认值
    - **Validates: Requirements 3.3, 3.4**

- [x] 2. P4: 实现换手率上限截断
  - [x] 2.1 在 `TopkDropoutWithRiskControlStrategy.generate_orders()` 中实现换手率截断逻辑
    - 合并止损信号 + 父类 `super().generate_orders()` 返回的信号后，计算总换手率（卖出市值 / portfolio_value）
    - 当换手率超过 `max_daily_turnover_pct` 时，按优先级截断：止损 > force_sell > dropout_sell，低优先级信号先移除
    - 实现 `_apply_turnover_cap()` 方法封装截断逻辑
    - 截断卖出信号后相应减少买入信号数量，保持买卖平衡
    - 在日志中记录实际换手率和被截断的信号数量
    - _Requirements: 1.4, 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.2 编写 Property 2 属性测试：规则优先级与 Dropout 排除
    - **Property 2: 规则优先级与 Dropout 排除**
    - 生成包含止损触发股票的持仓集合，验证止损股票不出现在 dropout 卖出信号中，无重复信号
    - **Validates: Requirements 1.4**

  - [x] 2.3 编写 Property 3 属性测试：换手率上限截断
    - **Property 3: 换手率上限截断**
    - 生成超过换手率上限的信号集，验证截断后卖出市值占比不超过 max_daily_turnover_pct，且买入数量相应减少
    - **Validates: Requirements 2.2, 2.3, 2.4**

- [x] 3. 检查点 — P4 完成验证
  - Ensure all tests pass, ask the user if questions arise.


- [x] 4. P1: 实现 SectorHMMTrainer 训练模块
  - [x] 4.1 创建 `AIstock/backend/quant_models/hmm/sector_hmm.py`，实现 `SectorHMMConfig`、`SectorHMMTrainer` 类
    - 定义 `SectorHMMConfig` dataclass（n_states=2, history_years=3.0, min_trading_days=120, cooldown_days=3, trending_coeff=1.5, fading_coeff=0.5, neutral_coeff=1.0）
    - 实现 `_build_observation_matrix(sector_code)` 方法：从 `market.sw_daily` 和沪深 300 指数查询数据，构建 4 列观测矩阵（日收益率、超额收益 20 日均值、成交量占比、涨停家数占比），过滤 NaN
    - 实现 `_label_states(model)` 方法：比较两个状态均值向量中日收益率分量，高者标记为 "trending"，低者标记为 "fading"
    - 实现 `train_all_sectors()` 方法：从 `market.sw_index_classify` 获取申万一级行业列表，逐行业训练 GaussianHMM，数据不足 120 天的跳过并记录警告
    - 实现 `save_models()` 和 `load_models()` 方法：JSON 格式持久化 HMM 参数（transmat、means、covars、state_labels）
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 4.2 编写 Property 5 属性测试：HMM 观测矩阵结构
    - **Property 5: HMM 观测矩阵结构**
    - 生成随机行业数据，验证 `_build_observation_matrix()` 返回矩阵恰好 4 列，行数等于交易日数量，不含 NaN
    - **Validates: Requirements 4.2**

  - [x] 4.3 编写 Property 6 属性测试：HMM 模型持久化往返
    - **Property 6: HMM 模型持久化往返**
    - 生成随机 HMM 参数（transmat、means、covars），save_models 后 load_models，验证数值差异 < 1e-10
    - **Validates: Requirements 4.4**

  - [x] 4.4 编写 Property 7 属性测试：HMM 状态标记一致性
    - **Property 7: HMM 状态标记一致性**
    - 生成随机 2 状态 HMM 均值向量，验证 "trending" 状态的日收益率分量严格大于 "fading" 状态
    - **Validates: Requirements 4.6**

- [x] 5. P1: 实现 SectorHMMInference 推断模块
  - [x] 5.1 在 `sector_hmm.py` 中实现 `SectorHMMInference` 类
    - 实现 `__init__()` 加载已训练的 HMM 模型参数，初始化冷却期状态缓存
    - 实现 `_decode_state(sector_code, trade_date)` 方法：构建截至 trade_date 的观测序列，Viterbi 解码最后一天的隐状态
    - 实现 `_check_cooldown(sector_code, new_state, trade_date)` 方法：检查距上次状态切换是否超过 cooldown_days 个交易日
    - 实现 `get_sector_coefficients(trade_date)` 方法：遍历所有已训练行业，解码状态并映射热度系数（trending→1.5, fading→0.5），冷却期内保持上次值，无模型行业返回 1.0
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x] 5.2 编写 Property 8 属性测试：HMM 推断输出有效性
    - **Property 8: HMM 推断输出有效性**
    - 生成随机模型参数和日期，验证 `get_sector_coefficients()` 返回值均为 0.5、1.0 或 1.5，且所有已训练行业代码都作为键出现
    - **Validates: Requirements 5.1, 5.2, 5.3**

  - [x] 5.3 编写 Property 9 属性测试：HMM 冷却期行为
    - **Property 9: HMM 冷却期行为**
    - 模拟状态切换序列，验证切换后 cooldown_days 内热度系数保持为切换前的值
    - **Validates: Requirements 5.4**

- [x] 6. P1: 集成行业热度系数到 TopkDropoutWithRiskControlStrategy
  - [x] 6.1 在 `topk_dropout_rc.py` 的 `generate_orders()` 中集成 HMM 热度调整
    - 新增 `_get_stock_sector_map(symbols, trade_date)` 辅助方法：查询 `market.sw_index_member` 获取股票→申万一级行业映射
    - 当 `config.get("enable_sector_hmm")` 为 True 时，调用 `SectorHMMInference.get_sector_coefficients(signal_date)` 获取热度系数
    - 将 `score_items` 中每只股票的 score 乘以其所属行业热度系数，使用调整后评分传递给 `super().generate_orders()`
    - 当 `enable_sector_hmm` 为 False 或未设置时，跳过热度调整，使用原始评分
    - 在日志中记录热态和冷态行业列表
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 6.2 编写 Property 10 属性测试：行业热度系数调整评分
    - **Property 10: 行业热度系数调整评分**
    - 生成随机 score_items 和热度系数字典，验证调整后评分 = 原始 score × 热度系数，TopK 排序基于调整后评分
    - **Validates: Requirements 6.2, 6.3**

  - [x] 6.3 编写 Property 11 属性测试：行业热度开关关闭时评分不变
    - **Property 11: 行业热度开关关闭时评分不变**
    - 验证 enable_sector_hmm=False 时 generate_orders 使用的评分与原始 score_items 完全一致
    - **Validates: Requirements 6.4**

- [x] 7. 检查点 — P1 完成验证
  - Ensure all tests pass, ask the user if questions arise.


- [x] 8. P5: 实现 OptunaHyperparamOptimizer 核心类
  - [x] 8.1 创建 `AIstock/backend/services/quantevolver/optuna_optimizer.py`，实现 `OptunaHyperparamOptimizer` 类
    - 实现 `__init__(task_id, model_type)` 方法：初始化 task_id、model_type，设置 Study 存储路径 `{QE_SOTA_ASSETS_DIR}/optuna_studies/{task_id}_{model_type}.db`
    - 实现 `get_or_create_study()` 方法：使用 `JournalFileStorage` 创建或加载持久化 Study（TPE 采样器），首次创建时调用 `_inject_historical_trials()` 和 `_inject_cross_task_trials()`
    - 实现 `_inject_historical_trials(study)` 方法：从 `qe_evolution_loops` 查询同 task_id 下 action_type="param_tune" 且 status="completed" 的记录，通过 `study.add_trial()` 注入为已完成 trial，目标值为 IC
    - 实现 `_inject_cross_task_trials(study)` 方法：查询同 model_type 跨 task 的最优 trial（IC 最高前 20 条，超过 50 条时截断），注入时在 user_attrs 中标记 source_task_id
    - 实现 `_define_search_space(trial)` 方法：根据 `HYPERPARAM_RANGES` 为每个超参数调用 `suggest_int`（max_depth、num_leaves、n_epochs、batch_size、early_stop）或 `suggest_float`（learning_rate、lr、weight_decay 等，log=True）
    - 实现 `ask()` 方法：调用 `study.ask()` 获取 trial，通过 `_define_search_space()` 生成候选超参数，记录日志（trial_number、参数值、已有 trial 数量），返回 (trial, suggested_params)
    - 实现 `tell(trial, ic_value)` 方法：调用 `study.tell(trial, ic_value)` 反馈结果
    - 异常处理：optuna 未安装时 import 失败优雅降级，Study 加载失败时创建新 Study，ask()/tell() 异常时记录日志不中断流程
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.4_

  - [x] 8.2 编写 Property 12 属性测试：Optuna Study 幂等创建
    - **Property 12: Optuna Study 幂等创建**
    - 对同一 (task_id, model_type) 连续两次调用 get_or_create_study()，验证 trial 数量相同，不重复注入
    - **Validates: Requirements 7.1**

  - [x] 8.3 编写 Property 13 属性测试：历史 Trial 注入
    - **Property 13: 历史 Trial 注入**
    - 模拟 qe_evolution_loops 中 N 条 param_tune 记录，验证新 Study 包含至少 N 条已完成 trial，目标值等于对应 IC
    - **Validates: Requirements 7.2, 7.3**

  - [x] 8.4 编写 Property 14 属性测试：跨 Task Trial 注入与上限
    - **Property 14: 跨 Task Trial 注入与上限**
    - 生成大量跨 task trial（>50 条），验证注入不超过 20 条，每条 user_attrs 包含 source_task_id
    - **Validates: Requirements 7.4, 10.1, 10.2, 10.3**

  - [x] 8.5 编写 Property 15 属性测试：Optuna ask() 参数范围有效性
    - **Property 15: Optuna ask() 参数范围有效性**
    - 对所有 HYPERPARAM_RANGES 中的 model_type 调用 ask()，验证每个参数在 (min, max) 范围内，整数参数为 int，浮点参数为 float
    - **Validates: Requirements 8.1, 8.2, 8.5**

  - [x] 8.6 编写 Property 16 属性测试：Optuna tell() 反馈更新
    - **Property 16: Optuna tell() 反馈更新**
    - 调用 ask() 获取 trial 后调用 tell(trial, ic_value)，验证 Study 中已完成 trial 数量增加 1，最新 trial 目标值等于 ic_value
    - **Validates: Requirements 8.4**

- [x] 9. P5: 集成 Optuna 到 QE 演进流程
  - [x] 9.1 修改 `qe_evolution_agents.py` 的 `EvolutionModelAgent.run()` Step 6 集成 Optuna 建议
    - 当 `model_decision == "tune_hyperparams"` 时，实例化 `OptunaHyperparamOptimizer(task_id, selected_model_type)`
    - 调用 `ask()` 获取 Optuna 建议的超参数
    - 将建议注入 Step 6 LLM prompt 的 `optuna_suggestion` 字段
    - 异常处理：Optuna 不可用时回退到纯 LLM 模式，记录警告日志
    - 将 trial 对象附加到返回结果中（供 process_completed_loop 调用 tell()）
    - _Requirements: 8.1, 8.3, 9.2, 9.4, 9.5_

  - [x] 9.2 修改 `qe_evolution_service.py` 的 `process_completed_loop()` 集成 Optuna 反馈
    - 在 Loop 完成处理流程中，检测 `action_type == "param_tune"` 时调用 `OptunaHyperparamOptimizer.tell(trial, IC_value)` 反馈结果
    - 异常处理：tell() 失败时记录错误日志，不影响演进流程继续
    - _Requirements: 9.1_

- [x] 10. 检查点 — P5 完成验证
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. 最终集成与全量验证
  - [x] 11.1 确保 P4、P1、P5 三个模块在 `topk_dropout_rc.py` 和 QE 演进流程中无冲突
    - 验证 `TopkDropoutWithRiskControlStrategy` 中 P1 热度调整在止损检查之前执行（先调整评分，再检查止损）
    - 验证原有 `TopkDropoutStrategy`（`TOPK_DROPOUT`）不受任何影响，可随时切换回去
    - 验证 P5 Optuna 集成不影响非 param_tune 方向的演进流程
    - 确保所有新增 import 和依赖正确声明
    - _Requirements: 1.1-1.4, 2.1-2.5, 3.1-3.4, 4.1-4.6, 5.1-5.6, 6.1-6.5, 7.1-7.5, 8.1-8.5, 9.1-9.5, 10.1-10.4_

  - [x] 11.2 编写单元测试覆盖关键边界条件
    - P4: 冷启动无持仓时风控规则不触发、close_price_fn 返回 None 时跳过止损检查、portfolio_value ≤ 0 时跳过换手率计算
    - P1: 训练数据不足 120 天跳过行业、无模型行业返回中性系数 1.0、enable_sector_hmm 未设置时默认行为
    - P5: optuna 未安装时回退、空历史冷启动、Study 文件路径格式验证
    - _Requirements: 3.2, 4.5, 5.6, 6.4, 7.5, 9.4_

- [x] 12. 最终检查点 — 全量测试通过
  - Ensure all tests pass, ask the user if questions arise.

## 备注

- 标记 `*` 的任务为可选，可跳过以加速 MVP 交付
- 每个任务引用具体需求编号以确保可追溯性
- 检查点确保增量验证，避免问题累积
- 属性测试验证通用正确性属性，单元测试验证具体示例和边界条件
- P4 通过新建 `topk_dropout_rc.py` 子类实现，不修改原有 `topk_dropout.py`，可通过策略代码 `TOPK_DROPOUT_RC` 切换使用
- P4 和 P5 可并行实施，P1 依赖 P4 中的新策略类创建完成后再集成
