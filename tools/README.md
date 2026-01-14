# RD-Agent 工具脚本文档

**日期**: 2026-01-13  
**目的**: 汇总 RD-Agent 项目中的工具脚本，包括 Qlib 回测分析工具和调试工具

---

## 目录

- [一、Qlib 回测分析工具 (tools/)](#一qlib-回测分析工具-tools)
- [二、调试工具 (debug_tools/)](#二调试工具-debug_tools)

---

## 一、Qlib 回测分析工具 (tools/)

本目录包含用于分析 Qlib 回测结果的 Python 脚本，主要用于评估策略表现、分析交易行为、诊断回测问题等。

### 1. analyze_qlib_backtest.py

**作用**: Qlib 回测交易记录详细分析脚本，在 WSL Qlib 环境中执行，生成详细的分析报告

**使用方式**:
```bash
conda activate rdagent-gpu
python analyze_qlib_backtest.py
```

**参数**:
- 无命令行参数
- 配置路径在脚本内部硬编码：
  - `WORKSPACE_PATH`: 工作空间路径
  - `EXPERIMENT_ID`: 实验 ID

**功能**:
- 自动选择最新的运行结果进行分析
- 读取持仓数据、预测数据、回测结果汇总
- 分析交易行为、持仓统计、风险指标
- 生成详细的分析报告并保存到输出目录

**输出**:
- 在 `WORKSPACE_PATH/backtest_analysis_report/` 目录下生成分析报告
- 包含 JSON 格式的详细分析结果

---

### 2. analyze_backtest_result.py

**作用**: 分析回测结果，找出策略表现差的原因

**使用方式**:
```bash
python analyze_backtest_result.py
```

**参数**:
- 无命令行参数
- 配置路径在脚本内部硬编码：
  - `workspace_path`: 工作空间路径
  - `run_id`: 运行 ID

**功能**:
- 读取投资组合分析数据
- 读取持仓数据并分析持仓统计
- 分析每日持仓股票数量
- 提供策略表现差的诊断信息

**输出**:
- 控制台输出分析结果
- 显示持仓数据形状、列名、统计信息

---

### 3. analyze_cash_curve.py

**作用**: 分析资金曲线，支持调试特定日期的异常

**使用方式**:
```bash
python analyze_cash_curve.py [workspace_path] [experiment_id] [debug_date]
```

**参数**:
- `workspace_path` (可选): 工作空间路径，默认使用预设候选路径
- `experiment_id` (可选): 实验 ID，默认为 "891791629306182420"
- `debug_date` (可选): 调试日期，用于分析特定日期的异常

**功能**:
- 自动选择最新的运行结果进行分析
- 分析资金曲线的变化趋势
- 支持调试特定日期的异常情况
- 分析持仓变化、资金流动

**输出**:
- 控制台输出资金曲线分析结果
- 显示特定日期的详细信息（如果指定了 debug_date）

---

### 4. analyze_trade_performance.py

**作用**: 分析交易表现，基于 trading_events 构造完整的持仓回合

**使用方式**:
```bash
python analyze_trade_performance.py <trading_events_csv> [options]
```

**参数**:
- `trading_events_csv`: 交易事件 CSV 文件路径（必需）
- 支持通过 `argparse` 传递其他参数（需查看脚本实现）

**功能**:
- 基于 trading_events 构造每个股票的完整持仓回合（从持仓为 0 -> >0 -> 回到 0）
- 计算每笔交易的盈亏、持仓时长
- 分析交易成功率、平均盈亏
- 统计止盈止损情况

**输入格式**:
trading_events CSV 应包含以下列：
- `date`: 交易日期
- `stock_id`: 股票 ID
- `action`: BUY/SELL
- `amount_change`: 成交股数（正数）
- `trade_value_no_cost`: 成交金额（正数）

**输出**:
- 控制台输出交易表现分析结果
- 显示每笔交易的详细信息

---

### 5. analyze_trades.py

**作用**: 分析 RD-Agent 回测交易记录，统计选股数量、止盈止损次数、评分阈值合理性

**使用方式**:
```bash
python analyze_trades.py
```

**参数**:
- 无命令行参数
- 配置路径在脚本内部硬编码：
  - `workspace_path`: 工作空间路径
  - `positions_file`: 持仓数据文件路径
  - `pred_file`: 预测数据文件路径

**功能**:
- 读取持仓数据并分析持仓统计
- 读取预测数据并分析评分分布
- 统计选股数量、持仓变化
- 评估评分阈值的合理性

**输出**:
- 控制台输出交易记录分析结果
- 显示持仓数据类型、形状、列名
- 显示预测数据统计信息

---

### 6. analyze_qlib_backtest_fixed.py

**作用**: Qlib 回测交易记录详细分析脚本（修正版），正确读取持仓数据

**使用方式**:
```bash
python analyze_qlib_backtest_fixed.py
```

**参数**:
- 无命令行参数
- 配置路径在脚本内部硬编码：
  - `WORKSPACE_PATH`: 工作空间路径
  - `ARTIFACTS_PATH`: 实验结果路径
  - `OUTPUT_PATH`: 输出目录路径

**功能**:
- 读取回测结果汇总（qlib_res.csv）
- 读取持仓数据并分析持仓统计
- 读取预测数据并分析预测分布
- 分析交易行为、风险指标
- 生成详细的分析报告

**输出**:
- 在 `OUTPUT_PATH` 目录下生成分析报告
- 包含 JSON 格式的详细分析结果
- 包括汇总信息、持仓分析、指标分析、预测分析、交易行为分析、风险分析、建议

---

### 7. analyze_qlib_records.py

**作用**: 分析 Qlib 回测交易记录，读取持仓和指标数据，进行详细分析

**使用方式**:
```bash
python analyze_qlib_records.py
```

**参数**:
- 无命令行参数
- 配置路径在脚本内部硬编码：
  - `workspace_path`: 工作空间路径
  - `artifacts_path`: 实验结果路径

**功能**:
- 读取回测结果汇总
- 读取持仓数据并分析持仓结构
- 分析持仓统计、持仓价值
- 显示持仓数据的前几行

**输出**:
- 控制台输出回测交易记录分析结果
- 显示回测结果汇总、持仓数据结构、持仓统计

---

### 8. analyze_bundles.py

**作用**: 分析 production_bundles 目录中的文件分布

**使用方式**:
```bash
python analyze_bundles.py
```

**参数**:
- 无命令行参数
- 配置路径在脚本内部硬编码：
  - `bundles_path`: bundles 目录路径

**功能**:
- 统计空目录和非空目录数量
- 统计文件类型分布
- 计算总大小
- 显示有内容的目录示例

**输出**:
- 控制台输出 bundles 分析结果
- 显示目录统计、文件类型分布、总大小

---

### 9. analyze_bundles_detailed.py

**作用**: 详细分析 production_bundles 目录中的文件命名模式和大小分布

**使用方式**:
```bash
python analyze_bundles_detailed.py
```

**参数**:
- 无命令行参数
- 配置路径在脚本内部硬编码：
  - `bundles_path`: bundles 目录路径

**功能**:
- 分析文件命名模式（去掉前缀 hash）
- 统计文件大小分布
- 按大小分类（小、中、大、超大文件）
- 计算平均文件大小、中位数文件大小

**输出**:
- 控制台输出详细分析结果
- 显示文件类型模式、大小分布统计

---

### 10. analyze_score_distribution.py

**作用**: 详细分析预测评分分布，特别关注低于 0.2 的评分分布

**使用方式**:
```bash
python analyze_score_distribution.py
```

**参数**:
- 无命令行参数
- 配置路径在脚本内部硬编码：
  - `ARTIFACTS_PATH`: 实验结果路径
  - `OUTPUT_PATH`: 输出目录路径

**功能**:
- 读取预测数据
- 分析评分基本统计（均值、中位数、标准差、最小值、最大值）
- 特别关注低于 0.2 的评分分布
- 分析评分分位数分布

**输出**:
- 控制台输出评分分布分析结果
- 在 `OUTPUT_PATH` 目录下生成详细分析报告

---

## 二、调试工具 (debug_tools/)

本目录包含用于调试和诊断 RD-Agent 运行问题的 Python 脚本，主要用于分析日志、检查数据异常、诊断编码错误等。

**注意**: 本目录已在 `.gitignore` 中配置为忽略，不会提交到 Git 仓库。

### 1. analyze_aistock_h5_inf.py

**作用**: 分析 AIstock H5 文件中的无穷大值

**使用方式**:
```bash
python analyze_aistock_h5_inf.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 检测 H5 文件中的无穷大值
- 分析无穷大值的分布
- 生成诊断报告

---

### 2. analyze_coding_errors.py / analyze_coding_errors_v2.py / analyze_coding_errors_v3.py

**作用**: 分析编码错误，不同版本提供不同的分析深度

**使用方式**:
```bash
python analyze_coding_errors.py
python analyze_coding_errors_v2.py
python analyze_coding_errors_v3.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 分析编码错误日志
- 统计错误类型和频率
- 识别常见错误模式
- 提供错误修复建议

---

### 3. analyze_error_pattern.py / analyze_error_patterns.py

**作用**: 分析错误模式，识别重复出现的错误

**使用方式**:
```bash
python analyze_error_pattern.py
python analyze_error_patterns.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 识别错误模式
- 统计错误频率
- 分析错误趋势

---

### 4. analyze_h5_export_range.py

**作用**: 分析 H5 文件导出的时间范围

**使用方式**:
```bash
python analyze_h5_export_range.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 检查 H5 文件的时间范围
- 验证数据完整性
- 识别缺失的时间段

---

### 5. analyze_hypothesis_actions.py

**作用**: 分析 Hypothesis 的 action 分布（factor vs model）

**使用方式**:
```bash
python analyze_hypothesis_actions.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 统计每个 loop 的 action 分布
- 分析 action 切换模式
- 识别异常的 action 切换

---

### 6. analyze_inf_records.py

**作用**: 分析无穷大值记录

**使用方式**:
```bash
python analyze_inf_records.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 检测无穷大值记录
- 分析无穷大值的来源
- 提供修复建议

---

### 7. analyze_llm_errors.py

**作用**: 分析 LLM 调用错误

**使用方式**:
```bash
python analyze_llm_errors.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 分析 LLM 调用失败的原因
- 统计错误类型
- 提供错误修复建议

---

### 8. analyze_nav_anomaly.py

**作用**: 分析净值异常

**使用方式**:
```bash
python analyze_nav_anomaly.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 检测净值异常
- 分析异常原因
- 提供修复建议

---

### 9. analyze_pickle_structure.py

**作用**: 分析 pickle 文件的结构

**使用方式**:
```bash
python analyze_pickle_structure.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 分析 pickle 文件的内部结构
- 显示数据类型、形状、列名
- 帮助理解数据格式

---

### 10. analyze_system_prompts.py

**作用**: 分析系统提示词

**使用方式**:
```bash
python analyze_system_prompts.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 分析系统提示词的使用情况
- 统计提示词类型
- 识别提示词问题

---

### 11. analyze_factor_errors.py / analyze_factor_errors_v2.py

**作用**: 分析因子错误

**使用方式**:
```bash
python analyze_factor_errors.py
python analyze_factor_errors_v2.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 分析因子计算错误
- 统计错误类型
- 提供错误修复建议

---

### 12. analyze_evo_loop_distribution.py

**作用**: 分析 evolution loop 的分布

**使用方式**:
```bash
python analyze_evo_loop_distribution.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 统计 evolution loop 的数量
- 分析 loop 分布
- 识别异常 loop

---

### 13. analyze_unit_mismatch.py

**作用**: 分析单位不匹配问题

**使用方式**:
```bash
python analyze_unit_mismatch.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 检测单位不匹配
- 分析不匹配的原因
- 提供修复建议

---

### 14. analyze_trading_days.py

**作用**: 分析交易日数据

**使用方式**:
```bash
python analyze_trading_days.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 统计交易日数量
- 分析交易日分布
- 识别缺失的交易日

---

### 15. check_anomaly_stocks_*.py 系列

**作用**: 检查异常股票，不同版本提供不同的检查逻辑

**文件列表**:
- `check_anomaly_stocks_prefix_final.py`
- `check_anomaly_stocks_final.py`
- `check_anomaly_stocks_final_v3.py`
- `check_anomaly_stocks_debug.py`
- `check_anomaly_date_distribution_detail.py`
- `check_anomaly_stocks_prefix.py`
- `check_anomaly_stocks_logic_difference.py`
- `check_anomaly_stocks_list.py`
- `check_anomaly_date_distribution.py`
- `check_anomaly_stocks_range.py`
- `check_anomaly_stocks_recheck.py`
- `check_error_stocks_characteristics.py`
- `check_error_chuangyeboard_stocks.py`
- `check_error_stocks_full_data.py`
- `check_error_stocks_date_range.py`
- `check_error_stocks_full_list.py`

**使用方式**:
```bash
python check_anomaly_stocks_*.py
```

**参数**:
- 需查看具体脚本实现确认参数

**功能**:
- 检测异常股票
- 分析异常股票的特征
- 统计异常股票的数量和分布
- 提供异常股票列表

---

### 16. check_aistock_h5_anomalies.py

**作用**: 检查 AIstock H5 文件异常

**使用方式**:
```bash
python check_aistock_h5_anomalies.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 检测 H5 文件异常
- 分析异常原因
- 提供修复建议

---

### 17. check_20210406_anomaly.py

**作用**: 检查特定日期（2021-04-06）的异常

**使用方式**:
```bash
python check_20210406_anomaly.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 分析特定日期的异常
- 识别异常原因
- 提供修复建议

---

### 18. check_debug_h5_anomalies.py

**作用**: 检查调试用的 H5 文件异常

**使用方式**:
```bash
python check_debug_h5_anomalies.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 检测 H5 文件异常
- 分析异常原因
- 提供修复建议

---

### 19. check_csv_structure.py

**作用**: 检查 CSV 文件结构

**使用方式**:
```bash
python check_csv_structure.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 验证 CSV 文件结构
- 检查列名、数据类型
- 识别结构问题

---

### 20. check_bin_factor_column.py

**作用**: 检查 bin 文件中的因子列

**使用方式**:
```bash
python check_bin_factor_column.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 检查 bin 文件中的因子列
- 验证因子列的完整性
- 识别缺失的因子列

---

### 21. check_batch_3_4_logic.py

**作用**: 检查批次 3 和 4 的逻辑

**使用方式**:
```bash
python check_batch_3_4_logic.py
```

**参数**:
- 需查看脚本实现确认具体参数

**功能**:
- 验证批次逻辑
- 识别逻辑错误
- 提供修复建议

---

## 三、使用建议

### 3.1 Qlib 回测分析工具使用流程

1. **运行回测后**，使用 `analyze_qlib_backtest_fixed.py` 生成详细分析报告
2. **查看资金曲线**，使用 `analyze_cash_curve.py` 分析资金流动
3. **分析交易表现**，使用 `analyze_trade_performance.py` 分析每笔交易
4. **检查评分分布**，使用 `analyze_score_distribution.py` 评估评分合理性

### 3.2 调试工具使用流程

1. **遇到数据异常**，使用对应的 `check_anomaly_stocks_*.py` 脚本检查
2. **遇到编码错误**，使用 `analyze_coding_errors.py` 系列脚本分析
3. **遇到 LLM 错误**，使用 `analyze_llm_errors.py` 分析
4. **遇到提示词问题**，使用 `analyze_system_prompts.py` 分析
5. **需要了解数据结构**，使用 `analyze_pickle_structure.py` 查看

### 3.3 注意事项

1. **路径配置**: 大部分脚本在内部硬编码了路径，使用前需要根据实际情况修改
2. **环境依赖**: 部分脚本需要在 WSL Qlib 环境中运行
3. **输出位置**: 分析结果通常输出到控制台或指定的输出目录
4. **Git 忽略**: `debug_tools/` 目录已在 `.gitignore` 中配置，不会提交到 Git 仓库

---

## 四、维护说明

### 4.1 添加新工具

1. **Qlib 回测分析工具**: 放到 `tools/` 目录，并在此文档中添加说明
2. **调试工具**: 放到 `debug_tools/` 目录，并在此文档中添加说明

### 4.2 更新文档

当工具脚本的功能发生变化时，需要及时更新此文档，包括：
- 工具的作用描述
- 使用方式和参数
- 功能说明
- 输出说明

### 4.3 清理旧工具

对于不再使用的工具脚本：
1. 移动到 `debug_tools/` 目录的子目录（如 `archive/`）
2. 在文档中标注为"已废弃"
3. 保留一段时间后再删除

---

**文档版本**: v1.0  
**最后更新**: 2026-01-13
