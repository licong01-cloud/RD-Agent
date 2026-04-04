# Requirements Document

## Introduction

因子指标计算日志功能，用于记录每次因子指标计算的完整过程（成功/跳过/失败），解决当前系统中计算状态不透明的问题。当前 engine.py 在 `mask.sum() == 0` 时直接 `continue` 不记录，`except` 块只打 `logger.warning` 不返回到 API 响应，前端 SQL 只能统计有指标的因子数量，无法区分"没算过"和"算了但失败了"。本功能通过新建 `aistock_factor_calc_log` 表、修改计算引擎返回结构、扩展 API 响应模型、新增后端查询接口、修改前端展示，实现因子×窗口级别的计算状态全链路可追溯。

## Glossary

- **Calc_Log_Table**: `aistock_factor_calc_log` 数据库表，记录每个因子在每个评估窗口下的计算状态（ok/skipped/error）
- **Factor_Metrics_Table**: `aistock_factor_metrics` 数据库表，存储因子的17项量化指标
- **Calc_Batch_ID**: 批次标识符，同一次计算请求中所有因子×窗口记录共享同一个 calc_batch_id，用于关联 Calc_Log_Table 和 Factor_Metrics_Table
- **Calc_Engine**: 因子指标计算引擎，即 `engine.py` 中的 `compute_all_factors_metrics` 函数，负责从 parquet 文件和收盘价数据计算17项指标
- **Eval_Window**: 评估窗口，包含 full（全量）、out_sample（样本外）、recent_6m（近6月）、recent_3m（近3月）四种
- **Factor_Report**: Calc_Engine 返回的因子计算状态报告，包含每个因子在每个 Eval_Window 下的状态（ok/skipped/error）及详细信息
- **Sync_Service**: `rdagent_factor_metrics_sync.py` 同步服务，负责从 RD-Agent API 获取指标并写入数据库
- **Source_Task_View**: 前端按 source_task_id 分组的 Task 视图，展示每个 Task 下因子的计算覆盖情况
- **Factor_Window_Matrix**: 前端因子×窗口矩阵 UI，以 ✅/⚠️/❌ 图标展示每个因子在每个窗口的计算状态
- **Required_Days**: 每个 Eval_Window 所需的最少交易日数，full=0, out_sample=0, recent_6m=126, recent_3m=63

## Requirements

### Requirement 1: 创建因子计算日志表

**User Story:** As a 系统管理员, I want 系统自动创建 aistock_factor_calc_log 表, so that 因子计算的完整过程可以被持久化记录和追溯。

#### Acceptance Criteria

1. WHEN init_quant_schema 函数执行时, THE Calc_Log_Table SHALL 被创建，包含以下字段: id (BIGSERIAL PRIMARY KEY), calc_batch_id (TEXT NOT NULL), source_task_id (TEXT), factor_name (TEXT NOT NULL), eval_window (TEXT NOT NULL), status (TEXT NOT NULL), error_message (TEXT), n_trading_days (INTEGER), required_days (INTEGER), data_start (DATE), data_end (DATE), data_source (TEXT NOT NULL DEFAULT 'parquet'), calc_engine (TEXT NOT NULL DEFAULT 'rdagent'), calculated_at (TIMESTAMPTZ NOT NULL), created_at (TIMESTAMPTZ NOT NULL DEFAULT NOW())
2. THE Calc_Log_Table SHALL 具有唯一约束 UNIQUE (calc_batch_id, factor_name, eval_window)，防止同一批次内重复记录
3. THE Calc_Log_Table SHALL 在 calc_batch_id 和 source_task_id 字段上创建索引，以支持按批次和任务的高效查询
4. THE Calc_Log_Table SHALL 对每个字段设置中文 COMMENT 描述，说明字段含义和取值范围
5. WHEN init_quant_schema 函数执行时, THE Factor_Metrics_Table SHALL 通过 ALTER TABLE 新增 calc_batch_id (TEXT) 字段，用于关联 Calc_Log_Table

### Requirement 2: 计算引擎返回因子计算报告

**User Story:** As a 后端开发者, I want Calc_Engine 返回每个因子每个窗口的计算状态报告, so that 调用方可以获知哪些因子窗口成功、跳过或失败。

#### Acceptance Criteria

1. WHEN Calc_Engine 完成计算时, THE Calc_Engine SHALL 返回一个包含 metrics 列表和 factor_reports 列表的字典，其中 factor_reports 记录每个因子在每个 Eval_Window 下的计算状态
2. WHEN 某个 Eval_Window 的 mask.sum() 等于 0 时, THE Calc_Engine SHALL 生成一条 status 为 "skipped" 的 Factor_Report 记录，error_message 说明跳过原因为数据天数不足，并记录 n_trading_days 和 required_days
3. WHEN 某个因子在某个 Eval_Window 的计算过程中抛出异常时, THE Calc_Engine SHALL 生成一条 status 为 "error" 的 Factor_Report 记录，error_message 包含异常信息
4. WHEN 某个因子在某个 Eval_Window 计算成功时, THE Calc_Engine SHALL 生成一条 status 为 "ok" 的 Factor_Report 记录，包含 n_trading_days、data_start 和 data_end
5. THE Calc_Engine SHALL 为每次调用生成一个唯一的 Calc_Batch_ID（UUID v4 格式），并在返回结果中包含该 Calc_Batch_ID
6. WHEN recent_6m 窗口的实际交易日数少于 126 天时, THE Calc_Engine SHALL 将该窗口的 Factor_Report status 设为 "skipped"，required_days 设为 126
7. WHEN recent_3m 窗口的实际交易日数少于 63 天时, THE Calc_Engine SHALL 将该窗口的 Factor_Report status 设为 "skipped"，required_days 设为 63

### Requirement 3: API 响应模型扩展

**User Story:** As a 前端开发者, I want API 响应中包含因子计算状态明细, so that 前端可以展示每个因子每个窗口的计算结果。

#### Acceptance Criteria

1. THE FactorMetricsResponse 模型 SHALL 新增 factor_reports 字段（列表类型），包含每个因子在每个 Eval_Window 下的 status、error_message、n_trading_days、required_days、data_start、data_end
2. THE FactorMetricsResponse 模型 SHALL 新增 calc_batch_id 字段（字符串类型），标识本次计算的批次
3. WHEN API 端点 /v2/{task_id}/factor_metrics 被调用时, THE API SHALL 在响应中返回 factor_reports 和 calc_batch_id 字段

### Requirement 4: 同步服务写入计算日志

**User Story:** As a 系统运维人员, I want 同步服务在写入指标数据时同时写入计算日志, so that 每次同步的计算过程都有完整记录。

#### Acceptance Criteria

1. WHEN Sync_Service 从 RD-Agent API 获取到因子指标响应时, THE Sync_Service SHALL 将响应中的 factor_reports 逐条写入 Calc_Log_Table
2. WHEN Sync_Service 写入 Factor_Metrics_Table 时, THE Sync_Service SHALL 在每条指标记录中填入对应的 Calc_Batch_ID
3. IF Calc_Log_Table 写入过程中发生数据库异常, THEN THE Sync_Service SHALL 记录错误日志但继续完成 Factor_Metrics_Table 的写入，确保日志写入失败不影响核心指标同步
4. THE Sync_Service SHALL 使用 UPSERT 语义写入 Calc_Log_Table，当 (calc_batch_id, factor_name, eval_window) 冲突时更新已有记录

### Requirement 5: 后端查询接口

**User Story:** As a 前端开发者, I want 后端提供查询接口获取 Task 下每个因子的窗口计算明细, so that 前端可以展示因子×窗口矩阵。

#### Acceptance Criteria

1. WHEN GET /rdagent/catalogs/factors/source-tasks/{task_id}/calc-detail 被调用时, THE API SHALL 返回该 Task 下所有因子在所有 Eval_Window 下的计算状态记录，按 factor_name 和 eval_window 排序
2. THE API 响应 SHALL 包含 factors 列表，每个因子包含 factor_name 和 windows 子列表，每个 window 包含 eval_window、status、error_message、n_trading_days、required_days、data_start、data_end、calculated_at
3. THE API 响应 SHALL 包含 summary 对象，统计 ok_count、skipped_count、error_count 三种状态的总数
4. IF 指定的 source_task_id 在 Calc_Log_Table 中无记录, THEN THE API SHALL 返回空的 factors 列表和全零的 summary

### Requirement 6: 修改 source-tasks SQL 统计

**User Story:** As a 用户, I want source-tasks 列表显示真实的指标统计（成功/跳过/失败数）, so that 可以快速了解每个 Task 的计算覆盖情况。

#### Acceptance Criteria

1. WHEN GET /rdagent/catalogs/factors/source-tasks 被调用时, THE API SHALL 返回每个 source_task_id 的 ok_count、skipped_count、error_count 统计，数据来源于 Calc_Log_Table
2. THE API 响应中每个 Task 项 SHALL 包含 factor_count（因子总数，来自 factor_catalog）、ok_count（成功窗口数）、skipped_count（跳过窗口数）、error_count（失败窗口数）
3. WHEN Calc_Log_Table 中某个 source_task_id 无记录时, THE API SHALL 返回该 Task 的 ok_count、skipped_count、error_count 均为 0

### Requirement 7: 前端因子×窗口矩阵展示

**User Story:** As a 用户, I want 在 Task 分组视图展开后看到因子×窗口矩阵, so that 可以直观了解每个因子在每个窗口的计算状态。

#### Acceptance Criteria

1. WHEN 用户在 Source_Task_View 中展开某个 Task 时, THE Factor_Window_Matrix SHALL 以表格形式展示该 Task 下所有因子在 full、out_sample、recent_6m、recent_3m 四个窗口的计算状态
2. THE Factor_Window_Matrix SHALL 使用 ✅ 图标表示 status 为 "ok" 的窗口，⚠️ 图标表示 status 为 "skipped" 的窗口，❌ 图标表示 status 为 "error" 的窗口，空白表示无记录
3. WHEN 用户将鼠标悬停在 ⚠️ 或 ❌ 图标上时, THE Factor_Window_Matrix SHALL 显示 tooltip，包含 error_message 和 n_trading_days/required_days 信息
4. THE Source_Task_View 中每个 Task 行 SHALL 显示 ok_count、skipped_count、error_count 的统计数字，替代当前仅显示 metrics_count 的方式
5. WHEN Calc_Log_Table 中某个 Task 无记录时, THE Factor_Window_Matrix SHALL 显示"暂无计算日志"的提示信息
