# Implementation Plan: 因子计算日志 (factor-calc-log)

## Overview

按数据流方向实现：先建表 → 改引擎返回结构 → 扩展 API 模型 → 同步服务写入日志 → 后端查询接口 → 前端矩阵展示。每个阶段完成后有检查点，确保增量可验证。

## Tasks

- [x] 1. 数据库表创建与 Schema 变更
  - [x] 1.1 在 `AIstock/backend/db/init_quant_schema.py` 中添加 `aistock_factor_calc_log` 建表 DDL
    - 添加 CREATE TABLE 语句，包含 id, calc_batch_id, source_task_id, factor_name, eval_window, status, error_message, n_trading_days, required_days, data_start, data_end, data_source, calc_engine, calculated_at, created_at 字段
    - 添加 UNIQUE (calc_batch_id, factor_name, eval_window) 约束
    - 添加 idx_calc_log_batch 和 idx_calc_log_task 索引
    - 添加所有字段的中文 COMMENT
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [x] 1.2 在 `AIstock/backend/db/init_quant_schema.py` 的 `init_quant_schema()` 中通过 ALTER TABLE 为 `aistock_factor_metrics` 添加 `calc_batch_id TEXT` 字段
    - 使用 `add_column_if_not_exists` 模式，添加 calc_batch_id 列和中文 COMMENT
    - _Requirements: 1.5_

- [x] 2. 计算引擎返回结构改造
  - [x] 2.1 修改 `RD-Agent-main/rdagent/app/factor_metrics/engine.py` 中 `compute_all_factors_metrics` 函数
    - 添加 `REQUIRED_DAYS` 常量映射 `{"full": 0, "out_sample": 0, "recent_6m": 126, "recent_3m": 63}`
    - 函数开头生成 `calc_batch_id = str(uuid.uuid4())`
    - 将 `mask.sum() == 0` 的 `continue` 改为生成 `status="skipped"` 的 factor_report
    - 新增 `mask.sum() < REQUIRED_DAYS[window_name]` 检查，不足时生成 `status="skipped"` report
    - `except` 块生成 `status="error"` 的 factor_report（含 `error_message=str(e)`）
    - 计算成功时生成 `status="ok"` 的 factor_report（含 n_trading_days, data_start, data_end）
    - 每条 metrics dict 中新增 `calc_batch_id` 字段
    - 返回类型从 `list[dict]` 改为 `dict`，包含 `metrics`, `factor_reports`, `calc_batch_id`, `summary`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_
  - [ ]* 2.2 编写 Property 1 属性测试：Engine 返回结构完整性
    - **Property 1: Engine returns complete structure**
    - 使用 Hypothesis 生成随机 MultiIndex DataFrame，验证返回 dict 包含 metrics, factor_reports, calc_batch_id(UUID v4), summary，且 ok_count + skipped_count + error_count == len(factor_reports)
    - **Validates: Requirements 2.1, 2.5**
  - [ ]* 2.3 编写 Property 2 属性测试：交易日不足时 skipped 状态
    - **Property 2: Insufficient trading days produce skipped status**
    - 生成日期范围短于阈值的 DataFrame，验证 recent_6m(<126天) 和 recent_3m(<63天) 窗口产生 status="skipped"
    - **Validates: Requirements 2.2, 2.6, 2.7**
  - [ ]* 2.4 编写 Property 5 属性测试：factor_reports 覆盖所有因子×窗口组合
    - **Property 5: Factor reports cover all factor×window combinations**
    - 生成 N 个因子的 DataFrame，验证 len(factor_reports) == N × 4，每个 (factor_name, eval_window) 恰好出现一次
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 3. Checkpoint - 引擎改造验证
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. API 响应模型扩展
  - [x] 4.1 在 `RD-Agent-main/rdagent/app/api_endpoints/sota_factors_api.py` 中新增 `FactorReportItem` Pydantic 模型
    - 包含 factor_name, eval_window, status, error_message, n_trading_days, required_days, data_start, data_end, data_source, calc_engine, calculated_at 字段
    - _Requirements: 3.1_
  - [x] 4.2 修改 `FactorMetricsResponse` 模型，新增 `factor_reports: List[FactorReportItem]` 和 `calc_batch_id: Optional[str]` 字段
    - _Requirements: 3.1, 3.2_
  - [x] 4.3 修改 `_compute_task_factor_metrics` 函数，从 `compute_all_factors_metrics` 返回的 dict 中提取 metrics, factor_reports, calc_batch_id 并填入 FactorMetricsResponse
    - _Requirements: 3.3_

- [x] 5. 同步服务扩展
  - [x] 5.1 修改 `AIstock/backend/services/rdagent_factor_metrics_sync.py` 中 `_UPSERT_SQL` 和 `_insert_metrics_batch`，新增 `calc_batch_id` 列
    - 在 INSERT 和 params 中添加 calc_batch_id 字段
    - _Requirements: 4.2_
  - [x] 5.2 在 `rdagent_factor_metrics_sync.py` 中新增 `_UPSERT_CALC_LOG_SQL` 和 `_insert_calc_log_batch` 函数
    - 实现 UPSERT 语义，ON CONFLICT (calc_batch_id, factor_name, eval_window) DO UPDATE
    - _Requirements: 4.1, 4.4_
  - [x] 5.3 修改 `sync_factor_metrics_for_task`，从 API 响应提取 factor_reports 和 calc_batch_id，写入 calc_log
    - calc_log 写入用 try/except 包裹，失败只记日志不阻塞 metrics 写入
    - 兼容旧版 API 响应（无 factor_reports 字段时跳过）
    - _Requirements: 4.1, 4.2, 4.3_
  - [ ]* 5.4 编写 Property 6 属性测试：UPSERT 幂等性
    - **Property 6: Calc log UPSERT idempotency**
    - 生成随机 report 记录，写入两次，验证表中只有一行且反映最新值
    - **Validates: Requirements 4.4**

- [x] 6. Checkpoint - 同步服务验证
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. 后端查询接口
  - [x] 7.1 修改 `AIstock/backend/routers/rdagent_catalog_admin.py` 中 `list_factor_source_tasks` 的 SQL
    - 改为 LEFT JOIN aistock_factor_calc_log，统计 ok_count, skipped_count, error_count
    - 无 calc_log 记录时返回 0
    - _Requirements: 6.1, 6.2, 6.3_
  - [x] 7.2 在 `rdagent_catalog_admin.py` 中新增 `GET /factors/source-tasks/{task_id}/calc-detail` 端点
    - 查询 aistock_factor_calc_log WHERE source_task_id = task_id
    - 按 factor_name 分组，每个因子包含 windows 子列表
    - 返回 summary 统计 ok_count, skipped_count, error_count
    - 无记录时返回空 factors 列表和全零 summary
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [ ]* 7.3 编写 Property 7 属性测试：calc-detail summary 一致性
    - **Property 7: Calc-detail summary counts match actual records**
    - 生成随机 calc_log 记录集，验证 summary 中各 count 与实际记录数一致
    - **Validates: Requirements 5.1, 5.2, 5.3**
  - [ ]* 7.4 编写 Property 8 属性测试：source-tasks 统计一致性
    - **Property 8: Source-tasks statistics reflect calc_log data**
    - 生成多 task 的 calc_log 记录，验证 source-tasks 端点返回的统计与实际数据一致
    - **Validates: Requirements 6.1, 6.2**

- [x] 8. Checkpoint - 后端接口验证
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. 前端因子×窗口矩阵展示
  - [x] 9.1 修改 `AIstock/frontend/src/app/quantevolver/components/FactorList.tsx` 中 `SourceTask` 类型和 Task 行展示
    - 新增 ok_count, skipped_count, error_count 字段
    - Task 行显示 ✅{ok} ⚠️{skipped} ❌{error} 替代 metrics_count
    - _Requirements: 7.4_
  - [x] 9.2 在 FactorList.tsx 中实现因子×窗口矩阵组件
    - 展开 Task 时调用 GET /factors/source-tasks/{task_id}/calc-detail
    - 渲染表格：行=因子名，列=full/out_sample/recent_6m/recent_3m
    - 单元格图标：✅(ok) / ⚠️(skipped) / ❌(error) / 空白(无记录)
    - ⚠️ 和 ❌ 图标 hover 显示 tooltip（error_message + n_trading_days/required_days）
    - 无记录时显示"暂无计算日志"提示
    - _Requirements: 7.1, 7.2, 7.3, 7.5_
  - [ ]* 9.3 编写 Property 9 属性测试：status→icon 映射
    - **Property 9: Status-to-icon mapping is deterministic**
    - 使用 fast-check 生成随机 status 字符串，验证 ok→✅, skipped→⚠️, error→❌, 其他→空
    - **Validates: Requirements 7.2**

- [x] 10. Final checkpoint - 全链路验证
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- 跨工作区：Tasks 1, 5, 7, 9 修改 AIstock 侧文件；Tasks 2, 4 修改 RD-Agent-main 侧文件
- Property tests 使用 Hypothesis (Python) 和 fast-check (TypeScript)
- 每个 task 引用具体的 requirements 条款，确保全覆盖
- Checkpoints 在引擎改造、同步服务、后端接口三个关键节点后设置
