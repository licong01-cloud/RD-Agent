# 实施计划：QE 实验模块整改 (qe-experiment-overhaul)

## 概述

按三阶段依赖关系实施：Phase 1 DB迁移 + RDAgent路由重构（可并行）→ Phase 2 AIstock后端核心改动 → Phase 3 前端适配。涉及 Python/FastAPI 后端、TypeScript/Next.js 前端、RDAgent 侧 API 三个代码仓库。

## Tasks

- [x] 1. Phase 1a：数据库迁移与DDL更新
  - [x] 1.1 更新 `init_catalog_db.py` 中 qe_experiments 表的 DDL 定义
    - 将 `rdagent_task_id` 改为 `qe_task_id`，`rdagent_loop_id` 改为 `qe_loop_id`
    - 新增 `loop_index INTEGER DEFAULT 1`、`parent_experiment_id TEXT`、`is_evolution_loop BOOLEAN DEFAULT FALSE` 字段
    - 同步更新 qe_evolution_tasks、qe_evolution_loops 表中相关列名
    - _Requirements: 9.3, 7.1, 4.1_

  - [x] 1.2 编写并执行 DB 迁移 SQL 脚本
    - `ALTER TABLE RENAME COLUMN` 重命名 rdagent_task_id → qe_task_id, rdagent_loop_id → qe_loop_id
    - `ADD COLUMN IF NOT EXISTS` 新增 loop_index, parent_experiment_id, is_evolution_loop
    - 历史数据回填：qe_task_id = experiment_name, qe_loop_id = 'Loop1', loop_index = 1
    - 使用 IF EXISTS 检查确保幂等性
    - _Requirements: 9.1, 9.2, 9.4_

  - [ ]* 1.3 编写迁移幂等性属性测试
    - **Property 11: 迁移幂等性**
    - **Validates: Requirement 9.4**

- [x] 2. Phase 1b：RDAgent 侧 qe_evolution_api.py 路由重构
  - [x] 2.1 重构所有 Loop 相关路由为双参数嵌套格式
    - 将路由从单参数改为 `/tasks/{task_id}/loops/{loop_id}/*` 嵌套格式
    - 包括 status、metrics、assets/download、submit 等端点
    - 删除所有 `rsplit("_L", 1)` 解析逻辑
    - workspace 路径拼接改为 `WORKSPACE_BASE / task_id / loop_id`
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 2.2 新增 DELETE /tasks/{task_id} 端点
    - 实现删除整个 task workspace 目录的功能
    - 返回 `{ok: true, task_id: ...}` 或错误信息
    - _Requirements: 5.3, 8.1_

  - [ ]* 2.3 编写 workspace 路径一致性属性测试
    - **Property 2: Workspace路径一致性**
    - **Validates: Requirements 2.1, 2.3, 2.4, 3.1, 8.3**

- [x] 3. Phase 1 检查点
  - 确保所有测试通过，如有疑问请向用户确认。验证 DB 迁移成功、RDAgent 路由可正常响应。

- [x] 4. Phase 2a：文件重命名 + QEWorkspaceClient 双参数改造
  - [x] 4.1 重命名文件和类
    - `qe_rdagent_api_client.py` → `qe_workspace_client.py`
    - 类名 `RdagentApiClient` → `QEWorkspaceClient`
    - _Requirements: 7.2_

  - [x] 4.2 改造所有方法为双参数签名
    - `get_loop_status(task_id, loop_id)`、`get_loop_metrics(task_id, loop_id)`、`download_loop_assets(task_id, loop_id)` 等
    - URL 构造改为 `f"{base_url}/tasks/{task_id}/loops/{loop_id}/..."`
    - 新增 `cleanup_task_workspace(task_id)` 方法，调用 `DELETE /tasks/{task_id}`
    - _Requirements: 8.4, 5.3_

- [x] 5. Phase 2b：qe_evolution_service.py 改造
  - [x] 5.1 更新 import 路径和属性名
    - import 从 `qe_rdagent_api_client` 改为 `qe_workspace_client`
    - 属性名 `rdagent_client` → `workspace_client`
    - _Requirements: 7.3_

  - [x] 5.2 实现 task_id 复用和 loop_id 格式统一
    - `create_task()` 中 task_id = base_experiment_id（复用基础实验ID）
    - current_loop 初始值设为 1，演进从 Loop2 开始
    - loop_id 格式统一为 `f"Loop{loop_index}"`
    - 所有 API 调用适配双参数 `(task_id, loop_id)`
    - _Requirements: 3.2, 3.3, 3.4, 6.2_

  - [x] 5.3 实现演进 Loop 结果统一写入 qe_experiments 表
    - 演进 Loop 完成时插入子记录：experiment_id = `{task_id}_L{loop_index}`
    - 设置 parent_experiment_id = task_id, is_evolution_loop = TRUE
    - 使用 `ON CONFLICT DO UPDATE` 处理重复插入
    - _Requirements: 4.2, 4.4, 4.5, 6.3_

  - [x] 5.4 添加演进前置条件校验
    - 基础实验状态非 completed 时抛出 ValueError
    - _Requirements: 6.1_

  - [ ]* 5.5 编写演进任务ID复用属性测试
    - **Property 4: 演进任务ID复用**
    - **Validates: Requirements 3.2, 6.2**

  - [ ]* 5.6 编写 Loop 序号单调递增属性测试
    - **Property 3: Loop序号单调递增**
    - **Validates: Requirements 3.3, 4.2**

  - [ ]* 5.7 编写演进前置条件属性测试
    - **Property 8: 演进前置条件**
    - **Validates: Requirement 6.1**

- [x] 6. Phase 2c：config_composer.py 改造
  - [x] 6.1 实现基于日期时间的实验ID生成
    - `_generate_unique_experiment_id()` 返回 `qe_YYYYMMDD_HHMMSS` 格式
    - 冲突时追加 `_2`、`_3` 后缀，100次仍冲突则抛出 RuntimeError
    - experiment_name = experiment_id（统一）
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 6.2 重命名内部变量和方法
    - `_rdagent_config_cache` → `_workspace_config_cache`
    - `_fetch_rdagent_config` → `_fetch_workspace_config`
    - 更新 SELECT 查询中的列名（rdagent_task_id → qe_task_id 等）
    - _Requirements: 7.4, 9.3_

  - [ ]* 6.3 编写实验ID格式与唯一性属性测试
    - **Property 1: 实验ID格式与唯一性**
    - **Validates: Requirements 1.1, 1.2, 1.3**

- [x] 7. Phase 2d：quantevolver.py 路由层改造
  - [x] 7.1 修复 task_id 赋值和变量重命名
    - `qe_task_id = experiment_name`（不再拼接 `f"{experiment_name}_{experiment_id}"`）
    - loop_index = 1, qe_loop_id = "Loop1"
    - 所有 `rdagent_*` 变量名改为 `qe_*` 或 `workspace_*`
    - _Requirements: 2.1, 2.2, 2.4, 7.1_

  - [x] 7.2 适配所有 QEWorkspaceClient 双参数调用
    - `create_task(qe_task_id)` → 保持单参数
    - `submit_loop(qe_task_id, qe_loop_id, config)` → 双参数
    - `get_loop_status(qe_task_id, qe_loop_id)` → 双参数
    - `get_loop_metrics(qe_task_id, qe_loop_id)` → 双参数
    - `download_loop_assets(qe_task_id, qe_loop_id)` → 双参数
    - _Requirements: 8.5, 2.3_

  - [x] 7.3 实现 DELETE /experiments/{experiment_id} 端点
    - 检查实验状态：running → 409, 不存在 → 404
    - 调用 `cleanup_task_workspace(experiment_id)` 清理 workspace
    - workspace 清理失败时记录 warning 并继续
    - 单事务级联删除：qe_evolution_tasks → qe_factor_experiment_metrics → 子Loop → 主实验
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 7.4 编写删除完整性属性测试
    - **Property 7: 删除完整性**
    - **Validates: Requirements 5.5, 5.6**

  - [ ]* 7.5 编写数据完整性约束属性测试
    - **Property 6: 数据完整性约束**
    - **Validates: Requirements 4.4, 4.5**

  - [ ]* 7.6 编写实验结果记录完整性属性测试
    - **Property 5: 实验结果记录完整性**
    - **Validates: Requirements 4.3, 4.2**

- [x] 8. Phase 2e-2f：辅助文件改造
  - [x] 8.1 qe_file_sync_client.py 函数重命名
    - `_get_rdagent_api_base` → `_get_qe_api_base`
    - _Requirements: 7.5_

  - [ ]* 8.2 编写 QE Catalog 隔离性属性测试
    - **Property 9: QE Catalog隔离性**
    - **Validates: Requirement 10.1**

  - [ ]* 8.3 编写 QE 命名一致性属性测试
    - **Property 10: QE命名一致性**
    - **Validates: Requirement 7.7**

- [x] 9. Phase 2 检查点
  - 确保所有测试通过，如有疑问请向用户确认。验证后端所有改动协调一致，API 调用链路完整。

- [x] 10. Phase 3a：前端 experiments/page.tsx 适配
  - [x] 10.1 更新 TypeScript 类型定义
    - `rdagent_task_id` → `qe_task_id`，`rdagent_loop_id` → `qe_loop_id`
    - 新增 `loop_index`、`parent_experiment_id`、`is_evolution_loop` 字段
    - _Requirements: 7.6, 4.1_

  - [x] 10.2 实现实验删除按钮和交互
    - 添加删除按钮，调用 `DELETE /experiments/{experiment_id}?cleanup_workspace=true`
    - running 状态实验禁用删除按钮
    - 删除成功后刷新列表
    - _Requirements: 5.1, 5.3_

  - [x] 10.3 实现按 parent_experiment_id 分组展示
    - 主实验与其演进 Loop 分组显示
    - 按 loop_index 排序
    - _Requirements: 4.3_

- [x] 11. Phase 3b：useExperimentSSE.ts 适配
  - [x] 11.1 更新 SSE 数据流中的字段名
    - 所有 `rdagent_task_id` → `qe_task_id`，`rdagent_loop_id` → `qe_loop_id`
    - _Requirements: 7.6_

- [x] 12. 最终检查点
  - 确保所有测试通过，如有疑问请向用户确认。验证端到端流程：创建实验 → 执行 → 查询metrics → 演进 → 删除。

## 备注

- 标记 `*` 的任务为可选，可跳过以加速 MVP 交付
- 每个任务均引用具体需求编号，确保可追溯性
- 属性测试使用 Python hypothesis 库
- Phase 1 的两个子阶段（1a DB迁移、1b RDAgent路由）可并行执行
- QE 模块严格隔离：只读引用 catalog 表，不写入 catalog 表，不触碰 rdagent_workspace
