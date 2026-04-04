# 需求文档

## 简介

本文档定义 QE（QuantEvolver）实验模块整改的全部需求。整改涵盖七个核心改进：metrics 404 关键Bug修复、实验ID格式改造、统一workspace结构、统一结果统计、实验删除功能、rdagent命名清理、API路由重构。所有需求均基于已批准的设计文档推导，遵循 EARS 模式和 INCOSE 质量标准。

## 术语表

- **QE系统**：QuantEvolver 实验模块，负责因子实验的创建、执行、演进和管理
- **ConfigComposer**：实验配置组装组件，负责ID生成和配置缓存管理
- **QEWorkspaceClient**：与 RDAgent 侧 API 通信的客户端组件（原 RdagentApiClient）
- **QEEvolutionService**：演进任务生命周期管理组件
- **RDAgent_QE_API**：RDAgent 侧提供的 workspace 管理 API 服务
- **qe_experiments表**：存储所有实验记录（含单次实验和演进Loop）的数据库表
- **qe_evolution_tasks表**：存储演进任务元数据的数据库表
- **qe_factor_experiment_metrics表**：存储因子实验指标的数据库表
- **Workspace**：RDAgent 侧的实验文件目录，路径格式为 `qe_workspace/{task_id}/Loop{N}/`
- **Loop**：实验的一次执行单元，编号从1开始（Loop1为单次实验，Loop2+为演进轮次）
- **experiment_id**：实验唯一标识符，格式为 `qe_YYYYMMDD_HHMMSS` 或冲突时 `qe_YYYYMMDD_HHMMSS_N`
- **task_id**：workspace 层面的任务标识符，与 experiment_id 统一
- **loop_id**：Loop 标识符，格式为 `Loop{N}`
- **catalog表**：`aistock_factor_catalog`、`aistock_model_catalog`、`aistock_strategy_catalog` 的统称，属于 RDAgent Task 同步模块

## 需求

### 需求 1：实验ID生成

**用户故事：** 作为开发者，我希望实验ID采用基于日期时间的可读格式，以便直观识别实验创建时间并消除 experiment_id 与 experiment_name 的分离。

#### 验收标准

1. WHEN ConfigComposer 生成新实验ID时，THE ConfigComposer SHALL 返回格式为 `qe_YYYYMMDD_HHMMSS` 的字符串，其中日期时间取自当前系统时间
2. WHEN 生成的基础ID在 qe_experiments表 中已存在时，THE ConfigComposer SHALL 依次追加 `_2`、`_3` 等后缀直到找到唯一值
3. THE ConfigComposer SHALL 将 experiment_name 设置为与 experiment_id 相同的值
4. WHEN 冲突检测循环达到100次仍未找到唯一ID时，THE ConfigComposer SHALL 抛出 RuntimeError 异常

### 需求 2：Workspace路径一致性（metrics 404 Bug修复）

**用户故事：** 作为开发者，我希望实验的 workspace 物理路径与 API 查询路径始终一致，以便消除 metrics 404 错误。

#### 验收标准

1. WHEN QE系统 执行单次实验时，THE QE系统 SHALL 将 qe_task_id 设置为 experiment_name（即 experiment_id），不再使用 `f"{experiment_name}_{experiment_id}"` 拼接
2. WHEN QE系统 执行单次实验时，THE QE系统 SHALL 将 loop_index 设置为 1，qe_loop_id 设置为 `"Loop1"`
3. THE QE系统 SHALL 确保 workspace 物理路径 `qe_workspace/{qe_task_id}/Loop{loop_index}/` 与 API 查询路径 `/tasks/{qe_task_id}/loops/Loop{loop_index}/*` 中的参数值完全一致
4. WHEN `_poll_and_sync` 轮询实验状态时，THE QE系统 SHALL 使用与 workspace 创建时相同的 `(qe_task_id, qe_loop_id)` 参数对

### 需求 3：统一Workspace结构

**用户故事：** 作为用户，我希望单次实验和演进实验共用同一 workspace 结构，以便从已完成的单次实验无缝开始演进。

#### 验收标准

1. WHEN 执行单次实验时，THE QE系统 SHALL 在 `qe_workspace/{experiment_id}/Loop1/` 路径下创建实验文件
2. WHEN 用户对已完成的单次实验发起演进时，THE QEEvolutionService SHALL 复用基础实验的 experiment_id 作为 task_id
3. WHEN 演进任务执行新 Loop 时，THE QEEvolutionService SHALL 在同一 workspace 下创建 `Loop{N}/` 子目录，其中 N = current_loop + 1，从 Loop2 开始
4. THE QEEvolutionService SHALL 将演进任务的 current_loop 初始值设置为 1，使第一个演进 Loop 编号为 2

### 需求 4：统一实验结果统计

**用户故事：** 作为用户，我希望所有实验结果（含演进Loop）统一存储在同一张表中，以便通过单次查询获取完整的实验结果。

#### 验收标准

1. THE qe_experiments表 SHALL 包含 `loop_index`（INTEGER DEFAULT 1）、`parent_experiment_id`（TEXT）、`is_evolution_loop`（BOOLEAN DEFAULT FALSE）三个字段
2. WHEN 演进 Loop 完成时，THE QEEvolutionService SHALL 在 qe_experiments表 中插入一条记录，其 experiment_id 为 `{task_id}_L{loop_index}`，parent_experiment_id 为 task_id，is_evolution_loop 为 TRUE
3. WHEN 查询某实验的完整结果时，THE QE系统 SHALL 通过 `WHERE experiment_id = :id OR parent_experiment_id = :id ORDER BY loop_index ASC` 返回主实验及所有演进 Loop 的记录
4. WHEN parent_experiment_id 为 NULL 时，THE qe_experiments表 中该记录 SHALL 表示主实验；WHEN parent_experiment_id 非 NULL 时，该记录 SHALL 表示演进子 Loop
5. WHEN is_evolution_loop 为 TRUE 时，THE qe_experiments表 中该记录的 parent_experiment_id SHALL 不为 NULL

### 需求 5：实验删除

**用户故事：** 作为用户，我希望能够删除失败或不需要的实验记录及其 workspace 文件，以便清理系统资源。

#### 验收标准

1. WHEN 用户请求删除一个状态为 `running` 的实验时，THE QE系统 SHALL 返回 HTTP 409 Conflict 并拒绝删除
2. WHEN 用户请求删除一个不存在的实验时，THE QE系统 SHALL 返回 HTTP 404
3. WHEN 用户请求删除一个非运行中的实验且 cleanup_workspace=true 时，THE QE系统 SHALL 通过 QEWorkspaceClient 调用 `DELETE /tasks/{experiment_id}` 删除 workspace 目录
4. WHEN workspace 清理失败（RDAgent API 不可达或目录不存在）时，THE QE系统 SHALL 记录 warning 日志并继续执行 DB 清理，在返回结果中包含 warnings 字段
5. WHEN 删除实验的 DB 清理阶段，THE QE系统 SHALL 在同一事务中按顺序删除 qe_evolution_tasks、qe_factor_experiment_metrics、子 Loop 记录（parent_experiment_id 匹配）、主实验记录
6. WHEN 删除操作成功完成后，THE QE系统 SHALL 确保 qe_experiments表、qe_evolution_tasks表、qe_factor_experiment_metrics表 中不存在任何与该 experiment_id 关联的记录

### 需求 6：演进任务管理

**用户故事：** 作为用户，我希望基于已完成的单次实验发起演进任务，以便在同一 workspace 中持续优化实验。

#### 验收标准

1. WHEN 用户对一个状态非 `completed` 的实验发起演进时，THE QEEvolutionService SHALL 抛出 ValueError 并返回 HTTP 400
2. WHEN 创建演进任务时，THE QEEvolutionService SHALL 在 qe_evolution_tasks表 中插入记录，task_id 等于 base_experiment_id，current_loop 为 1
3. WHEN 演进 Loop 完成时，THE QEEvolutionService SHALL 同时更新 qe_evolution_loops 和在 qe_experiments表 中插入子 Loop 记录，使用 `ON CONFLICT DO UPDATE` 处理重复插入

### 需求 7：rdagent 命名清理

**用户故事：** 作为开发者，我希望 QE 模块内部代码中语义为"QE实验"的 `rdagent_*` 命名全部改为 `qe_*` 前缀，以便消除命名混淆并明确模块边界。

#### 验收标准

1. THE QE系统 SHALL 将 qe_experiments表 中的 `rdagent_task_id` 列重命名为 `qe_task_id`，`rdagent_loop_id` 列重命名为 `qe_loop_id`
2. THE QE系统 SHALL 将文件 `qe_rdagent_api_client.py` 重命名为 `qe_workspace_client.py`，类名 `RdagentApiClient` 改为 `QEWorkspaceClient`
3. THE QE系统 SHALL 将 `qe_evolution_service.py` 中的属性名 `rdagent_client` 改为 `workspace_client`
4. THE QE系统 SHALL 将 `config_composer.py` 中的 `_rdagent_config_cache` 改为 `_workspace_config_cache`，`_fetch_rdagent_config` 改为 `_fetch_workspace_config`
5. THE QE系统 SHALL 将 `qe_file_sync_client.py` 中的 `_get_rdagent_api_base` 改为 `_get_qe_api_base`
6. THE QE系统 SHALL 将前端 TypeScript 类型中的 `rdagent_task_id` 改为 `qe_task_id`，`rdagent_loop_id` 改为 `qe_loop_id`
7. THE QE系统 SHALL 保留语义确实指向 RDAgent 数据的命名（如 `rdagent_task_sync`、`rdagent_sota`、`RDAGENT_FACTOR_DATA_WSL` 等）不做修改

### 需求 8：API路由重构

**用户故事：** 作为开发者，我希望 RDAgent 侧 QE API 采用双参数嵌套路由结构，以便消除单参数编码中 `rsplit("_L", 1)` 的脆弱解析逻辑。

#### 验收标准

1. THE RDAgent_QE_API SHALL 将所有 Loop 相关路由重构为 `/tasks/{task_id}/loops/{loop_id}/*` 嵌套格式
2. THE RDAgent_QE_API SHALL 删除所有 `rsplit("_L", 1)` 解析逻辑，直接从路径参数获取 task_id 和 loop_id
3. WHEN 构造 workspace 路径时，THE RDAgent_QE_API SHALL 使用 `WORKSPACE_BASE / task_id / loop_id` 拼接
4. THE QEWorkspaceClient SHALL 将所有方法签名从单参数 `(loop_id)` 改为双参数 `(task_id, loop_id)`
5. THE QE系统 SHALL 将所有调用 QEWorkspaceClient 的代码（quantevolver.py、qe_evolution_service.py）适配为双参数调用

### 需求 9：数据库迁移

**用户故事：** 作为开发者，我希望数据库迁移脚本能安全地完成列名重命名、新增字段和历史数据回填，以便支持新的数据模型。

#### 验收标准

1. THE QE系统 SHALL 按顺序执行迁移：先列名重命名（`rdagent_task_id` → `qe_task_id`、`rdagent_loop_id` → `qe_loop_id`），再新增字段（`loop_index`、`parent_experiment_id`、`is_evolution_loop`），最后回填历史数据
2. WHEN 回填历史数据时，THE QE系统 SHALL 将已有记录的 qe_task_id 设置为 experiment_name，qe_loop_id 设置为 `'Loop1'`，loop_index 设置为 1
3. THE QE系统 SHALL 更新 `init_catalog_db.py` 中的 DDL 定义，使新环境初始化时直接使用新列名和新字段
4. IF 迁移脚本重复执行，THEN THE QE系统 SHALL 通过 `IF EXISTS` 检查或异常捕获跳过已完成的步骤，避免报错

### 需求 10：QE与RDAgent隔离性

**用户故事：** 作为系统架构师，我希望 QE 实验模块与 RDAgent Task 同步模块完全隔离，以便防止数据污染和功能耦合。

#### 验收标准

1. THE QE系统 SHALL 仅以只读方式引用 catalog表（`aistock_factor_catalog`、`aistock_model_catalog`、`aistock_strategy_catalog`），不执行任何 INSERT、UPDATE 或 DELETE 操作
2. THE QE系统 SHALL 仅使用 QEWorkspaceClient 与 RDAgent 侧通信，不导入或使用 `RDAgentResultsApiClient`
3. THE QE系统 SHALL 仅在 `qe_workspace/` 路径下操作 workspace 文件，不触碰 `rdagent_workspace/` 路径
4. THE QE系统 SHALL 确保 qe_experiments表 与 `rdagent_candidate_tasks` 表之间不存在外键约束
