# 实施计划：HMM 模型训练管理

## 概述

在 QE 系统中新增「模型训练」模块，按增量方式实现：先建数据库表结构，再实现后端服务层和 API 路由，接着实现训练脚本和定时调度，然后集成 QE 实验，最后实现前端页面。每个步骤构建在前一步之上，确保无孤立代码。

## 任务

- [ ] 1. 数据库表结构与初始化
  - [ ] 1.1 在 `AIstock/backend/db/init_quant_schema.py` 的 DDL 列表中新增三张表的建表语句
    - 创建 `model_train_configs` 表：config_id（TEXT PK, gen_random_uuid()）、model_type（TEXT NOT NULL）、display_name（TEXT NOT NULL）、config_json（JSONB NOT NULL）、cron_expression（TEXT）、cron_enabled（BOOLEAN DEFAULT FALSE）、created_at（TIMESTAMPTZ DEFAULT NOW()）
    - 创建 `model_train_snapshots` 表：snapshot_id（TEXT PK）、config_id（TEXT FK → model_train_configs ON DELETE RESTRICT）、trained_at（TIMESTAMPTZ DEFAULT NOW()）、model_path（TEXT NOT NULL）、sector_count（INTEGER DEFAULT 0）、status（TEXT DEFAULT 'pending'）、metrics_json（JSONB）
    - 创建 `model_train_jobs` 表：job_id（TEXT PK）、config_id（TEXT FK → model_train_configs ON DELETE RESTRICT）、snapshot_id（TEXT FK → model_train_snapshots）、status（TEXT DEFAULT 'pending'）、started_at（TIMESTAMPTZ）、completed_at（TIMESTAMPTZ）、error_message（TEXT）
    - 创建 (model_type, display_name) 联合唯一索引和 config_id 外键索引
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 2. 后端服务层：HMMTrainingService
  - [ ] 2.1 创建 `AIstock/backend/services/hmm_training_service.py`，实现 HMMTrainingService 类
    - 实现 `__init__`：从环境变量 `HMM_MODELS_DIR` 读取模型根目录，默认 `AIstock/data/hmm_models/`
    - 实现 `_fill_default_config(config_json)`：用 SectorHMMConfig 默认值（n_states=2, history_years=3.0, min_trading_days=120, cooldown_days=3, trending_coeff=1.5, fading_coeff=0.5, neutral_coeff=1.0）填充缺失字段，忽略未知字段
    - 实现 `_build_model_path(config_id, snapshot_date)`：生成 `{models_dir}/{config_id}/{snapshot_date}/models.json` 路径
    - 实现 `create_config(model_type, display_name, config_json)`：根据 model_type 填充默认值后插入 model_train_configs，返回完整记录
    - 实现 `list_configs(model_type)`：查询指定 model_type 的配置，附带 snapshot_count，按 created_at 降序
    - 实现 `delete_config(config_id)`：检查关联快照，有则拒绝（409），无则删除
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 7.1, 7.2_

  - [ ]* 2.2 编写 Property 1 属性测试：超参配置创建往返
    - **Property 1: 超参配置创建往返**
    - 使用 hypothesis 生成随机 display_name 和部分 config_json，验证创建后查询结果包含所有必需字段，缺失字段被默认值填充
    - **Validates: Requirements 1.1, 1.2, 1.6**

  - [ ]* 2.3 编写 Property 2 属性测试：超参版本列表按创建时间降序
    - **Property 2: 超参版本列表按创建时间降序**
    - 创建多个配置，验证 list_configs 返回列表中 created_at 单调递减
    - **Validates: Requirements 1.3**

  - [ ]* 2.4 编写 Property 3 属性测试：超参版本删除约束
    - **Property 3: 超参版本删除约束**
    - 创建配置，可选创建快照，验证有快照时删除被拒绝，无快照时删除成功
    - **Validates: Requirements 1.4, 1.5, 8.5**

  - [ ]* 2.5 编写 Property 11 属性测试：display_name 唯一性约束
    - **Property 11: display_name 唯一性约束**
    - 同一 model_type 下用相同 display_name 创建两次，验证第二次失败；不同 model_type 允许相同名称
    - **Validates: Requirements 8.4**

  - [ ]* 2.6 编写 Property 5 属性测试：模型文件路径生成
    - **Property 5: 模型文件路径生成**
    - 生成随机 config_id 和 snapshot_date，验证路径严格等于 `{HMM_MODELS_DIR}/{config_id}/{snapshot_date}/models.json`
    - **Validates: Requirements 2.5, 7.1, 7.2**

- [ ] 3. 检查点 — 服务层基础功能验证
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. 训练任务执行与快照管理
  - [ ] 4.1 在 HMMTrainingService 中实现训练任务相关方法
    - 实现 `trigger_training(config_id)`：检查是否有 pending/running job，有则拒绝（409）；无则插入 model_train_jobs（status='pending'），返回 job_id
    - 实现 `run_training(job_id, config_id)`：更新 status='running' + started_at，通过 `subprocess.Popen` 执行 `wsl python hmm_train_script.py --config-json ... --output-path ...`，成功时更新 status='completed' + completed_at 并创建 snapshot 记录，失败时更新 status='failed' + error_message
    - 实现 `list_jobs(config_id)`：查询该配置的训练任务列表
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ] 4.2 在 HMMTrainingService 中实现快照管理方法
    - 实现 `list_snapshots(config_id)`：按 trained_at 降序返回快照列表
    - 实现 `get_snapshot(snapshot_id)`：返回快照完整信息含 metrics_json
    - 实现 `delete_snapshot(snapshot_id)`：删除 DB 记录 + 模型文件 + 空目录，文件不存在时标注 file_missing=True
    - 实现 `resolve_model_path(snapshot_id)`：根据 snapshot_id 查询 model_path
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 7.3, 7.4_

  - [ ]* 4.3 编写 Property 4 属性测试：训练任务并发约束
    - **Property 4: 训练任务并发约束**
    - 创建配置和活跃 job（pending/running），验证重复触发被拒绝
    - **Validates: Requirements 2.7, 4.5**

  - [ ]* 4.4 编写 Property 12 属性测试：训练任务初始状态
    - **Property 12: 训练任务初始状态**
    - 触发训练，验证初始 status='pending'，started_at/completed_at/error_message 为空
    - **Validates: Requirements 2.1**

  - [ ]* 4.5 编写 Property 6 属性测试：快照列表按训练时间降序
    - **Property 6: 快照列表按训练时间降序**
    - 创建多个快照，验证 list_snapshots 返回列表中 trained_at 单调递减
    - **Validates: Requirements 3.1**

  - [ ]* 4.6 编写 Property 7 属性测试：快照详情往返
    - **Property 7: 快照详情往返**
    - 创建快照，验证 get_snapshot 返回所有必需字段且值一致
    - **Validates: Requirements 3.2, 3.3**

  - [ ]* 4.7 编写 Property 8 属性测试：快照删除同步清理
    - **Property 8: 快照删除同步清理**
    - 创建快照（可选创建文件），验证删除后 DB 记录和文件均被清理，文件不存在时 file_missing=True
    - **Validates: Requirements 3.4, 3.5, 7.4**

- [ ] 5. 训练脚本
  - [ ] 5.1 创建 `AIstock/scripts/hmm_train_script.py` 独立训练脚本
    - 接收 `--config-json` 和 `--output-path` 命令行参数
    - 解析 config_json 为 SectorHMMConfig，实例化 SectorHMMTrainer
    - 调用 `train_all_sectors()` 训练，`save_models()` 保存到 output-path
    - 输出 JSON 摘要到 stdout：`{"sector_count": N, "status": "ok"}`
    - 异常时输出 `{"status": "error", "message": "..."}` 并以非零退出码退出
    - _Requirements: 2.3, 2.5_

- [ ] 6. 检查点 — 训练任务与快照管理验证
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. FastAPI 路由层
  - [ ] 7.1 创建 `AIstock/backend/routers/hmm_training.py`，定义 Pydantic 模型和所有端点
    - 定义请求/响应模型：ConfigCreateRequest、ConfigResponse、SnapshotResponse、JobResponse、CronUpdateRequest
    - 实现超参版本 CRUD 端点：POST /configs、GET /configs、DELETE /configs/{config_id}
    - 实现训练任务端点：POST /configs/{config_id}/trigger-training（使用 BackgroundTasks 异步执行 run_training）、GET /configs/{config_id}/jobs
    - 实现快照端点：GET /configs/{config_id}/snapshots、GET /snapshots/{snapshot_id}、DELETE /snapshots/{snapshot_id}
    - 实现滚动训练端点：PUT /configs/{config_id}/cron
    - 实现辅助端点：GET /snapshots/{snapshot_id}/model-path
    - Router 使用 `prefix="/api/v1/hmm-training"`, `tags=["HMM Training"]`
    - _Requirements: 1.1-1.6, 2.1-2.7, 3.1-3.5, 4.1, 4.4_

  - [ ] 7.2 在 `AIstock/backend/main.py` 中注册 hmm_training router
    - 导入 hmm_training router 并调用 `app.include_router(hmm_training.router, prefix="/api/v1")`
    - _Requirements: 1.1_

- [ ] 8. 定期滚动训练调度
  - [ ] 8.1 在 HMMTrainingService 中实现 `update_cron(config_id, cron_expr, enabled)` 方法
    - 更新 model_train_configs 的 cron_expression 和 cron_enabled 字段
    - _Requirements: 4.1, 4.4_

  - [ ] 8.2 在 `AIstock/backend/routers/hmm_training.py` 或独立模块中实现 APScheduler 调度逻辑
    - 实现 `rolling_training_tick()` 异步函数：遍历所有 cron_enabled=True 的配置，调用 trigger_training，已有活跃任务时跳过并记录日志
    - 在 FastAPI app startup 事件中初始化 AsyncIOScheduler，添加 rolling_training_tick 定时任务
    - 在 FastAPI app shutdown 事件中关闭 scheduler
    - _Requirements: 4.2, 4.3, 4.5_

  - [ ]* 8.3 编写 Property 9 属性测试：滚动训练计划配置往返
    - **Property 9: 滚动训练计划配置往返**
    - 设置 cron 表达式和启用状态，验证查询结果一致；禁用后不被调度器选中
    - **Validates: Requirements 4.1, 4.4**

- [ ] 9. 检查点 — 后端 API 与调度验证
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. QE 实验集成
  - [ ] 10.1 修改 `AIstock/backend/routers/quantevolver_evolution.py` 的 EvolutionTaskCreateRequest
    - 新增 `enable_sector_hmm: bool = False` 和 `hmm_model_version_id: Optional[str] = None` 字段
    - 在 `create_evolution_task` 端点中新增验证逻辑：enable_sector_hmm=True 时必须提供 hmm_model_version_id；验证 snapshot 状态为 "completed"；验证模型文件存在于文件系统；解析 model_path 注入策略配置中的 `sector_hmm_model_path`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6_

  - [ ]* 10.2 编写 Property 10 属性测试：QE 实验 HMM 验证规则
    - **Property 10: QE 实验 HMM 验证规则**
    - 生成各种 snapshot 状态和文件存在组合，验证：(a) enable_sector_hmm=True 且无 version_id 时报错；(b) snapshot 非 completed 时报错；(c) 文件不存在时报错；(d) 验证通过时 model_path 一致
    - **Validates: Requirements 6.2, 6.3, 6.4, 6.6**

- [ ] 11. 前端页面：通用模型训练管理
  - [ ] 11.1 创建 `AIstock/frontend/src/app/quantevolver/model-training/page.tsx` 主页面
    - 页面标题「模型训练」，顶部模型类型 Tab 切换（当前仅 sector_hmm「行业 HMM」，未来可扩展 market_hmm、rl_execution 等）
    - 模型类型 Tab 配置通过前端常量数组定义，新增类型只需添加一项（包含 model_type、display_label、config_fields 定义）
    - 每个 Tab 下展示该 model_type 的 Config 列表：display_name、关键超参摘要、快照数量、创建时间
    - 点击 Config 展开显示该版本下的 Snapshot 列表（trained_at、sector_count、status）
    - 每个 Config 提供「触发训练」按钮，有 pending/running job 时禁用并显示状态
    - 每个 Config 提供 cron 配置入口（cron 表达式输入 + 启用/禁用开关）
    - 创建配置对话框：display_name + 根据当前 model_type 动态渲染的超参字段（sector_hmm 显示 n_states、history_years、min_trading_days、cooldown_days、trending_coeff、fading_coeff、neutral_coeff 带默认值）
    - 对用户隐藏模型文件物理路径，仅展示配置名称和快照日期
    - API 调用时传入当前选中的 model_type 作为查询参数
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 7.5_

  - [ ] 11.2 在 QE 实验创建表单中集成 HMM 模型选择器
    - 在 Evolution Task 创建表单中新增 HMM 开关（enable_sector_hmm）
    - 开关启用时显示两级联动选择器：先选 Config_Version（display_name），再选该版本下的 Time_Snapshot（trained_at + status）
    - 仅显示 status='completed' 的快照供选择
    - 将选中的 snapshot_id 作为 hmm_model_version_id 提交
    - _Requirements: 6.5_

  - [ ] 11.3 在 QE 主菜单导航中添加「模型训练」菜单项
    - 路由路径 `/quantevolver/model-training`
    - 确保菜单项在正确位置显示
    - _Requirements: 5.1_

- [ ] 12. 检查点 — 前端页面与集成验证
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. 最终集成与边界条件测试
  - [ ] 13.1 编写单元测试覆盖关键边界条件
    - config_json 为空字典时所有字段使用默认值
    - config_json 包含额外未知字段时被忽略
    - WSL subprocess 返回非零退出码时 job 标记为 failed
    - 删除快照时文件不存在的降级行为（file_missing=True）
    - enable_sector_hmm=False 时不验证 hmm_model_version_id
    - 数据库表 CREATE IF NOT EXISTS 幂等性
    - _Requirements: 1.6, 2.6, 3.5, 6.3, 8.1_

  - [ ] 13.2 确保所有模块端到端无冲突
    - 验证 hmm_training router 与现有 quantevolver_evolution router 无路径冲突
    - 验证 model_train_configs/snapshots/jobs 表与现有 app.model_train_run 表无冲突
    - 验证前端 hmm-training 页面与现有 QE 页面导航一致
    - 确保所有新增 import 和依赖正确声明
    - _Requirements: 1.1-1.6, 2.1-2.7, 3.1-3.5, 4.1-4.5, 5.1-5.7, 6.1-6.6, 7.1-7.5, 8.1-8.5_

- [ ] 14. 最终检查点 — 全量测试通过
  - Ensure all tests pass, ask the user if questions arise.

## 备注

- 标记 `*` 的任务为可选，可跳过以加速 MVP 交付
- 每个任务引用具体需求编号以确保可追溯性
- 检查点确保增量验证，避免问题累积
- 属性测试验证通用正确性属性（Property 1-12），单元测试验证具体示例和边界条件
- 数据库表为通用设计（model_train_configs/snapshots/jobs），通过 model_type 字段区分模型类型，当前仅实现 sector_hmm
- 后端使用 Python（FastAPI），前端使用 TypeScript（Next.js App Router）
