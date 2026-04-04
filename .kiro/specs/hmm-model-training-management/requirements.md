# 需求文档：HMM 模型训练管理

## 简介

在 QE（QuantEvolver）主菜单下新增「模型训练」页面，实现通用模型训练的完整生命周期管理。页面通过 `model_type` 字段区分不同模型类型（行业 HMM、大盘 HMM、日内 RL 执行策略等），提供统一的两维度版本管理（超参版本 × 时间版本）。当前首先实现行业 HMM（sector_hmm）的训练管理，但 UI 和后端架构支持未来扩展到任意模型类型。

## 术语表

- **HMM_Training_Page**: QE 主菜单下的「模型训练」前端页面，路径为 `/quantevolver/hmm-training`
- **HMM_API**: FastAPI 后端路由，挂载于 `/api/v1/hmm-training`，提供模型训练管理的所有 REST 接口
- **Config_Version**: 超参版本，对应一组模型超参设置，存储于 `model_train_configs` 表，通过 `model_type` 字段区分不同模型类型
- **Time_Snapshot**: 时间版本，同一超参版本下不同训练日期产生的模型快照，存储于 `model_train_snapshots` 表
- **Training_Job**: 训练任务，记录一次模型训练的执行状态和结果，存储于 `model_train_jobs` 表
- **Model_File**: 训练产出的 JSON 模型文件，路径格式为 `{HMM_MODELS_DIR}/{config_id}/{snapshot_date}/models.json`
- **HMM_MODELS_DIR**: 模型文件根目录，通过环境变量配置，默认为 `AIstock/data/hmm_models/`
- **Rolling_Training**: 定期滚动训练，按 cron 表达式或手动触发，在同一超参版本下创建新的时间快照
- **SectorHMMTrainer**: 已实现的 HMM 训练器类，位于 `AIstock/backend/quant_models/hmm/sector_hmm.py`
- **QE_Experiment**: QE 实验，通过 `EvolutionTaskCreateRequest` 创建，可集成 HMM 模型选择

## 需求

### 需求 1：超参版本管理

**用户故事：** 作为量化研究员，我希望创建和管理不同的 HMM 超参配置版本，以便对比不同超参设置下的模型效果。

#### 验收标准

1. WHEN 用户提交包含显示名称和 SectorHMMConfig 超参 JSON 的创建请求时，THE HMM_API SHALL 在 `model_train_configs` 表中创建一条新记录（model_type='sector_hmm'）并返回生成的 config_id
2. THE HMM_API SHALL 对每个 Config_Version 存储以下字段：config_id、model_type、display_name、config_json（包含 n_states、history_years、min_trading_days、cooldown_days、trending_coeff、fading_coeff、neutral_coeff）、created_at
3. WHEN 用户请求超参版本列表时，THE HMM_API SHALL 返回所有 Config_Version 记录，按 created_at 降序排列
4. WHEN 用户请求删除一个没有关联 Time_Snapshot 的 Config_Version 时，THE HMM_API SHALL 删除该记录并返回成功
5. IF 用户请求删除一个已有关联 Time_Snapshot 的 Config_Version，THEN THE HMM_API SHALL 拒绝删除并返回错误信息，说明存在关联快照
6. WHEN 用户提交的 config_json 缺少必填超参字段时，THE HMM_API SHALL 使用 SectorHMMConfig 的默认值填充缺失字段

### 需求 2：训练任务执行

**用户故事：** 作为量化研究员，我希望针对指定的超参版本触发 HMM 模型训练，以便获得最新的行业状态模型。

#### 验收标准

1. WHEN 用户针对某个 Config_Version 触发训练时，THE HMM_API SHALL 在 `model_train_jobs` 表中创建一条状态为 "pending" 的记录，并返回 job_id
2. WHEN Training_Job 开始执行时，THE HMM_API SHALL 将任务状态更新为 "running"，并记录 started_at 时间戳
3. THE HMM_API SHALL 通过 WSL 执行 Python 脚本调用 `SectorHMMTrainer.train_all_sectors()` 完成训练
4. WHEN 训练成功完成时，THE HMM_API SHALL 将任务状态更新为 "completed"，记录 completed_at 时间戳，并在 `model_train_snapshots` 表中创建对应的 Time_Snapshot 记录
5. WHEN 训练成功完成时，THE HMM_API SHALL 调用 `SectorHMMTrainer.save_models()` 将模型保存到 `{HMM_MODELS_DIR}/{config_id}/{snapshot_date}/models.json`
6. IF 训练过程中发生错误，THEN THE HMM_API SHALL 将任务状态更新为 "failed"，并在 error_message 字段中记录错误详情
7. WHILE 某个 Config_Version 已有一个状态为 "pending" 或 "running" 的 Training_Job 时，THE HMM_API SHALL 拒绝为该 Config_Version 创建新的训练任务

### 需求 3：时间版本（快照）管理

**用户故事：** 作为量化研究员，我希望查看和管理同一超参版本下不同训练日期的模型快照，以便进行版本对比和回滚。

#### 验收标准

1. WHEN 用户请求某个 Config_Version 的快照列表时，THE HMM_API SHALL 返回该版本下所有 Time_Snapshot 记录，按 trained_at 降序排列
2. THE HMM_API SHALL 对每个 Time_Snapshot 存储以下字段：snapshot_id、config_id、trained_at、model_path、sector_count（成功训练的行业数量）、status、metrics_json
3. WHEN 用户请求某个 Time_Snapshot 的详情时，THE HMM_API SHALL 返回该快照的完整信息，包括 metrics_json 中的训练指标
4. WHEN 用户请求删除一个 Time_Snapshot 时，THE HMM_API SHALL 删除数据库记录并同时删除对应的 Model_File
5. IF 删除 Model_File 时文件不存在，THEN THE HMM_API SHALL 仅删除数据库记录并在响应中标注文件已缺失

### 需求 4：定期滚动训练

**用户故事：** 作为量化研究员，我希望为超参版本配置定期自动训练计划，以便模型能持续跟踪最新的市场状态。

#### 验收标准

1. WHEN 用户为某个 Config_Version 设置滚动训练计划时，THE HMM_API SHALL 存储 cron 表达式和启用状态
2. WHEN 滚动训练计划触发时，THE HMM_API SHALL 自动创建一个新的 Training_Job，流程与手动触发训练一致
3. WHEN 滚动训练成功完成时，THE HMM_API SHALL 在同一 Config_Version 下创建新的 Time_Snapshot，保留所有历史快照
4. WHEN 用户禁用某个 Config_Version 的滚动训练计划时，THE HMM_API SHALL 停止该计划的后续自动触发
5. IF 滚动训练触发时该 Config_Version 已有正在执行的 Training_Job，THEN THE HMM_API SHALL 跳过本次触发并记录日志

### 需求 5：前端模型训练页面

**用户故事：** 作为量化研究员，我希望在 QE 主菜单中有一个统一的模型训练管理页面，支持多种模型类型的两维度版本管理，以便方便地管理所有模型的全生命周期。

#### 验收标准

1. THE HMM_Training_Page SHALL 在 QE 主菜单中显示为「模型训练」菜单项，路由路径为 `/quantevolver/model-training`
2. THE HMM_Training_Page SHALL 顶部提供模型类型筛选器（如 Tab 或下拉框），支持按 model_type 筛选配置列表，当前可选类型包括「行业 HMM」（sector_hmm），未来可扩展
3. THE HMM_Training_Page SHALL 展示当前选中 model_type 下的 Config_Version 列表，每项显示 display_name、关键超参摘要、快照数量、创建时间
4. WHEN 用户点击某个 Config_Version 时，THE HMM_Training_Page SHALL 展开显示该版本下的 Time_Snapshot 列表
5. THE HMM_Training_Page SHALL 提供创建新 Config_Version 的表单，表单字段根据当前选中的 model_type 动态渲染（sector_hmm 显示 SectorHMMConfig 字段，未来其他类型显示对应字段）
6. THE HMM_Training_Page SHALL 为每个 Config_Version 提供「触发训练」按钮
7. WHILE 某个 Config_Version 有正在执行的 Training_Job 时，THE HMM_Training_Page SHALL 显示训练进度状态（pending/running），并禁用该版本的「触发训练」按钮
8. THE HMM_Training_Page SHALL 为每个 Config_Version 提供滚动训练计划的配置入口，支持设置 cron 表达式和启用/禁用开关

### 需求 6：QE 实验集成

**用户故事：** 作为量化研究员，我希望在创建 QE 实验时选择是否启用 HMM 模型及具体版本，以便将行业状态调整纳入策略回测。

#### 验收标准

1. THE HMM_API SHALL 在 `EvolutionTaskCreateRequest` 中新增 `enable_sector_hmm`（布尔值）和 `hmm_model_version_id`（字符串，引用 snapshot_id）字段
2. WHEN 用户创建实验时启用 HMM 并选择了 hmm_model_version_id，THE HMM_API SHALL 根据 snapshot_id 自动解析出对应的 model_path，并将 `enable_sector_hmm` 和 `sector_hmm_model_path` 注入到策略配置中
3. IF 用户启用了 HMM 但未提供 hmm_model_version_id，THEN THE HMM_API SHALL 返回验证错误，提示需要选择模型版本
4. IF 用户提供的 hmm_model_version_id 对应的 Time_Snapshot 状态不是 "completed"，THEN THE HMM_API SHALL 返回验证错误，提示所选模型版本尚未完成训练
5. WHEN 创建实验的前端表单中启用 HMM 开关时，THE HMM_Training_Page SHALL 显示两级联动选择器：先选 Config_Version，再选该版本下的 Time_Snapshot
6. IF 用户提供的 hmm_model_version_id 对应的 Model_File 不存在于文件系统中，THEN THE HMM_API SHALL 返回验证错误，提示模型文件缺失

### 需求 7：模型文件路径管理

**用户故事：** 作为量化研究员，我希望模型文件的存储路径由系统自动管理，以便我只需关注模型版本选择而无需关心文件位置。

#### 验收标准

1. THE HMM_API SHALL 按照 `{HMM_MODELS_DIR}/{config_id}/{snapshot_date}/models.json` 的路径规则自动生成和管理模型文件路径
2. THE HMM_API SHALL 通过环境变量 `HMM_MODELS_DIR` 配置模型根目录，未设置时默认使用 `AIstock/data/hmm_models/`
3. WHEN 创建新的 Time_Snapshot 时，THE HMM_API SHALL 自动创建所需的目录层级
4. WHEN 删除 Time_Snapshot 时，THE HMM_API SHALL 同时删除对应的模型文件和空目录
5. THE HMM_Training_Page SHALL 对用户隐藏模型文件的物理路径，仅展示 Config_Version 名称和快照日期

### 需求 8：数据库表结构

**用户故事：** 作为开发者，我希望有清晰的数据库表结构支撑 HMM 模型版本管理，以便数据持久化和查询高效。

#### 验收标准

1. THE HMM_API SHALL 创建 `model_train_configs` 表，包含字段：config_id（UUID 主键）、model_type（文本，如 'sector_hmm'）、display_name（同 model_type 下唯一）、config_json（JSONB）、created_at（带时区时间戳）
2. THE HMM_API SHALL 创建 `model_train_snapshots` 表，包含字段：snapshot_id（UUID 主键）、config_id（外键引用 model_train_configs）、trained_at（带时区时间戳）、model_path（文本）、sector_count（整数）、status（枚举：pending/completed/failed）、metrics_json（JSONB）
3. THE HMM_API SHALL 创建 `model_train_jobs` 表，包含字段：job_id（UUID 主键）、config_id（外键引用 model_train_configs）、status（枚举：pending/running/completed/failed）、started_at（可空时间戳）、completed_at（可空时间戳）、error_message（可空文本）
4. THE HMM_API SHALL 在 `model_train_configs` 上创建 (model_type, display_name) 联合唯一索引
5. THE HMM_API SHALL 在 `model_train_snapshots.config_id` 和 `model_train_jobs.config_id` 上创建外键约束，级联行为为 RESTRICT（防止误删有关联数据的配置）
