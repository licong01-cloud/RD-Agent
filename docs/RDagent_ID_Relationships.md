# RDAgent ID 关联关系说明文档 (Phase 3)

## 1. 核心 ID 定义

| ID 名称 | 格式 | 作用 | 存储位置 |
| :--- | :--- | :--- | :--- |
| **Task Run ID** | `str` (如 `20250106...`) | 代表一次完整的 RDAgent 任务运行。 | `task_runs` 表 |
| **Loop ID** | `int` (0, 1, 2...) | 代表任务中的一次迭代（循环）。 | `loops` 表 |
| **Workspace ID** | `str` (32位 UUID) | 代表一个物理实验目录。一个 Loop 可能产生多个 Workspace。 | `workspaces` 表 |
| **Asset Bundle ID** | `str` (32位 UUID) | 代表一个固化后的资产包。一个成功完成的 Loop 对应一个 Asset Bundle。 | `loops.asset_bundle_id` |

## 2. 实体对应关系

### 1:N 关系
- **Task Run : Loop** = 1 : N (一个任务跑多次循环)
- **Loop : Workspace** = 1 : N (一个循环可能有多个阶段，如 coding -> factor -> model)
- **Asset Bundle : Factor** = 1 : N (一个资产包包含多个因子定义)

### 1:1 关系 (逻辑上)
- **Loop : Asset Bundle** = 1 : 1 (仅对 `has_result=1` 的成功 Loop 进行固化)
- **Workspace : Artifacts** = 1 : N (一个工作区产出多类中间产物)

## 3. 关联查询路径

1. **从 Loop 找资产文件**: 
   `loops` -> `asset_bundle_id` -> 物理路径 `RDagentDB/production_bundles/{asset_bundle_id}/`
2. **从因子找来源 Loop**:
   `factor_registry` -> `task_run_id`, `loop_id` -> `loops` 表
3. **从资产包找原始 Workspace**:
   `factor_registry` -> `workspace_id` -> `workspaces` 表 (物理路径 `workspace_path`)

## 4. 目录结构简化 (Phase 3)

为了方便 AIstock 访问，固化后的资产目录已简化为：
`RDagentDB/production_bundles/{asset_bundle_id}/`
- `*.yaml` (策略配置文件)
- `*.py` (因子/模型代码)
- `{workspace_id}_model.pkl` (模型权重)
- `{workspace_id}_mlruns/` (MLflow 跟踪数据)
