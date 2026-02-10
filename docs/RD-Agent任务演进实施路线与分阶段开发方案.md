# RD-Agent 任务演进实施路线与分阶段开发方案

> 依据文档：`docs/RD-Agent任务演进解决方案设计.md`（现状与理论基础）

## 1. 总体目标与原则

### 1.1 总体目标

1. 以 **app_tpl** 为核心实现提示词/配置模板的版本化与可回退替换。
2. 先在 RD-Agent 侧完成最小可行验证，再逐步引入多版本管理与 API 集成。
3. 最终与 AIstock UI/DB 打通，实现模板全生命周期管理、审计与复用。

### 1.2 实施原则

- **先验证、后扩展**：先做单场景自定义验证，再做多版本管理。
- **基线优先**：默认模板保持“基线版本”，所有定制在 app_tpl 中完成。
- **接口稳定**：对 RD-Agent 主流程改动最小化，新增能力通过可回退方式接入。

### 1.3 核心代码不变声明（本方案前提）

- **不修改核心执行链路**：`rdagent/app/qlib_rd_loop/*`、`rdagent/scenarios/qlib/*`、`rdagent/components/*` 保持不变。
- **允许新增/改动范围**：`app_tpl` 模板包、scheduler API 接口层、外部启动/模板渲染脚本、TaskConfig 示例/Schema。
- **默认模式不受影响**：不设置 `app_tpl` 或不使用外部启动器时，RD-Agent 仍按原有模式运行。

## 2. 分阶段实施概览（概要设计）

| 阶段 | 目标 | 关键产物 | 验收标准 |
| --- | --- | --- | --- |
| Phase 0 | 基线固化与模板快照 | app_tpl/qlib/v0、manifest 结构 | 能在不改默认模板的情况下完成替换与回退 |
| Phase 1 | app_tpl 自定义场景验证 | 自定义 prompts/config 版本、验证脚本 | 指定场景可成功运行并可回退 |
| Phase 2 | 多版本模板管理 + API 发布/回滚 | publish/rollback API、版本校验 | 多版本可切换、任务日志可追溯 |
| Phase 3 | AIstock 模板入库与 UI 管理 | DB 模型、发布流程、UI 管理 | AIstock 可发布模板并驱动 RD-Agent 替换 |
| Phase 4 | 全场景复用与治理 | 多场景模板库、策略治理 | 多场景复用可控、审计与回退可用 |

### 2.1 文件级改动总览（按阶段）

| 阶段 | 新增/修改文件 | 用途 | 是否影响核心 |
| --- | --- | --- | --- |
| Phase 0 | `app_tpl/qlib/v0/**`、`app_tpl/qlib/v0/manifest.json`、`template_tools/template_snapshot.py` | 基线模板快照与清单生成 | 否 |
| Phase 1 | `app_tpl/qlib/v1/**`、`template_tools/template_bundle_builder.py`、`template_tools/aistock_task_runner.py`、`configs/task_config.schema.json`、`configs/task_config.example.json` | 自定义模板渲染与外部启动器 | 否 |
| Phase 2 | `rdagent/app/scheduler/server.py`、`rdagent/app/scheduler/api_stub.py`、`rdagent/app/scheduler/template_service.py`、`docs/api/templates.md` | 模板发布/回滚 API | 否 |
| Phase 3 | AIstock 侧工程文件（不在本仓库） | 模板入库/UI 管理 | 否 |
| Phase 4 | `tests/template_bundle_smoke_test.py`、`docs/governance/template_policy.md` | 多场景治理与回归 | 否 |

### 2.2 外部脚本与 API 职责边界（代码级）

- `template_tools/template_snapshot.py`：仅复制默认模板并生成 `manifest.json`，不改动任何 RDLoop。
- `template_tools/template_bundle_builder.py`：根据 TaskConfig 渲染/组装模板包（输出到 `app_tpl/...`）。
- `template_tools/aistock_task_runner.py`：作为“外部启动器”，设置环境变量后调用原始 `python -m rdagent.app.qlib_rd_loop.*`。
- `rdagent/app/scheduler/*`（API 层）：新增发布/回滚入口，不影响任务执行链路。

## 3. Phase 0：基线固化与模板快照

### 3.1 目标

- 固化 RD-Agent 当前默认模板为“基线版本”。

### 3.2 关键任务

1. 从默认模板目录生成 `app_tpl/qlib/v0` 快照（提示词 + Qlib 配置模板）。
2. 规范 `manifest.json` 结构（记录模板列表、hash、时间戳）。
3. 在任务日志中记录 `template_version` 字段（仅记录，不影响流程）。

### 3.3 验收标准

- 在不修改默认模板的情况下，通过设置 `RD_AGENT_SETTINGS__APP_TPL` 可完成 v0 覆盖与回退。

### 3.4 代码级落地（文件清单）

新增：

- `app_tpl/qlib/v0/rdagent/scenarios/qlib/prompts.yaml`
- `app_tpl/qlib/v0/rdagent/scenarios/qlib/experiment/prompts.yaml`
- `app_tpl/qlib/v0/rdagent/scenarios/qlib/experiment/model_template/*.yaml`
- `app_tpl/qlib/v0/rdagent/scenarios/qlib/experiment/factor_template/*.yaml`
- `app_tpl/qlib/v0/manifest.json`（模板清单与 hash）
- `template_tools/template_snapshot.py`（从默认模板生成 v0 快照）

### 3.5 代码级执行说明（template_snapshot.py）

功能流程：

1. 读取默认模板路径：
   - `rdagent/scenarios/qlib/prompts.yaml`
   - `rdagent/scenarios/qlib/experiment/prompts.yaml`
   - `rdagent/scenarios/qlib/experiment/model_template/*.yaml`
   - `rdagent/scenarios/qlib/experiment/factor_template/*.yaml`
2. 复制到 `app_tpl/qlib/v0/` 对应路径。
3. 计算每个文件 hash，生成 `manifest.json`：
   - `version` / `created_at` / `files[{path, sha256}]`。

## 4. Phase 1：app_tpl 自定义场景验证（PoC）

### 4.1 目标

- 以一个自定义场景为起点，验证 app_tpl 的可用性与不侵入性。

### 4.2 关键任务

1. 新建 `app_tpl/qlib/v1`：
   - 覆盖 `rdagent/scenarios/qlib/prompts.yaml` 与关键 Qlib 模板（示例：限定模型范围）。
2. 通过环境变量或 TaskConfig 指定 `RD_AGENT_SETTINGS__APP_TPL`。
3. 运行一次 Qlib RDLoop（如模型演进）并验证结果。

### 4.3 验收标准

- 指定模板版本后能正常运行；
- 取消 `app_tpl` 设置即恢复默认行为。

### 4.4 代码级落地（文件清单）

新增：

- `app_tpl/qlib/v1/**`（自定义 prompts 与 qlib 模板，限定模型/因子范围）
- `app_tpl/qlib/v1/manifest.json`（模板清单与 hash）
- `template_tools/template_bundle_builder.py`：读取 TaskConfig，渲染 prompts/config 并生成模板包
- `template_tools/aistock_task_runner.py`：读取 TaskConfig -> 设置 `RD_AGENT_SETTINGS__APP_TPL`/环境变量 -> 启动原有 RDLoop
- `configs/task_config.schema.json`：TaskConfig 校验 Schema
- `configs/task_config.example.json`：示例任务配置

### 4.5 代码级执行说明（Phase 1 外部启动器）

`template_tools/template_bundle_builder.py`：

1. 读取 `TaskConfig`（JSON）。
2. 依据 `scenario/version` 确定输出目录 `app_tpl/<scenario>/<version>/`。
3. 若 TaskConfig 含 `model_allowlist`/`factor_allowlist`：
   - 覆盖对应 prompts 模板（如 `rdagent/scenarios/qlib/prompts.yaml`）。
4. 输出 `manifest.json`（同 Phase 0 规则）。

`template_tools/aistock_task_runner.py`：

1. 读取 `TaskConfig` 并设置环境变量：
   - `RD_AGENT_SETTINGS__APP_TPL=app_tpl/<scenario>/<version>`
   - `QLIB_SOTA_FACTOR_PATH=<task_dir>`（若指定 SOTA 来源）
   - 其他 `runtime_env` 透传
2. 根据 `mode` 调用原有 RDLoop：
   - `python -m rdagent.app.qlib_rd_loop.model --loop_n N`
   - `python -m rdagent.app.qlib_rd_loop.factor --loop_n N`
   - `python -m rdagent.app.qlib_rd_loop.quant --loop_n N`

说明：外部启动器 **不修改** RDLoop 逻辑，仅负责环境变量与模板版本注入。

## 4.6 模板版本化流程说明（v0 基线 -> vN 迭代）

### v0 是否基线？

- 是。`v0` 用于固化“默认模板快照”，只生成一次，不再改动。

### 后续版本如何生成？

- 推荐流程：
  1. **从 v0 复制模板基线**（`template_base=default` 时自动拷贝默认模板快照）。
  2. 在 TaskConfig 中声明变更：
     - `template_files`（替换/新增模板文件）
     - `prompt_patch`（补丁式修改提示词）
  3. 生成新版本 `v1/v2/v3...` 并写入 `app_tpl/<scenario>/<version>/`。

### 版本说明字段与适用场景

- 当前 `manifest.json` 已包含：`scenario`、`version`、`task_id`、`created_at`、`files[]`、
  `description`、`base_version`、`changed_files`。

### 是否全量替换？

- 现有方案**始终输出全量模板包**：
  - 先复制默认模板（或基线）
  - 再应用补丁/替换
  - 生成完整 `app_tpl/<scenario>/<version>`
- 因此**不需要判断改了哪些文件**，也不会出现“部分替换导致缺失”的问题。

## 5. Phase 2：多版本模板管理与 API 发布/回滚

### 5.1 目标

- 在 RD-Agent 侧形成多版本模板发布/回滚能力。

### 5.2 关键任务

1. 在 scheduler API 中新增：
   - `POST /templates/publish`
   - `POST /templates/history`
   - `POST /templates/rollback`
2. 在发布时执行：
   - YAML/Jinja 校验、文件完整性校验
   - 生成/更新 `manifest.json`
3. 任务启动时写入 `template_version` 和 `manifest_hash`。

### 5.2.x 实施进度（2026-01-20）

- [x] 模板发布/历史/回滚 API 已实现并接入 FastAPI。
- [x] 发布前 YAML/JSON/Jinja 校验 + manifest 生成。
- [x] manifest 版本说明字段（description/base_version/changed_files）已落地。
- [x] 模板发布 API 文档已补充（`docs/api/templates.md`）。

### 5.3 验收标准

- 多版本模板可切换、可回滚；
- 任务日志能追溯模板版本。

### 5.4 代码级落地（文件清单）

新增：

- `rdagent/app/scheduler/template_service.py`：模板发布/回滚逻辑封装（写入 `app_tpl`、生成 manifest）
- `docs/api/templates.md`：模板发布与回滚 API 文档

修改：

- `rdagent/app/scheduler/server.py`：新增 `/templates/publish|history|rollback`
- `rdagent/app/scheduler/api_stub.py`：新增模板接口 stub

### 5.5 API 代码级落地说明（Phase 2）

`rdagent/app/scheduler/template_service.py`（新增）：

- `publish_templates(bundle)`：
  1. 校验文件路径与 YAML/Jinja 合法性；
  2. 写入 `app_tpl/<scenario>/<version>/...`；
  3. 生成 `manifest.json`；
  4. 写入 history 记录。
- `list_template_history()`：读取 `history/.meta_history.jsonl`。
- `rollback_template(version|backup_path)`：调用 `storage.rollback_file` 回退。

`rdagent/app/scheduler/server.py`（修改）：

- 路由新增 `/templates/publish`、`/templates/history`、`/templates/rollback`。

说明：所有新增接口仅操作 `app_tpl` 与 history 目录，不影响任务执行链路。

## 6. Phase 3：AIstock 模板入库与 UI 管理

### 6.1 目标

- AIstock 侧实现模板的入库、发布、回滚与 UI 管理。

### 6.2 关键任务

1. 模板数据模型设计（版本、场景、文件、hash、发布状态）。
2. 发布流程：AIstock -> RD-Agent `/templates/publish`。
3. UI 支持：编辑、diff、发布、回滚、历史查看。

### 6.3 验收标准

- AIstock UI 能完成模板发布；
- RD-Agent 通过 `app_tpl` 生效。

### 6.4 代码级落地（文件清单，AIstock 侧）

说明：该阶段改动位于 AIstock 仓库，不影响 RD-Agent 核心。

### 6.5 Phase 3 开发与完成情况（AIstock 侧）

> 结论：AIstock 模板管理 UI 已完成开发，左侧导航已提供入口；但**尚未进行完整联调验证**（需按环境执行发布/回滚/文件编辑验证）。

#### 6.5.1 已完成内容（开发完成）

- [x] 前端模板管理 UI（调度页）：模板列表、筛选、详情、文件列表、拖拽排序、内容编辑与保存、diff 预览。
- [x] v0 版本只读限制与提示。
- [x] 历史查询、回滚与发布流程表单。
- [x] 错误提示、未保存变更保护、保存/发布/回滚反馈提示。
- [x] 时间戳展示统一为北京时间（`Asia/Shanghai`）。
- [x] 左侧导航栏入口：`/scheduler`（RD-Agent 调度页）。

#### 6.5.2 后端接口（AIstock 侧）

- [x] `/api/v1/rdagent/templates` 模板列表
- [x] `/api/v1/rdagent/templates/{scenario}/{version}/files` 文件列表
- [x] `/api/v1/rdagent/templates/{scenario}/{version}/file` 文件读取/保存（v0 禁止保存）
- [x] 已注册路由模块 `backend/routers/rdagent_templates.py`

#### 6.5.3 关键文件（AIstock 仓库）

- `frontend/src/app/scheduler/page.tsx`：模板管理 UI 主体
- `backend/routers/rdagent_templates.py`：模板管理 API
- `backend/main.py`：路由注册
- `backend/routers/__init__.py`：路由聚合

#### 6.5.4 验证状态（需补充）

- [x] Scheduler API 与 Results API 共用服务：通过 Results API 挂载 `/scheduler` 完成统一服务验证。
- [x] 模板发布：UI -> RD-Agent `/templates/publish` 联调验证（经 `/scheduler` 路由）。
- [x] 模板历史查询：UI -> RD-Agent `/templates/history` 联调验证（经 `/scheduler` 路由）。
- [x] 模板回滚：UI -> RD-Agent `/templates/rollback` 联调验证。
- [x] 文件编辑保存：AIstock API -> 实际文件内容写入与读取一致性验证。
- [x] v0 版本编辑禁用与后端 403 校验验证。

### 6.6 Phase 3 v2：模板管理 UI/流程升级（设计方案）

#### 6.6.1 目标

- 模板列表独占一行展示，默认显示全量模板。
- 发布与编辑合并为单卡片，编辑完成后直接填写发布信息并发布。
- 引入“是否应用”状态，同一时间仅允许一个模板处于应用状态。
- 模板统一为 RD-Agent + qlib 全量模板，不再保存仅 qlib 子集版本。

#### 6.6.2 交互与布局

- 所有卡片独立成行。
- 模板列表卡片排在第一位，展示全部模板。
- 模板编辑卡片独占一行：左侧文件列表，右侧文本编辑器；每个文件可单独编辑、保存。
- 发布表单嵌入编辑卡片下方（填写描述、版本等后点击发布）。

#### 6.6.3 模板状态与规则

- 新增模板应用状态字段：`is_active`（UI 展示“已应用/未应用”）。
- 同一时间仅允许一个模板为应用状态。
- 列表中：
  - v0 模板默认显示为“已应用”，且不可编辑。
  - 非 v0 模板提供“编辑”“应用”按钮。
- 点击“编辑”会在下方编辑卡片中加载该模板。
- 点击“应用”仅切换应用状态，不改变发布语义。

#### 6.6.4 模板创建策略

- 创建新模板必须基于一个源模板。
- 默认源模板为“RD-Agent + qlib v0”。
- 当模板数量增多时，源模板通过下拉框选择。
- 创建时自动复制源模板的所有文件内容作为初始内容。

#### 6.6.5 数据规范

- 所有模板必须为 RD-Agent + qlib 的全量模板。
- 若已确认全量 v0 模板存在，则删除现有 qlib v0 及其衍生版本。
- 保证 RD-Agent 运行始终依赖 qlib 统一模板，避免仅修改 qlib 子集导致运行不可预测。

#### 6.6.6 后端配合（Phase 3 v2 对应扩展）

- 模板列表接口需返回 `is_active`。
- 新增应用切换接口（例如：`POST /rdagent/templates/{scenario}/{version}/activate`）。
- 保障 v0 模板只读策略不变。

## 7. Phase 4：多场景复用与治理

### 7.1 目标

- 形成跨场景模板库与治理机制。

### 7.2 关键任务

1. 按场景拆分模板版本（qlib/kaggle/finetune 等）。
2. 多环境回归测试与模板合规校验。
3. 增加审计与角色权限控制。

### 7.3 验收标准

- 多场景模板复用可控；
- 回滚与审计流程稳定。

### 7.4 代码级落地（文件清单）

新增：

- `tests/template_bundle_smoke_test.py`：模板包完整性/渲染回归
- `docs/governance/template_policy.md`：模板治理规范与发布流程

## 8. 风险与回退策略

- **模板冲突风险**：统一通过 app_tpl 管理，不直接改默认模板。
- **发布误配置**：通过 manifest 校验与回滚机制降低影响。
- **跨场景复用风险**：强制场景隔离，避免模板混用。

## 9. 里程碑推进建议（滚动式）

- 每阶段完成后进入下阶段；若阶段验收失败，优先回滚或延长验证周期。
- Phase 0/1 可先在 RD-Agent 侧闭环，Phase 2 之后再引入 AIstock。
