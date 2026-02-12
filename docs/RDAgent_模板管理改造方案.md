# RDAgent 模板管理改造方案

> 创建时间: 2025-02-11
> 状态: 实施中

---

## 一、已完成工作

### 1.1 v4 模板同步
通过 `debug_tools/sync_v4_templates.py` 脚本，已将当前 `rdagent/` 下的提示词文件同步覆盖到 `app_tpl/all/v4/`，共 10 个有差异的文件。

### 1.2 APP_TPL 参数机制分析
- `load_content()` 函数（`rdagent/utils/agent/tpl.py`）根据 `RD_AGENT_SETTINGS.app_tpl` 优先从 app_tpl 目录加载模板
- 正确的 `app_tpl` 值：`../app_tpl/all/v4/rdagent`（相对于 PROJ_PATH 即 `rdagent/` 目录）
- 设置后 RDAgent 自动从 v4 目录加载提示词，无需文件拷贝

---

## 二、发现的问题

### 2.1 source_data 字段信息重复（运行时代码问题）

**问题描述**：`get_data_folder_intro()` 扫描数据目录中所有文件并为每个文件生成描述，导致同一组字段在 LLM 提示词中重复出现 4-5 次。

**重复来源**：
| 文件 | 重复内容 |
|------|---------|
| `daily_basic.h5` | 输出 db_ 前缀列名 + 附带 schema |
| `moneyflow.h5` | 输出 mf_ 前缀列名 + 附带 schema |
| `bak_basic.h5` | 输出 bb_ 前缀列名 + 附带 schema |
| `cyq_perf.h5` | 输出 cp_ 前缀列名 + 附带 schema |
| `static_factors.parquet` | 包含所有 H5 字段的聚合，再次输出 |
| `static_factors_schema.csv` | 独立文件，再次输出 schema |
| `static_factors_schema.json` | 独立文件，再次输出 schema |
| `README.md` | 包含所有字段的文档说明 |

**修复方案**：在 `get_data_folder_intro()` 中排除纯 schema 文件（`*_schema.csv`、`*_schema.json`），因为 H5 和 parquet 文件的 `get_file_desc()` 已通过 `_candidate_schema_paths_for_file()` 自动附带 schema 信息。

**涉及文件**：`rdagent/scenarios/qlib/experiment/utils.py`

### 2.2 feedback 阶段 background 叠加

**问题描述**：feedback 阶段调用 `get_scenario_all_desc(filtered_tag="feedback")`，内部 `common_description()` 调用 `background(tag=None)` 返回 quant + factor + model 三段背景。这是设计意图（feedback 需要全局上下文），不是 bug。

但 `common_description()` 中包含完整的 `source_data` 描述，意味着 feedback 阶段的 system prompt 也包含了重复的字段信息。修复 2.1 后此问题自动解决。

### 2.3 QlibModelExperiment2Feedback 重复调用 LLM

**问题描述**：`feedback.py` 第159-179行，`QlibModelExperiment2Feedback.generate_feedback()` 中对同一个 prompt 调用了两次 `build_messages_and_create_chat_completion()`，浪费 token 和时间。

**修复方案**：删除第二次重复调用。

### 2.4 AIstock 侧模板管理架构问题

#### 问题 A：未使用 APP_TPL 参数
- 当前通过文件拷贝覆盖 `rdagent/` 下的文件实现"应用模板"
- 应改为设置 `RD_AGENT_SETTINGS__APP_TPL` 环境变量

#### 问题 B：直接操作 WSL 文件
- AIstock 后端通过 Windows 路径直接读写 RDAgent 侧文件
- 应统一通过 RDAgent API 操作

#### 问题 C：状态记录不统一
- 激活状态：AIstock JSON vs RDAgent 无记录
- 模板列表：PG 表 vs manifest.json
- 备份：`backups/` vs `history/template_bundles/`

#### 问题 D：根目录污染
- `backups/`（370 items）、`history/`、`.env.backup*` 等未在 `.gitignore` 中
- 大量临时脚本和调试文件散落在根目录

### 2.5 aistock_task_runner.py 中 app_tpl 值计算错误

**问题描述**：`_resolve_app_tpl()` 生成的值格式为 `app_tpl/{scenario}/{version}`（如 `app_tpl/qlib/v4`），无法正确映射到 v4 模板文件。

**正确格式**：`../app_tpl/{scenario}/{version}/rdagent`

---

## 三、改造方案

### 3.1 P0：source_data 去重（已实施 ✓）

**修改文件**：`rdagent/scenarios/qlib/experiment/utils.py`

**修改内容**：
1. `get_data_folder_intro()` 排除独立 schema 文件（`*_schema.csv`、`*_schema.json`）
2. 两轮处理：先处理 H5 文件并收集列名，再处理 parquet 时排除已描述的列
3. 新增 `_extract_schema_column_names()` 辅助函数，从 schema 文件提取列名作为 H5 元数据提取失败时的补充
4. `get_file_desc()` 新增 `exclude_columns` 参数，parquet 分支（schema 和 metadata 两条路径）均支持列过滤

**修改文件**：`rdagent/scenarios/qlib/developer/feedback.py`

**修改内容**：
5. 删除 `QlibModelExperiment2Feedback.generate_feedback()` 中对同一 prompt 的第二次重复 LLM 调用

**验证结果**：
- 文件数：9 → 7（排除 2 个独立 schema 文件）
- parquet schema 列：92 → 34（排除 58 个与 H5 重复的列）
- 总输出：7785 字符，7 个文件段

### 3.2 P1：基础修复（已实施 ✓）

**修改文件**：`template_tools/aistock_task_runner.py`
- 修复 `_resolve_app_tpl()` 路径计算：`app_tpl/{scenario}/{version}` → `../app_tpl/{scenario}/{version}/rdagent`
- 默认 scenario 从 `qlib` 改为 `all`

**修改文件**：`.gitignore`
- 补充 `backups/`、`history/`、`scheduler_data/` 目录
- 补充根目录临时文件模式

**根目录清理**：
- 临时脚本移至 `debug_tools/`
- 临时数据文件移至 `debug_tools/`

### 3.3 P2：API 统一（待实施）

**目标**：AIstock 侧所有对 RDAgent 文件的操作改为通过 RDAgent API 完成。

**AIstock 侧直接操作 RDAgent 文件的操作清单**：

| 操作 | 函数/API | 直接文件操作 | 改造方案 |
|------|---------|-------------|---------|
| 模板列表 | `_collect_template_items()` | 读取 `app_tpl/manifest.json` | 新增 RDAgent API: `GET /templates/list` |
| 发布模板 | `publish_template()` | 写入 `app_tpl/` | 已有 RDAgent API: `POST /templates/publish` |
| 读取文件 | `get_template_file()` | 读取 `app_tpl/{path}` | 新增 RDAgent API: `GET /templates/{sc}/{ver}/file` |
| 保存文件 | `save_template_file()` | 写入 `app_tpl/{path}` | 新增 RDAgent API: `POST /templates/{sc}/{ver}/file` |
| 删除模板 | `delete_template()` | 删除 `app_tpl/` 目录 | 新增 RDAgent API: `DELETE /templates/{sc}/{ver}` |
| 刷新SHA256 | `refresh_template_sha256()` | 读写 manifest | 新增 RDAgent API: `POST /templates/{sc}/{ver}/refresh-sha256` |
| **应用模板** | `_apply_template_files()` | 拷贝到 `rdagent/` | **P3 替代：设置 APP_TPL 参数** |
| **创建备份** | `_create_backup()` | 拷贝 `rdagent/` | **P3 替代：APP_TPL 无需备份运行时文件** |
| **回滚** | `_rollback_from_backup()` | 拷贝到 `rdagent/` | **P3 替代：切换 APP_TPL 参数** |
| **验证** | `_verify_template_applied()` | 读取 `rdagent/` SHA256 | **P3 替代：APP_TPL 自动加载** |
| **同步状态** | `get_sync_status()` | 读取 `rdagent/` | **P3 替代：检查 APP_TPL 值** |

**RDAgent 侧需新增 API**：
1. `GET /templates/list?scenario=xxx` - 列出模板（替代 AIstock 直接扫描 app_tpl/）
2. `GET /templates/{sc}/{ver}/files` - 文件列表
3. `GET /templates/{sc}/{ver}/file?path=xxx` - 读取文件
4. `POST /templates/{sc}/{ver}/file?path=xxx` - 保存文件
5. `DELETE /templates/{sc}/{ver}` - 删除模板
6. `POST /templates/{sc}/{ver}/refresh-sha256` - 刷新 SHA256
7. `GET /templates/active` - 获取当前激活模板（APP_TPL 值）
8. `POST /templates/activate` - 激活模板（设置 APP_TPL）

### 3.4 P3：参数化模板切换（待实施）

**目标**：使用 `APP_TPL` 参数替代文件拷贝，消除"应用模板"操作。

**改造要点**：
- "应用模板" = 设置 `RD_AGENT_SETTINGS__APP_TPL=../app_tpl/{sc}/{ver}/rdagent`
- "回滚" = 切换 APP_TPL 到之前的版本
- "验证" = 检查 APP_TPL 值是否指向正确目录
- "备份" = 不再需要（app_tpl/ 目录本身就是版本化的）
- 激活状态统一存储在 RDAgent 侧（.env 或运行时配置）

### 3.5 P4：备份目录迁移（待实施）
- `backups/` → `git_ignore_folder/template_backups/`（或直接删除，P3 后不再需要）
- `history/` → `git_ignore_folder/template_history/`
