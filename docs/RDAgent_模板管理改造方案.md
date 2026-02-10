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

### 3.1 P0：source_data 去重（已实施）
- 修改 `get_data_folder_intro()` 排除纯 schema 文件
- 修复 `QlibModelExperiment2Feedback` 重复 LLM 调用

### 3.2 P1：基础修复
- 修复 `aistock_task_runner.py` 中 `app_tpl` 值计算
- `.gitignore` 补充 `backups/`、`history/`、`.env.backup*` 等
- 根目录清理

### 3.3 P2：API 统一
- RDAgent 侧新增模板管理 REST API
- AIstock 侧改造为调用 RDAgent API

### 3.4 P3：参数化模板切换
- 使用 `APP_TPL` 参数替代文件拷贝
- 统一配置状态管理

### 3.5 P4：备份目录迁移
- `backups/` → `git_ignore_folder/template_backups/`
- `history/` → `git_ignore_folder/template_history/`
