# RD-Agent 任务演进解决方案设计

> 目标：在 **不改动 RD-Agent 整体运行框架与既有行为** 的前提下，提供可配置、可回退、可对接 AIstock 的任务演进方案；支持选择 SOTA/Alpha 因子、指定模型、按场景切换提示词与配置模板，并保证每轮 loop 的结果可被 UI 观察。

## 1. 现状分析（基于代码扫描）

### 1.1 任务与演进流程

RD-Agent 在 Qlib 场景下包含三类演进入口：

- **因子演进**：`rdagent/app/qlib_rd_loop/factor.py` 使用 `FACTOR_PROP_SETTING` 驱动 `RDLoop`。@rdagent/app/qlib_rd_loop/factor.py#1-61
- **模型演进**：`rdagent/app/qlib_rd_loop/model.py` 使用 `MODEL_PROP_SETTING` 驱动 `RDLoop`。@rdagent/app/qlib_rd_loop/model.py#1-43
- **量化联合演进（因子+模型）**：`rdagent/app/qlib_rd_loop/quant.py` 根据 `QUANT_PROP_SETTING` 同时初始化因子与模型的 proposal/coder/runner/feedback，并在每轮 loop 中按 action 分支运行。@rdagent/app/qlib_rd_loop/quant.py#1-140

`QUANT_PROP_SETTING.action_selection` 当前支持 `bandit | llm | random`，且实际 action 由 `QlibQuantHypothesisGen` 产生，默认受历史反馈和 LLM 影响。@rdagent/app/qlib_rd_loop/conf.py#74-121 @rdagent/scenarios/qlib/proposal/quant_proposal.py#50-180

### 1.2 模型/因子任务生成方式

- 因子任务由 LLM 输出 JSON 后转成 `FactorTask`，由 `QlibFactorHypothesis2Experiment` 生成实验。@rdagent/scenarios/qlib/proposal/factor_proposal.py#61-132
- 模型任务同理，由 `QlibModelHypothesis2Experiment` 将 LLM 输出转为 `ModelTask`。@rdagent/scenarios/qlib/proposal/model_proposal.py#73-159

当前 **模型选择由 LLM 生成**，不存在显式“固定模型列表”机制。

### 1.3 SOTA/Alpha 因子处理

- `QlibFactorRunner` 在因子演进中聚合历史 SOTA 因子与新因子，并写入 `combined_factors_df.parquet`。@rdagent/scenarios/qlib/developer/factor_runner.py#223-312
- `QlibModelRunner` 在模型演进中读取历史 SOTA 因子（`process_factor_data`），并可叠加 `conf_baseline_factors_model.yaml` 中的 Alpha 因子列表。@rdagent/scenarios/qlib/developer/model_runner.py#150-200

因此 **SOTA 因子和 Alpha 因子已经具备“可叠加”能力，但不可被选择性筛选**。

### 1.4 配置与提示词加载机制

提示词和 YAML 模板加载统一走 `rdagent/utils/agent/tpl.py` 的 `T()`/`load_content()`，支持 **app_tpl 覆盖机制**：

- 若 `RD_AGENT_SETTINGS.app_tpl` 设置，将优先从 `app_tpl/...` 目录加载同名 YAML/模板，允许“覆盖默认模板”。@rdagent/utils/agent/tpl.py#33-78
- `RD_AGENT_SETTINGS` 内提供 `app_tpl` 配置入口。@rdagent/core/conf.py#128-132

#### 1.4.1 app_tpl 机制澄清与可用能力

- **性质**：`app_tpl` 是 RD-Agent **内置** 的模板覆盖机制（`tpl.py` 内实现），并非本方案新增能力。@rdagent/utils/agent/tpl.py#33-78
- **覆盖优先级**：当 `RD_AGENT_SETTINGS.app_tpl` 设置后，`T()` 会先尝试 `app_tpl/...` 路径，再回落到默认模板。
- **可实现能力**：
  1. **提示词版本管理**：不同场景/版本目录并存，通过切换 `app_tpl` 实现版本回退。
  2. **配置模板管理**：Qlib YAML 模板可按场景覆盖，不改动默认模板。
  3. **场景化定制**：为不同策略/任务提供不同的提示词与配置组合。
- **结论**：`app_tpl` 可以作为本方案 **主要落地方法**，满足“最小改动、可回退、与上游兼容”的要求。

#### 1.4.2 app_tpl 目录可包含的文件范围（答复整理）

- **可包含内容**：任何由 `T()` 或 `load_content()` 加载的模板文件（提示词 YAML、Qlib YAML 配置、模板片段等）。
- **典型覆盖范围**：
  1. `rdagent/scenarios/**/prompts*.yaml`（提示词）
  2. `rdagent/scenarios/qlib/experiment/*_template/*.yaml`（Qlib 配置模板）
  3. `rdagent/components/**/prompts*.yaml`（通用组件提示词）
  4. `rdagent/app/**/tpl/**` 或特定场景自带模板目录（如 finetune 场景）
- **结论**：原则上 **可以包含所有“提示词模板 + 配置模板”**，只要这些文件是通过模板加载入口读取。

#### 1.4.3 app_tpl 的版本管理方式与一致性保障（答复整理）

1. **目录分版本**：`app_tpl/<scenario>/<version>/...`，每个版本是完整模板集。
2. **一致性保障**：单任务启动前只设置一次 `RD_AGENT_SETTINGS__APP_TPL` 指向版本根目录，保证一次性全量替换。
3. **建议 Manifest**：在版本目录内放 `manifest.json`，记录模板清单与 hash，用于发布校验与回滚对比。
4. **回退方式**：切换 `RD_AGENT_SETTINGS__APP_TPL` 到旧版本目录即可回退。

#### 1.4.4 现有 app_tpl 模板情况与管理方式（答复整理）

- **已存在的模板目录**：
  - `app/finetune/llm/tpl`（finetune-llm 场景默认模板）
  - `app/finetune/data_science/tpl`（finetune-data_science 场景默认模板）
  由对应场景在启动时设置 `RD_AGENT_SETTINGS.app_tpl`。@rdagent/app/finetune/llm/conf.py#30-38 @rdagent/app/finetune/data_science/conf.py#27-35
- **QLib 场景**：当前仓库**未内置**多个版本的 `app_tpl` 目录，需要由 AIstock 或部署侧提供。
- **管理工具/命令**：当前 RD-Agent **没有独立的 app_tpl 管理命令**；模板覆盖由 `RD_AGENT_SETTINGS.app_tpl` 驱动，`config_service` 只负责模板文件读写/备份，不负责版本切换。@rdagent/app/scheduler/config_service.py#24-66

#### 1.4.5 与 AIstock 的 API 集成可行性（答复整理）

- **可行性**：可在 AIstock 侧存储模板包，发布时通过 API 写入 `app_tpl/<scenario>/<version>` 并设置 `RD_AGENT_SETTINGS__APP_TPL`。
- **复用性**：不同任务复用同一 `scenario/version` 即可复用目标与提示词/配置模板集。
- **运行参数**：仅需在任务启动前设置 `RD_AGENT_SETTINGS__APP_TPL`；如模板新增变量，需在 `T(...).r(...)` 处补充上下文，否则无额外参数要求。

### 1.5 Qlib 配置模板现状

当前 Qlib 模板主要来自以下路径：

- 因子演进模板：`rdagent/scenarios/qlib/experiment/factor_template/*.yaml`，默认 LGBModel。@rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml#1-109
- 模型演进模板：`rdagent/scenarios/qlib/experiment/model_template/conf_*`，包括 baseline 与 sota（GeneralPTNN + PyTorch）。@rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml#1-101 @rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml#1-121

当前 **ML 模型明确支持的是 LGBModel（LightGBM）**，其他传统 ML 模型未在模板中声明。

### 1.6 结果与对接 API 现状

`results_api_server.py` 已提供任务与 SOTA 锚点、资产读取等接口，适合 AIstock 拉取任务与权重：

- `/tasks`、`/tasks/{task_id}/summary`、`/tasks/{task_id}/sota_factor_anchor`
- `/tasks/{task_id}/asset_bytes`：按 `file_dict` key 读取模型权重/因子入口代码
- `/catalog/*`：读取 catalog JSON（如果存在）

见 `rdagent/app/results_api_server.py` 的路由实现。@rdagent/app/results_api_server.py#640-879

### 1.7 配置文件清单（RD-Agent 与 Qlib）

#### 1.7.1 RD-Agent 全局/环境配置

| 文件路径 | 使用场景 | 主要控制/配置 | 备注 |
| --- | --- | --- | --- |
| `.env` | RD-Agent 启动与运行全局配置 | 数据路径、日志路径、API Key、并发等全局环境参数 | `config_service` 支持备份/回滚。@rdagent/app/scheduler/config_service.py#24-166 |

#### 1.7.2 Qlib 场景 YAML 模板（qrun 配置）

| 文件路径 | 使用场景 | 主要控制/配置 | 实际调用位置 |
| --- | --- | --- | --- |
| `rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml` | 因子演进首轮（无基线因子） | LGBModel 参数、Alpha158 数据处理、回测区间/策略 | `factor_runner`：无 based_experiments 时使用。@rdagent/scenarios/qlib/developer/factor_runner.py#351-363 |
| `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_dynamic.yaml` | 因子演进（已有基线/SOTA 因子） | 动态因子加载 `combined_factors_df.parquet`、LGBModel 参数、回测区间 | `factor_runner`：有 based_experiments 时使用。@rdagent/scenarios/qlib/developer/factor_runner.py#351-363 |
| `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml` | 因子演进 + SOTA 模型评估 | 同时说明 Alpha158 与动态因子拼接、模型评估 | `factor_runner`：SOTA 模型评估分支。@rdagent/scenarios/qlib/developer/factor_runner.py#346-349 |
| `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml` | 静态因子离线实验（含 `static_path`） | 读取外部静态因子表、LGBModel 参数 | 当前 `QlibFBWorkspace` 会拒绝包含 `static_path` 的配置（不用于在线 loop）。@rdagent/scenarios/qlib/experiment/workspace.py#76-87 |
| `rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml` | 模型演进（无 SOTA 因子） | PyTorch GeneralPTNN、Alpha158 处理、回测参数 | `model_runner`：无 SOTA 因子时使用。@rdagent/scenarios/qlib/developer/model_runner.py#242-257 |
| `rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml` | 模型演进（有 SOTA 因子） | NestedDataLoader + `combined_factors_df.parquet` | `model_runner`：有 SOTA 因子时使用。@rdagent/scenarios/qlib/developer/model_runner.py#242-257 |
| `rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model_standalone.yaml` | 固定模型/手动训练 | LGBModel（Tabular 基线）、回测参数 | 当前未在 `model_runner` 默认分支中调用，可作为固定模型策略模板。 |

> 备注：`config_service.TEMPLATE_FILES` 仅包含部分模板（baseline/sota），其余模板可按需纳入模板管理。@rdagent/app/scheduler/config_service.py#24-66

### 1.8 提示词文件清单与动态替换（重点：QLib 场景）

#### 1.8.1 Qlib 场景提示词（核心）

| 文件路径 | 使用场景 | 主要目标 | 动态替换点（Jinja 变量） |
| --- | --- | --- | --- |
| `rdagent/scenarios/qlib/prompts.yaml` | 假设生成/动作选择/反馈总结 | 统一假设格式、反馈格式、action 选择 | `trace`、`experiment`、`feedback`、`hypothesis_text`、`task_details`、`combined_result`、`sota_hypothesis`、`exp`、`exp_result` 等。@rdagent/scenarios/qlib/proposal/quant_proposal.py#50-165 |
| `rdagent/scenarios/qlib/experiment/prompts.yaml` | Qlib 场景背景、接口、输出格式 | 定义因子/模型场景与代码规范 | `runtime_environment`，以及 include 的各子模板。@rdagent/scenarios/qlib/experiment/factor_experiment.py#25-40 |
| `rdagent/scenarios/qlib/experiment/prompts_core_constraints.yaml` | 因子代码骨架约束 | 约束因子脚本结构与输出 | `{function_name}` 等占位符（由因子任务名称映射）。 |
| `rdagent/scenarios/qlib/experiment/prompts_data_loading.yaml` | 因子数据加载约束 | 数据源字段/静态因子读取规范 | 无显式变量；可通过模板覆盖调整字段白名单。 |
| `rdagent/scenarios/qlib/experiment/prompts_error_prevention.yaml` | 因子错误预防 | 常见错误示例与约束 | 无显式变量。 |
| `rdagent/scenarios/qlib/experiment/prompts_dataset_info.yaml` | 数据集说明 | 股票池、结果格式与预计算因子提示 | 无显式变量。 |
| `rdagent/scenarios/qlib/experiment/prompts_language_spec.yaml` | 语言规范 | 代码注释/错误信息规范 | 无显式变量。 |
| `rdagent/scenarios/qlib/factor_experiment_loader/prompts.yaml` | 研报抽因子/模型、分类/可行性 | 抽取结构化因子与模型信息 | `report_content`、`factor_dict` 等。 |
| `rdagent/app/qlib_rd_loop/prompts.yaml` | 研报场景假设生成（备用） | 研报因子描述生成假设 | `factor_descriptions`、`report_content`（当前未发现直接调用，保留作备用）。 |

#### 1.8.2 RD-Agent 通用/组件级提示词（与 Qlib 直接相关）

| 文件路径 | 使用场景 | 主要目标 | 动态替换点 |
| --- | --- | --- | --- |
| `rdagent/components/proposal/prompts.yaml` | 通用假设生成/实验生成 | 统一 hypothesis/experiment 结构 | `scenario`、`hypothesis_specification`、`hypothesis_output_format`、`hypothesis_and_feedback` 等。 |
| `rdagent/components/coder/factor_coder/prompts.yaml` | 因子代码生成/评审 | 生成 factor.py、纠错 | `scenario`、`factor_information_str`、`code`、`execution_feedback` 等。 |
| `rdagent/components/coder/model_coder/prompts.yaml` | 模型代码生成/评审 | 生成 model.py、纠错 | `scenario`、`model_information_str`、`current_code`、`model_execution_feedback` 等。 |
| `rdagent/components/coder/CoSTEER/prompts.yaml` | 组件识别 | 组件拆解与路由 | `all_component_content`。 |

#### 1.8.3 非 Qlib 场景提示词（列举，非本方案主线）

> 这些文件服务于 Kaggle/DataScience/Finetune/CI 等场景，均可沿用 `app_tpl` 覆盖机制进行版本化替换。

`rdagent/scenarios/data_science/*/prompts*.yaml`、`rdagent/scenarios/kaggle/*/prompts*.yaml`、`rdagent/scenarios/general_model/prompts.yaml`、`rdagent/app/finetune/**/prompts*.yaml`、`rdagent/app/CI/prompts.yaml`、`rdagent/app/utils/prompts.yaml`。

### 1.9 提示词一致性检查（矛盾/不合理点）

1. **`hypothesis_output_format_with_action` JSON 引号冲突**：
   - 内容中出现 `"... follow "hypothesis_specification" ..."`，内部双引号未转义，容易生成非法 JSON。@rdagent/scenarios/qlib/prompts.yaml#77-83
2. **模型提示词与执行能力不一致**：
   - `model_hypothesis_specification` 要求专注 PyTorch 架构，但 `prompts.yaml` 中又给出 XGBoost 具体实现指导，且 `model_type` 实际只接受 `Tabular/TimeSeries`。@rdagent/scenarios/qlib/prompts.yaml#85-93 @rdagent/scenarios/qlib/experiment/prompts.yaml#215-255
3. **`qlib_model_output_format` 与实际 Qlib 训练流程不一致**：
   - 提示词要求输出 `output.pth` 且固定“8 个数”，但 `QlibModelRunner` 实际通过 `qrun` 训练并输出模型权重，不依赖 `output.pth`。@rdagent/scenarios/qlib/experiment/prompts.yaml#292-296 @rdagent/scenarios/qlib/developer/model_runner.py#150-289
4. **静态字段强约束与数据可用性风险**：
   - 因子假设要求每轮至少 2 个静态/资金流因子，但 `static_factors.parquet` 并非必然存在；若数据缺失将导致无法满足约束。@rdagent/scenarios/qlib/prompts.yaml#109-112 @rdagent/scenarios/qlib/experiment/prompts_data_loading.yaml#5-19
5. **`daily_pv.h5` 字段假设可能过宽**：
   - 提示词默认包含 `factor` 列，但实际数据未必存在，建议与数据导出侧核对。@rdagent/scenarios/qlib/experiment/prompts_data_loading.yaml#3-4

### 1.10 提示词模板动态替换细化（落地方法）

1. **基线方式（无需改代码）**：
   - 使用 `app_tpl/<scenario>/<version>/...` 覆盖对应 YAML 文件；通过环境变量 `RD_AGENT_SETTINGS__APP_TPL` 选择版本。
2. **需要新增变量时**：
   - 在调用 `T("...").r(...)` 的位置添加新的上下文字段（如 `allowed_models`、`factor_allowlist`），并在模板中以 `{{ allowed_models }}` 访问。
3. **建议的动态替换要点（按文件）**：
   - `scenarios/qlib/prompts.yaml`：
     - `hypothesis_and_feedback` / `last_hypothesis_and_feedback`：使用 `trace/experiment/feedback` 注入历史结果。
     - `action_gen.user`：注入 `hypothesis_and_feedback` 与 `last_hypothesis_and_feedback`，可扩展 `strategy_constraints`。
     - `factor_hypothesis_specification` / `model_hypothesis_specification`：可增加 `factor_allowlist`、`model_allowlist`、`data_constraints`。
   - `scenarios/qlib/experiment/prompts*.yaml`：
     - `*_interface`：注入 `schema` 或 `static_fields_whitelist`；
     - `*_output_format`：按任务需要自定义输出 JSON schema。
   - `components/coder/*/prompts.yaml`：
     - 引入 `code_policy`/`lint_rules`/`data_constraints` 作为可扩展上下文，统一约束因子/模型实现。
   - `factor_experiment_loader/prompts.yaml`：
     - 支持 `extraction_schema_version` 与 `field_alias_map`，保证研报抽取的一致性。

## 2. 目标与设计约束

### 2.1 需求拆解

1. **从 task/多 task 中选择 SOTA 与 Alpha 因子，在 RD-Agent 侧进行模型演进**。
2. **模型演进可指定模型（非随机 LLM 选择）**。
3. **基于指定的 SOTA 因子集/子集重新训练模型，生成可供 AIstock 实盘选股的权重**。
4. **提示词与配置文件的模板/版本管理方案**：不修改主框架，仅最小可回退改动，AIstock UI 侧可选择场景与模板版本。
5. **通过 RD-Agent API 实现 AIstock 配置下发**，模板存储在 AIstock 数据库。
6. **每轮 loop 结果可在 RD-Agent UI 查看**。
7. **仅支持当前 RD-Agent 已支持的 ML 模型**。

### 2.2 设计约束

- 不破坏现有 RD-Agent 运行框架；新增能力需 **可回退**。
- 模板与提示词须支持 **“覆盖式”扩展**（app_tpl），不侵入默认模板。
- AIstock 与 RD-Agent 通过 **明确契约** 对接。

## 3. 总体方案架构

### 3.1 架构分层

```text
AIstock（UI + DB）
  ├─ 场景模板管理（Prompt/Config/ModelPolicy）
  ├─ Task 配置下发（TaskConfig JSON）
  └─ 结果消费（Results API）

RD-Agent（最小改动）
  ├─ TaskConfig 解析器（可选、可回退）
  ├─ app_tpl 覆盖加载（已有机制）
  ├─ 因子/模型选择策略注入（可选）
  ├─ 训练执行（原 RDLoop）
  └─ Results API 输出与 UI 显示
```

### 3.2 关键原则

1. **不改默认行为**：无 TaskConfig 时行为完全一致。
2. **模板与提示词通过 app_tpl 覆盖**，并可按 task 选择版本。
3. **指定模型/因子为“软覆盖”**：仅在配置存在时生效。
4. **所有新配置可落盘**，记录在日志/结果 API，便于 UI 展示。

## 4. 功能详细设计

### 4.1 SOTA/Alpha 因子选择与模型演进

#### 4.1.1 需求目标

- 支持从指定 task 的 SOTA 因子中 **选择一组或子集** 用于模型演进。
- 支持选择 Alpha 因子（当前由 `conf_baseline_factors_model.yaml` 提供）进行叠加。

#### 4.1.2 设计方案

新增 `FactorSelectionPolicy`（配置驱动）：

```yaml
factor_selection:
  source_task_id: "<task_id>"       # SOTA 因子来源任务
  mode: "subset"                     # all | subset
  include_factors: ["factor_a", "factor_b"]
  exclude_factors: ["factor_c"]
alpha_factors:
  enabled: true
  include_alpha: ["RESI5", "WVMA5"]
```

实现方式（最小改动）：

1. **在 model_runner 中读取该配置**，对 `process_factor_data(...)` 结果进行列筛选；若无配置则保持原逻辑。@rdagent/scenarios/qlib/developer/model_runner.py#150-200
2. `alpha_factors` 通过现有 `load_alpha_factors_from_yaml()` 读取配置文件，再根据 `include_alpha` 做筛选即可。@rdagent/scenarios/qlib/developer/model_runner.py#18-59

#### 4.1.3 对接 AIstock

AIstock 侧配置来源：

- 通过 Results API `/tasks/{task_id}/sota_factor_anchor` 获取 SOTA 因子信息。@rdagent/app/results_api_server.py#698-846
- 选出子集后形成 `TaskConfig` 下发给 RD-Agent。

### 4.2 指定模型（非 LLM 随机）

#### 4.2.1 设计目标

在模型演进任务中，允许指定模型与超参，不再由 LLM 随机生成。

#### 4.2.2 设计方案（两种模式）

##### 模式 A：固定模型直出（推荐）

新增 `ModelPolicy`：

```yaml
model_policy:
  mode: fixed
  models:
    - name: LGBModel
      model_type: Tabular
      hyperparameters: {...}
      training_hyperparameters: {...}
      config_template: conf_baseline_factors_model_standalone.yaml
```

实现方式：

- 在 `QlibModelHypothesis2Experiment.convert_response` 前增加一个 **“短路分支”**，若 `model_policy.mode = fixed`，则直接构建 `ModelTask`，跳过 LLM 输出。@rdagent/scenarios/qlib/proposal/model_proposal.py#134-159

##### 模式 B：LLM 受限候选集

通过 prompt 注入 `allowed_models`，要求 LLM 只能从候选集中选择。该方式不改变代码结构，只需模板覆盖。

#### 4.2.3 当前支持的 ML 模型范围

现有 Qlib 模板中明确使用的是 `LGBModel`（LightGBM）。@rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml#63-78

因此在 **“仅支持已支持 ML 模型”** 约束下，默认仅开放 `LGBModel`。若未来扩展 XGBoost/CatBoost，应先新增模板再启用。

### 4.3 基于 SOTA 因子集/子集重新训练（实盘选股）

#### 4.3.1 设计目标

从指定 SOTA 因子集生成模型权重，并可由 AIstock 使用实盘数据完成选股。

#### 4.3.2 设计方案

新增任务类型：`model_retrain`（不走 LLM 模型生成）。

流程：

1. AIstock 选择 SOTA 因子集（来源 task）。
2. RD-Agent 执行模型训练：
   - 使用 `conf_baseline_factors_model_standalone.yaml` 或 `conf_sota_factors_model.yaml` 作为模板。@rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model_standalone.yaml#1-96 @rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml#1-121
   - 组合因子 DataFrame 写入 `combined_factors_df.parquet`（与现有模型演进一致）。@rdagent/scenarios/qlib/developer/model_runner.py#150-200
3. 结果权重通过 Results API 输出，AIstock 拉取 `model.pkl/params.pkl`。@rdagent/app/results_api_server.py#848-879

#### 4.3.3 输出契约（实盘选股）

RD-Agent 侧输出：

- **模型权重**：`/tasks/{task_id}/asset_bytes?key=model.pkl` 或 `params.pkl`
- **训练元数据**：可新增 `model_meta.json`（建议）
- **因子入口**：`factor_entry.py`（用于 AIstock 重放因子）

### 4.4 提示词与配置模板管理（版本化）

#### 4.4.1 设计目标

- AIstock UI 能创建模板/版本并选择场景。
- RD-Agent 不修改核心框架，仅使用最小覆盖方式。
- 模板可回退，避免与上游更新冲突。

#### 4.4.2 方案核心：app_tpl 覆盖机制

`T()` 在加载模板时会优先尝试 `app_tpl` 目录。@rdagent/utils/agent/tpl.py#33-78

设计约定：

```text
repo_root/
  app_tpl/
    <scenario>/<version>/
      rdagent/scenarios/qlib/experiment/prompts.yaml
      rdagent/scenarios/qlib/prompts.yaml
      rdagent/scenarios/qlib/experiment/model_template/*.yaml
      rdagent/scenarios/qlib/experiment/factor_template/*.yaml
```

选择模板方式：

- 任务启动前设置 `RD_AGENT_SETTINGS__APP_TPL=app_tpl/<scenario>/<version>`。@rdagent/core/conf.py#128-132
- 若未设置，则仍走默认模板路径。

#### 4.4.3 模板版本管理与回退

现有 `config_service` 已提供模板文件备份能力，可用于回退：

- `read_template/write_template` 具备历史备份能力。@rdagent/app/scheduler/config_service.py#24-166

在此基础上建议新增：

- `POST /templates/publish`：接收 AIstock 下发的模板 bundle，写入 `app_tpl/...`。
- `GET /templates/history`：查询备份历史（基于 `history/.meta_history.jsonl`）。@rdagent/app/scheduler/storage.py#1-56
- `POST /templates/rollback`：指定版本回退（调用 `rollback_file`）。@rdagent/app/scheduler/storage.py#47-53

#### 4.4.4 模板发布/回滚标准流程（API）

**发布流程（建议）**：

1. **AIstock 侧校验**：对模板包执行 YAML/Jinja 校验与必备文件完整性检查（prompts + qlib 模板）。
2. **发布接口**：`POST /templates/publish`，写入 `app_tpl/<scenario>/<version>`，同时生成 `manifest.json`（含 hash/时间戳）。@rdagent/utils/agent/tpl.py#33-78
3. **激活版本**：任务启动时设置 `RD_AGENT_SETTINGS__APP_TPL` 指向该版本目录（或由 TaskConfig 下发）。@rdagent/core/conf.py#128-132
4. **日志记录**：每次任务写入 `template_version` 与 `manifest_hash`，便于 UI 对齐回溯。@rdagent/app/qlib_rd_loop/quant.py#34-118

**回滚流程（建议）**：

1. 调用 `POST /templates/rollback` 指定版本或时间戳。
2. RD-Agent 侧通过备份/历史记录恢复对应模板（`config_service`/`storage`）。@rdagent/app/scheduler/config_service.py#24-66 @rdagent/app/scheduler/storage.py#47-53
3. 后续任务继续通过 `RD_AGENT_SETTINGS__APP_TPL` 指向回滚后的版本目录。

### 4.5 AIstock ↔ RD-Agent 接口契约

#### 4.5.1 TaskConfig 下发接口（建议）

```json
POST /aistock/task-config
{
  "task_id": "<uuid>",
  "mode": "model_retrain|factor_evolve|model_evolve|quant_evolve",
  "app_tpl": "app_tpl/qlib/v1",
  "factor_selection": {...},
  "alpha_factors": {...},
  "model_policy": {...},
  "runtime_env": {"USE_ALPHA_FACTORS": "true"}
}
```

#### 4.5.2 模板发布接口（建议）

```json
POST /aistock/templates/publish
{
  "scenario": "qlib",
  "version": "v1",
  "files": [
    {"path": "rdagent/scenarios/qlib/experiment/prompts.yaml", "content": "..."},
    {"path": "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml", "content": "..."}
  ]
}
```

#### 4.5.3 结果消费接口（现有）

- `/tasks/{task_id}/sota_factor_anchor`：获取 SOTA 因子锚点
- `/tasks/{task_id}/asset_bytes`：下载权重与因子代码
- `/tasks/{task_id}/summary`：任务摘要

均已在 Results API 中存在。@rdagent/app/results_api_server.py#640-879

### 4.6 debug_tools/run_model_comparison_with_sota.py 启动模式与 app_tpl 迁移评估

**脚本启动模式**（`--model XGBoost`）：

- 通过 `python -m rdagent.app.qlib_rd_loop.model --loop_n N` 启动 **模型演进 RDLoop**。
- 使用环境变量注入 SOTA 因子路径：`QLIB_SOTA_FACTOR_PATH=<task_dir>`，并按模型分别创建 workspace 目录。@debug_tools/run_model_comparison_with_sota.py#146-178
- 脚本包含 `modify_prompts_for_model`，用于直接改写 `rdagent/scenarios/qlib/prompts.yaml`，但当前默认注释不启用。@debug_tools/run_model_comparison_with_sota.py#91-149

**是否可用 app_tpl 替代“脚本改模板”的方式**：

- **可以**：将模型约束/提示词改动写入 `app_tpl/<scenario>/<version>/rdagent/scenarios/qlib/prompts.yaml`，任务启动前设置 `RD_AGENT_SETTINGS__APP_TPL`，无需脚本写入默认模板。@rdagent/utils/agent/tpl.py#33-78
- **结论**：app_tpl 可替代“脚本级改模板”，脚本仅保留为批量运行器。
- **命令/参数方式**：当前无独立的 `app_tpl` 命令，建议通过环境变量或 TaskConfig 设置；如需命令式调用可封装 `rdagent.app.scheduler` 的模板发布/激活 API。

**个性化定制与冲突分析**：

- 该脚本属于“定制化任务运行器”，但**当前未通过 app_tpl 做版本化管理**。
- 脚本的个性化主要来自 `QLIB_SOTA_FACTOR_PATH` 与 `loop_n/workspace`，提示词/配置默认仍是基础版本；若启用 `modify_prompts_for_model` 才会进入“直接改模板”的定制模式。
- 若同时修改默认模板 + 启用 app_tpl，会导致优先级覆盖与版本混乱；建议统一以 app_tpl 管理版本。

### 4.7 模板基线回归策略与 AIstock 集成开发边界

**是否需要先恢复 RD-Agent 模板到基础版本？**

- **建议恢复**：保持默认模板为“基线版本”，再在 `app_tpl` 创建定制版本，避免历史改动污染基线与后续对比。
- **操作建议**：以当前默认模板创建 `app_tpl/qlib/v0` 作为“基准快照”，后续版本从 v0 派生。

**AIstock 侧是否必须加入工作区？**

- **现阶段不必**：先在 RD-Agent 侧实现 `templates/publish` 等 API 与 app_tpl 落盘流程即可。
- **后续集成阶段**：AIstock UI/DB 完成后再纳入工作区，接入模板包发布、版本管理与审计能力。

### 4.8 每轮 Loop 结果在 UI 展示

现有 RDLoop 会在每一步记录 `logger.log_object(...)`，并通过日志文件供 UI 读取。@rdagent/app/qlib_rd_loop/quant.py#34-118

为满足“个性化演进任务可查看”的需求，建议在每轮 loop 开始时记录：

- `task_config_snapshot`（包含 model_policy、factor_selection 等）
- `template_version`（app_tpl 版本）

可实现为 **新增 log tag**，不影响原有逻辑。

## 5. 最小化改动清单（RD-Agent 侧）

1. **增加 TaskConfig 解析器（可选）**：
   - 读取 `task_config.json`（路径可固定在 log 或 workspace）。
   - 当配置缺失时，不影响现有行为。

2. **model_policy 固定模型分支**：
   - 在 `model_proposal.py` 增加“固定模型短路”。@rdagent/scenarios/qlib/proposal/model_proposal.py#134-159

3. **factor_selection & alpha_factors 筛选**：
   - 在 `model_runner.py` 对 `combined_factors` 做列过滤。@rdagent/scenarios/qlib/developer/model_runner.py#150-200

4. **模板发布接口**：
   - 在 scheduler API 中新增模板发布与回滚接口，沿用 `config_service`。@rdagent/app/scheduler/config_service.py#24-166

以上均为“可回退”的局部改动，不改变默认流程。

## 6. AIstock 侧配合要点（供对接）

1. **模板与提示词版本化管理**：存储在 AIstock DB，发布时推送到 RD-Agent 的 `app_tpl`。
2. **任务配置落库**：形成 `TaskConfig`，选择场景、模板版本、因子集、模型策略。
3. **结果同步**：使用 Results API 拉取模型权重和因子入口代码。

## 7. 实施步骤建议（RD-Agent 侧）

1. 实现 `TaskConfig` 解析与注入（支持 JSON 文件或 API）。
2. 实现 `model_policy` 固定模型分支。
3. 增加因子筛选逻辑。
4. 扩展 scheduler API 完成模板发布/回滚。
5. 在 UI 日志记录 TaskConfig 快照。

## 8. 风险与回退策略

- **风险：模板覆盖与上游冲突** → 使用 `app_tpl` 覆盖，仅在任务配置中启用。
- **风险：模型选择错误** → 通过 `model_policy` 限定 LGBModel，避免 LLM 随机输出。
- **风险：因子筛选造成空特征** → 加入最小因子数检查，失败即回退到全量。
- **回退**：删除 `app_tpl` 或不传 `TaskConfig` 即可恢复默认行为。

## 9. 分阶段实施路线（代码级落地）

### 9.1 核心代码不变声明

- **不修改核心执行链路**：`rdagent/app/qlib_rd_loop/*`、`rdagent/scenarios/qlib/*`、`rdagent/components/*` 保持不变。
- **允许新增/改动范围**：`app_tpl` 模板包、scheduler API 接口层、外部启动/模板渲染脚本、TaskConfig 示例/Schema。
- **默认模式不受影响**：不设置 `app_tpl` 或不使用外部启动器时，RD-Agent 仍按原有模式运行。

### 9.1.1 全量模板清单规范与快照/打包扩展（仅配置与提示词）

**原则**：模板快照与模板包只包含**配置文件与提示词文件**，不得包含任何 RD-Agent 核心程序文件（`.py` 等）。

**全量模板清单（Template Registry）**：统一登记所有可版本化模板范围，并按场景分组。

- 统一清单文件：`configs/template_registry.yaml`（新增）
- 仅纳入：
  - `rdagent/scenarios/**/prompts.yaml`
  - `rdagent/scenarios/**/experiment/prompts.yaml`
  - `rdagent/scenarios/**/experiment/**/model_template/*.yaml`
  - `rdagent/scenarios/**/experiment/**/*.yaml`
- 排除规则（必须）：
  - `**/*.py`
  - `**/*.bak`、`**/*~`、`**/backup/**`、`**/__pycache__/**`

**快照扩展（Phase 0）**：

- 读取 `template_registry.yaml`，按 include/exclude 扫描模板文件。
- 生成基线快照目录（仅配置与提示词）：`snapshots/template_snapshot_<timestamp>/rdagent/...`
- 输出 `manifest.json`：记录版本、文件清单、hash、创建时间（北京时间）。

**打包扩展（Phase 1）**：

- 基于 TaskConfig + Registry 生成模板包，输出到 `app_tpl/<scenario>/<version>/rdagent/...`
- 生成 `manifest.json`：记录 base_snapshot、patch_summary、文件清单与创建时间（北京时间）。

### 9.2 Phase 0：基线模板快照（不改核心）

新增文件：

- `app_tpl/qlib/v0/rdagent/scenarios/qlib/prompts.yaml`
- `app_tpl/qlib/v0/rdagent/scenarios/qlib/experiment/prompts.yaml`
- `app_tpl/qlib/v0/rdagent/scenarios/qlib/experiment/model_template/*.yaml`
- `app_tpl/qlib/v0/rdagent/scenarios/qlib/experiment/factor_template/*.yaml`
- `app_tpl/qlib/v0/manifest.json`
- `template_tools/template_snapshot.py`

职责说明：`template_snapshot.py` 仅复制默认模板并生成 `manifest.json`，不改动 RDLoop。

### 9.3 Phase 1：app_tpl 自定义场景验证（PoC）

新增文件：

- `app_tpl/qlib/v1/**`（自定义 prompts 与 qlib 模板）
- `app_tpl/qlib/v1/manifest.json`
- `template_tools/template_bundle_builder.py`
- `template_tools/aistock_task_runner.py`
- `configs/task_config.schema.json`
- `configs/task_config.example.json`

职责说明：

- `template_bundle_builder.py`：根据 TaskConfig 渲染/组装模板包输出到 `app_tpl/<scenario>/<version>`。
- `aistock_task_runner.py`：设置 `RD_AGENT_SETTINGS__APP_TPL`/`QLIB_SOTA_FACTOR_PATH` 等环境变量，直接调用原始 `python -m rdagent.app.qlib_rd_loop.*`。

### 9.4 Phase 2：模板发布/回滚 API（scheduler 层）

新增文件：

- `rdagent/app/scheduler/template_service.py`
- `docs/api/templates.md`

修改文件：

- `rdagent/app/scheduler/server.py`：新增 `/templates/publish|history|rollback`
- `rdagent/app/scheduler/api_stub.py`：新增模板接口 stub

职责说明：API 仅操作 `app_tpl` 与 `history` 目录，不影响任务执行链路。@rdagent/app/scheduler/server.py#16-76

### 9.5 Phase 3：AIstock 模板入库与 UI 管理

说明：该阶段改动位于 AIstock 仓库，不影响 RD-Agent 核心代码。

### 9.6 Phase 4：多场景治理与回归

新增文件：

- `tests/template_bundle_smoke_test.py`
- `docs/governance/template_policy.md`

说明：用于模板包回归与治理，不改动 RDLoop。

---

**交付说明**：本方案仅覆盖 RD-Agent 侧开发，AIstock UI 设计另行实施；但已提供清晰的接口与契约，以确保双方对接稳定。
