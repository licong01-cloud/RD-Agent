# RD-Agent × AIstock Phase 2 详细设计补充：因子共享包与成果导出 API

> 本补充文档在《2025-12-23_Phase2_Detail_Design_RD-Agent_AIstock.md》基础上，细化 Phase 2 中：
> - RD-Agent 侧如何组织因子实现源码（因子共享包）；
> - 如何将因子/策略/模型训练结果通过只读 API 暴露给 AIstock；
> - AIstock 侧如何消费这些成果并落库/展示。
> 不改变原有 Phase 2 目标，仅对“成果导出与访问方式”做增强。

---

## 1. RD-Agent 侧：因子共享包维护与元数据扩展

### 1.1 因子共享包结构

- 在 RD-Agent 仓库平级目录下维护一个独立 Python 包目录，例如：

  ```text
  F:\Dev\rd-factors-lib\
    rd_factors_lib\
      __init__.py
      alpha158.py
      momentum.py
      volume.py
      cross_section.py
      generated.py
      VERSION
  ```

- 安装方式：

  ```bash
  # 在 RD-Agent 与 AIstock 的各自虚拟环境中执行一次
  pip install -e F:\Dev\rd-factors-lib
  ```

- 版本管理：
  - `VERSION` 文件中记录当前版本字符串（如 `"1.0.7"`），或在 `__init__.py` 中定义 `__version__`；
  - RD-Agent 在导出因子元数据时，会读取该版本号并写入 `impl_version` 字段。

### 1.2 因子演进流程对共享包的更新

在 RD-Agent 因子演进 loop 的末尾（即某个候选因子通过验证被标记为“成功”时），增加一步“共享包入库”逻辑：

1. 从当前 loop 的因子实现（原始生成的 `factor.py` 片段或字符串）中抽取出核心函数，实现签名标准化：

   ```python
   def factor_xxx(df: pd.DataFrame) -> pd.Series | pd.DataFrame:
       ...
   ```

2. 将该函数写入或更新到 `rd_factors_lib/generated.py`：
   - 若函数名已存在，则覆盖旧实现（并在 commit message 或内部注释中保留历史信息）；
   - 若是新因子，则追加新的函数定义。

3. 更新版本号：
   - 自动递增 `VERSION` 中的次版本号或补丁号（例如从 `1.0.7` → `1.0.8`）。

4. 在本 loop 对应的 `factor_meta.json` 记录该因子的实现指针：

   ```json
   {
     "name": "FACTOR_XXX",
     "source": "rd_agent",
     "impl_module": "rd_factors_lib.generated",
     "impl_func": "factor_xxx",
     "impl_version": "1.0.8",
     ...
   }
   ```

> 注：上述 1–4 步由 RD-Agent 内部 Python 代码自动完成，**不依赖人工编辑共享包文件**。

### 1.3 Alpha158 因子与共享包

- Alpha158 因子元信息仍由 `tools/export_alpha158_meta.py` 从 Qlib 配置中导出，形成 `alpha158_meta.json`；
- 在 Phase 2 内：
  - RD-Agent 不强制在共享包中实现 Alpha158 的完整因子函数集合；
  - 仅保证 Expression（Qlib 表达式）和元信息完整；
- 若需对部分 Alpha158 因子提供 Python 参考实现，可在 `alpha158.py` 中提供对应函数，并在导出元数据时为这些因子补充 `impl_module/impl_func` 字段。

---

## 2. RD-Agent 侧只读成果 API 设计（Phase 2 范围）

### 2.1 服务定位与部署

- 服务名称（示意）：`rdagent-results-api`；
- 部署位置：
  - 运行在 RD-Agent 同一环境（例如 WSL 内部），监听本机端口（如 `http://localhost:9000`）；
- 安全与边界：
  - 只提供只读接口，不执行交易或生成实时信号；
  - 访问范围限定在本机或受控内网。

### 2.2 核心接口定义（示意）

> 这里给出推荐接口形态，具体路径与参数可以在实现时微调，但语义应保持一致。

#### 2.2.1 Catalog 相关

- `GET /catalog/factors`
  - 返回内容：等价于当前 `factor_catalog.json` 的结构；
  - 支持查询参数：
    - `source`（如 `qlib_alpha158` / `rd_agent`）
    - `name` 前缀过滤等（可选）。

- `GET /catalog/strategies`
  - 返回内容：等价于 `strategy_catalog.json`；
  - 关键字段包括：`strategy_id`, `step_name`, `action`, `template_files`, `data_config`, `model_config` 等。

- `GET /catalog/loops`
  - 返回内容：等价于 `loop_catalog.json`；
  - 支持分页或按 `task_run_id` / 时间过滤（可选）。

#### 2.2.2 因子与实现指针

- `GET /factors/{name}`
  - 返回内容：单个因子在 `factor_meta.json` 中的完整记录；
  - 字段包括：
    - `name`, `source`, `description_cn`, `formula_hint`, `tags`；
    - 表现指标（来自 `factor_perf`）；
    - 以及实现指针：`impl_module`, `impl_func`, `impl_version`（如有）。

- `GET /alpha158/meta`
  - 返回内容：`alpha158_meta.json` 的完整内容或子集视图；
  - 供 AIstock 构建 Alpha158 因子库视图和后续迁移使用。

#### 2.2.3 实验与 artifacts

- `GET /task_runs` / `GET /loops` / `GET /workspaces`
  - 封装 registry.sqlite：返回任务、循环、workspace 的元信息；
  - 提供必要过滤参数（如状态、时间区间等）。

- `GET /loops/{task_run_id}/{loop_id}/artifacts`
  - 返回指定 loop 所有关键 artifacts 的路径与摘要信息：
    - `factor_meta`, `factor_perf`, `feedback`, `ret_curve`, `dd_curve` 等；
  - 字段包括：
    - 文件相对路径、类型、更新时间戳、大小等；
    - 对应在 `artifacts` / `artifact_files` 中的 ID。

#### 2.2.4 可选：因子包归档

- `GET /factor_package/bundle?version={version}`
  - 返回指定版本的因子共享包归档（tar/zip）；
  - 用于离线备份，并非日常同步主通道。

---

## 3. AIstock 侧：Phase 2 中对成果的接入与落库

### 3.1 与 RD-Agent API 的交互流程

1. AIstock 后端提供一个“RD-Agent 同步任务”模块：
   - 定时任务或 UI 触发：
     - 调用 `GET /catalog/*` 与 `GET /factors/*` 等接口，拉取最新的因子/策略/loop/Alpha158 信息；
   - 将增量/全量结果写入本地数据库：
     - 使用 upsert 策略，按主键（如 `(name, source)` / `strategy_id` / `(task_run_id, loop_id, workspace_id)`）覆盖更新。

2. 因子共享包版本对齐：
   - 从因子元数据或专门接口中读取当前生效的 `impl_version`；
   - 将该版本号记录在 AIstock 本地配置/DB 中；
   - 若 RD-Agent 与 AIstock 共享同一物理目录（如 `F:\Dev\rd-factors-lib`）：
     - AIstock 只需在虚拟环境中执行一次 `pip install -e`，即可获得该包实现；
     - 版本号用于“对齐判断”，而非下载逻辑；
   - 若未来需要支持不同版本并存，可在本地维护多个安装路径，并通过 `impl_version` 选择对应环境（此为 Phase 3/4 可选增强）。

### 3.2 本地数据库结构扩展建议

- 在已有 Phase 2 设计的基础上，对因子表/策略表/实验表增加以下字段：

- 因子表（如 `factor_registry`）：
  - 新增：
    - `impl_module: text`
    - `impl_func: text`
    - `impl_version: text`

- 策略表（如 `strategy_registry`）：
  - 可新增：
    - `model_type: text`
    - `train_start`, `train_end`: text / date
    - `val_start`, `val_end`: text / date
    - `test_start`, `test_end`: text / date

- 实验表（如 `loop_result`）：
  - 可新增：
    - `model_run_id: text`（对应 RD-Agent / Qlib / mlflow 中的 run 标识）
    - `factor_impl_version: text`（记录本次 loop 使用的因子包版本，可选）

> 以上字段具体命名可由 AIstock 实现时微调，但含义应与本补充设计保持一致。

### 3.3 Phase 2 中的功能目标（对 AIstock）

- 在 Phase 2 完成后，AIstock 应具备：
  - **完整的 RD-Agent 成果视图**：
    - 因子库：可浏览 RD-Agent 演进因子与 Alpha158 因子，查看元数据、表现与实现指针；
    - 策略库：可浏览策略配置与模型信息；
    - 实验库：可按 loop 查看指标、曲线与反馈。
  - **与因子共享包的初步联动**：
    - 知道某个因子的“参考实现”在本地哪个模块/函数；
    - 可以在研究/调试环境中导入这些函数进行分析（无需进入模拟盘/实盘执行栈）。

> Phase 2 不要求 AIstock 已经能在自身执行栈中直接运行这些因子/策略/模型，执行迁移工作留待 Phase 3 完成。

---

## 4. 当前实现进度与验证记录（2025-12-27）

本节记录截至 2025-12-27 已在 RD-Agent 仓库中完成的与因子共享包 / 成果导出 API / AIstock 数据接入相关的实现与本地自测步骤，供后续开发与联调参考。

### 4.1 RD-Agent 侧已实现能力

- **因子共享包骨架与写入逻辑（占位版）**
  - 在仓库根目录下新增 `rd_factors_lib` 包（含 `__init__.py`, `generated.py`, `VERSION`）。
  - 在 `rdagent.utils.artifacts_writer.write_loop_artifacts` 中集成 `_sync_factor_impl_to_shared_lib`：
    - 为通过验收的演进因子在 `rd_factors_lib.generated` 中写入占位实现 stub；
    - 将原始 `factor.py` 源码片段以字符串形式写入共享包，提供参考实现源码；
    - 在对应 loop 的 `factor_meta.json` 中记录实现指针字段（如 `impl_module`, `impl_func`, `impl_version`）。

- **只读成果 API 服务**
  - 新增 FastAPI 应用：`rdagent.app.results_api_server.create_app`。
  - 通过 `rdagent.app.cli` 增加 CLI 启动入口：
    - 示例：`python -m rdagent.app.cli results_api --host 127.0.0.1 --port 9000`。
  - 提供的核心只读接口：
    - `GET /health`：健康检查；
    - `GET /catalog/factors`：返回 `RDagentDB/aistock/factor_catalog.json`；
    - `GET /catalog/strategies`：返回 `RDagentDB/aistock/strategy_catalog.json`；
    - `GET /catalog/loops`：返回 `RDagentDB/aistock/loop_catalog.json`；
    - `GET /alpha158/meta`：返回 `RDagentDB/aistock/alpha158_meta.json`；
    - `GET /factors/{name}`：在 factor_catalog 中按 `name` 返回单条因子详细信息；
    - `GET /loops/{task_run_id}/{loop_id}/artifacts`：直接查询 `registry.sqlite` 中 `artifacts` / `artifact_files`，汇总指定 loop 的工件及文件清单（包括文件路径、大小、mtime、summary_json 等）。

- **Alpha158 元数据导出**
  - 工具脚本：`tools/export_alpha158_meta.py`。
  - 从 Qlib 配置（例如 `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_dynamic.yaml`）中解析 `alpha158_config.feature`，导出 `alpha158_meta.json`：
    - 字段包括：`name`, `expression`, `source="qlib_alpha158"`, `region`, `tags=["alpha158"]`。
  - factor_catalog 支持将 Alpha158 合并为统一因子库：
    - `tools/export_aistock_factor_catalog.py` 支持 `--alpha-meta` 参数，将 `alpha158_meta.json` 中的因子合并入最终的 `factor_catalog.json`。

- **AIstock-facing 四大 Catalog 导出**
  - `tools/export_aistock_factor_catalog.py`：
    - 输入：`--registry-sqlite RDagentDB/registry.sqlite`，可选 `--alpha-meta RDagentDB/aistock/alpha158_meta.json`；
    - 输出：`RDagentDB/aistock/factor_catalog.json`；
    - 结构：`{"version": "v1", "generated_at_utc": "...", "source": "rdagent_tools", "factors": [...]}`。
  - `tools/export_aistock_strategy_catalog.py`：
    - 从 registry 中收集有结果的 loops/workspaces，扫描 YAML 模板；
    - 提取 `data_config` / `dataset_config` / `portfolio_config` / `backtest_config` / `model_config` 等配置；
    - 输出：`RDagentDB/aistock/strategy_catalog.json`。
  - `tools/export_aistock_loop_catalog.py`：
    - 汇总每个有结果的 loop 的指标、决策与关键文件路径；
    - 字段包括：`task_run_id`, `loop_id`, `workspace_id`, `strategy_id`, `factor_names`, `metrics`, `decision`, `summary_texts`, `paths`；
    - 输出：`RDagentDB/aistock/loop_catalog.json`。
  - `tools/export_aistock_model_catalog.py`：
    - 从 registry 与各 workspace 中收集可复用模型的配置与 artifacts 信息；
    - 字段包括：`task_run_id`, `loop_id`, `workspace_id`, `workspace_path`, `model_config`, `dataset_config`, `model_artifacts` 等；
    - 输出：`RDagentDB/aistock/model_catalog.json`，供 AIstock 在 Phase 3 中直接同步模型 registry 与 artifacts。

### 4.2 已执行的本地验证步骤

- **Alpha158 元数据导出与 API 验证**

  在项目根目录（WSL）执行：

  ```bash
  mkdir -p RDagentDB/aistock

  python tools/export_alpha158_meta.py \
    --conf-yaml rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_dynamic.yaml \
    --output RDagentDB/aistock/alpha158_meta.json
  ```

  启动只读 API：

  ```bash
  python -m rdagent.app.cli results_api --host 127.0.0.1 --port 9000
  ```

  在另一个终端验证：

  ```bash
  curl http://127.0.0.1:9000/alpha158/meta | head
  curl http://127.0.0.1:9000/catalog/factors | head
  ```

  预期：`/alpha158/meta` 返回刚导出的 `alpha158_meta.json` 内容，不再是 `"alpha158_meta.json not found"` 错误；`/catalog/factors` 能看到合并了 Alpha158 的因子列表。

- **四大 Catalog 导出与文件存在性验证**

  示例命令：

  ```bash
  python tools/export_aistock_factor_catalog.py \
    --registry-sqlite RDagentDB/registry.sqlite \
    --output RDagentDB/aistock/factor_catalog.json \
    --alpha-meta RDagentDB/aistock/alpha158_meta.json

  python tools/export_aistock_strategy_catalog.py \
    --registry-sqlite RDagentDB/registry.sqlite \
    --output RDagentDB/aistock/strategy_catalog.json

  python tools/export_aistock_loop_catalog.py \
    --registry-sqlite RDagentDB/registry.sqlite \
    --output RDagentDB/aistock/loop_catalog.json

  python tools/export_aistock_model_catalog.py \
    --registry-sqlite RDagentDB/registry.sqlite \
    --output RDagentDB/aistock/model_catalog.json
  ```

  预期：`RDagentDB/aistock` 目录下存在四份 JSON 文件，且通过 `results_api` 对应的 `/catalog/*` 接口可以直接返回这些内容（新增 `/catalog/models` 对应 `model_catalog.json`）。

- **因子共享包写入逻辑验证（占位版）**

  - 执行一次带有演进因子的训练 loop（通过现有 `fin_quant` 或其它任务入口），确保正常生成 `factor_meta.json`；
  - 检查：
    - `rd_factors_lib/generated.py` 中出现对应因子函数 stub 以及源代码字符串；
    - 该因子在 `factor_meta.json` 中带有正确的 `impl_module`, `impl_func`, `impl_version` 字段（当前为 Phase 2 占位实现，为 Phase 3 因子迁移做准备）。

