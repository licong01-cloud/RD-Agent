# RD-Agent × AIstock 功能缺口补齐详细设计与执行步骤

> 文件：2025-12-30_Gap_Closure_Design_and_Execution_Plan.md  
> 目的：在项目规范与 Phase1–3 设计更新的基础上，列出当前实现与 REQ 之间的缺口，并给出补齐方案与执行步骤。

---

## 1. 总览

### 1.1 范围

- RD-Agent 仓库内：
  - 因子共享包与元数据；
  - registry / artifacts / 各类 catalog exporter；
  - results-api。
- AIstock 仓库内（仅列出需要对接/新增的能力，不给出实现细节）：
  - 成果同步与落库；
  - DataService 与 qlib runtime 集成；
  - 基于 loop 的一键重放能力。

### 1.2 当前关键缺口（摘要）

- 自研因子缺少实现指针与表达式（REQ-FACTOR-P2-001/002 未满足）；
- loop 状态与指标未回填（REQ-LOOP-P2-001/002 未满足）；
- 模型/策略/loop catalog 字段不够完备（部分 REQ-MODEL/STRATEGY 未满足）；
- results-api 返回结构未完全对齐 REQ（例如未暴露因子实现信息）；
- AIstock 侧 DataService 与 qlib runtime 集成尚未落地（REQ-DATASVC-P3-001, REQ-MODEL-P3-010 未满足）。

---

## 2. RD-Agent 侧缺口与补齐方案

### 2.1 因子实现指针与表达式（REQ-FACTOR-P2-001/002/010/011）

#### 2.1.1 现状

- `scan_workspace_factors.py` 基于 `combined_factors_df.parquet` 列名生成 `factor_meta.json`；
- `rdagent_generated` 因子在 AIstock 侧只看到 `name` 与 `source`，`expression` 与 `impl_*` 为空。

#### 2.1.2 目标

- 所有 `rdagent_generated` 因子在 `factor_meta.json` / `factor_catalog.json` 中：
  - 必须包含 `impl_module`, `impl_func`, `impl_version`；
  - 至少有 `expression` 或 `formula_hint`；
- 新验收因子自动写入共享包与 meta；
- 历史因子通过迁移脚本尽可能补全，实现可追踪。

#### 2.1.3 设计与执行步骤

1. **梳理因子共享包写入流程**
   - 位置：`rdagent.utils.artifacts_writer.write_loop_artifacts`（示例）
   - 行为：
     - 为通过验收的因子，将函数写入 `rd_factors_lib.generated`；
     - 更新 `rd_factors_lib/VERSION`；
     - 在当前 loop 的 `factor_meta.json` 中写入 `impl_module`, `impl_func`, `impl_version`；
   - 行动：
     - [ ] 检查现有逻辑是否为“占位版”，如是，需要升级为真正可用实现；
     - [ ] 确认所有演进因子入口（不同场景/任务）都调用了该逻辑。

2. **新增历史因子迁移脚本**（例如：`tools/migrate_factor_impl_meta.py`）
   - 输入：
     - `RDagentDB/registry.sqlite`（用于定位所有有因子的 workspace）；
     - 各 workspace 内的 `factor_meta.json` 与 `factor.py` 等源文件；
     - `rd_factors_lib/generated.py`。
   - 行为：
     - 遍历所有 `source = "rdagent_generated"` 的因子：
       - 若能在 `rd_factors_lib.generated` 中找到同名函数：
         - 补写 `impl_module/impl_func/impl_version`；
       - 若能在 workspace 原始 `factor.py` 中找到实现：
         - 选定统一的导出策略（如写入 `generated.py` 并分配函数名）；
       - 不能找到实现的：
         - 在 meta 中标记 `legacy_no_impl = true`；
     - 尝试为每个因子补写 `expression` 或 `formula_hint`：
       - 对部分标准模板类因子，可以从模板或命名约定推导表达式；
       - 对复杂/自定义因子，至少写入自然语言公式说明。
   - 输出：
     - 更新后的 `factor_meta.json`；
     - 更新后的 `rd_factors_lib/generated.py` 与 `VERSION`。

3. **更新 factor_catalog 导出逻辑**
   - 文件：`tools/export_aistock_factor_catalog.py`
   - 行为：
     - 合并 workspace `factor_meta.json` 与 Alpha meta；
     - 将 `impl_module`, `impl_func`, `impl_version`, `expression`, `formula_hint`, `legacy_no_impl` 等字段一并写入 `factor_catalog.json`；
   - 同步更新 `results-api`：
     - `GET /catalog/factors` / `GET /factors/{name}` 必须返回上述字段。

4. **增加契约测试（tests/contracts/test_factors_contracts.py）**
   - 检查：
     - 所有 `rdagent_generated` 因子是否有 `impl_*`（或明确标记 `legacy_no_impl`）；
     - 是否有 `expression` 或 `formula_hint`。

---

### 2.2 loop 状态与指标（REQ-LOOP-P2-001/002）

#### 2.2.1 现状

- `bootstrap_registry_*` 将大量 loop 插入为 `status = "unknown"`，metrics 为空；
- `backfill_registry_artifacts.py` 目前主要补 artifacts，不回填 `loops` 表中的指标与状态。

#### 2.2.2 目标

- 所有 `loops.has_result = 1` 的记录：
  - `status` 不为 `"unknown"`；
  - 关键指标字段至少一项非空（如 `ic_mean` 或 `ann_return`）；
- `loop_catalog.json` 中可用于筛选“可上线/可验证”的实验。

#### 2.2.3 设计与执行步骤

1. **新增 `tools/backfill_loop_metrics_from_summaries.py`**
   - 输入：
     - `RDagentDB/registry.sqlite`；
     - 各 workspace 内的 `experiment_summary.json`、`qlib_res.csv`、`ret.pkl` 等文件。
   - 行为：
     - 遍历 `loops` 表中 `has_result = 1` 的记录；
     - 对应 workspace：
       - 解析 summary/结果文件，提取：`ic_mean`, `rank_ic_mean`, `ann_return`, `mdd`, `turnover`, 其他关键指标；
       - 根据成功/失败标记更新 `status` 字段；
     - 以事务方式批量更新 `loops` 表。

2. **增强 `export_aistock_loop_catalog.py`**
   - 在 loop catalog 中输出：
     - `status`；
     - 各种指标（IC/收益/回撤/换手等）；
     - 关联的 `factor_names`、`model_id`、`strategy_id`、`workspace_path`；
     - 关键 artifacts 路径（回测曲线、因子表现等）。

3. **新增契约测试（tests/contracts/test_loops_contracts.py）**
   - 校验：
     - `loops.has_result = 1` 的记录中，`status != "unknown"`；
     - 至少一个指标字段非空。

---

### 2.3 模型/策略/loop catalog 字段完备性（REQ-MODEL-P2-001, REQ-STRATEGY-P2-001 等）

#### 2.3.1 现状

- exporter 已存在，但字段不一定完备、与 REQ 不完全对齐；
- 模型/策略/loop 之间的关联关系在部分历史数据中不完整。

#### 2.3.2 目标

- `strategy_catalog.json` / `model_catalog.json` / `loop_catalog.json` 三者之间能够完整描述：
  - 某个 loop 使用了哪些因子、哪个模型、哪套策略配置；
  - 足够信息让 AIstock 一键重放该 loop（结合 DataService + qlib runtime）。

#### 2.3.3 设计与执行步骤

1. **补齐模型 catalog 字段**
   - 对 `export_aistock_model_catalog.py`：
     - 确保每条记录包含：`model_type`, `model_conf`, `dataset_conf`, `feature_names`, `window`, `freq`；
     - 可从 `experiment_summary`, 模型配置 YAML、Qlib 配置等源文件解析获得。

2. **补齐策略 catalog 字段**
   - 对 `export_aistock_strategy_catalog.py`：
     - 对齐 Phase2 补充设计中的字段要求：
       - `data_config`, `dataset_config`, `portfolio_config`, `backtest_config`, `model_config`；
     - 为每条策略记录引入主键（如 `strategy_id`），并与 loop 建立映射。

3. **增强 loop catalog**
   - 对 `export_aistock_loop_catalog.py`：
     - 为每个 loop 记录其 `strategy_id` 与 `model_id`；
     - 输出 `factor_names`，并可在后续版本中链接到 `factor_catalog` 中的实现信息。

4. **新增契约测试（tests/contracts/test_models_contracts.py 等）**
   - 检查 JSON 中是否存在所有 REQ 要求的字段；
   - 检查关键关联字段不为空。

---

### 2.4 results-api 字段与契约（REQ-API-P2-001 等）

#### 2.4.1 现状

- `results-api` 已实现 `/catalog/*` 与 `/loops/{...}/artifacts` 等端点；
- 但返回的 JSON 结构尚未严格对齐 REQ（例如因子实现指针/表达式字段缺失）。

#### 2.4.2 目标

- API 返回内容与对应 catalog JSON 完全一致或为其视图子集；
- 对 AIstock 来说，调用 API 即可获得所有设计要求的字段，无需额外查询。

#### 2.4.3 设计与执行步骤

1. **同步 API 与 catalog 结构**
   - 调整 FastAPI 路由中返回的数据模型，使其直接基于 catalog JSON 或共享的 Pydantic 模型；
   - 确保：
     - `GET /catalog/factors` / `/factors/{name}` 暴露因子实现与表达式字段；
     - `/catalog/models` / `/catalog/strategies` / `/catalog/loops` 字段齐全。

2. **新增 API 契约测试（tests/contracts/test_results_api_contracts.py）**
   - 使用 FastAPI TestClient 或集成测试：
     - 对关键端点发起请求；
     - 校验返回 JSON 中字段和类型满足 REQ。

---

## 3. AIstock 侧缺口与补齐方向（概要）

> 具体实现由 AIstock 仓库负责，此处仅列出与 RD-Agent 设计耦合的关键点。

### 3.1 成果同步与落库（REQ-AISTOCK-P2-001/002/003）

- 实现“RD-Agent 同步任务”模块：
  - 定时调用 `results-api`；
  - 将 catalog 与因子元数据 UPSET 到本地 DB；
  - 存储 `impl_module/impl_func/impl_version` 等字段。

### 3.2 DataService 与 qlib runtime 集成（REQ-DATASVC-P2-001, REQ-DATASVC-P3-001, REQ-MODEL-P3-010）

- 实现数据服务层接口：`get_history_window` / `get_realtime_snapshot` 等；
- 集成固定版本 qlib runtime：
  - 能根据 `model_conf` / `dataset_conf` 加载 RD-Agent 导出的模型；
  - 使用 DataService 提供的 DataFrame 作为输入；
- 确保因子共享包 `rd_factors_lib` 在 AIstock 执行环境中可 import。

### 3.3 基于 loop 的一键重放（REQ-LOOP-P3-001）

- 在 AIstock 中实现：
  - 选定 loop → 从本地 DB 中取出该 loop 的因子/模型/策略信息；
  - 利用 DataService + qlib runtime + 因子共享包，重放该策略，至少支持模拟盘级别验证。

---

## 4. 执行节奏建议

1. **第一阶段（RD-Agent 优先）**
   - 完成：
     - 因子实现指针与表达式补齐；
     - loop 状态与指标回填；
     - catalog 字段补齐；
     - results-api 对齐；
     - 契约测试落地。

2. **第二阶段（AIstock 对接）**
   - 在 AIstock 仓库中：
     - 实现成果同步模块与本地落库；
     - 扩展 DB Schema 以容纳 impl/模型/loop 信息；
     - 部分实现 DataService 接口。

3. **第三阶段（在线复用与一键重放）**
   - 集成 qlib runtime 与 DataService；
   - 实现基于 loop 的一键重放逻辑；
   - 完成模拟盘/实盘接入前的端到端联调与自动化测试。

---

## 5. 验收与关闭

- 每一阶段结束时：
  - 对照 `2025-12-30_Project_Development_Spec_RD-Agent_AIstock.md` 与 `2025-12-30_Phase1-3_Design_Update_RD-Agent_AIstock.md` 中的 REQ 清单；
  - 确认：
    - 所有相关 REQ 已有实现映射；
    - 所有契约测试通过；
    - 无未解释的“临时精简”。

- 本文档将随着实现推进持续更新，记录：
  - 已完成的补齐项；
  - 新发现的缺口；
  - 变更后的执行步骤。
