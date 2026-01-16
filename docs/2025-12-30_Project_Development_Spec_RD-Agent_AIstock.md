# RD-Agent × AIstock 项目开发规范（不得精简版）

> 文件：2025-12-30_Project_Development_Spec_RD-Agent_AIstock.md  
> 适用范围：RD-Agent 仓库、AIstock 仓库，以及两侧联合开发  
> 适用阶段：Phase 1–3 及后续所有版本

---

## 1. 总则

- **1.1 不得精简原则（GOV-001）**
  - 所有实现必须以最新的顶层设计 / Phase 设计 / 数据服务层设计为**硬约束**。  
  - 不得在未更新设计文档和评审通过的前提下，以任何形式进行“功能精简”“临时实现”“占位实现”并投入主干。  
  - 如确需阶段性折衷，必须：
    - 先更新设计文档，新增/调整对应 REQ；
    - 在 PR 中显式声明偏离点、原因和补齐计划；
    - 通过评审后方可合入主干。

- **1.2 设计-实现-测试三位一体（GOV-002）**
  - 所有功能需求必须：
    - 在设计文档中以 **REQ ID** 形式记录；
    - 在实现中对照 REQ ID 进行注释或关联；
    - 在测试（尤其是契约测试）中对照 REQ ID 覆盖验证。

- **1.3 版本与文档同步（GOV-003）**
  - 任一影响接口、数据结构或行为的变更，必须同步更新：
    - 对应设计文档（顶层 / Phase / 数据服务层）；
    - REQ Checklist；
    - 契约测试用例。

---

## 2. REQ ID 规范

- **2.1 命名规则（GOV-010）**
  - 统一格式：`REQ-<域>-<阶段>-<序号>`，例如：`REQ-FACTOR-P2-001`。
  - 域（示例）：
    - `FACTOR`：因子与因子共享包
    - `LOOP`：实验 loop 与 registry
    - `MODEL`：模型
    - `STRATEGY`：策略
    - `API`：只读成果 API / DataService API
    - `DATASVC`：数据服务层
    - `GOV`：项目治理相关
  - 阶段：`P1` / `P2` / `P3` 等，对应顶层架构分阶段。

- **2.2 REQ 内容要求（GOV-011）**
  - 每条 REQ 至少包含：
    - 标题：一句话说明硬性要求；
    - 描述：补充细节、字段要求、行为约束；
    - 适用范围：RD-Agent / AIstock / 双方；
    - 关联接口 / 文件：例如 `factor_catalog.json`，`GET /catalog/factors`；
    - 验证方式：指向对应的契约测试或手工验收步骤。

- **2.3 REQ 生命周期（GOV-012）**
  - 新增 REQ：必须通过架构/业务评审；
  - 修改 REQ：需说明影响范围及兼容性策略；
  - 废弃 REQ：必须提供替代方案，并在代码与测试中清理旧逻辑。

---

## 3. 契约测试（CI Contract Tests）规范

- **3.1 范围（GOV-020）**
  - 契约测试覆盖以下对象：
    - JSON 产物：`factor_catalog.json` / `strategy_catalog.json` / `model_catalog.json` / `loop_catalog.json` / Alpha meta 等；
    - SQLite Registry：`registry.sqlite` 中的关键表与字段；
    - HTTP API：`results-api`、未来的 DataService API；
    - 因子共享包：`rd_factors_lib` 结构与版本号。

- **3.2 目录结构（GOV-021）**
  - 建议目录：`tests/contracts/`，按域拆分：
    - `test_factors_contracts.py`
    - `test_loops_contracts.py`
    - `test_models_contracts.py`
    - `test_results_api_contracts.py`
    - `test_datasvc_contracts.py`（未来扩展）

- **3.3 执行与门禁（GOV-022）**
  - CI pipeline 必须包含：

    ```bash
    pytest tests/contracts
    ```

  - 任一契约测试失败，视为违反 REQ 契约，不得合入主干。
  - 如需临时跳过某个契约测试：
    - 必须在对应测试上方写明 REQ ID 和跳过原因；
    - 必须在 PR 描述中单独说明；
    - 必须有明确补齐时间点。

---

## 4. PR 模板与评审 Checklist 规范

- **4.1 PR 模板（GOV-030）**

  - 仓库根目录必须提供统一 PR 模板（如 `/.github/pull_request_template.md`），至少包含：
    - 变更说明与影响范围；
    - 关联设计文档列表（顶层 / Phase / 数据服务层）；
    - 关联 REQ ID 列表；
    - “是否存在与设计不一致或精简的实现？”勾选项：
      - 若是，必须写明偏离内容、原因、风险、补齐计划；
    - 测试情况：单元测试 / 集成测试 / 契约测试。

- **4.2 评审 Checklist（GOV-031）**

  - 评审人必须按以下清单进行审核：
    - [ ] 变更是否列出所有相关 REQ ID？
    - [ ] 是否存在未在 PR 中声明的“功能精简”或“临时实现”？
    - [ ] 契约测试是否全部通过？如跳过，是否有充分理由？
    - [ ] 是否删除或弱化了任何设计中已定义的字段/行为？若有，设计文档和 REQ 是否同步更新？

---

## 5. RD-Agent 侧关键 REQ 范例（摘录）

> 详细 REQ 清单将在各设计文档（Phase1–3、数据服务层）中维护，以下仅列出与当前成果导出链路相关的核心条目示例。

- **REQ-FACTOR-P2-001：因子实现指针**
  - 所有 `source = "rdagent_generated"` 的因子，在 `factor_meta.json` 与 `factor_catalog.json` 中必须包含：
    - `impl_module`, `impl_func`, `impl_version`；
  - 以支持 AIstock 侧通过 Python import 直接复用因子实现。

- **REQ-FACTOR-P2-002：因子表达式 / 公式提示**
  - RD-Agent 希望在 AIstock 上复用的所有自研因子，必须在 meta 中提供：
    - `expression`（若可抽象为 Qlib 表达式）；或
    - `formula_hint`（自然语言公式说明）。

- **REQ-LOOP-P2-001：loop 状态**
  - `loops.has_result = 1` 的行，`status` 字段不得为 `"unknown"`；
  - 必须明确标记为 `"succeeded"` / `"failed"` / `"partial"` 等。

- **REQ-LOOP-P2-002：loop 指标**
  - `loops.has_result = 1` 的行，至少一个指标非空：
    - `ic_mean` 或 `ann_return` 等；

- **REQ-MODEL-P2-001：模型元数据**
  - `model_catalog.json` 中每个可复用模型必须包含：
    - `model_type`, `model_conf`, `dataset_conf`, `feature_names`, `window`, `freq` 等关键字段。

- **REQ-API-P2-001：因子 Catalog API 字段齐全**
  - `GET /catalog/factors`、`GET /factors/{name}` 返回内容，必须包含：
    - `name`, `source`, `expression`/`formula_hint`（如有）、`impl_module`/`impl_func`/`impl_version`（如有）。

---

## 6. AIstock 侧关键 REQ 范例（摘录）

- **REQ-DATASVC-P2-001：统一数据形态**
  - DataService 对上暴露的历史/实时数据接口，必须统一为：
    - `pd.DataFrame`，MultiIndex `(datetime, instrument)`；
    - columns：标准行情字段 + 因子字段。

- **REQ-DATASVC-P3-001：仅通过 DataService 获取数据**
  - 所有进入模拟盘/实盘执行栈的策略/模型，必须通过 DataService 接口获取行情与因子数据；
  - 禁止直接访问 HDF5/bin/DB 等底层实现。

- **REQ-MODEL-P3-010：qlib runtime 集成**
  - AIstock 后端必须集成固定版本的 qlib runtime；
  - 能根据 `model_conf` / `dataset_conf` / 特征列表，在自身执行栈中加载并运行 RD-Agent 导出的模型。

---

## 7. 实施与验收

- **7.1 渐进落地**
  - 现有代码需对照本规范和更新后的 Phase1–3 设计逐步补齐：
    - 优先完成因子实现指针与 loop 状态/指标回填；
    - 随后完善 model/strategy/loop catalog 字段；
    - 最后补全 AIstock 侧 DataService + qlib runtime 集成。

- **7.2 验收标准**
  - 对每个阶段/里程碑：
    - 设计文档中的 REQ Checklist 无未实现项；
    - 契约测试全部通过；
    - PR 记录中无未解决的“临时精简”条目。

- **7.3 变更流程**
  - 任何对本规范的修改，必须：
    - 开单 / 开 PR；
    - 经项目 Owner 与架构 Owner 共同评审通过；
    - 同步更新相关设计文档与测试用例。
