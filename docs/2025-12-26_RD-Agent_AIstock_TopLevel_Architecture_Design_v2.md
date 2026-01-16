# RD-Agent × AIstock 顶层架构设计（v2，2025-12-26 整合版）

> 本文件作为《2025-12-23_RD-Agent_AIstock_TopLevel_Architecture_Design.md》的整合升级版，
> 将 2025-12-26 新增的“因子共享包 + 只读成果 API + Phase2/Phase3 执行分工补充方案”一并纳入。
>
> 为避免直接修改原始文档导致信息丢失：
> - 原始顶层设计（v1）完整保留：`2025-12-23_RD-Agent_AIstock_TopLevel_Architecture_Design.md`（仅作历史溯源）
>
> **当前生效的顶层架构设计以本 v2 文档为准。关于 Registry/因子共享包/只读成果 API 的实现细节与字段合同，请参考 Phase2 最终版设计文档：**
> - `2025-12-29_Phase2_Detail_Design_RD-Agent_AIstock_Final.md`（含 Registry 附录 A/B）


---

## 1. 顶层设计主文（v1）

完整内容参见历史版本：

- `docs/2025-12-23_RD-Agent_AIstock_TopLevel_Architecture_Design.md`（仅作溯源使用）

其定义了：

- RD-Agent × AIstock 的总体架构分层；
- 分阶段实施路线（Phase 1–5）的目标与验收标准；
- Registry 设计与三大 Catalog 的原始协议；
- Phase 2/3/4/5 的初始角色划分与能力演进。

本 v2 文件**不重拷**该内容，而是以补充说明的方式，对其中与“因子/策略/模型成果交付”和“执行权边界”相关的部分进行增强约束。

---

## 2. 2025-12-26 顶层补充：因子共享包与成果同步 API（概要）

> 详细设计参见：
> - `docs/2025-12-26_RD-Agent_AIstock_TopLevel_Supp_FactorPackage_and_API.md`
>
> 本节仅做概要性摘录，帮助读者在一份文档中掌握关键决策点。

### 2.1 因子实现资产化：因子共享包

- RD-Agent 与 AIstock 在本地共享一个独立的 Python 包目录（例如 `F:\Dev\rd-factors-lib`），内含：
  - `rd_factors_lib/alpha158.py` / `momentum.py` / `volume.py` / `generated.py` 等模块；
  - `VERSION` 或 `__version__` 字段用于标记当前实现版本（如 `"1.0.7"`）。
- 每个因子在元数据中记录：
  - `impl_module`: 如 `"rd_factors_lib.generated"`
  - `impl_func`: 如 `"factor_xxx"`
  - `impl_version`: 如 `"1.0.8"`
- RD-Agent 因子演进流程在 loop 收尾时，自动将通过验证的因子实现写入共享包并 bump 版本；
- AIstock 通过本地安装该包并读取版本号，对齐因子实现，用于后续迁移与测试（执行迁移在 Phase 3 完成）。

### 2.2 RD-Agent 只读成果 API

- 在 RD-Agent 环境中部署一个只读服务（示意名：`rdagent-results-api`），监听本机端口；
- 提供只读接口：
  - `GET /catalog/factors` / `/catalog/strategies` / `/catalog/loops`
  - `GET /factors/{name}`（含实现指针）
  - `GET /alpha158/meta`
  - `GET /loops/{id}/artifacts` 等
- 该 API **只用于研发成果同步**，不返回实时信号/订单，**不参与模拟盘/实盘交易链路**。

### 2.3 Phase 2 vs Phase 3：执行责任划分

- **Phase 2：**
  - RD-Agent：保证 artifacts 完整，维护 registry，输出三大 Catalog + 因子实现指针 + 只读 API；
  - AIstock：拉取并落库 RD-Agent 的因子/策略/loop/Alpha158/训练结果等成果，用于研究与决策；
  - **不**在 AIstock 执行栈中直接运行 RD-Agent 的模型/因子/策略。

- **Phase 3：**
  - RD-Agent：继续作为“研发工厂”，按 ExperimentRequest 执行实验并输出成果；
  - AIstock：基于因子共享包 + 元数据，在自身数据服务和执行引擎中实现等价的因子/策略/模型，并用于模拟盘/实盘；
  - 对候选组合在同一历史数据集上做对账测试，确保执行行为与 RD-Agent 回测结果在可接受误差内一致。

> 更详细的字段、接口与工作流描述，请参阅补充文档 `2025-12-26_RD-Agent_AIstock_TopLevel_Supp_FactorPackage_and_API.md`。

---

## 3. 历史成果迁移到 AIstock 的标准路径（backfill + 导出/API）

在顶层设计层面，明确以下事实：

- RD-Agent 通过扩展的 backfill 工具，对历史与新任务统一补齐 Phase 2 所需 artifacts，并更新 registry（含 `has_result` 标记）；
- 在此基础上：
  - 通过三大导出脚本（factor/strategy/loop catalog），可一次性导出“历史 + 新任务”的全量视图；
  - 或通过只读成果 API 分页拉取同一视图；
- AIstock 侧以 **全量 upsert** 的方式将这些数据导入本地数据库，作为后续 Phase 3 执行迁移与重训的唯一数据源。

上述路径在 Phase 2/3 中保持稳定，后续阶段（多策略、多窗口、工作流与多 Agent）在此基础上增量演进，不改变已有协议的含义。

---

## 4. Phase 2 / Phase 3 RD-Agent × AIstock 分工与数据流总览

本节从更宏观的角度，对比说明 Phase 2 与 Phase 3 中 RD-Agent 与 AIstock 的开发分工和数据传输约定。

### 4.1 Phase 2 分工

- **RD-Agent 侧负责：**
  - 产生并管理所有研发成果：
    - loops / workspaces / registry.sqlite；
    - Phase 2 artifacts（factor_meta / factor_perf / feedback / 回测曲线等）；
    - 三大 Catalog（factor / strategy / loop）；
  - 维护因子共享包（rd_factors_lib）：
    - 将演进通过验证的因子实现写入共享包模块；
    - 维护 VERSION / __version__；
    - 在因子元数据中记录 impl_module / impl_func / impl_version；
  - 提供只读成果 API：
    - /catalog/* 视图；
    - /factors/{name} /alpha158/meta；
    - /loops/*/artifacts 等；
  - backfill 工具：
    - 对历史任务补齐 Phase 2 所需 artifacts 和 registry 记录；
    - 提供 --task-run-id / --all-task-runs / --since-date 等命令行入口。

- **AIstock 侧负责：**
  - 实现 RD-Agent API 的调用与数据同步：
    - 周期性或按需拉取 catalog / factor_meta / loop / alpha158；
  - 将数据导入本地数据库与文件系统：
    - 建立因子库/策略库/实验库/Alpha158 等表结构；
  - 提供研究与管理 UI：
    - 浏览和筛选 RD-Agent 演进出来的因子/策略/实验结果；
    - 查看因子实现指针和训练元数据等。

- **Phase 2 不做的事：**
  - RD-Agent 不参与任何模拟盘/实盘交易执行，只是“结果生产者”；
  - AIstock 只在研究域使用这些成果，**不在自身执行栈中直接运行 RD-Agent 的模型/策略/因子**。

### 4.2 Phase 3 分工

- **RD-Agent 侧继续负责：**
  - 因子/策略/模型的演进、回测与评估；
  - 维护 registry / artifacts / Catalog / 因子共享包；
  - 提供只读成果 API；
  - 接收来自 AIstock 的 ExperimentRequest / FactorSetConfig / StrategyConfig 等研发域请求。

- **AIstock 侧新增负责：**
  - 在自身代码库中实现：
    - 因子执行引擎（FactorEngine），基于 AIstock Data Service；
    - 策略执行引擎（StrategyEngine），实现与 RD-Agent 相同策略逻辑；
    - 模型训练与推理组件，基于从 RD-Agent 导入的 model_config / 数据窗口等；
  - 利用因子共享包作为参考实现源：
    - 为需要上线的因子编写等价实现；
    - 通过标准测试集完成数值对齐；
  - 基于 RD-Agent 回测结果做对账：
    - 在同一历史数据集上验证策略/模型行为是否与 RD-Agent 一致；
  - 将通过验收的组合接入模拟盘与实盘执行链路。

- **执行域的硬边界：**
  - 模拟盘/实盘的执行：
    - **只依赖 AIstock 的数据服务与执行引擎**；
    - 不直接调用 RD-Agent 的 API 或运行 RD-Agent 进程；
  - RD-Agent 在 Phase 3 中仍旧仅处于“研发域”，通过 artifacts 与 API 提供信息，完全不介入交易系统的运行时路径。

以上分工与数据流约定在 Phase 2/3 中保持稳定，为后续多策略、多窗口与工作流/多 Agent 的进一步演进提供清晰边界。

---

## 5. 硬性要求（REQ Checklist，按 2025-12-30 项目规范对齐）

> 顶层架构文档作为所有 Phase 详细设计的上位约束，本节以 REQ ID 形式固定关键原则。
> 详细描述见：`docs/2025-12-30_Project_Development_Spec_RD-Agent_AIstock.md` 与
> `docs/2025-12-30_Phase1-3_Design_Update_RD-Agent_AIstock.md`。

- **REQ-GOV-PX-001：不得精简原则**  
  任一功能实现不得在未更新设计与 REQ、未在 PR 中显式声明的情况下擅自精简或弱化。

- **REQ-FACTOR-P2-XXX & REQ-FACTOR-P3-XXX：因子资产化与复用**  
  RD-Agent 必须通过因子共享包与因子 Catalog 将演进因子的实现与元数据资产化；AIstock 必须在
  Phase 3 中基于这些资产实现等价因子执行。

- **REQ-API-P2-001：只读成果 API 契约**  
  RD-Agent 只读成果 API 是 AIstock 同步成果的唯一来源接口，其字段与行为必须与 Phase 2 最终版
  设计保持兼容，未经评审不得做破坏性修改或精简。

- **REQ-DATASVC-P3-001：执行数据流硬边界**  
  顶层架构明确：模拟盘/实盘所有执行数据流一律通过 AIstock 数据服务层与执行引擎完成，RD-Agent
  不得直接参与执行时路径，只作为研发域系统提供 artifacts 与配置.
