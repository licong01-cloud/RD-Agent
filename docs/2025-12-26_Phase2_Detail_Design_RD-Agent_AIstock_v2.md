# RD-Agent × AIstock Phase 2 详细设计（v2，2025-12-26 整合版）

> 本文件整合：
> - 原始 Phase 2 详细设计：`2025-12-23_Phase2_Detail_Design_RD-Agent_AIstock.md`
> - 补充文档：`2025-12-26_Phase2_Detail_Design_Supp_FactorPackage_and_API.md`
>
> 用于给出一份可直接参考实现的 Phase 2 设计总览。旧版文件完整保留，本文件不删除任何历史内容。

> **权威性与入口约定：**
> - 对 AIstock 后端研发团队而言，Phase 2 相关的需求与接口以本文件为**唯一入口文档**；
> - 其它文档（顶层架构、数据服务层设计、因子共享包与 API 补充等）均通过本文件进行索引与引用，不单独作为需求来源；
> - 如遇描述不一致时，以本文件的总结性约定为准，再回溯到被引用文档做对齐与更新。

---

## 1. 原始 Phase 2 详细设计

完整内容参见：

- `docs/2025-12-23_Phase2_Detail_Design_RD-Agent_AIstock.md`

原文已经定义了：

- 统一 `write_loop_artifacts` 的职责与字段；
- Phase 2 所需的 JSON/图表 artifacts（factor_meta / factor_perf / feedback / 回测图像等）；
- SQLite registry 与四大 Catalog 的基础约定（因子/策略/loop/模型）；
- AIstock 在 Phase 2 中应实现的因子库/策略库/实验库 UI 与导入逻辑。

本 v2 文件在此基础上，通过补充章节对以下内容做进一步细化：

- 因子实现源码的组织方式（因子共享包）；
- RD-Agent 只读成果 API 的接口形态与边界；
- backfill 工具在“历史成果 → AIstock 导入”路径中的角色；
- AIstock 在 Phase 2 中如何接入这些成果。

---

## 2. 因子共享包与成果导出 API（补充概要）

> 详细设计与字段列表参见：
> - `docs/2025-12-26_Phase2_Detail_Design_Supp_FactorPackage_and_API.md`

这里只摘录关键设计点，方便在阅读原文时联动理解。

### 2.1 因子共享包结构与演进流程

- RD-Agent 与 AIstock 共用一个本地因子共享包目录（例如 `F:\Dev\rd-factors-lib`）：
  - 包含 `rd_factors_lib/alpha158.py`, `momentum.py`, `volume.py`, `generated.py` 等模块；
  - 包内通过 `VERSION` 或 `__version__` 维护因子实现版本号。
- RD-Agent 因子演进 loop 在结束时：
  - 将通过验证的因子实现函数写入/更新 `generated.py`；
  - 自动 bump 版本号；
  - 在对应的 `factor_meta.json` / Catalog 中写入：
    - `impl_module`, `impl_func`, `impl_version` 字段。
- Alpha158 因子：
  - 继续通过 `export_alpha158_meta.py` 暴露完整表达式与元信息；
  - 是否在共享包中提供 Python 参考实现可按需决定，不是 Phase 2 的硬性要求。

### 2.2 RD-Agent 只读成果 API（Phase 2 范围）

- 在 RD-Agent 环境（如 WSL）启动 `rdagent-results-api`：
  - 监听本机端口，提供只读 JSON 接口；
  - 封装原有的 registry.sqlite 与四大 Catalog 视图。
- 核心接口包括：
  - `GET /catalog/factors` / `/catalog/strategies` / `/catalog/loops` / `/catalog/models`
  - `GET /factors/{name}`（含因子实现指针）
  - `GET /alpha158/meta`
  - `GET /loops/{task_run_id}/{loop_id}/artifacts`（列出该 loop 的 artifacts 清单）。
- 明确边界：
  - 所有接口只读，不返回实时信号/订单；
  - 不参与模拟盘/实盘的执行链路，仅服务于“成果导入与分析”。

### 2.3 扩展 backfill 工具：历史成果补齐与导出

- 在现有 `tools/backfill_registry_artifacts.py` 的基础上，Phase 2 要求其具备：
  - **遍历与检查历史任务**：
    - 扫描指定日志/结果目录下全部历史 task_run / loop / workspace；
    - 检查每个 workspace 是否具备 Phase 2 所需 artifacts；
  - **补齐缺失 artifacts 与 registry 记录**：
    - 自动生成缺失的 `factor_meta.json` / `factor_perf.json` / `feedback.json` / 回测图表；
    - 在 `artifacts` / `artifact_files` 中登记相应记录；
    - 对符合条件的 loop，将 `has_result` 更新为 `1`；
  - **为 AIstock 的一次性导入做准备**：
    - backfill 完成后，API 与三大导出脚本看到的是“历史 + 新任务”的完整视图。
- 使用方式（示例，命令行）：

  ```bash
  cd /mnt/f/Dev/RD-Agent-main

  # 对历史任务执行一次全量 backfill
  python tools/backfill_registry_artifacts.py \
    --log-root rdagent_log_root_path \
    --workspace-root workspace_root_path \
    --mode full
  ```

- 建议：
  - 在首次对接 AIstock 之前，先针对历史数据执行一次 `--mode full`；
  - 后续根据需要周期性执行 `--mode incremental`，补齐增量任务。

---

## 3. AIstock 在 Phase 2 中的接入与落库（补充概要）

- AIstock 通过 HTTP 调用 RD-Agent 的只读成果 API：
  - 执行“全量拉取 + upsert 导入”流程，将因子/策略/loop/Alpha158/模型元数据写入本地数据库；
  - 在此基础上：
  - 构建因子库/策略库/实验库 UI；
  - 支持对 RD-Agent 演进成果的浏览、筛选与标记；
- Phase 2 的总体目标是：**数据与成果全部打通到 AIstock**，为后续 Phase 3/4/5 提供统一数据基础。
- **权威性与入口约定：**
  - 本文件作为 Phase 2 相关的需求与接口的唯一入口文档。
- **与数据服务层详细设计的关系：**
  - 数据服务层接口与视图的形态，统一参见：
    - `2025-12-24_DataServiceLayer_Detail_Design_RD-Agent_AIstock.md`
  - 本 Phase 2 设计对 AIstock 的要求是：
    - **至少在离线/研究场景中，完成数据服务层的最小可用实现**：
      - 提供以 `DataFrame(MultiIndex(datetime, instrument))` 为基础的 tabular 因子/行情矩阵视图（如文中定义的 `get_history_window` 等接口）；
      - 保证字段命名与索引规范与 RD-Agent/qlib 的离线视图（如 `daily_pv.h5`、Alpha158 因子）保持一致；
    - 这些接口在 Phase 2 阶段主要服务于：
      - 因子共享包（`rd_factors_lib`）的本地调试与验证；
      - 在 AIstock 环境中进行基于 tabular 因子矩阵的深度模型训练与回测对齐（不进入真实执行栈）。
### 3.1 Phase 2 的总体目标（面向 AIstock）

- **数据与成果全部打通到 AIstock：**
  - RD-Agent 侧通过四大 Catalog（factor/strategy/loop/model）与只读成果 API，将因子/策略/loop/Alpha158/模型元数据与关键 artifacts 全部暴露给 AIstock；
  - AIstock 侧在 Phase 2 内完成一次性“全量导入 + 定期增量同步”链路，使后续各阶段不再依赖 RD-Agent 的内部数据结构即可独立访问这些成果。
- **为后续 Phase 3/4/5 提供统一数据基础：**
  - 无论是 Phase 3 的选股功能与策略预览，还是后续的模拟盘/实盘交易与 LiveFeedback，都必须直接基于 Phase 2 已导入的这些表与文件视图构建；
  - Phase 2 完成后，即使 RD-Agent 暂时不可用，AIstock 也应具备独立浏览与分析既有科研成果的能力。

### 3.2 与数据服务层详细设计的关系

- 数据服务层接口与视图的形态，统一参见：
  - `2025-12-24_DataServiceLayer_Detail_Design_RD-Agent_AIstock.md`
- 本 Phase 2 设计对 AIstock 的要求是：
  - **至少在离线/研究场景中，完成数据服务层的最小可用实现**：
    - 提供以 `DataFrame(MultiIndex(datetime, instrument))` 为基础的 tabular 因子/行情矩阵视图（如文中定义的 `get_history_window` 等接口）；
    - 保证字段命名与索引规范与 RD-Agent/qlib 的离线视图（如 `daily_pv.h5`、Alpha158 因子）保持一致；
  - 这些接口在 Phase 2 阶段主要服务于：
    - 因子共享包（`rd_factors_lib`）的本地调试与验证；
    - 在 AIstock 环境中进行基于 tabular 因子矩阵的深度模型训练与回测对齐（不进入真实执行栈）。
- 在线/实盘执行路径中对数据服务层的依赖（实时 snapshot、账户视图等），将在 Phase 3 详细设计中进一步固化为硬性要求，这里仅作为 Phase 2 的**预备工作与建议实现**。

---

## 4. Phase 2 实施进度跟踪（简要）

> 本小节只用于记录关键能力的实现状态，便于 RD-Agent × AIstock 双方同步进度。

- **backfill 工具扩展（RD-Agent）**  
  - 目标：支持对历史所有 task_run 做 Phase 2 artifacts 补录，并更新 `loops.has_result`。  
  - 状态：**已完成**  
    - 已在 `tools/backfill_registry_artifacts.py` 中增加 `--all-task-runs` 参数；  
    - 兼容原有 `--task-run-id` / `--log-path` / `--since-date` / `--max-loops` 行为。

- **因子共享包骨架（RD-Agent）**  
  - 目标：建立 `rd_factors_lib` 包的最小结构，提供版本号与 `generated.py` 占位模块。  
  - 状态：**已完成（骨架）**  
    - 新建：`rd_factors_lib/__init__.py`、`rd_factors_lib/generated.py`、`rd_factors_lib/VERSION`。  
    - 后续由演进/工具逻辑将通过验证的因子函数写入 `generated.py`。

- **演进因子写入共享包（RD-Agent）**  
  - 目标：在演进任务成功后，自动将因子实现写入 `rd_factors_lib.generated`，并 bump VERSION。  
  - 状态：**已完成（占位 stub 实现）**  
    - 在 `rdagent.utils.artifacts_writer.write_loop_artifacts` 中调用 `_sync_factor_impl_to_shared_lib`：  
      - 对有结果的因子 loop，从内存中的 `factor_meta_payload` 中读取因子名；  
      - 在 `rd_factors_lib.generated` 中为每个因子生成一个占位函数 `factor_xxx(df)`；  
      - 自动 bump `rd_factors_lib/VERSION` 的补丁号，并将 `impl_module` / `impl_func` / `impl_version` 回写至 `factor_meta.json`。  
    - 当前实现生成的是 **占位 stub**（抛出 `NotImplementedError`），用于 Phase 2 中：
      - 保证 AIstock 可以依赖稳定的函数入口与版本号做对账；  
      - 具体数值行为仍以 RD-Agent workspace 中的 `factor.py` 为准，完整 reference 实现留待 Phase 3 增强。

- **只读成果 API（RD-Agent）**  
  - 目标：提供 `/catalog/*`、`/factors/{name}`、`/alpha158/meta`、`/loops/*/artifacts` 等只读接口。  
  - 状态：**已完成**  
    - 实现入口：`rdagent.app.results_api_server:create_app`，通过 CLI 命令 `python -m rdagent.app.cli results_api --host 127.0.0.1 --port 9000` 启动；  
    - `/catalog/*` 与 `/alpha158/meta` 直接读取 `<registry_dir>/aistock/` 目录下由导出脚本生成的 JSON 文件；  
    - `/factors/{name}` 在 factor_catalog 中按 `name` 返回单条因子记录；  
    - `/loops/{task_run_id}/{loop_id}/artifacts` 直接查询 registry.sqlite 中 `artifacts` / `artifact_files` 表，返回指定 loop 的 artifacts 视图。  

- **成果同步与落库（AIstock）**  
  - 目标：实现调用 RD-Agent 只读 API 的客户端，将因子/策略/loop/Alpha158/训练元数据导入本地 DB。  
  - 状态：**待实现（AIstock 侧）**  
    - 参考本文件第 3 章与顶层/Registry 设计中的建表建议与字段合同。

