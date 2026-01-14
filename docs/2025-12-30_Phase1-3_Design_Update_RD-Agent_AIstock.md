# RD-Agent × AIstock Phase1–3 设计更新（对齐项目开发规范，已废弃，仅作溯源）

> 文件：2025-12-30_Phase1-3_Design_Update_RD-Agent_AIstock.md  
> 状态：本文件中的 REQ 内容已全部合并进以下设计文档的 REQ Checklist 中，仅保留作历史溯源，**不再作为需求或接口的设计入口**：  
> - 顶层架构设计：`2025-12-26_RD-Agent_AIstock_TopLevel_Architecture_Design_v2.md`  
> - Phase 1 详细设计：`2025-12-23_Phase1_Detail_Design_RD-Agent_AIstock.md`  
> - Phase 2 详细设计最终版：`2025-12-29_Phase2_Detail_Design_RD-Agent_AIstock_Final.md`  
> - Phase 3 详细设计最终版：`2025-12-29_Phase3_Detail_Design_RD-Agent_AIstock_Final.md`  
> - 数据服务层详细设计：`2025-12-24_DataServiceLayer_Detail_Design_RD-Agent_AIstock.md`  
> 任何新的实现或验收，都应以上述文档及其中的 REQ Checklist 为准，而非本文件。  
> 下面保留原始内容仅供追溯参考。

---

## 1. Phase 1：离线因子/模型实验链路

### 1.1 目标与范围

- RD-Agent 在本地/研究环境中完成因子/模型的离线训练与回测；
- 使用 AIstock 提供的离线数据视图（HDF5/Qlib bin 等）。

### 1.2 关键 REQ

- **REQ-FACTOR-P1-001：统一离线数据视图**
  - RD-Agent 在 Phase 1 中使用的行情与因子数据，必须来自 AIstock 侧标准导出的：
    - `daily_pv.h5` / `moneyflow.h5` / `qlib_bin` 等；
  - 禁止直接访问非规范化的临时数据源。

- **REQ-MODEL-P1-001：模型接口兼容性**
  - 拟在未来 AIstock 复用的模型，必须遵守 qlib Model 接口：
    - `fit`, `predict`, `save`, `load` 等；
  - 并在保存时写明 `model_conf` 与 `dataset_conf`。

- **REQ-LOOP-P1-001：实验元数据记录**
  - 每个实验 loop 必须生成包含主要训练/验证信息的 JSON：
    - 训练起止日期、验证集、特征列表、目标字段、评估指标等；
  - 为后续 Phase 2/3 的 registry 与 catalog 打基础。

### 1.3 AIstock 侧职责（Phase 1）

- 提供离线数据导出脚本：保证字段/索引规范与 RD-Agent 设计文档一致；
- 不要求在 Phase 1 内实现在线 DataService 或 qlib runtime 集成。

---

## 2. Phase 2：成果导出与只读 API

### 2.1 RD-Agent 侧 REQ

- **REQ-FACTOR-P2-001：因子实现指针（详见项目规范）**
- **REQ-FACTOR-P2-002：因子表达式 / 公式提示**
- **REQ-FACTOR-P2-010：因子共享包结构**
  - 必须存在 `rd_factors_lib` 包，并包含：`__init__.py`, `generated.py`, `VERSION`；
  - 新验收的因子函数写入 `generated.py` 并更新版本号。

- **REQ-FACTOR-P2-011：loop → 因子共享包写入**
  - RD-Agent 因子演进 loop 通过验收时，自动完成：
    - 因子函数写入/更新 `rd_factors_lib.generated`；
    - 更新 `VERSION`；
    - 更新当前 loop 的 `factor_meta.json`：`impl_module`, `impl_func`, `impl_version`。

- **REQ-LOOP-P2-001/002：loop 状态与指标**
  - `loops.has_result = 1` 的记录：
    - `status` 不得为 `"unknown"`；
    - 至少一个指标非空（`ic_mean`/`ann_return` 等）。

- **REQ-MODEL-P2-001：模型 catalog 字段**
  - `model_catalog.json` 每条记录必须包含：
    - `task_run_id`, `loop_id`, `workspace_id`, `workspace_path`；
    - `model_type`, `model_conf`, `dataset_conf`；
    - `feature_names`, `window`, `freq`；
    - 模型文件相关 artifacts 的引用。

- **REQ-STRATEGY-P2-001：策略 catalog 字段**
  - `strategy_catalog.json` 每条记录必须包含：
    - `strategy_id`, `step_name`, `action`；
    - `data_config`, `dataset_config`, `portfolio_config`, `backtest_config`, `model_config`；
    - 与特定 loop/模型的关联信息。

- **REQ-API-P2-001：只读成果 API 字段齐全**
  - `results-api` 中：
    - `GET /catalog/factors` / `GET /factors/{name}`：必须暴露因子元数据中的实现指针与表达式相关字段；
    - `GET /catalog/models` / `/catalog/strategies` / `/catalog/loops`：字段与对应 catalog JSON 保持一致，不得做精简。

### 2.2 AIstock 侧 REQ

- **REQ-AISTOCK-P2-001：成果同步任务**
  - AIstock 后端必须有“RD-Agent 同步任务”：
    - 定期调用 `results-api` 的 `/catalog/*` 与 `/factors/*` 等接口；
    - 将增量/全量结果写入本地数据库（upsert）。

- **REQ-AISTOCK-P2-002：因子共享包版本对齐**
  - 对于带有 `impl_version` 的因子：
    - AIstock 必须记录当前使用的 `rd_factors_lib` 版本；
    - 确保与 RD-Agent 侧版本一致或兼容。

- **REQ-AISTOCK-P2-003：本地 Schema 扩展**
  - 本地因子/策略/实验表结构必须包含：
    - 对应的 `impl_module`, `impl_func`, `impl_version` 等字段。

---

## 3. Phase 3：在线数据服务与模型复用

### 3.1 RD-Agent 侧 REQ

- **REQ-FACTOR-P3-001：因子函数标准形态**
  - RD-Agent 演进因子需提供统一形态函数：

    ```python
    def factor_xxx(df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        ...
    ```

  - 支持离线（HDF5/Qlib bin）与在线（DataService `get_history_window`）复用。

- **REQ-STRATEGY-P3-001：策略函数标准形态**
  - 策略函数需消费：
    - `factors: DataFrame`；
    - `prices: DataFrame`；
    - `portfolio: PortfolioState`；
  - 返回调仓权重或订单计划。

### 3.2 AIstock 侧 REQ

- **REQ-DATASVC-P2-001：统一数据形态（已在项目规范中定义）**

- **REQ-DATASVC-P3-001：仅通过 DataService 获取数据**

- **REQ-MODEL-P3-010：qlib runtime 集成**
  - AIstock 必须：
    - 集成固定版本 qlib；
    - 能根据 `model_conf` / `dataset_conf` / 特征列表，从 RD-Agent 导出的模型文件中加载模型；
    - 通过 DataService 提供的 DataFrame 喂给模型 `predict`。

- **REQ-LOOP-P3-001：基于 loop 的一键重放能力**
  - AIstock 需支持：
    - 在 UI 或 API 中选定某个 loop；
    - 自动解析其因子/模型/策略配置；
    - 使用 DataService + qlib runtime 在自己的执行栈中重放该策略（至少可在模拟盘环境中验证）。

---

## 4. 数据服务层设计对齐（概述）

> 详细接口定义仍在《2025-12-24_DataServiceLayer_Detail_Design_RD-Agent_AIstock.md》中维护，此处只补充 REQ ID 与 Phase 对齐。

- **REQ-DATASVC-P2-001**：
  - `get_history_window` / `get_realtime_snapshot` / `get_portfolio_state` 等接口的签名与数据结构，必须按设计文档实现。

- **REQ-DATASVC-P3-001**：
  - 所有进入模拟盘/实盘执行栈的策略/模型，消费数据时一律通过 DataService 接口。

- **REQ-DATASVC-P3-002**：
  - DataService 必须为 RD-Agent × qlib runtime 提供与离线环境一致的数据形态（字段名、频率、缺失值处理等）。

---

## 5. 小结与后续工作

- 本文件建立在已有三份关键设计文档（顶层/Phase2补充/数据服务层）之上，将其中的“隐性要求”显式化为 REQ ID；
- 后续：
  - 顶层设计文档需要在末尾增加 "硬性要求（REQ Checklist）" 章节，引用本文件中的 REQ ID；
  - RD-Agent 与 AIstock 的实现与测试需要逐步对齐这些 REQ；
  - 当前已知的功能缺口与补齐计划将记录在 `2025-12-30_Gap_Closure_Design_and_Execution_Plan.md` 中。
