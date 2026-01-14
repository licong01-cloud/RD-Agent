# RD-Agent × AIstock Phase 2 详细设计最终版（2026-01-02，权威版）

> 本文件为 RD-Agent × AIstock Phase 2 后续研发的**唯一权威设计方案**。
>
> 核心要求：
> - **最终对外数据与 API 字段必须与既有 Phase 2 版本一致**：不得丢字段、不得改名、不得改变字段语义；允许延迟补齐，但最终必须齐全。
> - **在线最轻量**：RD-Agent 在线执行阶段（workflow/loop finally）只写“标识与状态”，不进行目录遍历/大文件读取/大量文件写入/拷贝/共享库同步。
> - **离线可重试**：所有成果采集（materialize）、共享库同步、catalog 生成均由离线脚本执行，并具备幂等与失败可重试能力。

---

## 1. 范围与总体目标

### 1.1 范围

- **RD-Agent 侧**：
  - 将 `write_loop_artifacts(...)` 的在线职责收敛为 **marker-only**（仅写标识/状态/索引）。
  - 引入并维护 **采集状态机**（materialization status），用于 AIstock 判定“是否已采集完成”。
  - 扩展离线工具 `tools/backfill_registry_artifacts.py`：
    - 新增 `materialize-pending` 模式，对 pending/failed 的 loop 执行完整采集与回填。
    - 保证生成并登记 Phase2 标准 artifacts（JSON/图表/registry artifact_files）与既有版本保持一致。
  - 继续提供只读成果 API：`rdagent-results-api`。
  - 继续提供四大 Catalog 导出脚本：factor/strategy/loop/model。

- **AIstock 侧**：
  - 通过 Results API 完成全量 + 增量同步（同 Phase2 既有模式）。
  - 增加“采集状态判定 + 手工触发补齐”的 UI/后端能力：
    - 以 `materialization_status` 为权威信号，判断哪些任务可消费/不可消费。
    - 对 pending/failed 提供“一键补齐/重试采集”能力。

### 1.2 总体目标

- **成果资产化 + 完整打通到 AIstock**：
  - RD-Agent 产出标准化 artifacts（JSON/图表）与四大 Catalog，并通过 Results API 暴露。
  - AIstock 消费 Catalog + artifacts，形成因子库/策略库/实验库/模型库视图。

- **可靠性优先（科研环境）**：
  - 允许采集延迟，但必须保证最终一致与可重试。
  - 在线环节尽可能不引入额外内存/IO/阻塞，降低 WSL OOM 与残留进程风险。

---

## 2. RD-Agent 侧：Registry 与采集状态机（Materialization State Machine）

### 2.1 背景

- RD-Agent workflow 可能在并发与子进程场景下出现：
  - 子进程树复杂（qrun/factor/worker）
  - 父进程在 finally 阶段执行重 IO 时更易卡死/被 OOM killer 杀死
  - 父进程异常退出会导致子进程残留

因此：Phase2 采集链路必须把重活迁移至离线。

### 2.2 新增字段（Registry 权威）

在 `registry.sqlite` 的 `loops` 表增加：

- `materialization_status`：`pending | running | done | failed`
- `materialization_error`：`TEXT`（最近一次失败摘要，可空）
- `materialization_updated_at_utc`：`TEXT`（推荐，便于观测/排障）

语义：
- `pending`：已标记需要采集，等待离线任务处理
- `running`：离线采集中（用于互斥与 UI 展示）
- `done`：采集完成（对外数据已齐全，可同步/可消费）
- `failed`：采集失败（可重试；错误原因写入 error 字段）

### 2.3 幂等与断点继续

- 断点继续依赖离线采集实现：
  - 对每个目标文件（如 `factor_meta.json`）做“存在且校验通过则跳过/否则重写”的策略
  - 对 registry artifacts/artifact_files 采用稳定 ID 或 upsert，避免重复记录
- `materialization_error` 用于：
  - AIstock UI 展示失败原因
  - 运维快速定位失败根因
  - 下一次重试时确认是否需要清理/覆盖

---

## 3. RD-Agent 侧：在线 `write_loop_artifacts`（marker-only）

### 3.1 位置与调用方式（保持 workflow 结构不变）

- 调用点：`rdagent/utils/workflow/loop.py` 的 step `finally`（保持既有调用结构）。
- 函数位置：`rdagent/utils/artifacts_writer.py`。

### 3.2 在线职责（严格最小集合）

在线 `write_loop_artifacts` 仅做：

1. **定位信息写入/更新**（若已存在则 upsert）：
   - task_run/loop/workspace 的基础行（已有逻辑保持）
   - best_workspace_id、workspace_path、log_trace_path、action、status

2. **采集状态写入**：
   - 当 loop 到达“可采集终态”时（例如 status 进入 success/failed 且 workspace_path 可定位）：
     - `materialization_status = pending`
     - `materialization_error = NULL`
     - `materialization_updated_at_utc = now`

3. **轻量 metrics（可选）**：
   - 仅写入调用方已持有的 `metrics`（内存 dict），禁止为 enrich 而读 `qlib_res.csv`。

### 3.3 在线禁止操作（强约束）

在线 `write_loop_artifacts` 严禁：
- 递归遍历 workspace：`glob/rglob/listdir`
- 读取大文件：`combined_factors_df.parquet`、`ret.pkl`、`signals.parquet`、`mlruns/` 等
- 生成/写入 Phase2 JSON：`factor_meta.json/factor_perf.json/feedback.json`
- 生成图表：`ret_curve.png/dd_curve.png`
- 共享库同步：写 `rd_factors_lib/generated.py`、更新 `VERSION`

---

## 4. RD-Agent 侧：离线 Materialize（统一采集与回填）

### 4.1 离线采集的权威定义

离线采集（materialize）的目标：
- 以 `(task_run_id, loop_id)` 定位到 experiment_workspace
- 生成并登记 Phase2 规定的全部 artifacts（文件 + registry 记录）
- 保证输出的文件名/路径/字段集合与既有 Phase2 版本一致

### 4.2 工具入口：`tools/backfill_registry_artifacts.py`

新增或增强模式：

- `--mode materialize-pending`：
  - 查询 registry 中 `materialization_status IN ('pending','failed')`
  - 对每条 loop：
    1) 置 `running`
    2) materialize 生成并登记所有 artifacts
    3) 成功置 `done`
    4) 失败置 `failed` 并写 error

并保留既有：
- `--mode backfill`：全量历史补齐
- `--mode check`：只读检查

### 4.3 materialize 必须生成/保证存在的文件（与既有 Phase2 一致）

- workspace 根目录：
  - `workspace_meta.json`
  - `experiment_summary.json`
  - `manifest.json`
  - `factor_meta.json`
  - `factor_perf.json`
  - `feedback.json`
  - `ret_curve.png`（以及可选 `dd_curve.png`）
  - 对 model loop：`model_meta.json`（若原版要求）

### 4.4 materialize 必须登记的 registry 记录

- `artifacts` / `artifact_files`：完整登记所有上述文件
- `loops.has_result` 与关键指标字段（遵循 REQ-LOOP-P2-001）

### 4.5 生成逻辑复用原则（保证最终一致）

- 离线脚本必须复用 `rdagent/utils/artifacts_writer.py` 的 payload 构建 helper
- 禁止在 AIstock 侧“补字段/猜字段”
- 允许延迟，但最终必须齐全

---

## 5. RD-Agent 侧：因子共享包（rd_factors_lib）与 impl_* 指针（离线同步）

### 5.1 Phase2 的最终口径

- `impl_module/impl_func/impl_version` 字段 **最终必须齐全**（因子元数据与 catalog 必须暴露）。
- 共享库同步与 impl_* 回写 **作为离线任务执行**：
  - 可集成在 materialize 完成后
  - 或作为独立 `--mode sync-shared-lib`

### 5.2 对外一致性要求

- 同一因子的：
  - `factor_meta.json` 中 impl_* 字段
  - `factor_catalog.json` 中 impl_* 字段
  - `rd_factors_lib` 实际实现与 VERSION

必须一致。

---

## 6. RD-Agent 侧：只读成果 API（保持只读接口不变）

### 6.1 现有接口保持

- `/catalog/factors|strategies|loops|models`
- `/alpha158/meta`
- `/factors/{name}`
- `/loops/{task_run_id}/{loop_id}/artifacts`

### 6.2 采集状态对外暴露

- `materialization_status/materialization_error` 推荐进入 `loop_catalog.json` 并通过 `/catalog/loops` 暴露。
- Phase2 可不新增写接口；AIstock 触发补齐采用“执行离线脚本”的方式。

---

## 7. 四大 Catalog（字段集合必须保持一致）

- 因子/策略/loop/模型四大 catalog 的 schema 与字段集合，保持与既有 Phase2 版本一致。
- `loop_catalog` 建议新增（或透传）字段：
  - `materialization_status`
  - `materialization_error`

> 若新增字段会影响既有 AIstock schema，则改为 AIstock 通过 registry 查询；但推荐走 catalog（便于 UI）

---

## 8. AIstock 侧：采集状态判定、同步与手工补齐

### 8.1 判定规则

- 已采集：`materialization_status == done`
- 未采集：`pending/failed`
- 采集中：`running`

### 8.2 UI 功能

- 在实验/回测列表中展示采集状态
- 对 pending/failed 提供“触发补齐/重试”按钮
- failed 展示 error 摘要

### 8.3 触发补齐（Phase2 最小侵入）

- AIstock 触发 RD-Agent 机器执行离线命令：
  1) `materialize-pending`（可带 task_run_id/loop_id 过滤）
  2) 重新 export catalog
  3) AIstock 再同步 Results API

---

## 9. 开发执行顺序（建议按此顺序落地，保证最小风险）

1) **Registry schema 升级**：loops 表新增 materialization 字段；提供 upsert 方法
2) **在线 write_loop_artifacts 轻量化**：改为仅写 pending 标识
3) **离线 backfill 扩展**：实现 `--mode materialize-pending`，并把现有 Phase2 产物生成逻辑集中到离线路径
4) **Catalog export 适配**：从 registry/文件读取 materialization 状态并写入 loop_catalog（可选）
5) **AIstock 同步逻辑**：同步时以 done 为可消费条件；UI 增加触发补齐入口
6) **共享库同步（Phase3能力）**：离线实现 sync-shared-lib 并回写 impl_*

---

## 10. REQ Checklist（更新后口径）

- REQ-FACTOR-P2-001：impl_* 字段最终齐全且与共享库一致（允许离线延迟）
- REQ-FACTOR-P2-002：factor_meta / catalog 字段齐全（freq/align/nan_policy 不得省略）
- REQ-LOOP-P2-001：has_result=1 的 loop 至少一个关键指标非空（由离线 materialize 填充）
- REQ-LOOP-P2-002：历史与新 loop 的 artifacts/artifact_files 最终必须齐全（离线幂等补齐）
- REQ-API-P2-001：Results API 字段集合不得裁剪
- 新增：REQ-ARTIFACT-P2-MAT-001：materialization 状态机字段与语义

---

## 附录：历史文档列表（仅用于追溯，不再作为权威来源）

- `docs/2025-12-29_Phase2_Detail_Design_RD-Agent_AIstock_Final.md`
- `docs/2025-12-23_Phase2_Detail_Design_RD-Agent_AIstock.md`

