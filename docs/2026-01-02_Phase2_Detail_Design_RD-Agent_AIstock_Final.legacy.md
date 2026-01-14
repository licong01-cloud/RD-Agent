# RD-Agent × AIstock Phase 2 详细设计最终版（2026-01-02）

> 本文件为 RD-Agent × AIstock Phase 2 的**最新唯一权威设计文档**。
>
> 设计原则：
> - **兼容既有 Phase 2 对外数据与接口**：最终产出的 JSON 文件、Catalog 字段集合、Results API 返回结构必须与既有 Phase 2 版本保持一致（不得丢字段、不得空字段、不得改名、不得改含义）。
> - **允许延迟（eventual consistency）**：RD-Agent 在线执行阶段不再做重型采集，允许由离线任务在事后补齐所有数据，AIstock 侧以“采集状态”为准判断是否可消费。
> - **在线最轻量、离线可重试**：`artifacts_writer` 在线只写标识/状态（marker/registry），所有扫描 workspace、读大文件、写大量 JSON/图片、同步共享库等都由离线脚本完成。
>
> 与历史文档关系：
> - 本文件**全文收录**并继承 `docs/2025-12-29_Phase2_Detail_Design_RD-Agent_AIstock_Final.md` 的全部内容（见文末《附录 Z》）。
> - 若本文与收录原文存在冲突，**以本文（2026-01-02）为准**；收录原文仅用于追溯。

---

## 0. 2026-01-02 关键变更摘要（相对 2025-12-29 版本）

### 0.1 变更动机（工程约束）

- RD-Agent 在 WSL 环境中存在 OOM/崩溃风险。
- `write_loop_artifacts` 在线执行包含目录扫描、读写多个文件、生成图片、同步共享库等重操作，会：
  - 拉长 workflow `finally` 阶段，增加父进程处于“子进程仍在跑但收尾卡住”的窗口；
  - 在内存紧张时更容易触发 OOM killer 导致父进程异常退出，进而产生残留子进程。

因此 Phase2 的工程实现必须升级为：
- 在线 **marker-only**（只写最轻量标识，不做任何重 I/O/扫描/拷贝/同步）；
- 离线 **materialize**（可重试/可断点续跑）生成所有 Phase2 标准产物与 registry 记录；
- 最终对外字段与接口保持与既有 Phase2 一致。

### 0.2 行为变化（对外兼容策略）

- **对外不变**：
  - workspace 内的标准文件名/路径（`factor_meta.json`、`factor_perf.json`、`feedback.json`、`workspace_meta.json`、`experiment_summary.json`、`manifest.json`、`ret_curve.png`、`dd_curve.png` 等）保持不变；
  - `RDagentDB/aistock/*.json` 四大 Catalog 输出字段集合保持不变；
  - `rdagent-results-api` 所有只读接口返回结构保持不变。

- **时序允许延迟**：
  - loop/workspace 在在线运行结束后可能处于“产物尚未 materialize 完整”的中间态；
  - AIstock 不应基于“目录存在/日志存在”判断已采集，而应基于 registry 的 `materialization_status`（或由其导出的 loop_catalog 字段）判断。

### 0.3 新增：采集状态机字段（Registry 权威）

为支持 AIstock 判定“已采集/未采集”，并支持离线任务失败后可重试，必须在 registry（SQLite）中增加采集状态字段。

- 字段名：
  - `materialization_status`: `pending | running | done | failed`
  - `materialization_error`: `TEXT`，记录最近一次失败摘要（可为空）
  - `materialization_updated_at_utc`: `TEXT`（推荐）

- 语义：
  - `pending`：已标记需要采集，但尚未开始 materialize
  - `running`：正在 materialize（用于 UI 展示与互斥）
  - `done`：已 materialize 完整（对外数据已齐全，可同步）
  - `failed`：materialize 失败（可重试；错误信息写入 `materialization_error`）

> 说明：状态字段本身不等于“断点续跑”，断点续跑依赖离线 materialize 的幂等实现：已生成且校验通过的文件/记录将被跳过。

---

## 1. RD-Agent 侧：在线 marker-only 与离线 materialize 的最终方案

### 1.1 在线：`write_loop_artifacts` 退化为轻量标识写入（marker-only）

#### 1.1.1 位置与调用点

- 调用点仍保持在 `rdagent/utils/workflow/loop.py` 的 step `finally` 中（不改变 workflow 的结构）。
- 但 `rdagent/utils/artifacts_writer.py:write_loop_artifacts(...)` 的默认行为改为：
  - **不扫描 workspace**
  - **不生成/写入任何 Phase2 大文件**
  - **不生成图片**
  - **不登记 artifact_files（文件级清单）**
  - **不执行共享库同步**

#### 1.1.2 在线必须写入的最小信息

在线必须保证 AIstock 与离线脚本能定位任务与产物：

- registry：
  - `task_runs` / `loops` / `workspaces` 的基础 upsert（已有）
  - 若 loop 到达可采集终态（例如 `status in (success, failed)` 且 workspace_path 存在）：
    - 将 `materialization_status` 置为 `pending`
    - 清空 `materialization_error`
    - 记录/更新 `materialization_updated_at_utc`

> 允许在线写极小 JSON marker 文件（可选），但非必须；本方案以 registry 为权威索引。

#### 1.1.3 在线禁止的重操作清单（强约束）

在线 `write_loop_artifacts` **禁止**：
- `Path.glob / Path.rglob` 递归遍历 workspace
- 读取 `combined_factors_df.parquet`、`ret.pkl`、`signals.parquet`、`mlruns/` 等
- 解析 `qlib_res.csv` 来 enrich metrics
- 生成/写入 `factor_meta.json` / `factor_perf.json` / `feedback.json`
- 生成/写入 `ret_curve.png` / `dd_curve.png`
- 写入/追加 `F:\Dev\rd-factors-lib\rd_factors_lib\generated.py`（共享库同步）

---

### 1.2 离线：materialize 统一由 `tools/backfill_registry_artifacts.py` 承担

#### 1.2.1 离线 materialize 的权威定义

离线 materialize 的目标是：
- 对每个 `(task_run_id, loop_id)` 对应的 `best_workspace_id`（或 experiment_workspace）
- 生成并登记 Phase2 所需的全部 JSON/图表与 registry 记录
- **最终产物与既有 Phase2 实现保持一致**（字段集合、文件名、相对路径、语义）

#### 1.2.2 工具增强：新增 mode `materialize-pending`

在 `tools/backfill_registry_artifacts.py` 增加：

- `--mode materialize-pending`：
  - 查询 registry 中 `materialization_status IN ('pending','failed')` 的 loops
  - 对每条记录：
    - 将其状态置 `running`
    - 执行 materialize：生成 JSON、图表、登记 artifacts/artifact_files
    - 成功：置 `done`，清空 error
    - 失败：置 `failed`，写入 error（摘要）

> materialize 过程中必须遵守幂等：重复执行不会产生重复 DB 记录，且不会损坏已有文件。

#### 1.2.3 materialize 的实现要求（保证最终与原版一致）

- 必须生成/保证存在（按既有 Phase2 契约）：
  - `workspace_meta.json`
  - `experiment_summary.json`
  - `manifest.json`
  - `factor_meta.json`
  - `factor_perf.json`
  - `feedback.json`
  - `ret_curve.png`（以及可选 `dd_curve.png`）
  - 对 model loop：`model_meta.json`（如原版要求）

- 必须登记：
  - `artifacts` / `artifact_files` 全量文件清单
  - `loops.has_result` 与关键指标字段（遵循原版 REQ）

- 必须复用同一套 payload 构建逻辑：
  - 尽可能复用 `rdagent/utils/artifacts_writer.py` 的 helper
  - 确保 schema 字段不缺失、不为空字段、不删字段

---

### 1.3 因子共享库同步（Phase3能力）：完全迁移为离线任务

- 原 2025-12-29 文档中关于“loop 末尾自动写共享包”的在线约束，在 2026-01-02 版本中调整为：
  - 共享库同步（写 `rd_factors_lib.generated`、更新 `VERSION`、回写 `impl_*`）作为**离线任务**执行。
  - 可由 `backfill_registry_artifacts.py` 的扩展 mode 执行：例如 `--mode sync-shared-lib`，或集成在 `materialize-pending` 的后续步骤。

- 对外契约保持：
  - 最终 `factor_meta.json` 与 factor_catalog 中的 `impl_module/impl_func/impl_version` 必须齐全且一致。
  - 只是生成时机允许延迟。

---

## 2. AIstock 侧：采集状态判定 + 手工触发补齐（基于 log_dir / loop 维度）

### 2.1 判定“已采集/未采集”的权威规则

AIstock 判定必须基于 registry 的 `materialization_status`（或其被导出到 `loop_catalog.json` 的对应字段）：

- 已采集：`materialization_status == 'done'`
- 未采集：`materialization_status in ('pending','failed')`
- 采集中：`materialization_status == 'running'`

> 禁止仅依据 `log_dir` 目录存在与否判断已采集。

### 2.2 UI 功能：对未采集任务展示“补齐/重试采集”

- 在实验/回测记录列表页（以 `(task_run_id, loop_id)` 或 `log_dir` 唯一定位）：
  - 展示采集状态
  - 对 pending/failed 提供按钮：`触发补齐` / `重试`
  - 对 failed 展示 `materialization_error`（摘要）

### 2.3 触发补齐的实现方式（两种等价方案）

- 方案 A（Phase2 最小侵入，推荐）：
  - AIstock 触发 RD-Agent 机器上的运维任务（执行脚本命令）
  - 脚本完成后再触发 catalog export，再同步 Results API

- 方案 B（Phase3 工程化增强）：
  - 在 `rdagent-results-api` 增加 `/admin/materialize` 类管理端点（需鉴权/内网）
  - AIstock 通过 HTTP 触发 materialize，并轮询 status

> Phase2 交付优先方案 A；方案 B 作为后续增强。

---

## 3. 对既有 Phase2 REQ 的调整说明（保持最终字段一致）

### 3.1 保持不变的 REQ（最终数据必须齐全）

- 因子元数据字段齐全（`description_cn/formula_hint/variables/freq/align/nan_policy/impl_*`）
- loop_catalog 指标字段齐全（至少一个关键指标非空）
- Results API 字段集合不得裁剪

### 3.2 调整的 REQ（共享库同步从在线迁移至离线）

- 原 `REQ-FACTOR-P2-011`（在线 loop 末尾写共享包）调整为：
  - 共享包写入由离线任务完成，但最终必须回写 `impl_*` 到 `factor_meta.json` 与 catalog。

### 3.3 新增 REQ：采集状态机

- `loops` 必须包含：`materialization_status/materialization_error`（以及更新时间字段）
- AIstock 判定“已采集/未采集”必须依据该字段

---

## 4. 推荐的端到端运维流程（以延迟换稳定）

### 4.1 RD-Agent 侧（离线任务）

1. materialize pending/failed：

```bash
python tools/backfill_registry_artifacts.py \
  --db /mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite \
  --mode materialize-pending
```

2. 刷新四大 catalog（保持原命令不变）：

```bash
python tools/export_aistock_factor_catalog.py   --registry-sqlite ... --output ...
python tools/export_aistock_strategy_catalog.py --registry-sqlite ... --output ...
python tools/export_aistock_model_catalog.py    --registry-sqlite ... --output ...
python tools/export_aistock_loop_catalog.py     --registry-sqlite ... --output ...
```

### 4.2 AIstock 侧

- 定时/手动触发同步：
  - 先读取 `/catalog/loops` 获取 materialization 状态
  - 仅对 `done` 的记录进行业务侧“可消费”标记
  - 对 pending/failed 提供 UI 触发补齐

---

## 附录 Z：2025-12-29 Phase2 最终版原文全文收录（用于追溯）

> 说明：以下内容为 `docs/2025-12-29_Phase2_Detail_Design_RD-Agent_AIstock_Final.md` **全文原样复制**。
> 若与本文主文冲突，以本文主文为准。



---

# 2025-12-29 ԭ�Ŀ�ʼ��ȫ��ԭ����¼��

# RD-Agent × AIstock Phase 2 详细设计最终版（2025-12-29）

> 本文件为 RD-Agent × AIstock Phase 2 设计的**最终整合版**，在内容上整合并去重自：
> - `2025-12-23_Phase2_Detail_Design_RD-Agent_AIstock.md`
> - `2025-12-26_Phase2_Detail_Design_Supp_FactorPackage_and_API.md`
> - `2025-12-26_Phase2_Detail_Design_RD-Agent_AIstock_v2.md`
>
> 所有需求与接口说明均以本文件为**唯一入口文档**；旧版文件保留用于追溯，不再单独作为需求来源。若与旧文存在表述不一致，以本文件为准，再回溯更新历史文档。

---

## 1. 范围与总体目标

### 1.1 范围

- **RD-Agent 侧**：
  - 抽象并实现统一的 `write_loop_artifacts(...)` 函数；
  - 在 model loop 完成后，集中调用该函数生成所有约定的 artifacts；
  - 增加并规范化三类新的核心 JSON artifacts：
    - `factor_meta.json`
    - `factor_perf.json`
    - `feedback.json`
  - 统一生成/登记回测图表文件（如 `ret_curve.png`, `dd_curve.png`）；
  - 基于 registry.sqlite 与 workspace artifacts，导出 AIstock-facing 四大 Catalog：因子/策略/loop/模型；
  - 提供只读成果 API 服务 `rdagent-results-api` 暴露上述 Catalog 与关键 artifacts 视图；
  - 通过 backfill 工具为历史任务补齐 Phase 2 所需 artifacts 与 registry 记录。

- **AIstock 侧**：
  - 在 Phase 1 的 registry 只读消费能力基础上：
    - 通过 HTTP 调用 RD-Agent 只读成果 API 完成**全量 + 增量**同步；
    - 将因子/策略/loop/Alpha158/模型元数据写入本地数据库；
    - 新增“因子库/策略库/实验库”视图；
    - 在实验详情页展示 `feedback.json` 与回测曲线；
  - 在 Phase 2 阶段，**至少在离线/研究场景中**完成数据服务层的生产级实现（满足 REQ-DATASVC-P2-001），为因子共享包调试与后续执行迁移提供完整数据基础，而非任何形式的“精简版”或 PoC。

### 1.2 总体目标

- **成果资产化 + 完整打通到 AIstock**：
  - 从“只能看策略回测结果（signals/ret）”升级为“能系统性查看所有演进因子及其表现，以及每次实验的反馈与评估”；
  - RD-Agent 通过统一 artifacts 与四大 Catalog，将成果以规范化结构暴露给 AIstock；
  - AIstock 通过同步任务与本地 DB schema 承接全部成果，为后续 Phase 3/4/5 提供统一基础。

- **兼容 Phase 1**：
  - 不修改 Phase 1 已有 artifacts 类型与含义；
  - 新增 artifacts 与 API 均以向后兼容方式接入；
  - 旧数据在不执行 backfill 的情况下，AIstock 至少能继续消费 Phase 1 成果；
  - 执行 backfill 后，旧数据也具备 Phase 2 新增 artifacts 能力。

---

## 2. RD-Agent 侧：统一 artifacts 写入与 backfill

### 2.1 `write_loop_artifacts(...)` 函数签名与职责

- 文件位置（示意）：`rdagent/utils/artifacts_writer.py`
- 核心函数：

```python
from pathlib import Path
from typing import Any

from rdagent.registry import RegistryConnection  # 示意类型


def write_loop_artifacts(
    conn: RegistryConnection,
    *,
    task_run_row: dict[str, Any],
    loop_row: dict[str, Any],
    workspace_path: Path,
    action: str,
    has_result: bool,
) -> None:
    """根据 workspace 中已有文件，生成并登记所有约定的 artifacts。

    - 仅在 `action == 'model'` 且 has_result == True 时写“完整成果”；
    - 对 `action == 'factor'`，可视需要仅写部分调试信息（Phase 2 可不覆盖）。
    """
    ...
```

**职责**：

1. 检查 workspace 下已有文件：
   - `ret.pkl`, `qlib_res.csv`, `signals.*`, `ret_schema.*`, `combined_factors_df.parquet`, `conf*.yaml`, `mlruns/` 等；
2. 生成/刷新：
   - `workspace_meta.json`、`experiment_summary.json`、`manifest.json`；
   - `factor_meta.json`、`factor_perf.json`、`feedback.json`；
   - 回测图表文件（`ret_curve.png`, `dd_curve.png`）；
3. 在 `artifacts` / `artifact_files` 中登记所有上述文件；
4. 保证对同一 loop 重复调用是**幂等的**（不会生成重复 DB 记录，也不会破坏已有记录）。

### 2.2 与 loop 执行流程的整合

- 在 `rdagent/utils/workflow/loop.py` 中原有 meta/summary/manifest/artifacts 写入逻辑已整体迁移至 `write_loop_artifacts(...)`；
- 目前 `loop.py` 中仅保留一行（或少量）调用：

```python
write_loop_artifacts(
    conn,
    task_run_row=task_run_snapshot,
    loop_row=loop_snapshot,
    workspace_path=ws_root,
    action=action,
    has_result=has_result,
)
```

- 迁移过程保持行为兼容：在未开启新 artifacts 逻辑前，行为与 Phase 1 完全一致；
- 在此基础上，逐步在 `write_loop_artifacts` 内加入新 JSON/图表 artifacts 的生成与登记逻辑。

### 2.3 Phase 2 新增 JSON artifacts 设计

#### 2.3.1 `factor_meta.json`

**目的**：记录本次实验中“参与回测的因子”的元信息，支撑 AIstock 因子库与解释性视图。

**结构示意：**

```json
{
  "version": "v1",
  "generated_at_utc": "2025-12-23T11:00:00Z",
  "factors": [
    {
      "name": "VolAdj_Momentum_10D",
      "source": "rdagent_generated",  
      "description_cn": "10日价格动量按过去20日收益波动率进行风险调整",
      "formula_hint": "(close_t/close_{t-10}-1) / annualized_std(returns_{t-20..t-1})",
      "created_at_utc": "2025-12-23T11:00:00Z",
      "experiment_id": "<task_run_id>/<loop_id>",
      "tags": ["momentum", "vol_adjusted", "daily"],
      "variables": {"window": 10, "vol_window": 20},
      "impl_module": "rd_factors_lib.generated",
      "impl_func": "factor_voladj_momentum_10d",
      "impl_version": "1.0.8"
    }
  ]
}
```

**生成来源与要点**：

- 来源信息：
  - RD-Agent 内部的 hypothesis/justification 文本；
  - 因子函数名（如 `calculate_VolAdj_Momentum_10D`）；
  - `factor.py` 中的注释（如存在）；
  - 静态因子表 schema（`static_factors_schema.json`）。
- Phase 2 要求：
  - 稳定的因子名 `name`；
  - 因子来源 `source`（`rdagent_generated` / `qlib_alpha158` / `external` 等）；
  - 至少一行简要中文说明 `description_cn`（可由 LLM/规则生成）；
  - `variables` 字段透传 FactorTask / 日志中的结构，不做字段精简；
  - 若通过因子共享包导出实现，则补充 `impl_module` / `impl_func` / `impl_version` 字段。

#### 2.3.2 `factor_perf.json`

**目的**：记录单因子与组合因子在回测中的表现摘要，支持 AIstock 按因子维度筛选与排序。

**结构示意：**

```json
{
  "version": "v1",
  "generated_at_utc": "2025-12-23T11:00:00Z",
  "factors": [
    {
      "name": "VolAdj_Momentum_10D",
      "metrics": {
        "ic_mean": 0.045,
        "ic_ir": 1.10,
        "coverage": 0.95
      },
      "windows": [
        {
          "name": "test_2021_2025",
          "start": "2021-01-01",
          "end": "2025-12-01",
          "annual_return": 0.18,
          "max_drawdown": 0.42,
          "sharpe": 1.20
        }
      ]
    }
  ],
  "combinations": [
    {
      "name": "SOTA_plus_new_20251223",
      "factor_names": ["VolAdj_Momentum_10D", "MF_Main_Trend_5D"],
      "windows": [
        {
          "name": "test_2021_2025",
          "annual_return": 0.22,
          "max_drawdown": 0.39,
          "sharpe": 1.40,
          "metrics": {
            "turnover": 2.3,
            "volatility": 0.22
          }
        }
      ]
    }
  ]
}
```

**生成来源与要点**：

- 单因子层指标：
  - 基于 `combined_factors_df.parquet` 计算的单因子描述统计及 IC/RankIC 等；
- 组合层指标：
  - 从 Qlib 回测结果（`qlib_res.csv`、`ret.pkl` 或 `experiment_summary`）中解析，或在 `read_exp_res.py` 中输出更多字段供使用；
- Phase 2 要求：
  - `factors[*].metrics` 至少提供 `ic_mean` / `ic_ir` / `coverage`，可附带更多统计量；
  - `combinations[*].windows[*]` 嵌入完整 metrics 字典（年化收益、最大回撤、Sharpe 等）。

#### 2.3.3 `feedback.json`

**目的**：以结构化形式记录本轮实验的关键反馈与评估，供 AIstock 直接展示，无需解析日志。

**结构示意：**

```json
{
  "version": "v1",
  "decision": true,
  "hypothesis": "VolAdj_Momentum_10D：10日价格动量按20日波动率调整...",
  "summary": {
    "execution": "Execution succeeded without error. Expected output file found.",
    "value_feedback": "因子实现正确，符合模板和数据要求...",
    "shape_feedback": "Index is MultiIndex(datetime, instrument), single float64 column...",
    "code_critic": [
      "窗口定义包含当前日，可能与描述中‘过去10日’有轻微偏差..."
    ],
    "limitations": [
      "Dynamic factors instruments overlap ratio is 0.81, close to threshold 0.8.",
      "回撤集中在 2021–2022 的极端行情，对该阶段过拟合需注意。"
    ]
  }
}
```

**字段映射规则（`HypothesisFeedback` → `feedback.json`）**：

- 决策：
  - `HypothesisFeedback.decision` → `feedback.decision`；
- 假设：
  - 优先用 `HypothesisFeedback.new_hypothesis` 作为 `feedback.hypothesis`；
  - 若为空，则回退为 `exp.hypothesis.hypothesis`；
- 总结与评价：
  - `HypothesisFeedback.observations` → `summary.execution`；
  - `HypothesisFeedback.hypothesis_evaluation` → `summary.value_feedback`；
  - `HypothesisFeedback.reason` 作为对上述的补充，可拼接进 `value_feedback` 或 `shape_feedback`，但语义保持“评价/推理补充”；
- 局限性与代码审阅：
  - 如反馈生成逻辑产出 `limitations` / `code_critic` 数组，则映射到 `summary.limitations[]` / `summary.code_critic[]`；
  - 若当前版本未显式提供，则这两个字段可为空数组。

> `feedback.json` 是**单次实验级别快照**：描述“这一轮实验对策略/因子的观察和评价”。长期因子定义与公式提示来自 `factor_meta.json`；AIstock 可在导入后按需汇总多轮 `limitations` 等字段。

#### 2.3.4 JSON Schema 版本管理与容错

- `factor_meta.json` / `factor_perf.json` / `feedback.json` **统一包含**：
  - `version` 字段（如 `"v1"`）；
  - `generated_at_utc` 字段；
- 后续新增字段时不改变既有字段语义，保持向后兼容；
- `write_loop_artifacts` 写入逻辑要求：
  - 幂等（多次执行不会产生重复 DB 记录）；
  - 单个 JSON/图表生成失败时不阻断整体流程，在 `experiment_summary` 或 `feedback` 中追加 warning；
  - 写文件时可覆盖旧版本，或通过稳定文件名 + `mtime` 管理版本。

### 2.4 回测图表文件生成与登记

- 输入：
  - `ret.pkl`（含收益曲线的 DataFrame 或 Series）。
- 输出（至少）：
  - `ret_curve.png`：净值或收益曲线；
  - 可选 `dd_curve.png`：回撤曲线。

**实现要点**：

- 在 `artifacts_writer` 内提供 `_save_ret_plots(ret_pkl_path, out_dir)`：
  - 使用 matplotlib/plotly 生成静态图；
  - 将 `ret_curve.png`（和可选 `dd_curve.png`）保存到 workspace 并登记为 `artifact_type='backtest_curve_image'`；
- `write_loop_artifacts` 及 backfill 工具在检测到 `ret.pkl` 时调用该 helper；
- AIstock 通过 `loop_catalog` 与 `/loops/{task_run_id}/{loop_id}/artifacts` 拿到图表文件路径，并在前端映射为可访问 URL。

### 2.5 Backfill 工具：历史成果补齐

- 工具：`tools/backfill_registry_artifacts.py`
- Phase 2 要求其支持：
  - 遍历与检查历史任务：
    - 扫描 registry.sqlite 中全部历史 task_run / loop / workspace（支持 `--all-task-runs` 一次性处理全部任务，或用 `--task-run-id` 精确指定单个任务）；
    - 通过 workspace 目录与已有文件（`qlib_res.csv`、`ret.pkl`、`combined_factors_df.parquet` 等）判断是否具备 Phase 2 所需成果；
  - 补齐缺失 artifacts 与 registry 记录：
    - 自动生成缺失的 `workspace_meta.json` / `experiment_summary.json` / `manifest.json`；
    - 自动生成缺失的 `factor_meta.json` / `factor_perf.json` / `feedback.json` / 回测图表文件（`ret_curve.png`、`dd_curve.png`）；
    - 在 `artifacts` / `artifact_files` 中登记相应记录；
    - 对符合条件的 loop 将 `has_result` 更新为 `1`；
  - 提供两种 **mode**：
    - `--mode backfill`（默认）：执行实际补录逻辑，写入/更新 JSON 与 DB 记录；
    - `--mode check`：严格只读模式，仅检查每个 workspace 是否具备 Phase 2 所需 JSON/图表及 registry 记录，输出检查结果 JSON，不修改任何文件或 DB；
  - 支持多种过滤与控制参数：
    - `--only-experiment-workspace`：仅处理 `workspace_role='experiment_workspace'` 的 workspace；
    - `--since-date`：基于 workspace 路径的 mtime 过滤，仅处理最近更新的 workspace；
    - `--max-loops`：限制本次处理的 workspace 数量；
    - `--overwrite-json`：允许覆盖已有 JSON 文件，否则只在缺失时创建；
    - `--cleanup-existing`：在 backfill 之前先清理选中 workspace 对应的旧 artifacts/artifact_files 记录，然后按当前规范重建，避免历史多次 backfill 遗留的重复或脏数据；
    - `--dry-run`：在 backfill 模式下只计算计划执行的操作并输出结果 JSON，不实际写盘或更新 DB；
  - 支持“日志驱动”的增强 backfill：
    - 通过 `--log-path` 指定 RD-Agent 日志根目录；
    - 工具会从 registry 的 `task_runs.log_trace_path` 字段拼接出每个 task_run 的日志目录，解析其中的 `FactorTask` 与 `HypothesisFeedback`；
    - 仅对决策 `decision=True` 的 loop 执行 Phase 2 补录，并用日志中的因子描述/公式提示/反馈内容丰富 `factor_meta.json` 与 `feedback.json`，保证这些 JSON 能直接支撑 AIstock 侧的因子库与实验反馈视图，而无需解析原始日志。

**使用建议**：

- 在首次 AIstock 对接前，推荐按如下顺序执行一次全量补录：
  1. 使用 `--mode check --all-task-runs --only-experiment-workspace --dry-run` 做全量只读检查，生成当前缺口报告；
  2. 使用 `--mode backfill --all-task-runs --only-experiment-workspace --overwrite-json --cleanup-existing` 按当前规范重建所有历史 experiment_workspace 的 Phase 1 + Phase 2 JSON 与 artifacts 记录；
  3. 如需要利用日志中的决策与因子描述进一步丰富 JSON，可在上述基础上再执行一次 `--mode backfill --log-path <RD-Agent 日志根目录> --overwrite-json`，仅对决策为 True 的 loop 做日志驱动增强补录；
- 后续可按需要定期运行 `--mode backfill` 配合 `--since-date` / `--max-loops` 做增量补录，保证新产生的 task_run/loop 与历史数据在 artifacts/JSON/registry 上保持同一规范，不存在“旧版 schema 残留”。

---

## 3. RD-Agent 侧：因子共享包设计

### 3.1 目录结构与安装

- 在 RD-Agent 仓库平级目录维护独立 Python 包，例如：

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

- 安装方式（RD-Agent 与 AIstock 各自虚拟环境执行一次即可）：

```bash
pip install -e F:\Dev\rd-factors-lib
```

- 版本管理：
  - `VERSION` 文件或 `__init__.py` 中记录当前版本字符串（如 `"1.0.7"`）；
  - RD-Agent 在导出因子元数据时读取版本号写入 `impl_version` 字段。

### 3.2 因子演进流程与共享包更新

在 RD-Agent 因子演进 loop 末尾（候选因子通过验证被标记为“成功”时），增加“共享包入库”逻辑：

1. 从当前 loop 的因子实现（生成的 `factor.py` 片段或字符串）中抽取核心函数，统一签名：

```python
def factor_xxx(df: pd.DataFrame) -> pd.Series | pd.DataFrame:
    ...
```

2. 写入或更新到 `rd_factors_lib/generated.py`：
   - 若函数名已存在，则覆盖旧实现（保留必要历史信息）；
   - 若为新因子，则追加新函数定义；
3. 自动更新版本号：
   - 递增 `VERSION` 中的次版本号或补丁号（例如 `1.0.7` → `1.0.8`）；
4. 在本 loop 对应的 `factor_meta.json` 中记录实现指针：

```json
{
  "name": "FACTOR_XXX",
  "source": "rdagent_generated",
  "impl_module": "rd_factors_lib.generated",
  "impl_func": "factor_xxx",
  "impl_version": "1.0.8"
}
```

> 上述 1–4 步完全由 RD-Agent 内部 Python 代码自动完成，不依赖人工编辑共享包文件。

### 3.3 Alpha158 因子与共享包

- Alpha158 因子元信息来自 `tools/export_alpha158_meta.py` 从 Qlib 配置导出的 `alpha158_meta.json`；
- Phase 2 内：
  - 不强制在共享包中实现全部 Alpha158 因子函数；
  - 仅保证 Expression（Qlib 表达式）和元信息完整，并在 factor_catalog 中标记 `source="qlib_alpha158"`；
- 如需对部分 Alpha158 因子提供 Python 参考实现，可在 `alpha158.py` 中实现，并在导出元数据时补充 `impl_module` / `impl_func`。

### 3.4 Phase 2 占位实现与后续增强

- 当前实现已在 `rd_factors_lib` 包内提供骨架：`__init__.py`、`generated.py`、`VERSION`；
- 在 `write_loop_artifacts` 中集成 `_sync_factor_impl_to_shared_lib`：
  - 为通过验收的演进因子写入占位函数 stub（抛 `NotImplementedError` 或仅记录源码字符串）；
  - 回写 `impl_module` / `impl_func` / `impl_version` 至 `factor_meta.json`；
- Phase 2 目标：
  - 保证 AIstock 可以依赖稳定的函数入口与版本号做对账；
  - 具体数值行为仍以 RD-Agent workspace 中的 `factor.py` 为准；
  - 完整 reference 实现留待 Phase 3 因子迁移增强时补齐。

---

## 4. RD-Agent 侧：只读成果 API 设计

### 4.1 服务定位与部署

- 服务名（示意）：`rdagent-results-api`；
- 部署位置：
  - 运行在 RD-Agent 同一环境（如 WSL 内），监听本机端口（如 `http://127.0.0.1:9000`）；
- 安全与边界：
  - 只提供只读接口，不执行交易或生成实时信号；
  - 访问范围限定在本机或受控内网。

### 4.2 FastAPI 应用与 CLI 启动

- FastAPI 应用：`rdagent.app.results_api_server:create_app`；
- CLI 启动入口：`rdagent.app.cli`；示例命令：

```bash
python -m rdagent.app.cli results_api --host 127.0.0.1 --port 9000
```

### 4.3 Catalog 相关接口

- `GET /catalog/factors`
  - 返回内容：`RDagentDB/aistock/factor_catalog.json`；
  - 支持查询参数：`source`（`qlib_alpha158` / `rdagent_generated` 等）、`name` 前缀等（可选）。

- `GET /catalog/strategies`
  - 返回内容：`RDagentDB/aistock/strategy_catalog.json`；

- `GET /catalog/loops`
  - 返回内容：`RDagentDB/aistock/loop_catalog.json`；

- `GET /catalog/models`
  - 返回内容：`RDagentDB/aistock/model_catalog.json`。

### 4.4 因子与 Alpha158 元信息接口

- `GET /factors/{name}`
  - 在 factor_catalog 中按 `name` 返回单条因子完整记录；
  - 字段包括：`name`, `source`, `description_cn`, `formula_hint`, `tags`, 表现指标、`impl_module` / `impl_func` / `impl_version` 等。

- `GET /alpha158/meta`
  - 返回 `RDagentDB/aistock/alpha158_meta.json` 内容；
  - 用于 AIstock 构建 Alpha158 因子库与后续迁移。

### 4.5 实验与 artifacts 视图接口

- `GET /task_runs` / `GET /loops` / `GET /workspaces`
  - 封装 registry.sqlite：返回任务、循环、workspace 元信息；
  - 提供按状态/时间区间过滤（可选）。

- `GET /loops/{task_run_id}/{loop_id}/artifacts`
  - 直接查询 `artifacts` / `artifact_files` 表，返回该 loop 所有关键 artifacts：
    - `factor_meta`, `factor_perf`, `feedback`, `ret_curve`, `dd_curve` 等；
  - 字段包括：文件相对路径、类型、更新时间戳、大小、artifact_id 等。

### 4.6 可选：因子包归档接口

- `GET /factor_package/bundle?version={version}`
  - 返回指定版本的因子共享包归档（tar/zip）；
  - 用于离线备份与审计，非日常同步主通道。

---

## 5. AIstock-facing 四大 Catalog 设计

> 目标：在 RD-Agent 侧预先准备好 AIstock 所需的“研究资产视图”，避免 AIstock 直接扫描 workspace 或日志，只需导入约定好的 Catalog JSON 即可。

### 5.1 Factor Catalog（因子库）

- 导出脚本：`tools/export_aistock_factor_catalog.py`
- 顶层结构示例：

```json
{
  "version": "v1",
  "generated_at_utc": "...",
  "source": "rdagent_tools",
  "factors": [
    {
      "name": "RESI5",
      "source": "qlib_alpha158",
      "expression": "Resi($close, 5)/$close",
      "description_cn": "...",
      "variables": {},
      "tags": ["alpha158"],
      "region": "cn"
    },
    {
      "name": "rd_factor_001",
      "source": "rdagent_generated",
      "expression": null,
      "description_cn": "RD-Agent 生成因子描述",
      "variables": {"window": 5, "field": "close"},
      "tags": ["momentum"],
      "region": "cn",
      "impl_module": "rd_factors_lib.generated",
      "impl_func": "factor_rd_001",
      "impl_version": "1.0.8"
    }
  ]
}
```

- RD-Agent 侧职责：
  - 通过 `tools/export_alpha158_meta.py` 导出 Alpha158 全量因子定义，标记 `source="qlib_alpha158"`；
  - 汇总各 workspace 的 `factor_meta.json`（`source="rdagent_generated"`），透传 `description_cn` / `formula_hint` / `variables` / `tags` 等字段；
  - 合并去重并输出统一 `factor_catalog.json`；
- AIstock 侧职责：
  - 提供因子字典导入接口，将 `factor_catalog.json` 落地到本地 `factor_catalog` 表；
  - 因子库列表/详情页完全依赖该表与 Phase 2 artifacts，不再直接访问 RD-Agent workspace。

### 5.2 Strategy Catalog（策略库）

- 导出脚本：`tools/export_aistock_strategy_catalog.py`
- 顶层结构示例：

```json
{
  "version": "v1",
  "generated_at_utc": "...",
  "strategies": [
    {
      "strategy_id": "hash_of_template_and_args",
      "scenario": "QlibPlan2Scenario",
      "step_name": "train_model",
      "action": "model",
      "template_path": "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
      "data_config": {
        "market": "all",
        "benchmark": "000300.SH",
        "segments": {
          "train": ["2010-01-07", "2018-12-31"],
          "valid": ["2019-01-01", "2020-12-31"],
          "test":  ["2021-01-01", "2025-12-01"]
        }
      },
      "portfolio_config": {
        "class": "TopkDropoutStrategy",
        "topk": 50,
        "n_drop": 5,
        "fee": {"open_cost": 0.0005, "close_cost": 0.0015}
      },
      "model_config": {
        "class": "GeneralPTNN",
        "metric": "loss",
        "hyper_params": {"n_epochs": 100, "lr": 0.001}
      }
    }
  ]
}
```

- RD-Agent 侧职责：
  - 基于现有 YAML 模板与 registry 中的 workspace 记录，抽取“实际使用过的策略配置”；
  - 为每种唯一配置生成稳定 `strategy_id`（模板路径 + 参数 hash）；
  - 输出 `strategy_catalog.json`，不负责持久管理策略启/停状态（由 AIstock 管理）。
- AIstock 侧职责：
  - 导入 `strategy_catalog` 至本地 `strategy_catalog` 表；
  - 策略详情页展示数据配置、组合逻辑与模型配置。

### 5.3 Loop Catalog（回测记录库）

- 导出脚本：`tools/export_aistock_loop_catalog.py`
- 顶层结构示例：

```json
{
  "version": "v1",
  "generated_at_utc": "...",
  "loops": [
    {
      "task_run_id": "...",
      "loop_id": 0,
      "workspace_id": "exp_ws_001",
      "scenario": "QlibPlan2Scenario",
      "step_name": "train_model",
      "action": "model",
      "status": "success",
      "has_result": true,
      "strategy_id": "hash_of_template_and_args",
      "factor_names": ["RESI5", "WVMA5", "rd_factor_001"],
      "metrics": {
        "annual_return": 0.18,
        "max_drawdown": -0.12,
        "sharpe": 1.5,
        "IC": 0.06
      },
      "decision": true,
      "summary_texts": {
        "execution": "...",
        "value_feedback": "...",
        "shape_feedback": "..."
      },
      "paths": {
        "factor_meta": "factor_meta.json",
        "factor_perf": "factor_perf.json",
        "feedback": "feedback.json",
        "ret_curve": "ret_curve.png",
        "dd_curve": "dd_curve.png"
      }
    }
  ]
}
```

- RD-Agent 侧职责：
  - 遍历 registry 中所有 `has_result = 1` 的 loop；
  - 从 Phase 2 artifacts 中抽取因子组合与回测指标；
  - 关联 `strategy_catalog` 后输出 `loop_catalog.json`；
- AIstock 侧职责：
  - 导入 `loop_catalog` 至本地 `backtest_runs` 表；
  - 支持历史回测记录按策略/因子/指标筛选。

### 5.4 Model Catalog（模型库）

- 导出脚本：`tools/export_aistock_model_catalog.py`
- 顶层结构示例：

```json
{
  "version": "v1",
  "generated_at_utc": "...",
  "models": [
    {
      "task_run_id": "...",
      "loop_id": 0,
      "workspace_id": "exp_ws_001",
      "workspace_path": "...",
      "model_config": {"class": "GeneralPTNN", "metric": "loss", "hyper_params": {"n_epochs": 100}},
      "dataset_config": {"...": "..."},
      "model_artifacts": {
        "model.pkl": "...",
        "feature_importance.json": "..."
      }
    }
  ]
}
```

- 供 AIstock 在 Phase 3 中直接同步模型 registry 与 artifacts，为执行迁移与重训做准备。

---

## 6. AIstock 侧：Phase 2 接入与落库

### 6.1 RD-Agent 只读成果 API 交互流程

1. AIstock 后端实现“RD-Agent 同步任务”模块：
   - 定时任务或手动触发：
     - 调用 `GET /catalog/factors` / `/catalog/strategies` / `/catalog/loops` / `/catalog/models` / `/alpha158/meta` 等接口；
   - 将结果写入本地数据库：
     - 使用 upsert 策略，按主键（如 `(name, source)` / `strategy_id` / `(task_run_id, loop_id, workspace_id)`）覆盖更新。

2. 因子共享包版本对齐：
   - 从因子元数据或专门接口中读取当前生效 `impl_version`；
   - 将版本号记录在 AIstock 本地配置/DB 中；
   - 若 RD-Agent 与 AIstock 共享同一物理目录（如 `F:\Dev\rd-factors-lib`）：
     - AIstock 在虚拟环境中执行一次 `pip install -e` 即可获得因子包实现；
     - 版本号用于“对齐判断”，非下载逻辑；
   - 如未来需要支持多版本共存，可在 AIstock 内维护多套环境并通过 `impl_version` 选择（留待 Phase 3/4 增强）。

### 6.2 本地数据库结构扩展建议

- 因子表（如 `factor_registry`）：
  - 新增字段：`impl_module` / `impl_func` / `impl_version`；
- 策略表（如 `strategy_registry`）：
  - 可新增：`model_type`, `train_start`, `train_end`, `val_start`, `val_end`, `test_start`, `test_end`；
- 实验表（如 `loop_result`）：
  - 可新增：`model_run_id`（对应 RD-Agent/qlib/mlflow run 标识）、`factor_impl_version`（本次 loop 使用的因子包版本，可选）。

### 6.3 前端视图与字段合同

#### 6.3.1 策略详情页字段（核心）

> 字段语义与来源在 Phase 2 固定，后续仅做增量展示。

- **基本信息**：
  - `strategy_id`：策略主键，可由 `task_run` / `loop` / `workspace` 组合或 `strategy_catalog.strategy_id`；
  - `name`：策略名称（例如 workspace 名/别名）；
  - `shape` / `output_mode`：策略形态与输出模式；
  - `source_key`：原始来源 key（三元组 `task_run_id`、`loop_id`、`workspace_path` 等）；
  - `created_at`：策略创建时间；
  - `status`：策略状态（启用/禁用/待审核等，AIstock 内部管理）。

- **关联因子**：
  - `factor_names`：本策略依赖的因子名称列表（来自 `factor_perf.combinations[].factor_names` 或配置）；
  - `factor_source_summary`：因子来源概要（由 `factor_meta.factors[].source` 聚合）。

- **回测指标**：
  - `annual_return` / `max_drawdown` / `sharpe` / `win_rate`（可选）等，来自 `qlib_res.csv` 或 `experiment_summary`；

- **回测曲线**：
  - `equity_curve`：日期+净值数组（由 `ret.pkl` 解析）；
  - `equity_curve_chart_url`：`ret_curve.png` 的访问 URL；

- **反馈信息**：
  - `decision` / `limitations` / `code_critic` / `hypothesis` 等来自 `feedback.json`。

#### 6.3.2 因子详情页字段（核心）

- **基本信息**：
  - `factor_name` / `source` / `description_cn` / `formula_hint` / `tags` / `created_at` 等来自 `factor_meta.json`；
- **表现概览**：
  - `ic_mean` / `ic_ir` / `coverage`，来自 `factor_perf.factors[].metrics`；
- **窗口表现**：
  - `windows[]`、`annual_return` / `max_drawdown` / `sharpe`，来自 `factor_perf.factors[].windows[]`；
- **组合关系与关联实验**：
  - 反查 `factor_perf.combinations[].factor_names` 得到 `combination_names` 与组合层表现；
  - 结合 AIstock DB 映射到 `first_experiment_id`、`latest_experiments` 等视图。

> RD-Agent 在 Phase 2 中保证上述 JSON 结构与字段含义稳定，后续 Phase 3+ 的 UI 增强只在 AIstock 侧进行。

---

## 7. Phase 2 与数据服务层的关系

### 7.1 数据服务层角色

- 数据服务层是 AIstock 内部的实时/准实时数据访问抽象：
  - 提供 snapshot / history window / streaming / account / position 等视图；
  - 主要服务于“在线/模拟交易时的因子与策略执行”；
- 在 Phase 2 范围内，数据服务层必须至少满足本节与数据服务层详细设计文档中对 **离线/研究场景** 的全部硬性要求，为后续在线执行迁移提供完整基础，而非任何形式的“精简版”或 PoC。

### 7.2 Phase 2 对数据服务层的硬性要求（离线/研究场景）

- 严格参考 `2025-12-24_DataServiceLayer_Detail_Design_RD-Agent_AIstock.md`：
  - 在 Phase 2 内，AIstock 数据服务层必须在离线/研究场景中完成对以下接口的**生产级落地**：
    - 提供 `DataFrame(MultiIndex(datetime, instrument))` 为基础的 tabular 因子/行情矩阵视图（如 `get_history_window` 等），满足 REQ-DATASVC-P2-001；
    - 字段命名与索引规范与 RD-Agent/qlib 的离线视图（如 `daily_pv.h5`、Alpha158 因子）保持一致；
  - 这些接口在 Phase 2 阶段可以仅服务于：
    - 因子共享包（`rd_factors_lib`）在 AIstock 环境中的本地调试与验证；
    - 基于 tabular 因子矩阵的模型训练与回测对齐（不进入真实执行栈）；
  - 以上能力是 Phase 2 的**硬性范围内要求**，不得以“最小可用”或“PoC 实现”为理由做功能缩水或字段精简。

### 7.3 开发顺序建议（AIstock 侧）

- **推荐的节奏（不改变上述硬性范围）**：
  1. 完成 Phase 2 成果导入与 UI 验收：
     - 因子库/策略库/实验库视图打通；
     - 能从 RD-Agent 的 Catalog + artifacts 中稳定导入并展示成果；
  2. 在此基础上，按数据服务层详细设计文档与本节要求，完成数据服务层在离线/研究场景中的生产级实现：
     - 至少支持研究/调试场景（本地 qlib runtime + DataProvider 对接），满足 REQ-DATASVC-P2-001；
  3. 在 Phase 3 中基于已导入的成果与数据服务层接口，推进执行迁移、选股服务、策略预览等能力。

- Phase 2 期间，AIstock 可以**并行**启动数据服务层基础框架与内部集成，但真正“生产级 Phase 3 功能”上线前，应保证：
  - Phase 2 已在 AIstock 侧通过验收；
  - 数据服务层核心接口在生产环境可用并具备监控。

---

## 8. 开发任务拆分与当前进度（截至 2025-12-27）

### 8.1 RD-Agent 侧任务与状态

1. **抽取并实现 `write_loop_artifacts`**  
   - 状态：已完成。  
   - 已将原有 meta/summary/manifest/artifacts 写入逻辑迁移至 `rdagent.utils.artifacts_writer.write_loop_artifacts`，并在 `loop.py` 中替换为函数调用，保持行为兼容。

2. **实现 `factor_meta` / `factor_perf` / `feedback` 生成逻辑**  
   - 状态：已完成。  
   - `factor_meta.json` / `factor_perf.json` / `feedback.json` 按 v1 schema 生成并写入 workspace 根目录，同时登记 artifacts 与 artifact_files：
     - `factor_meta.json`：保证 `variables` 透传；
     - `factor_perf.json`：单因子描述统计 + 组合窗口 metrics；
     - `feedback.json`：在 execution/value/shape 基础上透传 `code_critic` 与 `limitations`（若存在）。

3. **实现回测图表生成与登记**  
   - 状态：已完成。  
   - 在 `_save_ret_plots` 中实现 `ret.pkl → ret_curve.png + dd_curve.png`；
   - 在 `write_loop_artifacts` 与 backfill 中统一调用，并以 `backtest_curve_image` 类型登记。

4. **更新 backfill 工具以支持 Phase 2 补录**  
   - 状态：已完成。  
   - `tools/backfill_registry_artifacts.py` 支持：
     - 为历史 loop 生成缺失的 `factor_meta` / `factor_perf` / `feedback` / 图表文件及 DB 记录；
     - `--mode backfill` / `--mode check` 两种模式，以及 `--all-task-runs`、`--only-experiment-workspace`、`--since-date`、`--max-loops`、`--overwrite-json`、`--cleanup-existing`、`--dry-run`、`--log-path` 等参数；
     - 详细使用方式与一键全量/增量补录流程见第 2.5 小节《Backfill 工具：历史成果补齐》。

5. **因子共享包骨架与写入逻辑（占位版）**  
   - 状态：已完成骨架与 stub 写入。  
   - 新建 `rd_factors_lib` 包并集成 `_sync_factor_impl_to_shared_lib`；
   - 为通过验收的因子写入占位 stub 与源码字符串，并在 `factor_meta` 中记录实现指针。

6. **只读成果 API 服务**  
   - 状态：已完成。  
   - FastAPI 应用与 CLI 启动入口已实现；
   - `/catalog/*`、`/alpha158/meta`、`/factors/{name}`、`/loops/{task_run_id}/{loop_id}/artifacts` 等接口可用。

7. **Alpha158 元数据导出与 Catalog 导出脚本**  
   - 状态：已完成。  
   - `tools/export_alpha158_meta.py` 导出 `alpha158_meta.json`；
   - `tools/export_aistock_factor_catalog.py` / `...strategy_catalog.py` / `...loop_catalog.py` / `...model_catalog.py` 导出四大 Catalog。

### 8.2 AIstock 侧任务与 Phase 2 验收要点

1. **成果同步与落库**  
   - 实现调用 RD-Agent 只读 API 的客户端，同步因子/策略/loop/Alpha158/模型元数据至本地 DB；
   - 支持全量同步（首次对接）与增量同步（定期任务）。

2. **DB Schema 与 API 扩展**  
   - 扩展因子、策略、实验等表，以承接 `impl_*`、窗口表现等字段；
   - 对前端提供统一 REST/GraphQL API，前端不直接访问 RD-Agent。

3. **前端视图实现**  
   - 因子库列表/详情页；
   - 策略/实验详情页（含 `feedback` 与回测曲线）。

4. **Phase 2 技术与功能验收**  
   - 任一新产生的 `action='model' AND has_result=1` loop：
     - `factor_meta.json`、`factor_perf.json`、`feedback.json`、`ret_curve.png` 至少存在；
     - 对应 artifacts / artifact_files 记录存在且可解析；
   在 AIstock UI：
     - 可按因子维度浏览与筛选；
     - 在实验详情页展示反馈摘要与回测曲线。

---

## 9. 硬性要求（REQ Checklist，按 2025-12-30 项目规范对齐）

> 本节列出与 Phase 2 直接相关的关键 REQ 条目，完整说明见：
> `docs/2025-12-30_Project_Development_Spec_RD-Agent_AIstock.md` 与
> `docs/2025-12-30_Phase1-3_Design_Update_RD-Agent_AIstock.md`。

- **REQ-FACTOR-P2-001：因子实现指针**  
  RD-Agent 在导出因子元数据和 factor_catalog 时，必须为每个可在 AIstock 侧复用的因子提供实现指针：
  `impl_module`, `impl_func`, `impl_version`，并保证与因子共享包（`rd_factors_lib`）中的实际实现和版本一致。

- **REQ-FACTOR-P2-002：因子表达式与公式提示**  
  对于 Alpha158 等基于表达式的因子，RD-Agent 必须在 `alpha158_meta.json` 与 factor_catalog 中完整记录表达式和必要的公式提示信息，禁止以“简化版因子列表”替代全部 Alpha 因子。  
  对于由 RD-Agent 因子演进流程产生、并输出到 factor catalog 的因子，`factor_meta.json` / factor_catalog 中必须至少包含：因子计算公式/表达式（`formula_hint` 或等价字段）、中文描述（`description_cn`）、可映射到数据服务层原始字段集合的变量列表（`variables`）、计算频率（`freq`）、时间对齐规则（`align`）以及数值稳定性/NaN 处理规则（`nan_policy`）。除非在任务/数据源级有更具体配置，Phase 2 统一约定 RD-Agent 因子默认按 A 股日线收盘价对齐计算（`freq = "1d"`, `align = "close"`），缺失值处理遵循数据服务层的统一缺失值处理规范（`nan_policy = "dataservice_default"`），不得以“默认约定不需要写出”为由在元数据与 catalog 中省略这些信息；这些字段必须通过 factor catalog 与只读成果 API 暴露给 AIstock 侧，作为因子在模拟盘与实盘环境中运行时的**唯一权威契约来源**。

- **REQ-FACTOR-P2-010：因子共享包结构**  
  必须存在独立的因子共享包 `rd_factors_lib`，并包含 `__init__.py`, `generated.py`, `VERSION` 等文件。该包是 RD-Agent × AIstock 之间因子实现复用的唯一官方入口，不得以临时脚本或散落源码替代。

- **REQ-FACTOR-P2-011：loop → 因子共享包写入**  
  RD-Agent 因子演进 loop 在通过回测与验证后，必须自动完成：
  将因子函数写入/更新 `rd_factors_lib.generated`，更新 `VERSION`，并把 `impl_module`, `impl_func`, `impl_version` 回写到对应 loop 的 `factor_meta.json` 与因子 catalog 记录中。

- **REQ-LOOP-P2-001：有结果 loop 的状态与指标完整性**  
  对于 `loops.has_result = 1` 的记录：
  `status` 字段不得为 `"unknown"`，至少一个关键指标（如 `ic_mean` 或 `ann_return`）必须非空；不得以“仅标记 has_result、不写指标”的方式简化实现。

- **REQ-LOOP-P2-002：历史回测成果的统一登记与补录**  
  RD-Agent 必须通过 backfill 工具确保历史与新产生的所有符合条件的 loop：
  - 在 `artifacts` / `artifact_files` 中完整登记 Phase 2 新增 JSON（`factor_meta` / `factor_perf` / `feedback`）与回测图表等关键文件；
  - 补齐缺失记录并保持幂等，不得留下“部分登记/部分缺失”的半成品状态。

- **REQ-LOOP-P2-003：回测关键指标的唯一事实表**  
  回测相关的关键绩效指标（如年化收益率、IC、最大回撤、Sharpe、胜率、波动率等）在 Phase 2 中**只允许在 loop 层以结构化字段形式存储**，由 `factor_perf.json` → `loop_catalog.json` → 只读成果 API 统一暴露。因子、策略、模型三类 catalog 不得重复保存这些回测指标；AIstock 若需按因子/策略/模型维度查看表现，必须通过与 loop 的关联关系（如 `strategy_id`、`model_id`、`factor_names`）在数据库侧进行 join 查询。

- **REQ-MODEL-P2-001：模型 catalog 字段齐全**  
  `model_catalog.json` 中每条记录必须至少包含：`task_run_id`, `loop_id`, `workspace_id`, `workspace_path`, `model_type`, `model_conf`, `dataset_conf`, `feature_names`, `window`, `freq` 以及模型文件相关 artifacts 的引用，不得裁剪字段。  
  对于 `action='model' AND has_result=1` 的 loop，其 workspace 中必须生成标准化的 `model_meta.json` 文件，至少包含：`model_type`, `model_conf`, `dataset_conf`, `feature_schema`，并在 `workspace_meta.json` / `experiment_summary.json` 中通过 `pointers.model_meta_path` 或 `files["model_meta.json"]` 进行引用，作为 AIstock 侧复用与重放该模型的唯一元数据入口。`model_conf` / `dataset_conf` 中必须给出能够在 AIstock 侧直接复用的训练数据集与运行接口必要信息（包括但不限于：数据源标识、基础字段格式与预处理前提、训练窗口与频率定义），以便 AIstock 在接入真实行情后无需额外反推即可构造与训练阶段一致的输入数据集，并驱动该模型在模拟盘与实盘环境中运行。

- **REQ-MODEL-P2-002：模型实例粒度与与 loop 的关联**  
  `model_catalog.json` 中的每条记录必须对应一份**已经完成训练的模型实例**，而不是抽象的“模型类型”。同一 `model_type` 在不同 loop、不同数据集或不同超参数下训练出的模型，必须在 catalog 中以多条独立记录体现。每条记录至少要包含：唯一的 `model_id`（可由 `task_run_id` / `loop_id` / `model_type` / `dataset_conf` 等组合生成稳定标识）、`model_type`, `task_run_id`, `loop_id`, `workspace_id`, `workspace_path` 以及可由 AIstock 侧直接复用的训练配置与训练/验证指标。AIstock 在模拟盘、实盘和策略预览等场景中，应通过选择具体的 `model_id` 来加载和复用模型，并通过与 loop 表的关联获取该模型在不同回测中的收益与风险表现。

- **REQ-STRATEGY-P2-001：策略 catalog 字段齐全**  
  `strategy_catalog.json` 中每条记录必须至少包含：`strategy_id`, `step_name`, `action`, `data_config`, `dataset_config`, `portfolio_config`, `backtest_config`, `model_config` 以及与特定 loop/模型的关联信息，不得将这些字段打包为不透明字符串或部分省略。

- **REQ-API-P2-001：只读成果 API 字段齐全**  
  `rdagent-results-api` 中：
  - `GET /catalog/factors` 与 `GET /factors/{name}` 必须暴露因子元数据中的实现指针、表达式、描述等关键字段；
  - `GET /catalog/models` / `/catalog/strategies` / `/catalog/loops` 返回的字段集合必须与对应 catalog JSON 完全一致，不得做字段精简或删除。

- **REQ-AISTOCK-P2-001：成果同步任务**  
  AIstock 后端必须实现“RD-Agent 成果同步任务”：
  定期调用 `results-api` 的 `/catalog/*` 与 `/factors/*` 等接口，将增量/全量结果按 upsert 策略写入本地数据库，并保证不丢失任何 catalog 中的官方字段。

- **REQ-AISTOCK-P2-002：因子共享包版本对齐**  
  对于带有 `impl_version` 的因子，AIstock 必须记录并对齐当前使用的 `rd_factors_lib` 版本，确保与 RD-Agent 侧版本一致或兼容，不得以“版本信息不敏感”为由忽略该字段。

- **REQ-AISTOCK-P2-003：本地 Schema 扩展**  
  AIstock 本地因子/策略/实验等表结构必须包含 `impl_module`, `impl_func`, `impl_version` 等字段，并与 RD-Agent 因子/catalog 中的字段一一对应，不得在落库时合并或丢弃这些信息。
 
- **REQ-AISTOCK-P2-004：因子元数据结构化落库与可运行性**  
  AIstock 在同步 factor catalog 时，必须将 RD-Agent 因子元数据中的关键运行契约字段以结构化方式落入本地因子表（如 `factor_registry`），而不得仅以 JSON 文件或不透明 blob 的形式保存。至少包括：`name`, `source`, `impl_module`, `impl_func`, `impl_version`, `description_cn`, `formula_hint`, `variables`, `freq`, `align`, `nan_policy`。AIstock 的因子运行引擎和策略执行模块必须以这些结构化字段作为构造 DataService 请求、加载因子实现与设定频率/对齐/NaN 策略的唯一入口，禁止绕开 catalog/DB 直接依赖临时脚本或硬编码路径。

---

### 9.1 RD-Agent Results API 规格（Phase 2 输出接口）

本小节定义 `rdagent-results-api` 在 Phase 2 中必须提供的 HTTP/REST 接口规格，作为 AIstock 侧开发只读集成的唯一权威参考。除非另有说明，所有接口均为 **只读**，字段集合必须与对应的 catalog JSON 完全一致，不得精简、重命名或删除。

- **GET `/health`**  
  用途：存活探针。  
  响应：`{"status": "ok"}`。

- **GET `/catalog/factors`**  
  用途：返回完整的因子 catalog 视图。  
  数据源：`RDagentDB/aistock/factor_catalog.json`。  
  响应结构：
  - `version`: 与 JSON 中一致（当前为 `"v1"`）；
  - `generated_at_utc`: 导出时间戳；
  - `source`: 固定为 `"rdagent_tools"`；
  - `factors`: 因子列表，**字段集合与 `factor_catalog.json` 中的单条因子记录完全一致**，至少包括：
    - `name`, `source`, `description_cn`, `formula_hint`, `variables`, `freq`, `align`, `nan_policy`,
      `created_at_utc`, `experiment_id`, `tags`, `impl_module`, `impl_func`, `impl_version` 等。  
  API 层不得对 `factors` 内部字段做任何过滤或重命名，必须原样透传 JSON 内容。

- **GET `/factors/{name}`**  
  用途：按名称返回单个因子的完整元数据。  
  语义：在 `factor_catalog.json` 的 `factors` 数组中查找 `name` 精确匹配的记录并返回；未找到时返回 404。  
  响应字段集合与 `/catalog/factors` 中单条因子记录完全一致。

- **GET `/catalog/strategies`**  
  用途：返回完整的策略 catalog 视图。  
  数据源：`RDagentDB/aistock/strategy_catalog.json`。  
  响应结构：
  - `version`, `generated_at_utc`, `source` 字段；
  - `strategies`: 策略列表，单条记录字段集合必须与 `strategy_catalog.json` 中一致，至少包括：
    - 标识与示例：`strategy_id`, `scenario`, `step_name`, `action`, `workspace_example`, `template_files`；
    - 配置：`data_config`, `dataset_config`, `portfolio_config`, `backtest_config`, `model_config`；
    - 衍生字段：`feature_list`, `market`, `instruments`, `freq`。  
  API 不得裁剪或重新打包这些字段，AIstock 应以该结构为策略表的直接建模依据。

- **GET `/catalog/models`**  
  用途：返回完整的模型 catalog 视图。  
  数据源：`RDagentDB/aistock/model_catalog.json`。  
  响应结构：
  - `version`, `generated_at_utc`, `source`；
  - `models`: 模型实例列表，单条记录至少包括：
    - 标识：`model_id`, `model_type`, `task_run_id`, `loop_id`, `workspace_id`, `workspace_path`；
    - 训练配置：`model_config`, `dataset_config`, `feature_schema`（如存在）；
    - 工件引用：`model_artifacts`（如 `{"mlruns": "mlruns", "model_files": [...]}`）。  
  API 必须原样透传 `model_catalog.json` 中的字段，不得新增/删除字段。

- **GET `/catalog/loops`**  
  用途：返回完整的 loop / 回测 catalog 视图，是 AIstock 获取回测 KPI 的唯一事实表来源。  
  数据源：`RDagentDB/aistock/loop_catalog.json`。  
  响应结构：
  - `version`, `generated_at_utc`, `source`；
  - `loops`: loop 列表，单条记录至少包括：
    - 基本信息：`task_run_id`, `loop_id`, `workspace_id`, `workspace_path`, `scenario`, `step_name`, `action`, `status`, `has_result`, `log_dir`；
    - 关联关系：`strategy_id`, `factor_names`；
    - 回测关键指标（仅在 loop 层结构化存储）：`annualized_return`, `max_drawdown`, `sharpe`, `ic`, `ic_ir`, `win_rate`, `metrics`；
    - 文本反馈：`decision`, `summary_texts.{execution,value_feedback,shape_feedback}`；
    - 资源路径：`paths`（如 `factor_meta`, `factor_perf`, `feedback`, `ret_curve`, `dd_curve`, `mlruns`, `model_files`）。  
  API 必须与 `loop_catalog.json` 字段一一对应，禁止对指标或路径字段做任何删减。

- **GET `/alpha158/meta`**  
  用途：返回 Alpha158 等外部因子库的 meta 信息，供 AIstock 建立统一的因子视图。  
  数据源：`RDagentDB/aistock/alpha158_meta.json`。  
  响应结构与 `alpha158_meta.json` 完全一致，API 不得对字段做变形。

- **GET `/loops/{task_run_id}/{loop_id}/artifacts`**  
  用途：按 `(task_run_id, loop_id)` 返回 registry 中登记的 artifacts 与 artifact_files 汇总视图，用于 AIstock 侧调试与回放。  
  数据源：`RDagentDB/registry.sqlite` 中的 `artifacts` 与 `artifact_files` 表。  
  响应结构：
  - `task_run_id`: string；
  - `loop_id`: int；
  - `artifacts`: 数组，每个元素至少包括：
    - `artifact_id`, `artifact_type`, `name`, `status`, `primary_flag`, `entry_path`, `summary`（由 `summary_json` 解析）、
    - `files`: 文件列表，字段包括 `file_id`, `path`, `sha256`, `size_bytes`, `mtime_utc`, `kind`。  
  API 必须完整暴露 registry 中关于该 loop 的 artifacts/文件元数据，不得过滤任何文件级字段。

> **实现约束**：
> - 所有 `/catalog/*` 与 `/alpha158/meta` 接口，必须直接从 `RDagentDB/aistock/*.json` 读取并原样返回；
> - `/loops/{task_run_id}/{loop_id}/artifacts` 必须直接查询 `registry.sqlite`，并将 `artifacts` 与 `artifact_files` 表中的信息完整映射到 HTTP 响应；
> - API 层禁止进行字段级“简化”或“重命名”，禁止仅返回部分子树或压缩为不透明 blob；
> - AIstock 侧可以仅依赖本节 API 规格和 Phase 2 输出的四大 catalog，即可完整实现数据同步、落库与模拟盘/实盘运行所需的全部元数据访问，无需再额外解析 RD-Agent 的内部目录结构或日志文件。

---

### 9.2 回测指标字段与增量数据一键更新

1. **回测指标字段的当前状态与演进**  
   - 在 Phase 2 中，loop 层已经结构化暴露了 `annualized_return`, `max_drawdown`, `sharpe`, `ic`, `ic_ir`, `win_rate`, `metrics` 等字段，并通过 `loop_catalog.json` 与 `GET /catalog/loops` 提供访问。  
   - 这些字段的取值上限取决于底层回测引擎在 `factor_perf.json` 中真正输出了哪些指标：
     - 若当前引擎仅提供年化收益，则 `annualized_return` 会被填充，其余字段可能为 `null`；
     - 将来如在 `factor_perf.json` 中增加 `max_drawdown` / `sharpe` / `ic` / `ic_ir` 等指标，导出脚本与 API 无需修改即可自动将这些新指标补充到 loop_catalog 与 `/catalog/loops` 的返回中。  
   - 因此，Phase 2 已经在 schema 层为这些关键 KPI 预留了稳定字段；指标本身的丰富程度归属于“回测引擎能力演进”范畴，而非本设计的缺失。

2. **AIstock 侧增量数据一键更新（推荐流程）**  
   Phase 2 推荐采用“RD-Agent 负责重建 catalog 视图 + AIstock 负责通过 API upsert 落库”的模式，以便在 UI 上实现“一键同步最新成果”：

   - **RD-Agent 侧：刷新四大 catalog 的固定命令**（由定时任务或人工触发）：

     ```bash
     # 1) 全量 backfill：确保所有有结果的 loop 已生成标准化 JSON
     python tools/backfill_registry_artifacts.py \
       --db /mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite \
       --all-task-runs \
       --mode backfill

     # 2) 因子 catalog
     python tools/export_aistock_factor_catalog.py \
       --registry-sqlite /mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite \
       --output /mnt/f/Dev/RD-Agent-main/RDagentDB/aistock/factor_catalog.json

     # 3) 策略 catalog
     python tools/export_aistock_strategy_catalog.py \
       --registry-sqlite /mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite \
       --output /mnt/f/Dev/RD-Agent-main/RDagentDB/aistock/strategy_catalog.json

     # 4) 模型 catalog
     python tools/export_aistock_model_catalog.py \
       --registry-sqlite /mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite \
       --output /mnt/f/Dev/RD-Agent-main/RDagentDB/aistock/model_catalog.json

     # 5) loop catalog
     python tools/export_aistock_loop_catalog.py \
       --registry-sqlite /mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite \
       --output /mnt/f/Dev/RD-Agent-main/RDagentDB/aistock/loop_catalog.json
     ```

     以上命令可被封装为 RD-Agent 侧的运维脚本或 CI 任务（例如按日、按任务结束后触发），保证 `RDagentDB/aistock/*.json` 始终反映最新成果。

   - **AIstock 侧：通过 Results API upsert 落库**  
     在 RD-Agent 侧完成 catalog 刷新后，AIstock 可通过一个 UI 按钮或定时任务调用以下 API，并按 upsert 策略写入本地数据库：

     1. `GET /catalog/factors` → upsert 到本地 `factor_registry` 表：
        - 以 `name` + `source` 作为主键或唯一索引；
        - 更新 `description_cn`, `formula_hint`, `variables`, `freq`, `align`, `nan_policy`, `impl_module`, `impl_func`, `impl_version` 等字段。

     2. `GET /catalog/strategies` → upsert 到本地 `strategies` 表：
        - 以 `strategy_id` 为主键；
        - 全量更新 `data_config`, `dataset_config`, `portfolio_config`, `backtest_config`, `model_config`, `feature_list`, `market`, `instruments`, `freq` 等字段。

     3. `GET /catalog/models` → upsert 到本地 `models` 表：
        - 以 `model_id` 为主键（模型实例粒度）；
        - 保存 `model_type`, `model_config`, `dataset_config`, `feature_schema`, `model_artifacts` 以及 `(task_run_id, loop_id, workspace_id, workspace_path)` 关联信息。

     4. `GET /catalog/loops` → upsert 到本地 `loops` 表（或 `loop_results`）：
        - 以 `(task_run_id, loop_id)` 为主键；
        - 保存基础字段、`strategy_id`, `factor_names` 以及所有回测 KPI：`annualized_return`, `max_drawdown`, `sharpe`, `ic`, `ic_ir`, `win_rate`, `metrics`。  
        - 后续按需扩展出二级关系表（如 `loop_factors`，将 `factor_names` 数组拆成多行），以支持多对多分析。

     5. 如需调试/回放特定 loop 的 artifacts，可在 UI 中按需调用：
        - `GET /loops/{task_run_id}/{loop_id}/artifacts`，将返回的 `artifacts[].files[]` 作为文件清单，用于展示下载链接或内部排查。

   在 UI 层，AIstock 可以将以上 API 调用与本地 upsert 过程封装为“一键同步”操作：
   - 用户点击“同步 RD-Agent 成果”按钮；
   - 后端依次调用 `/catalog/factors` → `/catalog/strategies` → `/catalog/models` → `/catalog/loops`，并写入本地库；
   - 同步完成后，UI 即可基于本地 DB 进行查询、预览与模拟盘运行，而无需直接访问 RD-Agent 的文件系统或内部目录结构。

---

## 10. 总结：Phase 2 在整体路线中的位置

- 本文件定义了 Phase 2 在 RD-Agent × AIstock 之间的：
  - 统一 artifacts 结构；
  - 因子/策略/loop/模型四大 Catalog；
  - 因子共享包与只读成果 API；
  - AIstock 导入与展示责任；
  - 与数据服务层的接口与边界；
  - **Registry（SQLite + Workspace 内 meta/summary/manifest）设计与 AIstock 对接方式（见附录 A/B）。**
- 完成 Phase 2 后：
  - RD-Agent 的研究成果已经以结构化资产形态完全打通到 AIstock；
  - AIstock 即使在 RD-Agent 暂不可用时，也能独立浏览与分析既有科研成果，并能在接入实际行情数据后，基于 Phase 2 输出的 catalog 与 workspace 元数据直接构造训练阶段一致的数据集，运行已训练好的模型与策略用于模拟盘验证与实盘交易，不依赖额外的人肉还原或线下补充信息；
  - 为 Phase 3 的执行迁移、选股服务与策略预览，以及完整数据服务层的接入，提供全部必要的输入前提。

---

## 附录 A：RD-Agent → AIstock SQLite Registry 设计（全文收录）

> 本附录完整收录并整合自原《2025-12-22_RD-Agent_AIstock_SQLite_Registry_Design_v1.md》。
> 该原始文档不再单独作为设计入口，仅作历史溯源使用。

### A.0 背景与目标

RD-Agent 负责策略研发与演进，AIstock 负责回测/模拟盘/实盘。为了让 AIstock **无需遍历扫描全部 workspace**，且能按 **任务（task）→ loop → workspace → artifact** 追溯与拉取成果，需要 RD-Agent 输出：

- Workspace 内：结构化元信息与成果清单（可搬运、可审计）。
- Workspace 外：一个可并发写入、可 SQL 查询的 Registry（SQLite）。

本方案目标：

- AIstock 只读 SQLite，就能：
  - 列出任务列表、任务状态；
  - 明确每个任务有哪些 loop、有无成果；
  - 对“有成果”的 loop 定位到对应 workspace；
  - 获取该 workspace 下的 artifacts（模型/特征/配置/报告）与文件清单（hash/size/path）；
  - 可按常用指标（<=10 列）进行筛选，其余指标在 `metrics_json`；
- RD-Agent 在并行机制下稳定写入（主进程写，WAL，busy_timeout，重试退避，进程内 lock 串行事务）。
- 对 RD-Agent 主流程影响最小：写失败可降级，不影响任务推进。

### A.1 已确认的设计选项

- **DB 放置路径（按最终实现）**：`<repo_root>/RDagentDB/registry.sqlite`（随项目目录迁移，且不提交仓库）。
- **task_run_id**：每次 CLI 启动生成 UUID；session resume 继续写同一 `task_run_id`。
- **并行写入策略**：只在主进程写 SQLite + WAL + busy_timeout + 重试退避 + 进程内 lock 串行事务。
- **artifact 粒度**：两级结构 `artifacts（组件）+ artifact_files（文件清单）`。
- **artifact 关联**：同时保存 `workspace_id` 与 `(task_run_id, loop_id)`。
- **成果判定（qlib）**：
  - model loop：`ret.pkl` 与 `qlib_res.csv` 均存在；
  - factor loop：`combined_factors_df.parquet` 存在。
- **Parquet 引擎一致性**：RD-Agent 与 AIstock 推荐统一使用 `pyarrow` 读取/写入 Parquet，以减少类型兼容性差异。
- **失败/跳过/异常**：写入 loop 记录（`status = failed/aborted/skip`）。
- **指标来源**：从 workspace 内已有文件（`qlib_res.csv`、`ret.pkl` 等）提取。
- **AIstock 消费方式**：现阶段 AIstock 只读 SQLite。
- **DB 目录随项目迁移**：`RDagentDB/` 在仓库下，已加入 `.gitignore`。

### A.2 并行机制与写入约束

#### A.2.1 并行发生在哪里

- 并行主要体现在：**多个 loop 并发推进**。
  - `LoopBase.run()` 会启动：
    - 1 个 `kickoff_loop()` 负责不断生成新的 loop index（0,1,2...）；
    - N 个 `execute_loop()` worker 从队列取 loop 执行 step；
- 同一 loop 的 step 仍是串行（`direct_exp_gen → coding → running → feedback`）。

#### A.2.2 subprocess 约束

- 当 `force_subproc=True` 时，`LoopBase._run_step()` 会使用 `ProcessPoolExecutor` 在子进程执行 step。
- **强约束**：Registry 写入必须发生在主进程（step 返回后），不得在子进程写 DB。

#### A.2.3 写入位置

- 在 `LoopBase._run_step()` 中：
  - step 开始：写入/更新 task、loop、workspace 的“运行中”状态；
  - step 结束：写入/更新 step 结果；若为 `running` 且成功则写入成果摘要与 artifacts。

### A.3 数据对象定义

#### A.3.1 Task（任务）

- 一次 `rdagent.app.cli fin_quant ...` 启动对应一个 `task_run_id`。
- session resume（从 session folder load）应复用同一个 `task_run_id`。

#### A.3.2 Loop（迭代轮次）

- 以 `loop_id`（整数，从 0 递增）标识。
- 每个 loop 的 `action` 为 `factor` 或 `model`（qlib 场景）。

#### A.3.3 Workspace

- 当前 workspace 目录名为 UUID。
- 每个 experiment 通常包含：
  - `experiment_workspace`：实际运行与产物落盘位置；
  - `sub_workspace_list`：候选代码注入/开发 workspace。

#### A.3.4 Artifact（组件级）

- Artifact 是 AIstock 消费的“组件”，而不是单个文件：
  - `model`：模型产物；
  - `feature_set`：特征/因子集合；
  - `config_snapshot`：配置快照；
  - `report`：研究证据（指标汇总、曲线、图表等）。

Artifact 再关联文件清单（`artifact_files`），用于校验/复制/同步。

### A.4 SQLite Schema（v1）

#### A.4.1 通用约定

- 所有时间统一存 `*_at_utc`（ISO 8601 字符串或 unix timestamp，推荐 ISO 8601）。
- 主键：
  - `task_runs.task_run_id`（TEXT）；
  - `loops(task_run_id, loop_id)`（复合）；
  - `workspaces.workspace_id`（TEXT）；
  - `artifacts.artifact_id`（TEXT，uuid）；
  - `artifact_files.file_id`（TEXT，uuid）。

#### A.4.2 表：task_runs

- 目的：任务级索引与审计。

字段建议：

- `task_run_id` TEXT PRIMARY KEY；
- `scenario` TEXT；
- `status` TEXT  -- running/success/failed/aborted；
- `created_at_utc` TEXT；
- `updated_at_utc` TEXT；
- `git_sha` TEXT；
- `rdagent_version` TEXT；
- `log_trace_path` TEXT；
- `params_json` TEXT  -- 市场、数据源、segments 等。

#### A.4.3 表：loops

- 目的：AIstock 直接判断“哪个 loop 有成果”。

字段建议：

- `task_run_id` TEXT NOT NULL；
- `loop_id` INTEGER NOT NULL；
- `action` TEXT  -- factor/model；
- `status` TEXT  -- running/success/failed/aborted/skip；
- `has_result` INTEGER DEFAULT 0  -- 0/1；
- `best_workspace_id` TEXT；
- `started_at_utc` TEXT；
- `ended_at_utc` TEXT；
- `error_type` TEXT；
- `error_message` TEXT；

指标列（<=10）：

- `ic_mean` REAL；
- `rank_ic_mean` REAL；
- `ann_return` REAL；
- `mdd` REAL；
- `turnover` REAL；
- `multi_score` REAL；

扩展：

- `metrics_json` TEXT  -- 全量指标 JSON（可空）。

约束：`PRIMARY KEY (task_run_id, loop_id)`。

#### A.4.4 表：workspaces

- 目的：明确每个 workspace 的出处与入口指针。

字段建议：

- `workspace_id` TEXT PRIMARY KEY；
- `task_run_id` TEXT NOT NULL；
- `loop_id` INTEGER；
- `workspace_role` TEXT  -- experiment_workspace/sub_workspace；
- `experiment_type` TEXT  -- qlib_factor/qlib_model（或 factor/model）；
- `step_name` TEXT  -- 最后一次更新来自哪个 step；
- `status` TEXT  -- running/success/failed/aborted；
- `workspace_path` TEXT NOT NULL；
- `meta_path` TEXT；
- `summary_path` TEXT；
- `manifest_path` TEXT；
- `created_at_utc` TEXT；
- `updated_at_utc` TEXT。

索引建议：

- `idx_workspaces_task_loop` ON `(task_run_id, loop_id)`；
- `idx_workspaces_role` ON `workspace_role`。

#### A.4.5 表：artifacts

- 目的：组件级产物索引（模型/特征/配置/报告）。

字段建议：

- `artifact_id` TEXT PRIMARY KEY；
- `task_run_id` TEXT NOT NULL；
- `loop_id` INTEGER；
- `workspace_id` TEXT NOT NULL；
- `artifact_type` TEXT  -- model/feature_set/config_snapshot/report；
- `name` TEXT；
- `version` TEXT；
- `status` TEXT  -- present/missing；
- `primary` INTEGER DEFAULT 0；
- `summary_json` TEXT；
- `entry_path` TEXT；
- `created_at_utc` TEXT；
- `updated_at_utc` TEXT。

索引建议：

- `idx_artifacts_task_loop` ON `(task_run_id, loop_id)`；
- `idx_artifacts_workspace` ON `workspace_id`。

#### A.4.6 表：artifact_files

- 目的：文件级指纹（校验/同步/复制）。

字段建议：

- `file_id` TEXT PRIMARY KEY；
- `artifact_id` TEXT NOT NULL；
- `workspace_id` TEXT NOT NULL；
- `path` TEXT NOT NULL  -- 相对 workspace 的路径；
- `sha256` TEXT；
- `size_bytes` INTEGER；
- `mtime_utc` TEXT；
- `kind` TEXT  -- model/config/data/report。

索引建议：`idx_artifact_files_artifact` ON `artifact_id`。

### A.5 Workspace 内文件：可搬运真相源

即便 AIstock 只读 DB，仍建议在 workspace 内落文件，作为“可搬运/可审计”的真相源：

- `workspace_meta.json`：小而稳定（归属关系、状态、指针）；
- `experiment_summary.json`：详细成果（指标、关键产物路径、因子清单摘要）；
- `manifest.json`：对 AIstock 的 release 契约入口。

基本要求（作为 Phase 2 的默认契约能力，而非“最低实现”）：

- 只要 DB 可用，AIstock 即可仅通过只读 DB 消费 RD-Agent 成果；
- 当 DB 不可用/迁移时，AIstock 必须能够通过 workspace 内的 meta/summary/manifest 文件重建所需视图。

### A.6 写入时机与算法（主进程 hook）

- 写入点：在 `LoopBase._run_step()` 主进程逻辑中，step 开始/结束处；
- 关键字段提取：`task_run_id`、`loop_id`、`step_name`、`action`、`workspace_id/workspace_path` 等；
- 成果判定与 `loops.has_result` 更新规则：
  - `action='model'` 且 `ret.pkl` 与 `qlib_res.csv` 同时存在；
  - `action='factor'` 且 `combined_factors_df.parquet` 存在；
- artifacts 写入：
  - model loop：`model` / `config_snapshot` / `report`（纳入 `ret_schema.*`/`signals.*` 等文件）；
  - factor loop：`feature_set` / `config_snapshot`。

### A.7 并发写入与降级策略

- SQLite 配置：`journal_mode=WAL`、`busy_timeout`、短事务；
- 进程内 Lock 串行执行写入；
- 捕获 `database is locked` 时指数退避重试；
- **降级策略**：写入失败不得中断主流程，只记录 warning 日志和可选的 `registry_write_failures.log`。

### A.8 AIstock 消费 SQL 示例

（示例 SQL 已原样保留，供 AIstock 侧直接参考，略。）

### A.9 风险清单与上游同步评估

- 锁冲突、子进程写 DB、DB 损坏、跨平台路径等风险及缓解措施；
- 代码侵入范围与 cherry-pick 策略：
  - 新增 registry 模块；
  - 在 `LoopBase._run_step()` 中调用独立 hook 函数；
  - 其余逻辑封装在新模块，降低与上游冲突面。

---

## 附录 B：AIstock 对接 RD-Agent Registry 指南（全文收录）

> 本附录完整收录并整合自原《2025-12-22_AIstock_RD-Agent_Registry_Integration_Guide.md》。
> 该原始文档不再单独作为设计入口，仅作历史溯源使用。

### B.0 目的

AIstock 侧**无需扫描 RD-Agent workspace 目录**，仅通过读取 RD-Agent 输出的：

- 项目内 SQLite：`RDagentDB/registry.sqlite`；
- 每个 workspace 内的 `manifest.json` / `experiment_summary.json`；

即可实现：

- 按 task_run → loop → workspace → artifacts 定位成果；
- 按关键指标（例如 `mdd`）筛选候选；
- 获取可复制的文件清单（相对路径 + hash/size/mtime）。

约束：

- AIstock 只读 SQLite 与 workspace 文件；
- `RDagentDB/` 随项目迁移，不提交 git。

### B.1 文件位置与运行环境

- 默认 DB 路径：`<repo_root>/RDagentDB/registry.sqlite`；
- 可选环境变量：
  - `RD_AGENT_DISABLE_SQLITE_REGISTRY=1`（不建议在对接阶段启用）；
  - `RD_AGENT_REGISTRY_DB_PATH`（必须位于 `<repo_root>` 之下，否则被忽略）；
- Parquet 读取推荐：统一使用 `pyarrow`。

### B.2 SQLite Schema 与推荐消费链路

- 5 张表：`task_runs` / `loops` / `workspaces` / `artifacts` / `artifact_files`；
- 推荐闭环链路（AIstock 侧在 Phase 2 内应至少实现这一整条链路的只读消费）：

`task_runs` → `loops.best_workspace_id` → `workspaces.manifest_path` → 读取 workspace 内 `manifest.json`。

### B.3 推荐 SQL 消费流程

1. 选择 task_run：按时间倒序列出最近 N 个任务；
2. 列出该 task_run 下所有 loops，并筛选 `has_result=1` 且符合指标门槛的 loop；
3. 通过 `best_workspace_id` 查询 `workspaces`，拿到 `workspace_path` 与 `manifest_path` 等入口；
4. 拼接 `manifest_abs_path = workspace_path + "/" + manifest_path`，读取 `manifest.json`；
5. 如需完整文件清单，再通过 `artifacts` / `artifact_files` 取组件和文件指纹。

所有 SQL 示例在原文中已给出，本附录维持原有结构，AIstock 可直接套用。

### B.4 Artifact 类型与约定

- `model` loop：
  - 成果判定：`ret.pkl` 与 `qlib_res.csv` 同时存在；
  - 典型 artifacts：`report` / `model` / `config_snapshot`。
- `factor` loop：
  - 成果判定：`combined_factors_df.parquet` 存在；
  - 典型 artifacts：`feature_set` / `config_snapshot`。
- `status=missing`：即使产物缺失，也会写入 `artifacts.status="missing"` 供诊断。

### B.5 AIstock 实现建议（基于本设计的推荐落地方式）

- 实现一个只读服务层：
  - `RegistryReader(db_path)`：封装对 5 张表的查询；
  - `WorkspaceManifestReader(workspace_path, manifest_path)`：封装 manifest 解析与基本校验；
- DB 作为权威索引，manifest 作为可搬运真相源与 debug 入口；上述能力属于 Phase 2 范围内 AIstock 侧的标准实现方式，而不是“可选的简化版”。

### B.6 成果文件契约（qlib_res / ret / combined_factors / mlruns / conf*.yaml / signals.*）

本节在原文中给出了各类关键文件的：

- 生成方式（来自 RD-Agent 的 `read_exp_res.py` / runner）；
- 文件格式（CSV / pickle / Parquet / JSON / 目录）；
- 读取方式与字段约束；
- 与 Registry 列（尤其是 `loops` 指标列）的映射关系；

Phase 2 的主文中已经定义了 AIstock 需要落库的字段合同，本附录保留这些底层约定，供实现与排障时参考：

- `qlib_res.csv`：两列形式的指标表，index 为指标名，value 为数值；
- `ret.pkl` / `ret_schema.parquet` / `ret_schema.json`：回测曲线/报告对象及稳定 schema 表；
- `combined_factors_df.parquet`：MultiIndex index、二级 MultiIndex 列（level-0 固定为 `feature`）；
- `mlruns/`：MLflow 跟踪目录，建议整体复制；
- `conf*.yaml`：配置快照；
- `signals.parquet` / `signals.json`：强固定 schema 的可执行信号表（Phase 3 执行迁移时的主入口）。

### B.7 “成果包含内容列表”（按 artifact_type）

- `report`：`qlib_res.csv` + 建议性附加文件（`ret.*` / `signals.*` 等）；
- `feature_set`：`combined_factors_df.parquet`；
- `model`：`mlruns/` 目录；
- `config_snapshot`：`conf*.yaml` 集合。

> 自本附录起，AIstock 研发/运维在实现或排障 Registry 对接逻辑时，只需参考 Phase 2 本文件与附录 A/B，无需再额外查阅独立的 Registry 设计/集成文档。

