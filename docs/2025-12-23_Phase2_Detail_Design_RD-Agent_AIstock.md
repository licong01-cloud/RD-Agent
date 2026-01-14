# Phase 2 详细设计：统一 artifacts 函数 + 因子/策略成果结构化暴露

> 目标：在 Phase 1 的基础上，抽象出统一的结果写入函数，将因子与策略的关键成果结构化为标准化 artifacts（factor_meta/factor_perf/feedback/图表），并通过 registry 暴露给 AIstock，构建完整的“因子库/策略库/回测视图”能力。

---

## 1. 范围与目标

### 1.1 范围

- RD-Agent 侧：
  - 抽象并实现统一的 `write_loop_artifacts(...)` 函数；
  - 在 model loop 完成后，集中调用该函数生成所有约定的 artifacts；
  - 增加三类新的核心 JSON artifacts：
    - `factor_meta.json`
    - `factor_perf.json`
    - `feedback.json`
  - 统一生成/登记回测图表文件（如 `ret_curve.png`, `dd_curve.png`）。

- AIstock 侧：
  - 在 Phase 1 的基础上，增加：
    - 因子库视图（基于 factor_meta/factor_perf）；
    - 策略结果详细视图（结合 feedback 与图表）。

### 1.2 目标

- 从“只能看到策略层回测结果（signals/ret）”升级为：“能系统性查看所有演进因子及其表现，以及每次实验的反馈与评估”。
- 保持与 Phase 1 的兼容：
  - 不修改现有 artifacts 类型与含义；
  - 新增 artifacts 通过向后兼容方式接入。

---

## 2. RD-Agent 侧：统一 artifacts 函数设计

### 2.1 函数签名与职责

建议在 RD-Agent 内部新增一个模块（示意）：

- 文件：`rdagent/utils/artifacts_writer.py`
- 主要函数：

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

职责：

1. 检查 workspace 下已有文件：
   - `ret.pkl`, `qlib_res.csv`, `signals.*`, `ret_schema.*`, `combined_factors_df.parquet`, `conf*.yaml`, `mlruns/` 等；
2. 生成/刷新：
   - `workspace_meta.json`、`experiment_summary.json`、`manifest.json`；
   - `factor_meta.json`、`factor_perf.json`、`feedback.json`；
   - 回测图表文件（`ret_curve.png`, `dd_curve.png`）；
3. 在 `artifacts` / `artifact_files` 中登记所有上述文件；
4. 保证对同一 loop 重复调用是幂等的（不会生成重复 DB 记录，也不会破坏已有记录）。

### 2.2 与现有 loop 写入逻辑的整合

- 在 `rdagent/utils/workflow/loop.py` 中，当前已经存在一段写 meta/summary/manifest 与 artifacts 的逻辑。
- Phase 2 计划：
  1. 将该段逻辑几乎“原样搬迁”到 `write_loop_artifacts(...)` 中；
  2. 在 `loop.py` 中仅保留一行调用：

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

  3. 确保行为与 Phase 1 完全一致（即使还未开启新 artifacts 逻辑）。

- 完成上述“重构不改行为”后，再在 `write_loop_artifacts` 内增量加入新 artifacts 生成逻辑。

---

## 3. 新增 JSON artifacts 设计

### 3.1 factor_meta.json

**目的**：记录本次实验中“参与回测的因子”的元信息，便于 AIstock 构建因子库与解释性视图。

**建议结构（简化版）：**

```json
{
  "factors": [
    {
      "name": "VolAdj_Momentum_10D",
      "source": "rdagent_generated",  // 或 "external", "alpha158" 等
      "description_cn": "10日价格动量按过去20日收益波动率进行风险调整",
      "formula_hint": "(close_t/close_{t-10}-1) / annualized_std(returns_{t-20..t-1})",
      "created_at_utc": "2025-12-23T11:00:00Z",
      "experiment_id": "<task_run_id>/<loop_id>",
      "tags": ["momentum", "vol_adjusted", "daily"]
    }
  ]
}
```

**生成来源：**

- 可以从以下信息中综合提取：
  - RD-Agent 内部的 hypothesis/justification 文本；
  - 因子函数名（如 `calculate_VolAdj_Momentum_10D`）；
  - factor.py 中的注释（如存在）；
  - 静态因子表 schema（`static_factors_schema.json`）。

**实现要点：**

- Phase 2 不要求 100% 精准自然语言说明，但需提供：
  - 稳定的因子名；
  - 基本来源类型（rdagent_generated/external/alpha158）；
  - 至少一行简要说明（可由 LLM/规则生成）。

### 3.2 factor_perf.json

**目的**：记录单因子与组合因子在回测中的表现摘要，支持 AIstock 按因子维度筛选与排序。

**建议结构（示意）：**

```json
{
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
          "sharpe": 1.40
        }
      ]
    }
  ]
}
```

**生成来源：**

- 单因子层面的 IC/RankIC 等：
  - 可由 RD-Agent 在获得因子矩阵与 label 后，做轻量级统计；
- 组合层面的收益/回撤/Sharpe：
  - 可从 Qlib 回测结果（`qlib_res.csv` 或其他记录）中解析，或在 `read_exp_res.py` 中输出更多字段供使用。

**实现要点：**

- Phase 2 可以先仅支持：
  - `factors` 部分的少量指标（如 ic_mean/ic_ir/coverage）；
  - `combinations` 部分只记录整体组合的年化收益/最大回撤/Sharpe。
- 后续 Phase 可逐步丰富字段。

### 3.3 feedback.json

**目的**：以结构化形式记录本轮实验的关键反馈与评估，便于 AIstock UI 直接展示，无需解析日志。

**建议结构（示意）：**

```json
{
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

#### 3.3.1 HypothesisFeedback → feedback.json 字段映射规则

RD-Agent 现有的反馈对象主要包括：

- `ExperimentFeedback` / `HypothesisFeedback`（位于 `rdagent.core.proposal`）；
- Qlib 场景下的 `Qlib*Experiment2Feedback`，会根据实验结果生成 `HypothesisFeedback`，其典型字段包含：
  - `observations`
  - `hypothesis_evaluation`
  - `new_hypothesis`
  - `reason`
  - `decision`

Phase 2 中约定以下**稳定映射关系**：

- **决策层面：**
  - `HypothesisFeedback.decision` → `feedback.json.decision`
- **假设本身：**
  - `HypothesisFeedback.new_hypothesis`（如有）优先映射到 `feedback.json.hypothesis`
  - 若 `new_hypothesis` 为空，则回退使用 `exp.hypothesis.hypothesis` 作为 `feedback.json.hypothesis`
- **总结与评价：**
  - `HypothesisFeedback.observations`
    - 主要描述本轮实验的结果观察与数据表现；
    - 映射到 `feedback.json.summary.execution`
  - `HypothesisFeedback.hypothesis_evaluation`
    - 对当前假设是否被支持/证伪的评价；
    - 映射到 `feedback.json.summary.value_feedback`
  - `HypothesisFeedback.reason`
    - 作为对上述评价的补充说明；
    - 可拼接到 `feedback.json.summary.shape_feedback` 或附加在 `value_feedback` 后（实现可按需细化，但语义应保持为“评价/推理补充”）。
- **局限性与代码审阅：**
  - 若 Qlib 侧的反馈生成 prompt 额外产出“局限/告警”和“代码审阅意见”（如 `limitations` / `code_critic` 数组），则：
    - 映射到 `feedback.json.summary.limitations[]`、`feedback.json.summary.code_critic[]`；
  - 若当前版本未显式提供，则这两个字段可以为空数组，等待后续扩展。

> 说明：
>
> - `feedback.json` 是**单次实验级别的快照**，描述“这一轮实验对策略/因子的观察和评价”；
> - 因子库/策略库中的长期描述（例如因子定义、公式提示）来自 `factor_meta.json` 等文件；
> - AIstock 在导入时：
>   - 策略/实验详情页直接消费 `feedback.json` 中的 `decision`、`hypothesis` 与 `summary.*` 字段；
>   - 因子库可按需汇总多轮实验的 `summary.limitations` 等字段，用于展示“长期局限与风险提示”。

**生成来源：**

- CoSTEER 的多轮 feedback 对象（`HypothesisFeedback` 等）；
- RD-Agent 的日志消息（经规则/LLM 提炼补充到 `limitations`/`code_critic`）。

**实现要点：**

- Phase 2 目标是“有结构化骨架”，不要求自然语言完美；
- 保证 JSON 可解析，字段含义清晰，便于 AIstock 前端直接使用；
- `write_loop_artifacts` 内部应通过统一的 helper（如 `_build_feedback_dict(...)`），根据上述映射规则从 `HypothesisFeedback` / 实验对象中组装 payload，随后再决定是否落盘与登记 artifacts（支持幂等重跑）。

---

## 4. 回测图表文件生成

### 4.1 生成规则

- 输入：`ret.pkl`（通常是一个含收益曲线的 DataFrame 或 Series）。
- 输出：至少一张 PNG 图：
  - `ret_curve.png`：净值或收益曲线；
  - 可选：`dd_curve.png`：回撤曲线。

### 4.2 技术实现建议

- 依赖 matplotlib 或 plotly（静态导出）：

```python
import matplotlib.pyplot as plt
import pandas as pd


def save_ret_plots(ret_pkl_path: Path, out_dir: Path) -> None:
    df = pd.read_pickle(ret_pkl_path)
    # 假设 df 有一列 "portfolio" 或类似字段
    series = df.iloc[:, 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    series.cumsum().plot(ax=ax)
    ax.set_title("Cumulative Return")
    ax.grid(True)
    (out_dir / "ret_curve.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "ret_curve.png", bbox_inches="tight")
    plt.close(fig)
```

- `write_loop_artifacts` 在检测到 `ret.pkl` 存在时，调用该函数生成图表文件并登记到 artifacts 中：
  - `artifact_type = 'backtest_curve_image'` 等。

---

## 5. AIstock 侧：因子库/策略视图消费逻辑

### 5.1 因子库视图

- 数据来源：
  - `factor_meta.json` + `factor_perf.json`；
- 功能：
  - 按因子名/标签/来源筛选；
  - 展示单因子指标（ic_mean/ic_ir/coverage 等）；
  - 跳转到关联的回测结果与策略视图。

### 5.2 策略/实验详细视图

- 数据来源：
  - `experiment_summary.json` + `feedback.json` + 回测图表 + Phase 1 的 `qlib_res.csv` / `ret.pkl`；
- 功能：
  - 展示本轮实验的：
    - 决策（Decision）与简要说明；
    - 回测指标（年化收益/最大回撤/Sharpe 等）；
    - 回测曲线图；
    - 关键限制与告警。

---

### 5.3 最小成果归档到 AIstock 数据库

1. **扩展 registry 访问封装**
   - 在 Phase 1 基础上，增加对 `factor_meta` / `factor_perf` / `feedback` / 图表的查询与读取能力。

2. **实现因子库与策略详细视图**
   - 前端与后端联动：
     - 新增“因子库”列表页与详情页；
     - 在“策略/实验详情”页上展示 feedback 与图表。

---

## 6. 职责划分与与数据服务层关系（Phase 2）

### 6.1 Phase 2 RD-Agent / AIstock 职责划分

- **RD-Agent 侧（Phase 2 主责任方）**：
  - **统一 artifacts 写入**：
    - 抽取并实现 `write_loop_artifacts(...)`，将 `loop.py` 中现有 meta/summary/manifest/artifacts 写入逻辑迁移进来；
    - 在 `action == "model" AND has_result == 1` 的 loop 上，保证统一生成：
      - `workspace_meta.json`、`experiment_summary.json`、`manifest.json`；
      - `factor_meta.json`、`factor_perf.json`、`feedback.json`；
      - 回测图表文件（`ret_curve.png`，可选 `dd_curve.png`）；
    - 在 `artifacts` / `artifact_files` 中登记所有上述文件，且保证幂等。
  - **覆盖前端字段合同所需数据**：
    - 确保“策略详情页字段表（5.4）”与“因子详情页字段表（5.5）”中列出的字段，在 Phase 2 结束时：
      - 要么可以直接从上述 JSON / CSV / PKL 中获取；
      - 要么可以通过 registry.sqlite 中的结构化字段组合得到。
  - **历史 workspace 补录工具**：
    - 扩展并使用 `tools/backfill_registry_artifacts.py`，尽可能为已有 `action='model' AND has_result=1` 的历史 loop：
      - 生成缺失的 `factor_meta.json` / `factor_perf.json` / `feedback.json` / 图表文件；
      - 更新 `artifacts` / `artifact_files` 记录，使 AIstock 可以统一消费。

- **AIstock 侧（Phase 2 支撑方）**：
  - **导入/落库能力**：
    - 基于 Phase 1 的 registry 读取封装，扩展一个“RD-Agent 实验导入任务”（示意）：
      - 支持 **全量导入**：从某个起始日期/loop id 开始，扫描 registry.sqlite + workspace artifacts，将符合条件的实验全部导入 AIstock DB（因子库 + 策略库 + 回测结果）；
      - 支持 **增量导入**：基于 `created_at`/`updated_at` 或自增 id，每次只导入新增部分；
      - 记录导入状态（成功/失败/跳过原因），便于重复执行与排错。
  - **DB Schema 与 API**：
    - 结合第 5 章字段合同，在 AIstock 侧：
      - 设计或扩展因子表、策略表、实验表及其关系表；
      - 暴露对应的查询 API（REST/GraphQL 等），前端只需要调用 AIstock API，而无需直接读 RD-Agent 文件。
  - **前端视图实现**：
    - 在 UI 上落地：
      - 因子库列表页 + 详情页（消费 `factor_meta` / `factor_perf` 对应字段）；
      - 策略/实验详情页（消费 `experiment_summary` / `feedback` / 回测曲线等）。

> 小结：Phase 2 中，**RD-Agent 负责“生产并标准化成果”**，**AIstock 负责“导入并展示成果”**。只要双方遵守 artifacts 结构与字段合同，后续 Phase 3+ 的归档与多策略扩展均可在 AIstock 侧迭代完成，而不再频繁修改 RD-Agent。

### 6.2 与数据服务层的关系与并行开发

- **数据服务层的角色**：
  - 数据服务层是 AIstock 内部的实时/准实时数据访问抽象（snapshot/stream/history/account 等），
  - 主要服务于“在线/模拟交易时的因子与策略执行”，而非离线回测 artifacts 的生成。

- **Phase 2 与数据服务层的边界**：
  - Phase 2 的重点仍在于：
    - 从 RD-Agent 侧产出结构化 artifacts；
    - 在 AIstock 中建立“因子库/策略库 + 回测视图”的本地化存储与展示能力；
  - 对于数据服务层：
    - **Phase 2 不要求 RD-Agent 立即切换为通过数据服务层获取行情**；
    - 但应在顶层架构与函数签名层面，已经明确：
      - 未来 RD-Agent 产出的策略/因子，在实盘/模拟交易时，
      - 必须通过数据服务层暴露的 DataFrame / view，而不是直接读取 h5/bin 文件。

- **AIstock 侧在 Phase 2 可并行开展的数据服务层工作**：
  - 在不依赖 RD-Agent 改动的前提下，AIstock 可以开始：
    - 搭建数据服务层的基础框架与模块划分；
    - 对接现有行情源/账户系统，初步实现 snapshot / history window / streaming 接口；
    - 在内部或现有策略引擎中试用数据服务层，验证性能与易用性。
  - 与 RD-Agent 的对接（即“从 AIstock 因子/策略库选择某个策略，挂接到数据服务层并下发到交易执行”）
    - 规划为 Phase 3+ 的工作，在 Phase 3 详细设计文档中单独展开。

> 小结：Phase 2 期间，**AIstock 完全可以并行启动数据服务层的设计与基础实现**，且不阻塞 RD-Agent 侧的 artifacts 改造；真正的“在线策略接入数据服务层”集成工作，可在 Phase 3 结合归档与多策略管理一起落地。

---

## 7. 规范 / 规则 / 限制（Phase 2）

### 6.1 JSON Schema 版本管理

- 为 `factor_meta.json` / `factor_perf.json` / `feedback.json` 设计简化版 schema：
  - 至少包含 `version` 字段（如 `"v1"`），便于后续扩展时做向后兼容处理；
  - 未来新增字段时，不破坏现有解析逻辑。

### 6.2 幂等与容错

- `write_loop_artifacts` 必须幂等：
  - 重复执行不会生成重复 DB 记录；
  - 写文件时可以覆盖旧版本，或在文件名中加入稳定标识。
- 遇到生成某个 JSON/图表失败时：
  - 不应阻断整个回写流程；
  - 在 experiment_summary 或 feedback 中记录 warning，便于后续排查。

---

## 7. 面向 AIstock 的三大 Catalog（Factor/Strategy/Loop）

> 目标：在 RD-Agent 侧预先准备好 AIstock 所需的“研究资产视图”，避免 AIstock 自行扫描 workspace / 日志，只需导入约定好的 Catalog 文件即可。

### 7.1 Factor Catalog（因子库）

- 由 RD-Agent 通过 tools 脚本生成统一的因子字典文件，例如 `aistock_factor_catalog.json`。
- 顶层结构（示意）：

  ```json
  {
    "version": "v1",
    "generated_at_utc": "...",
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
        "region": "cn"
      }
    ]
  }
  ```

- RD-Agent 侧职责：
  - 通过 qlib 的 Alpha158 handler / 配置导出 Alpha158 全量因子定义（表达式 + 短名），标记为 `source="qlib_alpha158"`；
  - 汇总所有 workspace 中的 `factor_meta.json`（`source="rdagent_generated"`），透传 `description_cn` / `formula_hint` / `variables` / `tags` 等字段；
  - 去重并输出统一的 `factor_catalog` 文件。
- AIstock 侧职责：
  - 提供因子字典导入接口，读取 `factor_catalog` 并落地到内部 `factor_catalog` 表；
  - 在因子库列表页 / 详情页仅依赖该表与 Phase2 artifacts，不直接访问 RD-Agent workspace。

### 7.2 Strategy Catalog（策略库）

- 由 RD-Agent 通过 tools 脚本扫描 registry 与 YAML 模板，生成 `aistock_strategy_catalog.json`。
- 顶层结构（示意）：

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
            "test": ["2021-01-01", "2025-12-01"]
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
  - 为每种唯一配置生成稳定的 `strategy_id`（模板路径 + 参数 hash）；
  - 输出 `strategy_catalog`，不负责持久管理策略状态（启用/禁用等由 AIstock 管理）。
- AIstock 侧职责：
  - 导入 `strategy_catalog` 到本地 `strategy_catalog` 表；
  - 在策略详情页，通过 `strategy_id` 展示数据配置、组合逻辑与模型配置；
  - 后续 Phase3/4 可在此基础上做策略推荐与相似检索。

### 7.3 Loop / Backtest Catalog（回测记录库）

- 由 RD-Agent 通过 tools 脚本读取 registry.sqlite + workspace Phase2 artifacts，生成 `aistock_loop_catalog.json`。
- 顶层结构（示意）：

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
        "metrics": {"annual_return": 0.18, "max_drawdown": -0.12, "sharpe": 1.5, "IC": 0.06, "...": "..."},
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
  - 遍历 registry 中所有 `has_result=True` 的 `experiment_workspace`；
  - 从对应 workspace 的 Phase2 artifacts 中抽取因子组合与完整回测指标；
  - 与 `strategy_catalog` 关联后输出 `loop_catalog`，只记录**最终有结果的 loop**，不记录中间迭代过程。
- AIstock 侧职责：
  - 导入 `loop_catalog` 到本地 `backtest_runs` 表；
  - 在 UI 中展示“历史回测记录”，按策略/因子/指标筛选，用作未来组合设计与推荐的参考。

---

## 8. 开发计划与任务拆分

### 8.1 RD-Agent 侧任务

1. **抽取并实现 `write_loop_artifacts`（已完成）**
   - 已将 `loop.py` 中原有 meta/summary/manifest/artifacts 写入逻辑迁移到 `rdagent.utils.artifacts_writer.write_loop_artifacts`；
   - 在 `loop.py` 中替换为函数调用，并保持行为兼容。

2. **实现 factor_meta/factor_perf/feedback 生成逻辑（已完成）**
   - `factor_meta.json` / `factor_perf.json` / `feedback.json` 已在 `write_loop_artifacts` 中按 v1 schema 生成并写入 workspace 根目录，同时登记对应 artifacts 与 artifact_files；
   - `factor_meta.json`：
     - 确保 `factors[*].variables` 透传 FactorTask / 日志中的结构，不做字段精简；
   - `factor_perf.json`：
     - `factors[*]` 包含基于 `combined_factors_df.parquet` 计算的单因子描述统计（coverage/mean/std/分位数等）；
     - `combinations[*].windows[*].metrics` 嵌入完整 metrics 字典；
   - `feedback.json`：
     - 在保持原有 execution/value/shape 字段的基础上，透传 `code_critic` 与 `limitations`（若上游对象已挂载）。

3. **实现回测图表生成（已完成）**
   - 在 `artifacts_writer._save_ret_plots` 中实现 `ret.pkl → ret_curve.png + dd_curve.png` 的图表生成逻辑；
   - 在 `write_loop_artifacts` 与 backfill 中调用 `_save_ret_plots`，将图表以 `artifact_type='backtest_curve_image'` 的形式登记到 `artifacts` / `artifact_files`。

4. **更新 backfill 工具（已完成 Phase2 补录）**
   - `tools/backfill_registry_artifacts.py` 已支持：
     - 为历史 workspace 生成缺失的 `factor_meta.json` / `factor_perf.json` / `feedback.json` / 回测图表；
     - 在 DB 中登记对应 artifacts 与 artifact_files 记录；
     - 对因子实验复用在线逻辑，从 `combined_factors_df.parquet` 计算单因子描述统计，保证历史补录与在线结构一致。

5. **新增：AIstock-facing Catalog 导出脚本（待实现）**

   - `tools/export_alpha158_meta.py`
     - 基于 qlib Alpha158 handler / 配置，导出 Alpha158 全量因子（表达式 + 短名）为内部中间文件；
   - `tools/export_aistock_factor_catalog.py`
     - 汇总 Alpha158 中间文件与各 workspace 的 `factor_meta.json`，生成统一的 `factor_catalog`；
   - `tools/export_aistock_strategy_catalog.py`
     - 扫描 YAML 模板与 registry 中的实际使用记录，生成 `strategy_catalog`；
   - `tools/export_aistock_loop_catalog.py`（或统一合并为 `export_aistock_catalogs.py`）
     - 基于 registry.sqlite 与 Phase2 artifacts 汇总所有 `has_result=True` 的 loop，生成 `loop_catalog`。

### 5.4 策略详情页字段表（前端字段合同）

> 说明：以下为 Phase 2 建议实现的“长期稳定字段集”。后续 Phase 3/4 仅在这些基础上做增量展示，不应改变字段含义。

| 字段分组 | 前端字段名（示意） | 含义说明 | 数据来源文件 | 示例字段/路径 |
|----------|--------------------|----------|--------------|---------------|
| 基本信息 | `strategy_id` | RD-Agent 在 AIstock 中的策略主键（可用 task_run/loop/workspace 组装） | registry.sqlite | `task_runs.id` + `loops.id` + `workspaces.id` |
| 基本信息 | `name` | 策略名称（通常为 RD-Agent 生成的 workspace 名/别名） | `workspace_meta.json` | `name` 或 `workspace_path` |
| 基本信息 | `shape` | 策略形态（如 `portfolio`/`signal`） | AIstock 策略表 / Experiment 配置 | `shape` |
| 基本信息 | `output_mode` | 输出模式（如 `target_weight` 等） | AIstock 策略表 / Experiment 配置 | `output_mode` |
| 基本信息 | `source_key` | 原始来源 key（`task_run/loop/workspace` 三元组） | registry.sqlite | `task_runs.id`、`loops.id`、`workspaces.workspace_path` |
| 基本信息 | `created_at` | 策略创建时间 | registry.sqlite / `workspace_meta.json` | `task_runs.created_at` 或 `created_at_utc` |
| 基本信息 | `status` | 策略状态（启用/禁用/待审核等） | AIstock 策略表 | `status` |
| 关联因子 | `factor_names` | 本策略依赖的因子名称列表 | `factor_perf.json` / Experiment 配置 | `combinations[].factor_names` |
| 关联因子 | `factor_source_summary` | 因子来源概要（RD-Agent/Alpha/外部） | `factor_meta.json` | 聚合 `factors[].source` |
| 回测指标 | `annual_return` | 主测试区间年化收益 | `qlib_res.csv` / `experiment_summary.json` | 如 `excess_return` 或自定义列 |
| 回测指标 | `max_drawdown` | 主测试区间最大回撤 | 同上 | `max_drawdown` |
| 回测指标 | `sharpe` | Sharpe 比率 | 同上 | `sharpe` |
| 回测指标 | `win_rate`（可选） | 盈利期占比/胜率 | `qlib_res.csv` 或衍生计算 | 自定义 |
| 回测曲线 | `equity_curve` | 净值/收益曲线数据（日期+数值数组） | `ret.pkl` | 解析为 `[{date, nav}, ...]` |
| 回测曲线 | `equity_curve_chart_url` | 曲线图静态图片访问路径 | `ret_curve.png` | 文件路径经 AIstock 侧映射为 URL |
| 反馈信息 | `decision` | CoSTEER 对本策略的最终决策（是否推荐） | `feedback.json` | `decision` |
| 反馈信息 | `limitations` | 关键限制/告警列表 | `feedback.json` | `summary.limitations[]` |
| 反馈信息 | `code_critic` | 重要代码审阅意见 | `feedback.json` | `summary.code_critic[]` |
| 反馈信息 | `hypothesis` | 策略/因子的核心假设摘要 | `feedback.json` | `hypothesis` |

> 注：具体列名可由 AIstock 自行调整，但应保持字段语义和来源不变，以保证 Phase 3+ 的归档与前端增强无需回到 RD-Agent 侧改动。

### 5.5 因子详情页字段表（前端字段合同）

> 说明：因子详情页主要面向“因子库浏览与分析”，字段来自 `factor_meta.json` / `factor_perf.json` 及 Experiment 信息。

| 字段分组 | 前端字段名（示意） | 含义说明 | 数据来源文件 | 示例字段/路径 |
|----------|--------------------|----------|--------------|---------------|
| 基本信息 | `factor_name` | 因子名称（唯一标识） | `factor_meta.json` | `factors[].name` |
| 基本信息 | `source` | 因子来源（`rdagent_generated`/`alpha_library`/`external_manual` 等） | `factor_meta.json` | `factors[].source` |
| 基本信息 | `description_cn` | 因子的中文描述/含义 | `factor_meta.json` | `factors[].description_cn` |
| 基本信息 | `formula_hint` | 公式提示/实现思路（人类可读） | `factor_meta.json` | `factors[].formula_hint` |
| 基本信息 | `tags` | 标签列表（如 `momentum`/`vol_adjusted` 等） | `factor_meta.json` | `factors[].tags` |
| 基本信息 | `created_at` | 因子首次产生时间 | `factor_meta.json` | `factors[].created_at_utc` |
| 表现概览 | `ic_mean` | 全局 IC 均值 | `factor_perf.json` | `factors[].metrics.ic_mean` |
| 表现概览 | `ic_ir` | IC Information Ratio | `factor_perf.json` | `factors[].metrics.ic_ir` |
| 表现概览 | `coverage` | 覆盖率 | `factor_perf.json` | `factors[].metrics.coverage` |
| 窗口表现 | `windows[]` | 各时间窗口上的表现列表 | `factor_perf.json` | `factors[].windows[]` |
| 窗口表现 | `window_name` | 窗口名称（如 `test_2021_2025`） | 同上 | `windows[].name` |
| 窗口表现 | `annual_return` | 该窗口年化收益 | 同上 | `windows[].annual_return` |
| 窗口表现 | `max_drawdown` | 该窗口最大回撤 | 同上 | `windows[].max_drawdown` |
| 窗口表现 | `sharpe` | 该窗口 Sharpe | 同上 | `windows[].sharpe` |
| 组合关系 | `combination_names` | 包含该因子的组合名称列表 | `factor_perf.json` | 反查 `combinations[].factor_names` |
| 组合关系 | `combination_perf_summary` | 该因子参与的组合表现概要 | `factor_perf.json` | 选取若干 `combinations[]` 的指标 |
| 关联实验 | `first_experiment_id` | 首次引入本因子的 experiment ID | `factor_meta.json` / AIstock DB | `experiment_id` |
| 关联实验 | `latest_experiments` | 最近若干涉及本因子的实验列表 | AIstock DB（通过 factors↔experiments 关联） | 查询生成 |

> 注：AIstock 可以在因子详情页中只展示概览信息（ic_mean/ic_ir/coverage + 某个代表窗口），也可以提供“展开全部窗口/组合详情”的交互，但底层字段都应来自上述标准结构。RD-Agent 侧在 Phase 2 中确保这些 JSON 的结构与字段稳定，后续 Phase 3+ 的扩展仅在 AIstock 侧进行。

3. **实现回测图表生成**
   - 实现 `save_ret_plots` 或类似工具函数；
   - 在 `write_loop_artifacts` 中调用并登记图表 artifacts。

4. **更新 backfill 工具**
   - 使 `tools/backfill_registry_artifacts.py` 能为历史 loop 生成缺失的 factor_meta/factor_perf/feedback/图表文件与 DB 记录。

### 5.3 最小成果归档到 AIstock 数据库

1. **扩展 registry 访问封装**
   - 在 Phase 1 基础上，增加对 `factor_meta` / `factor_perf` / `feedback` / 图表的查询与读取能力。

2. **实现因子库与策略详细视图**
   - 前端与后端联动：
     - 新增“因子库”列表页与详情页；
     - 在“策略/实验详情”页上展示 feedback 与图表。

---

## 9. Phase 2 验收标准

1. **技术验收**：
   - 对任一新产生的 `action='model' AND has_result=1` loop：
     - `factor_meta.json`、`factor_perf.json`、`feedback.json`、`ret_curve.png` 至少存在；
     - 对应 artifacts 与 artifact_files 记录存在且可解析。

2. **AIstock 功能验收**：
   - 在 AIstock UI 中：
     - 能按因子维度浏览与筛选（来自 factor_meta + factor_perf）；
     - 在实验详情页展示反馈摘要与回测曲线；
   - 所有这些功能只依赖 registry 与 artifacts，不依赖额外日志。

3. **兼容性验收**：
   - Phase 1 的脚本与流程在 Phase 2 环境下仍然可用；
   - 旧数据在不执行 backfill 的情况下，AIstock 至少能继续消费 Phase 1 的成果；
   - 执行 backfill 后，旧数据也具备 Phase 2 的新增 artifacts 能力。
