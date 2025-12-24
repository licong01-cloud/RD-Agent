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

**生成来源：**

- CoSTEER 的多轮 feedback 对象（execution/value/shape/code critic 等）；
- RD-Agent 的日志消息（经规则/LLM 提炼）。

**实现要点：**

- Phase 2 目标是“有结构化骨架”，不要求自然语言完美；
- 保证 JSON 可解析，字段含义清晰，便于 AIstock 前端直接使用。

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

## 6. 规范 / 规则 / 限制（Phase 2）

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

## 7. 开发计划与任务拆分

### 7.1 RD-Agent 侧任务

1. **抽取并实现 `write_loop_artifacts`（不改行为版）**
   - 将 `loop.py` 里现有 meta/summary/manifest/artifacts 写入逻辑迁移到新函数；
   - 在 `loop.py` 中替换为函数调用，确保回归测试通过。

2. **实现 factor_meta/factor_perf/feedback 生成逻辑**
   - 在 `write_loop_artifacts` 中：
     - 解析当前 workspace 的因子列表与回测结果；
     - 生成 JSON 并写入 workspace 根目录；
     - 在 `artifacts` / `artifact_files` 中登记。

3. **实现回测图表生成**
   - 实现 `save_ret_plots` 或类似工具函数；
   - 在 `write_loop_artifacts` 中调用并登记图表 artifacts。

4. **更新 backfill 工具**
   - 使 `tools/backfill_registry_artifacts.py` 能为历史 loop 生成缺失的 factor_meta/factor_perf/feedback/图表文件与 DB 记录。

### 7.2 AIstock 侧任务

1. **扩展 registry 访问封装**
   - 在 Phase 1 基础上，增加对 `factor_meta` / `factor_perf` / `feedback` / 图表的查询与读取能力。

2. **实现因子库与策略详细视图**
   - 前端与后端联动：
     - 新增“因子库”列表页与详情页；
     - 在“策略/实验详情”页上展示 feedback 与图表。

---

## 8. Phase 2 验收标准

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
