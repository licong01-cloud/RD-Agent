# 2025-12-14 备忘录：因子演进 debug（static_factors rolling 补齐 + retry 触发条件修复 + 数据集集成验证）

## 1. 背景与目标
- **目标**：解决 Qlib 因子演进在若干轮次中反复失败（常见为缺列/输出全 NaN），并确保 `daily_basic.h5`、`moneyflow.h5` 的字段能稳定进入因子研发与执行链路。
- **关键约束**：因子实现必须以 `static_factors_schema.csv/json` 为列名白名单，禁止 LLM 编造字段。

## 2. 今日诊断出的核心问题

### 2.1 rolling 资金流派生列未落地（schema 中为 0）
- **现象**：`git_ignore_folder/factor_implementation_source_data/static_factors_schema.csv` 中 `_5d/_20d` 派生列数量为 0。
- **直接原因**：`tools/generate_static_factors_bundle.py` 的 `_derive_moneyflow_features()` 在 `moneyflow.h5` 原始表缺少分档买卖额/量字段时会返回空表，导致 rolling 派生列未被 append 到最终合并结果。
- **根因解释**：不同导出快照中 `moneyflow_raw` 字段覆盖可能不一致；当缺少 `mf_sm_buy_amt/mf_md_buy_amt/...` 等分档字段时，无法从 raw 直接构造 `mf_total_net_amt_{w}d`。

### 2.2 debug 数据目录缺少 schema 文件
- **现象**：
  - `git_ignore_folder/factor_implementation_source_data` 中 `static_factors_schema.csv/json` 存在且包含 rolling。
  - 但 `git_ignore_folder/factor_implementation_source_data_debug` 中 **缺少** `static_factors_schema.csv/json`。
- **风险**：debug 模式下 LLM 获取不到 schema 描述，更容易猜列名导致缺列失败。

### 2.3 因子 auto-repair / retry 在历史 run 中未触发
- **证据**：对 `log/2025-12-13_18-07-27-156205` 全目录 4529 个 `.pkl` 做二进制扫描：
  - `FactorAutoRepair hits: 0`
- **结论**：该 run 中 factor runner 的 auto-repair 分支 **从未执行**。
- **原因**：`_is_factor_autorepair_eligible()` 触发签名过窄，很多“缺列/全 NaN/空输出”等失败文案无法命中。

## 3. 今日完成的代码修改（文件级）

### 3.1 生成 static_factors：moneyflow 派生列 fallback + schema debug 同步
- **文件**：`tools/generate_static_factors_bundle.py`
- **修改点 A（fallback 派生）**：
  - 优先从 `df_mf_raw` 派生。
  - 若派生为空，则从已加载的可选表（尤其是 `combined_static_factors.parquet`）中筛选 `mf_` 列作为 fallback，再尝试派生。
  - 目标：即使 raw moneyflow 缺分档列，只要 combined 表包含分档列，就能生成 rolling。
- **修改点 B（debug schema 写出）**：
  - 将 `static_factors_schema.csv/json` 同时写入：
    - `git_ignore_folder/factor_implementation_source_data/`
    - `git_ignore_folder/factor_implementation_source_data_debug/`

### 3.2 因子执行数据目录：优先复制 repo 生成的 parquet+schema
- **文件**：`rdagent/scenarios/qlib/experiment/utils.py`
- **函数**：`generate_data_folder_from_qlib()`
- **修改**：
  - 优先复制：`git_ignore_folder/factor_implementation_source_data/static_factors.parquet` + `static_factors_schema.csv/json`
  - fallback：`/mnt/c/.../AIstock/factors/combined_static_factors.parquet`
  - 同样对 `data_folder` 与 `data_folder_debug` 生效。

### 3.3 retry（auto-repair）触发签名扩展
- **文件**：`rdagent/scenarios/qlib/developer/factor_runner.py`
- **函数**：`_is_factor_autorepair_eligible()`
- **修改**：扩展触发关键字，覆盖常见失败：
  - 缺列：`missing columns / columns are missing / not in index / column not found / does not exist`
  - 空输出/全 NaN：`empty dataframe / all nan / all values are nan`
  - 文件类：`no such file or directory`

## 4. 今日完成的验证（证据链）

### 4.1 rolling 列生成验证
- **重新生成后结果**：
  - `schema cols: 66`
  - `rolling cols: 12`
  - rolling 列示例：`mf_total_net_amt_5d`, `mf_total_net_amt_20d`, `mf_main_net_amt_ratio_5d`, ...

### 4.2 数值正确性 spot-check
- 对 `000001.SZ`：
  - `mf_total_net_amt_5d` 与按 instrument 分组的 5 日滚动和对比：`max_abs_diff = 0.0`

### 4.3 NaN 比例检查
- 派生列 NaN 率约 0.51~0.55：
  - 主要由 `min_periods=window`（前置期）与 ratio 分母为 0/缺失导致，属于可解释范围。

### 4.4 执行目录静态验证（不启动任务）
- `factor_implementation_source_data`：schema 存在且 rolling=12。
- `factor_implementation_source_data_debug`：发现 schema 缺失（已通过生成脚本补写修复）。

### 4.5 历史 run retry 验证
- 对 `log/2025-12-13_18-07-27-156205` 全量 `.pkl` 扫描：`FactorAutoRepair hits: 0`，说明当时未触发 auto-repair。

## 5. 可复现的检查脚本片段（供后续使用）

### 5.1 检查 schema 是否包含 rolling 列
```bash
python - <<'PY'
import pandas as pd
p = 'git_ignore_folder/factor_implementation_source_data/static_factors_schema.csv'
df = pd.read_csv(p)
cols = df['name'].astype(str).tolist()
hit = [c for c in cols if ('_5d' in c or '_20d' in c)]
print('schema cols:', len(cols))
print('rolling cols:', len(hit))
print(hit)
PY
```

### 5.2 spot-check：rolling 正确性
```bash
python - <<'PY'
import pandas as pd
pq = 'git_ignore_folder/factor_implementation_source_data/static_factors.parquet'
df = pd.read_parquet(pq).sort_index()
inst = df.index.get_level_values('instrument').unique()[0]
sub = df.xs(inst, level='instrument')
calc5 = sub['mf_total_net_amt'].astype('float64').rolling(5, min_periods=5).sum()
got5  = sub['mf_total_net_amt_5d'].astype('float64')
mx = (calc5 - got5).abs().dropna().max()
print('instrument:', inst)
print('max_abs_diff:', float(mx) if mx==mx else mx)
PY
```

### 5.3 扫描日志中是否触发 auto-repair
```bash
python - <<'PY'
from pathlib import Path
root = Path('log/2025-12-13_18-07-27-156205')
pkls = list(root.rglob('*.pkl'))
print('pkl count:', len(pkls))
print('FactorAutoRepair hits:', sum(b'FactorAutoRepair' in p.read_bytes() for p in pkls))
PY
```

## 6. 下一步建议（待办）
- 运行一轮新的 `fin_factor` 或 `fin_quant` 任务，确认：
  - 新 log 中 `static_factors_schema.csv/json` 在 prompt 侧被引用（可用二进制扫描命中次数作证据）。
  - 生成的 `factor.py` 实际 `read_parquet('static_factors.parquet')` 并使用 `mf_*/db_*` 列。
  - 在出现缺列/空输出时，`FactorAutoRepair` 能在新 log 中出现（证明 retry 生效）。

## 7. 备注（环境/工具）
- WSL 曾出现 `Wsl/Service/CreateInstance/E_UNEXPECTED`（疑似磁盘空间不足导致），释放空间后恢复。
- WSL `rdagent-gpu` 环境确认可用：Python 3.10.19, pandas 2.3.3, tables 3.10.1。
