# SOTA 因子原理备忘录

## 1. 目的与适用范围

本备忘录用于澄清 RD-Agent 在 **Qlib 因子研发（Factor Loop）** 中所说的 “SOTA 因子（组合）” 的准确含义、来源、筛选/去重逻辑、在每轮回测中的使用方式，以及如何在 workspace 中定位与查看。

本备忘录只描述**代码层面**已经实现的逻辑，不包含任何“理想设计”或“未来规划”。


## 2. SOTA 因子在系统中的准确含义

在当前代码实现中，“SOTA 因子（SOTA_factor / SOTA_feature）”并不是一个固定文件或静态列表。

- **SOTA 因子 = 来自 `based_experiments` 中历史 `QlibFactorExperiment` 的因子产物集合**
- 其核心来源是 `rdagent/scenarios/qlib/developer/utils.py::process_factor_data()` 对历史 experiment 的执行结果（`result.h5` 被读取为 DataFrame）进行收集、拼接。

换句话说：

- SOTA 因子是“历史轮次被保留并作为基底沿用”的因子集合（动态累积）。
- 系统不会在 repo 内维护一个名为 `sota_factors.csv` / `sota_factors.parquet` 的静态名单文件。


## 3. SOTA 因子从哪里来（数据流/对象流）

### 3.1 入口：`QlibFactorRunner.develop()`

主要逻辑位于：

- `rdagent/scenarios/qlib/developer/factor_runner.py::QlibFactorRunner.develop()`

在 `develop()` 中，如果 `exp.based_experiments` 存在，会先从中筛出所有历史的 `QlibFactorExperiment`：

- `sota_factor_experiments_list = [base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)]`

当列表长度大于 1 时，触发 SOTA 因子处理：

- `SOTA_factor = process_factor_data(sota_factor_experiments_list)`

### 3.2 SOTA 因子的构建方式：`process_factor_data()`

实现位于：

- `rdagent/scenarios/qlib/developer/utils.py::process_factor_data()`

其核心行为：

- 遍历传入的一个或多个 `QlibFactorExperiment`
- 对每个 experiment：
  - 只处理 “有 sub_tasks 的 experiment”（表示是研发任务产物）
  - 依赖 `prop_dev_feedback`（仅对执行成功反馈项进行收集）
  - 对每个成功的 `implementation` 执行 `implementation.execute("All")` 拿到 DataFrame
  - 将所有成功产物 `pd.concat(..., axis=1)` 拼成一个多列的因子面板

因此，SOTA 因子并不是“从某个磁盘目录直接扫出来”，而是 **从历史 experiment 对象的实现产物中收集出来**。


## 4. SOTA 因子如何筛选/去重

### 4.1 核心筛选：只保留“能成功执行 + 输出合法”的因子

在 SOTA 侧：

- 只有 `implementation.execute("All")` 返回的 df 非空、且索引 level 包含 `datetime`，并且不是分钟级（代码中用 `pd.Timedelta(minutes=1)` 过滤）时，才会进入 SOTA 拼接集合。

### 4.2 新因子去重：按与 SOTA 的相关性过滤

当 `SOTA_factor` 存在时，新因子会先执行去重：

- `new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)`

相关性计算方式：

- 对 `concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)`
- 按 `datetime` 分组，在每个日期内对每一对列计算相关系数
- 然后对日期维度取均值
- 最终得到每个新因子的最大相关系数 `IC_max`

过滤阈值：

- 若新因子与任一 SOTA 因子的相关性均值 **>= 0.995**，则视为“几乎重复”，会被删除
- 若所有新因子都被判定为重复，则为了保证本轮仍有产出，代码会“放宽”并保留全部新因子（避免新因素一列不剩）

备注：这里变量名使用了 `IC`，但语义是“相关性去重阈值”，并非回测中常见的横截面 IC 指标。


## 5. 每轮回测中 SOTA 因子如何参与（与 Alpha158 的关系）

### 5.1 每轮回测默认只跑一次

在 `QlibFactorRunner.develop()` 中，每轮最终只会运行一次 `exp.experiment_workspace.execute(...)`（即一次 `qrun conf_*.yaml`）。

### 5.2 每轮都包含 Alpha158

无论 baseline 还是 combined 模式，配置中都会包含 Alpha158 作为基础特征：

- baseline：`conf_baseline.yaml` 的 dataset handler 为 `Alpha158`
- combined：`conf_combined_factors_dynamic.yaml` / `conf_combined_factors_sota_model.yaml` 使用 `CombinedAlpha158DynamicFactorsLoader`：
  - 读取 Alpha158 的静态 feature
  - 再将动态因子文件 `combined_factors_df.parquet` join 进来

因此实际输入特征 = **Alpha158 + 动态因子（SOTA + new）**。

### 5.3 动态因子文件的唯一权威入口

在回测前，系统会将本轮动态因子写到当前 experiment workspace：

- `combined_factors_df.parquet`

该文件包含：

- SOTA 因子（基于 `based_experiments` 拼接得到）
- 本轮新因子（去重后）

并作为回测配置 `dynamic_path` 被加载。


## 6. 在哪里可以查到 SOTA 因子（最实用的定位方式）

### 6.1 查“本轮用于回测的动态因子全集”（推荐）

查看当前轮 experiment workspace 下的：

- `combined_factors_df.parquet`

你可以通过读取该 parquet 来查看列名（即因子名集合）以及因子值。

### 6.2 追溯“哪些来自 SOTA，哪些来自 new”（需要结合对象/日志）

当前实现不会在 parquet 的列上直接标注来源（SOTA/new）。

如果需要区分来源：

- “new 因子”：来自当前 `exp.sub_workspace_list` 的 `factor.py` 输出列
- “SOTA 因子”：来自 `exp.based_experiments` 中 `QlibFactorExperiment` 的 `factor.py` 输出列

实践中通常要通过：

- runner 日志中打印的 experiment 信息
- 或在 workspace 内逐个查看历史 experiment 的产物（见下条）

来完成追溯。

### 6.3 查历史保留因子的原始实现（factor.py）与产物（result.h5）

SOTA 因子来自历史 `QlibFactorExperiment`，每个因子实现本质上由 `FactorFBWorkspace` 执行，产物是：

- `factor.py`（实现代码）
- `result.h5`（输出 DataFrame，MultiIndex(datetime, instrument)）

这些文件位于对应历史 experiment 的 workspace 目录中（通常在 `git_ignore_folder/RD-Agent_workspace/<id>/...` 下面）。


## 7. 关键配置与约束（与 SOTA 相关）

- **不允许配置里出现 `static_path:`**：
  - `QlibFBWorkspace.execute()` 会拒绝包含 `static_path:` 的 qlib config
  - 目的：确保每轮回测必须用当前轮的 `combined_factors_df.parquet`，避免“永远跑旧静态因子”造成评估失真。

- **相关性去重阈值**：`0.995`（几乎相同才删）

- **动态因子 join 质量阈值（loader 参数）**：
  - 在 `conf_combined_factors_dynamic.yaml` / `conf_combined_factors_sota_model.yaml` 中
  - `min_dynamic_non_nan_ratio: 0.01`
  - `min_instrument_overlap_ratio: 0.8`
  - `enforce_instrument_format: true`


## 8. 常见误解澄清

- “SOTA 因子 = 静态资金流因子文件”
  - 不准确。SOTA 因子是基于 `based_experiments` 的历史产物拼接。
  - 资金流相关字段可能来自 `static_factors.parquet` 或派生逻辑，但它不是 SOTA 定义本身。

- “每轮会跑多次回测用于对照”
  - 当前实现不会。每轮默认只跑一次回测。


## 9. 相关代码索引（便于进一步查阅）

- SOTA 构建与回测入口：
  - `rdagent/scenarios/qlib/developer/factor_runner.py::QlibFactorRunner.develop()`
- 历史因子产物收集：
  - `rdagent/scenarios/qlib/developer/utils.py::process_factor_data()`
- 新因子去重：
  - `rdagent/scenarios/qlib/developer/factor_runner.py::QlibFactorRunner.deduplicate_new_factors()`
- 回测执行（qrun）：
  - `rdagent/scenarios/qlib/experiment/workspace.py::QlibFBWorkspace.execute()`
- 回测配置模板：
  - `rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml`
  - `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_dynamic.yaml`
  - `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml`

