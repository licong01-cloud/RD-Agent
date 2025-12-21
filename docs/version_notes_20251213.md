# 版本说明（2025-12-13）

本次更新聚焦于 **Qlib 场景的可用性与鲁棒性**：降低因子/模型实验的常见失败率、增强数据字段可用性、并修复启动阶段的阻断性问题（循环导入、数据目录介绍对 parquet 不支持）。

## 1. 主要变更概览

### 1.1 因子自动修复（Factor Auto-Repair）
- **位置**：`rdagent/scenarios/qlib/developer/factor_runner.py`
- **内容**：在因子开发/执行失败时，自动触发最多 `N` 次的“修复-重试”流程（默认 `2` 次）。
- **触发条件**：仅对常见且可修复的错误签名触发（例如：缺失文件、字段 KeyError、部分 pandas MultiIndex 相关 NotImplementedError、预期产物缺失等），避免对不可修复问题进行无意义重试。
- **配置**：
  - 环境变量 `QLIB_FACTOR_MAX_REPAIR_ATTEMPTS` 控制最大修复次数。

### 1.2 模型运行自动重试（Model Retry）
- **位置**：`rdagent/scenarios/qlib/developer/model_runner.py`
- **内容**：模型执行失败时按预设组合进行最多 3 次重试（例如在 `DatasetH` / `TSDatasetH` 间切换、调整 `step_len/num_timesteps`），用于缓解模型与 dataset/window 不匹配导致的失败。
- **触发条件**：仅当错误签名与 qlib dataset/window 相关时才触发，避免扩大运行成本。

### 1.3 静态因子/原始字段融合数据（static_factors.parquet）生成工具
- **位置**：`tools/generate_static_factors_bundle.py`
- **内容**：将 `daily_basic` / `moneyflow` 等原始字段与已有 precomputed 因子融合，生成可被因子代码直接 join 的 `static_factors.parquet`，并输出 schema（JSON/CSV）。
- **目的**：让 LLM 在因子演进中能可靠使用原始字段（含中文含义），减少“猜字段名/猜含义”。

### 1.4 修复循环导入导致的启动失败
- **问题**：`ImportError: cannot import name 'rdagent_logger' from partially initialized module 'rdagent.log'`
- **根因**：`rdagent.core.utils` 与 `rdagent.log.logger` 顶层互相 import，形成循环依赖。
- **修复**：
  - **位置**：`rdagent/core/utils.py`
  - **方式**：移除顶层 `rdagent_logger` 导入，改为在 `cache_with_pickle` 且 `QLIB_QUANT_CACHE_DEBUG=1` 时才进行 lazy import。

### 1.5 支持 `.parquet` 数据文件描述（避免 quant 初始化阶段报错）
- **问题**：`NotImplementedError: file type static_factors.parquet is not supported`
- **根因**：`get_data_folder_intro()` 遍历数据目录生成 prompt 时，`get_file_desc()` 仅支持 `.h5/.md`。
- **修复**：
  - **位置**：`rdagent/scenarios/qlib/experiment/utils.py`
  - **方式**：
    - 支持所有 `.parquet` 的轻量描述（文件大小 + schema/列名），不加载全量数据。
    - 对 `static_factors.parquet` 做特判：补充索引/字段前缀约定说明（`db_`/`mf_`/`ae_`）。
    - 优先读取同目录的 `*_schema.csv/json`，否则 fallback 读取 parquet 元数据。

## 2. 兼容性与行为变化说明

- **不会影响回测/训练的真实计算**：上述 `.parquet` 描述与 logger lazy import 都属于“提示/日志/初始化信息构造”阶段的改动，不改变 Qlib 实际数据读取与计算结果。
- **自动修复/重试会增加少量运行时长**：仅在触发错误且满足签名条件时执行；可通过环境变量控制。

## 3. 建议的最小 smoke test

在 WSL conda 环境中执行：

```bash
cd /mnt/c/Users/lc999/RD-Agent-main
conda activate rdagent-gpu
export QLIB_FACTOR_MAX_REPAIR_ATTEMPTS=2
export QLIB_QUANT_DISABLE_CACHE=1
dotenv run -- python -m rdagent.app.qlib_rd_loop.quant --loop_n 1 --step_n 1
```

预期：
- 不再出现循环导入 ImportError
- 不再出现 `static_factors.parquet is not supported`
- 若触发因子修复或模型重试，会看到类似日志：
  - `[FactorAutoRepair] ...`
  - `[ModelRetry] attempt ...`

## 4. 相关文件列表

- `rdagent/scenarios/qlib/developer/factor_runner.py`
- `rdagent/scenarios/qlib/developer/model_runner.py`
- `tools/generate_static_factors_bundle.py`
- `rdagent/core/utils.py`
- `rdagent/scenarios/qlib/experiment/utils.py`
