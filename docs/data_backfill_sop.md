# RD-Agent 数据补录标准操作流程 (SOP)

本文档旨在指导如何将历史或生产环境中的 RD-Agent 实验数据补录到 `registry.sqlite` 数据库，并导出为 AIStock API 可消费的结构化 JSON，同时实现因子代码的同步共享。

## 适用场景
1. 存在大量历史 `RD-Agent_workspace` 目录，但数据库记录缺失。
2. 实验在不同机器/环境运行后，需要汇总成果到 AIStock。
3. 更新了采集过滤逻辑（如：回测指标强校验），需要重刷存量数据。
4. 需要将存量实验中的因子 Python 实现代码同步到共享库 `rd-factors-lib`。

## 环境配置
在执行补录前，请确保 `.env` 文件中已配置共享库路径，补录脚本会将因子和策略代码同步到这些库中：
```env
# 因子实现代码同步的目标库路径 (指向本地克隆的 rd-factors-lib 仓库根目录)
RD_FACTORS_LIB_ROOT=F:\Dev\rd-factors-lib

# 策略配置代码同步的目标库路径 (指向本地克隆的 rd-strategies-lib 仓库根目录)
RD_STRATEGIES_LIB_ROOT=F:\Dev\rd-strategies-lib
```

## 核心步骤

### 1. 物理目录扫描与注册
首先需要确保所有 Workspace 目录都在数据库中挂号。
- **存量数据**：若已有大量实验目录，需先运行扫描脚本。
- **增量数据**：RD-Agent 运行时会自动注册新的 Workspace。

```powershell
# --ws-root: Workspace 根目录路径
# --db: 目标 registry.sqlite 路径
python tools/scan_and_register_all_workspaces.py `
    --ws-root git_ignore_folder/RD-Agent_workspace `
    --db RDagentDB/registry.sqlite
```

### 2. 结构化成果采集与代码同步 (Backfill)
对标记为有成果的 Workspace 执行深度采集和 Phase 3 标准化补齐。
- **Phase 3 补全**：脚本会强制为所有 Workspace 补齐 `strategy_meta.json` (策略 Python 化) 和 `model_meta.json` (模型权威元数据)。
- **指标校验**：对于因子实验，根据 `factor_perf.json` 校验回测指标，只有具备有效指标的才会生成 `factor_meta.json`。
- **代码同步**：提取因子 Wrapper 代码及策略配置函数，同步到共享库。

```powershell
# 设置 PYTHONPATH 确保导入正常
$env:PYTHONPATH="."

# 执行全量补录
python tools/backfill_registry_artifacts.py `
    --db RDagentDB/registry.sqlite `
    --mode backfill `
    --all-task-runs `
    --only-experiment-workspace `
    --overwrite-json
```


### 3. 导出 AIStock Catalog
将数据库中的结构化信息同步到 AIStock 前端消费的 JSON 文件中。

```powershell
# 1. 导出因子目录 (去重且包含历史补齐因子)
python tools/export_aistock_factor_catalog.py --registry-sqlite RDagentDB/registry.sqlite --output RDagentDB/aistock/factor_catalog.json

# 2. 导出实验轮次 (Loop) 目录
python tools/export_aistock_loop_catalog.py --registry-sqlite RDagentDB/registry.sqlite --output RDagentDB/aistock/loop_catalog.json

# 3. 导出模型 (Model) 目录
python tools/export_aistock_model_catalog.py --registry-sqlite RDagentDB/registry.sqlite --output RDagentDB/aistock/model_catalog.json

# 4. 导出策略 (Strategy) 目录
python tools/export_aistock_strategy_catalog.py --registry-sqlite RDagentDB/registry.sqlite --output RDagentDB/aistock/strategy_catalog.json
```

### 4. 数据完整性验证
执行验证脚本，确保输出的 Catalog 数量符合预期且逻辑一致。

```powershell
python tools/verify_catalog_counts.py
```

## 5. AIstock 全量数据更新步骤 (交付指引)

AIstock 侧应用 RD-Agent Phase 3 成果的完整更新步骤如下：

### 第一步：同步共享库

1. 拉取最新的 `rd-factors-lib` 代码，确保 `rd_factors_lib/generated.py` 包含最新的因子 Wrapper。
2. 拉取最新的 `rd-strategies-lib` 代码，确保 `rd_strategies_lib/generated.py` 包含最新的策略配置函数。

### 第二步：导入最新的 Catalog

将 RD-Agent 导出的 JSON 文件复制到 AIstock 的数据目录（如 `RDagentDB/aistock/`）：

- `factor_catalog.json`：包含 Phase 3 因子接口信息。
- `strategy_catalog.json`：包含 Phase 3 策略 Python 实现信息。
- `model_catalog.json`：包含 Phase 3 模型权威配置。

### 第三步：配置执行引擎

在 AIstock 的执行引擎中，根据 Catalog 中的元数据进行以下逻辑更新：

1. **因子执行**：检测 `interface_info.type == 'class'`，则调用 `interface_info.standard_wrapper` 指定的函数。
2. **策略加载**：通过 `python_implementation` 字段动态加载策略配置函数，替换原有的 YAML 直接解析逻辑。
3. **模型重构**：从 `model_catalog.json` 中的 `model_config` 和 `dataset_config` 字段读取权威配置进行模型实例重建。

## 注意事项
- **WSL 路径转换**：采集脚本已内置 `_to_native_path` 逻辑，支持在 Windows 宿主机上处理 WSL 格式的数据库路径。
- **代码同步原理**：同步逻辑会读取 Workspace 下的 `*.py` 文件，提取类或函数定义。若库路径不存在，同步将自动跳过。
- **数据完整性**：Phase 3 补录确保了即使没有回测结果的 Workspace 也会补齐元数据，确保资产 100% 覆盖。
