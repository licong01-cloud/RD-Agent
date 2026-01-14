# 数据预处理与验证模块初始化指南

## 概述

本文档提供 RD-Agent × AIstock 数据预处理与验证模块的完整初始化步骤，确保系统能够正确导出和消费增强的元数据。

## 前置条件

### 环境要求

- Python 3.9+
- RD-Agent 环境
- AIstock 环境
- PostgreSQL 数据库
- 必要的 Python 依赖包

### 目录结构

```
F:/Dev/
├── RD-Agent-main/          # RD-Agent 项目根目录
│   ├── rdagent/
│   │   └── utils/
│   │       └── artifacts_writer.py  # 已更新
│   ├── tools/
│   │   └── backfill_registry_artifacts.py  # 已支持新字段
│   └── log/                 # 实验日志目录
├── AIstock/                # AIstock 项目根目录
│   ├── backend/
│   │   ├── data_service/
│   │   │   └── preprocessor.py  # 新建
│   │   └── inference_engine.py  # 已更新
│   └── rdagent_data/         # RD-Agent 数据同步目录
└── rd-strategies-lib/       # 策略共享库
    └── rd_strategies_lib/
        └── generated.py      # 自动生成
```

## 初始化步骤

### 步骤 1：验证 RD-Agent 侧代码更新

#### 1.1 检查 artifacts_writer.py

```bash
cd F:/Dev/RD-Agent-main
python -c "
from rdagent.utils.artifacts_writer import (
    _extract_model_metadata_from_workspace,
    _build_factor_meta_dict,
    _generate_input_schema,
    _sync_strategy_impl_to_shared_lib
)
print('✅ 所有函数导入成功')

# 测试 input_schema 生成
schema = _generate_input_schema('feature_PriceStrength_10D', ['rdagent'])
print('✅ input_schema 生成测试通过:', schema['required_fields'])
"
```

**预期输出**：
```
✅ 所有函数导入成功
✅ input_schema 生成测试通过: ['open', 'high', 'low', 'close', 'volume']
```

#### 1.2 检查 backfill_registry_artifacts.py

```bash
cd F:/Dev/RD-Agent-main
python -c "
from tools.backfill_registry_artifacts import (
    _maybe_backfill_phase2_for_workspace,
    _maybe_backfill_phase2_for_loop
)
print('✅ backfill 函数导入成功')
"
```

### 步骤 2：运行 RD-Agent backfill 生成增强元数据

#### 2.1 备份现有数据（可选但推荐）

```bash
cd F:/Dev/RD-Agent-main
# 备份 registry.sqlite
copy RDagentDB\\registry.sqlite RDagentDB\\registry.sqlite.backup

# 备份现有 workspace 元数据（可选）
# 如果需要保留原始的 factor_meta.json 等文件，请先备份
```

#### 2.2 运行 backfill 脚本

```bash
cd F:/Dev/RD-Agent-main

# 方式 1：全量 backfill（推荐用于首次初始化）
python tools/backfill_registry_artifacts.py --mode solidify-all --all-task-runs

# 方式 2：仅 backfill 指定 task_run_id
python tools/backfill_registry_artifacts.py --mode backfill --task-run-id <task_run_id>

# 方式 3：仅 backfill 指定 loop_id
python tools/backfill_registry_artifacts.py --mode backfill --loop-id <loop_id>
```

**预期输出**：
```
[INFO] Starting backfill process...
[INFO] Processing workspace: <workspace_id>
[INFO] Generated model_meta.json with preprocess_config
[INFO] Generated factor_meta.json with input_schema
[INFO] Generated strategy_meta.json with portfolio_config
[INFO] Backfill completed successfully
```

#### 2.3 验证生成的元数据文件

```bash
cd F:/Dev/RD-Agent-main

# 检查 model_meta.json
python -c "
import json
from pathlib import Path

# 找一个 workspace 目录
ws_dirs = list(Path('log').glob('*/workspace_*'))
if ws_dirs:
    ws = ws_dirs[0]
    model_meta_path = ws / 'model_meta.json'
    if model_meta_path.exists():
        with open(model_meta_path) as f:
            meta = json.load(f)
            print('✅ model_meta.json 包含 preprocess_config:', 'preprocess_config' in meta)
            if 'preprocess_config' in meta:
                print('   预处理配置:', meta['preprocess_config'])
    else:
        print('⚠️  model_meta.json 不存在')
"
```

**预期输出**：
```
✅ model_meta.json 包含 preprocess_config: True
   预处理配置: {'normalize': 'zscore', 'fillna': 'forward_fill', 'clip': None, 'standardize_features': True}
```

```bash
# 检查 factor_meta.json
python -c "
import json
from pathlib import Path

ws_dirs = list(Path('log').glob('*/workspace_*'))
if ws_dirs:
    ws = ws_dirs[0]
    factor_meta_path = ws / 'factor_meta.json'
    if factor_meta_path.exists():
        with open(factor_meta_path) as f:
            meta = json.load(f)
            if meta.get('factors'):
                first_factor = meta['factors'][0]
                print('✅ factor_meta.json 包含 input_schema:', 'input_schema' in first_factor)
                if 'input_schema' in first_factor:
                    print('   输入 Schema:', first_factor['input_schema'])
    else:
        print('⚠️  factor_meta.json 不存在')
"
```

**预期输出**：
```
✅ factor_meta.json 包含 input_schema: True
   输入 Schema: {'required_fields': ['open', 'high', 'low', 'close', 'volume'], 'optional_fields': ['amount', 'pct_chg', 'turnover_rate'], 'lookback_days': 10, 'index_type': 'MultiIndex(datetime, instrument)', 'description': '因子计算所需的输入数据结构定义 (lookback: 10天)'}
```

```bash
# 检查 strategy_meta.json
python -c "
import json
from pathlib import Path

ws_dirs = list(Path('log').glob('*/workspace_*'))
if ws_dirs:
    ws = ws_dirs[0]
    strategy_meta_path = ws / 'strategy_meta.json'
    if strategy_meta_path.exists():
        with open(strategy_meta_path) as f:
            meta = json.load(f)
            print('✅ strategy_meta.json 包含 portfolio_config:', 'portfolio_config' in meta)
            if 'portfolio_config' in meta:
                print('   策略配置:', meta['portfolio_config'])
    else:
        print('⚠️  strategy_meta.json 不存在')
"
```

**预期输出**：
```
✅ strategy_meta.json 包含 portfolio_config: True
   策略配置: {'signal_config': {'top_k': 50, 'min_score': 0.5, 'max_positions': 50, 'score_field': 'score'}, 'weight_config': {'method': 'equal_weight', 'max_single_weight': 0.05}, 'rebalance_config': {'freq': '1d', 'rebalance_threshold': 0.1}, 'risk_config': {'max_drawdown': 0.2, 'max_single_loss': 0.05}}
```

### 步骤 3：导出增强的 Catalog 到 AIstock

#### 3.1 运行 Catalog 导出脚本

```bash
cd F:/Dev/RD-Agent-main

# 导出所有 Catalog
python sync_all_to_aistock.py

# 或者分别导出
python tools/export_aistock_factor_catalog.py
python tools/export_aistock_model_catalog.py
python tools/export_aistock_strategy_catalog.py
python tools/export_aistock_loop_catalog.py
```

**预期输出**：
```
[INFO] Exporting factor catalog...
[INFO] Exported 158 factors
[INFO] Exporting model catalog...
[INFO] Exported 50 models
[INFO] Exporting strategy catalog...
[INFO] Exported 30 strategies
[INFO] Exporting loop catalog...
[INFO] Exported 230 loops
[INFO] All catalogs exported successfully
```

#### 3.2 验证导出的 Catalog 文件

```bash
cd F:/Dev/RD-Agent-main

# 检查 factor_catalog.json
python -c "
import json
with open('rdagent_data/factor_catalog.json') as f:
    catalog = json.load(f)
    if catalog.get('factors'):
        first_factor = catalog['factors'][0]
        print('✅ factor_catalog.json 包含 input_schema:', 'input_schema' in first_factor)
    else:
        print('⚠️  factor_catalog.json 为空')
"
```

**预期输出**：
```
✅ factor_catalog.json 包含 input_schema: True
```

```bash
# 检查 model_catalog.json
python -c "
import json
with open('rdagent_data/model_catalog.json') as f:
    catalog = json.load(f)
    if catalog.get('models'):
        first_model = catalog['models'][0]
        print('✅ model_catalog.json 包含 preprocess_config:', 'preprocess_config' in first_model)
        if 'preprocess_config' in first_model:
            print('   预处理配置:', first_model['preprocess_config'])
    else:
        print('⚠️  model_catalog.json 为空')
"
```

**预期输出**：
```
✅ model_catalog.json 包含 preprocess_config: True
   预处理配置: {'normalize': 'zscore', 'fillna': 'forward_fill', 'clip': None, 'standardize_features': True}
```

```bash
# 检查 strategy_catalog.json
python -c "
import json
with open('rdagent_data/strategy_catalog.json') as f:
    catalog = json.load(f)
    if catalog.get('strategies'):
        first_strategy = catalog['strategies'][0]
        print('✅ strategy_catalog.json 包含 portfolio_config:', 'portfolio_config' in first_strategy)
    else:
        print('⚠️  strategy_catalog.json 为空')
"
```

**预期输出**：
```
✅ strategy_catalog.json 包含 portfolio_config: True
```

### 步骤 4：验证 AIstock 侧预处理模块

#### 4.1 验证 DataPreprocessor 模块

```bash
cd F:/Dev/AIstock/backend

python -c "
from data_service.preprocessor import DataPreprocessor
import pandas as pd
import numpy as np

# 创建测试数据
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100) * 10 + 5,
    'feature3': np.random.rand(100)
})

# 添加一些缺失值
df.loc[0:5, 'feature1'] = np.nan
df.loc[10:15, 'feature2'] = np.nan

print('原始数据统计:')
print(df.describe())

# 测试预处理
preprocessor = DataPreprocessor()
config = {
    'normalize': 'zscore',
    'fillna': 'forward_fill',
    'clip': [-2, 2],
    'standardize_features': True
}

df_processed = preprocessor.apply_model_preprocess(df, config)
print('\\n预处理后数据统计:')
print(df_processed.describe())

# 测试数据验证
schema = {
    'required_fields': ['feature1', 'feature2', 'feature3'],
    'lookback_days': 10
}
preprocessor.validate_factor_input(df, schema)
print('\\n✅ 数据验证通过')

# 测试数据质量检查
quality_report = preprocessor.check_data_quality(df)
print('\\n数据质量报告:')
print(f'总行数: {quality_report[\"total_rows\"]}')
print(f'缺失率: {quality_report[\"missing_ratio\"]}')
print(f'异常值数量: {quality_report[\"outlier_count\"]}')
"
```

**预期输出**：
```
原始数据统计:
           feature1     feature2     feature3
count  100.000000  100.000000  100.000000
mean     0.059312    5.082415    0.498850
std      0.992134    9.717642    0.287890
min     -2.990845   -24.649931    0.001436
25%     -0.617016    -1.917378    0.254715
50%      0.050788     5.491878    0.498850
75%      0.724529    11.466688    0.743819
max      2.634616    27.073081    0.997874

预处理后数据统计:
           feature1     feature2     feature3
count  100.000000  100.000000  100.000000
mean    -0.000000    0.000000    0.000000
std      1.000000    1.000000    1.000000
min     -2.000000   -2.000000   -2.000000
25%     -0.680000   -0.720000   -0.850000
50%      0.000000    0.040000    0.000000
75%      0.670000    0.660000    0.850000
max      2.000000    2.000000    2.000000

✅ 数据验证通过

数据质量报告:
总行数: 100
缺失率: {'feature1': 6, 'feature2': 6, 'feature3': 0}
异常值数量: {'feature1': 0, 'feature2': 0, 'feature3': 0}
```

#### 4.2 验证 InferenceEngine 集成

```bash
cd F:/Dev/AIstock/backend

python -c "
from inference_engine import InferenceEngine
from data_service.preprocessor import DataPreprocessor

# 创建 InferenceEngine 实例
engine = InferenceEngine()

# 验证 preprocessor 已初始化
print('✅ InferenceEngine.preprocessor 已初始化:', isinstance(engine.preprocessor, DataPreprocessor))

# 验证 preprocess_config 应用
test_config = {
    'normalize': 'zscore',
    'fillna': 'forward_fill',
    'clip': [-3, 3]
}
print('✅ 预处理配置测试通过:', test_config)
"
```

**预期输出**：
```
✅ InferenceEngine.preprocessor 已初始化: True
✅ 预处理配置测试通过: {'normalize': 'zscore', 'fillna': 'forward_fill', 'clip': [-3, 3]}
```

### 步骤 5：同步 Catalog 到 AIstock 数据库

#### 5.1 启动 AIstock 后端

```bash
cd F:/Dev/AIstock/backend

# 启动后端服务
python main.py
```

#### 5.2 触发 Catalog 同步

通过 AIstock 后端 API 触发 Catalog 同步：

```bash
# 方式 1：通过 API 调用
curl -X POST http://localhost:8000/api/rdagent/catalog/sync

# 方式 2：通过前端界面
# 访问 http://localhost:3000/rdagent/factors
# 点击"同步 Catalog"按钮
```

#### 5.3 验证数据库中的元数据

```bash
# 连接到 PostgreSQL 数据库
psql -U postgres -d aistock

# 检查 factor_catalog 表
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'aistock_factor_catalog';

# 检查是否有 input_schema 字段
SELECT COUNT(*) FROM aistock_factor_catalog WHERE input_schema IS NOT NULL;

# 检查 model_catalog 表
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'aistock_model_catalog';

# 检查是否有 preprocess_config 字段
SELECT COUNT(*) FROM aistock_model_catalog WHERE preprocess_config IS NOT NULL;

# 检查 strategy_catalog 表
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'aistock_strategy_catalog';

# 检查是否有 portfolio_config 字段
SELECT COUNT(*) FROM aistock_strategy_catalog WHERE portfolio_config IS NOT NULL;
```

**预期输出**：
```
 column_name  | data_type
--------------+-----------
 id           | integer
 name         | text
 ...
 input_schema | jsonb

 count
-------
    158

 column_name      | data_type
------------------+-----------
 id               | integer
 model_type       | text
 ...
 preprocess_config | jsonb

 count
-------
     50

 column_name       | data_type
-------------------+-----------
 id                | integer
 strategy_name     | text
 ...
 portfolio_config  | jsonb

 count
-------
     30
```

### 步骤 6：完整选股流程测试

#### 6.1 通过前端触发推理

1. 访问 AIstock 前端：`http://localhost:3000/rdagent/multi-selection`
2. 选择一个策略
3. 点击"执行选股"按钮
4. 观察后端日志

#### 6.2 检查后端日志

```bash
# 在 AIstock 后端终端查看日志
```

**预期日志输出**：
```
INFO:aistock.inference:加载策略资产: strategy_xxx@v1
INFO:aistock.inference:因子加载验证成功
INFO:aistock.inference:使用 Catalog 指定的标准化因子 Wrapper: factor_xxx
INFO:aistock.inference:获取推理数据成功: 2026-01-07
INFO:aistock.inference:因子计算成功
INFO:aistock.inference:应用数据预处理配置成功
INFO:aistock.data_service:Applied fillna method: forward_fill
INFO:aistock.data_service:Applied normalization method: zscore
INFO:aistock.data_service:Applied clip range: [-3, 3]
INFO:aistock.inference:过滤到 158 个特征用于模型推理
INFO:aistock.inference:模型推理成功: strategy_xxx, 日期: 2026-01-07, 获得 5000 条预测分数
INFO:aistock.inference:信号持久化成功
```

#### 6.3 验证推理结果

```bash
# 查询数据库中的推理结果
psql -U postgres -d aistock

SELECT 
    strategy_id,
    trade_date,
    COUNT(*) as stock_count,
    AVG(score) as avg_score,
    MAX(score) as max_score,
    MIN(score) as min_score
FROM trading.rdagent_signal
WHERE trade_date = '2026-01-07'
GROUP BY strategy_id, trade_date
ORDER BY avg_score DESC;
```

**预期输出**：
```
 strategy_id |  trade_date  | stock_count |     avg_score     |    max_score    |    min_score
--------------+--------------+-------------+-------------------+-----------------+------------------
 strategy_xxx  | 2026-01-07   |        5000 | 0.523456789012345 | 0.987654321098765 | 0.0123456789012345
```

## 故障排查

### 问题 1：backfill 脚本失败

**症状**：
```
[ERROR] Failed to extract model metadata
```

**解决方案**：
1. 检查 workspace 目录是否存在
2. 检查 YAML 配置文件格式是否正确
3. 查看详细错误日志

```bash
cd F:/Dev/RD-Agent-main
python tools/backfill_registry_artifacts.py --mode backfill --task-run-id <task_run_id> --verbose
```

### 问题 2：Catalog 导出失败

**症状**：
```
[ERROR] Failed to export factor catalog
```

**解决方案**：
1. 检查 registry.sqlite 数据库是否完整
2. 检查 workspace 元数据文件是否正确生成
3. 检查输出目录权限

```bash
cd F:/Dev/RD-Agent-main
python tools/export_aistock_factor_catalog.py --verbose
```

### 问题 3：AIstock 推理失败

**症状**：
```
[ERROR] 因子计算失败
[ERROR] 模型推理失败
```

**解决方案**：
1. 检查数据服务是否正常运行
2. 检查因子文件是否存在且可加载
3. 检查模型文件是否存在且可加载
4. 检查预处理配置是否正确

```bash
cd F:/Dev/AIstock/backend

# 测试数据服务
python -c "
from data_service.api import get_history_window
from datetime import datetime, timedelta

end = datetime.now()
start = end - timedelta(days=60)
df = get_history_window(
    universe=['000001.SZ'],
    start=start,
    end=end,
    freq='1d',
    adj='front'
)
print('✅ 数据服务正常，获取到', len(df), '条数据')
"
```

### 问题 4：预处理配置未生效

**症状**：
```
[WARNING] 未找到预处理配置，跳过预处理步骤
```

**解决方案**：
1. 检查 model_meta.json 是否包含 preprocess_config 字段
2. 检查 AIstock 是否正确加载了 Catalog 数据
3. 重新运行 backfill 和 Catalog 导出

```bash
cd F:/Dev/RD-Agent-main

# 重新生成元数据
python tools/backfill_registry_artifacts.py --mode solidify-all --all-task-runs

# 重新导出 Catalog
python sync_all_to_aistock.py

# 重新同步到 AIstock
curl -X POST http://localhost:8000/api/rdagent/catalog/sync
```

## 验收检查清单

### RD-Agent 侧

- [ ] `model_meta.json` 包含 `preprocess_config` 字段
- [ ] `factor_meta.json` 包含 `input_schema` 字段
- [ ] `strategy_meta.json` 包含 `portfolio_config` 字段
- [ ] backfill 脚本能够正确生成增强的元数据文件
- [ ] Catalog 导出脚本能够正确导出增强的元数据

### AIstock 侧

- [ ] DataPreprocessor 模块能够正确应用预处理配置
- [ ] InferenceEngine 能够正确调用预处理模块
- [ ] 数据验证逻辑能够正确检测缺失字段
- [ ] 完整选股流程能够成功执行并产生正确结果
- [ ] 数据库中的元数据包含新字段

### 集成验收

- [ ] AIstock 推理结果与 RD-Agent 回测结果在数值上对齐
- [ ] 预处理配置缺失时能够使用默认配置
- [ ] 数据验证失败时能够给出明确的错误提示
- [ ] 后端日志显示预处理步骤已正确执行

## 后续维护

### 定期更新元数据

```bash
cd F:/Dev/RD-Agent-main

# 每次有新的 Loop 完成后，运行 backfill
python tools/backfill_registry_artifacts.py --mode solidify-all

# 导出最新 Catalog
python sync_all_to_aistock.py

# 同步到 AIstock
curl -X POST http://localhost:8000/api/rdagent/catalog/sync
```

### 监控预处理性能

```bash
cd F:/Dev/AIstock/backend

# 查看预处理性能日志
grep "应用数据预处理" logs/aistock.log | tail -20
```

### 数据质量监控

```bash
cd F:/Dev/AIstock/backend

# 定期检查数据质量报告
python -c "
from data_service.preprocessor import DataPreprocessor
import pandas as pd

# 从数据库加载最新数据
df = pd.read_sql('SELECT * FROM market.stock_quotes ORDER BY date DESC LIMIT 1000', 'postgresql://postgres:password@localhost/aistock')

preprocessor = DataPreprocessor()
report = preprocessor.check_data_quality(df)
print('数据质量报告:', report)
"
```

## 总结

完成以上所有步骤后，RD-Agent × AIstock 数据预处理与验证模块将完全初始化并可用。系统将能够：

1. ✅ 自动导出模型预处理配置
2. ✅ 自动导出因子输入 Schema
3. ✅ 自动导出策略配置 Schema
4. ✅ 在推理时自动应用预处理
5. ✅ 自动验证输入数据完整性
6. ✅ 提供数据质量报告

系统已具备完整的数据预处理与验证能力，可以支持实时选股功能。
