# RD-Agent × AIstock 完整初始化与增量补充指南

## 目录

1. [方案完整性分析](#方案完整性分析)
2. [强关联关系保证](#强关联关系保证)
3. [完整初始化步骤](#完整初始化步骤)
4. [增量补充步骤](#增量补充步骤)
5. [验证检查清单](#验证检查清单)
6. [故障排查](#故障排查)

---

## 方案完整性分析

### 当前方案覆盖范围

#### 1. 元数据完整性

| 元数据类型 | 包含字段 | 关联字段 | 状态 |
|-----------|---------|---------|------|
| **Loop** | task_run_id, loop_id, workspace_id, workspace_path, log_dir, paths, asset_bundle_id | task_run_id, loop_id, workspace_id | ✅ 完整 |
| **Factor** | name, input_schema, task_run_id, loop_id, workspace_id, workspace_path, asset_bundle_id | task_run_id, loop_id, workspace_id | ✅ 完整 |
| **Model** | model_type, preprocess_config, task_run_id, loop_id, workspace_id, workspace_path, model_artifacts, asset_bundle_id | task_run_id, loop_id, workspace_id | ✅ 完整 |
| **Strategy** | strategy_id, portfolio_config, task_run_id, loop_id, workspace_id, workspace_path, python_implementation, asset_bundle_id | task_run_id, loop_id, workspace_id | ✅ 完整 |

#### 2. 文件资产完整性

| 文件类型 | 存储位置 | 关联方式 | 状态 |
|---------|---------|---------|------|
| **Workspace 文件** | `F:/Dev/RD-Agent-main/log/<task_run_id>/workspace_<workspace_id>` | workspace_path | ✅ 完整 |
| **资产包** | `F:/Dev/RD-Agent-main/production_bundles/<asset_bundle_id>.tar.gz` | asset_bundle_id | ✅ 完整 |
| **AIstock 缓存** | `F:/Dev/AIstock/rdagent_assets/bundles/<asset_bundle_id>` | asset_bundle_id | ✅ 完整 |
| **策略共享库** | `F:/Dev/rd-strategies-lib/rd_strategies_lib/generated.py` | strategy_name | ✅ 完整 |

#### 3. 强关联关系保证

**三层关联机制**：

1. **第一层：Registry 数据库关联**
   - `loops` 表 ↔ `workspaces` 表（通过 task_run_id）
   - `loops` 表 ↔ `task_runs` 表（通过 task_run_id）
   - `loops` 表 ↔ `production_bundles` 表（通过 asset_bundle_id）

2. **第二层：Catalog JSON 关联**
   - Loop Catalog 包含 `task_run_id`, `loop_id`, `workspace_id`, `workspace_path`, `asset_bundle_id`
   - Factor Catalog 包含 `task_run_id`, `loop_id`, `workspace_id`, `workspace_path`, `asset_bundle_id`
   - Model Catalog 包含 `task_run_id`, `loop_id`, `workspace_id`, `workspace_path`, `asset_bundle_id`
   - Strategy Catalog 包含 `task_run_id`, `loop_id`, `workspace_id`, `workspace_path`, `asset_bundle_id`

3. **第三层：文件系统关联**
   - Workspace 目录结构：`log/<task_run_id>/workspace_<workspace_id>/`
   - 资产包文件：`production_bundles/<asset_bundle_id>.tar.gz`
   - AIstock 缓存：`rdagent_assets/bundles/<asset_bundle_id>/`

**关联查询路径**：

```
Loop Catalog (task_run_id, loop_id)
    ↓
Registry Database (loops table)
    ↓
Workspace (workspace_path) ←→ Asset Bundle (asset_bundle_id)
    ↓                              ↓
Files in Workspace          Files in Asset Bundle
    ↓                              ↓
factor_meta.json           model.pkl
factor_perf.json           factor_*.py
model_meta.json            strategy_*.py
strategy_meta.json         YAML configs
```

---

## 强关联关系保证

### 关键关联字段说明

#### 1. Loop 层面关联

```json
{
  "task_run_id": "20250107_001",
  "loop_id": 1,
  "workspace_id": "ws_12345",
  "workspace_path": "F:/Dev/RD-Agent-main/log/20250107_001/workspace_ws_12345",
  "log_dir": "F:/Dev/RD-Agent-main/log/20250107_001",
  "asset_bundle_id": "bundle_20250107_001_1",
  "paths": {
    "factor_meta": "factor_meta.json",
    "factor_perf": "factor_perf.json",
    "feedback": "feedback.json",
    "model_files": ["model.pkl"],
    "mlruns": "mlruns"
  }
}
```

#### 2. Factor 层面关联

```json
{
  "name": "feature_PriceStrength_10D",
  "task_run_id": "20250107_001",
  "loop_id": 1,
  "workspace_id": "ws_12345",
  "workspace_path": "F:/Dev/RD-Agent-main/log/20250107_001/workspace_ws_12345",
  "asset_bundle_id": "bundle_20250107_001_1",
  "input_schema": {
    "required_fields": ["open", "high", "low", "close", "volume"],
    "lookback_days": 10
  }
}
```

#### 3. Model 层面关联

```json
{
  "model_type": "LGBModel",
  "task_run_id": "20250107_001",
  "loop_id": 1,
  "workspace_id": "ws_12345",
  "workspace_path": "F:/Dev/RD-Agent-main/log/20250107_001/workspace_ws_12345",
  "asset_bundle_id": "bundle_20250107_001_1",
  "preprocess_config": {
    "normalize": "zscore",
    "fillna": "forward_fill"
  },
  "model_artifacts": {
    "model_files": ["model.pkl"],
    "mlruns": "mlruns"
  }
}
```

#### 4. Strategy 层面关联

```json
{
  "strategy_id": "strategy_20250107_001_1",
  "task_run_id": "20250107_001",
  "loop_id": 1,
  "workspace_id": "ws_12345",
  "workspace_path": "F:/Dev/RD-Agent-main/log/20250107_001/workspace_ws_12345",
  "asset_bundle_id": "bundle_20250107_001_1",
  "portfolio_config": {
    "signal_config": {
      "top_k": 50
    }
  },
  "python_implementation": {
    "module": "rd_strategies_lib.generated",
    "func": "get_strategy_20250107_001_1_config"
  }
}
```

### 关联查询示例

#### 查询 Loop 的所有关联数据

```sql
-- 查询 Loop 的完整信息
SELECT
    l.task_run_id,
    l.loop_id,
    l.workspace_id,
    l.workspace_path,
    l.log_dir,
    pb.bundle_id as asset_bundle_id,
    pb.bundle_path as asset_bundle_path
FROM loops l
LEFT JOIN production_bundles pb ON l.task_run_id = pb.task_run_id AND l.loop_id = pb.loop_id
WHERE l.task_run_id = '20250107_001' AND l.loop_id = 1;
```

#### 查询 Factor 的所有关联数据

```sql
-- 查询 Factor 的完整信息
SELECT
    f.name,
    f.task_run_id,
    f.loop_id,
    f.workspace_id,
    f.workspace_path,
    l.log_dir,
    pb.bundle_id as asset_bundle_id
FROM aistock_factor_catalog f
JOIN loops l ON f.task_run_id = l.task_run_id AND f.loop_id = l.loop_id
LEFT JOIN production_bundles pb ON f.task_run_id = pb.task_run_id AND f.loop_id = pb.loop_id
WHERE f.name = 'feature_PriceStrength_10D';
```

---

## 完整初始化步骤

### 前置条件检查

#### 1. 环境检查

```bash
# 检查 RD-Agent 环境
cd F:/Dev/RD-Agent-main
python -c "import rdagent; print('✅ RD-Agent 环境正常')"

# 检查 AIstock 环境
cd F:/Dev/AIstock/backend
python -c "import aistock; print('✅ AIstock 环境正常')"

# 检查 PostgreSQL 连接
psql -U postgres -d aistock -c "SELECT version();"
```

#### 2. 目录结构检查

```bash
# 检查 RD-Agent 目录结构
cd F:/Dev/RD-Agent-main
dir log
dir RDagentDB
dir production_bundles

# 检查 AIstock 目录结构
cd F:/Dev/AIstock
dir rdagent_assets
dir rdagent_data
```

#### 3. 数据库表检查

```bash
# 连接到 RD-Agent 数据库
psql -U postgres -d rdagent

# 检查关键表
\d loops
\d workspaces
\d task_runs
\d production_bundles
\d artifacts

# 退出
\q

# 连接到 AIstock 数据库
psql -U postgres -d aistock

# 检查关键表
\d aistock_factor_catalog
\d aistock_model_catalog
\d aistock_strategy_catalog
\d aistock_loop_catalog

# 退出
\q
```

### 步骤 1：RD-Agent 侧元数据生成

#### 1.1 备份现有数据

```bash
cd F:/Dev/RD-Agent-main

# 备份 Registry 数据库
copy RDagentDB\\registry.sqlite RDagentDB\\registry.sqlite.backup.%date:~0,4%%date:~5,2%%date:~8,2%

# 备份 production_bundles 目录（如果存在）
if exist production_bundles (
    xcopy production_bundles production_bundles.backup.%date:~0,4%%date:~5,2%%date:~8,2% /E /I /Y
)

# 备份策略共享库（如果存在）
if exist F:\\Dev\\rd-strategies-lib (
    xcopy F:\\Dev\\rd-strategies-lib F:\\Dev\\rd-strategies-lib.backup.%date:~0,4%%date:~5,2%%date:~8,2% /E /I /Y
)
```

#### 1.2 运行 Backfill 生成增强元数据

```bash
cd F:/Dev/RD-Agent-main

# 方式 1：全量 backfill（推荐首次初始化）
python tools/backfill_registry_artifacts.py --mode solidify-all --all-task-runs --overwrite-json

# 方式 2：指定时间范围 backfill
python tools/backfill_registry_artifacts.py --mode solidify-all --start-date 2025-01-01 --end-date 2025-12-31 --overwrite-json

# 方式 3：指定 task_run_id backfill
python tools/backfill_registry_artifacts.py --mode backfill --task-run-id 20250107_001 --overwrite-json
```

**预期输出**：
```
[INFO] Starting backfill process...
[INFO] Processing task_run: 20250107_001
[INFO] Processing loop: 1
[INFO] Workspace: workspace_ws_12345
[INFO] Generated model_meta.json with preprocess_config
[INFO] Generated factor_meta.json with input_schema
[INFO] Generated strategy_meta.json with portfolio_config
[INFO] Registered artifacts in database
[INFO] Backfill completed successfully
```

#### 1.3 验证生成的元数据

```bash
cd F:/Dev/RD-Agent-main

# 查找最新的 workspace
python -c "
from pathlib import Path
import json

# 找到最新的 workspace 目录
log_dirs = sorted([d for d in Path('log').iterdir() if d.is_dir()], reverse=True)
if log_dirs:
    latest_log = log_dirs[0]
    ws_dirs = list(latest_log.glob('workspace_*'))
    if ws_dirs:
        ws = ws_dirs[0]
        print(f'检查 workspace: {ws}')
        
        # 检查 model_meta.json
        model_meta_path = ws / 'model_meta.json'
        if model_meta_path.exists():
            with open(model_meta_path) as f:
                meta = json.load(f)
                print(f'✅ model_meta.json: preprocess_config={\"preprocess_config\" in meta}')
        
        # 检查 factor_meta.json
        factor_meta_path = ws / 'factor_meta.json'
        if factor_meta_path.exists():
            with open(factor_meta_path) as f:
                meta = json.load(f)
                if meta.get('factors'):
                    first_factor = meta['factors'][0]
                    print(f'✅ factor_meta.json: input_schema={\"input_schema\" in first_factor}')
        
        # 检查 strategy_meta.json
        strategy_meta_path = ws / 'strategy_meta.json'
        if strategy_meta_path.exists():
            with open(strategy_meta_path) as f:
                meta = json.load(f)
                print(f'✅ strategy_meta.json: portfolio_config={\"portfolio_config\" in meta}')
"
```

### 步骤 2：资产包创建与文件拷贝

#### 2.1 创建资产包目录

```bash
cd F:/Dev/RD-Agent-main

# 创建 production_bundles 目录
if not exist production_bundles (
    mkdir production_bundles
)
```

#### 2.2 为每个 Loop 创建资产包

```bash
cd F:/Dev/RD-Agent-main

# 获取所有有结果的 Loop
python -c "
import sqlite3
import tarfile
from pathlib import Path
import uuid

conn = sqlite3.connect('RDagentDB/registry.sqlite')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 查询所有有结果的 Loop
cur.execute('''
    SELECT l.task_run_id, l.loop_id, w.workspace_id, w.workspace_path, l.log_dir
    FROM loops l
    JOIN workspaces w ON l.task_run_id = w.task_run_id
    WHERE l.has_result = 1 OR l.has_result = '1'
    ORDER BY l.task_run_id, l.loop_id
''')

loops = cur.fetchall()
print(f'找到 {len(loops)} 个有结果的 Loop')

for loop in loops:
    task_run_id = loop['task_run_id']
    loop_id = loop['loop_id']
    workspace_id = loop['workspace_id']
    workspace_path = loop['workspace_path']
    log_dir = loop['log_dir']
    
    # 生成 asset_bundle_id
    asset_bundle_id = f'bundle_{task_run_id}_{loop_id}'
    
    print(f'处理 Loop: {task_run_id}/{loop_id}, workspace: {workspace_id}')
    
    ws_root = Path(workspace_path)
    if not ws_root.exists():
        print(f'  ⚠️  Workspace 不存在，跳过')
        continue
    
    # 创建资产包
    bundle_path = Path('production_bundles') / f'{asset_bundle_id}.tar.gz'
    
    # 收集要打包的文件
    files_to_pack = []
    
    # 元数据文件
    for meta_file in ['factor_meta.json', 'factor_perf.json', 'feedback.json', 'model_meta.json', 'strategy_meta.json']:
        file_path = ws_root / meta_file
        if file_path.exists():
            files_to_pack.append(file_path)
    
    # 模型文件
    for model_file in ['model.pkl', 'model.joblib', 'model.bin', 'model.onnx', 'model.pt', 'model.pth']:
        file_path = ws_root / model_file
        if file_path.exists():
            files_to_pack.append(file_path)
    
    # 因子实现文件
    for factor_file in ws_root.glob('factor_*.py'):
        files_to_pack.append(factor_file)
    
    # YAML 配置文件
    for yaml_file in ws_root.glob('*.yaml'):
        files_to_pack.append(yaml_file)
    for yaml_file in ws_root.glob('*.yml'):
        files_to_pack.append(yaml_file)
    
    # 日志文件
    log_root = Path(log_dir) if log_dir else ws_root.parent
    for log_file in ['loop.log', 'task.log']:
        file_path = log_root / log_file
        if file_path.exists():
            files_to_pack.append(file_path)
    
    if not files_to_pack:
        print(f'  ⚠️  没有找到文件，跳过')
        continue
    
    # 创建 tar.gz 包
    with tarfile.open(bundle_path, 'w:gz') as tar:
        for file_path in files_to_pack:
            arcname = file_path.name
            tar.add(file_path, arcname=arcname)
    
    print(f'  ✅ 创建资产包: {bundle_path.name} ({len(files_to_pack)} 个文件)')
    
    # 更新 production_bundles 表
    try:
        cur.execute('''
            INSERT OR REPLACE INTO production_bundles
            (task_run_id, loop_id, bundle_id, bundle_path, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        ''', (task_run_id, loop_id, asset_bundle_id, str(bundle_path)))
        conn.commit()
    except Exception as e:
        print(f'  ⚠️  更新数据库失败: {e}')

conn.close()
print('\\n资产包创建完成')
"
```

**预期输出**：
```
找到 230 个有结果的 Loop
处理 Loop: 20250107_001/1, workspace: ws_12345
  ✅ 创建资产包: bundle_20250107_001_1.tar.gz (15 个文件)
处理 Loop: 20250107_001/2, workspace: ws_12346
  ✅ 创建资产包: bundle_20250107_001_2.tar.gz (12 个文件)
...
资产包创建完成
```

#### 2.3 验证资产包

```bash
cd F:/Dev/RD-Agent-main

# 列出所有资产包
dir production_bundles

# 验证资产包内容
python -c "
import tarfile
from pathlib import Path

bundle_dir = Path('production_bundles')
bundles = list(bundle_dir.glob('*.tar.gz'))

print(f'找到 {len(bundles)} 个资产包')

for bundle in bundles[:5]:  # 只显示前 5 个
    print(f'\\n{bundle.name}:')
    with tarfile.open(bundle, 'r:gz') as tar:
        members = tar.getnames()
        for member in members[:10]:  # 只显示前 10 个文件
            print(f'  - {member}')
        if len(members) > 10:
            print(f'  ... 共 {len(members)} 个文件')
"
```

### 步骤 3：Catalog 导出

#### 3.1 导出所有 Catalog

```bash
cd F:/Dev/RD-Agent-main

# 确保输出目录存在
if not exist rdagent_data (
    mkdir rdagent_data
)

# 导出 Factor Catalog
python tools/export_aistock_factor_catalog.py

# 导出 Model Catalog
python tools/export_aistock_model_catalog.py

# 导出 Strategy Catalog
python tools/export_aistock_strategy_catalog.py

# 导出 Loop Catalog
python tools/export_aistock_loop_catalog.py
```

**预期输出**：
```
[INFO] Exporting factor catalog...
[INFO] Exported 158 factors
[INFO] Output: rdagent_data/factor_catalog.json

[INFO] Exporting model catalog...
[INFO] Exported 50 models
[INFO] Output: rdagent_data/model_catalog.json

[INFO] Exporting strategy catalog...
[INFO] Exported 30 strategies
[INFO] Output: rdagent_data/strategy_catalog.json

[INFO] Exporting loop catalog...
[INFO] Exported 230 loops
[INFO] Output: rdagent_data/loop_catalog.json
```

#### 3.2 验证 Catalog 包含 asset_bundle_id

```bash
cd F:/Dev/RD-Agent-main

# 检查 Loop Catalog
python -c "
import json

with open('rdagent_data/loop_catalog.json') as f:
    catalog = json.load(f)
    
loops = catalog.get('loops', [])
print(f'Loop Catalog: {len(loops)} 个 Loop')

# 统计包含 asset_bundle_id 的 Loop
with_bundle = [l for l in loops if l.get('asset_bundle_id')]
print(f'包含 asset_bundle_id: {len(with_bundle)} 个 Loop ({len(with_bundle)/len(loops)*100:.1f}%)')

# 显示前 3 个 Loop 的关联信息
for loop in loops[:3]:
    print(f'\\nLoop: {loop[\"task_run_id\"]}/{loop[\"loop_id\"]}')
    print(f'  workspace_path: {loop.get(\"workspace_path\")}')
    print(f'  asset_bundle_id: {loop.get(\"asset_bundle_id\")}')
"
```

**预期输出**：
```
Loop Catalog: 230 个 Loop
包含 asset_bundle_id: 230 个 Loop (100.0%)

Loop: 20250107_001/1
  workspace_path: F:/Dev/RD-Agent-main/log/20250107_001/workspace_ws_12345
  asset_bundle_id: bundle_20250107_001_1
```

### 步骤 4：AIstock 侧数据导入

#### 4.1 拷贝 Catalog 文件到 AIstock

```bash
# 方式 1：直接拷贝
copy F:\\Dev\\RD-Agent-main\\rdagent_data\\factor_catalog.json F:\\Dev\\AIstock\\rdagent_data\\
copy F:\\Dev\\RD-Agent-main\\rdagent_data\\model_catalog.json F:\\Dev\\AIstock\\rdagent_data\\
copy F:\\Dev\\RD-Agent-main\\rdagent_data\\strategy_catalog.json F:\\Dev\\AIstock\\rdagent_data\\
copy F:\\Dev\\RD-Agent-main\\rdagent_data\\loop_catalog.json F:\\Dev\\AIstock\\rdagent_data\\

# 方式 2：使用 xcopy（推荐）
xcopy F:\\Dev\\RD-Agent-main\\rdagent_data\\*.json F:\\Dev\\AIstock\\rdagent_data\\ /Y
```

#### 4.2 导入 Catalog 到数据库

```bash
cd F:/Dev/AIstock/backend

# 启动 Python 环境
python -c "
import sys
sys.path.insert(0, '.')

from services.rdagent_catalog_etl_service import (
    import_factor_catalog_from_file,
    import_model_catalog_from_file,
    import_strategy_catalog_from_file,
    import_loop_catalog_from_file
)

# 导入 Factor Catalog
print('导入 Factor Catalog...')
import_factor_catalog_from_file('../rdagent_data/factor_catalog.json')
print('✅ Factor Catalog 导入完成')

# 导入 Model Catalog
print('导入 Model Catalog...')
import_model_catalog_from_file('../rdagent_data/model_catalog.json')
print('✅ Model Catalog 导入完成')

# 导入 Strategy Catalog
print('导入 Strategy Catalog...')
import_strategy_catalog_from_file('../rdagent_data/strategy_catalog.json')
print('✅ Strategy Catalog 导入完成')

# 导入 Loop Catalog
print('导入 Loop Catalog...')
import_loop_catalog_from_file('../rdagent_data/loop_catalog.json')
print('✅ Loop Catalog 导入完成')
"
```

**预期输出**：
```
导入 Factor Catalog...
✅ Factor Catalog 导入完成
导入 Model Catalog...
✅ Model Catalog 导入完成
导入 Strategy Catalog...
✅ Strategy Catalog 导入完成
导入 Loop Catalog...
✅ Loop Catalog 导入完成
```

#### 4.3 验证数据库导入

```bash
# 连接到 AIstock 数据库
psql -U postgres -d aistock

# 检查导入的记录数
SELECT 'Factor Catalog' as catalog_type, COUNT(*) as count FROM aistock_factor_catalog
UNION ALL
SELECT 'Model Catalog', COUNT(*) FROM aistock_model_catalog
UNION ALL
SELECT 'Strategy Catalog', COUNT(*) FROM aistock_strategy_catalog
UNION ALL
SELECT 'Loop Catalog', COUNT(*) FROM aistock_loop_catalog;

# 检查 asset_bundle_id 字段
SELECT 
    'Loop Catalog' as catalog_type,
    COUNT(*) as total,
    COUNT(asset_bundle_id) as with_bundle_id,
    ROUND(COUNT(asset_bundle_id)::numeric / COUNT(*) * 100, 1) as coverage_pct
FROM aistock_loop_catalog
UNION ALL
SELECT 
    'Factor Catalog',
    COUNT(*),
    COUNT(asset_bundle_id),
    ROUND(COUNT(asset_bundle_id)::numeric / COUNT(*) * 100, 1)
FROM aistock_factor_catalog
UNION ALL
SELECT 
    'Model Catalog',
    COUNT(*),
    COUNT(asset_bundle_id),
    ROUND(COUNT(asset_bundle_id)::numeric / COUNT(*) * 100, 1)
FROM aistock_model_catalog
UNION ALL
SELECT 
    'Strategy Catalog',
    COUNT(*),
    COUNT(asset_bundle_id),
    ROUND(COUNT(asset_bundle_id)::numeric / COUNT(*) * 100, 1)
FROM aistock_strategy_catalog;

# 退出
\q
```

**预期输出**：
```
 catalog_type   | count
----------------+-------
 Factor Catalog |   158
 Model Catalog  |    50
 Strategy Catalog|    30
 Loop Catalog   |   230

 catalog_type   | total | with_bundle_id | coverage_pct
----------------+-------+----------------+--------------
 Loop Catalog   |   230 |            230 |        100.0
 Factor Catalog |   158 |            158 |        100.0
 Model Catalog  |    50 |             50 |        100.0
 Strategy Catalog|    30 |             30 |        100.0
```

### 步骤 5：资产包同步到 AIstock

#### 5.1 创建 AIstock 资产目录

```bash
cd F:/Dev/AIstock

# 创建资产目录
if not exist rdagent_assets (
    mkdir rdagent_assets
)

if not exist rdagent_assets\\bundles (
    mkdir rdagent_assets\\bundles
)
```

#### 5.2 拷贝资产包到 AIstock

```bash
# 方式 1：直接拷贝
xcopy F:\\Dev\\RD-Agent-main\\production_bundles\\*.tar.gz F:\\Dev\\AIstock\\rdagent_assets\\bundles\\ /Y

# 方式 2：使用 robocopy（推荐，支持断点续传）
robocopy F:\\Dev\\RD-Agent-main\\production_bundles F:\\Dev\\AIstock\\rdagent_assets\\bundles *.tar.gz /E /XO
```

#### 5.3 验证资产包拷贝

```bash
cd F:/Dev/AIstock/rdagent_assets/bundles

# 统计资产包数量
dir *.tar.gz | find /c ".tar.gz"

# 验证资产包完整性
python -c "
import tarfile
from pathlib import Path

bundle_dir = Path('.')
bundles = list(bundle_dir.glob('*.tar.gz'))

print(f'找到 {len(bundles)} 个资产包')

# 随机抽查 5 个资产包
import random
sample_bundles = random.sample(bundles, min(5, len(bundles)))

for bundle in sample_bundles:
    print(f'\\n{bundle.name}:')
    try:
        with tarfile.open(bundle, 'r:gz') as tar:
            members = tar.getnames()
            print(f'  ✅ 资产包有效，包含 {len(members)} 个文件')
    except Exception as e:
        print(f'  ❌ 资产包损坏: {e}')
"
```

### 步骤 6：完整关联关系验证

#### 6.1 验证 Loop → Workspace → Asset Bundle 关联

```bash
cd F:/Dev/RD-Agent-main

python -c "
import sqlite3
import json
from pathlib import Path

conn = sqlite3.connect('RDagentDB/registry.sqlite')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 查询一个 Loop 的完整信息
cur.execute('''
    SELECT 
        l.task_run_id, l.loop_id, l.workspace_id, l.workspace_path, l.log_dir,
        pb.bundle_id, pb.bundle_path
    FROM loops l
    LEFT JOIN production_bundles pb ON l.task_run_id = pb.task_run_id AND l.loop_id = pb.loop_id
    WHERE l.has_result = 1
    LIMIT 1
''')

loop = cur.fetchone()
if loop:
    print('Loop 关联信息:')
    print(f'  task_run_id: {loop[\"task_run_id\"]}')
    print(f'  loop_id: {loop[\"loop_id\"]}')
    print(f'  workspace_id: {loop[\"workspace_id\"]}')
    print(f'  workspace_path: {loop[\"workspace_path\"]}')
    print(f'  log_dir: {loop[\"log_dir\"]}')
    print(f'  bundle_id: {loop[\"bundle_id\"]}')
    print(f'  bundle_path: {loop[\"bundle_path\"]}')
    
    # 验证 Workspace 存在
    ws_root = Path(loop['workspace_path'])
    print(f'\\nWorkspace 验证:')
    print(f'  存在: {ws_root.exists()}')
    if ws_root.exists():
        # 检查元数据文件
        for meta_file in ['factor_meta.json', 'model_meta.json', 'strategy_meta.json']:
            file_path = ws_root / meta_file
            print(f'  {meta_file}: {file_path.exists()}')
    
    # 验证 Asset Bundle 存在
    bundle_path = Path(loop['bundle_path']) if loop['bundle_path'] else None
    print(f'\\nAsset Bundle 验证:')
    print(f'  存在: {bundle_path.exists() if bundle_path else False}')
    if bundle_path and bundle_path.exists():
        import tarfile
        with tarfile.open(bundle_path, 'r:gz') as tar:
            members = tar.getnames()
            print(f'  文件数: {len(members)}')
            print(f'  文件列表: {members[:5]}...')

conn.close()
"
```

#### 6.2 验证 Catalog 数据库关联

```bash
# 连接到 AIstock 数据库
psql -U postgres -d aistock

-- 验证 Loop 关联
SELECT 
    lc.task_run_id,
    lc.loop_id,
    lc.workspace_id,
    lc.workspace_path,
    lc.asset_bundle_id,
    fc.count as factor_count,
    mc.count as model_count
FROM aistock_loop_catalog lc
LEFT JOIN LATERAL (
    SELECT COUNT(*) as count
    FROM aistock_factor_catalog fc
    WHERE fc.task_run_id = lc.task_run_id AND fc.loop_id = lc.loop_id
) fc ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*) as count
    FROM aistock_model_catalog mc
    WHERE mc.task_run_id = lc.task_run_id AND mc.loop_id = lc.loop_id
) mc ON true
WHERE lc.has_result = true
LIMIT 10;

-- 退出
\q
```

#### 6.3 验证文件系统关联

```bash
cd F:/Dev/AIstock

python -c "
import json
from pathlib import Path

# 读取 Loop Catalog
with open('rdagent_data/loop_catalog.json') as f:
    loop_catalog = json.load(f)

loops = loop_catalog.get('loops', [])

print('验证文件系统关联:\\n')

for loop in loops[:3]:  # 只验证前 3 个
    task_run_id = loop['task_run_id']
    loop_id = loop['loop_id']
    workspace_path = loop.get('workspace_path')
    asset_bundle_id = loop.get('asset_bundle_id')
    
    print(f'Loop: {task_run_id}/{loop_id}')
    
    # 验证 Workspace
    if workspace_path:
        ws_root = Path(workspace_path)
        print(f'  Workspace: {ws_root.exists()}')
        if ws_root.exists():
            # 检查元数据文件
            for meta_file in ['factor_meta.json', 'model_meta.json', 'strategy_meta.json']:
                file_path = ws_root / meta_file
                print(f'    {meta_file}: {file_path.exists()}')
    
    # 验证 Asset Bundle
    if asset_bundle_id:
        bundle_path = Path('rdagent_assets/bundles') / f'{asset_bundle_id}.tar.gz'
        print(f'  Asset Bundle: {bundle_path.exists()}')
        if bundle_path.exists():
            import tarfile
            with tarfile.open(bundle_path, 'r:gz') as tar:
                members = tar.getnames()
                print(f'    文件数: {len(members)}')
    
    print()
"
```

---

## 增量补充步骤

### 场景 1：新增 Loop 完成

#### 1.1 在 RD-Agent 侧生成元数据

```bash
cd F:/Dev/RD-Agent-main

# 指定新的 task_run_id 和 loop_id
python tools/backfill_registry_artifacts.py --mode backfill --task-run-id <new_task_run_id> --loop-id <new_loop_id> --overwrite-json
```

#### 1.2 创建资产包

```bash
cd F:/Dev/RD-Agent-main

python -c "
import sqlite3
import tarfile
from pathlib import Path

conn = sqlite3.connect('RDagentDB/registry.sqlite')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 查询新的 Loop
cur.execute('''
    SELECT l.task_run_id, l.loop_id, w.workspace_id, w.workspace_path, l.log_dir
    FROM loops l
    JOIN workspaces w ON l.task_run_id = w.task_run_id
    WHERE l.task_run_id = ? AND l.loop_id = ?
''', ('<new_task_run_id>', <new_loop_id>))

loop = cur.fetchone()
if loop:
    task_run_id = loop['task_run_id']
    loop_id = loop['loop_id']
    workspace_id = loop['workspace_id']
    workspace_path = loop['workspace_path']
    log_dir = loop['log_dir']
    
    asset_bundle_id = f'bundle_{task_run_id}_{loop_id}'
    
    print(f'创建资产包: {asset_bundle_id}')
    
    ws_root = Path(workspace_path)
    if ws_root.exists():
        bundle_path = Path('production_bundles') / f'{asset_bundle_id}.tar.gz'
        
        # 收集文件（同初始化步骤）
        files_to_pack = []
        for meta_file in ['factor_meta.json', 'factor_perf.json', 'feedback.json', 'model_meta.json', 'strategy_meta.json']:
            file_path = ws_root / meta_file
            if file_path.exists():
                files_to_pack.append(file_path)
        
        for model_file in ['model.pkl', 'model.joblib', 'model.bin', 'model.onnx', 'model.pt', 'model.pth']:
            file_path = ws_root / model_file
            if file_path.exists():
                files_to_pack.append(file_path)
        
        for factor_file in ws_root.glob('factor_*.py'):
            files_to_pack.append(factor_file)
        
        for yaml_file in ws_root.glob('*.yaml'):
            files_to_pack.append(yaml_file)
        for yaml_file in ws_root.glob('*.yml'):
            files_to_pack.append(yaml_file)
        
        # 创建资产包
        with tarfile.open(bundle_path, 'w:gz') as tar:
            for file_path in files_to_pack:
                arcname = file_path.name
                tar.add(file_path, arcname=arcname)
        
        print(f'✅ 资产包创建成功: {bundle_path}')
        
        # 更新数据库
        cur.execute('''
            INSERT OR REPLACE INTO production_bundles
            (task_run_id, loop_id, bundle_id, bundle_path, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        ''', (task_run_id, loop_id, asset_bundle_id, str(bundle_path)))
        conn.commit()
    else:
        print('❌ Workspace 不存在')
else:
    print('❌ Loop 不存在')

conn.close()
"
```

#### 1.3 导出增量 Catalog

```bash
cd F:/Dev/RD-Agent-main

# 导出所有 Catalog（增量导出会自动合并）
python tools/export_aistock_factor_catalog.py
python tools/export_aistock_model_catalog.py
python tools/export_aistock_strategy_catalog.py
python tools/export_aistock_loop_catalog.py
```

#### 1.4 同步到 AIstock

```bash
# 拷贝 Catalog 文件
xcopy F:\\Dev\\RD-Agent-main\\rdagent_data\\*.json F:\\Dev\\AIstock\\rdagent_data\\ /Y

# 拷贝新的资产包
xcopy F:\\Dev\\RD-Agent-main\\production_bundles\\bundle_<new_task_run_id>_<new_loop_id>.tar.gz F:\\Dev\\AIstock\\rdagent_assets\\bundles\\ /Y

# 导入到数据库
cd F:/Dev/AIstock/backend

python -c "
import sys
sys.path.insert(0, '.')

from services.rdagent_catalog_etl_service import (
    import_factor_catalog_from_file,
    import_model_catalog_from_file,
    import_strategy_catalog_from_file,
    import_loop_catalog_from_file
)

import_factor_catalog_from_file('../rdagent_data/factor_catalog.json')
import_model_catalog_from_file('../rdagent_data/model_catalog.json')
import_strategy_catalog_from_file('../rdagent_data/strategy_catalog.json')
import_loop_catalog_from_file('../rdagent_data/loop_catalog.json')

print('✅ 增量 Catalog 导入完成')
"
```

### 场景 2：通过 AIstock UI 触发增量同步

#### 2.1 启动 AIstock 后端

```bash
cd F:/Dev/AIstock/backend

# 启动后端服务
python main.py
```

#### 2.2 通过 UI 触发同步

1. 访问 AIstock 前端：`http://localhost:3000/rdagent/factors`
2. 点击"同步 Catalog"按钮
3. 等待同步完成

#### 2.3 验证同步结果

```bash
# 查看后端日志
# 应该看到类似以下日志：
# [INFO] Starting incremental sync...
# [INFO] Downloading asset bundle: bundle_xxx
# [INFO] Extracting asset bundle: bundle_xxx
# [INFO] Importing factor metadata...
# [INFO] Importing model metadata...
# [INFO] Sync completed successfully
```

### 场景 3：通过 API 触发增量同步

#### 3.1 调用同步 API

```bash
# 触发全量同步
curl -X POST http://localhost:8000/api/rdagent/catalog/sync

# 触发增量同步（指定 last_sync_time）
curl -X POST http://localhost:8000/api/rdagent/catalog/sync?last_sync_time=2025-01-07T00:00:00Z
```

#### 3.2 查看同步状态

```bash
# 查询同步状态
curl http://localhost:8000/api/rdagent/catalog/sync-status
```

---

## 验证检查清单

### RD-Agent 侧验证

- [ ] Registry 数据库包含所有 Loop 记录
- [ ] Workspace 目录包含所有元数据文件
- [ ] `model_meta.json` 包含 `preprocess_config` 字段
- [ ] `factor_meta.json` 包含 `input_schema` 字段
- [ ] `strategy_meta.json` 包含 `portfolio_config` 字段
- [ ] `production_bundles` 目录包含所有资产包
- [ ] `production_bundles` 表包含所有资产包记录
- [ ] Catalog JSON 文件包含 `asset_bundle_id` 字段

### AIstock 侧验证

- [ ] `rdagent_data` 目录包含所有 Catalog JSON 文件
- [ ] `rdagent_assets/bundles` 目录包含所有资产包
- [ ] 数据库表包含所有 Catalog 记录
- [ ] 所有记录包含 `asset_bundle_id` 字段
- [ ] `asset_bundle_id` 覆盖率达到 100%

### 关联关系验证

- [ ] Loop Catalog ↔ Workspace Path 关联正确
- [ ] Loop Catalog ↔ Asset Bundle ID 关联正确
- [ ] Factor Catalog ↔ Loop Catalog 关联正确
- [ ] Model Catalog ↔ Loop Catalog 关联正确
- [ ] Strategy Catalog ↔ Loop Catalog 关联正确
- [ ] Workspace 文件存在且可访问
- [ ] Asset Bundle 文件存在且可解压

### 功能验证

- [ ] 可以通过 task_run_id 和 loop_id 查询到完整信息
- [ ] 可以通过 asset_bundle_id 下载并解压资产包
- [ ] 可以通过 workspace_path 访问原始文件
- [ ] 增量同步能够正确添加新数据
- [ ] 重复同步不会产生重复数据

---

## 故障排查

### 问题 1：资产包创建失败

**症状**：
```
❌ Workspace 不存在
```

**解决方案**：
1. 检查 workspace_path 是否正确
2. 检查 workspace 目录是否存在
3. 检查文件权限

```bash
cd F:/Dev/RD-Agent-main

# 检查 workspace 目录
python -c "
from pathlib import Path
ws_path = 'F:/Dev/RD-Agent-main/log/20250107_001/workspace_ws_12345'
ws_root = Path(ws_path)
print(f'Workspace 存在: {ws_root.exists()}')
print(f'Workspace 可读: {ws_root.exists() and any(ws_root.iterdir())}')
"
```

### 问题 2：Catalog 导入失败

**症状**：
```
[ERROR] Failed to import catalog
```

**解决方案**：
1. 检查 JSON 文件格式是否正确
2. 检查数据库表结构是否匹配
3. 查看详细错误日志

```bash
cd F:/Dev/AIstock/backend

# 验证 JSON 格式
python -c "
import json

with open('../rdagent_data/loop_catalog.json') as f:
    try:
        catalog = json.load(f)
        print('✅ JSON 格式正确')
        print(f'包含 {len(catalog.get(\"loops\", []))} 个 Loop')
    except json.JSONDecodeError as e:
        print(f'❌ JSON 格式错误: {e}')
"
```

### 问题 3：asset_bundle_id 缺失

**症状**：
```
coverage_pct < 100%
```

**解决方案**：
1. 检查 production_bundles 表是否正确更新
2. 重新运行资产包创建脚本
3. 重新导出 Catalog

```bash
cd F:/Dev/RD-Agent-main

# 检查 production_bundles 表
python -c "
import sqlite3

conn = sqlite3.connect('RDagentDB/registry.sqlite')
cur = conn.cursor()

cur.execute('SELECT COUNT(*) FROM production_bundles')
count = cur.fetchone()[0]
print(f'production_bundles 表包含 {count} 条记录')

cur.execute('''
    SELECT l.task_run_id, l.loop_id, pb.bundle_id
    FROM loops l
    LEFT JOIN production_bundles pb ON l.task_run_id = pb.task_run_id AND l.loop_id = pb.loop_id
    WHERE l.has_result = 1 AND pb.bundle_id IS NULL
    LIMIT 5
''')

missing = cur.fetchall()
if missing:
    print(f'\\n缺少 asset_bundle_id 的 Loop:')
    for row in missing:
        print(f'  {row[0]}/{row[1]}')
else:
    print('\\n✅ 所有 Loop 都有 asset_bundle_id')

conn.close()
"
```

### 问题 4：资产包下载失败

**症状**：
```
[ERROR] Failed to download asset bundle
```

**解决方案**：
1. 检查资产包文件是否存在
2. 检查文件权限
3. 检查网络连接

```bash
cd F:/Dev/AIstock/rdagent_assets/bundles

# 检查资产包文件
python -c "
from pathlib import Path

bundle_dir = Path('.')
bundles = list(bundle_dir.glob('*.tar.gz'))

print(f'找到 {len(bundles)} 个资产包')

for bundle in bundles[:5]:
    print(f'{bundle.name}: {bundle.stat().st_size / 1024:.2f} KB')
"
```

---

## 总结

### 方案完整性

✅ **完全满足**：当前方案能够保证 Loop、因子、策略、模型数据与文件资产的强关联，确保可以获取明确的对应关系。

**三层关联机制**：
1. Registry 数据库关联（task_run_id, loop_id, workspace_id）
2. Catalog JSON 关联（workspace_path, asset_bundle_id）
3. 文件系统关联（workspace 目录, 资产包文件）

**关联查询路径**：
```
Loop Catalog (task_run_id, loop_id)
    ↓
Registry Database (loops table)
    ↓
Workspace (workspace_path) ←→ Asset Bundle (asset_bundle_id)
    ↓                              ↓
Files in Workspace          Files in Asset Bundle
```

### 初始化步骤总结

1. ✅ RD-Agent 侧元数据生成（backfill）
2. ✅ 资产包创建与文件拷贝
3. ✅ Catalog 导出
4. ✅ AIstock 侧数据导入
5. ✅ 资产包同步到 AIstock
6. ✅ 完整关联关系验证

### 增量补充步骤总结

1. ✅ 新 Loop 完成后生成元数据
2. ✅ 创建新的资产包
3. ✅ 导出增量 Catalog
4. ✅ 同步到 AIstock（UI 或 API）

### 验证检查清单

- ✅ RD-Agent 侧验证（8 项）
- ✅ AIstock 侧验证（6 项）
- ✅ 关联关系验证（7 项）
- ✅ 功能验证（5 项）

按照本指南操作，可以完成 RD-Agent × AIstock 的完整初始化和后续增量补充，确保数据与文件的强关联关系。
