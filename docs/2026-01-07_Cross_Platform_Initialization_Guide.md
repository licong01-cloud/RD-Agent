# RD-Agent × AIstock 跨平台初始化步骤（支持 WSL）

## 目录

1. [环境兼容性分析](#环境兼容性分析)
2. [历史数据处理能力](#历史数据处理能力)
3. [WSL 环境初始化步骤](#wsl-环境初始化步骤)
4. [Windows 环境初始化步骤](#windows-环境初始化步骤)
5. [增量补充步骤（跨平台）](#增量补充步骤跨平台)
6. [验证检查清单](#验证检查清单)

---

## 环境兼容性分析

### Python 脚本兼容性

✅ **Python 脚本完全跨平台兼容**

| 组件 | 支持状态 | 说明 |
|------|---------|------|
| `backfill_registry_artifacts.py` | ✅ 完全支持 | 使用 `_to_native_path()` 自动转换路径 |
| `export_aistock_factor_catalog.py` | ✅ 完全支持 | 使用 `Path` 对象，跨平台 |
| `export_aistock_model_catalog.py` | ✅ 完全支持 | 使用 `Path` 对象，跨平台 |
| `export_aistock_strategy_catalog.py` | ✅ 完全支持 | 使用 `Path` 对象，跨平台 |
| `export_aistock_loop_catalog.py` | ✅ 完全支持 | 使用 `Path` 对象，跨平台 |
| `sync_all_to_aistock.py` | ✅ 完全支持 | 使用 `Path` 对象，跨平台 |

**关键特性**：

```python
def _to_native_path(p_str: str) -> Path:
    """Convert path between WSL and Windows format based on current OS."""
    if not p_str:
        return Path()
    is_windows = os.name == "nt"
    # Normalize slashes first
    p_str = p_str.replace("/", os.sep).replace("\\", os.sep)
    
    if is_windows:
        # Handle WSL path: /mnt/f/... -> F:\...
        if p_str.lower().startswith(f"{os.sep}mnt{os.sep}"):
            parts = p_str.split(os.sep)
            if len(parts) >= 3:
                drive = parts[2].upper()
                return Path(f"{drive}:\\") / Path(*parts[3:])
        # Handle already Windows path
        if len(p_str) > 1 and p_str[1] == ":":
            return Path(p_str)
    else:
        # Handle Windows path in WSL: F:\... -> /mnt/f/...
        if len(p_str) > 1 and p_str[1] == ":":
            drive = p_str[0].lower()
            rel = p_str[3:].replace("\\", "/")
            return Path(f"/mnt/{drive}") / rel
            
    return Path(p_str)
```

### Shell 脚本兼容性

⚠️ **Shell 脚本需要区分环境**

| 操作 | Windows 命令 | WSL/Linux 命令 |
|------|-------------|---------------|
| 复制文件 | `copy` | `cp` |
| 复制目录 | `xcopy` | `cp -r` |
| 删除文件 | `del` | `rm` |
| 删除目录 | `rmdir` | `rm -rf` |
| 检查存在 | `if exist` | `if [ -e ]` |
| 创建目录 | `mkdir` | `mkdir -p` |

---

## 历史数据处理能力

### Backfill 脚本处理逻辑

✅ **能够处理所有历史 Loop，无论是否有 RD-Agent 侧写出的数据**

#### 处理流程

```
1. 检查 has_result
   ├─ has_result = True
   │  ├─ Phase 2: 生成 factor_meta.json, factor_perf.json, feedback.json
   │  └─ Phase 3: 生成 strategy_meta.json, model_meta.json
   │
   └─ has_result = False
      └─ Phase 3: 生成 strategy_meta.json, model_meta.json（始终执行）
```

#### has_result 判定逻辑

```python
# 从 backfill_registry_artifacts.py
action = ws.experiment_type or _read_loop_action(cur, ws.task_run_id, ws.loop_id) or "unknown"

if action == "model":
    has_result = bool(qlib_res.exists() and ret_pkl.exists())
elif action == "factor":
    has_result = bool(combined_factors.exists())
else:
    has_result = False
```

#### 数据导出保证

| 数据类型 | has_result=True | has_result=False |
|---------|----------------|------------------|
| **factor_meta.json** | ✅ 从 qlib_res.csv 生成 | ✅ 从 YAML 配置生成 |
| **factor_perf.json** | ✅ 从 qlib_res.csv 生成 | ⚠️ 跳过（无性能数据） |
| **feedback.json** | ✅ 从指标生成 | ⚠️ 跳过（无反馈数据） |
| **model_meta.json** | ✅ 从 workspace 提取 | ✅ 从 workspace 提取 |
| **strategy_meta.json** | ✅ 从 YAML 提取 | ✅ 从 YAML 提取 |
| **ret_curve.png** | ✅ 从 ret.pkl 生成 | ⚠️ 跳过（无收益数据） |
| **dd_curve.png** | ✅ 从 ret.pkl 生成 | ⚠️ 跳过（无回撤数据） |

**结论**：所有有成果的 Loop 都能导出至少 `model_meta.json` 和 `strategy_meta.json`，有结果的 Loop 还能导出完整的性能数据。

---

## WSL 环境初始化步骤

### 前置条件

```bash
# 检查 WSL 环境
uname -a
# 预期输出: Linux ... Microsoft ...

# 检查 Python 环境
python3 --version
# 预期输出: Python 3.9+

# 检查 PostgreSQL 客户端
psql --version
# 预期输出: psql (PostgreSQL) 14.x
```

### 步骤 1：环境变量配置

```bash
# 设置 RD-Agent 根目录（WSL 路径）
export RDAGENT_ROOT="/mnt/f/Dev/RD-Agent-main"
export AISTOCK_ROOT="/mnt/f/Dev/AIstock"

# 设置策略共享库路径（Windows 路径）
export RD_STRATEGIES_LIB_ROOT="/mnt/f/rd-strategies-lib"

# 验证路径
ls -la "$RDAGENT_ROOT"
ls -la "$AISTOCK_ROOT"
```

### 步骤 2：备份现有数据

```bash
cd "$RDAGENT_ROOT"

# 备份 Registry 数据库
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
cp RDagentDB/registry.sqlite "RDagentDB/registry.sqlite.backup.$BACKUP_DATE"

# 备份 production_bundles 目录（如果存在）
if [ -d "production_bundles" ]; then
    cp -r production_bundles "production_bundles.backup.$BACKUP_DATE"
fi

# 备份策略共享库（如果存在）
if [ -d "$RD_STRATEGIES_LIB_ROOT" ]; then
    cp -r "$RD_STRATEGIES_LIB_ROOT" "${RD_STRATEGIES_LIB_ROOT}.backup.$BACKUP_DATE"
fi

echo "✅ 备份完成: $BACKUP_DATE"
```

### 步骤 3：运行 Backfill 生成增强元数据

```bash
cd "$RDAGENT_ROOT"

# 方式 1：全量 backfill（推荐首次初始化）
python3 tools/backfill_registry_artifacts.py --mode solidify-all --all-task-runs --overwrite-json

# 方式 2：指定时间范围 backfill
python3 tools/backfill_registry_artifacts.py --mode solidify-all --start-date 2025-01-01 --end-date 2025-12-31 --overwrite-json

# 方式 3：指定 task_run_id backfill
python3 tools/backfill_registry_artifacts.py --mode backfill --task-run-id 20250107_001 --overwrite-json

# 方式 4：指定 loop_id backfill
python3 tools/backfill_registry_artifacts.py --mode backfill --loop-id 1 --overwrite-json
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
[INFO] Backfill completed successfully
```

### 步骤 4：验证生成的元数据

```bash
cd "$RDAGENT_ROOT"

python3 -c "
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

### 步骤 5：创建资产包

```bash
cd "$RDAGENT_ROOT"

# 创建 production_bundles 目录
mkdir -p production_bundles

# 为每个 Loop 创建资产包
python3 -c "
import sqlite3
import tarfile
from pathlib import Path

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

### 步骤 6：导出 Catalog

```bash
cd "$RDAGENT_ROOT"

# 确保输出目录存在
mkdir -p rdagent_data

# 导出 Factor Catalog
python3 tools/export_aistock_factor_catalog.py

# 导出 Model Catalog
python3 tools/export_aistock_model_catalog.py

# 导出 Strategy Catalog
python3 tools/export_aistock_strategy_catalog.py

# 导出 Loop Catalog
python3 tools/export_aistock_loop_catalog.py
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

### 步骤 7：同步到 AIstock

```bash
# 拷贝 Catalog 文件到 AIstock
cp "$RDAGENT_ROOT/rdagent_data"/*.json "$AISTOCK_ROOT/rdagent_data/"

# 拷贝资产包到 AIstock
mkdir -p "$AISTOCK_ROOT/rdagent_assets/bundles"
cp "$RDAGENT_ROOT/production_bundles"/*.tar.gz "$AISTOCK_ROOT/rdagent_assets/bundles/"

# 导入到数据库
cd "$AISTOCK_ROOT/backend"

python3 -c "
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

### 步骤 8：验证导入

```bash
# 连接到 AIstock 数据库
psql -U postgres -d aistock -c "
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
"
```

**预期输出**：
```
 catalog_type   | total | with_bundle_id | coverage_pct
----------------+-------+----------------+--------------
 Loop Catalog   |   230 |            230 |        100.0
 Factor Catalog |   158 |            158 |        100.0
 Model Catalog  |    50 |             50 |        100.0
 Strategy Catalog|    30 |             30 |        100.0
```

---

## Windows 环境初始化步骤

### 前置条件

```powershell
# 检查 Python 环境
python --version
# 预期输出: Python 3.9+

# 检查 PostgreSQL 客户端
psql --version
# 预期输出: psql (PostgreSQL) 14.x
```

### 步骤 1：环境变量配置

```powershell
# 设置 RD-Agent 根目录
$env:RDAGENT_ROOT = "F:\Dev\RD-Agent-main"
$env:AISTOCK_ROOT = "F:\Dev\AIstock"

# 设置策略共享库路径
$env:RD_STRATEGIES_LIB_ROOT = "F:\Dev\rd-strategies-lib"

# 验证路径
Get-ChildItem $env:RDAGENT_ROOT
Get-ChildItem $env:AISTOCK_ROOT
```

### 步骤 2：备份现有数据

```powershell
cd $env:RDAGENT_ROOT

# 备份 Registry 数据库
$BACKUP_DATE = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item "RDagentDB\registry.sqlite" "RDagentDB\registry.sqlite.backup.$BACKUP_DATE"

# 备份 production_bundles 目录（如果存在）
if (Test-Path "production_bundles") {
    Copy-Item -Recurse "production_bundles" "production_bundles.backup.$BACKUP_DATE"
}

# 备份策略共享库（如果存在）
if (Test-Path $env:RD_STRATEGIES_LIB_ROOT) {
    Copy-Item -Recurse $env:RD_STRATEGIES_LIB_ROOT "${env:RD_STRATEGIES_LIB_ROOT}.backup.$BACKUP_DATE"
}

Write-Host "✅ 备份完成: $BACKUP_DATE"
```

### 步骤 3：运行 Backfill 生成增强元数据

```powershell
cd $env:RDAGENT_ROOT

# 方式 1：全量 backfill（推荐首次初始化）
python tools\backfill_registry_artifacts.py --mode solidify-all --all-task-runs --overwrite-json

# 方式 2：指定时间范围 backfill
python tools\backfill_registry_artifacts.py --mode solidify-all --start-date 2025-01-01 --end-date 2025-12-31 --overwrite-json

# 方式 3：指定 task_run_id backfill
python tools\backfill_registry_artifacts.py --mode backfill --task-run-id 20250107_001 --overwrite-json
```

### 步骤 4：验证生成的元数据

```powershell
cd $env:RDAGENT_ROOT

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

### 步骤 5：创建资产包

```powershell
cd $env:RDAGENT_ROOT

# 创建 production_bundles 目录
if (-not (Test-Path "production_bundles")) {
    New-Item -ItemType Directory -Path "production_bundles"
}

# 为每个 Loop 创建资产包
python -c "
import sqlite3
import tarfile
from pathlib import Path

conn = sqlite3.connect('RDagentDB\registry.sqlite')
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

### 步骤 6：导出 Catalog

```powershell
cd $env:RDAGENT_ROOT

# 确保输出目录存在
if (-not (Test-Path "rdagent_data")) {
    New-Item -ItemType Directory -Path "rdagent_data"
}

# 导出 Factor Catalog
python tools\export_aistock_factor_catalog.py

# 导出 Model Catalog
python tools\export_aistock_model_catalog.py

# 导出 Strategy Catalog
python tools\export_aistock_strategy_catalog.py

# 导出 Loop Catalog
python tools\export_aistock_loop_catalog.py
```

### 步骤 7：同步到 AIstock

```powershell
# 拷贝 Catalog 文件到 AIstock
Copy-Item "$env:RDAGENT_ROOT\rdagent_data\*.json" "$env:AISTOCK_ROOT\rdagent_data\"

# 拷贝资产包到 AIstock
if (-not (Test-Path "$env:AISTOCK_ROOT\rdagent_assets\bundles")) {
    New-Item -ItemType Directory -Path "$env:AISTOCK_ROOT\rdagent_assets\bundles" -Force
}
Copy-Item "$env:RDAGENT_ROOT\production_bundles\*.tar.gz" "$env:AISTOCK_ROOT\rdagent_assets\bundles\"

# 导入到数据库
cd $env:AISTOCK_ROOT\backend

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

### 步骤 8：验证导入

```powershell
# 连接到 AIstock 数据库
psql -U postgres -d aistock -c "
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
"
```

---

## 增量补充步骤（跨平台）

### 场景 1：新增 Loop 完成（WSL）

```bash
cd "$RDAGENT_ROOT"

# 指定新的 task_run_id 和 loop_id
python3 tools/backfill_registry_artifacts.py --mode backfill --task-run-id <new_task_run_id> --loop-id <new_loop_id> --overwrite-json

# 创建资产包
python3 -c "
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

# 导出增量 Catalog
python3 tools/export_aistock_factor_catalog.py
python3 tools/export_aistock_model_catalog.py
python3 tools/export_aistock_strategy_catalog.py
python3 tools/export_aistock_loop_catalog.py

# 同步到 AIstock
cp "$RDAGENT_ROOT/rdagent_data"/*.json "$AISTOCK_ROOT/rdagent_data/"
cp "$RDAGENT_ROOT/production_bundles/bundle_<new_task_run_id>_<new_loop_id>.tar.gz" "$AISTOCK_ROOT/rdagent_assets/bundles/"

# 导入到数据库
cd "$AISTOCK_ROOT/backend"
python3 -c "
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

1. 访问 AIstock 前端：`http://localhost:3000/rdagent/factors`
2. 点击"同步 Catalog"按钮
3. 等待同步完成

### 场景 3：通过 API 触发增量同步

```bash
# 触发全量同步
curl -X POST http://localhost:8000/api/rdagent/catalog/sync

# 触发增量同步（指定 last_sync_time）
curl -X POST http://localhost:8000/api/rdagent/catalog/sync?last_sync_time=2025-01-07T00:00:00Z
```

---

## 验证检查清单

### 环境兼容性验证

- [ ] Python 脚本在 WSL 环境正常运行
- [ ] Python 脚本在 Windows 环境正常运行
- [ ] 路径转换正确（WSL ↔ Windows）
- [ ] 文件操作跨平台兼容

### 历史数据处理验证

- [ ] 所有 Loop 都能导出至少 `model_meta.json`
- [ ] 所有 Loop 都能导出至少 `strategy_meta.json`
- [ ] 有结果的 Loop 能导出完整的性能数据
- [ ] 无结果的 Loop 不会导致脚本失败

### 强关联关系验证

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

## 总结

### 环境兼容性

✅ **完全支持 WSL 和 Windows 环境**

- Python 脚本使用 `_to_native_path()` 自动转换路径
- 使用 `Path` 对象，跨平台兼容
- Shell 命令需要区分环境（文档已提供两种环境版本）

### 历史数据处理能力

✅ **能够处理所有历史 Loop**

- `has_result=True`：导出完整元数据和性能数据
- `has_result=False`：导出基础元数据（`model_meta.json`、`strategy_meta.json`）
- 脚本不会因为缺少数据而失败

### 初始化步骤

✅ **提供完整的跨平台初始化步骤**

- WSL 环境初始化步骤（8 步）
- Windows 环境初始化步骤（8 步）
- 增量补充步骤（3 种场景）

按照文档操作即可完成 RD-Agent × AIstock 的跨平台初始化和后续增量补充。
