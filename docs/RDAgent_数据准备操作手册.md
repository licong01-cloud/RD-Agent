# RDAgent 数据准备操作手册

> 本手册详细描述从 AIstock 侧导出 H5 和 bin 文件后，如何在 RDAgent 侧完成全部的数据准备工作。

---

## 目录

1. [前置条件](#1-前置条件)
2. [AIstock 侧数据导出要求](#2-aistock-侧数据导出要求)
3. [数据准备流程概览](#3-数据准备流程概览)
4. [详细操作步骤](#4-详细操作步骤)
5. [验证与检查](#5-验证与检查)
6. [常见问题与解决方案](#6-常见问题与解决方案)
7. [附录](#7-附录)

---

## 1. 前置条件

### 1.1 环境要求

- **操作系统**：Windows 10/11 或 WSL2
- **Python 版本**：3.9+
- **依赖包**：
  - pandas >= 1.5.0
  - numpy >= 1.23.0
  - tables >= 3.8.0
  - pyarrow >= 12.0.0
  - qlib >= 0.9.0

### 1.2 目录结构约定

```
F:/Dev/
├── AIstock/                          # AIstock 项目根目录
│   ├── qlib_snapshots/               # H5 快照目录
│   │   └── qlib_export_20251209/    # 具体快照版本
│   │       ├── daily_pv.h5          # 日线价量
│   │       ├── moneyflow.h5         # 资金流
│   │       ├── daily_basic.h5       # 每日指标
│   │       ├── instruments/         # 标的信息
│   │       │   └── all.txt
│   │       ├── calendars/           # 交易日历
│   │       │   └── day.txt
│   │       └── metadata/            # 元数据（可选）
│   │           └── aistock_field_map.csv
│   ├── qlib_bin/                    # Qlib bin 目录
│   │   └── qlib_bin_20251209/
│   └── factors/                     # 预计算因子输出目录
│       ├── daily_basic_factors/
│       ├── moneyflow_factors/
│       └── ae_recon_error_10d/
│
└── RD-Agent-main/                   # RDAgent 项目根目录
    ├── git_ignore_folder/
    │   └── factor_implementation_source_data/       # 全量数据目录
    │       ├── daily_pv.h5
    │       ├── daily_basic.h5
    │       ├── moneyflow.h5
    │       ├── static_factors.parquet
    │       ├── static_factors_schema.csv
    │       └── static_factors_schema.json
    │   └── factor_implementation_source_data_debug/ # 调试数据目录
    │       ├── daily_pv.h5
    │       ├── daily_basic.h5
    │       ├── moneyflow.h5
    │       ├── static_factors.parquet
    │       ├── static_factors_schema.csv
    │       └── static_factors_schema.json
    ├── tools/                       # 工具脚本目录
    │   ├── generate_static_factors_bundle.py
    │   ├── generate_factor_schemas.py
    │   └── precompute_moneyflow_factors.py
    ├── precompute_daily_basic_factors.py
    └── regenerate_debug_dataset.py
```

### 1.3 环境变量配置

在 `.env` 文件或系统环境变量中配置：

```bash
# AIstock 数据路径（WSL 格式）
QLIB_DATA_PATH=/mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209
AIstock_SNAPSHOT_ROOT=/mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209
AISTOCK_FACTORS_ROOT=/mnt/f/Dev/AIstock/factors
AISTOCK_DATA_GOVERNANCE_DIR=/mnt/f/Dev/AIstock/data_governance

# Qlib bin 路径
QLIB_BIN_ROOT_WIN=F:/Dev/AIstock/qlib_bin/qlib_bin_20251209
QLIB_BIN_ROOT_WSL=/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209

# RD-Agent 路径
QLIB_RDAGENT_ROOT_WIN=F:/Dev/RD-Agent-main
QLIB_RDAGENT_ROOT_WSL=/mnt/f/Dev/RD-Agent-main
```

---

## 2. AIstock 侧数据导出要求

### 2.1 H5 文件要求

#### 2.1.1 必需文件

| 文件名 | 用途 | 数据结构 | 时间范围要求 |
|--------|------|---------|-------------|
| `daily_pv.h5` | 日线价量数据 | MultiIndex(datetime, instrument)<br>Columns: open, close, high, low, volume, factor, amount | 建议 2010 年之后 |
| `moneyflow.h5` | 资金流数据 | MultiIndex(datetime, instrument)<br>Columns: mf_* 系列（如 mf_sm_buy_vol, mf_lg_buy_amt 等） | 建议 2010 年之后 |
| `daily_basic.h5` | 每日指标数据 | MultiIndex(datetime, instrument)<br>Columns: db_* 系列（如 db_pe, db_pb, db_total_mv 等） | 建议 2010 年之后 |

#### 2.1.2 数据格式要求

**索引格式**：
- 必须是 `pandas.MultiIndex`
- 索引名称必须是 `['datetime', 'instrument']`
- `datetime` 必须是 `datetime64[ns]` 类型
- `instrument` 必须是标准 Qlib 格式（如 `000001.SZ`、`600000.SH`）

**列名规范**：
- 价格/成交量列：`open`, `close`, `high`, `low`, `volume`, `factor`, `amount`
- 资金流列：以 `mf_` 前缀开头（如 `mf_sm_buy_vol`）
- 基本面列：以 `db_` 前缀开头（如 `db_pe`, `db_pb`）

**数据类型**：
- 数值列必须是 `float64` 或 `float32`
- 不允许有 `object` 或 `string` 类型的数据列

#### 2.1.3 数据完整性要求

- ✅ **无缺失索引**：所有 (datetime, instrument) 组合必须有数据
- ✅ **无重复索引**：MultiIndex 必须唯一
- ✅ **时间连续性**：每个股票的时间序列应连续（允许停牌日缺失）
- ✅ **股票代码规范**：必须使用 Qlib 标准格式（`000001.SZ`、`600000.SH`）

### 2.2 Qlib Bin 文件要求

#### 2.2.1 必需目录结构

```
qlib_bin_20251209/
├── instruments/
│   ├── all.txt              # 全部标的列表
│   ├── sh_stock.txt         # 上交所股票
│   └── sz_stock.txt         # 深交所股票
├── calendars/
│   └── day.txt              # 交易日历
└── market_data/
    └── cn_stock/
        ├── day/
        │   ├── open/
        │   ├── close/
        │   ├── high/
        │   ├── low/
        │   ├── volume/
        │   └── factor/
        └── min/
            └── 1min/        # 可选：1 分钟数据
```

#### 2.2.2 Bin 文件格式要求

- 使用 Qlib 标准的 `bin` 格式（`np.float32`）
- 文件命名：`{instrument}.bin`
- 数据按日期排序
- 缺失值使用 `np.nan`

### 2.3 元数据文件要求（可选但推荐）

#### 2.3.1 字段映射文件

**文件路径**：`metadata/aistock_field_map.csv`

**格式要求**：

```csv
name,meaning_cn,unit,source_table,comment
mf_sm_buy_vol,小单买入量,股,moneyflow,
mf_lg_buy_amt,大单买入金额,元,moneyflow,
db_pe,市盈率,倍,daily_basic,
db_pb,市净率,倍,daily_basic,
```

**字段说明**：
- `name`：字段名（必须与 H5 文件中的列名一致）
- `meaning_cn`：中文含义
- `unit`：单位
- `source_table`：来源表（`daily_basic`、`moneyflow`）
- `comment`：备注（可选）

---

## 3. 数据准备流程概览

### 3.1 流程图

```
AIstock 侧导出
    │
    ├─→ H5 快照 (daily_pv.h5, moneyflow.h5, daily_basic.h5)
    │
    ├─→ Qlib Bin 文件
    │
    └─→ 元数据 (aistock_field_map.csv)
         │
         ▼
┌─────────────────────────────────────────┐
│  步骤 1: 预计算因子                     │
│  - daily_basic_factors                  │
│  - moneyflow_factors                    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  步骤 2: 生成静态因子 Bundle            │
│  - static_factors.parquet               │
│  - static_factors_schema.csv/json       │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  步骤 3: 生成 Debug 数据集              │
│  - 从全量数据提取子集                   │
│  - 100 只股票，2 年数据                 │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  步骤 4: 生成 Factor Set Schema         │
│  - factor set 元数据                    │
│  - 写入 AIstock data_governance         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  步骤 5: 验证数据完整性                 │
│  - 检查文件格式                         │
│  - 检查数据范围                         │
│  - 检查股票集一致性                     │
└─────────────────────────────────────────┘
         │
         ▼
    ✅ 数据准备完成
```

### 3.2 步骤说明

| 步骤 | 操作 | 脚本 | 输入 | 输出 | 预计耗时 |
|------|------|------|------|------|---------|
| 1 | 预计算 daily_basic 因子 | `precompute_daily_basic_factors.py` | `daily_basic.h5` | `daily_basic_factors/result.h5/pkl/parquet` | 5-10 分钟 |
| 2 | 预计算 moneyflow 因子 | `tools/precompute_moneyflow_factors.py` | `moneyflow.h5` + `daily_pv.h5` | `moneyflow_factors/result.h5/pkl/parquet` | 10-20 分钟 |
| 3 | 生成静态因子 Bundle | `tools/generate_static_factors_bundle.py` | `daily_basic.h5` + `moneyflow.h5` + 预计算因子 | `static_factors.parquet` + schema | 15-30 分钟 |
| 4 | 生成 Debug 数据集 | `regenerate_debug_dataset.py` | 全量数据 | Debug 数据集 | 5-10 分钟 |
| 5 | 生成 Factor Set Schema | `tools/generate_factor_schemas.py` | 预计算因子 | Factor set schema | 5-10 分钟 |
| 6 | 验证数据完整性 | 手动检查或脚本 | 所有数据 | 验证报告 | 5-10 分钟 |

**总计耗时**：约 45-90 分钟（取决于数据量）

---

## 4. 详细操作步骤

### 步骤 1：预计算 daily_basic 因子

#### 1.1 脚本说明

**脚本路径**：`precompute_daily_basic_factors.py`

**功能**：从 `daily_basic.h5` 中预计算基本面因子，如 PE、PB、市值等衍生指标。

#### 1.2 操作步骤

```bash
# 进入 RDAgent 根目录
cd F:/Dev/RD-Agent-main

# 运行脚本
python precompute_daily_basic_factors.py
```

#### 1.3 输出文件

```
F:/Dev/AIstock/factors/daily_basic_factors/
├── result.h5          # HDF5 格式
├── result.pkl         # Pickle 格式
└── result.parquet     # Parquet 格式（推荐）
```

#### 1.4 验证输出

```bash
# 检查文件是否存在
ls -lh F:/Dev/AIstock/factors/daily_basic_factors/

# 检查数据形状
python -c "
import pandas as pd
df = pd.read_parquet('F:/Dev/AIstock/factors/daily_basic_factors/result.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Time range: {df.index.get_level_values(\"datetime\").min()} to {df.index.get_level_values(\"datetime\").max()}')
print(f'Instruments: {df.index.get_level_values(\"instrument\").nunique()}')
"
```

---

### 步骤 2：预计算 moneyflow 因子

#### 2.1 脚本说明

**脚本路径**：`tools/precompute_moneyflow_factors.py`

**功能**：从 `moneyflow.h5` 和 `daily_pv.h5` 中预计算资金流因子，如净流入、大单占比等。

#### 2.2 操作步骤

```bash
# 进入 RDAgent 根目录
cd F:/Dev/RD-Agent-main

# 运行脚本
python tools/precompute_moneyflow_factors.py
```

#### 2.3 输出文件

```
F:/Dev/AIstock/factors/moneyflow_factors/
├── result.h5          # HDF5 格式
├── result.pkl         # Pickle 格式
└── result.parquet     # Parquet 格式（推荐）
```

#### 2.4 验证输出

```bash
# 检查文件是否存在
ls -lh F:/Dev/AIstock/factors/moneyflow_factors/

# 检查数据形状
python -c "
import pandas as pd
df = pd.read_parquet('F:/Dev/AIstock/factors/moneyflow_factors/result.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Time range: {df.index.get_level_values(\"datetime\").min()} to {df.index.get_level_values(\"datetime\").max()}')
print(f'Instruments: {df.index.get_level_values(\"instrument\").nunique()}')
"
```

---

### 步骤 3：生成静态因子 Bundle

#### 3.1 脚本说明

**脚本路径**：`tools/generate_static_factors_bundle.py`

**功能**：
1. 合并原始 H5 数据（`daily_basic.h5`、`moneyflow.h5`）
2. 合并预计算因子（`daily_basic_factors`、`moneyflow_factors`）
3. 派生 rolling 聚合因子（5D、20D）
4. 生成 schema 文件（CSV + JSON）

#### 3.2 操作步骤

```bash
# 进入 RDAgent 根目录
cd F:/Dev/RD-Agent-main

# 基础运行（不使用字段映射）
python tools/generate_static_factors_bundle.py \
  --snapshot-root F:/Dev/AIstock/qlib_snapshots/qlib_export_20251209 \
  --aistock-factors-root F:/Dev/AIstock/factors

# 高级运行（使用字段映射）
python tools/generate_static_factors_bundle.py \
  --snapshot-root F:/Dev/AIstock/qlib_snapshots/qlib_export_20251209 \
  --aistock-factors-root F:/Dev/AIstock/factors \
  --field-map F:/Dev/AIstock/qlib_snapshots/qlib_export_20251209/metadata/aistock_field_map.csv

# 仅生成 schema（不重新生成 parquet）
python tools/generate_static_factors_bundle.py \
  --snapshot-root F:/Dev/AIstock/qlib_snapshots/qlib_export_20251209 \
  --aistock-factors-root F:/Dev/AIstock/factors \
  --schema-only
```

#### 3.3 输出文件

```
git_ignore_folder/factor_implementation_source_data/
├── static_factors.parquet              # 静态因子数据
├── static_factors_schema.csv          # Schema（CSV 格式）
├── static_factors_schema.json         # Schema（JSON 格式）
└── README.md                          # 说明文档

git_ignore_folder/factor_implementation_source_data_debug/
├── static_factors.parquet              # Debug 静态因子数据
├── static_factors_schema.csv          # Debug Schema（CSV 格式）
├── static_factors_schema.json         # Debug Schema（JSON 格式）
└── README.md                          # 说明文档
```

#### 3.4 验证输出

```bash
# 检查文件是否存在
ls -lh git_ignore_folder/factor_implementation_source_data/

# 检查数据形状
python -c "
import pandas as pd
df = pd.read_parquet('git_ignore_folder/factor_implementation_source_data/static_factors.parquet')
print(f'Shape: {df.shape}')
print(f'Columns count: {len(df.columns)}')
print(f'Time range: {df.index.get_level_values(\"datetime\").min()} to {df.index.get_level_values(\"datetime\").max()}')
print(f'Instruments: {df.index.get_level_values(\"instrument\").nunique()}')
"

# 检查 schema
python -c "
import json
with open('git_ignore_folder/factor_implementation_source_data/static_factors_schema.json', 'r') as f:
    schema = json.load(f)
print(f'Schema fields: {len(schema)}')
print(f'Sample fields: {schema[:3]}')
"
```

---

### 步骤 4：生成 Debug 数据集

#### 4.1 脚本说明

**脚本路径**：`regenerate_debug_dataset.py`

**功能**：从全量数据中提取子集，生成用于调试的小数据集：
- 时间范围：2018-01-01 至 2019-12-31
- 股票数量：100 只
- 文件：`daily_pv.h5`、`daily_basic.h5`、`moneyflow.h5`

#### 4.2 操作步骤

```bash
# 进入 RDAgent 根目录
cd F:/Dev/RD-Agent-main

# 运行脚本
python regenerate_debug_dataset.py
```

#### 4.3 输出文件

```
git_ignore_folder/factor_implementation_source_data_debug/
├── daily_pv.h5                          # Debug 日线价量
├── daily_basic.h5                       # Debug 每日指标
├── moneyflow.h5                         # Debug 资金流
├── daily_pv.h5.backup                   # 备份文件
├── daily_basic.h5.backup                # 备份文件
└── moneyflow.h5.backup                  # 备份文件
```

#### 4.4 验证输出

```bash
# 检查文件是否存在
ls -lh git_ignore_folder/factor_implementation_source_data_debug/

# 检查数据形状
python -c "
import pandas as pd
df = pd.read_hdf('git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5', key='data')
print(f'Shape: {df.shape}')
print(f'Time range: {df.index.get_level_values(\"datetime\").min()} to {df.index.get_level_values(\"datetime\").max()}')
print(f'Instruments: {df.index.get_level_values(\"instrument\").nunique()}')
"
```

---

### 步骤 5：生成 Factor Set Schema

#### 5.1 脚本说明

**脚本路径**：`tools/generate_factor_schemas.py`

**功能**：为预计算的因子集合生成 schema 元数据，写入 AIstock 的 data_governance 目录。

#### 5.2 操作步骤

```bash
# 进入 RDAgent 根目录
cd F:/Dev/RD-Agent-main

# 运行脚本
python tools/generate_factor_schemas.py \
  --factors-root F:/Dev/AIstock/factors \
  --out-dir-governance F:/Dev/AIstock/data_governance
```

#### 5.3 输出文件

```
F:/Dev/AIstock/data_governance/
└── factor_sets/
    ├── daily_basic_factors/
    │   └── schema.json
    ├── moneyflow_factors/
    │   └── schema.json
    └── ae_recon_error_10d/
        └── schema.json
```

#### 5.4 验证输出

```bash
# 检查文件是否存在
ls -lh F:/Dev/AIstock/data_governance/factor_sets/

# 检查 schema 内容
python -c "
import json
with open('F:/Dev/AIstock/data_governance/factor_sets/daily_basic_factors/schema.json', 'r') as f:
    schema = json.load(f)
print(f'Factor set: {schema.get(\"name\")}')
print(f'Description: {schema.get(\"description\")}')
print(f'Fields count: {len(schema.get(\"fields\", []))}')
"
```

---

### 步骤 6：验证数据完整性

#### 6.1 验证脚本

创建验证脚本 `verify_data_preparation.py`：

```python
#!/usr/bin/env python3
"""验证数据准备完整性"""

import pandas as pd
from pathlib import Path

def verify_h5_file(path: Path, name: str) -> bool:
    """验证 H5 文件"""
    print(f"\n验证 {name}: {path}")
    
    if not path.exists():
        print(f"  ❌ 文件不存在")
        return False
    
    try:
        df = pd.read_hdf(path, key="data")
        
        # 检查索引
        if not isinstance(df.index, pd.MultiIndex):
            print(f"  ❌ 索引不是 MultiIndex")
            return False
        
        if list(df.index.names) != ["datetime", "instrument"]:
            print(f"  ❌ 索引名称不正确: {df.index.names}")
            return False
        
        # 检查数据
        if df.empty:
            print(f"  ❌ 数据为空")
            return False
        
        print(f"  ✅ 形状: {df.shape}")
        print(f"  ✅ 时间范围: {df.index.get_level_values('datetime').min()} 至 {df.index.get_level_values('datetime').max()}")
        print(f"  ✅ 股票数量: {df.index.get_level_values('instrument').nunique()}")
        print(f"  ✅ 列数: {len(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return False

def verify_parquet_file(path: Path, name: str) -> bool:
    """验证 Parquet 文件"""
    print(f"\n验证 {name}: {path}")
    
    if not path.exists():
        print(f"  ❌ 文件不存在")
        return False
    
    try:
        df = pd.read_parquet(path)
        
        # 检查索引
        if not isinstance(df.index, pd.MultiIndex):
            print(f"  ❌ 索引不是 MultiIndex")
            return False
        
        if list(df.index.names) != ["datetime", "instrument"]:
            print(f"  ❌ 索引名称不正确: {df.index.names}")
            return False
        
        # 检查数据
        if df.empty:
            print(f"  ❌ 数据为空")
            return False
        
        print(f"  ✅ 形状: {df.shape}")
        print(f"  ✅ 时间范围: {df.index.get_level_values('datetime').min()} 至 {df.index.get_level_values('datetime').max()}")
        print(f"  ✅ 股票数量: {df.index.get_level_values('instrument').nunique()}")
        print(f"  ✅ 列数: {len(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("验证数据准备完整性")
    print("=" * 80)
    
    results = {}
    
    # 验证全量数据
    full_data_dir = Path("git_ignore_folder/factor_implementation_source_data")
    results["daily_pv.h5"] = verify_h5_file(full_data_dir / "daily_pv.h5", "daily_pv.h5")
    results["daily_basic.h5"] = verify_h5_file(full_data_dir / "daily_basic.h5", "daily_basic.h5")
    results["moneyflow.h5"] = verify_h5_file(full_data_dir / "moneyflow.h5", "moneyflow.h5")
    results["static_factors.parquet"] = verify_parquet_file(full_data_dir / "static_factors.parquet", "static_factors.parquet")
    
    # 验证 debug 数据
    debug_data_dir = Path("git_ignore_folder/factor_implementation_source_data_debug")
    results["debug_daily_pv.h5"] = verify_h5_file(debug_data_dir / "daily_pv.h5", "debug_daily_pv.h5")
    results["debug_daily_basic.h5"] = verify_h5_file(debug_data_dir / "daily_basic.h5", "debug_daily_basic.h5")
    results["debug_moneyflow.h5"] = verify_h5_file(debug_data_dir / "moneyflow.h5", "debug_moneyflow.h5")
    results["debug_static_factors.parquet"] = verify_parquet_file(debug_data_dir / "static_factors.parquet", "debug_static_factors.parquet")
    
    # 输出总结
    print(f"\n{'=' * 80}")
    print("验证总结")
    print(f"{'=' * 80}")
    
    for name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name}: {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print(f"\n✅ 所有验证通过！数据准备完成。")
    else:
        print(f"\n⚠️  部分验证失败，请检查上述日志。")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    exit(main())
```

#### 6.2 运行验证

```bash
# 运行验证脚本
python verify_data_preparation.py
```

---

## 5. 验证与检查

### 5.1 数据一致性检查

#### 5.1.1 股票集一致性

```bash
python -c "
import pandas as pd

# 读取全量数据
df_pv = pd.read_hdf('git_ignore_folder/factor_implementation_source_data/daily_pv.h5', key='data')
df_basic = pd.read_hdf('git_ignore_folder/factor_implementation_source_data/daily_basic.h5', key='data')
df_mf = pd.read_hdf('git_ignore_folder/factor_implementation_source_data/moneyflow.h5', key='data')

# 获取股票集
instruments_pv = set(df_pv.index.get_level_values('instrument').unique())
instruments_basic = set(df_basic.index.get_level_values('instrument').unique())
instruments_mf = set(df_mf.index.get_level_values('instrument').unique())

print(f'daily_pv 股票数: {len(instruments_pv)}')
print(f'daily_basic 股票数: {len(instruments_basic)}')
print(f'moneyflow 股票数: {len(instruments_mf)}')
print(f'共同股票: {len(instruments_pv & instruments_basic & instruments_mf)}')
print(f'daily_pv 独有: {len(instruments_pv - instruments_basic - instruments_mf)}')
print(f'daily_basic 独有: {len(instruments_basic - instruments_pv - instruments_mf)}')
print(f'moneyflow 独有: {len(instruments_mf - instruments_pv - instruments_basic)}')
"
```

#### 5.1.2 时间范围一致性

```bash
python -c "
import pandas as pd

# 读取全量数据
df_pv = pd.read_hdf('git_ignore_folder/factor_implementation_source_data/daily_pv.h5', key='data')
df_basic = pd.read_hdf('git_ignore_folder/factor_implementation_source_data/daily_basic.h5', key='data')
df_mf = pd.read_hdf('git_ignore_folder/factor_implementation_source_data/moneyflow.h5', key='data')

# 获取时间范围
print(f'daily_pv: {df_pv.index.get_level_values(\"datetime\").min()} 至 {df_pv.index.get_level_values(\"datetime\").max()}')
print(f'daily_basic: {df_basic.index.get_level_values(\"datetime\").min()} 至 {df_basic.index.get_level_values(\"datetime\").max()}')
print(f'moneyflow: {df_mf.index.get_level_values(\"datetime\").min()} 至 {df_mf.index.get_level_values(\"datetime\").max()}')
"
```

### 5.2 数据质量检查

#### 5.2.1 缺失值检查

```bash
python -c "
import pandas as pd

# 读取数据
df = pd.read_parquet('git_ignore_folder/factor_implementation_source_data/static_factors.parquet')

# 检查缺失值
missing_counts = df.isnull().sum()
missing_ratio = (missing_counts / len(df)) * 100

print('缺失值统计:')
for col in df.columns:
    if missing_counts[col] > 0:
        print(f'  {col}: {missing_counts[col]} ({missing_ratio[col]:.2f}%)')
"
```

#### 5.2.2 异常值检查

```bash
python -c "
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_parquet('git_ignore_folder/factor_implementation_source_data/static_factors.parquet')

# 检查无穷大值
inf_counts = np.isinf(df).sum()
print('无穷大值统计:')
for col in df.columns:
    if inf_counts[col] > 0:
        print(f'  {col}: {inf_counts[col]}')

# 检查极端值（超过 3 个标准差）
numeric_cols = df.select_dtypes(include=[np.number]).columns
print('\n极端值统计（超过 3 个标准差）:')
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    extreme = ((df[col] - mean).abs() > 3 * std).sum()
    if extreme > 0:
        print(f'  {col}: {extreme} ({extreme/len(df)*100:.2f}%)')
"
```

### 5.3 Schema 验证

```bash
python -c "
import json
import pandas as pd

# 读取 schema
with open('git_ignore_folder/factor_implementation_source_data/static_factors_schema.json', 'r') as f:
    schema = json.load(f)

# 读取数据
df = pd.read_parquet('git_ignore_folder/factor_implementation_source_data/static_factors.parquet')

# 验证列名匹配
schema_fields = {field['name'] for field in schema}
data_columns = set(df.columns)

print(f'Schema 字段数: {len(schema_fields)}')
print(f'数据列数: {len(data_columns)}')
print(f'共同字段: {len(schema_fields & data_columns)}')
print(f'Schema 独有: {len(schema_fields - data_columns)}')
print(f'数据独有: {len(data_columns - schema_fields)}')

if schema_fields - data_columns:
    print(f'\nSchema 独有字段: {schema_fields - data_columns}')
if data_columns - schema_fields:
    print(f'\n数据独有字段: {data_columns - schema_fields}')
"
```

---

## 6. 常见问题与解决方案

### 6.1 H5 文件读取失败

**问题**：`pandas.read_hdf` 报错 `KeyError: No object named data in the file`

**原因**：H5 文件的 key 名称不是 `data`

**解决方案**：

```bash
# 检查 H5 文件的 key
python -c "
import pandas as pd
with pd.HDFStore('path/to/file.h5') as store:
    print(store.keys())
"

# 使用正确的 key 读取
df = pd.read_hdf('path/to/file.h5', key='your_key')
```

### 6.2 MultiIndex 损坏

**问题**：读取 H5 文件后，MultiIndex 损坏或索引名称不正确

**原因**：H5 文件写入时索引格式不正确

**解决方案**：

```python
import pandas as pd

# 读取数据
df = pd.read_hdf('path/to/file.h5', key='data')

# 修复索引
if isinstance(df.index, pd.MultiIndex):
    df.index.set_names(['datetime', 'instrument'], inplace=True)
else:
    # 如果不是 MultiIndex，需要重新构建
    df = df.reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(['datetime', 'instrument']).sort_index()

# 重新写入
df.to_hdf('path/to/file.h5', key='data', mode='w')
```

### 6.3 内存不足

**问题**：处理大文件时出现 OOM（Out of Memory）

**原因**：数据量过大，内存不足

**解决方案**：

```python
import pandas as pd

# 分块读取和处理
chunk_size = 1000000  # 每块 100 万行
chunks = pd.read_hdf('path/to/large_file.h5', key='data', chunksize=chunk_size)

for chunk in chunks:
    # 处理每个 chunk
    processed = process_chunk(chunk)
    # 写入结果
    processed.to_parquet('output.parquet', append=True)
```

### 6.4 股票代码格式不一致

**问题**：股票代码格式不统一（如 `000001.SZ` vs `SZ000001`）

**原因**：不同数据源使用不同的股票代码格式

**解决方案**：

```python
import pandas as pd

# 标准化股票代码
def normalize_instrument(code: str) -> str:
    """标准化股票代码为 Qlib 格式"""
    code = str(code).strip()
    
    # SH600000 -> 600000.SH
    if len(code) == 9 and code[:2] in ['SH', 'SZ']:
        return code[2:] + '.' + code[:2]
    
    # 600000.SH -> 保持不变
    if '.' in code:
        return code
    
    # 600000 -> 假设是沪市
    if code.startswith('6'):
        return code + '.SH'
    # 000001 -> 假设是深市
    else:
        return code + '.SZ'

# 应用标准化
df.index = df.index.set_levels(
    df.index.get_level_values('instrument').map(normalize_instrument),
    level='instrument'
)
```

### 6.5 时间范围不匹配

**问题**：不同文件的时间范围不一致

**原因**：数据源更新时间不同

**解决方案**：

```python
import pandas as pd

# 读取多个文件
df_pv = pd.read_hdf('daily_pv.h5', key='data')
df_basic = pd.read_hdf('daily_basic.h5', key='data')
df_mf = pd.read_hdf('moneyflow.h5', key='data')

# 获取共同时间范围
start_date = max(
    df_pv.index.get_level_values('datetime').min(),
    df_basic.index.get_level_values('datetime').min(),
    df_mf.index.get_level_values('datetime').min()
)
end_date = min(
    df_pv.index.get_level_values('datetime').max(),
    df_basic.index.get_level_values('datetime').max(),
    df_mf.index.get_level_values('datetime').max()
)

# 筛选共同时间范围
df_pv = df_pv.loc[start_date:end_date]
df_basic = df_basic.loc[start_date:end_date]
df_mf = df_mf.loc[start_date:end_date]
```

---

## 7. 附录

### 7.1 环境变量完整列表

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `QLIB_DATA_PATH` | Qlib 数据路径（H5） | `/mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209` |
| `AIstock_SNAPSHOT_ROOT` | AIstock 快照根目录 | `/mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209` |
| `AISTOCK_FACTORS_ROOT` | AIstock 因子输出目录 | `/mnt/f/Dev/AIstock/factors` |
| `AISTOCK_DATA_GOVERNANCE_DIR` | AIstock 数据治理目录 | `/mnt/f/Dev/AIstock/data_governance` |
| `QLIB_BIN_ROOT_WIN` | Qlib bin 路径（Windows） | `F:/Dev/AIstock/qlib_bin/qlib_bin_20251209` |
| `QLIB_BIN_ROOT_WSL` | Qlib bin 路径（WSL） | `/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209` |
| `QLIB_RDAGENT_ROOT_WIN` | RD-Agent 根目录（Windows） | `F:/Dev/RD-Agent-main` |
| `QLIB_RDAGENT_ROOT_WSL` | RD-Agent 根目录（WSL） | `/mnt/f/Dev/RD-Agent-main` |

### 7.2 相关文档

- [AIstock Qlib 数据集使用备忘录](./AIstock_Qlib_数据集使用备忘录.md)
- [2025-12-21 AIstock 因子数据链路全流程](./2025-12-21_AIstock因子数据链路全流程.md)
- [AIstock field map 导出规范](./AIstock_field_map_export_spec.md)
- [20251214 因子演进 debug static_factors rolling retry 备忘录](./20251214_因子演进_debug_static_factors_rolling_retry备忘录.md)

### 7.3 脚本清单

| 脚本路径 | 功能 | 依赖 |
|---------|------|------|
| `precompute_daily_basic_factors.py` | 预计算基本面因子 | `daily_basic.h5` |
| `tools/precompute_moneyflow_factors.py` | 预计算资金流因子 | `moneyflow.h5`, `daily_pv.h5` |
| `tools/generate_static_factors_bundle.py` | 生成静态因子 Bundle | `daily_basic.h5`, `moneyflow.h5`, 预计算因子 |
| `regenerate_debug_dataset.py` | 生成 Debug 数据集 | 全量数据 |
| `tools/generate_factor_schemas.py` | 生成 Factor Set Schema | 预计算因子 |

### 7.4 文件格式对照表

| 文件类型 | 格式 | 读取方式 | 写入方式 |
|---------|------|---------|---------|
| H5 | HDF5 | `pd.read_hdf(path, key='data')` | `df.to_hdf(path, key='data', mode='w')` |
| Parquet | Parquet | `pd.read_parquet(path)` | `df.to_parquet(path)` |
| CSV | CSV | `pd.read_csv(path)` | `df.to_csv(path)` |
| JSON | JSON | `json.load(open(path))` | `json.dump(data, open(path, 'w'))` |

### 7.5 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-01-07 | 初始版本 |

---

## 联系与支持

如有问题，请联系开发团队或查阅相关文档。
