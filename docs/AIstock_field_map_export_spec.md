# AIstock -> RD-Agent 字段含义映射 CSV 导出规范

本文档用于指导 **AIstock 侧**导出一份“字段含义映射表”，供 **RD-Agent** 的脚本 `tools/generate_static_factors_bundle.py` 读取，以便在不修改任何 `.h5` 文件格式的前提下，将字段中文含义/单位/口径写入 RD-Agent 生成的 `static_factors_schema.csv/json`。

## 1. 导出目标

- **目标文件名（固定）**：`aistock_field_map.csv`
- **保存位置（推荐，固定）**：
  - Windows 路径：`C:/Users/lc999/NewAIstock/AIstock/metadata/aistock_field_map.csv`
  - WSL 路径：`/mnt/c/Users/lc999/NewAIstock/AIstock/metadata/aistock_field_map.csv`

说明：
- 若 AIstock 希望改路径/文件名，也可以，但 RD-Agent 执行生成脚本时需要用 `--field-map` 指向该文件。

## 2. 覆盖范围

导出的映射表需要至少覆盖以下两类数据源里出现的字段：

- `daily_basic.h5` 的所有列
- `moneyflow.h5` 的所有列

若能额外覆盖 RD-Agent 合并进 `static_factors.parquet` 的其他表（如预计算因子表）也可，但非必须。

## 3. 字段命名必须严格一致（最关键）

映射表中的字段名必须与 RD-Agent 侧将要合并的 DataFrame 列名 **完全一致**（区分大小写、下划线、前缀）。

- 例如：`db_pe_ttm`、`db_circ_mv`、`mf_net_amt` 等。
- 如果 AIstock 的原始数据库字段名不是以 `db_` / `mf_` 前缀组织，请在导出 CSV 时完成**映射/重命名**，保证最终 `name` 与 `.h5` 的列名一致。

## 4. CSV 文件格式要求

- 文件编码：`UTF-8`（建议无 BOM）
- 分隔符：逗号 `,`
- 第一行为表头

### 4.1 必填列

| 列名 | 类型 | 含义 |
| --- | --- | --- |
| `name` | string | **字段名**，必须与 `.h5` 列名一致 |
| `meaning_cn` | string | 字段中文含义，建议来自 DB 字段 comment |

### 4.2 强烈建议列（可选但推荐）

| 列名 | 类型 | 含义 |
| --- | --- | --- |
| `unit` | string | 单位，例如：`元`、`股`、`手`、`%`、`万元` 等 |
| `source_table` | string | 来源表/数据域，例如：`daily_basic`、`moneyflow` |
| `comment` | string | 原始 DB comment（可与 `meaning_cn` 相同，用于保留原文） |
| `dtype_hint` | string | 类型提示（如 `float64`/`int64`），可选；最终 dtype 以 RD-Agent 读到的 parquet 为准 |

### 4.3 最小示例

```csv
name,meaning_cn,unit,source_table,comment
db_pe_ttm,市盈率TTM,,daily_basic,市盈率TTM
db_circ_mv,流通市值,元,daily_basic,流通市值
mf_net_amt,资金净流入金额,元,moneyflow,买入金额-卖出金额
```

## 5. 生成逻辑建议（AIstock 侧实现参考）

AIstock 可按以下思路从数据库表结构导出：

- 遍历 `daily_basic`、`moneyflow`（或实际对应的源表）字段列表
- 读取字段 comment（中文含义）
- 输出到 CSV：
  - `name`：映射为 `.h5` 中的最终列名（含 `db_` / `mf_` 前缀）
  - `meaning_cn`：comment 文本（去除换行，trim）
  - `unit`：如果能从元数据中推断/维护则填
  - `source_table`：`daily_basic` 或 `moneyflow`

## 6. RD-Agent 侧如何使用（给 AIstock 侧验收用）

RD-Agent 侧执行：

```bash
python tools/generate_static_factors_bundle.py \
  --field-map /mnt/c/Users/lc999/NewAIstock/AIstock/metadata/aistock_field_map.csv
```

执行完成后，RD-Agent 会输出：

- `git_ignore_folder/factor_implementation_source_data/static_factors_schema.csv`
- `git_ignore_folder/factor_implementation_source_data/static_factors_schema.json`

其中 schema 的 `meaning` 列会被 `meaning_cn` 补全，并可能额外包含 `unit`、`source_table`。

## 7. 常见问题与约束

- 不修改 `.h5`：本方案完全不需要改 `.h5` 的结构或写入 attribute。
- 字段名不一致：这是最常见失败原因。请确保导出的 `name` 与 `.h5` 列名一致。
- 同名字段冲突：若同一 `name` 出现多次，RD-Agent 会以最后一条为准（建议 AIstock 侧保证唯一）。
