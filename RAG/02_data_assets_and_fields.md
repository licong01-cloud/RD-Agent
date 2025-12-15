# Data Assets & Field Naming (Ground Truth)

This document summarizes the *actual* data assets and the naming conventions that factor/model code must follow.

## Core Snapshot Panels (Factor R&D)
### daily_pv.h5
- Format: HDF5 (key usually `data`)
- Index: `MultiIndex(datetime, instrument)`
- Columns (Qlib-style, common):
  - `$open`, `$high`, `$low`, `$close`, `$volume`, `$amount`, `$factor` (if present)
- In factor scripts, columns should be normalized to **no `$` prefix** (e.g., `close`, `volume`).

### moneyflow.h5
- Index: `MultiIndex(datetime, instrument)`
- Columns: `mf_*` fields (e.g., `mf_net_amt`, `mf_net_vol`, segmented buy/sell amounts/volumes, etc.)

### daily_basic.h5
- Index: `MultiIndex(datetime, instrument)`
- Columns: `db_*` fields (valuation, market cap, turnover/volume ratio, etc.)
- Note: some values may already be converted to base units during export (e.g., market cap 万元→元).

## Unified Static Panel for Factor Runtime
### static_factors.parquet
- A unified table used by factor scripts at runtime.
- It is expected to include `db_*` and `mf_*` columns and optional derived rolling columns (e.g., `*_5d`, `*_20d`).

### static_factors_schema.csv / static_factors_schema.json
- Schema/whitelist of available fields.
- Factor generation should treat the schema as the authoritative field list.

## Field Meaning Map (AIstock → RD-Agent)
### aistock_field_map.csv
- Exported by AIstock and read by `tools/generate_static_factors_bundle.py`.
- Minimum required columns:
  - `name`: must exactly match the final dataframe column name used by RD-Agent (`db_*`, `mf_*`, etc.)
  - `meaning_cn`: Chinese meaning
- Recommended optional columns: `unit`, `source_table`, `comment`, `dtype_hint`.

## Naming Rules
- Column names are case-sensitive and must match exactly.
- Use prefixes:
  - price/volume panel normalized names: `open/high/low/close/volume/amount/factor`
  - daily_basic: `db_*`
  - moneyflow: `mf_*`
- Do not invent names. If unsure, do not use the field.
