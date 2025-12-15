# System Hard Rules (AIstock ↔ RD-Agent ↔ Qlib)

## Rule Priority
- If external quant articles conflict with local rules, **local rules win**.
- If upstream RD-Agent docs conflict with local integration constraints, **local integration constraints win**.

## Data Source Split
- **Qlib bin** is the authoritative data source for backtest/model training.
  - Current provider_uri points to: `.../qlib_bin/qlib_bin_20251209` (WSL path used in templates).
- **HDF5 snapshots + Parquet** are the authoritative sources for factor implementation scripts.
  - Factor scripts read `daily_pv.h5` from the current working directory.
  - Static/fund-flow/basic panels should be provided via `static_factors.parquet` (+ schema) in the factor runtime folder.

## Market / Universe
- Use `market: all` as the unified universe.
- Universe is exported with filters already applied:
  - ST / *ST excluded
  - delisted / suspended excluded
- Factor code should not re-filter by ST/delisting unless explicitly required by strategy design.

## Time Range / Segments
- Templates use long-range data with train/valid/test segments and a backtest end date slightly before last trading day to avoid qlib calendar edge issues.

## Output & Format Constraints (LLM-facing)
- When a step requires JSON, output **only valid JSON**.
- Do not invent field names. Only use fields present in the runtime schema / field map.

## Engineering Constraints (Factor Implementation)
- Do not import `h5py`.
- Avoid broad try/except wrapping main logic; let errors surface for logs/auto-repair.
- Use `pandas.DataFrame.to_hdf` to write `result.h5`.
