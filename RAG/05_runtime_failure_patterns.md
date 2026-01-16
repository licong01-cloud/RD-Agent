# Runtime Failure Patterns & Fixes (from logs)

This document is a distilled checklist of the most common failure patterns observed in RD-Agent multi-round runs.

## Pattern A: Missing Columns / KeyError
- **Symptom**: `KeyError`, `Missing columns`, `not in index`
- **Root cause**:
  - hallucinated field names
  - schema not visible in the run folder
  - wrong join panel (using `daily_pv` only when field is in `static_factors`)
- **Fix**:
  - enforce schema whitelist
  - join `static_factors.parquet` before referencing `db_*`/`mf_*`

## Pattern B: All NaN / Empty DataFrame
- **Symptom**: output factor column all NaN; value evaluator reports empty
- **Root cause**:
  - rolling windows with min_periods too strict combined with short sample
  - denominator zero / missing values
  - index misalignment after unstack/stack
- **Fix**:
  - keep index alignment, use `reindex(df.index)`
  - add epsilon for ratio denominators
  - allow NaNs only for early windows, ensure later periods have values

## Pattern C: Index Misalignment (boolean mask mismatch)
- **Symptom**: `IndexError: boolean index did not match indexed array`
- **Root cause**: losing index when converting to numpy arrays (e.g., scaler outputs)
- **Fix**:
  - avoid converting to raw numpy without preserving index
  - if using numpy arrays, carefully align lengths and map back to MultiIndex

## Pattern D: Using Unsupported APIs
- **Symptom**: `AttributeError` (e.g. Series has no `reshape`), or using nonexistent pandas rolling methods
- **Fix**:
  - stick to known pandas/numpy APIs; avoid inventing methods

## Pattern E: Files Not Found
- **Symptom**: `FileNotFoundError` for expected factor outputs
- **Root cause**: wrong relative paths or missing data bundle copy into run folder
- **Fix**:
  - ensure factor run folder contains `daily_pv.h5` and `static_factors.parquet` (+ schema)

## Auto-Repair Note
- Auto-repair only triggers when failure signatures match eligibility checks.
- Keep error messages explicit (missing columns/all NaN/empty dataframe) to improve repair hit-rate.
