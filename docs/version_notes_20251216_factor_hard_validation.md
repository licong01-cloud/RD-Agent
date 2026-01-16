# Version Notes (20251216) - Factor Hard Validation & RAG Rules

## Summary
- Added minimal hard validation for factor execution outputs (`result.h5`) in `FactorFBWorkspace.execute()`.
- Added a strict ban on look-ahead operations via `shift(-k)` (k>0) static scan before running factor code.
- Updated RAG knowledge base documents to clarify:
  - allowed factor runtime data assets (whitelist)
  - future-row / future-operation ban (including forbidden phrasing)
  - recommendation to prefer `static_factors.parquet` (+ schema) for `db_*`/`mf_*` fields (not mandatory)

## Code Changes
### Factor runtime hard validation
- File: `rdagent/components/coder/factor_coder/factor.py`
- Behavior:
  - Refuse to run `factor.py` if it contains `shift(-k)` (k>0).
  - After reading `result.h5`, validate:
    - DataFrame not empty
    - Index is `MultiIndex` and includes `datetime` and `instrument`
    - `datetime` is parseable to timestamps
    - Output max datetime does not exceed allowed max datetime
      - prefer using the last row of `daily_pv.h5` when accessible; otherwise fall back to today's date
    - Output is not too sparse: `non_nan_row_ratio >= QLIB_FACTOR_MIN_NON_NAN_ROW_RATIO`
      - default: `0.10`

## RAG Document Updates
- `RAG/01_system_hard_rules.md`
  - Added factor runtime allowed assets whitelist.
  - Added future-row / future-operation ban (including forbidden phrasing).
- `RAG/02_data_assets_and_fields.md`
  - Clarified schema as authoritative field list and meanings.
  - Added recommendation to prefer `static_factors.parquet` for `db_*`/`mf_*` fields (not mandatory).
- `RAG/03_factor_implementation_contract.md`
  - Added explicit ban on future-row / future-operation descriptions (in addition to `shift(-k)` ban).

## Configuration
- `QLIB_FACTOR_MIN_NON_NAN_ROW_RATIO`
  - Default: `0.10`
  - Meaning: minimum ratio of rows where at least one factor column is non-NaN.

## Notes / Known Limitations
- This change does not sandbox filesystem access; it focuses on output contract validation and basic leakage prevention signals.
- The `daily_pv.h5` max datetime check attempts a light-weight read; if unavailable, the check falls back to today's date.
