# Factor Implementation Contract (CoSTEER / RD-Agent)

## I/O Contract
- Input: read `daily_pv.h5` (key=`data`) in the **current working directory**.
- Optional input: read and join `static_factors.parquet` from the current working directory.
- Output: write factor values to `result.h5` (key=`data`, mode=`w`) in the current working directory.

## Index & Shape
- All factor outputs must preserve the same MultiIndex shape as the source `daily_pv.h5` dataframe:
  - Index: `MultiIndex(datetime, instrument)`
  - Set `result_df.index.names = df.index.names` (inherit, do not hardcode names).

## Column Naming
- Each factor column name must be English and contain no spaces (use underscores).
- Values must be `float64`.

## Column Normalization
- Recommended pattern:
  - rename `$open/$close/...` to `open/close/...` once at the top
  - all subsequent logic uses normalized names.

## Time-Series Safety
- Avoid look-ahead bias:
  - do not use `shift(-k)` for factor features
  - when computing rolling stats over returns, prefer `ret.shift(1).rolling(window=n)`
- Do not use any future rows or future outcomes as feature inputs.
- Forbidden descriptions include any phrasing like:
  - "use next day"
  - "tomorrow's price/close"
  - "future return"
  - "future price"
  - "t+1" as a feature input

## Numerical Safety
- For divisions, guard denominator with small epsilon.
- Allow NaNs for early window periods; do not fabricate values.

## Dependencies
- Allowed: stdlib, `numpy`, `pandas`.
- Not allowed: `h5py`.
- Avoid extra libraries unless guaranteed installed.

## Error Handling
- Do not wrap the entire main logic in try/except.
- When required columns are missing, raise a clear exception like:
  - `ValueError("Missing columns: [...]")`
