"""Compare tabular backtest artifacts with explicit keys and tolerances."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ComparisonInputError(RuntimeError):
    pass


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ComparisonInputError(f"input file does not exist: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix in {".pkl", ".pickle"}:
            frame = pd.read_pickle(path)
        elif suffix == ".json":
            frame = pd.read_json(path)
        else:
            raise ComparisonInputError(f"unsupported input format: {path.suffix}")
    except ComparisonInputError:
        raise
    except Exception as exc:
        raise ComparisonInputError(f"failed to read input file: {path}: {exc}") from exc
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=frame.name or "value")
    if not isinstance(frame, pd.DataFrame):
        raise ComparisonInputError(f"input is not a table: {path}")
    if any(name is not None for name in frame.index.names):
        frame = frame.reset_index()
    return frame


def _canonical_key(row: pd.Series, keys: list[str]) -> tuple[str, ...]:
    return tuple(str(row[key]) for key in keys)


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def compare_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    keys: list[str],
    values: list[str],
    atol: float,
    rtol: float,
    max_mismatches: int,
) -> dict[str, Any]:
    if not keys or not values:
        raise ComparisonInputError("at least one --key and one --value are required")
    if atol < 0 or rtol < 0 or max_mismatches <= 0:
        raise ComparisonInputError("atol/rtol must be non-negative and max-mismatches must be positive")
    required = set(keys + values)
    for side, frame in (("left", left), ("right", right)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ComparisonInputError(f"{side} input missing columns: {missing}")
        duplicate = frame.duplicated(keys, keep=False)
        if bool(duplicate.any()):
            sample = frame.loc[duplicate, keys].head(5).to_dict(orient="records")
            raise ComparisonInputError(f"{side} input has duplicate keys: {sample}")

    left_indexed = left.set_index(keys, drop=False).sort_index()
    right_indexed = right.set_index(keys, drop=False).sort_index()
    left_keys = set(left_indexed.index.tolist())
    right_keys = set(right_indexed.index.tolist())
    only_left = sorted(left_keys - right_keys, key=str)
    only_right = sorted(right_keys - left_keys, key=str)
    common = sorted(left_keys & right_keys, key=str)

    mismatches: list[dict[str, Any]] = []
    field_mismatch_counts = {field: 0 for field in values}
    max_abs_diff = {field: 0.0 for field in values}
    for key in common:
        left_row = left_indexed.loc[key]
        right_row = right_indexed.loc[key]
        if isinstance(left_row, pd.DataFrame) or isinstance(right_row, pd.DataFrame):
            raise ComparisonInputError(f"duplicate key escaped validation: {key}")
        for field in values:
            left_value = left_row[field]
            right_value = right_row[field]
            try:
                left_number = float(left_value)
                right_number = float(right_value)
            except (TypeError, ValueError) as exc:
                raise ComparisonInputError(
                    f"non-numeric value: key={key} field={field} left={left_value} right={right_value}"
                ) from exc
            both_nan = math.isnan(left_number) and math.isnan(right_number)
            one_nan = math.isnan(left_number) != math.isnan(right_number)
            if math.isinf(left_number) or math.isinf(right_number):
                raise ComparisonInputError(
                    f"non-finite value: key={key} field={field} left={left_value} right={right_value}"
                )
            abs_diff = 0.0 if both_nan else abs(left_number - right_number)
            if math.isfinite(abs_diff):
                max_abs_diff[field] = max(max_abs_diff[field], abs_diff)
            within = both_nan or (
                not one_nan
                and math.isfinite(left_number)
                and math.isfinite(right_number)
                and abs_diff <= atol + rtol * abs(right_number)
            )
            if within:
                continue
            field_mismatch_counts[field] += 1
            if len(mismatches) < max_mismatches:
                key_payload = {name: str(value) for name, value in zip(keys, key if isinstance(key, tuple) else (key,))}
                mismatches.append(
                    {
                        "key": key_payload,
                        "field": field,
                        "left": _json_number(left_number),
                        "right": _json_number(right_number),
                        "abs_diff": _json_number(abs_diff),
                    }
                )

    mismatch_count = sum(field_mismatch_counts.values())
    passed = not only_left and not only_right and mismatch_count == 0
    return {
        "schema_version": "backtest_artifact_comparison_v1",
        "passed": passed,
        "keys": keys,
        "values": values,
        "atol": atol,
        "rtol": rtol,
        "left_row_count": len(left),
        "right_row_count": len(right),
        "common_key_count": len(common),
        "only_left_count": len(only_left),
        "only_right_count": len(only_right),
        "only_left_sample": [str(item) for item in only_left[:max_mismatches]],
        "only_right_sample": [str(item) for item in only_right[:max_mismatches]],
        "field_mismatch_counts": field_mismatch_counts,
        "max_abs_diff": max_abs_diff,
        "mismatch_count": mismatch_count,
        "mismatch_sample": mismatches,
    }


def _write_report(report: dict[str, Any], output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None
    resolved = output_dir.resolve()
    if _is_relative_to(resolved, PROJECT_ROOT.resolve()):
        raise ComparisonInputError(f"output-dir must be outside repository: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    report_path = resolved / "comparison_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="Left CSV/Parquet/Pickle/JSON table.")
    parser.add_argument("--right", type=Path, required=True, help="Right CSV/Parquet/Pickle/JSON table.")
    parser.add_argument("--key", action="append", required=True, help="Key column; repeat for compound keys.")
    parser.add_argument("--value", action="append", required=True, help="Numeric column; repeat for multiple values.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance.")
    parser.add_argument("--max-mismatches", type=int, default=100, help="Maximum mismatch samples in the report.")
    parser.add_argument("--output-dir", type=Path, help="Optional external artifact directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        report = compare_tables(
            _load_table(args.left),
            _load_table(args.right),
            keys=list(args.key),
            values=list(args.value),
            atol=args.atol,
            rtol=args.rtol,
            max_mismatches=args.max_mismatches,
        )
        report_path = _write_report(report, args.output_dir)
    except ComparisonInputError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 2
    output = dict(report)
    if report_path is not None:
        output["report_path"] = str(report_path)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
