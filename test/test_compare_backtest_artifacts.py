from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.compare_backtest_artifacts import ComparisonInputError, compare_tables, main


def _frame(values=(10.0, 11.0)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-07-23", "2026-07-24"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "pre_close": values,
            "deal_price": [10.1, 11.1],
        }
    )


def test_same_inputs_are_reproducible():
    first = compare_tables(
        _frame(),
        _frame(),
        keys=["trade_date", "symbol"],
        values=["pre_close", "deal_price"],
        atol=1e-8,
        rtol=1e-6,
        max_mismatches=10,
    )
    second = compare_tables(
        _frame(),
        _frame(),
        keys=["trade_date", "symbol"],
        values=["pre_close", "deal_price"],
        atol=1e-8,
        rtol=1e-6,
        max_mismatches=10,
    )
    assert first == second
    assert first["passed"] is True


def test_mismatch_and_missing_key_are_reported():
    right = _frame((10.0, 12.0)).iloc[:1]
    report = compare_tables(
        _frame(),
        right,
        keys=["trade_date", "symbol"],
        values=["pre_close"],
        atol=0.0,
        rtol=0.0,
        max_mismatches=10,
    )
    assert report["passed"] is False
    assert report["only_left_count"] == 1


def test_numeric_mismatch_uses_explicit_tolerance():
    close = _frame((10.0, 11.000001))
    assert compare_tables(
        _frame(), close, keys=["trade_date", "symbol"], values=["pre_close"], atol=1e-4, rtol=0.0, max_mismatches=10
    )["passed"]
    assert not compare_tables(
        _frame(), close, keys=["trade_date", "symbol"], values=["pre_close"], atol=0.0, rtol=0.0, max_mismatches=10
    )["passed"]
    missing = _frame()
    missing.loc[0, "pre_close"] = float("nan")
    report = compare_tables(
        missing, _frame(), keys=["trade_date", "symbol"], values=["pre_close"], atol=0.0, rtol=0.0, max_mismatches=10
    )
    assert report["mismatch_sample"][0]["left"] is None
    json.dumps(report, allow_nan=False)


def test_duplicate_keys_and_missing_columns_fail_fast():
    duplicate = pd.concat([_frame(), _frame().iloc[:1]], ignore_index=True)
    with pytest.raises(ComparisonInputError, match="duplicate keys"):
        compare_tables(
            duplicate, _frame(), keys=["trade_date", "symbol"], values=["pre_close"], atol=0.0, rtol=0.0, max_mismatches=10
        )
    with pytest.raises(ComparisonInputError, match="missing columns"):
        compare_tables(
            _frame(), _frame(), keys=["missing"], values=["pre_close"], atol=0.0, rtol=0.0, max_mismatches=10
        )
    infinite = _frame()
    infinite.loc[0, "pre_close"] = float("inf")
    with pytest.raises(ComparisonInputError, match="non-finite value"):
        compare_tables(
            infinite, _frame(), keys=["trade_date", "symbol"], values=["pre_close"], atol=0.0, rtol=0.0, max_mismatches=10
        )


def test_cli_uses_external_output_and_nonzero_mismatch_code(tmp_path, capsys):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.parquet"
    output = tmp_path / "artifacts"
    _frame().to_csv(left, index=False)
    _frame((10.0, 12.0)).to_parquet(right, index=False)
    code = main(
        [
            "--left", str(left),
            "--right", str(right),
            "--key", "trade_date",
            "--key", "symbol",
            "--value", "pre_close",
            "--atol", "0",
            "--rtol", "0",
            "--output-dir", str(output),
        ]
    )
    assert code == 1
    report = json.loads((output / "comparison_report.json").read_text(encoding="utf-8"))
    assert report["mismatch_count"] == 1
    assert not any(Path.cwd().glob("mlruns_compare*"))
    assert not any(Path.cwd().glob("mlruns_verify*"))
    assert not any(Path.cwd().glob("compare_result*"))
    assert json.loads(capsys.readouterr().out)["report_path"].endswith("comparison_report.json")


def test_cli_returns_input_error_code(tmp_path, capsys):
    code = main(
        [
            "--left", str(tmp_path / "missing-left.csv"),
            "--right", str(tmp_path / "missing-right.csv"),
            "--key", "trade_date",
            "--value", "deal_price",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["passed"] is False
    assert "does not exist" in payload["error"]


def test_cli_refuses_repository_output(tmp_path, capsys):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _frame().to_csv(left, index=False)
    _frame().to_csv(right, index=False)
    code = main(
        [
            "--left", str(left),
            "--right", str(right),
            "--key", "trade_date",
            "--key", "symbol",
            "--value", "deal_price",
            "--output-dir", str(Path(__file__).resolve().parents[1] / "compare_result"),
        ]
    )
    assert code == 2
    assert "outside repository" in json.loads(capsys.readouterr().err)["error"]
