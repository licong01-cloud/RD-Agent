"""Focused contract tests for the static-factor bundle generator."""

# ruff: noqa: SLF001

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "generate_static_factors_bundle.py"
_SPEC = importlib.util.spec_from_file_location("generate_static_factors_bundle_under_test", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(_MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_downcast_preserves_l2_category_dtype() -> None:
    frame = pd.DataFrame(
        {
            "continuous": pd.Series([1.0, 2.0], dtype="float64"),
            "l2_code_id": pd.Series([0, 133], dtype="int16"),
        },
    )

    result = _MODULE._downcast_factor_columns(frame)

    assert result["continuous"].dtype == np.dtype("float32")
    assert result["l2_code_id"].dtype == np.dtype("int16")


def test_normalize_l2_category_maps_join_null_to_minus1_and_emits_receipt() -> None:
    series = pd.Series([0.0, 133.0, np.nan, -1.0, 7.0], name="l2_code_id")

    normalized, receipt = _MODULE._normalize_l2_code_id(series)

    assert normalized.dtype == np.dtype("int16")
    assert normalized.tolist() == [0, 133, -1, -1, 7]
    assert receipt == {
        "column": "l2_code_id",
        "dtype": "int16",
        "rows": 5,
        "known_rows": 3,
        "unknown_minus1_rows": 2,
        "null_to_minus1_rows": 1,
        "known_coverage": 0.6,
        "known_sector_ids": 3,
        "min": -1,
        "max": 133,
    }


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([1.5], "non-integer"),
        ([np.inf], "non-finite"),
        ([-2], "below the unknown sentinel"),
        ([2**31], "exceeds supported signed int32 range"),
    ],
)
def test_normalize_l2_category_rejects_invalid_values(values: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _MODULE._normalize_l2_code_id(pd.Series(values, name="l2_code_id"))


def test_l2_schema_is_explicitly_categorical_and_sector_sourced() -> None:
    frame = pd.DataFrame({"l2_code_id": pd.Series([0, -1], dtype="int16")})

    schema = _MODULE._fill_derived_meanings(_MODULE._build_schema(frame))

    assert schema == [
        {
            "name": "l2_code_id",
            "dtype": "int16",
            "meaning": _MODULE._L2_CODE_ID_MEANING,
            "source": "sector_data_raw",
            "semantic_type": "categorical_id",
            "unknown_value": -1,
        },
    ]


def test_margin_detail_schema_uses_raw_source() -> None:
    frame = pd.DataFrame({"md_rzye": pd.Series([1.0], dtype="float32")})

    schema = _MODULE._build_schema(frame)

    assert schema[0]["source"] == "margin_detail_raw"


def test_left_join_normalization_keeps_integer_semantics() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-04-27"), "000001.SZ"),
            (pd.Timestamp("2026-04-27"), "000002.SZ"),
            (pd.Timestamp("2026-04-28"), "000001.SZ"),
        ],
        names=["datetime", "instrument"],
    )
    base = pd.DataFrame({"db_close": [1.0, 2.0, 3.0]}, index=index)
    sector = pd.DataFrame(
        {"l2_code_id": pd.Series([3, 4], index=index[:2], dtype="int16")},
        index=index[:2],
    )

    merged = _MODULE._downcast_factor_columns(base).join(
        _MODULE._downcast_factor_columns(sector), how="left",
    )
    merged["l2_code_id"], receipt = _MODULE._normalize_l2_code_id(merged["l2_code_id"])

    assert merged["l2_code_id"].dtype == np.dtype("int16")
    assert merged["l2_code_id"].tolist() == [3, 4, -1]
    assert receipt["null_to_minus1_rows"] == 1
    assert receipt["known_coverage"] == pytest.approx(2 / 3)
