from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from model_training.hmm import precompute_coefficients as subject


def _model(*, feature_count: int = 7) -> dict:
    return {
        "model_schema_version": subject.MODEL_SCHEMA_VERSION,
        "sector_name": "银行",
        "n_states": 3,
        "covariance_type": "diag",
        "startprob": [0.2, 0.3, 0.5],
        "transmat": [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        "means": [[0.0] * feature_count, [0.1] * feature_count, [-0.1] * feature_count],
        "covars": [[1.0] * feature_count for _ in range(3)],
        "state_labels": {"0": "neutral", "1": "trending", "2": "fading"},
        "random_seed": 42,
        "rolling_window": 5,
        "use_limit_down": feature_count == 8,
        "zscore_mean": [0.0] * feature_count,
        "zscore_std": [1.0] * feature_count,
    }


def test_load_model_bundle_is_utf8_and_validates_complete_schema(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"801010.SI": _model()}, ensure_ascii=False), encoding="utf-8")

    models, contract = subject.load_model_bundle(str(path))

    assert models["801010.SI"]["sector_name"] == "银行"
    assert contract == {
        "rolling_window": 5,
        "use_limit_down": False,
        "feature_count": 7,
        "zscore_enabled": True,
        "zscore_mean": [0.0] * 7,
        "zscore_std": [1.0] * 7,
        "random_seed": 42,
    }


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda models: models.clear(), "non-empty object"),
        (lambda models: models["801010.SI"].pop("startprob"), "missing required fields"),
        (lambda models: models["801010.SI"].__setitem__("covars", [[float("nan")] * 7] * 3), "finite"),
        (lambda models: models["801010.SI"].__setitem__("random_seed", "42"), "random_seed"),
    ],
)
def test_model_bundle_rejects_empty_missing_nonfinite_or_nondeterministic_contract(mutate, match) -> None:
    models = {"801010.SI": _model()}
    mutate(models)
    with pytest.raises(subject.HMMCoefficientContractError, match=match):
        subject.validate_model_bundle(models)


def test_load_model_bundle_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_bytes(b"\xff\xfe{}")
    with pytest.raises(subject.HMMCoefficientContractError, match="UTF-8 models JSON"):
        subject.load_model_bundle(str(path))


@pytest.mark.parametrize("covariance_type", ["diag", "full", "tied", "spherical"])
def test_training_covariance_types_have_one_to_one_precompute_contract(covariance_type: str) -> None:
    model = _model(feature_count=7)
    model["covariance_type"] = covariance_type
    identity = np.eye(7).tolist()
    if covariance_type in {"full", "tied", "spherical"}:
        model["covars"] = [identity, identity, identity]

    subject.validate_model_bundle({"801010.SI": model})
    hmm, _labels = subject.build_hmm_objects({"801010.SI": model})["801010.SI"]

    assert hmm.covariance_type == covariance_type
    assert np.isfinite(hmm.covars_).all()


def test_unknown_preset_is_not_silently_replaced() -> None:
    with pytest.raises(subject.HMMCoefficientContractError, match="unknown HMM coefficient preset"):
        subject.resolve_preset("not-approved")


def test_build_coefficient_artifact_emits_pit_dates_states_and_positive_coefficients() -> None:
    result = subject.build_coefficient_artifact(
        model_path="models.json",
        preset_key="preset_A",
        start_date=date(2026, 1, 5),
        end_date=date(2026, 1, 5),
        expected_sector_codes=("801010.SI",),
        sector_date_labels={"801010.SI": {"2026-01-05": "trending"}},
        stock_sector_map_by_date={"2026-01-05": {"000001.SZ": "801010.SI"}},
    )

    assert result["schema_version"] == subject.COEFFICIENT_SCHEMA_VERSION
    assert result["mapping_mode"] == "pit_by_trade_date_v1"
    assert result["daily_states"]["2026-01-05"]["801010.SI"] == "trending"
    assert result["daily_coefficients"]["2026-01-05"]["801010.SI"] == 1.05
    assert result["stock_sector_map_by_date"]["2026-01-05"]["000001.SZ"] == "801010.SI"


def test_build_coefficient_artifact_rejects_unknown_state_or_missing_model_sector() -> None:
    kwargs = {
        "model_path": "models.json",
        "preset_key": "preset_A",
        "start_date": date(2026, 1, 5),
        "end_date": date(2026, 1, 5),
        "expected_sector_codes": ("801010.SI",),
        "sector_date_labels": {"801010.SI": {"2026-01-05": "unknown"}},
        "stock_sector_map_by_date": {"2026-01-05": {"000001.SZ": "801010.SI"}},
    }
    with pytest.raises(subject.HMMCoefficientContractError, match="missing or unknown state"):
        subject.build_coefficient_artifact(**kwargs)
    kwargs["sector_date_labels"] = {"801010.SI": {"2026-01-05": "trending"}}
    kwargs["stock_sector_map_by_date"] = {"2026-01-05": {"000001.SZ": "801020.SI"}}
    with pytest.raises(subject.HMMCoefficientContractError, match="sectors without models"):
        subject.build_coefficient_artifact(**kwargs)


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def execute(self, query, params):
        assert "market.sw_index_member" in query
        assert params == (date(2026, 1, 6), date(2026, 1, 5))

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, rows):
        self.cursor_instance = _Cursor(rows)

    def cursor(self):
        return self.cursor_instance


def test_pit_stock_sector_mapping_changes_only_on_effective_date() -> None:
    conn = _Connection(
        [
            ("000001.SZ", "801010.SI", date(2020, 1, 1), date(2026, 1, 5)),
            ("000001.SZ", "801020.SI", date(2026, 1, 6), None),
        ],
    )
    result = subject.load_pit_stock_sector_map(conn, [date(2026, 1, 5), date(2026, 1, 6)])
    assert result == {
        "2026-01-05": {"000001.SZ": "801010.SI"},
        "2026-01-06": {"000001.SZ": "801020.SI"},
    }
    assert conn.cursor_instance.closed is True


def test_limit_ratio_uses_pit_membership_for_each_date(tmp_path: Path, monkeypatch) -> None:
    features = tmp_path / "features"
    for symbol in ("000001.sz",):
        directory = features / symbol
        directory.mkdir(parents=True)
        (directory / "limit_up.day.bin").write_bytes(b"x")
        (directory / "limit_down.day.bin").write_bytes(b"x")
    monkeypatch.setattr(
        subject,
        "read_qlib_bin",
        lambda path: (0, np.array([1.0, 0.0], dtype=np.float32))
        if "limit_up" in path
        else (0, np.array([0.0, 1.0], dtype=np.float32)),
    )
    calendar = [date(2026, 1, 5), date(2026, 1, 6)]
    mapping = {
        "2026-01-05": {"000001.SZ": "801010.SI"},
        "2026-01-06": {"000001.SZ": "801020.SI"},
    }
    up, down = subject.get_limit_ratios_by_pit_sector(
        str(tmp_path), mapping, calendar, calendar[0], calendar[-1],
    )
    assert up == {"801010.SI": {calendar[0]: 1.0}, "801020.SI": {calendar[1]: 0.0}}
    assert down == {"801010.SI": {calendar[0]: 0.0}, "801020.SI": {calendar[1]: 1.0}}
