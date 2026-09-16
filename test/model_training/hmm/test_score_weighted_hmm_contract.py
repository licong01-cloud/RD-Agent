from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

TEMPLATE = (
    Path(__file__).parents[3]
    / "app_tpl/all/v4/rdagent/scenarios/qlib/experiment/factor_template/score_weighted_strategy.py"
)


def _load_template(monkeypatch):
    qlib = types.ModuleType("qlib")
    contrib = types.ModuleType("qlib.contrib")
    strategy = types.ModuleType("qlib.contrib.strategy")
    signal_strategy = types.ModuleType("qlib.contrib.strategy.signal_strategy")
    signal_strategy.TopkDropoutStrategy = object
    backtest = types.ModuleType("qlib.backtest")
    decision = types.ModuleType("qlib.backtest.decision")
    decision.Order = decision.OrderDir = decision.TradeDecisionWO = object
    for name, module in {
        "qlib": qlib,
        "qlib.contrib": contrib,
        "qlib.contrib.strategy": strategy,
        "qlib.contrib.strategy.signal_strategy": signal_strategy,
        "qlib.backtest": backtest,
        "qlib.backtest.decision": decision,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("score_weighted_hmm_contract", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _strategy(module, path: Path):
    instance = module.ScoreWeightedTopkStrategy.__new__(module.ScoreWeightedTopkStrategy)
    instance.enable_sector_hmm = True
    instance.hmm_coefficients_file = str(path)
    instance._hmm_config = None
    instance._hmm_config_loaded = False
    instance._last_hmm_adjustment_trace = None
    return instance


def _v2_payload():
    return {
        "schema_version": "hmm_sector_coefficients_v2",
        "mapping_mode": "pit_by_trade_date_v1",
        "preset_key": "preset_A",
        "preset_coeffs": {"trending": 1.05, "neutral": 1.0, "fading": 0.96},
        "daily_states": {"2026-01-05": {"801010.SI": "trending"}},
        "daily_coefficients": {"2026-01-05": {"801010.SI": 1.05}},
        "stock_sector_map_by_date": {"2026-01-05": {"000001.SZ": "801010.SI"}},
    }


def _qe_entry(*, status="applied", sector="801010.SI"):
    applied = status == "applied"
    return {
        "status": status,
        "sector_code": sector if applied else None,
        "reason_code": None if applied else "classification:classification_authority_unavailable",
        "adjustment_applied": applied,
        "classification_receipt_hash": "a" * 64,
        "index_membership_receipt_hash": "b" * 64,
        "classification_row_hashes": ["c" * 64] if applied else [],
        "index_membership_row_hashes": ["d" * 64] if applied else [],
    }


def _qe_payload():
    sectors = [f"801{index:03d}.SI" for index in range(10, 41)]
    payload = {
        "schema_version": "hmm_risk_qe_assistance_coefficients_v1",
        "source_model_contract": "hmm_risk_rotation_l1_g2a_v1_6",
        "model_hash": "3956107600a3aef4b51ac1da0c56f7940ce49a34777c836e974d14a5b45fbee6",
        "source_mapping_sha256": "e478722f700535ac4e37744a651291bc6d179cb899dccd28cdb957ed4491b82f",
        "source_prediction_file_sha256": "0957ae8a6527fb28ba337a449ce0f72dfe9f43513003492329d7e770aa9da8e2",
        "source_prediction_row_sha256": "2" * 64,
        "authority_identity": {
            "bundle_hash": "203effb611d00edde4c0ee9c40f205759097628b8c5eb249907f3b33e6932ddf"
        },
        "canonical_l1_codes": sectors,
        "window_start": "2024-07-02",
        "window_end": "2026-03-31",
        "daily_coefficients": {
            "2026-01-05": {sector: (1.02 if sector == "801010.SI" else 1.0) for sector in sectors},
        },
        "stock_sector_applicability_by_date": {
            "2026-01-05": {
                "000001.SZ": _qe_entry(),
                "000002.SZ": _qe_entry(
                    status="not_applicable_authority_unavailable",
                    sector=None,
                ),
            },
        },
        "adapter_formula": {
            "text": "adjusted_score=raw_score+(coefficient-1.0)*abs(raw_score)",
            "sha256": hashlib.sha256(
                b"adjusted_score=raw_score+(coefficient-1.0)*abs(raw_score)",
            ).hexdigest(),
        },
        "adjustment_mode": "sign_safe_magnitude_v1",
        "tail_accessed": False,
        "prediction_row_count": 2,
        "date_count": 1,
        "sector_denominator": 31,
        "applied_row_count": 1,
        "not_applicable_row_count": 1,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _rehash_qe_payload(payload):
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()


def test_v2_payload_adjustment_is_pit_exact_and_traceable(tmp_path: Path, monkeypatch) -> None:
    module = _load_template(monkeypatch)
    path = tmp_path / "coefficients.json"
    path.write_text(json.dumps(_v2_payload()), encoding="utf-8")
    strategy = _strategy(module, path)

    adjusted = strategy._apply_hmm_adjustment(pd.Series({"000001.SZ": 2.0}), "2026-01-05")

    assert adjusted["000001.SZ"] == pytest.approx(2.1)
    assert strategy._last_hmm_adjustment_trace == {
        "trade_date": "2026-01-05",
        "mapping_mode": "pit_by_trade_date_v1",
        "row_count": 1,
        "rows": [
            {
                "stock_id": "000001.SZ",
                "sector_code": "801010.SI",
                "state": "trending",
                "coefficient": 1.05,
                "raw_score": 2.0,
                "adjusted_score": 2.1,
                "reason": "hmm_sector_coefficient_applied",
            },
        ],
    }


def test_missing_date_stock_sector_or_coefficient_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    module = _load_template(monkeypatch)
    path = tmp_path / "coefficients.json"
    payload = _v2_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    strategy = _strategy(module, path)
    with pytest.raises(RuntimeError, match="缺少交易日"):
        strategy._apply_hmm_adjustment(pd.Series({"000001.SZ": 2.0}), "2026-01-06")

    strategy = _strategy(module, path)
    with pytest.raises(RuntimeError, match="股票行业映射缺失"):
        strategy._apply_hmm_adjustment(pd.Series({"000002.SZ": 2.0}), "2026-01-05")

    payload["daily_coefficients"]["2026-01-05"] = {"801020.SI": 1.05}
    path.write_text(json.dumps(payload), encoding="utf-8")
    strategy = _strategy(module, path)
    with pytest.raises(RuntimeError, match="state/sector sets differ"):
        strategy._apply_hmm_adjustment(pd.Series({"000001.SZ": 2.0}), "2026-01-05")


def test_empty_hmm_score_input_fails_instead_of_becoming_noop(tmp_path: Path, monkeypatch) -> None:
    module = _load_template(monkeypatch)
    strategy = _strategy(module, tmp_path / "unused.json")

    with pytest.raises(RuntimeError, match="input score is empty"):
        strategy._apply_hmm_adjustment(pd.Series(dtype=float), "2026-01-05")


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), 0.0, -1.0])
def test_abnormal_coefficient_never_becomes_neutral_success(tmp_path: Path, monkeypatch, coefficient) -> None:
    module = _load_template(monkeypatch)
    payload = _v2_payload()
    payload["daily_coefficients"]["2026-01-05"]["801010.SI"] = coefficient
    path = tmp_path / "coefficients.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    strategy = _strategy(module, path)
    with pytest.raises(RuntimeError, match="finite and positive"):
        strategy._apply_hmm_adjustment(pd.Series({"000001.SZ": 2.0}), "2026-01-05")


def test_legacy_static_payload_remains_explicitly_detected(tmp_path: Path, monkeypatch) -> None:
    module = _load_template(monkeypatch)
    payload = {
        "daily_coefficients": {"2026-01-05": {"801010.SI": 1.05}},
        "stock_sector_map": {"000001.SZ": "801010.SI"},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    strategy = _strategy(module, path)
    adjusted = strategy._apply_hmm_adjustment(pd.Series({"000001.SZ": 2.0}), "2026-01-05")
    assert adjusted["000001.SZ"] == pytest.approx(2.1)
    assert strategy._last_hmm_adjustment_trace["mapping_mode"] == "static_legacy_v1"
    assert strategy._last_hmm_adjustment_trace["rows"][0]["state"] is None


def test_qe_assistance_uses_sign_safe_formula_and_explicit_non_applicable(tmp_path: Path, monkeypatch) -> None:
    module = _load_template(monkeypatch)
    path = tmp_path / "qe-assistance.json"
    path.write_text(json.dumps(_qe_payload()), encoding="utf-8")
    strategy = _strategy(module, path)

    adjusted = strategy._apply_hmm_adjustment(
        pd.Series({"000001.SZ": -2.0, "000002.SZ": 3.0}),
        "2026-01-05",
    )

    assert adjusted["000001.SZ"] == pytest.approx(-1.96)
    assert adjusted["000002.SZ"] == 3.0
    assert strategy._last_hmm_adjustment_trace["mapping_mode"] == "qe_assistance_by_trade_date_v1"
    assert [row["reason"] for row in strategy._last_hmm_adjustment_trace["rows"]] == [
        "hmm_qe_assistance_applied",
        "not_applicable_authority_unavailable",
    ]


@pytest.mark.parametrize(
    "mutation, expected_reason",
    [
        (
            lambda payload: payload.update(artifact_sha256="0" * 64),
            "hmm_risk_qe_assistance_authority_identity_mismatch",
        ),
        (
            lambda payload: (
                payload["stock_sector_applicability_by_date"]["2026-01-05"]["000002.SZ"].update(
                    reason_code="unknown",
                ),
                _rehash_qe_payload(payload),
            ),
            "hmm_risk_qe_assistance_pit_mapping_missing",
        ),
    ],
)
def test_qe_assistance_rejects_hash_or_entry_drift(
    tmp_path: Path,
    monkeypatch,
    mutation,
    expected_reason,
) -> None:
    module = _load_template(monkeypatch)
    payload = _qe_payload()
    mutation(payload)
    path = tmp_path / "qe-assistance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    strategy = _strategy(module, path)

    with pytest.raises(module.HMMQEAssistanceContractError) as exc_info:
        strategy._apply_hmm_adjustment(
            pd.Series({"000001.SZ": -2.0, "000002.SZ": 3.0}),
            "2026-01-05",
        )
    assert exc_info.value.reason_code == expected_reason


def test_qe_assistance_requires_exact_daily_prediction_denominator(tmp_path: Path, monkeypatch) -> None:
    module = _load_template(monkeypatch)
    path = tmp_path / "qe-assistance.json"
    path.write_text(json.dumps(_qe_payload()), encoding="utf-8")
    strategy = _strategy(module, path)

    with pytest.raises(RuntimeError, match="applicability denominator differs"):
        strategy._apply_hmm_adjustment(pd.Series({"000001.SZ": -2.0}), "2026-01-05")


def test_qe_assistance_duplicate_json_identity_fails_closed(tmp_path: Path, monkeypatch) -> None:
    module = _load_template(monkeypatch)
    payload = _qe_payload()
    raw = json.dumps(payload)
    raw = raw[:-1] + ', "schema_version": "hmm_risk_qe_assistance_coefficients_v1"}'
    path = tmp_path / "qe-assistance-duplicate.json"
    path.write_text(raw, encoding="utf-8")
    strategy = _strategy(module, path)

    with pytest.raises(module.HMMQEAssistanceContractError) as exc_info:
        strategy._apply_hmm_adjustment(
            pd.Series({"000001.SZ": -2.0, "000002.SZ": 3.0}),
            "2026-01-05",
        )
    assert exc_info.value.reason_code == "hmm_risk_qe_assistance_input_invalid"


def test_disabled_hmm_does_not_read_artifact(tmp_path: Path, monkeypatch) -> None:
    module = _load_template(monkeypatch)
    strategy = _strategy(module, tmp_path / "missing.json")
    strategy.enable_sector_hmm = False
    scores = pd.Series({"000001.SZ": 1.0})

    assert strategy._apply_hmm_adjustment(scores, "2026-01-05") is scores
