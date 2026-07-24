from __future__ import annotations

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
