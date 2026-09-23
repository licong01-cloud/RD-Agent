from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rdagent.scenarios.qlib.experiment.factor_template.minute_execution_contract import (
    DayFeatureArtifact,
    MarketAction,
    MinuteExecutionContractError,
    REALLOC_OFFSET,
    TAIL_START_OFFSET,
    classify_market_state,
    normalize_trade_step,
    raw_price,
    select_tail_substitute_candidates,
    strict_remaining_fraction,
    v24_stage,
    v25_stage,
)


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FACTOR = ROOT / "rdagent/scenarios/qlib/experiment/factor_template"
RUNTIME_MODEL = ROOT / "rdagent/scenarios/qlib/experiment/model_template"
V4_FACTOR = ROOT / "app_tpl/all/v4/rdagent/scenarios/qlib/experiment/factor_template"
V4_MODEL = ROOT / "app_tpl/all/v4/rdagent/scenarios/qlib/experiment/model_template"


@pytest.mark.parametrize(
    ("length", "step", "expected"),
    [
        (240, 0, 0),
        (240, 210, TAIL_START_OFFSET),
        (240, 235, REALLOC_OFFSET),
        (240, 239, 239),
        (241, 0, None),
        (241, 1, 0),
        (241, 211, TAIL_START_OFFSET),
        (241, 236, REALLOC_OFFSET),
        (241, 240, 239),
    ],
)
def test_240_and_241_minute_calendars_share_one_normalized_grid(length, step, expected):
    assert normalize_trade_step(trade_step=step, start_idx=0, end_idx=length - 1) == expected


@pytest.mark.parametrize("length", [0, 239, 242])
def test_invalid_historical_minute_counts_fail_fast(length):
    with pytest.raises(MinuteExecutionContractError, match="minute_calendar_length_data_error"):
        normalize_trade_step(trade_step=0, start_idx=0, end_idx=length - 1)


def test_v25_two_stage_transition_is_exact_and_reproducible():
    assert [v25_stage(step) for step in (0, 29, 30, 239)] == ["EARLY", "EARLY", "LATE", "LATE"]
    with pytest.raises(MinuteExecutionContractError, match="v25_stage_step_data_error"):
        v25_stage(240)


def test_v24_two_stage_transition_is_exact_and_reproducible():
    assert [v24_stage(step) for step in (0, 29, 30, 239)] == ["WARMUP", "WARMUP", "PLAN", "PLAN"]
    with pytest.raises(MinuteExecutionContractError, match="v24_stage_step_data_error"):
        v24_stage(240)


def test_raw_price_conversion_prevents_adjusted_limit_comparison():
    assert raw_price(5.5, 0.5) == pytest.approx(11.0)
    with pytest.raises(MinuteExecutionContractError, match="factor_missing_data_error"):
        raw_price(5.5, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "action", "reason"),
    [
        ({"suspended_by_suspend_d": True}, MarketAction.NO_FILL, "suspended_by_suspend_d"),
        ({"suspended_by_exchange": True}, MarketAction.NO_FILL, "suspended_by_exchange"),
        ({"side": "BUY", "price": 11.0}, MarketAction.NO_FILL, "limit_up_buy_blocked"),
        ({"side": "SELL", "price": 9.0}, MarketAction.NO_FILL, "limit_down_sell_blocked"),
        ({"side": "BUY", "price": 9.0}, MarketAction.P0_FORCE, "p0_limit_buy_at_down_limit"),
        ({"side": "SELL", "price": 11.0}, MarketAction.P0_FORCE, "p0_limit_sell_at_up_limit"),
        ({"volume": 0.0}, MarketAction.NO_FILL, "intraday_halt_or_no_bar"),
    ],
)
def test_explicit_market_states_are_not_data_fallbacks(kwargs, action, reason):
    inputs = {
        "side": "BUY",
        "price": 10.0,
        "prev_close": 10.0,
        "limit_up": 11.0,
        "limit_down": 9.0,
        "volume": 100.0,
    }
    inputs.update(kwargs)
    state = classify_market_state(**inputs)
    assert state.action == action
    assert state.reason == reason


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("price", 0.0, "price_missing_data_error"),
        ("prev_close", None, "prev_close_missing_data_error"),
        ("limit_up", None, "limit_price_missing_data_error"),
        ("limit_down", float("nan"), "limit_price_missing_data_error"),
    ],
)
def test_unexplained_missing_market_data_fails_fast(field, value, message):
    inputs = {
        "side": "BUY",
        "price": 10.0,
        "prev_close": 10.0,
        "limit_up": 11.0,
        "limit_down": 9.0,
        "volume": 100.0,
    }
    inputs[field] = value
    with pytest.raises(MinuteExecutionContractError, match=message):
        classify_market_state(**inputs)


def test_strict_plan_fraction_is_reproducible_and_never_uniform_fallback():
    plan = [1.0, 2.0, 3.0]
    first = strict_remaining_fraction(plan, 1, expected_length=3)
    second = strict_remaining_fraction(plan, 1, expected_length=3)
    assert first == second == pytest.approx(0.4)
    with pytest.raises(MinuteExecutionContractError, match="execution_plan_remaining_weight_error"):
        strict_remaining_fraction([1.0, 0.0, 0.0], 1, expected_length=3)
    with pytest.raises(MinuteExecutionContractError, match="execution_plan_length_error"):
        strict_remaining_fraction([1.0, 2.0], 1, expected_length=3)


def test_tail_substitute_switches_to_next_tradable_ranked_candidate():
    tradable = {"D", "E"}
    selected = select_tail_substitute_candidates(
        [("A", 5.0), ("B", 4.0), ("C", 3.0), ("D", 2.0), ("E", 1.0)],
        blocked_count=2,
        current_holdings={"A"},
        already_added={"B"},
        selling_count=1,
        topk=4,
        is_tradable=lambda symbol: symbol in tradable,
    )
    assert selected == [("D", 2.0), ("E", 1.0)]


def test_tail_substitute_respects_topk_capacity():
    selected = select_tail_substitute_candidates(
        [("D", 2.0), ("E", 1.0)],
        blocked_count=2,
        current_holdings={"A", "B", "C"},
        already_added=set(),
        selling_count=0,
        topk=3,
        is_tradable=lambda _symbol: True,
    )
    assert selected == []


def test_v25_day_feature_artifact_is_explicit_and_fail_closed(tmp_path):
    artifact_path = tmp_path / "qe_v25_day_features.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "qe_v25_model_inputs_v1",
                "features_by_date": {"2026-07-24": {"000001.SZ": list(range(10))}},
            }
        ),
        encoding="utf-8",
    )
    artifact = DayFeatureArtifact(artifact_path, expected_schema_version="qe_v25_model_inputs_v1")
    assert artifact.vector(trade_date="2026-07-24T09:30:00", symbol="000001") == list(range(10))
    with pytest.raises(MinuteExecutionContractError, match="day_features_missing_data_error"):
        artifact.vector(trade_date="2026-07-24", symbol="000002.SZ")
    with pytest.raises(MinuteExecutionContractError, match="day_features_schema_mismatch"):
        DayFeatureArtifact(artifact_path, expected_schema_version="paper_v2_v25_day_features_v2")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "filename",
    [
        "minute_execution_contract.py",
        "tail_twap_strategy.py",
        "tail_twap_v24_strategy.py",
        "tail_twap_v25_strategy.py",
    ],
)
def test_factor_model_and_v4_strategy_copies_are_byte_identical(filename):
    hashes = {
        _sha256(directory / filename)
        for directory in (RUNTIME_FACTOR, RUNTIME_MODEL, V4_FACTOR, V4_MODEL)
    }
    assert len(hashes) == 1


def test_v25_source_requires_two_stage_weights_and_real_day_features():
    source = (RUNTIME_FACTOR / "tail_twap_v25_strategy.py").read_text(encoding="utf-8")
    assert "EARLY_WEIGHT = 0.8879" in source
    assert "LATE_WEIGHT = 0.1121" in source
    assert "day_features_file and day_features_schema_version" in source
    assert "day_features = np.zeros(10" not in source
    assert "refusing to fall back to TWAP" in source


def test_v24_source_contains_no_plan_or_data_fallback():
    source = (RUNTIME_FACTOR / "tail_twap_v24_strategy.py").read_text(encoding="utf-8")
    forbidden = ("plan 生成失败, 降级 TWAP", "均匀兜底", "_v24_plan_failed")
    assert not any(token in source for token in forbidden)
    assert "$factor" in source
    assert "strict_remaining_fraction" in source
