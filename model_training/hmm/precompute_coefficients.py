#!/usr/bin/env python
"""预计算 HMM 行业热度系数，输出 JSON 文件供 Qlib 回测策略使用.

对回测期间每个交易日、每个行业做前向滤波（forward filtering）解码，输出:
{date_str: {sector_code: coefficient, ...}, ...}

前向滤波 vs Viterbi:
  - Viterbi 在完整序列上做全局最优路径解码，T 日状态受 T+1...T+N 日数据影响（前瞻偏差）
  - 前向滤波只用 ≤T 日的观测计算 T 日状态后验概率，严格因果，无前瞻偏差

用法:
  cd /mnt/f/Dev/RD-Agent-main
  conda activate rdagent-gpu
  python -m model_training.hmm.precompute_coefficients \
    --model-path /home/lc999/model_training_ws/hmm/564b407f-.../models.json \
    --preset preset_A \
    --start-date 2024-07-01 --end-date 2026-03-10 \
    --output /path/to/hmm_sector_coefficients.json \
    --db-password "$TDX_DB_PASSWORD" \
    [--decode-method forward]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model_training.common.data_loader import (
    get_db_conn,
    load_csi300_daily,
    load_l2_sector_data,
    load_market_total_volume,
    read_qlib_bin,
    read_qlib_calendar,
)
from model_training.hmm.train_sector_hmm import build_observation_matrix

QLIB_BIN_DIR = "/home/lc999/data/qlib_bin"


def forward_filter_states(hmm, obs: np.ndarray) -> np.ndarray:
    """前向滤波：逐步计算每日状态后验概率，严格因果无前瞻偏差.

    使用 hmmlearn 内部前向算法，fwd_lattice[t] 仅依赖 obs[0:t+1]。

    Args:
        hmm: 已训练的 GaussianHMM 对象
        obs: (T, D) 观测矩阵

    Returns:
        (T,) 每日最大后验概率状态索引
    """
    from hmmlearn import _hmmc
    from hmmlearn.utils import normalize as hmm_normalize

    log_startprob = np.log(hmm.startprob_ + 1e-300)
    log_transmat = np.log(hmm.transmat_ + 1e-300)
    log_frameprob = hmm._compute_log_likelihood(obs)

    _, fwd_lattice = _hmmc.forward_log(
        log_startprob, log_transmat, log_frameprob,
    )
    # fwd_lattice[t] = log P(state_j, o_1:t+1) — 只依赖 ≤t 的观测
    # 转换为后验概率
    posteriors = np.exp(fwd_lattice)
    hmm_normalize(posteriors, axis=1)

    return posteriors.argmax(axis=1)


# 预设系数
PRESETS = {
    "preset_A": {"trending": 1.05, "neutral": 1.00, "fading": 0.96},
    "preset_B": {"trending": 1.10, "neutral": 1.00, "fading": 0.92},
}

COEFFICIENT_SCHEMA_VERSION = "hmm_sector_coefficients_v2"
MODEL_SCHEMA_VERSION = "sector_hmm_model_v2"
_STATE_LABELS_BY_COUNT = {
    2: {"trending", "fading"},
    3: {"trending", "neutral", "fading"},
}


class HMMCoefficientContractError(ValueError):
    """Raised when a model or coefficient artifact violates the HMM contract."""


def _finite_array(value: Any, field: str, *, ndim: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HMMCoefficientContractError(f"{field} must be a numeric array") from exc
    if ndim is not None and result.ndim != ndim:
        raise HMMCoefficientContractError(f"{field} must have ndim={ndim}; actual={result.ndim}")
    if result.size == 0 or not np.isfinite(result).all():
        raise HMMCoefficientContractError(f"{field} must be non-empty and finite")
    return result


def resolve_preset(name: str) -> dict[str, float]:
    if name not in PRESETS:
        raise HMMCoefficientContractError(f"unknown HMM coefficient preset: {name}")
    values = PRESETS[name]
    if set(values) != {"trending", "neutral", "fading"}:
        raise HMMCoefficientContractError(f"preset {name} has an invalid state set")
    if any(not np.isfinite(value) or value <= 0 for value in values.values()):
        raise HMMCoefficientContractError(f"preset {name} coefficients must be finite and positive")
    return dict(values)


def validate_model_bundle(models: Any) -> dict[str, Any]:
    if not isinstance(models, dict) or not models:
        raise HMMCoefficientContractError("models JSON must be a non-empty object")

    common_contract: dict[str, Any] | None = None
    for sector_code, info in sorted(models.items()):
        field = f"models[{sector_code!r}]"
        if not isinstance(sector_code, str) or not sector_code.strip() or not isinstance(info, dict):
            raise HMMCoefficientContractError(f"{field} must be a model object with a non-empty sector code")
        required = {
            "model_schema_version",
            "n_states",
            "covariance_type",
            "startprob",
            "transmat",
            "means",
            "covars",
            "state_labels",
            "random_seed",
            "rolling_window",
            "use_limit_down",
        }
        missing = sorted(required - set(info))
        if missing:
            raise HMMCoefficientContractError(f"{field} missing required fields: {missing}")
        if info["model_schema_version"] != MODEL_SCHEMA_VERSION:
            raise HMMCoefficientContractError(
                f"{field}.model_schema_version must be {MODEL_SCHEMA_VERSION}",
            )
        n_states = info["n_states"]
        if not isinstance(n_states, int) or n_states not in _STATE_LABELS_BY_COUNT:
            raise HMMCoefficientContractError(f"{field}.n_states must be 2 or 3")
        covariance_type = info["covariance_type"]
        if covariance_type not in {"diag", "full", "tied", "spherical"}:
            raise HMMCoefficientContractError(
                f"{field}.covariance_type must be diag, full, tied, or spherical",
            )
        means = _finite_array(info["means"], f"{field}.means", ndim=2)
        if means.shape[0] != n_states:
            raise HMMCoefficientContractError(f"{field}.means state dimension is invalid")
        feature_count = int(means.shape[1])
        expected_features = 8 if info["use_limit_down"] else 7
        if not isinstance(info["use_limit_down"], bool) or feature_count != expected_features:
            raise HMMCoefficientContractError(
                f"{field} feature dimension must be {expected_features} for use_limit_down={info['use_limit_down']}",
            )
        rolling_window = info["rolling_window"]
        if not isinstance(rolling_window, int) or rolling_window < 2:
            raise HMMCoefficientContractError(f"{field}.rolling_window must be an integer >= 2")
        random_seed = info["random_seed"]
        if not isinstance(random_seed, int):
            raise HMMCoefficientContractError(f"{field}.random_seed must be an integer")

        startprob = _finite_array(info["startprob"], f"{field}.startprob", ndim=1)
        transmat = _finite_array(info["transmat"], f"{field}.transmat", ndim=2)
        if startprob.shape != (n_states,) or transmat.shape != (n_states, n_states):
            raise HMMCoefficientContractError(f"{field} probability dimensions are invalid")
        if np.any(startprob < 0) or np.any(transmat < 0):
            raise HMMCoefficientContractError(f"{field} probabilities must be non-negative")
        if not np.isclose(startprob.sum(), 1.0, atol=1e-10) or not np.allclose(
            transmat.sum(axis=1), 1.0, atol=1e-10,
        ):
            raise HMMCoefficientContractError(f"{field} probabilities must be normalized")

        covars = _finite_array(info["covars"], f"{field}.covars")
        if covariance_type == "diag":
            if covars.ndim == 3 and covars.shape == (n_states, feature_count, feature_count):
                off_diagonal = covars - np.array([np.diag(np.diag(item)) for item in covars])
                if not np.allclose(off_diagonal, 0.0, atol=1e-12):
                    raise HMMCoefficientContractError(f"{field}.covars contains off-diagonal values for diag HMM")
                covars = np.array([np.diag(item) for item in covars])
            if covars.shape != (n_states, feature_count) or np.any(covars <= 0):
                raise HMMCoefficientContractError(f"{field}.covars diag variances must be finite and positive")
        elif covariance_type == "full":
            if covars.shape != (n_states, feature_count, feature_count):
                raise HMMCoefficientContractError(f"{field}.covars full covariance shape is invalid")
            for state, matrix in enumerate(covars):
                if not np.allclose(matrix, matrix.T, atol=1e-10):
                    raise HMMCoefficientContractError(f"{field}.covars[{state}] must be symmetric")
                if np.any(np.linalg.eigvalsh(matrix) <= 0):
                    raise HMMCoefficientContractError(f"{field}.covars[{state}] must be positive definite")
        elif covariance_type == "tied":
            if covars.ndim == 3 and covars.shape == (n_states, feature_count, feature_count):
                if not all(np.allclose(covars[0], matrix, atol=1e-12) for matrix in covars[1:]):
                    raise HMMCoefficientContractError(f"{field}.covars tied matrices must be identical")
                covars = covars[0]
            if covars.shape != (feature_count, feature_count):
                raise HMMCoefficientContractError(f"{field}.covars tied covariance shape is invalid")
            if not np.allclose(covars, covars.T, atol=1e-10) or np.any(np.linalg.eigvalsh(covars) <= 0):
                raise HMMCoefficientContractError(f"{field}.covars tied covariance must be symmetric positive definite")
        else:
            if covars.ndim == 3 and covars.shape == (n_states, feature_count, feature_count):
                diagonals = np.array([np.diag(matrix) for matrix in covars])
                off_diagonal = covars - np.array([np.diag(row) for row in diagonals])
                if not np.allclose(off_diagonal, 0.0, atol=1e-12):
                    raise HMMCoefficientContractError(f"{field}.covars spherical covariance must be diagonal")
                if not np.allclose(diagonals, diagonals[:, :1], atol=1e-12):
                    raise HMMCoefficientContractError(
                        f"{field}.covars spherical feature variances must be equal within each state",
                    )
                covars = diagonals[:, 0]
            if covars.shape != (n_states,) or np.any(covars <= 0):
                raise HMMCoefficientContractError(
                    f"{field}.covars spherical variances must be finite and positive",
                )

        labels = info["state_labels"]
        expected_keys = {str(index) for index in range(n_states)}
        if not isinstance(labels, dict) or set(labels) != expected_keys:
            raise HMMCoefficientContractError(f"{field}.state_labels must cover every hidden state exactly once")
        if set(labels.values()) != _STATE_LABELS_BY_COUNT[n_states]:
            raise HMMCoefficientContractError(f"{field}.state_labels has an invalid semantic label set")

        has_mean = "zscore_mean" in info
        has_std = "zscore_std" in info
        if has_mean != has_std:
            raise HMMCoefficientContractError(f"{field} must provide both zscore_mean and zscore_std")
        if has_mean:
            zscore_mean = _finite_array(info["zscore_mean"], f"{field}.zscore_mean", ndim=1)
            zscore_std = _finite_array(info["zscore_std"], f"{field}.zscore_std", ndim=1)
            if zscore_mean.shape != (feature_count,) or zscore_std.shape != (feature_count,):
                raise HMMCoefficientContractError(f"{field} z-score dimensions are invalid")
            if np.any(zscore_std <= 0):
                raise HMMCoefficientContractError(f"{field}.zscore_std must be strictly positive")
        contract = {
            "rolling_window": rolling_window,
            "use_limit_down": info["use_limit_down"],
            "feature_count": feature_count,
            "zscore_enabled": has_mean,
            "zscore_mean": info.get("zscore_mean"),
            "zscore_std": info.get("zscore_std"),
            "random_seed": random_seed,
        }
        if common_contract is None:
            common_contract = contract
        elif common_contract != contract:
            raise HMMCoefficientContractError(f"{field} differs from the family-wide preprocessing/seed contract")
    assert common_contract is not None
    return common_contract


def load_model_bundle(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as model_file:
            models = json.load(model_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HMMCoefficientContractError(f"cannot load UTF-8 models JSON {path}: {exc}") from exc
    return models, validate_model_bundle(models)


def build_hmm_objects(models: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from hmmlearn.hmm import GaussianHMM

    result = {}
    for code, info in sorted(models.items()):
        hmm = GaussianHMM(n_components=info["n_states"], covariance_type=info["covariance_type"])
        hmm.startprob_ = np.asarray(info["startprob"], dtype=np.float64)
        hmm.transmat_ = np.asarray(info["transmat"], dtype=np.float64)
        hmm.means_ = np.asarray(info["means"], dtype=np.float64)
        hmm.n_features = int(hmm.means_.shape[1])
        covars = np.asarray(info["covars"], dtype=np.float64)
        if info["covariance_type"] == "diag" and covars.ndim == 3:
            covars = np.array([np.diag(item) for item in covars])
        elif info["covariance_type"] == "tied" and covars.ndim == 3:
            covars = covars[0]
        elif info["covariance_type"] == "spherical" and covars.ndim == 3:
            covars = np.array([np.diag(item)[0] for item in covars])
        hmm.covars_ = covars
        result[code] = (hmm, dict(info["state_labels"]))
    return result


def load_pit_stock_sector_map(conn, trade_dates: Sequence[date]) -> dict[str, dict[str, str]]:
    dates = tuple(sorted(set(trade_dates)))
    if not dates:
        raise HMMCoefficientContractError("coefficient output has no trading dates")
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT ts_code, l2_code, in_date, out_date
            FROM market.sw_index_member
            WHERE in_date <= %s AND (out_date IS NULL OR out_date >= %s)
            ORDER BY ts_code, in_date, l2_code
            """,
            (dates[-1], dates[0]),
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
    result: dict[str, dict[str, str]] = {}
    for trade_date in dates:
        mapping: dict[str, str] = {}
        for symbol, sector_code, in_date, out_date in rows:
            if not symbol or not sector_code or in_date is None:
                raise HMMCoefficientContractError("PIT stock-sector membership contains empty identity fields")
            if in_date <= trade_date and (out_date is None or out_date >= trade_date):
                existing = mapping.get(symbol)
                if existing is not None and existing != sector_code:
                    raise HMMCoefficientContractError(
                        f"PIT stock-sector membership is ambiguous for {symbol} on {trade_date}",
                    )
                mapping[symbol] = sector_code
        if not mapping:
            raise HMMCoefficientContractError(f"PIT stock-sector mapping is empty for {trade_date}")
        result[trade_date.isoformat()] = dict(sorted(mapping.items()))
    return result


def get_limit_ratios_by_pit_sector(
    qlib_dir: str,
    stock_sector_map_by_date: Mapping[str, Mapping[str, str]],
    calendar: Sequence[date],
    start_date: date,
    end_date: date,
) -> tuple[dict[str, dict[date, float]], dict[str, dict[date, float]]]:
    """Aggregate limit flags using the sector membership effective on each date."""

    features_dir = Path(qlib_dir) / "features"
    if not features_dir.is_dir():
        raise HMMCoefficientContractError(f"Qlib features directory does not exist: {features_dir}")
    selected_dates = tuple(day for day in calendar if start_date <= day <= end_date)
    if not selected_dates:
        raise HMMCoefficientContractError("Qlib calendar has no dates in the requested range")
    missing_maps = [day.isoformat() for day in selected_dates if day.isoformat() not in stock_sector_map_by_date]
    if missing_maps:
        raise HMMCoefficientContractError(f"PIT sector mapping is missing dates: {missing_maps[:10]}")

    all_symbols = sorted(
        {symbol for day in selected_dates for symbol in stock_sector_map_by_date[day.isoformat()]},
    )
    date_to_index = {day: index for index, day in enumerate(calendar)}
    counts: dict[str, dict[str, dict[date, list[int]]]] = {"up": {}, "down": {}}
    for symbol in all_symbols:
        symbol_dir = features_dir / symbol.lower()
        paths = {
            "up": symbol_dir / "limit_up.day.bin",
            "down": symbol_dir / "limit_down.day.bin",
        }
        values_by_side = {}
        for side, path in paths.items():
            if not path.is_file():
                raise HMMCoefficientContractError(f"missing Qlib {side} flag for {symbol}: {path}")
            values_by_side[side] = read_qlib_bin(str(path))
        for trade_date in selected_dates:
            mapping = stock_sector_map_by_date[trade_date.isoformat()]
            sector_code = mapping.get(symbol)
            if sector_code is None:
                continue
            calendar_index = date_to_index[trade_date]
            for side, (start_index, values) in values_by_side.items():
                value_index = calendar_index - start_index
                if not 0 <= value_index < len(values) or not np.isfinite(values[value_index]):
                    raise HMMCoefficientContractError(
                        f"missing finite Qlib {side} flag for {symbol} on {trade_date}",
                    )
                bucket = counts[side].setdefault(sector_code, {}).setdefault(trade_date, [0, 0])
                bucket[1] += 1
                if values[value_index] >= 0.5:
                    bucket[0] += 1

    output = []
    for side in ("up", "down"):
        ratios: dict[str, dict[date, float]] = {}
        for sector_code, daily in sorted(counts[side].items()):
            ratios[sector_code] = {}
            for trade_date, (flagged, total) in sorted(daily.items()):
                if total <= 0:
                    raise HMMCoefficientContractError(
                        f"PIT {side} denominator is zero for {sector_code} on {trade_date}",
                    )
                ratios[sector_code][trade_date] = flagged / total
        output.append(ratios)
    return output[0], output[1]


def build_coefficient_artifact(
    *,
    model_path: str,
    preset_key: str,
    start_date: date,
    end_date: date,
    expected_sector_codes: Sequence[str],
    sector_date_labels: Mapping[str, Mapping[str, str]],
    stock_sector_map_by_date: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    preset_coeffs = resolve_preset(preset_key)
    sectors = tuple(sorted(expected_sector_codes))
    if not sectors or tuple(sorted(sector_date_labels)) != sectors:
        raise HMMCoefficientContractError("decoded sector set must exactly match the model sector set")
    dates = tuple(sorted(stock_sector_map_by_date))
    if not dates:
        raise HMMCoefficientContractError("stock_sector_map_by_date must be non-empty")
    daily_coefficients: dict[str, dict[str, float]] = {}
    daily_states: dict[str, dict[str, str]] = {}
    for date_key in dates:
        try:
            parsed_date = date.fromisoformat(date_key)
        except ValueError as exc:
            raise HMMCoefficientContractError(f"invalid ISO coefficient date: {date_key}") from exc
        if not start_date <= parsed_date <= end_date:
            raise HMMCoefficientContractError(f"coefficient date {date_key} is outside the requested range")
        mapping = stock_sector_map_by_date[date_key]
        if not isinstance(mapping, Mapping) or not mapping:
            raise HMMCoefficientContractError(f"stock-sector mapping is empty for {date_key}")
        states: dict[str, str] = {}
        coefficients: dict[str, float] = {}
        for sector_code in sectors:
            label = sector_date_labels[sector_code].get(date_key)
            if label not in preset_coeffs:
                raise HMMCoefficientContractError(
                    f"missing or unknown state for sector={sector_code} date={date_key}: {label!r}",
                )
            states[sector_code] = label
            coefficients[sector_code] = float(preset_coeffs[label])
        referenced_sectors = set(mapping.values())
        missing_sector_models = sorted(referenced_sectors - set(sectors))
        if missing_sector_models:
            raise HMMCoefficientContractError(
                f"stock-sector mapping references sectors without models on {date_key}: {missing_sector_models[:10]}",
            )
        daily_states[date_key] = states
        daily_coefficients[date_key] = coefficients
    return {
        "schema_version": COEFFICIENT_SCHEMA_VERSION,
        "mapping_mode": "pit_by_trade_date_v1",
        "model_path": model_path,
        "preset_key": preset_key,
        "preset_coeffs": preset_coeffs,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "sector_count": len(sectors),
        "date_count": len(dates),
        "daily_states": daily_states,
        "daily_coefficients": daily_coefficients,
        "stock_sector_map_by_date": {key: dict(value) for key, value in stock_sector_map_by_date.items()},
    }


def load_precompute_inputs(password: str, start_date: date, end_date: date):
    """Load DB and Qlib inputs while guaranteeing that the DB connection closes."""

    conn = get_db_conn(password=password)
    try:
        sector_data = load_l2_sector_data(conn, start_date, end_date)
        csi300 = load_csi300_daily(conn, start_date, end_date)
        market_vol = load_market_total_volume(conn, start_date, end_date)
        calendar = read_qlib_calendar(QLIB_BIN_DIR)
        analysis_dates = tuple(day for day in calendar if start_date <= day <= end_date)
        pit_mapping = load_pit_stock_sector_map(conn, analysis_dates)
        limit_up, limit_down = get_limit_ratios_by_pit_sector(
            QLIB_BIN_DIR,
            pit_mapping,
            calendar,
            start_date,
            end_date,
        )
        return sector_data, csi300, market_vol, calendar, pit_mapping, limit_up, limit_down
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="preset_A", choices=sorted(PRESETS))
    parser.add_argument("--start-date", default="2024-07-01")
    parser.add_argument("--end-date", default="2026-03-10")
    parser.add_argument("--output", required=True)
    parser.add_argument("--db-password", default="")
    parser.add_argument("--decode-method", default="forward", choices=["forward"],
                        help="解码方式固定为 forward，禁止使用未来观测的全局 Viterbi 路径")
    args = parser.parse_args()

    preset_coeffs = resolve_preset(args.preset)
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        raise HMMCoefficientContractError("start-date must not be after end-date")

    print(f"加载模型: {args.model_path}")
    models, model_contract = load_model_bundle(args.model_path)
    rw = model_contract["rolling_window"]
    uld = model_contract["use_limit_down"]
    has_zscore = model_contract["zscore_enabled"]
    zscore_mean = np.array(model_contract["zscore_mean"]) if has_zscore else None
    zscore_std = np.array(model_contract["zscore_std"]) if has_zscore else None

    print(f"行业数: {len(models)}, 滚动窗口: {rw}, zscore: {has_zscore}")
    print(f"预设: {args.preset} = {preset_coeffs}")
    print(f"日期范围: {start} ~ {end}")
    print(f"解码方式: {args.decode_method}")

    # 加载数据
    ds = start - timedelta(days=int(2.5 * 365 + 30))
    de = end + timedelta(days=5)
    print("加载数据...")
    sector_data, csi300, market_vol, calendar, pit_mapping, lu, ld = load_precompute_inputs(
        args.db_password,
        ds,
        de,
    )

    # 重建 HMM 对象
    hmm_objs = build_hmm_objects(models)

    # 对每个行业构建完整观测序列并解码
    print("解码所有行业...")
    # {sector_code: {date_str: label}}
    sector_date_labels: dict[str, dict[str, str]] = {}

    for code in models:
        if code not in sector_data:
            raise HMMCoefficientContractError(f"sector data is missing for model sector {code}")
        dl = sorted(sector_data[code], key=lambda x: x["trade_date"])
        obs, obs_dates = build_observation_matrix(
            dl, csi300, market_vol, lu.get(code, {}), ld.get(code, {}),
            rolling_window=rw, use_limit_down=uld,
        )
        if obs.shape[0] < 20:
            raise HMMCoefficientContractError(f"sector {code} has insufficient observations: {obs.shape[0]}")
        if zscore_mean is not None:
            obs = (obs - zscore_mean) / zscore_std

        hmm, labels = hmm_objs[code]
        try:
            states = forward_filter_states(hmm, obs)
        except Exception as exc:
            raise HMMCoefficientContractError(f"forward decoding failed for sector {code}: {exc}") from exc

        date_labels = {}
        for i, td in enumerate(obs_dates):
            if start <= td <= end:
                state_idx = states[i]
                label = labels.get(str(state_idx))
                if label not in preset_coeffs:
                    raise HMMCoefficientContractError(
                        f"decoded unknown state for sector={code} date={td}: index={state_idx} label={label!r}",
                    )
                date_labels[td.isoformat()] = label
        sector_date_labels[code] = date_labels

    print(f"解码完成: {len(sector_date_labels)} 个行业 (方式: {args.decode_method})")

    output_dates = tuple(day for day in calendar if start <= day <= end)
    output_mapping = {day.isoformat(): pit_mapping[day.isoformat()] for day in output_dates}
    result = build_coefficient_artifact(
        model_path=args.model_path,
        preset_key=args.preset,
        start_date=start,
        end_date=end,
        expected_sector_codes=tuple(models),
        sector_date_labels=sector_date_labels,
        stock_sector_map_by_date=output_mapping,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"输出: {args.output} ({result['date_count']} 天, {result['sector_count']} 行业)")


if __name__ == "__main__":
    main()
