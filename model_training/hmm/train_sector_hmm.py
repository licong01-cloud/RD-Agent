#!/usr/bin/env python
"""行业 HMM 模型训练主脚本 — WSL 环境执行.

数据源：
  - sector_data 表（L2 行业价格/成交量/资金流）
  - Qlib bin 文件（涨停占比）
  - index_daily 表（CSI300 基准）

观测量（7 维，默认）：
  1. 日收益率
  2. 超额收益 N 日均值（N=rolling_window，默认 5）
  3. 成交量占比
  4. 涨停占比（从 Qlib bin）
  5. N 日波动率
  6. 净资金流入占比
  7. 超大单净流入占比

优化特性：
  - Z-score 标准化：使训练集各特征均值=0、标准差=1，符合 HMM 高斯假设
  - 可配置滚动窗口：默认 5 日（匹配热态持续中位数 1.3-1.7 天）
  - 支持 full 协方差类型：捕捉特征间交互关系

用法:
  cd /mnt/f/Dev/RD-Agent-main
  conda activate rdagent-gpu
  python -m model_training.hmm.train_sector_hmm [--n-states 2] [--cov-type full] [--rolling-window 5] [--zscore]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root on path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model_training.common.data_loader import (
    get_db_conn, load_l2_sector_data, load_csi300_daily,
    load_market_total_volume, load_sector_stock_mapping,
    get_limit_up_ratio_by_sector, read_qlib_calendar,
)
from model_training.hmm.config import HMMTrainConfig

MODEL_SCHEMA_VERSION = "sector_hmm_model_v2"
HMM_RANDOM_SEED = 42

# ─── 观测矩阵构建 ───

def build_observation_matrix(
    sector_data: List[Dict],
    csi300: Dict[date, float],
    market_vol: Dict[date, float],
    limit_up: Dict[date, float],
    limit_down: Dict[date, float] = None,
    rolling_window: int = 5,
    use_limit_down: bool = False,
) -> Tuple[np.ndarray, List[date]]:
    """构建观测矩阵（7 维或 8 维）.

    Args:
        rolling_window: 超额收益和波动率的滚动窗口天数（默认 5）
        use_limit_down: 是否包含跌停占比维度

    Returns: (obs_matrix, dates_list)
    """
    if not sector_data:
        raise ValueError("HMM observation source has no sector rows")
    if not isinstance(rolling_window, int) or rolling_window < 2:
        raise ValueError("rolling_window must be an integer >= 2")
    if limit_down is None:
        limit_down = {}
    rows = []
    dates_out = []

    sorted_data = sorted(sector_data, key=lambda x: x["trade_date"])
    win = max(rolling_window, 2)  # 至少 2 天才能算标准差

    for i, rec in enumerate(sorted_data):
        td = rec["trade_date"]
        pct = rec["pct_change"]
        vol = rec["vol"]
        amount = rec["amount"]
        mf_net = rec["mf_net_amt"]
        mf_buy_elg = rec["mf_buy_elg_amt"]
        mf_sell_elg = rec["mf_sell_elg_amt"]
        raw_values = [pct, vol, amount, mf_net, mf_buy_elg, mf_sell_elg]
        if not np.isfinite(raw_values).all():
            raise ValueError(f"sector observation contains non-finite source values on {td}")
        if amount <= 0:
            raise ValueError(f"sector observation amount must be positive on {td}")

        csi_pct = csi300.get(td)
        mvol = market_vol.get(td)
        if csi_pct is None or mvol is None:
            raise ValueError(f"sector observation is missing benchmark or market volume on {td}")
        if not np.isfinite([csi_pct, mvol]).all() or mvol <= 0:
            raise ValueError(f"sector observation benchmark/market volume is invalid on {td}")

        # o1: 日收益率
        daily_ret = pct / 100.0

        # o2: 超额收益 N 日均值
        csi_window = []
        for j in range(max(0, i - win + 1), i + 1):
            d2 = sorted_data[j]["trade_date"]
            c2 = csi300.get(d2)
            if c2 is None or not np.isfinite(c2):
                raise ValueError(f"sector observation rolling benchmark is missing or invalid on {d2}")
            csi_window.append(sorted_data[j]["pct_change"] / 100.0 - c2 / 100.0)
        excess_nd = np.mean(csi_window)

        # o3: 成交量占比
        vol_ratio = vol / mvol

        # o4: 涨停占比
        lu_ratio = limit_up.get(td)
        if lu_ratio is None or not np.isfinite(lu_ratio):
            raise ValueError(f"sector observation limit-up ratio is missing or invalid on {td}")

        # o5: N 日波动率
        ret_window = [sorted_data[j]["pct_change"] / 100.0
                      for j in range(max(0, i - win + 1), i + 1)]
        volatility = np.std(ret_window) if len(ret_window) > 1 else 0.0

        # o6: 净资金流入占比
        mf_net_ratio = mf_net / amount

        # o7: 超大单净流入占比
        elg_net = mf_buy_elg - mf_sell_elg
        elg_ratio = elg_net / amount

        row = [daily_ret, excess_nd, vol_ratio, lu_ratio, volatility, mf_net_ratio, elg_ratio]
        if use_limit_down:
            ld_ratio = limit_down.get(td)
            if ld_ratio is None or not np.isfinite(ld_ratio):
                raise ValueError(f"sector observation limit-down ratio is missing or invalid on {td}")
            row.insert(4, ld_ratio)  # 插在 lu_ratio 后面

        if any(np.isnan(v) or np.isinf(v) for v in row):
            raise ValueError(f"sector observation contains a non-finite derived value on {td}")
        rows.append(row)
        dates_out.append(td)

    return np.array(rows, dtype=np.float64), dates_out


def apply_zscore(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对观测矩阵做 z-score 标准化.

    Args:
        obs: (T, D) 原始观测矩阵

    Returns:
        (standardized_obs, feature_means, feature_stds)
        feature_means/stds 需要保存到模型中，推断时使用相同的标准化参数
    """
    if obs.shape[0] == 0:
        d = obs.shape[1] if obs.ndim == 2 else 0
        return obs, np.zeros(d), np.ones(d)
    means = obs.mean(axis=0)
    stds = obs.std(axis=0)
    # 防止除零：std=0 的特征保持原值
    stds = np.where(stds < 1e-10, 1.0, stds)
    standardized = (obs - means) / stds
    return standardized, means, stds


# ─── 优化函数 ───

def _infer_n_features(hmm) -> int:
    means = np.asarray(getattr(hmm, "means_", np.empty((0, 0))))
    if means.ndim == 2 and means.shape[1] > 0:
        return int(means.shape[1])
    covars = np.asarray(hmm.covars_)
    if covars.ndim == 3:
        return int(covars.shape[1])
    if covars.ndim == 2 and hmm.covariance_type != "tied":
        return int(covars.shape[1])
    if covars.ndim == 2:
        return int(covars.shape[0])
    raise ValueError("Cannot infer HMM feature dimension for covariance validation")


def _covars_as_full_matrices(hmm) -> np.ndarray:
    """Return covariance as full matrices, independent of hmmlearn storage shape."""
    covars = np.asarray(hmm.covars_, dtype=np.float64)
    n_components = int(hmm.n_components)
    n_features = _infer_n_features(hmm)

    if covars.ndim == 3:
        return covars.copy()
    if covars.ndim == 2:
        if hmm.covariance_type == "tied" and covars.shape == (n_features, n_features):
            return np.repeat(covars[None, :, :], n_components, axis=0)
        return np.array([np.diag(row) for row in covars], dtype=np.float64)
    if covars.ndim == 1:
        return np.array([np.eye(n_features) * value for value in covars], dtype=np.float64)

    raise ValueError(f"Unsupported covariance shape for {hmm.covariance_type}: {covars.shape}")


def _set_hmm_covars_from_full_matrices(hmm, full_covars: np.ndarray) -> None:
    """Store fixed covariances through hmmlearn's setter using the expected shape."""
    cov_type = hmm.covariance_type
    if cov_type == "diag":
        hmm.covars_ = np.diagonal(full_covars, axis1=1, axis2=2).copy()
    elif cov_type == "full":
        hmm.covars_ = full_covars
    elif cov_type == "tied":
        hmm.covars_ = full_covars[0]
    elif cov_type == "spherical":
        hmm.covars_ = np.diagonal(full_covars, axis1=1, axis2=2).mean(axis=1)
    else:
        raise ValueError(f"Unsupported covariance_type: {cov_type}")


def covariance_bound_stats(hmm, max_covar: float = 10.0, min_covar: float = 1e-3) -> Dict[str, Any]:
    """Validate saved covariance bounds and return compact diagnostics."""
    full_covars = _covars_as_full_matrices(hmm)
    cov_type = hmm.covariance_type

    if cov_type in {"diag", "spherical"}:
        values = np.diagonal(full_covars, axis1=1, axis2=2)
        value_name = "variance"
    else:
        eigvals = [np.linalg.eigvalsh((mat + mat.T) / 2) for mat in full_covars]
        values = np.concatenate(eigvals) if eigvals else np.array([], dtype=np.float64)
        value_name = "eigenvalue"

    if values.size == 0:
        raise ValueError("Cannot validate empty HMM covariance values")

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    max_abs_entry = float(np.max(np.abs(full_covars)))
    within_bounds = min_value >= min_covar - 1e-9 and max_value <= max_covar + 1e-9
    if not within_bounds:
        raise ValueError(
            f"HMM covariance {value_name}s out of bounds after fix: "
            f"min={min_value:.6g}, max={max_value:.6g}, "
            f"expected=[{min_covar}, {max_covar}]"
        )

    return {
        "covariance_bound_type": value_name,
        "covariance_min_after": min_value,
        "covariance_max_after": max_value,
        "covariance_max_abs_entry_after": max_abs_entry,
    }


def validate_and_fix_covariance(hmm, max_covar: float = 10.0, min_covar: float = 1e-3):
    """Validate and persistently fix abnormal covariance values.

    Do not mutate ``hmm.covars_[i]`` in place: for hmmlearn this may be a
    detached view, especially for ``covariance_type='diag'``. Instead rebuild
    the full covariance representation and write it back through the setter.
    """
    if not np.isfinite([min_covar, max_covar]).all() or min_covar <= 0 or max_covar < min_covar:
        raise ValueError("Covariance bounds must be finite, positive, and ordered")

    cov_type = hmm.covariance_type
    full_covars = _covars_as_full_matrices(hmm)
    if not np.isfinite(full_covars).all():
        raise ValueError("HMM covariance contains non-finite values")
    fixed = False
    anomaly_count = 0

    if cov_type in {"diag", "spherical"}:
        diag_vals = np.diagonal(full_covars, axis1=1, axis2=2).copy()
        if np.any(diag_vals <= 0):
            raise ValueError("HMM covariance contains non-positive values")
        anomaly_mask = (diag_vals > max_covar) | (diag_vals < min_covar)
        anomaly_count = int(np.sum(anomaly_mask))
        if anomaly_count:
            diag_vals = np.clip(diag_vals, min_covar, max_covar)
            full_covars = np.array([np.diag(row) for row in diag_vals], dtype=np.float64)
            fixed = True
    elif cov_type in {"full", "tied"}:
        matrices = [full_covars[0]] if cov_type == "tied" else list(full_covars)
        fixed_matrices = []
        for mat in matrices:
            sym = (mat + mat.T) / 2
            eigvals, eigvecs = np.linalg.eigh(sym)
            if np.any(eigvals <= 0):
                raise ValueError("HMM covariance is not positive definite")
            anomaly_mask = (eigvals > max_covar) | (eigvals < min_covar)
            anomaly_count += int(np.sum(anomaly_mask))
            eigvals_fixed = np.clip(eigvals, min_covar, max_covar)
            if np.any(anomaly_mask):
                fixed = True
            fixed_mat = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T
            fixed_matrices.append((fixed_mat + fixed_mat.T) / 2)
        full_covars = (
            np.repeat(fixed_matrices[0][None, :, :], hmm.n_components, axis=0)
            if cov_type == "tied"
            else np.array(fixed_matrices, dtype=np.float64)
        )
    else:
        raise ValueError(f"Unsupported covariance_type: {cov_type}")

    if fixed:
        _set_hmm_covars_from_full_matrices(hmm, full_covars)

    covariance_bound_stats(hmm, max_covar=max_covar, min_covar=min_covar)
    return fixed, anomaly_count


def smooth_transition_matrix(transmat: np.ndarray, alpha: float = 0.1, min_self_trans: float = 0.3) -> np.ndarray:
    """平滑转移矩阵."""
    n_states = transmat.shape[0]
    smoothed = transmat.copy()

    # Dirichlet 平滑
    for i in range(n_states):
        smoothed[i] = (transmat[i] + alpha) / (1 + alpha * n_states)

    # 约束最小自转移概率
    for i in range(n_states):
        if smoothed[i, i] < min_self_trans:
            old_self = smoothed[i, i]
            smoothed[i, i] = min_self_trans
            remaining = 1 - min_self_trans
            other_sum = smoothed[i].sum() - old_self
            if other_sum > 0:
                for j in range(n_states):
                    if i != j:
                        smoothed[i, j] = smoothed[i, j] / other_sum * remaining
            else:
                for j in range(n_states):
                    if i != j:
                        smoothed[i, j] = remaining / (n_states - 1)

    # 确保每行和为 1
    for i in range(n_states):
        smoothed[i] /= smoothed[i].sum()

    return smoothed


# ─── 训练 ───

def train_all_sectors(cfg: HMMTrainConfig) -> Tuple[Dict[str, Any], Dict]:
    """训练所有 L2 行业的 HMM 模型.

    Returns: (models_dict, metadata)
    """
    from hmmlearn.hmm import GaussianHMM

    print(f"  连接数据库 {cfg.db_host}:{cfg.db_port}/{cfg.db_name}")
    conn = get_db_conn(cfg.db_host, cfg.db_port, cfg.db_user, cfg.db_password, cfg.db_name)

    # 加载数据（训练期 + 一些 lookback）
    lookback = cfg.train_start - timedelta(days=30)
    print(f"  加载 L2 行业数据 {lookback} ~ {cfg.train_end}")
    sector_data = load_l2_sector_data(conn, lookback, cfg.train_end)
    csi300 = load_csi300_daily(conn, lookback, cfg.train_end)
    market_vol = load_market_total_volume(conn, lookback, cfg.train_end)

    # 加载涨停数据（从 Qlib bin）
    print(f"  加载涨停数据 (Qlib bin: {cfg.qlib_bin_dir})")
    sector_stocks = load_sector_stock_mapping(conn, cfg.sector_level)
    calendar = read_qlib_calendar(cfg.qlib_bin_dir)
    limit_up_data, limit_down_data = get_limit_up_ratio_by_sector(
        cfg.qlib_bin_dir, sector_stocks, calendar, cfg.train_start, cfg.train_end,
    )
    conn.close()

    if not sector_data:
        raise ValueError("HMM training source has no sector data")
    if not csi300 or not market_vol or not calendar:
        raise ValueError("HMM training source is missing benchmark, market-volume, or calendar data")

    print(f"  行业数: {len(sector_data)}, CSI300 天数: {len(csi300)}")
    print(f"  涨停数据行业数: {len(limit_up_data)}, 跌停数据行业数: {len(limit_down_data)}")
    print(f"  滚动窗口: {cfg.rolling_window}日, Z-score: {cfg.zscore}, 跌停: {cfg.use_limit_down}")
    print(f"  优化: min_covar=1e-3, 转移矩阵平滑(alpha=0.1), 最小自转移=0.3")

    models = {}
    skipped = 0
    fixed_count = 0
    total_anomalies = 0

    # 第一遍：收集所有行业的原始观测矩阵（用于全局 z-score 统计）
    sector_obs_raw = {}
    for sector_code, data_list in sorted(sector_data.items()):
        train_data = [r for r in data_list if cfg.train_start <= r["trade_date"] <= cfg.train_end]
        if len(train_data) < cfg.min_trading_days:
            raise ValueError(
                f"training data is insufficient for sector {sector_code}: "
                f"rows={len(train_data)} required={cfg.min_trading_days}"
            )

        lu_data = limit_up_data.get(sector_code, {})
        ld_data = limit_down_data.get(sector_code, {})

        obs, obs_dates = build_observation_matrix(
            train_data, csi300, market_vol, lu_data, ld_data,
            rolling_window=cfg.rolling_window,
            use_limit_down=cfg.use_limit_down,
        )
        if obs.shape[0] < cfg.min_trading_days:
            raise ValueError(
                f"training observations are insufficient for sector {sector_code}: "
                f"rows={obs.shape[0]} required={cfg.min_trading_days}"
            )

        sector_obs_raw[sector_code] = (obs, obs_dates, train_data)

    # 计算全局 z-score 统计量（跨所有行业的训练数据）
    global_zscore_mean = None
    global_zscore_std = None
    if cfg.zscore and sector_obs_raw:
        all_obs = np.vstack([obs for obs, _, _ in sector_obs_raw.values()])
        global_zscore_mean = all_obs.mean(axis=0)
        global_zscore_std = all_obs.std(axis=0)
        global_zscore_std = np.where(global_zscore_std < 1e-10, 1.0, global_zscore_std)
        print(f"  Z-score 全局统计: mean={np.round(global_zscore_mean, 4)}, std={np.round(global_zscore_std, 4)}")

    # 第二遍：训练每个行业
    skipped = 0  # reset
    for sector_code, (obs, obs_dates, train_data) in sorted(sector_obs_raw.items()):
        sector_name = train_data[0].get("l2_name", sector_code)

        # 应用 z-score 标准化
        if cfg.zscore and global_zscore_mean is not None:
            obs_train = (obs - global_zscore_mean) / global_zscore_std
        else:
            obs_train = obs

        try:
            hmm = GaussianHMM(
                n_components=cfg.n_states,
                covariance_type=cfg.covariance_type,
                n_iter=cfg.n_iter,
                min_covar=1e-3,  # 新增: 最小协方差阈值
                random_state=HMM_RANDOM_SEED,
            )
            hmm.fit(obs_train)

            # 验证并修复协方差
            fixed, anomaly_count = validate_and_fix_covariance(hmm, max_covar=10.0, min_covar=1e-3)
            cov_stats = covariance_bound_stats(hmm, max_covar=10.0, min_covar=1e-3)
            if fixed:
                fixed_count += 1
                total_anomalies += anomaly_count
                print(f"    {sector_code} ({sector_name}): 修复 {anomaly_count} 个异常协方差")

            # 平滑转移矩阵
            hmm.transmat_ = smooth_transition_matrix(hmm.transmat_, alpha=0.1, min_self_trans=0.3)

        except Exception as e:
            raise ValueError(f"HMM training failed for sector {sector_code} ({sector_name}): {e}") from e

        # 状态标记：日收益率分量最高的为 trending
        means_ret = hmm.means_[:, 0]  # 第 0 维是日收益率
        if cfg.n_states == 2:
            trending_idx = int(np.argmax(means_ret))
            fading_idx = 1 - trending_idx
            labels = {str(trending_idx): "trending", str(fading_idx): "fading"}
        else:
            # 3 状态：最高=trending, 最低=fading, 中间=neutral
            sorted_idx = np.argsort(means_ret)
            labels = {
                str(sorted_idx[2]): "trending",
                str(sorted_idx[1]): "neutral",
                str(sorted_idx[0]): "fading",
            }

        # 分析热态平均持续天数
        transmat = hmm.transmat_
        trending_state = int([k for k, v in labels.items() if v == "trending"][0])
        p_stay = transmat[trending_state, trending_state]
        avg_duration = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else float("inf")

        model_info = {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "sector_code": sector_code,
            "sector_name": sector_name,
            "n_states": cfg.n_states,
            "covariance_type": cfg.covariance_type,
            "startprob": hmm.startprob_.tolist(),
            "transmat": hmm.transmat_.tolist(),
            "means": hmm.means_.tolist(),
            "covars": hmm.covars_.tolist(),
            "state_labels": labels,
            "trending_avg_duration_days": round(avg_duration, 1),
            "training_days": int(obs.shape[0]),
            "obs_features": cfg.obs_features,
            "rolling_window": cfg.rolling_window,
            "use_limit_down": cfg.use_limit_down,
            "random_seed": HMM_RANDOM_SEED,
            "covariance_fixed": bool(fixed),
            "covariance_anomaly_count": int(anomaly_count),
            **cov_stats,
        }
        # 保存 z-score 参数（推断时需要使用相同的标准化）
        if cfg.zscore and global_zscore_mean is not None:
            model_info["zscore_mean"] = global_zscore_mean.tolist()
            model_info["zscore_std"] = global_zscore_std.tolist()

        models[sector_code] = model_info

    if len(models) != len(sector_data):
        raise ValueError(f"HMM training coverage is incomplete: trained={len(models)}/{len(sector_data)}")
    print(f"  训练完成: {len(models)} 个行业, 跳过 {skipped} 个")
    print(f"  协方差修复: {fixed_count} 个行业, 共 {total_anomalies} 个异常值")
    return models, {
        "sector_data": sector_data, "csi300": csi300, "market_vol": market_vol,
        "limit_up_data": limit_up_data, "limit_down_data": limit_down_data,
    }


# ─── 验证 ───

def validate_model(
    models: Dict[str, Any],
    cfg: HMMTrainConfig,
    cached_data: Dict,
) -> Dict[str, Any]:
    """在验证集上评估模型效果.

    Strategy: load data from train_start through val_end+30d,
    build full observation sequence, decode states, but only
    evaluate predictions in the validation window.
    Uses the same z-score parameters from training for standardization.
    """
    from hmmlearn.hmm import GaussianHMM

    if not models:
        raise ValueError("HMM validation requires at least one serialized sector model")

    conn = get_db_conn(cfg.db_host, cfg.db_port, cfg.db_user, cfg.db_password, cfg.db_name)

    # Load data from train_start (for full Viterbi context) through val_end + 30 days (for future returns)
    data_end = cfg.val_end + timedelta(days=45)
    print(f"  加载数据 {cfg.train_start} ~ {data_end}")
    sector_data = load_l2_sector_data(conn, cfg.train_start - timedelta(days=30), data_end)
    csi300 = load_csi300_daily(conn, cfg.train_start - timedelta(days=30), data_end)
    market_vol = load_market_total_volume(conn, cfg.train_start - timedelta(days=30), data_end)
    sector_stocks = load_sector_stock_mapping(conn, cfg.sector_level)
    calendar = read_qlib_calendar(cfg.qlib_bin_dir)
    limit_up_data, limit_down_data = get_limit_up_ratio_by_sector(
        cfg.qlib_bin_dir, sector_stocks, calendar, cfg.train_start, data_end,
    )
    conn.close()

    print(f"  行业数: {len(sector_data)}, CSI300 天数: {len(csi300)}")

    # 获取 z-score 参数（从第一个模型中读取，所有行业共享全局统计量）
    zscore_mean = None
    zscore_std = None
    first_model = next(iter(models.values()))
    if first_model and "zscore_mean" in first_model:
        zscore_mean = np.array(first_model["zscore_mean"])
        zscore_std = np.array(first_model["zscore_std"])
        print(f"  使用训练集 z-score 参数进行标准化")

    # Reconstruct HMM objects
    hmm_objects = {}
    for code, info in models.items():
        required = {
            "model_schema_version", "n_states", "covariance_type", "startprob", "transmat",
            "means", "covars", "state_labels", "rolling_window", "use_limit_down", "random_seed",
        }
        missing = sorted(required - set(info))
        if missing:
            raise ValueError(f"serialized HMM model {code} is missing fields: {missing}")
        if info["model_schema_version"] != MODEL_SCHEMA_VERSION:
            raise ValueError(f"serialized HMM model {code} has unsupported schema version")
        if info["n_states"] not in {2, 3}:
            raise ValueError(f"serialized HMM model {code} has unsupported state count")
        expected_labels = {"trending", "fading"} if info["n_states"] == 2 else {"trending", "neutral", "fading"}
        expected_keys = {str(index) for index in range(info["n_states"])}
        if set(info["state_labels"]) != expected_keys or set(info["state_labels"].values()) != expected_labels:
            raise ValueError(f"serialized HMM model {code} has invalid state labels")
        hmm = GaussianHMM(n_components=info["n_states"], covariance_type=info["covariance_type"])
        startprob = np.asarray(info["startprob"], dtype=np.float64)
        if startprob.shape != (info["n_states"],) or not np.isfinite(startprob).all() or np.any(startprob < 0):
            raise ValueError(f"serialized HMM model {code} has invalid startprob")
        if not np.isclose(startprob.sum(), 1.0, atol=1e-10):
            raise ValueError(f"serialized HMM model {code} startprob is not normalized")
        hmm.startprob_ = startprob
        transmat = np.asarray(info["transmat"], dtype=np.float64)
        means = np.asarray(info["means"], dtype=np.float64)
        if transmat.shape != (info["n_states"], info["n_states"]) or not np.isfinite(transmat).all():
            raise ValueError(f"serialized HMM model {code} has invalid transition matrix")
        if np.any(transmat < 0) or not np.allclose(transmat.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError(f"serialized HMM model {code} transition matrix is not normalized")
        if means.ndim != 2 or means.shape[0] != info["n_states"] or not np.isfinite(means).all():
            raise ValueError(f"serialized HMM model {code} has invalid means")
        hmm.transmat_ = transmat
        hmm.means_ = means
        hmm.n_features = int(means.shape[1])
        covars = np.array(info["covars"], dtype=np.float64)
        if not np.isfinite(covars).all():
            raise ValueError(f"serialized HMM model {code} covariance is non-finite")
        if info["covariance_type"] == "full":
            if covars.shape != (info["n_states"], means.shape[1], means.shape[1]):
                raise ValueError(f"serialized HMM model {code} full covariance shape is invalid")
            for i in range(covars.shape[0]):
                if not np.allclose(covars[i], covars[i].T, atol=1e-10):
                    raise ValueError(f"serialized HMM model {code} covariance is not symmetric")
                if np.any(np.linalg.eigvalsh(covars[i]) <= 0):
                    raise ValueError(f"serialized HMM model {code} covariance is not positive definite")
        elif info["covariance_type"] == "diag":
            covars_raw = np.array(covars, dtype=np.float64)
            if covars_raw.ndim == 3:
                if covars_raw.shape != (info["n_states"], means.shape[1], means.shape[1]):
                    raise ValueError(f"serialized HMM model {code} diagonal covariance shape is invalid")
                off_diagonal = covars_raw - np.array([np.diag(np.diag(matrix)) for matrix in covars_raw])
                if not np.allclose(off_diagonal, 0.0, atol=1e-12):
                    raise ValueError(f"serialized HMM model {code} diagonal covariance has off-diagonal values")
                covars = np.array([np.diag(covars_raw[i]) for i in range(covars_raw.shape[0])])
            elif covars_raw.ndim == 2:
                if covars_raw.shape != (info["n_states"], means.shape[1]):
                    raise ValueError(f"serialized HMM model {code} diagonal covariance shape is invalid")
                covars = covars_raw
            else:
                raise ValueError(f"serialized HMM model {code} diagonal covariance shape is invalid")
            if np.any(covars <= 0):
                raise ValueError(f"serialized HMM model {code} covariance is non-positive")
        elif info["covariance_type"] == "tied":
            if covars.ndim == 3:
                if covars.shape[0] != info["n_states"] or not all(
                    np.allclose(covars[0], matrix, atol=1e-12) for matrix in covars[1:]
                ):
                    raise ValueError(f"serialized HMM model {code} tied covariance matrices differ")
                covars = covars[0]
            if covars.shape != (means.shape[1], means.shape[1]):
                raise ValueError(f"serialized HMM model {code} tied covariance shape is invalid")
            if not np.allclose(covars, covars.T, atol=1e-10) or np.any(np.linalg.eigvalsh(covars) <= 0):
                raise ValueError(f"serialized HMM model {code} tied covariance is not positive definite")
        elif info["covariance_type"] == "spherical":
            if covars.ndim == 3:
                diagonals = np.array([np.diag(matrix) for matrix in covars])
                off_diagonal = covars - np.array([np.diag(row) for row in diagonals])
                if not np.allclose(off_diagonal, 0.0, atol=1e-12) or not np.allclose(
                    diagonals,
                    diagonals[:, :1],
                    atol=1e-12,
                ):
                    raise ValueError(f"serialized HMM model {code} spherical covariance shape is invalid")
                covars = diagonals[:, 0]
            if covars.shape != (info["n_states"],) or np.any(covars <= 0):
                raise ValueError(f"serialized HMM model {code} spherical covariance is non-positive")
        else:
            raise ValueError(f"serialized HMM model {code} has unsupported covariance type")
        hmm.covars_ = covars
        hmm_objects[code] = (hmm, info["state_labels"])

    # Collect results per eval window
    results_by_window = {w: {"trending": [], "fading": [], "neutral": []} for w in cfg.eval_windows}

    decoded_sectors = 0
    total_predictions = 0

    # 获取模型的 rolling_window 和 use_limit_down 参数
    model_rolling_window = first_model["rolling_window"]
    model_use_limit_down = first_model["use_limit_down"]

    for code in models:
        if code not in sector_data:
            raise ValueError(f"validation sector data is missing for model sector {code}")

        data_list = sorted(sector_data[code], key=lambda x: x["trade_date"])
        lu_data = limit_up_data.get(code, {})
        ld_data = limit_down_data.get(code, {})

        # Build full observation matrix (train + val period)
        obs, obs_dates = build_observation_matrix(
            data_list, csi300, market_vol, lu_data, ld_data,
            rolling_window=model_rolling_window,
            use_limit_down=model_use_limit_down,
        )

        if obs.shape[0] < 20:
            raise ValueError(f"validation observations are insufficient for sector {code}: {obs.shape[0]}")

        # 应用 z-score（使用训练集的统计量）
        if zscore_mean is not None:
            obs = (obs - zscore_mean) / zscore_std

        hmm, labels = hmm_objects[code]
        try:
            states = hmm.predict(obs)
        except Exception as e:
            raise ValueError(f"validation prediction failed for sector {code}: {e}") from e

        decoded_sectors += 1

        # Map dates to labels (only validation period)
        date_label = {}
        for i, td in enumerate(obs_dates):
            if cfg.val_start <= td <= cfg.val_end:
                state_idx = states[i]
                label = labels.get(str(state_idx))
                if label not in {"trending", "neutral", "fading"}:
                    raise ValueError(f"validation decoded unknown state for sector {code}: {state_idx}")
                date_label[td] = label

        if not date_label:
            raise ValueError(f"validation window has no decoded dates for sector {code}")

        # Build pct_change lookup for future returns
        pct_map = {r["trade_date"]: r["pct_change"] for r in data_list}
        extended_dates = sorted(set(r["trade_date"] for r in data_list if r["trade_date"] >= cfg.val_start))

        for td, label in date_label.items():
            td_idx = -1
            for idx, d in enumerate(extended_dates):
                if d == td:
                    td_idx = idx
                    break
            if td_idx < 0:
                raise ValueError(f"validation date {td} is absent from sector {code} return history")

            total_predictions += 1

            for window in cfg.eval_windows:
                future = []
                for offset in range(1, window + 1):
                    fi = td_idx + offset
                    if fi < len(extended_dates):
                        ret = pct_map.get(extended_dates[fi])
                        if ret is not None:
                            future.append(ret)
                if len(future) >= window:
                    cum_ret = sum(future)
                    results_by_window[window][label].append(cum_ret)

    print(f"  解码行业: {decoded_sectors}, 总预测数: {total_predictions}")
    if decoded_sectors != len(models) or total_predictions <= 0:
        raise ValueError(
            f"validation coverage is incomplete: decoded={decoded_sectors}/{len(models)} predictions={total_predictions}",
        )
    expected_labels = set().union(*(set(info["state_labels"].values()) for info in models.values()))
    missing_evidence = [
        (window, label)
        for window in cfg.eval_windows
        for label in sorted(expected_labels)
        if not results_by_window[window][label]
    ]
    if missing_evidence:
        raise ValueError(f"validation return evidence is incomplete: {missing_evidence}")

    # Compute metrics
    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else None

    metrics = {
        "train_period": f"{cfg.train_start} ~ {cfg.train_end}",
        "val_period": f"{cfg.val_start} ~ {cfg.val_end}",
        "sector_count": len(models),
        "n_states": cfg.n_states,
        "covariance_type": cfg.covariance_type,
        "rolling_window": cfg.rolling_window,
        "zscore": cfg.zscore,
        "decoded_sectors": decoded_sectors,
        "total_predictions": total_predictions,
    }

    # Duration analysis
    durations = [m["trending_avg_duration_days"] for m in models.values()
                 if m["trending_avg_duration_days"] < 1000]
    if durations:
        metrics["trending_avg_duration_median"] = round(float(np.median(durations)), 1)
        metrics["trending_avg_duration_mean"] = round(float(np.mean(durations)), 1)

    for window in cfg.eval_windows:
        t = results_by_window[window]["trending"]
        f = results_by_window[window]["fading"]
        n = results_by_window[window].get("neutral", [])

        t_mean = safe_mean(t)
        f_mean = safe_mean(f)
        spread = round(t_mean - f_mean, 4) if t_mean is not None and f_mean is not None else None

        metrics[f"trending_{window}d"] = t_mean
        metrics[f"fading_{window}d"] = f_mean
        metrics[f"spread_{window}d"] = spread
        metrics[f"trending_{window}d_n"] = len(t)
        metrics[f"fading_{window}d_n"] = len(f)
        if n:
            metrics[f"neutral_{window}d"] = safe_mean(n)

    return metrics


# ─── 保存 & 注册 ───

def save_and_register(
    models: Dict[str, Any],
    metrics: Dict[str, Any],
    cfg: HMMTrainConfig,
    display_name: str,
) -> Dict[str, str]:
    """保存模型到 workspace 并注册到数据库."""
    from datetime import datetime, timezone

    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    config_json = {
        "n_states": cfg.n_states,
        "covariance_type": cfg.covariance_type,
        "n_iter": cfg.n_iter,
        "sector_level": cfg.sector_level,
        "cooldown_days": cfg.cooldown_days,
        "trending_coeff": cfg.trending_coeff,
        "fading_coeff": cfg.fading_coeff,
        "neutral_coeff": cfg.neutral_coeff,
        "obs_features": cfg.obs_features,
        "min_trading_days": cfg.min_trading_days,
        "rolling_window": cfg.rolling_window,
        "zscore": cfg.zscore,
        "use_limit_down": cfg.use_limit_down,
    }

    conn = get_db_conn(cfg.db_host, cfg.db_port, cfg.db_user, cfg.db_password, cfg.db_name)
    cur = conn.cursor()

    # Find or create config
    cur.execute(
        "SELECT config_id FROM model_train_configs WHERE model_type = 'sector_hmm' AND display_name = %s",
        (display_name,),
    )
    row = cur.fetchone()
    if row:
        config_id = row[0]
    else:
        cur.execute(
            "INSERT INTO model_train_configs (model_type, display_name, config_json) VALUES ('sector_hmm', %s, %s) RETURNING config_id",
            (display_name, json.dumps(config_json)),
        )
        config_id = cur.fetchone()[0]
        conn.commit()

    # Save model file
    ws_dir = os.path.join(cfg.workspace_dir, "hmm", config_id, snapshot_date)
    os.makedirs(ws_dir, exist_ok=True)
    model_path = os.path.join(ws_dir, "models.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)

    # Create snapshot
    cur.execute(
        """INSERT INTO model_train_snapshots (config_id, model_path, sector_count, status, metrics_json)
           VALUES (%s, %s, %s, 'completed', %s) RETURNING snapshot_id""",
        (config_id, model_path, len(models), json.dumps(metrics)),
    )
    snapshot_id = cur.fetchone()[0]

    # Create job record
    cur.execute(
        "INSERT INTO model_train_jobs (config_id, snapshot_id, status, started_at, completed_at) VALUES (%s, %s, 'completed', NOW(), NOW())",
        (config_id, snapshot_id),
    )
    conn.commit()
    cur.close()
    conn.close()

    return {"config_id": config_id, "snapshot_id": snapshot_id, "model_path": model_path}


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="行业 HMM 模型训练（WSL 环境）")
    parser.add_argument("--n-states", type=int, default=2)
    parser.add_argument("--cov-type", default="full", choices=["full", "diag", "tied", "spherical"])
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--rolling-window", type=int, default=5, help="滚动窗口天数（超额收益、波动率）")
    parser.add_argument("--zscore", action="store_true", default=True, help="启用 z-score 标准化")
    parser.add_argument("--no-zscore", action="store_true", help="禁用 z-score 标准化")
    parser.add_argument("--use-limit-down", action="store_true", default=False, help="包含跌停占比维度")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--db-password", default=os.getenv("TDX_DB_PASSWORD", ""))
    parser.add_argument("--output-json", default=None, help="输出结果 JSON 路径（供 API 解析）")
    args = parser.parse_args()

    use_zscore = not args.no_zscore

    cfg = HMMTrainConfig(
        n_states=args.n_states,
        covariance_type=args.cov_type,
        n_iter=args.n_iter,
        rolling_window=args.rolling_window,
        zscore=use_zscore,
        use_limit_down=args.use_limit_down,
        db_password=args.db_password,
    )

    n_dim = 8 if cfg.use_limit_down else 7
    zscore_tag = "zscore" if cfg.zscore else "raw"
    display_name = args.display_name or f"L2_{cfg.n_states}状态_{cfg.covariance_type}_{n_dim}维_w{cfg.rolling_window}_{zscore_tag}"

    print("=" * 70)
    print("行业 HMM 模型训练（WSL 环境）")
    print("=" * 70)
    print(f"配置: {display_name}")
    print(f"状态数: {cfg.n_states}, 协方差: {cfg.covariance_type}, 迭代: {cfg.n_iter}")
    print(f"滚动窗口: {cfg.rolling_window}日, Z-score: {cfg.zscore}, 跌停: {cfg.use_limit_down}")
    print(f"观测量: {cfg.obs_features}")
    print(f"训练集: {cfg.train_start} ~ {cfg.train_end}")
    print(f"验证集: {cfg.val_start} ~ {cfg.val_end}")
    print(f"Qlib bin: {cfg.qlib_bin_dir}")
    print(f"Workspace: {cfg.workspace_dir}")
    print()

    # Step 1: 训练
    print("[Step 1/3] 训练模型")
    models, cached = train_all_sectors(cfg)
    if not models:
        print("  错误: 没有训练出任何行业模型")
        sys.exit(1)

    # 打印热态持续天数分析
    durations = [(m["sector_name"], m["trending_avg_duration_days"])
                 for m in models.values() if m["trending_avg_duration_days"] < 1000]
    durations.sort(key=lambda x: x[1])
    print(f"\n  热态平均持续天数:")
    print(f"    最短: {durations[0][0]} = {durations[0][1]} 天")
    print(f"    最长: {durations[-1][0]} = {durations[-1][1]} 天")
    print(f"    中位数: {np.median([d for _, d in durations]):.1f} 天")
    print()

    # Step 2: 验证
    print("[Step 2/3] 验证模型")
    metrics = validate_model(models, cfg, cached)

    print(f"\n  {'窗口':<8} {'热态':>10} {'冷态':>10} {'差值':>10} {'热态N':>8} {'冷态N':>8}")
    print("  " + "-" * 58)
    for w in cfg.eval_windows:
        t = metrics.get(f"trending_{w}d", "N/A")
        f = metrics.get(f"fading_{w}d", "N/A")
        s = metrics.get(f"spread_{w}d", "N/A")
        tn = metrics.get(f"trending_{w}d_n", 0)
        fn = metrics.get(f"fading_{w}d_n", 0)
        t_str = f"{t:.4f}" if isinstance(t, (int, float)) else str(t)
        f_str = f"{f:.4f}" if isinstance(f, (int, float)) else str(f)
        s_str = f"{s:.4f}" if isinstance(s, (int, float)) else str(s)
        print(f"  {w}日{'':<5} {t_str:>10} {f_str:>10} {s_str:>10} {tn:>8} {fn:>8}")

    if metrics.get("trending_avg_duration_median"):
        print(f"\n  热态持续天数中位数: {metrics['trending_avg_duration_median']} 天")

    spread_5d = metrics.get("spread_5d")
    if spread_5d is not None and spread_5d > 0:
        print(f"\n  ✓ 模型有效: 热态 5 日收益高于冷态 {spread_5d}%")
    elif spread_5d is not None:
        print(f"\n  ✗ 模型无效: 热态 5 日收益低于冷态 {spread_5d}%")
    print()

    # Step 3: 注册
    print("[Step 3/3] 保存 & 注册")
    result = save_and_register(models, metrics, cfg, display_name)
    print(f"  config_id: {result['config_id']}")
    print(f"  snapshot_id: {result['snapshot_id']}")
    print(f"  model_path: {result['model_path']}")

    # 输出 JSON（供 API 解析）
    if args.output_json:
        output = {**result, "metrics": metrics, "sector_count": len(models)}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  output_json: {args.output_json}")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
