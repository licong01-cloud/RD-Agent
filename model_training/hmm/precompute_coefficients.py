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
    --db-password lc78080808 \
    [--decode-method forward|viterbi]
"""
from __future__ import annotations
import argparse, json, os, sys
from datetime import date, timedelta
from typing import Any, Dict, Optional
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model_training.hmm.train_sector_hmm import build_observation_matrix
from model_training.common.data_loader import (
    get_db_conn, load_l2_sector_data, load_csi300_daily,
    load_market_total_volume, load_sector_stock_mapping,
    get_limit_up_ratio_by_sector, read_qlib_calendar,
)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="preset_A")
    parser.add_argument("--start-date", default="2024-07-01")
    parser.add_argument("--end-date", default="2026-03-10")
    parser.add_argument("--output", required=True)
    parser.add_argument("--db-password", default="")
    parser.add_argument("--decode-method", default="forward", choices=["forward", "viterbi"],
                        help="解码方式: forward=前向滤波(因果,默认), viterbi=全局解码(有前瞻偏差)")
    args = parser.parse_args()

    preset_coeffs = PRESETS.get(args.preset, PRESETS["preset_A"])
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    print(f"加载模型: {args.model_path}")
    with open(args.model_path) as f:
        models = json.load(f)

    first = next(iter(models.values()))
    rw = first.get("rolling_window", 5)
    uld = first.get("use_limit_down", False)
    has_zscore = "zscore_mean" in first
    zscore_mean = np.array(first["zscore_mean"]) if has_zscore else None
    zscore_std = np.array(first["zscore_std"]) if has_zscore else None

    print(f"行业数: {len(models)}, 滚动窗口: {rw}, zscore: {has_zscore}")
    print(f"预设: {args.preset} = {preset_coeffs}")
    print(f"日期范围: {start} ~ {end}")
    print(f"解码方式: {args.decode_method}")

    # 加载数据
    conn = get_db_conn(password=args.db_password)
    ds = start - timedelta(days=int(2.5 * 365 + 30))
    de = end + timedelta(days=5)
    print("加载数据...")
    sector_data = load_l2_sector_data(conn, ds, de)
    csi300 = load_csi300_daily(conn, ds, de)
    market_vol = load_market_total_volume(conn, ds, de)
    sector_stocks = load_sector_stock_mapping(conn, "L2")
    calendar = read_qlib_calendar(QLIB_BIN_DIR)
    lu, ld = get_limit_up_ratio_by_sector(QLIB_BIN_DIR, sector_stocks, calendar, ds, de)
    # 加载股票→行业映射
    cur = conn.cursor()
    cur.execute("SELECT ts_code, l2_code FROM market.sw_index_member WHERE out_date IS NULL OR out_date >= CURRENT_DATE")
    stock_sector_map = {}
    for ts_code, l2_code in cur.fetchall():
        stock_sector_map[ts_code] = l2_code
    cur.close()
    conn.close()

    # 重建 HMM 对象
    from hmmlearn.hmm import GaussianHMM
    hmm_objs = {}
    for code, info in models.items():
        hmm = GaussianHMM(n_components=info["n_states"], covariance_type=info["covariance_type"])
        hmm.startprob_ = np.array([1.0 / info["n_states"]] * info["n_states"])
        hmm.transmat_ = np.array(info["transmat"])
        hmm.means_ = np.array(info["means"])
        covars = np.array(info["covars"], dtype=np.float64)
        if info["covariance_type"] == "diag":
            if covars.ndim == 3:
                covars = np.array([np.diag(covars[i]) for i in range(covars.shape[0])])
            covars = np.maximum(covars, 1e-6)
        elif info["covariance_type"] == "full":
            for i in range(covars.shape[0]):
                covars[i] = (covars[i] + covars[i].T) / 2
                covars[i] += np.eye(covars[i].shape[0]) * 1e-6
        hmm.covars_ = covars
        hmm_objs[code] = (hmm, info["state_labels"])

    # 对每个行业构建完整观测序列并解码
    print("解码所有行业...")
    # {sector_code: {date_str: label}}
    sector_date_labels: Dict[str, Dict[str, str]] = {}

    for code in models:
        if code not in sector_data:
            continue
        dl = sorted(sector_data[code], key=lambda x: x["trade_date"])
        obs, obs_dates = build_observation_matrix(
            dl, csi300, market_vol, lu.get(code, {}), ld.get(code, {}),
            rolling_window=rw, use_limit_down=uld,
        )
        if obs.shape[0] < 20:
            continue
        if zscore_mean is not None:
            obs = (obs - zscore_mean) / zscore_std

        hmm, labels = hmm_objs[code]
        try:
            if args.decode_method == "forward":
                states = forward_filter_states(hmm, obs)
            else:
                states = hmm.predict(obs)
        except Exception:
            continue

        date_labels = {}
        for i, td in enumerate(obs_dates):
            if start <= td <= end:
                state_idx = states[i]
                date_labels[td.isoformat()] = labels.get(str(state_idx), "neutral")
        sector_date_labels[code] = date_labels

    print(f"解码完成: {len(sector_date_labels)} 个行业 (方式: {args.decode_method})")

    # 构建输出: {date_str: {sector_code: coefficient}}
    all_dates = set()
    for dl in sector_date_labels.values():
        all_dates.update(dl.keys())

    daily_coefficients = {}
    for d in sorted(all_dates):
        day_coeffs = {}
        for code, dl in sector_date_labels.items():
            label = dl.get(d, "neutral")
            day_coeffs[code] = preset_coeffs.get(label, 1.0)
        daily_coefficients[d] = day_coeffs

    # 同时输出股票→行业映射
    result = {
        "preset": args.preset,
        "preset_coeffs": preset_coeffs,
        "model_path": args.model_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "sector_count": len(sector_date_labels),
        "date_count": len(daily_coefficients),
        "stock_sector_map": stock_sector_map,
        "daily_coefficients": daily_coefficients,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"输出: {args.output} ({len(daily_coefficients)} 天, {len(sector_date_labels)} 行业)")


if __name__ == "__main__":
    main()
