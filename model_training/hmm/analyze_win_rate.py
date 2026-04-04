#!/usr/bin/env python
"""分析最佳 HMM 模型的胜率和收益分布.

计算热态/冷态/中性下：
- 次日上涨概率（胜率）
- 收益分布（均值、中位数、25/75分位）
- 不同窗口的累计收益胜率

用法:
  cd /mnt/f/Dev/RD-Agent-main
  conda activate rdagent-gpu
  python -m model_training.hmm.analyze_win_rate --db-password lc78080808
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from typing import Any, Dict, List

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

TEST_START = date(2025, 4, 1)
TEST_END = date(2026, 3, 10)
EVAL_WINDOWS = [1, 2, 3, 5, 10, 20]
QLIB_BIN_DIR = "/home/lc999/data/qlib_bin"

# 最佳模型 config_id
BEST_CONFIG_ID = "564b407f"  # L2_3状态_diag_7维_w3_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-password", default="")
    args = parser.parse_args()

    conn = get_db_conn(password=args.db_password)

    # 查找最佳模型
    cur = conn.cursor()
    cur.execute(
        """SELECT c.config_id, c.display_name, s.model_path
           FROM model_train_configs c
           JOIN model_train_snapshots s ON s.config_id = c.config_id
           WHERE c.config_id LIKE %s AND s.status = 'completed'
           ORDER BY s.trained_at DESC LIMIT 1""",
        (BEST_CONFIG_ID + "%",),
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        print("找不到模型"); return

    config_id, display_name, model_path = row
    print(f"分析模型: {display_name}")
    print(f"测试期: {TEST_START} ~ {TEST_END}\n")

    with open(model_path) as f:
        models = json.load(f)

    first_model = next(iter(models.values()))
    rolling_window = first_model.get("rolling_window", 5)
    use_limit_down = first_model.get("use_limit_down", False)

    # 加载数据
    data_start = date(2022, 1, 1) - timedelta(days=30)
    data_end = TEST_END + timedelta(days=30)
    sector_data = load_l2_sector_data(conn, data_start, data_end)
    csi300 = load_csi300_daily(conn, data_start, data_end)
    market_vol = load_market_total_volume(conn, data_start, data_end)
    sector_stocks = load_sector_stock_mapping(conn, "L2")
    calendar = read_qlib_calendar(QLIB_BIN_DIR)
    limit_up_data, limit_down_data = get_limit_up_ratio_by_sector(
        QLIB_BIN_DIR, sector_stocks, calendar, date(2022, 1, 1), data_end,
    )
    conn.close()

    # 重建 HMM 并解码
    from hmmlearn.hmm import GaussianHMM

    # {window: {label: [cum_returns]}}
    results = {w: {"trending": [], "fading": [], "neutral": []} for w in EVAL_WINDOWS}

    for code, info in models.items():
        if code not in sector_data:
            continue

        hmm = GaussianHMM(n_components=info["n_states"], covariance_type=info["covariance_type"])
        hmm.startprob_ = np.array([1.0 / info["n_states"]] * info["n_states"])
        hmm.transmat_ = np.array(info["transmat"])
        hmm.means_ = np.array(info["means"])
        covars = np.array(info["covars"], dtype=np.float64)
        if info["covariance_type"] == "diag":
            if covars.ndim == 3:
                covars = np.array([np.diag(covars[i]) for i in range(covars.shape[0])])
            covars = np.maximum(covars, 1e-6)
        hmm.covars_ = covars

        data_list = sorted(sector_data[code], key=lambda x: x["trade_date"])
        lu = limit_up_data.get(code, {})
        ld = limit_down_data.get(code, {})

        obs, obs_dates = build_observation_matrix(
            data_list, csi300, market_vol, lu, ld,
            rolling_window=rolling_window, use_limit_down=use_limit_down,
        )
        if obs.shape[0] < 20:
            continue

        try:
            states = hmm.predict(obs)
        except Exception:
            continue

        labels = info["state_labels"]
        pct_map = {r["trade_date"]: r["pct_change"] for r in data_list}
        ext_dates = sorted(set(r["trade_date"] for r in data_list if r["trade_date"] >= TEST_START))

        for i, td in enumerate(obs_dates):
            if not (TEST_START <= td <= TEST_END):
                continue
            label = labels.get(str(states[i]), "unknown")
            if label == "unknown":
                continue

            td_idx = -1
            for idx, d in enumerate(ext_dates):
                if d == td:
                    td_idx = idx
                    break
            if td_idx < 0:
                continue

            for window in EVAL_WINDOWS:
                future = []
                for offset in range(1, window + 1):
                    fi = td_idx + offset
                    if fi < len(ext_dates):
                        ret = pct_map.get(ext_dates[fi])
                        if ret is not None:
                            future.append(ret)
                if len(future) >= window:
                    cum_ret = sum(future)
                    results[window][label].append(cum_ret)

    # 打印分析
    print("=" * 90)
    print(f"{'窗口':<6} {'状态':<8} {'样本N':>7} {'均值%':>8} {'中位%':>8} {'胜率%':>7} {'P25%':>8} {'P75%':>8}")
    print("-" * 90)

    for w in EVAL_WINDOWS:
        for label in ["trending", "neutral", "fading"]:
            data = results[w][label]
            if not data:
                continue
            arr = np.array(data)
            n = len(arr)
            mean = np.mean(arr)
            median = np.median(arr)
            win_rate = np.sum(arr > 0) / n * 100
            p25 = np.percentile(arr, 25)
            p75 = np.percentile(arr, 75)

            label_cn = {"trending": "热态", "neutral": "中性", "fading": "冷态"}[label]
            print(f"{w}日{'':<3} {label_cn:<6} {n:>7} {mean:>+8.4f} {median:>+8.4f} {win_rate:>7.1f} {p25:>+8.4f} {p75:>+8.4f}")
        # spread
        t = results[w]["trending"]
        f = results[w]["fading"]
        if t and f:
            t_wr = np.sum(np.array(t) > 0) / len(t) * 100
            f_wr = np.sum(np.array(f) > 0) / len(f) * 100
            spread_mean = np.mean(t) - np.mean(f)
            print(f"{'':>6} {'差值':<6} {'':>7} {spread_mean:>+8.4f} {'':>8} {t_wr - f_wr:>+7.1f}")
        print()

    # 额外：热态 vs 全市场基准
    print("=" * 90)
    print("热态 vs 全市场基准（所有状态合并）")
    print("-" * 90)
    for w in EVAL_WINDOWS:
        all_data = results[w]["trending"] + results[w]["neutral"] + results[w]["fading"]
        t = results[w]["trending"]
        if all_data and t:
            all_mean = np.mean(all_data)
            all_wr = np.sum(np.array(all_data) > 0) / len(all_data) * 100
            t_mean = np.mean(t)
            t_wr = np.sum(np.array(t) > 0) / len(t) * 100
            print(f"{w}日  全市场: 均值={all_mean:+.4f}% 胜率={all_wr:.1f}%  "
                  f"热态: 均值={t_mean:+.4f}% 胜率={t_wr:.1f}%  "
                  f"提升: 均值{t_mean - all_mean:+.4f}% 胜率{t_wr - all_wr:+.1f}%")


if __name__ == "__main__":
    main()
