#!/usr/bin/env python
"""在测试集（2025-04-01 ~ 2026-03-10）上评估多个 HMM 模型的真实效果.

自动从 DB 中查询所有已注册的 sector_hmm 模型，加载模型文件，
在测试集上解码状态并计算热态/冷态未来收益 spread。

用法:
  cd /mnt/f/Dev/RD-Agent-main
  conda activate rdagent-gpu
  python -m model_training.hmm.test_on_testset --db-password lc78080808
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

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


def list_all_configs(conn) -> List[Dict]:
    """列出所有 sector_hmm 配置（每个 config 取最新 snapshot）."""
    cur = conn.cursor()
    cur.execute("""
        SELECT c.config_id, c.display_name, c.config_json,
               s.model_path, s.trained_at
        FROM model_train_configs c
        JOIN model_train_snapshots s ON s.config_id = c.config_id
        WHERE c.model_type = 'sector_hmm' AND s.status = 'completed'
        ORDER BY c.config_id, s.trained_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    seen = set()
    result = []
    for config_id, display_name, config_json, model_path, trained_at in rows:
        if config_id in seen:
            continue
        seen.add(config_id)
        cj = config_json if isinstance(config_json, dict) else json.loads(config_json)
        result.append({
            "config_id": config_id,
            "display_name": display_name,
            "config_json": cj,
            "model_path": model_path,
        })
    return result



def evaluate_model(
    models: Dict[str, Any],
    display_name: str,
    sector_data: Dict,
    csi300: Dict,
    market_vol: Dict,
    limit_up_data: Dict,
    limit_down_data: Dict,
) -> Dict[str, Any]:
    """在测试集上评估单个模型."""
    from hmmlearn.hmm import GaussianHMM

    # 从模型中读取参数
    first_model = next(iter(models.values()), None)
    if not first_model:
        return {"error": "empty model"}

    n_states = first_model.get("n_states", 2)
    cov_type = first_model.get("covariance_type", "diag")
    rolling_window = first_model.get("rolling_window", 20)
    use_limit_down = first_model.get("use_limit_down", False)

    # z-score 参数
    zscore_mean = None
    zscore_std = None
    if "zscore_mean" in first_model:
        zscore_mean = np.array(first_model["zscore_mean"])
        zscore_std = np.array(first_model["zscore_std"])

    # 判断特征维度（旧模型可能是 8 维含跌停，新模型 7 维不含跌停）
    n_features = len(first_model["means"][0])

    # 重建 HMM 对象
    hmm_objects = {}
    for code, info in models.items():
        hmm = GaussianHMM(n_components=info["n_states"], covariance_type=info["covariance_type"])
        hmm.startprob_ = np.array([1.0 / info["n_states"]] * info["n_states"])
        hmm.transmat_ = np.array(info["transmat"])
        hmm.means_ = np.array(info["means"])
        covars = np.array(info["covars"])
        if info["covariance_type"] == "full":
            for i in range(covars.shape[0]):
                covars[i] = (covars[i] + covars[i].T) / 2
                covars[i] += np.eye(covars[i].shape[0]) * 1e-6
        elif info["covariance_type"] == "diag":
            covars_raw = np.array(covars, dtype=np.float64)
            if covars_raw.ndim == 3:
                covars = np.array([np.diag(covars_raw[i]) for i in range(covars_raw.shape[0])])
            elif covars_raw.ndim == 2:
                expected_dim = len(info["means"][0])
                if covars_raw.shape[1] != expected_dim:
                    covars = covars_raw.reshape(info["n_states"], -1)[:, :expected_dim]
            covars = np.maximum(covars, 1e-6)
        hmm.covars_ = covars
        hmm_objects[code] = (hmm, info["state_labels"])

    # 解码 & 评估
    results_by_window = {w: {"trending": [], "fading": [], "neutral": []} for w in EVAL_WINDOWS}
    decoded_sectors = 0
    total_predictions = 0

    # 判断旧模型是否需要 8 维（含跌停）
    old_8dim = (n_features == 8 and not use_limit_down)
    # 旧 8 维模型：build_observation_matrix 需要 use_limit_down=True
    actual_use_ld = use_limit_down or old_8dim

    for code in models:
        if code not in sector_data:
            continue
        data_list = sorted(sector_data[code], key=lambda x: x["trade_date"])
        lu_data = limit_up_data.get(code, {})
        ld_data = limit_down_data.get(code, {})

        obs, obs_dates = build_observation_matrix(
            data_list, csi300, market_vol, lu_data, ld_data,
            rolling_window=rolling_window,
            use_limit_down=actual_use_ld,
        )
        if obs.shape[0] < 20:
            continue

        # 应用 z-score
        if zscore_mean is not None:
            obs = (obs - zscore_mean) / zscore_std

        hmm, labels = hmm_objects[code]
        try:
            states = hmm.predict(obs)
        except Exception:
            continue

        decoded_sectors += 1

        # 只看测试期
        date_label = {}
        for i, td in enumerate(obs_dates):
            if TEST_START <= td <= TEST_END:
                state_idx = states[i]
                date_label[td] = labels.get(str(state_idx), "unknown")

        if not date_label:
            continue

        pct_map = {r["trade_date"]: r["pct_change"] for r in data_list}
        extended_dates = sorted(set(r["trade_date"] for r in data_list if r["trade_date"] >= TEST_START))

        for td, label in date_label.items():
            if label == "unknown":
                continue
            td_idx = -1
            for idx, d in enumerate(extended_dates):
                if d == td:
                    td_idx = idx
                    break
            if td_idx < 0:
                continue
            total_predictions += 1
            for window in EVAL_WINDOWS:
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

    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else None

    metrics = {
        "display_name": display_name,
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "n_states": n_states,
        "cov_type": cov_type,
        "rolling_window": rolling_window,
        "zscore": zscore_mean is not None,
        "n_features": n_features,
        "decoded_sectors": decoded_sectors,
        "total_predictions": total_predictions,
    }

    for window in EVAL_WINDOWS:
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
            metrics[f"neutral_{window}d_n"] = len(n)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="HMM 模型测试集评估")
    parser.add_argument("--db-password", default=os.getenv("TDX_DB_PASSWORD", ""))
    args = parser.parse_args()

    print("=" * 80)
    print("HMM 模型测试集评估")
    print(f"测试期: {TEST_START} ~ {TEST_END}")
    print("=" * 80)

    conn = get_db_conn(password=args.db_password)

    # 列出所有模型
    configs = list_all_configs(conn)
    print(f"\n找到 {len(configs)} 个模型配置:")
    for i, cfg in enumerate(configs):
        print(f"  [{i+1}] {cfg['display_name']}  (config_id: {cfg['config_id'][:8]}...)")
    print()

    # 加载公共数据（一次性加载，所有模型共享）
    print("[1/2] 加载公共数据...")
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
    print(f"  行业数: {len(sector_data)}, CSI300 天数: {len(csi300)}")
    print(f"  涨停行业数: {len(limit_up_data)}, 跌停行业数: {len(limit_down_data)}")

    # 逐个评估
    print(f"\n[2/2] 评估 {len(configs)} 个模型...\n")
    all_metrics = []

    for cfg in configs:
        model_path = cfg["model_path"]
        display_name = cfg["display_name"]

        if not os.path.exists(model_path):
            print(f"  ✗ {display_name}: 模型文件不存在 ({model_path})")
            continue

        with open(model_path, "r") as f:
            models = json.load(f)

        print(f"  评估: {display_name} ({len(models)} 个行业)...")
        metrics = evaluate_model(
            models, display_name,
            sector_data, csi300, market_vol,
            limit_up_data, limit_down_data,
        )
        all_metrics.append(metrics)

    # 打印汇总表
    print("\n" + "=" * 80)
    print("测试集评估结果汇总")
    print(f"测试期: {TEST_START} ~ {TEST_END}")
    print("=" * 80)

    # 表头
    header = f"{'模型':<40} {'1d':>8} {'2d':>8} {'3d':>8} {'5d':>8} {'10d':>8} {'20d':>8} {'热N':>6} {'冷N':>6}"
    print(header)
    print("-" * len(header))

    # 按 1d spread 降序排列
    sorted_metrics = sorted(
        all_metrics,
        key=lambda m: m.get("spread_1d") or -999,
        reverse=True,
    )

    for m in sorted_metrics:
        name = m["display_name"][:38]
        vals = []
        for w in EVAL_WINDOWS:
            s = m.get(f"spread_{w}d")
            vals.append(f"{s:+.4f}" if s is not None else "   N/A")
        tn = m.get("trending_1d_n", 0)
        fn = m.get("fading_1d_n", 0)
        print(f"{name:<40} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {vals[5]:>8} {tn:>6} {fn:>6}")

    # 打印每个模型的详细信息
    print("\n" + "=" * 80)
    print("各模型详细指标")
    print("=" * 80)

    for m in sorted_metrics:
        print(f"\n── {m['display_name']} ──")
        print(f"   状态数: {m['n_states']}, 协方差: {m['cov_type']}, "
              f"窗口: {m['rolling_window']}d, zscore: {m['zscore']}, "
              f"特征维度: {m['n_features']}")
        print(f"   解码行业: {m['decoded_sectors']}, 总预测: {m['total_predictions']}")
        print(f"   {'窗口':<6} {'热态':>10} {'冷态':>10} {'差值':>10} {'热N':>8} {'冷N':>8}", end="")
        # 检查是否有 neutral
        has_neutral = any(m.get(f"neutral_{w}d") is not None for w in EVAL_WINDOWS)
        if has_neutral:
            print(f" {'中性':>10} {'中N':>8}", end="")
        print()
        print("   " + "-" * 70)
        for w in EVAL_WINDOWS:
            t = m.get(f"trending_{w}d")
            f = m.get(f"fading_{w}d")
            s = m.get(f"spread_{w}d")
            tn = m.get(f"trending_{w}d_n", 0)
            fn = m.get(f"fading_{w}d_n", 0)
            t_s = f"{t:.4f}" if t is not None else "N/A"
            f_s = f"{f:.4f}" if f is not None else "N/A"
            s_s = f"{s:+.4f}" if s is not None else "N/A"
            line = f"   {w}日{'':<3} {t_s:>10} {f_s:>10} {s_s:>10} {tn:>8} {fn:>8}"
            if has_neutral:
                n = m.get(f"neutral_{w}d")
                nn = m.get(f"neutral_{w}d_n", 0)
                n_s = f"{n:.4f}" if n is not None else "N/A"
                line += f" {n_s:>10} {nn:>8}"
            print(line)

    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
