#!/usr/bin/env python
"""为最佳模型写入按天数分级的两套信号系数预设.

QE 创建实验时可选择 preset_A（保守，最高1.05）或 preset_B（激进，最高1.10），
系统根据 rebalance_days 自动匹配对应天数的系数。

用法:
  cd /mnt/f/Dev/RD-Agent-main
  conda activate rdagent-gpu
  python model_training/hmm/update_signal_coefficients.py lc78080808
"""
import json
import sys

import psycopg2
import psycopg2.extras

DB_PASSWORD = sys.argv[1] if len(sys.argv) > 1 else ""

# 两套系数预设
SIGNAL_PRESETS = {
    "preset_A": {
        "label": "保守档（最高+5%）",
        "description": "基于测试集胜率差2-4%设计，热态最高加成5%，适合稳健策略",
        "coefficients": {
            "1":  {"trending": 1.05, "neutral": 1.00, "fading": 0.96},
            "2":  {"trending": 1.05, "neutral": 1.00, "fading": 0.96},
            "3":  {"trending": 1.05, "neutral": 1.00, "fading": 0.95},
            "5":  {"trending": 1.03, "neutral": 1.00, "fading": 0.97},
            "10": {"trending": 1.02, "neutral": 1.00, "fading": 0.98},
            "20": {"trending": 1.04, "neutral": 1.00, "fading": 0.96},
        },
    },
    "preset_B": {
        "label": "激进档（最高+10%）",
        "description": "热态最高加成10%，冷态最低-10%，适合追求超额收益",
        "coefficients": {
            "1":  {"trending": 1.10, "neutral": 1.00, "fading": 0.92},
            "2":  {"trending": 1.10, "neutral": 1.00, "fading": 0.92},
            "3":  {"trending": 1.10, "neutral": 1.00, "fading": 0.90},
            "5":  {"trending": 1.06, "neutral": 1.00, "fading": 0.94},
            "10": {"trending": 1.05, "neutral": 1.00, "fading": 0.96},
            "20": {"trending": 1.08, "neutral": 1.00, "fading": 0.93},
        },
    },
}

# 最佳模型 config_id 前缀
BEST_CONFIG_PREFIX = "564b407f"  # L2_3状态_diag_7维_w3_raw


def main():
    conn = psycopg2.connect(
        host="127.0.0.1", port=5432, user="postgres",
        password=DB_PASSWORD, dbname="aistock",
    )
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # 查找最佳模型
    cur.execute(
        "SELECT config_id, display_name, config_json FROM model_train_configs "
        "WHERE config_id LIKE %s",
        (BEST_CONFIG_PREFIX + "%",),
    )
    row = cur.fetchone()
    if not row:
        print(f"找不到 config_id 以 {BEST_CONFIG_PREFIX} 开头的配置")
        return

    config_id = row["config_id"]
    display_name = row["display_name"]
    cj = row["config_json"]
    if isinstance(cj, str):
        cj = json.loads(cj)

    print(f"更新模型: {display_name} ({config_id})")

    # 写入信号系数预设
    cj["signal_presets"] = SIGNAL_PRESETS

    # 同时更新旧的固定系数为 preset_A 的默认值（向后兼容）
    cj["trending_coeff"] = 1.05
    cj["fading_coeff"] = 0.96
    cj["neutral_coeff"] = 1.00

    cur.execute(
        "UPDATE model_train_configs SET config_json = %s WHERE config_id = %s",
        (json.dumps(cj, ensure_ascii=False), config_id),
    )
    conn.commit()

    print(f"已写入两套信号系数预设:")
    for key, preset in SIGNAL_PRESETS.items():
        print(f"  {key}: {preset['label']}")
        for days, coeffs in sorted(preset["coefficients"].items(), key=lambda x: int(x[0])):
            print(f"    {days:>2}d: 热态={coeffs['trending']:.2f}  中性={coeffs['neutral']:.2f}  冷态={coeffs['fading']:.2f}")

    cur.close()
    conn.close()
    print("\n完成!")


if __name__ == "__main__":
    main()
