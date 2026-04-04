#!/usr/bin/env python
"""为所有已注册的 HMM 模型添加说明，并为最佳模型设置按天数分级的信号系数.

用法:
  cd /mnt/f/Dev/RD-Agent-main
  conda activate rdagent-gpu
  python model_training/hmm/update_model_descriptions.py lc78080808
"""
import json
import sys

import psycopg2
import psycopg2.extras

DB_PASSWORD = sys.argv[1] if len(sys.argv) > 1 else ""

# ─── 模型说明 ───
# display_name -> description
DESCRIPTIONS = {
    # ★ 新一代优化模型（推荐）
    "L2_3状态_diag_7维_w3_raw": (
        "【推荐】3状态(热/中/冷) + diag协方差 + 3日滚动窗口 + 无标准化。"
        "测试集(2025-04~2026-03)全窗口正spread: 1d+0.156%, 2d+0.225%, 5d+0.138%, 20d+0.853%。"
        "综合表现最佳，短期和长期信号均有效。"
    ),
    "L2_3状态_diag_7维_w5_raw": (
        "3状态(热/中/冷) + diag协方差 + 5日滚动窗口 + 无标准化。"
        "测试集全窗口正spread: 1d+0.108%, 5d+0.058%, 20d+0.851%。"
        "长期信号强，热态样本最精准(1970个)。"
    ),
    "L2_3状态_full_7维_w5_raw": (
        "3状态 + full协方差 + 5日窗口 + 无标准化。"
        "测试集: 1d+0.110%, 5d+0.046%, 20d+0.615%。"
        "full协方差捕捉特征交互，中长期表现好。"
    ),
    "L2_2状态_full_7维_w5_zscore": (
        "2状态(热/冷) + full协方差 + 5日窗口 + z-score标准化。"
        "测试集: 1d+0.113%, 5d+0.042%, 20d+0.203%。"
        "信号稳定，热态样本多(4913个)，适合保守策略。"
    ),
    "L2_2状态_diag_7维_w5_zscore": (
        "2状态 + diag协方差 + 5日窗口 + z-score标准化。"
        "测试集: 1d+0.103%, 5d+0.037%, 20d+0.212%。"
        "与full版本类似，计算更快。"
    ),
    "L2_3状态_diag_7维_w5_zscore": (
        "3状态 + diag + 5日窗口 + z-score。"
        "测试集: 1d+0.073%, 5d-0.117%。"
        "z-score对3状态模型不利，5d以上信号失效。"
    ),
    # 4状态模型（过拟合，不推荐）
    "L2_4状态_diag_7维_w5_raw": (
        "4状态 + diag + 5日窗口。验证集表现好但测试集过拟合: 5d-0.005%。"
        "9个行业transmat退化。不推荐使用。"
    ),
    "L2_4状态_diag_7维_w3_raw": (
        "4状态 + diag + 3日窗口。测试集过拟合: 5d-0.113%。"
        "不推荐使用。"
    ),
    # 旧版模型（20日窗口，已被新版超越）
    "L2_2状态_diag_7维_v6": "旧版: 2状态+diag+20日窗口。测试集1d+0.033%, 5d-0.167%。已被w5/w3版本超越。",
    "L2_2状态_diag_7维_v5": "旧版: 同v6，重复训练。",
    "L2_2状态_diag_7维_v4": "旧版: 同v6，重复训练。",
    "L2_2状态_diag_7维_v3": "旧版: 同v6，重复训练。",
    "L2_2状态_diag_7维_v2": "旧版: 同v6，重复训练。",
    "L2_2状态_diag_7维": "旧版: 最早的2状态模型，20日窗口。",
    "L2_2状态_diag_8维": "旧版: 含跌停占比的8维模型。测试集1d+0.025%，跌停维度降低效果。",
    "L2_3状态_diag_7维": "旧版: 3状态+20日窗口+8维(含跌停)。测试集1d+0.024%。",
    "L2二级行业2状态配置": "早期配置，模型文件已丢失。",
    "默认2状态配置": "早期默认配置，模型文件已丢失。",
}


def main():
    conn = psycopg2.connect(
        host="127.0.0.1", port=5432, user="postgres",
        password=DB_PASSWORD, dbname="aistock",
    )
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # 查询所有配置
    cur.execute(
        "SELECT config_id, display_name, config_json FROM model_train_configs "
        "WHERE model_type='sector_hmm'"
    )
    rows = cur.fetchall()

    updated = 0
    for row in rows:
        config_id = row["config_id"]
        name = row["display_name"]
        cj = row["config_json"]
        if isinstance(cj, str):
            cj = json.loads(cj)

        desc = DESCRIPTIONS.get(name)
        if desc is None:
            print(f"  跳过: {name} (无说明)")
            continue

        cj["description"] = desc
        cur.execute(
            "UPDATE model_train_configs SET config_json = %s WHERE config_id = %s",
            (json.dumps(cj, ensure_ascii=False), config_id),
        )
        updated += 1
        print(f"  更新: {name}")

    conn.commit()
    print(f"\n共更新 {updated} 个配置的说明。")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
