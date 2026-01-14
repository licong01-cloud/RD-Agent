#!/usr/bin/env python3
"""
详细分析预测评分分布
特别关注低于 0.2 的评分分布
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# 配置路径
ARTIFACTS_PATH = Path("/mnt/f/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/dcf014c2ce2a4255bb795ee0f7eb7d9f/mlruns/891791629306182420/a334523060ec40d998f521fe3db0df87/artifacts")
OUTPUT_PATH = Path("/mnt/f/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/dcf014c2ce2a4255bb795ee0f7eb7d9f/backtest_analysis_report")

OUTPUT_PATH.mkdir(exist_ok=True)

print("=" * 80)
print("预测评分详细分布分析")
print("=" * 80)

# 读取预测数据
pred_file = ARTIFACTS_PATH / "pred.pkl"
print(f"\n读取预测数据: {pred_file}")
with open(pred_file, 'rb') as f:
    pred = pickle.load(f)

print(f"数据形状: {pred.shape}")
print(f"数据类型: {type(pred)}")
print(f"索引类型: {type(pred.index)}")
print(f"索引名称: {pred.index.names}")

# 如果索引是 MultiIndex，展开它
if isinstance(pred.index, pd.MultiIndex):
    pred = pred.reset_index()
    print(f"列名: {pred.columns.tolist()}")
    print(f"\n前5行:")
    print(pred.head())

# 评分统计
scores = pred['score']
print("\n" + "=" * 80)
print("【1】评分基本统计")
print("=" * 80)
print(f"总记录数: {len(scores)}")
print(f"均值: {scores.mean():.6f}")
print(f"中位数: {scores.median():.6f}")
print(f"标准差: {scores.std():.6f}")
print(f"最小值: {scores.min():.6f}")
print(f"最大值: {scores.max():.6f}")
print(f"25%分位数: {scores.quantile(0.25):.6f}")
print(f"75%分位数: {scores.quantile(0.75):.6f}")

# 详细分布
print("\n" + "=" * 80)
print("【2】评分详细分布")
print("=" * 80)

# 负值分布
negative_scores = scores[scores < 0]
print(f"\n负值评分:")
print(f"  数量: {len(negative_scores)} ({len(negative_scores)/len(scores)*100:.2f}%)")
print(f"  均值: {negative_scores.mean():.6f}")
print(f"  最小值: {negative_scores.min():.6f}")
print(f"  最大值: {negative_scores.max():.6f}")

# 0-0.2 分布
zero_to_02 = scores[(scores >= 0) & (scores < 0.2)]
print(f"\n0-0.2 评分:")
print(f"  数量: {len(zero_to_02)} ({len(zero_to_02)/len(scores)*100:.2f}%)")
print(f"  均值: {zero_to_02.mean():.6f}")
print(f"  中位数: {zero_to_02.median():.6f}")

# 细分 0-0.2 区间
bins_0_02 = [-0.1, 0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
labels_0_02 = ['<0', '0-0.02', '0.02-0.04', '0.04-0.06', '0.06-0.08', '0.08-0.10', 
               '0.10-0.12', '0.12-0.14', '0.14-0.16', '0.16-0.18', '0.18-0.20']
score_bins_0_02 = pd.cut(scores, bins=bins_0_02, labels=labels_0_02, include_lowest=True)
score_dist_0_02 = score_bins_0_02.value_counts().sort_index()

print(f"\n0-0.2 区间细分分布:")
for label, count in score_dist_0_02.items():
    pct = count / len(scores) * 100
    print(f"  {label}: {count:>10,} ({pct:>6.2f}%)")

# 0.2 以上分布
above_02 = scores[scores >= 0.2]
print(f"\n0.2 以上评分:")
print(f"  数量: {len(above_02)} ({len(above_02)/len(scores)*100:.2f}%)")

bins_above = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
labels_above = ['0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0+']
score_bins_above = pd.cut(scores, bins=bins_above, labels=labels_above)
score_dist_above = score_bins_above.value_counts().sort_index()

print(f"\n0.2 以上区间分布:")
for label, count in score_dist_above.items():
    pct = count / len(scores) * 100
    print(f"  {label}: {count:>10,} ({pct:>6.2f}%)")

# 时序分析（如果有日期信息）
print("\n" + "=" * 80)
print("【3】时序分析")
print("=" * 80)

if 'datetime' in pred.columns:
    pred['datetime'] = pd.to_datetime(pred['datetime'])
    pred['date'] = pred['datetime'].dt.date
    
    # 每日评分统计
    daily_stats = pred.groupby('date').agg({
        'score': ['mean', 'std', 'min', 'max', 'count']
    })
    daily_stats.columns = ['mean', 'std', 'min', 'max', 'count']
    
    print(f"\n每日评分统计:")
    print(f"  日期范围: {daily_stats.index.min()} ~ {daily_stats.index.max()}")
    print(f"  总天数: {len(daily_stats)}")
    print(f"\n前10天:")
    print(daily_stats.head(10))
    
    # 每日评分变化趋势
    print(f"\n评分变化趋势:")
    print(f"  早期均值（前100天）: {daily_stats['mean'].head(100).mean():.6f}")
    print(f"  后期均值（后100天）: {daily_stats['mean'].tail(100).mean():.6f}")
    
    # 每日评分超过0.2的数量
    daily_above_02 = pred[pred['score'] >= 0.2].groupby('date').size()
    print(f"\n每日评分>=0.2的股票数:")
    print(f"  平均: {daily_above_02.mean():.1f}")
    print(f"  最大: {daily_above_02.max()}")
    print(f"  最小: {daily_above_02.min()}")
    print(f"  中位数: {daily_above_02.median():.1f}")
    
    # 保存每日统计
    daily_stats.to_csv(OUTPUT_PATH / "daily_score_stats.csv")

# 最高评分分析
print("\n" + "=" * 80)
print("【4】最高评分分析")
print("=" * 80)

top_100 = scores.nlargest(100)
print(f"\nTop 100 评分:")
print(f"  最小值: {top_100.min():.6f}")
print(f"  最大值: {top_100.max():.6f}")
print(f"  均值: {top_100.mean():.6f}")
print(f"  中位数: {top_100.median():.6f}")

# 找出评分最高的记录
if 'datetime' in pred.columns and 'instrument' in pred.columns:
    top_10_records = pred.nlargest(10, 'score')
    print(f"\nTop 10 评分记录:")
    print(top_10_records[['datetime', 'instrument', 'score']].to_string(index=False))

# 保存详细分布
print("\n" + "=" * 80)
print("【5】保存结果")
print("=" * 80)

# 保存完整分布统计
dist_summary = pd.DataFrame({
    '区间': list(score_dist_0_02.index) + list(score_dist_above.index),
    '数量': list(score_dist_0_02.values) + list(score_dist_above.values),
    '占比': [c/len(scores)*100 for c in list(score_dist_0_02.values) + list(score_dist_above.values)]
})
dist_summary.to_csv(OUTPUT_PATH / "score_distribution_detail.csv", index=False)

# 保存评分统计
stats_summary = pd.DataFrame({
    '指标': ['总记录数', '均值', '中位数', '标准差', '最小值', '最大值', '25%分位数', '75%分位数',
             '负值数量', '负值占比', '0-0.2数量', '0-0.2占比', '0.2+数量', '0.2+占比'],
    '值': [len(scores), scores.mean(), scores.median(), scores.std(), scores.min(), scores.max(),
           scores.quantile(0.25), scores.quantile(0.75),
           len(negative_scores), len(negative_scores)/len(scores)*100,
           len(zero_to_02), len(zero_to_02)/len(scores)*100,
           len(above_02), len(above_02)/len(scores)*100]
})
stats_summary.to_csv(OUTPUT_PATH / "score_statistics_summary.csv", index=False)

print(f"\n已保存:")
print(f"  1. {OUTPUT_PATH / 'score_distribution_detail.csv'} - 评分详细分布")
print(f"  2. {OUTPUT_PATH / 'score_statistics_summary.csv'} - 评分统计汇总")
if 'datetime' in pred.columns:
    print(f"  3. {OUTPUT_PATH / 'daily_score_stats.csv'} - 每日评分统计")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
