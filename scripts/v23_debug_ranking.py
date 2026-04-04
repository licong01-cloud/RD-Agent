"""Debug: 分析持仓股排名与 pred 排名的差异"""
import pickle, numpy as np, pandas as pd, yaml
from pathlib import Path

base = Path("qe_workspace/qe_20260403_012723")
with open(list((base / "Loop1/mlruns").rglob("pred.pkl"))[0], "rb") as f:
    pred = pickle.load(f)
with open(list((base / "Loop1/mlruns").rglob("positions_normal_1day.pkl"))[0], "rb") as f:
    positions = pickle.load(f)

sample_date = pd.Timestamp("2024-07-02")
day_pred = pred.loc[sample_date].copy().dropna()
day_pred = day_pred.sort_values("score", ascending=False)
day_pred["rank_all"] = range(1, len(day_pred) + 1)

print(f"当天有效预测: {len(day_pred)} 只股票")

pos = positions[sample_date]
pos_dict = pos.position if hasattr(pos, "position") else {}
held = set(s for s in pos_dict.keys() if s != "cash" and s != "now_account_value")
print(f"当天持仓: {len(held)} 只")

# 持仓股在全量排名中的位置
held_ranks = []
for s in held:
    if s in day_pred.index:
        r = int(day_pred.loc[s, "rank_all"])
        held_ranks.append((s, r, float(day_pred.loc[s, "score"])))
held_ranks.sort(key=lambda x: x[1])

print(f"\n持仓在pred中: {len(held_ranks)} 只")
ranks_only = [r for _, r, _ in held_ranks]
print(f"持仓排名: mean={np.mean(ranks_only):.0f} median={np.median(ranks_only):.0f}")
print(f"Top50中: {sum(1 for r in ranks_only if r <= 50)}")
print(f"Top100中: {sum(1 for r in ranks_only if r <= 100)}")
print(f"Top200中: {sum(1 for r in ranks_only if r <= 200)}")
print(f"500名外: {sum(1 for r in ranks_only if r > 500)}")

# Top 20
print(f"\n--- 全量 Top 20 ---")
for i, (idx, row) in enumerate(day_pred.head(20).iterrows()):
    mark = "HELD" if idx in held else ""
    print(f"  {i+1:3d}. {idx} score={row['score']:.4f} {mark}")

# 持仓中排名最好和最差
print(f"\n--- 持仓排名最好 10 只 ---")
for s, r, sc in held_ranks[:10]:
    print(f"  rank {r:4d}: {s} score={sc:.4f}")
print(f"\n--- 持仓排名最差 10 只 ---")
for s, r, sc in held_ranks[-10:]:
    print(f"  rank {r:4d}: {s} score={sc:.4f}")

# 看 score 分布: 持仓 vs Top50
top50_scores = day_pred.head(50)["score"]
held_scores = [sc for _, _, sc in held_ranks]
print(f"\n--- Score 分布 ---")
print(f"Top50 score: mean={top50_scores.mean():.4f} min={top50_scores.min():.4f}")
print(f"持仓 score:  mean={np.mean(held_scores):.4f} min={np.min(held_scores):.4f} max={np.max(held_scores):.4f}")
print(f"全量 score:  mean={day_pred['score'].mean():.4f} std={day_pred['score'].std():.4f}")

# 核心问题: pred 是否包含不可交易的股票?
# TopkDropout 在 Qlib 中使用 Exchange 过滤不可交易股票
# pred.pkl 包含所有有预测的股票, 不管是否可交易
# 所以排名 1988 可能是因为 pred 包含 4269 只股票,
# 但实际可交易的只有一部分, TopkDropout 只在可交易股票中选 Top50

# 检查策略配置
with open(base / "conf.yaml") as f:
    conf = yaml.safe_load(f)

strat = conf.get("port_analysis_config", {}).get("strategy", {})
print(f"\n--- 策略配置 ---")
print(f"strategy: {strat}")

# 检查多天的趋势
print(f"\n=== 多日分析: 持仓股在全量pred中的排名 ===")
dates = sorted(positions.keys())
for date in dates[:5] + dates[200:205]:
    if date not in pred.index.get_level_values(0):
        continue
    dp = pred.loc[date].copy().dropna().sort_values("score", ascending=False)
    dp["r"] = range(1, len(dp) + 1)

    pos_d = positions[date]
    pd2 = pos_d.position if hasattr(pos_d, "position") else {}
    h = set(s for s in pd2.keys() if s != "cash" and s != "now_account_value")

    ranks = [int(dp.loc[s, "r"]) for s in h if s in dp.index]
    if ranks:
        in50 = sum(1 for r in ranks if r <= 50)
        print(f"  {date.date()}: 持仓{len(ranks)}只, 排名mean={np.mean(ranks):.0f} "
              f"Top50={in50} 总预测={len(dp)}")
