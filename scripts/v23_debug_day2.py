"""分析正常交易日的选股排名"""
import pickle, numpy as np, pandas as pd
from pathlib import Path

base = Path("qe_workspace/qe_20260403_012723")
with open(list((base / "Loop1/mlruns").rglob("pred.pkl"))[0], "rb") as f:
    pred = pickle.load(f)
with open(list((base / "Loop1/mlruns").rglob("positions_normal_1day.pkl"))[0], "rb") as f:
    positions = pickle.load(f)

dates = sorted(positions.keys())

# 关键问题: pred 的日期是T日, 但 TopkDropout 用的是T-1日的pred来决定T日持仓!
# Qlib 的选股逻辑: 用 D-1 的 pred score 在 D 日执行买卖
# 所以应该用 D-1 的 pred 排名来看 D 日的持仓

print("=" * 70)
print("关键: pred 日期 vs 持仓日期的对应关系")
print("TopkDropout 用 D-1 的 pred 在 D 日买入")
print("=" * 70)

# 分析几个典型日期
for target_idx in [5, 10, 50, 100, 200, 300]:
    if target_idx >= len(dates):
        break
    date = dates[target_idx]
    prev_date = dates[target_idx - 1]

    # 用前一天的pred
    if prev_date not in pred.index.get_level_values(0):
        continue
    dp = pred.loc[prev_date].dropna().sort_values("score", ascending=False)
    dp_rank = {s: i+1 for i, s in enumerate(dp.index)}

    # 当天持仓
    pos = positions[date]
    pd2 = pos.position if hasattr(pos, "position") else {}
    held = set(s for s in pd2 if s != "cash" and s != "now_account_value")

    # 前一天持仓
    prev_pos = positions[prev_date]
    prev_pd2 = prev_pos.position if hasattr(prev_pos, "position") else {}
    prev_held = set(s for s in prev_pd2 if s != "cash" and s != "now_account_value")

    new_today = held - prev_held
    sold_today = prev_held - held
    kept = held & prev_held

    # 用当天pred排名
    dp_today = pred.loc[date].dropna().sort_values("score", ascending=False)
    dp_today_rank = {s: i+1 for i, s in enumerate(dp_today.index)}

    # 持仓排名 (用前一天pred)
    ranks_prev = sorted([dp_rank.get(s, 9999) for s in held if s in dp_rank])
    # 持仓排名 (用当天pred)
    ranks_today = sorted([dp_today_rank.get(s, 9999) for s in held if s in dp_today_rank])

    print(f"\n--- {date.date()} (Day {target_idx}) ---")
    print(f"  持仓: {len(held)} 只, 新买: {len(new_today)}, 卖出: {len(sold_today)}, 保持: {len(kept)}")
    print(f"  用D-1({prev_date.date()})pred排名: mean={np.mean(ranks_prev):.0f} median={np.median(ranks_prev):.0f} Top50={sum(1 for r in ranks_prev if r<=50)}")
    print(f"  用D  ({date.date()})pred排名: mean={np.mean(ranks_today):.0f} median={np.median(ranks_today):.0f} Top50={sum(1 for r in ranks_today if r<=50)}")

    # 新买入的排名 (用D-1 pred)
    if new_today:
        new_ranks = sorted([dp_rank.get(s, 9999) for s in new_today if s in dp_rank])
        print(f"  新买入排名(D-1 pred): {new_ranks}")

    # 持仓中排名最差的 (用D-1 pred)
    worst = [(s, dp_rank.get(s, 9999)) for s in held if s in dp_rank]
    worst.sort(key=lambda x: -x[1])
    if len(worst) > 3:
        print(f"  排名最差3只(D-1 pred): {[(s, r) for s, r in worst[:3]]}")

# 进一步: 用D-1 pred重新计算全部日期的持仓排名
print("\n" + "=" * 70)
print("全量分析: 持仓排名 (用D-1 pred vs 用D pred)")
print("=" * 70)

all_ranks_prev = []
all_ranks_today = []
for i in range(1, len(dates)):
    date = dates[i]
    prev_date = dates[i-1]
    if prev_date not in pred.index.get_level_values(0):
        continue
    if date not in pred.index.get_level_values(0):
        continue

    pos = positions[date]
    pd2 = pos.position if hasattr(pos, "position") else {}
    held = set(s for s in pd2 if s != "cash" and s != "now_account_value")

    dp_prev = pred.loc[prev_date].dropna().sort_values("score", ascending=False)
    dp_today = pred.loc[date].dropna().sort_values("score", ascending=False)
    prev_rank = {s: r+1 for r, s in enumerate(dp_prev.index)}
    today_rank = {s: r+1 for r, s in enumerate(dp_today.index)}

    for s in held:
        if s in prev_rank:
            all_ranks_prev.append(prev_rank[s])
        if s in today_rank:
            all_ranks_today.append(today_rank[s])

all_ranks_prev = np.array(all_ranks_prev)
all_ranks_today = np.array(all_ranks_today)

print(f"\n用 D-1 pred 看持仓排名:")
print(f"  mean={all_ranks_prev.mean():.0f} median={np.median(all_ranks_prev):.0f}")
print(f"  Top50:  {(all_ranks_prev<=50).mean()*100:.1f}%")
print(f"  Top100: {(all_ranks_prev<=100).mean()*100:.1f}%")
print(f"  Top200: {(all_ranks_prev<=200).mean()*100:.1f}%")
print(f"  >500:   {(all_ranks_prev>500).mean()*100:.1f}%")

print(f"\n用 D 当天 pred 看持仓排名:")
print(f"  mean={all_ranks_today.mean():.0f} median={np.median(all_ranks_today):.0f}")
print(f"  Top50:  {(all_ranks_today<=50).mean()*100:.1f}%")
print(f"  Top100: {(all_ranks_today<=100).mean()*100:.1f}%")
print(f"  >500:   {(all_ranks_today>500).mean()*100:.1f}%")

# 分布
print(f"\n用 D-1 pred 排名分布:")
for lo, hi in [(1,50),(51,100),(101,200),(201,500),(501,1000),(1001,9999)]:
    cnt = ((all_ranks_prev>=lo)&(all_ranks_prev<=hi)).sum()
    print(f"  {lo:>4d}-{hi:<4d}: {cnt:6d} ({cnt/len(all_ranks_prev)*100:5.1f}%)")
