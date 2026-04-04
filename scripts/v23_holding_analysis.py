"""持仓周期深度分析: 排名衰减、n_drop效应、持仓优化空间"""
import pickle, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

base = Path("qe_workspace/qe_20260403_012723")
with open(list((base / "Loop1/mlruns").rglob("pred.pkl"))[0], "rb") as f:
    pred = pickle.load(f)
with open(list((base / "Loop1/mlruns").rglob("positions_normal_1day.pkl"))[0], "rb") as f:
    positions = pickle.load(f)

# =============================================
# 1. 理解 pred 中的排名: 全量 vs 可交易
# =============================================
print("=" * 60)
print("1. pred 排名说明")
print("=" * 60)
print(f"pred 包含 {pred.index.get_level_values(1).nunique()} 只股票的预测")
print(f"但 TopkDropout 只在可交易股票中选 Top50")
print(f"pred 全量排名 ≠ 可交易排名, 下面分析全量排名中的持仓分布")

# 提取每日持仓
daily_holdings = {}
for date, pos in sorted(positions.items()):
    try:
        pd2 = pos.position if hasattr(pos, "position") else {}
        stocks = set(s for s in pd2.keys() if s != "cash" and s != "now_account_value")
    except:
        stocks = set()
    daily_holdings[date] = stocks

dates = sorted(daily_holdings.keys())

# =============================================
# 2. 分析持仓股入选和退出时的排名
# =============================================
print("\n" + "=" * 60)
print("2. 持仓股入选/退出排名分析")
print("=" * 60)

# 追踪每只股票的入选/退出
active = {}  # stock -> entry_date_idx
entry_exit_data = []  # (stock, entry_date, exit_date, days, entry_rank, exit_rank)

for i, date in enumerate(dates):
    current = daily_holdings[date]
    if date not in pred.index.get_level_values(0):
        continue
    dp = pred.loc[date].dropna().sort_values("score", ascending=False)
    dp_rank = {s: r+1 for r, s in enumerate(dp.index)}

    # 新入选
    for s in current:
        if s not in active:
            active[s] = (i, dp_rank.get(s, 9999))

    # 退出
    exited = [s for s in active if s not in current]
    for s in exited:
        entry_i, entry_rank = active.pop(s)
        exit_rank = dp_rank.get(s, 9999)
        days = i - entry_i
        entry_exit_data.append({
            "stock": s,
            "entry_date": dates[entry_i],
            "exit_date": date,
            "days": days,
            "entry_rank": entry_rank,
            "exit_rank": exit_rank,
        })

# 仍在持仓的
for s, (entry_i, entry_rank) in active.items():
    dp = pred.loc[dates[-1]].dropna().sort_values("score", ascending=False)
    dp_rank = {s2: r+1 for r, s2 in enumerate(dp.index)}
    entry_exit_data.append({
        "stock": s,
        "entry_date": dates[entry_i],
        "exit_date": dates[-1],
        "days": len(dates) - entry_i,
        "entry_rank": entry_rank,
        "exit_rank": dp_rank.get(s, 9999),
    })

df_ee = pd.DataFrame(entry_exit_data)
print(f"总持仓记录: {len(df_ee)}")
print(f"\n入选时排名 (在全量{pred.loc[dates[0]].dropna().shape[0]}只中):")
print(f"  mean={df_ee['entry_rank'].mean():.0f} median={df_ee['entry_rank'].median():.0f}")
print(f"  Top50:  {(df_ee['entry_rank']<=50).sum()} ({(df_ee['entry_rank']<=50).mean()*100:.1f}%)")
print(f"  Top100: {(df_ee['entry_rank']<=100).sum()} ({(df_ee['entry_rank']<=100).mean()*100:.1f}%)")
print(f"  Top200: {(df_ee['entry_rank']<=200).sum()} ({(df_ee['entry_rank']<=200).mean()*100:.1f}%)")

print(f"\n退出时排名:")
print(f"  mean={df_ee['exit_rank'].mean():.0f} median={df_ee['exit_rank'].median():.0f}")
print(f"  仍在Top50:  {(df_ee['exit_rank']<=50).sum()} ({(df_ee['exit_rank']<=50).mean()*100:.1f}%)")
print(f"  跌出Top200: {(df_ee['exit_rank']>200).sum()} ({(df_ee['exit_rank']>200).mean()*100:.1f}%)")
print(f"  跌出Top500: {(df_ee['exit_rank']>500).sum()} ({(df_ee['exit_rank']>500).mean()*100:.1f}%)")

# 排名变化
df_ee["rank_change"] = df_ee["exit_rank"] - df_ee["entry_rank"]
print(f"\n排名变化 (exit - entry):")
print(f"  mean={df_ee['rank_change'].mean():.0f} (排名下降=正数)")
print(f"  排名上升: {(df_ee['rank_change']<0).mean()*100:.1f}%")
print(f"  排名下降: {(df_ee['rank_change']>0).mean()*100:.1f}%")

# =============================================
# 3. 按持仓天数分组分析
# =============================================
print("\n" + "=" * 60)
print("3. 持仓天数 vs 入选/退出排名")
print("=" * 60)

bins = [(1,2,"1-2天"), (3,5,"3-5天"), (6,10,"6-10天"), (11,20,"11-20天"), (21,50,"21-50天"), (51,999,"50+天")]
print(f"{'持仓天数':>8s} {'数量':>6s} {'入选排名':>8s} {'退出排名':>8s} {'排名变化':>8s} {'退出>500':>8s}")
for lo, hi, label in bins:
    sub = df_ee[(df_ee["days"]>=lo) & (df_ee["days"]<=hi)]
    if len(sub) < 5:
        continue
    print(f"  {label:>6s}  {len(sub):5d}  {sub['entry_rank'].median():7.0f}  "
          f"{sub['exit_rank'].median():7.0f}  {sub['rank_change'].median():+7.0f}  "
          f"{(sub['exit_rank']>500).mean()*100:6.1f}%")

# =============================================
# 4. n_drop=5 导致的"排队等卖"效应
# =============================================
print("\n" + "=" * 60)
print("4. n_drop=5 的排队效应")
print("=" * 60)

# 对于每个交易日, 看有多少持仓股已跌出Top50但还在等待被卖出
queue_data = []
for i, date in enumerate(dates):
    current = daily_holdings[date]
    if not current or date not in pred.index.get_level_values(0):
        continue
    dp = pred.loc[date].dropna().sort_values("score", ascending=False)
    dp_rank = {s: r+1 for r, s in enumerate(dp.index)}

    ranks = [dp_rank.get(s, 9999) for s in current if s in dp_rank]
    if not ranks:
        continue
    outside_50 = sum(1 for r in ranks if r > 50)
    outside_100 = sum(1 for r in ranks if r > 100)
    outside_500 = sum(1 for r in ranks if r > 500)
    queue_data.append({
        "date": date,
        "held": len(ranks),
        "outside_50": outside_50,
        "outside_100": outside_100,
        "outside_500": outside_500,
    })

df_q = pd.DataFrame(queue_data)
print(f"每日持仓中排名在Top50之外的股票数:")
print(f"  排名>50:  mean={df_q['outside_50'].mean():.1f} / {df_q['held'].mean():.0f}")
print(f"  排名>100: mean={df_q['outside_100'].mean():.1f}")
print(f"  排名>500: mean={df_q['outside_500'].mean():.1f}")
print(f"\n这些股票本应被替换,但n_drop=5每天只能卖5只,导致排队等卖")
print(f"如果n_drop=10, 理论上清队速度翻倍")
print(f"如果n_drop=50(无限制), 每天直接换为当日Top50")

# =============================================
# 5. 持仓股在可交易范围内的排名 (更准确)
# =============================================
print("\n" + "=" * 60)
print("5. 持仓股在可交易股票中的排名估算")
print("=" * 60)

# 假设约50%的股票当天可交易(非停牌/非ST等)
# 实际可交易数估计: 4269总预测 * ~85%可交易 ≈ 3600
# TopkDropout 选 Top50 / 3600 ≈ Top 1.4%

sample_dates = [dates[0], dates[100], dates[200], dates[300], dates[-1]]
for date in sample_dates:
    if date not in pred.index.get_level_values(0):
        continue
    current = daily_holdings[date]
    dp = pred.loc[date].dropna().sort_values("score", ascending=False)
    dp_rank = {s: r+1 for r, s in enumerate(dp.index)}
    total = len(dp)

    ranks = sorted([dp_rank[s] for s in current if s in dp_rank])
    if not ranks:
        continue
    in50 = sum(1 for r in ranks if r <= 50)
    in100 = sum(1 for r in ranks if r <= 100)
    pct_top = np.mean([r/total*100 for r in ranks])
    print(f"  {date.date()}: {len(ranks)}只, Top50={in50} Top100={in100} "
          f"平均百分位={pct_top:.1f}% 总预测={total}")

# =============================================
# 6. 持仓周期 vs 收益贡献 (用 pred score 近似)
# =============================================
print("\n" + "=" * 60)
print("6. 不同持仓周期的平均 score (alpha 近似)")
print("=" * 60)

# 计算每个持仓记录的平均 daily score
period_scores = defaultdict(list)
for _, row in df_ee.iterrows():
    s = row["stock"]
    start = row["entry_date"]
    end = row["exit_date"]
    days = row["days"]
    scores = []
    for date in dates:
        if date < start or date > end:
            continue
        if date in pred.index.get_level_values(0) and s in pred.loc[date].index:
            scores.append(float(pred.loc[date].loc[s, "score"]))
    if scores:
        avg_score = np.mean(scores)
        if days <= 2:
            period_scores["1-2天"].append(avg_score)
        elif days <= 5:
            period_scores["3-5天"].append(avg_score)
        elif days <= 10:
            period_scores["6-10天"].append(avg_score)
        elif days <= 20:
            period_scores["11-20天"].append(avg_score)
        elif days <= 50:
            period_scores["21-50天"].append(avg_score)
        else:
            period_scores["50+天"].append(avg_score)

print(f"{'持仓周期':>8s} {'数量':>6s} {'平均score':>10s} {'中位score':>10s}")
for label in ["1-2天", "3-5天", "6-10天", "11-20天", "21-50天", "50+天"]:
    if label in period_scores:
        sc = period_scores[label]
        print(f"  {label:>6s}  {len(sc):5d}  {np.mean(sc):+.4f}    {np.median(sc):+.4f}")

print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)
