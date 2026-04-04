"""分析持仓周期 vs 实际收益的关系"""
import pickle, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

base = Path("qe_workspace/qe_20260403_012723")
with open(list((base / "Loop1/mlruns").rglob("pred.pkl"))[0], "rb") as f:
    pred = pickle.load(f)
with open(list((base / "Loop1/mlruns").rglob("positions_normal_1day.pkl"))[0], "rb") as f:
    positions = pickle.load(f)
with open(list((base / "Loop1/mlruns").rglob("label.pkl"))[0], "rb") as f:
    label = pickle.load(f)

print(f"label shape: {label.shape}, columns: {label.columns.tolist()}")
print(f"label index names: {label.index.names}")
print(f"label sample:\n{label.head()}")

# 提取每日持仓
daily_holdings = {}
daily_amounts = {}  # stock -> amount/value
for date, pos in sorted(positions.items()):
    try:
        pd2 = pos.position if hasattr(pos, "position") else {}
        stocks = {}
        for s, v in pd2.items():
            if s == "cash" or s == "now_account_value":
                continue
            if hasattr(v, "amount"):
                stocks[s] = {"amount": v.amount, "price": v.price if hasattr(v, "price") else 0}
            elif isinstance(v, dict):
                stocks[s] = v
            else:
                stocks[s] = {"amount": v}
    except:
        stocks = {}
    daily_holdings[date] = set(stocks.keys())

dates = sorted(daily_holdings.keys())

# 追踪每次持仓的实际收益
print("\n" + "=" * 70)
print("持仓周期 vs 实际收益分析")
print("=" * 70)

# 用 label (通常是 T+1 或 T+2 的收益率) 来计算每只持仓的真实收益
label_col = label.columns[0]
print(f"label 列名: {label_col}")

# 追踪每个持仓段的累计收益
active = {}  # stock -> (entry_date_idx, cumulative_return, daily_returns)
completed = []  # (stock, days, cum_return, daily_rets)

for i, date in enumerate(dates):
    current = daily_holdings[date]

    # 新入选
    for s in current:
        if s not in active:
            active[s] = (i, [])

    # 在持仓中的, 记录当天收益
    for s in list(active.keys()):
        if s in current and date in label.index.get_level_values(0):
            try:
                ret = float(label.loc[(date, s), label_col])
                if np.isfinite(ret):
                    active[s][1].append(ret)
            except (KeyError, TypeError):
                pass

    # 退出的
    exited = [s for s in active if s not in current]
    for s in exited:
        entry_i, rets = active.pop(s)
        days = i - entry_i
        if days > 0 and len(rets) > 0:
            cum_ret = np.sum(rets)  # 简单累加日收益
            completed.append((s, days, cum_ret, rets))

# 仍在持仓的
for s, (entry_i, rets) in active.items():
    days = len(dates) - entry_i
    if days > 0 and len(rets) > 0:
        cum_ret = np.sum(rets)
        completed.append((s, days, cum_ret, rets))

print(f"\n完成的持仓段: {len(completed)}")

# 按持仓天数分组
print(f"\n{'持仓周期':>8s} {'数量':>6s} {'累计收益mean':>12s} {'累计收益median':>14s} {'日均收益':>10s} {'正收益%':>8s}")
bins = [(1,2,"1-2天"), (3,5,"3-5天"), (6,10,"6-10天"), (11,20,"11-20天"),
        (21,50,"21-50天"), (51,100,"51-100天"), (101,999,"100+天")]
for lo, hi, lbl in bins:
    sub = [c for c in completed if lo <= c[1] <= hi]
    if len(sub) < 3:
        continue
    cum_rets = [c[2] for c in sub]
    daily_rets = [c[2]/c[1] for c in sub]
    pos_pct = sum(1 for r in cum_rets if r > 0) / len(cum_rets) * 100
    print(f"  {lbl:>7s}  {len(sub):5d}  {np.mean(cum_rets):+10.4f}    {np.median(cum_rets):+12.4f}  "
          f"{np.mean(daily_rets):+8.5f}  {pos_pct:6.1f}%")

# 收益最好的 Top 20 持仓段
print(f"\n--- 收益最好的 Top 20 ---")
completed.sort(key=lambda x: -x[2])
for s, days, cum_ret, rets in completed[:20]:
    print(f"  {s} {days:4d}天 累计={cum_ret:+.4f} 日均={cum_ret/days:+.5f}")

# 收益最差的 Top 20
print(f"\n--- 收益最差的 Top 20 ---")
for s, days, cum_ret, rets in completed[-20:]:
    print(f"  {s} {days:4d}天 累计={cum_ret:+.4f} 日均={cum_ret/days:+.5f}")

# 收益 vs 持仓天数的相关性
days_arr = np.array([c[1] for c in completed])
rets_arr = np.array([c[2] for c in completed])
daily_rets_arr = np.array([c[2]/c[1] for c in completed])

print(f"\n--- 相关性分析 ---")
print(f"持仓天数 vs 累计收益 相关系数: {np.corrcoef(days_arr, rets_arr)[0,1]:.4f}")
print(f"持仓天数 vs 日均收益 相关系数: {np.corrcoef(days_arr, daily_rets_arr)[0,1]:.4f}")

# 关键: 如果只持有 Top50 信号强的，持仓天数自然更长
# 但这些股票是因为持有久所以赚钱, 还是因为信号持续强所以持有久且赚钱?
print(f"\n--- 因果分析: 信号持续性 vs 持仓周期 ---")
# 计算每个持仓段中, pred score 保持在 Top50 的天数比例
print("(待补充: 需要分析信号持续在Top排名的持仓段 vs 信号快速衰减的持仓段)")

# 按 TopK 设计逻辑分析: 哪些持仓贡献了正收益
total_pos_ret = sum(c[2] for c in completed if c[2] > 0)
total_neg_ret = sum(c[2] for c in completed if c[2] < 0)
print(f"\n--- 收益贡献 ---")
print(f"正收益总和: {total_pos_ret:+.2f}")
print(f"负收益总和: {total_neg_ret:+.2f}")

for lo, hi, lbl in bins:
    sub = [c for c in completed if lo <= c[1] <= hi]
    if not sub:
        continue
    pos = sum(c[2] for c in sub if c[2] > 0)
    neg = sum(c[2] for c in sub if c[2] < 0)
    net = pos + neg
    print(f"  {lbl:>8s}: 正={pos:+.2f} 负={neg:+.2f} 净={net:+.2f} "
          f"(占总正收益 {pos/total_pos_ret*100:.1f}%)")
