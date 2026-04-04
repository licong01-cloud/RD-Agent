"""分析 n_drop + 10%止损 的潜在效果"""
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

label_col = label.columns[0]

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
# 1. 每个持仓段的日收益序列 + 最大回撤
# =============================================
print("=" * 70)
print("1. 持仓段亏损分析")
print("=" * 70)

active = {}  # stock -> (entry_idx, [daily_rets])
completed = []  # (stock, entry_idx, days, daily_rets)

for i, date in enumerate(dates):
    current = daily_holdings[date]
    for s in current:
        if s not in active:
            active[s] = (i, [])
    for s in list(active.keys()):
        if s in current and date in label.index.get_level_values(0):
            try:
                ret = float(label.loc[(date, s), label_col])
                if np.isfinite(ret):
                    active[s][1].append(ret)
            except (KeyError, TypeError):
                active[s][1].append(0.0)
    exited = [s for s in active if s not in current]
    for s in exited:
        entry_i, rets = active.pop(s)
        days = i - entry_i
        if days > 0 and len(rets) > 0:
            completed.append((s, entry_i, days, rets))

for s, (entry_i, rets) in active.items():
    days = len(dates) - entry_i
    if days > 0 and len(rets) > 0:
        completed.append((s, entry_i, days, rets))

print(f"总持仓段: {len(completed)}")

# =============================================
# 2. 模拟止损: 如果在累计亏损达到X%时卖出
# =============================================
print("\n" + "=" * 70)
print("2. 止损模拟")
print("=" * 70)

for stop_pct in [0.05, 0.08, 0.10, 0.15, 0.20]:
    total_ret_no_stop = 0.0
    total_ret_with_stop = 0.0
    n_stopped = 0
    stopped_saved = 0.0  # 止损避免的亏损
    stopped_missed = 0.0  # 止损错过的后续收益
    n_stop_correct = 0  # 止损后继续跌的
    n_stop_wrong = 0    # 止损后反弹的

    for s, entry_i, days, rets in completed:
        cum_ret = np.sum(rets)
        total_ret_no_stop += cum_ret

        # 模拟止损
        cum = 0.0
        stopped = False
        stop_day = -1
        for d, r in enumerate(rets):
            cum += r
            if cum <= -stop_pct:
                stopped = True
                stop_day = d
                break

        if stopped:
            n_stopped += 1
            ret_at_stop = cum  # 止损时的累计收益
            ret_after_stop = np.sum(rets[stop_day+1:])  # 止损后的收益
            total_ret_with_stop += ret_at_stop

            if ret_after_stop < 0:
                n_stop_correct += 1  # 止损正确: 后续继续跌
            else:
                n_stop_wrong += 1    # 止损错误: 后续反弹

            stopped_saved += max(0, -ret_after_stop)  # 避免的亏损
            stopped_missed += max(0, ret_after_stop)   # 错过的收益
        else:
            total_ret_with_stop += cum_ret

    print(f"\n  止损阈值: -{stop_pct*100:.0f}%")
    print(f"  触发止损: {n_stopped} / {len(completed)} ({n_stopped/len(completed)*100:.1f}%)")
    print(f"  无止损总收益: {total_ret_no_stop:+.2f}")
    print(f"  有止损总收益: {total_ret_with_stop:+.2f}")
    print(f"  收益差: {total_ret_with_stop - total_ret_no_stop:+.2f}")
    if n_stopped > 0:
        print(f"  止损正确(后续继续跌): {n_stop_correct} ({n_stop_correct/n_stopped*100:.1f}%)")
        print(f"  止损错误(后续反弹):   {n_stop_wrong} ({n_stop_wrong/n_stopped*100:.1f}%)")
        print(f"  避免的亏损: +{stopped_saved:.2f}")
        print(f"  错过的收益: -{stopped_missed:.2f}")

# =============================================
# 3. 最大回撤分析 (持仓段级别)
# =============================================
print("\n" + "=" * 70)
print("3. 持仓段最大回撤分析")
print("=" * 70)

max_dd_list = []
for s, entry_i, days, rets in completed:
    cum = np.cumsum(rets)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()
    max_dd_list.append((s, days, np.sum(rets), max_dd))

max_dd_arr = np.array([x[3] for x in max_dd_list])
print(f"持仓段最大回撤:")
print(f"  mean={max_dd_arr.mean()*100:.2f}% median={np.median(max_dd_arr)*100:.2f}%")
print(f"  < -5%:  {(max_dd_arr < -0.05).sum()} ({(max_dd_arr < -0.05).mean()*100:.1f}%)")
print(f"  < -10%: {(max_dd_arr < -0.10).sum()} ({(max_dd_arr < -0.10).mean()*100:.1f}%)")
print(f"  < -15%: {(max_dd_arr < -0.15).sum()} ({(max_dd_arr < -0.15).mean()*100:.1f}%)")
print(f"  < -20%: {(max_dd_arr < -0.20).sum()} ({(max_dd_arr < -0.20).mean()*100:.1f}%)")

# 回撤 >10% 但最终正收益的
deep_dd = [(s, d, r, dd) for s, d, r, dd in max_dd_list if dd < -0.10]
recovered = [(s, d, r, dd) for s, d, r, dd in deep_dd if r > 0]
print(f"\n回撤 >10% 的持仓: {len(deep_dd)}")
print(f"  最终正收益(回撤后反弹): {len(recovered)} ({len(recovered)/max(len(deep_dd),1)*100:.1f}%)")
print(f"  最终负收益: {len(deep_dd)-len(recovered)} ({(len(deep_dd)-len(recovered))/max(len(deep_dd),1)*100:.1f}%)")

if deep_dd:
    print(f"\n回撤>10%的持仓详情 (前20):")
    deep_dd.sort(key=lambda x: x[3])
    for s, d, r, dd in deep_dd[:20]:
        outcome = "反弹" if r > 0 else "亏损"
        print(f"  {s} {d:4d}天 最大回撤={dd*100:+.1f}% 最终={r*100:+.1f}% [{outcome}]")

# =============================================
# 4. n_drop + 止损 组合效果
# =============================================
print("\n" + "=" * 70)
print("4. n_drop + 止损 组合效果估算")
print("=" * 70)

# 当前: n_drop=5, 无止损
# 方案A: n_drop=5 + 10%止损
# 方案B: n_drop=3 + 10%止损 (减少换手+止损)
# 方案C: n_drop=5 + 15%止损 (更宽松止损)

print("\n注意: 止损释放的资金会重新投入(买入当日Top排名股票)")
print("此处简化假设: 止损释放的资金的后续收益 = 全策略平均日收益")

# 计算平均日收益
all_daily_rets = []
for s, entry_i, days, rets in completed:
    all_daily_rets.extend(rets)
avg_daily_ret = np.mean(all_daily_rets)
print(f"全策略平均日收益: {avg_daily_ret*100:.4f}%/天")

for stop_pct in [0.10, 0.15]:
    total_original = 0.0
    total_stopped = 0.0
    n_stopped = 0
    freed_days = 0  # 止损后释放的资金天数

    for s, entry_i, days, rets in completed:
        total_original += np.sum(rets)

        cum = 0.0
        stopped = False
        for d, r in enumerate(rets):
            cum += r
            if cum <= -stop_pct:
                stopped = True
                total_stopped += cum  # 止损时已有的亏损
                remaining_days = len(rets) - d - 1
                freed_days += remaining_days
                # 释放资金按平均收益再投资
                total_stopped += remaining_days * avg_daily_ret
                n_stopped += 1
                break
        if not stopped:
            total_stopped += np.sum(rets)

    print(f"\n  止损 {stop_pct*100:.0f}%: 触发{n_stopped}次, 释放{freed_days}天资金")
    print(f"    原始总收益: {total_original:+.2f}")
    print(f"    止损后总收益: {total_stopped:+.2f}")
    print(f"    差异: {total_stopped-total_original:+.2f}")

# =============================================
# 5. 回撤分档 vs 后续走势 (判断止损的有效性)
# =============================================
print("\n" + "=" * 70)
print("5. 触及特定亏损后的后续走势概率")
print("=" * 70)

for threshold in [0.05, 0.08, 0.10, 0.15]:
    n_hit = 0
    n_continue_down = 0  # 继续跌 >5%
    n_recover = 0        # 最终回到入场价以上
    n_partial = 0        # 部分回收但仍亏损
    after_rets = []

    for s, entry_i, days, rets in completed:
        cum = 0.0
        for d, r in enumerate(rets):
            cum += r
            if cum <= -threshold:
                n_hit += 1
                after = rets[d+1:]
                if len(after) == 0:
                    n_continue_down += 1
                    after_rets.append(0)
                    break
                after_cum = np.sum(after)
                after_rets.append(after_cum)
                final = cum + after_cum
                if final >= 0:
                    n_recover += 1
                elif after_cum < -0.05:
                    n_continue_down += 1
                else:
                    n_partial += 1
                break

    if n_hit == 0:
        continue
    print(f"\n  触及 -{threshold*100:.0f}% 后 ({n_hit} 次):")
    print(f"    最终回本:     {n_recover:4d} ({n_recover/n_hit*100:.1f}%)")
    print(f"    继续跌>5%:    {n_continue_down:4d} ({n_continue_down/n_hit*100:.1f}%)")
    print(f"    部分回收:     {n_partial:4d} ({n_partial/n_hit*100:.1f}%)")
    print(f"    后续平均收益: {np.mean(after_rets):+.4f}")
