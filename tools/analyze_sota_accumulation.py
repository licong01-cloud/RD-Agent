"""
分析不同任务目录下的 SOTA 因子和模型累积情况
"""
import json
import os
from collections import defaultdict

# 加载 factor_catalog
fc = json.load(open('RDagentDB/aistock/factor_catalog.json', 'r', encoding='utf-8'))

# 统计每个 experiment_id 的因子数量
exp_id_to_factors = defaultdict(list)
for factor in fc['factors']:
    if factor['source'] == 'rdagent_generated' and factor.get('experiment_id'):
        exp_id = factor['experiment_id']
        exp_id_to_factors[exp_id].append(factor['name'])

print("=" * 80)
print("不同任务（experiment_id）的因子分布")
print("=" * 80)

# 按 experiment_id 排序
sorted_exp_ids = sorted(exp_id_to_factors.keys(), key=lambda x: x.split('/')[0])

for exp_id in sorted_exp_ids[:15]:  # 显示前15个
    factor_count = len(exp_id_to_factors[exp_id])
    print(f"\n{exp_id}")
    print(f"  因子数量: {factor_count}")
    if factor_count <= 5:
        print(f"  因子列表: {exp_id_to_factors[exp_id]}")
    else:
        print(f"  前5个因子: {exp_id_to_factors[exp_id][:5]}")

# 分析 experiment_id 的时间顺序
print("\n" + "=" * 80)
print("Experiment ID 时间顺序分析")
print("=" * 80)

# 提取日期部分
exp_dates = {}
for exp_id in exp_id_to_factors.keys():
    parts = exp_id.split('/')
    if len(parts) >= 1:
        date_part = parts[0]
        exp_dates[exp_id] = date_part

# 按日期排序
sorted_by_date = sorted(exp_dates.items(), key=lambda x: x[1])

print(f"\n总共有 {len(sorted_by_date)} 个不同的实验任务")
print(f"\n最早的任务: {sorted_by_date[0][0]} ({sorted_by_date[0][1]})")
print(f"最晚的任务: {sorted_by_date[-1][0]} ({sorted_by_date[-1][1]})")

# 分析因子是否累积
print("\n" + "=" * 80)
print("因子累积分析")
print("=" * 80)

# 检查是否有重复的因子名称
all_factor_names = []
for factors in exp_id_to_factors.values():
    all_factor_names.extend(factors)

unique_factor_names = set(all_factor_names)
print(f"\n总因子实例数: {len(all_factor_names)}")
print(f"唯一因子名称数: {len(unique_factor_names)}")

if len(all_factor_names) != len(unique_factor_names):
    print(f"✅ 有重复因子: {len(all_factor_names) - len(unique_factor_names)} 个重复")
    
    # 找出重复的因子
    from collections import Counter
    factor_counts = Counter(all_factor_names)
    duplicates = {name: count for name, count in factor_counts.items() if count > 1}
    
    print(f"\n重复因子示例 (前5个):")
    for name, count in list(duplicates.items())[:5]:
        print(f"  {name}: 出现 {count} 次")
else:
    print(f"❌ 无重复因子")

# 分析每个任务的因子是否包含之前的因子
print("\n" + "=" * 80)
print("任务间因子关系分析")
print("=" * 80)

# 按时间顺序分析
task_sequence = []
for exp_id, date_part in sorted_by_date:
    task_sequence.append({
        'exp_id': exp_id,
        'date': date_part,
        'factors': set(exp_id_to_factors[exp_id])
    })

# 分析累积情况
print(f"\n按时间顺序分析因子变化:")
print(f"{'序号':<6} {'日期':<25} {'因子数':<8} {'新增因子':<10}")
print("-" * 60)

cumulative_factors = set()
for i, task in enumerate(task_sequence[:10]):  # 只显示前10个
    current_factors = task['factors']
    new_factors = current_factors - cumulative_factors
    
    print(f"{i+1:<6} {task['date']:<25} {len(current_factors):<8} {len(new_factors):<10}")
    
    cumulative_factors.update(current_factors)

print(f"\n累积因子总数: {len(cumulative_factors)}")

# 检查最新任务是否包含所有之前的因子
print("\n" + "=" * 80)
print("最新任务是否包含所有之前的因子？")
print("=" * 80)

if len(task_sequence) > 0:
    latest_task = task_sequence[-1]
    all_previous_factors = set()
    
    for task in task_sequence[:-1]:
        all_previous_factors.update(task['factors'])
    
    latest_factors = latest_task['factors']
    
    print(f"\n最新任务: {latest_task['exp_id']}")
    print(f"最新任务因子数: {len(latest_factors)}")
    print(f"之前所有任务因子数: {len(all_previous_factors)}")
    print(f"唯一因子总数: {len(unique_factor_names)}")
    
    # 检查是否包含
    contained = all_previous_factors.issubset(latest_factors)
    print(f"\n最新任务是否包含所有之前的因子: {'✅ 是' if contained else '❌ 否'}")
    
    if not contained:
        missing = all_previous_factors - latest_factors
        print(f"缺失的因子数: {len(missing)}")
        print(f"缺失的因子示例: {list(missing)[:5]}")
    
    # 检查最新任务是否是所有因子的并集
    union_all = set()
    for task in task_sequence:
        union_all.update(task['factors'])
    
    is_union = latest_factors == union_all
    print(f"\n最新任务是否是所有任务的因子并集: {'✅ 是' if is_union else '❌ 否'}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print(f"\n1. 总共有 {len(exp_id_to_factors)} 个不同的实验任务")
print(f"2. 总共有 {len(unique_factor_names)} 个唯一的 RD-Agent 演进因子")
print(f"3. 每个任务包含的因子数量不同")
print(f"4. {'最新任务包含所有之前的因子' if contained else '最新任务不包含所有之前的因子'}")
print(f"5. {'最新任务是所有任务的因子并集' if is_union else '最新任务不是所有任务的因子并集'}")
