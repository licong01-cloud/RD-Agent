"""
分析 RD-Agent 因子与 Loop/Workspace 的追溯关系
"""
import json

# 加载数据
fc = json.load(open('RDagentDB/aistock/factor_catalog.json', 'r', encoding='utf-8'))
lc = json.load(open('RDagentDB/aistock/loop_catalog.json', 'r', encoding='utf-8'))

# 分析 RD-Agent 因子的 experiment_id
rdagent_factors = [f for f in fc['factors'] if f['source'] == 'rdagent_generated']
exp_ids = set(f['experiment_id'] for f in rdagent_factors if f.get('experiment_id'))

print("=" * 80)
print("Factor Catalog 分析")
print("=" * 80)
print(f"RD-Agent 因子总数: {len(rdagent_factors)}")
print(f"有 experiment_id 的因子数: {len(exp_ids)}")
print(f"唯一的 experiment_id 数量: {len(exp_ids)}")
print(f"\n示例 experiment_id:")
for exp_id in list(exp_ids)[:5]:
    print(f"  - {exp_id}")

# 分析 Loop Catalog
print("\n" + "=" * 80)
print("Loop Catalog 分析")
print("=" * 80)
print(f"Loop 总数: {len(lc['loops'])}")

# 统计 Loop 的 action 类型
actions = {}
for loop in lc['loops']:
    action = loop.get('action', 'unknown')
    actions[action] = actions.get(action, 0) + 1

print(f"\nLoop action 分布:")
for action, count in actions.items():
    print(f"  - {action}: {count}")

# 统计有 factor_names 的 loop
loops_with_factors = [loop for loop in lc['loops'] if loop.get('factor_names')]
print(f"\n有 factor_names 的 loop 数: {len(loops_with_factors)}")

# 分析关联关系
print("\n" + "=" * 80)
print("关联关系分析")
print("=" * 80)

# 方法1: 通过 experiment_id 关联
# experiment_id 格式: "YYYY-MM-DD_HH-MM-SS-XXXXXX/loop_number"
# 需要解析出 log_folder 和 loop_number

exp_id_to_loop = {}
for factor in rdagent_factors:
    exp_id = factor.get('experiment_id')
    if exp_id and '/' in exp_id:
        parts = exp_id.split('/')
        log_folder = parts[0]
        loop_num = parts[1] if len(parts) > 1 else None

        # 在 loop_catalog 中查找匹配的 loop
        for loop in lc['loops']:
            if loop.get('loop_id') == int(loop_num):
                exp_id_to_loop[exp_id] = {
                    'factor_name': factor['name'],
                    'loop_id': loop['loop_id'],
                    'workspace_path': loop.get('workspace_path'),
                    'factor_names_in_loop': loop.get('factor_names', [])
                }
                break

print(f"\n通过 experiment_id 关联到的 loop 数: {len(exp_id_to_loop)}")

# 方法2: 通过 factor_names 关联
# loop 中的 factor_names 字段包含该 loop 使用的因子名称
factor_to_loops = {}
for loop in lc['loops']:
    factor_names = loop.get('factor_names', [])
    for factor_name in factor_names:
        if factor_name not in factor_to_loops:
            factor_to_loops[factor_name] = []
        factor_to_loops[factor_name].append({
            'loop_id': loop['loop_id'],
            'workspace_path': loop.get('workspace_path'),
            'action': loop.get('action')
        })

print(f"通过 factor_names 关联到的因子数: {len(factor_to_loops)}")

# 分析 RD-Agent 因子的追溯情况
print("\n" + "=" * 80)
print("RD-Agent 因子追溯情况")
print("=" * 80)

factors_with_exp_id = [f for f in rdagent_factors if f.get('experiment_id')]
factors_without_exp_id = [f for f in rdagent_factors if not f.get('experiment_id')]

print(f"\n有 experiment_id 的因子数: {len(factors_with_exp_id)}")
print(f"无 experiment_id 的因子数: {len(factors_without_exp_id)}")

# 示例：有 experiment_id 的因子
if factors_with_exp_id:
    print(f"\n示例 - 有 experiment_id 的因子:")
    for factor in factors_with_exp_id[:3]:
        print(f"  因子名: {factor['name']}")
        print(f"  experiment_id: {factor['experiment_id']}")
        print(f"  created_at: {factor['created_at_utc']}")

# 示例：无 experiment_id 的因子
if factors_without_exp_id:
    print(f"\n示例 - 无 experiment_id 的因子:")
    for factor in factors_without_exp_id[:3]:
        print(f"  因子名: {factor['name']}")
        print(f"  created_at: {factor['created_at_utc']}")

# 分析 workspace_path 的可用性
print("\n" + "=" * 80)
print("Workspace 信息可用性")
print("=" * 80)

loops_with_workspace = [loop for loop in lc['loops'] if loop.get('workspace_path')]
print(f"\n有 workspace_path 的 loop 数: {len(loops_with_workspace)}")
print(f"总 loop 数: {len(lc['loops'])}")

if loops_with_workspace:
    print(f"\n示例 workspace_path:")
    for loop in loops_with_workspace[:3]:
        print(f"  Loop {loop['loop_id']}: {loop['workspace_path']}")

# 总结
print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"\n✅ RD-Agent 因子总数: {len(rdagent_factors)}")
print(f"✅ 有 experiment_id 的因子: {len(factors_with_exp_id)} ({len(factors_with_exp_id)/len(rdagent_factors)*100:.1f}%)")
print(f"✅ Loop catalog 中的 loop 数: {len(lc['loops'])}")
print(f"✅ 有 workspace_path 的 loop: {len(loops_with_workspace)} ({len(loops_with_workspace)/len(lc['loops'])*100:.1f}%)")
print(f"✅ 有 factor_names 的 loop: {len(loops_with_factors)} ({len(loops_with_factors)/len(lc['loops'])*100:.1f}%)")

print(f"\n追溯可行性:")
print(f"  - 通过 experiment_id: {'✅ 可行' if len(exp_id_to_loop) > 0 else '❌ 不可行'}")
print(f"  - 通过 factor_names: {'✅ 可行' if len(factor_to_loops) > 0 else '❌ 不可行'}")
print(f"  - 获取 workspace_path: {'✅ 可行' if len(loops_with_workspace) > 0 else '❌ 不可行'}")
