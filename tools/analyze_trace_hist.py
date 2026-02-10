"""
分析Trace.hist，统计因子总数
"""
import pickle
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def analyze_trace_hist(trace_file_path: str) -> Dict[str, Any]:
    """
    分析Trace.hist，统计因子总数

    Args:
        trace_file_path: Trace文件的路径

    Returns:
        统计结果字典
    """
    # 加载Trace
    with open(trace_file_path, "rb") as f:
        trace = pickle.load(f)

    # 统计信息
    stats = {
        "total_experiments": 0,
        "sota_experiments": 0,
        "non_sota_experiments": 0,
        "total_factors": 0,
        "sota_factors": 0,
        "non_sota_factors": 0,
        "factor_names": [],
        "sota_factor_names": [],
        "non_sota_factor_names": [],
        "experiments_detail": [],
    }

    # 遍历Trace.hist
    for idx, (experiment, feedback) in enumerate(trace.hist):
        stats["total_experiments"] += 1

        # 判断是否是SOTA实验
        is_sota = feedback.decision if hasattr(feedback, "decision") else False
        if is_sota:
            stats["sota_experiments"] += 1
        else:
            stats["non_sota_experiments"] += 1

        # 获取因子信息
        factor_names = []
        if hasattr(experiment, "sub_tasks") and experiment.sub_tasks:
            for task in experiment.sub_tasks:
                if hasattr(task, "get_task_information"):
                    task_info = task.get_task_information()
                    if isinstance(task_info, dict):
                        factor_name = task_info.get("factor_name", "Unknown")
                        factor_names.append(factor_name)

        # 统计因子数量
        stats["total_factors"] += len(factor_names)
        if is_sota:
            stats["sota_factors"] += len(factor_names)
            stats["sota_factor_names"].extend(factor_names)
        else:
            stats["non_sota_factors"] += len(factor_names)
            stats["non_sota_factor_names"].extend(factor_names)

        # 记录实验详情
        experiment_detail = {
            "index": idx,
            "is_sota": is_sota,
            "factor_count": len(factor_names),
            "factor_names": factor_names,
            "has_result": hasattr(experiment, "result") and experiment.result is not None,
        }

        # 如果有回测结果，记录关键指标
        if hasattr(experiment, "result") and experiment.result is not None:
            try:
                result_df = pd.DataFrame(experiment.result)
                important_metrics = ["IC", "1day.excess_return_with_cost.annualized_return", 
                                   "1day.excess_return_with_cost.max_drawdown"]
                for metric in important_metrics:
                    if metric in result_df.index:
                        experiment_detail[metric] = float(result_df.loc[metric, 0])
            except Exception as e:
                experiment_detail["result_error"] = str(e)

        stats["experiments_detail"].append(experiment_detail)

    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    """打印统计结果"""
    print("=" * 80)
    print("Trace.hist 统计结果")
    print("=" * 80)
    print(f"总实验数量: {stats['total_experiments']}")
    print(f"SOTA实验数量: {stats['sota_experiments']}")
    print(f"非SOTA实验数量: {stats['non_sota_experiments']}")
    print()
    print(f"总因子数量: {stats['total_factors']}")
    print(f"SOTA因子数量: {stats['sota_factors']}")
    print(f"非SOTA因子数量: {stats['non_sota_factors']}")
    print()
    print("SOTA因子列表:")
    for i, factor_name in enumerate(stats["sota_factor_names"], 1):
        print(f"  {i}. {factor_name}")
    print()
    print("非SOTA因子列表:")
    for i, factor_name in enumerate(stats["non_sota_factor_names"], 1):
        print(f"  {i}. {factor_name}")
    print()
    print("=" * 80)
    print("实验详情")
    print("=" * 80)
    for exp_detail in stats["experiments_detail"]:
        print(f"\n实验 #{exp_detail['index']}")
        print(f"  是否SOTA: {'是' if exp_detail['is_sota'] else '否'}")
        print(f"  因子数量: {exp_detail['factor_count']}")
        print(f"  因子名称: {', '.join(exp_detail['factor_names'])}")
        print(f"  有回测结果: {'是' if exp_detail['has_result'] else '否'}")
        if exp_detail['has_result']:
            for key in ["IC", "1day.excess_return_with_cost.annualized_return", 
                       "1day.excess_return_with_cost.max_drawdown"]:
                if key in exp_detail:
                    print(f"  {key}: {exp_detail[key]:.6f}")


def main():
    """主函数"""
    import sys

    # 获取Trace文件路径
    if len(sys.argv) > 1:
        trace_path = sys.argv[1]
    else:
        # 尝试多个可能的路径
        possible_paths = [
            "log/*/trace/*.pkl",
            "log/**/trace/*.pkl",
            "log/**/*.pkl",
        ]
        
        # 搜索trace文件
        import glob
        trace_files = []
        for pattern in possible_paths:
            trace_files.extend(glob.glob(pattern))
        
        if trace_files:
            # 选择最新的trace文件
            trace_path = max(trace_files, key=lambda p: p.stat().st_mtime)
            print(f"找到Trace文件: {trace_path}")
        else:
            print("错误: 找不到Trace文件")
            print("请尝试以下命令：")
            print("  python tools/analyze_trace_hist.py <trace_file_path>")
            print("或确保Trace文件存在于log/目录下")
            sys.exit(1)

    # 分析Trace.hist
    print(f"正在分析Trace文件: {trace_path}")
    stats = analyze_trace_hist(trace_path)

    # 打印统计结果
    print_stats(stats)

    # 保存统计结果到JSON
    import json
    output_file = Path(trace_path).parent / "trace_hist_stats.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n统计结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
