"""
提取所有 SOTA 因子信息
"""
import pickle
from pathlib import Path
import pandas as pd
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibFactorExperiment, QlibModelExperiment


def extract_sota_factors(log_folder: Path):
    """
    从日志文件夹中提取所有 SOTA 因子
    
    Args:
        log_folder: 日志文件夹路径，例如 f:/Dev/RD-Agent-main/log/qlib_quant_2025-12-21_10-30-00/
    
    Returns:
        sota_experiments: 所有 SOTA 因子实验列表
        sota_factor_names: 所有 SOTA 因子名称列表
        sota_results: 所有 SOTA 实验结果列表
    """
    session_folder = log_folder / "__session__"
    
    # 查找最新的 session 文件
    session_files = sorted(session_folder.glob("*.pkl"), key=lambda x: x.stat().st_mtime)
    if not session_files:
        print(f"未找到 session 文件在 {session_folder}")
        return None, None, None
    
    latest_session = session_files[-1]
    print(f"加载 session 文件: {latest_session}")
    
    # 加载 session
    with open(latest_session, "rb") as f:
        session = pickle.load(f)
    
    # 获取 trace.hist
    trace = session.trace
    hist = trace.hist
    
    print(f"\n总实验数量: {len(hist)}")
    
    # 提取 SOTA 因子
    sota_experiments = []
    sota_factor_names = []
    sota_results = []
    
    for idx, (exp, feedback) in enumerate(hist):
        if isinstance(exp, QlibFactorExperiment):
            if feedback.decision:  # SOTA 实验
                print(f"\n=== SOTA 实验 #{len(sota_experiments) + 1} (Loop {idx}) ===")
                print(f"Hypothesis: {exp.hypothesis.hypothesis}")
                print(f"Decision: {feedback.decision}")
                
                # 提取因子名称
                factor_names = []
                for task in exp.sub_tasks:
                    task_info = task.get_task_information()
                    factor_names.append(task_info.get("factor_name", "Unknown"))
                
                print(f"因子数量: {len(factor_names)}")
                print(f"因子名称: {factor_names}")
                
                # 提取结果
                if exp.result is not None:
                    result_df = pd.DataFrame(exp.result)
                    print(f"\n关键指标:")
                    important_metrics = ["IC", "1day.excess_return_with_cost.annualized_return", 
                                       "1day.excess_return_with_cost.max_drawdown"]
                    for metric in important_metrics:
                        if metric in result_df.index:
                            print(f"  {metric}: {result_df.loc[metric, 0]:.6f}")
                    
                    sota_results.append(result_df)
                else:
                    print("无结果数据")
                    sota_results.append(None)
                
                sota_experiments.append(exp)
                sota_factor_names.extend(factor_names)
    
    print(f"\n\n=== 总结 ===")
    print(f"SOTA 实验数量: {len(sota_experiments)}")
    print(f"SOTA 因子总数: {len(sota_factor_names)}")
    print(f"SOTA 因子列表: {sota_factor_names}")
    
    return sota_experiments, sota_factor_names, sota_results


def extract_all_loops(log_folder: Path):
    """
    提取所有 Loop 的信息
    
    Args:
        log_folder: 日志文件夹路径
    
    Returns:
        loops_info: 所有 Loop 的信息列表
    """
    session_folder = log_folder / "__session__"
    
    # 查找最新的 session 文件
    session_files = sorted(session_folder.glob("*.pkl"), key=lambda x: x.stat().st_mtime)
    if not session_files:
        print(f"未找到 session 文件在 {session_folder}")
        return None
    
    latest_session = session_files[-1]
    print(f"加载 session 文件: {latest_session}")
    
    # 加载 session
    with open(latest_session, "rb") as f:
        session = pickle.load(f)
    
    # 获取 loop_trace
    loop_trace = session.loop_trace
    
    print(f"\nLoop 数量: {len(loop_trace)}")
    
    loops_info = []
    for loop_id, steps in loop_trace.items():
        print(f"\n=== Loop {loop_id} ===")
        print(f"步骤数量: {len(steps)}")
        
        loop_info = {
            "loop_id": loop_id,
            "steps": len(steps),
            "start_time": steps[0].start if steps else None,
            "end_time": steps[-1].end if steps else None,
            "duration": (steps[-1].end - steps[0].start).total_seconds() if steps else 0
        }
        
        print(f"开始时间: {loop_info['start_time']}")
        print(f"结束时间: {loop_info['end_time']}")
        print(f"持续时间: {loop_info['duration']:.2f} 秒")
        
        loops_info.append(loop_info)
    
    return loops_info


def extract_all_experiments_with_decision(log_folder: Path):
    """
    提取所有有结论的实验（decision 不为 None）
    
    Args:
        log_folder: 日志文件夹路径
    
    Returns:
        experiments_with_decision: 所有有结论的实验列表
    """
    session_folder = log_folder / "__session__"
    
    # 查找最新的 session 文件
    session_files = sorted(session_folder.glob("*.pkl"), key=lambda x: x.stat().st_mtime)
    if not session_files:
        print(f"未找到 session 文件在 {session_folder}")
        return None
    
    latest_session = session_files[-1]
    print(f"加载 session 文件: {latest_session}")
    
    # 加载 session
    with open(latest_session, "rb") as f:
        session = pickle.load(f)
    
    # 获取 trace.hist
    trace = session.trace
    hist = trace.hist
    
    print(f"\n总实验数量: {len(hist)}")
    
    # 提取所有有结论的实验
    experiments_with_decision = []
    
    for idx, (exp, feedback) in enumerate(hist):
        if feedback.decision is not None:  # 有结论
            exp_type = "Factor" if isinstance(exp, QlibFactorExperiment) else "Model"
            print(f"\n=== 实验 #{idx} ({exp_type}) ===")
            print(f"Decision: {feedback.decision}")
            print(f"Reason: {feedback.reason}")
            
            exp_info = {
                "index": idx,
                "type": exp_type,
                "decision": feedback.decision,
                "reason": feedback.reason,
                "hypothesis": exp.hypothesis.hypothesis if hasattr(exp, 'hypothesis') else None,
                "result": exp.result
            }
            
            # 提取关键指标
            if exp.result is not None:
                result_df = pd.DataFrame(exp.result)
                important_metrics = ["IC", "1day.excess_return_with_cost.annualized_return", 
                                   "1day.excess_return_with_cost.max_drawdown"]
                for metric in important_metrics:
                    if metric in result_df.index:
                        exp_info[metric] = result_df.loc[metric, 0]
            
            experiments_with_decision.append(exp_info)
    
    print(f"\n\n=== 总结 ===")
    print(f"有结论的实验数量: {len(experiments_with_decision)}")
    accepted = sum(1 for exp in experiments_with_decision if exp['decision'])
    rejected = sum(1 for exp in experiments_with_decision if not exp['decision'])
    print(f"接受的实验: {accepted}")
    print(f"拒绝的实验: {rejected}")
    
    return experiments_with_decision


if __name__ == "__main__":
    # 设置日志文件夹路径
    log_folder = Path("f:/Dev/RD-Agent-main/log/qlib_quant_2025-12-21_10-30-00")
    
    # 如果日志文件夹不存在，提示用户
    if not log_folder.exists():
        print(f"日志文件夹不存在: {log_folder}")
        print("\n可用的日志文件夹:")
        log_root = Path("f:/Dev/RD-Agent-main/log")
        if log_root.exists():
            for folder in sorted(log_root.iterdir()):
                if folder.is_dir() and folder.name.startswith("qlib"):
                    print(f"  {folder}")
        exit(1)
    
    print("=" * 80)
    print("1. 提取所有 SOTA 因子")
    print("=" * 80)
    sota_experiments, sota_factor_names, sota_results = extract_sota_factors(log_folder)
    
    print("\n" + "=" * 80)
    print("2. 提取所有 Loop 信息")
    print("=" * 80)
    loops_info = extract_all_loops(log_folder)
    
    print("\n" + "=" * 80)
    print("3. 提取所有有结论的实验")
    print("=" * 80)
    experiments_with_decision = extract_all_experiments_with_decision(log_folder)
    
    # 保存结果到文件
    if sota_factor_names:
        output_file = log_folder / "sota_factors_summary.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("SOTA 因子汇总\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"SOTA 实验数量: {len(sota_experiments)}\n")
            f.write(f"SOTA 因子总数: {len(sota_factor_names)}\n\n")
            f.write("SOTA 因子列表:\n")
            for i, name in enumerate(sota_factor_names, 1):
                f.write(f"  {i}. {name}\n")
        
        print(f"\nSOTA 因子汇总已保存到: {output_file}")
    
    if experiments_with_decision:
        output_file = log_folder / "all_experiments_summary.csv"
        df = pd.DataFrame(experiments_with_decision)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"所有实验汇总已保存到: {output_file}")
