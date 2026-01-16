"""
获取所有有结论的 Loop 列表
"""
import pickle
from pathlib import Path
import pandas as pd
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibFactorExperiment, QlibModelExperiment


def get_loops_with_decision(log_folder: Path):
    """
    获取所有有结论的 Loop 列表
    
    Args:
        log_folder: 日志文件夹路径
    
    Returns:
        loops_with_decision: 所有有结论的 Loop 信息列表
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
    
    # 获取 trace.hist 和 loop_trace
    trace = session.trace
    hist = trace.hist
    loop_trace = session.loop_trace
    
    print(f"\n总实验数量: {len(hist)}")
    print(f"总 Loop 数量: {len(loop_trace)}")
    
    # 提取所有有结论的 Loop
    loops_with_decision = []
    
    for loop_id, steps in loop_trace.items():
        print(f"\n=== Loop {loop_id} ===")
        
        # 获取该 Loop 对应的实验索引
        # 注意：hist 的索引不一定等于 loop_id，因为可能有并行实验
        # 这里假设每个 Loop 对应一个实验（非并行模式）
        
        if loop_id < len(hist):
            exp, feedback = hist[loop_id]
            
            if feedback.decision is not None:  # 有结论
                exp_type = "Factor" if isinstance(exp, QlibFactorExperiment) else "Model"
                
                loop_info = {
                    "loop_id": loop_id,
                    "type": exp_type,
                    "decision": feedback.decision,
                    "reason": feedback.reason,
                    "start_time": steps[0].start if steps else None,
                    "end_time": steps[-1].end if steps else None,
                    "duration": (steps[-1].end - steps[0].start).total_seconds() if steps else 0,
                    "steps": len(steps)
                }
                
                print(f"类型: {exp_type}")
                print(f"Decision: {feedback.decision}")
                print(f"Reason: {feedback.reason}")
                print(f"持续时间: {loop_info['duration']:.2f} 秒")
                
                # 提取关键指标
                if exp.result is not None:
                    result_df = pd.DataFrame(exp.result)
                    important_metrics = ["IC", "1day.excess_return_with_cost.annualized_return", 
                                       "1day.excess_return_with_cost.max_drawdown"]
                    for metric in important_metrics:
                        if metric in result_df.index:
                            loop_info[metric] = result_df.loc[metric, 0]
                            print(f"{metric}: {result_df.loc[metric, 0]:.6f}")
                
                loops_with_decision.append(loop_info)
            else:
                print(f"无结论（decision = {feedback.decision}）")
        else:
            print(f"Loop {loop_id} 超出 hist 范围")
    
    print(f"\n\n=== 总结 ===")
    print(f"有结论的 Loop 数量: {len(loops_with_decision)}")
    accepted = sum(1 for loop in loops_with_decision if loop['decision'])
    rejected = sum(1 for loop in loops_with_decision if not loop['decision'])
    print(f"接受的 Loop: {accepted}")
    print(f"拒绝的 Loop: {rejected}")
    
    return loops_with_decision


def get_loops_with_decision_parallel(log_folder: Path):
    """
    获取所有有结论的 Loop 列表（支持并行模式）
    
    在并行模式下，一个 Loop 可能产生多个 Experiment
    需要通过 idx2loop_id 映射来找到对应的 Loop
    
    Args:
        log_folder: 日志文件夹路径
    
    Returns:
        loops_with_decision: 所有有结论的 Loop 信息列表
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
    
    # 获取 trace.hist、loop_trace 和 idx2loop_id
    trace = session.trace
    hist = trace.hist
    loop_trace = session.loop_trace
    idx2loop_id = trace.idx2loop_id
    
    print(f"\n总实验数量: {len(hist)}")
    print(f"总 Loop 数量: {len(loop_trace)}")
    print(f"idx2loop_id 映射: {idx2loop_id}")
    
    # 按 Loop ID 分组实验
    loop_experiments = {}
    for idx, (exp, feedback) in enumerate(hist):
        loop_id = idx2loop_id.get(idx, idx)  # 如果没有映射，默认使用 idx
        
        if loop_id not in loop_experiments:
            loop_experiments[loop_id] = []
        
        loop_experiments[loop_id].append({
            "exp": exp,
            "feedback": feedback,
            "idx": idx
        })
    
    # 提取所有有结论的 Loop
    loops_with_decision = []
    
    for loop_id, exps in loop_experiments.items():
        print(f"\n=== Loop {loop_id} ===")
        print(f"实验数量: {len(exps)}")
        
        # 获取 Loop 的步骤信息
        steps = loop_trace.get(loop_id, [])
        
        # 检查是否有实验有结论
        has_decision = any(exp['feedback'].decision is not None for exp in exps)
        
        if has_decision:
            # 获取有结论的实验
            decision_exps = [exp for exp in exps if exp['feedback'].decision is not None]
            
            for exp_info in decision_exps:
                exp = exp_info['exp']
                feedback = exp_info['feedback']
                exp_type = "Factor" if isinstance(exp, QlibFactorExperiment) else "Model"
                
                loop_info = {
                    "loop_id": loop_id,
                    "exp_idx": exp_info['idx'],
                    "type": exp_type,
                    "decision": feedback.decision,
                    "reason": feedback.reason,
                    "start_time": steps[0].start if steps else None,
                    "end_time": steps[-1].end if steps else None,
                    "duration": (steps[-1].end - steps[0].start).total_seconds() if steps else 0,
                    "steps": len(steps)
                }
                
                print(f"实验 #{exp_info['idx']} ({exp_type})")
                print(f"Decision: {feedback.decision}")
                print(f"Reason: {feedback.reason}")
                
                # 提取关键指标
                if exp.result is not None:
                    result_df = pd.DataFrame(exp.result)
                    important_metrics = ["IC", "1day.excess_return_with_cost.annualized_return", 
                                       "1day.excess_return_with_cost.max_drawdown"]
                    for metric in important_metrics:
                        if metric in result_df.index:
                            loop_info[metric] = result_df.loc[metric, 0]
                            print(f"{metric}: {result_df.loc[metric, 0]:.6f}")
                
                loops_with_decision.append(loop_info)
        else:
            print(f"无结论")
    
    print(f"\n\n=== 总结 ===")
    print(f"有结论的 Loop 数量: {len(set(loop['loop_id'] for loop in loops_with_decision))}")
    print(f"有结论的实验数量: {len(loops_with_decision)}")
    accepted = sum(1 for loop in loops_with_decision if loop['decision'])
    rejected = sum(1 for loop in loops_with_decision if not loop['decision'])
    print(f"接受的实验: {accepted}")
    print(f"拒绝的实验: {rejected}")
    
    return loops_with_decision


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
    print("获取所有有结论的 Loop 列表")
    print("=" * 80)
    
    # 尝试并行模式
    loops_with_decision = get_loops_with_decision_parallel(log_folder)
    
    # 保存结果到文件
    if loops_with_decision:
        output_file = log_folder / "loops_with_decision.csv"
        df = pd.DataFrame(loops_with_decision)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nLoop 列表已保存到: {output_file}")
