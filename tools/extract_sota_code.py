"""
提取所有 SOTA 因子的源代码
"""
import pickle
from pathlib import Path
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibFactorExperiment, QlibModelExperiment


def extract_sota_factor_code(log_folder: Path, output_dir: Path = None):
    """
    提取所有 SOTA 因子的源代码
    
    Args:
        log_folder: 日志文件夹路径
        output_dir: 输出目录，默认为 log_folder/sota_factors_code/
    
    Returns:
        factor_codes: 因子代码字典 {factor_name: code}
    """
    if output_dir is None:
        output_dir = log_folder / "sota_factors_code"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 提取 SOTA 因子代码
    factor_codes = {}
    factor_info = []
    
    for idx, (exp, feedback) in enumerate(hist):
        if isinstance(exp, QlibFactorExperiment):
            if feedback.decision:  # SOTA 实验
                print(f"\n=== SOTA 实验 #{len(factor_info) + 1} (Loop {idx}) ===")
                print(f"Hypothesis: {exp.hypothesis.hypothesis}")
                
                # 提取因子代码
                for task_idx, sub_ws in enumerate(exp.sub_workspace_list):
                    if sub_ws is not None and hasattr(sub_ws, "file_dict"):
                        factor_code = sub_ws.file_dict.get("factor.py")
                        if factor_code:
                            # 获取因子名称
                            task_info = task.get_task_information()
                            factor_name = task_info.get("factor_name", f"factor_{len(factor_codes)}")
                            
                            # 保存因子代码
                            factor_file = output_dir / f"{factor_name}_loop{idx}.py"
                            with open(factor_file, "w", encoding="utf-8") as f:
                                f.write(factor_code)
                            
                            print(f"  因子 #{task_idx}: {factor_name}")
                            print(f"  代码已保存到: {factor_file}")
                            
                            factor_codes[factor_name] = factor_code
                            factor_info.append({
                                "loop_id": idx,
                                "factor_name": factor_name,
                                "task_idx": task_idx,
                                "hypothesis": exp.hypothesis.hypothesis,
                                "code_file": str(factor_file),
                                "reason": feedback.reason
                            })
    
    # 保存因子信息汇总
    if factor_info:
        import json
        info_file = output_dir / "factor_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(factor_info, f, ensure_ascii=False, indent=2)
        print(f"\n因子信息已保存到: {info_file}")
    
    print(f"\n\n=== 总结 ===")
    print(f"SOTA 因子数量: {len(factor_codes)}")
    print(f"源代码已保存到: {output_dir}")
    
    return factor_codes


def extract_sota_model_code(log_folder: Path, output_dir: Path = None):
    """
    提取所有 SOTA 模型的源代码
    
    Args:
        log_folder: 日志文件夹路径
        output_dir: 输出目录，默认为 log_folder/sota_models_code/
    
    Returns:
        model_codes: 模型代码字典 {model_name: code}
    """
    if output_dir is None:
        output_dir = log_folder / "sota_models_code"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 提取 SOTA 模型代码
    model_codes = {}
    model_info = []
    
    for idx, (exp, feedback) in enumerate(hist):
        if isinstance(exp, QlibModelExperiment):
            if feedback.decision:  # SOTA 实验
                print(f"\n=== SOTA 实验 #{len(model_info) + 1} (Loop {idx}) ===")
                print(f"Hypothesis: {exp.hypothesis.hypothesis}")
                
                # 提取模型代码
                for task_idx, sub_ws in enumerate(exp.sub_workspace_list):
                    if sub_ws is not None and hasattr(sub_ws, "file_dict"):
                        model_code = sub_ws.file_dict.get("model.py")
                        if model_code:
                            # 获取模型名称
                            task_info = task.get_task_information()
                            model_name = task_info.get("name", f"model_{len(model_codes)}")
                            
                            # 保存模型代码
                            model_file = output_dir / f"{model_name}_loop{idx}.py"
                            with open(model_file, "w", encoding="utf-8") as f:
                                f.write(model_code)
                            
                            print(f"  模型 #{task_idx}: {model_name}")
                            print(f"  代码已保存到: {model_file}")
                            
                            model_codes[model_name] = model_code
                            model_info.append({
                                "loop_id": idx,
                                "model_name": model_name,
                                "task_idx": task_idx,
                                "hypothesis": exp.hypothesis.hypothesis,
                                "code_file": str(model_file),
                                "reason": feedback.reason
                            })
    
    # 保存模型信息汇总
    if model_info:
        import json
        info_file = output_dir / "model_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print(f"\n模型信息已保存到: {info_file}")
    
    print(f"\n\n=== 总结 ===")
    print(f"SOTA 模型数量: {len(model_codes)}")
    print(f"源代码已保存到: {output_dir}")
    
    return model_codes


def extract_all_sota_code(log_folder: Path):
    """
    提取所有 SOTA 因子和模型的源代码
    
    Args:
        log_folder: 日志文件夹路径
    """
    print("=" * 80)
    print("提取 SOTA 因子源代码")
    print("=" * 80)
    factor_codes = extract_sota_factor_code(log_folder)
    
    print("\n" + "=" * 80)
    print("提取 SOTA 模型源代码")
    print("=" * 80)
    model_codes = extract_sota_model_code(log_folder)
    
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"SOTA 因子数量: {len(factor_codes) if factor_codes else 0}")
    print(f"SOTA 模型数量: {len(model_codes) if model_codes else 0}")


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
    
    # 提取所有 SOTA 代码
    extract_all_sota_code(log_folder)
