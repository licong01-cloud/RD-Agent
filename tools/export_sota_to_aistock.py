"""
导出 SOTA 因子和模型到结构化数据
"""
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibFactorExperiment, QlibModelExperiment


def export_sota_to_structured_data(log_folder: Path, output_dir: Path = None):
    """
    导出 SOTA 因子和模型到结构化数据
    
    Args:
        log_folder: 日志文件夹路径
        output_dir: 输出目录，默认为 log_folder/workspace/
    
    Returns:
        export_files: 导出的文件列表
    """
    if output_dir is None:
        output_dir = log_folder / "workspace"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (output_dir / "factors").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
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
    
    # 提取 SOTA 因子
    sota_factors = []
    for idx, (exp, feedback) in enumerate(hist):
        if isinstance(exp, QlibFactorExperiment):
            if feedback.decision:
                for task_idx, sub_ws in enumerate(exp.sub_workspace_list):
                    if sub_ws is not None and hasattr(sub_ws, "file_dict"):
                        factor_code = sub_ws.file_dict.get("factor.py")
                        if factor_code:
                            task_info = exp.sub_tasks[task_idx].get_task_information()
                            factor_name = task_info.get("factor_name", f"factor_{len(sota_factors)}")
                            
                            # 保存因子代码
                            factor_file = output_dir / "factors" / f"{factor_name}.py"
                            with open(factor_file, "w", encoding="utf-8") as f:
                                f.write(factor_code)
                            
                            sota_factors.append({
                                "factor_name": factor_name,
                                "loop_id": idx,
                                "factor_type": "cross_section",
                                "description": exp.hypothesis.hypothesis,
                                "code_file": f"factors/{factor_name}.py",
                                "performance": {
                                    "IC": float(exp.result.get("IC", 0)),
                                    "ICIR": float(exp.result.get("ICIR", 0))
                                } if exp.result else {}
                            })
    
    # 提取 SOTA 模型
    sota_models = []
    for idx, (exp, feedback) in enumerate(hist):
        if isinstance(exp, QlibModelExperiment):
            if feedback.decision:
                for task_idx, sub_ws in enumerate(exp.sub_workspace_list):
                    if sub_ws is not None and hasattr(sub_ws, "file_dict"):
                        model_code = sub_ws.file_dict.get("model.py")
                        if model_code:
                            task_info = exp.sub_tasks[task_idx].get_task_information()
                            model_name = task_info.get("name", f"model_{len(sota_models)}")
                            
                            # 保存模型代码
                            model_file = output_dir / "models" / f"{model_name}.py"
                            with open(model_file, "w", encoding="utf-8") as f:
                                f.write(model_code)
                            
                            sota_models.append({
                                "model_name": model_name,
                                "loop_id": idx,
                                "model_type": task_info.get("model_type", "Tabular"),
                                "architecture": task_info.get("architecture", ""),
                                "code_file": f"models/{model_name}.py",
                                "hyperparameters": task_info.get("hyperparameters", {}),
                                "training_hyperparameters": task_info.get("training_hyperparameters", {}),
                                "performance": {
                                    "IC": float(exp.result.get("IC", 0)),
                                    "annualized_return": float(exp.result.get("1day.excess_return_with_cost.annualized_return", 0)),
                                    "max_drawdown": float(exp.result.get("1day.excess_return_with_cost.max_drawdown", 0))
                                } if exp.result else {}
                            })
    
    # 提取实验汇总
    experiments = []
    for idx, (exp, feedback) in enumerate(hist):
        exp_type = "factor" if isinstance(exp, QlibFactorExperiment) else "model"
        experiments.append({
            "loop_id": idx,
            "action": exp_type,
            "hypothesis": exp.hypothesis.hypothesis if hasattr(exp, "hypothesis") else "",
            "decision": feedback.decision,
            "reason": feedback.reason,
            "result": exp.result if exp.result else {}
        })
    
    # 生成 workspace_meta.json
    best_metrics = {}
    if sota_models:
        best_metrics = sota_models[-1]["performance"]
    elif sota_factors:
        best_metrics = sota_factors[-1]["performance"]
    
    workspace_meta = {
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "experiment_id": log_folder.name,
        "scenario": "qlib_quant",
        "total_loops": len(hist),
        "sota_factor_count": len(sota_factors),
        "sota_model_count": len(sota_models),
        "best_metrics": best_metrics
    }
    
    with open(output_dir / "workspace_meta.json", "w", encoding="utf-8") as f:
        json.dump(workspace_meta, f, ensure_ascii=False, indent=2)
    
    # 生成 sota_factors.json
    sota_factors_json = {
        "version": "1.0",
        "factors": sota_factors,
        "combined_factors_file": "data/combined_factors_df.parquet",
        "total_factors": len(sota_factors)
    }
    
    with open(output_dir / "sota_factors.json", "w", encoding="utf-8") as f:
        json.dump(sota_factors_json, f, ensure_ascii=False, indent=2)
    
    # 生成 sota_models.json
    sota_models_json = {
        "version": "1.0",
        "models": sota_models,
        "current_sota_model": sota_models[-1]["model_name"] if sota_models else None,
        "total_models": len(sota_models)
    }
    
    with open(output_dir / "sota_models.json", "w", encoding="utf-8") as f:
        json.dump(sota_models_json, f, ensure_ascii=False, indent=2)
    
    # 生成 experiment_summary.json
    experiment_summary = {
        "version": "1.0",
        "experiments": experiments,
        "total_experiments": len(experiments),
        "accepted_experiments": sum(1 for exp in experiments if exp["decision"]),
        "rejected_experiments": sum(1 for exp in experiments if not exp["decision"])
    }
    
    with open(output_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, ensure_ascii=False, indent=2)
    
    # 生成 data_profile.json
    data_profile = {
        "version": "1.0",
        "data_source": "AIstock A-share",
        "data_type": "daily_pv.h5",
        "market": "CN",
        "instruments": "non-ST, non-delisted",
        "data_split": {
            "train": {
                "start": "2010-01-07",
                "end": "2018-12-31"
            },
            "valid": {
                "start": "2019-01-01",
                "end": "2020-12-31"
            },
            "test": {
                "start": "2021-01-01",
                "end": "2025-12-01"
            }
        },
        "features": {
            "initial_factor_library_size": 158,
            "sota_factor_count": len(sota_factors)
        }
    }
    
    with open(output_dir / "data_profile.json", "w", encoding="utf-8") as f:
        json.dump(data_profile, f, ensure_ascii=False, indent=2)
    
    # 复制 combined_factors_df.parquet
    for loop_dir in log_folder.glob("loop_*"):
        combined_factors_file = loop_dir / "experiment_workspace" / "combined_factors_df.parquet"
        if combined_factors_file.exists():
            import shutil
            shutil.copy(combined_factors_file, output_dir / "data" / "combined_factors_df.parquet")
            break
    
    export_files = [
        "workspace_meta.json",
        "sota_factors.json",
        "sota_models.json",
        "experiment_summary.json",
        "data_profile.json"
    ]
    
    print(f"\n\n=== 导出完成 ===")
    print(f"导出目录: {output_dir}")
    print(f"导出文件: {export_files}")
    print(f"SOTA 因子数量: {len(sota_factors)}")
    print(f"SOTA 模型数量: {len(sota_models)}")
    
    return export_files


if __name__ == "__main__":
    import sys
    
    # 使用命令行参数或默认值
    if len(sys.argv) > 1:
        log_folder = Path(sys.argv[1])
    else:
        log_folder = Path("f:/Dev/RD-Agent-main/log/2026-01-13_06-56-49-446055")
    
    if not log_folder.exists():
        print(f"日志文件夹不存在: {log_folder}")
        exit(1)
    
    export_sota_to_structured_data(log_folder)
