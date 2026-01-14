"""
统计所有RD-Agent任务中进入SOTA的模型类型
"""
import pickle
from pathlib import Path
from collections import Counter
import json

def analyze_sota_models(log_dir: str = "log"):
    """分析所有任务中的SOTA模型类型"""
    log_path = Path(log_dir)
    model_stats = Counter()
    model_details = []
    
    for task_dir in log_path.iterdir():
        if not task_dir.is_dir():
            continue
            
        # 查找session文件
        session_file = task_dir / "__session__" / "0" / "1_coding"
        if not session_file.exists():
            continue
        
        try:
            with open(session_file, 'rb') as f:
                session = pickle.load(f)
            
            # 遍历trace历史，查找SOTA模型
            for exp, feedback in session.trace.hist:
                if feedback.decision and hasattr(exp, 'sub_workspace_list'):
                    # 检查是否是模型实验
                    is_model_exp = False
                    model_type = None
                    model_code = None
                    
                    for sub_workspace in exp.sub_workspace_list:
                        if 'model.py' in sub_workspace.file_dict:
                            is_model_exp = True
                            model_code = sub_workspace.file_dict['model.py']
                            break
                    
                    if is_model_exp:
                        # 分析模型类型
                        if 'LightGBM' in model_code or 'lgb' in model_code:
                            model_type = 'LightGBM'
                        elif 'CatBoost' in model_code or 'cat' in model_code:
                            model_type = 'CatBoost'
                        elif 'LSTM' in model_code or 'torch.nn.LSTM' in model_code:
                            model_type = 'LSTM'
                        elif 'MLP' in model_code or 'torch.nn.Linear' in model_code:
                            model_type = 'MLP'
                        elif 'Transformer' in model_code or 'torch.nn.Transformer' in model_code:
                            model_type = 'Transformer'
                        elif 'XGBoost' in model_code or 'xgb' in model_code:
                            model_type = 'XGBoost'
                        else:
                            model_type = 'Unknown'
                        
                        model_stats[model_type] += 1
                        
                        model_details.append({
                            "task_id": task_dir.name,
                            "model_type": model_type,
                            "loop_id": session.trace.hist.index((exp, feedback)),
                            "performance": exp.result if hasattr(exp, 'result') else None,
                            "decision_reason": feedback.reason if hasattr(feedback, 'reason') else None
                        })
        
        except Exception as e:
            print(f"Error processing {task_dir.name}: {e}")
            continue
    
    return model_stats, model_details

if __name__ == "__main__":
    # 分析SOTA模型类型
    model_stats, model_details = analyze_sota_models()
    
    print("=" * 80)
    print("SOTA模型类型统计")
    print("=" * 80)
    print(f"\n总共有 {sum(model_stats.values())} 个SOTA模型")
    print("\n模型类型分布:")
    for model_type, count in model_stats.most_common():
        print(f"  {model_type}: {count} ({count/sum(model_stats.values())*100:.1f}%)")
    
    # 保存详细结果
    result = {
        "summary": {
            "total_models": sum(model_stats.values()),
            "model_types": dict(model_stats)
        },
        "details": model_details
    }
    
    output_file = "sota_model_types_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 打印前5个模型详情
    print("\n前5个SOTA模型详情:")
    for i, detail in enumerate(model_details[:5]):
        print(f"\n{i+1}. 任务ID: {detail['task_id']}")
        print(f"   模型类型: {detail['model_type']}")
        print(f"   Loop ID: {detail['loop_id']}")
        if detail['performance']:
            print(f"   性能指标: {detail['performance']}")
