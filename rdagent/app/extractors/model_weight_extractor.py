"""模型权重提取模块

从RD-Agent session中提取模型权重文件(model.pkl)
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from ..path_utils import get_rdagent_log_root

logger = logging.getLogger(__name__)


class ModelWeightExtractor:
    """模型权重提取器
    
    从RD-Agent session中提取最后一个SOTA因子实验的模型权重文件
    """
    
    def __init__(self):
        self.rdagent_log_root = get_rdagent_log_root()
    
    def extract_model_weight(self, task_id: str) -> Dict[str, Any]:
        """提取task的模型权重
        
        Args:
            task_id: RD-Agent task ID
            
        Returns:
            {
                'success': bool,
                'task_id': str,
                'model_weight': bytes or None,  # 模型权重二进制数据
                'source': str or None,  # 来源说明
                'error': str or None
            }
        """
        result = {
            'success': False,
            'task_id': task_id,
            'model_weight': None,
            'source': None,
            'error': None
        }
        
        try:
            task_log_dir = self.rdagent_log_root / task_id
            if not task_log_dir.exists():
                result['error'] = f"Task log目录不存在: {task_log_dir}"
                return result
            
            session_dir = task_log_dir / "__session__"
            if not session_dir.exists():
                result['error'] = f"Session目录不存在: {session_dir}"
                return result
            
            # 查找最新的session文件
            session_files = sorted(session_dir.glob("*.pkl"))
            if not session_files:
                result['error'] = "未找到session文件"
                return result
            
            latest_session = session_files[-1]
            
            # 加载session对象
            import pickle
            with open(latest_session, 'rb') as f:
                session_obj = pickle.load(f)
            
            # 从session中提取模型权重
            # 查找最后一个SOTA因子实验的模型权重
            if not hasattr(session_obj, 'hist') or not session_obj.hist:
                result['error'] = "Session中没有hist数据"
                return result
            
            # 倒序查找decision=True的实验
            for i in range(len(session_obj.hist) - 1, -1, -1):
                exp, decision = session_obj.hist[i]
                
                if not decision:
                    continue
                
                # 找到SOTA因子实验，查找对应的模型实验
                # 模型实验通常在SOTA因子实验之后
                model_exp_index = i + 1
                if model_exp_index >= len(session_obj.hist):
                    continue
                
                model_exp, _ = session_obj.hist[model_exp_index]
                
                # 从模型实验中提取model.pkl
                if hasattr(model_exp, "sub_workspace_list") and model_exp.sub_workspace_list:
                    file_dict = getattr(model_exp.sub_workspace_list[0], "file_dict", {})
                    
                    if "model.pkl" in file_dict:
                        model_bytes = file_dict["model.pkl"]
                        if isinstance(model_bytes, bytes):
                            result['success'] = True
                            result['model_weight'] = model_bytes
                            result['source'] = f"session_hist[{model_exp_index}].sub_workspace_list[0].file_dict['model.pkl']"
                            logger.info(f"[{task_id}] 成功提取模型权重: {len(model_bytes)} bytes")
                            return result
            
            result['error'] = "未找到模型权重文件"
            return result
            
        except Exception as e:
            result['error'] = f"提取失败: {str(e)}"
            logger.error(f"[{task_id}] 模型权重提取失败: {e}", exc_info=True)
            return result
    
    def get_model_weight_info(self, task_id: str) -> Dict[str, Any]:
        """获取模型权重信息（不返回二进制数据）
        
        Args:
            task_id: RD-Agent task ID
            
        Returns:
            {
                'success': bool,
                'task_id': str,
                'has_model_weight': bool,
                'size': int or None,  # 字节数
                'source': str or None,
                'error': str or None
            }
        """
        extract_result = self.extract_model_weight(task_id)
        
        return {
            'success': extract_result['success'],
            'task_id': task_id,
            'has_model_weight': extract_result['model_weight'] is not None,
            'size': len(extract_result['model_weight']) if extract_result['model_weight'] else None,
            'source': extract_result['source'],
            'error': extract_result['error']
        }
