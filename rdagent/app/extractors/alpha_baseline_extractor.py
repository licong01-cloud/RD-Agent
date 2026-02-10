"""
Alpha基线因子提取模块

功能：
1. 从model_meta.json提取模型训练时真实使用的Alpha158因子
2. 数据源：model_meta.json -> dataset_conf -> handler -> infer_processors -> FilterCol.col_list

注意：
- 严格使用模型训练时的FilterCol配置，不使用任何兜底方案
- 如果无法从model_meta.json提取，返回错误而非使用默认值
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AlphaBaselineExtractor:
    """Alpha基线因子提取器"""
    
    def __init__(self, rdagent_log_root: str):
        """
        初始化Alpha基线因子提取器
        
        Args:
            rdagent_log_root: RD-Agent log目录路径
        """
        self.rdagent_log_root = Path(rdagent_log_root)
        if not self.rdagent_log_root.exists():
            raise ValueError(f"RD-Agent log目录不存在: {rdagent_log_root}")
    
    def extract_alpha_factors_from_model_meta(self, model_meta: Dict) -> List[str]:
        """
        从model_meta.json的FilterCol配置提取Alpha158因子
        
        Args:
            model_meta: model_meta.json解析后的字典
            
        Returns:
            Alpha158因子列表
            
        Raises:
            ValueError: 如果无法提取FilterCol配置
        """
        try:
            # 严格从 dataset_conf.kwargs.handler.kwargs.infer_processors 提取
            dataset_conf = model_meta.get("dataset_conf", {})
            if not dataset_conf:
                raise ValueError("model_meta.json缺少dataset_conf字段")
            
            dataset_kwargs = dataset_conf.get("kwargs", {})
            handler = dataset_kwargs.get("handler", {})
            handler_kwargs = handler.get("kwargs", {})
            infer_processors = handler_kwargs.get("infer_processors", [])
            
            if not infer_processors:
                raise ValueError("model_meta.json中infer_processors为空")
            
            # 查找FilterCol处理器
            for processor in infer_processors:
                if processor.get("class") == "FilterCol":
                    proc_kwargs = processor.get("kwargs", {})
                    if proc_kwargs.get("fields_group") == "feature":
                        col_list = proc_kwargs.get("col_list", [])
                        if isinstance(col_list, list) and len(col_list) > 0:
                            logger.info(f"从FilterCol提取到{len(col_list)}个Alpha158因子")
                            return col_list.copy()
            
            raise ValueError(
                "无法从model_meta.json的infer_processors中找到FilterCol.col_list配置。"
                "这是模型训练时真实使用的Alpha158因子列表。"
            )
            
        except KeyError as e:
            raise ValueError(
                f"model_meta.json结构不正确，缺少必需字段: {e}. "
                "期望路径: dataset_conf.kwargs.handler.kwargs.infer_processors"
            )
    
    def extract_from_model_meta(self, task_id: str) -> Optional[Dict]:
        """
        从model_meta.json提取Alpha基线因子信息
        
        Args:
            task_id: task ID
            
        Returns:
            {
                'alpha_baseline_factors': List[str],  # 真实的Alpha158因子
                'model_meta': Dict,  # 原始model_meta
                'source': str  # 数据来源说明
            }
            如果文件不存在或提取失败返回None
        """
        task_folder = self.rdagent_log_root / task_id
        model_meta_file = task_folder / "model_meta.json"
        
        if not model_meta_file.exists():
            logger.warning(f"[{task_id}] model_meta.json不存在: {model_meta_file}")
            return None
        
        try:
            with open(model_meta_file, 'r', encoding='utf-8') as f:
                model_meta = json.load(f)
            
            # 从FilterCol提取真实Alpha158因子
            alpha_factors = self.extract_alpha_factors_from_model_meta(model_meta)
            
            logger.info(f"[{task_id}] 从model_meta.json提取{len(alpha_factors)}个真实Alpha158因子")
            
            return {
                'alpha_baseline_factors': alpha_factors,
                'model_meta': model_meta,
                'source': 'model_meta.json/dataset_conf/infer_processors/FilterCol.col_list'
            }
            
        except Exception as e:
            logger.error(f"[{task_id}] 从model_meta.json提取Alpha158因子失败: {e}")
            return None
    
    def extract_task_alpha_baseline(self, task_id: str) -> Dict:
        """
        提取单个task的Alpha基线因子信息
        
        严格从model_meta.json的FilterCol配置提取真实Alpha158因子
        
        Args:
            task_id: task ID
            
        Returns:
            {
                'task_id': str,
                'success': bool,
                'alpha_baseline_factors': List[str],  # 真实的Alpha158因子
                'factor_count': int,
                'model_meta': Optional[Dict],
                'source': Optional[str],  # 数据来源
                'error': Optional[str]
            }
        """
        result = {
            'task_id': task_id,
            'success': False,
            'alpha_baseline_factors': [],
            'factor_count': 0,
            'model_meta': None,
            'source': None,
            'error': None
        }
        
        try:
            # 从model_meta.json提取真实的Alpha158因子
            meta_result = self.extract_from_model_meta(task_id)
            
            if meta_result is None:
                result['error'] = f"无法从{task_id}/model_meta.json提取Alpha158因子。"
                result['error'] += " 要求：必须从model_meta.json的FilterCol配置中提取真实因子。"
                logger.error(f"[{task_id}] {result['error']}")
                return result
            
            alpha_factors = meta_result['alpha_baseline_factors']
            
            if not alpha_factors:
                result['error'] = f"从model_meta.json提取的Alpha158因子列表为空"
                logger.error(f"[{task_id}] {result['error']}")
                return result
            
            # 成功提取
            result['success'] = True
            result['alpha_baseline_factors'] = alpha_factors
            result['factor_count'] = len(alpha_factors)
            result['model_meta'] = meta_result['model_meta']
            result['source'] = meta_result['source']
            
            logger.info(
                f"[{task_id}] 成功提取{len(alpha_factors)}个真实Alpha158因子，"
                f"前5个: {alpha_factors[:5]}"
            )
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"[{task_id}] 提取Alpha158因子失败: {e}", exc_info=True)
        
        return result
