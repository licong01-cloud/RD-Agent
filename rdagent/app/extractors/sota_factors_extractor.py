"""
SOTA因子提取模块（V2 - 基于based_experiments链对齐parquet）

功能：
1. 从最后SOTA因子实验的based_experiments链+自身提取与parquet完全对齐的SOTA因子
2. 提取每个因子的源代码(factor.py)
3. 提取每个因子进入SOTA时那个LOOP的回测数据(exp.result)
4. 兼容中文因子名映射（从括号中提取英文名）

数据来源：
- RD-Agent log目录下的task目录
- session文件：__session__/*/3_feedback
- 因子代码：sub_workspace_list[idx].file_dict['factor.py']
- 回测数据：exp.result (pandas Series)

对齐原理：
- parquet中的因子列 = based_experiments链中所有final_decision=True的因子 + 当前实验自身final_decision=True的因子
- 这与factor_runner.py中process_factor_data的逻辑一致
"""

import logging
import pickle
import re
from collections import OrderedDict
from pathlib import Path, WindowsPath, PosixPath
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os

logger = logging.getLogger(__name__)


@dataclass
class FactorLoopMetrics:
    """因子LOOP回测指标"""
    ic: Optional[float] = None
    icir: Optional[float] = None
    rank_ic: Optional[float] = None
    rank_icir: Optional[float] = None
    annualized_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    information_ratio: Optional[float] = None
    sharpe: Optional[float] = None
    annualized_return_with_cost: Optional[float] = None
    max_drawdown_with_cost: Optional[float] = None
    information_ratio_with_cost: Optional[float] = None


@dataclass
class FactorDetails:
    """因子详细信息（包含LOOP指标）"""
    factor_name: str
    factor_formulation: str
    factor_description: str
    source_loop_id: int
    source_loop_tag: str
    metrics: FactorLoopMetrics


@dataclass
class AlignedFactorInfo:
    """与parquet对齐的SOTA因子完整信息"""
    factor_name: str
    source_code: Optional[str] = None
    source_code_length: int = 0
    source_from: str = ""
    exp_result: Optional[Any] = None
    exp_result_dict: Optional[Dict[str, Any]] = None
    factor_formulation: str = ""
    factor_description: str = ""
    variables: Optional[Dict] = None


class PathCompatUnpickler(pickle.Unpickler):
    """跨平台路径兼容的Unpickler
    
    处理pathlib在不同操作系统间序列化的兼容性问题
    用于加载在WSL（Linux）环境下创建的session文件到Windows环境
    """
    def find_class(self, module: str, name: str):
        if module == "pathlib":
            if name == "PosixPath":
                return WindowsPath if os.name == 'nt' else PosixPath
            elif name == "WindowsPath":
                return WindowsPath if os.name == 'nt' else PosixPath
            elif name == "Path":
                return Path
        return super().find_class(module, name)


class SOTAFactorsExtractor:
    """SOTA因子提取器"""
    
    def __init__(self, rdagent_log_root: str):
        """
        初始化SOTA因子提取器
        
        Args:
            rdagent_log_root: RD-Agent log目录路径
        """
        self.rdagent_log_root = Path(rdagent_log_root)
        if not self.rdagent_log_root.exists():
            raise ValueError(f"RD-Agent log目录不存在: {rdagent_log_root}")
    
    def load_session(self, task_id: str) -> Optional[object]:
        """
        加载task的session文件（使用数值排序找最大loop_id）
        
        Args:
            task_id: task ID
            
        Returns:
            session对象，如果加载失败返回None
        """
        task_folder = self.rdagent_log_root / task_id
        session_folder = task_folder / '__session__'
        
        if not session_folder.exists():
            logger.warning(f"[{task_id}] Session目录不存在: {session_folder}")
            return None
        
        # 获取所有loop文件夹，使用数值排序
        loop_dirs = []
        for d in session_folder.iterdir():
            if d.is_dir() and d.name.isdigit():
                loop_dirs.append((int(d.name), d))
        
        if not loop_dirs:
            logger.warning(f"[{task_id}] 未找到loop目录")
            return None
        
        # 按数值排序，从大到小尝试加载
        loop_dirs.sort(key=lambda x: x[0], reverse=True)
        
        for loop_id, loop_dir in loop_dirs:
            # 优先尝试 3_feedback
            feedback_file = loop_dir / '3_feedback'
            if feedback_file.exists():
                try:
                    with feedback_file.open('rb') as f:
                        session = PathCompatUnpickler(f).load()
                    logger.info(f"[{task_id}] 成功加载session from loop {loop_id}/3_feedback")
                    return session
                except Exception as e:
                    logger.warning(f"[{task_id}] 加载loop {loop_id}/3_feedback失败: {e}")
            
            # 尝试 1_coding
            coding_file = loop_dir / '1_coding'
            if coding_file.exists():
                try:
                    with coding_file.open('rb') as f:
                        session = PathCompatUnpickler(f).load()
                    logger.info(f"[{task_id}] 成功加载session from loop {loop_id}/1_coding")
                    return session
                except Exception as e:
                    logger.warning(f"[{task_id}] 加载loop {loop_id}/1_coding失败: {e}")
        
        logger.error(f"[{task_id}] 所有loop目录都无法加载session")
        return None
    
    def extract_sota_factors(self, session: object) -> List[str]:
        """
        从session提取所有decision=True的SOTA因子
        
        Args:
            session: session对象
            
        Returns:
            SOTA因子名称列表（去重但保持顺序）
        """
        if not hasattr(session, 'trace') or not hasattr(session.trace, 'hist'):
            logger.warning("Session没有trace.hist属性")
            return []
        
        factors = []
        
        for loop_id in range(len(session.trace.hist)):
            try:
                loop_exp, feedback = session.trace.hist[loop_id]
                exp_type = type(loop_exp).__name__
                
                # 必须是Factor类型且decision=True
                if 'Factor' not in exp_type:
                    continue
                if not (hasattr(feedback, 'decision') and feedback.decision):
                    continue
                
                # 获取sub_tasks因子名列表
                hyp_factors = []
                if hasattr(loop_exp, 'sub_tasks') and loop_exp.sub_tasks:
                    for task in loop_exp.sub_tasks:
                        fname = getattr(task, 'factor_name', None) or getattr(task, 'name', None)
                        hyp_factors.append(fname)

                if not hyp_factors:
                    continue

                # 通过prop_dev_feedback中的final_decision过滤
                # 只有decision=True且final_decision=True的因子才进入SOTA
                if hasattr(loop_exp, 'prop_dev_feedback') and loop_exp.prop_dev_feedback is not None:
                    pdf = loop_exp.prop_dev_feedback
                    feedback_list = getattr(pdf, 'feedback_list', None)
                    if feedback_list is None and hasattr(pdf, '__iter__'):
                        feedback_list = list(pdf)

                    if feedback_list:
                        for i, fb_item in enumerate(feedback_list):
                            if i >= len(hyp_factors) or not hyp_factors[i]:
                                continue
                            if fb_item is not None and hasattr(fb_item, 'final_decision'):
                                if fb_item.final_decision:
                                    factors.append(hyp_factors[i])
                    else:
                        for fname in hyp_factors:
                            if fname:
                                factors.append(fname)
                else:
                    for fname in hyp_factors:
                        if fname:
                            factors.append(fname)
            except Exception as e:
                logger.warning(f"Loop {loop_id}提取因子失败: {e}")
                continue
        
        # 去重但保持顺序
        seen = set()
        unique_factors = []
        for f in factors:
            if f not in seen:
                seen.add(f)
                unique_factors.append(f)
        
        return unique_factors
    
    def extract_factor_codes(self, session: object, factor_names: List[str]) -> Dict[str, str]:
        """
        提取因子源代码
        
        Args:
            session: session对象
            factor_names: 需要提取的因子名称列表
            
        Returns:
            {factor_name: code_content}字典
        """
        if not hasattr(session, 'trace') or not hasattr(session.trace, 'hist'):
            return {}
        
        factor_codes = {}
        
        for loop_id in range(len(session.trace.hist)):
            try:
                loop_exp, feedback = session.trace.hist[loop_id]
                exp_type = type(loop_exp).__name__
                
                # 必须是Factor类型且decision=True
                if 'Factor' not in exp_type:
                    continue
                if not (hasattr(feedback, 'decision') and feedback.decision):
                    continue
                
                # 从sub_workspace_list提取因子代码
                if hasattr(loop_exp, 'sub_tasks') and hasattr(loop_exp, 'sub_workspace_list'):
                    for idx, task in enumerate(loop_exp.sub_tasks):
                        if not hasattr(task, 'factor_name') or not task.factor_name:
                            continue
                        
                        factor_name = task.factor_name
                        
                        # 如果不在需要提取的列表中，跳过
                        if factor_name not in factor_names:
                            continue
                        
                        # 如果已经提取过，跳过
                        if factor_name in factor_codes:
                            continue
                        
                        # 从sub_workspace_list提取代码
                        if idx < len(loop_exp.sub_workspace_list):
                            workspace = loop_exp.sub_workspace_list[idx]
                            if hasattr(workspace, 'file_dict') and 'factor.py' in workspace.file_dict:
                                code_content = workspace.file_dict['factor.py']
                                if isinstance(code_content, bytes):
                                    code_content = code_content.decode('utf-8')
                                factor_codes[factor_name] = code_content
            except Exception as e:
                logger.warning(f"Loop {loop_id}提取因子代码失败: {e}")
                continue
        
        return factor_codes
    
    def _safe_get_float(self, result: Dict[str, Any], key: str) -> Optional[float]:
        """安全获取浮点数值"""
        try:
            val = result.get(key)
            if val is not None:
                return round(float(val), 5)
        except Exception:
            pass
        return None
    
    def _extract_loop_metrics(self, loop_exp: object) -> FactorLoopMetrics:
        """从loop_exp.result提取回测指标"""
        metrics = FactorLoopMetrics()
        
        if not hasattr(loop_exp, 'result') or loop_exp.result is None:
            return metrics
        
        result = loop_exp.result
        
        try:
            # 字典格式
            if hasattr(result, 'get'):
                metrics.ic = self._safe_get_float(result, 'IC')
                metrics.annualized_return = self._safe_get_float(
                    result, '1day.excess_return_without_cost.annualized_return'
                ) or self._safe_get_float(
                    result, '1day.excess_return_with_cost.annualized_return'
                )
                metrics.max_drawdown = self._safe_get_float(
                    result, '1day.excess_return_without_cost.max_drawdown'
                ) or self._safe_get_float(
                    result, '1day.excess_return_with_cost.max_drawdown'
                )
                metrics.information_ratio = self._safe_get_float(
                    result, '1day.excess_return_without_cost.information_ratio'
                ) or self._safe_get_float(
                    result, '1day.excess_return_with_cost.information_ratio'
                )
            # pandas Series格式
            elif hasattr(result, 'index'):
                if 'IC' in result.index:
                    metrics.ic = round(float(result['IC']), 5)
                # 其他字段类似处理
        except Exception as e:
            logger.warning(f"提取LOOP指标失败: {e}")
        
        return metrics
    
    def extract_factor_details(self, session: object) -> Dict[str, FactorDetails]:
        """提取所有SOTA因子的详细信息和LOOP指标（首次成为SOTA的LOOP）"""
        factor_details = {}
        
        for loop_id in range(len(session.trace.hist)):
            try:
                loop_exp, feedback = session.trace.hist[loop_id]
                exp_type = type(loop_exp).__name__
                
                # 必须是Factor类型且decision=True
                if 'Factor' not in exp_type:
                    continue
                if not (hasattr(feedback, 'decision') and feedback.decision):
                    continue
                
                # 提取该LOOP的回测指标
                loop_metrics = self._extract_loop_metrics(loop_exp)
                
                # 获取loop_tag
                loop_tag = getattr(loop_exp, 'loop_tag', f'{loop_id}_feedback')
                
                # 遍历该LOOP的所有因子
                if hasattr(loop_exp, 'sub_tasks') and loop_exp.sub_tasks:
                    for task in loop_exp.sub_tasks:
                        if not hasattr(task, 'factor_name') or not task.factor_name:
                            continue
                        
                        factor_name = task.factor_name
                        
                        # 只记录首次成为SOTA的LOOP（保持原始最佳表现）
                        if factor_name in factor_details:
                            continue
                        
                        details = FactorDetails(
                            factor_name=factor_name,
                            factor_formulation=getattr(task, 'factor_formulation', ''),
                            factor_description=getattr(task, 'factor_description', ''),
                            source_loop_id=loop_id,
                            source_loop_tag=loop_tag,
                            metrics=loop_metrics
                        )
                        factor_details[factor_name] = details
                        
            except Exception as e:
                logger.warning(f"Loop {loop_id}提取详情失败: {e}")
                continue
        
        return factor_details
    
    def extract_task_sota_factors(self, task_id: str) -> Dict:
        """
        提取单个task的完整SOTA因子信息
        
        Args:
            task_id: task ID
            
        Returns:
            {
                'task_id': str,
                'success': bool,
                'sota_factors': List[str],
                'factor_codes': Dict[str, str],
                'factor_count': int,
                'error': Optional[str]
            }
        """
        result = {
            'task_id': task_id,
            'success': False,
            'sota_factors': [],
            'factor_codes': {},
            'factor_details': {},  # 新增: 因子详情（包含LOOP指标和表达式）
            'factor_count': 0,
            'error': None
        }
        
        try:
            # 1. 加载session
            session = self.load_session(task_id)
            if not session:
                result['error'] = "无法加载session文件"
                return result
            
            # 2. 提取SOTA因子列表
            sota_factors = self.extract_sota_factors(session)
            if not sota_factors:
                result['error'] = "未找到SOTA因子"
                return result
            
            # 3. 提取因子源代码
            factor_codes = self.extract_factor_codes(session, sota_factors)
            
            # 4. 提取因子详情（新增: LOOP指标和表达式）
            factor_details = self.extract_factor_details(session)
            
            # 5. 返回结果
            result['success'] = True
            result['sota_factors'] = sota_factors
            result['factor_codes'] = factor_codes
            # 将FactorDetails对象转换为字典
            result['factor_details'] = {
                k: {
                    'factor_name': v.factor_name,
                    'factor_formulation': v.factor_formulation,
                    'factor_description': v.factor_description,
                    'source_loop_id': v.source_loop_id,
                    'source_loop_tag': v.source_loop_tag,
                    'metrics': {
                        'ic': v.metrics.ic,
                        'annualized_return': v.metrics.annualized_return,
                        'max_drawdown': v.metrics.max_drawdown,
                        'information_ratio': v.metrics.information_ratio,
                        'sharpe': v.metrics.sharpe
                    }
                }
                for k, v in factor_details.items()
            }
            result['factor_count'] = len(sota_factors)
            
            logger.info(f"[{task_id}] 成功提取{len(sota_factors)}个SOTA因子，包含LOOP指标和表达式")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"[{task_id}] 提取SOTA因子失败: {e}")
        
        return result

    # ================================================================
    # V2 核心方法：基于based_experiments链提取与parquet完全对齐的SOTA因子
    # ================================================================

    @staticmethod
    def _is_factor_experiment(exp: object) -> bool:
        try:
            return "factor" in type(exp).__name__.lower()
        except Exception:
            return False

    @staticmethod
    def _normalize_factor_name(name: str) -> str:
        """处理中文因子名映射：从括号中提取英文名
        
        例如：'主力资金净流入强度5日滚动（MainNetAmtRatio_5D）' -> 'MainNetAmtRatio_5D'
        """
        if not name:
            return name
        # 匹配中文括号或英文括号中的英文名
        m = re.search(r'[（(]([A-Za-z_][A-Za-z0-9_]*)[）)]', name)
        if m:
            return m.group(1)
        return name

    @staticmethod
    def _get_factor_info_from_exp(exp: object, idx: int) -> Tuple[Optional[str], Optional[bool]]:
        """从实验对象获取因子名和final_decision"""
        fname = None
        fd = None
        if hasattr(exp, "sub_tasks") and exp.sub_tasks and idx < len(exp.sub_tasks):
            task = exp.sub_tasks[idx]
            fname = getattr(task, "factor_name", None) or getattr(task, "name", None)
        if hasattr(exp, "prop_dev_feedback") and exp.prop_dev_feedback is not None:
            pdf = exp.prop_dev_feedback
            feedback_list = getattr(pdf, "feedback_list", None)
            if feedback_list is None and hasattr(pdf, "__iter__"):
                feedback_list = list(pdf)
            if feedback_list and idx < len(feedback_list):
                fb_item = feedback_list[idx]
                if fb_item is not None and hasattr(fb_item, "final_decision"):
                    fd = fb_item.final_decision
        return fname, fd

    @staticmethod
    def _get_factor_source_code(exp: object, idx: int) -> Optional[str]:
        """从实验对象获取因子源代码"""
        if hasattr(exp, "sub_workspace_list") and exp.sub_workspace_list:
            if idx < len(exp.sub_workspace_list):
                ws = exp.sub_workspace_list[idx]
                if ws is not None and hasattr(ws, "file_dict") and isinstance(ws.file_dict, dict):
                    code = ws.file_dict.get("factor.py", None)
                    if isinstance(code, bytes):
                        code = code.decode("utf-8", errors="ignore")
                    return code
        return None

    @staticmethod
    def _get_factor_task_details(exp: object, idx: int) -> Tuple[str, str, Optional[Dict]]:
        """从实验对象获取因子的formulation、description、variables"""
        formulation = ""
        description = ""
        variables = None
        if hasattr(exp, "sub_tasks") and exp.sub_tasks and idx < len(exp.sub_tasks):
            task = exp.sub_tasks[idx]
            formulation = getattr(task, "factor_formulation", "") or ""
            description = getattr(task, "factor_description", "") or ""
            variables = getattr(task, "variables", None)
        return formulation, description, variables

    def _extract_metrics_from_result(self, result: Any) -> Dict[str, Any]:
        """从exp.result (pandas Series或dict) 提取完整回测指标"""
        metrics = {}
        if result is None:
            return metrics
        
        try:
            # 转换为dict
            if hasattr(result, "to_dict"):
                rd = result.to_dict()
            elif isinstance(result, dict):
                rd = result
            else:
                return metrics
            
            # 提取所有指标
            key_map = {
                "IC": "ic",
                "ICIR": "icir",
                "Rank IC": "rank_ic",
                "Rank ICIR": "rank_icir",
                "1day.excess_return_without_cost.annualized_return": "annualized_return",
                "1day.excess_return_without_cost.max_drawdown": "max_drawdown",
                "1day.excess_return_without_cost.information_ratio": "information_ratio",
                "1day.excess_return_with_cost.annualized_return": "annualized_return_with_cost",
                "1day.excess_return_with_cost.max_drawdown": "max_drawdown_with_cost",
                "1day.excess_return_with_cost.information_ratio": "information_ratio_with_cost",
            }
            
            for src_key, dst_key in key_map.items():
                val = rd.get(src_key)
                if val is not None:
                    try:
                        metrics[dst_key] = round(float(val), 6)
                    except (ValueError, TypeError):
                        pass
            
            # 保留完整的原始指标字典
            metrics["_raw"] = {str(k): (round(float(v), 6) if isinstance(v, (int, float)) else str(v)) for k, v in rd.items()}
        except Exception as e:
            logger.warning(f"提取回测指标失败: {e}")
        
        return metrics

    def extract_aligned_sota_factors(self, task_id: str) -> Dict[str, Any]:
        """
        V2核心方法：从最后SOTA因子实验的based_experiments链+自身提取与parquet完全对齐的SOTA因子
        
        算法：
        1. 找到hist中最后一个decision=True的因子实验
        2. 从该实验的based_experiments链中提取所有final_decision=True的因子
        3. 从该实验自身提取final_decision=True的因子
        4. 每个因子包含：源代码、进入SOTA时的回测数据、因子详情
        
        Returns:
            {
                'task_id': str,
                'success': bool,
                'aligned_factors': OrderedDict[str, dict],  # 与parquet对齐的因子
                'factor_names': List[str],                    # 因子名列表（与parquet列顺序一致）
                'factor_count': int,
                'last_sota_hist_index': int,                  # 最后SOTA因子实验在hist中的索引
                'based_experiments_count': int,
                'error': Optional[str]
            }
        """
        result = {
            "task_id": task_id,
            "success": False,
            "aligned_factors": OrderedDict(),
            "factor_names": [],
            "factor_count": 0,
            "last_sota_hist_index": None,
            "based_experiments_count": 0,
            "error": None,
        }

        try:
            # 1. 加载session
            session = self.load_session(task_id)
            if not session:
                result["error"] = "无法加载session文件"
                return result

            if not hasattr(session, "trace") or not hasattr(session.trace, "hist"):
                result["error"] = "session没有trace.hist属性"
                return result

            hist = session.trace.hist

            # 2. 找到最后一个decision=True的因子实验
            last_sota_idx = None
            for i in range(len(hist) - 1, -1, -1):
                exp, feedback = hist[i]
                if not self._is_factor_experiment(exp):
                    continue
                decision = getattr(feedback, "decision", None) if feedback else None
                if decision is True:
                    last_sota_idx = i
                    break

            if last_sota_idx is None:
                result["error"] = "未找到decision=True的因子实验"
                return result

            result["last_sota_hist_index"] = last_sota_idx
            last_exp, last_fb = hist[last_sota_idx]

            # 3. 从based_experiments链提取SOTA因子
            aligned_factors = OrderedDict()
            based_exps = getattr(last_exp, "based_experiments", []) or []
            result["based_experiments_count"] = len(based_exps)

            for be_idx, be in enumerate(based_exps):
                if not self._is_factor_experiment(be):
                    continue
                sub_count = len(be.sub_tasks) if hasattr(be, "sub_tasks") and be.sub_tasks else 0
                if sub_count == 0:
                    continue

                be_result = getattr(be, "result", None)

                for si in range(sub_count):
                    fname, fd = self._get_factor_info_from_exp(be, si)
                    if fd is not True or not fname:
                        continue
                    
                    # 规范化因子名（处理中文括号映射）
                    normalized_name = self._normalize_factor_name(fname)
                    
                    if normalized_name in aligned_factors:
                        continue

                    source_code = self._get_factor_source_code(be, si)
                    formulation, description, variables = self._get_factor_task_details(be, si)
                    metrics = self._extract_metrics_from_result(be_result)

                    aligned_factors[normalized_name] = {
                        "factor_name": normalized_name,
                        "original_name": fname,
                        "source_code": source_code,
                        "source_code_length": len(source_code) if source_code else 0,
                        "source_from": f"based_exp[{be_idx}]",
                        "factor_formulation": formulation,
                        "factor_description": description,
                        "variables": variables,
                        "metrics": metrics,
                    }

            # 4. 从当前实验自身提取SOTA因子
            sub_count = len(last_exp.sub_tasks) if hasattr(last_exp, "sub_tasks") and last_exp.sub_tasks else 0
            last_result = getattr(last_exp, "result", None)

            for si in range(sub_count):
                fname, fd = self._get_factor_info_from_exp(last_exp, si)
                if fd is not True or not fname:
                    continue
                
                normalized_name = self._normalize_factor_name(fname)
                
                if normalized_name in aligned_factors:
                    continue

                source_code = self._get_factor_source_code(last_exp, si)
                formulation, description, variables = self._get_factor_task_details(last_exp, si)
                metrics = self._extract_metrics_from_result(last_result)

                aligned_factors[normalized_name] = {
                    "factor_name": normalized_name,
                    "original_name": fname,
                    "source_code": source_code,
                    "source_code_length": len(source_code) if source_code else 0,
                    "source_from": f"hist[{last_sota_idx}]",
                    "factor_formulation": formulation,
                    "factor_description": description,
                    "variables": variables,
                    "metrics": metrics,
                }

            # 5. 构建结果
            result["success"] = True
            result["aligned_factors"] = aligned_factors
            result["factor_names"] = list(aligned_factors.keys())
            result["factor_count"] = len(aligned_factors)

            # 统计
            has_code = sum(1 for f in aligned_factors.values() if f["source_code"])
            has_metrics = sum(1 for f in aligned_factors.values() if f["metrics"])
            logger.info(
                f"[{task_id}] V2提取完成: {len(aligned_factors)}个对齐因子, "
                f"有源代码:{has_code}/{len(aligned_factors)}, "
                f"有回测数据:{has_metrics}/{len(aligned_factors)}"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"[{task_id}] V2提取SOTA因子失败: {e}", exc_info=True)

        return result

    def extract_task_sota_factors_v2(self, task_id: str) -> Dict:
        """
        V2版本的完整SOTA因子提取（兼容旧接口格式，内部使用aligned方法）
        
        返回格式与extract_task_sota_factors一致，但使用based_experiments链对齐逻辑
        """
        aligned_result = self.extract_aligned_sota_factors(task_id)
        
        if not aligned_result["success"]:
            return {
                "task_id": task_id,
                "success": False,
                "sota_factors": [],
                "factor_codes": {},
                "factor_details": {},
                "factor_count": 0,
                "error": aligned_result.get("error"),
            }
        
        aligned_factors = aligned_result["aligned_factors"]
        
        # 转换为旧格式
        sota_factors = list(aligned_factors.keys())
        factor_codes = {}
        factor_details = {}
        
        for fname, info in aligned_factors.items():
            if info["source_code"]:
                factor_codes[fname] = info["source_code"]
            
            metrics = info.get("metrics", {})
            factor_details[fname] = {
                "factor_name": fname,
                "factor_formulation": info.get("factor_formulation", ""),
                "factor_description": info.get("factor_description", ""),
                "source_loop_id": aligned_result.get("last_sota_hist_index", -1),
                "source_loop_tag": info.get("source_from", ""),
                "metrics": {
                    "ic": metrics.get("ic"),
                    "icir": metrics.get("icir"),
                    "rank_ic": metrics.get("rank_ic"),
                    "rank_icir": metrics.get("rank_icir"),
                    "annualized_return": metrics.get("annualized_return"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "information_ratio": metrics.get("information_ratio"),
                    "sharpe": metrics.get("sharpe"),
                    "annualized_return_with_cost": metrics.get("annualized_return_with_cost"),
                    "max_drawdown_with_cost": metrics.get("max_drawdown_with_cost"),
                    "information_ratio_with_cost": metrics.get("information_ratio_with_cost"),
                },
            }
        
        return {
            "task_id": task_id,
            "success": True,
            "sota_factors": sota_factors,
            "factor_codes": factor_codes,
            "factor_details": factor_details,
            "factor_count": len(sota_factors),
            "error": None,
        }

# Trigger reload
