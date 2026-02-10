"""完整TASK资产提取器

复用验证脚本的成功逻辑，从Session文件中提取：
- SOTA因子列表
- 因子代码
- 模型权重（双重定位：file_dict + mlruns）
- 特征序列（从combined_factors_df.parquet）
"""
import os
import pickle
from pathlib import Path, WindowsPath, PosixPath
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

try:
    from .path_utils import normalize_path
except ImportError:
    from path_utils import normalize_path


def _print_if_not_silent(*args, **kwargs):
    """只在非静默模式下打印"""
    if not os.environ.get('TASK_EXTRACTOR_SILENT'):
        print(*args, **kwargs)


class PathCompatUnpickler(pickle.Unpickler):
    """跨平台路径兼容的Unpickler
    
    处理pathlib在不同操作系统间序列化的兼容性问题
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


def load_pickle_compat(file_path: Path) -> Any:
    """使用兼容的Unpickler加载pickle文件"""
    with file_path.open("rb") as f:
        return PathCompatUnpickler(f).load()


class TaskAssetsExtractor:
    """TASK资产提取器
    
    基于验证脚本的成功模式，从Session文件直接提取所有资产
    """
    
    # Alpha基线因子（固定20个）
    ALPHA_BASELINE_FACTORS = [
        'RESI5', 'WVMA5', 'RSQR5', 'KLEN', 'RSQR10',
        'CORR5', 'CORD5', 'CORR10', 'ROC60', 'RESI10',
        'VSTD5', 'RSQR60', 'CORR60', 'WVMA60', 'STD5',
        'RSQR20', 'CORD60', 'CORD10', 'CORR20', 'KLOW'
    ]
    
    def __init__(self, log_root: Path, workspace_root: Path):
        self.log_root = Path(log_root)
        self.workspace_root = Path(workspace_root)
    
    def _load_session_numeric_sort(self, task_id: str) -> Tuple[Any, int, str]:
        """加载Session，选择hist最长的LOOP
        
        关键修复: 不再按loop_id数值排序，而是遍历所有loop目录，
        加载每个session，选择hist长度最长的那个。
        
        这是因为SOTA因子进入的顺序不一定与LOOP id一致，
        hist最长的LOOP才是包含全量SOTA因子的LOOP。
        
        Returns:
            (session对象, loop_id, session文件名)
        """
        task_folder = self.log_root / task_id
        session_folder = task_folder / "__session__"
        
        if not session_folder.exists():
            raise FileNotFoundError(f"Session文件夹不存在: {session_folder}")
        
        # 获取所有loop文件夹，使用数值排序
        loop_dirs = []
        for d in session_folder.iterdir():
            if d.is_dir() and d.name.isdigit():
                loop_dirs.append((int(d.name), d))
        
        if not loop_dirs:
            raise ValueError(f"未找到任何loop文件夹: {session_folder}")
        
        # 遍历所有loop目录，加载每个session，记录hist长度
        session_candidates = []
        
        for loop_id, loop_dir in loop_dirs:
            # 尝试加载该loop目录下的所有session文件
            session_files = []
            
            # 优先尝试 3_feedback
            feedback_file = loop_dir / "3_feedback"
            if feedback_file.exists():
                session_files.append(("3_feedback", feedback_file))
            
            # 尝试 1_coding
            coding_file = loop_dir / "1_coding"
            if coding_file.exists():
                session_files.append(("1_coding", coding_file))
            
            # 尝试其他文件
            for file in loop_dir.iterdir():
                if file.is_file() and file.name not in ["3_feedback", "1_coding"]:
                    session_files.append((file.name, file))
            
            # 加载每个session文件，记录hist长度
            for file_name, file_path in session_files:
                try:
                    session = load_pickle_compat(file_path)
                    hist_len = 0
                    if hasattr(session, 'trace') and hasattr(session.trace, 'hist'):
                        hist_len = len(session.trace.hist)
                    
                    session_candidates.append({
                        'loop_id': loop_id,
                        'loop_dir': loop_dir,
                        'file_name': file_name,
                        'file_path': file_path,
                        'session': session,
                        'hist_len': hist_len
                    })
                except Exception as e:
                    print(f"    警告: loop_id={loop_id} {file_name}加载失败: {e}")
                    continue
        
        if not session_candidates:
            raise RuntimeError(f"所有loop文件夹的session文件都加载失败")
        
        # 按hist_len降序排序，选择hist最长的
        session_candidates.sort(key=lambda x: x['hist_len'], reverse=True)
        best = session_candidates[0]
        
        print(f"    选择session: loop_id={best['loop_id']}, hist_len={best['hist_len']}, file={best['file_name']}")
        print(f"    所有候选: {[(c['loop_id'], c['hist_len']) for c in session_candidates[:5]]}")
        
        return best['session'], best['loop_id'], best['file_name']
    
    def _find_last_sota_factor_loop(self, session: Any) -> int:
        """找到最后一个decision=True的Factor LOOP的ID
        
        Returns:
            loop_id (在trace.hist中的索引)
        """
        if not hasattr(session, 'trace') or not hasattr(session.trace, 'hist'):
            raise ValueError("Session对象缺少trace.hist属性")
        
        hist = session.trace.hist
        
        # 从后往前找
        for loop_id in range(len(hist) - 1, -1, -1):
            try:
                loop_exp, feedback = hist[loop_id]
                exp_type = type(loop_exp).__name__
                
                # 必须是Factor类型
                if 'Factor' not in exp_type:
                    continue
                
                # 必须decision=True
                if not (hasattr(feedback, 'decision') and feedback.decision):
                    continue
                
                # 必须有sub_tasks
                if hasattr(loop_exp, 'sub_tasks') and loop_exp.sub_tasks:
                    return loop_id
            except Exception:
                continue
        
        raise ValueError("未找到任何decision=True的Factor LOOP")
    
    def _extract_sota_factors(self, session: Any) -> List[str]:
        """提取所有decision=True的LOOP中final_decision=True的SOTA因子(去重)

        核心逻辑:
        - decision=True: 回测成功,实验进入SOTA(factor_proposal.py:112-114)
        - final_decision=True: 因子代码执行成功,可生成数据(utils.py:39-40)
        - 只有decision=True且final_decision=True的因子才真正进入SOTA并参与训练
        - final_decision=True但decision=False的因子不进入SOTA(回测失败)
        - decision=True但final_decision=False的因子不进入SOTA(代码执行失败)

        Returns:
            因子名称列表(去重但保持顺序)
        """
        if not hasattr(session, 'trace') or not hasattr(session.trace, 'hist'):
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
                        # 无feedback_list,回退到所有因子(兼容旧数据)
                        for fname in hyp_factors:
                            if fname:
                                factors.append(fname)
                else:
                    # 无prop_dev_feedback,回退到所有因子(兼容旧数据)
                    for fname in hyp_factors:
                        if fname:
                            factors.append(fname)
            except Exception as e:
                print(f"    警告: loop_id={loop_id}提取因子失败: {e}")
                continue

        # 去重但保持顺序
        seen = set()
        unique_factors = []
        for f in factors:
            if f not in seen:
                seen.add(f)
                unique_factors.append(f)

        return unique_factors
    
    def _get_session_info(self, session: Any, loop_id: int, session_dir_id: str) -> Dict[str, Any]:
        """获取Session元信息"""
        hist_len = 0
        if hasattr(session, 'trace') and hasattr(session.trace, 'hist'):
            hist_len = len(session.trace.hist)
        
        return {
            "source_session_dir_id": session_dir_id,
            "hist_len": hist_len,
            "last_sota_factor_loop_id": loop_id
        }
    
    def _extract_factor_codes(self, session: Any, last_loop_id: int = None) -> List[Dict[str, Any]]:
        """从所有decision=True的Factor LOOP中提取因子代码
        
        Args:
            session: Session对象
            last_loop_id: 最后一个SOTA因子LOOP的ID（可选，用于验证）
        
        Returns:
            [
                {
                    "factor_name": "factor_1",
                    "code": "def calculate_factor_1():\\n    ...",
                    "loop_id": 3,
                    "workspace_index": 0
                },
                ...
            ]
        """
        if not hasattr(session, 'trace') or not hasattr(session.trace, 'hist'):
            return []
        
        hist = session.trace.hist
        factor_codes = []
        seen_factors = set()
        
        # 遍历所有LOOP，提取所有decision=True的Factor的代码
        for loop_id in range(len(hist)):
            try:
                loop_exp, feedback = hist[loop_id]
                exp_type = type(loop_exp).__name__
                
                # 必须是Factor类型且decision=True
                if 'Factor' not in exp_type:
                    continue
                if not (hasattr(feedback, 'decision') and feedback.decision):
                    continue
                
                # 必须有sub_workspace_list
                if not hasattr(loop_exp, 'sub_workspace_list') or not loop_exp.sub_workspace_list:
                    continue
                
                # 获取每个因子的final_decision状态
                final_decisions = []
                if hasattr(loop_exp, 'prop_dev_feedback') and loop_exp.prop_dev_feedback is not None:
                    pdf = loop_exp.prop_dev_feedback
                    feedback_list = getattr(pdf, 'feedback_list', None)
                    if feedback_list is None and hasattr(pdf, '__iter__'):
                        feedback_list = list(pdf)
                    if feedback_list:
                        for fb_item in feedback_list:
                            if fb_item is not None and hasattr(fb_item, 'final_decision'):
                                final_decisions.append(bool(fb_item.final_decision))
                            else:
                                final_decisions.append(None)

                # 遍历sub_workspace，只提取final_decision=True的因子代码
                for idx, workspace in enumerate(loop_exp.sub_workspace_list):
                    if workspace is None:
                        continue

                    # 只提取final_decision=True的因子代码
                    if idx < len(final_decisions) and final_decisions[idx] is not None:
                        if not final_decisions[idx]:
                            continue

                    if not hasattr(workspace, 'file_dict'):
                        continue
                    
                    file_dict = workspace.file_dict
                    if not isinstance(file_dict, dict):
                        continue
                    
                    # 查找factor.py
                    factor_key = None
                    for key in file_dict.keys():
                        if str(key).lower() == 'factor.py':
                            factor_key = key
                            break
                    
                    if not factor_key:
                        continue
                    
                    # 获取代码内容
                    code_content = file_dict[factor_key]
                    if isinstance(code_content, bytes):
                        code_content = code_content.decode('utf-8')
                    elif not isinstance(code_content, str):
                        continue
                    
                    # 尝试从sub_tasks获取因子名
                    factor_name = f"factor_{idx}"
                    if hasattr(loop_exp, 'sub_tasks') and loop_exp.sub_tasks:
                        if idx < len(loop_exp.sub_tasks):
                            task = loop_exp.sub_tasks[idx]
                            if hasattr(task, 'factor_name') and task.factor_name:
                                factor_name = task.factor_name
                    
                    # 去重
                    if factor_name in seen_factors:
                        continue
                    seen_factors.add(factor_name)
                    
                    factor_codes.append({
                        "factor_name": factor_name,
                        "code": code_content,
                        "loop_id": loop_id,
                        "workspace_index": idx
                    })
            except Exception:
                continue
        
        return factor_codes
    
    def _extract_model_weight_from_file_dict(
        self, 
        loop_exp: Any
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """方法1: 从sub_workspace_list.file_dict提取模型权重"""
        
        if not hasattr(loop_exp, 'sub_workspace_list') or not loop_exp.sub_workspace_list:
            return None, {"error": "No sub_workspace_list"}
        
        # 遍历所有sub_workspace
        for idx, workspace in enumerate(loop_exp.sub_workspace_list):
            if workspace is None:
                continue
            
            if not hasattr(workspace, 'file_dict'):
                continue
            
            file_dict = workspace.file_dict
            if not isinstance(file_dict, dict):
                continue
            
            # 查找模型权重文件
            for key in file_dict.keys():
                key_lower = str(key).lower()
                if key_lower in ['params.pkl', 'model.pkl']:
                    content = file_dict[key]
                    if isinstance(content, (bytes, bytearray)):
                        return bytes(content), {
                            "source": "file_dict",
                            "file_name": str(key),
                            "workspace_index": idx,
                            "size_bytes": len(content)
                        }
        
        return None, {"error": "Model weight not found in file_dict"}
    
    def _extract_model_weight_from_mlruns(
        self,
        loop_exp: Any
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """方法2: 从experiment_workspace/mlruns提取模型权重"""
        
        # 使用experiment_workspace而不是sub_workspace_list
        if not hasattr(loop_exp, 'experiment_workspace'):
            return None, {"error": "No experiment_workspace"}
        
        if not hasattr(loop_exp.experiment_workspace, 'workspace_path'):
            return None, {"error": "No workspace_path in experiment_workspace"}
        
        ws_path_raw = loop_exp.experiment_workspace.workspace_path
        if not ws_path_raw:
            return None, {"error": "workspace_path is empty"}
        
        # 路径转换
        ws_path = normalize_path(ws_path_raw)
        
        if not ws_path.exists():
            return None, {"error": f"Workspace not exists: {ws_path}"}
        
        # 查找mlruns目录
        mlruns_dir = ws_path / "mlruns"
        if not mlruns_dir.exists():
            return None, {"error": f"mlruns not exists: {mlruns_dir}"}
        
        # 递归查找params.pkl（优先）
        candidates = []
        for pkl_file in mlruns_dir.rglob("params.pkl"):
            if pkl_file.is_file():
                candidates.append(pkl_file)
        
        # 如果没有params.pkl，查找model.pkl
        if not candidates:
            for pkl_file in mlruns_dir.rglob("model.pkl"):
                if pkl_file.is_file():
                    candidates.append(pkl_file)
        
        if not candidates:
            return None, {"error": "No params.pkl or model.pkl in mlruns"}
        
        # 按修改时间排序，取最新的
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        try:
            content = candidates[0].read_bytes()
            return content, {
                "source": "mlruns",
                "file_name": candidates[0].name,
                "size_bytes": len(content),
                "file_path": str(candidates[0]),
                "workspace_path": str(ws_path)
            }
        except Exception as e:
            return None, {"error": f"Failed to read {candidates[0]}: {e}"}
    
    def _extract_model_weight(
        self,
        session: Any,
        last_loop_id: int
    ) -> Dict[str, Any]:
        """双重定位提取模型权重
        
        优先级: file_dict > mlruns
        """
        if not hasattr(session, 'trace') or not hasattr(session.trace, 'hist'):
            raise ValueError("Session对象缺少trace.hist属性")
        
        if last_loop_id >= len(session.trace.hist):
            raise ValueError(f"loop_id {last_loop_id} 超出范围")
        
        loop_exp, _ = session.trace.hist[last_loop_id]
        
        # 方法1: file_dict
        content, info = self._extract_model_weight_from_file_dict(loop_exp)
        if content is not None:
            result = {
                "found": True,
                "content": content,  # 保留bytes用于后续使用
                **info
            }
            return result
        
        # 方法2: mlruns
        content, info = self._extract_model_weight_from_mlruns(loop_exp)
        if content is not None:
            result = {
                "found": True,
                "content": content,  # 保留bytes用于后续使用
                **info
            }
            return result
        
        raise RuntimeError("模型权重文件未找到（file_dict和mlruns都失败）")
    
    def _extract_feature_sequence(
        self,
        session: Any,
        last_loop_id: int
    ) -> Dict[str, Any]:
        """从combined_factors_df.parquet提取特征序列
        
        Returns:
            {
                "alpha_baseline": ["RESI5", "WVMA5", ...],  # 20个
                "dynamic_factors": ["factor_1", "factor_2", ...],
                "total_count": 25,
                "source": "combined_factors_df.parquet"
            }
        """
        if not hasattr(session, 'trace') or not hasattr(session.trace, 'hist'):
            raise ValueError("Session对象缺少trace.hist属性")
        
        if last_loop_id >= len(session.trace.hist):
            raise ValueError(f"loop_id {last_loop_id} 超出范围")
        
        loop_exp, _ = session.trace.hist[last_loop_id]
        
        # 使用experiment_workspace.workspace_path
        if not hasattr(loop_exp, 'experiment_workspace'):
            raise ValueError(f"loop_id {last_loop_id} 没有experiment_workspace")
        
        if not hasattr(loop_exp.experiment_workspace, 'workspace_path'):
            raise ValueError("experiment_workspace没有workspace_path")
        
        ws_path_raw = loop_exp.experiment_workspace.workspace_path
        if not ws_path_raw:
            raise ValueError("workspace_path为空")
        
        # 路径转换
        ws_path = normalize_path(ws_path_raw)
        
        if not ws_path.exists():
            raise ValueError(f"Workspace不存在: {ws_path}")
        
        # 查找parquet文件
        parquet_file = ws_path / "combined_factors_df.parquet"
        if not parquet_file.exists():
            raise RuntimeError(f"未找到combined_factors_df.parquet: {parquet_file}")
        
        try:
            # 使用pyarrow读取parquet，完全复用验证脚本逻辑
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_file)
            columns = table.column_names
            
            # 过滤出特征列（排除索引列）
            feature_columns = [col for col in columns if col not in ['datetime', 'instrument']]
            
            # 验证脚本的逻辑：feature_sequence = ALPHA_BASELINE_FACTORS + dynamic_factors
            # 所以我们返回完整的特征序列，前20个是Alpha基线因子，后面是动态因子
            return {
                "alpha_baseline": self.ALPHA_BASELINE_FACTORS,
                "dynamic_factors": feature_columns,
                "total_count": len(self.ALPHA_BASELINE_FACTORS) + len(feature_columns),
                "source": "combined_factors_df.parquet",
                "parquet_file": str(parquet_file),
                "workspace_path": str(ws_path)
            }
            
        except Exception as e:
            raise RuntimeError(f"读取parquet文件失败: {e}")
    
    def _validate_consistency(
        self,
        sota_factors: List[str],
        factor_codes: List[Dict[str, Any]],
        model_weight: Dict[str, Any],
        feature_sequence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """严格验证数据一致性"""
        
        validation = {
            "all_ok": True,
            "errors": [],
            "warnings": []
        }
        
        # 验证1: 因子代码数量
        factor_code_count = len(factor_codes)
        sota_factor_count = len(sota_factors)
        
        validation["factor_code_count"] = factor_code_count
        validation["sota_factor_count"] = sota_factor_count
        
        # 允许因子代码数量 >= SOTA因子数量（因为可能有重复版本）
        if factor_code_count < sota_factor_count:
            validation["all_ok"] = False
            validation["errors"].append(
                f"因子代码数量({factor_code_count}) < SOTA因子数量({sota_factor_count})"
            )
        elif factor_code_count > sota_factor_count:
            validation["warnings"].append(
                f"因子代码数量({factor_code_count}) > SOTA因子数量({sota_factor_count})，"
                "这是正常的（不同LOOP可能生成相同因子的不同版本）"
            )
        
        validation["factor_alignment_ok"] = factor_code_count >= sota_factor_count
        
        # 验证2: 模型权重存在
        validation["model_weight_found"] = model_weight.get("found", False)
        validation["model_weight_size"] = model_weight.get("size_bytes", 0)
        
        if not validation["model_weight_found"]:
            validation["all_ok"] = False
            validation["errors"].append("模型权重文件未找到")
        
        # 验证3: 特征序列
        alpha_count = len(feature_sequence.get("alpha_baseline", []))
        dynamic_count = len(feature_sequence.get("dynamic_factors", []))
        total_feature_count = feature_sequence.get("total_count", 0)
        
        validation["alpha_baseline_count"] = alpha_count
        validation["dynamic_factors_count"] = dynamic_count
        validation["total_feature_count"] = total_feature_count
        
        # Alpha基线因子应该是20个
        if alpha_count != 20:
            validation["warnings"].append(
                f"Alpha基线因子数量({alpha_count}) != 20"
            )
        
        # 总特征数应该 = Alpha基线 + 动态因子
        if total_feature_count != alpha_count + dynamic_count:
            validation["all_ok"] = False
            validation["errors"].append(
                f"总特征数({total_feature_count}) != Alpha基线({alpha_count}) + 动态因子({dynamic_count})"
            )
        
        validation["feature_sequence_ok"] = (total_feature_count == alpha_count + dynamic_count)
        
        # 验证4: 因子名称匹配
        # 从因子代码中提取因子名
        factor_code_set = set(fc["factor_name"] for fc in factor_codes)
        
        # 从动态因子中提取因子名（处理tuple字符串格式）
        # parquet中的列名格式：字符串 "('feature', 'factor_name')"
        dynamic_factor_names = set()
        for df in feature_sequence["dynamic_factors"]:
            if isinstance(df, str) and df.startswith("('feature',"):
                # 从 "('feature', 'factor_name')" 提取 'factor_name'
                try:
                    import ast
                    parsed = ast.literal_eval(df)
                    if isinstance(parsed, tuple) and len(parsed) >= 2:
                        dynamic_factor_names.add(parsed[1])
                except:
                    dynamic_factor_names.add(df)
            else:
                dynamic_factor_names.add(df)
        
        missing_in_code = dynamic_factor_names - factor_code_set
        extra_in_code = factor_code_set - dynamic_factor_names
        
        if missing_in_code:
            validation["warnings"].append(
                f"特征序列中的动态因子在因子代码中缺失: {list(missing_in_code)}"
            )
        
        if extra_in_code:
            validation["warnings"].append(
                f"因子代码中有额外的因子（未在特征序列中）: {list(extra_in_code)}"
            )
        
        validation["factor_name_alignment_ok"] = len(missing_in_code) == 0
        
        return validation
    
    def extract_complete_assets(self, task_id: str) -> Dict[str, Any]:
        """一次性提取完整TASK资产
        
        这是核心入口函数，整合所有提取逻辑
        """
        try:
            _print_if_not_silent(f"\n开始提取TASK资产: {task_id}")
            
            # 1. 加载Session（数值排序）
            _print_if_not_silent("  [1/7] 加载Session...")
            session, loop_id, session_file = self._load_session_numeric_sort(task_id)
            _print_if_not_silent(f"    ✓ Session加载成功: loop_id={loop_id}, file={session_file}")
            
            # 2. 找到最后一个SOTA因子LOOP
            _print_if_not_silent("  [2/7] 查找最后SOTA因子LOOP...")
            last_loop_id = self._find_last_sota_factor_loop(session)
            _print_if_not_silent(f"    ✓ 最后SOTA因子LOOP: {last_loop_id}")
            
            # 3. 提取SOTA因子列表
            _print_if_not_silent("  [3/7] 提取SOTA因子列表...")
            sota_factors = self._extract_sota_factors(session)
            _print_if_not_silent(f"    ✓ SOTA因子数量: {len(sota_factors)}")
            
            # 4. 提取所有因子代码
            _print_if_not_silent("  [4/7] 提取因子代码...")
            factor_codes = self._extract_factor_codes(session, last_loop_id)
            _print_if_not_silent(f"    ✓ 因子代码数量: {len(factor_codes)}")
            
            # 5. 提取模型权重（双重定位）
            _print_if_not_silent("  [5/7] 提取模型权重...")
            model_weight = self._extract_model_weight(session, last_loop_id)
            _print_if_not_silent(f"    ✓ 模型权重: {model_weight['source']}, {model_weight['size_bytes']} bytes")
            
            # 6. 提取特征序列
            _print_if_not_silent("  [6/7] 提取特征序列...")
            feature_sequence = self._extract_feature_sequence(session, last_loop_id)
            _print_if_not_silent(f"    ✓ 特征序列: Alpha={len(feature_sequence['alpha_baseline'])}, "
                  f"动态={len(feature_sequence['dynamic_factors'])}, "
                  f"总计={feature_sequence['total_count']}")
            
            # 7. 获取Session信息
            session_info = self._get_session_info(session, last_loop_id, str(loop_id))
            
            # 8. 严格验证
            _print_if_not_silent("  [7/7] 验证数据一致性...")
            validation = self._validate_consistency(
                sota_factors, factor_codes, model_weight, feature_sequence
            )
            
            if validation["all_ok"]:
                _print_if_not_silent("    ✓ 验证通过")
            else:
                _print_if_not_silent(f"    ✗ 验证失败: {validation['errors']}")
            
            if validation.get("warnings"):
                for warning in validation["warnings"]:
                    _print_if_not_silent(f"    ⚠ {warning}")
            
            # 移除bytes数据，避免JSON序列化问题
            model_weight_info = {k: v for k, v in model_weight.items() if k != 'content'}
            
            return {
                "ok": validation["all_ok"],
                "task_id": task_id,
                "session_info": session_info,
                "sota_factors": {
                    "count": len(sota_factors),
                    "factors": sota_factors
                },
                "factor_codes": factor_codes,
                "model_weight": model_weight_info,
                "feature_sequence": feature_sequence,
                "validation": validation
            }
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            print(f"    ✗ 提取失败: {error_msg}")
            return {
                "ok": False,
                "task_id": task_id,
                "error": error_msg,
                "traceback": traceback_str
            }
