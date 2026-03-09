"""
QE (QuantEvolver) 实验工作区设置API端点

提供以下接口：
1. POST /api/qe/experiments/{exp_id}/setup - 创建实验工作区并链接数据文件
2. GET /api/qe/experiments/{exp_id}/status - 获取实验工作区状态
3. GET /api/qe/factor-data-files - 获取因子数据文件列表

用途：
- AIstock 通过此API在RDAgent (WSL) 环境中创建实验工作区
- 数据文件链接在WSL环境执行，确保跨文件系统兼容性
- 因子执行时直接读取链接的数据文件，无需复制

关键：
- 所有文件操作在WSL/Linux环境执行
- 使用 os.link() 或 os.symlink() 链接数据文件
- 大文件使用 .pathmap 机制，避免复制
"""

import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qe", tags=["QE Experiment Setup"])


# ================================================================
# 配置
# ================================================================

def _get_qe_workspace_root() -> Path:
    """获取QE workspace根目录"""
    env_val = os.environ.get("QE_WORKSPACE_ROOT")
    if env_val:
        return Path(env_val)
    # 默认: RD-Agent-main/qe_workspace
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "qe_workspace"


def _get_factor_data_root() -> Path:
    """获取因子数据目录根路径（通过环境变量配置，不依赖RD-Agent内部模块）"""
    env_val = os.environ.get("QE_FACTOR_DATA_DIR")
    if env_val:
        return Path(env_val)
    # 回退：尝试从RD-Agent配置读取（兼容过渡期）
    try:
        from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
        return Path(FACTOR_COSTEER_SETTINGS.data_folder)
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / "git_ignore_folder" / "factor_implementation_source_data"


def _get_factor_data_debug_root() -> Path:
    """获取因子数据调试目录根路径（通过环境变量配置，不依赖RD-Agent内部模块）"""
    env_val = os.environ.get("QE_FACTOR_DATA_DEBUG_DIR")
    if env_val:
        return Path(env_val)
    # 回退：尝试从RD-Agent配置读取（兼容过渡期）
    try:
        from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
        return Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / "git_ignore_folder" / "factor_implementation_source_data_debug"


QE_WORKSPACE_ROOT = _get_qe_workspace_root()
FACTOR_DATA_ROOT = _get_factor_data_root()
FACTOR_DATA_DEBUG_ROOT = _get_factor_data_debug_root()


# ================================================================
# 数据模型
# ================================================================

class FactorFileSpec(BaseModel):
    """因子文件规格"""
    name: str = Field(..., description="因子名称")
    source_code: str = Field(..., description="因子原始源代码（完整脚本，不做任何修改）")


class ExperimentSetupRequest(BaseModel):
    """实验工作区设置请求"""
    factors: List[FactorFileSpec] = Field(default_factory=list, description="因子文件列表")
    conf_yaml: Optional[str] = Field(None, description="conf.yaml 配置内容（简化模式可省略）")
    strategy_code: Optional[str] = Field(None, description="策略源码 (custom_strategy.py)")
    model_code: Optional[str] = Field(None, description="模型源码 (model.py)")
    use_debug_data: bool = Field(False, description="是否使用调试数据集（小数据集）")
    link_data_files: bool = Field(True, description="是否链接数据文件")
    overwrite: bool = Field(False, description="是否覆盖已存在的实验目录")
    additional_files: Dict[str, str] = Field(default_factory=dict, description="额外文件 {filename: content}")


class LinkedFileInfo(BaseModel):
    """链接文件信息"""
    name: str
    source_path: str
    target_path: str
    size: int
    link_type: str  # "hard", "symlink", "pathmap", "copied"


class ExperimentSetupResponse(BaseModel):
    """实验工作区设置响应"""
    success: bool
    exp_id: str
    workspace_path: str
    wsl_path: str
    linked_files: List[LinkedFileInfo]
    factor_files: List[str]
    created_files: List[str]
    errors: List[str]
    message: str


class ExperimentStatusResponse(BaseModel):
    """实验工作区状态响应"""
    exp_id: str
    exists: bool
    workspace_path: str
    files: List[Dict[str, Any]]
    linked_data_files: List[Dict[str, Any]]
    factor_files: List[str]


class FactorDataFilesResponse(BaseModel):
    """因子数据文件列表响应"""
    data_root: str
    debug_root: str
    files: List[Dict[str, Any]]


# ================================================================
# 核心函数
# ================================================================

def _validate_exp_id(exp_id: str) -> None:
    """验证实验ID格式，防止路径遍历攻击"""
    if not exp_id:
        raise HTTPException(status_code=400, detail="实验ID不能为空")
    import re
    if not re.match(r'^[a-zA-Z0-9_\-]+$', exp_id):
        raise HTTPException(status_code=400, detail=f"实验ID格式无效: {exp_id}")


def _link_all_files_to_workspace(
    source_dir: Path,
    target_dir: Path,
    linked_files: List[LinkedFileInfo],
    errors: List[str],
) -> None:
    """
    将源目录中的所有文件链接到目标目录。
    
    与 RDAgent core/experiment.py link_all_files_in_folder_to_workspace 完全一致：
    1. 优先使用 os.link() 硬链接
    2. 失败则使用 os.symlink() 软链接（仅 Linux/macOS）
    3. 大文件（>50MB）创建 .pathmap 映射文件
    4. 小文件最后手段复制
    
    注意：此函数必须在 WSL/Linux 环境中执行，不能在 Windows 环境执行！
    """
    if not source_dir.exists():
        errors.append(f"源数据目录不存在: {source_dir}")
        return
    
    source_dir = source_dir.absolute()
    
    for item in source_dir.iterdir():
        # 跳过输出文件和锁文件
        if item.name in ("result.h5", "execution.lock"):
            continue
        
        if not item.is_file():
            continue
        
        src_path = item
        dst_path = target_dir / item.name
        
        # 移除已存在的目标文件
        if dst_path.exists() or dst_path.is_symlink():
            try:
                dst_path.unlink()
            except Exception as e:
                errors.append(f"无法删除已存在文件 {dst_path}: {e}")
                continue
        
        # 获取文件大小
        try:
            file_size = src_path.stat().st_size
        except OSError as e:
            errors.append(f"无法获取文件大小 {src_path}: {e}")
            continue
        
        # 尝试硬链接
        try:
            os.link(src_path, dst_path)
            linked_files.append(LinkedFileInfo(
                name=item.name,
                source_path=str(src_path),
                target_path=str(dst_path),
                size=file_size,
                link_type="hard",
            ))
            logger.info(f"[QE] 硬链接成功: {item.name} ({file_size / 1024 / 1024:.1f}MB)")
            continue
        except OSError as e:
            logger.debug(f"[QE] 硬链接失败 {item.name}: {e}")
        
        # 尝试软链接（仅 Linux/macOS）
        if platform.system() in ("Linux", "Darwin"):
            try:
                os.symlink(src_path, dst_path)
                linked_files.append(LinkedFileInfo(
                    name=item.name,
                    source_path=str(src_path),
                    target_path=str(dst_path),
                    size=file_size,
                    link_type="symlink",
                ))
                logger.info(f"[QE] 软链接成功: {item.name} ({file_size / 1024 / 1024:.1f}MB)")
                continue
            except OSError as e:
                logger.debug(f"[QE] 软链接失败 {item.name}: {e}")
        
        # 大文件（>=50MB）：创建 .pathmap 映射文件，绝不复制
        if file_size >= 50 * 1024 * 1024:
            pathmap_path = dst_path.with_suffix(dst_path.suffix + ".pathmap")
            try:
                pathmap_path.write_text(str(src_path))
                linked_files.append(LinkedFileInfo(
                    name=item.name,
                    source_path=str(src_path),
                    target_path=str(pathmap_path),
                    size=file_size,
                    link_type="pathmap",
                ))
                logger.info(f"[QE] 大文件pathmap: {item.name} ({file_size / 1024 / 1024:.1f}MB) -> {src_path}")
            except Exception as e:
                errors.append(f"无法创建pathmap {item.name}: {e}")
            continue
        
        # 小文件：最后手段复制
        try:
            shutil.copy2(src_path, dst_path)
            linked_files.append(LinkedFileInfo(
                name=item.name,
                source_path=str(src_path),
                target_path=str(dst_path),
                size=file_size,
                link_type="copied",
            ))
            logger.info(f"[QE] 复制成功: {item.name} ({file_size / 1024:.1f}KB)")
        except Exception as e:
            errors.append(f"无法复制文件 {item.name}: {e}")


# ================================================================
# API端点
# ================================================================

@router.post("/experiments/{exp_id}/setup", response_model=ExperimentSetupResponse)
async def setup_experiment_workspace(
    exp_id: str,
    request: ExperimentSetupRequest,
):
    """
    创建实验工作区并链接数据文件。
    
    此API在WSL/Linux环境中执行，确保：
    1. 实验目录在WSL文件系统中创建
    2. 数据文件使用 os.link()/os.symlink() 正确链接
    3. 因子源码保持原始格式，不做任何修改
    4. 大文件使用 .pathmap 机制，避免复制
    
    Args:
        exp_id: 实验ID (如 qe_exp_abc123)
        request: 实验设置请求
        
    Returns:
        设置结果，包含链接文件列表和错误信息
    """
    _validate_exp_id(exp_id)
    
    # 创建实验目录
    exp_dir = QE_WORKSPACE_ROOT / exp_id
    
    # 处理已存在的目录
    if exp_dir.exists():
        if request.overwrite:
            shutil.rmtree(exp_dir)
            logger.info(f"[QE] 覆盖已存在的实验目录: {exp_dir}")
        else:
            # 目录已存在，不覆盖，仅链接数据文件
            logger.info(f"[QE] 实验目录已存在，跳过创建: {exp_dir}")
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建因子目录
    factors_dir = exp_dir / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    
    linked_files: List[LinkedFileInfo] = []
    created_files: List[str] = []
    errors: List[str] = []
    
    # 1. 链接数据文件（从因子数据目录）
    if request.link_data_files:
        data_source = FACTOR_DATA_DEBUG_ROOT if request.use_debug_data else FACTOR_DATA_ROOT
        _link_all_files_to_workspace(data_source, exp_dir, linked_files, errors)
    
    # 2. 写入 conf.yaml（如果有）
    if request.conf_yaml:
        conf_yaml_content: str = request.conf_yaml  # 类型收窄
        conf_path = exp_dir / "conf.yaml"
        try:
            conf_path.write_text(conf_yaml_content, encoding="utf-8")
            created_files.append("conf.yaml")
            logger.info(f"[QE] 写入 conf.yaml: {conf_path}")
        except Exception as e:
            errors.append(f"写入 conf.yaml 失败: {e}")
    
    # 3. 写入因子源码文件（保持原始格式，不做任何修改）
    factor_names = []
    for factor_spec in request.factors:
        factor_name = factor_spec.name
        factor_code = factor_spec.source_code
        
        # 因子文件名：factors/{name}.py
        factor_path = factors_dir / f"{factor_name}.py"
        try:
            factor_path.write_text(factor_code, encoding="utf-8")
            created_files.append(f"factors/{factor_name}.py")
            factor_names.append(factor_name)
            logger.info(f"[QE] 写入因子源码: {factor_path} ({len(factor_code)} bytes)")
        except Exception as e:
            errors.append(f"写入因子 {factor_name} 失败: {e}")
    
    # 4. 写入策略源码（如果有）
    if request.strategy_code:
        strategy_path = exp_dir / "custom_strategy.py"
        try:
            strategy_path.write_text(request.strategy_code, encoding="utf-8")
            created_files.append("custom_strategy.py")
            logger.info(f"[QE] 写入策略源码: {strategy_path}")
        except Exception as e:
            errors.append(f"写入策略源码失败: {e}")
    
    # 5. 写入模型源码（如果有）
    if request.model_code:
        model_path = exp_dir / "model.py"
        try:
            model_path.write_text(request.model_code, encoding="utf-8")
            created_files.append("model.py")
            logger.info(f"[QE] 写入模型源码: {model_path}")
        except Exception as e:
            errors.append(f"写入模型源码失败: {e}")
    
    # 6. 写入额外文件
    for filename, content in request.additional_files.items():
        # 安全检查：防止路径遍历
        if ".." in filename or "/" in filename or "\\" in filename:
            errors.append(f"非法文件名: {filename}")
            continue
        
        extra_path = exp_dir / filename
        try:
            extra_path.write_text(content, encoding="utf-8")
            created_files.append(filename)
            logger.info(f"[QE] 写入额外文件: {extra_path}")
        except Exception as e:
            errors.append(f"写入额外文件 {filename} 失败: {e}")
    
    # 生成 WSL 路径
    wsl_path = str(exp_dir)
    # 如果是 Windows 路径，转换为 WSL 路径
    if platform.system() == "Linux" and wsl_path.startswith("/mnt/"):
        pass  # 已经是 WSL 路径
    elif platform.system() == "Linux":
        # 可能是原生 Linux 路径
        pass
    else:
        # Windows 环境，生成 WSL 路径格式
        wsl_path = wsl_path.replace("\\", "/").replace("F:", "/mnt/f").replace("f:", "/mnt/f")
    
    success = len(errors) == 0
    
    return ExperimentSetupResponse(
        success=success,
        exp_id=exp_id,
        workspace_path=str(exp_dir),
        wsl_path=wsl_path,
        linked_files=linked_files,
        factor_files=factor_names,
        created_files=created_files,
        errors=errors,
        message="实验工作区创建成功" if success else f"创建完成但有 {len(errors)} 个错误",
    )


@router.get("/experiments/{exp_id}/status", response_model=ExperimentStatusResponse)
async def get_experiment_status(exp_id: str):
    """
    获取实验工作区状态。
    
    Args:
        exp_id: 实验ID
        
    Returns:
        工作区状态信息
    """
    _validate_exp_id(exp_id)
    
    exp_dir = QE_WORKSPACE_ROOT / exp_id
    
    if not exp_dir.exists():
        return ExperimentStatusResponse(
            exp_id=exp_id,
            exists=False,
            workspace_path=str(exp_dir),
            files=[],
            linked_data_files=[],
            factor_files=[],
        )
    
    # 收集文件信息
    files = []
    linked_data_files = []
    factor_files = []
    
    for item in exp_dir.iterdir():
        if item.is_file():
            stat = item.stat()
            file_info = {
                "name": item.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
            files.append(file_info)
            
            # 检查是否是链接的数据文件
            if item.is_symlink():
                linked_data_files.append({
                    "name": item.name,
                    "link_target": str(item.resolve()),
                    "size": stat.st_size,
                })
            elif item.suffix == ".pathmap":
                # pathmap 文件
                target = item.read_text().strip()
                linked_data_files.append({
                    "name": item.stem,
                    "link_type": "pathmap",
                    "real_path": target,
                })
        
        elif item.is_dir() and item.name == "factors":
            # 收集因子文件
            for factor_file in item.iterdir():
                if factor_file.is_file() and factor_file.suffix == ".py":
                    factor_files.append(factor_file.stem)
    
    return ExperimentStatusResponse(
        exp_id=exp_id,
        exists=True,
        workspace_path=str(exp_dir),
        files=files,
        linked_data_files=linked_data_files,
        factor_files=factor_files,
    )


@router.get("/factor-data-files", response_model=FactorDataFilesResponse)
async def get_factor_data_files():
    """
    获取因子数据文件列表。
    
    Returns:
        数据文件列表，包含生产数据和调试数据
    """
    def _collect_files(root: Path) -> List[Dict[str, Any]]:
        if not root.exists():
            return []
        
        files = []
        for item in root.iterdir():
            if item.is_file():
                stat = item.stat()
                files.append({
                    "name": item.name,
                    "size": stat.st_size,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": stat.st_mtime,
                })
        return sorted(files, key=lambda x: x["name"])
    
    return FactorDataFilesResponse(
        data_root=str(FACTOR_DATA_ROOT),
        debug_root=str(FACTOR_DATA_DEBUG_ROOT),
        files=_collect_files(FACTOR_DATA_ROOT),
    )


@router.get("/tasks/{task_id}/factors/{factor_name}/source")
async def get_factor_source_code(
    task_id: str,
    factor_name: str,
):
    """
    获取指定因子的原始完整源码。
    
    通过调用 results_api_server 的 complete_assets API 获取因子源码。
    
    Args:
        task_id: 任务ID
        factor_name: 因子名称（如 elg_flow_acceleration_pb_trend）
        
    Returns:
        因子源码内容
    """
    import requests
    
    # 获取 API 基础 URL
    api_base = os.environ.get("RDAGENT_API_BASE", "http://127.0.0.1:9000")
    
    try:
        # 调用 complete_assets API 获取完整资产
        resp = requests.get(
            f"{api_base}/tasks/{task_id}/complete_assets",
            timeout=120,
        )
        
        if resp.status_code == 404:
            return {
                "success": False,
                "task_id": task_id,
                "factor_name": factor_name,
                "error": f"Task not found: {task_id}",
            }
        
        resp.raise_for_status()
        assets_data = resp.json()
        
        # 从 complete_assets 响应中提取因子源码
        factors = assets_data.get("factors", [])
        
        for factor in factors:
            if factor.get("name") == factor_name or factor.get("factor_name") == factor_name:
                source_code = factor.get("source_code") or factor.get("code")
                if source_code:
                    return {
                        "success": True,
                        "task_id": task_id,
                        "factor_name": factor_name,
                        "source_code": source_code,
                        "source": "complete_assets",
                    }
        
        # 因子未找到
        return {
            "success": False,
            "task_id": task_id,
            "factor_name": factor_name,
            "error": f"Factor not found: {factor_name}",
            "available_factors": [f.get("name") or f.get("factor_name") for f in factors],
        }
        
    except requests.exceptions.ConnectionError:
        logger.warning(f"[QE] API 连接失败: {api_base}")
        return {
            "success": False,
            "task_id": task_id,
            "factor_name": factor_name,
            "error": "API connection failed",
        }
    except Exception as e:
        logger.error(f"[QE] 获取因子源码失败: task={task_id}, factor={factor_name}, {e}")
        return {
            "success": False,
            "task_id": task_id,
            "factor_name": factor_name,
            "error": str(e),
        }


@router.get("/tasks/{task_id}/factors")
async def list_task_factors(task_id: str):
    """
    列出task中所有可用的因子名称。
    
    通过调用 results_api_server 的现有 API 获取 session anchor 信息。
    
    Args:
        task_id: 任务ID
        
    Returns:
        因子名称列表
    """
    import requests
    
    # 获取 API 基础 URL
    api_base = os.environ.get("RDAGENT_API_BASE", "http://127.0.0.1:9000")
    
    try:
        # 调用 session_anchor API 获取 SOTA 因子信息
        resp = requests.get(
            f"{api_base}/tasks/{task_id}/session_anchor",
            timeout=60,
        )
        
        if resp.status_code == 404:
            return {
                "success": False,
                "task_id": task_id,
                "factors": [],
                "error": f"Task not found: {task_id}",
            }
        
        resp.raise_for_status()
        anchor_data = resp.json()
        
        # 从 session_anchor 响应中提取因子列表
        # 格式参考 results_api_server.py 的 task_session_anchor 端点
        factor_names = anchor_data.get("factor_names", [])
        
        if not factor_names:
            # 尝试从 sota_factor_names 获取
            factor_names = anchor_data.get("sota_factor_names", [])
        
        return {
            "success": True,
            "task_id": task_id,
            "factors": factor_names,
            "count": len(factor_names),
            "source": "session_anchor",
        }
        
    except requests.exceptions.ConnectionError:
        logger.warning(f"[QE] API 连接失败: {api_base}")
        return {
            "success": False,
            "task_id": task_id,
            "factors": [],
            "error": "API connection failed",
        }
    except Exception as e:
        logger.error(f"[QE] 列出任务因子失败: {task_id}, {e}")
        return {
            "success": False,
            "task_id": task_id,
            "factors": [],
            "error": str(e),
        }


@router.get("/experiments/{exp_id}/linked-files")
async def get_linked_files_status(exp_id: str):
    """
    获取实验工作区中数据文件的链接状态。
    
    用于验证数据文件是否正确链接，是否可以正常读取。
    
    Args:
        exp_id: 实验ID
        
    Returns:
        链接文件状态列表
    """
    _validate_exp_id(exp_id)
    
    exp_dir = QE_WORKSPACE_ROOT / exp_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"实验目录不存在: {exp_id}")
    
    # 需要检查的数据文件列表
    required_data_files = [
        "daily_pv.h5",
        "static_factors.parquet",
        "daily_basic.h5",
        "moneyflow.h5",
        "bak_basic.h5",
        "cyq_perf.h5",
        "README.md",
    ]
    
    file_status = []
    for fname in required_data_files:
        fpath = exp_dir / fname
        pathmap_path = exp_dir / (fname + ".pathmap")
        
        status = {
            "name": fname,
            "exists": False,
            "link_type": None,
            "size": 0,
            "readable": False,
            "real_path": None,
        }
        
        if fpath.exists() and fpath.is_file():
            status["exists"] = True
            status["size"] = fpath.stat().st_size
            
            if fpath.is_symlink():
                status["link_type"] = "symlink"
                status["real_path"] = str(fpath.resolve())
            else:
                # 检查是否是硬链接（通过比较 inode）
                try:
                    # 硬链接的 nlink > 1
                    nlink = fpath.stat().st_nlink
                    if nlink > 1:
                        status["link_type"] = "hard"
                    else:
                        status["link_type"] = "copied"
                except:
                    status["link_type"] = "unknown"
            
            # 检查是否可读
            try:
                with open(fpath, "rb") as f:
                    f.read(1024)
                status["readable"] = True
            except Exception as e:
                status["readable"] = False
                status["error"] = str(e)
        
        elif pathmap_path.exists():
            status["exists"] = True
            status["link_type"] = "pathmap"
            status["real_path"] = pathmap_path.read_text().strip()
            # 检查真实路径是否可读
            try:
                real_path = Path(status["real_path"])
                if real_path.exists():
                    status["size"] = real_path.stat().st_size
                    with open(real_path, "rb") as f:
                        f.read(1024)
                    status["readable"] = True
            except Exception as e:
                status["readable"] = False
                status["error"] = str(e)
        
        file_status.append(status)
    
    return {
        "exp_id": exp_id,
        "workspace_path": str(exp_dir),
        "files": file_status,
        "all_ready": all(f["exists"] and f["readable"] for f in file_status if f["name"] in ["daily_pv.h5", "static_factors.parquet"]),
    }
