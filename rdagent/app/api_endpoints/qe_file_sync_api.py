"""
QE (QuantEvolver) 文件同步API端点

提供以下接口：
1. POST /api/qe/experiments/{exp_id}/files - 上传单个文件到实验目录
2. POST /api/qe/experiments/{exp_id}/files/batch - 批量上传多个文件到实验目录
3. GET  /api/qe/experiments/{exp_id}/files - 列出实验目录中的文件
4. GET  /api/qe/experiments - 列出所有QE实验目录
5. DELETE /api/qe/experiments/{exp_id} - 删除实验目录

用途：
- AIstock (Windows) 通过HTTP API将QE生成的实验文件同步到RDAgent (WSL) 侧
- 解决未来两个环境独立运行时的文件同步问题
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qe", tags=["QE File Sync"])

# QE实验目录根路径（RDAgent侧）
def _get_qe_workspace_root() -> Path:
    """获取QE workspace根目录"""
    env_val = os.environ.get("QE_WORKSPACE_ROOT")
    if env_val:
        return Path(env_val)
    # 默认: RD-Agent-main/qe_workspace
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "qe_workspace"

QE_WORKSPACE_ROOT = _get_qe_workspace_root()

# 允许上传的文件名白名单（安全限制）
ALLOWED_FILENAMES = {
    "conf.yaml",
    "factor.py",
    "read_exp_res.py",
    "custom_strategy.py",
    "model.py",
    "requirements.txt",
}

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {".yaml", ".yml", ".py", ".txt", ".json", ".csv", ".b64"}

# 单文件最大大小 (10MB — parquet 等二进制文件 base64 编码后较大)
MAX_FILE_SIZE = 10 * 1024 * 1024


def _validate_exp_id(exp_id: str) -> None:
    """验证实验ID格式，防止路径遍历攻击"""
    if not exp_id:
        raise HTTPException(status_code=400, detail="实验ID不能为空")
    # 只允许字母数字、下划线、连字符
    import re
    if not re.match(r'^[a-zA-Z0-9_\-]+$', exp_id):
        raise HTTPException(status_code=400, detail=f"实验ID格式无效: {exp_id}")


def _validate_filename(filename: str) -> None:
    """验证文件名安全性"""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    # 防止路径遍历
    if ".." in filename:
        raise HTTPException(status_code=400, detail=f"文件名包含非法路径遍历字符: {filename}")
    # 检查扩展名
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不允许的文件扩展名: {ext}，允许: {ALLOWED_EXTENSIONS}"
        )


class FileUploadResponse(BaseModel):
    """文件上传响应"""
    success: bool
    exp_id: str
    filename: str
    size: int
    path: str
    message: str


class BatchFileUploadRequest(BaseModel):
    """批量文件上传请求（JSON方式）"""
    files: Dict[str, str]  # filename -> content


class BatchFileUploadResponse(BaseModel):
    """批量文件上传响应"""
    success: bool
    exp_id: str
    uploaded: List[str]
    failed: List[Dict[str, str]]
    total: int
    success_count: int


class ExperimentInfo(BaseModel):
    """实验信息"""
    exp_id: str
    path: str
    files: List[str]
    total_size: int


class ExperimentListResponse(BaseModel):
    """实验列表响应"""
    experiments: List[ExperimentInfo]
    total: int


# ================================================================
# 文件上传接口
# ================================================================

@router.post("/experiments/{exp_id}/files", response_model=FileUploadResponse)
async def upload_file(
    exp_id: str,
    file: UploadFile = File(...),
):
    """
    上传单个文件到QE实验目录
    
    Args:
        exp_id: 实验ID (如 qe_exp_abc123)
        file: 上传的文件
        
    Returns:
        上传结果
    """
    _validate_exp_id(exp_id)
    _validate_filename(file.filename)
    
    # 创建实验目录
    exp_dir = QE_WORKSPACE_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取文件内容
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {len(content)} bytes，最大允许 {MAX_FILE_SIZE} bytes"
        )
    
    # 写入文件
    target_path = exp_dir / file.filename
    target_path.write_bytes(content)
    
    logger.info(f"[QE] 文件上传成功: {exp_id}/{file.filename} ({len(content)} bytes)")
    
    return FileUploadResponse(
        success=True,
        exp_id=exp_id,
        filename=file.filename,
        size=len(content),
        path=str(target_path),
        message=f"文件 {file.filename} 上传成功"
    )


@router.post("/experiments/{exp_id}/files/batch", response_model=BatchFileUploadResponse)
async def upload_files_batch(
    exp_id: str,
    request: BatchFileUploadRequest,
):
    """
    批量上传文件到QE实验目录（JSON方式，适合文本文件）
    
    Args:
        exp_id: 实验ID
        request: 包含 files 字典 {filename: content}
        
    Returns:
        批量上传结果
    """
    _validate_exp_id(exp_id)
    
    # 创建实验目录
    exp_dir = QE_WORKSPACE_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded = []
    failed = []
    
    for filename, content in request.files.items():
        try:
            _validate_filename(filename)
            
            content_bytes = content.encode("utf-8")
            if len(content_bytes) > MAX_FILE_SIZE:
                failed.append({
                    "filename": filename,
                    "error": f"文件过大: {len(content_bytes)} bytes"
                })
                continue
            
            target_path = exp_dir / filename
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            uploaded.append(filename)
            logger.info(f"[QE] 批量上传: {exp_id}/{filename} ({len(content_bytes)} bytes)")
            
        except HTTPException as e:
            failed.append({"filename": filename, "error": e.detail})
        except Exception as e:
            failed.append({"filename": filename, "error": str(e)})
    
    success = len(failed) == 0
    
    return BatchFileUploadResponse(
        success=success,
        exp_id=exp_id,
        uploaded=uploaded,
        failed=failed,
        total=len(request.files),
        success_count=len(uploaded),
    )


# ================================================================
# 查询接口
# ================================================================

@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments():
    """
    列出所有QE实验目录
    
    Returns:
        实验列表
    """
    experiments = []
    
    if not QE_WORKSPACE_ROOT.exists():
        return ExperimentListResponse(experiments=[], total=0)
    
    for item in sorted(QE_WORKSPACE_ROOT.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            files = []
            total_size = 0
            for f in item.iterdir():
                if f.is_file():
                    files.append(f.name)
                    total_size += f.stat().st_size
            
            experiments.append(ExperimentInfo(
                exp_id=item.name,
                path=str(item),
                files=sorted(files),
                total_size=total_size,
            ))
    
    return ExperimentListResponse(
        experiments=experiments,
        total=len(experiments),
    )


@router.get("/experiments/{exp_id}/files")
async def list_experiment_files(exp_id: str):
    """
    列出实验目录中的文件
    
    Args:
        exp_id: 实验ID
        
    Returns:
        文件列表及详情
    """
    _validate_exp_id(exp_id)
    
    exp_dir = QE_WORKSPACE_ROOT / exp_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"实验目录不存在: {exp_id}")
    
    files = []
    for f in sorted(exp_dir.iterdir()):
        if f.is_file():
            stat = f.stat()
            files.append({
                "filename": f.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
    
    return {
        "exp_id": exp_id,
        "path": str(exp_dir),
        "files": files,
        "total": len(files),
    }


# ================================================================
# 删除接口
# ================================================================

@router.delete("/experiments/{exp_id}")
async def delete_experiment(exp_id: str):
    """
    删除QE实验目录
    
    Args:
        exp_id: 实验ID
        
    Returns:
        删除结果
    """
    _validate_exp_id(exp_id)
    
    exp_dir = QE_WORKSPACE_ROOT / exp_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"实验目录不存在: {exp_id}")
    
    try:
        shutil.rmtree(exp_dir)
        logger.info(f"[QE] 实验目录已删除: {exp_id}")
        return {
            "success": True,
            "exp_id": exp_id,
            "message": f"实验目录 {exp_id} 已删除"
        }
    except Exception as e:
        logger.error(f"[QE] 删除实验目录失败: {exp_id}, {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
