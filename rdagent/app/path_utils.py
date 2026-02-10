"""WSL/Windows路径转换工具

用于处理RD-Agent在WSL环境和AIstock在Windows环境之间的路径转换问题。
"""
import os
from pathlib import Path


def convert_wsl_to_windows(wsl_path: str) -> Path:
    """将WSL路径转换为Windows路径
    
    例如: /mnt/f/Dev/... -> F:/Dev/...
    
    Args:
        wsl_path: WSL格式的路径
        
    Returns:
        Windows格式的Path对象
    """
    normalized = str(wsl_path).replace('\\', '/')
    
    if normalized.startswith('/mnt/'):
        parts = normalized.split('/')
        if len(parts) >= 3:
            drive = parts[2].upper()
            rest = '/'.join(parts[3:]) if len(parts) > 3 else ''
            return Path(f"{drive}:/{rest}")
    
    return Path(wsl_path)


def convert_windows_to_wsl(win_path: str) -> Path:
    """将Windows路径转换为WSL路径
    
    例如: F:/Dev/... -> /mnt/f/Dev/...
    
    Args:
        win_path: Windows格式的路径
        
    Returns:
        WSL格式的Path对象
    """
    normalized = str(win_path).replace('\\', '/')
    
    # 检查是否是Windows驱动器路径 (如 F:/ 或 F:\)
    if len(normalized) > 1 and normalized[1] == ':':
        drive = normalized[0].lower()
        rest = normalized[3:] if len(normalized) > 3 else ''
        return Path(f"/mnt/{drive}/{rest}")
    
    return Path(win_path)


def normalize_path(path: str | Path) -> Path:
    """根据当前系统自动转换路径
    
    - 在Windows系统上：将WSL路径转换为Windows路径
    - 在WSL/Linux系统上：将Windows路径转换为WSL路径
    
    Args:
        path: 任意格式的路径
        
    Returns:
        适合当前系统的Path对象
    """
    if isinstance(path, Path):
        path = str(path)
    
    if not path:
        return Path()
    
    path_str = str(path)
    
    # 当前系统是Windows
    if os.name == 'nt':
        # 如果是WSL路径，转换为Windows路径
        if path_str.replace('\\', '/').startswith('/mnt/'):
            return convert_wsl_to_windows(path_str)
        # 已经是Windows路径，直接返回
        return Path(path_str)
    else:
        # 当前系统是WSL/Linux
        # 如果是Windows路径，转换为WSL路径
        if len(path_str) > 1 and path_str[1] == ':':
            return convert_windows_to_wsl(path_str)
        # 已经是WSL路径，直接返回
        return Path(path_str)
