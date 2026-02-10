#!/usr/bin/env python
"""检查pickle文件中引用的模块信息"""
import pickle
import sys
from pathlib import Path

# 测试读取pickle文件
pkl_file = Path("/mnt/f/Dev/RD-Agent-main/log/model_comparison_20260117_193508/log/2026-01-17_11-35-08-830195/Loop_0/running/Qlib_execute_log/34208-34667/2026-01-17_13-11-08-273248.pkl")

if not pkl_file.exists():
    print(f"文件不存在: {pkl_file}")
    sys.exit(1)

try:
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    print(f"成功读取pickle文件")
    print(f"数据类型: {type(data)}")
    if hasattr(data, '__dict__'):
        print(f"属性: {list(data.__dict__.keys())}")
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print(f"尝试分析pickle文件中的模块引用...")
    
    # 尝试读取pickle文件但不反序列化
    import pickletools
    with open(pkl_file, 'rb') as f:
        pickletools.dis(f)
except Exception as e:
    print(f"错误: {e}")
    print(f"错误类型: {type(e).__name__}")
