#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

def fix_env_encoding():
    env_file = Path('.env')
    
    # 读取原始文件
    with open(env_file, 'rb') as f:
        content = f.read()
    
    print(f'原始文件大小: {len(content)} bytes')
    
    # 查找UTF-16编码的开始位置（查找乱码的开始）
    # UTF-16 LE的特征是每个字符后面有\x00
    # 查找"QLIB_SKIP_CODE_LOGGING"的UTF-16编码
    utf16_start = content.find(b'Q\x00L\x00I\x00B\x00_\x00S\x00K\x00I\x00P\x00_\x00C\x00O\x00D\x00E\x00_\x00L\x00O\x00G\x00G\x00I\x00N\x00G\x00')
    
    if utf16_start != -1:
        print(f'找到UTF-16编码内容，位置: {utf16_start}')
        
        # 保留UTF-16之前的内容
        new_content = content[:utf16_start]
        
        # 追加正确的UTF-8编码内容
        new_content += b'\n# 禁用qlib代码记录，避免编码问题\n'
        new_content += b'QLIB_SKIP_CODE_LOGGING=true\n'
        new_content += b'MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false\n'
        
        # 写回文件
        with open(env_file, 'wb') as f:
            f.write(new_content)
        
        print(f'修复后文件大小: {len(new_content)} bytes')
        print('已修复.env文件编码问题')
        
        # 验证修复结果
        with open(env_file, 'rb') as f:
            fixed_content = f.read()
        
        print('最后100 bytes:')
        print(fixed_content[-100:])
    else:
        print('未找到UTF-16编码内容，可能已经修复或文件格式不同')
        
        # 检查文件末尾是否已经有正确的配置
        if b'QLIB_SKIP_CODE_LOGGING=true' in content:
            print('文件中已包含QLIB_SKIP_CODE_LOGGING=true配置')
        else:
            print('文件中不包含QLIB_SKIP_CODE_LOGGING=true配置')
            print('建议手动检查文件内容')

if __name__ == '__main__':
    fix_env_encoding()
