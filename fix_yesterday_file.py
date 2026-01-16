#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修复昨天提交的文件编码问题"""

# 读取昨天提交的文件
with open('yesterday_model_runner.py', 'rb') as f:
    content = f.read()

# 移除null bytes
content = content.replace(b'\x00', b'')

# 保存修复后的文件
with open('yesterday_model_runner_fixed.py', 'wb') as f:
    f.write(content)

print('Fixed file created')
