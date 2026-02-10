#!/usr/bin/env python3
"""修复 markdownlint 问题"""

import re
from pathlib import Path

def fix_markdownlint(file_path: Path):
    """修复 markdownlint 问题"""
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # MD040: 代码块缺少语言标识
        if line.strip() == '```' and i > 0:
            # 检查是否是代码块结束
            prev_line = lines[i-1].strip()
            if not prev_line.startswith('```'):
                # 这是代码块开始，添加语言标识
                # 尝试从上下文推断语言
                if 'python' in content.lower() or 'def ' in content or 'import ' in content:
                    line = '```python'
                elif 'yaml' in content.lower() or ': ' in content:
                    line = '```yaml'
                elif 'json' in content.lower():
                    line = '```json'
                elif 'text' in content.lower() or 'Loop' in content or 'based_experiments' in content:
                    line = '```text'
                else:
                    line = '```text'

        # MD012: 多个连续空行
        if i > 0 and lines[i-1].strip() == '' and line.strip() == '':
            # 跳过连续空行
            i += 1
            continue

        # MD032: 列表周围缺少空行
        if line.strip().startswith(('- ', '* ', '+ ')) or re.match(r'^\d+\.', line.strip()):
            # 列表项前需要空行
            if i > 0 and lines[i-1].strip() != '' and not lines[i-1].strip().startswith(('#', '```', '-', '*', '+')):
                # 检查前一行是否是空行或标题
                if not lines[i-1].strip().startswith('#'):
                    fixed_lines.append('')

        # 添加当前行
        fixed_lines.append(line)
        i += 1

    # 修复后处理
    fixed_content = '\n'.join(fixed_lines)

    # MD001: 标题层级递增问题
    # 检查 #### 是否出现在 ### 之后
    lines = fixed_content.split('\n')
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('#### '):
            # 检查前一个标题
            for j in range(i-1, -1, -1):
                if lines[j].startswith('#'):
                    prev_level = lines[j].count('#')
                    if prev_level == 3:
                        # 需要改为 ###
                        lines[i] = '### ' + line[5:]
                    break

    fixed_content = '\n'.join(lines)

    # 写回文件
    file_path.write_text(fixed_content, encoding='utf-8')
    print(f"已修复: {file_path}")

if __name__ == '__main__':
    doc_path = Path(r'f:/Dev/RD-Agent-main/docs/模型权重文件定位方案_v2.md')
    fix_markdownlint(doc_path)
