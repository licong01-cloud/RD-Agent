@echo off
echo 正在统计 RD-Agent 演进因子数量...
echo.

REM 设置 Python 路径
set PYTHONPATH=F:\Dev\RD-Agent-main

REM 运行统计脚本
python tools\count_evolved_factors.py

echo.
echo 统计完成！
pause
