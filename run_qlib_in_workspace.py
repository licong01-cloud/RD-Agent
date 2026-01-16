#!/usr/bin/env python3
"""在 workspace 中运行 Qlib 来复现错误"""
import subprocess
import sys
from pathlib import Path

workspace_path = Path(r"F:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/6257164b31854b49b41f60c3f531da55")
config_file = workspace_path / "conf_combined_factors_sota_model.yaml"

print(f"工作目录: {workspace_path}")
print(f"配置文件: {config_file}")
print("=" * 80)

# 检查配置文件是否存在
if not config_file.exists():
    print(f"错误: 配置文件不存在: {config_file}")
    sys.exit(1)

# 设置环境变量
env = {
    "PYTHONPATH": str(workspace_path),
    "QLIB_WORKER": "1",
}

# 设置运行参数
run_env = {
    "n_epochs": "20",
    "lr": "1e-3",
    "early_stop": "5",
    "batch_size": "256",
    "weight_decay": "1e-4",
    "num_features": "39",  # 20 alpha158 + 19 dynamic factors
}

env.update(run_env)

print(f"运行参数: {run_env}")
print("=" * 80)

# 运行 Qlib
print("\n开始运行 Qlib...")
try:
    result = subprocess.run(
        ["python", "-m", "qlib.workflow.cli", "run", "config", str(config_file)],
        cwd=str(workspace_path),
        env={**subprocess.os.environ, **env},
        capture_output=True,
        timeout=300,  # 5分钟超时
        text=True
    )

    print(f"返回码: {result.returncode}")
    print(f"\n标准输出:")
    print(result.stdout)

    if result.stderr:
        print(f"\n标准错误:")
        print(result.stderr)

    # 检查是否有 UnicodeDecodeError
    if "UnicodeDecodeError" in result.stdout or "UnicodeDecodeError" in result.stderr:
        print("\n" + "=" * 80)
        print("⚠️ 发现 UnicodeDecodeError!")
        print("=" * 80)

except subprocess.TimeoutExpired:
    print("错误: 执行超时 (5分钟)")
except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 80)
print("执行完成")
