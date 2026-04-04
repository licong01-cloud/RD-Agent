"""qrun 的薄包装：Jinja2 渲染 → YAML 解析 → limit_threshold 转换 → Qlib workflow。

YAML 模板含 Jinja2 变量 ({{ n_epochs }} 等) 和条件 ({% if %}),
需要先用环境变量渲染 Jinja2，再解析 YAML。

YAML 中 limit_threshold: ["$limit_up", "$limit_down"] 被解析为 Python list，
但 Qlib Exchange 的 LT_TP_EXP 模式需要 tuple。本脚本做这个转换。

用法：python qrun_limit.py conf.yaml
"""
import os
import sys
from pathlib import Path

from jinja2 import Template, meta
from ruamel.yaml import YAML
from qlib.model.trainer import task_train
try:
    from qlib.workflow.cli import sys_config
except ModuleNotFoundError:
    # qlib ≥ 0.9.7 removed cli.py; inline the function
    def sys_config(config, config_path):
        sc = config.get("sys", {})
        for p in ([sc["path"]] if isinstance(sc.get("path"), str) else list(sc.get("path", []))):
            sys.path.append(p)
        for p in ([sc["rel_path"]] if isinstance(sc.get("rel_path"), str) else list(sc.get("rel_path", []))):
            sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))
import qlib
from qlib.config import C


def render_yaml_template(yaml_path: str) -> str:
    """用环境变量渲染 Jinja2 模板，返回渲染后的 YAML 字符串。"""
    with open(yaml_path, "r") as f:
        content = f.read()
    template = Template(content)
    env = template.environment
    parsed = env.parse(content)
    variables = meta.find_undeclared_variables(parsed)
    context = {var: os.environ[var] for var in variables if var in os.environ}
    return template.render(context)


def patch_backtest_config(config: dict):
    """递归修补 backtest 配置：
    1. exchange_kwargs.limit_threshold: list → tuple (Qlib LT_TP_EXP 模式)
    2. benchmark: null 保留为 None（Qlib create_account_instance 已修补为传 None 而非 {}）
    """
    if isinstance(config, dict):
        for key, val in config.items():
            if key == 'exchange_kwargs' and isinstance(val, dict):
                lt = val.get('limit_threshold')
                if isinstance(lt, list):
                    val['limit_threshold'] = tuple(lt)
            elif key == 'backtest' and isinstance(val, dict):
                patch_backtest_config(val)
            else:
                patch_backtest_config(val)
    elif isinstance(config, list):
        for item in config:
            patch_backtest_config(item)


def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "conf.yaml"

    # Jinja2 渲染 → YAML 解析
    rendered = render_yaml_template(yaml_path)
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(rendered)

    patch_backtest_config(config)
    sys_config(config, config_path=yaml_path)

    # Init qlib — 使用 MLFLOW_TRACKING_URI 环境变量（由 workspace 设置），或默认 cwd/mlruns
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = str(Path(os.getcwd()).resolve() / "mlruns")
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + tracking_uri
    qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    # Run training + backtesting
    experiment_name = config.get("experiment_name", "workflow")
    recorder = task_train(config.get("task"), experiment_name=experiment_name)
    recorder.save_objects(config=config)


if __name__ == '__main__':
    main()
