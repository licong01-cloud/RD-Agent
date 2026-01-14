import argparse
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore


def _parse_alpha158_from_yaml(conf_path: Path) -> list[dict[str, Any]]:
    """从 qlib Alpha158 配置 YAML 中解析出因子名和表达式。

    约定结构类似：

    data_handler_config:
      data_loader:
        kwargs:
          alpha158_config:
            feature:
              - [expr_list]
              - [name_list]

    本函数只依赖上述结构，不关心其它字段。
    """

    obj = yaml.safe_load(conf_path.read_text(encoding="utf-8"))
    dh = obj.get("data_handler_config") or obj.get("data_handler") or {}
    if not isinstance(dh, dict):
        raise ValueError("YAML 中缺少 data_handler_config/data_handler 字段")

    # 兼容 conf_combined_factors_dynamic.yaml 的结构
    data_loader = (dh.get("data_loader") or {}).get("kwargs") or {}
    alpha_conf = data_loader.get("alpha158_config") or data_loader.get("alpha158")
    if not isinstance(alpha_conf, dict):
        raise ValueError("YAML 中缺少 alpha158_config/alpha158 配置")

    feat = alpha_conf.get("feature")
    if not (isinstance(feat, list) and len(feat) >= 2):
        raise ValueError("alpha158_config.feature 结构不符合预期，应为 [expr_list, name_list]")

    expr_list, name_list = feat[0], feat[1]
    if not (isinstance(expr_list, list) and isinstance(name_list, list)):
        raise ValueError("feature[0] 和 feature[1] 必须都是列表")
    if len(expr_list) != len(name_list):
        raise ValueError(
            f"表达式数量({len(expr_list)})与名称数量({len(name_list)})不一致"
        )

    factors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expr, name in zip(expr_list, name_list):
        if not isinstance(name, str) or not name:
            continue
        if name in seen:
            # 去重：同名因子只保留第一次出现
            continue
        seen.add(name)

        factors.append(
            {
                "name": name,
                "expression": str(expr),
                "source": "qlib_alpha158",
                "region": "cn",
                "tags": ["alpha158"],
            }
        )

    return factors


def run(conf_path: Path, output_path: Path) -> None:
    factors = _parse_alpha158_from_yaml(conf_path)

    payload = {
        "version": "v1",
        "source": "qlib_alpha158",
        "generated_at_utc": None,  # 由调用方或后处理填充，如需精确时间可改为 datetime.utcnow().isoformat()
        "factors": factors,
        "config_path": str(conf_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"已从 {conf_path} 解析出 {len(factors)} 个 Alpha 因子，写入 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 qlib Alpha158 配置 YAML 生成 alpha158_meta.json",
    )
    parser.add_argument(
        "--conf-yaml",
        required=True,
        help="包含 Alpha158 feature 定义的 qlib YAML 路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 alpha158_meta.json 路径",
    )
    args = parser.parse_args()

    conf_path = Path(args.conf_yaml)
    if not conf_path.exists():
        raise SystemExit(f"配置文件不存在: {conf_path}")

    output_path = Path(args.output)
    run(conf_path=conf_path, output_path=output_path)


if __name__ == "__main__":
    main()
