import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _extract_alpha158_factors(conf: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract Alpha158 factor definitions (expression + name) from Qlib config.

    约定结构同 conf_combined_factors_dynamic.yaml：

    data_handler_config:
      alpha158_config:
        feature:
          - [expr_list]
          - [name_list]
    """

    dh_cfg = conf.get("data_handler_config") or {}
    if not isinstance(dh_cfg, dict):
        return []

    # 兼容两种结构：
    # 1) data_handler_config.alpha158_config.feature
    # 2) data_handler_config.data_loader.kwargs.alpha158_config.feature
    alpha_cfg: dict[str, Any] | None = None

    direct_alpha = dh_cfg.get("alpha158_config")
    if isinstance(direct_alpha, dict):
        alpha_cfg = direct_alpha
    else:
        data_loader = dh_cfg.get("data_loader") or {}
        if isinstance(data_loader, dict):
            kwargs = data_loader.get("kwargs") or {}
            if isinstance(kwargs, dict):
                nested_alpha = kwargs.get("alpha158_config")
                if isinstance(nested_alpha, dict):
                    alpha_cfg = nested_alpha

    if not isinstance(alpha_cfg, dict):
        return []

    feature = alpha_cfg.get("feature") or []
    if not isinstance(feature, list) or len(feature) < 2:
        return []

    expr_list = feature[0] or []
    name_list = feature[1] or []
    if not (isinstance(expr_list, list) and isinstance(name_list, list)):
        return []

    n = min(len(expr_list), len(name_list))
    factors: list[dict[str, Any]] = []
    for i in range(n):
        expr = expr_list[i]
        name = name_list[i]
        if not isinstance(expr, str) or not isinstance(name, str):
            continue
        factors.append(
            {
                "name": name,
                "expression": expr,
                "source": "qlib_alpha158",
                "region": conf.get("qlib_init", {}).get("region", "cn"),
                "tags": ["alpha158"],
            }
        )
    return factors


def run(conf_yaml: Path, output_path: Path) -> None:
    conf = _load_yaml(conf_yaml)
    factors = _extract_alpha158_factors(conf)

    payload = {
        "version": "v1",
        "generated_at_utc": _utc_now_iso(),
        "source": "qlib_alpha158",
        "factors": factors,
        "config_path": str(conf_yaml),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full Alpha158 factor metadata (expression + name) as JSON.")
    parser.add_argument(
        "--conf-yaml",
        required=True,
        help="Path to Qlib experiment YAML that contains alpha158_config (e.g., conf_combined_factors_dynamic.yaml).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for alpha158 factor meta.",
    )
    args = parser.parse_args()

    conf_yaml = Path(args.conf_yaml)
    if not conf_yaml.exists():
        raise SystemExit(f"YAML config not found: {conf_yaml}")

    output_path = Path(args.output)

    run(conf_yaml=conf_yaml, output_path=output_path)


if __name__ == "__main__":
    main()
