import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

"""
检查最近一次 RD-Agent Qlib workspace 中，qrun 使用的 YAML 里所有 StaticDataLoader 的因子表索引结构。

使用方式（在 WSL 中示例）：

    cd /mnt/f/dev/RD-Agent-main
    conda activate rdagent-gpu  # 按你的环境名调整
    python tools/inspect_latest_qlib_workspace_static_loaders.py

脚本行为：
1. 在 git_ignore_folder/RD-Agent_workspace 下找到按修改时间排序最新的 workspace 目录；
2. 在该目录下搜索类似 qrun_*.yaml 或 .yaml 的配置文件；
3. 解析 YAML，找到 data_handler_config.data_loader.kwargs.dataloader_l 中
   所有 class 为 qlib.data.dataset.loader.StaticDataLoader 的条目，读取其 kwargs.config 路径；
4. 对每个找到的 config 路径：
   - 打印路径
   - 读取 DataFrame（自动识别 parquet / pkl）
   - 打印 index 类型、层级数、名称和前几行示例；

只读不写，用于定位是否存在 index 只有一层的静态因子表，从而解释 MergeError。
"""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT / "git_ignore_folder" / "RD-Agent_workspace"


def find_latest_workspace(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # 按修改时间排序，取最近一个
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_yaml_configs(ws: Path) -> list[Path]:
    """在 workspace 目录下查找可能的 qrun YAML 配置文件。"""
    yamls: list[Path] = []
    for p in ws.rglob("*.yaml"):
        yamls.append(p)
    return yamls


def _safe_get(d: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def inspect_yaml(yaml_path: Path) -> None:
    print("==============================")
    print("YAML 文件:", yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Qlib 的 task 配置通常在 task -> dataset -> handler -> data_loader
    # 这里做一个尽量鲁棒的递归搜索：找所有包含 data_handler_config 或 handler 的 dict

    def find_handler_dicts(obj: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(obj, dict):
            # 可能是 handler 或 data_handler_config
            if "data_loader" in obj and isinstance(obj["data_loader"], dict):
                found.append(obj)
            for v in obj.values():
                found.extend(find_handler_dicts(v))
        elif isinstance(obj, list):
            for it in obj:
                found.extend(find_handler_dicts(it))
        return found

    handler_dicts = find_handler_dicts(cfg)
    if not handler_dicts:
        print("[提示] 未在该 YAML 中找到包含 data_loader 的 handler 配置。")
        return

    for h_idx, handler_cfg in enumerate(handler_dicts, start=1):
        print(f"-- Handler #{h_idx} --")
        dl_cfg = handler_cfg.get("data_loader") or handler_cfg.get("data_loader_config")
        if not isinstance(dl_cfg, dict):
            print("  [提示] data_loader 不是 dict，跳过。")
            continue

        kwargs = dl_cfg.get("kwargs", {})
        dataloader_l = kwargs.get("dataloader_l")
        if not isinstance(dataloader_l, list):
            print("  [提示] 未找到 kwargs.dataloader_l，跳过。")
            continue

        for i, dl in enumerate(dataloader_l, start=1):
            dl_class = dl.get("class")
            dl_kwargs = dl.get("kwargs", {})
            print(f"  dataloader_l[{i}] class: {dl_class}")
            if not dl_class:
                continue

            if dl_class != "qlib.data.dataset.loader.StaticDataLoader":
                continue

            config_path = dl_kwargs.get("config")
            print("    [StaticDataLoader] config:", config_path)
            if not config_path:
                continue

            p = Path(config_path)
            # 处理相对路径的情况
            if not p.is_absolute():
                p = (yaml_path.parent / p).resolve()

            if not p.exists():
                print("    [警告] 文件不存在:", p)
                continue

            # 读取并打印索引信息
            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    # 简单起见，非 parquet 一律按 pickle 读
                    df = pd.read_pickle(p)
            except Exception as e:  # noqa: BLE001
                print("    [错误] 读取失败:", e)
                continue

            print("    index type:", type(df.index))
            print("    index.nlevels:", getattr(df.index, "nlevels", "N/A"))
            print("    index.names:", getattr(df.index, "names", None))
            try:
                print("    head index sample:", list(df.index[:5]))
            except Exception as e:  # noqa: BLE001
                print("    [警告] 打印 index 样例失败:", e)
        print()


def main() -> None:
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("WORKSPACE_ROOT:", WORKSPACE_ROOT)

    ws = find_latest_workspace(WORKSPACE_ROOT)
    if ws is None:
        print("[错误] 未找到任何 RD-Agent_workspace 目录。")
        return

    print("[信息] 最近的 workspace 目录:", ws)

    yaml_files = find_yaml_configs(ws)
    if not yaml_files:
        print("[错误] 在该 workspace 下未找到任何 .yaml 文件。")
        return

    print("[信息] 在该 workspace 下找到的 YAML 文件:")
    for yp in yaml_files:
        print("  -", yp)
    print()

    # 按文件名排序，确保输出顺序稳定
    for yp in sorted(yaml_files):
        inspect_yaml(yp)


if __name__ == "__main__":
    main()
