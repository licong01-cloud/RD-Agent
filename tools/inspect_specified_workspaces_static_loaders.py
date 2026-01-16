import pandas as pd
from pathlib import Path
from typing import Any

import yaml

"""
检查指定多个 RD-Agent Qlib workspace 中，所有 YAML/YML 里 StaticDataLoader 的因子表索引结构。

使用方式（在 WSL 中示例）：

    cd /mnt/f/dev/RD-Agent-main
    conda activate rdagent-gpu  # 按你的环境名调整
    python tools/inspect_specified_workspaces_static_loaders.py

脚本行为：
1. 使用预先写死的 workspace 路径列表：
   - /mnt/f/dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/47dc878efd6f40b89b7a32cd25755bbb
   - /mnt/f/dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/bca21629d58243ad950a9c8682a9a073
   - /mnt/f/dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/d693c64d7947480ea2ac1a8f5b21db01
2. 在每个 workspace 目录下搜索所有 .yaml 和 .yml 文件；
3. 解析 YAML，找到 data_loader.kwargs.dataloader_l 中
   所有 class 为 qlib.data.dataset.loader.StaticDataLoader 的条目，读取其 kwargs.config 路径；
4. 对每个找到的 config 路径：
   - 打印路径
   - 读取 DataFrame（自动识别 parquet / pkl）
   - 打印 index 类型、层级数、名称和前几行示例；

只读不写，用于定位是否存在 index 只有一层的静态因子表，从而解释 MergeError。
"""


WORKSPACE_PATHS = [
    Path("/mnt/f/dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/47dc878efd6f40b89b7a32cd25755bbb"),
    Path("/mnt/f/dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/bca21629d58243ad950a9c8682a9a073"),
    Path("/mnt/f/dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/d693c64d7947480ea2ac1a8f5b21db01"),
]


def find_yaml_configs(ws: Path) -> list[Path]:
    """在 workspace 目录下查找 qrun 相关的 YAML/YML 配置文件。"""
    yamls: list[Path] = []
    for suffix in ("*.yaml", "*.yml"):
        for p in ws.rglob(suffix):
            yamls.append(p)
    return yamls


def inspect_yaml(yaml_path: Path) -> None:
    print("==============================")
    print("YAML/YML 文件:", yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def find_handler_dicts(obj: Any) -> list[dict[str, Any]]:
        """递归查找包含 data_loader 的 handler / data_handler_config 字典。"""
        found: list[dict[str, Any]] = []
        if isinstance(obj, dict):
            if "data_loader" in obj and isinstance(obj["data_loader"], dict):
                found.append(obj)
            if "data_handler_config" in obj and isinstance(obj["data_handler_config"], dict):
                dhc = obj["data_handler_config"]
                if "data_loader" in dhc and isinstance(dhc["data_loader"], dict):
                    found.append(dhc)
            for v in obj.values():
                found.extend(find_handler_dicts(v))
        elif isinstance(obj, list):
            for it in obj:
                found.extend(find_handler_dicts(it))
        return found

    handler_dicts = find_handler_dicts(cfg)
    if not handler_dicts:
        print("[提示] 未在该 YAML/YML 中找到包含 data_loader 的 handler 配置。")
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
            if not p.is_absolute():
                p = (yaml_path.parent / p).resolve()

            if not p.exists():
                print("    [警告] 文件不存在:", p)
                continue

            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
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
    for ws in WORKSPACE_PATHS:
        print("============================================================")
        print("检查 workspace:", ws)
        if not ws.exists():
            print("[错误] 该 workspace 目录不存在。")
            continue

        yaml_files = find_yaml_configs(ws)
        if not yaml_files:
            print("[错误] 在该 workspace 下未找到任何 .yaml/.yml 文件。")
            continue

        print("[信息] 在该 workspace 下找到的 YAML/YML 文件:")
        for yp in yaml_files:
            print("  -", yp)
        print()

        for yp in sorted(yaml_files):
            inspect_yaml(yp)


if __name__ == "__main__":
    main()
