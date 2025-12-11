import os
from pathlib import Path

import pandas as pd
import yaml

import qlib
from qlib.utils import init_instance_by_config


def main() -> None:
    """检查当前 workspace 下 conf_combined_factors.yaml 中各个 dataloader 的输出索引结构。

    使用方式（在 WSL 中）：

        cd /mnt/c/Users/lc999/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/5223c0ad21f442ddb64c887b2c0a9d09
        conda activate rdagent-gpu
        python ../../tools/inspect_conf_combined_factors_dataloaders.py
    """

    ws_dir = Path(__file__).resolve().parents[1] / "git_ignore_folder" / "RD-Agent_workspace" / "5223c0ad21f442ddb64c887b2c0a9d09"
    conf_path = ws_dir / "conf_combined_factors.yaml"

    print("[信息] workspace:", ws_dir)
    print("[信息] 使用配置:", conf_path)

    with open(conf_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    handler_conf = conf["task"]["dataset"]["kwargs"]["handler"]
    dl_list_conf = handler_conf["kwargs"]["data_loader"]["kwargs"]["dataloader_l"]

    print("[信息] 共发现 dataloader 个数:", len(dl_list_conf))

    # 初始化 qlib（client 模式）
    # 当前版本没有 is_initialized，直接调用一次 init 即可（幂等）
    qlib.init()

    start_time = handler_conf["kwargs"].get("start_time") or "2015-01-01"
    end_time = handler_conf["kwargs"].get("end_time") or "2020-01-01"
    handler_instruments = handler_conf["kwargs"].get("instruments")
    print("[信息] handler instruments 配置:", handler_instruments)

    for i, dl_conf in enumerate(dl_list_conf):
        print("\n========== dataloader", i, "==========")
        print("config:", dl_conf)

        dl = init_instance_by_config(dl_conf)

        # 1) instruments=None 时，由 loader 自己决定股票池，主要看原始索引结构
        print("-- load with instruments=None --")
        try:
            df = dl.load(instruments=None, start_time=start_time, end_time=end_time)
        except Exception as e:
            print("[错误] load(instruments=None) 失败:", repr(e))
        else:
            print("类型:", type(dl))
            print("DataFrame 形状:", df.shape)
            print("index type:", type(df.index))
            print("index nlevels:", getattr(df.index, "nlevels", "N/A"))
            print("index names:", getattr(df.index, "names", None))

            if isinstance(df.index, pd.MultiIndex):
                print("index level0 sample (前 5 个):", list(df.index.levels[0][:5]))
                print("index level1 sample (前 5 个):", list(df.index.levels[1][:5]))

            print("columns type:", type(df.columns))
            print("columns nlevels:", getattr(df.columns, "nlevels", "N/A"))
            print("columns names:", getattr(df.columns, "names", None))
            print("columns sample (前 5 个):", list(df.columns[:5]))

        # 2) 使用 handler 中配置的 instruments，再调一次 load，看是否发生索引降维
        if handler_instruments is not None:
            print("-- load with handler_instruments=", handler_instruments, "--")
            try:
                df_inst = dl.load(instruments=handler_instruments, start_time=start_time, end_time=end_time)
            except Exception as e:
                print("[错误] load(instruments=handler_instruments) 失败:", repr(e))
            else:
                print("[with instruments] DataFrame 形状:", df_inst.shape)
                print("[with instruments] index type:", type(df_inst.index))
                print("[with instruments] index nlevels:", getattr(df_inst.index, "nlevels", "N/A"))
                print("[with instruments] index names:", getattr(df_inst.index, "names", None))

                if isinstance(df_inst.index, pd.MultiIndex):
                    print("[with instruments] index level0 sample (前 5 个):", list(df_inst.index.levels[0][:5]))
                    print("[with instruments] index level1 sample (前 5 个):", list(df_inst.index.levels[1][:5]))

                print("[with instruments] columns type:", type(df_inst.columns))
                print("[with instruments] columns nlevels:", getattr(df_inst.columns, "nlevels", "N/A"))
                print("[with instruments] columns names:", getattr(df_inst.columns, "names", None))
                print("[with instruments] columns sample (前 5 个):", list(df_inst.columns[:5]))


if __name__ == "__main__":
    main()
