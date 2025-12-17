import os
from pathlib import Path

import pandas as pd


def main():
    """Convert existing AE factor result.h5 to result.parquet for Qlib StaticDataLoader.

    - Input H5:  F:/Dev/AIstock/factors/ae_recon_error_10d/result.h5
    - Output:    F:/Dev/AIstock/factors/ae_recon_error_10d/result.parquet

    The H5 is assumed to contain a DataFrame under key="data" with
    MultiIndex(datetime, instrument) and one column `ae_recon_error_10d`.
    """

    snapshot_root = r"F:/Dev/AIstock/factors/ae_recon_error_10d"
    # 在 Windows 下直接用 <Drive>:/ 路径；在 WSL/Linux 下，将 <Drive>:/ 前缀映射到 /mnt/<drive>/
    if os.name != "nt" and len(snapshot_root) >= 3 and snapshot_root[1:3] == ":/":
        drive = snapshot_root[0].lower()
        root = Path("/mnt") / drive / snapshot_root[3:]
    else:
        root = Path(snapshot_root)
    h5_path = root / "result.h5"
    parquet_path = root / "result.parquet"
    pkl_path = root / "result.pkl"

    if not h5_path.exists():
        raise FileNotFoundError(f"AE factor H5 not found at {h5_path}")

    print(f"[INFO] Loading AE factor from {h5_path}")
    df = pd.read_hdf(h5_path, key="data")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving AE factor parquet to {parquet_path} with shape={df.shape}")
    df.to_parquet(parquet_path)

    print(f"[INFO] Saving AE factor pickle to {pkl_path}")
    df.to_pickle(pkl_path)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
