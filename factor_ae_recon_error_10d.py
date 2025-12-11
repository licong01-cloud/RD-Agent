import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class AE10D(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def build_sequences_with_index(df: pd.DataFrame, feature_cols: List[str], window: int):
    """Build rolling sequences and return (seqs, idx_end).

    seqs: np.ndarray, shape (n_samples, window * n_features)
    idx_end: list of (datetime, instrument) for each sample (窗口结束日对应的索引)
    """
    panels = []
    for col in feature_cols:
        wide = df[col].unstack("instrument").sort_index()
        panels.append(wide)

    data_3d = np.stack([p.values for p in panels], axis=-1)  # (T, N, F)
    dates = panels[0].index
    instruments = panels[0].columns

    T, N, F = data_3d.shape
    if T < window:
        return np.zeros((0, window * F), dtype="float32"), []

    seqs = []
    idx_end = []
    for t in range(window - 1, T):
        window_slice = data_3d[t - window + 1 : t + 1]  # (window, N, F)
        window_flat = window_slice.reshape(window, N, F).transpose(1, 0, 2).reshape(N, window * F)
        # 对应结束日期为 dates[t]
        dt = dates[t]
        for i, inst in enumerate(instruments):
            seq = window_flat[i]
            if np.isfinite(seq).all():
                seqs.append(seq)
                idx_end.append((dt, inst))

    if not seqs:
        return np.zeros((0, window * F), dtype="float32"), []

    return np.vstack(seqs).astype("float32"), idx_end


def main():
    # 本脚本遵循 CoSTEER 因子规范：从当前目录读取 daily_pv.h5，输出 result.h5
    h5_path = "daily_pv.h5"
    model_path = os.environ.get("AE10D_MODEL_PATH", "models/ae_10d.pth")
    factor_name = "ae_recon_error_10d"

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"daily_pv.h5 not found in current directory: {os.getcwd()}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained AE model not found: {model_path}")

    print(f"[INFO] Loading HDF5 from {h5_path}")
    df = pd.read_hdf(h5_path, key="data").sort_index()

    rename_map = {
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$close": "close",
        "$volume": "volume",
        "$amount": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    print(f"[INFO] Loading pretrained AE model from {model_path}")
    payload = torch.load(model_path, map_location="cpu")
    window = int(payload["window"])
    feature_cols = payload["features"]
    mean = payload["mean"]  # shape (1, input_dim)
    std = payload["std"]
    input_dim = int(payload["input_dim"])

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in daily_pv.h5 after rename.")

    seqs, idx_end = build_sequences_with_index(df, feature_cols, window)
    if seqs.shape[0] == 0:
        raise RuntimeError("No valid sequences built from daily_pv.h5; check data and window size.")

    # 归一化需与训练时一致
    seqs_norm = (seqs - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE10D(input_dim=input_dim)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    print(f"[INFO] Running AE forward on {seqs_norm.shape[0]} sequences, device={device}")
    batch_size = 4096
    n = seqs_norm.shape[0]
    recon_errors = np.zeros(n, dtype="float32")

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.from_numpy(seqs_norm[start:end]).to(device)
            recon = model(batch)
            diff = (batch - recon) ** 2
            # 均值重构误差作为异常程度
            err = diff.mean(dim=1).cpu().numpy()
            recon_errors[start:end] = err

    # 将结果组装回 MultiIndex DataFrame
    idx = pd.MultiIndex.from_tuples(idx_end, names=["datetime", "instrument"])
    result_df = pd.DataFrame({factor_name: recon_errors}, index=idx)
    result_df = result_df.sort_index()

    print(f"[INFO] Saving factor to result.h5 with shape={result_df.shape}")
    result_df.to_hdf("result.h5", key="data", mode="w")


if __name__ == "__main__":
    main()
