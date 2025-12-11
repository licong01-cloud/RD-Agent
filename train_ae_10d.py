import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data.astype("float32")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])


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


def build_sequences(df: pd.DataFrame, feature_cols: List[str], window: int) -> np.ndarray:
    """Build rolling 10-day sequences across all instruments.

    df: MultiIndex (datetime, instrument) sorted by index.
    Returns array of shape (n_samples, window * n_features).
    """
    # unstack to wide format: index=datetime, columns=(feature, instrument)
    # 为避免内存过大，只用 close 一列是默认行为，可通过 feature_cols 控制
    panels = []
    for col in feature_cols:
        wide = df[col].unstack("instrument").sort_index()
        panels.append(wide)

    # stack features on last axis: (time, n_instr, n_feat)
    data_3d = np.stack([p.values for p in panels], axis=-1)  # (T, N, F)

    T, N, F = data_3d.shape
    if T < window:
        return np.zeros((0, window * F), dtype="float32")

    seqs = []
    for t in range(window - 1, T):
        window_slice = data_3d[t - window + 1 : t + 1]  # (window, N, F)
        # reshape: (N, window * F)
        window_flat = window_slice.reshape(window, N, F).transpose(1, 0, 2).reshape(N, window * F)
        seqs.append(window_flat)

    seqs_arr = np.concatenate(seqs, axis=0)  # (T_window * N, window*F)
    # 去掉含 nan 的样本
    mask = np.isfinite(seqs_arr).all(axis=1)
    return seqs_arr[mask]


def main():
    parser = argparse.ArgumentParser(description="Pre-train global 10-day AE on AIstock daily_pv.h5")
    parser.add_argument("--h5-path", type=str, default="daily_pv.h5", help="Path to daily_pv.h5")
    parser.add_argument("--window", type=int, default=10, help="Sequence window length")
    parser.add_argument("--features", type=str, default="close", help="Comma-separated feature columns, after rename")
    parser.add_argument("--output", type=str, default="models/ae_10d.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=500_000, help="Subsample upper bound for training")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[INFO] Loading HDF5 from {args.h5_path}")
    df = pd.read_hdf(args.h5_path, key="data").sort_index()

    # 默认重命名与项目规范保持一致（只处理我们可能用到的列）
    rename_map = {
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$close": "close",
        "$volume": "volume",
        "$amount": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in daily_pv.h5 after rename.")

    print(f"[INFO] Building {args.window}-day sequences with features: {feature_cols}")
    seqs = build_sequences(df, feature_cols, args.window)
    if seqs.shape[0] == 0:
        raise RuntimeError("No valid sequences built from daily_pv.h5; check data and window size.")

    print(f"[INFO] Total sequences before subsample: {seqs.shape[0]}")
    if seqs.shape[0] > args.max_samples:
        idx = np.random.choice(seqs.shape[0], size=args.max_samples, replace=False)
        seqs = seqs[idx]
        print(f"[INFO] Subsampled to {seqs.shape[0]} sequences")

    # 简单标准化
    mean = seqs.mean(axis=0, keepdims=True)
    std = seqs.std(axis=0, keepdims=True) + 1e-12
    seqs_norm = (seqs - mean) / std

    dataset = SequenceDataset(seqs_norm)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = seqs_norm.shape[1]
    model = AE10D(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"[INFO] Start training AE on device={device}, input_dim={input_dim}")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batch = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1
        avg_loss = total_loss / max(n_batch, 1)
        print(f"[INFO] Epoch {epoch}/{args.epochs}, loss={avg_loss:.6f}")

    # 保存模型和归一化参数
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "window": args.window,
        "features": feature_cols,
        "mean": mean.astype("float32"),
        "std": std.astype("float32"),
    }
    torch.save(payload, args.output)
    print(f"[INFO] Saved AE model and stats to {args.output}")


if __name__ == "__main__":
    main()
