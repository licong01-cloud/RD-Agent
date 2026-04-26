"""v25 方案B Stage 2: 后210分钟执行计划网络。

输入:
  gap_bucket_idx: [B] — 归一化缺口桶索引 (0~8)
  gap_ratio: [B] — 归一化缺口值
  is_buy: [B] — 方向 (1.0=买, 0.0=卖)
  early_weight: [B] — 前30分钟总权重占比
  early_peak_pos: [B] — 前30分钟峰值位置 (0~1归一化)
  early_concentration: [B] — 前30分钟集中度 (max/mean)

输出: [B, 210] softmax 执行权重分布 (后210分钟)

架构: MLP
  - gap_bucket embedding (9 → 16)
  - 6个标量特征 (gap_ratio, is_buy, 3个前30分钟统计)
  - 隐藏层: [128, 256]
  - 输出: 210维 softmax
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatePlanNet(nn.Module):

    def __init__(self, gap_buckets: int = 9, gap_emb_dim: int = 16,
                 late_len: int = 210):
        super().__init__()
        self.late_len = late_len

        # Gap embedding
        self.gap_embedding = nn.Embedding(gap_buckets, gap_emb_dim)

        # MLP: gap_emb(16) + gap_ratio(1) + is_buy(1)
        #      + early_weight(1) + early_peak_pos(1) + early_concentration(1) = 22
        input_dim = gap_emb_dim + 5

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, late_len),
        )

    def forward(self, gap_bucket_idx: torch.Tensor, gap_ratio: torch.Tensor,
                is_buy: torch.Tensor, early_weight: torch.Tensor,
                early_peak_pos: torch.Tensor, early_concentration: torch.Tensor) -> torch.Tensor:
        # Gap embedding
        gap_emb = self.gap_embedding(gap_bucket_idx)  # [B, 16]

        # Extra scalars → [B, 5]
        extra = torch.stack([gap_ratio, is_buy, early_weight,
                             early_peak_pos, early_concentration], dim=1)

        # Fuse
        fused = torch.cat([gap_emb, extra], dim=1)  # [B, 21]
        logits = self.mlp(fused)                     # [B, 210]
        return F.softmax(logits, dim=-1)
