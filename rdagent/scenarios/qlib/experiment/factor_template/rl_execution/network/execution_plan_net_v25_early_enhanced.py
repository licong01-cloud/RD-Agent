"""v25 Early Plan Net - 增强版（添加历史特征）"""
import torch
import torch.nn as nn


class EarlyPlanNetEnhanced(nn.Module):
    """前30分钟执行计划网络 - 增强版

    输入特征：
    - gap_ratio, gap_ratio_signed, limit_pct, is_buy, gap_bucket (5维)
    - day_features (10维历史特征)
    总计：15维
    """
    def __init__(self):
        super().__init__()
        # 输入：5个标量 + 10个历史特征 = 15维
        self.mlp = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 30),
            nn.Softmax(dim=-1)
        )

    def forward(self, gap_bucket, gap_ratio, gap_ratio_signed, limit_pct, is_buy, day_features):
        # gap_bucket归一化
        gap_bucket_norm = gap_bucket.float() / 8.0

        # 拼接所有特征
        x = torch.cat([
            gap_ratio.unsqueeze(1),
            gap_ratio_signed.unsqueeze(1),
            limit_pct.unsqueeze(1),
            is_buy.unsqueeze(1),
            gap_bucket_norm.unsqueeze(1),
            day_features  # [batch, 10]
        ], dim=1)  # [batch, 15]

        return self.mlp(x)
