"""HMM 训练配置."""
from dataclasses import dataclass, field
from datetime import date
from typing import List


@dataclass
class HMMTrainConfig:
    """行业 HMM 训练超参配置."""

    # HMM 模型参数
    n_states: int = 2                    # 隐状态数: 2(热/冷) 或 3(强/震/弱)
    covariance_type: str = "diag"        # full / diag / tied / spherical
    n_iter: int = 300                    # EM 最大迭代次数

    # 数据参数
    sector_level: str = "L2"             # L1 / L2
    cooldown_days: int = 3               # 状态切换冷却期

    # 信号系数
    trending_coeff: float = 1.5
    fading_coeff: float = 0.5
    neutral_coeff: float = 1.0           # 3 状态时震荡态系数

    # 观测量构建参数
    rolling_window: int = 5              # 滚动窗口天数（超额收益、波动率）
    zscore: bool = True                  # 是否对观测量做 z-score 标准化
    use_limit_down: bool = False         # 是否使用跌停占比（之前实验表明会降低效果）

    # 数据划分
    train_start: date = date(2022, 1, 1)
    train_end: date = date(2024, 6, 30)
    val_start: date = date(2024, 7, 1)
    val_end: date = date(2025, 3, 31)
    # 测试集: 2025-04-01 ~ 至今，留给 QE 回测

    # 路径
    qlib_bin_dir: str = "/home/lc999/data/qlib_bin"
    workspace_dir: str = "/home/lc999/model_training_ws"

    # 数据库
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_user: str = "postgres"
    db_password: str = ""
    db_name: str = "aistock"

    # 观测量选择
    obs_features: List[str] = field(default_factory=lambda: [
        "daily_return",          # sw2_pct_change / 100
        "excess_return_Nd",      # 超额收益 N 日均值（N=rolling_window）
        "volume_ratio",          # sw2_vol / 全市场 vol
        "limit_up_ratio",        # 行业内涨停占比（从 Qlib bin）
        "volatility_Nd",         # N 日波动率（N=rolling_window）
        "net_mf_ratio",          # sw2_mf_net_amt / sw2_amount（净资金流入占比）
        "elg_net_mf_ratio",      # 超大单净流入占比
    ])

    # 评估窗口
    eval_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])

    # 最小训练天数
    min_trading_days: int = 120
