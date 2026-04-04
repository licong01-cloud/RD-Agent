# 模型训练目录

统一的模型训练源代码和脚本目录，位于 RD-Agent 项目下。
所有训练脚本在 WSL 环境中执行，workspace 使用 WSL 本地文件系统。

## 目录结构

```
RD-Agent-main/model_training/     ← 源代码（git 跟踪）
├── hmm/                           ← 行业 HMM 模型
│   ├── train_sector_hmm.py        ← 训练主脚本（含验证）
│   └── config.py                  ← 训练配置
├── common/                        ← 通用工具
│   ├── data_loader.py             ← 数据加载（DB + Qlib bin）
│   └── registry.py                ← 模型注册到 DB
└── README.md

/home/lc999/model_training_ws/     ← WSL 本地 workspace（git 忽略）
├── hmm/{config_id}/{date}/        ← HMM 模型产出
├── logs/                          ← 训练日志
└── tmp/                           ← 临时文件
```

## 数据源

- **行业数据**: `market.sector_data` JOIN `market.sw_index_member`（L2 行业级别）
- **涨停数据**: Qlib bin 文件 `/home/lc999/data/qlib_bin/features/*/limit_up.day.bin`
- **CSI300 基准**: `market.index_daily`
- **资金流**: `sector_data` 表中的 `sw2_mf_*` 字段

## WSL 执行

```bash
cd /mnt/f/Dev/RD-Agent-main
conda activate rdagent-gpu
python -m model_training.hmm.train_sector_hmm --config-json '{"n_states":2}'
```

## API 调度

AIstock 后端通过 `HMMTrainingService.run_training()` 调用
`subprocess.run(["wsl", "python", "-m", "model_training.hmm.train_sector_hmm", ...])` 执行。
Workspace 路径: `/home/lc999/model_training_ws/`
