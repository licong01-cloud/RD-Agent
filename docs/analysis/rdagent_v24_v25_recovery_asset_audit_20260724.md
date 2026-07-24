# RD-Agent V24/V25 历史恢复资产审计

日期：2026-07-24

## 1. 审计边界与基线

本次审计只处理 V24/V25 分钟执行、v4 的 5D/10D 派生表达、回测一致性诊断工具以及相关文档和测试。HMM 训练与系数、QE 多 Alpha 持久编排、Scheduler、因子指标平台、通用 v1/v2/V3/v5/v6 模板、工作流规范、数据库和运行时均不在范围内。

对比基线固定为：

- RD-Agent `origin/main`：`293c9d086bceb858f7b884d37b7d0365628334c7`；
- `recovery/rdagent-root-wip-20260721`：`a03f028ad55807bd264f87ceafc659b34959a08e`；
- `recovery/rdagent-root-snapshot-20260724`：`c1bef1035e2c4ee7c76b7f7973b378891f3b64ee`。

两个 recovery 引用仅通过 Git object 读取，未修改、移动、删除、reset 或 clean。下文的“采用”均指提取仍有效的语义后进入独立修复 PR，不表示原文件可以原样恢复，也不表示已经部署或正在运行。

## 2. V24/V25 策略与配置

| 历史资产 | 最新 main 对比 | 结论 |
| --- | --- | --- |
| `rdagent/scenarios/qlib/experiment/factor_template/conf_v25_example.yaml` | 仅 snapshot 存在；把执行策略放在非 `NestedExecutor.inner_strategy.kwargs` 层，并含当前 wrapper 不支持的参数和固定模型路径 | 放弃原文件。由 AIstock `test_qe_config_truth.py` 验证真实生成 YAML 的 inner strategy 和 kwargs |
| `rdagent/scenarios/qlib/experiment/factor_template/conf_v25_1_example.yaml` | 仅 snapshot 存在；与 V25 示例有相同层级问题 | 放弃原文件，使用配置真值测试 |
| `rdagent/scenarios/qlib/experiment/factor_template/tail_twap_v24_strategy.py` | WIP/snapshot 为 `2dadea9...`，main 的 v4 factor 副本为 `3b7053b...`；历史版本包含 plan/data/index 的均匀或 TWAP 降级，以及 adjusted/raw 价格混用 | 采用 V24 Plan Net 和 warmup/plan 骨架，修复后恢复；拒绝任何降级路径 |
| `rdagent/scenarios/qlib/experiment/model_template/tail_twap_v24_strategy.py` | main 缺失，snapshot 为 `2dadea9...` | 以修复后的 factor/model 同源副本恢复，并用 byte-identical 测试约束 |
| `app_tpl/all/v4/**/tail_twap_v24_strategy.py` | main 只含 factor 副本；snapshot 同时复制到 factor/model | 只恢复修复后的必需运行副本，不恢复旧 blob |
| `app_tpl/all/v4/**/tail_twap_strategy.py` | main 的 factor/model 与 runtime model 存在漂移；snapshot 的相关副本是旧逻辑 | 统一为同一修复源码：raw 价格、显式 suspend/limit 状态、严格 240/241 日历和可追踪尾盘替补 |
| `app_tpl/all/v4/**/score_weighted_strategy.py` | main、WIP 和 snapshot 的 v4 factor blob 均为 `6f56590...` | 分钟执行入口已核对，无需修改；HMM 系数契约不在本窗口 |
| `app_tpl/all/v4/**/score_weighted_strategy_v2.py` | snapshot blob `23b58ed...` 早于 main 的 `38c5528...` | 不回滚 current main；只核对分钟策略衔接，HMM 逻辑不认领 |
| `rdagent/scenarios/qlib/experiment/model_template/tail_twap_strategy.py` | main blob `2cb30ef...`，与 runtime factor 和 snapshot 的 `623a6d0...` 均不同 | 统一到修复后的权威父策略，不保留旧的缺数据跳过行为 |

明确拒绝 snapshot 中将 `TAIL_START_OFFSET` 或 `REALLOC_OFFSET` 减一的处理。241 长度只允许首位为可选 09:25 auction；去掉 auction 后，14:30 和 14:55 的 normalized index 仍分别为 `210` 和 `235`。

本批次存在有意的算法语义修正：

- V24 的 plan 失败、索引异常、缺数据不再切换到均匀/TWAP；
- Qlib adjusted OHLC 必须用 `$factor` 转换为 raw RMB 后，才能与 raw `prev_close`、涨跌停价比较；
- `suspend_d`、交易所停牌、涨停买入阻断、跌停卖出阻断是显式 no-fill；缺 `pre_close`、limit、有效分钟价、factor、模型或 plan 是错误；
- V25 不再使用全零 `day_features`，必须显式提供带 schema version 的特征 artifact；
- `TAIL_SUBSTITUTE` 保留 ranked candidate 切换和 top-k 容量，不再在候选为空或不可交易时静默改为 `TAIL_BOOST`；
- 订单、no-fill 和替补原因保存在策略 diagnostics 字典并写结构化日志。

## 3. v4-5d / v4-10d 派生模板

snapshot 的 `app_tpl/all/v4-5d/**` 和 `app_tpl/all/v4-10d/**` 各自复制完整 v4 树。逐文件 blob 对比显示，大多数源码、prompt、parquet 输入和运行辅助文件与 base v4 相同，不能将其作为约 108 个新源码文件重新合入。

真正的派生差异收敛为：

| 派生版本 | 标签表达 | 其他允许差异 |
| --- | --- | --- |
| `v4-5d` | `Ref($close, -6) / $close - 1` | variant/manifest version 与确定性文件 hash |
| `v4-10d` | `Ref($close, -11) / $close - 1` | variant/manifest version 与确定性文件 hash |

处理结论：

- 用 `variant.json` 声明 base `v4`、目标 label expression 和严格替换数量；
- 用版本 materializer 在 repo 外的新目录生成完整模板和 deterministic manifest；
- 删除 main 中 5D/10D 目录下 16 个零散重复执行文件，避免派生目录成为部分覆盖的隐式模板；
- `benchmark_sh000300.parquet` 继续从 base v4 继承，是模板运行输入，不加入全局 parquet ignore；
- 生成结果必须包含 base v4 的 V24/V25 执行资产；fixed-hold 标签只改变监督目标，不改变分钟执行状态机。

## 4. 文档、测试和 prompt 资产

| 历史资产 | 最新 main 对比 | 结论 |
| --- | --- | --- |
| `HOW_TO_USE_V25.md` | 仅 snapshot 根目录存在，包含旧模型路径和旧调用方式 | 不原样迁移；持久契约提炼到当前用户指南 |
| `REMOTE_SYNC_COMPLETE.md` | 仅 snapshot 根目录存在，是一次性远端同步完成通知 | 放弃，不将历史完成状态写入 main |
| `STRATEGY_UPDATE_GUIDE.md` | 仅 snapshot 根目录存在，包含过时替换/同步步骤 | 放弃；旧 workspace 必须显式选择复现或迁移，不允许静默覆盖 |
| `V25_DEPLOYMENT_COMPLETE.md` | 仅 snapshot 根目录存在，记录旧部署状态 | 放弃；部署和 runtime 必须现场核查 |
| `V25_INTEGRATION_REPORT.md` | 仅 snapshot 根目录存在，包含过时集成结论 | 只提炼两阶段和市场约束，不迁移完成声明 |
| `docs/analysis/qe_v25_strategy_issues_analysis_20260501.md` | 仅 snapshot 存在 | 提炼尾盘替补、候选切换和配置真实性；旧状态不迁移 |
| `docs/tasks/qe_v25_strategy_fix_tasks_20260501.md` | 仅 snapshot 存在 | 已完成/过期任务不恢复；有效验收项转为 contract tests |
| `test_v25_strategy.py` | 仅 snapshot 根目录存在，依赖本机路径/运行环境 | 不迁移；测试进入 `test/qlib/test_minute_execution_contract.py` |
| `test_v25_wsl.yaml` | 仅 snapshot 根目录存在，是临时 WSL 配置 | 不迁移；由生成 YAML 真值测试替代 |
| `rdagent/scenarios/qlib/prompts.yaml.backup_20260426_172603` | 仅 snapshot 存在；与 current prompts 有大量普通因子研发和反馈格式差异 | 不原样恢复。分钟成交约束和回测准确性要求已由 contract、测试及本指南覆盖；未发现必须从 backup 恢复的独有分钟条款 |

## 5. 历史诊断脚本分类

下列文件均只存在于 snapshot，current main 不含对应工具。审计发现它们没有稳定 CLI，并硬编码了本机/WSL/远端路径、主机、实验 ID 或 repo 根输出中的至少一项。

| 文件 | 分类与结论 |
| --- | --- |
| `scripts/check_300573.py` | 单股票一次性定位；留在 recovery |
| `scripts/check_limit_accuracy.py` | 固定数据路径的限价核对；能力由通用 artifact compare 覆盖 |
| `scripts/check_price_accuracy.py` | 固定数据路径的价格核对；能力由通用 artifact compare 覆盖 |
| `scripts/check_remote_limits.py` | 远端 SSH/主机绑定；留在 recovery |
| `scripts/compare_data.py` | 硬编码数据与输出目录；由参数化 compare CLI 取代 |
| `scripts/compare_deal_prices.py` | 固定实验成交价对比；由参数化 compare CLI 取代 |
| `scripts/deep_compare_prev_close.py` | 一次性 prev-close 深挖；留在 recovery |
| `scripts/deep_diagnose_divergence.py` | 固定实验 divergence 排查；留在 recovery |
| `scripts/deterministic_compare.py` | 可复现比较思路有价值，但实现绑定实验；提炼到通用 compare CLI |
| `scripts/deterministic_same_pred.py` | 固定 prediction 实验；留在 recovery |
| `scripts/final_verify.py` | 历史批次聚合验证；留在 recovery |
| `scripts/pred_swap_test.py` | 固定预测互换实验；留在 recovery |
| `scripts/swap_test.py` | 一次性互换实验；留在 recovery |
| `scripts/swap_test_v2.py` | 一次性互换实验；留在 recovery |
| `scripts/test_deal_prices.py` | 根脚本式临时测试；不迁移 |
| `scripts/test_model_transfer.py` | 固定模型资产测试；受保护资产不在本窗口 |
| `scripts/trace_day1_divergence.py` | 固定实验首日追踪；留在 recovery |
| `scripts/verify_day_data.py` | 固定日数据核对；留在 recovery |
| `scripts/verify_qlib_reads.py` | 固定 Qlib 环境 smoke；不作为通用测试迁移 |
| `scripts/verify_v4_backtest.py` | 固定 v4 实验回测；留在 recovery |
| `scripts/verify_v4_compare.py` | 固定 v4 目录对比；由参数化 compare CLI 取代 |
| `scripts/_tmp/analyze_20241231.py` | 一次性日期分析；留在 recovery |
| `scripts/_tmp/check_1min_indicators.py` | 临时指标检查；留在 recovery |
| `scripts/_tmp/fix_backup_candidates.py` | 带修复语义的一次性脚本；禁止迁移 |
| `scripts/_tmp/reconstruct_loop5.py` | 固定 Loop 重建；QE workspace 不在本窗口 |
| `scripts/_tmp/verify_backup_candidates.py` | 一次性候选核对；留在 recovery |

可复用能力收敛为 `scripts/compare_backtest_artifacts.py`：输入路径、复合 key、数值列、容差和可选输出目录均由 CLI 提供；输出目录必须位于调用者指定的外部 artifact 位置，不再向 repo 根写 `mlruns_compare`、`mlruns_verify` 或 `compare_result`。退出码 `0/1/2` 分别表示一致、存在差异和输入错误，JSON 使用 UTF-8。

## 6. 验证与仍需主窗口处理的事项

本恢复批次的测试按 PR 边界分别记录，不能把未运行的真实 Qlib/模型回测写成通过。Windows 环境缺少 Qlib/Torch 时，只能确认 pure-contract、source contract、配置生成和 Python 编译；真实模型加载、historical replay、Paper v2/QE 同输入结果对账必须在具备权威依赖和 artifact 的环境执行。

仍需 QE 平台主窗口处理：

1. 定义并生产 `day_features_file` 的权威 schema、PIT 来源、hash、workspace 注入和恢复迁移 receipt；本窗口只实现 fail-closed consumer 和配置透传。
2. 对真实 240/241 Qlib 数据运行 V24/V25 回测，并与 Paper v2 的相同 MarketContext/原因码/plan 对账。
3. 旧 QE workspace 不自动替换策略文件；恢复时必须明确选择“旧文件复现”或“注入当前修复模板并记录迁移”。
4. 模型权重、StrategyPackage、QE workspace、数据库、部署和 runtime 均未修改；其当前可用性必须单独现场核查。

