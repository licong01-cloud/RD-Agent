# QE V24/V25 分钟执行使用与恢复契约

本文描述 RD-Agent/Qlib 模板与 AIstock QE/Paper v2 之间必须保持一致的分钟执行语义。它是源码使用说明，不是部署完成、模型可用或 runtime 在线证明。

## 1. 单一执行语义

V24/V25 的 Qlib adapter 必须与 AIstock 分钟执行标准保持同一逻辑契约：

- Qlib `$open/$close/$high/$low` 是 adjusted price；先用 `raw = adjusted / $factor` 转为未复权人民币价格，再与 raw `$prev_close/$up_limit_price/$down_limit_price` 比较；
- `suspended_by_suspend_d`、`suspended_by_exchange`、`limit_up_buy_blocked`、`limit_down_sell_blocked` 和 `intraday_halt_or_no_bar` 是显式 no-fill/等待市场状态；
- 无停牌证据时缺 `pre_close`、涨跌停价、分钟线或有效 `$factor`，以及模型/plan/day features 缺失，必须 fail-fast；
- 禁止回退到另一个 TWAP、收盘价、日线、默认价格、默认仓位、空订单伪成功或均匀 plan；
- 所有策略副本必须使用同一 `minute_execution_contract.py`，并由 hash/byte-identical 测试约束。

Paper v2 本批次不改源码。它继续使用自己的权威 MarketContext/provider/persistence adapter；所谓“一致”指市场状态、价格基准、plan 和 fail-fast 合同一致，不表示两个框架共享 Qlib 对象或模型文件。

## 2. V24 与 V25 状态机

### V24

V24 保留 30 分钟 warmup 和后续 210 分钟 Plan Net 执行。warmup 中的权威市场阻断产生显式 no-plan/no-fill；它不能触发替代算法。plan 长度、索引、值或剩余权重不合法时直接失败。

### V25

V25 的 normalized 240 分钟网格严格划分为：

- `EARLY`：index `0..29`，权重总和 `0.8879`；
- `LATE`：index `30..239`，权重总和 `0.1121`；
- 尾盘起点：14:30，index `210`；
- 替补/再分配检查：14:55，index `235`。

长度 241 只表示首位还有 09:25 auction。auction step 返回等待/无订单，之后减一映射到相同 240 分钟网格；不能把 14:30/14:55 常量整体减一。

V25 模型输入禁止用全零 day features。调用方必须在 `NestedExecutor.inner_strategy.kwargs` 中提供：

```yaml
day_features_file: qe_v25_day_features.json
day_features_schema_version: qe_v25_model_inputs_v1
```

当前 consumer 接受的 JSON 形状为：

```json
{
  "schema_version": "qe_v25_model_inputs_v1",
  "features_by_date": {
    "2026-07-24": {
      "600000.SH": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
  }
}
```

缺文件、schema 漂移、缺日期/股票、不是 10 维或含非有限数值均为明确错误。artifact 的权威生产、PIT lineage、hash 和 workspace receipt 仍须由 QE 平台侧补齐，不能由策略 adapter 猜测。

## 3. 尾盘未成交与候选切换

`TAIL_BOOST` 和 `TAIL_SUBSTITUTE` 是显式配置的两种业务策略，不是互为异常 fallback：

- `TAIL_BOOST` 只按其既定持仓再分配语义执行；
- `TAIL_SUBSTITUTE` 按 score 排序跳过已有持仓、已加入候选和不可交易候选，并继续切换到下一只可交易股票；
- candidate 为空、top-k 容量耗尽或候选均不可交易时记录对应 no-fill reason，不得偷偷切换成 `TAIL_BOOST`；
- 缺权威 `topk` 或 trade position 是配置/状态错误，不得填默认值。

市场约束导致的零成交与数据错误必须分开：前者保留订单剩余量和 reason，后者终止当前运行。`WAIT` 只用于实时尚无新 bar 或 09:25 auction 等确切等待状态。

## 4. QE 配置真实性

有效配置必须把执行类放在 Qlib `NestedExecutor` 的 `inner_strategy` 中，所有算法参数放在其 `kwargs`：

```yaml
executor:
  class: NestedExecutor
  module_path: qlib.backtest.executor
  kwargs:
    inner_strategy:
      class: TailTWAPWithV25TwoStageStrategy
      module_path: tail_twap_v25_strategy
      kwargs:
        filter_suspended_on_signal: true
        suspend_filter_file: qe_suspend_filter.json
        suspend_filter_strict: true
        day_features_file: qe_v25_day_features.json
        day_features_schema_version: qe_v25_model_inputs_v1
```

根目录历史 `conf_v25*_example.yaml` 的非 NestedExecutor 层级不是兼容配置，不能用于证明算法真实执行。以 AIstock ConfigComposer 生成结果和 `test_qe_config_truth.py` 为准。

## 5. fixed 5D/10D 模板

5D/10D 只改变监督标签期限，不改变分钟执行算法：

- 5D：`Ref($close, -6) / $close - 1`；
- 10D：`Ref($close, -11) / $close - 1`。

使用 `scripts/materialize_template_variant.py` 从 base v4 生成 repo 外的独立目录，例如：

```powershell
python scripts/materialize_template_variant.py `
  --spec app_tpl/all/v4-5d/variant.json `
  --output-dir F:\artifacts\rdagent-v4-5d
```

工具拒绝 repo 内输出和覆盖已有目录，严格检查标签替换数量，并为完整输出写 deterministic manifest。`benchmark_sh000300.parquet` 是从 base v4 继承的运行输入，不能用全局 `*.parquet` ignore 隐藏。

## 6. 历史 workspace 恢复

模板提交不会改写已存在的 QE workspace。恢复旧实验时只能显式选择：

1. 使用 workspace 中的旧策略文件复现实验，并保留旧文件 hash；
2. 注入当前修复模板，记录新 hash、配置、artifact 和“恢复迁移”身份后重新运行。

禁止在恢复流程中静默替换策略，禁止修改模型权重、StrategyPackage、validated policy、数据库资产或既有回测结果。源码合入也不代表依赖、部署、runtime 或模型已经可用；这些状态必须在目标环境独立核查。
