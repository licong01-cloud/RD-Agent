"""完整分钟线回测验证：内存 patch + benchmark 注入 + 指标验证。

在独立目录运行，使用 Alpha158 子集 + 简单 GRU 模型，
验证：
1. 内存占用 < 4GB (无 swap thrashing)
2. 回测指标非 NaN (benchmark 注入生效)
3. IC 等信号指标正常
"""
import gc
import os
import sys
import time
import shutil
import threading
from pathlib import Path

# ── 配置 ──
TEST_DIR = Path("/tmp/minute_backtest_test")
QLIB_DAY = "/home/lc999/data/qlib_bin"
QLIB_MIN = "/home/lc999/data/qlib_minute_bin"

# 使用较短的回测区间 (30 天) 加速验证
CONF_YAML = """
qlib_init:
    provider_uri:
        day: "{qlib_day}"
        1min: "{qlib_min}"
    region: cn
    dataset_cache: null
    expression_cache: null

market: &market all
benchmark: &benchmark 000300.SH

data_handler_config: &data_handler_config
    start_time: 2024-01-01
    end_time: 2025-06-30
    instruments: *market
    data_loader:
        class: qlib.contrib.data.loader.Alpha158DL
        kwargs:
            config:
                label:
                    - ["Ref($close, -2)/Ref($close, -1) - 1"]
                    - ["LABEL0"]
                feature:
                    - ["Resi($close, 5)/$close", "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
                       "Rsquare($close, 5)", "($high-$low)/$open", "Rsquare($close, 10)", "Corr($close, Log($volume+1), 5)",
                       "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)", "Corr($close, Log($volume+1), 10)",
                       "Ref($close, 60)/$close", "Resi($close, 10)/$close"]
                    - ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", "ROC60", "RESI10"]
    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
              fit_start_time: 2024-01-01
              fit_end_time: 2025-03-31
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSZScoreNorm
          kwargs:
              fields_group: label

port_analysis_config: &port_analysis_config
    executor:
        class: NestedExecutor
        module_path: qlib.backtest.executor
        kwargs:
            time_per_step: day
            inner_executor:
                class: SimulatorExecutor
                module_path: qlib.backtest.executor
                kwargs:
                    time_per_step: 1min
                    generate_portfolio_metrics: false
            inner_strategy:
                class: TWAPStrategy
                module_path: qlib.contrib.strategy.rule_strategy
            generate_portfolio_metrics: true
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 30
            n_drop: 3
    backtest:
        start_time: 2025-04-01
        end_time: 2025-05-30
        account: 100000000
        benchmark: ~
        exchange_kwargs:
            freq: 1min
            limit_threshold: ["$limit_up", "$limit_down"]
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5
            trade_unit: 100

task:
    model:
        class: GeneralPTNN
        module_path: qlib.contrib.model.pytorch_general_nn
        kwargs:
            n_epochs: 5
            lr: 0.0003
            early_stop: 3
            batch_size: 4096
            weight_decay: 0.00001
            metric: loss
            loss: mse
            n_jobs: 4
            GPU: 0
            pt_model_uri: "model.model_cls"
            pt_model_kwargs:
                num_features: 10
                num_timesteps: 20
    dataset:
        class: TSDatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: DataHandlerLP
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2024-01-01, 2025-03-31]
                valid: [2025-04-01, 2025-04-30]
                test: [2025-05-01, 2025-05-30]
            step_len: 20
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            model: <MODEL>
            dataset: <DATASET>
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            config: *port_analysis_config
""".format(qlib_day=QLIB_DAY, qlib_min=QLIB_MIN)

# ── 简单 GRU 模型 ──
MODEL_PY = '''
import torch
import torch.nn as nn


class model_cls(nn.Module):
    """Simple GRU model for testing."""
    def __init__(self, num_features=10, num_timesteps=20, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, timesteps, features)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
'''


def memory_monitor(stop_event, peak_rss):
    """后台线程监控 RSS 内存峰值 (MB)。"""
    while not stop_event.is_set():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        rss_mb = rss_kb / 1024
                        if rss_mb > peak_rss[0]:
                            peak_rss[0] = rss_mb
                        break
        except Exception:
            pass
        time.sleep(1)


def main():
    # 准备测试目录
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)

    # 写入配置文件
    (TEST_DIR / "conf.yaml").write_text(CONF_YAML)
    (TEST_DIR / "model.py").write_text(MODEL_PY)

    # 复制 qrun_limit_minute.py 和 benchmark
    src_dir = Path("/mnt/f/Dev/RD-Agent-main/rdagent/scenarios/qlib/experiment/model_template")
    shutil.copy2(src_dir / "qrun_limit_minute.py", TEST_DIR / "qrun_limit_minute.py")
    shutil.copy2(src_dir / "benchmark_sh000300.parquet", TEST_DIR / "benchmark_sh000300.parquet")

    os.chdir(TEST_DIR)

    # 启动内存监控
    stop_event = threading.Event()
    peak_rss = [0.0]
    monitor = threading.Thread(target=memory_monitor, args=(stop_event, peak_rss), daemon=True)
    monitor.start()

    print("=" * 60)
    print("MINUTE BACKTEST VERIFICATION TEST")
    print(f"Working dir: {TEST_DIR}")
    print(f"Backtest: 2025-04-01 ~ 2025-05-30 (~40 trading days)")
    print("=" * 60)

    t0 = time.time()

    # 导入并执行 qrun_limit_minute 的逻辑
    sys.path.insert(0, str(TEST_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location("qlm", str(TEST_DIR / "qrun_limit_minute.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 手动执行 main 逻辑
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load((TEST_DIR / "conf.yaml").read_text())

    mod.patch_backtest_config(config)

    from qlib.workflow.cli import sys_config, task_train
    import qlib
    from qlib.config import C

    sys_config(config, config_path=str(TEST_DIR / "conf.yaml"))

    tracking_uri = str(TEST_DIR / "mlruns")
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + tracking_uri
    qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    # 注入 benchmark
    benchmark_series = mod.load_benchmark_series(config)
    mod.inject_benchmark(config, benchmark_series)

    # 应用内存 patch
    mod.apply_minute_memory_patch()

    print(f"\n[{time.time()-t0:.1f}s] Starting training + backtest...")

    # 执行训练和回测
    recorder = task_train(config.get("task"), experiment_name="minute_test")

    elapsed = time.time() - t0

    # 停止内存监控
    stop_event.set()
    monitor.join(timeout=2)

    print(f"\n{'=' * 60}")
    print(f"EXECUTION COMPLETED in {elapsed:.1f}s")
    print(f"Peak RSS: {peak_rss[0]:.0f} MB")
    print(f"{'=' * 60}")

    # 读取结果
    print("\n── Metrics ──")
    try:
        metrics = recorder.list_metrics()
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

        # 验证关键指标
        ic = metrics.get("IC", None)
        ann_ret = metrics.get("excess_return_annualized", None) or metrics.get("return_annualized", None)

        print("\n── Validation ──")
        if ic is not None and not (ic != ic):  # not NaN
            print(f"  IC = {ic:.4f} ✓ (non-NaN)")
        else:
            print(f"  IC = {ic} ✗ (NaN or missing)")

        # 检查是否有 NaN 的回测指标
        nan_metrics = [k for k, v in metrics.items() if isinstance(v, float) and v != v]
        if nan_metrics:
            print(f"  NaN metrics: {nan_metrics} ✗")
        else:
            print(f"  All metrics non-NaN ✓")

        if peak_rss[0] < 4096:
            print(f"  Peak memory: {peak_rss[0]:.0f} MB < 4GB ✓")
        else:
            print(f"  Peak memory: {peak_rss[0]:.0f} MB >= 4GB ✗")

    except Exception as e:
        print(f"  Error reading metrics: {e}")

    # 检查 swap 使用
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSwap:"):
                    swap_kb = int(line.split()[1])
                    print(f"  Current swap: {swap_kb/1024:.0f} MB {'✓' if swap_kb < 1024*1024 else '✗'}")
                    break
    except:
        pass


if __name__ == "__main__":
    main()
