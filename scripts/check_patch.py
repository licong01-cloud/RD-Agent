#!/usr/bin/env python3
"""Check if the minute memory patch would be applied correctly."""
import sys
sys.path.insert(0, '/mnt/f/Dev/RD-Agent-main')

# Simulate what qrun_limit_minute.py does
import qlib
qlib.init(
    provider_uri={'day': '/home/lc999/data/qlib_bin', '1min': '/home/lc999/data/qlib_minute_bin'},
    region='cn',
    dataset_cache=None,
    expression_cache=None,
)

from qlib.backtest.exchange import Exchange
print(f"Before patch: Exchange.get_quote_from_qlib.__name__ = {Exchange.get_quote_from_qlib.__name__}")

# Now apply the patch
import importlib.util
spec = importlib.util.spec_from_file_location(
    'qlm', '/mnt/f/Dev/RD-Agent-main/rdagent/scenarios/qlib/experiment/model_template/qrun_limit_minute.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.apply_minute_memory_patch()

print(f"After patch: Exchange.get_quote_from_qlib.__name__ = {Exchange.get_quote_from_qlib.__name__}")

from qlib.backtest.executor import NestedExecutor
print(f"NestedExecutor._init_sub_trading.__name__ = {NestedExecutor._init_sub_trading.__name__}")

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
print(f"TopkDropoutStrategy.generate_trade_decision.__name__ = {TopkDropoutStrategy.generate_trade_decision.__name__}")

# Check Qlib mem cache config
from qlib.config import C
print(f"mem_cache_size_limit = {C.mem_cache_size_limit}")
print(f"mem_cache_limit_type = {C.mem_cache_limit_type}")
