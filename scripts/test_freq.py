"""Check Exchange.freq value in current qlib version"""
import qlib
qlib.init(
    provider_uri={'day': '/home/lc999/data/qlib_bin', '1min': '/home/lc999/data/qlib_minute_bin'},
    region='cn', dataset_cache=None, expression_cache=None,
)

from qlib.backtest.exchange import Exchange
from qlib.data import D

# Create a minimal Exchange with freq=1min to check what self.freq becomes
codes = D.instruments('all')
ex = Exchange(
    freq='1min',
    start_time='2024-07-01',
    end_time='2024-07-02',
    codes=['000001.SZ'],
    deal_price='close',
    limit_threshold=None,
    open_cost=0.0001,
    close_cost=0.0006,
    min_cost=5,
)
print(f"Exchange.freq = {repr(ex.freq)}")
print(f"Exchange.freq type = {type(ex.freq)}")
print(f"Exchange.freq == '1min': {ex.freq == '1min'}")
print(f"Exchange.freq in ('1min', '5min'): {ex.freq in ('1min', '5min')}")
