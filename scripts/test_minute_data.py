"""测试分钟线按天加载的数据一致性。"""
import qlib
from qlib.data import D
import pandas as pd

qlib.init(provider_uri={'day': '/home/lc999/data/qlib_bin', '1min': '/home/lc999/data/qlib_minute_bin'}, region='cn')

# 测试: end_time 需要 +1 天才能获取当天的 1min 数据
fields = ["$close", "$open", "$high", "$low", "$volume"]

# 方案1: end_time 比 start_time 多一天
df1 = D.features(['000001.SZ'], fields, start_time='2024-07-01', end_time='2024-07-02', freq='1min', disk_cache=False)
print(f'[+1 day] shape={df1.shape}')
if not df1.empty:
    dates = df1.index.get_level_values(1).normalize().unique()
    print(f'  Unique dates: {dates.tolist()}')
    print(f'  First bar: {df1.index.get_level_values(1).min()}, Last: {df1.index.get_level_values(1).max()}')

# 方案2: 使用 23:59:59
df2 = D.features(['000001.SZ'], fields, start_time='2024-07-01', end_time='2024-07-01 23:59:59', freq='1min', disk_cache=False)
print(f'\n[23:59:59] shape={df2.shape}')
if not df2.empty:
    print(f'  First bar: {df2.index.get_level_values(1).min()}, Last: {df2.index.get_level_values(1).max()}')

# 方案3: 使用 15:00:00
df3 = D.features(['000001.SZ'], fields, start_time='2024-07-01', end_time='2024-07-01 15:00:00', freq='1min', disk_cache=False)
print(f'\n[15:00:00] shape={df3.shape}')
if not df3.empty:
    print(f'  First bar: {df3.index.get_level_values(1).min()}, Last: {df3.index.get_level_values(1).max()}')

# 验证数据正确性: 多只股票
codes = ['000001.SZ', '600000.SH', '000002.SZ']
df_multi = D.features(codes, fields, start_time='2024-07-01', end_time='2024-07-01 15:00:00', freq='1min', disk_cache=False)
print(f'\nMulti-stock: shape={df_multi.shape}')
if not df_multi.empty:
    for code in codes:
        try:
            sub = df_multi.loc[code.lower()]
            print(f'  {code}: {len(sub)} bars')
        except KeyError:
            try:
                sub = df_multi.loc[code]
                print(f'  {code} (uppercase): {len(sub)} bars')
            except KeyError:
                print(f'  {code}: NOT FOUND')
