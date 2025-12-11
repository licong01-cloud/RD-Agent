import os
from pathlib import Path

import numpy as np
import pandas as pd


def _to_unix_path(p: Path) -> Path:
    # 简单的 Windows → WSL 路径转换，避免在 WSL 内写到本地 C:\ 根目录
    s = str(p)
    if os.name != "nt" and s[1:3] == ":/":  # e.g. C:/...
        drive = s[0].lower()
        return Path("/mnt") / drive / s[3:]
    return p


def main() -> None:
    """预计算个股资金流向因子表 moneyflow_factors.

    输入：
      - moneyflow.h5: 含 net_mf_amount/net_mf_vol/buy_elg_*/sell_elg_* 等字段
      - daily_pv.h5:  含 amount/volume 等基础成交字段

    输出：
      - moneyflow_factors/result.pkl
        索引: MultiIndex(datetime, instrument)
        列: 一组以 mf_* 开头的资金流因子

    默认路径可以按需修改，也可以在后续任务中参数化.
    """

    # 默认使用 20251209 这期 snapshot，可按需调整
    snapshot_root = Path("C:/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_20251209")
    moneyflow_path = snapshot_root / "moneyflow.h5"
    daily_pv_path = snapshot_root / "daily_pv.h5"

    # 因子输出目录
    factors_root = Path("C:/Users/lc999/NewAIstock/AIstock/factors/moneyflow_factors")
    output_path = factors_root / "result.pkl"

    moneyflow_path = _to_unix_path(moneyflow_path)
    daily_pv_path = _to_unix_path(daily_pv_path)
    output_path = _to_unix_path(output_path)

    print("[信息] moneyflow 源:", moneyflow_path)
    print("[信息] daily_pv 源:", daily_pv_path)
    print("[信息] 输出因子表:", output_path)

    if not moneyflow_path.exists():
        raise FileNotFoundError(f"moneyflow.h5 不存在: {moneyflow_path}")
    if not daily_pv_path.exists():
        raise FileNotFoundError(f"daily_pv.h5 不存在: {daily_pv_path}")

    print("[信息] 载入 moneyflow.h5 ...")
    df_mf = pd.read_hdf(moneyflow_path)
    print("  moneyflow 形状:", df_mf.shape)

    print("[信息] 载入 daily_pv.h5 ...")
    df_pv = pd.read_hdf(daily_pv_path)
    print("  daily_pv 形状:", df_pv.shape)

    # 确保两边都是 MultiIndex(datetime, instrument)
    for name, df in [("moneyflow", df_mf), ("daily_pv", df_pv)]:
        if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
            raise ValueError(f"{name} 索引不是 2 层 MultiIndex(datetime, instrument)，当前: {type(df.index)}, nlevels={getattr(df.index, 'nlevels', None)}")
        if list(df.index.names) != ["datetime", "instrument"]:
            df.index.set_names(["datetime", "instrument"], inplace=True)

    # 将 moneyflow 的 instrument 从 'SH600000'/'SZ000001' 风格映射为 '600000.SH'/'000001.SZ'，以便与 daily_pv 对齐
    inst_mf = df_mf.index.get_level_values("instrument")
    if any(code.startswith(("SH", "SZ")) for code in inst_mf[:10]):
        def _map_inst(code: str) -> str:
            if code.startswith("SH") or code.startswith("SZ"):
                market = code[:2]
                num = code[2:]
                suffix = ".SH" if market == "SH" else ".SZ"
                return f"{num}.{suffix}"
            return code

        mapped = [
            _map_inst(c) for c in inst_mf
        ]
        df_mf.index = pd.MultiIndex.from_arrays(
            [df_mf.index.get_level_values("datetime"), mapped],
            names=["datetime", "instrument"],
        )

    # 仅保留因子计算所需列
    needed_mf_cols = [
        "mf_lg_buy_vol", "mf_lg_buy_amt", "mf_lg_sell_vol", "mf_lg_sell_amt",
        "mf_elg_buy_vol", "mf_elg_buy_amt", "mf_elg_sell_vol", "mf_elg_sell_amt",
        "mf_net_vol", "mf_net_amt",
    ]
    needed_pv_cols_sets = [["$amount", "$volume"], ["amount", "volume"]]

    missing_mf = [c for c in needed_mf_cols if c not in df_mf.columns]
    if missing_mf:
        raise KeyError(f"moneyflow 中缺少列: {missing_mf}")

    # 兼容 daily_pv 列名：优先使用带 $ 的命名，否则退回无 $ 命名
    chosen_pv_cols = None
    for cand in needed_pv_cols_sets:
        if all(c in df_pv.columns for c in cand):
            chosen_pv_cols = cand
            break
    if chosen_pv_cols is None:
        raise KeyError(f"daily_pv 中既不存在 ['$amount','$volume']，也不存在 ['amount','volume']，当前列: {list(df_pv.columns)}")

    df_mf = df_mf[needed_mf_cols]
    df_pv = df_pv[chosen_pv_cols]

    print("[信息] 按 (datetime, instrument) 对齐 moneyflow 与 daily_pv ...")
    df = df_mf.join(df_pv, how="inner")
    df = df.sort_index()
    print("  合并后形状:", df.shape)

    # 避免 0 除
    # 根据前面选择的列名，统一取得成交额/量
    amt_col, vol_col = chosen_pv_cols
    amount = df[amt_col].replace(0, np.nan)
    volume = df[vol_col].replace(0, np.nan)

    # 总体资金净流入：mf_net_amt / mf_net_vol 已由 qlib 导出直接给出
    total_net_mf_amt = df["mf_net_amt"]  # 净流入金额（买入 - 卖出），单位：元
    total_net_mf_vol = df["mf_net_vol"]  # 净流入量（买入 - 卖出），单位：股/手

    # 金额占成交额比例：表示每日资金净流入占成交额的比重
    total_net_mf_amt_ratio = total_net_mf_amt / (amount.replace(0, pd.NA))
    # 成交量占比：资金净流入对应的成交量占总成交量的比例
    total_net_mf_vol_ratio = total_net_mf_vol / (volume.replace(0, pd.NA))

    # 主力（大单+超大单）净流入：代表“主力资金”行为
    lg_buy_amt = df["mf_lg_buy_amt"]
    lg_sell_amt = df["mf_lg_sell_amt"]
    elg_buy_amt = df["mf_elg_buy_amt"]
    elg_sell_amt = df["mf_elg_sell_amt"]

    lg_buy_vol = df["mf_lg_buy_vol"]
    lg_sell_vol = df["mf_lg_sell_vol"]
    elg_buy_vol = df["mf_elg_buy_vol"]
    elg_sell_vol = df["mf_elg_sell_vol"]

    # 主力（大单+超大单）净流入金额/成交量
    main_net_mf_amt = (lg_buy_amt + elg_buy_amt) - (lg_sell_amt + elg_sell_amt)
    main_net_mf_vol = (lg_buy_vol + elg_buy_vol) - (lg_sell_vol + elg_sell_vol)

    main_net_mf_amt_ratio = main_net_mf_amt / (amount.replace(0, pd.NA))
    main_net_mf_vol_ratio = main_net_mf_vol / (volume.replace(0, pd.NA))

    # 组装结果表
    df_factors = pd.DataFrame(index=df.index)
    # 总体净流入水平
    df_factors["mf_total_net_amt"] = total_net_mf_amt.astype("float64")
    df_factors["mf_total_net_vol"] = total_net_mf_vol.astype("float64")
    df_factors["mf_total_net_amt_ratio"] = total_net_mf_amt_ratio.astype("float64")
    df_factors["mf_total_net_vol_ratio"] = total_net_mf_vol_ratio.astype("float64")
    # 主力资金（大单+超大单）净流入
    df_factors["mf_main_net_amt"] = main_net_mf_amt.astype("float64")
    df_factors["mf_main_net_vol"] = main_net_mf_vol.astype("float64")
    df_factors["mf_main_net_amt_ratio"] = main_net_mf_amt_ratio.astype("float64")
    df_factors["mf_main_net_vol_ratio"] = main_net_mf_vol_ratio.astype("float64")

    # === 超大单(elg)优先刻画 ===
    # 超大单自身的净流入
    elg_net_mf_amt = elg_buy_amt - elg_sell_amt
    elg_net_mf_vol = elg_buy_vol - elg_sell_vol

    df_factors["mf_elg_net_amt"] = elg_net_mf_amt.astype("float64")
    df_factors["mf_elg_net_vol"] = elg_net_mf_vol.astype("float64")
    df_factors["mf_elg_net_amt_ratio"] = (elg_net_mf_amt / amount).astype("float64")
    df_factors["mf_elg_net_vol_ratio"] = (elg_net_mf_vol / volume).astype("float64")

    # 超大单在“主力净流入”中的占比（体现优先级高于大单）
    # 使用安全除法，防止主力净流入为 0 时异常
    main_amt_safe = main_net_mf_amt.replace(0, np.nan)
    main_vol_safe = main_net_mf_vol.replace(0, np.nan)
    df_factors["mf_elg_share_in_main_amt"] = (elg_net_mf_amt / main_amt_safe).astype("float64")
    df_factors["mf_elg_share_in_main_vol"] = (elg_net_mf_vol / main_vol_safe).astype("float64")

    # 当前版本输出 8 个基础净流入因子 + 6 个超大单优先相关因子，不做额外滚动窗口聚合，防止引入不必要的复杂度
    print("[信息] 最终资金流因子表形状:", df_factors.shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_factors.to_pickle(output_path)
    print("[成功] 已写出资金流因子表:", output_path)


if __name__ == "__main__":
    main()
