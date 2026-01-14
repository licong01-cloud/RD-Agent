import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: Path, **read_csv_kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **read_csv_kwargs)


def build_round_trips(trading_events: pd.DataFrame) -> pd.DataFrame:
    """基于 trading_events 构造每个股票的完整持仓回合（从持仓为 0 -> >0 -> 回到 0）。

    要求 trading_events 包含列：
      - date: datetime64
      - stock_id: str
      - action: BUY/SELL
      - amount_change: 成交股数（正数）
      - trade_value_no_cost: 成交金额（正数）
    """
    df = trading_events.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date", "action"], ascending=[True, True, True])

    rounds = []

    for stock_id, g in df.groupby("stock_id", sort=False):
        pos = 0.0
        round_id = 0
        # 当前回合信息
        buy_value = 0.0
        sell_value = 0.0
        buy_shares = 0.0
        sell_shares = 0.0
        entry_date = None
        exit_date = None

        for _, row in g.iterrows():
            date = row["date"]
            action = row["action"]
            amt = float(row["amount_change"])
            value = float(row["trade_value_no_cost"])

            if action == "BUY":
                if pos == 0:
                    # 新开一个回合
                    round_id += 1
                    buy_value = 0.0
                    sell_value = 0.0
                    buy_shares = 0.0
                    sell_shares = 0.0
                    entry_date = date
                    exit_date = None
                pos += amt
                buy_value += value
                buy_shares += amt
            elif action == "SELL":
                pos -= amt
                sell_value += value
                sell_shares += amt
                exit_date = date

                if pos < 1e-6:
                    # 视为本回合平仓完成
                    if buy_value > 0:
                        pnl = sell_value - buy_value
                        ret = pnl / buy_value
                    else:
                        pnl = 0.0
                        ret = np.nan

                    holding_days = None
                    if entry_date is not None and exit_date is not None:
                        holding_days = (exit_date - entry_date).days + 1

                    rounds.append(
                        {
                            "stock_id": stock_id,
                            "round_id": round_id,
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                            "buy_value": buy_value,
                            "sell_value": sell_value,
                            "buy_shares": buy_shares,
                            "sell_shares": sell_shares,
                            "pnl": pnl,
                            "return": ret,
                            "holding_days": holding_days,
                        }
                    )
                    # 重置
                    pos = 0.0
                    buy_value = 0.0
                    sell_value = 0.0
                    buy_shares = 0.0
                    sell_shares = 0.0
                    entry_date = None
                    exit_date = None

        # 如果最后还有未平仓的持仓，本轮忽略（无完整回合）

    return pd.DataFrame(rounds)


def summarize_round_trips(round_trips: pd.DataFrame) -> dict:
    if round_trips.empty:
        return {
            "total_rounds": 0,
            "win_rounds": 0,
            "loss_rounds": 0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
            "return_quantiles": {},
        }

    df = round_trips.copy()
    df = df[df["return"].notna()]

    total_rounds = len(df)
    win_rounds = int((df["pnl"] > 0).sum())
    loss_rounds = int((df["pnl"] <= 0).sum())
    win_rate = win_rounds / total_rounds if total_rounds > 0 else None

    avg_return = float(df["return"].mean()) if not df["return"].empty else None
    median_return = float(df["return"].median()) if not df["return"].empty else None
    qs = df["return"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    return_quantiles = {str(k): float(v) for k, v in qs.items()}

    return {
        "total_rounds": int(total_rounds),
        "win_rounds": win_rounds,
        "loss_rounds": loss_rounds,
        "win_rate": float(win_rate) if win_rate is not None else None,
        "avg_return": avg_return,
        "median_return": median_return,
        "return_quantiles": return_quantiles,
    }


def diagnose_stop_loss(
    round_trips: pd.DataFrame,
    positions_detail: pd.DataFrame,
    stop_loss_pct: float,
) -> tuple[pd.DataFrame, dict]:
    """从数据角度诊断止损是否“本应触发却未触发”。

    近似方法：
      - 对每个完整回合，取入场日收盘价为 entry_price；
      - 从入场日至平仓日，查看该股票的价格序列；
      - 如果期间曾经出现过 price <= entry_price * (1 - stop_loss_pct)，
        但最终回合收益率并非接近 -stop_loss_pct，而是显著高于该水平，说明：
          * 数据上出现过“跌破止损线但又拉回”，策略可能只按收盘或其它口径判断，
            或止损逻辑不易触发。

    返回：
      - touched_df: 所有曾跌破理论止损线的回合明细；
      - summary: 聚合统计信息。
    """
    if round_trips.empty or positions_detail.empty:
        return pd.DataFrame(), {
            "rounds_with_theoretical_stop_loss": 0,
            "ratio_rounds_with_theoretical_stop_loss": 0.0,
        }

    pos = positions_detail.copy()
    pos["date"] = pd.to_datetime(pos["date"])
    pos = pos.sort_values(["stock_id", "date"])  # type: ignore[arg-type]

    # 仅使用有价格的记录
    pos["price"] = pd.to_numeric(pos["price"], errors="coerce")
    pos = pos[pos["price"].notna()]

    records = []

    for _, r in round_trips.iterrows():
        stock_id = r["stock_id"]
        entry_date = pd.to_datetime(r["entry_date"]) if pd.notna(r["entry_date"]) else None
        exit_date = pd.to_datetime(r["exit_date"]) if pd.notna(r["exit_date"]) else None
        if entry_date is None or exit_date is None:
            continue

        sub = pos[(pos["stock_id"] == stock_id) & (pos["date"] >= entry_date) & (pos["date"] <= exit_date)]
        if sub.empty:
            continue

        sub = sub.sort_values("date")
        entry_price = float(sub.iloc[0]["price"])
        min_price = float(sub["price"].min())
        min_price_date = sub.loc[sub["price"].idxmin(), "date"]
        min_return_from_entry = min_price / entry_price - 1.0 if entry_price > 0 else np.nan

        touched = min_price <= entry_price * (1.0 - stop_loss_pct)

        records.append(
            {
                "stock_id": stock_id,
                "round_id": int(r["round_id"]),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_return": float(r["return"]),
                "exit_pnl": float(r["pnl"]),
                "holding_days": int(r["holding_days"]) if not pd.isna(r["holding_days"]) else None,
                "min_price": min_price,
                "min_price_date": min_price_date,
                "min_return_from_entry": min_return_from_entry,
                "touched_theoretical_stop_loss": bool(touched),
            }
        )

    diag_df = pd.DataFrame(records)
    if diag_df.empty:
        return diag_df, {
            "rounds_with_theoretical_stop_loss": 0,
            "ratio_rounds_with_theoretical_stop_loss": 0.0,
        }

    touched_df = diag_df[diag_df["touched_theoretical_stop_loss"]]
    n_touched = len(touched_df)
    total_rounds = len(diag_df)
    ratio = n_touched / total_rounds if total_rounds > 0 else 0.0

    # 再区分：
    #   - 真正以较大亏损离场（例如 exit_return <= -stop_loss_pct * 0.8）
    #   - 明显拉回后盈利或小亏离场
    hard_stop_like = touched_df[touched_df["exit_return"] <= -stop_loss_pct * 0.8]
    recovered_after_touch = touched_df[touched_df["exit_return"] > -stop_loss_pct * 0.2]

    summary = {
        "total_rounds_considered": int(total_rounds),
        "rounds_with_theoretical_stop_loss": int(n_touched),
        "ratio_rounds_with_theoretical_stop_loss": float(ratio),
        "rounds_hard_stop_like": int(len(hard_stop_like)),
        "rounds_recovered_after_touch": int(len(recovered_after_touch)),
        "avg_min_return_from_entry_on_touched": float(touched_df["min_return_from_entry"].mean())
        if not touched_df.empty
        else None,
        "avg_exit_return_on_touched": float(touched_df["exit_return"].mean())
        if not touched_df.empty
        else None,
    }

    return touched_df, summary


def analyze_cash_and_exposure(daily_positions_summary: pd.DataFrame) -> dict:
    df = daily_positions_summary.copy()
    df["date"] = pd.to_datetime(df["date"])

    total_days = len(df)
    if total_days == 0:
        return {}

    # 空仓日：stock_count == 0
    if "stock_count" in df.columns:
        cash_only = df[df["stock_count"] == 0]
    else:
        cash_only = pd.DataFrame(columns=df.columns)

    cash_only_days = len(cash_only)
    cash_only_ratio = cash_only_days / total_days if total_days > 0 else 0.0

    # 统计最长连续空仓区间
    longest_streak = 0
    current_streak = 0
    streaks = []
    start_date = None

    for _, row in df.sort_values("date").iterrows():
        is_cash_only = (
            row["stock_count"] == 0 if "stock_count" in df.columns else False
        )
        date = row["date"]
        if is_cash_only:
            if current_streak == 0:
                start_date = date
            current_streak += 1
        else:
            if current_streak > 0:
                end_date = date - pd.Timedelta(days=1)
                streaks.append(
                    {
                        "start_date": start_date,
                        "end_date": end_date,
                        "length": current_streak,
                    }
                )
                longest_streak = max(longest_streak, current_streak)
                current_streak = 0
                start_date = None

    if current_streak > 0 and start_date is not None:
        end_date = df["date"].max()
        streaks.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "length": current_streak,
            }
        )
        longest_streak = max(longest_streak, current_streak)

    # 初始与结束资产
    first_row = df.sort_values("date").iloc[0]
    last_row = df.sort_values("date").iloc[-1]

    def _get_or_none(row, col):
        return float(row[col]) if col in row.index else None

    initial_total_value = _get_or_none(first_row, "total_value")
    final_total_value = _get_or_none(last_row, "total_value")

    result = {
        "total_days": int(total_days),
        "cash_only_days": int(cash_only_days),
        "cash_only_ratio": float(cash_only_ratio),
        "cash_only_streaks": [
            {
                "start_date": str(s["start_date"]),
                "end_date": str(s["end_date"]),
                "length": int(s["length"]),
            }
            for s in streaks
        ],
        "longest_cash_only_streak": int(longest_streak),
        "initial_total_value": initial_total_value,
        "final_total_value": final_total_value,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "分析单只股票买入到卖出的实际涨幅、策略胜率、空仓天数，"
            "并从数据角度诊断止损为何很少触发，为策略调整提供定量依据。"
        )
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        required=True,
        help=(
            "backtest_analysis_report 目录路径，"
            "例如 f:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/.../backtest_analysis_report"
        ),
    )
    parser.add_argument(
        "--stop_loss_pct",
        type=float,
        default=0.10,
        help="止损百分比，例如 0.10 表示 -10% 止损线（用于理论止损诊断）",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        raise FileNotFoundError(f"report_dir not found: {report_dir}")

    trading_events_path = report_dir / "trading_events.csv"
    daily_pos_summary_path = report_dir / "daily_positions_summary.csv"
    positions_detail_path = report_dir / "positions_detail.csv"
    analysis_report_path = report_dir / "analysis_report.json"

    print(f"[INFO] 加载 trading_events: {trading_events_path}")
    trading_events = load_csv(trading_events_path)

    print(f"[INFO] 加载 daily_positions_summary: {daily_pos_summary_path}")
    daily_pos_summary = load_csv(daily_pos_summary_path)

    print(f"[INFO] 加载 positions_detail: {positions_detail_path}")
    positions_detail = load_csv(positions_detail_path)

    # 可选：加载原始 analysis_report.json 以便后续融合
    original_summary = None
    if analysis_report_path.exists():
        try:
            original_summary = json.loads(analysis_report_path.read_text(encoding="utf-8"))
        except Exception:
            original_summary = None

    # 1) 构造回合并计算每个回合的涨幅 / PnL
    print("[INFO] 构造每只股票的完整持仓回合（round-trips）...")
    round_trips = build_round_trips(trading_events)

    round_trips_path = report_dir / "trade_round_trips.csv"
    round_trips.to_csv(round_trips_path, index=False)
    print(f"[INFO] 已保存回合明细到: {round_trips_path}")

    round_trips_summary = summarize_round_trips(round_trips)

    # 2) 止损诊断（理论止损 vs 实际行为）
    print("[INFO] 基于 positions_detail 做理论止损诊断...")
    stop_loss_df, stop_loss_summary = diagnose_stop_loss(
        round_trips=round_trips,
        positions_detail=positions_detail,
        stop_loss_pct=args.stop_loss_pct,
    )

    if not stop_loss_df.empty:
        stop_loss_diag_path = report_dir / "stop_loss_rounds.csv"
        stop_loss_df.to_csv(stop_loss_diag_path, index=False)
        print(f"[INFO] 已保存触及理论止损线的回合明细到: {stop_loss_diag_path}")
    else:
        print("[INFO] 未发现价格曾跌破理论止损线的回合，或数据不足。")

    # 3) 空仓天数、初/末资产
    print("[INFO] 分析空仓天数和初/末资产...")
    cash_exposure_summary = analyze_cash_and_exposure(daily_pos_summary)

    # 4) 汇总到一个新的诊断报告 JSON
    diagnostics = {
        "round_trips_summary": round_trips_summary,
        "stop_loss_diagnostics": stop_loss_summary,
        "cash_and_exposure": cash_exposure_summary,
        "stop_loss_pct_assumed": args.stop_loss_pct,
    }

    if original_summary is not None:
        diagnostics["original_analysis_report_summary"] = original_summary.get("summary")

    diag_path = report_dir / "trade_performance_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 已输出诊断报告到: {diag_path}")


if __name__ == "__main__":
    main()
