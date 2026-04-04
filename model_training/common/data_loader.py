"""通用数据加载器 — 支持 DB 查询 + Qlib bin 文件读取."""
from __future__ import annotations

import logging
import struct
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


def get_db_conn(host="127.0.0.1", port=5432, user="postgres", password="", dbname="aistock"):
    return psycopg2.connect(host=host, port=port, user=user, password=password, dbname=dbname)


# ─── Qlib bin 文件读取 ───

def read_qlib_bin(path: str) -> Tuple[int, np.ndarray]:
    """读取 Qlib bin 文件，返回 (start_index, values_array).

    Qlib bin 格式: 前 4 字节为 uint32 start_index，后续为 float32 数组。
    """
    p = Path(path)
    if not p.exists():
        return 0, np.array([], dtype=np.float32)
    with open(p, "rb") as f:
        buf = f.read()
    if len(buf) < 4:
        return 0, np.array([], dtype=np.float32)
    start_idx = struct.unpack("<I", buf[:4])[0]
    values = np.frombuffer(buf[4:], dtype=np.float32)
    return start_idx, values


def read_qlib_calendar(qlib_dir: str) -> List[date]:
    """读取 Qlib 日线 calendar."""
    cal_path = Path(qlib_dir) / "calendars" / "day.txt"
    if not cal_path.exists():
        logger.warning("Calendar not found: %s", cal_path)
        return []
    dates = []
    with open(cal_path) as f:
        for line in f:
            line = line.strip()
            if line:
                dates.append(date.fromisoformat(line[:10]))
    return sorted(dates)


def get_limit_up_ratio_by_sector(
    qlib_dir: str,
    sector_stocks: Dict[str, List[str]],
    calendar: List[date],
    start_date: date,
    end_date: date,
) -> Tuple[Dict[str, Dict[date, float]], Dict[str, Dict[date, float]]]:
    """从 Qlib bin 文件计算每个行业每天的涨停占比和跌停占比.

    Returns:
        (limit_up_ratios, limit_down_ratios)
        Each: {sector_code: {date: ratio}}
    """
    features_dir = Path(qlib_dir) / "features"
    if not features_dir.exists():
        logger.warning("Qlib features dir not found: %s", features_dir)
        return {}, {}

    date_to_idx = {d: i for i, d in enumerate(calendar)}
    cal_dates = sorted(date_to_idx.keys())
    start_idx = next((date_to_idx[d] for d in cal_dates if d >= start_date), None)
    end_idx = next((date_to_idx[d] for d in reversed(cal_dates) if d <= end_date), None)
    if start_idx is None or end_idx is None:
        return {}, {}

    lu_result: Dict[str, Dict[date, float]] = {}
    ld_result: Dict[str, Dict[date, float]] = {}

    for sector_code, stocks in sector_stocks.items():
        daily_lu = {}  # {date: [limit_up_count, total]}
        daily_ld = {}  # {date: [limit_down_count, total]}

        for stock in stocks:
            stock_lower = stock.lower()
            lu_path = features_dir / stock_lower / "limit_up.day.bin"
            ld_path = features_dir / stock_lower / "limit_down.day.bin"

            lu_start, lu_vals = (0, np.array([], dtype=np.float32))
            ld_start, ld_vals = (0, np.array([], dtype=np.float32))

            if lu_path.exists():
                lu_start, lu_vals = read_qlib_bin(str(lu_path))
            if ld_path.exists():
                ld_start, ld_vals = read_qlib_bin(str(ld_path))

            for cal_idx in range(start_idx, end_idx + 1):
                if cal_idx >= len(calendar):
                    break
                td = calendar[cal_idx]

                # Limit up
                lu_val_idx = cal_idx - lu_start
                if 0 <= lu_val_idx < len(lu_vals):
                    val = lu_vals[lu_val_idx]
                    if not np.isnan(val):
                        if td not in daily_lu:
                            daily_lu[td] = [0, 0]
                        daily_lu[td][1] += 1
                        if val >= 0.5:
                            daily_lu[td][0] += 1

                # Limit down
                ld_val_idx = cal_idx - ld_start
                if 0 <= ld_val_idx < len(ld_vals):
                    val = ld_vals[ld_val_idx]
                    if not np.isnan(val):
                        if td not in daily_ld:
                            daily_ld[td] = [0, 0]
                        daily_ld[td][1] += 1
                        if val >= 0.5:
                            daily_ld[td][0] += 1

        lu_result[sector_code] = {td: c[0] / c[1] if c[1] > 0 else 0.0 for td, c in daily_lu.items()}
        ld_result[sector_code] = {td: c[0] / c[1] if c[1] > 0 else 0.0 for td, c in daily_ld.items()}

    return lu_result, ld_result


# ─── DB 数据加载 ───

def load_l2_sector_data(
    conn,
    start_date: date,
    end_date: date,
) -> Dict[str, List[Dict[str, Any]]]:
    """从 sector_data 表加载 L2 行业级别数据.

    Returns:
        {l2_code: [{trade_date, sw2_pct_change, sw2_vol, sw2_amount,
                     sw2_mf_net_amt, sw2_mf_buy_elg_amt, sw2_mf_sell_elg_amt}, ...]}
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT DISTINCT ON (m.l2_code, sd.trade_date)
            m.l2_code, m.l2_name, sd.trade_date,
            sd.sw2_pct_change, sd.sw2_vol, sd.sw2_amount,
            sd.sw2_mf_net_amt, sd.sw2_mf_buy_elg_amt, sd.sw2_mf_sell_elg_amt,
            sd.sw2_total_mv
        FROM market.sector_data sd
        JOIN market.sw_index_member m ON sd.ts_code = m.ts_code
        WHERE sd.trade_date BETWEEN %s AND %s
          AND m.in_date <= sd.trade_date
          AND (m.out_date IS NULL OR m.out_date >= sd.trade_date)
        ORDER BY m.l2_code, sd.trade_date, sd.ts_code
    """, (start_date, end_date))

    result: Dict[str, List[Dict]] = {}
    for row in cur.fetchall():
        code = row["l2_code"]
        if code not in result:
            result[code] = []
        result[code].append({
            "trade_date": row["trade_date"],
            "l2_name": row["l2_name"],
            "pct_change": float(row["sw2_pct_change"] or 0),
            "vol": float(row["sw2_vol"] or 0),
            "amount": float(row["sw2_amount"] or 0),
            "mf_net_amt": float(row["sw2_mf_net_amt"] or 0),
            "mf_buy_elg_amt": float(row["sw2_mf_buy_elg_amt"] or 0),
            "mf_sell_elg_amt": float(row["sw2_mf_sell_elg_amt"] or 0),
            "total_mv": float(row["sw2_total_mv"] or 0),
        })
    cur.close()
    return result


def load_csi300_daily(conn, start_date: date, end_date: date) -> Dict[date, float]:
    """加载 CSI300 日收益率."""
    cur = conn.cursor()
    cur.execute("""
        SELECT trade_date, pct_chg FROM market.index_daily
        WHERE ts_code = '399300.SZ' AND trade_date BETWEEN %s AND %s
        ORDER BY trade_date
    """, (start_date, end_date))
    result = {td: float(pct or 0) for td, pct in cur.fetchall()}
    cur.close()
    return result


def load_market_total_volume(conn, start_date: date, end_date: date) -> Dict[date, float]:
    """加载全市场总成交量（从 sw_daily 聚合）."""
    cur = conn.cursor()
    cur.execute("""
        SELECT trade_date, SUM(vol) FROM market.sw_daily
        WHERE trade_date BETWEEN %s AND %s
        GROUP BY trade_date ORDER BY trade_date
    """, (start_date, end_date))
    result = {td: float(v or 0) for td, v in cur.fetchall()}
    cur.close()
    return result


def load_sector_stock_mapping(conn, sector_level: str = "L2") -> Dict[str, List[str]]:
    """加载行业→个股映射（用于从 Qlib bin 计算涨停占比）.

    Returns:
        {sector_code: [stock_code, ...]}
    """
    col = "l2_code" if sector_level == "L2" else "l1_code"
    cur = conn.cursor()
    cur.execute(f"""
        SELECT {col}, ts_code FROM market.sw_index_member
        WHERE out_date IS NULL OR out_date >= CURRENT_DATE
    """)
    result: Dict[str, List[str]] = {}
    for sector_code, ts_code in cur.fetchall():
        if sector_code not in result:
            result[sector_code] = []
        result[sector_code].append(ts_code)
    cur.close()
    return result
