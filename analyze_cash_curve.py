import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WORKSPACE_CANDIDATES = [
    Path("f:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/dcf014c2ce2a4255bb795ee0f7eb7d9f"),
    Path("/mnt/f/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/dcf014c2ce2a4255bb795ee0f7eb7d9f"),
]
DEFAULT_EXPERIMENT_ID = "891791629306182420"


def _parse_args() -> tuple[Path, str]:
    workspace_arg = sys.argv[1].strip() if len(sys.argv) >= 2 else ""
    experiment_id = sys.argv[2].strip() if len(sys.argv) >= 3 else DEFAULT_EXPERIMENT_ID

    if workspace_arg:
        ws = Path(workspace_arg)
        return ws, experiment_id

    for p in DEFAULT_WORKSPACE_CANDIDATES:
        if p.exists():
            return p, experiment_id

    # 兜底：用第一个候选（即使不存在也让后续报错信息更明确）
    return DEFAULT_WORKSPACE_CANDIDATES[0], experiment_id


def _get_debug_date_arg() -> str:
    return sys.argv[3].strip() if len(sys.argv) >= 4 else ""


def _pick_latest_run_artifacts(workspace_path: Path, experiment_id: str) -> tuple[Path | None, str | None]:
    exp_dir = workspace_path / "mlruns" / experiment_id
    if not exp_dir.exists():
        return None, None

    best_run_id = None
    best_mtime = None
    best_artifacts = None

    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        artifacts_dir = run_dir / "artifacts"
        positions_file = artifacts_dir / "portfolio_analysis" / "positions_normal_1day.pkl"
        if not positions_file.exists():
            continue
        try:
            mtime = positions_file.stat().st_mtime
        except Exception:
            continue
        if best_mtime is None or mtime > best_mtime:
            best_mtime = mtime
            best_run_id = run_dir.name
            best_artifacts = artifacts_dir

    return best_artifacts, best_run_id


def _to_dt(x) -> pd.Timestamp:
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.Timestamp(str(x))


def _is_account_field(k: str) -> bool:
    kl = str(k).lower()
    return kl in {
        "cash",
        "now_account_value",
        "account_value",
        "init_cash",
        "total_value",
    } or kl.endswith("_account_value")


def _extract_position_obj(pos_obj):
    if pos_obj is None:
        return None
    if hasattr(pos_obj, "position"):
        try:
            return getattr(pos_obj, "position")
        except Exception:
            return None
    return pos_obj


def _extract_cash_and_account_value(pos_dict: dict) -> tuple[float | None, float | None]:
    cash = None
    nav = None

    for k in ["cash", "CASH"]:
        if k in pos_dict:
            try:
                cash = float(pos_dict[k])
            except Exception:
                pass

    for k in ["now_account_value", "account_value", "NOW_ACCOUNT_VALUE", "ACCOUNT_VALUE"]:
        if k in pos_dict:
            try:
                nav = float(pos_dict[k])
            except Exception:
                pass

    return cash, nav


def _extract_stock_value(pos_dict: dict) -> float:
    total = 0.0
    for k, v in pos_dict.items():
        if _is_account_field(k):
            continue
        if v is None:
            continue

        amt = None
        px = None

        if isinstance(v, dict):
            if "amount" in v:
                amt = v.get("amount")
            elif "amt" in v:
                amt = v.get("amt")

            if "price" in v:
                px = v.get("price")
            elif "close" in v:
                px = v.get("close")
            elif "last_price" in v:
                px = v.get("last_price")
        else:
            # 有些版本可能直接存 amount
            try:
                amt = float(v)
            except Exception:
                amt = None

        try:
            amt_f = float(amt) if amt is not None else 0.0
        except Exception:
            amt_f = 0.0

        try:
            px_f = float(px) if px is not None else np.nan
        except Exception:
            px_f = np.nan

        if not np.isfinite(px_f) or px_f <= 0:
            continue
        total += amt_f * px_f

    return float(total)


def _load_positions_curve(positions_file: Path) -> tuple[pd.DataFrame, dict]:
    with open(positions_file, "rb") as f:
        positions = pickle.load(f)

    meta: dict[str, object] = {
        "positions_file": str(positions_file),
        "positions_type": str(type(positions)),
        "init_cash_from_obj": None,
        "first_dt": None,
        "first_cash": None,
        "first_nav": None,
    }

    # positions 常见结构：dict[dt] -> Position
    if not isinstance(positions, dict):
        raise TypeError(f"Unsupported positions type: {type(positions)}")

    rows = []

    for dt_raw, pos_obj in positions.items():
        dt = _to_dt(dt_raw)

        init_cash = None
        if hasattr(pos_obj, "init_cash"):
            try:
                init_cash = float(getattr(pos_obj, "init_cash"))
            except Exception:
                init_cash = None

        pos_dict = _extract_position_obj(pos_obj)
        if not isinstance(pos_dict, dict):
            continue

        cash, nav = _extract_cash_and_account_value(pos_dict)
        stock_value = _extract_stock_value(pos_dict)

        # 最稳的净值：如果 nav 有值，优先用 nav；否则用 cash + stock_value
        if nav is not None and np.isfinite(nav):
            total_value = float(nav)
        else:
            total_value = float((cash or 0.0) + stock_value)

        rows.append(
            {
                "date": dt,
                "cash": cash,
                "now_account_value": nav,
                "stock_value": stock_value,
                "total_value_from_pos": total_value,
                "init_cash_from_obj": init_cash,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("positions 文件解析后为空")

    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # meta
    meta["init_cash_from_obj"] = (
        float(df["init_cash_from_obj"].dropna().iloc[0]) if df["init_cash_from_obj"].notna().any() else None
    )
    meta["first_dt"] = str(df.iloc[0]["date"])
    meta["first_cash"] = float(df.iloc[0]["cash"]) if pd.notna(df.iloc[0]["cash"]) else None
    meta["first_nav"] = (
        float(df.iloc[0]["now_account_value"]) if pd.notna(df.iloc[0]["now_account_value"]) else None
    )

    return df, meta


def _debug_dump_position(positions_file: Path, debug_date: str) -> None:
    if not debug_date:
        return

    try:
        debug_dt = pd.to_datetime(debug_date)
    except Exception:
        print(f"  - debug_date 解析失败: {debug_date}")
        return

    with open(positions_file, "rb") as f:
        positions = pickle.load(f)
    if not isinstance(positions, dict):
        print(f"  - positions 类型不支持 debug: {type(positions)}")
        return

    target = None
    for dt_raw, pos_obj in positions.items():
        dt = _to_dt(dt_raw)
        if dt.normalize() == debug_dt.normalize():
            target = pos_obj
            break

    if target is None:
        print(f"  - 未找到 debug_date 对应的 position: {debug_date}")
        return

    pos_dict = _extract_position_obj(target)
    if not isinstance(pos_dict, dict):
        print(f"  - debug_date 的 position 结构非 dict: {type(pos_dict)}")
        return

    keys = list(pos_dict.keys())
    print(f"  [debug] date={debug_dt.date()} pos_dict_keys_cnt={len(keys)} keys_preview={keys[:20]}")

    cash, nav = _extract_cash_and_account_value(pos_dict)
    print(f"  [debug] cash={cash} now_account_value={nav}")

    sample_items = []
    for k, v in pos_dict.items():
        if _is_account_field(k):
            continue
        sample_items.append((k, v))
        if len(sample_items) >= 3:
            break

    for idx, (k, v) in enumerate(sample_items, start=1):
        if isinstance(v, dict):
            v_preview = {kk: v.get(kk) for kk in list(v.keys())[:12]}
            print(f"  [debug] holding#{idx} instrument={k} type=dict keys={list(v.keys())[:20]} preview={v_preview}")
        else:
            print(f"  [debug] holding#{idx} instrument={k} type={type(v)} value_preview={v}")


def _load_indicator_curve(indicators_file: Path) -> pd.DataFrame:
    with open(indicators_file, "rb") as f:
        ind = pickle.load(f)

    # qlib 里通常是 DataFrame
    if isinstance(ind, pd.DataFrame):
        df = ind.copy()
    elif isinstance(ind, dict):
        df = pd.DataFrame(ind)
    else:
        raise TypeError(f"Unsupported indicators type: {type(ind)}")

    if df.index is not None:
        df = df.reset_index().rename(columns={"index": "date"})
    if "date" not in df.columns:
        raise ValueError("indicators 中找不到 date")

    df["date"] = pd.to_datetime(df["date"])
    if "value" not in df.columns:
        raise ValueError("indicators 中找不到 value")

    out = df[["date", "value"]].rename(columns={"value": "total_value_from_indicators"})
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def _load_daily_positions_summary(daily_csv: Path) -> pd.DataFrame:
    if not daily_csv.exists():
        return pd.DataFrame(columns=["date", "total_value_from_daily_positions_summary", "stock_count"])

    df = pd.read_csv(daily_csv)
    if df.empty:
        return pd.DataFrame(columns=["date", "total_value_from_daily_positions_summary", "stock_count"])

    if "date" not in df.columns or "total_value" not in df.columns:
        return pd.DataFrame(columns=["date", "total_value_from_daily_positions_summary", "stock_count"])

    df["date"] = pd.to_datetime(df["date"])
    out = df[["date", "total_value"]].rename(columns={"total_value": "total_value_from_daily_positions_summary"})
    if "stock_count" in df.columns:
        out["stock_count"] = df["stock_count"]
    return out.sort_values("date").reset_index(drop=True)


def _describe_series(name: str, s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    res = {
        "name": name,
        "count": int(s.notna().sum()),
        "min": float(s.min()) if s.notna().any() else None,
        "max": float(s.max()) if s.notna().any() else None,
        "median": float(s.median()) if s.notna().any() else None,
        "mean": float(s.mean()) if s.notna().any() else None,
        "std": float(s.std()) if s.notna().any() else None,
        "non_positive_cnt": int((s <= 0).sum()) if s.notna().any() else 0,
    }
    return res


def main() -> int:
    workspace_path, experiment_id = _parse_args()
    debug_date = _get_debug_date_arg()
    artifacts_dir, run_id = _pick_latest_run_artifacts(workspace_path, experiment_id)
    if artifacts_dir is None:
        raise FileNotFoundError(f"No valid artifacts found under {workspace_path}/mlruns/{experiment_id}")

    pa_dir = artifacts_dir / "portfolio_analysis"
    positions_file = pa_dir / "positions_normal_1day.pkl"
    indicators_file = pa_dir / "indicators_normal_1day.pkl"

    output_dir = workspace_path / "backtest_analysis_report"
    daily_positions_summary_csv = output_dir / "daily_positions_summary.csv"

    print(f"Python: {sys.executable}")
    print(f"PythonVersion: {sys.version.splitlines()[0]}")
    print(f"Workspace: {workspace_path}")
    print(f"ExperimentId: {experiment_id}")
    print(f"RunId: {run_id}")
    print(f"Artifacts: {artifacts_dir}")
    print()

    print("[1/4] 读取 positions 并计算净值曲线 (口径B: cash + stock_value 或 now_account_value)...")
    pos_curve, pos_meta = _load_positions_curve(positions_file)
    print("  ✓ positions 解析成功")
    for k, v in pos_meta.items():
        print(f"    - {k}: {v}")
    if debug_date:
        _debug_dump_position(positions_file, debug_date)
    print()

    print("[2/4] 读取 indicators 并提取净值曲线 (口径A: indicators.value)...")
    ind_curve = _load_indicator_curve(indicators_file)
    print(f"  ✓ indicators 解析成功, 行数={len(ind_curve)}")
    print()

    print("[3/4] 读取 daily_positions_summary.csv (口径C: total_value)...")
    dps_curve = _load_daily_positions_summary(daily_positions_summary_csv)
    if dps_curve.empty:
        print("  - daily_positions_summary.csv 不存在或为空，跳过")
    else:
        print(f"  ✓ daily_positions_summary 读取成功, 行数={len(dps_curve)}")
    print()

    print("[4/4] 合并三种口径并做一致性诊断...")
    merged = pos_curve[["date", "total_value_from_pos", "cash", "stock_value", "now_account_value"]].merge(
        ind_curve, on="date", how="outer"
    )
    if not dps_curve.empty:
        merged = merged.merge(dps_curve, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # 反推 init_cash（多种候选）
    init_cash_candidates = {
        "init_cash_from_position_obj": pos_meta.get("init_cash_from_obj"),
        "first_total_value_from_pos": float(merged["total_value_from_pos"].dropna().iloc[0])
        if merged["total_value_from_pos"].notna().any()
        else None,
        "first_total_value_from_indicators": float(merged["total_value_from_indicators"].dropna().iloc[0])
        if merged["total_value_from_indicators"].notna().any()
        else None,
        "first_total_value_from_daily_positions_summary": float(
            merged["total_value_from_daily_positions_summary"].dropna().iloc[0]
        )
        if "total_value_from_daily_positions_summary" in merged.columns
        and merged["total_value_from_daily_positions_summary"].notna().any()
        else None,
    }

    print("  init_cash 反推/读取结果：")
    for k, v in init_cash_candidates.items():
        print(f"    - {k}: {v}")
    print()

    # 统计描述
    series_to_describe = {
        "total_value_from_pos": merged.get("total_value_from_pos"),
        "total_value_from_indicators": merged.get("total_value_from_indicators"),
    }
    if "total_value_from_daily_positions_summary" in merged.columns:
        series_to_describe["total_value_from_daily_positions_summary"] = merged.get(
            "total_value_from_daily_positions_summary"
        )

    print("  各口径数值范围（用于判断是否爆炸/为0/为负）：")
    for name, s in series_to_describe.items():
        if s is None:
            continue
        desc = _describe_series(name, s)
        print(
            f"    - {name}: count={desc['count']} min={desc['min']} max={desc['max']} median={desc['median']} mean={desc['mean']} non_positive_cnt={desc['non_positive_cnt']}"
        )
    print()

    # 比值诊断（避免数量级问题）
    if merged["total_value_from_pos"].notna().any() and merged["total_value_from_indicators"].notna().any():
        ratio_pi = merged["total_value_from_pos"] / merged["total_value_from_indicators"]
        ratio_pi = pd.to_numeric(ratio_pi, errors="coerce")
        ratio_desc = _describe_series("ratio_pos_over_indicators", ratio_pi)
        print(
            "  口径B/口径A 比值（pos / indicators.value）: "
            f"count={ratio_desc['count']} min={ratio_desc['min']} max={ratio_desc['max']} median={ratio_desc['median']} mean={ratio_desc['mean']}"
        )

    if "total_value_from_daily_positions_summary" in merged.columns and merged["total_value_from_daily_positions_summary"].notna().any():
        if merged["total_value_from_pos"].notna().any():
            ratio_dp = merged["total_value_from_daily_positions_summary"] / merged["total_value_from_pos"]
            ratio_dp = pd.to_numeric(ratio_dp, errors="coerce")
            ratio_desc = _describe_series("ratio_daily_positions_over_pos", ratio_dp)
            print(
                "  口径C/口径B 比值（daily_positions_summary.total_value / pos）: "
                f"count={ratio_desc['count']} min={ratio_desc['min']} max={ratio_desc['max']} median={ratio_desc['median']} mean={ratio_desc['mean']}"
            )

        if merged["total_value_from_indicators"].notna().any():
            ratio_di = merged["total_value_from_daily_positions_summary"] / merged["total_value_from_indicators"]
            ratio_di = pd.to_numeric(ratio_di, errors="coerce")
            ratio_desc = _describe_series("ratio_daily_positions_over_indicators", ratio_di)
            print(
                "  口径C/口径A 比值（daily_positions_summary.total_value / indicators.value）: "
                f"count={ratio_desc['count']} min={ratio_desc['min']} max={ratio_desc['max']} median={ratio_desc['median']} mean={ratio_desc['mean']}"
            )

    # 输出合并结果用于你人工核对
    out_csv = output_dir / "cash_curve_compare.csv"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)
        print()
        print(f"  ✓ 已输出: {out_csv}")
    except Exception as e:
        print()
        print(f"  - 输出 {out_csv} 失败: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
