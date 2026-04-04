#!/usr/bin/env python
"""把测试集评估结果批量写入所有 snapshot 的 metrics_json.test_metrics 字段.

同时计算排名（按 1d spread 降序）写入 metrics_json.test_rank。

用法:
  cd /mnt/f/Dev/RD-Agent-main
  conda activate rdagent-gpu
  python -m model_training.hmm.backfill_test_metrics --db-password lc78080808
"""
from __future__ import annotations
import argparse, json, os, sys
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model_training.hmm.train_sector_hmm import build_observation_matrix
from model_training.common.data_loader import (
    get_db_conn, load_l2_sector_data, load_csi300_daily,
    load_market_total_volume, load_sector_stock_mapping,
    get_limit_up_ratio_by_sector, read_qlib_calendar,
)

TEST_START = date(2025, 4, 1)
TEST_END = date(2026, 3, 10)
EVAL_WINDOWS = [1, 2, 3, 5, 10, 20]
QLIB_BIN_DIR = "/home/lc999/data/qlib_bin"


def evaluate_single(models, sector_data, csi300, market_vol, limit_up_data, limit_down_data):
    from hmmlearn.hmm import GaussianHMM
    first = next(iter(models.values()), None)
    if not first:
        return None
    rw = first.get("rolling_window", 20)
    uld = first.get("use_limit_down", False)
    n_feat = len(first["means"][0])
    old_8d = (n_feat == 8 and not uld)
    actual_uld = uld or old_8d
    zscore_mean = np.array(first["zscore_mean"]) if "zscore_mean" in first else None
    zscore_std = np.array(first["zscore_std"]) if "zscore_std" in first else None

    hmm_objs = {}
    for code, info in models.items():
        hmm = GaussianHMM(n_components=info["n_states"], covariance_type=info["covariance_type"])
        hmm.startprob_ = np.array([1.0 / info["n_states"]] * info["n_states"])
        hmm.transmat_ = np.array(info["transmat"])
        hmm.means_ = np.array(info["means"])
        covars = np.array(info["covars"], dtype=np.float64)
        if info["covariance_type"] == "diag":
            if covars.ndim == 3:
                covars = np.array([np.diag(covars[i]) for i in range(covars.shape[0])])
            covars = np.maximum(covars, 1e-6)
        elif info["covariance_type"] == "full":
            for i in range(covars.shape[0]):
                covars[i] = (covars[i] + covars[i].T) / 2
                covars[i] += np.eye(covars[i].shape[0]) * 1e-6
        hmm.covars_ = covars
        hmm_objs[code] = (hmm, info["state_labels"])

    results = {w: {"trending": [], "fading": [], "neutral": []} for w in EVAL_WINDOWS}
    decoded = 0
    total_pred = 0

    for code in models:
        if code not in sector_data:
            continue
        dl = sorted(sector_data[code], key=lambda x: x["trade_date"])
        lu = limit_up_data.get(code, {})
        ld = limit_down_data.get(code, {})
        obs, od = build_observation_matrix(dl, csi300, market_vol, lu, ld, rolling_window=rw, use_limit_down=actual_uld)
        if obs.shape[0] < 20:
            continue
        if zscore_mean is not None:
            obs = (obs - zscore_mean) / zscore_std
        hmm, labels = hmm_objs[code]
        try:
            states = hmm.predict(obs)
        except:
            continue
        decoded += 1
        pct_map = {r["trade_date"]: r["pct_change"] for r in dl}
        ext = sorted(set(r["trade_date"] for r in dl if r["trade_date"] >= TEST_START))
        for i, td in enumerate(od):
            if not (TEST_START <= td <= TEST_END):
                continue
            label = labels.get(str(states[i]), "unknown")
            if label == "unknown":
                continue
            ti = -1
            for idx, d in enumerate(ext):
                if d == td:
                    ti = idx
                    break
            if ti < 0:
                continue
            total_pred += 1
            for w in EVAL_WINDOWS:
                fut = []
                for off in range(1, w + 1):
                    fi = ti + off
                    if fi < len(ext):
                        r = pct_map.get(ext[fi])
                        if r is not None:
                            fut.append(r)
                if len(fut) >= w:
                    results[w][label].append(sum(fut))

    def sm(lst):
        return round(float(np.mean(lst)), 4) if lst else None
    def wr(lst):
        return round(float(np.sum(np.array(lst) > 0) / len(lst) * 100), 1) if lst else None

    m = {"test_period": f"{TEST_START} ~ {TEST_END}", "decoded_sectors": decoded, "total_predictions": total_pred}
    for w in EVAL_WINDOWS:
        t, f = results[w]["trending"], results[w]["fading"]
        tm, fm = sm(t), sm(f)
        m[f"spread_{w}d"] = round(tm - fm, 4) if tm is not None and fm is not None else None
        m[f"trending_{w}d"] = tm
        m[f"fading_{w}d"] = fm
        m[f"trending_{w}d_n"] = len(t)
        m[f"fading_{w}d_n"] = len(f)
        m[f"trending_{w}d_winrate"] = wr(t)
        m[f"fading_{w}d_winrate"] = wr(f)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-password", default="")
    args = parser.parse_args()

    conn = get_db_conn(password=args.db_password)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.config_id, c.display_name, s.snapshot_id, s.model_path, s.metrics_json
        FROM model_train_configs c
        JOIN model_train_snapshots s ON s.config_id = c.config_id
        WHERE c.model_type='sector_hmm' AND s.status='completed'
        ORDER BY c.config_id, s.trained_at DESC
    """)
    rows = cur.fetchall()
    cur.close()

    seen = set()
    snapshots = []
    for cid, dn, sid, mp, mj in rows:
        if cid in seen:
            continue
        seen.add(cid)
        snapshots.append({"config_id": cid, "display_name": dn, "snapshot_id": sid, "model_path": mp, "metrics_json": mj})

    print(f"找到 {len(snapshots)} 个模型，加载公共数据...")
    ds = date(2022, 1, 1) - timedelta(days=30)
    de = TEST_END + timedelta(days=30)
    sd = load_l2_sector_data(conn, ds, de)
    c3 = load_csi300_daily(conn, ds, de)
    mv = load_market_total_volume(conn, ds, de)
    ss = load_sector_stock_mapping(conn, "L2")
    cal = read_qlib_calendar(QLIB_BIN_DIR)
    lu, ld = get_limit_up_ratio_by_sector(QLIB_BIN_DIR, ss, cal, date(2022, 1, 1), de)
    conn.close()

    all_test = []
    for snap in snapshots:
        mp = snap["model_path"]
        dn = snap["display_name"]
        if not os.path.exists(mp):
            print(f"  跳过 {dn}: 文件不存在")
            all_test.append((snap, None))
            continue
        with open(mp) as f:
            models = json.load(f)
        print(f"  评估 {dn}...")
        tm = evaluate_single(models, sd, c3, mv, lu, ld)
        all_test.append((snap, tm))

    # 计算排名（按 1d spread 降序）
    valid = [(s, t) for s, t in all_test if t and t.get("spread_1d") is not None]
    valid.sort(key=lambda x: x[1]["spread_1d"], reverse=True)
    rank_map = {}
    for i, (s, t) in enumerate(valid):
        rank_map[s["snapshot_id"]] = i + 1

    # 写入 DB
    conn = get_db_conn(password=args.db_password)
    cur = conn.cursor()
    updated = 0
    for snap, tm in all_test:
        mj = snap["metrics_json"]
        if isinstance(mj, str):
            mj = json.loads(mj) if mj else {}
        elif mj is None:
            mj = {}
        if tm:
            mj["test_metrics"] = tm
            mj["test_rank"] = rank_map.get(snap["snapshot_id"])
            mj["test_rank_total"] = len(valid)
        cur.execute(
            "UPDATE model_train_snapshots SET metrics_json = %s WHERE snapshot_id = %s",
            (json.dumps(mj, ensure_ascii=False), snap["snapshot_id"]),
        )
        updated += 1
        r = rank_map.get(snap["snapshot_id"], "-")
        s1 = tm["spread_1d"] if tm and tm.get("spread_1d") is not None else "N/A"
        print(f"    {snap['display_name']:<42} rank={r}  1d_spread={s1}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n已更新 {updated} 个 snapshot 的 metrics_json")


if __name__ == "__main__":
    main()
