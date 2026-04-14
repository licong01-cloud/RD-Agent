#!/usr/bin/env python3
"""
Backfill feature_importance into qlib_results_enhanced.json for existing QE loops.

For PyTorch models (GeneralPTNN/SimpleGRU) that lack LightGBM native feature_importance,
computes |corr(factor, prediction)| as a model-agnostic importance proxy.

Usage:
    python scripts/backfill_feature_importance.py <task_id>
    python scripts/backfill_feature_importance.py qe_20260410_173242_d073

Optional flags:
    --update-db     Also update AIStock DB qe_evolution_loops.metrics_json
    --loop <N>      Only process LoopN (default: all loops)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pickle

QE_WORKSPACE = os.environ.get(
    "QE_WORKSPACE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qe_workspace"),
)


def find_recorder_artifacts(loop_dir: str) -> str | None:
    """Find the mlruns artifacts directory for the loop."""
    mlruns = os.path.join(loop_dir, "mlruns")
    if not os.path.isdir(mlruns):
        return None
    for exp_id in os.listdir(mlruns):
        exp_path = os.path.join(mlruns, exp_id)
        if exp_id == "0" or not os.path.isdir(exp_path):
            continue
        for run_id in os.listdir(exp_path):
            artifacts_path = os.path.join(exp_path, run_id, "artifacts")
            if os.path.isdir(artifacts_path):
                return artifacts_path
    return None


def extract_lgbm_importance(artifacts_dir: str) -> list | None:
    """Try LightGBM native feature_importance."""
    params_path = os.path.join(artifacts_dir, "params.pkl")
    if not os.path.exists(params_path):
        return None
    try:
        with open(params_path, "rb") as f:
            params = pickle.load(f)
    except Exception as e:
        print(f"  [INFO] Cannot unpickle params.pkl ({e}), skipping LightGBM path")
        return None
    model = getattr(params, "model", None)
    if model is None or not hasattr(model, "feature_importance"):
        return None
    fi_gain = model.feature_importance(importance_type="gain")
    fi_split = model.feature_importance(importance_type="split")
    names = model.feature_name() if hasattr(model, "feature_name") else [f"f{i}" for i in range(len(fi_gain))]
    total = sum(fi_gain) or 1.0
    result = []
    for i, name in enumerate(names):
        result.append({
            "name": name,
            "gain": round(float(fi_gain[i]), 4),
            "gain_pct": round(float(fi_gain[i]) / total * 100, 2),
            "split": int(fi_split[i]),
            "method": "lightgbm_gain",
        })
    result.sort(key=lambda x: -x["gain"])
    return result


def extract_correlation_importance(loop_dir: str, artifacts_dir: str) -> list:
    """Compute |corr(factor, pred)| from parquet + pred.pkl."""
    pred_path = os.path.join(artifacts_dir, "pred.pkl")
    factors_path = os.path.join(loop_dir, "combined_factors_df.parquet")

    if not os.path.exists(pred_path):
        print(f"  [SKIP] pred.pkl not found: {pred_path}")
        return []
    if not os.path.exists(factors_path):
        print(f"  [SKIP] combined_factors_df.parquet not found: {factors_path}")
        return []

    try:
        with open(pred_path, "rb") as f:
            pred = pickle.load(f)
    except Exception as e:
        # Fallback: pandas can often read pickled DataFrame directly
        try:
            pred = pd.read_pickle(pred_path)
        except Exception as e2:
            print(f"  [SKIP] Failed to load pred.pkl: {e} / {e2}")
            return []

    if isinstance(pred, pd.DataFrame):
        pred_series = pred["score"] if "score" in pred.columns else pred.iloc[:, 0]
    else:
        pred_series = pred

    factors_df = pd.read_parquet(factors_path)
    print(f"  pred shape: {pred_series.shape}, factors shape: {factors_df.shape}")

    common_idx = pred_series.index.intersection(factors_df.index)
    if len(common_idx) == 0:
        print("  [SKIP] No common index between pred and factors")
        return []
    print(f"  common samples: {len(common_idx)}")

    pred_aligned = pred_series.loc[common_idx]
    factors_aligned = factors_df.loc[common_idx].reindex(pred_aligned.index)
    pred_arr = pred_aligned.to_numpy(dtype=float, copy=False)

    scores = []
    skipped = []
    for col in factors_aligned.columns:
        try:
            col_vals = factors_aligned[col].to_numpy(dtype=float, copy=False)
        except (ValueError, TypeError) as e:
            skipped.append((col, f"non-numeric: {e}"))
            continue
        mask = np.isfinite(col_vals) & np.isfinite(pred_arr)
        n_valid = int(mask.sum())
        if n_valid < 100:
            skipped.append((col, f"insufficient samples: {n_valid}"))
            continue
        cv = col_vals[mask]
        pv = pred_arr[mask]
        if cv.std() == 0 or pv.std() == 0:
            skipped.append((col, "zero variance"))
            continue
        c = np.corrcoef(cv, pv)[0, 1]
        if not np.isfinite(c):
            skipped.append((col, "non-finite corr"))
            continue
        scores.append((col, abs(float(c))))

    if skipped:
        print(f"  [WARN] Skipped {len(skipped)} factors:")
        for name, reason in skipped[:10]:
            print(f"         - {name}: {reason}")

    def _clean_col_name(col) -> str:
        """Handle MultiIndex column tuples like ('feature', 'factor_name')."""
        if isinstance(col, tuple):
            return str(col[-1])
        return str(col)

    total = sum(s for _, s in scores) or 1.0
    result = []
    for name, s in scores:
        result.append({
            "name": _clean_col_name(name),
            "gain": round(float(s), 6),
            "gain_pct": round(float(s) / total * 100, 2),
            "split": 0,
            "method": "pytorch_correlation",
        })
    result.sort(key=lambda x: -x["gain"])
    return result


def process_loop(task_id: str, loop_name: str, update_db: bool = False):
    """Process a single loop: compute importance and update enhanced JSON."""
    loop_dir = os.path.join(QE_WORKSPACE, task_id, loop_name)
    if not os.path.isdir(loop_dir):
        print(f"  [SKIP] Loop dir not found: {loop_dir}")
        return False

    artifacts_dir = find_recorder_artifacts(loop_dir)
    if not artifacts_dir:
        print(f"  [SKIP] No mlruns artifacts found in {loop_dir}")
        return False
    print(f"  artifacts: {artifacts_dir}")

    # Try LightGBM first, then correlation-based
    importance = extract_lgbm_importance(artifacts_dir)
    if importance:
        print(f"  -> LightGBM importance: {len(importance)} features")
    else:
        importance = extract_correlation_importance(loop_dir, artifacts_dir)
        if importance:
            print(f"  -> Correlation importance: {len(importance)} features")
        else:
            print(f"  [FAIL] No importance data could be extracted")
            return False

    # Update qlib_results_enhanced.json
    enhanced_path = os.path.join(loop_dir, "qlib_results_enhanced.json")
    if os.path.exists(enhanced_path):
        with open(enhanced_path, "r", encoding="utf-8") as f:
            enhanced = json.load(f)
    else:
        enhanced = {}

    enhanced["factor_analysis"] = {"feature_importance": importance}

    with open(enhanced_path, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, ensure_ascii=False)
    print(f"  -> Updated {enhanced_path}")

    # Update DB if requested
    if update_db:
        _update_db_cache(task_id, loop_name, enhanced)

    return True


def _update_db_cache(task_id: str, loop_name: str, enhanced: dict):
    """Update qe_evolution_loops.metrics_json in AIStock DB."""
    try:
        import psycopg2
        import psycopg2.extras

        db_url = os.environ.get("AISTOCK_DB_URL") or os.environ.get("DATABASE_URL")
        if not db_url:
            # Load AIStock .env if present
            aistock_env = os.environ.get("AISTOCK_ENV_FILE", "F:/Dev/AIstock/.env")
            env_vars = {}
            if os.path.exists(aistock_env):
                with open(aistock_env, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip().strip('"').strip("'")
            # Prefer TDX_DB_* (AIStock convention), fall back to PG* standard
            db_host = env_vars.get("TDX_DB_HOST") or os.environ.get("PGHOST", "127.0.0.1")
            db_port = env_vars.get("TDX_DB_PORT") or os.environ.get("PGPORT", "5432")
            db_name = env_vars.get("TDX_DB_NAME") or os.environ.get("PGDATABASE", "aistock")
            db_user = env_vars.get("TDX_DB_USER") or os.environ.get("PGUSER", "postgres")
            db_pass = env_vars.get("TDX_DB_PASSWORD") or os.environ.get("PGPASSWORD")
            if not db_pass:
                raise RuntimeError("DB password not found in TDX_DB_PASSWORD / PGPASSWORD")
            db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

        loop_id = f"{task_id}_{loop_name}"
        fi_payload = json.dumps(
            {"feature_importance": enhanced.get("factor_analysis", {}).get("feature_importance", [])},
            ensure_ascii=False,
        )
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Two-step jsonb_set: ensure 'enhanced_metrics' key exists first,
                # then set the nested 'factor_analysis' inside it.
                # jsonb_set with create_missing=true only creates the LEAF key,
                # so we must guarantee the intermediate path exists.
                cur.execute(
                    """UPDATE qe_evolution_loops
                       SET metrics_json = jsonb_set(
                           jsonb_set(
                               COALESCE(metrics_json, '{}'::jsonb),
                               '{enhanced_metrics}',
                               COALESCE(metrics_json->'enhanced_metrics', '{}'::jsonb)
                           ),
                           '{enhanced_metrics,factor_analysis}',
                           %s::jsonb
                       ),
                       updated_at = NOW()
                       WHERE loop_id = %s""",
                    (fi_payload, loop_id),
                )
                if cur.rowcount > 0:
                    print(f"  -> DB updated: qe_evolution_loops.loop_id={loop_id}")
                else:
                    print(f"  -> DB: no row found for loop_id={loop_id}")
            conn.commit()
    except Exception as e:
        # Surface DB failures clearly — do NOT silently swallow.
        print(f"  -> [ERROR] DB update failed for {task_id}/{loop_name}: {e}")
        import traceback
        traceback.print_exc()


def process_task(task_id: str, update_db: bool = False, specific_loop: int | None = None):
    task_dir = os.path.join(QE_WORKSPACE, task_id)
    if not os.path.isdir(task_dir):
        print(f"  [SKIP] Task directory not found: {task_dir}")
        return 0, 0

    if specific_loop is not None:
        loops = [f"Loop{specific_loop}"]
    else:
        loops = sorted([d for d in os.listdir(task_dir) if d.startswith("Loop") and os.path.isdir(os.path.join(task_dir, d))])

    print(f"Task: {task_id}  ({len(loops)} loops)")
    success = 0
    for loop_name in loops:
        print(f"  {loop_name}...")
        if process_loop(task_id, loop_name, update_db=update_db):
            success += 1
    return success, len(loops)


def main():
    parser = argparse.ArgumentParser(description="Backfill feature importance for QE experiment(s)")
    parser.add_argument("task_id", nargs="?", help="QE task ID (e.g. qe_20260410_173242_d073); omit with --all")
    parser.add_argument("--all", action="store_true", help="Process all qe_* tasks in workspace")
    parser.add_argument("--update-db", action="store_true", help="Also update AIStock DB cache")
    parser.add_argument("--loop", type=int, default=None, help="Only process specific loop number (e.g. 1)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tasks whose enhanced.json already has factor_analysis")
    args = parser.parse_args()

    if args.all:
        task_ids = sorted([d for d in os.listdir(QE_WORKSPACE) if d.startswith("qe_") and os.path.isdir(os.path.join(QE_WORKSPACE, d))])
        print(f"Found {len(task_ids)} tasks in {QE_WORKSPACE}")
    elif args.task_id:
        task_ids = [args.task_id]
    else:
        print("Error: provide <task_id> or --all")
        sys.exit(1)

    total_success = 0
    total_loops = 0
    skipped = 0
    for tid in task_ids:
        # Skip-existing logic
        if args.skip_existing:
            has_data = False
            tdir = os.path.join(QE_WORKSPACE, tid)
            if os.path.isdir(tdir):
                for ln in os.listdir(tdir):
                    if not ln.startswith("Loop"):
                        continue
                    ep = os.path.join(tdir, ln, "qlib_results_enhanced.json")
                    if os.path.exists(ep):
                        try:
                            with open(ep, "r", encoding="utf-8") as f:
                                d = json.load(f)
                            fi = d.get("factor_analysis", {}).get("feature_importance", [])
                            if fi:
                                has_data = True
                                break
                        except Exception:
                            pass
            if has_data:
                print(f"Task: {tid}  [SKIP: factor_analysis already exists]")
                skipped += 1
                continue

        s, n = process_task(tid, update_db=args.update_db, specific_loop=args.loop)
        total_success += s
        total_loops += n
        print()

    print(f"Done. {total_success}/{total_loops} loops updated across {len(task_ids) - skipped} tasks ({skipped} skipped).")


if __name__ == "__main__":
    main()
