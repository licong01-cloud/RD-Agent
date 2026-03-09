import pickle
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import qlib
import yaml
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

_conf_path = Path(__file__).resolve().parent / "conf_baseline.yaml"
if not _conf_path.exists():
    _conf_path = Path(__file__).resolve().parent / "conf.yaml"
if not _conf_path.exists():
    _conf_path = Path.cwd() / "conf_baseline.yaml"
if not _conf_path.exists():
    _conf_path = Path.cwd() / "conf.yaml"

_provider_uri = None
_region = None
if _conf_path.exists():
    try:
        _conf_obj = yaml.safe_load(_conf_path.read_text(encoding="utf-8"))
        _qi = (_conf_obj or {}).get("qlib_init", {})
        _provider_uri = _qi.get("provider_uri")
        _region = _qi.get("region")
    except Exception:
        _provider_uri = None
        _region = None

if _provider_uri and _region:
    qlib.init(provider_uri=_provider_uri, region=_region)
else:
    qlib.init()

from qlib.workflow import R

# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

# TODO: list all the recorder and metrics

_cwd = Path.cwd()
_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not _tracking_uri:
    _local_mlruns = _cwd / "mlruns"
    if _local_mlruns.exists():
        os.environ["MLFLOW_TRACKING_URI"] = str(_local_mlruns)

# Assuming you have already listed the experiments
experiments = R.list_experiments()

# Iterate through each experiment to find the latest recorder
experiment_name = None
latest_recorder = None
for experiment in experiments:
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is not None:
            experiment_name = experiment
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            end_time = recorder.info["end_time"]
            try:
                # Check if the recorder has a valid end time
                if end_time is not None:
                    if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                        latest_recorder = recorder
                else:
                    print(f"Warning: Recorder {recorder_id} has no valid end time")
            except Exception as e:
                print(f"Error: {e}")

# Check if the latest recorder is found
if latest_recorder is None:
    print("No recorders found")
else:
    print(f"Latest recorder: {latest_recorder}")

    # Load the specified file from the latest recorder
    metrics = pd.Series(latest_recorder.list_metrics())

    output_path = Path.cwd() / "qlib_res.csv"
    metrics.to_csv(output_path)

    print(f"Output has been saved to {output_path}")

    try:
        ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        ret_data_frame.to_pickle("ret.pkl")

        def _normalize_ret_df(obj: object) -> pd.DataFrame:
            if isinstance(obj, pd.Series):
                df = obj.to_frame(name=obj.name or "value")
            elif isinstance(obj, pd.DataFrame):
                df = obj
            else:
                try:
                    df = pd.DataFrame(obj)  # type: ignore[arg-type]
                except Exception:
                    df = pd.DataFrame({"value": [obj]})

            if isinstance(df.index, pd.MultiIndex):
                idx_names = [n if n else f"index_{i}" for i, n in enumerate(df.index.names)]
                df = df.reset_index()
                for n in idx_names:
                    if n in df.columns:
                        if pd.api.types.is_datetime64_any_dtype(df[n]):
                            df[n] = pd.to_datetime(df[n], utc=True, errors="coerce")
            else:
                idx_name = df.index.name if df.index.name else "index"
                df = df.reset_index().rename(columns={"index": idx_name})
                if idx_name in df.columns and pd.api.types.is_datetime64_any_dtype(df[idx_name]):
                    df[idx_name] = pd.to_datetime(df[idx_name], utc=True, errors="coerce")

            for c in list(df.columns):
                if df[c].dtype == object:
                    try:
                        df[c] = pd.to_numeric(df[c], errors="ignore")
                    except Exception:
                        pass
            return df

        ret_schema_df = _normalize_ret_df(ret_data_frame)

        try:
            ret_schema_df.to_parquet("ret_schema.parquet", index=False)
        except Exception as e:
            print(f"Warning: failed to write ret_schema.parquet: {e}")

        try:
            Path("ret_schema.json").write_text(
                ret_schema_df.to_json(orient="table", date_format="iso"),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Warning: failed to write ret_schema.json: {e}")

        try:
            pred_obj = latest_recorder.load_object("pred.pkl")

            def _normalize_pred_df(obj: object) -> pd.DataFrame:
                if isinstance(obj, pd.Series):
                    df = obj.to_frame(name=obj.name or "score")
                elif isinstance(obj, pd.DataFrame):
                    df = obj
                else:
                    df = pd.DataFrame(obj)  # type: ignore[arg-type]

                if "score" not in df.columns:
                    if df.shape[1] >= 1:
                        df = df.rename(columns={df.columns[0]: "score"})
                    else:
                        df["score"] = pd.NA

                if isinstance(df.index, pd.MultiIndex):
                    idx_names = [n if n else f"index_{i}" for i, n in enumerate(df.index.names)]
                    df = df.reset_index()
                    cols = set(df.columns)
                    if "datetime" not in cols:
                        for n in idx_names:
                            if "date" in str(n).lower() and n in cols:
                                df = df.rename(columns={n: "datetime"})
                                break
                    if "instrument" not in cols:
                        for n in idx_names:
                            if "inst" in str(n).lower() and n in cols:
                                df = df.rename(columns={n: "instrument"})
                                break
                else:
                    idx_name = df.index.name if df.index.name else "index"
                    df = df.reset_index().rename(columns={"index": idx_name})

                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
                return df

            pred_df = _normalize_pred_df(pred_obj)

            _topk = 50
            _n_drop = 0
            try:
                _pa = (_conf_obj or {}).get("port_analysis_config", {})
                _st = (_pa or {}).get("strategy", {})
                _kw = (_st or {}).get("kwargs", {})
                if isinstance(_kw, dict):
                    _topk = int(_kw.get("topk", _topk))
                    _n_drop = int(_kw.get("n_drop", _n_drop))
            except Exception:
                _topk = 50
                _n_drop = 0

            if "datetime" not in pred_df.columns or "instrument" not in pred_df.columns:
                raise ValueError("pred.pkl missing required index columns for signals (datetime/instrument)")

            pred_df = pred_df[["datetime", "instrument", "score"]].copy()
            pred_df = pred_df.dropna(subset=["datetime", "instrument"]).copy()

            pred_df["trade_date"] = pred_df["datetime"].dt.date.astype(str)
            pred_df["score"] = pd.to_numeric(pred_df["score"], errors="coerce")

            pred_df["rank"] = (
                pred_df.groupby("trade_date")["score"].rank(ascending=False, method="first").astype("Int64")
            )

            topk_df = pred_df[pred_df["rank"].notna() & (pred_df["rank"] <= _topk)].copy()

            topk_df["signal"] = topk_df["score"]
            topk_df["target_weight"] = 1.0 / float(_topk) if _topk > 0 else 0.0
            topk_df["target_position"] = pd.NA
            topk_df["price_ref"] = pd.NA
            topk_df["universe_flag"] = 1
            topk_df["pred_return"] = topk_df["score"]
            topk_df["confidence"] = pd.NA
            topk_df["volatility_est"] = pd.NA
            topk_df["max_weight"] = pd.NA
            topk_df["min_weight"] = pd.NA
            topk_df["sector"] = pd.NA
            topk_df["industry"] = pd.NA

            now_utc = datetime.now(tz=timezone.utc).isoformat()
            topk_df["generated_at_utc"] = now_utc
            topk_df["task_run_id"] = pd.NA
            topk_df["loop_id"] = pd.NA
            topk_df["workspace_id"] = pd.NA
            topk_df["model_version"] = pd.NA

            topk_df["weight_method"] = "topk_equal_weight"
            topk_df["topk"] = _topk
            topk_df["n_drop"] = _n_drop
            topk_df["rebalance_freq"] = "1d"

            signals_cols = [
                "trade_date",
                "instrument",
                "signal",
                "target_weight",
                "target_position",
                "price_ref",
                "universe_flag",
                "score",
                "rank",
                "pred_return",
                "confidence",
                "volatility_est",
                "max_weight",
                "min_weight",
                "sector",
                "industry",
                "generated_at_utc",
                "task_run_id",
                "loop_id",
                "workspace_id",
                "model_version",
            ]
            weight_meta_cols = ["weight_method", "topk", "n_drop", "rebalance_freq"]

            signals_df = topk_df[signals_cols + weight_meta_cols].copy()

            try:
                signals_df.to_parquet("signals.parquet", index=False)
            except Exception as e:
                print(f"Warning: failed to write signals.parquet: {e}")

            try:
                Path("signals.json").write_text(
                    signals_df.to_json(orient="table", date_format="iso"),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"Warning: failed to write signals.json: {e}")
        except Exception as e:
            print(f"Warning: failed to generate signals from pred.pkl: {e}")
    except Exception as e:
        print(f"Warning: failed to load portfolio_analysis/report_normal_1day.pkl: {e}")

# ============================================================
# Phase 3: Enhanced Diagnostics Output
# ============================================================
import json as _json
import re as _re
import math as _math

import numpy as _np


def _json_safe(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        v = float(obj)
        if _math.isnan(v) or _math.isinf(v):
            return None
        return v
    if isinstance(obj, _np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float):
        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return obj
    return obj


def _extract_ic_diagnostics(recorder) -> dict:
    """Extract IC time series diagnostics from sig_analysis pkl files."""
    result = {}
    try:
        ic_obj = recorder.load_object("sig_analysis/ic.pkl")
        if isinstance(ic_obj, pd.Series):
            ic_s = ic_obj.dropna()
        elif isinstance(ic_obj, pd.DataFrame):
            ic_s = ic_obj.iloc[:, 0].dropna()
        else:
            ic_s = pd.Series(ic_obj).dropna()

        dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in ic_s.index]
        vals = ic_s.values.astype(float)
        rolling_mean = pd.Series(vals).rolling(30, min_periods=5).mean().tolist()
        rolling_std = pd.Series(vals).rolling(30, min_periods=5).std().tolist()

        result["ic_series"] = vals.tolist()
        result["ic_dates"] = dates
        result["ic_mean"] = float(vals.mean())
        result["ic_std"] = float(vals.std())
        result["ic_positive_ratio"] = float((vals > 0).sum() / len(vals)) if len(vals) > 0 else None
        result["ic_rolling_30d_mean"] = rolling_mean
        result["ic_rolling_30d_std"] = rolling_std
    except Exception as e:
        print(f"Warning: failed to extract IC diagnostics: {e}")

    try:
        ric_obj = recorder.load_object("sig_analysis/ric.pkl")
        if isinstance(ric_obj, pd.Series):
            ric_s = ric_obj.dropna()
        elif isinstance(ric_obj, pd.DataFrame):
            ric_s = ric_obj.iloc[:, 0].dropna()
        else:
            ric_s = pd.Series(ric_obj).dropna()

        ric_vals = ric_s.values.astype(float)
        result["rank_ic_series"] = ric_vals.tolist()
        result["rank_ic_mean"] = float(ric_vals.mean())
        result["rank_ic_std"] = float(ric_vals.std())
        result["rank_ic_positive_ratio"] = (
            float((ric_vals > 0).sum() / len(ric_vals)) if len(ric_vals) > 0 else None
        )
    except Exception as e:
        print(f"Warning: failed to extract Rank IC diagnostics: {e}")

    return result


def _extract_training_diagnostics() -> dict:
    """Extract training loss curves from stdout/log files via regex."""
    result = {}
    # Search for log files in current directory
    log_candidates = list(Path.cwd().glob("*.log")) + list(Path.cwd().glob("*.txt"))
    # Also check stdout captured by RD-Agent
    for name in ("stdout.log", "output.log", "run.log", "exp_output.log"):
        p = Path.cwd() / name
        if p.exists() and p not in log_candidates:
            log_candidates.append(p)

    all_text = ""
    for lf in log_candidates:
        try:
            all_text += lf.read_text(encoding="utf-8", errors="replace") + "\n"
        except Exception:
            pass

    if not all_text:
        return result

    # Pattern 1: PyTorch-style "Epoch X: train_loss=Y, valid_loss=Z"
    train_losses, val_losses = [], []
    for m in _re.finditer(
        r"[Ee]poch\s*(\d+).*?train[_ ]?loss[=:\s]+([\d.]+).*?val(?:id)?[_ ]?loss[=:\s]+([\d.]+)",
        all_text,
    ):
        train_losses.append(float(m.group(2)))
        val_losses.append(float(m.group(3)))

    # Pattern 2: qlib TabNet/ALSTM style "[Epoch N]: train_loss=X, valid_loss=Y"
    if not train_losses:
        for m in _re.finditer(
            r"\[Epoch\s+(\d+)\].*?train_loss[=:\s]+([\d.eE+-]+).*?valid_loss[=:\s]+([\d.eE+-]+)",
            all_text,
        ):
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))

    # Pattern 3: generic "epoch N ... loss X ... val Y"
    if not train_losses:
        for m in _re.finditer(
            r"epoch[:\s]+(\d+).*?loss[:\s]+([\d.eE+-]+).*?(?:val|test)[_ ]?loss[:\s]+([\d.eE+-]+)",
            all_text, _re.IGNORECASE,
        ):
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))

    # Pattern 4: GeneralPTNN style "Epoch0: train 0.994649, valid 0.992367" (single line)
    if not train_losses:
        for m in _re.finditer(
            r"[Ee]poch\s*(\d+):\s*train\s+([\d.eE+-]+),?\s*valid\s+([\d.eE+-]+)",
            all_text,
        ):
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))

    # Pattern 5: Most Qlib models (ALSTM/GRU/LSTM/GATs/TCN/Transformer/TabNet/etc.)
    # print "train X, valid Y" on a standalone line (Epoch header is on a separate line)
    if not train_losses:
        for m in _re.finditer(
            r"^train\s+([\d.eE+-]+),?\s*valid\s+([\d.eE+-]+)",
            all_text, _re.MULTILINE,
        ):
            train_losses.append(float(m.group(1)))
            val_losses.append(float(m.group(2)))

    if train_losses:
        result["train_loss_curve"] = train_losses
        result["val_loss_curve"] = val_losses
        result["total_epochs"] = len(train_losses)
        best_epoch = int(_np.argmin(val_losses)) + 1 if val_losses else len(train_losses)
        result["best_epoch"] = best_epoch
        result["final_train_loss"] = train_losses[-1]
        result["final_val_loss"] = val_losses[-1] if val_losses else None

        if train_losses[-1] > 0 and val_losses:
            result["overfit_ratio"] = round(val_losses[-1] / train_losses[-1], 4)
        else:
            result["overfit_ratio"] = None

        result["convergence_ratio"] = round(best_epoch / len(train_losses), 4)

        # Check early stopping
        result["early_stop_triggered"] = False
        if _re.search(r"early.?stop", all_text, _re.IGNORECASE):
            result["early_stop_triggered"] = True

    return result


def _extract_return_curves(recorder) -> dict:
    """Extract cumulative return curves from report_normal_1day.pkl."""
    result = {}
    try:
        report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        if isinstance(report, pd.DataFrame):
            df = report
        elif isinstance(report, pd.Series):
            df = report.to_frame()
        else:
            return result

        # Reset index to get dates
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # Find date column
        date_col = None
        for c in df.columns:
            if "date" in str(c).lower() or pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break
        if date_col is None and df.shape[0] > 0:
            # Use index as dates
            dates = [str(i) for i in range(len(df))]
        else:
            dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in df[date_col]]

        result["dates"] = dates

        # Extract return columns (qlib report format)
        col_map = {
            "excess_return_without_cost": "cumulative_excess_no_cost",
            "excess_return_with_cost": "cumulative_excess_with_cost",
            "return": "cumulative_benchmark",
        }
        for src, dst in col_map.items():
            matched = [c for c in df.columns if src in str(c).lower()]
            if matched:
                series = pd.to_numeric(df[matched[0]], errors="coerce").fillna(0)
                result[dst] = (1 + series).cumprod().subtract(1).tolist()

        # Compute drawdown from excess_with_cost
        if "cumulative_excess_with_cost" in result:
            cum = _np.array(result["cumulative_excess_with_cost"])
            running_max = _np.maximum.accumulate(cum + 1)
            dd = (cum + 1) / running_max - 1
            result["drawdown_series"] = dd.tolist()

    except Exception as e:
        print(f"Warning: failed to extract return curves: {e}")
    return result


def _extract_trade_diagnostics(recorder, summary_metrics: dict) -> dict:
    """Extract trading efficiency diagnostics from positions and report."""
    result = {}
    try:
        # Get annualized returns with/without cost from summary
        ann_no_cost = summary_metrics.get("1day.excess_return_without_cost.annualized_return")
        ann_with_cost = summary_metrics.get("1day.excess_return_with_cost.annualized_return")
        if ann_no_cost is not None and ann_with_cost is not None:
            result["cost_drag_annualized"] = round(float(ann_no_cost) - float(ann_with_cost), 6)

        # Load positions for turnover analysis
        pos_obj = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        if isinstance(pos_obj, pd.DataFrame):
            pos_df = pos_obj
        elif isinstance(pos_obj, pd.Series):
            pos_df = pos_obj.to_frame()
        else:
            pos_df = pd.DataFrame(pos_obj)

        if isinstance(pos_df.index, pd.MultiIndex):
            pos_df = pos_df.reset_index()

        # Compute average turnover from position changes
        if "instrument" in [str(c).lower() for c in pos_df.columns]:
            inst_col = [c for c in pos_df.columns if "inst" in str(c).lower()][0]
            date_col = [c for c in pos_df.columns if "date" in str(c).lower()]
            if date_col:
                date_col = date_col[0]
                daily_counts = pos_df.groupby(date_col)[inst_col].nunique()
                result["daily_trade_count_avg"] = round(float(daily_counts.mean()), 2)

        # Use turnover from summary if available
        turnover_key = [k for k in summary_metrics if "turnover" in str(k).lower()]
        if turnover_key:
            result["avg_turnover"] = float(summary_metrics[turnover_key[0]])

    except Exception as e:
        print(f"Warning: failed to extract trade diagnostics: {e}")
    return result


def _extract_prediction_diagnostics(recorder) -> dict:
    """Extract prediction behavior diagnostics from pred.pkl."""
    result = {}
    try:
        pred_obj = recorder.load_object("pred.pkl")
        if isinstance(pred_obj, pd.Series):
            pdf = pred_obj.to_frame(name="score")
        elif isinstance(pred_obj, pd.DataFrame):
            pdf = pred_obj
        else:
            pdf = pd.DataFrame(pred_obj)

        if "score" not in pdf.columns and pdf.shape[1] >= 1:
            pdf = pdf.rename(columns={pdf.columns[0]: "score"})

        scores = pd.to_numeric(pdf["score"], errors="coerce").dropna()
        result["pred_std"] = round(float(scores.std()), 6)

        # Prediction autocorrelation (1-day lag)
        if isinstance(pdf.index, pd.MultiIndex):
            pdf_flat = pdf.reset_index()
            date_cols = [c for c in pdf_flat.columns if "date" in str(c).lower()]
            inst_cols = [c for c in pdf_flat.columns if "inst" in str(c).lower()]
            if date_cols and inst_cols:
                dc, ic = date_cols[0], inst_cols[0]
                pivoted = pdf_flat.pivot_table(
                    index=dc, columns=ic, values="score", aggfunc="first"
                )
                # Rank turnover: correlation of ranks between consecutive days
                ranks = pivoted.rank(axis=1)
                rank_corrs = []
                for i in range(1, len(ranks)):
                    prev = ranks.iloc[i - 1].dropna()
                    curr = ranks.iloc[i].dropna()
                    common = prev.index.intersection(curr.index)
                    if len(common) > 5:
                        rank_corrs.append(float(prev[common].corr(curr[common])))
                if rank_corrs:
                    result["pred_autocorr_1d"] = round(float(_np.mean(rank_corrs)), 4)
                    result["pred_rank_turnover"] = round(1.0 - float(_np.mean(rank_corrs)), 4)

                # Top30 stability
                top30_sets = []
                for i in range(len(ranks)):
                    row = ranks.iloc[i].dropna().nlargest(30)
                    top30_sets.append(set(row.index))
                if len(top30_sets) > 1:
                    overlaps = []
                    for i in range(1, len(top30_sets)):
                        overlap = len(top30_sets[i] & top30_sets[i - 1]) / 30.0
                        overlaps.append(overlap)
                    result["top30_stability"] = round(float(_np.mean(overlaps)), 4)

    except Exception as e:
        print(f"Warning: failed to extract prediction diagnostics: {e}")
    return result


def _generate_llm_summary(enhanced: dict) -> dict:
    """Generate a compact LLM-friendly summary (no long time series)."""
    summary = {}

    # Copy original summary metrics
    if "summary" in enhanced:
        summary["summary"] = enhanced["summary"]

    # IC diagnostics summary (no series)
    ic = enhanced.get("ic_diagnostics", {})
    if ic:
        ic_mean = ic.get("ic_mean")
        ic_std = ic.get("ic_std")
        icir = round(ic_mean / ic_std, 4) if ic_mean and ic_std and ic_std > 0 else None

        # Determine IC trend from rolling mean
        rolling = ic.get("ic_rolling_30d_mean", [])
        valid_rolling = [v for v in rolling if v is not None]
        trend = "stable"
        if len(valid_rolling) >= 10:
            recent = _np.mean(valid_rolling[-10:])
            earlier = _np.mean(valid_rolling[:10])
            if recent > earlier * 1.1:
                trend = "rising"
            elif recent < earlier * 0.9:
                trend = "declining"

        summary["ic_diagnostics_summary"] = {
            "ic_mean": ic.get("ic_mean"),
            "ic_std": ic.get("ic_std"),
            "ic_positive_ratio": ic.get("ic_positive_ratio"),
            "ic_stability_score": icir,
            "rank_ic_mean": ic.get("rank_ic_mean"),
            "ic_trend": trend,
        }

    # Training diagnostics summary
    td = enhanced.get("training_diagnostics", {})
    if td:
        diag_parts = []
        ofr = td.get("overfit_ratio")
        if ofr is not None:
            if ofr > 1.3:
                diag_parts.append("严重过拟合")
            elif ofr > 1.1:
                diag_parts.append("轻微过拟合")
            else:
                diag_parts.append("拟合正常")
        cr = td.get("convergence_ratio")
        if cr is not None:
            if cr < 0.3:
                diag_parts.append("过早收敛")
            elif cr > 0.95:
                diag_parts.append("可能未充分收敛")
            else:
                diag_parts.append("收敛正常")

        summary["training_diagnostics_summary"] = {
            "overfit_ratio": ofr,
            "convergence_ratio": cr,
            "early_stop_triggered": td.get("early_stop_triggered"),
            "diagnosis": "，".join(diag_parts) if diag_parts else None,
        }

    # Trade diagnostics summary
    trd = enhanced.get("trade_diagnostics", {})
    if trd:
        diag_parts = []
        turnover = trd.get("avg_turnover")
        if turnover is not None and turnover > 0.5:
            diag_parts.append(f"换手率过高({turnover:.2f}>0.5)")
        cost_drag = trd.get("cost_drag_annualized")
        if cost_drag is not None and cost_drag > 0.05:
            diag_parts.append(f"成本侵蚀严重({cost_drag*100:.1f}pp)")

        summary["trade_diagnostics_summary"] = {
            "avg_turnover": turnover,
            "cost_drag": cost_drag,
            "diagnosis": "，".join(diag_parts) if diag_parts else "交易效率正常",
        }

    # Prediction diagnostics summary
    pred = enhanced.get("prediction_diagnostics", {})
    if pred:
        stability = pred.get("top30_stability")
        diag = "预测排名稳定" if stability and stability > 0.7 else "预测排名中等稳定" if stability and stability > 0.4 else "预测排名不稳定"
        summary["prediction_diagnostics_summary"] = {
            "pred_rank_turnover": pred.get("pred_rank_turnover"),
            "top30_stability": stability,
            "diagnosis": diag,
        }

    # Auto prescription
    issues = []
    recommendations = []
    dims = []

    if trd.get("avg_turnover") and trd["avg_turnover"] > 0.5:
        issues.append("high_turnover")
        dims.append("loss_function")
        recommendations.append("切换到Huber Loss或增加换手率惩罚项")

    ic_summary = summary.get("ic_diagnostics_summary", {})
    if ic_summary.get("ic_trend") == "declining":
        issues.append("ic_unstable")
        dims.append("non_stationarity")
        recommendations.append("增加RevIN处理因子非平稳性")

    td_summary = summary.get("training_diagnostics_summary", {})
    if td_summary.get("overfit_ratio") and td_summary["overfit_ratio"] > 1.3:
        issues.append("overfitting")
        dims.append("regularization")
        recommendations.append("增加Dropout或L2正则化")

    summary["auto_prescription"] = {
        "issues": issues,
        "recommended_dimensions": dims,
        "recommended_actions": recommendations,
    }

    return summary


# ============================================================
# Generate enhanced output files
# ============================================================
if latest_recorder is not None:
    try:
        _summary_dict = metrics.to_dict() if metrics is not None else {}

        _enhanced = {"summary": _summary_dict}
        _enhanced["ic_diagnostics"] = _extract_ic_diagnostics(latest_recorder)
        _enhanced["training_diagnostics"] = _extract_training_diagnostics()
        _enhanced["return_curves"] = _extract_return_curves(latest_recorder)
        _enhanced["trade_diagnostics"] = _extract_trade_diagnostics(latest_recorder, _summary_dict)
        _enhanced["prediction_diagnostics"] = _extract_prediction_diagnostics(latest_recorder)

        # Write full enhanced file (with time series, for dashboard charts)
        _enhanced_path = Path.cwd() / "qlib_results_enhanced.json"
        _enhanced_path.write_text(
            _json.dumps(_json_safe(_enhanced), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Enhanced results saved to {_enhanced_path}")

        # Write compact LLM summary (no long series, for feedback prompt)
        _llm_summary = _generate_llm_summary(_enhanced)
        _llm_path = Path.cwd() / "qlib_results_llm.json"
        _llm_path.write_text(
            _json.dumps(_json_safe(_llm_summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"LLM summary saved to {_llm_path}")

    except Exception as e:
        print(f"Warning: failed to generate enhanced output files: {e}")
