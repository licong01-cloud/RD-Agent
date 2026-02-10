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
