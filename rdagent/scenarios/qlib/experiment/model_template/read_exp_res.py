import pickle
import os
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
    except Exception as e:
        print(f"Warning: failed to load portfolio_analysis/report_normal_1day.pkl: {e}")
