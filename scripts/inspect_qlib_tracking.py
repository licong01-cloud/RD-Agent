import json
import os
from pathlib import Path


def _safe_repr(obj) -> str:
    try:
        return repr(obj)
    except Exception:
        return f"<{type(obj).__name__}: repr_failed>"


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    return _safe_repr(obj)


def main() -> None:
    out = {
        "env": {
            "PWD": os.getcwd(),
            "HOME": os.path.expanduser("~"),
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI"),
            "QLIB_LOG_DIR": os.environ.get("QLIB_LOG_DIR"),
        },
        "mlflow": None,
        "qlib": None,
        "qlib_R": None,
    }

    # MLflow
    try:
        import mlflow  # type: ignore

        out["mlflow"] = {
            "tracking_uri": mlflow.get_tracking_uri(),
        }
    except Exception as e:
        out["mlflow"] = {"error": f"{type(e).__name__}: {e}"}

    # Qlib + R
    try:
        import qlib  # type: ignore

        # Try init with defaults; if user has provider_uri/region in cwd conf.yaml, their read_exp_res already handles it.
        # Here we avoid making assumptions and just init default.
        qlib.init()

        from qlib.workflow import R  # type: ignore

        out["qlib"] = {
            "provider_uri": getattr(getattr(qlib, "config", None), "C", {}).get("provider_uri")
            if hasattr(qlib, "config")
            else None,
            "region": getattr(getattr(qlib, "config", None), "C", {}).get("region") if hasattr(qlib, "config") else None,
            "exp_manager": _safe_repr(getattr(R, "exp_manager", None)),
        }

        # Try to list experiments/recorders and sample a recorder info
        try:
            experiments_raw = R.list_experiments()
        except Exception as e:
            experiments_raw = f"{type(e).__name__}: {e}"

        if isinstance(experiments_raw, list):
            experiments = [str(x) for x in experiments_raw]
        else:
            experiments = experiments_raw

        out["qlib_R"] = {
            "experiments": experiments,
        }

        if isinstance(experiments_raw, list) and experiments_raw:
            exp0 = experiments[0]
            try:
                rec_ids_raw = R.list_recorders(experiment_name=exp0)
            except Exception as e:
                rec_ids_raw = f"{type(e).__name__}: {e}"

            if isinstance(rec_ids_raw, list):
                rec_ids = [str(x) for x in rec_ids_raw]
            else:
                rec_ids = rec_ids_raw

            out["qlib_R"]["sample_experiment"] = exp0
            out["qlib_R"]["sample_recorders"] = rec_ids

            # Try to open one recorder and dump key info
            if isinstance(rec_ids_raw, list) and rec_ids_raw:
                rid0 = rec_ids[0]
                try:
                    rec = R.get_recorder(recorder_id=rid0, experiment_name=exp0)
                    info = getattr(rec, "info", None)
                    out["qlib_R"]["sample_recorder_info"] = info

                    # Try MLflow artifact uri if this is MLflow-based recorder
                    try:
                        client = getattr(rec, "client", None)
                        out["qlib_R"]["sample_recorder_client"] = _safe_repr(client)
                    except Exception:
                        pass
                except Exception as e:
                    out["qlib_R"]["sample_recorder_error"] = f"{type(e).__name__}: {e}"

    except Exception as e:
        out["qlib"] = {"error": f"{type(e).__name__}: {e}"}

    print(json.dumps(out, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
