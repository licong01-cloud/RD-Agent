import re
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QTDockerEnv


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    @staticmethod
    def _should_fallback_lightgbm_gpu(stdout: str) -> bool:
        if not stdout:
            return False
        signatures = [
            "No OpenCL device found",
            "CUDA Tree Learner was not enabled",
            "GPU Tree Learner was not enabled",
            "OpenCL",
            "device_type=cuda",
            "device_type=gpu",
        ]
        return any(s in stdout for s in signatures)

    @staticmethod
    def _disable_lightgbm_gpu_in_config(conf_path: Path) -> bool:
        if not conf_path.exists():
            return False
        try:
            text = conf_path.read_text(encoding="utf-8")
        except Exception:
            return False

        old = text

        # Remove common LightGBM GPU-related kwargs lines.
        # Keep it simple and safe: delete these keys anywhere in the yaml.
        text = re.sub(r"^\s*device_type\s*:\s*gpu\s*$\n?", "", text, flags=re.M)
        text = re.sub(r"^\s*device_type\s*:\s*cuda\s*$\n?", "", text, flags=re.M)
        text = re.sub(r"^\s*device\s*:\s*gpu\s*$\n?", "", text, flags=re.M)
        text = re.sub(r"^\s*device\s*:\s*cuda\s*$\n?", "", text, flags=re.M)
        text = re.sub(r"^\s*gpu_use_dp\s*:\s*(true|false)\s*$\n?", "", text, flags=re.M | re.I)
        text = re.sub(r"^\s*max_bin\s*:\s*\d+\s*$\n?", "", text, flags=re.M)

        if text == old:
            return False

        conf_path.write_text(text, encoding="utf-8")
        return True

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs) -> str:
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv()
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            logger.error(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
            return None, "Unknown environment type"
        qtde.prepare()

        effective_env = dict(run_env or {})
        mlruns_dir = self.workspace_path / "mlruns"
        try:
            mlruns_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        effective_env.setdefault("MLFLOW_TRACKING_URI", str(mlruns_dir))

        # Temporarily force LightGBM to CPU mode to avoid GPU/OpenCL/CUDA backend issues.
        conf_path = self.workspace_path / qlib_config_name
        if conf_path.exists():
            try:
                conf_text = conf_path.read_text(encoding="utf-8")
            except Exception:
                conf_text = ""
            if "static_path:" in conf_text:
                raise ValueError(
                    f"Refusing to run qlib config '{qlib_config_name}' because it contains 'static_path:'; "
                    "this violates the requirement that each loop must backtest using the current loop's generated factors."
                )
        if self._disable_lightgbm_gpu_in_config(conf_path):
            logger.info("[LightGBM] Forced CPU mode by removing GPU/CUDA kwargs from config.")

        # Run the Qlib backtest
        execute_qlib_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=effective_env,
        )

        # Prefer GPU if configured, but allow CPU fallback when LightGBM GPU is unavailable.
        if self._should_fallback_lightgbm_gpu(execute_qlib_log):
            if self._disable_lightgbm_gpu_in_config(conf_path):
                logger.warning(
                    "[GPUFallback] Detected LightGBM GPU/OpenCL failure; removed LightGBM GPU params and retrying qrun on CPU."
                )
                execute_qlib_log = qtde.check_output(
                    local_path=str(self.workspace_path),
                    entry=f"qrun {qlib_config_name}",
                    env=effective_env,
                )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        execute_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=effective_env,
        )

        quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            logger.error("No result file found.")
            return None, execute_qlib_log

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            # Here, we ensure that the qlib experiment has run successfully before extracting information from execute_qlib_log using regex; otherwise, we keep the original experiment stdout.
            logger.info(
                f"[QlibFBWorkspace] Reading qlib results from: {qlib_res_path} (mtime={qlib_res_path.stat().st_mtime})"
            )
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            execute_qlib_log = "\n".join(matches)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            return None, execute_qlib_log
