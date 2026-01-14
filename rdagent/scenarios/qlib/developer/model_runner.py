import os
from pathlib import Path

import pandas as pd
import yaml

from rdagent.components.runner import CachedRunner
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import ModelEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import md5_hash
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment


def load_alpha_factors_from_yaml() -> list[str] | None:
    """
    从配置文件加载Alpha因子列表

    Returns:
        Alpha因子名称列表，如果加载失败则返回None
    """
    try:
        # Alpha因子配置文件路径
        alpha_config_path = Path(__file__).parent.parent / "experiment" / "model_template" / "conf_baseline_factors_model.yaml"
        
        if not alpha_config_path.exists():
            logger.warning(f"Alpha因子配置文件不存在: {alpha_config_path}")
            return None
        
        with open(alpha_config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式提取col_list，避免Jinja2模板解析问题
        import re
        # 匹配col_list: [...]格式
        match = re.search(r'col_list:\s*\[(.*?)\]', content, re.DOTALL)
        if match:
            # 提取因子名称
            factors_str = match.group(1)
            # 去除引号和空格，分割成列表
            alpha_factors = [f.strip().strip('"').strip("'") for f in factors_str.split(',')]
            alpha_factors = [f for f in alpha_factors if f]  # 过滤空字符串
            
            if alpha_factors:
                logger.info(f"成功加载{len(alpha_factors)}个Alpha因子: {alpha_factors[:5]}...")
                return alpha_factors
            else:
                logger.warning("配置文件中未找到Alpha因子列表")
                return None
        else:
            logger.warning("配置文件中未找到col_list配置")
            return None
            
    except Exception as e:
        logger.error(f"加载Alpha因子配置失败: {e}")
        return None


def qlib_model_cache_key(runner: "QlibModelRunner", exp: QlibModelExperiment) -> str | None:  # type: ignore[name-defined]
    return runner.get_cache_key(exp)


class QlibModelRunner(CachedRunner[QlibModelExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - Pytorch `model.py`
    - results in `mlflow`

    https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_nn.py
    - pt_model_uri:  hard-code `model.py:Net` in the config
    - let LLM modify model.py
    """

    def get_cache_key(self, exp: QlibModelExperiment) -> str | None:
        """Cache key sensitive to model implementation and factor inputs.

        - If env QLIB_QUANT_DISABLE_CACHE=1, disable cache.
        - Otherwise include:
          * default task-based key
          * current model.py source
          * combined_factors_df.parquet signature (size + mtime) if exists
          * training hyperparameters dict
        """

        if os.getenv("QLIB_QUANT_DISABLE_CACHE", "0") == "1":
            return None

        base_key = CachedRunner.get_cache_key(self, exp)
        parts: list[str] = [base_key]

        if exp.sub_workspace_list and exp.sub_workspace_list[0] is not None:
            sub_ws = exp.sub_workspace_list[0]
            if isinstance(sub_ws.file_dict, dict):
                model_src = sub_ws.file_dict.get("model.py")
                if isinstance(model_src, str) and model_src:
                    parts.append(model_src)

        ws_path = getattr(exp.experiment_workspace, "workspace_path", None)
        if isinstance(ws_path, Path):
            cf_path = ws_path / "combined_factors_df.parquet"
            if cf_path.exists():
                stat = cf_path.stat()
                parts.append(f"cf_size={stat.st_size}")
                parts.append(f"cf_mtime={stat.st_mtime}")

        training_hyperparameters = getattr(exp.sub_tasks[0], "training_hyperparameters", None)
        if isinstance(training_hyperparameters, dict) and training_hyperparameters:
            parts.append(str(sorted(training_hyperparameters.items())))

        return md5_hash("\n".join(parts))

    def _is_retryable_model_failure(self, stdout: str) -> bool:
        if not isinstance(stdout, str) or not stdout:
            return False
        s = stdout.lower()
        needles = [
            "this type of input is not supported",
            "notimplementederror",
            "_get_row_col",
            "step_len",
            "num_timesteps",
        ]
        return any(n in s for n in needles)

    def _execute_with_retry(
        self,
        exp: QlibModelExperiment,
        qlib_config_name: str,
        base_env: dict,
        attempts: list[dict],
    ):
        last_result, last_stdout = None, ""
        for idx, env_patch in enumerate(attempts, start=1):
            run_env = dict(base_env)
            run_env.update(env_patch)
            logger.info(f"[ModelRetry] attempt {idx}/{len(attempts)} run_env={env_patch}")
            result, stdout = exp.experiment_workspace.execute(qlib_config_name=qlib_config_name, run_env=run_env)
            last_result, last_stdout = result, stdout
            if result is not None:
                return result, stdout
            if not self._is_retryable_model_failure(stdout):
                break
        return last_result, last_stdout

    @cache_with_pickle(qlib_model_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        exist_sota_factor_exp = False
        if exp.based_experiments:
            SOTA_factor = None
            # Filter and retain only QlibFactorExperiment instances
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            # 修改：降低SOTA因子使用条件，从>1改为>=1，确保只要有SOTA因子就使用
            if len(sota_factor_experiments_list) >= 1:
                logger.info(f"SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            if SOTA_factor is not None and not SOTA_factor.empty:
                exist_sota_factor_exp = True
                combined_factors = SOTA_factor
                combined_factors = combined_factors.sort_index()
                combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
                new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
                combined_factors.columns = new_columns
                
                # 叠加Alpha因子
                use_alpha_factors = os.getenv("USE_ALPHA_FACTORS", "true") == "true"
                if use_alpha_factors:
                    alpha_factor_names = load_alpha_factors_from_yaml()
                    if alpha_factor_names:
                        logger.info(f"叠加Alpha因子: {len(alpha_factor_names)}个")
                        # 从SOTA因子中提取Alpha因子（如果存在）
                        alpha_factors_from_sota = []
                        for factor_name in alpha_factor_names:
                            if ("feature", factor_name) in combined_factors.columns:
                                alpha_factors_from_sota.append(combined_factors[("feature", factor_name)])
                        
                        # 记录叠加前后的因子数量
                        sota_factor_count = len([col for col in combined_factors.columns if col[0] == "feature"])
                        logger.info(f"SOTA因子数量: {sota_factor_count}")
                        
                        num_features = str(RD_AGENT_SETTINGS.initial_fator_library_size + len([col for col in combined_factors.columns if col[0] == "feature"]))
                    else:
                        num_features = str(RD_AGENT_SETTINGS.initial_fator_library_size + len([col for col in combined_factors.columns if col[0] == "feature"]))
                else:
                    num_features = str(RD_AGENT_SETTINGS.initial_fator_library_size + len([col for col in combined_factors.columns if col[0] == "feature"]))

                target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

                # Save the combined factors to the workspace
                combined_factors.to_parquet(target_path, engine="pyarrow")

        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        training_hyperparameters = exp.sub_tasks[0].training_hyperparameters
        if training_hyperparameters:
            env_to_use.update(
                {
                    "n_epochs": str(training_hyperparameters.get("n_epochs", "100")),
                    "lr": str(training_hyperparameters.get("lr", "2e-4")),
                    "early_stop": str(training_hyperparameters.get("early_stop", 10)),
                    "batch_size": str(training_hyperparameters.get("batch_size", 256)),
                    "weight_decay": str(training_hyperparameters.get("weight_decay", 0.0001)),
                }
            )
        else:
            # Default settings tuned for end-to-end pipeline stability.
            # Long PT trainings can take hours and are often interrupted, leaving no artifacts for read_exp_res.py.
            env_to_use.update(
                {
                    "n_epochs": "20",
                    "lr": "1e-3",
                    "early_stop": "5",
                    "batch_size": "256",
                    "weight_decay": "1e-4",
                }
            )

        logger.info(f"start to run {exp.sub_tasks[0].name} model")

        model_type = getattr(exp.sub_tasks[0], "model_type", None)
        if model_type not in ("TimeSeries", "Tabular"):
            raise ModelEmptyError(
                f"Unsupported model_type '{model_type}'. It must be either 'TimeSeries' or 'Tabular' so that "
                "the runner can choose a compatible dataset format (TSDatasetH vs DatasetH)."
            )

        if model_type == "TimeSeries":
            qlib_config_name = "conf_sota_factors_model.yaml" if exist_sota_factor_exp else "conf_baseline_factors_model.yaml"
            attempts = [
                {"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20, **({"num_features": num_features} if exist_sota_factor_exp else {})},
                {"dataset_cls": "DatasetH", "step_len": 20, "num_timesteps": 20, **({"num_features": num_features} if exist_sota_factor_exp else {})},
                {"dataset_cls": "TSDatasetH", "step_len": 60, "num_timesteps": 60, **({"num_features": num_features} if exist_sota_factor_exp else {})},
            ]
            result, stdout = self._execute_with_retry(exp, qlib_config_name=qlib_config_name, base_env=env_to_use, attempts=attempts)
        elif model_type == "Tabular":
            qlib_config_name = "conf_sota_factors_model.yaml" if exist_sota_factor_exp else "conf_baseline_factors_model.yaml"
            attempts = [
                {"dataset_cls": "DatasetH", **({"num_features": num_features} if exist_sota_factor_exp else {})},
                {"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20, **({"num_features": num_features} if exist_sota_factor_exp else {})},
                {"dataset_cls": "TSDatasetH", "step_len": 60, "num_timesteps": 60, **({"num_features": num_features} if exist_sota_factor_exp else {})},
            ]
            result, stdout = self._execute_with_retry(exp, qlib_config_name=qlib_config_name, base_env=env_to_use, attempts=attempts)

        exp.result = result
        exp.stdout = stdout

        if result is None:
            ws_path = getattr(getattr(exp, "experiment_workspace", None), "workspace_path", None)
            ws_info = f" (experiment_workspace={ws_path})" if ws_path is not None else ""
            logger.error(f"Failed to run {exp.sub_tasks[0].name}{ws_info}, because {stdout}")
            raise ModelEmptyError(f"Failed to run {exp.sub_tasks[0].name} model{ws_info}, because {stdout}")

        return exp
