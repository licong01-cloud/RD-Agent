from pathlib import Path
import os

import pandas as pd
from pandarallel import pandarallel

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.oai.llm_utils import APIBackend

pandarallel.initialize(verbose=1)

from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment


def qlib_factor_cache_key(runner: "QlibFactorRunner", exp: QlibFactorExperiment) -> str | None:  # type: ignore[name-defined]
    return runner.get_cache_key(exp)

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()

# class QlibFactorExpWorkspace:

#     def prepare():
#         # create a folder;
#         # copy template
#         # place data inside the folder `combined_factors`
#         #
#     def execute():
#         de = DockerEnv()
#         de.run(local_path=self.ws_path, entry="qrun conf_baseline.yaml")

# TODO: supporting multiprocessing and keep previous results


class QlibFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`
    """

    def calculate_information_coefficient(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    ) -> pd.DataFrame:
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1
                ].corr(concat_feature.iloc[:, col2])
        return res

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        # calculate the IC between each column of SOTA_feature and new_feature
        # if the IC is larger than a threshold, remove the new_feature column
        # return the new_feature

        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        IC_max = (
            concat_feature.groupby("datetime")
            .parallel_apply(
                lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
            )
            .mean()
        )
        IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        IC_max = IC_max.unstack().max(axis=0)
        # Relax de-duplication: only drop nearly-identical new factors.
        keep_idx = IC_max[IC_max < 0.995].index
        # If all new features are considered too similar, keep them all to ensure
        # the round still contributes genuinely new factors.
        if len(keep_idx) == 0:
            return new_feature
        return new_feature.iloc[:, keep_idx]

    def get_cache_key(self, exp: QlibFactorExperiment) -> str | None:
        """Cache key sensitive to factor implementation changes.

        - If env QLIB_QUANT_DISABLE_CACHE=1, disable cache.
        - Otherwise include:
          * default task-based key
          * all current factor.py sources
          * combined_factors_df.parquet signature (size + mtime) if exists
        """

        if os.getenv("QLIB_QUANT_DISABLE_CACHE", "0") == "1":
            return None

        base_key = CachedRunner.get_cache_key(self, exp)
        parts: list[str] = [base_key]

        for sub_ws in getattr(exp, "sub_workspace_list", []) or []:
            if sub_ws and isinstance(sub_ws.file_dict, dict):
                factor_src = sub_ws.file_dict.get("factor.py")
                if isinstance(factor_src, str) and factor_src:
                    parts.append(factor_src)

        cf_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"
        if cf_path.exists():
            stat = cf_path.stat()
            parts.append(f"cf_size={stat.st_size}")
            parts.append(f"cf_mtime={stat.st_mtime}")

        return md5_hash("\n".join(parts))

    def _get_factor_max_repair_attempts(self) -> int:
        raw = os.getenv("QLIB_FACTOR_MAX_REPAIR_ATTEMPTS", "2")
        try:
            v = int(raw)
        except Exception:
            v = 2
        return max(0, v)

    def _is_factor_autorepair_eligible(self, execution_feedback: str) -> bool:
        if not isinstance(execution_feedback, str) or not execution_feedback:
            return False
        s = execution_feedback.lower()
        needles = [
            "filenotfounderror",
            "keyerror",
            "expected output file not found",
            "isna is not defined for multiindex",
            "missing columns",
            "columns are missing",
            "not in index",
            "column not found",
            "does not exist",
            "no such file or directory",
            "empty dataframe",
            "all nan",
            "all values are nan",
        ]
        return any(n in s for n in needles)

    def _repair_factor_code_with_llm(self, factor_code: str, execution_feedback: str) -> str:
        system_prompt = (
            "You are fixing a Python factor implementation file named factor.py. "
            "Return ONLY valid Python source code for factor.py (no markdown, no explanations). "
            "Do not add any comments. Keep the same public function entry points as much as possible. "
            "The script must write the factor result to result.h5 in the current working directory."
        )
        user_prompt = (
            "The current factor.py failed. Fix it based on the execution feedback.\n\n"
            "[Execution feedback]\n"
            f"{execution_feedback}\n\n"
            "[Current factor.py]\n"
            f"{factor_code}\n"
        )
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )
        if not isinstance(resp, str):
            return ""
        return resp.strip()

    def _collect_valid_factor_dfs(self, exp: QlibFactorExperiment, data_type: str = "All") -> list[pd.DataFrame]:
        factor_dfs: list[pd.DataFrame] = []
        for implementation in getattr(exp, "sub_workspace_list", []) or []:
            if implementation is None:
                continue
            message, df = implementation.execute(data_type)
            if df is not None and "datetime" in df.index.names:
                time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                if pd.Timedelta(minutes=1) not in time_diff:
                    factor_dfs.append(df)
                else:
                    logger.warning(f"Factor data from {exp.hypothesis.concise_justification} is not generated.")
            else:
                logger.warning(
                    f"Factor data from {exp.hypothesis.concise_justification} is not generated because of {message}"
                )
        return factor_dfs

    def _attempt_factor_autorepair(self, exp: QlibFactorExperiment, max_repair_attempts: int) -> pd.DataFrame | None:
        if max_repair_attempts <= 0:
            return None

        for attempt in range(1, max_repair_attempts + 1):
            repaired_any = False
            for implementation in getattr(exp, "sub_workspace_list", []) or []:
                if implementation is None or not isinstance(getattr(implementation, "file_dict", None), dict):
                    continue
                factor_code = implementation.file_dict.get("factor.py")
                if not isinstance(factor_code, str) or not factor_code:
                    continue

                execution_feedback, df = implementation.execute("All")
                if df is not None:
                    continue
                if not self._is_factor_autorepair_eligible(execution_feedback):
                    continue

                logger.warning(
                    f"[FactorAutoRepair] attempt {attempt}/{max_repair_attempts} repairing factor.py due to failure signature"
                )
                new_code = self._repair_factor_code_with_llm(factor_code, execution_feedback)
                if not isinstance(new_code, str) or not new_code:
                    continue
                implementation.inject_files(**{"factor.py": new_code})
                repaired_any = True

            if not repaired_any:
                return None

            factor_dfs = self._collect_valid_factor_dfs(exp, data_type="All")
            if factor_dfs:
                return pd.concat(factor_dfs, axis=1)

        return None

    @cache_with_pickle(qlib_factor_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        """
        Generate the experiment by processing and combining factor data,
        then passing the combined data to Docker for backtest results.
        """
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            logger.info(f"Baseline experiment execution ...")
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        if exp.based_experiments:
            SOTA_factor = None
            # Filter and retain only QlibFactorExperiment instances
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info(f"SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            logger.info(f"New factor processing ...")
            # Process the new factors data
            try:
                new_factors = process_factor_data(exp)
            except FactorEmptyError as e:
                max_repair_attempts = self._get_factor_max_repair_attempts()
                logger.warning(
                    f"[FactorAutoRepair] new factor processing failed; max_repair_attempts={max_repair_attempts}; error={e}"
                )
                repaired = self._attempt_factor_autorepair(exp, max_repair_attempts=max_repair_attempts)
                if repaired is None or repaired.empty:
                    raise
                new_factors = repaired

            if new_factors.empty:
                raise FactorEmptyError("Factors failed to run on the full sample, this round of experiment failed.")

            # Combine the SOTA factor and new factors if SOTA factor exists
            if SOTA_factor is not None and not SOTA_factor.empty:
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                if new_factors.empty:
                    raise FactorEmptyError(
                        "The factors generated in this round are highly similar to the previous factors. Please change the direction for creating new factors."
                    )
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors

            if combined_factors.empty:
                raise FactorEmptyError(
                    "Combined factors became empty after alignment/dropna. Refusing to run backtest with incomplete factor data."
                )

            non_nan_row_ratio = float(combined_factors.notna().any(axis=1).mean())
            if non_nan_row_ratio <= 0.0:
                cols_preview = list(map(str, combined_factors.columns[:20].tolist()))
                raise FactorEmptyError(
                    "Combined factors contain only NaN values (no usable samples). "
                    "Refusing to write parquet/run backtest. "
                    f"columns_preview={cols_preview}"
                )

            # Sort and nest the combined factors under 'feature'
            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            if combined_factors.empty:
                raise FactorEmptyError(
                    "Combined factors became empty after de-duplication. Refusing to run backtest with incomplete factor data."
                )

            non_nan_row_ratio = float(combined_factors.notna().any(axis=1).mean())
            if non_nan_row_ratio <= 0.0:
                cols_preview = list(map(str, combined_factors.columns[:20].tolist()))
                raise FactorEmptyError(
                    "Combined factors contain only NaN values after de-duplication (no usable samples). "
                    "Refusing to write parquet/run backtest. "
                    f"columns_preview={cols_preview}"
                )
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns
            num_features = RD_AGENT_SETTINGS.initial_fator_library_size + len(combined_factors.columns)
            logger.info(f"Factor data processing completed.")

            # Due to the rdagent and qlib docker image in the numpy version of the difference,
            # the `combined_factors_df.pkl` file could not be loaded correctly in qlib dokcer,
            # so we changed the file type of `combined_factors_df` from pkl to parquet.
            target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

            # Save the combined factors to the workspace
            combined_factors.to_parquet(target_path, engine="pyarrow")

            # If model exp exists in the previous experiment
            exist_sota_model_exp = False
            for base_exp in reversed(exp.based_experiments):
                if isinstance(base_exp, QlibModelExperiment):
                    sota_model_exp = base_exp
                    exist_sota_model_exp = True
                    break
            logger.info(f"Experiment execution ...")
            if exist_sota_model_exp:
                exp.experiment_workspace.inject_files(
                    **{"model.py": sota_model_exp.sub_workspace_list[0].file_dict["model.py"]}
                )
                env_to_use = {"PYTHONPATH": "./"}
                sota_training_hyperparameters = sota_model_exp.sub_tasks[0].training_hyperparameters
                if sota_training_hyperparameters:
                    env_to_use.update(
                        {
                            "n_epochs": str(sota_training_hyperparameters.get("n_epochs", "100")),
                            "lr": str(sota_training_hyperparameters.get("lr", "2e-4")),
                            "early_stop": str(sota_training_hyperparameters.get("early_stop", 10)),
                            "batch_size": str(sota_training_hyperparameters.get("batch_size", 256)),
                            "weight_decay": str(sota_training_hyperparameters.get("weight_decay", 0.0001)),
                        }
                    )
                sota_model_type = sota_model_exp.sub_tasks[0].model_type
                if sota_model_type == "TimeSeries":
                    env_to_use.update(
                        {"dataset_cls": "TSDatasetH", "num_features": num_features, "step_len": 20, "num_timesteps": 20}
                    )
                elif sota_model_type == "Tabular":
                    env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})

                # model + combined factors
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors_sota_model.yaml", run_env=env_to_use
                )
            else:
                # LGBM + combined factors
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name=(
                        f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors_dynamic.yaml"
                    )
                )
        else:
            logger.info(f"Experiment execution ...")
            result, stdout = exp.experiment_workspace.execute(
                qlib_config_name=(
                    f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors_dynamic.yaml"
                )
            )

        if result is None:
            logger.error(f"Failed to run this experiment, because {stdout}")
            raise FactorEmptyError(f"Failed to run this experiment, because {stdout}")

        exp.result = result
        exp.stdout = stdout

        # Log key metrics used for factor feedback so that the chosen run's results are explicit in logs.
        try:
            ic = result.get("IC", None)
            ann_ret = result.get("1day.excess_return_with_cost.annualized_return", None)
            mdd = result.get("1day.excess_return_with_cost.max_drawdown", None)
            logger.info(
                f"[QlibFactorRunner] Final factor metrics: IC={ic}, "
                f"1day.excess_return_with_cost.annualized_return={ann_ret}, "
                f"1day.excess_return_with_cost.max_drawdown={mdd}"
            )
        except Exception:
            # Best-effort logging; do not break the pipeline if result is not dict-like.
            pass

        return exp
