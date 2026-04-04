import json
from pathlib import Path
from typing import Dict

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent

IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]


def process_results(current_result, sota_result):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # Set the metric as the index
    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # Rename the value column to reflect the result type
    current_df.rename(columns={"0": "Current Result"}, inplace=True)
    sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Filter the combined DataFrame to retain only the important metrics
    filtered_combined_df = combined_df.loc[IMPORTANT_METRICS]

    def format_filtered_combined_df(filtered_combined_df: pd.DataFrame) -> str:
        results = []
        for metric, row in filtered_combined_df.iterrows():
            current = row["Current Result"]
            sota = row["SOTA Result"]
            results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
        return "; ".join(results)

    return format_filtered_combined_df(filtered_combined_df)


class QlibFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        if isinstance(self.scen, (QlibQuantScenario, QlibFactorScenario)):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(filtered_tag="factor_feedback")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)

        # Case-insensitive key lookup helper
        def _get_ci(d, target_key, default=""):
            """Case-insensitive dict get with fuzzy matching."""
            val = d.get(target_key)
            if val is not None:
                return val
            target_lower = target_key.lower()
            for k, v in d.items():
                if k.lower() == target_lower:
                    return v
            # Fuzzy: check if target words appear in key
            target_words = [w for w in target_lower.split() if len(w) > 3]
            for k, v in d.items():
                k_lower = k.lower()
                if all(w in k_lower for w in target_words):
                    return v
            return default

        # Extract fields from JSON response with fuzzy matching
        observations = _get_ci(response_json, "Observations", "No observations provided")
        hypothesis_evaluation = _get_ci(response_json, "Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = _get_ci(response_json, "New Hypothesis", "No new hypothesis provided")
        reason = _get_ci(response_json, "Reasoning", "No reasoning provided")
        decision = convert2bool(_get_ci(response_json, "Replace Best Result", "no"))

        # Hard veto: if AnnRet < baseline × 80%, reject regardless of LLM decision.
        # "baseline" = the Alpha158-only experiment (sub_tasks=[], no generated factors).
        # In loop 0 it is exp.based_experiments[0] (result populated by runner).
        # In loop 1+ based_experiments[0].result is None (runner only fills [-1]),
        # so we fall back to the first historical experiment's baseline via trace.
        if decision and current_result is not None:
            annret_key = "1day.excess_return_with_cost.annualized_return"
            try:
                baseline_annret = None
                # Try current experiment's baseline first (works for loop 0)
                if exp.based_experiments and not exp.based_experiments[0].sub_tasks:
                    br = exp.based_experiments[0].result
                    if br is not None:
                        baseline_annret = float(br.loc[annret_key])
                # Fallback: search trace history for the baseline result (loop 1+)
                if baseline_annret is None:
                    for hist_exp, _ in trace.hist:
                        if hasattr(hist_exp, "based_experiments") and hist_exp.based_experiments:
                            b0 = hist_exp.based_experiments[0]
                            if not b0.sub_tasks and b0.result is not None:
                                baseline_annret = float(b0.result.loc[annret_key])
                                break
                if baseline_annret is not None:
                    current_annret = float(current_result.loc[annret_key])
                    threshold = baseline_annret * 0.80
                    if current_annret < threshold:
                        logger.warning(
                            f"[HARD VETO] AnnRet {current_annret:.4f} < baseline×80% ({threshold:.4f}). "
                            f"Overriding LLM ACCEPT → REJECT."
                        )
                        decision = False
                        reason = (
                            f"{reason} | [HARD VETO] AnnRet ({current_annret:.4f}) dropped below "
                            f"80% of baseline ({baseline_annret:.4f}), threshold={threshold:.4f}. "
                            f"Factor combination is destructive to Alpha158 signal."
                        )
            except Exception as e:
                logger.warning(f"[HARD VETO] Could not check AnnRet threshold: {e}")

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class QlibModelExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibModelExperiment): The experiment to generate feedback for.
            hypothesis (QlibModelHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            HypothesisFeedback: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            sota_result=SOTA_experiment.result.loc[IMPORTANT_METRICS] if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=exp.result.loc[IMPORTANT_METRICS] if exp.result is not None else "execution failed",
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response)

        # Case-insensitive key lookup helper
        def _get_ci(d, target_key, default=""):
            """Case-insensitive dict get with fuzzy matching."""
            val = d.get(target_key)
            if val is not None:
                return val
            target_lower = target_key.lower()
            for k, v in d.items():
                if k.lower() == target_lower:
                    return v
            # Fuzzy: check if target words appear in key
            target_words = [w for w in target_lower.split() if len(w) > 3]
            for k, v in d.items():
                k_lower = k.lower()
                if all(w in k_lower for w in target_words):
                    return v
            return default

        decision = convert2bool(_get_ci(response_json_hypothesis, "Decision", "false"))
        reason = _get_ci(response_json_hypothesis, "Reasoning", "No reasoning provided")

        # Hard veto: if AnnRet < baseline × 80%, reject regardless of LLM decision.
        # For model experiments, baseline = Alpha158-only factor experiment (sub_tasks=[]).
        # Search trace history for a factor experiment's baseline result.
        if decision and exp.result is not None:
            annret_key = "1day.excess_return_with_cost.annualized_return"
            try:
                from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
                current_annret = float(exp.result.loc[annret_key])
                baseline_annret = None
                # Search trace for the Alpha158-only baseline (empty factor experiment)
                for hist_exp, _ in trace.hist:
                    if hasattr(hist_exp, "based_experiments") and hist_exp.based_experiments:
                        b0 = hist_exp.based_experiments[0]
                        if isinstance(b0, QlibFactorExperiment) and not b0.sub_tasks and b0.result is not None:
                            baseline_annret = float(b0.result.loc[annret_key])
                            break
                # Fallback: use SOTA experiment's result (model loop without factor history)
                if baseline_annret is None and SOTA_experiment is not None and SOTA_experiment.result is not None:
                    baseline_annret = float(SOTA_experiment.result.loc[annret_key])

                if baseline_annret is not None:
                    threshold = baseline_annret * 0.80
                    if current_annret < threshold:
                        logger.warning(
                            f"[HARD VETO] AnnRet {current_annret:.4f} < baseline×80% ({threshold:.4f}). "
                            f"Overriding LLM ACCEPT → REJECT."
                        )
                        decision = False
                        reason = (
                            f"{reason} | [HARD VETO] AnnRet ({current_annret:.4f}) dropped below "
                            f"80% of baseline ({baseline_annret:.4f}), threshold={threshold:.4f}."
                        )
            except Exception as e:
                logger.warning(f"[HARD VETO] Could not check AnnRet threshold: {e}")

        return HypothesisFeedback(
            observations=_get_ci(response_json_hypothesis, "Observations", "No observations provided"),
            hypothesis_evaluation=_get_ci(response_json_hypothesis, "Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=_get_ci(response_json_hypothesis, "New Hypothesis", "No new hypothesis provided"),
            reason=reason,
            decision=decision,
        )
