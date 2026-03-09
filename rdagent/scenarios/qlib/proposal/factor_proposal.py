import json
import logging
from typing import List, Tuple

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.scenarios.qlib.proposal.field_utils import (
    collect_used_fields,
    format_field_usage_summary,
    format_unused_field_whitelist,
    validate_variables,
)
from rdagent.utils.agent.tpl import T

logger = logging.getLogger(__name__)

QlibFactorHypothesis = Hypothesis


class QlibFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                trace=trace,
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        last_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        # --- 2A + 4A: 字段统计 + 未使用白名单 ---
        field_usage_summary = ""
        if len(trace.hist) > 0:
            prefix_fields = collect_used_fields(trace.hist, factor_only=True)
            field_usage_summary = format_field_usage_summary(
                prefix_fields, n_rounds=len(trace.hist), language="zh"
            )
            field_usage_summary += format_unused_field_whitelist(prefix_fields, language="zh")

        # --- 2B: SOTA 上下文 ---
        sota_hypothesis_and_feedback = ""
        if len(trace.hist) > 0:
            for i in range(len(trace.hist) - 1, -1, -1):
                if trace.hist[i][1].decision:
                    sota_hypothesis_and_feedback = T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
                        experiment=trace.hist[i][0], feedback=trace.hist[i][1]
                    )
                    break
            if not sota_hypothesis_and_feedback:
                sota_hypothesis_and_feedback = "No SOTA available — all previous experiments were rejected."

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "sota_hypothesis_and_feedback": sota_hypothesis_and_feedback,
            # [AIstock] 原始代码鼓励ML因子，与因子层禁止ML训练的约束冲突，已注释
            # "RAG": (
            #     "Try the easiest and fastest factors to experiment with from various perspectives first."
            #     if len(trace.hist) < 15
            #     else "Now, you need to try factors that can achieve high IC (e.g., machine learning-based factors)."
            # ),
            "RAG": (
                "Try the easiest and fastest factors to experiment with from various perspectives first."
                if len(trace.hist) < 15
                else "Now, you need to try factors that can achieve high IC (e.g., cross-sectional ranking signals, non-linear price-volume combinations, multi-scale statistical features). Do NOT use machine learning training in factor scripts."
            ),
            "hypothesis_output_format": T("scenarios.qlib.prompts:factor_hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.qlib.prompts:factor_hypothesis_specification").r()
                + field_usage_summary,
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
        )
        return hypothesis


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        if isinstance(trace.scen, QlibQuantScenario):
            scenario = trace.scen.get_scenario_all_desc(action="factor")
        else:
            scenario = trace.scen.get_scenario_all_desc()

        experiment_output_format = T("scenarios.qlib.prompts:factor_experiment_output_format").r()

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
        else:
            specific_trace = Trace(trace.scen)
            for i in range(len(trace.hist) - 1, -1, -1):
                if not hasattr(trace.hist[i][0].hypothesis, "action") or trace.hist[i][0].hypothesis.action == "factor":
                    specific_trace.hist.insert(0, trace.hist[i])
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                    trace=specific_trace,
                )
            else:
                hypothesis_and_feedback = "No previous hypothesis and feedback available."

        # --- 3A + 4A: 字段统计 + 未使用白名单注入 Experiment Generation ---
        field_usage_rag = None
        if len(trace.hist) > 0:
            prefix_fields = collect_used_fields(trace.hist, factor_only=True)
            field_usage_rag = format_field_usage_summary(
                prefix_fields, n_rounds=len(trace.hist), language="en"
            )
            field_usage_rag += format_unused_field_whitelist(prefix_fields, language="en")

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": [],
            "RAG": field_usage_rag,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            # --- 4C: 幻觉字段检测 ---
            variables, field_warnings = validate_variables(variables, factor_name=factor_name)
            for w in field_warnings:
                logger.warning(w)
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    variables=variables,
                )
            )

        exp = QlibFactorExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
            t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
        ]

        unique_tasks = []
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                if isinstance(based_exp, QlibModelExperiment):
                    continue
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp
