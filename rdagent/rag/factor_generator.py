from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from rdagent.rag.factory import get_retriever
from rdagent.rag.knowledge_context import format_knowledge_context
from rdagent.rag.interfaces import RetrieverService
from rdagent.rag.retry_gate import decide_retry
from rdagent.rag.trace import RagTraceEvent, RagTraceWriter
from rdagent.rag.types import RetrievedContext
from rdagent.rag.validators import CodeValidator, ValidatorResult


@dataclass(frozen=True)
class FactorGenTask:
    query: str
    required_precomputed_columns: Sequence[str]
    forbidden_columns: Sequence[str]


class FactorCodeGenerator:
    def __init__(
        self,
        *,
        llm_generate: Callable[[str, str], str],
        validators: Sequence[CodeValidator],
        trace_path: Path,
        retriever: RetrieverService | None = None,
        max_retries: int = 1,
        top_k: int = 6,
        kb_version: str = "default",
    ):
        self._llm_generate = llm_generate
        self._validators = list(validators)
        self._trace = RagTraceWriter(trace_path)
        self._retriever = retriever
        self._max_retries = max_retries
        self._top_k = top_k
        self._kb_version = kb_version

    def generate(self, task: FactorGenTask) -> str:
        retriever = self._retriever or get_retriever()

        last_code = ""
        for attempt_idx in range(self._max_retries + 1):
            t0 = time.time()
            contexts: Sequence[RetrievedContext] = retriever.retrieve(task.query, top_k=self._top_k)
            t_retrieve = time.time()

            knowledge_context = format_knowledge_context(contexts)

            # Minimal prompt scaffold for offline testing.
            system_prompt = "You are a code generator. Follow the constraints strictly."
            user_prompt = (
                f"Task: {task.query}\n\n"
                f"Required precomputed columns: {list(task.required_precomputed_columns)}\n"
                f"Forbidden columns: {list(task.forbidden_columns)}\n\n"
                f"Knowledge context:\n{knowledge_context}\n\n"
                "Output Python code only."
            )

            code = self._llm_generate(system_prompt, user_prompt)
            t_llm = time.time()

            validator_results: list[ValidatorResult] = []
            for v in self._validators:
                validator_results.append(
                    v.validate(code, required_columns=task.required_precomputed_columns)
                )

            decision = decide_retry(
                validator_results=validator_results,
                attempt_idx=attempt_idx,
                max_retries=self._max_retries,
            )

            event = RagTraceEvent(
                attempt_idx=attempt_idx,
                query=task.query,
                top_k=self._top_k,
                kb_version=self._kb_version,
                retrieved_contexts=[c.to_dict() for c in contexts],
                validator_results=[v.to_dict() for v in validator_results],
                decision=decision.decision,
                timing_ms={
                    "retrieve": int((t_retrieve - t0) * 1000),
                    "llm": int((t_llm - t_retrieve) * 1000),
                    "validate": int((time.time() - t_llm) * 1000),
                },
            )
            self._trace.append(event)

            last_code = code
            if decision.decision == "accept":
                return code
            if decision.decision == "abort":
                return code

        return last_code
