from __future__ import annotations

from pathlib import Path

import pytest

from rdagent.rag.factor_generator import FactorCodeGenerator, FactorGenTask
from rdagent.rag.interfaces import RetrieverService
from rdagent.rag.types import RetrievedContext
from rdagent.rag.validators import OutputProtocolValidator, PrecomputedColumnValidator


class _LLMStub:
    def __init__(self):
        self.calls = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            # Wrong: uses mf_main_net_amt_ratio rolling instead of mf_main_net_amt_ratio_5d
            return """
import pandas as pd

def main():
    df = pd.read_hdf('daily_pv.h5', key='data')
    # WRONG: rolling substitute
    x = df['mf_main_net_amt_ratio'].groupby(level=1).rolling(5).mean().reset_index(level=0, drop=True)
    out = x.to_frame('factor1')
    out.to_hdf('result.h5', key='data', mode='w')

if __name__ == '__main__':
    main()
"""
        # Correct: references mf_main_net_amt_ratio_5d
        return """
import pandas as pd

def main():
    df = pd.read_hdf('daily_pv.h5', key='data')
    if 'mf_main_net_amt_ratio_5d' not in df.columns:
        raise ValueError('Missing columns: [mf_main_net_amt_ratio_5d]')
    x = df['mf_main_net_amt_ratio_5d']
    out = x.to_frame('factor1').astype('float64')
    out.to_hdf('result.h5', key='data', mode='w')

if __name__ == '__main__':
    main()
"""


class _RetrieverStub(RetrieverService):
    def retrieve(self, query: str, *, top_k: int, filters=None):  # type: ignore[override]
        return [
            RetrievedContext(
                text="contract: must directly use mf_main_net_amt_ratio_5d; rolling substitute is forbidden",
                doc_id="doc_test",
                source_path="RAG/03_factor_implementation_contract.md",
                chunk_id="doc_test#chunk0",
                score=1.0,
                type="contract",
                tags=["precomputed_column"],
                kb_version="default",
                doc_format="md",
            )
        ][:top_k]

    def disable_doc(self, doc_id: str) -> None:
        return

    def enable_doc(self, doc_id: str) -> None:
        return


def test_golden_offline(tmp_path: Path):
    trace_path = tmp_path / 'rag_trace.jsonl'

    llm = _LLMStub()
    validators = [
        OutputProtocolValidator(),
        PrecomputedColumnValidator(forbidden_columns=['mf_main_net_amt_ratio']),
    ]

    gen = FactorCodeGenerator(
        llm_generate=llm,
        validators=validators,
        trace_path=trace_path,
        retriever=_RetrieverStub(),
        max_retries=1,
        top_k=3,
        kb_version='default',
    )

    task = FactorGenTask(
        query='Implement factor using precomputed column mf_main_net_amt_ratio_5d only.',
        required_precomputed_columns=['mf_main_net_amt_ratio_5d'],
        forbidden_columns=['mf_main_net_amt_ratio'],
    )

    code = gen.generate(task)
    assert 'mf_main_net_amt_ratio_5d' in code
    assert llm.calls == 2
    assert trace_path.exists()
