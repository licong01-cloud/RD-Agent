from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rdagent.rag.validators import ValidatorResult


@dataclass(frozen=True)
class RetryDecision:
    decision: str  # accept|retry|abort
    reason: str


RECOVERABLE_ERROR_CODES: set[str] = {
    "OUTPUT_PROTOCOL_INVALID",
    "PRECOMPUTED_COLUMN_REQUIRED",
}


def decide_retry(
    *,
    validator_results: Sequence[ValidatorResult],
    attempt_idx: int,
    max_retries: int,
) -> RetryDecision:
    if all(v.passed for v in validator_results):
        return RetryDecision(decision="accept", reason="all_validators_passed")

    if attempt_idx >= max_retries:
        return RetryDecision(decision="abort", reason="retry_limit_reached")

    # If any failure is not recoverable, abort.
    for v in validator_results:
        if v.passed:
            continue
        if v.error_code is None or v.error_code not in RECOVERABLE_ERROR_CODES:
            return RetryDecision(decision="abort", reason=f"unrecoverable:{v.error_code}")

    return RetryDecision(decision="retry", reason="recoverable_validation_failure")
