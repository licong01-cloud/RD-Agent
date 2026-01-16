from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ValidatorResult:
    name: str
    passed: bool
    error_code: str | None = None
    error_summary: str | None = None
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "error_code": self.error_code,
            "error_summary": self.error_summary,
            "details": dict(self.details or {}),
        }


class CodeValidator:
    name: str

    def validate(self, code: str, *, required_columns: Sequence[str] | None = None) -> ValidatorResult:
        raise NotImplementedError


class OutputProtocolValidator(CodeValidator):
    name = "output_protocol"

    def validate(self, code: str, *, required_columns: Sequence[str] | None = None) -> ValidatorResult:
        # P0: keep it lightweight. Enforce result.h5 + to_hdf usage at least.
        checks = {
            "writes_result_h5": "result.h5" in code,
            "uses_to_hdf": ".to_hdf(" in code,
        }
        if all(checks.values()):
            return ValidatorResult(name=self.name, passed=True)
        return ValidatorResult(
            name=self.name,
            passed=False,
            error_code="OUTPUT_PROTOCOL_INVALID",
            error_summary="Factor code must write result.h5 using pandas.DataFrame.to_hdf.",
            details=checks,
        )


class PrecomputedColumnValidator(CodeValidator):
    name = "precomputed_column_required"

    def __init__(self, *, forbidden_columns: Sequence[str] | None = None):
        self._forbidden_columns = list(forbidden_columns or [])

    def validate(self, code: str, *, required_columns: Sequence[str] | None = None) -> ValidatorResult:
        required_columns = list(required_columns or [])
        missing = [c for c in required_columns if c not in code]

        forbidden_hits: list[str] = []
        for c in self._forbidden_columns:
            if c and c in code:
                forbidden_hits.append(c)

        # Optional: detect suspicious rolling usage when required column is missing.
        suspicious_rolling = bool(re.search(r"\.rolling\(.*\)\.mean\(\)", code))

        if not missing and not forbidden_hits:
            return ValidatorResult(name=self.name, passed=True)

        details: dict[str, Any] = {
            "required_columns": required_columns,
            "missing_required": missing,
            "forbidden_columns": self._forbidden_columns,
            "forbidden_hits": forbidden_hits,
            "suspicious_rolling": suspicious_rolling,
        }
        return ValidatorResult(
            name=self.name,
            passed=False,
            error_code="PRECOMPUTED_COLUMN_REQUIRED",
            error_summary="Code must directly reference required precomputed columns and must not use forbidden substitutes.",
            details=details,
        )
