"""PID-reuse-safe, durable QE loop cancellation receipts.

The legacy ``/kill`` route deliberately remains outside this contract.  Durable
multi-alpha cancellation is bound to the submission intent and the exact Linux
process incarnation.  The receipt is persisted before any signal is attempted,
so a transport/process crash can never cause a retry to signal an unrelated PID.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdagent.app.api_endpoints.qe_submission_receipt import (
    SubmissionReceiptError,
    SubmissionReceiptValidationError,
    capture_process_identity,
    get_submission_receipt_locked,
    loop_lifecycle_lock,
    observe_result_artifact,
    promote_submission_receipt_to_completed_from_verified_result_locked,
    read_loop_status_locked,
    transition_submission_receipt_locked,
    validate_process_identity,
    validate_submission_intent_hash,
    write_loop_status_locked,
)

KILL_RECEIPT_SCHEMA_VERSION = "qe_kill_receipt_v1"
KILL_RECEIPT_DIR_NAME = ".kill_receipts"
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOOP_ID_RE = re.compile(r"^Loop[1-9][0-9]*$")
_COMMAND_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_RECEIPT_STATUSES = frozenset(
    {"requested", "signal_sent", "reconciling", "completed", "cancelled", "failed"},
)
_TERMINAL_RECEIPT_STATUSES = frozenset({"completed", "cancelled", "failed"})
_ACTIVE_RECEIPT_STATUSES = frozenset({"requested", "signal_sent", "reconciling"})
_NEXT_GENERATION_REASONS = frozenset(
    {"kill_execution_incarnation_mismatch", "kill_process_started_race"},
)


class KillReceiptError(RuntimeError):
    """Base error for typed QE kill receipt persistence or execution."""


class KillReceiptConflictError(KillReceiptError):
    """A durable cancellation command conflicts with an active exact target."""


class KillReceiptValidationError(KillReceiptError):
    """The typed cancellation request has no canonical, safe identity."""


@dataclass(frozen=True)
class _KillPreparation:
    receipt: dict[str, Any] | None
    process_identity: dict[str, int] | None


def execute_typed_kill_intent(
    loop_dir: Path,
    *,
    task_id: str,
    loop_id: str,
    command_id: str,
    kill_intent_generation: int,
    kill_intent_hash: str,
    expected_submission_intent_hash: str,
    expected_process_identity: Mapping[str, Any] | None,
    expected_phase: str | None,
) -> dict[str, Any]:
    """Persist and deliver one exact cancellation intent.

    The lifecycle lock protects all receipt/status writers.  It is intentionally
    released for ``killpg`` itself: the pre-signal receipt makes the delivery
    ambiguous rather than risking a second signal if this process crashes.  The
    service always re-acquires the lock and re-observes submission/result facts
    after signal delivery; a completed result wins over cancellation.
    """

    intent = _normalize_kill_intent(
        task_id=task_id,
        loop_id=loop_id,
        command_id=command_id,
        kill_intent_generation=kill_intent_generation,
        kill_intent_hash=kill_intent_hash,
        expected_submission_intent_hash=expected_submission_intent_hash,
        expected_process_identity=expected_process_identity,
        expected_phase=expected_phase,
    )

    with loop_lifecycle_lock(loop_dir, intent["loop_id"]):
        preparation = _prepare_signal_delivery_unlocked(loop_dir, intent)
        if preparation.receipt is not None:
            return preparation.receipt
        live_identity = preparation.process_identity
        assert live_identity is not None

    signal_error: BaseException | None = None
    try:
        _signal_process_group(live_identity["pgid"], signal.SIGTERM)
    except OSError as exc:  # Reconciliation below must persist every delivery fact.
        signal_error = exc

    with loop_lifecycle_lock(loop_dir, intent["loop_id"]):
        current = _read_exact_receipt_unlocked(loop_dir, intent)
        if current is None:
            message = "typed QE kill receipt disappeared after durable signal reservation"
            raise KillReceiptError(message)
        if str(current.get("status") or "") in _TERMINAL_RECEIPT_STATUSES:
            return current
        submission = get_submission_receipt_locked(
            loop_dir,
            loop_id=intent["loop_id"],
            submission_intent_hash=intent["expected_submission_intent_hash"],
        )
        if submission is None:
            return _persist_receipt_unlocked(
                loop_dir,
                intent,
                status="failed",
                terminal_reason="kill_execution_incarnation_mismatch",
                existing=current,
                process_identity=live_identity,
                error={"message": "submission receipt disappeared after signal reservation"},
            )
        if signal_error is not None and not isinstance(signal_error, ProcessLookupError):
            return _persist_receipt_unlocked(
                loop_dir,
                intent,
                status="failed",
                terminal_reason="kill_signal_failed",
                existing=current,
                submission=submission,
                process_identity=live_identity,
                signal_attempt_count=1,
                signal_sent=False,
                error={"message": str(signal_error), "type": type(signal_error).__name__},
            )
        return _reobserve_after_signal_unlocked(
            loop_dir,
            intent,
            submission=submission,
            existing=current,
            process_identity=live_identity,
            signal_error=signal_error,
        )


def _prepare_signal_delivery_unlocked(
    loop_dir: Path,
    intent: Mapping[str, Any],
) -> _KillPreparation:
    existing = _find_existing_receipt_unlocked(loop_dir, intent)
    if existing is not None:
        return _KillPreparation(receipt=existing, process_identity=None)

    _assert_no_conflicting_active_receipt_unlocked(loop_dir, intent)
    submission = get_submission_receipt_locked(
        loop_dir,
        loop_id=intent["loop_id"],
        submission_intent_hash=intent["expected_submission_intent_hash"],
    )
    terminal_receipt = _pre_signal_terminal_receipt_unlocked(
        loop_dir,
        intent,
        submission=submission,
    )
    if terminal_receipt is not None:
        return _KillPreparation(receipt=terminal_receipt, process_identity=None)
    assert submission is not None

    identity_result = _validated_signal_identity_unlocked(
        loop_dir,
        intent,
        submission=submission,
    )
    if identity_result.receipt is not None:
        return identity_result
    live_identity = identity_result.process_identity
    assert live_identity is not None

    # This record is the durable delivery linearization point.  A crash after
    # this write is deliberately visible as unresolved rather than retried as
    # a second signal.
    _persist_receipt_unlocked(
        loop_dir,
        intent,
        status="reconciling",
        terminal_reason="signal_delivery_ambiguous",
        submission=submission,
        process_identity=live_identity,
        signal_attempt_count=1,
        signal_sent=False,
        error={"message": "SIGTERM delivery reserved; re-observe before any retry"},
    )
    return _KillPreparation(receipt=None, process_identity=live_identity)


def _pre_signal_terminal_receipt_unlocked(
    loop_dir: Path,
    intent: Mapping[str, Any],
    *,
    submission: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if submission is None:
        return _persist_receipt_unlocked(
            loop_dir,
            intent,
            status="failed",
            terminal_reason="kill_execution_incarnation_mismatch",
            error={"message": "submission receipt for expected_submission_intent_hash is absent"},
        )
    submission_status = str(submission.get("status") or "")
    if submission_status in {"completed", "failed", "cancelled"}:
        return _persist_terminal_from_submission_unlocked(loop_dir, intent, submission)
    if intent["expected_phase"] == "pre_process_start":
        return _cancel_before_process_start_unlocked(loop_dir, intent, submission)
    return None


def _validated_signal_identity_unlocked(
    loop_dir: Path,
    intent: Mapping[str, Any],
    *,
    submission: Mapping[str, Any],
) -> _KillPreparation:
    expected_identity = intent["expected_process_identity"]
    assert expected_identity is not None
    observed_identity = _process_identity_or_none(submission)
    if observed_identity is None:
        receipt = _persist_receipt_unlocked(
            loop_dir,
            intent,
            status="failed",
            terminal_reason="kill_execution_evidence_unavailable",
            submission=submission,
            error={"message": "submission receipt has no full pid/pgid/start_time_ticks identity"},
        )
        return _KillPreparation(receipt=receipt, process_identity=None)
    if observed_identity != expected_identity:
        receipt = _persist_receipt_unlocked(
            loop_dir,
            intent,
            status="failed",
            terminal_reason="kill_execution_incarnation_mismatch",
            submission=submission,
            process_identity=observed_identity,
            error={"message": "stored process identity differs from typed kill expectation"},
        )
        return _KillPreparation(receipt=receipt, process_identity=None)
    try:
        live_identity = capture_process_identity(expected_identity["pid"])
    except SubmissionReceiptError as exc:
        receipt = _persist_terminal_after_nonlive_process_unlocked(
            loop_dir,
            intent,
            submission=submission,
            process_identity=observed_identity,
            reason="kill_target_process_not_live",
            error={"message": str(exc)},
        )
        return _KillPreparation(receipt=receipt, process_identity=None)
    if live_identity != expected_identity:
        receipt = _persist_receipt_unlocked(
            loop_dir,
            intent,
            status="failed",
            terminal_reason="kill_execution_incarnation_mismatch",
            submission=submission,
            process_identity=live_identity,
            error={"message": "live process identity differs from typed kill expectation"},
        )
        return _KillPreparation(receipt=receipt, process_identity=None)
    return _KillPreparation(receipt=None, process_identity=live_identity)


def public_kill_receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable typed receipt without workspace implementation paths."""

    return {
        "schema_version": receipt["schema_version"],
        "task_id": receipt["task_id"],
        "loop_id": receipt["loop_id"],
        "command_id": receipt["command_id"],
        "kill_intent_generation": receipt["kill_intent_generation"],
        "kill_intent_hash": receipt["kill_intent_hash"],
        "expected_submission_intent_hash": receipt["expected_submission_intent_hash"],
        "expected_process_identity": receipt["expected_process_identity"],
        "expected_phase": receipt["expected_phase"],
        "process_identity": receipt.get("process_identity"),
        "status": receipt["status"],
        "signal_attempt_count": receipt["signal_attempt_count"],
        "signal_sent_at": receipt.get("signal_sent_at"),
        "signal_sent": receipt["signal_sent"],
        "process_observation": receipt.get("process_observation"),
        "result_observation": receipt.get("result_observation"),
        "submission_receipt_status": receipt.get("submission_receipt_status"),
        "terminal_reason": receipt.get("terminal_reason"),
        "error": receipt.get("error"),
        "created_at": receipt["created_at"],
        "updated_at": receipt["updated_at"],
        "completed_at": receipt.get("completed_at"),
    }


def _cancel_before_process_start_unlocked(
    loop_dir: Path,
    intent: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(submission.get("status") or "")
    if status != "reserved":
        reason = (
            "kill_process_started_race"
            if status in {"started", "running"}
            else "kill_execution_incarnation_mismatch"
        )
        return _persist_receipt_unlocked(
            loop_dir,
            intent,
            status="failed",
            terminal_reason=reason,
            submission=submission,
            process_identity=_process_identity_or_none(submission),
            error={"message": f"pre_process_start cancellation requires reserved submission; actual={status!r}"},
        )
    try:
        transition_submission_receipt_locked(
            loop_dir,
            loop_id=str(intent["loop_id"]),
            submission_intent_hash=str(intent["expected_submission_intent_hash"]),
            status="cancelled",
        )
        current_status = read_loop_status_locked(loop_dir)
        write_loop_status_locked(
            loop_dir,
            status="cancelled",
            expected_current={current_status},
        )
    except SubmissionReceiptError as exc:
        return _persist_receipt_unlocked(
            loop_dir,
            intent,
            status="failed",
            terminal_reason="kill_execution_incarnation_mismatch",
            submission=submission,
            error={"message": f"pre-process cancellation compare-and-set failed: {exc}"},
        )
    return _persist_receipt_unlocked(
        loop_dir,
        intent,
        status="cancelled",
        terminal_reason="cancelled_before_process_start",
        submission_status="cancelled",
        result_observation=_result_observation(loop_dir),
    )


def _reobserve_after_signal_unlocked(
    loop_dir: Path,
    intent: Mapping[str, Any],
    *,
    submission: Mapping[str, Any],
    existing: Mapping[str, Any],
    process_identity: Mapping[str, Any],
    signal_error: BaseException | None,
) -> dict[str, Any]:
    result_observation = _result_observation(loop_dir)
    current_submission_status = str(submission.get("status") or "")
    if current_submission_status in {"completed", "failed", "cancelled"}:
        return _persist_terminal_from_submission_unlocked(
            loop_dir,
            intent,
            submission,
            existing=existing,
            signal_sent=signal_error is None,
            signal_attempt_count=1,
            process_identity=process_identity,
            result_observation=result_observation,
        )
    try:
        live_identity = capture_process_identity(int(process_identity["pid"]))
    except SubmissionReceiptError as exc:
        if bool(result_observation.get("valid")):
            # A valid result beats a cancellation even when the background writer
            # has not yet advanced its receipt.
            promote_submission_receipt_to_completed_from_verified_result_locked(
                loop_dir,
                loop_id=str(intent["loop_id"]),
                submission_intent_hash=str(intent["expected_submission_intent_hash"]),
            )
            current_status = read_loop_status_locked(loop_dir)
            if current_status not in {"completed", "failed", "cancelled"}:
                write_loop_status_locked(loop_dir, status="completed", expected_current={current_status})
            return _persist_receipt_unlocked(
                loop_dir,
                intent,
                status="completed",
                terminal_reason="completed_result_wins_cancellation_race",
                existing=existing,
                process_identity=process_identity,
                submission_status="completed",
                result_observation=result_observation,
                signal_attempt_count=1,
                signal_sent=signal_error is None,
            )
        # The exact target was no longer live after the only reserved signal and
        # no valid result exists.  Under the shared lock this is the authoritative
        # cancellation terminalization point.
        transition_submission_receipt_locked(
            loop_dir,
            loop_id=str(intent["loop_id"]),
            submission_intent_hash=str(intent["expected_submission_intent_hash"]),
            status="cancelled",
        )
        current_status = read_loop_status_locked(loop_dir)
        if current_status not in {"completed", "failed", "cancelled"}:
            write_loop_status_locked(loop_dir, status="cancelled", expected_current={current_status})
        return _persist_receipt_unlocked(
            loop_dir,
            intent,
            status="cancelled",
            terminal_reason=(
                "cancelled_after_exact_signal"
                if signal_error is None
                else "cancelled_after_process_not_live"
            ),
            existing=existing,
            process_identity=process_identity,
            submission_status="cancelled",
            result_observation=result_observation,
            signal_attempt_count=1,
            signal_sent=signal_error is None,
            error={"message": str(exc)} if signal_error is not None else None,
        )
    if live_identity != dict(process_identity):
        return _persist_receipt_unlocked(
            loop_dir,
            intent,
            status="failed",
            terminal_reason="kill_execution_incarnation_mismatch",
            existing=existing,
            submission=submission,
            process_identity=live_identity,
            result_observation=result_observation,
            signal_attempt_count=1,
            signal_sent=signal_error is None,
            error={"message": "process identity changed after signal reservation"},
        )
    return _persist_receipt_unlocked(
        loop_dir,
        intent,
        status="reconciling",
        terminal_reason=None,
        existing=existing,
        submission=submission,
        process_identity=live_identity,
        result_observation=result_observation,
        signal_attempt_count=1,
        signal_sent=signal_error is None,
        error=(
            {"message": str(signal_error), "type": type(signal_error).__name__}
            if signal_error is not None
            else None
        ),
    )


def _persist_terminal_after_nonlive_process_unlocked(
    loop_dir: Path,
    intent: Mapping[str, Any],
    *,
    submission: Mapping[str, Any],
    process_identity: Mapping[str, Any] | None,
    reason: str,
    error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result_observation = _result_observation(loop_dir)
    if bool(result_observation.get("valid")):
        # A durable, parseable result is stronger evidence than a transient
        # process-liveness observation.  Keep submission/status receipts in
        # the same terminal state instead of exposing a completed typed receipt
        # beside a still-running submission receipt.
        promote_submission_receipt_to_completed_from_verified_result_locked(
            loop_dir,
            loop_id=str(intent["loop_id"]),
            submission_intent_hash=str(intent["expected_submission_intent_hash"]),
        )
        current_status = read_loop_status_locked(loop_dir)
        if current_status not in {"completed", "failed", "cancelled"}:
            write_loop_status_locked(loop_dir, status="completed", expected_current={current_status})
        return _persist_receipt_unlocked(
            loop_dir,
            intent,
            status="completed",
            terminal_reason="completed_result_wins_cancellation_race",
            submission=submission,
            process_identity=process_identity,
            result_observation=result_observation,
            submission_status="completed",
            error=error,
        )
    return _persist_receipt_unlocked(
        loop_dir,
        intent,
        status="failed",
        terminal_reason=reason,
        submission=submission,
        process_identity=process_identity,
        result_observation=result_observation,
        error=error,
    )


def _persist_terminal_from_submission_unlocked(
    loop_dir: Path,
    intent: Mapping[str, Any],
    submission: Mapping[str, Any],
    *,
    existing: Mapping[str, Any] | None = None,
    signal_sent: bool = False,
    signal_attempt_count: int = 0,
    process_identity: Mapping[str, Any] | None = None,
    result_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if result_observation is None:
        result_observation = _result_observation(loop_dir)
    if bool(result_observation.get("valid")):
        submission = promote_submission_receipt_to_completed_from_verified_result_locked(
            loop_dir,
            loop_id=str(intent["loop_id"]),
            submission_intent_hash=str(intent["expected_submission_intent_hash"]),
        )
    status = str(submission.get("status") or "")
    if status == "completed":
        receipt_status = "completed"
        reason = "completed_result_wins_cancellation_race"
    elif status == "cancelled":
        receipt_status = "cancelled"
        reason = "cancelled_before_or_during_signal"
    else:
        receipt_status = "failed"
        reason = "kill_target_already_failed"
    return _persist_receipt_unlocked(
        loop_dir,
        intent,
        status=receipt_status,
        terminal_reason=reason,
        existing=existing,
        submission=submission,
        process_identity=process_identity or _process_identity_or_none(submission),
        result_observation=result_observation or _result_observation(loop_dir),
        signal_attempt_count=signal_attempt_count,
        signal_sent=signal_sent,
    )


def _normalize_kill_intent(
    *,
    task_id: str,
    loop_id: str,
    command_id: str,
    kill_intent_generation: int,
    kill_intent_hash: str,
    expected_submission_intent_hash: str,
    expected_process_identity: Mapping[str, Any] | None,
    expected_phase: str | None,
) -> dict[str, Any]:
    normalized_loop_id = _validated_loop_id(loop_id)
    normalized_command_id = _validated_command_id(command_id)
    generation = _validated_generation(kill_intent_generation)
    normalized_hash = _validated_kill_hash(kill_intent_hash)
    normalized_identity = _validated_expected_identity(
        expected_phase=expected_phase,
        expected_process_identity=expected_process_identity,
    )
    return {
        "task_id": str(task_id),
        "loop_id": normalized_loop_id,
        "command_id": normalized_command_id,
        "kill_intent_generation": generation,
        "kill_intent_hash": normalized_hash,
        "expected_submission_intent_hash": validate_submission_intent_hash(expected_submission_intent_hash),
        "expected_process_identity": normalized_identity,
        "expected_phase": expected_phase,
    }


def _validated_loop_id(loop_id: str) -> str:
    normalized = str(loop_id or "")
    if not _LOOP_ID_RE.fullmatch(normalized):
        message = f"invalid QE loop id for typed kill receipt: {loop_id!r}"
        raise KillReceiptValidationError(message)
    return normalized


def _validated_command_id(command_id: str) -> str:
    normalized = str(command_id or "").strip()
    if not _COMMAND_ID_RE.fullmatch(normalized):
        message = "command_id must be a stable safe command identity"
        raise KillReceiptValidationError(message)
    return normalized


def _validated_generation(kill_intent_generation: int) -> int:
    message = "kill_intent_generation must be an integer >= 1"
    if isinstance(kill_intent_generation, bool):
        raise KillReceiptValidationError(message)
    try:
        generation = int(kill_intent_generation)
    except (TypeError, ValueError) as exc:
        raise KillReceiptValidationError(message) from exc
    if generation < 1:
        raise KillReceiptValidationError(message)
    return generation


def _validated_kill_hash(kill_intent_hash: str) -> str:
    normalized = str(kill_intent_hash or "").strip().lower()
    if not _HEX_SHA256_RE.fullmatch(normalized):
        message = "kill_intent_hash must be a lowercase SHA-256 hex digest"
        raise KillReceiptValidationError(message)
    return normalized


def _validated_expected_identity(
    *,
    expected_phase: str | None,
    expected_process_identity: Mapping[str, Any] | None,
) -> dict[str, int] | None:
    if expected_phase not in {None, "pre_process_start"}:
        message = "expected_phase must be null or pre_process_start"
        raise KillReceiptValidationError(message)
    if expected_phase == "pre_process_start":
        if expected_process_identity is not None:
            message = "pre_process_start cannot include expected_process_identity"
            raise KillReceiptValidationError(message)
        return None
    if expected_process_identity is None:
        message = "typed running-process cancellation requires expected_process_identity"
        raise KillReceiptValidationError(message)
    try:
        return validate_process_identity(expected_process_identity)
    except SubmissionReceiptValidationError as exc:
        raise KillReceiptValidationError(str(exc)) from exc


def _kill_receipt_path(loop_dir: Path, intent: Mapping[str, Any]) -> Path:
    command_digest = _safe_command_digest(str(intent["command_id"]))
    return (
        loop_dir.parent
        / KILL_RECEIPT_DIR_NAME
        / (
            f"{intent['loop_id']}.{intent['expected_submission_intent_hash']}."
            f"{command_digest}.{intent['kill_intent_generation']}.json"
        )
    )


def _safe_command_digest(command_id: str) -> str:
    # File names do not carry raw command identity even though it is validated.
    return hashlib.sha256(command_id.encode("utf-8")).hexdigest()


def _find_existing_receipt_unlocked(loop_dir: Path, intent: Mapping[str, Any]) -> dict[str, Any] | None:
    exact = _read_exact_receipt_unlocked(loop_dir, intent)
    if exact is not None:
        return exact
    root = loop_dir.parent / KILL_RECEIPT_DIR_NAME
    if not root.exists():
        return None
    for path in sorted(root.glob(f"{intent['loop_id']}.{intent['expected_submission_intent_hash']}.*.json")):
        payload = _read_receipt_path(path)
        if payload.get("kill_intent_hash") == intent["kill_intent_hash"]:
            return payload
        if (
            payload.get("command_id") == intent["command_id"]
            and int(payload.get("kill_intent_generation") or 0) == intent["kill_intent_generation"]
        ):
            message = "same command generation has a different durable kill intent hash"
            raise KillReceiptConflictError(message)
    return None


def _assert_no_conflicting_active_receipt_unlocked(loop_dir: Path, intent: Mapping[str, Any]) -> None:
    root = loop_dir.parent / KILL_RECEIPT_DIR_NAME
    if not root.exists():
        return
    prior: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"{intent['loop_id']}.{intent['expected_submission_intent_hash']}.*.json")):
        payload = _read_receipt_path(path)
        prior.append(payload)
        if str(payload.get("status") or "") in _ACTIVE_RECEIPT_STATUSES:
            message = "an active typed cancellation receipt already owns this loop execution"
            raise KillReceiptConflictError(message)
    if not prior:
        return
    newest = max(
        prior,
        key=lambda item: (
            int(item.get("kill_intent_generation") or 0),
            str(item.get("updated_at") or ""),
        ),
    )
    if int(intent["kill_intent_generation"]) <= int(newest.get("kill_intent_generation") or 0):
        message = "typed kill generation must advance after a completed receipt"
        raise KillReceiptConflictError(message)
    if (
        bool(newest.get("signal_sent"))
        or str(newest.get("terminal_reason") or "") not in _NEXT_GENERATION_REASONS
    ):
        message = "next typed kill generation is only valid after a no-signal incarnation/process-start race"
        raise KillReceiptConflictError(message)


def _read_exact_receipt_unlocked(loop_dir: Path, intent: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _kill_receipt_path(loop_dir, intent)
    if not path.exists():
        return None
    payload = _read_receipt_path(path)
    _validate_receipt_identity(payload, intent)
    return payload


def _read_receipt_path(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        message = f"typed QE kill receipt is unreadable: {path}: {exc}"
        raise KillReceiptError(message) from exc
    if not isinstance(payload, dict):
        message = f"typed QE kill receipt must be a JSON object: {path}"
        raise KillReceiptError(message)
    required = {
        "schema_version", "task_id", "loop_id", "command_id", "kill_intent_generation",
        "kill_intent_hash", "expected_submission_intent_hash", "expected_process_identity",
        "expected_phase", "status", "signal_attempt_count", "signal_sent", "created_at", "updated_at",
    }
    missing = sorted(required.difference(payload))
    if missing:
        message = f"typed QE kill receipt missing fields: {missing}"
        raise KillReceiptError(message)
    if payload.get("schema_version") != KILL_RECEIPT_SCHEMA_VERSION:
        message = f"unsupported typed QE kill receipt schema: {payload.get('schema_version')!r}"
        raise KillReceiptError(message)
    if str(payload.get("status") or "") not in _RECEIPT_STATUSES:
        message = f"unsupported typed QE kill receipt status: {payload.get('status')!r}"
        raise KillReceiptError(message)
    _normalize_kill_intent(
        task_id=str(payload.get("task_id") or ""),
        loop_id=str(payload.get("loop_id") or ""),
        command_id=str(payload.get("command_id") or ""),
        kill_intent_generation=payload.get("kill_intent_generation"),
        kill_intent_hash=str(payload.get("kill_intent_hash") or ""),
        expected_submission_intent_hash=str(payload.get("expected_submission_intent_hash") or ""),
        expected_process_identity=payload.get("expected_process_identity"),
        expected_phase=payload.get("expected_phase"),
    )
    if not isinstance(payload.get("signal_attempt_count"), int) or payload["signal_attempt_count"] < 0:
        message = "typed QE kill receipt signal_attempt_count must be a non-negative integer"
        raise KillReceiptError(message)
    if not isinstance(payload.get("signal_sent"), bool):
        message = "typed QE kill receipt signal_sent must be boolean"
        raise KillReceiptError(message)
    return payload


def _validate_receipt_identity(payload: Mapping[str, Any], intent: Mapping[str, Any]) -> None:
    comparable = (
        "task_id", "loop_id", "command_id", "kill_intent_generation", "kill_intent_hash",
        "expected_submission_intent_hash", "expected_process_identity", "expected_phase",
    )
    if any(payload.get(key) != intent.get(key) for key in comparable):
        message = "typed QE kill receipt identity conflicts with the requested kill intent"
        raise KillReceiptConflictError(message)


def _persist_receipt_unlocked(
    loop_dir: Path,
    intent: Mapping[str, Any],
    *,
    status: str,
    terminal_reason: str | None,
    existing: Mapping[str, Any] | None = None,
    submission: Mapping[str, Any] | None = None,
    submission_status: str | None = None,
    process_identity: Mapping[str, Any] | None = None,
    result_observation: Mapping[str, Any] | None = None,
    signal_attempt_count: int | None = None,
    signal_sent: bool | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in _RECEIPT_STATUSES:
        message = f"unsupported typed QE kill receipt status: {status!r}"
        raise KillReceiptError(message)
    current = dict(existing) if existing is not None else _read_exact_receipt_unlocked(loop_dir, intent)
    now = _utc_now_iso()
    prior_count = int(current.get("signal_attempt_count") or 0) if current else 0
    effective_count = prior_count if signal_attempt_count is None else int(signal_attempt_count)
    if effective_count < prior_count:
        message = "typed QE kill receipt signal_attempt_count cannot decrease"
        raise KillReceiptError(message)
    effective_sent = bool(current.get("signal_sent")) if current and signal_sent is None else bool(signal_sent)
    normalized_process = validate_process_identity(process_identity) if process_identity is not None else None
    receipt = {
        "schema_version": KILL_RECEIPT_SCHEMA_VERSION,
        **dict(intent),
        "process_identity": (
            normalized_process
            if normalized_process is not None
            else (current or {}).get("process_identity")
        ),
        "status": status,
        "signal_attempt_count": effective_count,
        "signal_sent_at": (
            (current or {}).get("signal_sent_at")
            or (now if effective_sent else None)
        ),
        "signal_sent": effective_sent,
        "process_observation": _process_observation(normalized_process or (current or {}).get("process_identity")),
        "result_observation": dict(result_observation or _result_observation(loop_dir)),
        "submission_receipt_status": submission_status
        or (
            str(submission.get("status"))
            if submission is not None
            else (current or {}).get("submission_receipt_status")
        ),
        "terminal_reason": terminal_reason,
        "error": dict(error) if error is not None else None,
        "created_at": (current or {}).get("created_at", now),
        "updated_at": now,
        "completed_at": now if status in _TERMINAL_RECEIPT_STATUSES else None,
    }
    _atomic_write_json(_kill_receipt_path(loop_dir, intent), receipt)
    return receipt


def _process_identity_or_none(submission: Mapping[str, Any]) -> dict[str, int] | None:
    raw = submission.get("process_identity")
    if raw is None:
        return None
    try:
        return validate_process_identity(raw)
    except SubmissionReceiptValidationError as exc:
        raise KillReceiptError(str(exc)) from exc


def _process_observation(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    return {"identity": dict(identity) if identity is not None else None}


def _result_observation(loop_dir: Path) -> dict[str, Any]:
    return observe_result_artifact(loop_dir)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _signal_process_group(pgid: int, signal_value: signal.Signals) -> None:
    """Signal the exact Linux process group; there is no PID-only fallback."""

    killpg = getattr(os, "killpg", None)
    if killpg is None:
        message = "typed QE kill requires os.killpg on the Linux execution node"
        raise OSError(message)
    killpg(pgid, signal_value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
