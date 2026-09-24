from __future__ import annotations

from typing import Any

from opentelemetry.trace import Status, StatusCode

from ..domain import EvaluationObservation, EvaluationScore, NormalizedRecord, ObservationStatus, ParseResult, ParseStatus
from ..protocols import EvaluationSuite
from ..telemetry import OperationTelemetry


class EvaluationExecutor:
    def __init__(self, telemetry: OperationTelemetry | None = None) -> None:
        self._telemetry = telemetry or OperationTelemetry.create("evaluation")

    def evaluate(self, suite: EvaluationSuite, record: NormalizedRecord, raw_output: Any) -> EvaluationObservation:
        operation = "evaluate_record"
        with self._telemetry.tracer.start_as_current_span("capability_anatomy.evaluation.record") as span:
            try:
                parsed = suite.parse(raw_output)
            except Exception as error:
                parsed = ParseResult(
                    status=ParseStatus.ERROR,
                    diagnostics={"error_type": type(error).__name__},
                )
                self._telemetry.record(span, operation=operation, outcome="recorded", reason="parse_failure_recorded")
                span.set_status(Status(StatusCode.ERROR, "parse_failure_recorded"))
                return EvaluationObservation(
                    example_id=record.id,
                    partition=record.partition,
                    status=ObservationStatus.ERROR,
                    raw_output=raw_output,
                    parsed=parsed,
                    scores={},
                    error_type=type(error).__name__,
                )
            if parsed.status is not ParseStatus.PARSED:
                self._telemetry.record(
                    span,
                    operation=operation,
                    outcome="recorded",
                    reason="parse_failure_returned",
                )
                span.set_status(Status(StatusCode.ERROR, "parse_failure_returned"))
                return EvaluationObservation(
                    example_id=record.id,
                    partition=record.partition,
                    status=ObservationStatus.ERROR,
                    raw_output=raw_output,
                    parsed=parsed,
                    scores={},
                    error_type=parsed.diagnostics.get("error_type", "ParseError"),
                )
            try:
                scores = dict(suite.score(record, parsed))
                if any(not isinstance(value, EvaluationScore) for value in scores.values()):
                    raise TypeError("evaluation scores must use EvaluationScore")
            except Exception as error:
                self._telemetry.record(span, operation=operation, outcome="recorded", reason="score_failure_recorded")
                span.set_status(Status(StatusCode.ERROR, "score_failure_recorded"))
                return EvaluationObservation(
                    example_id=record.id,
                    partition=record.partition,
                    status=ObservationStatus.ERROR,
                    raw_output=raw_output,
                    parsed=parsed,
                    scores={},
                    error_type=type(error).__name__,
                )
            self._telemetry.record(span, operation=operation, outcome="accepted", reason="observation_complete")
            return EvaluationObservation(
                example_id=record.id,
                partition=record.partition,
                status=ObservationStatus.COMPLETE,
                raw_output=raw_output,
                parsed=parsed,
                scores=scores,
            )

    def execution_failure(self, record: NormalizedRecord, error_type: str) -> EvaluationObservation:
        with self._telemetry.tracer.start_as_current_span(
            "capability_anatomy.evaluation.record",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self._telemetry.record(
                span,
                operation="evaluate_record",
                outcome="recorded",
                reason="model_execution_failed",
            )
            span.set_status(Status(StatusCode.ERROR, "model_execution_failed"))
        return EvaluationObservation(
            example_id=record.id,
            partition=record.partition,
            status=ObservationStatus.ERROR,
            raw_output=None,
            parsed=ParseResult(ParseStatus.ERROR, diagnostics={"error_type": error_type}),
            scores={},
            error_type=error_type,
        )
