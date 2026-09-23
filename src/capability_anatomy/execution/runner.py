from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Callable, Mapping

from opentelemetry.trace import Status, StatusCode

from ..errors import InterruptedRunError, InvalidEvidenceError
from ..telemetry import OperationTelemetry, identity_scope
from .store import FrozenRunStore
from .secure_fs import ensure_run_ownership


@dataclass(frozen=True)
class Task:
    id: str
    stage: str
    execute: Callable[[], Mapping[str, Any]]

    def __post_init__(self) -> None:
        if self.stage not in {"baseline", "scan", "control", "validation"}:
            raise ValueError("task stage must be baseline or scan/control/validation")


class ExperimentRunner:
    def __init__(
        self,
        store: FrozenRunStore,
        telemetry: OperationTelemetry | None = None,
        *,
        max_wall_seconds: float | None = None,
        max_memory_observation_bytes: int | None = None,
        max_task_retries: int | None = None,
        max_observation_errors: int | None = None,
        clock: Callable[[], float] | None = None,
        memory_bytes: Callable[[], int | None] | None = None,
        trace_checkpoint: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.store = store
        self.telemetry = telemetry or OperationTelemetry.create("execution")
        self.max_wall_seconds = max_wall_seconds
        self.max_memory_observation_bytes = max_memory_observation_bytes
        self.max_task_retries = max_task_retries
        self.max_observation_errors = max_observation_errors
        self.clock = clock
        self.memory_bytes = memory_bytes
        self.trace_checkpoint = trace_checkpoint

    @contextmanager
    def _task_span(self):
        with self.telemetry.tracer.start_as_current_span(
            "capability_anatomy.execution.task", record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except BaseException as error:
                if getattr(getattr(span, "status", None), "status_code", None) is not StatusCode.ERROR:
                    reason = getattr(error, "reason", "task_failed")
                    self.telemetry.record(span, operation="execute_task", outcome="failed", reason=reason)
                    span.set_status(Status(StatusCode.ERROR, reason))
                raise

    def _enforce_resources(self) -> None:
        if self.max_wall_seconds is None and self.max_memory_observation_bytes is None:
            return
        with self.telemetry.tracer.start_as_current_span(
            "capability_anatomy.execution.resource_budget",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            reason = None
            if (
                self.max_wall_seconds is not None
                and self.store.active_wall_seconds() > self.max_wall_seconds
            ):
                reason = "wall_budget_exceeded"
            elif self.max_memory_observation_bytes is not None:
                observed = self.memory_bytes() if self.memory_bytes is not None else None
                if observed is None:
                    reason = "memory_budget_unobservable"
                elif observed > self.max_memory_observation_bytes:
                    reason = "memory_budget_exceeded"
            if reason is not None:
                self.store.record_state("stopped", reason)
                self.telemetry.record(span, operation="enforce_resource_budget", outcome="refused", reason=reason)
                span.set_status(Status(StatusCode.ERROR, reason))
                message = (
                    "experiment wall-time budget exceeded"
                    if reason == "wall_budget_exceeded"
                    else "experiment memory budget cannot be enforced: runtime memory observation unavailable" if reason == "memory_budget_unobservable"
                    else "experiment memory budget exceeded"
                )
                raise InvalidEvidenceError(message)
            self.telemetry.record(
                span, operation="enforce_resource_budget", outcome="accepted",
                reason="resource_budget_available",
            )

    def run(
        self,
        tasks: tuple[Task, ...],
        *,
        frozen_plan_reason: str | None = None,
    ) -> dict[str, Mapping[str, Any]]:
        with ensure_run_ownership(self.store.root, self.telemetry):
            return self._run_owned(tasks, frozen_plan_reason=frozen_plan_reason)

    def _run_owned(
        self, tasks: tuple[Task, ...], *, frozen_plan_reason: str | None = None,
    ) -> dict[str, Mapping[str, Any]]:
        if not tasks or tasks[0].stage != "baseline":
            raise ValueError("task graph must start with a baseline")
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("task IDs must be unique")
        self.store.initialize()
        if frozen_plan_reason is not None:
            with self.telemetry.tracer.start_as_current_span(
                "capability_anatomy.execution.plan",
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                self.telemetry.record(
                    span,
                    operation="freeze_task_plan",
                    outcome="accepted",
                    reason=frozen_plan_reason,
                )
        results: dict[str, Mapping[str, Any]] = {}
        for task in tasks:
            self._enforce_resources()
            operation = "execute_task"
            with identity_scope(task=task.id), self._task_span() as span:
                span.set_attribute("capability_anatomy.stage", task.stage)
                cached = self.store.completed(task.id)
                if cached is not None:
                    self.telemetry.record(span, operation=operation, outcome="skipped", reason="compatible_task_complete")
                    results[task.id] = cached
                    continue
                failures = self.store.failure_attempts(task.id)
                prior_failure = self.store.failure_payload(task.id)
                if (
                    self.max_observation_errors == 0
                    and prior_failure is not None
                    and prior_failure.get("budget_failure") == "observation_error_budget_exceeded"
                ):
                    self.store.record_state("stopped", "observation_error_budget_exceeded")
                    self.telemetry.record(span, operation=operation, outcome="refused", reason="observation_error_budget_exceeded")
                    span.set_status(Status(StatusCode.ERROR, "observation_error_budget_exceeded"))
                    raise InvalidEvidenceError("experiment observation-error budget exceeded")
                if self.max_task_retries is not None and failures > self.max_task_retries:
                    self.store.record_state("stopped", "retry_budget_exceeded")
                    self.telemetry.record(
                        span, operation=operation, outcome="refused", reason="retry_budget_exceeded",
                    )
                    span.set_status(Status(StatusCode.ERROR, "retry_budget_exceeded"))
                    raise InvalidEvidenceError("experiment task retry budget exceeded")
                task_started = self.clock() if self.clock is not None else None
                try:
                    payload = dict(task.execute())
                    observation_errors = len(payload.get("failures", ()))
                    observed_memory = max(
                        (
                            int(value)
                            for observation in payload.get("observations", ())
                            for value in (
                                observation.get("memory_observation_bytes"),
                                observation.get("peak_memory_bytes"),
                            )
                            if isinstance(value, int)
                        ),
                        default=None,
                    )
                    if (
                        self.max_observation_errors is not None
                        and observation_errors > self.max_observation_errors
                    ):
                        payload["complete"] = False
                        payload["budget_failure"] = "observation_error_budget_exceeded"
                    if (
                        self.max_memory_observation_bytes is not None
                        and observed_memory is not None
                        and observed_memory > self.max_memory_observation_bytes
                    ):
                        payload["complete"] = False
                        payload["budget_failure"] = "memory_budget_exceeded"
                    if payload.get("complete") is False:
                        self.store.commit_failure(task.id, payload, attempt=failures + 1)
                        self.store.record_state("failed", "incomplete_task")
                        self.telemetry.record(
                            span, operation=operation, outcome="failed", reason="task_incomplete"
                        )
                        span.set_status(Status(StatusCode.ERROR, "task_incomplete"))
                        if payload.get("budget_failure") == "observation_error_budget_exceeded":
                            raise InvalidEvidenceError("experiment observation-error budget exceeded")
                        raise InvalidEvidenceError("experiment task evidence is incomplete")
                except KeyboardInterrupt as error:
                    self.store.record_state("interrupted", "task_interrupted")
                    self.telemetry.record(span, operation=operation, outcome="interrupted", reason="task_interrupted")
                    span.set_status(Status(StatusCode.ERROR, "task_interrupted"))
                    raise InterruptedRunError("run interrupted with resumable state") from error
                except InvalidEvidenceError:
                    # _task_span records unhandled failures once. A known
                    # incomplete-task decision already owns this terminal reason.
                    raise
                except Exception as error:
                    self.store.commit_failure(
                        task.id,
                        {"complete": False, "error_type": type(error).__name__},
                        attempt=failures + 1,
                    )
                    self.store.record_state("failed", "task_failed")
                    self.telemetry.record(span, operation=operation, outcome="failed", reason="task_failed")
                    span.set_status(Status(StatusCode.ERROR, "task_failed"))
                    raise
                finally:
                    if task_started is not None and self.clock is not None:
                        self.store.add_active_wall_seconds(max(0.0, self.clock() - task_started))
                self._enforce_resources()
                if self.trace_checkpoint is not None:
                    self.trace_checkpoint(task.id, payload)
                self.store.commit(task.id, payload)
                self.telemetry.record(span, operation=operation, outcome="accepted", reason="task_committed")
                results[task.id] = payload
        self.store.record_state("complete", "all_tasks_committed")
        return results
