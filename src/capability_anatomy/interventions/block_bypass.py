from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from types import TracebackType
from typing import Iterator

from opentelemetry.trace import Status, StatusCode

from ..errors import InvalidConfigurationError
from ..models.base import LoadedModel
from ..protocols import InterventionProvider, ModelAdapter
from ..telemetry import OperationTelemetry


class BlockBypass:
    name = "builtin.block-bypass"
    version = "1"
    api_version = "capability-anatomy/plugin-api/v1"
    capabilities = frozenset({"intervention.block_bypass", "intervention.scoped_cleanup"})

    def __init__(self, component_ids: tuple[str, ...], telemetry: OperationTelemetry | None = None) -> None:
        self.component_ids = component_ids
        self._telemetry = telemetry or OperationTelemetry.create("intervention")
        self._lock = Lock()
        self._active = False

    def _claim(self) -> None:
        with self._lock:
            if self._active:
                raise InvalidConfigurationError("intervention is already active")
            self._active = True

    def _release(self) -> None:
        with self._lock:
            self._active = False

    def validate(self, provider: InterventionProvider, adapter: ModelAdapter, loaded: LoadedModel) -> None:
        if not self.component_ids or len(self.component_ids) != len(set(self.component_ids)):
            raise InvalidConfigurationError("bypass component IDs must be non-empty and unique")
        topology = adapter.topology(loaded)
        available = {component.id for component in topology.components}
        if not set(self.component_ids) <= available:
            raise InvalidConfigurationError("bypass component is absent from model topology")
        provider.validate(adapter, loaded, self.component_ids)

    @contextmanager
    def apply(
        self,
        provider: InterventionProvider,
        adapter: ModelAdapter,
        loaded: LoadedModel,
    ) -> Iterator[None]:
        with self._telemetry.tracer.start_as_current_span(
            "capability_anatomy.intervention.block_bypass",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                self.validate(provider, adapter, loaded)
                self._claim()
            except BaseException:
                span.set_status(Status(StatusCode.ERROR, "validation_failed"))
                self._telemetry.record(span, operation="block_bypass", outcome="refused", reason="validation_failed")
                raise

            try:
                manager = provider.apply(adapter, loaded, self.component_ids)
                manager.__enter__()
            except BaseException:
                self._release()
                span.set_status(Status(StatusCode.ERROR, "hook_installation_failed"))
                self._telemetry.record(span, operation="block_bypass", outcome="failed", reason="hook_installation_failed")
                raise

            self._telemetry.record(span, operation="block_bypass", outcome="accepted", reason="scoped_bypass_active")
            body_error: BaseException | None = None
            body_traceback: TracebackType | None = None
            try:
                yield
            except BaseException as error:
                body_error = error
                body_traceback = error.__traceback__

            try:
                manager.__exit__(
                    type(body_error) if body_error is not None else None,
                    body_error,
                    body_traceback,
                )
            except BaseException:
                span.set_status(Status(StatusCode.ERROR, "cleanup_failed"))
                self._telemetry.record(span, operation="block_bypass", outcome="failed", reason="cleanup_failed")
                raise
            finally:
                self._release()

            if body_error is not None:
                span.set_status(Status(StatusCode.ERROR, "experiment_body_failed"))
                self._telemetry.record(span, operation="block_bypass", outcome="failed", reason="experiment_body_failed")
                raise body_error.with_traceback(body_traceback)

            self._telemetry.record(span, operation="block_bypass", outcome="completed", reason="hooks_removed")
