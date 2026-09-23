from __future__ import annotations

import time
from typing import Any

from opentelemetry.trace import Status, StatusCode

from .protocols import CORE_API_VERSION
from .telemetry import OperationTelemetry


class LocalRuntimePlugin:
    name = "builtin.local"
    version = "1"
    api_version = CORE_API_VERSION
    capabilities = frozenset(
        {
            "runtime.local",
            "runtime.execute",
            "runtime.memory",
            "runtime.synchronization",
            "runtime.timing",
            "runtime.tokens",
        }
    )

    def __init__(self, telemetry: OperationTelemetry | None = None) -> None:
        self._telemetry = telemetry or OperationTelemetry.create("runtime")

    def execute(self, adapter: Any, loaded: Any, request: Any) -> Any:
        with self._telemetry.tracer.start_as_current_span(
            "capability_anatomy.runtime.execute",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = adapter.execute(loaded, request)
            except Exception:
                self._telemetry.record(span, operation="execute", outcome="failed", reason="adapter_execution_failed")
                span.set_status(Status(StatusCode.ERROR, "adapter_execution_failed"))
                raise
            self._telemetry.record(span, operation="execute", outcome="accepted", reason="adapter_execution_complete")
            return result

    @staticmethod
    def synchronize(adapter: Any, loaded: Any) -> None:
        adapter.synchronize(loaded)

    @staticmethod
    def memory_bytes(adapter: Any, loaded: Any) -> int | None:
        return adapter.memory_bytes(loaded)

    @staticmethod
    def token_counts(adapter: Any, loaded: Any, request: Any, result: Any) -> tuple[int | None, int | None]:
        return adapter.token_counts(loaded, request, result)

    @staticmethod
    def clock() -> float:
        return time.perf_counter()
