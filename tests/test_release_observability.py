from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from capability_anatomy.runtime import LocalRuntimePlugin
from capability_anatomy.telemetry import OperationTelemetry, identity_scope, telemetry_scope


@pytest.fixture
def signals():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    meters = MeterProvider()
    yield provider, meters, exporter
    provider.shutdown()
    meters.shutdown()


def test_runtime_exception_is_redacted_at_the_sdk_export_boundary(signals):
    provider, meters, exporter = signals
    runtime = LocalRuntimePlugin(OperationTelemetry.create("runtime", provider.get_tracer("test"), meters.get_meter("test")))
    class Adapter:
        def execute(self, loaded, request):
            if request == "fail":
                raise RuntimeError("CANARY_RELEASE_CREDENTIAL")
            return "ok"
    assert runtime.execute(Adapter(), None, "positive") == "ok"
    with pytest.raises(RuntimeError):
        runtime.execute(Adapter(), None, "fail")
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].status.status_code is StatusCode.UNSET
    assert spans[1].status.status_code is StatusCode.ERROR
    assert spans[1].status.description == "adapter_execution_failed"
    assert spans[1].end_time > spans[1].start_time
    assert spans[1].events[0].attributes["capability_anatomy.reason"] == "adapter_execution_failed"
    serialized = json.dumps([json.loads(span.to_json()) for span in spans])
    assert "CANARY_RELEASE_CREDENTIAL" not in serialized
    assert "exception.stacktrace" not in serialized


def test_implicit_plugin_telemetry_uses_scoped_run_provider_and_safe_identity(signals):
    provider, meters, exporter = signals
    tracer = provider.get_tracer("run")
    with telemetry_scope(tracer, meters.get_meter("run")), identity_scope(record="private-record-canary"):
        with tracer.start_as_current_span("run") as root:
            runtime = LocalRuntimePlugin()
            class Adapter:
                def execute(self, loaded, request):
                    return "ok"
            assert runtime.execute(Adapter(), None, None) == "ok"
    child = next(span for span in exporter.get_finished_spans() if span.name.endswith("runtime.execute"))
    assert child.parent.span_id == root.get_span_context().span_id
    assert len(child.attributes["capability_anatomy.record_id_sha256"]) == 64
    assert "private-record-canary" not in child.to_json()
