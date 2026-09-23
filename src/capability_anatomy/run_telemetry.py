"""Local evidence plus bounded, observable asynchronous OTLP delivery."""
from __future__ import annotations

import os
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
import time
from threading import Lock

from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .errors import InvalidConfigurationError, InvalidEvidenceError, CapabilityAnatomyError


_EXPORT_LOG_CONTEXT = ContextVar("otlp_export_log_context", default=None)
_EXPORT_LOG_LOCK = Lock()
_EXPORT_LOG_USERS = {}


class _ExportLogFilter(logging.Filter):
    def filter(self, record):
        context = _EXPORT_LOG_CONTEXT.get()
        if context is not None:
            # SDK HTTP diagnostics include the server-controlled reason phrase.
            # Never format it, its arguments or exception details into a log.
            record.msg = json.dumps({"event": "otlp_export_diagnostic", "reason": "upstream_export_diagnostic",
                                     "signal": context[0], "trace_id": context[1]}, sort_keys=True)
            record.args = ()
            record.exc_info = record.exc_text = record.stack_info = None
        return True


_EXPORT_LOG_FILTER = _ExportLogFilter()


@contextmanager
def _export_diagnostics(logger_name, signal, trace_id):
    """Redact only this export's SDK logs; preserve concurrent caller logging."""
    names = tuple(dict.fromkeys((logger_name, "opentelemetry.util.re", "opentelemetry.exporter.otlp.proto.http")))
    loggers = [logging.getLogger(name) for name in names]
    with _EXPORT_LOG_LOCK:
        for logger in loggers:
            if not _EXPORT_LOG_USERS.get(logger.name):
                logger.addFilter(_EXPORT_LOG_FILTER)
            _EXPORT_LOG_USERS[logger.name] = _EXPORT_LOG_USERS.get(logger.name, 0) + 1
    token = _EXPORT_LOG_CONTEXT.set((signal, trace_id))
    try:
        yield
    finally:
        _EXPORT_LOG_CONTEXT.reset(token)
        with _EXPORT_LOG_LOCK:
            for logger in loggers:
                _EXPORT_LOG_USERS[logger.name] -= 1
                if not _EXPORT_LOG_USERS[logger.name]:
                    logger.removeFilter(_EXPORT_LOG_FILTER)
                    del _EXPORT_LOG_USERS[logger.name]


class OTLPHeaderError(InvalidConfigurationError):
    reason = "otlp_headers_invalid"


class TelemetryCapacityError(InvalidEvidenceError):
    reason = "telemetry_capacity_exceeded"


class TelemetryDeliveryError(CapabilityAnatomyError):
    reason = "telemetry_delivery_incomplete"


class BoundedSpanExporter(SpanExporter):
    """O(1) admission and bounded diagnostic retention, including terminal root."""
    def __init__(self, limit: int = 50000):
        self.limit = limit
        self.overflow = False
        self.dropped_spans = 0
        self._spans = []
        self._bound_lock = Lock()

    def export(self, spans):
        with self._bound_lock:
            for span in spans:
                if len(self._spans) < self.limit:
                    self._spans.append(span)
                else:
                    self.overflow = True
                    self.dropped_spans += 1
                    # Diagnostic-only retention replaces one already incomplete
                    # prefix entry so the actual invocation outcome remains visible.
                    if span.name in {"capability_anatomy.invocation", "capability_anatomy.cli.command"} and self.limit:
                        self._spans[-1] = span
            return SpanExportResult.FAILURE if self.overflow else SpanExportResult.SUCCESS

    def get_finished_spans(self):
        if self.overflow:
            raise TelemetryCapacityError("telemetry capacity exceeded; retained diagnostic trace is incomplete")
        return self.diagnostic_spans()

    def diagnostic_spans(self):
        with self._bound_lock:
            return tuple(self._spans)

    def shutdown(self):
        pass


def sdk_disabled() -> bool:
    return os.environ.get("OTEL_SDK_DISABLED", "false").strip().lower() == "true"


class _Delivery:
    """Accounting around the upstream exporters, never token/header logging."""
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.deadline = None
        self.trace_id = None
        self.failed = False
        self.sent = 0
        self.lost = 0
        self._lock = Lock()

    def begin(self, count):
        with self._lock:
            if self.failed or (self.deadline is not None and time.monotonic() >= self.deadline):
                self.failed = True
                self.lost += count
                return False
            return True

    def end(self, count, success):
        with self._lock:
            if success:
                self.sent += count
            else:
                self.lost += count
                self.failed = True  # circuit breaker: no per-span retry amplification


class _TraceDelivery(SpanExporter):
    def __init__(self, exporter, state):
        self.exporter, self.state = exporter, state

    def export(self, spans):
        if not self.state.begin(len(spans)):
            return SpanExportResult.FAILURE
        identities = {f"{span.context.trace_id:032x}" for span in spans}
        self.state.trace_id = next(iter(identities)) if len(identities) == 1 else None
        try:
            with _export_diagnostics("opentelemetry.exporter.otlp.proto.http.trace_exporter", "traces", self.state.trace_id):
                result = self.exporter.export(spans)
        except Exception:
            result = SpanExportResult.FAILURE
        self.state.end(len(spans), result is SpanExportResult.SUCCESS)
        return result

    def shutdown(self):
        with _export_diagnostics("opentelemetry.exporter.otlp.proto.http.trace_exporter", "traces", self.state.trace_id):
            self.exporter.shutdown()


class _CountedBatch(BatchSpanProcessor):
    def __init__(self, exporter):
        # Queue and local evidence have the same declared cardinality bound.
        super().__init__(exporter, max_queue_size=50000, max_export_batch_size=512,
                         schedule_delay_millis=1000)
        self.ended = 0
        self._count_lock = Lock()

    def on_end(self, span):
        if span.context and span.context.trace_flags.sampled:
            with self._count_lock:
                self.ended += 1
        super().on_end(span)


class OTLPDelivery:
    def __init__(self, processor, traces, metrics, metric_reader):
        self.processor, self.traces = processor, traces
        self.metrics, self.metric_reader = metrics, metric_reader

    def flush(self):
        # OTel Python's BatchSpanProcessor currently ignores force_flush's timeout.
        # Bound whole drains in the exporter adapters, in addition to each upstream
        # HTTP export's timeout. At most one in-flight call can cross this deadline.
        now = time.monotonic()
        for state in (self.traces, self.metrics):
            if state.deadline is None:
                state.deadline = now + state.timeout
        self.processor.force_flush()
        try:
            self.metric_reader.force_flush(timeout_millis=max(1, int(self.metrics.timeout * 1000)))
        except Exception:
            self.metrics.failed = True
        return self.snapshot()

    def snapshot(self):
        outstanding = max(0, self.processor.ended - self.traces.sent - self.traces.lost)
        return {"configured": True, "status": "incomplete" if self.traces.failed or self.metrics.failed or outstanding else "delivered",
                "spans_submitted": self.processor.ended, "spans_delivered": self.traces.sent,
                "spans_not_delivered": self.traces.lost + outstanding,
                "metric_exports_delivered": self.metrics.sent, "metric_exports_failed": self.metrics.lost}


def configure_otlp(tracer_provider, metric_readers):
    """Configure upstream HTTP exporters, async spans and explicit delivery health."""
    if sdk_disabled() or not any(os.environ.get(key) for key in (
        "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    )):
        return None
    from opentelemetry.util.re import parse_env_headers
    for signal in ("TRACES", "METRICS"):
        headers = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal}_HEADERS", os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", ""))
        with _export_diagnostics("opentelemetry.util.re", "configuration", None):
            # Compose the same upstream W3C/liberal parser used by the SDK.
            # Per-member validation catches ignored malformed credentials even
            # when the caller has disabled SDK warning logs; duplicates retain
            # the SDK's last-value semantics.
            for member in headers.split(","):
                if member.strip() and not parse_env_headers(member, liberal=True):
                    raise OTLPHeaderError("OTLP headers are malformed; use comma-separated name=value entries")
        protocol = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal}_PROTOCOL", os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"))
        if protocol != "http/protobuf":
            raise InvalidConfigurationError("OTLP signal protocol must be http/protobuf")
    def timeout(signal):
        raw = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal}_TIMEOUT", os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT", "5"))
        try:
            value = float(raw)
        except (ValueError, TypeError):
            raise InvalidConfigurationError("OTLP timeout must be finite positive seconds") from None
        import math
        if not math.isfinite(value) or value <= 0:
            raise InvalidConfigurationError("OTLP timeout must be finite positive seconds")
        return value
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, MetricExportResult, MetricExporter
    except ImportError as error:
        raise InvalidConfigurationError("OTLP HTTP exporter is unavailable; repair the capability-anatomy installation") from error
    traces, metrics = _Delivery(timeout("TRACES")), _Delivery(timeout("METRICS"))
    with _export_diagnostics("opentelemetry.exporter.otlp.proto.http.trace_exporter", "traces", None):
        trace_exporter = OTLPSpanExporter(timeout=traces.timeout)
    processor = _CountedBatch(_TraceDelivery(trace_exporter, traces))
    tracer_provider.add_span_processor(processor)
    class CountedMetrics(MetricExporter):
        def __init__(self):
            super().__init__()
            with _export_diagnostics("opentelemetry.exporter.otlp.proto.http.metric_exporter", "metrics", traces.trace_id):
                self.exporter = OTLPMetricExporter(timeout=metrics.timeout)
        def export(self, metrics_data, timeout_millis=10000, **kwargs):
            if not metrics.begin(1):
                return MetricExportResult.FAILURE
            try:
                with _export_diagnostics("opentelemetry.exporter.otlp.proto.http.metric_exporter", "metrics", traces.trace_id):
                    result = self.exporter.export(metrics_data, timeout_millis=timeout_millis, **kwargs)
            except Exception:
                result = MetricExportResult.FAILURE
            metrics.end(1, result is MetricExportResult.SUCCESS)
            return result
        def force_flush(self, timeout_millis=10000):
            return not metrics.failed
        def shutdown(self, timeout_millis=30000, **kwargs):
            with _export_diagnostics("opentelemetry.exporter.otlp.proto.http.metric_exporter", "metrics", traces.trace_id):
                self.exporter.shutdown(timeout_millis=timeout_millis, **kwargs)
    metric_reader = PeriodicExportingMetricReader(CountedMetrics(), export_interval_millis=60000,
                                                export_timeout_millis=int(metrics.timeout * 1000))
    metric_readers.append(metric_reader)
    return OTLPDelivery(processor, traces, metrics, metric_reader)
