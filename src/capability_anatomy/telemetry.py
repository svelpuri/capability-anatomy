from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from typing import Iterator, Mapping

from opentelemetry import metrics, trace
from opentelemetry.metrics import Counter, Meter
from opentelemetry.trace import Span, Status, StatusCode, Tracer


_ACTIVE_SIGNALS: ContextVar[tuple[Tracer, Meter] | None] = ContextVar("capability_anatomy.signals", default=None)
_IDENTITIES: ContextVar[Mapping[str, str]] = ContextVar("capability_anatomy.identities", default={})


@contextmanager
def telemetry_scope(tracer: Tracer, meter: Meter) -> Iterator[None]:
    """Pass one run's SDK through plugins without changing global providers."""
    token = _ACTIVE_SIGNALS.set((tracer, meter))
    try:
        yield
    finally:
        _ACTIVE_SIGNALS.reset(token)


@contextmanager
def identity_scope(**identifiers: str) -> Iterator[None]:
    # Identifiers originate in authored datasets/configs, so correlate by digest.
    values = dict(_IDENTITIES.get())
    for name, value in identifiers.items():
        if name not in {"experiment", "task", "record", "component"}:
            raise ValueError("unknown telemetry identity kind")
        values[f"capability_anatomy.{name}_id_sha256"] = hashlib.sha256(value.encode()).hexdigest()
    token = _IDENTITIES.set(values)
    try:
        yield
    finally:
        _IDENTITIES.reset(token)


def _providers(name: str, tracer: Tracer | None, meter: Meter | None) -> tuple[Tracer, Meter]:
    active = _ACTIVE_SIGNALS.get()
    return (
        tracer or (active[0] if active else trace.get_tracer(name)),
        meter or (active[1] if active else metrics.get_meter(name)),
    )


@dataclass(frozen=True)
class DecisionTelemetry:
    tracer: Tracer
    decisions: Counter

    @classmethod
    def create(cls, tracer: Tracer | None = None, meter: Meter | None = None) -> "DecisionTelemetry":
        actual_tracer, actual_meter = _providers("capability_anatomy.config", tracer, meter)
        return cls(
            tracer=actual_tracer,
            decisions=actual_meter.create_counter(
                "capability_anatomy.config.decisions",
                unit="{decision}",
                description="Configuration acceptance and refusal decisions",
            ),
        )

    def record(self, span: Span, *, outcome: str, reason: str, schema_version: str = "unknown") -> None:
        attributes = {
            "capability_anatomy.component": "config",
            "capability_anatomy.outcome": outcome,
            "capability_anatomy.reason": reason,
            "capability_anatomy.schema_version": schema_version,
        }
        span.set_attributes(_IDENTITIES.get())
        span.add_event("configuration.decision", attributes)
        span.set_attribute("capability_anatomy.outcome", outcome)
        span.set_attribute("capability_anatomy.reason", reason)
        if outcome == "refused":
            span.set_status(Status(StatusCode.ERROR, reason))
        self.decisions.add(1, attributes)


@dataclass(frozen=True)
class OperationTelemetry:
    tracer: Tracer
    decisions: Counter
    component: str

    @classmethod
    def create(
        cls,
        component: str,
        tracer: Tracer | None = None,
        meter: Meter | None = None,
    ) -> "OperationTelemetry":
        actual_tracer, actual_meter = _providers(f"capability_anatomy.{component}", tracer, meter)
        return cls(
            tracer=actual_tracer,
            decisions=actual_meter.create_counter(
                "capability_anatomy.operation.decisions",
                unit="{decision}",
                description="Plugin and operation lifecycle decisions",
            ),
            component=component,
        )

    def record(self, span: Span, *, operation: str, outcome: str, reason: str) -> None:
        attributes = {
            "capability_anatomy.component": self.component,
            "capability_anatomy.operation": operation,
            "capability_anatomy.outcome": outcome,
            "capability_anatomy.reason": reason,
        }
        span.set_attributes(_IDENTITIES.get())
        span.add_event("operation.decision", attributes)
        span.set_attribute("capability_anatomy.operation", operation)
        span.set_attribute("capability_anatomy.outcome", outcome)
        span.set_attribute("capability_anatomy.reason", reason)
        self.decisions.add(1, attributes)
