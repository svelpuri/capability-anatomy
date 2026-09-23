from __future__ import annotations

from dataclasses import replace

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.datasets import SyntheticDatasetProvider
from capability_anatomy.domain import EvaluationScore, NormalizedRecord, ObservationStatus, ParseResult, ParseStatus
from capability_anatomy.errors import InvalidConfigurationError, UnsupportedPluginError
from capability_anatomy.evaluations import EvaluationExecutor
from capability_anatomy.evaluations.plugins import SyntheticExactMatchSuite, ToolCallingSuite
from capability_anatomy.registry import PluginRegistry
from capability_anatomy.telemetry import OperationTelemetry


def _signals() -> tuple[OperationTelemetry, InMemorySpanExporter, InMemoryMetricReader]:
    exporter = InMemorySpanExporter()
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    return (
        OperationTelemetry.create(
            "evaluation",
            tracer=trace_provider.get_tracer("phase3-test"),
            meter=meter_provider.get_meter("phase3-test"),
        ),
        exporter,
        reader,
    )


def _record(record_id: str, expected: object, *, partition: str = "discovery") -> NormalizedRecord:
    return NormalizedRecord(record_id, partition, {"prompt": record_id}, expected, {})


def _values(observation) -> dict[str, float]:
    return {metric: score.value for metric, score in observation.scores.items()}


def test_evaluation_and_dataset_registries_are_independent() -> None:
    evaluation_registry = PluginRegistry(required_methods=("parse", "score", "aggregate"))
    dataset_registry = PluginRegistry(required_methods=("metadata", "records", "validate_for"))
    suite = SyntheticExactMatchSuite()
    provider = SyntheticDatasetProvider({"discovery": (_record("one", "yes"),)})

    evaluation_registry.register(suite)
    dataset_registry.register(provider)

    assert evaluation_registry.resolve(suite.name) is suite
    assert dataset_registry.resolve(provider.name) is provider
    with pytest.raises(UnsupportedPluginError, match="not installed"):
        evaluation_registry.resolve(provider.name)


def test_synthetic_dataset_has_stable_provenance_and_validates_suite_schema() -> None:
    records = (_record("one", "yes"), _record("two", "no"))
    provider = SyntheticDatasetProvider({"discovery": records})
    provider.validate_for(SyntheticExactMatchSuite())

    assert tuple(provider.records("discovery")) == records
    assert provider.stable_id(records[0]) == "one"
    assert provider.metadata().license == "CC0-1.0"
    assert len(provider.metadata().sha256) == 64
    with pytest.raises(InvalidConfigurationError, match="unavailable"):
        tuple(provider.records("final"))

    duplicate = SyntheticDatasetProvider({"discovery": (records[0], records[0])})
    with pytest.raises(InvalidConfigurationError, match="unique"):
        duplicate.validate_for(SyntheticExactMatchSuite())
    inconsistent = SyntheticDatasetProvider({"validation": (records[0],)})
    with pytest.raises(InvalidConfigurationError, match="inconsistent"):
        inconsistent.validate_for(SyntheticExactMatchSuite())


def test_aggregates_reconstruct_only_from_raw_observations() -> None:
    suite = SyntheticExactMatchSuite()
    executor = EvaluationExecutor()
    observations = (
        executor.evaluate(suite, _record("one", "yes"), "yes"),
        executor.evaluate(suite, _record("two", "yes"), "no"),
    )

    assert [_values(item) for item in observations] == [{"exact_match": 1.0}, {"exact_match": 0.0}]
    assert suite.aggregate(iter(observations)) == {
        "metrics": {"exact_match": 0.5},
        "sample_counts": {"exact_match": 2},
        "complete": 2,
        "errors": 0,
    }

    changed = (replace(observations[0], scores={"exact_match": EvaluationScore(0.0, 0.0, 1)}), observations[1])
    assert suite.aggregate(changed)["metrics"]["exact_match"] == 0.0


def test_parse_failure_is_a_durable_observation_not_an_exception() -> None:
    suite = ToolCallingSuite()
    observation = EvaluationExecutor().evaluate(suite, _record("bad", []), "not-json")

    assert observation.status is ObservationStatus.ERROR
    assert observation.parsed.status is ParseStatus.ERROR
    assert observation.parsed.diagnostics == {"error_type": "JSONDecodeError"}
    assert observation.error_type == "JSONDecodeError"
    assert observation.scores == {}
    assert suite.aggregate((observation,)) == {
        "metrics": {}, "sample_counts": {}, "complete": 0, "errors": 1
    }


def test_invalid_score_is_a_durable_error_observation() -> None:
    class NonFiniteSuite(SyntheticExactMatchSuite):
        def score(self, example, output):
            return {"exact_match": EvaluationScore(float("nan"))}

    observation = EvaluationExecutor().evaluate(NonFiniteSuite(), _record("bad-score", "yes"), "yes")

    assert observation.status is ObservationStatus.ERROR
    assert observation.parsed.status is ParseStatus.PARSED
    assert observation.scores == {}
    assert observation.error_type == "ValueError"


def test_tool_selection_binding_and_full_call_have_independent_populations() -> None:
    suite = ToolCallingSuite()
    executor = EvaluationExecutor()
    expected = [{"name": "weather", "arguments": {"city": "Paris"}}]

    wrong_arguments = executor.evaluate(
        suite,
        _record("positive", expected),
        '[{"name":"weather","arguments":{"city":"Rome"}}]',
    )
    partial_selection = executor.evaluate(
        suite,
        _record(
            "two-tools",
            [
                {"name": "weather", "arguments": {"city": "Paris"}},
                {"name": "calendar", "arguments": {"day": "Monday"}},
            ],
        ),
        '[{"name":"weather","arguments":{"city":"Paris"}}]',
    )

    assert _values(wrong_arguments) == {
        "tool_selection": 1.0,
        "argument_binding": 0.0,
        "full_call": 0.0,
    }
    assert _values(partial_selection) == {
        "tool_selection": 0.0,
        "full_call": 0.0,
        "argument_binding": 1.0,
    }
    aggregate = suite.aggregate((wrong_arguments, partial_selection))
    assert aggregate["metrics"] == {
        "argument_binding": 0.5,
        "full_call": 0.0,
        "tool_selection": 0.5,
    }
    assert aggregate["sample_counts"] == {
        "argument_binding": 2,
        "full_call": 2,
        "tool_selection": 2,
    }


def test_abstention_aggregate_uses_only_eligible_negative_examples() -> None:
    suite = ToolCallingSuite()
    executor = EvaluationExecutor()
    positive = executor.evaluate(
        suite,
        _record("positive", [{"name": "weather", "arguments": {}}]),
        '[{"name":"weather","arguments":{}}]',
    )
    abstained = executor.evaluate(suite, _record("negative", []), "[]")
    false_positive = executor.evaluate(
        suite, _record("negative-2", []), '[{"name":"weather","arguments":{}}]'
    )

    assert "abstention" not in positive.scores
    aggregate = suite.aggregate((positive, abstained, false_positive))
    assert aggregate["metrics"]["abstention"] == 0.5
    assert aggregate["sample_counts"]["abstention"] == 2


def test_returned_parse_error_cannot_become_a_complete_observation() -> None:
    class ReturnedErrorSuite(SyntheticExactMatchSuite):
        def parse(self, result):
            return ParseResult(ParseStatus.ERROR, diagnostics={"error_type": "SyntheticParseError"})

        def score(self, example, output):
            return {"exact_match": EvaluationScore(1.0, 1.0, 1)}

    observation = EvaluationExecutor().evaluate(ReturnedErrorSuite(), _record("returned", "yes"), "bad")

    assert observation.status is ObservationStatus.ERROR
    assert observation.parsed.status is ParseStatus.ERROR
    assert observation.error_type == "SyntheticParseError"
    assert observation.scores == {}


def test_evaluation_decisions_are_traceable_countable_and_redacted() -> None:
    telemetry, exporter, reader = _signals()
    executor = EvaluationExecutor(telemetry)
    secret = "private-output-do-not-record"

    complete = executor.evaluate(SyntheticExactMatchSuite(), _record("one", "yes"), "yes")
    failed = executor.evaluate(ToolCallingSuite(), _record("two", []), secret)

    assert complete.status is ObservationStatus.COMPLETE
    assert failed.status is ObservationStatus.ERROR
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "capability_anatomy.evaluation.record",
        "capability_anatomy.evaluation.record",
    ]
    events = [event for span in spans for event in span.events]
    assert [event.attributes["capability_anatomy.reason"] for event in events] == [
        "observation_complete",
        "parse_failure_recorded",
    ]
    assert spans[1].status.status_code.name == "ERROR"
    signal_text = repr(spans) + repr(reader.get_metrics_data())
    assert secret not in signal_text
    metric = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics[0]
    assert sum(point.value for point in metric.data.data_points) == 2
