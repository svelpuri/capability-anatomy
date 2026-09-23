"""Behavioral contracts for v2; legacy observations remain reconstruction-only."""
from dataclasses import asdict
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.discovery import PluginDiscovery
from capability_anatomy.domain import NormalizedRecord, ObservationStatus
from capability_anatomy.errors import InvalidEvidenceError
from capability_anatomy.evaluations import EvaluationExecutor
from capability_anatomy.evaluations.plugins.phase5 import Phase5EvaluationSuite
from capability_anatomy.execution.phase5_campaign import (
    _manifest_semantics_version, aggregate_baseline_controls, aggregate_observations,
    compute_damage_matrix, compute_validation_results, select_discovery_candidates,
)
from capability_anatomy.telemetry import OperationTelemetry


def record(kind, expected, identity="example", **metadata):
    return NormalizedRecord(id=identity, partition="discovery", input={}, expected=expected,
                            metadata={"kind": kind, "group_id": identity, **metadata})


def evaluated(suite, item, raw):
    observation = EvaluationExecutor().evaluate(suite, item, raw)
    assert observation.status is ObservationStatus.COMPLETE
    return observation


def durable(observation, repetition=0, condition="baseline"):
    return {"condition": condition, "component_id": "baseline", "partition": observation.partition,
            "example_id": observation.example_id, "group_id": observation.example_id,
            "repetition": repetition, "status": "complete", "expected_metric_ids": list(observation.scores),
            "scores": {metric: asdict(score) for metric, score in observation.scores.items()}}


@pytest.mark.parametrize("answer,expected,correct", [
    ("42", "42", True), (" \t42\n", "42", True), ("-42", "42", False),
    ("4.2", "42", False), ("4/2", "42", False), ("4^2", "42", False),
    ("4 2", "42", False), ("4,2", "42", False), ("４２", "42", False),
    ("−42", "42", False), ("4e2", "42", False), ("foo-bar", "foobar", False),
    ("A_1", "a1", False), ("Admin", "admin", False), ("a/b", "ab", False),
    ("43", "42", False),
])
def test_typed_reasoning_answers_through_executor_and_aggregate(answer, expected, correct):
    suite = Phase5EvaluationSuite()
    observation = evaluated(suite, record("reasoning", expected), answer)
    aggregate = aggregate_observations([durable(observation)], ["structured_reasoning"])
    assert aggregate["structured_reasoning"]["value"] == float(correct)
    assert suite.aggregate([observation])["metrics"]["structured_reasoning"] == float(correct)


@pytest.mark.parametrize("actual,expected,correct", [
    ("42", "42", True), ("-42", "42", False), ("4.2", "42", False),
    ("4/2", "42", False), ("4^2", "42", False), ("Admin", "admin", False),
    (42, 42, True), (42.0, 42, True), ("42", 42, False), (True, 1, False),
    ({"count": "-42"}, {"count": "42"}, False), (["4.2"], ["42"], False),
])
def test_typed_tool_arguments_through_public_executor(actual, expected, correct):
    suite = Phase5EvaluationSuite()
    item = record("simple", [{"lookup": {"value": [expected]}}],
                  scoring_schema={"argument_names": ["value"], "required": ["value"]})
    observation = evaluated(suite, item, json.dumps([{"name": "lookup", "arguments": {"value": actual}}]))
    scores = suite.aggregate([observation])["metrics"]
    assert scores["tool_selection"] == 1
    assert scores["argument_binding"] == scores["full_call"] == float(correct)


@pytest.mark.parametrize("raw,expected,reason", [
    ("[]", 1, "abstention_empty_call_list"),
    (" No applicable tool. ", 1, "abstention_explicit_form"),
    ("I cannot complete this request with the available tools.", 1, "abstention_explicit_form"),
    ("", 0, "abstention_empty_or_invalid_type"),
    ("  ", 0, "abstention_empty_or_invalid_type"),
    ("gibberish 9$%!", 0, "abstention_unrecognized_or_malformed"),
    ("[", 0, "abstention_unrecognized_or_malformed"),
    ('{"name": "lookup", "arguments": {}}', 0, "abstention_unrecognized_or_malformed"),
    ('<tool_call>{"name":', 0, "abstention_unrecognized_or_malformed"),
    ('[{"name":"lookup","arguments":{}}]', 0, "abstention_tool_invocation"),
    ('<tool_call>{"name":"lookup","arguments":{}}</tool_call>', 0, "abstention_tool_invocation"),
    ("No applicable tool. But I will call lookup.", 0, "abstention_unrecognized_or_malformed"),
])
def test_abstention_validity_denials_and_reason_spans(raw, expected, reason):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry = OperationTelemetry.create("evaluation", tracer=provider.get_tracer("release-evaluation"))
    suite = Phase5EvaluationSuite(telemetry=telemetry)
    observation = evaluated(suite, record("abstention", []), raw)
    assert suite.aggregate([observation])["metrics"]["abstention"] == expected
    assert aggregate_observations([durable(observation)], ["abstention"])["abstention"]["value"] == expected
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["capability_anatomy.evaluation.version"] == "2"
    assert reason in [event.attributes.get("capability_anatomy.reason") for event in spans[0].events]
    assert spans[0].end_time >= spans[0].start_time
    provider.shutdown()


def test_repetitions_and_baseline_positions_do_not_multiply_sample_counts():
    suite = Phase5EvaluationSuite()
    observations = [evaluated(suite, record("reasoning", "42", identity=identity), "43") for identity in ("a", "b")]
    rows = [durable(item, repetition) for repetition in range(20) for item in observations]
    aggregate = aggregate_observations(rows, ["structured_reasoning"])
    metric = aggregate["structured_reasoning"]
    assert metric["complete_count"] == 40
    assert metric["unique_record_count"] == metric["unique_group_count"] == 2
    assert metric["repetition_standard_deviation"] == 0
    assert metric["sample_uncertainty_95"] is None
    assert metric["sample_uncertainty_status"] == "not_estimated_records_not_assumed_independent"
    assert "uncertainty_95" not in metric
    assert metric["retained_competence"] == "not_established"
    baseline, drift = aggregate_baseline_controls(dict.fromkeys(("beginning", "middle", "end"), aggregate), aggregate)
    assert baseline["structured_reasoning"]["complete_count"] == 120
    assert baseline["structured_reasoning"]["unique_record_count"] == baseline["structured_reasoning"]["unique_group_count"] == 2
    assert baseline["structured_reasoning"]["sample_uncertainty_95"] is None
    assert baseline["structured_reasoning"]["sample_uncertainty_status"] == "not_estimated_records_not_assumed_independent"
    assert drift == {"structured_reasoning": 0}


RULE = {"target_absolute_damage_max": 0.05, "collateral_absolute_damage_max": 0.1,
        "perplexity_relative_damage_max": 0.1, "minimum_drift_margin": 0.025}
PROTOCOL = {"full_call": {"direction": "higher_is_better", "role": "target"}}


def matrix(baseline_value, condition_value, count=20):
    def metric(value):
        return {"value": value, "semantics_version": "2", "complete_count": count,
                "unique_record_count": count, "unique_group_count": count,
                "minimum": value, "maximum": value, "error_count": 0}
    return compute_damage_matrix({"full_call": metric(baseline_value)}, {"block": {"full_call": metric(condition_value)}}, PROTOCOL)


@pytest.mark.parametrize("damage,eligible", [(0.05, True), (1.0 - 0.95, True), (0.05 + 2e-9, False), (0.051, False)])
def test_discovery_validation_share_boundary_through_damage_pipeline(damage, eligible):
    evidence = matrix(1.0, 1.0 - damage)
    ranking = select_discovery_candidates(evidence, PROTOCOL, RULE, drift_by_metric={"full_call": 0},
                                         matched_random_damage={"block": {"full_call": evidence["block"]["full_call"]["absolute_damage"]}})
    validation = compute_validation_results(["block"], evidence, PROTOCOL, RULE)
    assert ranking["ranking"][0]["eligible"] is eligible
    assert validation["results"][0]["outcome"] == ("pass" if eligible else "fail")


@pytest.mark.parametrize("baseline_value,count,reason", [(0, 20, "baseline_competence_absent"), (1, 1, "insufficient_distinct_records_or_groups")])
def test_no_competence_or_one_group_cannot_be_eligible(baseline_value, count, reason):
    evidence = matrix(baseline_value, baseline_value, count)
    ranking = select_discovery_candidates(evidence, PROTOCOL, RULE, drift_by_metric={"full_call": 0}, matched_random_damage={"block": {"full_call": 0}})
    validation = compute_validation_results(["block"], evidence, PROTOCOL, RULE)
    assert ranking["candidates"] == []
    assert "full_call:" + reason in ranking["ranking"][0]["reasons"]
    assert validation["results"][0]["outcome"] == "inconclusive"


def test_legacy_aggregation_is_explicit_and_not_relabelled():
    row = durable(evaluated(Phase5EvaluationSuite(), record("reasoning", "42"), "43"))
    old = aggregate_observations([row, {**row, "repetition": 1}], ["structured_reasoning"], semantics_version="1")
    assert old == {"structured_reasoning": {"value": 0, "numerator": 0, "denominator": 2, "complete_count": 2,
        "error_count": 0, "repetition_values": [0, 0], "minimum": 0, "maximum": 0, "uncertainty_95": [0, 0]}}
    assert compute_validation_results(["block"], matrix(1, 0.95), PROTOCOL, RULE, semantics_version="1")["results"][0]["outcome"] == "fail"
    assert compute_validation_results(["block"], matrix(1, 0.95), PROTOCOL, RULE)["results"][0]["outcome"] == "pass"


@pytest.mark.parametrize("version", [None, "0", "3", "v2", 2, [], {}, True])
def test_unknown_manifest_evaluator_semantics_refuse(version):
    with pytest.raises(InvalidEvidenceError, match="version"):
        _manifest_semantics_version({"plugins": [{"role": "evaluation", "name": Phase5EvaluationSuite.name, "version": version}]})


def test_public_discovery_only_exposes_current_evaluator():
    resolved = PluginDiscovery().resolve("evaluation", Phase5EvaluationSuite.name,
                                        required_capabilities=frozenset({"evaluation.phase5"}), required_methods=("parse", "score"))
    assert resolved.plugin.version == "2"
    for version in ("1", "2"):
        assert _manifest_semantics_version({"plugins": [{"role": "evaluation", "name": Phase5EvaluationSuite.name, "version": version}]}) == version


@pytest.mark.parametrize("plugins", [None, {}, [None], [2], [], [{"role": "model"}]])
def test_malformed_evaluation_identities_fail_with_typed_refusal(plugins):
    with pytest.raises(InvalidEvidenceError, match="identity"):
        _manifest_semantics_version({"plugins": plugins})


def test_candidate_competence_decisions_have_child_spans_and_reason_names(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    signals = OperationTelemetry.create("evaluation", tracer=provider.get_tracer("candidate-decisions"))
    monkeypatch.setattr("capability_anatomy.execution.phase5_campaign.OperationTelemetry.create", lambda *_args: signals)
    evidence = matrix(0, 0)
    with signals.tracer.start_as_current_span("parent") as parent:
        parent.set_attribute("capability_anatomy.reason", "parent_reason")
        select_discovery_candidates(evidence, PROTOCOL, RULE, drift_by_metric={"full_call": 0}, matched_random_damage={"block": {"full_call": 0}})
        compute_validation_results(["block", "missing"], evidence, PROTOCOL, RULE)
    spans = exporter.get_finished_spans()
    parent_span = next(span for span in spans if span.name == "parent")
    assert parent_span.attributes["capability_anatomy.reason"] == "parent_reason"
    children = [span for span in spans if span.name != "parent"]
    assert len(children) == 3
    assert all(span.parent.span_id == parent_span.context.span_id for span in children)
    assert [span.attributes["capability_anatomy.reason"] for span in children] == ["baseline_competence_absent", "baseline_competence_absent", "validation_missing"]
    assert all(span.end_time >= span.start_time for span in children)
    provider.shutdown()
