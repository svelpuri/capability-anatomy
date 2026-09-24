from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.cli import main
from capability_anatomy.errors import InvalidEvidenceError
from capability_anatomy.execution.phase5_campaign import (
    _bypass_hooks_balanced,
    _derived_component_order,
    verify_phase5_aggregates,
    _verify_operational_evidence,
    aggregate_observations,
    aggregate_baseline_controls,
    build_campaign_plan,
    compute_damage_matrix,
    compute_validation_results,
    enforce_budgets,
    enforce_stop_rules,
    reconstruct_manifest,
    render_report_markdown,
    select_discovery_candidates,
    serialize_campaign_plan,
    summarize_measurements,
    write_final_artifacts,
)
from capability_anatomy.execution.runner import ExperimentRunner, Task
from capability_anatomy.execution.store import FrozenRunStore
from capability_anatomy.serialization import canonical_sha256
from capability_anatomy.telemetry import OperationTelemetry


METRICS = {
    "selection": {"role": "target", "direction": "higher_is_better"},
    "format": {"role": "collateral", "direction": "higher_is_better"},
    "perplexity": {"role": "collateral", "direction": "lower_is_better"},
}
RULE = {
    "target_absolute_damage_max": 0.05,
    "collateral_absolute_damage_max": 0.1,
    "perplexity_relative_damage_max": 0.1,
    "minimum_drift_margin": 0.025,
}


def test_campaign_plan_is_baseline_first_complete_seeded_and_result_independent() -> None:
    controls = {
        "no_op_hook": True,
        "repeated_baseline_positions": ["beginning", "middle", "end"],
        "matched_random_components": 1,
        "placement": {"middle_after_fraction": 0.5},
    }
    first = build_campaign_plan(("block.2", "block.0", "block.1"), controls, seed=91)
    assert first[0].id == "baseline.beginning"
    assert [task.id for task in first if task.condition == "baseline"] == [
        "baseline.beginning", "baseline.middle", "baseline.end"
    ]
    assert [task.component_id for task in first if task.stage == "scan"] == ["block.2", "block.0", "block.1"]
    assert len([task for task in first if task.condition == "no_op"]) == 1
    assert len([task for task in first if task.condition == "matched_random"]) == 1
    assert first[-1].id == "baseline.end"
    assert serialize_campaign_plan(first) == serialize_campaign_plan(
        build_campaign_plan(("block.2", "block.0", "block.1"), controls, seed=91)
    )
    assert serialize_campaign_plan(first) != serialize_campaign_plan(
        build_campaign_plan(("block.2", "block.0", "block.1"), controls, seed=9)
    )


def test_campaign_plan_rejects_incomplete_controls_and_duplicate_components() -> None:
    controls = {"repeated_baseline_positions": ["beginning", "middle", "end"], "matched_random_components": 0, "placement": {"middle_after_fraction": 0.5}}
    with pytest.raises(InvalidEvidenceError, match="duplicated"):
        build_campaign_plan(("a", "a"), controls, seed=1)
    broken = copy.deepcopy(controls)
    broken["repeated_baseline_positions"] = ["beginning", "end"]
    with pytest.raises(InvalidEvidenceError, match="positions"):
        build_campaign_plan(("a", "b"), broken, seed=1)


def test_one_component_campaign_still_places_all_three_baselines() -> None:
    controls = {"no_op_hook": True, "repeated_baseline_positions": ["beginning", "middle", "end"], "matched_random_components": 1, "placement": {"middle_after_fraction": 0.5}}
    plan = build_campaign_plan(("only.block",), controls, seed=1)
    assert [task.id for task in plan if task.condition == "baseline"] == [
        "baseline.beginning", "baseline.middle", "baseline.end",
    ]


def _observation(example: str, repetition: int, selection: float, numerator: float, status: str = "complete") -> dict:
    return {
        "condition": "baseline", "component_id": "none", "partition": "discovery",
        "example_id": example, "repetition": repetition, "status": status,
        "expected_metric_ids": ["selection"],
        "scores": {"selection": {"value": selection, "numerator": numerator, "denominator": 2}},
    }


def test_aggregation_preserves_values_fractions_counts_and_rejects_duplicates() -> None:
    rows = [_observation("a", 0, 0.5, 1), _observation("b", 0, 1, 2), _observation("c", 0, 0, 0, "error")]
    metric = aggregate_observations(rows, ("selection",), semantics_version="1")["selection"]
    assert metric == {
        "value": 0.75, "numerator": 3.0, "denominator": 4,
        "complete_count": 2, "error_count": 1,
        "repetition_values": [0.75], "minimum": 0.75, "maximum": 0.75,
        "uncertainty_95": None,
    }
    with pytest.raises(InvalidEvidenceError, match="duplicated"):
        aggregate_observations([rows[0], rows[0]], ("selection",))
    with pytest.raises(InvalidEvidenceError, match="missing|does not exist"):
        aggregate_observations([rows[0]], ("format",))
    missing = copy.deepcopy(rows)
    missing[1]["scores"] = {}
    with pytest.raises(InvalidEvidenceError, match="per-record"):
        aggregate_observations(missing, ("selection",))


def test_measurement_summary_names_observation_memory_and_retains_counts() -> None:
    rows = [
        {"status": "complete", "elapsed_seconds": 1.0, "memory_observation_bytes": 10, "input_tokens": 3, "output_tokens": 2},
        {"status": "complete", "elapsed_seconds": 3.0, "memory_observation_bytes": 14, "input_tokens": 5, "output_tokens": 4},
        {"status": "error", "elapsed_seconds": 100.0, "memory_observation_bytes": 100},
    ]
    summary = summarize_measurements(rows)
    assert summary["elapsed_seconds"] == {"count": 2, "mean": 2.0, "minimum": 1.0, "maximum": 3.0}
    assert summary["memory_observation_bytes"]["maximum"] == 14
    assert "peak_memory_bytes" not in summary


def _aggregate(value: float, *, errors: int = 0) -> dict:
    return {"value": value, "numerator": value * 4, "denominator": 4, "complete_count": 4, "error_count": errors, "minimum": value - 0.005, "maximum": value + 0.005, "repetition_values": [value - 0.005, value + 0.005]}


def test_baseline_reference_is_three_position_mean_and_drift_includes_no_op() -> None:
    baselines = {
        "beginning": {"selection": _aggregate(0.9)},
        "middle": {"selection": _aggregate(0.88)},
        "end": {"selection": _aggregate(0.86)},
    }
    reference, drift = aggregate_baseline_controls(baselines, {"selection": _aggregate(0.91)})
    assert reference["selection"]["value"] == pytest.approx(0.88)
    assert reference["selection"]["numerator"] == pytest.approx(10.56)
    assert reference["selection"]["denominator"] == 12
    assert drift["selection"] == pytest.approx(0.03)


def test_damage_direction_and_conservative_discovery_and_validation() -> None:
    baseline = {"selection": _aggregate(0.9), "format": _aggregate(0.8), "perplexity": _aggregate(10)}
    conditions = {
        "good": {"selection": _aggregate(0.86), "format": _aggregate(0.76), "perplexity": _aggregate(10.5)},
        "bad": {"selection": _aggregate(0.7), "format": _aggregate(0.6), "perplexity": _aggregate(13)},
    }
    matrix = compute_damage_matrix(baseline, conditions, METRICS)
    assert matrix["good"]["selection"]["absolute_damage"] == pytest.approx(0.04)
    assert matrix["good"]["perplexity"]["absolute_damage"] == pytest.approx(0.05)
    assert matrix["good"]["selection"]["baseline"]["numerator"] == 3.6
    matched = {"good": {metric: matrix["good"][metric]["absolute_damage"] for metric in METRICS}}
    ranking = select_discovery_candidates(
        matrix, METRICS, RULE,
        drift_by_metric={metric: 0.01 for metric in METRICS},
        matched_random_damage=matched,
    )
    assert ranking["candidates"] == ["good"]
    assert {row["component_id"]: row["classification"] for row in ranking["ranking"]} == {
        "bad": "broadly_sensitive", "good": "low_observed_sensitivity"
    }
    validation = compute_validation_results(("good", "missing"), {"good": matrix["good"]}, METRICS, RULE, semantics_version="1")
    assert [item["outcome"] for item in validation["results"]] == ["pass", "inconclusive"]


def test_candidate_is_inconclusive_without_controls_or_complete_evidence() -> None:
    baseline = {metric: _aggregate(1 if metric != "perplexity" else 10) for metric in METRICS}
    condition = {metric: _aggregate(0.99 if metric != "perplexity" else 10.1) for metric in METRICS}
    matrix = compute_damage_matrix(baseline, {"block": condition}, METRICS)
    result = select_discovery_candidates(matrix, METRICS, RULE, drift_by_metric={}, matched_random_damage={})
    assert result["candidates"] == []
    assert result["ranking"][0]["classification"] == "inconclusive"


def test_one_matched_random_repeat_is_a_global_candidate_gate() -> None:
    baseline = {metric: _aggregate(1 if metric != "perplexity" else 10) for metric in METRICS}
    conditions = {
        component: {metric: _aggregate(0.99 if metric != "perplexity" else 10.1) for metric in METRICS}
        for component in ("sampled", "not-sampled")
    }
    matrix = compute_damage_matrix(baseline, conditions, METRICS)
    bad_control = {"sampled": {metric: 0.5 for metric in METRICS}}
    result = select_discovery_candidates(
        matrix, METRICS, RULE,
        drift_by_metric={metric: 0.0 for metric in METRICS},
        matched_random_damage=bad_control,
    )
    assert result["candidates"] == []
    assert all(
        any("matched_control_not_reproduced" in reason for reason in row["reasons"])
        for row in result["ranking"]
    )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"elapsed_seconds": 11, "rss_bytes": 1, "observation_errors": 0, "task_retries": 0}, "wall_budget_exceeded"),
        ({"elapsed_seconds": 1, "rss_bytes": 101, "observation_errors": 0, "task_retries": 0}, "memory_budget_exceeded"),
        ({"elapsed_seconds": 1, "rss_bytes": 1, "observation_errors": 1, "task_retries": 0}, "observation_error_budget_exceeded"),
        ({"elapsed_seconds": 1, "rss_bytes": 1, "observation_errors": 0, "task_retries": 3}, "retry_budget_exceeded"),
    ],
)
def test_budget_enforcement(kwargs: dict, reason: str) -> None:
    budgets = {"max_wall_seconds": 10, "max_memory_observation_bytes": 100, "max_observation_errors": 0, "max_task_retries": 2}
    with pytest.raises(InvalidEvidenceError, match=reason):
        enforce_budgets(budgets, **kwargs)


def test_stop_rules_reject_triggered_and_unfrozen_conditions() -> None:
    enforce_stop_rules(("control_drift", "hook_cleanup_failure"), {"control_drift": False})
    with pytest.raises(InvalidEvidenceError, match="control_drift"):
        enforce_stop_rules(("control_drift",), {"control_drift": True})
    with pytest.raises(InvalidEvidenceError, match="unfrozen"):
        enforce_stop_rules(("control_drift",), {"new_rule": True})


def test_runner_enforces_cumulative_wall_and_memory_before_completion(tmp_path) -> None:
    class Clock:
        value = 0.0
        def __call__(self) -> float:
            return self.value

    clock = Clock()
    store = FrozenRunStore(tmp_path, {"seed": 1}, {"model": "fixed"})

    def first():
        clock.value += 6
        return {"complete": True, "failures": []}

    ExperimentRunner(
        store, max_wall_seconds=10, max_memory_observation_bytes=100, clock=clock,
        memory_bytes=lambda: 50,
    ).run((Task("baseline", "baseline", first),))
    assert store.active_wall_seconds() == 6

    def second():
        clock.value += 5
        return {"complete": True, "failures": []}

    with pytest.raises(InvalidEvidenceError, match="wall-time"):
        ExperimentRunner(
            store, max_wall_seconds=10, max_memory_observation_bytes=100, clock=clock,
            memory_bytes=lambda: 50,
        ).run((
            Task("baseline", "baseline", first),
            Task("scan.next", "scan", second),
        ))
    assert store.completed("scan.next") is None

    with pytest.raises(InvalidEvidenceError, match="memory"):
        ExperimentRunner(
            FrozenRunStore(tmp_path / "memory", {}, {}),
            max_memory_observation_bytes=100, memory_bytes=lambda: 101,
        ).run((Task("baseline", "baseline", lambda: {"complete": True}),))

    observation_store = FrozenRunStore(tmp_path / "observation-memory", {}, {})
    with pytest.raises(InvalidEvidenceError, match="incomplete"):
        ExperimentRunner(
            observation_store,
            max_memory_observation_bytes=100,
            memory_bytes=lambda: 50,
        ).run((Task("baseline", "baseline", lambda: {
            "complete": True,
            "observations": [{"memory_observation_bytes": 101}],
        }),))
    assert observation_store.completed("baseline") is None


def test_runner_retry_budget_means_initial_attempt_plus_frozen_retries(tmp_path) -> None:
    store = FrozenRunStore(tmp_path, {}, {})
    runner = ExperimentRunner(store, max_task_retries=2)
    task = Task("baseline", "baseline", lambda: {"complete": False, "failures": []})
    for expected_attempt in (1, 2, 3):
        with pytest.raises(InvalidEvidenceError, match="incomplete"):
            runner.run((task,))
        assert store.failure_attempts("baseline") == expected_attempt
    with pytest.raises(InvalidEvidenceError, match="retry budget"):
        runner.run((task,))


def test_runner_counts_raised_failures_and_zero_error_budget_is_terminal(tmp_path) -> None:
    raised_store = FrozenRunStore(tmp_path / "raised", {}, {})
    raised = ExperimentRunner(raised_store, max_task_retries=2)
    task = Task("baseline", "baseline", lambda: (_ for _ in ()).throw(RuntimeError("secret")))
    for attempt in (1, 2, 3):
        with pytest.raises(RuntimeError):
            raised.run((task,))
        assert raised_store.failure_attempts("baseline") == attempt
    assert [
        json.loads(path.read_text())["_attempt"]
        for path in sorted((tmp_path / "raised/failures").glob("baseline.attempt-*.json"))
    ] == [1, 2, 3]
    with pytest.raises(InvalidEvidenceError, match="retry budget"):
        raised.run((task,))

    error_store = FrozenRunStore(tmp_path / "errors", {}, {})
    error_runner = ExperimentRunner(error_store, max_observation_errors=0, max_task_retries=2)
    incomplete = Task("baseline", "baseline", lambda: {"complete": False, "failures": ["ParseError"]})
    with pytest.raises(InvalidEvidenceError, match="observation-error"):
        error_runner.run((incomplete,))
    with pytest.raises(InvalidEvidenceError, match="observation-error"):
        error_runner.run((Task("baseline", "baseline", lambda: {"complete": True, "failures": []}),))
    assert error_store.completed("baseline") is None


def test_resource_budget_refusal_is_exported_before_task_execution(tmp_path) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    telemetry = OperationTelemetry.create(
        "execution", provider.get_tracer("test.budget"), meters.get_meter("test.budget"),
    )
    executed = False

    def execute():
        nonlocal executed
        executed = True
        return {"complete": True}

    with pytest.raises(InvalidEvidenceError, match="memory"):
        ExperimentRunner(
            FrozenRunStore(tmp_path, {}, {}), telemetry,
            max_memory_observation_bytes=100, memory_bytes=lambda: 101,
        ).run((Task("baseline", "baseline", execute),))
    assert executed is False
    span, = [span for span in exporter.get_finished_spans() if span.name == "capability_anatomy.execution.resource_budget"]
    assert span.name == "capability_anatomy.execution.resource_budget"
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["capability_anatomy.reason"] == "memory_budget_exceeded"


def _bypass_decision(reason: str, outcome: str) -> dict:
    return {"name": "operation.decision", "attributes": {
        "capability_anatomy.reason": reason,
        "capability_anatomy.component": "intervention",
        "capability_anatomy.operation": "block_bypass",
        "capability_anatomy.outcome": outcome,
    }}


def _prior_segments(variant: str | None) -> dict:
    """A resumed run's earlier segment, as `block_bypass` actually records one.

    `interrupted_resume` mirrors the real Qwen3-1.7B bundle: the segment died
    mid-task, so its second activation terminates as `experiment_body_failed`
    instead of `hooks_removed`. `cleanup_failed` keeps the accounting balanced
    but reports a release that raised, which means hooks may still be installed.
    """
    if variant is None:
        return {}
    events = [_bypass_decision("scoped_bypass_active", "accepted"),
              _bypass_decision("hooks_removed", "completed")]
    if variant == "interrupted_resume":
        events += [_bypass_decision("scoped_bypass_active", "accepted"),
                   _bypass_decision("experiment_body_failed", "failed")]
    elif variant == "cleanup_failed":
        events += [_bypass_decision("cleanup_failed", "failed")]
    else:
        raise AssertionError(variant)
    return {"prior_segments": [{
        "trace_id": "b" * 32,
        "spans": [{"name": "capability_anatomy.phase5.scan", "span_id": "a" * 16,
                   "parent_span_id": None, "status": "ERROR", "events": events}],
        "metrics": [{"name": "capability_anatomy.operation.decisions",
                     "points": [{"attributes": event["attributes"], "value": 1} for event in events]}],
    }]}


@pytest.mark.parametrize(
    ("prior_variant", "expected_exit"),
    [
        (None, 0),
        ("interrupted_resume", 0),
        ("cleanup_failed", 2),
    ],
)
def test_final_artifacts_are_atomic_sanitized_and_offline_reconstructable(tmp_path, capsys, monkeypatch, prior_variant, expected_exit) -> None:
    required = ("protocol.json", "record-plan.json", "configuration.json", "compatibility.json", "topology.json", "component-scan-order.json", "discovery-task-plan.json", "validation-task-plan.json", "observations.jsonl", "metrics.json", "failures.jsonl", "controls.json", "provenance.json", "candidates.json", "validation.json", "trace.json", "report.json", "report.md", "evidence-manifest.json")
    values = {
        "baseline.beginning": 0.9, "baseline.middle": 0.9,
        "baseline.end": 0.9, "control.no-op": 0.9,
        "scan.block": 0.88, "control.random.block": 0.88,
        "validation.baseline": 0.9, "validation.block": 0.88,
    }
    rows = []
    for condition, value in values.items():
        for repetition in (0, 1):
            is_validation = condition.startswith("validation.")
            row = _observation("validation-example" if is_validation else "discovery-example", repetition, value, value * 2)
            row["condition"] = condition
            row["partition"] = "validation" if is_validation else "discovery"
            row["component_id"] = "block" if condition not in {"baseline.beginning", "baseline.middle", "baseline.end", "validation.baseline"} else "baseline"
            row["group_id"] = "validation-group" if is_validation else "discovery-group"
            row["expected_metric_ids"] = ["abstention"]
            row["scores"] = {"abstention": row["scores"].pop("selection")}
            rows.append(row)
    per_condition = {
        condition: aggregate_observations([row for row in rows if row["condition"] == condition], ("abstention",), semantics_version="1")
        for condition in values
    }
    metric_protocol = {"abstention": {"role": "target", "direction": "higher_is_better"}}
    rule = {"target_absolute_damage_max": 0.05, "collateral_absolute_damage_max": 0.1, "perplexity_relative_damage_max": 0.1, "minimum_drift_margin": 0.025}
    baseline, drift = aggregate_baseline_controls(
        {position: per_condition[f"baseline.{position}"] for position in ("beginning", "middle", "end")},
        per_condition["control.no-op"],
    )
    damage = compute_damage_matrix(baseline, {"block": per_condition["scan.block"]}, metric_protocol)
    random_matrix = compute_damage_matrix(baseline, {"block": per_condition["control.random.block"]}, metric_protocol)
    random_damage = {"block": {"abstention": random_matrix["block"]["abstention"]["absolute_damage"]}}
    candidates = select_discovery_candidates(damage, metric_protocol, rule, drift_by_metric=drift, matched_random_damage=random_damage)
    validation_damage = compute_damage_matrix(per_condition["validation.baseline"], {"block": per_condition["validation.block"]}, metric_protocol)
    validation = compute_validation_results(candidates["candidates"], validation_damage, metric_protocol, rule, semantics_version="1")
    frozen_controls = {"no_op_hook": True, "repeated_baseline_positions": ["beginning", "middle", "end"], "matched_random_components": 1, "placement": {"middle_after_fraction": 0.5}}
    discovery_rows = [{"source_id": "discovery-example", "group_key": "discovery-group", "kind": "abstention"}]
    validation_rows = [{"source_id": "validation-example", "group_key": "validation-group", "kind": "abstention"}]
    record_plan = {"schema_version": "capability-anatomy/phase5-record-plan/v1", "partitions": {"discovery": discovery_rows, "validation": validation_rows}, "rows_sha256": canonical_sha256([*discovery_rows, *validation_rows])}
    record_plan_sha256 = hashlib.sha256(json.dumps(record_plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plugin = lambda role, name, capabilities=(): {"role": role, "name": name, "version": "1", "api_version": "capability-anatomy/plugin-api/v1", "capabilities": list(capabilities)}
    protocol = {"schema_version": "capability-anatomy/phase5-protocol/v1", "experiment": {"id": "phase5", "seed": 4}, "model": {"plugin": "fixture.model", "version": "1", "source": "fixture/source", "revision": "fixed", "architecture_profile": "fixture-v1", "architecture_profile_sha256": "a" * 64}, "plugins": [plugin("model", "fixture.model"), plugin("intervention", "fixture.intervention"), plugin("evaluation", "reference.phase5-qwen-retention"), plugin("dataset", "fixture.dataset"), plugin("runtime", "fixture.runtime")], "runtime": {"deterministic": True, "warmup_runs": 0, "repetitions": 2, "randomized_record_order": True, "device": "cpu", "context_length": 32}, "dataset": {"record_plan": {"sha256": record_plan_sha256}, "manifest": {"sha256": "b" * 64}, "prompt_templates": {"sha256": "c" * 64}, "scorers": {"sha256": "d" * 64}}, "metrics": metric_protocol, "candidate_rule": rule, "controls": frozen_controls, "claim_boundary": "temporary sensitivity only", "required_artifacts": list(required)}
    manifest_plugins = [{**item, "distribution": "fixture", "distribution_version": "1"} for item in protocol["plugins"]]
    trace_id = "1" * 32
    measurements = {condition: summarize_measurements([row for row in rows if row["condition"] == condition]) for condition in values}
    report = {"status": "complete", "objective": "Measure temporary single-block sensitivity for tool calling and collateral capabilities under the frozen Phase 5 protocol.", "trace_id": trace_id, "claim_boundary": protocol["claim_boundary"], "provenance": None, "baseline": baseline, "per_component_damage": damage, "controls": {"drift": drift, "matched_random_damage": random_damage}, "candidate_ranking": candidates["ranking"], "measurements": measurements, "errors": [], "candidates": candidates["candidates"], "validation": validation, "limitations": ["fixture"], "reproduction_commands": ["fixture"]}
    topology = {"architecture": "fixture", "components": [{"id": "block", "kind": "transformer_block", "parent_id": None, "order": 0, "metadata": {"architecture": "fixture", "module_path": "model.layers.0", "original_order": 0, "parameter_count": 10, "architecture_profile": "fixture-v1", "architecture_profile_sha256": "a" * 64}}]}
    provenance = {"plugin": "fixture.model", "plugin_version": "1", "api_version": "capability-anatomy/plugin-api/v1", "architecture": "fixture", "model_source": "fixture/source", "model_revision": "fixed", "model_class": "Fixture", "implementation_metadata": {"architecture_profile": "fixture-v1", "architecture_profile_sha256": "a" * 64}, "libraries": {"fixture": "1"}}
    report["provenance"] = provenance
    discovery_plan = serialize_campaign_plan(build_campaign_plan(("block",), frozen_controls, seed=int.from_bytes(hashlib.sha256(b"4:controls").digest()[:8], "big")))
    decisions = [
        ("full_scan_authorized", "gate5a_authorization", "authorize_phase5_operation", "accepted", "capability_anatomy.gate5a.authorization"),
        ("plugin_contract_accepted", "plugin_discovery", "resolve_plugin", "accepted", "capability_anatomy.plugin.resolve"),
        ("resource_budget_available", "execution", "enforce_resource_budget", "accepted", "capability_anatomy.execution.resource_budget"),
        ("task_committed", "execution", "execute_task", "accepted", "capability_anatomy.execution.task"),
        ("runtime_plugin_execution_complete", "runtime", "execute", "accepted", "capability_anatomy.runtime.execute"),
        ("scoped_bypass_active", "intervention", "block_bypass", "accepted", "capability_anatomy.intervention.block_bypass"),
        ("hooks_removed", "intervention", "block_bypass", "completed", "capability_anatomy.intervention.block_bypass"),
        ("identity_no_op_active", "intervention", "identity_no_op", "accepted", "capability_anatomy.intervention.identity_no_op"),
        ("identity_no_op_hooks_removed", "intervention", "identity_no_op", "completed", "capability_anatomy.intervention.identity_no_op"),
        ("result_independent_campaign_plan_persisted", "execution", "freeze_discovery_plan", "accepted", "capability_anatomy.phase5.freeze_discovery_plan"),
        ("repeated_baseline_no_op_and_matched_controls_complete", "execution", "evaluate_controls", "accepted", "capability_anatomy.phase5.evaluate_controls"),
        ("ranking_derived_from_discovery_only", "execution", "freeze_discovery_ranking", "accepted", "capability_anatomy.phase5.freeze_discovery_ranking"),
        ("discovery_ranking_frozen_before_validation", "execution", "open_validation_once", "accepted", "capability_anatomy.phase5.open_validation_once"),
        ("required_artifacts_atomically_written_and_digest_bound", "execution", "finalize_evidence", "accepted", "capability_anatomy.phase5.finalize_evidence"),
        ("scan_evidence_validated", "phase5_scan", "phase5_full_scan", "completed", "capability_anatomy.phase5.scan"),
    ]
    decisions.extend([("task_committed", "execution", "execute_task", "accepted", "capability_anatomy.execution.task")] * 7)
    decisions.extend([("plugin_contract_accepted", "plugin_discovery", "resolve_plugin", "accepted", "capability_anatomy.plugin.resolve")] * 4)
    decisions.extend([("resource_budget_available", "execution", "enforce_resource_budget", "accepted", "capability_anatomy.execution.resource_budget")] * 15)
    decisions.extend([("runtime_plugin_execution_complete", "runtime", "execute", "accepted", "capability_anatomy.runtime.execute")] * 15)
    decisions.extend([("scoped_bypass_active", "intervention", "block_bypass", "accepted", "capability_anatomy.intervention.block_bypass")] * 2)
    decisions.extend([("hooks_removed", "intervention", "block_bypass", "completed", "capability_anatomy.intervention.block_bypass")] * 2)
    root_id = "0" * 15 + "1"
    trace_spans = [{"name": "capability_anatomy.phase5.scan", "span_id": root_id, "parent_span_id": None, "status": "UNSET", "events": []}]
    for index, (reason, component, operation, outcome, span_name) in enumerate(decisions):
        attributes = {"capability_anatomy.reason": reason, "capability_anatomy.component": component, "capability_anatomy.operation": operation, "capability_anatomy.outcome": outcome}
        if span_name == "capability_anatomy.phase5.scan":
            trace_spans[0]["events"].append({"name": "operation.decision", "attributes": attributes})
        else:
            trace_spans.append({"name": span_name, "span_id": f"{index + 2:016x}", "parent_span_id": root_id, "status": "UNSET", "events": [{"name": "operation.decision", "attributes": attributes}]})
    trace_points = [{"attributes": {"capability_anatomy.reason": reason, "capability_anatomy.component": component, "capability_anatomy.operation": operation, "capability_anatomy.outcome": outcome}, "value": 1} for reason, component, operation, outcome, _span in decisions]
    configuration = {"schema_version": "capability-anatomy/experiment-config/v1", "experiment_id": "phase5", "seed": 4, "model": {"plugin": "fixture.model", "source": "fixture/source", "revision": "fixed", "parameters": {"architecture_profile": "fixture-v1", "dtype": None}}, "runtime": {"executor": "fixture.runtime", "deterministic": True, "warmup_runs": 0, "randomized_execution_order": True, "repetitions": 2, "parameters": {"device": "cpu", "context_length": 32}}, "capability": {"evaluation_plugin": "reference.phase5-qwen-retention", "target_metrics": ["abstention"], "collateral_metrics": []}, "dataset": {"provider": "fixture.dataset"}, "intervention": {"plugin": "fixture.intervention", "parameters": {"component_ids": "all"}}, "output": {"retain_prompts": False, "retain_raw_outputs": False, "prompt_storage": "hash_only"}}
    artifacts = {
        "protocol.json": protocol,
        "record-plan.json": record_plan,
        "configuration.json": configuration,
        "compatibility.json": {"resolved_topology": topology, "model": provenance, "evaluation": {"plugin": "reference.phase5-qwen-retention", "version": "1"}, "intervention": {"plugin": "fixture.intervention", "version": "1"}, "discovered_plugins": manifest_plugins},
        "topology.json": topology,
        "component-scan-order.json": ["block"],
        "discovery-task-plan.json": discovery_plan,
        "validation-task-plan.json": [{"id": "validation.baseline", "condition": "baseline", "component_id": None}, {"id": "validation.block", "condition": "bypass", "component_id": "block"}],
        "observations.jsonl": rows,
        "metrics.json": {"per_condition": per_condition, "baseline": baseline, "damage_matrix": damage, "validation_damage": validation_damage, "measurements": measurements},
        "failures.jsonl": [],
        "controls.json": {"plan": discovery_plan, "drift": drift, "matched_random_damage": random_damage},
        "provenance.json": provenance,
        "candidates.json": candidates,
        "validation.json": validation,
        "trace.json": {"schema_version": "capability-anatomy/phase5-trace/v1", "trace_id": trace_id, "spans": trace_spans, "metrics": [{"name": "capability_anatomy.operation.decisions", "points": trace_points}], **_prior_segments(prior_variant)},
        "report.json": report,
        "report.md": render_report_markdown(report),
    }
    protocol_sha256 = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    compatibility = artifacts["compatibility.json"]
    authorization = {"schema_version": "capability-anatomy/gate5a-authorization/v1", "status": "approved", "conformance_authorized": True, "full_scan_authorized": True, "protocol_sha256": protocol_sha256, "approved_commit": "1" * 40, "approved_source_sha256": "2" * 64, "review_url": "https://reviews.example.org/approvals/fixture-1"}
    identity = {"run_id": "phase5", "config_sha256": hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "compatibility_sha256": hashlib.sha256(json.dumps(compatibility, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "metric_ids": ["abstention"], "protocol_sha256": protocol_sha256, "model_revision": "fixed", "dataset_sha256": "b" * 64, "prompt_sha256": "c" * 64, "scorer_sha256": "d" * 64, "plugins": manifest_plugins, "runtime": protocol["runtime"], "execution_state": "complete", "authorization": authorization}
    monkeypatch.setattr(
        "capability_anatomy.cli.validate_phase5_protocol",
        lambda _path, **_kwargs: protocol,
    )
    record_seed = int.from_bytes(hashlib.sha256(b"4:records").digest()[:8], "big")
    for condition in values:
        condition_rows = [row for row in rows if row["condition"] == condition]
        source_id = "validation-example" if condition.startswith("validation.") else "discovery-example"
        FrozenRunStore._atomic_write(
            tmp_path / "tasks" / f"{condition}.json",
            json.dumps({
                "execution_order": [source_id],
                "measurement_controls": {"seed": record_seed, "warmup_runs": 0, "repetitions": 2, "randomized_order": True},
                "observations": condition_rows,
            }, sort_keys=True, separators=(",", ":")).encode(),
        )
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)
    bound_paths = [item["path"] for item in manifest["artifacts"]] + ["evidence-manifest.json"]
    reconstruct_manifest(tmp_path, manifest, bound_paths)
    assert json.loads((tmp_path / "evidence-manifest.json").read_text()) == manifest
    assert len((tmp_path / "observations.jsonl").read_text().splitlines()) == 16
    assert [item["path"] for item in manifest["artifacts"]] == sorted(
        set(required) - {"evidence-manifest.json"}
        | {f"tasks/{condition}.json" for condition in values}
    )
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == expected_exit
    if expected_exit:
        assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
        return
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert main(("report", "--evidence", str(tmp_path), "--format", "markdown")) == 0
    rendered_markdown = capsys.readouterr().out
    assert rendered_markdown.startswith("# Phase 5 component scan report")
    for heading in (
        "## Objective", "## Provenance", "## Baseline and uncertainty",
        "## Per-component target and collateral damage", "## Controls and drift",
        "## Candidate ranking", "## Runtime, memory, and token observations",
        "## Errors", "## Validation", "## Limitations", "## Reproduction commands",
    ):
        assert heading in rendered_markdown

    task_path = tmp_path / "tasks/baseline.beginning.json"
    task_payload = json.loads(task_path.read_text())
    task_payload["measurement_controls"]["seed"] = 4
    FrozenRunStore._atomic_write(
        task_path, json.dumps(task_payload, sort_keys=True, separators=(",", ":")).encode(),
    )
    write_final_artifacts(tmp_path, {}, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    task_payload["measurement_controls"]["seed"] = record_seed
    FrozenRunStore._atomic_write(
        task_path, json.dumps(task_payload, sort_keys=True, separators=(",", ":")).encode(),
    )
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    monkeypatch.setattr(
        "capability_anatomy.cli.validate_phase5_protocol",
        lambda _path, **_kwargs: (_ for _ in ()).throw(InvalidEvidenceError("intrinsic-validation-sentinel")),
    )
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    monkeypatch.setattr("capability_anatomy.cli.validate_phase5_protocol", lambda _path, **_kwargs: protocol)

    wrong_identity = {**identity, "protocol_sha256": "0" * 64}
    write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=wrong_identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    incomplete_report = copy.deepcopy(artifacts)
    incomplete_report["report.json"].pop("baseline")
    incomplete_report["report.md"] = render_report_markdown(incomplete_report["report.json"])
    write_final_artifacts(tmp_path, incomplete_report, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    fabricated_identity = copy.deepcopy(identity)
    fabricated_identity["plugins"][0]["version"] = "invented"
    write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=fabricated_identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    # Rebinding a manifest cannot silently reinterpret v1 observations as v2,
    # and unknown scorer semantics must refuse before scientific reconstruction.
    for version, message in (("2", "aggregate reconstruction mismatch"), ("999", "semantics version is unsupported")):
        incompatible_evaluator = copy.deepcopy(identity)
        for item in incompatible_evaluator["plugins"]:
            if item["role"] == "evaluation":
                item["version"] = version
        write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=incompatible_evaluator)
        assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
        assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
        manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    incomplete_authorization = copy.deepcopy(identity)
    incomplete_authorization["authorization"].pop("approved_commit")
    write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=incomplete_authorization)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    invented_failure = copy.deepcopy(artifacts)
    invented_failure["failures.jsonl"] = [{"task_id": "invented", "attempt": 1}]
    invented_failure["report.json"]["errors"] = invented_failure["failures.jsonl"]
    invented_failure["report.md"] = render_report_markdown(invented_failure["report.json"])
    write_final_artifacts(tmp_path, invented_failure, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    malformed_trace_id = copy.deepcopy(artifacts)
    malformed_trace_id["trace.json"]["trace_id"] = "not-a-trace-id"
    malformed_trace_id["report.json"]["trace_id"] = "not-a-trace-id"
    malformed_trace_id["report.md"] = render_report_markdown(malformed_trace_id["report.json"])
    write_final_artifacts(tmp_path, malformed_trace_id, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    wrong_counter = copy.deepcopy(artifacts)
    task_point = next(point for point in wrong_counter["trace.json"]["metrics"][0]["points"] if point["attributes"]["capability_anatomy.reason"] == "task_committed")
    task_point["value"] = 2
    write_final_artifacts(tmp_path, wrong_counter, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    skeletal_trace = copy.deepcopy(artifacts)
    skeletal_trace["trace.json"] = {"schema_version": "capability-anatomy/phase5-trace/v1", "trace_id": trace_id, "spans": [], "metrics": []}
    write_final_artifacts(tmp_path, skeletal_trace, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    forged_trace = copy.deepcopy(artifacts)
    for span in forged_trace["trace.json"]["spans"]:
        for event in span["events"]:
            event["attributes"]["capability_anatomy.outcome"] = "failed"
    forged_trace["trace.json"]["metrics"][0]["name"] = "unrelated.metric"
    write_final_artifacts(tmp_path, forged_trace, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    wrong_metric_name = copy.deepcopy(artifacts)
    wrong_metric_name["trace.json"]["metrics"][0]["name"] = "unrelated.metric"
    write_final_artifacts(tmp_path, wrong_metric_name, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    wrong_outcome = copy.deepcopy(artifacts)
    for span in wrong_outcome["trace.json"]["spans"]:
        for event in span["events"]:
            event["attributes"]["capability_anatomy.outcome"] = "failed"
    for point in wrong_outcome["trace.json"]["metrics"][0]["points"]:
        point["attributes"]["capability_anatomy.outcome"] = "failed"
    write_final_artifacts(tmp_path, wrong_outcome, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    without_plan = {key: value for key, value in artifacts.items() if key != "record-plan.json"}
    write_final_artifacts(tmp_path, without_plan, required_paths=required, identity=identity)
    (tmp_path / "record-plan.json").write_bytes((tmp_path / "record-plan.json").read_bytes() + b"\n")
    write_final_artifacts(tmp_path, {}, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    omitted = copy.deepcopy(artifacts)
    omitted["observations.jsonl"] = [row for row in rows if row["condition"] != "scan.block"]
    omitted["metrics.json"]["per_condition"].pop("scan.block")
    omitted["metrics.json"]["measurements"].pop("scan.block")
    omitted["metrics.json"]["damage_matrix"] = {}
    omitted["candidates.json"] = {"schema_version": "capability-anatomy/phase5-discovery-ranking/v1", "candidates": [], "ranking": []}
    omitted["validation-task-plan.json"] = [{"id": "validation.baseline", "condition": "baseline", "component_id": None}]
    omitted["validation.json"] = {"schema_version": "capability-anatomy/phase5-validation/v1", "validation_once": True, "results": []}
    omitted["metrics.json"]["validation_damage"] = {}
    omitted["report.json"] = {**report, "candidates": [], "validation": omitted["validation.json"]}
    omitted["report.md"] = render_report_markdown(omitted["report.json"])
    write_final_artifacts(tmp_path, omitted, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    fabricated = copy.deepcopy(artifacts)
    fabricated["candidates.json"] = {**candidates, "candidates": ["fabricated"]}
    fabricated["report.json"] = {**report, "candidates": ["fabricated"]}
    write_final_artifacts(tmp_path, fabricated, required_paths=required, identity=identity)
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    manifest = write_final_artifacts(tmp_path, artifacts, required_paths=required, identity=identity)

    (tmp_path / "unbound.json").write_text("{}")
    bound_paths = [item["path"] for item in manifest["artifacts"]] + ["evidence-manifest.json"]
    with pytest.raises(InvalidEvidenceError, match="unbound"):
        reconstruct_manifest(tmp_path, manifest, bound_paths)
    (tmp_path / "unbound.json").unlink()

    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(InvalidEvidenceError, match="digest mismatch"):
        reconstruct_manifest(tmp_path, manifest, bound_paths)


def test_manifest_binds_store_task_state_and_failure_history(tmp_path) -> None:
    store = FrozenRunStore(tmp_path, {"seed": 1}, {"model": "fixed"})
    store.initialize()
    store.commit("baseline", {"complete": True})
    store.commit_failure("retry", {"complete": False}, attempt=1)
    store.record_state("complete", "all_tasks_committed")
    manifest = write_final_artifacts(
        tmp_path,
        {"report.json": {"status": "complete"}, "failures.jsonl": [{"task": "retry"}]},
        required_paths=("report.json", "failures.jsonl", "evidence-manifest.json"),
        identity={},
    )
    paths = {item["path"] for item in manifest["artifacts"]}
    assert {
        "experiment-config.json", "run-state.json", "execution-state.json",
        "tasks/baseline.json", "tasks/baseline.complete.json", "failures/retry.json",
        "failures/retry.attempt-0001.json",
    } <= paths
    required = [*paths, "evidence-manifest.json"]
    reconstruct_manifest(tmp_path, manifest, required)
    (tmp_path / "tasks/baseline.json").write_text("{}")
    with pytest.raises(InvalidEvidenceError, match="digest mismatch"):
        reconstruct_manifest(tmp_path, manifest, required)


def test_final_artifacts_reject_missing_unsafe_symlink_and_sensitive_content(tmp_path) -> None:
    with pytest.raises(InvalidEvidenceError, match="missing|does not exist"):
        write_final_artifacts(tmp_path, {}, required_paths=("report.json", "evidence-manifest.json"), identity={})
    with pytest.raises(InvalidEvidenceError, match="unsafe"):
        write_final_artifacts(tmp_path, {"../escape.json": {}}, required_paths=("evidence-manifest.json",), identity={})
    with pytest.raises(InvalidEvidenceError, match="forbidden"):
        write_final_artifacts(tmp_path, {"report.json": {"nested": {"raw_output": "secret"}}}, required_paths=("report.json", "evidence-manifest.json"), identity={})

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(InvalidEvidenceError, match="symlink"):
        write_final_artifacts(tmp_path, {"linked/report.json": {}}, required_paths=("evidence-manifest.json",), identity={})


def test_report_refuses_self_described_partial_manifest(tmp_path, capsys) -> None:
    write_final_artifacts(
        tmp_path,
        {"report.json": {"status": "complete"}},
        required_paths=("report.json", "evidence-manifest.json"),
        identity={"metric_ids": ["selection"]},
    )
    assert main(("report", "--evidence", str(tmp_path), "--format", "json")) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"


def _bypass_events(activated: int, removed: int, body_failed: int = 0, cleanup_failed: int = 0) -> dict:
    events = (
        [_bypass_decision("scoped_bypass_active", "accepted")] * activated
        + [_bypass_decision("hooks_removed", "completed")] * removed
        + [_bypass_decision("experiment_body_failed", "failed")] * body_failed
        + [_bypass_decision("cleanup_failed", "failed")] * cleanup_failed
    )
    return {"spans": [{"events": events}]}


@pytest.mark.parametrize(
    ("segments", "balanced", "reason"),
    [
        ([(28, 28, 0, 0)], True, "a clean run terminates every activation with hooks_removed"),
        ([(28, 27, 0, 0)], False, "an activation with no terminal decision is a possible leak"),
        ([(2, 1, 1, 0), (28, 28, 0, 0)], True, "an interrupted segment terminates as experiment_body_failed"),
        ([(2, 1, 0, 0), (28, 28, 0, 0)], False, "a resume does not excuse an unterminated activation"),
        ([(1, 1, 0, 1), (28, 28, 0, 0)], False, "cleanup_failed means hooks may still be installed"),
        ([(28, 28, 0, 1)], False, "cleanup_failed is prohibited even when the counts balance"),
        ([(1, 2, 0, 0)], False, "more terminals than activations is fabricated cleanup"),
        ([(0, 0, 0, 0), (2, 1, 1, 0), (28, 28, 0, 0)], True, "a refusal segment carries no bypass at all"),
    ],
)
def test_bypass_hooks_accounting_is_exact_per_segment_and_survives_resume(segments, balanced, reason) -> None:
    """No tolerance for dangling activations: block_bypass always records a terminal.

    `block_bypass` releases in a `finally` and then records exactly one of
    hooks_removed / experiment_body_failed / cleanup_failed, so a resumed run
    accounts exactly. Accepting a bare unterminated activation would admit a
    real cleanup_failed leak.
    """
    assert _bypass_hooks_balanced([_bypass_events(*t) for t in segments]) is balanced, reason


def test_gate5a_review_policy_is_shared_by_execution_and_reconstruction() -> None:
    from capability_anatomy.execution import orchestrator
    from capability_anatomy.review_policy import validate_review_reference

    assert orchestrator.validate_review_reference is validate_review_reference
    from capability_anatomy.execution import phase5_campaign
    assert phase5_campaign.validate_review_reference is validate_review_reference
    for function in (orchestrator._check_gate5a_authorization, _verify_operational_evidence):
        source = inspect.getsource(function)
        assert "validate_review_reference(" in source
        assert "github.com" not in source
        assert "pull/" not in source


def test_scan_order_is_verified_against_its_seed_not_just_its_membership() -> None:
    """`freeze_sequence` lets a stored order win over a resumed run's proposal.

    That is what makes a resume stable, but it also means an edited
    component-scan-order.json survives into a manifest-bound bundle unless the
    order is re-derived. Membership alone cannot catch a permutation.
    """
    topology = [{"id": f"transformer.block.{index:03d}"} for index in range(28)]
    protocol = {"experiment": {"seed": 20260904}, "runtime": {"randomized_execution_order": True}}
    derived = _derived_component_order(topology, protocol)

    assert sorted(derived) == sorted(component["id"] for component in topology)
    assert derived != [component["id"] for component in topology], "a seeded shuffle must reorder"
    assert derived == _derived_component_order(topology, protocol), "derivation must be deterministic"

    # The real run's frozen order is exactly this derivation.
    assert derived[:6] == [
        "transformer.block.004", "transformer.block.000", "transformer.block.001",
        "transformer.block.019", "transformer.block.023", "transformer.block.010",
    ]

    # A permutation keeps identical membership and must still be rejected.
    permuted = [derived[1], derived[0], *derived[2:]]
    assert set(permuted) == set(derived) and len(permuted) == len(derived)
    assert permuted != derived, "membership-only checks cannot see this"

    # A different seed derives a different order.
    assert _derived_component_order(topology, {"experiment": {"seed": 1}, "runtime": {"randomized_execution_order": True}}) != derived

    # The verifier must actually USE the derivation. A unit test of the helper
    # alone leaves deleting the call site silent, which is how the previous
    # round of regression tests failed to pin their own fix.
    assert "_derived_component_order" in inspect.getsource(verify_phase5_aggregates), (
        "verify_phase5_aggregates no longer checks the scan order against its seed"
    )
