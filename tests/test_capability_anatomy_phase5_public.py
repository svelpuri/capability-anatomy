"""Generic Phase 5 contracts requiring no historical model data or private evidence."""
from __future__ import annotations
import ast
import json
import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from capability_anatomy.datasets import Phase5DatasetProvider
from capability_anatomy.cli import main
from capability_anatomy.domain import NormalizedRecord, ObservationStatus
from capability_anatomy.errors import GateAuthorizationError, InvalidConfigurationError, InvalidEvidenceError
from capability_anatomy.evaluations import EvaluationExecutor
from capability_anatomy.execution.orchestrator import _identity_no_op_hook, _execute_phase5_conformance_runtime, _persist_phase5_conformance, _record_phase5_decision, _run_experiment, _validate_phase5_conformance
from capability_anatomy.execution.orchestrator import run_experiment, _governed_source_identity
from capability_anatomy.evaluations.plugins import Phase5EvaluationSuite
from capability_anatomy.models.base import GeneratedText, GenerationRequest, PerplexityRequest
from capability_anatomy.phase5_protocol import sha256_file
from capability_anatomy.telemetry import OperationTelemetry


def test_identity_no_op_control_emits_active_and_cleanup_evidence() -> None:
    class Hookable:
        def __init__(self):
            self._forward_hooks = {}

        def register_forward_hook(self, hook):
            self._forward_hooks[1] = hook
            return SimpleNamespace(remove=lambda: self._forward_hooks.pop(1))

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    telemetry = OperationTelemetry.create(
        "intervention", provider.get_tracer("test.no-op"), meters.get_meter("test.no-op"),
    )
    implementation = Hookable()
    with _identity_no_op_hook(SimpleNamespace(implementation=implementation), telemetry):
        assert len(implementation._forward_hooks) == 1
    assert not implementation._forward_hooks
    span, = exporter.get_finished_spans()
    assert span.name == "capability_anatomy.intervention.identity_no_op"
    assert [event.attributes["capability_anatomy.reason"] for event in span.events] == [
        "identity_no_op_active", "identity_no_op_hooks_removed",
    ]


def test_conformance_invariants_reject_false_positive_evidence() -> None:
    output = GeneratedText("same", 3)
    assert _validate_phase5_conformance(
        before_hooks=0, after_hooks=0, normal_calls=3, restored_calls=3,
        bypass_calls=3, bypass_identity_calls=3, sequence_widths=[12, 1, 1],
        normal=output, restored=GeneratedText("same", 3),
    ) == (1, 2)
    for widths, restored, message in (
        ([12, 8], output, "prefill/decode"),
        ([12, 1], GeneratedText("changed", 3), "restoration"),
    ):
        with pytest.raises(InvalidEvidenceError, match=message):
            _validate_phase5_conformance(
                before_hooks=0, after_hooks=0, normal_calls=2, restored_calls=2,
                bypass_calls=2, bypass_identity_calls=2, sequence_widths=widths,
                normal=output, restored=restored,
            )
    with pytest.raises(InvalidEvidenceError, match="cleanup"):
        _validate_phase5_conformance(
            before_hooks=0, after_hooks=1, normal_calls=2, restored_calls=2,
            bypass_calls=2, bypass_identity_calls=2, sequence_widths=[12, 1],
            normal=output, restored=output,
        )
    with pytest.raises(InvalidEvidenceError, match="prefill/decode"):
        _validate_phase5_conformance(
            before_hooks=0, after_hooks=0, normal_calls=2, restored_calls=2,
            bypass_calls=2, bypass_identity_calls=1, sequence_widths=[12, 1],
            normal=output, restored=output,
        )


def test_conformance_returns_before_scan_plan_construction() -> None:
    source = inspect.getsource(_run_experiment)
    bounded = source.index("if conformance_only:")
    bounded_return = source.index('"kind": "phase5_bounded_bypass_conformance"', bounded)
    scan_store = source.index("store = FrozenRunStore", bounded)
    scan_tasks = source.index('Task("baseline"', bounded)
    assert bounded < bounded_return < scan_store < scan_tasks


def test_conformance_evidence_is_atomically_digest_bound(tmp_path: Path) -> None:
    evidence = {"schema_version": "capability-anatomy/phase5-conformance/v1", "status": "complete"}
    manifest = _persist_phase5_conformance(tmp_path, evidence)
    evidence_path = tmp_path / "conformance/evidence.json"
    assert json.loads(evidence_path.read_text()) == evidence
    assert json.loads((tmp_path / "conformance/manifest.json").read_text()) == manifest
    assert manifest["artifacts"][0] == {
        "path": "evidence.json",
        "bytes": len(evidence_path.read_bytes()),
        "sha256": sha256_file(evidence_path),
    }


def test_public_conformance_persists_live_trace_identity_events_and_metrics(
    tmp_path: Path, monkeypatch,
) -> None:
    def bounded(
        _path, *, config, conformance_only, authorization_telemetry,
            intervention_telemetry, runtime_telemetry, execution_telemetry,
            discovery_telemetry,
    ):
        assert conformance_only is True
        assert execution_telemetry is not None and discovery_telemetry is not None
        with authorization_telemetry.tracer.start_as_current_span(
            "capability_anatomy.gate5a.authorization"
        ) as span:
            authorization_telemetry.record(
                span, operation="authorize_phase5_operation", outcome="accepted",
                reason="conformance_authorized",
            )
        with intervention_telemetry.tracer.start_as_current_span(
            "capability_anatomy.intervention.block_bypass"
        ) as span:
            intervention_telemetry.record(
                span, operation="block_bypass", outcome="accepted",
                reason="scoped_bypass_active",
            )
            intervention_telemetry.record(
                span, operation="block_bypass", outcome="completed",
                reason="hooks_removed",
            )
        for _ in range(3):
            with runtime_telemetry.tracer.start_as_current_span(
                "capability_anatomy.runtime.execute"
            ) as span:
                runtime_telemetry.record(
                    span, operation="execute", outcome="accepted",
                    reason="runtime_plugin_execution_complete",
                )
        return {
            "schema_version": "capability-anatomy/phase5-conformance/v1",
            "status": "complete",
            "_output_directory": str(tmp_path),
        }

    assert main(("example", "--output", str(tmp_path / "example"))) == 0
    config_path = tmp_path / "example/experiment.json"
    value = json.loads(config_path.read_text())
    value["output"]["directory"] = str(tmp_path)
    config_path.write_text(json.dumps(value))
    monkeypatch.setattr("capability_anatomy.execution.orchestrator._run_experiment", bounded)
    evidence = run_experiment(config_path, conformance_only=True)
    trace = json.loads((tmp_path / "conformance/trace.json").read_text())
    manifest = json.loads((tmp_path / "conformance/manifest.json").read_text())
    assert evidence["trace_id"] == trace["trace_id"]
    assert int(trace["trace_id"], 16) != 0
    assert {span["name"] for span in trace["spans"]} == {
        "capability_anatomy.phase5.conformance",
        "capability_anatomy.config.validate",
        "capability_anatomy.authored_input.load",
        "capability_anatomy.authored_input.read",
        "capability_anatomy.storage.probe_volume",
        "capability_anatomy.gate5a.authorization",
        "capability_anatomy.intervention.block_bypass",
        "capability_anatomy.runtime.execute",
    }
    assert sum(span["name"] == "capability_anatomy.runtime.execute" for span in trace["spans"]) == 3
    decisions = {
        (event["attributes"]["capability_anatomy.component"], event["attributes"]["capability_anatomy.reason"])
        for span in trace["spans"] for event in span["events"]
        if event["name"] == "operation.decision"
    }
    assert decisions == {
        ("gate5a_authorization", "conformance_authorized"),
        ("authored_input", "bounded_object_valid"),
        ("authored_input", "regular_file_read"),
        ("storage", "storage_operation_complete"),
        ("intervention", "hooks_removed"),
        ("intervention", "scoped_bypass_active"),
        ("phase5_conformance", "conformance_execution_complete"),
        ("runtime", "runtime_plugin_execution_complete"),
    }
    points = {
        (point["attributes"]["capability_anatomy.component"], point["value"])
        for metric in trace["metrics"] for point in metric["points"]
    }
    assert points >= {
        ("gate5a_authorization", 1), ("intervention", 1),
        ("phase5_conformance", 1), ("runtime", 3),
    }
    assert [item["path"] for item in manifest["artifacts"]] == ["evidence.json", "trace.json"]
    for item in manifest["artifacts"]:
        artifact = tmp_path / "conformance" / item["path"]
        assert item["bytes"] == len(artifact.read_bytes())
        assert item["sha256"] == sha256_file(artifact)


def test_real_conformance_composition_wires_all_live_telemetry() -> None:
    tree = ast.parse(inspect.getsource(_run_experiment))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    authorization = [
        call for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "_require_gate5a_authorization"
    ]
    assert len(authorization) == 1
    telemetry = next(
        keyword.value for keyword in authorization[0].keywords if keyword.arg == "telemetry"
    )
    assert isinstance(telemetry, ast.Name) and telemetry.id == "authorization_telemetry"

    bypasses = [
        call for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "BlockBypass"
    ]
    assert any(
        len(call.args) == 2
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "intervention_telemetry"
        for call in bypasses
    )
    no_op_calls = [
        call for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "_identity_no_op_hook"
    ]
    assert len(no_op_calls) == 1
    assert (
        len(no_op_calls[0].args) == 2
        and isinstance(no_op_calls[0].args[1], ast.Name)
        and no_op_calls[0].args[1].id == "intervention_telemetry"
    )

    runtime_calls = [
        call for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_execute_phase5_conformance_runtime"
    ]
    assert len(runtime_calls) == 4
    assert all(
        isinstance(call.args[-1], ast.Name) and call.args[-1].id == "runtime_telemetry"
        for call in runtime_calls
    )


def test_real_full_scan_consumes_frozen_campaign_and_evidence_boundaries() -> None:
    source = inspect.getsource(_run_experiment)
    assert "_experiment_relative_path(config_path)" in source
    assert "_experiment_relative_path(output)" in source
    assert "qwen3-0.6b-phase5" not in source
    assert 'FrozenRunStore._atomic_write(output / "protocol.json", read_regular_bytes(protocol_path, max_bytes=MAX_ARTIFACT_BYTES))' in source
    assert '"protocol.json": protocol' not in source
    assert '"seed": record_seed' in source
    assert '"seed": config.seed' not in source
    tree = ast.parse(source)
    rebound_telemetry = {
        target.id
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        for target in (
            assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
        )
        if isinstance(target, ast.Name) and target.id.endswith("_telemetry")
    }
    assert rebound_telemetry == set()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {
        call.func.id for call in calls if isinstance(call.func, ast.Name)
    }
    assert {
        "build_campaign_plan", "aggregate_observations",
        "aggregate_baseline_controls", "compute_damage_matrix",
        "select_discovery_candidates", "compute_validation_results",
        "write_final_artifacts",
    } <= names
    frozen_names = {
        call.args[0].value
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "freeze_payload"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }
    assert frozen_names == {
        "discovery-task-plan", "discovery-ranking", "validation-task-plan",
    }
    decision_operations = {
        keyword.value.value
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_record_phase5_decision"
        for keyword in call.keywords
        if keyword.arg == "operation" and isinstance(keyword.value, ast.Constant)
    }
    assert decision_operations == {
        "freeze_discovery_plan", "evaluate_controls", "freeze_discovery_ranking",
        "open_validation_once", "finalize_evidence",
    }
    runner_calls = [
        call for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "ExperimentRunner"
        and any(keyword.arg == "max_task_retries" for keyword in call.keywords)
        and call.keywords
    ]
    assert len(runner_calls) == 1
    assert {keyword.arg for keyword in runner_calls[0].keywords} >= {
        "max_wall_seconds", "max_memory_observation_bytes", "max_task_retries",
        "max_observation_errors", "clock", "memory_bytes",
    }


@pytest.mark.parametrize(
    ("failure", "outcome", "reason", "status"),
    [
        (False, "accepted", "runtime_plugin_execution_complete", "UNSET"),
        (True, "failed", "runtime_plugin_execution_failed", "ERROR"),
    ],
)
def test_phase5_runtime_wrapper_emits_live_success_and_failure_signals(
    failure: bool, outcome: str, reason: str, status: str,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    telemetry = OperationTelemetry.create(
        "runtime", provider.get_tracer("test.phase5.runtime"),
        meter_provider.get_meter("test.phase5.runtime"),
    )

    class Runtime:
        @staticmethod
        def execute(_adapter, _loaded, _request):
            if failure:
                raise RuntimeError("synthetic runtime failure")
            return "result"

    if failure:
        with pytest.raises(RuntimeError, match="synthetic runtime failure"):
            _execute_phase5_conformance_runtime(Runtime(), object(), object(), object(), telemetry)
    else:
        assert _execute_phase5_conformance_runtime(
            Runtime(), object(), object(), object(), telemetry,
        ) == "result"
    span, = exporter.get_finished_spans()
    assert span.name == "capability_anatomy.runtime.execute"
    assert span.status.status_code.name == status
    assert span.attributes["capability_anatomy.outcome"] == outcome
    assert span.attributes["capability_anatomy.reason"] == reason
    event, = span.events
    assert event.name == "operation.decision"
    assert event.attributes["capability_anatomy.reason"] == reason
    points = [
        point
        for resource in reader.get_metrics_data().resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        for point in metric.data.data_points
    ]
    assert len(points) == 1
    assert points[0].value == 1
    assert points[0].attributes["capability_anatomy.outcome"] == outcome


def test_phase5_campaign_decision_helper_emits_collector_readable_signal() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    telemetry = OperationTelemetry.create(
        "execution", provider.get_tracer("test.phase5.decision"),
        meter_provider.get_meter("test.phase5.decision"),
    )
    _record_phase5_decision(
        telemetry, operation="freeze_discovery_ranking", outcome="accepted",
        reason="ranking_derived_from_discovery_only",
    )
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "capability_anatomy.phase5.freeze_discovery_ranking"
    assert spans[0].attributes["capability_anatomy.reason"] == "ranking_derived_from_discovery_only"
    points = [
        point
        for resource in reader.get_metrics_data().resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        for point in metric.data.data_points
    ]
    assert len(points) == 1 and points[0].value == 1


def _record(kind: str, expected: object, *, record_id: str = "row-1") -> NormalizedRecord:
    inputs = {"text": "alpha beta"} if kind == "perplexity" else {
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "function", "function": {"name": "weather"}}] if kind in {"simple", "abstention"} else None,
    }
    metadata: dict[str, object] = {"kind": kind, "group_id": f"group-{record_id}"}
    if kind == "simple":
        metadata["scoring_schema"] = {"argument_names": ["city"], "required": ["city"]}
    return NormalizedRecord(record_id, "discovery", inputs, expected, metadata)


def test_phase5_binding_is_independent_of_selected_tool() -> None:
    suite = Phase5EvaluationSuite()
    record = _record("simple", [{"weather": {"city": ["Paris"]}}])
    observation = EvaluationExecutor().evaluate(
        suite,
        record,
        '[{"name":"wrong_tool","arguments":{"city":"Paris"}}]',
    )

    assert observation.status is ObservationStatus.COMPLETE
    assert {name: score.value for name, score in observation.scores.items()} == {
        "tool_selection": 0.0,
        "argument_binding": 1.0,
        "full_call": 0.0,
    }


def test_phase5_malformed_model_response_is_complete_capability_failure() -> None:
    observation = EvaluationExecutor().evaluate(
        Phase5EvaluationSuite(),
        _record("simple", [{"weather": {"city": ["Paris"]}}]),
        "not-json",
    )

    assert observation.status is ObservationStatus.COMPLETE
    assert {name: score.value for name, score in observation.scores.items()} == {
        "tool_selection": 0.0,
        "argument_binding": 0.0,
        "full_call": 0.0,
    }


def test_phase5_malformed_json_format_response_scores_zero() -> None:
    observation = EvaluationExecutor().evaluate(
        Phase5EvaluationSuite(),
        _record("format", {"kind": "json_keys", "value": ["answer"]}),
        "{malformed",
    )

    assert observation.status is ObservationStatus.COMPLETE
    assert observation.scores["instruction_format"].value == 0.0


def test_phase5_suite_emits_seven_distinct_metrics_and_generic_requests() -> None:
    suite = Phase5EvaluationSuite(max_new_tokens=17)
    cases = (
        (_record("simple", [{"weather": {"city": ["Paris"]}}], record_id="simple"), '[{"name":"weather","arguments":{"city":"Paris"}}]'),
        (_record("abstention", [], record_id="abstain"), "I cannot call a tool."),
        (_record("reasoning", ["42"], record_id="reason"), "42"),
        (_record("format", {"kind": "exact_prefix", "value": "PREFIX answer"}, record_id="format"), "PREFIX answer"),
        (_record("perplexity", None, record_id="ppl"), 2.5),
    )
    observations = [EvaluationExecutor().evaluate(suite, record, output) for record, output in cases]
    aggregate = suite.aggregate(observations)

    assert set(aggregate["metrics"]) == {
        "tool_selection", "argument_binding", "full_call", "abstention",
        "structured_reasoning", "instruction_format", "perplexity",
    }
    assert aggregate["fractions"]["tool_selection"] == {"numerator": 1.0, "denominator": 1}
    assert aggregate["fractions"]["perplexity"] is None
    assert isinstance(suite.build_request(cases[0][0]), GenerationRequest)
    assert suite.build_request(cases[0][0]).max_new_tokens == 17
    assert isinstance(suite.build_request(cases[-1][0]), PerplexityRequest)


def test_phase5_aggregation_reconstructs_fraction_across_repetitions() -> None:
    suite = Phase5EvaluationSuite()
    record = _record("simple", [{"weather": {"city": ["Paris"]}}])
    observations = [
        EvaluationExecutor().evaluate(suite, record, '[{"name":"weather","arguments":{"city":"Paris"}}]'),
        EvaluationExecutor().evaluate(suite, record, '[{"name":"weather","arguments":{"city":"London"}}]'),
    ]

    aggregate = suite.aggregate(observations)
    assert aggregate["metrics"]["argument_binding"] == 0.5
    assert aggregate["fractions"]["argument_binding"] == {"numerator": 1.0, "denominator": 2}


def test_phase5_dataset_requires_exact_frozen_role_membership() -> None:
    discovery = _record("reasoning", ["42"], record_id="discovery-1")
    validation = NormalizedRecord("validation-1", "validation", discovery.input, discovery.expected, {"kind": "reasoning", "group_id": "group-validation-1"})
    plan = {"partitions": {
        "discovery": [{"source_id": discovery.id, "group_key": "group-discovery-1", "kind": "reasoning"}],
        "validation": [{"source_id": validation.id, "group_key": "group-validation-1", "kind": "reasoning"}],
    }}
    provider = Phase5DatasetProvider({"discovery": (discovery,), "validation": (validation,)}, plan)
    provider.validate_for(Phase5EvaluationSuite())
    assert [record.id for record in provider.records("validation")] == ["validation-1"]

    wrong_group = NormalizedRecord(validation.id, validation.partition, validation.input, validation.expected, {**validation.metadata, "group_id": "leaked"})
    with pytest.raises(InvalidConfigurationError, match="frozen role plan"):
        Phase5DatasetProvider({"discovery": (discovery,), "validation": (wrong_group,)}, plan)

    mislabeled = NormalizedRecord(validation.id, "discovery", validation.input, validation.expected, validation.metadata)
    with pytest.raises(InvalidConfigurationError, match="frozen role plan"):
        Phase5DatasetProvider({"discovery": (discovery,), "validation": (mislabeled,)}, plan)


@pytest.mark.parametrize("standalone", [False, True])
def test_governed_source_identity_reads_commit_and_rejects_dirty_tree(tmp_path: Path, standalone: bool) -> None:
    lab = tmp_path if standalone else tmp_path / "packages/research/anatomy"
    paths = [
        "src/capability_anatomy/module.py", "src/capability_anatomy/execution/orchestrator.py",
        "schemas/schema.json", "pyproject.toml", "uv.lock",
        "config/phase0b_prompts.json", "configs/experiments/qwen3-0.6b-phase5-record-plan.json",
        "configs/experiments/qwen3-0.6b-phase5-protocol.json", "configs/experiments/qwen3-0.6b-phase5.yaml",
        "configs/experiments/qwen3-1.7b-phase5-protocol.json",
        "configs/experiments/qwen3-1.7b-phase5.yaml",
        "data/phase5/qwen3-0.6b-phase5-dataset.json",
    ]
    for relative in paths:
        path = lab / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "config", "user.email", "gate5a@example.invalid"), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "config", "user.name", "Gate 5A Test"), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "add", "."), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "commit", "-qm", "frozen"), check=True)
    commit = subprocess.run(
        ("git", "-C", str(tmp_path), "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    config_path = lab / "configs/experiments/qwen3-1.7b-phase5.yaml"
    protocol_path = lab / "configs/experiments/qwen3-1.7b-phase5-protocol.json"
    config = SimpleNamespace(dataset=SimpleNamespace(manifest=Path("../../data/phase5/qwen3-0.6b-phase5-dataset.json")))
    protocol = {"dataset": {
        "manifest": {"path": "../../data/phase5/qwen3-0.6b-phase5-dataset.json"},
        "record_plan": {"path": "qwen3-0.6b-phase5-record-plan.json"},
        "prompt_templates": {"path": "../../config/phase0b_prompts.json"},
        "scorers": {"path": "../../src/capability_anatomy/module.py"},
    }}
    implementation_path = lab / "src/capability_anatomy/execution/orchestrator.py"
    approved, current = _governed_source_identity(
        config_path, commit, config, protocol_path, protocol, implementation_path,
    )
    assert approved == current

    foreign_config = tmp_path / "lookalike/configs/experiments/qwen3-1.7b-phase5.yaml"
    foreign_config.parent.mkdir(parents=True)
    with pytest.raises(GateAuthorizationError, match="gate_source_identity_changed"):
        _governed_source_identity(
            foreign_config, commit, config, protocol_path, protocol, implementation_path,
        )

    unrelated = lab / "configs/experiments/qwen3-0.6b-phase5.yaml"
    unrelated.write_text("unrelated experiment changed\n", encoding="utf-8")
    assert _governed_source_identity(
        config_path, commit, config, protocol_path, protocol, implementation_path,
    ) == (approved, current)

    selected_protocol_bytes = protocol_path.read_bytes()
    protocol_path.write_text("selected experiment changed\n", encoding="utf-8")
    with pytest.raises(GateAuthorizationError, match="gate_source_identity_changed"):
        _governed_source_identity(
            config_path, commit, config, protocol_path, protocol, implementation_path,
        )
    protocol_path.write_bytes(selected_protocol_bytes)

    (lab / "src/capability_anatomy/module.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(GateAuthorizationError, match="gate_source_identity_changed"):
        _governed_source_identity(
            config_path, commit, config, protocol_path, protocol, implementation_path,
        )

    subprocess.run(("git", "-C", str(tmp_path), "add", "."), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "commit", "-qm", "post-approval drift"), check=True)
    approved, current = _governed_source_identity(
        config_path, commit, config, protocol_path, protocol, implementation_path,
    )
    assert approved != current

