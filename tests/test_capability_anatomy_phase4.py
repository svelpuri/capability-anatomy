from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.domain import (
    EvaluationObservation,
    EvaluationScore,
    NormalizedRecord,
    ObservationStatus,
    ParseResult,
    ParseStatus,
)
from capability_anatomy.discovery import ResolvedPlugin
from capability_anatomy.errors import InterruptedRunError, InvalidEvidenceError
from capability_anatomy.execution import ExperimentRunner, FrozenRunStore, MeasurementControls, Task
from capability_anatomy.execution.orchestrator import (
    _phase5_seed,
    _validate_phase5_model_contract,
    _validate_phase5_plugin_contract,
)
from capability_anatomy.datasets import Phase5DatasetProvider
from capability_anatomy.evaluations.plugins import Phase5EvaluationSuite
from capability_anatomy.evaluations import EvaluationExecutor
from capability_anatomy.serialization import canonical_json_bytes, canonical_sha256, serialize_observation
from capability_anatomy.telemetry import OperationTelemetry
from capability_anatomy.cli import main
from capability_anatomy.models.plugins.synthetic import SyntheticModelAdapter
from capability_anatomy.models.plugins.qwen3 import Qwen3Adapter, Qwen3BlockBypassProvider
from capability_anatomy.models.plugins.huggingface import ARCHITECTURE_PROFILES, HuggingFaceCausalLMAdapter
from capability_anatomy.protocols import MeasuredModelAdapter


def _store(root: Path, *, revision: str = "model-v1", seed: int = 7) -> FrozenRunStore:
    return FrozenRunStore(
        root,
        config={"seed": seed, "runtime": {"repetitions": 2}},
        compatibility={
            "model": {"plugin": "synthetic", "revision": revision},
            "suite": {"plugin": "exact", "version": "1"},
            "dataset_sha256": "a" * 64,
        },
    )


def test_synthetic_adapter_uses_declared_measurement_contract() -> None:
    adapter = SyntheticModelAdapter()
    assert isinstance(adapter, MeasuredModelAdapter)
    assert {
        "measurement.memory", "measurement.synchronization", "measurement.tokens"
    } <= adapter.capabilities


def _signals() -> tuple[OperationTelemetry, InMemorySpanExporter, InMemoryMetricReader]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    return (
        OperationTelemetry.create(
            "execution",
            tracer=provider.get_tracer("phase4-test"),
            meter=meter_provider.get_meter("phase4-test"),
        ),
        exporter,
        reader,
    )


def _e2e_config(tmp_path: Path, **updates) -> Path:
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / "configs/examples/synthetic-scan.yaml").read_text())
    value["dataset"]["manifest"] = str(root / "fixtures/synthetic/manifest.json")
    value["output"]["directory"] = str(tmp_path / "run")
    for section, replacement in updates.items():
        value[section].update(replacement)
    path = tmp_path / "experiment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_one_command_runs_baseline_interventions_and_resumes(tmp_path: Path, capsys) -> None:
    config = _e2e_config(tmp_path)
    assert main(("run", "--config", str(config))) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "complete"
    assert response["tasks"] == ["baseline", "scan.component-damage", "scan.component-neutral"]

    task_dir = tmp_path / "run/tasks"
    baseline = json.loads((task_dir / "baseline.json").read_text())
    neutral = json.loads((task_dir / "scan.component-neutral.json").read_text())
    damaged = json.loads((task_dir / "scan.component-damage.json").read_text())
    assert baseline["metrics"]["metrics"] == {"exact_match": 1.0}
    assert neutral["metrics"]["metrics"] == {"exact_match": 1.0}
    assert damaged["metrics"]["metrics"] == {"exact_match": 0.0}
    assert baseline["measurement_controls"] == {
        "randomized_order": True, "repetitions": 1, "seed": 20260903, "warmup_runs": 1
    }
    assert len(baseline["observations"]) == 2
    assert all(item["elapsed_seconds"] >= 0 for item in baseline["observations"])
    assert all(item["peak_memory_bytes"] > 0 for item in baseline["observations"])
    assert all((item["input_tokens"], item["output_tokens"]) == (1, 1) for item in baseline["observations"])
    assert baseline["observations"][0]["scores"]["exact_match"] == {
        "denominator": 1, "numerator": 1.0, "value": 1.0
    }
    assert baseline["provenance"]["plugin"] == "builtin.synthetic-model"
    assert baseline["failures"] == []
    before = {path.name: path.stat().st_mtime_ns for path in task_dir.glob("*.json")}
    assert main(("run", "--config", str(config))) == 0
    capsys.readouterr()
    after = {path.name: path.stat().st_mtime_ns for path in task_dir.glob("*.json")}
    assert before == after


def test_public_workflow_applies_warmups_repetitions_and_seeded_order(tmp_path: Path, capsys, monkeypatch) -> None:
    config = _e2e_config(
        tmp_path,
        experiment={"seed": 1},
        runtime={"warmup_runs": 1, "repetitions": 2, "randomized_execution_order": True},
    )
    original = SyntheticModelAdapter.execute
    calls = []
    synchronizations = 0

    def record(self, loaded, request):
        calls.append(request.payload["text"])
        return original(self, loaded, request)

    def synchronize(self, loaded) -> None:
        nonlocal synchronizations
        synchronizations += 1

    monkeypatch.setattr(SyntheticModelAdapter, "execute", record)
    monkeypatch.setattr(SyntheticModelAdapter, "synchronize", synchronize)
    assert main(("run", "--config", str(config))) == 0
    response = json.loads(capsys.readouterr().out)
    baseline = json.loads((tmp_path / "run/tasks/baseline.json").read_text())

    assert len(calls) == 18  # 3 tasks * 2 records * (1 warmup + 2 measured)
    assert synchronizations == 36  # before and after every model call
    assert response["component_scan_order"] == ["component.damage", "component.neutral"]
    assert json.loads((tmp_path / "run/component-scan-order.json").read_text()) == response["component_scan_order"]
    assert baseline["execution_order"] == ["discovery-2", "discovery-1"]
    assert [item["repetition"] for item in baseline["observations"]] == [0, 1, 0, 1]
    assert all(item["elapsed_seconds"] is not None for item in baseline["observations"])


def test_resume_uses_frozen_component_scan_order(tmp_path: Path, capsys, monkeypatch) -> None:
    config = _e2e_config(tmp_path, experiment={"seed": 1})
    assert main(("run", "--config", str(config))) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["component_scan_order"] == ["component.damage", "component.neutral"]

    task_dir = tmp_path / "run/tasks"
    (task_dir / "scan.component-neutral.json").unlink()
    (task_dir / "scan.component-neutral.complete.json").unlink()
    monkeypatch.setattr(MeasurementControls, "order", lambda _self, values: tuple(values))

    assert main(("run", "--config", str(config))) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["component_scan_order"] == first["component_scan_order"]
    assert json.loads((tmp_path / "run/component-scan-order.json").read_text()) == first["component_scan_order"]
    assert (task_dir / "scan.component-neutral.complete.json").exists()


def test_one_command_rejects_incompatible_plugin_and_incomplete_metrics(tmp_path: Path, capsys) -> None:
    incompatible = _e2e_config(tmp_path / "incompatible", model={"plugin": "missing.model"})
    assert main(("run", "--config", str(incompatible))) == 5
    assert "unsupported_plugin" in capsys.readouterr().err

    incomplete = _e2e_config(tmp_path / "incomplete", capability={"target_metrics": ["missing_metric"]})
    assert main(("run", "--config", str(incomplete))) == 2
    assert "incomplete" in capsys.readouterr().err
    payload = json.loads((tmp_path / "incomplete/run/failures/baseline.json").read_text())
    assert payload["complete"] is False
    assert payload["missing_metrics"] == ["missing_metric"]
    assert not (tmp_path / "incomplete/run/tasks/baseline.complete.json").exists()
    state = json.loads((tmp_path / "incomplete/run/execution-state.json").read_text())
    assert state == {"reason": "incomplete_task", "state": "failed"}


def test_public_qwen_composition_refuses_before_model_load_without_frozen_protocol(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    config = _e2e_config(
        tmp_path,
        model={
            "plugin": "huggingface.causal-lm",
            "source": "Qwen/Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "parameters": {"architecture_profile": "qwen3-dense-v1", "dtype": "float16"},
        },
        capability={
            "evaluation_plugin": "reference.phase5-qwen-retention",
            "suite_version": "1",
            "target_metrics": ["tool_selection", "argument_binding", "full_call", "abstention"],
            "collateral_metrics": ["structured_reasoning", "instruction_format", "perplexity"],
        },
        dataset={"provider": "reference.phase5-records"},
        intervention={"plugin": "huggingface.block-bypass"},
        runtime={"parameters": {"device": "mps", "context_length": 32768}},
    )
    loaded = False

    def forbidden_load(*_args, **_kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("model load crossed the protocol gate")

    monkeypatch.setattr(HuggingFaceCausalLMAdapter, "load", forbidden_load)
    assert main(("run", "--config", str(config))) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_evidence"
    assert loaded is False


def test_phase5_installed_plugin_contract_enforces_version_api_and_capability() -> None:
    from capability_anatomy.runtime import LocalRuntimePlugin

    plugin_types = {
        "model": Qwen3Adapter,
        "evaluation": Phase5EvaluationSuite,
        "dataset": Phase5DatasetProvider,
        "intervention": Qwen3BlockBypassProvider,
    }
    protocol = {"plugins": [
        {"role": role, "name": plugin.name, "version": plugin.version, "api_version": plugin.api_version, "capabilities": sorted(plugin.capabilities)}
        for role, plugin in plugin_types.items()
    ] + [
        {"role": "runtime", "name": LocalRuntimePlugin.name, "version": LocalRuntimePlugin.version, "api_version": LocalRuntimePlugin.api_version, "capabilities": sorted(LocalRuntimePlugin.capabilities)},
    ]}
    resolved = {
        item["role"]: ResolvedPlugin(item["role"], plugin_types[item["role"]] if item["role"] == "dataset" else plugin_types[item["role"]](), "test", "1")
        for item in protocol["plugins"] if item["role"] != "runtime"
    }
    resolved["runtime"] = ResolvedPlugin("runtime", LocalRuntimePlugin(), "test", "1")
    assert set(_validate_phase5_plugin_contract(protocol, resolved)) == {"model", "evaluation", "dataset", "intervention", "runtime"}

    for field, bad_value in (("version", "unsupported-version"), ("api_version", "old"), ("capabilities", ["model.invented"])):
        mutant = json.loads(json.dumps(protocol))
        mutant["plugins"][0][field] = bad_value
        with pytest.raises(InvalidEvidenceError, match="installed plugins"):
            _validate_phase5_plugin_contract(mutant, resolved)

    omitted = json.loads(json.dumps(protocol))
    omitted["plugins"][0]["capabilities"].pop()
    with pytest.raises(InvalidEvidenceError, match="installed plugins"):
        _validate_phase5_plugin_contract(omitted, resolved)


def test_phase5_top_level_model_identity_must_match_installed_adapter() -> None:
    config = type("Config", (), {"model": type("Model", (), {
        "source": "source", "revision": "a" * 40,
        "parameters": {"architecture_profile": "qwen3-dense-v1"},
    })()})()
    protocol = {"model": {
        "plugin": HuggingFaceCausalLMAdapter.name,
        "version": HuggingFaceCausalLMAdapter.version,
        "source": "source",
        "revision": "a" * 40,
        "architecture_profile": "qwen3-dense-v1",
        "architecture_profile_sha256": ARCHITECTURE_PROFILES["qwen3-dense-v1"].sha256,
    }}
    _validate_phase5_model_contract(protocol, config, HuggingFaceCausalLMAdapter())
    for field, value in (("plugin", "other"), ("version", "unsupported-version")):
        mutant = json.loads(json.dumps(protocol))
        mutant["model"][field] = value
        with pytest.raises(InvalidEvidenceError, match="installed adapter"):
            _validate_phase5_model_contract(mutant, config, HuggingFaceCausalLMAdapter())


def test_phase5_record_and_component_seed_domains_are_distinct_and_stable() -> None:
    assert _phase5_seed(20260904, "records") == _phase5_seed(20260904, "records")
    assert _phase5_seed(20260904, "records") != _phase5_seed(20260904, "components")


def test_one_command_atomically_records_model_failures(tmp_path: Path, capsys, monkeypatch) -> None:
    config = _e2e_config(tmp_path)

    def fail(self, loaded, request):
        raise RuntimeError("private model failure")

    monkeypatch.setattr(SyntheticModelAdapter, "execute", fail)
    assert main(("run", "--config", str(config))) == 2
    assert "private model failure" not in capsys.readouterr().err
    payload = json.loads((tmp_path / "run/failures/baseline.json").read_text())
    assert payload["complete"] is False
    assert payload["failures"] == ["RuntimeError", "RuntimeError"]
    assert payload["observations"][0]["status"] == "error"
    assert payload["observations"][0]["raw_output"] is None
    assert not (tmp_path / "run/tasks/baseline.complete.json").exists()


def test_transient_failed_scan_retries_while_valid_tasks_stay_cached(tmp_path: Path, capsys, monkeypatch) -> None:
    config = _e2e_config(tmp_path, runtime={"warmup_runs": 0, "repetitions": 1, "randomized_execution_order": False})
    original = SyntheticModelAdapter.execute
    calls = 0

    def transient(self, loaded, request):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("transient")
        return original(self, loaded, request)

    monkeypatch.setattr(SyntheticModelAdapter, "execute", transient)
    assert main(("run", "--config", str(config))) == 2
    capsys.readouterr()
    task_dir = tmp_path / "run/tasks"
    stable = {
        name: (task_dir / name).stat().st_mtime_ns
        for name in ("baseline.json", "scan.component-neutral.json")
    }
    assert not (task_dir / "scan.component-damage.complete.json").exists()

    monkeypatch.setattr(SyntheticModelAdapter, "execute", original)
    assert main(("run", "--config", str(config))) == 0
    capsys.readouterr()
    assert stable == {name: (task_dir / name).stat().st_mtime_ns for name in stable}
    assert (task_dir / "scan.component-damage.complete.json").exists()


def test_output_retention_hashes_secret_prompt_and_removes_raw_output(tmp_path: Path, capsys) -> None:
    secret = "secret-prompt-and-output"
    config = _e2e_config(
        tmp_path,
        output={"retain_prompts": True, "retain_raw_outputs": False, "prompt_storage": "hash_only"},
    )
    value = yaml.safe_load(config.read_text())
    manifest = tmp_path / "secret-manifest.json"
    manifest.write_text(json.dumps({
        "license": "private", "revision": "sealed-v1", "partitions": {
            "discovery": [{"id": "secret-1", "partition": "discovery", "input": {"text": secret}, "expected": secret, "metadata": {}}],
            "validation": [{"id": "safe-1", "partition": "validation", "input": {"text": "safe"}, "expected": "safe", "metadata": {}}],
        }
    }), encoding="utf-8")
    value["dataset"]["manifest"] = str(manifest)
    config.write_text(yaml.safe_dump(value), encoding="utf-8")

    assert main(("run", "--config", str(config))) == 0
    capsys.readouterr()
    run_text = "".join(path.read_text() for path in (tmp_path / "run").rglob("*.json"))
    assert secret not in run_text
    observation = json.loads((tmp_path / "run/tasks/baseline.json").read_text())["observations"][0]
    assert observation["raw_output"] is None
    assert observation["parsed"]["value"] is None
    assert observation["prompt"] == canonical_sha256({"text": secret})
    assert observation["scores"]["exact_match"] == {"value": 1.0, "numerator": 1.0, "denominator": 1}


def test_one_command_resumes_after_interruption(tmp_path: Path, capsys, monkeypatch) -> None:
    config = _e2e_config(tmp_path)
    original = SyntheticModelAdapter.execute
    interrupted = False

    def interrupt_once(self, loaded, request):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original(self, loaded, request)

    monkeypatch.setattr(SyntheticModelAdapter, "execute", interrupt_once)
    assert main(("run", "--config", str(config))) == 4
    assert "interrupted" in capsys.readouterr().err
    state = json.loads((tmp_path / "run/execution-state.json").read_text())
    assert state["state"] == "interrupted"

    monkeypatch.setattr(SyntheticModelAdapter, "execute", original)
    assert main(("run", "--config", str(config))) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "run/execution-state.json").read_text())["state"] == "complete"


def test_model_execution_failure_signal_is_sanitized() -> None:
    telemetry, exporter, _reader = _signals()
    observation = EvaluationExecutor(telemetry).execution_failure(
        NormalizedRecord("private-id", "discovery", {}, {}, {}), "RuntimeError"
    )
    assert observation.error_type == "RuntimeError"
    span = next(span for span in exporter.get_finished_spans() if span.name == "capability_anatomy.evaluation.record")
    assert span.status.status_code.name == "ERROR"
    assert span.events[0].attributes["capability_anatomy.reason"] == "model_execution_failed"
    assert "private-id" not in repr(span)


def test_observation_serialization_preserves_score_reconstruction_fields() -> None:
    observation = EvaluationObservation(
        example_id="example-1",
        partition="discovery",
        status=ObservationStatus.COMPLETE,
        raw_output="yes",
        parsed=ParseResult(ParseStatus.PARSED, "yes"),
        scores={
            "accuracy": EvaluationScore(0.5, 1.0, 2),
            "latency": EvaluationScore(0.125),
        },
    )

    serialized = serialize_observation(observation)
    assert serialized["scores"] == {
        "accuracy": {"value": 0.5, "numerator": 1.0, "denominator": 2},
        "latency": {"value": 0.125, "numerator": None, "denominator": None},
    }
    assert "prompt" not in serialized
    assert json.loads(canonical_json_bytes(serialized))["scores"] == serialized["scores"]


def test_task_store_preserves_observation_value_numerator_and_denominator(tmp_path: Path) -> None:
    observation = EvaluationObservation(
        example_id="example-1",
        partition="discovery",
        status=ObservationStatus.COMPLETE,
        raw_output="yes",
        parsed=ParseResult(ParseStatus.PARSED, "yes"),
        scores={"accuracy": EvaluationScore(0.5, 1.0, 2)},
    )
    store = _store(tmp_path)
    store.initialize()
    store.commit("baseline", {"observations": [observation]})

    durable = json.loads((tmp_path / "tasks" / "baseline.json").read_text())
    assert durable["observations"][0]["scores"]["accuracy"] == {
        "denominator": 2,
        "numerator": 1.0,
        "value": 0.5,
    }


def test_interrupted_run_resumes_only_compatible_completed_tasks(tmp_path: Path) -> None:
    calls = {"baseline": 0, "scan": 0}

    def baseline():
        calls["baseline"] += 1
        return {"kind": "baseline", "score": 1.0}

    def interrupt():
        calls["scan"] += 1
        raise KeyboardInterrupt

    runner = ExperimentRunner(_store(tmp_path))
    with pytest.raises(InterruptedRunError, match="resumable"):
        runner.run((Task("baseline", "baseline", baseline), Task("scan.a", "scan", interrupt)))

    assert json.loads((tmp_path / "execution-state.json").read_text())["state"] == "interrupted"

    def scan():
        calls["scan"] += 1
        return {"kind": "scan", "score": 0.75}

    result = ExperimentRunner(_store(tmp_path)).run(
        (Task("baseline", "baseline", baseline), Task("scan.a", "scan", scan))
    )
    assert calls == {"baseline": 1, "scan": 2}
    assert result["baseline"]["score"] == 1.0
    assert result["scan.a"]["score"] == 0.75
    assert json.loads((tmp_path / "execution-state.json").read_text())["state"] == "complete"


def test_resume_rejects_config_and_baseline_compatibility_changes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    assert json.loads((tmp_path / "experiment-config.json").read_text())["seed"] == 7
    assert json.loads((tmp_path / "compatibility.json").read_text())["model"]["revision"] == "model-v1"
    with pytest.raises(InvalidEvidenceError, match="changed"):
        _store(tmp_path, seed=8).initialize()
    with pytest.raises(InvalidEvidenceError, match="changed"):
        _store(tmp_path, revision="model-v2").initialize()


def test_task_graph_requires_one_baseline_identity(tmp_path: Path) -> None:
    runner = ExperimentRunner(_store(tmp_path))
    with pytest.raises(ValueError, match="start with a baseline"):
        runner.run((Task("scan.a", "scan", lambda: {}),))
    with pytest.raises(ValueError, match="unique"):
        runner.run(
            (
                Task("baseline", "baseline", lambda: {}),
                Task("baseline", "scan", lambda: {}),
            )
        )
    with pytest.raises(ValueError, match="baseline or scan"):
        Task("transform", "transform", lambda: {})


def test_corrupt_or_uncommitted_task_is_rerun(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    store.commit("baseline", {"score": 1.0})
    (tmp_path / "tasks" / "baseline.json").write_text('{"score":0.0}', encoding="utf-8")
    calls = 0

    def baseline():
        nonlocal calls
        calls += 1
        return {"score": 0.8}

    result = ExperimentRunner(_store(tmp_path)).run((Task("baseline", "baseline", baseline),))
    assert calls == 1
    assert result["baseline"] == {"score": 0.8}

    with pytest.raises(InvalidEvidenceError, match="task ID"):
        store.commit("../escape", {"score": 1.0})


def test_measurement_controls_are_reproducible_and_capture_runtime_signals() -> None:
    ticks = iter((1.0, 1.25, 2.0, 2.5))
    synchronizations = 0
    calls = 0

    def synchronize():
        nonlocal synchronizations
        synchronizations += 1

    def operation():
        nonlocal calls
        calls += 1
        return {"input_tokens": 3, "output_tokens": 2}

    controls = MeasurementControls(
        seed=17,
        warmup_runs=1,
        repetitions=2,
        randomized_order=True,
        synchronize=synchronize,
        memory_bytes=lambda: 4096,
        clock=lambda: next(ticks),
    )
    assert controls.order(range(8)) == MeasurementControls(
        seed=17, warmup_runs=0, repetitions=1, randomized_order=True
    ).order(range(8))
    measurements = controls.measure(
        operation,
        lambda value: (value["input_tokens"], value["output_tokens"]),
    )
    assert calls == 3
    assert synchronizations == 6
    assert [item.elapsed_seconds for item in measurements] == [0.25, 0.5]
    assert all(item.peak_memory_bytes == 4096 for item in measurements)
    assert [(item.input_tokens, item.output_tokens) for item in measurements] == [(3, 2), (3, 2)]


def test_task_lifecycle_is_traceable_countable_and_redacted(tmp_path: Path) -> None:
    telemetry, exporter, reader = _signals()
    runner = ExperimentRunner(_store(tmp_path), telemetry)
    secret = "private-task-payload"
    task = Task("baseline", "baseline", lambda: {"output": secret})

    runner.run((task,))
    ExperimentRunner(_store(tmp_path), telemetry).run((task,))

    events = [event for span in exporter.get_finished_spans() if span.name == "capability_anatomy.execution.task" for event in span.events]
    assert [event.attributes["capability_anatomy.reason"] for event in events] == [
        "task_committed",
        "compatible_task_complete",
    ]
    signal_text = repr(exporter.get_finished_spans()) + repr(reader.get_metrics_data())
    assert secret not in signal_text
    metric = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics[0]
    assert sum(point.value for point in metric.data.data_points if point.attributes.get("capability_anatomy.operation") == "execute_task") == 2


def test_frozen_component_plan_decision_is_traceable(tmp_path: Path) -> None:
    telemetry, exporter, _reader = _signals()
    runner = ExperimentRunner(_store(tmp_path), telemetry)
    runner.run(
        (Task("baseline", "baseline", lambda: {"complete": True}),),
        frozen_plan_reason="component_scan_order_frozen",
    )

    plan = next(span for span in exporter.get_finished_spans() if span.name.endswith(".plan"))
    assert plan.events[0].attributes["capability_anatomy.operation"] == "freeze_task_plan"
    assert plan.events[0].attributes["capability_anatomy.reason"] == "component_scan_order_frozen"


def test_task_failure_has_cause_and_error_status_without_payload_leak(tmp_path: Path) -> None:
    telemetry, exporter, _reader = _signals()

    def fail():
        raise RuntimeError("private failure detail")

    with pytest.raises(RuntimeError, match="private failure detail"):
        ExperimentRunner(_store(tmp_path), telemetry).run((Task("baseline", "baseline", fail),))

    span = next(span for span in exporter.get_finished_spans() if span.name == "capability_anatomy.execution.task")
    assert span.status.status_code.name == "ERROR"
    assert span.events[0].attributes["capability_anatomy.reason"] == "task_failed"
    assert "private failure detail" not in repr(span.events)
