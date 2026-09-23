from __future__ import annotations

import json
from pathlib import Path
import tomllib

import jsonschema
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.cli import main
from capability_anatomy.config import load_experiment_config
from capability_anatomy.domain import MetricDirection, MetricFraction, MetricResult, ObservationStatus, PolicyStatus
from capability_anatomy.errors import ExitCode, InvalidConfigurationError
from capability_anatomy.telemetry import DecisionTelemetry


LAB_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = LAB_ROOT / "configs/examples/synthetic-scan.yaml"
SCHEMA_ROOT = LAB_ROOT / "schemas"


def _schema(name: str) -> dict:
    return json.loads((SCHEMA_ROOT / name).read_text())


def _signals() -> tuple[DecisionTelemetry, InMemorySpanExporter, InMemoryMetricReader]:
    span_exporter = InMemorySpanExporter()
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    signals = DecisionTelemetry.create(
        tracer=trace_provider.get_tracer("test"),
        meter=meter_provider.get_meter("test"),
    )
    return signals, span_exporter, metric_reader


def test_example_config_loads_into_immutable_domain_model() -> None:
    config = load_experiment_config(EXAMPLE_CONFIG)

    assert config.experiment_id == "synthetic-component-scan"
    assert config.model.plugin == "builtin.synthetic-model"
    assert config.capability.target_metrics == ("exact_match",)
    assert config.intervention.plugin == "builtin.synthetic-intervention"
    assert config.runtime.executor == "builtin.local"
    assert config.output.prompt_storage == "hash_only"
    assert {status.value for status in PolicyStatus} == {"PASS", "FAIL", "INVALID"}
    result = MetricResult(
        metric_id="exact_match",
        aggregation_version="1",
        direction=MetricDirection.HIGHER_IS_BETTER,
        unit="proportion",
        status=ObservationStatus.COMPLETE,
        value=0.9,
        fraction=MetricFraction(numerator=9, denominator=10),
    )
    assert result.direction == "higher_is_better"


@pytest.mark.parametrize(
    ("original", "changed", "message"),
    [
        ("revision: fixture-v1", "revision: main", "revision must be immutable"),
        ("validation_partition: validation", "validation_partition: discovery", "partitions must be disjoint"),
        ("collateral_metrics: []", "collateral_metrics: [exact_match]", "metrics must be disjoint"),
        ("revision: fixture-v1", "revision: refs/heads/main", "revision must be immutable"),
    ],
)
def test_semantically_invalid_config_is_refused(
    tmp_path: Path, original: str, changed: str, message: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(EXAMPLE_CONFIG.read_text().replace(original, changed))

    with pytest.raises(InvalidConfigurationError, match=message):
        load_experiment_config(path)


def test_yaml_unsafe_constructor_is_never_executed(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:os.system ['false']\n")

    with pytest.raises(InvalidConfigurationError, match="not valid JSON or YAML"):
        load_experiment_config(path)


def test_configuration_decisions_are_trace_correlated_counted_and_redacted(tmp_path: Path) -> None:
    signals, span_exporter, metric_reader = _signals()
    valid = tmp_path / "valid.yaml"
    valid.write_text(EXAMPLE_CONFIG.read_text())
    invalid = tmp_path / "private-token-DO-NOT-EXPORT.yaml"
    invalid.write_text(EXAMPLE_CONFIG.read_text().replace("revision: fixture-v1", "revision: main"))

    load_experiment_config(valid, signals)
    with pytest.raises(InvalidConfigurationError):
        load_experiment_config(invalid, signals)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert {event.name for span in spans for event in span.events} == {"configuration.decision"}
    assert {span.attributes["capability_anatomy.outcome"] for span in spans} == {"accepted", "refused"}
    assert all(span.context.trace_id for span in spans)
    rendered = repr(spans)
    assert "private-token-DO-NOT-EXPORT" not in rendered
    assert "fixture-v1" not in rendered

    metrics = metric_reader.get_metrics_data()
    names = {
        metric.name
        for resource in metrics.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert "capability_anatomy.config.decisions" in names


def test_invalid_config_exits_before_adapter_resolution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}")

    assert main(["run", "--config", str(path)]) == ExitCode.INVALID
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "invalid_configuration"
    assert "adapter" not in error["message"]


def test_schema_failure_does_not_echo_private_value(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    private_value = "private-token-DO-NOT-ECHO"
    path = tmp_path / "invalid.yaml"
    path.write_text(EXAMPLE_CONFIG.read_text().replace("seed: 20260903", f"seed: {private_value}"))

    assert main(["run", "--config", str(path)]) == ExitCode.INVALID
    assert private_value not in capsys.readouterr().err


def test_removed_scan_command_is_not_advertised(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scan", "--config", str(EXAMPLE_CONFIG)]) == ExitCode.INVALID
    error = capsys.readouterr().err
    assert "invalid choice" in error


def test_doctor_reports_core_contract(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == ExitCode.PASSED
    result = json.loads(capsys.readouterr().out)
    assert result["core_api"] == "capability-anatomy/plugin-api/v1"
    assert result["status"] == "ok"
    assert result["version"] == "0.1.0"
    assert result["checks"] and all(check["status"] == "ok" for check in result["checks"])
    assert tuple(ExitCode) == (
        ExitCode.PASSED,
        ExitCode.INVALID,
        ExitCode.RUNTIME_FAILURE,
        ExitCode.INTERRUPTED,
        ExitCode.UNSUPPORTED,
    )


@pytest.mark.parametrize(
    ("schema_name", "instance"),
    [
        (
            "evidence-manifest.v1.schema.json",
            {
                "schema_version": "capability-anatomy/evidence-manifest/v1",
                "run_id": "synthetic-1",
                "config_sha256": "a" * 64,
                "execution": {"state": "complete", "completed_stages": ["baseline"]},
                "policy_result": {"policy_id": "retention-v1", "status": "PASS", "reasons": []},
                "artifacts": [{"path": "baseline/summary.json", "bytes": 42, "sha256": "b" * 64}],
            },
        ),
        (
            "metric-result.v1.schema.json",
            {
                "metric_id": "exact_match",
                "aggregation_version": "1",
                "direction": "higher_is_better",
                "unit": "proportion",
                "status": "complete",
                "value": 0.9,
                "fraction": {"numerator": 9, "denominator": 10},
            },
        ),
        (
            "transformation-recipe.v1.schema.json",
            {
                "schema_version": "capability-anatomy/transformation-recipe/v1",
                "plugin": {
                    "name": "community.rotate-components",
                    "version": "1",
                    "api_version": "capability-anatomy/plugin-api/v1",
                },
                "parameters": {"angle": 0.25},
                "input_artifacts": [{"uri": "file:model-in", "digest": f"sha256:{'b' * 64}"}],
                "output_artifacts": [{"uri": "file:model-out", "digest": f"sha256:{'c' * 64}"}],
                "component_changes": [
                    {"component_id": "encoder.branch.attention", "change": "rotated", "details": {}}
                ],
            },
        ),
    ],
)
def test_v1_schema_examples_validate(schema_name: str, instance: dict) -> None:
    jsonschema.Draft202012Validator(_schema(schema_name)).validate(instance)


def test_evidence_schema_rejects_parent_traversal() -> None:
    instance = {
        "schema_version": "capability-anatomy/evidence-manifest/v1",
        "run_id": "synthetic-1",
        "config_sha256": "a" * 64,
        "execution": {"state": "complete", "completed_stages": ["baseline"]},
        "artifacts": [{"path": "../secret", "bytes": 1, "sha256": "b" * 64}],
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_schema("evidence-manifest.v1.schema.json")).validate(instance)


def test_metric_schema_rejects_unknown_direction() -> None:
    instance = {
        "metric_id": "exact_match",
        "aggregation_version": "1",
        "direction": "ambiguous",
        "unit": "proportion",
        "status": "complete",
        "value": 0.9,
        "fraction": {"numerator": 9, "denominator": 10},
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_schema("metric-result.v1.schema.json")).validate(instance)


def test_typed_metric_rejects_non_finite_complete_result() -> None:
    with pytest.raises(ValueError, match="finite"):
        MetricResult(
            metric_id="perplexity",
            aggregation_version="1",
            direction=MetricDirection.LOWER_IS_BETTER,
            unit="ratio",
            status=ObservationStatus.COMPLETE,
            value=float("nan"),
            fraction=MetricFraction(numerator=1, denominator=1),
        )


def test_non_proportion_metric_needs_no_fraction() -> None:
    metric = MetricResult(
        metric_id="latency.p95",
        aggregation_version="1",
        direction=MetricDirection.LOWER_IS_BETTER,
        unit="ms",
        status=ObservationStatus.COMPLETE,
        value=18.4,
        sample_count=50,
        distribution={"p50": 12.1, "p95": 18.4},
        uncertainty={"method": "bootstrap", "lower": 17.2, "upper": 19.8},
    )
    assert metric.fraction is None


def test_execution_manifest_does_not_require_policy_verdict() -> None:
    instance = {
        "schema_version": "capability-anatomy/evidence-manifest/v1",
        "run_id": "execution-only",
        "config_sha256": "a" * 64,
        "execution": {"state": "interrupted", "completed_stages": ["baseline"]},
        "artifacts": [{"path": "baseline/summary.json", "bytes": 1, "sha256": "b" * 64}],
    }
    jsonschema.Draft202012Validator(_schema("evidence-manifest.v1.schema.json")).validate(instance)


def test_experiment_schema_keeps_plugin_settings_opaque() -> None:
    experiment = _schema("experiment-config.v1.schema.json")["properties"]
    assert set(experiment["model"]["properties"]) == {"plugin", "source", "revision", "parameters"}
    assert set(experiment["runtime"]["properties"]) == {
        "executor",
        "deterministic",
        "warmup_runs",
        "randomized_execution_order",
        "repetitions",
        "parameters",
    }
    assert set(experiment["intervention"]["properties"]) == {"plugin", "parameters"}


def test_transformation_schema_is_not_removal_specific() -> None:
    transformation = _schema("transformation-recipe.v1.schema.json")["properties"]
    assert not {"removed_component_ids", "retained_component_ids", "component_mapping"} & set(transformation)
    assert {"plugin", "parameters", "input_artifacts", "output_artifacts", "component_changes"} <= set(transformation)


def test_evidence_schema_separates_execution_and_policy() -> None:
    evidence = _schema("evidence-manifest.v1.schema.json")["properties"]
    assert "status" not in evidence
    assert "execution" in evidence
    assert "policy_result" in evidence


def test_core_distribution_has_no_model_runtime_dependencies() -> None:
    project = tomllib.loads((LAB_ROOT / "pyproject.toml").read_text())["project"]
    dependency_names = {dependency.split("=", 1)[0].lower() for dependency in project["dependencies"]}
    assert project["name"] == "capability-anatomy"
    assert not {"torch", "transformers", "huggingface-hub", "sacrebleu", "psutil"} & dependency_names
    assert {"torch==2.14.0", "transformers==5.17.0"} <= set(project["optional-dependencies"]["qwen3"])


def test_metric_schema_accepts_scalar_latency_without_fraction() -> None:
    instance = {
        "metric_id": "latency.p95",
        "aggregation_version": "1",
        "direction": "lower_is_better",
        "unit": "ms",
        "status": "complete",
        "value": 18.4,
        "sample_count": 50,
        "distribution": {"p50": 12.1, "p95": 18.4},
        "uncertainty": {"method": "bootstrap", "lower": 17.2, "upper": 19.8},
    }
    jsonschema.Draft202012Validator(_schema("metric-result.v1.schema.json")).validate(instance)
