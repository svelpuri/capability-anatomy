from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import json
import os
import signal
import threading
import sys
import uuid
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode, Link

from ..config import load_experiment_config
from ..authored_inputs import load_mapping, read_regular_bytes, AuthoredInputError
from ..discovery import PluginDiscovery, ResolvedPlugin
from ..errors import InvalidEvidenceError, InterruptedRunError
from ..evaluations import EvaluationExecutor
from ..interventions import BlockBypass
from ..models.base import ExecutionRequest
from ..phase5_protocol import sha256_file, validate_phase5_protocol
from ..protocols import MeasuredModelAdapter, RuntimePlugin
from ..serialization import canonical_json_bytes, canonical_sha256, serialize_observation, to_primitive
from ..telemetry import OperationTelemetry, telemetry_scope, identity_scope
from ..credentials import RuntimeCredentials, credential_scope
from .evidence_limits import MAX_ARTIFACT_BYTES, MAX_TRACE_BYTES
from .evidence_json import parse_mapping, encode_json
from ..errors import GateAuthorizationError
from ..review_policy import AUTHORIZATION_FIELDS, AUTHORIZATION_V2, validate_review_reference
from .secure_fs import run_ownership, read_bytes, exists
from .runner import ExperimentRunner, Task
from .store import FrozenRunStore
from .controls import MeasurementControls
from .phase5_campaign import (
    aggregate_baseline_controls,
    aggregate_observations,
    build_campaign_plan,
    compute_damage_matrix,
    compute_validation_results,
    render_report_markdown,
    select_discovery_candidates,
    serialize_campaign_plan,
    summarize_measurements,
    write_final_artifacts,
)


_TRACE_CAPTURE = ContextVar("capability_anatomy.trace_capture", default=None)


def _checkpoint_task_trace(output, task_id, payload):
    capture = _TRACE_CAPTURE.get()
    if capture is None:
        return
    exporter, reader, trace_id = capture
    evidence = _serialize_phase5_trace(exporter, reader, trace_id)
    task_hash = hashlib.sha256(task_id.encode()).hexdigest()
    evidence["spans"] = [item for item in evidence["spans"]
                         if item["attributes"].get("capability_anatomy.task_id_sha256") == task_hash]
    evidence["scope"] = "completed task record spans before the task completion marker"
    evidence["task_id_sha256"] = task_hash
    evidence["payload_sha256"] = canonical_sha256(payload)
    FrozenRunStore._atomic_write(output / "task-traces" / (task_id + ".json"), encode_json(evidence, max_bytes=MAX_TRACE_BYTES))


_PHASE5_METRICS_BY_KIND = {
    "simple": ("tool_selection", "argument_binding", "full_call"),
    "abstention": ("abstention",),
    "reasoning": ("structured_reasoning",),
    "format": ("instruction_format",),
    "perplexity": ("perplexity",),
}


def _phase5_seed(seed: int, domain: str) -> int:
    digest = hashlib.sha256(f"{seed}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _experiment_relative_path(path: Path) -> str:
    resolved = path.resolve()
    lab_root = Path(__file__).resolve().parents[3]
    try:
        return resolved.relative_to(lab_root).as_posix()
    except ValueError:
        return resolved.as_posix()


RUNTIME_OPERATION_CAPABILITIES = frozenset({
    "runtime.execute", "runtime.memory", "runtime.synchronization",
    "runtime.timing", "runtime.tokens",
})


def _validate_phase5_conformance(
    *, before_hooks: int, after_hooks: int, normal_calls: int, restored_calls: int,
    bypass_calls: int, bypass_identity_calls: int, sequence_widths: list[int],
    normal: Any, restored: Any,
) -> tuple[int, int]:
    if before_hooks != after_hooks or not normal_calls or not restored_calls:
        raise InvalidEvidenceError("Phase 5 conformance invocation or cleanup failed")
    prefill_calls = sum(width > 1 for width in sequence_widths)
    decode_calls = sum(width == 1 for width in sequence_widths)
    if not prefill_calls or not decode_calls or bypass_identity_calls != bypass_calls:
        raise InvalidEvidenceError("Phase 5 conformance did not prove prefill/decode bypass")
    if (str(restored) != str(normal) or getattr(restored, "generated_token_count", None)
            != getattr(normal, "generated_token_count", None)):
        raise InvalidEvidenceError("Phase 5 conformance deterministic restoration failed")
    return prefill_calls, decode_calls


def _execute_phase5_conformance_runtime(
    runtime_plugin: Any, adapter: Any, loaded: Any, request: Any,
    telemetry: OperationTelemetry | None,
) -> Any:
    if telemetry is None:
        return runtime_plugin.execute(adapter, loaded, request)
    with telemetry.tracer.start_as_current_span(
        "capability_anatomy.runtime.execute",
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            result = runtime_plugin.execute(adapter, loaded, request)
        except Exception:
            telemetry.record(
                span, operation="execute", outcome="failed",
                reason="runtime_plugin_execution_failed",
            )
            span.set_status(Status(StatusCode.ERROR, "runtime_plugin_execution_failed"))
            raise
        telemetry.record(
            span, operation="execute", outcome="accepted",
            reason="runtime_plugin_execution_complete",
        )
        return result


@contextmanager
def _identity_no_op_hook(component: Any, telemetry: OperationTelemetry | None = None):
    implementation = component.implementation
    if not callable(getattr(implementation, "register_forward_hook", None)):
        raise InvalidEvidenceError("Phase 5 no-op control cannot instrument component")
    before = len(implementation._forward_hooks)
    signals = telemetry or OperationTelemetry.create("intervention")
    with signals.tracer.start_as_current_span(
        "capability_anatomy.intervention.identity_no_op",
        record_exception=False, set_status_on_exception=False,
    ) as span:
        handle = implementation.register_forward_hook(lambda _module, _args, output: output)
        signals.record(span, operation="identity_no_op", outcome="accepted", reason="identity_no_op_active")
        try:
            yield
        finally:
            handle.remove()
            if len(implementation._forward_hooks) != before:
                span.set_status(Status(StatusCode.ERROR, "identity_no_op_cleanup_failed"))
                raise InvalidEvidenceError("Phase 5 no-op hook cleanup failed")
            signals.record(span, operation="identity_no_op", outcome="completed", reason="identity_no_op_hooks_removed")


def _persist_phase5_conformance(
    output: Path, evidence: dict[str, Any], trace_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_path = output / "conformance" / "evidence.json"
    encoded = encode_json(evidence)
    FrozenRunStore._atomic_write(evidence_path, encoded)
    artifacts = [{
        "path": "evidence.json", "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }]
    if trace_evidence is not None:
        trace_path = evidence_path.parent / "trace.json"
        trace_encoded = encode_json(trace_evidence, max_bytes=MAX_TRACE_BYTES)
        FrozenRunStore._atomic_write(trace_path, trace_encoded)
        artifacts.append({
            "path": "trace.json", "bytes": len(trace_encoded),
            "sha256": hashlib.sha256(trace_encoded).hexdigest(),
        })
    manifest = {
        "schema_version": "capability-anatomy/phase5-conformance-manifest/v1",
        "artifacts": artifacts,
    }
    FrozenRunStore._atomic_write(
        evidence_path.parent / "manifest.json", encode_json(manifest),
    )
    return manifest


def _serialize_phase5_trace(
    exporter: InMemorySpanExporter, reader: InMemoryMetricReader, trace_id: str | None, *, diagnostic: bool = False,
) -> dict[str, Any]:
    finished = exporter.diagnostic_spans() if diagnostic and hasattr(exporter, "diagnostic_spans") else exporter.get_finished_spans()
    spans = [span for span in finished if trace_id is None or f"{span.context.trace_id:032x}" == trace_id]
    metrics_data = reader.get_metrics_data()
    return {
        "schema_version": "capability-anatomy/phase5-trace/v2",
        "trace_id": trace_id,
        "complete": not getattr(exporter, "overflow", False),
        "dropped_spans": getattr(exporter, "dropped_spans", 0),
        "spans": [
            {
                "name": span.name,
                "trace_id": f"{span.context.trace_id:032x}",
                "start_time_unix_nano": span.start_time,
                "end_time_unix_nano": span.end_time,
                "duration_ns": span.end_time - span.start_time,
                "span_id": f"{span.context.span_id:016x}",
                "parent_span_id": (
                    f"{span.parent.span_id:016x}" if span.parent is not None else None
                ),
                "status": span.status.status_code.name,
                "links": [{"trace_id": f"{link.context.trace_id:032x}", "span_id": f"{link.context.span_id:016x}"} for link in span.links],
                "attributes": to_primitive(dict(span.attributes or {})),
                "events": [
                    {"name": event.name, "timestamp_unix_nano": event.timestamp, "attributes": to_primitive(dict(event.attributes or {}))}
                    for event in span.events
                ],
            }
            for span in spans
        ],
        "metrics": [
            {
                "name": metric.name,
                "points": [
                    {
                        "attributes": to_primitive(dict(point.attributes)),
                        "value": point.value,
                    }
                    for point in metric.data.data_points
                ],
            }
            for resource in metrics_data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        ],
    }


def _persist_phase5_failure_trace(
    output: Path, trace_id: str, trace_evidence: Mapping[str, Any], error: BaseException,
) -> None:
    root = output / "trace-failures" / trace_id
    failure = {
        "schema_version": "capability-anatomy/phase5-failure/v1",
        "status": "failed",
        "trace_id": trace_id,
        "error_type": type(error).__name__,
    }
    encoded_failure = encode_json(failure)
    encoded_trace = encode_json(trace_evidence, max_bytes=MAX_TRACE_BYTES)
    FrozenRunStore._atomic_write(root / "failure.json", encoded_failure)
    FrozenRunStore._atomic_write(root / "trace.json", encoded_trace)
    FrozenRunStore._atomic_write(
        root / "manifest.json",
        canonical_json_bytes({
            "schema_version": "capability-anatomy/phase5-failure-manifest/v1",
            "artifacts": [
                {"path": "failure.json", "bytes": len(encoded_failure), "sha256": hashlib.sha256(encoded_failure).hexdigest()},
                {"path": "trace.json", "bytes": len(encoded_trace), "sha256": hashlib.sha256(encoded_trace).hexdigest()},
            ],
        }),
    )


def _record_phase5_decision(
    telemetry: OperationTelemetry | None, *, operation: str, outcome: str, reason: str,
) -> None:
    if telemetry is None:
        return
    with telemetry.tracer.start_as_current_span(
        f"capability_anatomy.phase5.{operation}",
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        telemetry.record(span, operation=operation, outcome=outcome, reason=reason)
        if outcome in {"failed", "refused"}:
            span.set_status(Status(StatusCode.ERROR, reason))


def _governed_source_identity(
    config_path: Path,
    approved_commit: str,
    config: Any,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    implementation_path: Path | None = None,
) -> tuple[str, str]:
    try:
        repo_root_text = subprocess.run(
            ("git", "-C", str(config_path.parent), "rev-parse", "--show-toplevel"),
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        repo_root = Path(repo_root_text).resolve()
        lab_root = config_path.resolve().parents[2]
        implementation_root = (implementation_path or Path(__file__)).resolve().parents[3]
        if (
            lab_root != implementation_root
            or config_path.resolve().parent != lab_root / "configs/experiments"
        ):
            raise GateAuthorizationError("gate_source_identity_changed")
        relative_root = lab_root.relative_to(repo_root).as_posix()
        lab_prefix = "" if relative_root == "." else relative_root + "/"
        selected_paths = (
            config_path.resolve(),
            protocol_path.resolve(),
            (config_path.parent / config.dataset.manifest).resolve(),
            *(
                (protocol_path.parent / protocol["dataset"][key]["path"]).resolve()
                for key in ("manifest", "record_plan", "prompt_templates", "scorers")
            ),
        )
        selected = tuple(path.relative_to(repo_root).as_posix() for path in selected_paths)
        governed = tuple(dict.fromkeys((
            f"{lab_prefix}src/capability_anatomy",
            f"{lab_prefix}schemas",
            f"{lab_prefix}pyproject.toml",
            f"{lab_prefix}uv.lock",
            *selected,
        )))
        subprocess.run(
            ("git", "-C", str(repo_root), "cat-file", "-e", f"{approved_commit}^{{commit}}"),
            check=True, capture_output=True,
        )
        subprocess.run(
            ("git", "-C", str(repo_root), "merge-base", "--is-ancestor", approved_commit, "HEAD"),
            check=True, capture_output=True,
        )
        dirty = subprocess.run(
            ("git", "-C", str(repo_root), "status", "--porcelain=v1", "--", *governed),
            check=True, capture_output=True, text=True,
        ).stdout
        if dirty:
            raise GateAuthorizationError("gate_source_identity_changed")
        approved_paths = subprocess.run(
            ("git", "-C", str(repo_root), "ls-tree", "-r", "--name-only", approved_commit, "--", *governed),
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        current_paths = subprocess.run(
            ("git", "-C", str(repo_root), "ls-files", "--", *governed),
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        if approved_paths != current_paths or not approved_paths:
            raise GateAuthorizationError("gate_source_identity_changed")
        approved_digest = hashlib.sha256()
        current_digest = hashlib.sha256()
        for relative in approved_paths:
            approved_bytes = subprocess.run(
                ("git", "-C", str(repo_root), "show", f"{approved_commit}:{relative}"),
                check=True, capture_output=True,
            ).stdout
            current_bytes = (repo_root / relative).read_bytes()
            for digest, payload in ((approved_digest, approved_bytes), (current_digest, current_bytes)):
                digest.update(relative.encode("utf-8") + b"\0")
                digest.update(len(payload).to_bytes(8, "big") + payload)
        return approved_digest.hexdigest(), current_digest.hexdigest()
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as error:
        raise GateAuthorizationError("gate_approved_source_unavailable") from error


def _validate_phase5_plugin_contract(
    protocol: dict[str, Any], resolved: dict[str, ResolvedPlugin]
) -> dict[str, dict[str, Any]]:
    plugins = {item["role"]: item for item in protocol["plugins"]}
    mismatch = any(
        plugins[role]["name"] != item.plugin.name
        or plugins[role]["version"] != item.plugin.version
        or plugins[role]["api_version"] != item.plugin.api_version
        or set(plugins[role]["capabilities"]) != set(item.plugin.capabilities)
        for role, item in resolved.items()
    )
    if mismatch:
        raise InvalidEvidenceError("Phase 5 plugin contract disagrees with installed plugins")
    return plugins


def _validate_phase5_model_contract(protocol: dict[str, Any], config: Any, adapter: Any) -> None:
    if (
        "model.architecture_profile" not in adapter.capabilities
        or not callable(getattr(adapter, "profile_identity", None))
    ):
        raise InvalidEvidenceError("Phase 5 model plugin lacks profile identity capability")
    profile_id, profile_sha256 = adapter.profile_identity(config.model)
    if (
        protocol["model"]["plugin"] != adapter.name
        or protocol["model"]["version"] != adapter.version
        or protocol["model"]["source"] != config.model.source
        or protocol["model"]["revision"] != config.model.revision
        or protocol["model"]["architecture_profile"] != profile_id
        or protocol["model"]["architecture_profile_sha256"] != profile_sha256
    ):
        raise InvalidEvidenceError("Phase 5 model contract disagrees with installed adapter")


def _check_gate5a_authorization(
    config_path: Path, config: Any, protocol_path: Path, *, conformance_only: bool,
) -> dict[str, Any]:
    policy_path = config.policy_file
    if policy_path is None:
        raise GateAuthorizationError("gate_authorization_invalid")
    if not policy_path.is_absolute():
        policy_path = (config_path.parent / policy_path).resolve()
    try:
        authorization = load_mapping(policy_path, format="json")
    except (OSError, UnicodeError, json.JSONDecodeError, AuthoredInputError) as error:
        raise GateAuthorizationError("gate_authorization_invalid") from error
    expected = set(AUTHORIZATION_FIELDS) | {"review_policy_sha256"}
    if set(authorization) != expected or authorization["protocol_sha256"] != sha256_file(protocol_path):
        raise GateAuthorizationError("gate_authorization_invalid")
    if (
        authorization["schema_version"] != AUTHORIZATION_V2
        or authorization["status"] != "approved"
        or (authorization["conformance_authorized"] is not True if conformance_only
            else authorization["full_scan_authorized"] is not True)
        or not isinstance(authorization["approved_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", authorization["approved_commit"])
        or not isinstance(authorization["approved_source_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", authorization["approved_source_sha256"])
        or not isinstance(authorization["review_url"], str)
    ):
        raise GateAuthorizationError("gate_authorization_invalid")
    protocol = validate_phase5_protocol(protocol_path)
    try:
        validate_review_reference(protocol, authorization)
    except InvalidEvidenceError as error:
        raise GateAuthorizationError("gate_authorization_invalid") from error
    approved_source, current_source = _governed_source_identity(
        config_path, authorization["approved_commit"], config, protocol_path, protocol,
    )
    if (
        approved_source != authorization["approved_source_sha256"]
        or current_source != approved_source
    ):
        raise GateAuthorizationError("gate_source_identity_changed")
    return dict(authorization)


def _require_gate5a_authorization(
    config_path: Path, config: Any, protocol_path: Path, *, conformance_only: bool,
    telemetry: OperationTelemetry | None = None,
) -> dict[str, Any]:
    signals = telemetry or OperationTelemetry.create("gate5a_authorization")
    with signals.tracer.start_as_current_span(
        "capability_anatomy.gate5a.authorization",
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            approved_authorization = _check_gate5a_authorization(
                config_path, config, protocol_path, conformance_only=conformance_only,
            )
        except InvalidEvidenceError as error:
            reason = error.reason
            signals.record(
                span, operation="authorize_phase5_operation", outcome="refused", reason=reason,
            )
            span.set_status(Status(StatusCode.ERROR, reason))
            raise
        signals.record(
            span, operation="authorize_phase5_operation", outcome="accepted",
            reason="conformance_authorized" if conformance_only else "full_scan_authorized",
        )
        return approved_authorization


def _run_experiment(
    config_path: Path, *, conformance_only: bool = False, config: Any = None,
    authorization_telemetry: OperationTelemetry | None = None,
    intervention_telemetry: OperationTelemetry | None = None,
    runtime_telemetry: OperationTelemetry | None = None,
    execution_telemetry: OperationTelemetry | None = None,
    discovery_telemetry: OperationTelemetry | None = None,
) -> dict[str, Any]:
    config = config if config is not None else load_experiment_config(config_path)
    record_seed = config.seed
    component_seed = config.seed
    discovery = PluginDiscovery(discovery_telemetry)
    resolved = {
        "evaluation": discovery.resolve("evaluation", config.capability.evaluation_plugin, required_capabilities=frozenset(), required_methods=("required_record_schema", "build_request", "parse", "score", "aggregate")),
        "dataset": discovery.resolve("dataset", config.dataset.provider, required_capabilities=frozenset(), required_methods=("from_manifest", "metadata", "records", "stable_id", "validate_for")),
        "intervention": discovery.resolve("intervention", config.intervention.plugin, required_capabilities=frozenset({"intervention.block_bypass", "intervention.scoped_cleanup"}), required_methods=("validate", "apply")),
        "runtime": discovery.resolve(
            "runtime", config.runtime.executor,
            required_capabilities=RUNTIME_OPERATION_CAPABILITIES,
            required_methods=("execute", "synchronize", "memory_bytes", "token_counts", "clock"),
        ),
    }
    request_capabilities = resolved["evaluation"].plugin.capabilities
    model_capabilities = {
        "measurement.memory", "measurement.synchronization", "measurement.tokens",
        "model.topology.components",
    }
    if "request.chat_generation" in request_capabilities:
        model_capabilities.add("execution.autoregressive_text")
    if "request.native_tools" in request_capabilities:
        model_capabilities.add("model.native_tool_prompt")
    if "request.perplexity" in request_capabilities:
        model_capabilities.add("execution.perplexity")
    if "evaluation.phase5" in request_capabilities:
        model_capabilities.add("model.architecture_profile")
    resolved["model"] = discovery.resolve(
        "model", config.model.plugin,
        required_capabilities=frozenset(model_capabilities),
        required_methods=("load", "topology", "execute", "provenance", "synchronize", "memory_bytes", "token_counts"),
    )
    adapter = resolved["model"].plugin
    suite = resolved["evaluation"].plugin
    provider_factory = resolved["dataset"].plugin
    intervention_provider = resolved["intervention"].plugin
    runtime_plugin = resolved["runtime"].plugin
    if not isinstance(runtime_plugin, RuntimePlugin):
        raise InvalidEvidenceError("runtime plugin does not implement the public runtime protocol")
    if config.credentials:
        credentials = RuntimeCredentials(config.credentials)
        consumers = [item.plugin for item in resolved.values() if callable(getattr(item.plugin, "configure_credentials", None))]
        if not consumers:
            raise InvalidEvidenceError("configured credentials require a credential-aware selected plugin")
        for consumer in consumers:
            consumer.configure_credentials(credentials)
    protocol_ref = config.runtime.parameters.get("phase5_protocol")
    if "evaluation.phase5" in suite.capabilities and protocol_ref is None:
        raise InvalidEvidenceError("Phase 5 execution requires a frozen protocol")
    provider = provider_factory.from_manifest((config_path.parent / config.dataset.manifest).resolve())
    if protocol_ref is not None:
        if not isinstance(protocol_ref, str) or not protocol_ref:
            raise InvalidEvidenceError("Phase 5 execution requires a frozen protocol")
        protocol_path = (config_path.parent / protocol_ref).resolve()
        protocol = validate_phase5_protocol(protocol_path)
        plugins = _validate_phase5_plugin_contract(protocol, resolved)
        _validate_phase5_model_contract(protocol, config, adapter)
        if getattr(suite, "max_new_tokens", None) != protocol["runtime"]["max_new_tokens"]:
            raise InvalidEvidenceError("Phase 5 evaluator generation limit disagrees with frozen protocol")
        approved_authorization = _require_gate5a_authorization(
            config_path, config, protocol_path, conformance_only=conformance_only,
            telemetry=authorization_telemetry,
        )
        if (
            protocol["experiment"]["id"] != config.experiment_id
            or protocol["experiment"]["seed"] != config.seed
            or plugins["model"]["name"] != config.model.plugin
            or plugins["evaluation"]["name"] != config.capability.evaluation_plugin
            or plugins["dataset"]["name"] != config.dataset.provider
            or plugins["intervention"]["name"] != config.intervention.plugin
            or plugins["runtime"]["name"] != config.runtime.executor
            or protocol["runtime"]["deterministic"] != config.runtime.deterministic
            or protocol["runtime"]["warmup_runs"] != config.runtime.warmup_runs
            or protocol["runtime"]["repetitions"] != config.runtime.repetitions
            or protocol["runtime"]["randomized_record_order"] != config.runtime.randomized_execution_order
            or protocol["runtime"]["randomized_component_order"] != config.runtime.randomized_execution_order
            or protocol["runtime"]["device"] != config.runtime.parameters.get("device")
            or protocol["runtime"]["dtype"] != config.model.parameters.get("dtype")
            or protocol["runtime"]["context_length"] != config.runtime.parameters.get("context_length")
            or protocol["runtime"]["record_seed_derivation"] != "sha256(seed:records)"
            or protocol["runtime"]["component_seed_derivation"] != "sha256(seed:components)"
        ):
            raise InvalidEvidenceError("Phase 5 config disagrees with its frozen protocol")
        record_seed = _phase5_seed(config.seed, "records")
        component_seed = _phase5_seed(config.seed, "components")
        if getattr(provider, "plan_sha256", None) != protocol["dataset"]["record_plan"]["sha256"]:
            raise InvalidEvidenceError("Phase 5 dataset plan disagrees with its frozen protocol")
    if not isinstance(adapter, MeasuredModelAdapter):
        raise InvalidEvidenceError("model plugin does not implement declared measurements")
    provider.validate_for(suite)
    if config.capability.suite_version != suite.version:
        raise InvalidEvidenceError("evaluation suite version is incompatible")
    loaded = adapter.load(config.model, config.runtime)
    provenance = adapter.provenance(loaded)
    topology = adapter.topology(loaded)
    components = tuple(item.id for item in topology.components)
    resolved_topology = {
        "architecture": topology.architecture,
        "components": [
            {
                "id": item.id,
                "kind": item.kind,
                "parent_id": item.parent_id,
                "order": item.order,
                "metadata": to_primitive(item.metadata),
            }
            for item in topology.components
        ],
    }
    configured = config.intervention.parameters.get("component_ids")
    if configured == "all":
        targets = components
    elif isinstance(configured, list) and all(isinstance(item, str) for item in configured):
        targets = tuple(configured)
    else:
        raise InvalidEvidenceError("intervention components must be all or a list of IDs")
    if not targets:
        raise InvalidEvidenceError("intervention has no components")
    if not set(targets) <= set(components):
        raise InvalidEvidenceError("intervention component is incompatible with model topology")

    if conformance_only:
        component = topology.component(targets[0])
        records = provider.records(config.dataset.discovery_partition)
        record = next((item for item in records if "request.chat_generation" in suite.capabilities), None)
        if record is None:
            raise InvalidEvidenceError("Phase 5 conformance has no generation record")
        request = ExecutionRequest(suite.build_request(record))
        before_hooks = len(component.implementation._forward_hooks)
        normal_calls = 0
        def count_normal(_module: Any, _args: tuple[Any, ...], _output: Any) -> None:
            nonlocal normal_calls
            normal_calls += 1
        normal_probe = component.implementation.register_forward_hook(count_normal)
        try:
            normal = _execute_phase5_conformance_runtime(
                runtime_plugin, adapter, loaded, request, runtime_telemetry,
            )
        finally:
            normal_probe.remove()
        bypass_calls = 0
        bypass_identity_calls = 0
        bypass_sequence_widths: list[int] = []
        with BlockBypass((component.id,), intervention_telemetry).apply(
            intervention_provider, adapter, loaded,
        ):
            def observe_bypass(_module: Any, args: tuple[Any, ...], output: Any) -> None:
                nonlocal bypass_calls, bypass_identity_calls
                bypass_calls += 1
                hidden = output[0] if isinstance(output, tuple) else output
                if args and hasattr(args[0], "shape") and len(args[0].shape) >= 2:
                    bypass_sequence_widths.append(int(args[0].shape[-2]))
                if args and hidden is args[0]:
                    bypass_identity_calls += 1
            bypass_probe = component.implementation.register_forward_hook(observe_bypass)
            try:
                bypassed = _execute_phase5_conformance_runtime(
                    runtime_plugin, adapter, loaded, request, runtime_telemetry,
                )
            finally:
                bypass_probe.remove()
        after_hooks = len(component.implementation._forward_hooks)
        restored_calls = 0
        def count_restored(_module: Any, _args: tuple[Any, ...], _output: Any) -> None:
            nonlocal restored_calls
            restored_calls += 1
        restored_probe = component.implementation.register_forward_hook(count_restored)
        try:
            restored = _execute_phase5_conformance_runtime(
                runtime_plugin, adapter, loaded, request, runtime_telemetry,
            )
        finally:
            restored_probe.remove()
        prefill_calls, decode_calls = _validate_phase5_conformance(
            before_hooks=before_hooks, after_hooks=after_hooks, normal_calls=normal_calls,
            restored_calls=restored_calls, bypass_calls=bypass_calls,
            bypass_identity_calls=bypass_identity_calls, sequence_widths=bypass_sequence_widths,
            normal=normal, restored=restored,
        )
        evidence = {
            "schema_version": "capability-anatomy/phase5-conformance/v1",
            "status": "complete",
            "kind": "phase5_bounded_bypass_conformance",
            "component_id": component.id,
            "record_id": record.id,
            "normal_output_sha256": hashlib.sha256(str(normal).encode()).hexdigest(),
            "bypassed_output_sha256": hashlib.sha256(str(bypassed).encode()).hexdigest(),
            "restored_output_sha256": hashlib.sha256(str(restored).encode()).hexdigest(),
            "hook_count_before": before_hooks,
            "hook_count_after": after_hooks,
            "normal_invocations": normal_calls,
            "bypass_invocations": bypass_calls,
            "bypass_identity_invocations": bypass_identity_calls,
            "prefill_invocations": prefill_calls,
            "decode_invocations": decode_calls,
            "invocation_sequence_widths": bypass_sequence_widths,
            "restored_invocations": restored_calls,
            "prefill_decode_verified": True,
            "cleanup_verified": True,
            "model": to_primitive(provenance),
            "protocol_sha256": sha256_file(protocol_path),
        }
        output = config.output.directory
        if not output.is_absolute():
            output = Path(os.path.abspath(config_path.parent / output))
        evidence["_output_directory"] = str(output)
        evidence["_experiment_id"] = config.experiment_id
        return evidence

    compatibility = {
        "model": to_primitive(provenance),
        "resolved_topology": resolved_topology,
        "evaluation": {"plugin": suite.name, "version": suite.version},
        "dataset": to_primitive(provider.metadata()),
        "intervention": {"plugin": intervention_provider.name, "version": intervention_provider.version},
        "discovered_plugins": [resolved[role].identity() for role in sorted(resolved)],
        "operation_capabilities": {
            "model": sorted(model_capabilities),
            "evaluation": sorted(capability for capability in request_capabilities if capability.startswith("request.")),
            "intervention": ["intervention.block_bypass", "intervention.scoped_cleanup"],
            "runtime": sorted(RUNTIME_OPERATION_CAPABILITIES),
        },
    }
    output = config.output.directory
    if not output.is_absolute():
        output = Path(os.path.abspath(config_path.parent / output))
    store = FrozenRunStore(output, to_primitive(config), compatibility)
    evaluator = EvaluationExecutor()
    controls = MeasurementControls(
        seed=record_seed,
        warmup_runs=config.runtime.warmup_runs,
        repetitions=config.runtime.repetitions,
        randomized_order=config.runtime.randomized_execution_order,
        synchronize=lambda: runtime_plugin.synchronize(adapter, loaded),
        memory_bytes=lambda: runtime_plugin.memory_bytes(adapter, loaded),
        clock=runtime_plugin.clock,
    )
    store.initialize()
    component_controls = MeasurementControls(
        seed=component_seed,
        warmup_runs=0,
        repetitions=1,
        randomized_order=config.runtime.randomized_execution_order,
    )
    component_scan_order = store.freeze_sequence("component-scan-order", component_controls.order(targets))

    def evaluate(
        partition: str,
        component_ids: tuple[str, ...] = (),
        *,
        condition: str = "baseline",
        no_op_component: str | None = None,
    ) -> dict[str, Any]:
        if no_op_component is not None:
            context = _identity_no_op_hook(
                topology.component(no_op_component), intervention_telemetry,
            )
        else:
            context = (
                BlockBypass(component_ids, intervention_telemetry).apply(
                    intervention_provider, adapter, loaded,
                )
                if component_ids else nullcontext()
            )
        with context:
            observations = []
            prompts = {}
            execution_order = controls.order(provider.records(partition))
            for record in execution_order:
                capture = _TRACE_CAPTURE.get()
                if capture is not None and capture[0].overflow:
                    from ..run_telemetry import TelemetryCapacityError
                    raise TelemetryCapacityError("telemetry capacity exceeded before the next record")
                with identity_scope(record=record.id):
                    request = ExecutionRequest(suite.build_request(record))
                    prompts[record.id] = request.payload
                    try:
                        measurements = controls.measure(
                            lambda: _execute_phase5_conformance_runtime(
                                runtime_plugin, adapter, loaded, request, runtime_telemetry,
                            ),
                            lambda result: runtime_plugin.token_counts(adapter, loaded, request, result),
                        )
                    except Exception as error:
                        observations.append(evaluator.execution_failure(record, type(error).__name__))
                        continue
                    for repetition, measurement in enumerate(measurements):
                        observation = evaluator.evaluate(suite, record, measurement.value)
                        observations.append(
                            replace(
                                observation,
                                repetition=repetition,
                                elapsed_seconds=measurement.elapsed_seconds,
                                peak_memory_bytes=measurement.peak_memory_bytes,
                                input_tokens=measurement.input_tokens,
                                output_tokens=measurement.output_tokens,
                            )
                        )
        aggregate = suite.aggregate(observations)
        required = set(config.capability.target_metrics) | set(config.capability.collateral_metrics)
        missing = required - set(aggregate["metrics"])
        payload = {
            "partition": partition,
            "components": list(component_ids),
            "execution_order": [record.id for record in execution_order],
            "measurement_controls": {
                "seed": record_seed,
                "warmup_runs": config.runtime.warmup_runs,
                "repetitions": config.runtime.repetitions,
                "randomized_order": config.runtime.randomized_execution_order,
            },
            "observations": [],
            "metrics": aggregate,
            "provenance": to_primitive(provenance),
            "failures": [item.error_type for item in observations if item.error_type],
            "complete": not missing and not aggregate["errors"],
            "missing_metrics": sorted(missing),
        }
        for item in observations:
            serialized = serialize_observation(
                    item,
                    retain_raw_output=config.output.retain_raw_outputs,
                    prompt=prompts.get(item.example_id),
                    retain_prompt=config.output.retain_prompts,
                    prompt_storage=config.output.prompt_storage,
                )
            if protocol_ref is not None and not config.output.retain_raw_outputs:
                serialized.pop("raw_output", None)
                serialized["output_sha256"] = canonical_sha256(item.raw_output)
            if protocol_ref is not None:
                serialized.pop("prompt", None)
                serialized["prompt_sha256"] = canonical_sha256(prompts.get(item.example_id))
                serialized["memory_observation_bytes"] = serialized.pop("peak_memory_bytes")
                serialized["expected_metric_ids"] = list(_PHASE5_METRICS_BY_KIND[next(
                    record.metadata["kind"] for record in execution_order
                    if record.id == item.example_id
                )])
            serialized.update({
                "condition": condition,
                "component_id": component_ids[0] if component_ids else no_op_component or "baseline",
                "group_id": next(
                    record.metadata.get("group_id", record.id) for record in execution_order
                    if record.id == item.example_id
                ),
            })
            payload["observations"].append(serialized)
        return payload

    if protocol_ref is None:
        tasks = [Task("baseline", "baseline", lambda: evaluate(config.dataset.discovery_partition))]
        tasks.extend(Task(f"scan.{component.replace('.', '-')}", "scan", lambda component=component: evaluate(config.dataset.discovery_partition, (component,))) for component in component_scan_order)
        results = ExperimentRunner(store, trace_checkpoint=lambda task_id, payload: _checkpoint_task_trace(output, task_id, payload)).run(
            tuple(tasks), frozen_plan_reason="component_scan_order_frozen"
        )
        return {
            "component_scan_order": list(component_scan_order),
            "run_directory": str(output),
            "tasks": sorted(results),
            "status": "complete",
        }

    campaign_plan = build_campaign_plan(
        component_scan_order,
        protocol["controls"],
        seed=_phase5_seed(config.seed, "controls"),
    )
    store.freeze_payload("discovery-task-plan", serialize_campaign_plan(campaign_plan))
    _record_phase5_decision(
        execution_telemetry, operation="freeze_discovery_plan", outcome="accepted",
        reason="result_independent_campaign_plan_persisted",
    )

    def execute_campaign_task(task: Any, partition: str) -> dict[str, Any]:
        if task.condition == "baseline":
            return evaluate(partition, condition=task.id)
        if task.condition == "no_op":
            return evaluate(
                partition, condition=task.id, no_op_component=component_scan_order[0],
            )
        return evaluate(
            partition, (task.component_id,), condition=task.id,
        )

    runner = ExperimentRunner(
        store,
        trace_checkpoint=lambda task_id, payload: _checkpoint_task_trace(output, task_id, payload),
        telemetry=execution_telemetry,
        max_wall_seconds=protocol["budgets"]["max_wall_seconds"],
        max_memory_observation_bytes=protocol["budgets"]["max_memory_observation_bytes"],
        max_task_retries=protocol["budgets"]["max_task_retries"],
        max_observation_errors=protocol["budgets"]["max_observation_errors"],
        clock=runtime_plugin.clock,
        memory_bytes=lambda: runtime_plugin.memory_bytes(adapter, loaded),
    )
    tasks = tuple(
        Task(
            task.id,
            task.stage,
            lambda task=task: execute_campaign_task(task, config.dataset.discovery_partition),
        )
        for task in campaign_plan
    )
    results = runner.run(tasks, frozen_plan_reason="phase5_campaign_plan_frozen")

    required_metrics = tuple(protocol["metrics"])
    aggregated = {
        task_id: aggregate_observations(payload["observations"], required_metrics)
        for task_id, payload in results.items()
    }
    measurements = {
        task_id: summarize_measurements(payload["observations"])
        for task_id, payload in results.items()
    }
    baseline_reference, drift = aggregate_baseline_controls(
        {
            position: aggregated[f"baseline.{position}"]
            for position in ("beginning", "middle", "end")
        },
        aggregated["control.no-op"],
    )
    if any(value > protocol["candidate_rule"]["minimum_drift_margin"] for value in drift.values()):
        store.record_state("stopped", "control_drift")
        _record_phase5_decision(
            execution_telemetry, operation="evaluate_controls", outcome="refused",
            reason="control_drift",
        )
        raise InvalidEvidenceError("Phase 5 stop rule triggered: control_drift")
    _record_phase5_decision(
        execution_telemetry, operation="evaluate_controls", outcome="accepted",
        reason="repeated_baseline_no_op_and_matched_controls_complete",
    )
    scan_metrics = {
        component: aggregated[f"scan.{component}"] for component in component_scan_order
    }
    damage = compute_damage_matrix(baseline_reference, scan_metrics, protocol["metrics"])
    random_components = [task.component_id for task in campaign_plan if task.condition == "matched_random"]
    random_damage_matrix = compute_damage_matrix(
        baseline_reference,
        {component: aggregated[f"control.random.{component}"] for component in random_components},
        protocol["metrics"],
    )
    random_damage = {
        component: {
            metric: values["absolute_damage"] for metric, values in metrics.items()
        }
        for component, metrics in random_damage_matrix.items()
    }
    ranking = select_discovery_candidates(
        damage,
        protocol["metrics"],
        protocol["candidate_rule"],
        drift_by_metric=drift,
        matched_random_damage=random_damage,
    )
    store.freeze_payload("discovery-ranking", ranking)
    _record_phase5_decision(
        execution_telemetry, operation="freeze_discovery_ranking", outcome="accepted",
        reason="ranking_derived_from_discovery_only",
    )

    candidates = tuple(ranking["candidates"])
    validation_plan = [
        {"id": "validation.baseline", "condition": "baseline", "component_id": None}
    ] + [
        {"id": f"validation.{component}", "condition": "bypass", "component_id": component}
        for component in candidates
    ]
    store.freeze_payload("validation-task-plan", validation_plan)
    _record_phase5_decision(
        execution_telemetry, operation="open_validation_once", outcome="accepted",
        reason="discovery_ranking_frozen_before_validation",
    )
    validation_tasks = [
        Task(
            "validation.baseline",
            "baseline",
            lambda: evaluate(config.dataset.validation_partition, condition="validation.baseline"),
        )
    ]
    validation_tasks.extend(
        Task(
            f"validation.{component}",
            "validation",
            lambda component=component: evaluate(
                config.dataset.validation_partition,
                (component,),
                condition=f"validation.{component}",
            ),
        )
        for component in candidates
    )
    validation_results = runner.run(
        tuple(validation_tasks), frozen_plan_reason="phase5_validation_plan_frozen",
    )
    validation_baseline = aggregate_observations(
        validation_results["validation.baseline"]["observations"], required_metrics,
    )
    validation_conditions = {
        component: aggregate_observations(
            validation_results[f"validation.{component}"]["observations"], required_metrics,
        )
        for component in candidates
    }
    validation_damage = compute_damage_matrix(
        validation_baseline, validation_conditions, protocol["metrics"],
    )
    validation = compute_validation_results(
        candidates, validation_damage, protocol["metrics"], protocol["candidate_rule"],
    )
    validation_measurements = {
        task_id: summarize_measurements(payload["observations"])
        for task_id, payload in validation_results.items()
    }
    validation_aggregated = {
        "validation.baseline": validation_baseline,
        **{
            f"validation.{component}": value
            for component, value in validation_conditions.items()
        },
    }
    all_results = {**results, **validation_results}
    observations = [
        observation
        for task_id in sorted(all_results)
        for observation in all_results[task_id]["observations"]
    ]
    report_payload = {
        "status": "complete",
        "objective": "Measure temporary single-block sensitivity for tool calling and collateral capabilities under the frozen Phase 5 protocol.",
        "claim_boundary": protocol["claim_boundary"],
        "provenance": to_primitive(provenance),
        "baseline": baseline_reference,
        "per_component_damage": damage,
        "controls": {"drift": drift, "matched_random_damage": random_damage},
        "candidate_ranking": ranking["ranking"],
        "measurements": {**measurements, **validation_measurements},
        "candidates": ranking["candidates"],
        "validation": validation,
        "limitations": [
            "Memory values are synchronized observation-boundary allocations, not allocator peaks.",
            "Temporary single-block bypass does not establish that a block is unnecessary or safely removable.",
        ],
        "reproduction_commands": [
            f"uv run capability-anatomy run --config {_experiment_relative_path(config_path)}",
            f"uv run capability-anatomy report --evidence {_experiment_relative_path(output)} --format markdown",
        ],
    }
    historical_failures = []
    failures_root = output / "failures"
    if failures_root.exists():
        for failure_path in sorted(failures_root.glob("*.attempt-*.json")):
            historical_failures.append(parse_mapping(read_bytes(failure_path, max_bytes=MAX_ARTIFACT_BYTES)))
    report_payload["errors"] = historical_failures
    record_plan_path = (protocol_path.parent / protocol["dataset"]["record_plan"]["path"]).resolve()
    record_plan_bytes = read_regular_bytes(record_plan_path, max_bytes=MAX_ARTIFACT_BYTES)
    FrozenRunStore._atomic_write(output / "record-plan.json", record_plan_bytes)
    FrozenRunStore._atomic_write(output / "protocol.json", read_regular_bytes(protocol_path, max_bytes=MAX_ARTIFACT_BYTES))
    artifacts = {
        "configuration.json": to_primitive(config),
        "compatibility.json": compatibility,
        "topology.json": resolved_topology,
        "component-scan-order.json": list(component_scan_order),
        "observations.jsonl": observations,
        "metrics.json": {
            "baseline": baseline_reference,
            "per_condition": {**aggregated, **validation_aggregated},
            "damage_matrix": damage,
            "validation_damage": validation_damage,
            "measurements": {**measurements, **validation_measurements},
        },
        "failures.jsonl": historical_failures,
        "controls.json": {
            "plan": serialize_campaign_plan(campaign_plan),
            "drift": drift,
            "matched_random_damage": random_damage,
        },
        "provenance.json": to_primitive(provenance),
        "candidates.json": ranking,
        "validation.json": validation,
        "report.json": report_payload,
        "report.md": render_report_markdown(report_payload),
    }
    manifest_identity = {
        "config_sha256": store.config_sha256,
        "compatibility_sha256": store.compatibility_sha256,
        "protocol_sha256": sha256_file(protocol_path),
        "model_revision": config.model.revision,
        "dataset_sha256": protocol["dataset"]["manifest"]["sha256"],
        "prompt_sha256": protocol["dataset"]["prompt_templates"]["sha256"],
        "scorer_sha256": protocol["dataset"]["scorers"]["sha256"],
        "plugins": [resolved[role].identity() for role in sorted(resolved)],
        "runtime": protocol["runtime"],
        "execution_state": "complete",
        "metric_ids": sorted(protocol["metrics"]),
        "authorization": approved_authorization,
    }
    write_final_artifacts(
        output,
        artifacts,
        required_paths=[
            path for path in protocol["required_artifacts"] if path != "trace.json"
        ],
        identity=manifest_identity,
    )
    _record_phase5_decision(
        execution_telemetry, operation="finalize_evidence", outcome="accepted",
        reason="required_artifacts_atomically_written_and_digest_bound",
    )
    return {
        "component_scan_order": list(component_scan_order),
        "run_directory": str(output),
        "tasks": sorted(all_results),
        "candidates": list(candidates),
        "status": "complete",
        "_manifest_identity": manifest_identity,
        "_required_artifacts": protocol["required_artifacts"],
        "_experiment_id": config.experiment_id,
    }


@contextmanager
def _termination_scope():
    # Python signals remain cooperative. During the bounded post-lease drain,
    # retain termination intent until its diagnostic receipt has been committed.
    state = {"defer": False, "requested": False}
    if threading.current_thread() is not threading.main_thread():
        yield state
        return
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(_signum, _frame):
        state["requested"] = True
        if not state["defer"]:
            raise KeyboardInterrupt()
    for number in previous:
        signal.signal(number, interrupted)
    try:
        yield state
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def run_experiment(config_path: Path, *, conformance_only: bool = False) -> dict[str, Any]:
    with credential_scope():
        return _run_invocation(config_path, conformance_only=conformance_only)


def _prior_trace_segments(output: Path):
    from .evidence_inventory import inventory_paths
    from .core_evidence import CoreEvidenceError
    from .evidence_limits import MAX_ARTIFACTS, MAX_BUNDLE_BYTES
    paths = inventory_paths(output, max_entries=MAX_ARTIFACTS, max_depth=16, error=CoreEvidenceError)
    remaining = MAX_BUNDLE_BYTES
    segments = []
    for relative in paths:
        if relative != "trace.json" and not (relative.startswith("trace-failures/") and relative.endswith("/trace.json")):
            continue
        payload = read_bytes(output / relative, max_bytes=min(MAX_TRACE_BYTES, remaining))
        remaining -= len(payload)
        document = parse_mapping(payload, max_bytes=MAX_TRACE_BYTES)
        prior = document.pop("prior_segments", [])
        if not isinstance(prior, list) or any(not isinstance(segment, dict) for segment in prior):
            raise CoreEvidenceError("evidence_trace_missing")
        segments.extend(prior)
        segments.append(document)
    return segments


def _require_phase5_resume_trace(output: Path):
    """A cached scientific task needs its durable completed execution segment."""
    from .evidence_inventory import inventory_paths
    from .core_evidence import CoreEvidenceError
    from .evidence_limits import MAX_ARTIFACTS, MAX_BUNDLE_BYTES
    signals = OperationTelemetry.create("phase5_resume")
    with signals.tracer.start_as_current_span("capability_anatomy.phase5.resume_trace", record_exception=False,
                                              set_status_on_exception=False) as span:
        try:
            paths = inventory_paths(output, max_entries=MAX_ARTIFACTS, max_depth=16, error=CoreEvidenceError)
            tasks = {path.removeprefix("tasks/").removesuffix(".complete.json") for path in paths
                     if path.startswith("tasks/") and path.endswith(".complete.json")}
            committed = set()
            if tasks:
                for segment in _prior_trace_segments(output):
                    from .core_evidence import trace_completeness
                    trace_completeness(segment)
                    records = segment.get("spans")
                    if not isinstance(records, list) or any(not isinstance(record, dict) or not isinstance(record.get("attributes"), dict) for record in records):
                        raise CoreEvidenceError("evidence_trace_missing")
                    for record in records:
                        attrs = record["attributes"]
                        if (record.get("name") == "capability_anatomy.execution.task" and
                            record.get("status") in {"UNSET", "OK"} and
                            attrs.get("capability_anatomy.reason") == "task_committed"):
                            committed.add(attrs.get("capability_anatomy.task_id_sha256"))
                if not {hashlib.sha256(task.encode()).hexdigest() for task in tasks} <= committed:
                    error = InvalidEvidenceError("cached Phase 5 tasks lack a durable completion trace; preserve this run and start a new output directory")
                    error.reason = "phase5_resume_trace_missing"
                    raise error
        except BaseException as error:
            reason = getattr(error, "reason", "phase5_resume_trace_invalid")
            signals.record(span, operation="resume_phase5", outcome="refused", reason=reason)
            span.set_status(Status(StatusCode.ERROR, reason))
            raise
        signals.record(span, operation="resume_phase5", outcome="accepted", reason="phase5_resume_trace_available")


def _local_delivery_failure(root, trace_id, *, reason="telemetry_delivery_incomplete"):
    """Record failed terminal acknowledgement without exporting recursively."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    tracer = provider.get_tracer("capability_anatomy.delivery_diagnostic")
    signals = OperationTelemetry.create("telemetry_delivery", tracer, meter_provider.get_meter("capability_anatomy.delivery_diagnostic"))
    try:
        with otel_trace.use_span(root, end_on_exit=False), tracer.start_as_current_span(
            "capability_anatomy.telemetry.delivery", record_exception=False, set_status_on_exception=False,
        ) as span:
            span.set_attribute("capability_anatomy.export_scope", "local_only_delivery_failure")
            signals.record(span, operation="confirm_delivery", outcome="failed", reason=reason)
            span.set_status(Status(StatusCode.ERROR, reason))
        return _serialize_phase5_trace(exporter, reader, trace_id)
    finally:
        provider.shutdown()
        meter_provider.shutdown()


def _run_invocation(config_path: Path, *, conformance_only: bool = False) -> dict[str, Any]:
    from ..run_telemetry import BoundedSpanExporter, configure_otlp, sdk_disabled, TelemetryDeliveryError
    if sdk_disabled():
        error = InvalidEvidenceError("evidence-producing commands require telemetry; unset OTEL_SDK_DISABLED")
        error.reason = "telemetry_disabled"
        raise error
    exporter = BoundedSpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    readers = [reader]
    transport = configure_otlp(tracer_provider, readers)
    meter_provider = MeterProvider(metric_readers=readers)
    tracer = tracer_provider.get_tracer("capability_anatomy.invocation")
    meter = meter_provider.get_meter("capability_anatomy.invocation")
    output = None
    config = None
    invocation_id = uuid.uuid4().hex
    result = None
    failure = None
    trace_id = None
    delivery = {"configured": False, "status": "not_configured"}
    try:
        with telemetry_scope(tracer, meter), _termination_scope() as termination:
            signals = OperationTelemetry.create("invocation", tracer, meter)
            with tracer.start_as_current_span("capability_anatomy.invocation", record_exception=False,
                                              set_status_on_exception=False) as root:
                trace_id = f"{root.get_span_context().trace_id:032x}"
                root.set_attribute("capability_anatomy.outcome_scope", "evidence_publication_before_terminal_export")
                token = _TRACE_CAPTURE.set((exporter, reader, trace_id))
                try:
                    config = load_experiment_config(config_path)
                    root.set_attribute("capability_anatomy.experiment_id_sha256", hashlib.sha256(config.experiment_id.encode()).hexdigest())
                    output = Path(os.path.abspath(config.output.directory if config.output.directory.is_absolute() else config_path.parent / config.output.directory))
                    with identity_scope(experiment=config.experiment_id), run_ownership(output):
                        if "phase5_protocol" in config.runtime.parameters and not conformance_only:
                            _require_phase5_resume_trace(output)
                        result = _run_observed(config_path, config, conformance_only, exporter, reader, tracer_provider, meter_provider, tracer, meter, invocation_span_id=f"{root.get_span_context().span_id:016x}")
                except BaseException as error:
                    failure = InterruptedRunError("run interrupted with resumable evidence") if isinstance(error, KeyboardInterrupt) else error
                finally:
                    termination["defer"] = True
                    _TRACE_CAPTURE.reset(token)
                # Network draining is outside the run lease. One invocation root
                # remains open through finalization and transport outcome.
                if transport is not None:
                    delivery = transport.flush()
                    if delivery["status"] != "delivered" and failure is None:
                        failure = TelemetryDeliveryError("OTLP delivery incomplete; local evidence is retained")
                if termination["requested"] and (failure is None or isinstance(failure, TelemetryDeliveryError)):
                    failure = InterruptedRunError("invocation interrupted after local evidence publication")
                    failure.reason = "delivery_interrupted"
                reason = "invocation_failed" if failure else "evidence_publication_complete"
                signals.record(root, operation="publish_evidence", outcome="failed" if failure else "completed", reason=reason)
                if failure:
                    root.set_status(Status(StatusCode.ERROR, getattr(failure, "reason", "invocation_failed")))
            if transport is not None:
                delivery = transport.flush()
                if delivery["status"] != "delivered" and failure is None:
                    failure = TelemetryDeliveryError("OTLP delivery incomplete; local evidence is retained")
            if termination["requested"] and (failure is None or isinstance(failure, TelemetryDeliveryError)):
                failure = InterruptedRunError("invocation interrupted after local evidence publication")
                failure.reason = "delivery_interrupted"
            receipt_path = None
            if output is not None:
                receipt = _serialize_phase5_trace(exporter, reader, trace_id, diagnostic=True)
                if delivery["status"] == "incomplete" or termination["requested"]:
                    diagnostic = _local_delivery_failure(root, trace_id, reason="delivery_interrupted" if termination["requested"] else "telemetry_delivery_incomplete")
                    receipt["spans"].extend(diagnostic["spans"])
                    receipt["metrics"].extend(diagnostic["metrics"])
                receipt["final_delivery_outcome"] = {
                    "status": delivery["status"],
                    "reason": "telemetry_delivery_incomplete" if delivery["status"] == "incomplete" else "telemetry_delivery_complete" if delivery["status"] == "delivered" else "telemetry_remote_not_configured",
                    "scope": "configured pipeline through terminal invocation export; excludes local delivery diagnostic and receipt write",
                }
                receipt.update(schema_version="capability-anatomy/invocation-trace/v2", invocation_id=invocation_id,
                               status="interrupted" if isinstance(failure, InterruptedRunError) else "failed" if failure else "complete", transport=delivery,
                               evidence_status="complete" if result is not None else "incomplete",
                               interruption_requested=termination["requested"],
                               scope="ownership and evidence publication, followed by explicit terminal delivery accounting; excludes receipt write")
                if failure:
                    receipt["failure_reason"] = getattr(failure, "reason", "runtime_failure")
                receipt_path = output.parent / (output.name + ".invocations") / invocation_id / "trace.json"
                try:
                    with otel_trace.use_span(root, end_on_exit=False):
                        FrozenRunStore._atomic_write(receipt_path, encode_json(receipt, max_bytes=MAX_TRACE_BYTES))
                        if termination["requested"] and not receipt["interruption_requested"]:
                            # One bounded correction covers a request delivered
                            # during receipt publication. Repeated requests keep
                            # the same intent; they never start a rewrite loop.
                            if failure is None or isinstance(failure, TelemetryDeliveryError):
                                failure = InterruptedRunError("invocation interrupted during receipt publication")
                                failure.reason = "delivery_interrupted"
                            receipt.update(status="interrupted", interruption_requested=True,
                                           interruption_phase="receipt_publication",
                                           failure_reason=getattr(failure, "reason", "interrupted"))
                            diagnostic = _local_delivery_failure(root, trace_id, reason="delivery_interrupted")
                            receipt["spans"].extend(diagnostic["spans"])
                            receipt["metrics"].extend(diagnostic["metrics"])
                            FrozenRunStore._atomic_write(receipt_path, encode_json(receipt, max_bytes=MAX_TRACE_BYTES))
                except Exception as error:
                    receipt_path = None
                    if failure is None:
                        failure = InvalidEvidenceError("invocation diagnostic receipt could not be persisted")
                        failure.reason = "invocation_receipt_unwritable"
            if termination["requested"] and (failure is None or isinstance(failure, TelemetryDeliveryError)):
                failure = InterruptedRunError("invocation interrupted after receipt snapshot")
                failure.reason = "delivery_interrupted"
            context = {"trace_id": trace_id, "evidence_status": "complete" if result is not None else "incomplete",
                       "delivery_status": delivery["status"], "interruption_requested": termination["requested"],
                       "receipt_status": "written" if receipt_path is not None else "unavailable"}
            if output is not None:
                context["run_directory"] = str(output)
            if receipt_path is not None:
                context["invocation_trace"] = str(receipt_path)
            if result is not None:
                result.update(context)
            if failure is not None:
                failure.result = result
                failure.context = context
                raise failure
            return result
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()


def _run_observed(config_path, config, conformance_only, exporter, reader, tracer_provider, meter_provider, tracer, meter, *, invocation_span_id):
    arguments = (config_path, config, conformance_only, exporter, reader, tracer_provider, meter_provider, tracer, meter)
    if "phase5_protocol" in config.runtime.parameters or conformance_only:
        return _run_observed_execution(*arguments, invocation_span_id=invocation_span_id)
    signals = OperationTelemetry.create("core_run", tracer, meter)
    with tracer.start_as_current_span("capability_anatomy.run", record_exception=False, set_status_on_exception=False) as span:
        span.set_attribute("capability_anatomy.outcome_scope", "execution_and_evidence_publication")
        try:
            result = _run_observed_execution(*arguments, invocation_span_id=invocation_span_id)
        except BaseException as error:
            reason = getattr(error, "reason", "run_publication_failed")
            signals.record(span, operation="run", outcome="failed", reason=reason)
            span.set_status(Status(StatusCode.ERROR, reason))
            raise
        signals.record(span, operation="run", outcome="completed", reason="evidence_publication_complete")
        return result


def _run_observed_execution(config_path, config, conformance_only, exporter, reader, tracer_provider, meter_provider, tracer, meter, *, invocation_span_id):
    phase5 = "phase5_protocol" in config.runtime.parameters
    operation = "bounded_phase5_conformance" if conformance_only else "phase5_full_scan" if phase5 else "core_run"
    component = "phase5_conformance" if conformance_only else "phase5_scan"
    root_name = (
        "capability_anatomy.phase5.conformance"
        if conformance_only else "capability_anatomy.phase5.scan"
    )
    run_signals = OperationTelemetry.create(component, tracer, meter)
    authorization_signals = OperationTelemetry.create("gate5a_authorization", tracer, meter)
    intervention_signals = OperationTelemetry.create("intervention", tracer, meter)
    runtime_signals = OperationTelemetry.create("runtime", tracer, meter)
    execution_signals = OperationTelemetry.create("execution", tracer, meter)
    discovery_signals = OperationTelemetry.create("plugin_discovery", tracer, meter)

    if not phase5 and not conformance_only:
        root_name = "capability_anatomy.run.execution"
    failure: BaseException | None = None
    result: dict[str, Any] = {}
    lifecycle_context = otel_trace.get_current_span().get_span_context()
    with tracer.start_as_current_span(
        root_name,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        trace_id = f"{span.get_span_context().trace_id:032x}"
        try:
            result = _run_experiment(
                config_path, config=config,
                conformance_only=conformance_only,
                authorization_telemetry=authorization_signals,
                intervention_telemetry=intervention_signals,
                runtime_telemetry=runtime_signals,
                execution_telemetry=execution_signals,
                discovery_telemetry=discovery_signals,
            )
        except BaseException as error:
            failure = error
            run_signals.record(
                span, operation=operation, outcome="failed",
                reason="conformance_execution_failed" if conformance_only else "scan_execution_failed",
            )
            span.set_status(Status(
                StatusCode.ERROR,
                "conformance_execution_failed" if conformance_only else "scan_execution_failed",
            ))
        else:
            result["trace_id"] = trace_id
            run_signals.record(
                span, operation=operation, outcome="completed",
                reason="conformance_execution_complete" if conformance_only else "task_execution_complete",
            )
    trace_evidence = _serialize_phase5_trace(exporter, reader, trace_id, diagnostic=failure is not None)
    trace_evidence["execution_parent_span_id"] = f"{lifecycle_context.span_id:016x}"
    trace_evidence["external_parent_span_ids"] = list(dict.fromkeys((invocation_span_id, trace_evidence["execution_parent_span_id"])))
    trace_evidence["scope"] = "completed execution spans; final invocation outcome is in the adjacent invocation receipt"
    if failure is not None:
        failed_config = config
        output = failed_config.output.directory
        if not output.is_absolute():
            output = Path(os.path.abspath(config_path.parent / output))
        trace_evidence["run_id"] = failed_config.experiment_id
        if not exists(output / "evidence-manifest.json") and not exists(output / "core-manifest.json"):
            _persist_phase5_failure_trace(output, trace_id, trace_evidence, failure)
        raise failure
    output = Path(
        result.pop("_output_directory")
        if conformance_only else result["run_directory"]
    )
    trace_evidence["run_id"] = result.pop("_experiment_id", config.experiment_id)
    prior_segments = _prior_trace_segments(output)
    trace_evidence["prior_segments"] = list({segment["trace_id"]: segment for segment in prior_segments
                                           if segment.get("trace_id") != trace_id}.values())
    if conformance_only:
        _persist_phase5_conformance(output, result, trace_evidence)
    elif not phase5:
        from .core_evidence import finalize_core_evidence
        finalize_core_evidence(output, trace_evidence)
    else:
        required_artifacts = result.pop("_required_artifacts")
        manifest_identity = result.pop("_manifest_identity")
        report_path = output / "report.json"
        report = parse_mapping(read_bytes(report_path, max_bytes=MAX_ARTIFACT_BYTES))
        report["trace_id"] = trace_id
        write_final_artifacts(
            output,
            {"trace.json": trace_evidence, "report.json": report},
            required_paths=required_artifacts,
            identity=manifest_identity,
        )
    return result
