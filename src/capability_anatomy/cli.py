from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from importlib import import_module, metadata, resources
import os
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import load_experiment_config
from .discovery import PluginDiscovery
from .execution import secure_fs
from .execution.core_evidence import (MANIFEST as CORE_MANIFEST, _mapping, MAX_ARTIFACT_BYTES, render_core_report, verify_core_evidence, trace_evidence_summary)
from .serialization import canonical_json_bytes
from .telemetry import OperationTelemetry, telemetry_scope
from .run_telemetry import configure_otlp, TelemetryDeliveryError
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.trace import Status, StatusCode
from .errors import CapabilityAnatomyError, ExitCode, InvalidConfigurationError, InvalidEvidenceError, UnsupportedAdapterError, InterruptedRunError
from .execution import run_experiment
from .execution.phase5_campaign import reconstruct_manifest, verify_phase5_aggregates
from .phase5_protocol import sha256_file, validate_phase5_protocol, write_record_plan


CONFIG_COMMANDS = ("run", "conform-phase5")
EVIDENCE_COMMANDS = ("verify", "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capability-anatomy", description="Reproducible capability intervention evidence")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check core dependencies, schemas, plugins and platform support")
    doctor.add_argument("--output", type=Path, help="also probe locking and publication on this output volume")
    example = subparsers.add_parser("example", help="write a self-contained offline synthetic example")
    example.add_argument("--output", type=Path, required=True)
    freeze = subparsers.add_parser("freeze-phase5-records", help="freeze exact Phase 5 record identities")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    phase5 = subparsers.add_parser("validate-phase5", help="validate a frozen Phase 5 protocol without inference")
    phase5.add_argument("--protocol", type=Path, required=True)
    for command in CONFIG_COMMANDS:
        child = subparsers.add_parser(command, help=f"{command} from a frozen experiment configuration")
        child.add_argument("--config", type=Path, required=True)
    for command in EVIDENCE_COMMANDS:
        child = subparsers.add_parser(command, help=f"{command} an evidence bundle")
        child.add_argument("--evidence", type=Path, required=True)
        if command == "report":
            child.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser



def _verify_phase5(evidence: Path) -> dict:
    manifest_path = evidence / "evidence-manifest.json"
    manifest = _mapping(secure_fs.read_bytes(manifest_path, max_bytes=MAX_ARTIFACT_BYTES))
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("path"), str)
        for item in entries
    ):
        raise InvalidEvidenceError("Phase 5 evidence manifest entries are invalid")
    required = [item["path"] for item in entries]
    required.append("evidence-manifest.json")
    frozen_required = manifest.get("required_artifacts")
    protocol_path = evidence / "protocol.json"
    try:
        protocol = validate_phase5_protocol(
            protocol_path, validate_external_artifacts=False,
        )
    except InvalidConfigurationError as error:
        raise InvalidEvidenceError("Phase 5 evidence protocol is unreadable or invalid") from error
    if (
        not isinstance(frozen_required, list)
        or "report.json" not in frozen_required
        or "protocol.json" not in frozen_required
        or "observations.jsonl" not in frozen_required
        or "metrics.json" not in frozen_required
        or not set(frozen_required).issubset(required)
        or set(frozen_required) != set(protocol["required_artifacts"])
        or manifest.get("protocol_sha256") != sha256_file(protocol_path)
    ):
        raise InvalidEvidenceError("Phase 5 frozen required artifacts are incomplete")
    reconstruct_manifest(evidence, manifest, required)
    verify_phase5_aggregates(evidence, manifest)
    report = _mapping(secure_fs.read_bytes(evidence / "report.json", max_bytes=MAX_ARTIFACT_BYTES))
    from .execution.evidence_limits import MAX_TRACE_BYTES
    trace = _mapping(secure_fs.read_bytes(evidence / "trace.json", max_bytes=MAX_TRACE_BYTES),
                     max_bytes=MAX_TRACE_BYTES)
    return {**report, "trace_evidence": trace_evidence_summary(trace)}


def _doctor(output: Path | None = None) -> dict:
    checks = []
    signals = OperationTelemetry.create("doctor")

    def check(name, probe):
        with signals.tracer.start_as_current_span("capability_anatomy.doctor.check", record_exception=False,
                                                set_status_on_exception=False) as span:
            try:
                detail = probe()
            except Exception as error:
                reason = (error.reason if isinstance(error, CapabilityAnatomyError)
                          and error.reason in _PUBLIC_ERRORS else name + "_unavailable")
                code, message = _PUBLIC_ERRORS.get(reason, (2, "check unavailable; verify the dependency or platform configuration"))
                checks.append({"check": name, "status": "failed", "reason": reason,
                               "exit_code": code, "message": message})
                signals.record(span, operation=name, outcome="refused", reason=reason)
                span.set_status(Status(StatusCode.ERROR, reason))
            else:
                checks.append({"check": name, "status": "ok", "detail": detail})
                signals.record(span, operation=name, outcome="accepted", reason=name + "_available")

    def python_version():
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("unsupported interpreter")
        return ".".join(str(value) for value in sys.version_info[:3])

    check("python_312", python_version)
    check("posix_storage", lambda: (secure_fs._check_platform(), "platform APIs only: dir_fd, O_NOFOLLOW and flock; use doctor --output to probe a volume")[1])
    if output is not None:
        check("output_storage", lambda: secure_fs.probe_storage_volume(output))
    for distribution, module in (("jsonschema", "jsonschema"), ("PyYAML", "yaml"),
                                 ("opentelemetry-api", "opentelemetry.trace"),
                                 ("opentelemetry-sdk", "opentelemetry.sdk.trace"),
                                 ("opentelemetry-exporter-otlp-proto-http", "opentelemetry.exporter.otlp.proto.http.trace_exporter")):
        check("dependency_" + distribution, lambda distribution=distribution, module=module:
              (import_module(module), metadata.version(distribution))[1])

    def schema(name):
        packaged = resources.files("capability_anatomy").joinpath("_schemas/" + name)
        source = packaged if packaged.is_file() else Path(__file__).resolve().parents[2] / "schemas" / name
        import_module("jsonschema").Draft202012Validator.check_schema(json.loads(source.read_text(encoding="utf-8")))
        return "valid JSON Schema"

    for name in ("experiment-config.v1.schema.json", "evidence-manifest.v1.schema.json",
                 "metric-result.v1.schema.json", "transformation-recipe.v1.schema.json"):
        check("schema_" + name, lambda name=name: schema(name))
    plugins = (("model", "builtin.synthetic-model", ("load", "execute", "topology")),
               ("evaluation", "reference.synthetic-exact-match", ("parse", "score", "aggregate")),
               ("dataset", "reference.synthetic-dataset", ("from_manifest", "records", "validate_for")),
               ("intervention", "builtin.synthetic-intervention", ("validate", "apply")),
               ("runtime", "builtin.local", ("execute", "clock", "synchronize")))
    for role, name, methods in plugins:
        check("plugin_" + role, lambda role=role, name=name, methods=methods:
              PluginDiscovery().resolve(role, name, required_capabilities=frozenset(), required_methods=methods).identity())
    return {"core_api": "capability-anatomy/plugin-api/v1", "version": __version__,
            "status": "ok" if all(item["status"] == "ok" for item in checks) else "failed", "checks": checks,
            "inference_profile": "not probed; core checks do not load optional models",
            "next_step": "capability-anatomy example --output ./anatomy-example"}


def _example(output: Path) -> dict:
    # This fixture is generated, not located through a private checkout or optional package data.
    config = {
        "schema_version": "capability-anatomy/experiment-config/v1",
        "experiment": {"id": "synthetic-component-scan", "seed": 20260903},
        "model": {"plugin": "builtin.synthetic-model", "source": "synthetic-fixture", "revision": "fixture-v1",
                  "parameters": {"representation": "scalar"}},
        "runtime": {"executor": "builtin.local", "deterministic": True, "warmup_runs": 1,
                    "randomized_execution_order": True, "repetitions": 1, "parameters": {}},
        "capability": {"evaluation_plugin": "reference.synthetic-exact-match", "suite_version": "1",
                       "target_metrics": ["exact_match"], "collateral_metrics": []},
        "dataset": {"provider": "reference.synthetic-dataset", "manifest": "dataset.json",
                    "discovery_partition": "discovery", "validation_partition": "validation"},
        "intervention": {"plugin": "builtin.synthetic-intervention", "parameters": {"component_ids": "all"}},
        "policy": {},
        "output": {"directory": "run", "retain_prompts": False, "retain_raw_outputs": True, "prompt_storage": "hash_only"},
    }
    dataset = {"license": "CC0-1.0", "revision": "fixture-v1", "partitions": {}}
    for partition, values in (("discovery", ("alpha", "beta")), ("validation", ("gamma",))):
        dataset["partitions"][partition] = [{"id": f"{partition}-{index}", "partition": partition,
            "input": {"text": value}, "expected": value, "metadata": {}} for index, value in enumerate(values, 1)]
    output = Path(os.path.abspath(output))
    with secure_fs.run_ownership(output):
        for name, value in (("experiment.json", config), ("dataset.json", dataset)):
            secure_fs.create_once(output / name, canonical_json_bytes(value))
    return {"status": "created", "configuration": str(output / "experiment.json"),
            "run_command": ["capability-anatomy", "run", "--config", str(output / "experiment.json")],
            "verify_command": ["capability-anatomy", "verify", "--evidence", str(output / "run")],
            "report_command": ["capability-anatomy", "report", "--evidence", str(output / "run")]}


_PUBLIC_ERRORS = {
    "generic_policy_unsupported": (2, "policy.file is supported only for governed Phase5 authorization; remove it for generic experiments"),
    "delivery_interrupted": (4, "interrupted during telemetry delivery; inspect evidence_status and the invocation receipt before resuming"),
    "otlp_headers_invalid": (2, "OTLP headers are malformed; use comma-separated name=value entries with percent-encoded values"),
    "phase5_resume_trace_missing": (2, "cached Phase 5 tasks have no complete durable execution trace; preserve this evidence and start a new output directory"),
    "gate_authorization_invalid": (2, "authorization does not approve this protocol and operation; obtain a matching review under the frozen review policy"),
    "gate_approved_source_unavailable": (2, "approved source commit is unavailable; fetch the approved history in the experiment checkout"),
    "gate_source_identity_changed": (2, "governed source differs from approval; restore the approved tree or obtain review of the changed source"),
    "invalid_configuration": (2, "configuration is invalid; check required fields, immutable revisions and credential references"),
    "invalid_evidence": (2, "evidence is incomplete, unsafe or inconsistent; inspect the run trace and retain the original files"),
    "run_ownership_busy": (2, "output directory has an active run or evidence reader; wait for the owner to finish and retry"),
    "unsupported_adapter": (5, "requested model capability is unavailable; check the model plugin and optional inference installation"),
    "model_loading_selector_refused": (5, "custom model code, attention kernels and quantization are unsupported; use safe fixed-architecture settings"),
    "unsupported_plugin": (5, "requested plugin or platform capability is unavailable; run doctor and check the plugin contract"),
    "interrupted": (4, "run interrupted; completed tasks can be resumed"),
    "runtime_failure": (3, "operation failed; inspect the invocation receipt if available and run doctor"),
    "telemetry_capacity_exceeded": (2, "telemetry capacity exceeded; the invocation receipt is explicitly incomplete; split the experiment before retrying"),
    "telemetry_delivery_incomplete": (3, "OTLP delivery incomplete; local experiment evidence is retained; inspect the invocation receipt and collector connectivity"),
    "telemetry_disabled": (2, "evidence-producing commands require telemetry; unset OTEL_SDK_DISABLED before running"),
    "invocation_receipt_unwritable": (2, "invocation receipt could not be written; check output parent permissions and free space"),
}
from .execution.core_evidence import _REASONS as _CORE_REASONS
from .execution.evidence_json import _REASONS as _JSON_REASONS
from .evaluations.reduction import _REASONS as _REDUCTION_REASONS
for _reason, _message in (secure_fs._REASONS | _CORE_REASONS | _JSON_REASONS | _REDUCTION_REASONS).items():
    _PUBLIC_ERRORS[_reason] = (2, _message)
_PUBLIC_ERRORS["storage_filesystem_unsupported"] = (5, "output volume cannot provide locking, hard-link publication and directory sync; select a supported local filesystem")
for _reason in ("object_keys_must_be_strings", "duplicate_object_key", "document_node_limit", "document_depth_limit",
                "document_cycle", "document_requires_json_values", "document_requires_finite_numbers", "document_byte_limit",
                "document_format_unsupported", "document_root_must_be_object", "document_unreadable", "document_syntax_invalid",
                "document_requires_regular_file"):
    _PUBLIC_ERRORS[_reason] = (2, "authored input refused: " + _reason)


@contextmanager
def _command_telemetry(command: str):
    if command in CONFIG_COMMANDS:
        yield None  # run_experiment owns its complete SDK lifecycle.
        return
    provider = TracerProvider()
    meter = None
    transport = None
    try:
        readers = []
        transport = configure_otlp(provider, readers)
        meter = MeterProvider(metric_readers=readers)
        tracer = provider.get_tracer("capability_anatomy.cli")
        with telemetry_scope(tracer, meter.get_meter("capability_anatomy.cli")):
            signals = OperationTelemetry.create("cli")
            with tracer.start_as_current_span("capability_anatomy.cli.command", record_exception=False,
                                              set_status_on_exception=False) as span:
                span.set_attribute("capability_anatomy.command", command)
                result = {"success": True}
                try:
                    yield result
                except BaseException:
                    signals.record(span, operation=command, outcome="failed", reason="command_failed")
                    span.set_status(Status(StatusCode.ERROR, "command_failed"))
                    raise
                else:
                    signals.record(span, operation=command, outcome="completed" if result["success"] else "failed",
                                   reason="command_complete" if result["success"] else "command_failed")
                    if not result["success"]:
                        span.set_status(Status(StatusCode.ERROR, "command_failed"))
    finally:
        try:
            if transport is not None:
                delivery = transport.flush()
                if delivery["status"] != "delivered":
                    # Optional delivery is separate from a utility's result. These
                    # commands have no invocation receipt; never promise one.
                    print(json.dumps({"warning": "telemetry_delivery_incomplete",
                                      "command": command, "delivery_status": delivery["status"],
                                      "message": "optional remote telemetry delivery failed; command result is unchanged"},
                                     sort_keys=True), file=sys.stderr)
        finally:
            provider.shutdown()
            if meter is not None:
                meter.shutdown(timeout_millis=10000)


def _dispatch(args) -> int:
    if args.command == "doctor":
        result = _doctor(args.output)
        print(json.dumps(result, sort_keys=True))
        return max((item.get("exit_code", 0) for item in result["checks"]), default=0)
    if args.command == "example":
        result = _example(args.output)
        print(json.dumps(result, sort_keys=True))
        return int(ExitCode.PASSED)
    if args.command == "freeze-phase5-records":
        plan = write_record_plan(
            args.source,
            args.output,
            {"simple": 20, "abstention": 10, "reasoning": 10, "perplexity": 5, "format": 5},
        )
        print(json.dumps({"rows_sha256": plan["rows_sha256"], "status": "frozen"}, sort_keys=True))
        return int(ExitCode.PASSED)
    if args.command == "validate-phase5":
        protocol = validate_phase5_protocol(args.protocol)
        print(json.dumps({"experiment_id": protocol["experiment"]["id"], "status": "valid"}, sort_keys=True))
        return int(ExitCode.PASSED)
    if args.command in CONFIG_COMMANDS:
        result = run_experiment(args.config, conformance_only=args.command == "conform-phase5")
        print(json.dumps(result, sort_keys=True))
        return int(ExitCode.PASSED)
    if args.command in EVIDENCE_COMMANDS:
        with secure_fs.read_ownership(args.evidence):
            generic = secure_fs.exists(args.evidence / CORE_MANIFEST)
            report = verify_core_evidence(args.evidence) if generic else _verify_phase5(args.evidence)
            markdown = None if generic or args.command != "report" or args.format != "markdown" else secure_fs.read_bytes(args.evidence / "report.md", max_bytes=MAX_ARTIFACT_BYTES).decode("utf-8")
        if args.command == "verify":
            print(json.dumps({"status": "verified", "integrity_only": True,
                              "bundle": "core" if generic else "phase5",
                              **({"trace_evidence": report["trace_evidence"]} if "trace_evidence" in report else {})}, sort_keys=True))
        elif args.format == "json":
            print(json.dumps(report, sort_keys=True, ensure_ascii=True))
        elif generic:
            print(render_core_report(report), end="")
        else:
            print(markdown, end="")
        return int(ExitCode.PASSED)
    raise RuntimeError("unreachable command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        with _command_telemetry(args.command) as outcome:
            result = _dispatch(args)
            if outcome is not None:
                outcome["success"] = result == 0
            return result
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        print(json.dumps({"error": "invalid_input", "message": "input or evidence is unreadable or invalid"}), file=sys.stderr)
        return int(ExitCode.INVALID)
    except CapabilityAnatomyError as error:
        reason = error.reason if isinstance(error.reason, str) and error.reason in _PUBLIC_ERRORS else "runtime_failure"
        code, message = _PUBLIC_ERRORS[reason]
        diagnostic = {"error": reason, "message": message}
        context = getattr(error, "context", None)
        if isinstance(context, dict):
            diagnostic.update({key: context[key] for key in (
                "run_directory", "trace_id", "invocation_trace", "evidence_status",
                "delivery_status", "interruption_requested", "receipt_status",
            ) if key in context})
            completed = getattr(error, "result", None)
            if args.command in CONFIG_COMMANDS and completed is not None:
                print(json.dumps(completed, sort_keys=True))
        print(json.dumps(diagnostic, sort_keys=True), file=sys.stderr)
        return code
    except ModuleNotFoundError:
        print(json.dumps({"error": "unsupported_adapter", "message": "optional inference dependency is missing; install the qwen3 extra in this environment"}), file=sys.stderr)
        return int(ExitCode.UNSUPPORTED)
    except KeyboardInterrupt:
        print(json.dumps({"error": "interrupted", "message": "run interrupted; completed tasks can be resumed"}), file=sys.stderr)
        return int(ExitCode.INTERRUPTED)
    except Exception:
        print(json.dumps({"error": "runtime_failure", "message": "operation failed; inspect the run trace and doctor output"}), file=sys.stderr)
        return int(ExitCode.RUNTIME_FAILURE)


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
