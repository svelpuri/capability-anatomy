"""Offline consistency verification for model-neutral research runs.

The manifest is an integrity record, not a signature or a scientific claim.
No model or bundle-selected plugin code executes during verification.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from opentelemetry.trace import Status, StatusCode

from ..errors import InvalidEvidenceError
from ..evaluations.reduction import reduce_scores
from ..serialization import canonical_json_bytes
from ..telemetry import OperationTelemetry
from . import secure_fs
from .evidence_limits import MAX_ARTIFACT_BYTES, MAX_BUNDLE_BYTES, MAX_TRACE_BYTES, MAX_ARTIFACTS, is_trace
from .evidence_json import parse_json, parse_mapping, validate_tree, encode_json
from .evidence_inventory import inventory_paths

SCHEMA = "capability-anatomy/core-evidence/v1"
MANIFEST = "core-manifest.json"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_TASK = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


_REASONS = {
    "evidence_structure_invalid": "evidence structure is invalid; inspect the manifest and task records",
    "evidence_directory_depth_limit": "evidence directory exceeds the depth limit",
    "evidence_entry_limit": "evidence directory exceeds the entry limit",
    "evidence_path_invalid": "evidence manifest contains an invalid artifact path",
    "evidence_bundle_byte_limit": "evidence bundle exceeds the byte limit",
    "evidence_task_incomplete": "task observations are incomplete",
    "evidence_scores_missing": "task observation has no scores",
    "evidence_score_invalid": "observation score must be a finite numeric value",
    "evidence_completion_invalid": "run completion or frozen identities are inconsistent",
    "evidence_task_plan_invalid": "frozen component scan order or task identities are invalid",
    "evidence_task_coverage_mismatch": "task files differ from the frozen component plan",
    "evidence_task_commit_mismatch": "task completion marker does not match its payload and frozen identities",
    "evidence_required_metrics_missing": "task observations omit required metrics",
    "evidence_aggregate_contract_unsupported": "declared aggregate contract is unsupported; use metrics, sample_counts, complete and errors",
    "evidence_aggregate_mismatch": "declared task aggregate differs from the recorded observations",
    "evidence_trace_missing": "run trace is missing or malformed",
    "evidence_generated_byte_limit": "generated evidence exceeds its byte capacity; split the experiment",
    "evidence_manifest_invalid": "evidence manifest schema, entries or declared digests are invalid",
    "evidence_artifact_coverage_mismatch": "evidence files differ from the manifest inventory",
    "evidence_digest_mismatch": "evidence artifact bytes differ from the manifest digest",
    "evidence_report_mismatch": "stored report differs from task reconstruction",
    "evidence_trace_incomplete": "trace evidence declares loss or lacks its required completeness fields",
    "evidence_trace_schema_invalid": "trace completeness schema is unsupported",
    "evidence_trace_coverage_mismatch": "committed observations are not covered by their recorded execution spans",
}


class CoreEvidenceError(InvalidEvidenceError):
    def __init__(self, reason: str):
        super().__init__(_REASONS[reason])
        self.reason = reason


@contextmanager
def _decision(operation):
    signals = OperationTelemetry.create("core_evidence")
    with signals.tracer.start_as_current_span(
        f"capability_anatomy.evidence.{operation}", record_exception=False, set_status_on_exception=False,
    ) as span:
        try:
            yield
        except (InvalidEvidenceError, OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            refused = (error if isinstance(error, InvalidEvidenceError) else
                       secure_fs._os_error(error) if isinstance(error, OSError) else
                       CoreEvidenceError("evidence_structure_invalid"))
            reason = refused.reason
            signals.record(span, operation=operation, outcome="refused", reason=reason)
            span.set_status(Status(StatusCode.ERROR, reason))
            if refused is not error:
                raise refused from error
            raise
        else:
            signals.record(span, operation=operation, outcome="accepted", reason="core_evidence_consistent")


def _json(payload: bytes, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> Any:
    return parse_json(payload, max_bytes=max_bytes)


def _mapping(payload: bytes, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> dict[str, Any]:
    return parse_mapping(payload, max_bytes=max_bytes)


def _inventory(output: Path) -> list[str]:
    return inventory_paths(output, max_entries=MAX_ARTIFACTS, max_depth=16, error=CoreEvidenceError)


def _artifact_limit(relative: str) -> int:
    return MAX_TRACE_BYTES if is_trace(relative) else MAX_ARTIFACT_BYTES


def _snapshot(output: Path) -> dict[str, bytes]:
    payloads = {}
    size = 0
    for relative in _inventory(output):
        if relative == MANIFEST:
            continue
        payload = secure_fs.read_bytes(output / relative, max_bytes=_artifact_limit(relative))
        size += len(payload)
        if size > MAX_BUNDLE_BYTES:
            raise CoreEvidenceError("evidence_bundle_byte_limit")
        payloads[relative] = payload
    return payloads


def _number(value: Any) -> float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise CoreEvidenceError("evidence_score_invalid")
    return float(value)


def _summarize(task: Mapping[str, Any]) -> dict[str, Any]:
    observations = task.get("observations")
    if not isinstance(observations, list) or not observations or task.get("complete") is not True:
        raise CoreEvidenceError("evidence_task_incomplete")
    values: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        if not isinstance(observation, dict) or observation.get("status") != "complete":
            raise CoreEvidenceError("evidence_task_incomplete")
        scores = observation.get("scores")
        if not isinstance(scores, dict) or not scores:
            raise CoreEvidenceError("evidence_scores_missing")
        for metric, score in scores.items():
            if not isinstance(score, dict):
                raise CoreEvidenceError("evidence_score_invalid")
            _number(score.get("value"))
            values.setdefault(metric, []).append(score)
    summary = {metric: reduce_scores(scores) for metric, scores in sorted(values.items())}
    return {"observations": len(observations), "scores": summary}



def trace_completeness(segment: Mapping[str, Any]) -> str:
    """Read the versioned loss contract without certifying missing old fields."""
    schema = segment.get("schema_version")
    if schema not in {"capability-anatomy/phase5-trace/v1", "capability-anatomy/phase5-trace/v2"}:
        raise CoreEvidenceError("evidence_trace_schema_invalid")
    declared = "complete" in segment or "dropped_spans" in segment
    if schema.endswith("/v2") or declared:
        if segment.get("complete") is not True or type(segment.get("dropped_spans")) is not int or segment["dropped_spans"] != 0:
            raise CoreEvidenceError("evidence_trace_incomplete")
        return "complete"
    return "legacy_not_recorded"


def _trace_assessments(trace: Mapping[str, Any], *, inherited_schema: str | None = None) -> list[str]:
    if "schema_version" not in trace and inherited_schema == "capability-anatomy/phase5-trace/v1":
        trace = {"schema_version": inherited_schema, **trace}
    assessments = [trace_completeness(trace)]
    prior = trace.get("prior_segments", [])
    if not isinstance(prior, list) or any(not isinstance(segment, dict) for segment in prior):
        raise CoreEvidenceError("evidence_trace_coverage_mismatch")
    for segment in prior:
        assessments.extend(_trace_assessments(segment, inherited_schema=trace.get("schema_version")))
    return assessments


def trace_evidence_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    assessments = _trace_assessments(trace)
    complete = all(value == "complete" for value in assessments)
    return {"status": "complete" if complete else "legacy_not_recorded", "segments": len(assessments),
            "dropped_spans": 0 if complete else None, "scope": "retained execution segments"}


def _record_spans(segment: Mapping[str, Any]) -> dict[tuple[str, str], tuple[str, str]]:
    trace_id = segment.get("trace_id")
    if not isinstance(trace_id, str) or not re.fullmatch(r"[0-9a-f]{32}", trace_id) or not trace_id.strip("0"):
        raise CoreEvidenceError("evidence_trace_coverage_mismatch")
    spans = segment.get("spans")
    if not isinstance(spans, list):
        raise CoreEvidenceError("evidence_trace_coverage_mismatch")
    records = {}
    for span in spans:
        if not isinstance(span, dict) or not isinstance(span.get("attributes"), dict):
            raise CoreEvidenceError("evidence_trace_coverage_mismatch")
        attributes = span["attributes"]
        if attributes.get("capability_anatomy.reason") != "observation_complete":
            continue
        span_id = span.get("span_id")
        task = attributes.get("capability_anatomy.task_id_sha256")
        record = attributes.get("capability_anatomy.record_id_sha256")
        if (span.get("name") != "capability_anatomy.evaluation.record" or
            span.get("trace_id") != trace_id or span.get("status") not in {"UNSET", "OK"} or
            attributes.get("capability_anatomy.outcome") != "accepted" or
            not isinstance(span_id, str) or not re.fullmatch(r"[0-9a-f]{16}", span_id) or not span_id.strip("0") or
            not isinstance(task, str) or not _HEX.fullmatch(task) or
            not isinstance(record, str) or not _HEX.fullmatch(record)):
            raise CoreEvidenceError("evidence_trace_coverage_mismatch")
        identity = trace_id, span_id
        value = task, record
        if identity in records and records[identity] != value:
            raise CoreEvidenceError("evidence_trace_coverage_mismatch")
        records[identity] = value
    return records


def _trace_coverage(payloads: Mapping[str, bytes], tasks: list[str], trace: Mapping[str, Any]) -> dict[str, Any]:
    assessments = _trace_assessments(trace)
    expected = {}
    for task_id in tasks:
        task = _mapping(payloads[f"tasks/{task_id}.json"])
        ids = [observation.get("example_id") for observation in task["observations"]]
        if any(not isinstance(identifier, str) for identifier in ids):
            raise CoreEvidenceError("evidence_trace_coverage_mismatch")
        task_hash = hashlib.sha256(task_id.encode()).hexdigest()
        expected[task_id] = Counter((task_hash, hashlib.sha256(identifier.encode()).hexdigest()) for identifier in ids)
    checkpoints = {path for path in payloads if path.startswith("task-traces/")}
    if checkpoints:
        if checkpoints != {f"task-traces/{task_id}.json" for task_id in tasks}:
            raise CoreEvidenceError("evidence_trace_coverage_mismatch")
        for task_id in tasks:
            segment = _mapping(payloads[f"task-traces/{task_id}.json"], max_bytes=MAX_TRACE_BYTES)
            assessments.extend(_trace_assessments(segment))
            if (segment.get("task_id_sha256") != hashlib.sha256(task_id.encode()).hexdigest() or
                segment.get("payload_sha256") != hashlib.sha256(payloads[f"tasks/{task_id}.json"]).hexdigest() or
                Counter(_record_spans(segment).values()) != expected[task_id]):
                raise CoreEvidenceError("evidence_trace_coverage_mismatch")
    else:
        # Historical bundles predate task checkpoints. Their complete trace
        # remains acceptable only when it covers every committed observation.
        prior = trace.get("prior_segments", [])
        if not isinstance(prior, list) or any(not isinstance(item, dict) for item in prior):
            raise CoreEvidenceError("evidence_trace_coverage_mismatch")
        records = {}
        for segment in [trace, *prior]:
            for identity, value in _record_spans(segment).items():
                if identity in records and records[identity] != value:
                    raise CoreEvidenceError("evidence_trace_coverage_mismatch")
                records[identity] = value
        if Counter(records.values()) != sum(expected.values(), Counter()):
            raise CoreEvidenceError("evidence_trace_coverage_mismatch")
    return {"status": "complete" if all(value == "complete" for value in assessments) else "legacy_not_recorded",
            "segments": len(assessments), "dropped_spans": 0 if all(value == "complete" for value in assessments) else None,
            "scope": "retained execution segments and task checkpoints"}


def _report(payloads: Mapping[str, bytes], *, trace: Mapping[str, Any] | None = None, report_version: str = "capability-anatomy/core-report/v1") -> dict[str, Any]:
    config = _mapping(payloads["experiment-config.json"])
    compatibility = _mapping(payloads["compatibility.json"])
    state = _mapping(payloads["run-state.json"])
    execution = _mapping(payloads["execution-state.json"])
    config_hash = hashlib.sha256(payloads["experiment-config.json"]).hexdigest()
    compatibility_hash = hashlib.sha256(payloads["compatibility.json"]).hexdigest()
    if state != {"schema_version": "capability-anatomy/run-state/v1", "config_sha256": config_hash,
                 "compatibility_sha256": compatibility_hash} or execution != {
                     "state": "complete", "reason": "all_tasks_committed"}:
        raise CoreEvidenceError("evidence_completion_invalid")
    order = _json(payloads["component-scan-order.json"])
    if not isinstance(order, list) or not order or any(not isinstance(item, str) for item in order) or len(order) != len(set(order)):
        raise CoreEvidenceError("evidence_task_plan_invalid")
    tasks = ["baseline"] + [f"scan.{component.replace('.', '-')}" for component in order]
    if len(set(tasks)) != len(tasks) or any(not _TASK.fullmatch(task) for task in tasks):
        raise CoreEvidenceError("evidence_task_plan_invalid")
    expected_task_paths = {f"tasks/{task}{suffix}" for task in tasks for suffix in (".json", ".complete.json")}
    if {path for path in payloads if path.startswith("tasks/")} != expected_task_paths:
        raise CoreEvidenceError("evidence_task_coverage_mismatch")
    summaries = {}
    for task_id in tasks:
        payload = payloads[f"tasks/{task_id}.json"]
        marker = _mapping(payloads[f"tasks/{task_id}.complete.json"])
        if marker != {"task_id": task_id, "payload_sha256": hashlib.sha256(payload).hexdigest(),
                      "config_sha256": config_hash, "compatibility_sha256": compatibility_hash}:
            raise CoreEvidenceError("evidence_task_commit_mismatch")
        task = _mapping(payload)
        summary = _summarize(task)
        required_metrics = set(config["capability"]["target_metrics"]) | set(config["capability"]["collateral_metrics"])
        if not required_metrics <= set(summary["scores"]):
            raise CoreEvidenceError("evidence_required_metrics_missing")
        declared = task.get("metrics")
        if not isinstance(declared, dict) or not {"metrics", "errors"} <= set(declared):
            raise CoreEvidenceError("evidence_aggregate_contract_unsupported")
        expected_metrics = {metric: score["value"] for metric, score in summary["scores"].items()}
        expected_counts = {metric: score["denominator"] for metric, score in summary["scores"].items()}
        for field, expected in (("metrics", expected_metrics), ("sample_counts", expected_counts)):
            if field not in declared:
                continue  # Optional plugin counts are reconstructed from observations.
            actual = declared[field]
            if not isinstance(actual, dict) or set(actual) != set(expected) or any(_number(actual[metric]) != value for metric, value in expected.items()):
                raise CoreEvidenceError("evidence_aggregate_mismatch")
        if ("complete" in declared and (type(declared["complete"]) is not int or declared["complete"] != summary["observations"])) or type(declared["errors"]) is not int or declared["errors"] != 0:
            raise CoreEvidenceError("evidence_aggregate_mismatch")
        summaries[task_id] = summary
    if trace is None:
        trace = _mapping(payloads["trace.json"], max_bytes=MAX_TRACE_BYTES)
    else:
        validate_tree(trace, max_bytes=MAX_TRACE_BYTES)
    if not isinstance(trace.get("trace_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", trace["trace_id"]) or not trace["trace_id"].strip("0") or not isinstance(trace.get("spans"), list) or not trace["spans"]:
        raise CoreEvidenceError("evidence_trace_missing")
    assessment = _trace_coverage(payloads, tasks, trace)
    if report_version not in {"capability-anatomy/core-report/v1", "capability-anatomy/core-report/v2"}:
        raise CoreEvidenceError("evidence_report_mismatch")
    return {"schema_version": report_version,
            **({"trace_evidence": assessment} if report_version.endswith("/v2") else {}), "status": "complete",
            "experiment_id": config["experiment_id"], "trace_id": trace["trace_id"],
            "component_scan_order": order, "tasks": summaries,
            "claim_boundary": "Experimental temporary component sensitivity; no pruning, compression, generalization or authenticity claim.",
            "verification_boundary": "Artifact integrity and reconstruction from recorded scores; model outputs and scientific validity are not independently reproduced."}


def render_core_report(report: Mapping[str, Any]) -> str:
    lines = ["# Capability Anatomy experimental run", "", report["claim_boundary"], "", report["verification_boundary"], "",
             "| Task | Metric | Recorded-score summary | Reducer | Observations |", "| --- | --- | ---: | --- | ---: |"]
    if "trace_evidence" in report:
        lines[5:5] = ["Trace evidence: " + report["trace_evidence"]["status"] + ".", ""]
    ordered_tasks = sorted(report["tasks"].items()) if report["schema_version"].endswith("/v2") else report["tasks"].items()
    for task, summary in ordered_tasks:
        for metric, score in sorted(summary["scores"].items()):
            # JSON escaping prevents terminal controls from authored identifiers.
            safe_metric = json.dumps(metric, ensure_ascii=True).replace("|", "\\|")
            lines.append(f"| {task} | {safe_metric} | {score['value']:.6g} | {score['reducer']} | {score['observations']} |")
    return "\n".join(lines) + "\n"


def finalize_core_evidence(output: Path, trace: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a completed generic run, under the caller's full-lifecycle ownership."""
    with _decision("finalize"), secure_fs.ensure_run_ownership(output):
        payloads = _snapshot(output)
        payloads["trace.json"] = encode_json(trace, max_bytes=MAX_TRACE_BYTES)
        report = _report(payloads, trace=trace, report_version="capability-anatomy/core-report/v2")
        payloads["report.json"] = encode_json(report)
        payloads["report.md"] = render_core_report(report).encode()
        # Validate all generated byte sizes before the first publication.
        if any(len(payload) > _artifact_limit(name) for name, payload in payloads.items()) or sum(map(len, payloads.values())) > MAX_BUNDLE_BYTES:
            raise CoreEvidenceError("evidence_generated_byte_limit")
        for name in ("trace.json", "report.json", "report.md"):
            secure_fs.atomic_write(output / name, payloads[name])
        manifest = {"schema_version": SCHEMA, "integrity_only": True,
                    "artifacts": [{"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                                  for path, payload in sorted(payloads.items())]}
        secure_fs.atomic_write(output / MANIFEST, encode_json(manifest))
        return report


def verify_core_evidence(output: Path) -> dict[str, Any]:
    """Verify coverage, bytes, task commits and report reconstruction; read-only."""
    with _decision("verify"), secure_fs.read_ownership(output):
        manifest = _mapping(secure_fs.read_bytes(output / MANIFEST, max_bytes=MAX_ARTIFACT_BYTES))
        if manifest.get("schema_version") != SCHEMA or manifest.get("integrity_only") is not True:
            raise CoreEvidenceError("evidence_manifest_invalid")
        entries = manifest.get("artifacts")
        if not isinstance(entries, list) or not entries or len(entries) > MAX_ARTIFACTS:
            raise CoreEvidenceError("evidence_manifest_invalid")
        bound = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
                raise CoreEvidenceError("evidence_manifest_invalid")
            path = entry["path"]
            if not isinstance(path, str) or not path or Path(path).is_absolute() or any(part in ("", ".", "..") for part in path.split("/")) or "\\" in path or path == MANIFEST or path in bound:
                raise CoreEvidenceError("evidence_path_invalid")
            if type(entry["bytes"]) is not int or not 0 <= entry["bytes"] <= _artifact_limit(path) or not isinstance(entry["sha256"], str) or not _HEX.fullmatch(entry["sha256"]):
                raise CoreEvidenceError("evidence_manifest_invalid")
            bound[path] = entry
        payloads = _snapshot(output)
        if set(bound) != set(payloads):
            raise CoreEvidenceError("evidence_artifact_coverage_mismatch")
        for path, payload in payloads.items():
            if bound[path]["bytes"] != len(payload) or bound[path]["sha256"] != hashlib.sha256(payload).hexdigest():
                raise CoreEvidenceError("evidence_digest_mismatch")
        stored_report = _mapping(payloads["report.json"])
        report = _report(payloads, report_version=stored_report.get("schema_version"))
        if payloads["report.json"] != canonical_json_bytes(report) or payloads["report.md"] != render_core_report(report).encode():
            raise CoreEvidenceError("evidence_report_mismatch")
        if "trace_evidence" not in report:
            trace = _mapping(payloads["trace.json"], max_bytes=MAX_TRACE_BYTES)
            report = {**report, "trace_evidence": _trace_coverage(payloads, ["baseline"] + [f"scan.{component.replace('.', '-')}" for component in report["component_scan_order"]], trace)}
        return report
