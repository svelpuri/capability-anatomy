from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy import cli
from capability_anatomy.errors import InvalidEvidenceError
from capability_anatomy.execution import core_evidence, secure_fs
from capability_anatomy.serialization import canonical_json_bytes
from capability_anatomy.telemetry import telemetry_scope


def _cli(*arguments):
    return subprocess.run([sys.executable, "-m", "capability_anatomy.cli", *(str(arg) for arg in arguments)],
                          capture_output=True, text=True, timeout=20)


@pytest.fixture
def bundle(tmp_path):
    example = tmp_path / "example"
    result = _cli("example", "--output", example)
    assert result.returncode == 0, result.stderr
    result = _cli("run", "--config", example / "experiment.json")
    assert result.returncode == 0, result.stderr
    output = example / "run"
    assert (output / core_evidence.MANIFEST).is_file(), "public run did not finalize core evidence"
    return output


def _all_bytes(output):
    return {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}


def _rebind(output):
    path = output / core_evidence.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["artifacts"] = [{"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                             for name, payload in sorted(_all_bytes(output).items()) if name != core_evidence.MANIFEST]
    path.write_bytes(canonical_json_bytes(manifest))


def test_example_run_verify_report_and_resume_are_readable(bundle):
    before = _all_bytes(bundle)
    verified = _cli("verify", "--evidence", bundle)
    assert verified.returncode == 0, verified.stderr
    response = json.loads(verified.stdout)
    assert {key: response[key] for key in ("bundle", "integrity_only", "status")} == {"bundle": "core", "integrity_only": True, "status": "verified"}
    assert response["trace_evidence"]["status"] == "complete"
    assert response["trace_evidence"]["dropped_spans"] == 0
    markdown = _cli("report", "--evidence", bundle)
    assert markdown.returncode == 0, markdown.stderr
    assert "no pruning" in markdown.stdout
    report = _cli("report", "--evidence", bundle, "--format", "json")
    assert report.returncode == 0, report.stderr
    tasks = json.loads(report.stdout)["tasks"]
    assert tasks["baseline"]["scores"]["exact_match"]["value"] == 1
    assert tasks["scan.component-neutral"]["scores"]["exact_match"]["value"] == 1
    assert tasks["scan.component-damage"]["scores"]["exact_match"]["value"] == 0
    assert _all_bytes(bundle) == before
    task_before = {name: payload for name, payload in before.items() if name.startswith("tasks/")}
    resumed = _cli("run", "--config", bundle.parent / "experiment.json")
    assert resumed.returncode == 0, resumed.stderr
    assert {name: payload for name, payload in _all_bytes(bundle).items() if name.startswith("tasks/")} == task_before
    assert _cli("verify", "--evidence", bundle).returncode == 0


@pytest.mark.parametrize("kind", ["changed", "extra", "missing", "duplicate-entry", "path-traversal", "symlink"])
def test_manifest_rejects_artifact_corruption(bundle, kind, tmp_path):
    manifest_path = bundle / core_evidence.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if kind == "changed":
        (bundle / "report.md").write_text("unverified claim")
    elif kind == "extra":
        (bundle / "unexpected.json").write_text("{}")
    elif kind == "missing":
        (bundle / "report.md").unlink()
    elif kind == "duplicate-entry":
        manifest["artifacts"].append(manifest["artifacts"][0])
        manifest_path.write_text(json.dumps(manifest))
    elif kind == "path-traversal":
        manifest["artifacts"][0]["path"] = "../outside.json"
        manifest_path.write_text(json.dumps(manifest))
    else:
        (bundle / "report.md").unlink()
        external = tmp_path / "outside.md"
        external.write_text("outside")
        (bundle / "report.md").symlink_to(external)
    result = _cli("verify", "--evidence", bundle)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_rehashed_report_forgery_is_refused_by_observation_reconstruction(bundle):
    report_path = bundle / "report.json"
    report = json.loads(report_path.read_text())
    report["tasks"]["baseline"]["scores"]["exact_match"]["value"] = 0.123
    report_path.write_bytes(canonical_json_bytes(report))
    _rebind(bundle)
    with pytest.raises(InvalidEvidenceError):
        core_evidence.verify_core_evidence(bundle)


def test_rehashed_task_without_matching_commit_is_refused(bundle):
    path = bundle / "tasks/baseline.json"
    task = json.loads(path.read_text())
    task["debug_metadata"] = {"note": "changed after task commit"}
    path.write_bytes(canonical_json_bytes(task))
    _rebind(bundle)
    with pytest.raises(InvalidEvidenceError):
        core_evidence.verify_core_evidence(bundle)


def test_rehashed_missing_task_is_refused_by_frozen_plan(bundle):
    (bundle / "tasks/scan.component-neutral.json").unlink()
    (bundle / "tasks/scan.component-neutral.complete.json").unlink()
    _rebind(bundle)
    with pytest.raises(InvalidEvidenceError):
        core_evidence.verify_core_evidence(bundle)


def test_rehashed_incomplete_state_is_refused(bundle):
    (bundle / "execution-state.json").write_bytes(canonical_json_bytes({"state": "interrupted", "reason": "task_interrupted"}))
    _rebind(bundle)
    with pytest.raises(InvalidEvidenceError):
        core_evidence.verify_core_evidence(bundle)


@pytest.mark.parametrize("payload,reason", [
    (b"[]", "evidence_json_object_required"), (b"null", "evidence_json_object_required"),
    (b'{"artifacts":NaN}', "evidence_json_invalid"),
    (b'{"artifacts":[],"artifacts":[]}', "evidence_json_duplicate_key"),
    (b"\xff", "evidence_json_invalid"), (b"[" * 10000, "evidence_json_depth_limit"),
], ids=["array", "null", "nonfinite", "duplicate", "encoding", "recursion"])
def test_malformed_manifest_returns_typed_error_without_traceback(bundle, payload, reason):
    (bundle / core_evidence.MANIFEST).write_bytes(payload)
    result = _cli("verify", "--evidence", bundle)
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"] == reason
    assert "Traceback" not in result.stderr


def test_evidence_byte_limit_is_enforced_before_json_parsing(bundle, monkeypatch):
    monkeypatch.setattr(core_evidence, "MAX_ARTIFACT_BYTES", 16)
    with pytest.raises(InvalidEvidenceError):
        core_evidence.verify_core_evidence(bundle)


def test_failed_finalization_does_not_rewrite_existing_bound_files(bundle):
    before = _all_bytes(bundle)
    with pytest.raises(InvalidEvidenceError):
        core_evidence.finalize_core_evidence(bundle, {"trace_id": "bad", "spans": []})
    assert _all_bytes(bundle) == before
    assert core_evidence.verify_core_evidence(bundle)["status"] == "complete"


def test_help_advertises_only_implemented_commands():
    parser = cli.build_parser()
    commands = next(action.choices for action in parser._actions if getattr(action, "choices", None))
    assert set(commands) == {"doctor", "example", "freeze-phase5-records", "validate-phase5", "run", "conform-phase5", "verify", "report"}
    for command in ("baseline", "scan", "analyze", "transform", "validate"):
        result = _cli(command)
        assert result.returncode == 2
        assert "Traceback" not in result.stderr


def test_doctor_detects_missing_core_schema(monkeypatch, capsys):
    original = Path.read_text

    def missing(self, *args, **kwargs):
        if self.name == "experiment-config.v1.schema.json":
            raise OSError("fake-secret-missing-schema")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", missing)
    assert cli.main(["doctor"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert any(check["reason"] == "schema_experiment-config.v1.schema.json_unavailable"
               for check in result["checks"] if check["status"] == "failed")
    assert "fake-secret" not in json.dumps(result)


@pytest.mark.parametrize("module, distribution", [("yaml", "PyYAML"),
    ("opentelemetry.exporter.otlp.proto.http.trace_exporter", "opentelemetry-exporter-otlp-proto-http")])
def test_doctor_detects_missing_core_dependency(monkeypatch, capsys, module, distribution):
    original = cli.import_module

    def missing(name):
        if name == module:
            raise ImportError("fake-secret-dependency")
        return original(name)

    monkeypatch.setattr(cli, "import_module", missing)
    assert cli.main(["doctor"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert any(check["check"] == "dependency_" + distribution and check["status"] == "failed" for check in result["checks"])
    assert "fake-secret" not in json.dumps(result)


@pytest.mark.parametrize("error", [RuntimeError("fake-secret-token"), InvalidEvidenceError("fake-secret-token")])
def test_cli_never_echoes_plugin_exception_content(tmp_path, monkeypatch, capsys, error):
    cli._example(tmp_path)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "run_experiment", fail)
    assert cli.main(["run", "--config", str(tmp_path / "experiment.json")]) == (2 if isinstance(error, InvalidEvidenceError) else 3)
    result = capsys.readouterr()
    assert "fake-secret" not in result.err
    assert "Traceback" not in result.err
    assert json.loads(result.err)["message"]


def test_example_refuses_symlink_and_preserves_existing_changed_inputs(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    link = tmp_path / "link"
    link.symlink_to(external, target_is_directory=True)
    result = _cli("example", "--output", link)
    assert result.returncode == 2
    assert list(external.iterdir()) == []
    cli._example(external)
    path = external / "experiment.json"
    path.write_text("user-authored bytes")
    before = _all_bytes(external)
    result = _cli("example", "--output", external)
    assert result.returncode == 2
    assert _all_bytes(external) == before


def test_verification_emits_reason_and_counter(bundle):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader])
    try:
        with telemetry_scope(provider.get_tracer("test"), meter.get_meter("test")):
            core_evidence.verify_core_evidence(bundle)
        spans = [span for span in exporter.get_finished_spans() if span.name == "capability_anatomy.evidence.verify"]
        assert len(spans) == 1
        assert spans[0].end_time > spans[0].start_time
        assert spans[0].events[-1].attributes["capability_anatomy.reason"] == "core_evidence_consistent"
        points = [point for resource in reader.get_metrics_data().resource_metrics for scope in resource.scope_metrics
                  for metric in scope.metrics for point in metric.data.data_points]
        assert any(point.attributes.get("capability_anatomy.reason") == "core_evidence_consistent" and point.value == 1 for point in points)
    finally:
        provider.shutdown()
        meter.shutdown()


def test_digest_covers_bound_auxiliary_artifact(bundle):
    auxiliary = bundle / "auxiliary.json"
    auxiliary.write_bytes(b'{"note":"original"}')
    _rebind(bundle)
    assert core_evidence.verify_core_evidence(bundle)["status"] == "complete"
    auxiliary.write_bytes(b'{"note":"modified"}')
    with pytest.raises(InvalidEvidenceError):
        core_evidence.verify_core_evidence(bundle)


def test_manifest_covers_missing_auxiliary_artifact(bundle):
    auxiliary = bundle / "auxiliary.json"
    auxiliary.write_bytes(b'{"note":"original"}')
    _rebind(bundle)
    assert core_evidence.verify_core_evidence(bundle)["status"] == "complete"
    auxiliary.unlink()
    with pytest.raises(InvalidEvidenceError):
        core_evidence.verify_core_evidence(bundle)


def test_rehashed_unplanned_task_is_refused(bundle):
    (bundle / "tasks/scan.unplanned.json").write_bytes((bundle / "tasks/baseline.json").read_bytes())
    (bundle / "tasks/scan.unplanned.complete.json").write_bytes((bundle / "tasks/baseline.complete.json").read_bytes())
    _rebind(bundle)
    with pytest.raises(InvalidEvidenceError):
        core_evidence.verify_core_evidence(bundle)


def test_inventory_bounds_directory_entries_including_empty_directories(bundle, monkeypatch):
    for index in range(20):
        (bundle / f"empty-{index}").mkdir()
    monkeypatch.setattr(core_evidence, "MAX_ARTIFACTS", 30)
    with pytest.raises(InvalidEvidenceError):
        core_evidence.verify_core_evidence(bundle)


@pytest.mark.parametrize("broken", [False, True])
def test_nonrun_cli_scopes_exports_and_shuts_down_without_touching_host_sdk(tmp_path, monkeypatch, capsys, broken):
    from opentelemetry import trace, metrics
    exporter = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    global_tracer, global_meter = trace.get_tracer_provider(), metrics.get_meter_provider()

    def collector(provider, readers):
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        readers.append(reader)

    monkeypatch.setattr(cli, "configure_otlp", collector)
    before = _all_bytes(tmp_path)
    if broken:
        original = cli.import_module
        def missing(name):
            if name == "yaml":
                raise ImportError("fake-sensitive-value")
            return original(name)
        monkeypatch.setattr(cli, "import_module", missing)
    assert cli.main(["doctor"]) == (2 if broken else 0)
    assert json.loads(capsys.readouterr().out)["status"] == ("failed" if broken else "ok")
    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "capability_anatomy.cli.command")
    assert root.events[-1].attributes["capability_anatomy.reason"] == ("command_failed" if broken else "command_complete")
    assert root.status.status_code.name == ("ERROR" if broken else "UNSET")
    assert "fake-sensitive-value" not in repr(spans)
    assert any(span.name == "capability_anatomy.doctor.check" for span in spans)
    assert all(span.context.trace_id == root.context.trace_id for span in spans)
    assert trace.get_tracer_provider() is global_tracer
    assert metrics.get_meter_provider() is global_meter
    assert _all_bytes(tmp_path) == before


@pytest.mark.parametrize("reason, code", [("document_requires_regular_file", 2), ("model_loading_selector_refused", 5)])
def test_new_input_and_model_denials_keep_documented_exit_categories(tmp_path, monkeypatch, capsys, reason, code):
    from capability_anatomy.authored_inputs import AuthoredInputError
    from capability_anatomy.errors import UnsupportedAdapterError
    class SelectorRefused(UnsupportedAdapterError):
        reason = "model_loading_selector_refused"
    cli._example(tmp_path)
    def refuse(*args, **kwargs):
        if reason == "document_requires_regular_file":
            raise AuthoredInputError(reason)
        raise SelectorRefused("fake-private-model-selector")
    monkeypatch.setattr(cli, "run_experiment", refuse)
    assert cli.main(["run", "--config", str(tmp_path / "experiment.json")]) == code
    diagnostic = capsys.readouterr().err
    assert json.loads(diagnostic)["error"] == reason
    assert "fake-private-model-selector" not in diagnostic


def test_real_cli_refuses_fifo_promptly_and_regular_config_runs(tmp_path):
    import os
    fifo = tmp_path / "experiment.fifo"
    os.mkfifo(fifo)
    result = subprocess.run(
        [sys.executable, "-m", "capability_anatomy.cli", "run", "--config", str(fifo)],
        capture_output=True, text=True, timeout=3,
    )
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stderr)["error"] == "document_requires_regular_file"
    assert "Traceback" not in result.stderr
    example = tmp_path / "regular"
    assert _cli("example", "--output", example).returncode == 0
    positive = _cli("run", "--config", example / "experiment.json")
    assert positive.returncode == 0, positive.stderr
    assert (example / "run" / core_evidence.MANIFEST).is_file()
