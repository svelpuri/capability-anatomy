"""Real OS enforcement probes for the experimental release storage boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.errors import InvalidEvidenceError, UnsupportedPluginError
from capability_anatomy.execution import secure_fs
from capability_anatomy.execution.store import FrozenRunStore
from capability_anatomy.telemetry import OperationTelemetry


def _store(root: Path) -> FrozenRunStore:
    return FrozenRunStore(root, {"seed": 7}, {"revision": "fixed"})


@pytest.mark.parametrize("directory", ["tasks", "failures", "conformance", "trace-failures", "nested/deeper"])
def test_write_rejects_directory_symlink_without_modifying_external_files(tmp_path, directory):
    output, external = tmp_path / "run", tmp_path / "external"
    output.mkdir()
    external.mkdir()
    sentinel = external / "sentinel.json"
    sentinel.write_bytes(b"untouched")
    link = output / directory
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(external, target_is_directory=True)
    with secure_fs.run_ownership(output):
        with pytest.raises(InvalidEvidenceError):
            secure_fs.atomic_write(link / "sentinel.json", b"overwrite")
    assert sentinel.read_bytes() == b"untouched"
    assert sorted(p.name for p in external.iterdir()) == ["sentinel.json"]


@pytest.mark.parametrize("leaf", ["experiment-config.json", "compatibility.json", "run-state.json", "execution-state.json", "budget-state.json", "task-plan.json"])
def test_early_run_state_leaves_refuse_symlinks(tmp_path, leaf):
    output = tmp_path / "run"
    output.mkdir()
    external = tmp_path / "external.json"
    external.write_bytes(b"{}")
    (output / leaf).symlink_to(external)
    with secure_fs.run_ownership(output):
        with pytest.raises(InvalidEvidenceError):
            if leaf == "execution-state.json":
                _store(output).record_state("done", "complete")
            elif leaf == "budget-state.json":
                _store(output).add_active_wall_seconds(1)
            elif leaf == "task-plan.json":
                _store(output).freeze_sequence("task-plan", ("a", "b"))
            else:
                _store(output).initialize()
    assert external.read_bytes() == b"{}"


def test_output_root_symlink_is_rejected_before_initialization(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    output = tmp_path / "run"
    output.symlink_to(external, target_is_directory=True)
    with pytest.raises(InvalidEvidenceError):
        with secure_fs.run_ownership(output):
            _store(output).initialize()
    assert list(external.iterdir()) == []


def test_existing_leaf_hardlink_and_fifo_are_rejected(tmp_path):
    external = tmp_path / "external"
    external.write_bytes(b"private")
    output = tmp_path / "run"
    output.mkdir()
    os.link(external, output / "hardlink")
    os.mkfifo(output / "fifo")
    with secure_fs.run_ownership(output):
        for path in (output / "hardlink", output / "fifo"):
            with pytest.raises(InvalidEvidenceError):
                secure_fs.read_bytes(path)
            with pytest.raises(InvalidEvidenceError):
                secure_fs.atomic_write(path, b"overwrite")
    assert external.read_bytes() == b"private"


def test_store_roundtrip_and_interrupted_task_marker(tmp_path):
    with secure_fs.run_ownership(tmp_path):
        store = _store(tmp_path)
        store.initialize()
        store.commit("baseline", {"answer": 42})
        assert store.completed("baseline") == {"answer": 42}
        (tmp_path / "tasks" / "baseline.complete.json").unlink()
        assert store.completed("baseline") is None
        store.commit("baseline", {"answer": 43})
        store.commit_failure("scan.1", {"error_type": "RuntimeError"})
        assert store.failure_attempts("scan.1") == 1
        assert store.freeze_sequence("scan-order", ("a", "b")) == ("a", "b")
        assert store.freeze_sequence("scan-order", ("b", "a")) == ("a", "b")
        assert store.completed("baseline") == {"answer": 43}
    with secure_fs.run_ownership(tmp_path):
        resumed = _store(tmp_path)
        resumed.initialize()
        assert resumed.completed("baseline") == {"answer": 43}
    assert not any("lock" in path.name for path in tmp_path.rglob("*"))


def test_swapped_parent_symlink_does_not_redirect_open_descriptor(tmp_path, monkeypatch):
    output, external = tmp_path / "run", tmp_path / "external"
    (output / "tasks").mkdir(parents=True)
    external.mkdir()
    sentinel = external / "baseline.json"
    sentinel.write_bytes(b"untouched")
    original_write = secure_fs._write_at
    swapped = False

    def swap_after_parent_open(parent, name, payload):
        nonlocal swapped
        if not swapped:
            swapped = True
            (output / "tasks").rename(output / "original-tasks")
            (output / "tasks").symlink_to(external, target_is_directory=True)
        original_write(parent, name, payload)

    with secure_fs.run_ownership(output):
        monkeypatch.setattr(secure_fs, "_write_at", swap_after_parent_open)
        secure_fs.atomic_write(output / "tasks" / "baseline.json", b"safe")
    assert swapped
    assert sentinel.read_bytes() == b"untouched"
    assert (output / "original-tasks" / "baseline.json").read_bytes() == b"safe"


def _child(code, *args):
    return subprocess.run([sys.executable, "-c", code, *(str(arg) for arg in args)],
                          text=True, capture_output=True, timeout=15)


_CHILD_WRITE = '''
from pathlib import Path
import sys
from capability_anatomy.execution.secure_fs import run_ownership, RunBusyError
from capability_anatomy.execution.store import FrozenRunStore
root=Path(sys.argv[1])
try:
    with run_ownership(root):
        store=FrozenRunStore(root, {"seed": 7}, {"revision": "fixed"})
        store.initialize()
        store.commit("baseline", {"answer": 42})
except RunBusyError:
    print("busy-before-initialize")
    sys.exit(2)
print("completed")
'''


def test_same_output_subprocess_cannot_initialize_or_execute_under_owner(tmp_path):
    with secure_fs.run_ownership(tmp_path):
        result = _child(_CHILD_WRITE, tmp_path)
        assert result.returncode == 2, result.stdout + result.stderr
        assert result.stdout.strip() == "busy-before-initialize"
        assert list(tmp_path.iterdir()) == []
    result = _child(_CHILD_WRITE, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _store(tmp_path).completed("baseline") == {"answer": 42}


def test_abrupt_owner_exit_releases_lock_without_removing_evidence(tmp_path):
    result = _child('''
import os,sys
from pathlib import Path
from capability_anatomy.execution.secure_fs import run_ownership, atomic_write
with run_ownership(Path(sys.argv[1])):
    atomic_write(Path(sys.argv[1])/"before-exit.json", b"preserved")
    os._exit(17)
''', tmp_path)
    assert result.returncode == 17
    result = _child(_CHILD_WRITE, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "before-exit.json").read_bytes() == b"preserved"


def test_ownership_never_permits_escape_or_nested_run(tmp_path):
    output = tmp_path / "run"
    with secure_fs.run_ownership(output):
        with pytest.raises(InvalidEvidenceError, match="outside"):
            secure_fs.atomic_write(tmp_path / "outside.json", b"bad")
        with pytest.raises(InvalidEvidenceError, match="nested"):
            with secure_fs.run_ownership(output):
                pass
    assert not (tmp_path / "outside.json").exists()


def test_unsupported_platform_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_fs, "fcntl", None)
    with pytest.raises(UnsupportedPluginError, match="requires POSIX"):
        with secure_fs.run_ownership(tmp_path):
            pass


def test_ownership_telemetry_has_lifecycle_duration_reason_and_no_path(tmp_path):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader])
    signals = OperationTelemetry.create("storage", provider.get_tracer("test"), meter.get_meter("test"))
    try:
        with secure_fs.run_ownership(tmp_path, signals):
            pass
        spans = exporter.get_finished_spans()
        assert {span.name for span in spans} == {"capability_anatomy.storage.ownership", "capability_anatomy.storage.probe_volume"}
        span, = [span for span in spans if span.name == "capability_anatomy.storage.ownership"]
        assert span.end_time > span.start_time
        assert [event.attributes["capability_anatomy.reason"] for event in span.events] == [
            "run_ownership_acquired", "run_ownership_released",
        ]
        assert str(tmp_path) not in json.dumps([dict(event.attributes) for event in span.events])
        points = [point for resource in reader.get_metrics_data().resource_metrics
                  for scope in resource.scope_metrics for metric in scope.metrics
                  for point in metric.data.data_points]
        assert sum(point.value for point in points) == 3
    finally:
        provider.shutdown()
        meter.shutdown()


def test_direct_store_commit_holds_ownership_across_payload_and_marker(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.initialize()
    original = store._atomic_write
    competitors = []

    def compete_between_payload_and_marker(path, payload):
        original(path, payload)
        if path.name == "baseline.json":
            competitors.append(_child('''
from pathlib import Path
import sys
from capability_anatomy.execution.store import FrozenRunStore
from capability_anatomy.execution.secure_fs import RunBusyError
try:
    FrozenRunStore(Path(sys.argv[1]), {"seed": 7}, {"revision": "fixed"}).commit("baseline", {"answer": 99})
except RunBusyError:
    sys.exit(2)
''', tmp_path))

    monkeypatch.setattr(store, "_atomic_write", compete_between_payload_and_marker)
    store.commit("baseline", {"answer": 42})
    assert len(competitors) == 1
    assert competitors[0].returncode == 2, competitors[0].stderr
    assert store.completed("baseline") == {"answer": 42}
    successor = _child(_CHILD_WRITE, tmp_path)
    assert successor.returncode == 0, successor.stderr


def test_matching_ownership_reused_and_other_run_refused(tmp_path):
    output = tmp_path / "run"
    with secure_fs.run_ownership(output):
        assert secure_fs.ownership_active(output)
        with secure_fs.ensure_run_ownership(output):
            _store(output).initialize()
        with pytest.raises(InvalidEvidenceError):
            with secure_fs.ensure_run_ownership(tmp_path / "different-run"):
                pass
    assert not secure_fs.ownership_active(output)
    assert not (tmp_path / "different-run").exists()


def test_failed_immutable_write_never_publishes_partial_input(tmp_path, monkeypatch):
    def fail_fsync(descriptor):
        raise OSError("simulated flush failure after bytes were written")

    with secure_fs.run_ownership(tmp_path):
        with monkeypatch.context() as mutation:
            mutation.setattr(secure_fs.os, "fsync", fail_fsync)
            with pytest.raises(InvalidEvidenceError):
                secure_fs.create_once(tmp_path / "input.json", b"complete")
        assert list(tmp_path.iterdir()) == []
        secure_fs.create_once(tmp_path / "input.json", b"complete")
        secure_fs.create_once(tmp_path / "input.json", b"complete")
        assert secure_fs.read_bytes(tmp_path / "input.json") == b"complete"


def test_failed_owned_run_retains_failure_reason_after_release(tmp_path):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    meter = MeterProvider()
    signals = OperationTelemetry.create("storage", provider.get_tracer("test"), meter.get_meter("test"))
    try:
        with pytest.raises(RuntimeError):
            with secure_fs.run_ownership(tmp_path, signals):
                raise RuntimeError("fake-sensitive-incident-content")
        spans = exporter.get_finished_spans()
        assert {span.name for span in spans} == {"capability_anatomy.storage.ownership", "capability_anatomy.storage.probe_volume"}
        span, = [span for span in spans if span.name == "capability_anatomy.storage.ownership"]
        assert span.attributes["capability_anatomy.reason"] == "owned_run_failed"
        assert span.attributes["capability_anatomy.outcome"] == "failed"
        assert span.status.status_code.name == "ERROR"
        assert span.status.description == "owned_run_failed"
        assert "fake-sensitive" not in repr(span.events)
        assert span.events[-1].attributes["capability_anatomy.reason"] == "run_ownership_released"
    finally:
        provider.shutdown()
        meter.shutdown()
