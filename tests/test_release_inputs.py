from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.request import Request, urlopen

import pytest
import yaml
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.authored_inputs import AuthoredInputError, load_mapping
from capability_anatomy.config import load_experiment_config
from capability_anatomy.credentials import CredentialReference, RuntimeCredentials
from capability_anatomy.discovery import PluginDiscovery
from capability_anatomy.errors import InvalidConfigurationError
from capability_anatomy.serialization import canonical_json_bytes, to_primitive
from capability_anatomy.telemetry import OperationTelemetry

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/examples/synthetic-scan.yaml"
CANARY = "fake-credential-canary-only-9a9825"


def signals(component):
    exporter = InMemorySpanExporter()
    traces = TracerProvider()
    traces.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    return OperationTelemetry.create(component, tracer=traces.get_tracer("probe"), meter=meters.get_meter("probe")), exporter, reader


@pytest.mark.parametrize(("content", "reason"), [
    ("[]", "document_root_must_be_object"),
    ("null", "document_root_must_be_object"),
    ("a: &a [*a]", "document_cycle"),
    ("a: &a {b: *a}", "document_cycle"),
    ("a: [" * 70 + "0" + "]" * 70, "document_depth_limit"),
    ("a: 1\na: 2", "duplicate_object_key"),
    ("{\"a\": 1, \"a\": 2}", "duplicate_object_key"),
    ("a: .nan", "document_requires_finite_numbers"),
    ("a: 2026-01-01", "document_requires_json_values"),
    ("1: value", "object_keys_must_be_strings"),
])
def test_authored_input_adversarial_documents_are_structured(tmp_path, content, reason):
    path = tmp_path / "authored.yaml"
    path.write_text(content)
    telemetry, exporter, reader = signals("authored_input")
    with pytest.raises(AuthoredInputError, match=reason):
        load_mapping(path, telemetry=telemetry)
    span = exporter.get_finished_spans()[-1]
    assert span.attributes["capability_anatomy.reason"] == reason
    assert span.status.status_code.name == "ERROR"
    assert span.end_time >= span.start_time
    assert reader.get_metrics_data().resource_metrics


def test_byte_and_expanded_alias_budgets_with_positive_control(tmp_path):
    path = tmp_path / "authored.yaml"
    path.write_text("a: &a [1, 2]\nb: [*a, *a]\n")
    assert load_mapping(path) == {"a": [1, 2], "b": [[1, 2], [1, 2]]}
    with pytest.raises(AuthoredInputError, match="document_node_limit"):
        load_mapping(path, max_nodes=6)
    with pytest.raises(AuthoredInputError, match="document_byte_limit"):
        load_mapping(path, max_bytes=8)


def test_deep_json_is_refused_before_recursive_decode(tmp_path):
    path = tmp_path / "deep.json"
    path.write_text('{"a":' + '[' * 1200 + '0' + ']' * 1200 + '}')
    with pytest.raises(AuthoredInputError, match="document_depth_limit"):
        load_mapping(path)


@pytest.mark.parametrize("parameters", [
    {"api_key": CANARY}, {"headers": {"Authorization": "Bearer " + CANARY}},
    {"remotePassword": CANARY}, {"nested": [{"access_token": CANARY}]},
    {"secretAccessKey": CANARY}, {"headers": {"X-API-Key": CANARY}},
    {"source_url": "https://user:" + CANARY + "@example.invalid/model"},
    {"source_url": "https://example.invalid/model?token=" + CANARY},
])
def test_inline_credentials_refuse_before_evidence_with_redacted_message(tmp_path, parameters):
    value = yaml.safe_load(EXAMPLE.read_text())
    value["model"]["parameters"] = parameters
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    with pytest.raises(InvalidConfigurationError) as caught:
        load_experiment_config(path)
    assert "credential" in str(caught.value)
    assert CANARY not in str(caught.value)
    assert "environment" in str(caught.value) or "ENV_NAME" in str(caught.value)


def test_credential_references_preserve_scientific_and_legacy_metadata(tmp_path):
    value = yaml.safe_load(EXAMPLE.read_text())
    value["model"]["parameters"].update({"max_new_tokens": 128, "tokenizer": "plain", "full_scan_authorization_file": "approval.json", "authorization_policy": "frozen"})
    value["credentials"] = {"api_key": {"provider": "env", "name": "CA_TEST_API_KEY"}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    config = load_experiment_config(path)
    assert config.credentials["api_key"] == CredentialReference("env", "CA_TEST_API_KEY")
    assert to_primitive(config)["credentials"] == value["credentials"]
    assert to_primitive(config)["model"]["parameters"] == value["model"]["parameters"]
    assert "credentials" not in to_primitive(load_experiment_config(EXAMPLE))


def test_credential_plugin_live_transport_and_missing_environment(tmp_path, monkeypatch):
    telemetry, exporter, reader = signals("credentials")
    monkeypatch.setenv("CA_TEST_API_KEY", CANARY)
    provider = RuntimeCredentials({"api_key": CredentialReference("env", "CA_TEST_API_KEY")}, telemetry=telemetry)
    seen = []
    class Consumer(BaseHTTPRequestHandler):
        def do_GET(self):
            accepted = self.headers.get("Authorization") == "Bearer " + CANARY
            seen.append(accepted)
            self.send_response(200 if accepted else 403)
            self.end_headers()
            self.wfile.write(b"authenticated" if accepted else b"denied")
        def log_message(self, *_args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Consumer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    class CredentialPlugin:
        def configure_credentials(self, credentials):
            self.credentials = credentials
        def execute(self):
            secret = self.credentials.get("api_key")
            request = Request(f"http://127.0.0.1:{server.server_port}/", headers={"Authorization": "Bearer " + secret.reveal()})
            with urlopen(request, timeout=2) as response:
                return response.read().decode()
    try:
        plugin = CredentialPlugin()
        plugin.configure_credentials(provider)
        result = plugin.execute()
        assert result == "authenticated" and seen == [True]
        monkeypatch.setenv("CA_TEST_API_KEY", "wrong-canary")
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as caught:
            plugin.execute()
        assert caught.value.code == 403 and seen == [True, False]
        monkeypatch.delenv("CA_TEST_API_KEY")
        with pytest.raises(InvalidConfigurationError, match="credential_environment_missing"):
            plugin.execute()
        assert seen == [True, False]  # refusal before transport
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    serialized = json.dumps([span.to_json() for span in exporter.get_finished_spans()])
    assert CANARY not in serialized + result
    assert "credential_environment_resolved" in serialized
    assert "credential_environment_missing" in serialized
    assert reader.get_metrics_data().resource_metrics


def test_secret_wrappers_fail_serialization_and_redact_representations(monkeypatch):
    monkeypatch.setenv("CA_TEST_API_KEY", CANARY)
    provider = RuntimeCredentials({"api_key": CredentialReference("env", "CA_TEST_API_KEY")})
    value = provider.get("api_key")
    assert CANARY not in str(value) + repr(value) + repr(provider)
    assert value.reveal() == CANARY
    for secret in (value, provider):
        with pytest.raises(TypeError, match="cannot be serialized"):
            canonical_json_bytes({"nested": secret})
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(value)


def _install_plugin_metadata(tmp_path, distribution, name, module):
    folder = tmp_path / f"{distribution}-1.0.dist-info"
    folder.mkdir()
    (folder / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.0\n")
    (folder / "entry_points.txt").write_text(f"[capability_anatomy.models]\n{name} = {module}:Plugin\n")


@pytest.mark.parametrize("builtin_collision", [False, True])
def test_real_duplicate_entrypoints_refuse_before_import(tmp_path, monkeypatch, builtin_collision):
    marker = tmp_path / "side_effect.txt"
    name = "builtin.synthetic-model" if builtin_collision else "test.duplicate-model"
    module = "ca_duplicate_probe"
    (tmp_path / f"{module}.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nraise RuntimeError('must not import')\n")
    _install_plugin_metadata(tmp_path, "ca_probe_one", name, module)
    if not builtin_collision:
        _install_plugin_metadata(tmp_path, "ca_probe_two", name, module)
    monkeypatch.syspath_prepend(str(tmp_path))
    telemetry, exporter, _ = signals("plugin_discovery")
    with pytest.raises(InvalidConfigurationError, match="duplicate model plugin"):
        PluginDiscovery(telemetry).resolve("model", name, required_capabilities=frozenset(), required_methods=())
    assert not marker.exists() and module not in sys.modules
    assert exporter.get_finished_spans()[-1].attributes["capability_anatomy.outcome"] == "refused"


def test_unique_real_entrypoint_loads_and_executes(tmp_path, monkeypatch):
    module = "ca_unique_probe"
    (tmp_path / f"{module}.py").write_text("class Plugin:\n name = 'test.unique-model'\n version = '1'\n api_version = 'capability-anatomy/plugin-api/v1'\n capabilities = frozenset()\n def execute(self): return 'working'\n")
    from capability_anatomy.protocols import CORE_API_VERSION
    p = tmp_path / f"{module}.py"
    p.write_text(p.read_text().replace("capability-anatomy/plugin-api/v1", CORE_API_VERSION))
    _install_plugin_metadata(tmp_path, "ca_unique", "test.unique-model", module)
    monkeypatch.syspath_prepend(str(tmp_path))
    resolved = PluginDiscovery().resolve("model", "test.unique-model", required_capabilities=frozenset(), required_methods=("execute",))
    assert resolved.plugin.execute() == "working"
    sys.modules.pop(module, None)


@pytest.mark.parametrize("fragment", [
    "access_token=FRAGMENT_CANARY_ONLY_582793",
    "refresh_token=FRAGMENT_CANARY_ONLY_582793&token_type=bearer",
    "access%5Ftoken=FRAGMENT_CANARY_ONLY_582793",
    "?api_key=FRAGMENT_CANARY_ONLY_582793",
    "/callback?access_token=FRAGMENT_CANARY_ONLY_582793",
])
def test_public_run_refuses_credential_fragment_before_snapshot(tmp_path, monkeypatch, fragment):
    from capability_anatomy.execution import run_experiment
    value = yaml.safe_load(EXAMPLE.read_text())
    value["dataset"]["manifest"] = str(ROOT / "fixtures/synthetic/manifest.json")
    value["output"]["directory"] = str(tmp_path / "run")
    value["model"]["parameters"]["reference_url"] = "https://example.invalid/model#" + fragment
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    calls = []
    from capability_anatomy.models.plugins.synthetic import SyntheticModelAdapter
    original_load = SyntheticModelAdapter.load
    def observed_load(*args, **kwargs):
        calls.append("load")
        return original_load(*args, **kwargs)
    monkeypatch.setattr(SyntheticModelAdapter, "load", observed_load)
    with pytest.raises(InvalidConfigurationError, match="credential-bearing URL refused") as failure:
        run_experiment(path)
    assert "FRAGMENT_CANARY_ONLY_582793" not in str(failure.value)
    assert calls == []
    assert not (tmp_path / "run").exists()


def test_public_run_preserves_harmless_fragment_in_reproducibility_metadata(tmp_path):
    from capability_anatomy.execution import run_experiment
    value = yaml.safe_load(EXAMPLE.read_text())
    value["dataset"]["manifest"] = str(ROOT / "fixtures/synthetic/manifest.json")
    value["output"]["directory"] = str(tmp_path / "run")
    ordinary_url = "https://example.invalid/model#architecture-details"
    value["model"]["parameters"]["reference_url"] = ordinary_url
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    assert run_experiment(path)["status"] == "complete"
    snapshot = json.loads((tmp_path / "run/experiment-config.json").read_text())
    assert snapshot["model"]["parameters"]["reference_url"] == ordinary_url


def test_real_fifo_refused_without_waiting_for_writer_with_regular_file_control(tmp_path):
    import os
    import subprocess
    fifo = tmp_path / "config.yaml"
    os.mkfifo(fifo)
    code = """import sys,json
from pathlib import Path
from capability_anatomy.authored_inputs import load_mapping,AuthoredInputError
try:
 value=load_mapping(Path(sys.argv[1]))
except AuthoredInputError as error:
 print(json.dumps({'reason':error.reason}));raise SystemExit(2)
print(json.dumps(value))
"""
    # The subprocess timeout is a test failure if the old blocking open returns.
    result = subprocess.run([sys.executable, "-c", code, str(fifo)], capture_output=True, text=True, timeout=3)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout) == {"reason": "document_requires_regular_file"}
    positive = tmp_path / "regular.json"
    positive.write_text('{"answer":42}')
    result = subprocess.run([sys.executable, "-c", code, str(positive)], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"answer": 42}


def test_authored_directory_and_leaf_symlink_refused_with_static_signals(tmp_path):
    target = tmp_path / "target.json"
    target.write_text('{"answer":42}')
    link = tmp_path / "config.json"
    link.symlink_to(target)
    telemetry, exporter, _ = signals("authored_input")
    with pytest.raises(AuthoredInputError, match="document_unreadable"):
        load_mapping(link, telemetry=telemetry)
    with pytest.raises(AuthoredInputError, match="document_requires_regular_file"):
        load_mapping(tmp_path, telemetry=telemetry)
    assert all(span.attributes["capability_anatomy.outcome"] == "refused" for span in exporter.get_finished_spans())
    assert load_mapping(target) == {"answer": 42}


@pytest.mark.parametrize("url", [
    "//user:SCHEME_RELATIVE_CANARY@example.invalid/path",
    "//example.invalid/path#access_token=SCHEME_RELATIVE_CANARY",
    "//example.invalid/path?api_key=SCHEME_RELATIVE_CANARY",
])
def test_scheme_relative_credential_urls_are_refused(tmp_path, url):
    value = yaml.safe_load(EXAMPLE.read_text())
    value["model"]["parameters"]["reference_url"] = url
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(InvalidConfigurationError, match="credential-bearing URL refused"):
        load_experiment_config(path)


def test_public_run_dataset_fifo_refuses_before_model_load(tmp_path):
    import os
    import subprocess
    fifo = tmp_path / "dataset.json"
    os.mkfifo(fifo)
    value = yaml.safe_load(EXAMPLE.read_text())
    value["dataset"]["manifest"] = str(fifo)
    value["output"]["directory"] = str(tmp_path / "bad-run")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    marker = tmp_path / "model-loaded"
    code = """import sys
from pathlib import Path
from capability_anatomy.models.plugins.synthetic import SyntheticModelAdapter
from capability_anatomy.cli import main
original=SyntheticModelAdapter.load
def load(*args,**kwargs):
 Path(sys.argv[2]).write_text('loaded');return original(*args,**kwargs)
SyntheticModelAdapter.load=load
raise SystemExit(main(['run','--config',sys.argv[1]]))
"""
    denied = subprocess.run([sys.executable, "-c", code, str(path), str(marker)], capture_output=True, text=True, timeout=3)
    assert denied.returncode == 2, denied.stderr
    assert "document_requires_regular_file" in denied.stderr
    assert not marker.exists()
    value["dataset"]["manifest"] = str(ROOT / "fixtures/synthetic/manifest.json")
    value["output"]["directory"] = str(tmp_path / "good-run")
    path.write_text(yaml.safe_dump(value))
    accepted = subprocess.run([sys.executable, "-c", code, str(path), str(marker)], capture_output=True, text=True, timeout=3)
    assert accepted.returncode == 0, accepted.stderr
    assert marker.exists()
    assert json.loads(accepted.stdout)["status"] == "complete"


@pytest.mark.parametrize("consumer", ["synthetic", "phase5_manifest", "phase5_plan", "protocol", "record_source", "hash"])
def test_authored_consumers_refuse_fifo_at_actual_open(tmp_path, consumer):
    import os
    import subprocess
    fifo = tmp_path / "input.json"
    os.mkfifo(fifo)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"record_plan": str(fifo), "partitions": {}}))
    code = """import sys,json
from pathlib import Path
from capability_anatomy.authored_inputs import AuthoredInputError
from capability_anatomy.datasets.synthetic import SyntheticDatasetProvider
from capability_anatomy.datasets.phase5 import Phase5DatasetProvider
from capability_anatomy.phase5_protocol import validate_phase5_protocol,build_record_plan,sha256_file,KINDS
path=Path(sys.argv[1]);mode=sys.argv[2]
try:
 if mode=='synthetic': SyntheticDatasetProvider.from_manifest(path)
 elif mode=='phase5_manifest': Phase5DatasetProvider.from_manifest(path)
 elif mode=='phase5_plan': Phase5DatasetProvider.from_manifest(Path(sys.argv[3]))
 elif mode=='protocol': validate_phase5_protocol(path)
 elif mode=='record_source': build_record_plan(path,{kind:1 for kind in KINDS})
 elif mode=='hash': sha256_file(path)
except AuthoredInputError as error:
 print(json.dumps({'reason':error.reason}));raise SystemExit(2)
raise AssertionError('consumer unexpectedly accepted FIFO')
"""
    result = subprocess.run([sys.executable, "-c", code, str(fifo), consumer, str(manifest)], capture_output=True, text=True, timeout=3)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout) == {"reason": "document_requires_regular_file"}


def test_regular_record_plan_and_dataset_and_stream_hash_positive(tmp_path):
    import hashlib
    from capability_anatomy.datasets.phase5 import Phase5DatasetProvider
    from capability_anatomy.phase5_protocol import build_record_plan, KINDS, sha256_file
    rows = [{"source_id": f"{part}-{kind}", "group_key": f"group-{part}-{kind}", "kind": kind, "split": part}
            for part in ("discovery", "validation") for kind in KINDS]
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"rows": rows}))
    plan = build_record_plan(source, {kind: 1 for kind in KINDS})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    assert plan["source_manifest_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    records = {part: [{"id": row["source_id"], "partition": part, "input": {}, "expected": {},
                       "metadata": {"group_id": row["group_key"], "kind": row["kind"]}}
                      for row in rows if row["split"] == part] for part in ("discovery", "validation")}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"record_plan": "plan.json", "partitions": records}))
    provider = Phase5DatasetProvider.from_manifest(manifest)
    assert len(tuple(provider.records("discovery"))) == len(KINDS)
    assert provider.plan_sha256 == sha256_file(plan_path)
    large = tmp_path / "legacy-evidence.bin"
    content = b"regular historical evidence\n" * 350000
    assert len(content) > 8 * 1024 * 1024
    large.write_bytes(content)
    assert sha256_file(large) == hashlib.sha256(content).hexdigest()
