from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.cli import main
from capability_anatomy.discovery import PluginDiscovery
from capability_anatomy.errors import InvalidConfigurationError
from capability_anatomy.models.plugins.synthetic import SyntheticModelAdapter
from capability_anatomy.domain import ModelConfig, RuntimeConfig
from capability_anatomy.models.base import ExecutionRequest, GenerationRequest, LoadedModel
from capability_anatomy.models.plugins.huggingface import ARCHITECTURE_PROFILES, HuggingFaceCausalLMAdapter
from capability_anatomy.protocols import CORE_API_VERSION, RuntimePlugin
from capability_anatomy.runtime import LocalRuntimePlugin
from capability_anatomy.telemetry import OperationTelemetry


class ExternalModelPlugin(SyntheticModelAdapter):
    name = "external.fixture-model"


class _EntryPoint:
    name = ExternalModelPlugin.name
    dist = SimpleNamespace(name="external-fixture", version="3.2.1")

    @staticmethod
    def load():
        return ExternalModelPlugin


def _external_entry_points(*, group: str, name: str):
    if group == "capability_anatomy.models" and name == ExternalModelPlugin.name:
        return (_EntryPoint(),)
    return ()


def _config(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / "configs/examples/synthetic-scan.yaml").read_text())
    value["model"]["plugin"] = ExternalModelPlugin.name
    value["dataset"]["manifest"] = str(root / "fixtures/synthetic/manifest.json")
    value["output"]["directory"] = str(tmp_path / "run")
    path = tmp_path / "external.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def _external_runtime_config(tmp_path: Path) -> Path:
    path = _config(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["runtime"]["executor"] = "external.fixture-runtime"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_external_plugin_runs_without_core_or_orchestrator_change(tmp_path: Path, monkeypatch, capsys) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures/gate5m_external_plugin"
    monkeypatch.syspath_prepend(str(fixture))

    assert main(("run", "--config", str(_config(tmp_path)))) == 0
    capsys.readouterr()
    compatibility = json.loads((tmp_path / "run/compatibility.json").read_text())
    model = next(item for item in compatibility["discovered_plugins"] if item["role"] == "model")
    assert model["name"] == ExternalModelPlugin.name
    assert model["distribution"] == "external-gate5m-plugin"
    assert model["distribution_version"] == "3.2.1"
    assert compatibility["operation_capabilities"]["model"] == [
        "measurement.memory", "measurement.synchronization", "measurement.tokens", "model.topology.components",
    ]

    class ChangedEntryPoint(_EntryPoint):
        dist = SimpleNamespace(name="external-gate5m-plugin", version="3.2.2")

    monkeypatch.setattr(
        "capability_anatomy.discovery.metadata.entry_points",
        lambda *, group, name: (ChangedEntryPoint(),) if group == "capability_anatomy.models" and name == ExternalModelPlugin.name else (),
    )
    assert main(("run", "--config", str(_config(tmp_path)))) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "storage_frozen_input_changed"


def test_external_nonlocal_runtime_is_selected_and_used_by_configuration(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures/gate5m_external_plugin"
    monkeypatch.syspath_prepend(str(fixture))
    config = _external_runtime_config(tmp_path)

    from external_gate5m_plugin import ExternalRuntimePlugin

    ExternalRuntimePlugin.calls = dict.fromkeys(ExternalRuntimePlugin.calls, 0)
    assert isinstance(ExternalRuntimePlugin(), RuntimePlugin)
    assert "runtime.local" not in ExternalRuntimePlugin.capabilities
    assert main(("run", "--config", str(config))) == 0
    capsys.readouterr()
    assert all(count > 0 for count in ExternalRuntimePlugin.calls.values())

    compatibility = json.loads((tmp_path / "run/compatibility.json").read_text())
    runtime = next(item for item in compatibility["discovered_plugins"] if item["role"] == "runtime")
    assert runtime["name"] == "external.fixture-runtime"
    assert runtime["distribution"] == "external-gate5m-plugin"
    assert runtime["capabilities"] == sorted(ExternalRuntimePlugin.capabilities)
    assert compatibility["operation_capabilities"]["runtime"] == [
        "runtime.execute", "runtime.memory", "runtime.synchronization",
        "runtime.timing", "runtime.tokens",
    ]
    assert "runtime.fixture.remote" not in compatibility["operation_capabilities"]["runtime"]


@pytest.mark.parametrize("broken_contract", ["capability", "method"])
def test_external_runtime_contract_refuses_before_model_load(
    tmp_path: Path, monkeypatch, capsys, broken_contract: str,
) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures/gate5m_external_plugin"
    monkeypatch.syspath_prepend(str(fixture))
    from external_gate5m_plugin import ExternalModelPlugin, ExternalRuntimePlugin

    model_loads = 0

    def forbidden_model_load(self, spec, runtime):
        nonlocal model_loads
        model_loads += 1
        raise AssertionError("model loading crossed runtime negotiation")

    monkeypatch.setattr(ExternalModelPlugin, "load", forbidden_model_load)
    if broken_contract == "capability":
        monkeypatch.setattr(
            ExternalRuntimePlugin,
            "capabilities",
            ExternalRuntimePlugin.capabilities - {"runtime.tokens"},
        )
        expected = "lacks required capabilities"
    else:
        monkeypatch.setattr(ExternalRuntimePlugin, "token_counts", None)
        expected = "interface is incomplete"

    assert main(("run", "--config", str(_external_runtime_config(tmp_path)))) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_configuration"
    assert model_loads == 0


def test_external_runtime_distribution_drift_is_refused_on_resume(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures/gate5m_external_plugin"
    monkeypatch.syspath_prepend(str(fixture))
    config = _external_runtime_config(tmp_path)
    assert main(("run", "--config", str(config))) == 0
    capsys.readouterr()

    from external_gate5m_plugin import ExternalRuntimePlugin
    from capability_anatomy.discovery import metadata as discovery_metadata

    original_entry_points = discovery_metadata.entry_points

    class ChangedRuntimeEntryPoint:
        name = ExternalRuntimePlugin.name
        dist = SimpleNamespace(name="external-gate5m-plugin", version="3.2.2")

        @staticmethod
        def load():
            return ExternalRuntimePlugin

    def changed_entry_points(*, group: str, name: str):
        if group == "capability_anatomy.runtimes" and name == ExternalRuntimePlugin.name:
            return (ChangedRuntimeEntryPoint(),)
        return original_entry_points(group=group, name=name)

    monkeypatch.setattr(discovery_metadata, "entry_points", changed_entry_points)
    assert main(("run", "--config", str(config))) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "storage_frozen_input_changed"


def test_duplicate_discovered_name_is_rejected(monkeypatch) -> None:
    class Duplicate(_EntryPoint):
        name = "builtin.synthetic-model"

    monkeypatch.setattr(
        "capability_anatomy.discovery.metadata.entry_points",
        lambda *, group, name: (Duplicate(),) if group == "capability_anatomy.models" and name == Duplicate.name else (),
    )
    with pytest.raises(InvalidConfigurationError, match="duplicate model plugin"):
        PluginDiscovery().resolve(
            "model", Duplicate.name,
            required_capabilities=frozenset(), required_methods=("load",),
        )


def test_discovery_decision_is_traceable_and_redacted() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    telemetry = OperationTelemetry.create(
        "plugin_discovery",
        tracer=provider.get_tracer("gate5m-test"),
        meter=MeterProvider(metric_readers=[reader]).get_meter("gate5m-test"),
    )
    secret_name = "missing.secret-plugin"
    with pytest.raises(Exception):
        PluginDiscovery(telemetry).resolve(
            "model", secret_name,
            required_capabilities=frozenset(), required_methods=(),
        )
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"
    assert span.events[0].attributes["capability_anatomy.reason"] == "plugin_contract_refused"
    assert secret_name not in repr(span)


@pytest.mark.parametrize(
    ("api_version", "capabilities", "message"),
    [("old", ExternalModelPlugin.capabilities, "API version"),
     (ExternalModelPlugin.api_version, frozenset(), "lacks required capabilities")],
)
def test_discovery_rejects_api_or_capabilities_before_load(
    monkeypatch, api_version: str, capabilities: frozenset[str], message: str
) -> None:
    loaded = False

    class InvalidExternal(ExternalModelPlugin):
        pass

    InvalidExternal.api_version = api_version
    InvalidExternal.capabilities = capabilities

    def forbidden_load(self, spec, runtime):
        nonlocal loaded
        loaded = True
        raise AssertionError

    InvalidExternal.load = forbidden_load

    class InvalidEntryPoint(_EntryPoint):
        @staticmethod
        def load():
            return InvalidExternal

    monkeypatch.setattr(
        "capability_anatomy.discovery.metadata.entry_points",
        lambda *, group, name: (InvalidEntryPoint(),) if group == "capability_anatomy.models" and name == InvalidExternal.name else (),
    )
    with pytest.raises(InvalidConfigurationError, match=message):
        PluginDiscovery().resolve(
            "model", InvalidExternal.name,
            required_capabilities=frozenset({"measurement.tokens"}), required_methods=("load",),
        )
    assert loaded is False


def test_orchestrator_rejects_callable_topology_without_declared_capability(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    loaded = False

    class UndeclaredTopologyPlugin(ExternalModelPlugin):
        capabilities = ExternalModelPlugin.capabilities - {"model.topology.components"}

        def load(self, spec, runtime):
            nonlocal loaded
            loaded = True
            return super().load(spec, runtime)

    class UndeclaredEntryPoint(_EntryPoint):
        @staticmethod
        def load():
            return UndeclaredTopologyPlugin

    monkeypatch.setattr(
        "capability_anatomy.discovery.metadata.entry_points",
        lambda *, group, name: (UndeclaredEntryPoint(),)
        if group == "capability_anatomy.models" and name == ExternalModelPlugin.name else (),
    )
    assert main(("run", "--config", str(_config(tmp_path)))) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_configuration"
    assert callable(UndeclaredTopologyPlugin().topology)
    assert loaded is False


def test_public_runner_uses_runtime_for_every_operational_boundary(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures/gate5m_external_plugin"
    monkeypatch.syspath_prepend(str(fixture))
    calls = {name: 0 for name in ("execute", "synchronize", "memory", "tokens", "clock")}
    originals = {
        "execute": LocalRuntimePlugin.execute,
        "synchronize": LocalRuntimePlugin.synchronize,
        "memory": LocalRuntimePlugin.memory_bytes,
        "tokens": LocalRuntimePlugin.token_counts,
    }

    def execute(self, adapter, loaded, request):
        calls["execute"] += 1
        return originals["execute"](self, adapter, loaded, request)

    def synchronize(adapter, loaded):
        calls["synchronize"] += 1
        return originals["synchronize"](adapter, loaded)

    def memory_bytes(adapter, loaded):
        calls["memory"] += 1
        return originals["memory"](adapter, loaded)

    def token_counts(adapter, loaded, request, result):
        calls["tokens"] += 1
        return originals["tokens"](adapter, loaded, request, result)

    def clock():
        calls["clock"] += 1
        return float(calls["clock"])

    monkeypatch.setattr(LocalRuntimePlugin, "execute", execute)
    monkeypatch.setattr(LocalRuntimePlugin, "synchronize", staticmethod(synchronize))
    monkeypatch.setattr(LocalRuntimePlugin, "memory_bytes", staticmethod(memory_bytes))
    monkeypatch.setattr(LocalRuntimePlugin, "token_counts", staticmethod(token_counts))
    monkeypatch.setattr(LocalRuntimePlugin, "clock", staticmethod(clock))

    assert main(("run", "--config", str(_config(tmp_path)))) == 0
    capsys.readouterr()
    assert all(count > 0 for count in calls.values())


def test_runtime_execution_decision_is_traceable() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    telemetry = OperationTelemetry.create(
        "runtime",
        tracer=provider.get_tracer("gate5m-runtime-test"),
        meter=MeterProvider(metric_readers=[reader]).get_meter("gate5m-runtime-test"),
    )

    class Adapter:
        @staticmethod
        def execute(loaded, request):
            return "result"

    assert LocalRuntimePlugin(telemetry).execute(Adapter(), object(), object()) == "result"
    span = exporter.get_finished_spans()[0]
    assert span.name == "capability_anatomy.runtime.execute"
    assert span.events[0].attributes["capability_anatomy.operation"] == "execute"
    assert span.events[0].attributes["capability_anatomy.reason"] == "adapter_execution_complete"


def test_orchestrator_has_no_concrete_model_family_import() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/capability_anatomy/execution/orchestrator.py").read_text()
    assert "models.plugins.qwen3" not in source
    assert "Qwen3Adapter" not in source


def test_core_and_orchestrator_import_without_model_libraries() -> None:
    root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        [sys.executable, "-c", "import sys; import capability_anatomy; import capability_anatomy.execution.orchestrator; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
        cwd=root,
        env={"PYTHONPATH": str(root / "src")},
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr


class _LlamaBlock(torch.nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.self_attn = SimpleNamespace(layer_idx=index)
        self.mlp = torch.nn.Identity()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden, cache="cache"):
        return hidden + self.weight, cache


class _LlamaFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama")
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList((_LlamaBlock(0), _LlamaBlock(1)))
        self.generated = False

    def generate(self, **kwargs):
        self.generated = True
        return torch.cat((kwargs["input_ids"], torch.tensor([[8, 9]])), dim=1)


class _Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "rendered llama prompt"

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tokens, **kwargs):
        return "decoded"


def test_second_family_profile_generation_measurement_and_bypass() -> None:
    adapter = HuggingFaceCausalLMAdapter()
    profile = adapter._profile_for_spec(ModelConfig(
        "huggingface.causal-lm", "fixture", "a" * 40,
        {"architecture_profile": "llama-like-v1", "dtype": "float32"},
    ))
    model = _LlamaFixture()
    loaded = LoadedModel(model, "fixture", "a" * 40, {
        "tokenizer": _Tokenizer(), "context_length": 32, "architecture_profile": profile,
    })
    topology = adapter.topology(loaded)
    request = ExecutionRequest(GenerationRequest(({"role": "user", "content": "x"},), None, 2))

    assert topology.architecture == "llama-like"
    assert topology.components[0].metadata["architecture_profile"] == "llama-like-v1"
    assert len(topology.components[0].metadata["architecture_profile_sha256"]) == 64
    result = adapter.execute(loaded, request)
    assert result == "decoded"
    assert adapter.token_counts(loaded, request, result) == (3, 2)

    from capability_anatomy.models.plugins.huggingface import HuggingFaceBlockBypassProvider
    from capability_anatomy.interventions import BlockBypass
    baseline = model.model.layers[0](torch.tensor([1.0]))[0]
    with BlockBypass(("transformer.block.000",)).apply(HuggingFaceBlockBypassProvider(), adapter, loaded):
        bypassed = model.model.layers[0](torch.tensor([1.0]))[0]
    assert baseline.item() == 2.0
    assert bypassed.item() == 1.0
    assert not model.model.layers[0]._forward_hooks


def test_second_family_fixture_produces_complete_public_evidence(tmp_path: Path, monkeypatch, capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / "configs/examples/synthetic-scan.yaml").read_text())
    value["model"] = {
        "plugin": "huggingface.causal-lm", "source": "local/llama-fixture",
        "revision": "a" * 40,
        "parameters": {"architecture_profile": "llama-like-v1", "dtype": "float32"},
    }
    value["runtime"]["parameters"] = {"device": "cpu", "context_length": 32}
    value["runtime"]["warmup_runs"] = 0
    value["capability"]["evaluation_plugin"] = "reference.chat-exact-match"
    value["intervention"]["plugin"] = "huggingface.block-bypass"
    value["intervention"]["parameters"] = {"component_ids": ["transformer.block.000"]}
    manifest = tmp_path / "llama-records.json"
    rows = {
        partition: [{
            "id": f"{partition}-1", "partition": partition,
            "input": {"messages": [{"role": "user", "content": "hello"}], "max_new_tokens": 2},
            "expected": "decoded", "metadata": {},
        }]
        for partition in ("discovery", "validation")
    }
    manifest.write_text(json.dumps({"license": "fixture", "revision": "v1", "partitions": rows}), encoding="utf-8")
    value["dataset"]["manifest"] = str(manifest)
    value["output"]["directory"] = str(tmp_path / "run")
    config = tmp_path / "llama.yaml"
    config.write_text(yaml.safe_dump(value), encoding="utf-8")

    def fixture_load(self, spec, runtime):
        profile = self._profile_for_spec(spec)
        return LoadedModel(_LlamaFixture(), spec.source, spec.revision, {
            "tokenizer": _Tokenizer(), "context_length": 32, "architecture_profile": profile,
        })

    monkeypatch.setattr(HuggingFaceCausalLMAdapter, "load", fixture_load)
    assert main(("run", "--config", str(config))) == 0
    capsys.readouterr()
    baseline = json.loads((tmp_path / "run/tasks/baseline.json").read_text())
    compatibility = json.loads((tmp_path / "run/compatibility.json").read_text())
    assert baseline["complete"] is True
    assert baseline["metrics"]["metrics"] == {"exact_match": 1.0}
    assert compatibility["model"]["implementation_metadata"]["architecture_profile"] == "llama-like-v1"
    assert compatibility["model"]["implementation_metadata"]["model_type"] == "llama"
    assert compatibility["model"]["architecture"] == "llama-like"
    assert compatibility["resolved_topology"]["architecture"] == "llama-like"
    assert [item["id"] for item in compatibility["resolved_topology"]["components"]] == [
        "transformer.block.000", "transformer.block.001",
    ]


def test_resolved_topology_drift_refuses_completed_run_resume(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / "configs/examples/synthetic-scan.yaml").read_text())
    value["model"] = {
        "plugin": "huggingface.causal-lm", "source": "local/llama-fixture",
        "revision": "a" * 40,
        "parameters": {"architecture_profile": "llama-like-v1", "dtype": "float32"},
    }
    value["runtime"]["parameters"] = {"device": "cpu", "context_length": 32}
    value["runtime"]["warmup_runs"] = 0
    value["capability"]["evaluation_plugin"] = "reference.chat-exact-match"
    value["intervention"]["plugin"] = "huggingface.block-bypass"
    value["intervention"]["parameters"] = {"component_ids": "all"}
    manifest = tmp_path / "records.json"
    rows = {partition: [{
        "id": f"{partition}-1", "partition": partition,
        "input": {"messages": [{"role": "user", "content": "hello"}], "max_new_tokens": 2},
        "expected": "decoded", "metadata": {},
    }] for partition in ("discovery", "validation")}
    manifest.write_text(json.dumps({"license": "fixture", "revision": "v1", "partitions": rows}), encoding="utf-8")
    value["dataset"]["manifest"] = str(manifest)
    value["output"]["directory"] = str(tmp_path / "run")
    config = tmp_path / "topology.yaml"
    config.write_text(yaml.safe_dump(value), encoding="utf-8")
    layer_count = 2

    def fixture_load(self, spec, runtime):
        model = _LlamaFixture()
        model.model.layers = torch.nn.ModuleList(tuple(_LlamaBlock(index) for index in range(layer_count)))
        return LoadedModel(model, spec.source, spec.revision, {
            "tokenizer": _Tokenizer(), "context_length": 32,
            "architecture_profile": self._profile_for_spec(spec),
        })

    monkeypatch.setattr(HuggingFaceCausalLMAdapter, "load", fixture_load)
    assert main(("run", "--config", str(config))) == 0
    capsys.readouterr()
    layer_count = 1
    assert main(("run", "--config", str(config))) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "storage_frozen_input_changed"


def test_profiles_fail_closed_on_family_mismatch() -> None:
    adapter = HuggingFaceCausalLMAdapter()
    loaded = LoadedModel(_LlamaFixture(), "fixture", "a" * 40, {
        "tokenizer": _Tokenizer(), "context_length": 32,
        "architecture_profile": ARCHITECTURE_PROFILES["qwen3-dense-v1"],
    })
    with pytest.raises(Exception, match="does not match architecture profile"):
        adapter.topology(loaded)

    with pytest.raises(Exception, match="profile is unknown"):
        adapter._profile_for_spec(ModelConfig(
            "huggingface.causal-lm", "fixture", "a" * 40,
            {"architecture_profile": "guessed-profile", "dtype": "float32"},
        ))


def test_profile_family_mismatch_refuses_before_weight_load(monkeypatch) -> None:
    import transformers
    weight_loads = 0
    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", lambda *_args, **_kwargs: ({"model_type": "qwen3"}, {}))

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="qwen3"),
    )

    def load_weights(*args, **kwargs):
        nonlocal weight_loads
        weight_loads += 1
        raise AssertionError("weights loaded")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_weights)
    spec = ModelConfig(
        "huggingface.causal-lm", "fixture", "a" * 40,
        {"architecture_profile": "llama-like-v2", "dtype": "float32"},
    )
    runtime = RuntimeConfig("builtin.local", True, 0, True, 1, {"device": "cpu", "context_length": 32})
    with pytest.raises(Exception, match="does not match architecture profile"):
        HuggingFaceCausalLMAdapter().load(spec, runtime)
    assert weight_loads == 0
