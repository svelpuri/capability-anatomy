from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.domain import ModelConfig, RuntimeConfig
from capability_anatomy.errors import InvalidConfigurationError, UnsupportedAdapterError
from capability_anatomy.interventions import BlockBypass
from capability_anatomy.models.base import AdapterProvenance, ComponentHandle, ExecutionRequest, LoadedModel, ModelTopology, PerplexityRequest
from capability_anatomy.models.plugins.qwen3 import Qwen3Adapter, Qwen3BlockBypassProvider, QwenGenerationRequest
from capability_anatomy.models.plugins.huggingface import ARCHITECTURE_PROFILES
from capability_anatomy.protocols import CORE_API_VERSION, MeasuredModelAdapter, ModelAdapter
from capability_anatomy.registry import PluginRegistry
from capability_anatomy.telemetry import OperationTelemetry


class FakeAttention:
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx


class FakeBlock(torch.nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = FakeAttention(layer_idx)
        self.mlp = torch.nn.Identity()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, hidden: torch.Tensor, cache: str = "cache") -> tuple[torch.Tensor, str]:
        self.calls += 1
        return hidden + self.weight, cache


class FakeModel(torch.nn.Module):
    def __init__(self, layer_count: int = 2, model_type: str = "qwen3") -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(FakeBlock(index) for index in range(layer_count))
        self.generate_kwargs: dict | None = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.cat((kwargs["input_ids"], torch.tensor([[8, 9]])), dim=1)


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.template_kwargs: dict | None = None

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.template_kwargs = {"messages": messages, **kwargs}
        return "rendered"

    def __call__(self, _rendered: str, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tokens, **_kwargs) -> str:
        assert tokens.tolist() == [8, 9]
        return "decoded"


def _loaded(model: FakeModel | None = None) -> LoadedModel:
    return LoadedModel(
        model=model or FakeModel(),
        source="Qwen/Qwen3-fixture",
        revision="a" * 40,
        resources={"tokenizer": FakeTokenizer(), "context_length": 32},
    )


def _forward(model: FakeModel, value: float) -> tuple[torch.Tensor, str]:
    output: tuple[torch.Tensor, str] = (torch.tensor([value]), "cache")
    for layer in model.model.layers:
        output = layer(output[0], output[1])
    return output


def _signals() -> tuple[OperationTelemetry, InMemorySpanExporter, InMemoryMetricReader]:
    exporter = InMemorySpanExporter()
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    telemetry = OperationTelemetry.create(
        "intervention",
        tracer=trace_provider.get_tracer("test"),
        meter=meter_provider.get_meter("test"),
    )
    return telemetry, exporter, reader


def test_qwen_topology_is_stable_and_fails_closed() -> None:
    adapter = Qwen3Adapter()
    loaded = _loaded()

    topology = adapter.topology(loaded)
    assert topology.architecture == "qwen3-dense"
    assert tuple(component.id for component in topology.components) == (
        "transformer.block.000",
        "transformer.block.001",
    )
    assert topology.component("transformer.block.001").order == 1
    assert topology.component("transformer.block.001").parent_id is None
    assert topology.component("transformer.block.001").metadata == {
        "architecture": "qwen3-dense",
        "architecture_profile": "qwen3-dense-v2",
        "architecture_profile_sha256": ARCHITECTURE_PROFILES["qwen3-dense-v2"].sha256,
        "module_path": "model.layers.1",
        "original_order": 1,
        "parameter_count": 1,
    }

    stale = _loaded()
    stale.model.model.layers[1].self_attn.layer_idx = 0
    with pytest.raises(UnsupportedAdapterError, match="stale or ambiguous"):
        adapter.topology(stale)

    wrong_family = _loaded(FakeModel(model_type="qwen2"))
    with pytest.raises(UnsupportedAdapterError, match="does not match architecture profile"):
        adapter.topology(wrong_family)


def test_generic_topology_supports_hierarchy_and_unordered_components() -> None:
    topology = ModelTopology(
        architecture="mixture-graph",
        components=(
            ComponentHandle("layer.a", "layer", object()),
            ComponentHandle("layer.a/head.z", "attention_head", object(), parent_id="layer.a"),
            ComponentHandle("expert.free", "expert", object()),
        ),
    )
    assert topology.component("layer.a/head.z").parent_id == "layer.a"
    assert topology.component("expert.free").order is None

    with pytest.raises(ValueError, match="acyclic"):
        ModelTopology(
            architecture="cycle",
            components=(
                ComponentHandle("a", "node", object(), parent_id="b"),
                ComponentHandle("b", "node", object(), parent_id="a"),
            ),
        )


def test_scoped_bypass_executes_block_on_prefill_and_decode_and_preserves_cache() -> None:
    adapter = Qwen3Adapter()
    provider = Qwen3BlockBypassProvider()
    loaded = _loaded()
    intervention = BlockBypass(("transformer.block.000",))

    baseline, baseline_cache = _forward(loaded.model, 1.0)
    with intervention.apply(provider, adapter, loaded):
        prefill, prefill_cache = _forward(loaded.model, 1.0)
        decode, decode_cache = _forward(loaded.model, 2.0)

    restored, restored_cache = _forward(loaded.model, 1.0)
    assert baseline.item() == 3.0
    assert prefill.item() == 2.0
    assert decode.item() == 3.0
    assert restored.item() == baseline.item()
    assert {baseline_cache, prefill_cache, decode_cache, restored_cache} == {"cache"}
    assert loaded.model.model.layers[0].calls == 4
    assert not loaded.model.model.layers[0]._forward_hooks
    with intervention.apply(provider, adapter, loaded):
        assert loaded.model.model.layers[0]._forward_hooks
    assert not loaded.model.model.layers[0]._forward_hooks




def test_hooks_are_removed_after_body_failure_and_intervention_can_be_reused() -> None:
    adapter = Qwen3Adapter()
    provider = Qwen3BlockBypassProvider()
    loaded = _loaded()
    intervention = BlockBypass(("transformer.block.000",))

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with intervention.apply(provider, adapter, loaded):
            assert loaded.model.model.layers[0]._forward_hooks
            raise RuntimeError("synthetic failure")

    assert not loaded.model.model.layers[0]._forward_hooks


@pytest.mark.parametrize("error", [ValueError("parse failed"), KeyboardInterrupt()])
def test_hooks_are_removed_after_parser_error_or_interruption(error: BaseException) -> None:
    adapter = Qwen3Adapter()
    provider = Qwen3BlockBypassProvider()
    loaded = _loaded()

    with pytest.raises(type(error)):
        with BlockBypass(("transformer.block.000",)).apply(provider, adapter, loaded):
            raise error
    assert not loaded.model.model.layers[0]._forward_hooks


def test_nested_and_duplicate_bypasses_are_refused_without_leaking_hooks() -> None:
    adapter = Qwen3Adapter()
    provider = Qwen3BlockBypassProvider()
    loaded = _loaded()
    intervention = BlockBypass(("transformer.block.000",))

    with intervention.apply(provider, adapter, loaded):
        with pytest.raises(InvalidConfigurationError, match="already active"):
            with intervention.apply(provider, adapter, loaded):
                pass
        with pytest.raises(InvalidConfigurationError, match="already active"):
            with BlockBypass(("transformer.block.000",)).apply(provider, adapter, loaded):
                pass
        assert len(loaded.model.model.layers[0]._forward_hooks) == 1

    assert not loaded.model.model.layers[0]._forward_hooks
    with pytest.raises(InvalidConfigurationError, match="non-empty and unique"):
        BlockBypass(("transformer.block.000", "transformer.block.000")).validate(provider, adapter, loaded)


def test_bypass_rejects_unsupported_output_and_still_cleans_up() -> None:
    class InvalidOutputBlock(FakeBlock):
        def forward(self, hidden: torch.Tensor, cache: str = "cache") -> list[torch.Tensor]:
            return [hidden]

    loaded = _loaded()
    loaded.model.model.layers[0] = InvalidOutputBlock(0)
    adapter = Qwen3Adapter()
    provider = Qwen3BlockBypassProvider()

    with pytest.raises(RuntimeError, match="unsupported output structure"):
        with BlockBypass(("transformer.block.000",)).apply(provider, adapter, loaded):
            loaded.model.model.layers[0](torch.tensor([1.0]))
    assert not loaded.model.model.layers[0]._forward_hooks


def test_hook_registration_failure_does_not_poison_component() -> None:
    class RegistrationFailureBlock(FakeBlock):
        def register_forward_hook(self, hook, *args, **kwargs):
            raise RuntimeError("registration failed")

    loaded = _loaded()
    loaded.model.model.layers[0] = RegistrationFailureBlock(0)
    adapter = Qwen3Adapter()
    provider = Qwen3BlockBypassProvider()
    intervention = BlockBypass(("transformer.block.000",))

    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="registration failed"):
            with intervention.apply(provider, adapter, loaded):
                pass


def test_model_adapter_does_not_own_bypass_implementation() -> None:
    assert "bypass_component" not in ModelAdapter.__dict__
    assert not hasattr(Qwen3Adapter, "bypass_component")
    assert "intervention.block_bypass" not in Qwen3Adapter.capabilities
    assert "intervention.block_bypass" in Qwen3BlockBypassProvider.capabilities


def test_qwen_declares_and_implements_public_measurement_contract() -> None:
    adapter = Qwen3Adapter()
    loaded = _loaded()
    request = ExecutionRequest(
        QwenGenerationRequest(messages=({"role": "user", "content": "x"},), tools=None, max_new_tokens=2)
    )

    assert isinstance(adapter, MeasuredModelAdapter)
    assert {
        "measurement.memory", "measurement.synchronization", "measurement.tokens"
    } <= adapter.capabilities
    adapter.synchronize(loaded)
    assert adapter.memory_bytes(loaded) is None
    result = adapter.execute(loaded, request)
    assert adapter.token_counts(loaded, request, result) == (3, 2)


def test_cleanup_failure_releases_hook_and_provider_ownership() -> None:
    failures = iter((RuntimeError("cleanup failed"), None))

    def cleanup_probe() -> None:
        error = next(failures)
        if error is not None:
            raise error

    adapter = Qwen3Adapter()
    provider = Qwen3BlockBypassProvider(cleanup_probe=cleanup_probe)
    loaded = _loaded()
    intervention = BlockBypass(("transformer.block.000",))

    with pytest.raises(RuntimeError, match="cleanup failed"):
        with intervention.apply(provider, adapter, loaded):
            assert loaded.model.model.layers[0]._forward_hooks
    assert not loaded.model.model.layers[0]._forward_hooks
    with intervention.apply(provider, adapter, loaded):
        assert loaded.model.model.layers[0]._forward_hooks
    assert not loaded.model.model.layers[0]._forward_hooks


def test_qwen_native_render_generate_and_provenance_are_bound() -> None:
    adapter = Qwen3Adapter()
    loaded = _loaded()
    request = ExecutionRequest(
        QwenGenerationRequest(
            messages=({"role": "user", "content": "use the tool"},),
            tools=({"name": "lookup"},),
            max_new_tokens=4,
        )
    )

    assert adapter.execute(loaded, request) == "decoded"
    tokenizer = loaded.resources["tokenizer"]
    assert tokenizer.template_kwargs["enable_thinking"] is False
    assert tokenizer.template_kwargs["tools"] == [{"name": "lookup"}]
    assert loaded.model.generate_kwargs["do_sample"] is False
    assert loaded.model.generate_kwargs["max_new_tokens"] == 4

    provenance = adapter.provenance(loaded)
    assert provenance.plugin == "reference.qwen3-transformers"
    assert provenance.api_version == CORE_API_VERSION
    assert provenance.model_revision == "a" * 40
    assert set(provenance.libraries) == {"python", "torch", "transformers"}
    assert provenance.implementation_metadata == {
        "tokenizer_class": "FakeTokenizer",
        "model_type": "qwen3",
        "architecture_profile": "qwen3-dense-v2",
        "architecture_profile_sha256": ARCHITECTURE_PROFILES["qwen3-dense-v2"].sha256,
    }


def test_qwen_perplexity_uses_teacher_forced_loss_and_zero_output_tokens() -> None:
    class PerplexityModel(FakeModel):
        def forward(self, **kwargs):
            assert torch.equal(kwargs["labels"], kwargs["input_ids"])
            return SimpleNamespace(loss=torch.tensor(2.0).log())

    adapter = Qwen3Adapter()
    loaded = _loaded(PerplexityModel())
    request = ExecutionRequest(PerplexityRequest("alpha beta"))

    assert adapter.execute(loaded, request) == pytest.approx(2.0)
    assert adapter.token_counts(loaded, request, 2.0) == (3, 0)


def test_core_model_contract_supports_a_tokenizer_free_adapter() -> None:
    class SyntheticClassifierAdapter:
        name = "test.synthetic-classifier"
        version = "1"
        api_version = CORE_API_VERSION
        architecture = "linear-classifier"
        capabilities = frozenset({"execution.classification"})

        def load(self, spec: ModelConfig, runtime: RuntimeConfig) -> LoadedModel:
            return LoadedModel(model=lambda value: value > 0, source=spec.source, revision=spec.revision)

        def topology(self, loaded: LoadedModel) -> ModelTopology:
            return ModelTopology(
                architecture=self.architecture,
                components=(ComponentHandle("classifier", "classifier", loaded.model),),
            )

        def execute(self, loaded: LoadedModel, request: ExecutionRequest) -> bool:
            return loaded.model(request.payload)

        def provenance(self, loaded: LoadedModel) -> AdapterProvenance:
            return AdapterProvenance(
                plugin=self.name,
                plugin_version=self.version,
                api_version=self.api_version,
                architecture=self.architecture,
                model_source=loaded.source,
                model_revision=loaded.revision,
                model_class="synthetic_callable",
                implementation_metadata={"task": "binary_classification"},
                libraries={"python": "test"},
            )

    adapter = SyntheticClassifierAdapter()
    loaded = adapter.load(
        ModelConfig(plugin=adapter.name, source="fixture", revision="v1", parameters={}),
        RuntimeConfig(
            executor="local",
            deterministic=True,
            warmup_runs=0,
            randomized_execution_order=False,
            repetitions=1,
            parameters={},
        ),
    )

    assert loaded.resources == {}
    assert adapter.execute(loaded, ExecutionRequest(1)) is True
    assert adapter.provenance(loaded).implementation_metadata == {"task": "binary_classification"}


def test_qwen_requires_its_plugin_owned_tokenizer_resource() -> None:
    loaded = LoadedModel(model=FakeModel(), source="fixture", revision="v1")
    with pytest.raises(UnsupportedAdapterError, match="tokenizer resource is required"):
        Qwen3Adapter().execute(
            loaded,
            ExecutionRequest(QwenGenerationRequest(messages=(), tools=None, max_new_tokens=1)),
        )


def test_plugin_registry_rejects_incompatible_duplicate_and_underpowered_plugins() -> None:
    adapter = Qwen3Adapter()
    registry = PluginRegistry(
        required_capabilities=frozenset({"execution.autoregressive_text"}),
        required_methods=("load", "topology", "execute"),
    )
    registry.register(adapter)
    assert registry.resolve(adapter.name) is adapter
    assert registry.names() == (adapter.name,)

    with pytest.raises(InvalidConfigurationError, match="already registered"):
        registry.register(adapter)
    incompatible = SimpleNamespace(
        name="reference.incompatible",
        version="1",
        api_version="old-api",
        capabilities=frozenset({"execution.autoregressive_text"}),
    )
    with pytest.raises(InvalidConfigurationError, match="API version"):
        registry.register(incompatible)
    underpowered = SimpleNamespace(
        name="reference.underpowered",
        version="1",
        api_version=CORE_API_VERSION,
        capabilities=frozenset(),
    )
    with pytest.raises(InvalidConfigurationError, match="lacks required"):
        registry.register(underpowered)
    incomplete = SimpleNamespace(
        name="reference.incomplete",
        version="1",
        api_version=CORE_API_VERSION,
        capabilities=frozenset({"execution.autoregressive_text"}),
    )
    with pytest.raises(InvalidConfigurationError, match="interface is incomplete"):
        registry.register(incomplete)


def test_qwen_loader_disables_remote_code_and_binds_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    calls: list[tuple[str, str, dict]] = []
    model = FakeModel()
    tokenizer = FakeTokenizer()

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs):
            calls.append(("tokenizer", source, kwargs))
            return tokenizer

    class ConfigFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs):
            calls.append(("config", source, kwargs))
            return SimpleNamespace(model_type="qwen3")

    class ModelFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs):
            calls.append(("model", source, kwargs))
            return model

    monkeypatch.setattr(transformers, "AutoTokenizer", TokenizerFactory)
    monkeypatch.setattr(transformers, "AutoConfig", ConfigFactory)
    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", lambda *_args, **_kwargs: ({"model_type": "qwen3"}, {}))
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", ModelFactory)
    spec = ModelConfig(
        plugin="reference.qwen3-transformers",
        source="Qwen/Qwen3-fixture",
        revision="b" * 40,
        parameters={"dtype": "float32"},
    )
    runtime = RuntimeConfig(
        executor="builtin.local",
        deterministic=True,
        warmup_runs=0,
        randomized_execution_order=True,
        repetitions=1,
        parameters={"device": "cpu", "context_length": 32},
    )

    loaded = Qwen3Adapter().load(spec, runtime)
    assert loaded.model is model
    assert loaded.model.training is False
    assert [kind for kind, _source, _kwargs in calls] == ["config", "tokenizer", "model"]
    assert all(source == spec.source for _kind, source, _kwargs in calls)
    assert all(kwargs["revision"] == spec.revision for _kind, _source, kwargs in calls)
    assert all(kwargs["trust_remote_code"] is False for _kind, _source, kwargs in calls)


def test_qwen_refuses_rendered_prompt_plus_generation_over_context_limit() -> None:
    loaded = _loaded()
    loaded = LoadedModel(loaded.model, loaded.source, loaded.revision, {**loaded.resources, "context_length": 4})
    request = ExecutionRequest(
        QwenGenerationRequest(messages=({"role": "user", "content": "x"},), tools=None, max_new_tokens=2)
    )

    with pytest.raises(InvalidConfigurationError, match="frozen context length"):
        Qwen3Adapter().execute(loaded, request)
    assert loaded.model.generate_kwargs is None


def test_intervention_lifecycle_is_visible_and_redacted() -> None:
    telemetry, exporter, reader = _signals()
    loaded = _loaded()
    intervention = BlockBypass(("transformer.block.000",), telemetry)
    provider = Qwen3BlockBypassProvider()

    with intervention.apply(provider, Qwen3Adapter(), loaded):
        _forward(loaded.model, 1.0)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "capability_anatomy.intervention.block_bypass"
    assert [event.attributes["capability_anatomy.reason"] for event in spans[0].events] == [
        "scoped_bypass_active",
        "hooks_removed",
    ]
    assert spans[0].start_time < spans[0].end_time
    assert "Qwen/Qwen3-fixture" not in repr(spans)
    metric_names = {
        metric.name
        for resource in reader.get_metrics_data().resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert "capability_anatomy.operation.decisions" in metric_names


@pytest.mark.parametrize(
    ("failure", "reason", "outcome"),
    [
        ("validation", "validation_failed", "refused"),
        ("installation", "hook_installation_failed", "failed"),
        ("body", "experiment_body_failed", "failed"),
        ("cleanup", "cleanup_failed", "failed"),
    ],
)
def test_intervention_failure_telemetry_is_specific(failure: str, reason: str, outcome: str) -> None:
    telemetry, exporter, _reader = _signals()
    loaded = _loaded()
    adapter = Qwen3Adapter()
    provider = Qwen3BlockBypassProvider()
    component_ids = ("missing",) if failure == "validation" else ("transformer.block.000",)

    if failure == "installation":
        loaded.model.model.layers[0].register_forward_hook = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("registration failed")
        )
    elif failure == "cleanup":
        class CleanupFailureProvider:
            def validate(self, _adapter, _loaded, _component_ids):
                pass

            def apply(self, _adapter, _loaded, _component_ids):
                class Manager:
                    def __enter__(self):
                        return None

                    def __exit__(self, *_args):
                        raise RuntimeError("cleanup failed")

                return Manager()

        provider = CleanupFailureProvider()

    with pytest.raises((InvalidConfigurationError, RuntimeError)):
        with BlockBypass(component_ids, telemetry).apply(provider, adapter, loaded):
            if failure == "body":
                raise RuntimeError("body failed")

    decision = exporter.get_finished_spans()[0].events[-1].attributes
    assert decision["capability_anatomy.reason"] == reason
    assert decision["capability_anatomy.outcome"] == outcome
