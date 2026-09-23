"""Actual local Transformers models, tokenizer, cache and public plugins; no downloads."""
from dataclasses import replace

import pytest
import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from capability_anatomy.discovery import PluginDiscovery
from capability_anatomy.domain import ModelConfig, RuntimeConfig
from capability_anatomy.errors import UnsupportedAdapterError
from capability_anatomy.models.plugins.huggingface import ARCHITECTURE_PROFILES


@pytest.fixture(params=["Llama", "Mistral", "Qwen3"])
def real_profile(request, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        family = request.param
        config = getattr(transformers, family + "Config")(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
            head_dim=8, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        model = getattr(transformers, family + "ForCausalLM")(config).eval()
        model.save_pretrained(tmp_path, safe_serialization=True)
        vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "user": 4, "assistant": 5,
                      "hello": 6, "world": 7, "tool": 8}
        vocabulary.update({"word" + str(index): index for index in range(9, 32)})
        tokenizer_impl = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer_impl.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl, model_input_names=["input_ids", "attention_mask"], unk_token="[UNK]",
                                           bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]")
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if tools %}tool {% endif %}{% if add_generation_prompt %}assistant {% endif %}"
        tokenizer.save_pretrained(tmp_path)
        discovery = PluginDiscovery()
        adapter = discovery.resolve("model", "huggingface.causal-lm", required_capabilities=frozenset({"execution.autoregressive_text"}), required_methods=("load", "execute")).plugin
        provider = discovery.resolve("intervention", "huggingface.block-bypass", required_capabilities=frozenset({"intervention.block_bypass"}), required_methods=("apply",)).plugin
        profile_id = "qwen3-dense-v2" if family == "Qwen3" else "llama-like-v2"
        spec = ModelConfig(plugin="huggingface.causal-lm", source=str(tmp_path), revision="a" * 40,
                           parameters={"architecture_profile": profile_id, "dtype": "float32"})
        runtime = RuntimeConfig(executor="builtin.local", deterministic=True, warmup_runs=0, repetitions=1,
                                randomized_execution_order=False, parameters={"device": "cpu", "context_length": 64})
        loaded = adapter.load(spec, runtime)
        try:
            yield family, adapter, provider, loaded, spec, runtime
        finally:
            torch.set_num_threads(threads)








def test_real_pickle_only_checkpoint_refused_at_model_consumer(real_profile):
    from pathlib import Path
    _, adapter, _, loaded, spec, runtime = real_profile
    source = Path(spec.source)
    # Benign tensor dictionary in ordinary PyTorch pickle format, not a payload.
    torch.save(loaded.model.state_dict(), source / "pytorch_model.bin")
    (source / "model.safetensors").unlink()
    assert (source / "pytorch_model.bin").is_file()
    assert not list(source.glob("*.safetensors"))
    with pytest.raises(OSError, match="safetensors"):
        adapter.load(spec, runtime)


@pytest.mark.parametrize("selector", ["attn_implementation", "_attn_implementation", "_attn_implementation_internal", "attention_implementation"])
def test_real_checkpoint_custom_attention_refuses_before_factories(real_profile, monkeypatch, selector):
    import json
    from pathlib import Path
    _, adapter, _, _, spec, runtime = real_profile
    path = Path(spec.source) / "config.json"
    authored = json.loads(path.read_text())
    authored[selector] = "kernels-community/fake-selector-no-download"
    path.write_text(json.dumps(authored))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("factory")
        raise AssertionError("factory reached before selector refusal")
    for factory in (transformers.AutoConfig, transformers.AutoTokenizer, transformers.AutoModelForCausalLM):
        monkeypatch.setattr(factory, "from_pretrained", forbidden)
    with pytest.raises(UnsupportedAdapterError, match="model_loading_selector_refused"):
        adapter.load(spec, runtime)
    assert calls == []


@pytest.mark.parametrize("parameters", [{"attn_implementation": "remote/kernel"}, {"use_kernels": True}, {"trust_remote_code": True}])
def test_authored_parameter_selector_refuses_before_config_fetch(real_profile, monkeypatch, parameters):
    _, adapter, _, _, spec, runtime = real_profile
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("fetch")
        raise AssertionError("fetch before authored selector refusal")
    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", forbidden)
    unsafe = replace(spec, parameters={**spec.parameters, **parameters})
    with pytest.raises(UnsupportedAdapterError, match="model_loading_selector_refused"):
        adapter.load(unsafe, runtime)
    assert not calls


def test_safe_load_uses_eager_and_reports_duration_and_reason(real_profile):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from capability_anatomy.telemetry import OperationTelemetry
    _, adapter, _, _, spec, runtime = real_profile
    exporter = InMemorySpanExporter()
    traces = TracerProvider()
    traces.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    adapter._load_telemetry = OperationTelemetry.create("model_loading", tracer=traces.get_tracer("gate"), meter=meters.get_meter("gate"))
    loaded = adapter.load(spec, runtime)
    assert loaded.model.config._attn_implementation == "eager"
    unsafe = replace(spec, parameters={**spec.parameters, "attn_implementation": "secret-selector-never-log"})
    with pytest.raises(UnsupportedAdapterError):
        adapter.load(unsafe, runtime)
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].attributes["capability_anatomy.reason"] == "safetensors_eager_model_loaded"
    assert spans[1].attributes["capability_anatomy.reason"] == "model_loading_selector_refused"
    assert spans[1].status.status_code.name == "ERROR"
    assert all(span.end_time >= span.start_time for span in spans)
    assert "secret-selector-never-log" not in "".join(span.to_json() for span in spans)
    assert reader.get_metrics_data().resource_metrics


def test_legacy_architecture_profile_hashes_are_unchanged():
    assert ARCHITECTURE_PROFILES["qwen3-dense-v1"].sha256 == "f13b3beea33b6dad48a91f3264415e5cb080dfd360262b5510458838c9303c17"
    assert ARCHITECTURE_PROFILES["llama-like-v1"].sha256 == "5b2361c3618e3c14705bea3dc284b7b9e10bc4575122aa3fa4a9a84917558d6b"
