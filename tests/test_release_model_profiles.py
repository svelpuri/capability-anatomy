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
from capability_anatomy.errors import InvalidConfigurationError, UnsupportedAdapterError
from capability_anatomy.models.base import ExecutionRequest, GenerationRequest
from capability_anatomy.models.plugins.huggingface import ARCHITECTURE_PROFILES


@pytest.fixture
def isolated_torch_state():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(7)
            yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture(params=["Llama", "Mistral", "Qwen3"])
def real_profile(request, tmp_path, monkeypatch, isolated_torch_state):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
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
    yield family, adapter, provider, loaded, spec, runtime


def test_real_family_cached_generation_bypass_and_exact_restoration(real_profile):
    family, adapter, provider, loaded, _, _ = real_profile
    ids = torch.tensor([[1, 6, 7, 6]])
    model = loaded.model
    assert len(adapter.topology(loaded).components) == 2
    with torch.inference_mode():
        before = model(ids, use_cache=True)
        with provider.apply(adapter, loaded, ("transformer.block.000",)):
            changed = model(ids, use_cache=True)
            continued = model(torch.tensor([[7]]), past_key_values=changed.past_key_values, use_cache=True)
            generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False)
            assert all(continued.past_key_values.get_seq_length(index) == 5 for index in range(2))
            assert 5 <= generated.shape[-1] <= 7
        after = model(ids, use_cache=True)
    assert not torch.equal(before.logits, changed.logits), family
    assert torch.equal(before.logits, after.logits), family
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_real_family_public_adapter_template_tokens_context_and_failure_cleanup(real_profile):
    _, adapter, provider, loaded, _, _ = real_profile
    request = GenerationRequest(messages=({"role": "user", "content": "hello world"},), tools=(), max_new_tokens=3)
    output = adapter.execute(loaded, ExecutionRequest(request))
    assert isinstance(output, str)
    assert 1 <= output.generated_token_count <= 3
    rendered = adapter._render_prompt(loaded, request)
    assert rendered == "user hello world assistant "
    tokenizer = loaded.resources["tokenizer"]
    width = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    bounded = replace(loaded, resources={**loaded.resources, "context_length": width + 3})
    assert adapter.execute(bounded, ExecutionRequest(request)) == output
    too_short = replace(loaded, resources={**loaded.resources, "context_length": width + 2})
    with pytest.raises(InvalidConfigurationError, match="context"):
        with provider.apply(adapter, loaded, ("transformer.block.000",)):
            adapter.execute(too_short, ExecutionRequest(request))
    assert all(not layer._forward_hooks for layer in loaded.model.model.layers)
    assert adapter.execute(loaded, ExecutionRequest(request)) == output


def test_real_family_profile_mismatch_is_refused(real_profile):
    family, adapter, _, loaded, spec, runtime = real_profile
    wrong_profile = "llama-like-v2" if family == "Qwen3" else "qwen3-dense-v2"
    wrong = replace(spec, parameters={**spec.parameters, "architecture_profile": wrong_profile})
    with pytest.raises(UnsupportedAdapterError, match="match architecture profile"):
        adapter.load(wrong, runtime)
    forged = replace(loaded, resources={**loaded.resources, "architecture_profile": ARCHITECTURE_PROFILES[wrong_profile]})
    with pytest.raises(UnsupportedAdapterError, match="match architecture profile"):
        adapter.topology(forged)
