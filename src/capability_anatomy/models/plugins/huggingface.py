from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform
import json
from threading import Lock
from typing import Any, Callable, Iterator, Mapping

import torch

from ...domain import ModelConfig, RuntimeConfig
from ...errors import InvalidConfigurationError, UnsupportedAdapterError
from ...protocols import CORE_API_VERSION, ModelAdapter
from ...telemetry import OperationTelemetry
from opentelemetry.trace import Status, StatusCode
from ..base import AdapterProvenance, ComponentHandle, ExecutionRequest, GeneratedText, GenerationRequest, LoadedModel, ModelTopology, PerplexityRequest


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


QwenGenerationRequest = GenerationRequest


@dataclass(frozen=True)
class ArchitectureProfile:
    id: str
    model_types: tuple[str, ...]
    architecture: str
    layers_path: tuple[str, ...] = ("model", "layers")
    attention_attribute: str = "self_attn"
    mlp_attribute: str = "mlp"
    layer_index_attribute: str = "layer_idx"
    output_rule: str = "preserve_output_tail"
    transformers_versions: tuple[str, ...] = ("4.55.4",)

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


ARCHITECTURE_PROFILES = {
    "qwen3-dense-v1": ArchitectureProfile("qwen3-dense-v1", ("qwen3",), "qwen3-dense"),
    "llama-like-v1": ArchitectureProfile("llama-like-v1", ("llama", "mistral"), "llama-like"),
    "qwen3-dense-v2": ArchitectureProfile("qwen3-dense-v2", ("qwen3",), "qwen3-dense", output_rule="native_tensor", transformers_versions=("5.17.0",)),
    "llama-like-v2": ArchitectureProfile("llama-like-v2", ("llama", "mistral"), "llama-like", output_rule="native_tensor", transformers_versions=("5.17.0",)),
}


class _ModelSelectorRefused(UnsupportedAdapterError):
    reason = "model_loading_selector_refused"


class HuggingFaceCausalLMAdapter:
    name = "huggingface.causal-lm"
    version = "2"
    api_version = CORE_API_VERSION
    architecture = "profile-selected"
    capabilities = frozenset(
        {
            "execution.autoregressive_text",
            "execution.perplexity",
            "measurement.memory",
            "measurement.synchronization",
            "measurement.tokens",
            "model.native_tool_prompt",
            "model.architecture_profile",
            "model.topology.components",
            "model.topology.transformer_blocks",
        }
    )

    def __init__(self, default_profile: str | None = None) -> None:
        self._default_profile = default_profile
        self._load_telemetry = OperationTelemetry.create("model_loading")

    def _profile_for_spec(self, spec: ModelConfig) -> ArchitectureProfile:
        profile_id = spec.parameters.get("architecture_profile", self._default_profile)
        try:
            return ARCHITECTURE_PROFILES[profile_id]
        except (KeyError, TypeError) as error:
            raise UnsupportedAdapterError("Hugging Face architecture profile is unknown") from error

    def profile_identity(self, spec: ModelConfig) -> tuple[str, str]:
        profile = self._profile_for_spec(spec)
        return profile.id, profile.sha256

    def _profile(self, loaded: LoadedModel) -> ArchitectureProfile:
        profile = loaded.resources.get("architecture_profile")
        if profile is None and self._default_profile is not None:
            profile = ARCHITECTURE_PROFILES[self._default_profile]
        if not isinstance(profile, ArchitectureProfile):
            raise UnsupportedAdapterError("Hugging Face architecture profile is missing")
        return profile

    @staticmethod
    def _require_safe_selectors(value: Mapping[str, Any]) -> None:
        pending: list[Any] = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                for key, child in item.items():
                    normalized = key.lstrip("_")
                    if normalized in {"attn_implementation", "attn_implementation_internal", "attention_implementation"} and child not in (None, "eager"):
                        raise _ModelSelectorRefused("model_loading_selector_refused: only eager attention is supported")
                    if normalized in {"use_kernels", "allow_all_kernels", "kernel_config", "trust_remote_code", "custom_generate", "auto_map", "custom_pipelines", "quantization_config"} and child:
                        raise _ModelSelectorRefused("model_loading_selector_refused: custom model code, kernels and quantization are unsupported")
                    pending.append(child)
            elif isinstance(item, list):
                pending.extend(item)

    def load(self, spec: ModelConfig, runtime: RuntimeConfig) -> LoadedModel:
        with self._load_telemetry.tracer.start_as_current_span(
            "capability_anatomy.model.load", record_exception=False, set_status_on_exception=False,
        ) as span:
            try:
                loaded = self._load_safe(spec, runtime)
            except BaseException as error:
                reason = error.reason if isinstance(error, _ModelSelectorRefused) else "safe_model_loading_refused"
                self._load_telemetry.record(span, operation="load_model", outcome="refused", reason=reason)
                span.set_status(Status(StatusCode.ERROR, reason))
                raise
            self._load_telemetry.record(span, operation="load_model", outcome="accepted", reason="safetensors_eager_model_loaded")
            return loaded

    def _load_safe(self, spec: ModelConfig, runtime: RuntimeConfig) -> LoadedModel:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

        self._require_safe_selectors(dict(spec.parameters))
        self._require_safe_selectors(dict(runtime.parameters))
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype_name = spec.parameters.get("dtype")
        device = runtime.parameters.get("device")
        context_length = runtime.parameters.get("context_length")
        if (
            dtype_name not in dtypes
            or not isinstance(device, str)
            or not device
            or not isinstance(context_length, int)
            or context_length <= 0
        ):
            raise UnsupportedAdapterError("Qwen3 dtype or device parameter is unsupported")
        profile = self._profile_for_spec(spec)
        if _package_version("transformers") not in profile.transformers_versions:
            raise UnsupportedAdapterError("Hugging Face architecture profile does not support installed Transformers")
        # Read plain configuration data through the established library before
        # constructing any model/tokenizer or importing model-selected kernels.
        authored_config, _ = PretrainedConfig.get_config_dict(spec.source, revision=spec.revision)
        self._require_safe_selectors(authored_config)
        model_config = AutoConfig.from_pretrained(
            spec.source,
            revision=spec.revision,
            trust_remote_code=False,
        )
        if getattr(model_config, "model_type", None) not in profile.model_types:
            raise UnsupportedAdapterError("model config does not match architecture profile")
        tokenizer = AutoTokenizer.from_pretrained(
            spec.source,
            revision=spec.revision,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            spec.source,
            revision=spec.revision,
            dtype=dtypes[dtype_name],
            use_safetensors=True,
            use_kernels=False,
            allow_all_kernels=False,
            attn_implementation="eager",
            config=model_config,
            trust_remote_code=False,
        ).to(device).eval()
        loaded = LoadedModel(
            model=model,
            source=spec.source,
            revision=spec.revision,
            resources={"tokenizer": tokenizer, "context_length": context_length, "architecture_profile": profile},
        )
        self.topology(loaded)
        return loaded

    def _layers(self, loaded: LoadedModel) -> torch.nn.ModuleList:
        model = loaded.model
        profile = self._profile(loaded)
        if getattr(getattr(model, "config", None), "model_type", None) not in profile.model_types:
            raise UnsupportedAdapterError("model config does not match architecture profile")
        layers: Any = model
        for attribute in profile.layers_path:
            layers = getattr(layers, attribute, None)
        if not isinstance(layers, torch.nn.ModuleList) or not layers:
            raise UnsupportedAdapterError("Qwen3 transformer block topology is ambiguous")
        return layers

    def topology(self, loaded: LoadedModel) -> ModelTopology:
        profile = self._profile(loaded)
        layers = self._layers(loaded)
        components = []
        for order, layer in enumerate(layers):
            attention = getattr(layer, profile.attention_attribute, None)
            if attention is None or not hasattr(layer, profile.mlp_attribute):
                raise UnsupportedAdapterError("profile block structure is incomplete")
            if getattr(attention, profile.layer_index_attribute, None) != order:
                raise UnsupportedAdapterError("profile layer indices are stale or ambiguous")
            components.append(
                ComponentHandle(
                    id=f"transformer.block.{order:03d}",
                    kind="transformer_block",
                    order=order,
                    implementation=layer,
                    parent_id=None,
                    metadata={
                        "architecture": profile.architecture,
                        "architecture_profile": profile.id,
                        "architecture_profile_sha256": profile.sha256,
                        "module_path": f"model.layers.{order}",
                        "original_order": order,
                        "parameter_count": sum(parameter.numel() for parameter in layer.parameters()),
                    },
                )
            )
        return ModelTopology(architecture=profile.architecture, components=tuple(components))

    def _render_prompt(self, loaded: LoadedModel, request: GenerationRequest) -> str:
        kwargs = {"tools": list(request.tools)} if request.tools is not None else {}
        return self._tokenizer(loaded).apply_chat_template(
            list(request.messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            **kwargs,
        )

    def execute(self, loaded: LoadedModel, request: ExecutionRequest) -> str | float:
        if isinstance(request.payload, PerplexityRequest):
            tokenizer = self._tokenizer(loaded)
            encoded = tokenizer(request.payload.text, return_tensors="pt", add_special_tokens=False)
            device = next(loaded.model.parameters()).device
            inputs = {name: value.to(device) for name, value in encoded.items()}
            self._enforce_context(loaded, inputs["input_ids"], 0)
            with torch.inference_mode():
                loss = loaded.model(**inputs, labels=inputs["input_ids"]).loss
            return float(torch.exp(loss.float()).item())
        if not isinstance(request.payload, GenerationRequest):
            raise InvalidConfigurationError("Qwen3 execution request is incompatible")
        generation = request.payload
        tokenizer = self._tokenizer(loaded)
        rendered = self._render_prompt(loaded, generation)
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        try:
            device = next(loaded.model.parameters()).device
        except StopIteration as error:
            raise UnsupportedAdapterError("Qwen3 model has no parameters") from error
        inputs = {name: value.to(device) for name, value in encoded.items()}
        self._enforce_context(loaded, inputs["input_ids"], generation.max_new_tokens)
        with torch.inference_mode():
            generated = loaded.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=generation.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        generated_ids = generated[0, prompt_width:]
        return GeneratedText(
            tokenizer.decode(generated_ids, skip_special_tokens=True),
            int(generated_ids.numel()),
        )

    @staticmethod
    def _enforce_context(loaded: LoadedModel, input_ids: torch.Tensor, output_tokens: int) -> None:
        limit = loaded.resources.get("context_length")
        if not isinstance(limit, int) or input_ids.shape[-1] + output_tokens > limit:
            raise InvalidConfigurationError("Qwen3 rendered request exceeds frozen context length")

    def synchronize(self, loaded: LoadedModel) -> None:
        device = next(loaded.model.parameters()).device
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

    def memory_bytes(self, loaded: LoadedModel) -> int | None:
        device = next(loaded.model.parameters()).device
        if device.type == "mps":
            return int(torch.mps.driver_allocated_memory())
        if device.type == "cuda":
            return int(torch.cuda.memory_allocated(device))
        return None

    def token_counts(
        self,
        loaded: LoadedModel,
        request: ExecutionRequest,
        result: Any,
    ) -> tuple[int, int]:
        if isinstance(request.payload, PerplexityRequest) and isinstance(result, float):
            tokenizer = self._tokenizer(loaded)
            ids = tokenizer(request.payload.text, add_special_tokens=False)["input_ids"]
            return (int(ids.numel()) if isinstance(ids, torch.Tensor) else len(ids), 0)
        if not isinstance(request.payload, GenerationRequest) or not isinstance(result, GeneratedText):
            raise InvalidConfigurationError("Qwen3 token measurement input is incompatible")
        tokenizer = self._tokenizer(loaded)
        rendered = self._render_prompt(loaded, request.payload)
        def token_length(text: str) -> int:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if isinstance(ids, torch.Tensor):
                return int(ids.numel())
            if ids and isinstance(ids[0], list):
                return len(ids[0])
            return len(ids)

        input_tokens = token_length(rendered)
        return input_tokens, result.generated_token_count

    def _tokenizer(self, loaded: LoadedModel) -> Any:
        tokenizer = loaded.resources.get("tokenizer")
        if tokenizer is None:
            raise UnsupportedAdapterError("Qwen3 tokenizer resource is required")
        return tokenizer

    def provenance(self, loaded: LoadedModel) -> AdapterProvenance:
        return AdapterProvenance(
            plugin=self.name,
            plugin_version=self.version,
            api_version=self.api_version,
            architecture=self._profile(loaded).architecture,
            model_source=loaded.source,
            model_revision=loaded.revision,
            model_class=type(loaded.model).__name__,
            implementation_metadata={
                "tokenizer_class": type(self._tokenizer(loaded)).__name__,
                "model_type": str(getattr(getattr(loaded.model, "config", None), "model_type", "unknown")),
                "architecture_profile": self._profile(loaded).id,
                "architecture_profile_sha256": self._profile(loaded).sha256,
            },
            libraries={
                "python": platform.python_version(),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
            },
        )


class HuggingFaceBlockBypassProvider:
    name = "huggingface.block-bypass"
    version = "2"
    api_version = CORE_API_VERSION
    capabilities = frozenset({"intervention.block_bypass", "intervention.scoped_cleanup"})

    def __init__(self, cleanup_probe: Callable[[], None] | None = None) -> None:
        self._active_bypasses: set[tuple[int, str]] = set()
        self._lock = Lock()
        self._cleanup_probe = cleanup_probe

    def validate(
        self,
        adapter: ModelAdapter,
        loaded: LoadedModel,
        component_ids: tuple[str, ...],
    ) -> None:
        topology = adapter.topology(loaded)
        profile = loaded.resources.get("architecture_profile")
        if isinstance(profile, ArchitectureProfile) and profile.output_rule not in {"preserve_output_tail", "native_tensor"}:
            raise UnsupportedAdapterError("architecture profile bypass output rule is unsupported")
        if any(topology.component(component_id).kind != "transformer_block" for component_id in component_ids):
            raise UnsupportedAdapterError("Hugging Face bypass provider requires transformer blocks")

    @contextmanager
    def _bypass_one(self, loaded: LoadedModel, component: ComponentHandle) -> Iterator[None]:
        key = (id(loaded.model), component.id)
        with self._lock:
            if key in self._active_bypasses:
                raise InvalidConfigurationError("Qwen3 component bypass is already active")
            self._active_bypasses.add(key)

        handle = None
        try:
            def hook(_module: Any, args: tuple[Any, ...], output: Any) -> Any:
                if not args:
                    raise RuntimeError("Qwen3 block bypass requires hidden-state input")
                hidden = args[0]
                if isinstance(output, tuple):
                    return (hidden, *output[1:])
                if isinstance(output, torch.Tensor):
                    return hidden
                raise RuntimeError("Qwen3 block returned an unsupported output structure")

            handle = component.implementation.register_forward_hook(hook)
            yield
        finally:
            cleanup_error: BaseException | None = None
            try:
                if handle is not None:
                    handle.remove()
                if self._cleanup_probe is not None:
                    self._cleanup_probe()
            except BaseException as error:
                cleanup_error = error
            finally:
                with self._lock:
                    self._active_bypasses.discard(key)
            if cleanup_error is not None:
                raise cleanup_error

    @contextmanager
    def apply(
        self,
        adapter: ModelAdapter,
        loaded: LoadedModel,
        component_ids: tuple[str, ...],
    ) -> Iterator[None]:
        self.validate(adapter, loaded, component_ids)
        topology = adapter.topology(loaded)
        with ExitStack() as stack:
            for component_id in component_ids:
                stack.enter_context(self._bypass_one(loaded, topology.component(component_id)))
            yield
