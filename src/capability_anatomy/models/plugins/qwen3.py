from __future__ import annotations

from .huggingface import HuggingFaceBlockBypassProvider, HuggingFaceCausalLMAdapter, QwenGenerationRequest


class Qwen3Adapter(HuggingFaceCausalLMAdapter):
    """Thin compatibility extension selecting the validated Qwen profile."""

    name = "reference.qwen3-transformers"
    architecture = "qwen3-dense"

    def __init__(self) -> None:
        super().__init__("qwen3-dense-v2")


class Qwen3BlockBypassProvider(HuggingFaceBlockBypassProvider):
    """Thin compatibility name for the shared native-hook backend."""

    name = "reference.qwen3-block-bypass"
