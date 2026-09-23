from __future__ import annotations

from ...models.base import GenerationRequest
from ...domain import NormalizedRecord
from .synthetic import SyntheticExactMatchSuite


class ChatExactMatchSuite(SyntheticExactMatchSuite):
    """Small model-neutral chat fixture suite used for backend conformance."""

    name = "reference.chat-exact-match"
    capabilities = frozenset({"evaluation.parse", "evaluation.score", "evaluation.aggregate", "request.chat_generation"})

    def build_request(self, example: NormalizedRecord) -> GenerationRequest:
        return GenerationRequest(
            messages=tuple(example.input["messages"]),
            tools=tuple(example.input["tools"]) if example.input.get("tools") is not None else None,
            max_new_tokens=int(example.input.get("max_new_tokens", 2)),
        )
