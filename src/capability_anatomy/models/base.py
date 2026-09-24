from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ComponentHandle:
    id: str
    kind: str
    implementation: Any = field(repr=False, compare=False)
    parent_id: str | None = None
    order: int | None = None
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelTopology:
    architecture: str
    components: tuple[ComponentHandle, ...]

    def __post_init__(self) -> None:
        ids = [component.id for component in self.components]
        if not self.architecture:
            raise ValueError("topology architecture is required")
        if not ids:
            raise ValueError("topology must contain components")
        if len(ids) != len(set(ids)):
            raise ValueError("topology component IDs must be unique")
        by_id = {component.id: component for component in self.components}
        if any(component.parent_id is not None and component.parent_id not in by_id for component in self.components):
            raise ValueError("topology parent must reference a component")
        if any(component.order is not None and component.order < 0 for component in self.components):
            raise ValueError("topology order must be non-negative")
        for component in self.components:
            seen = {component.id}
            parent_id = component.parent_id
            while parent_id is not None:
                if parent_id in seen:
                    raise ValueError("topology hierarchy must be acyclic")
                seen.add(parent_id)
                parent_id = by_id[parent_id].parent_id

    def component(self, component_id: str) -> ComponentHandle:
        matches = [component for component in self.components if component.id == component_id]
        if len(matches) != 1:
            raise KeyError(component_id)
        return matches[0]


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    source: str
    revision: str
    resources: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class ExecutionRequest:
    payload: Any


class GeneratedText(str):
    """Decoded generation with its exact pre-decode token count."""

    def __new__(cls, value: str, generated_token_count: int) -> "GeneratedText":
        if not isinstance(generated_token_count, int) or generated_token_count < 0:
            raise ValueError("generated token count must be a non-negative integer")
        instance = super().__new__(cls, value)
        instance.generated_token_count = generated_token_count
        return instance


@dataclass(frozen=True)
class GenerationRequest:
    messages: tuple[Mapping[str, str], ...]
    tools: tuple[Mapping[str, Any], ...] | None
    max_new_tokens: int


@dataclass(frozen=True)
class PerplexityRequest:
    text: str


@dataclass(frozen=True)
class AdapterProvenance:
    plugin: str
    plugin_version: str
    api_version: str
    architecture: str
    model_source: str
    model_revision: str
    model_class: str
    implementation_metadata: Mapping[str, str]
    libraries: Mapping[str, str]
