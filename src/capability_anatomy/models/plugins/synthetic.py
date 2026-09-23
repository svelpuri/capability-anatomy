from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator
import resource
import sys

from ...domain import ModelConfig, RuntimeConfig
from ...errors import InvalidConfigurationError
from ...protocols import CORE_API_VERSION, ModelAdapter
from ..base import AdapterProvenance, ComponentHandle, ExecutionRequest, LoadedModel, ModelTopology


@dataclass
class _SyntheticModel:
    active: set[str] = field(default_factory=set)


class SyntheticModelAdapter:
    name = "builtin.synthetic-model"
    version = "1"
    api_version = CORE_API_VERSION
    architecture = "synthetic-two-component"
    capabilities = frozenset({
        "execution.synthetic",
        "measurement.memory",
        "measurement.synchronization",
        "measurement.tokens",
        "model.topology.components",
    })

    def load(self, spec: ModelConfig, runtime: RuntimeConfig) -> LoadedModel:
        return LoadedModel(_SyntheticModel(), spec.source, spec.revision)

    def topology(self, loaded: LoadedModel) -> ModelTopology:
        return ModelTopology(
            self.architecture,
            (
                ComponentHandle("component.neutral", "synthetic", object(), order=0),
                ComponentHandle("component.damage", "synthetic", object(), order=1),
            ),
        )

    def execute(self, loaded: LoadedModel, request: ExecutionRequest) -> str:
        if not isinstance(request.payload, dict) or not isinstance(request.payload.get("text"), str):
            raise InvalidConfigurationError("synthetic request requires text")
        if "component.damage" in loaded.model.active:
            return "damaged"
        return request.payload["text"]

    def synchronize(self, loaded: LoadedModel) -> None:
        return None

    def memory_bytes(self, loaded: LoadedModel) -> int:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024

    def token_counts(self, loaded: LoadedModel, request: ExecutionRequest, result: str) -> tuple[int, int]:
        return len(request.payload["text"].split()), len(result.split())

    def provenance(self, loaded: LoadedModel) -> AdapterProvenance:
        return AdapterProvenance(
            self.name, self.version, self.api_version, self.architecture,
            loaded.source, loaded.revision, type(loaded.model).__name__, {}, {"python": "builtin"},
        )


class SyntheticInterventionProvider:
    name = "builtin.synthetic-intervention"
    version = "1"
    api_version = CORE_API_VERSION
    capabilities = frozenset({"intervention.block_bypass", "intervention.scoped_cleanup"})

    def validate(self, adapter: ModelAdapter, loaded: LoadedModel, component_ids: tuple[str, ...]) -> None:
        available = {item.id for item in adapter.topology(loaded).components}
        if not set(component_ids) <= available:
            raise InvalidConfigurationError("synthetic intervention component is unavailable")

    @contextmanager
    def apply(self, adapter: ModelAdapter, loaded: LoadedModel, component_ids: tuple[str, ...]) -> Iterator[None]:
        self.validate(adapter, loaded, component_ids)
        loaded.model.active.update(component_ids)
        try:
            yield
        finally:
            loaded.model.active.difference_update(component_ids)
