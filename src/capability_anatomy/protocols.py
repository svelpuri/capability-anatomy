from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from .domain import DatasetMetadata, EvaluationObservation, EvaluationScore, NormalizedRecord, ParseResult
from .models.base import AdapterProvenance, ExecutionRequest, LoadedModel, ModelTopology


CORE_API_VERSION = "capability-anatomy/plugin-api/v1"


@runtime_checkable
class Plugin(Protocol):
    name: str
    version: str
    api_version: str
    capabilities: frozenset[str]


class ModelAdapter(Plugin, Protocol):
    architecture: str

    def load(self, spec: Any, runtime: Any) -> LoadedModel: ...
    def topology(self, loaded: LoadedModel) -> ModelTopology: ...
    def execute(self, loaded: LoadedModel, request: ExecutionRequest) -> Any: ...
    def provenance(self, loaded: LoadedModel) -> AdapterProvenance: ...


@runtime_checkable
class MeasuredModelAdapter(ModelAdapter, Protocol):
    """Model adapter with the public measurement boundary required by runners."""

    def synchronize(self, loaded: LoadedModel) -> None: ...
    def memory_bytes(self, loaded: LoadedModel) -> int | None: ...
    def token_counts(
        self,
        loaded: LoadedModel,
        request: ExecutionRequest,
        result: Any,
    ) -> tuple[int | None, int | None]: ...


@runtime_checkable
class RuntimePlugin(Plugin, Protocol):
    """Execution and measurement boundary selected independently of location."""

    def execute(self, adapter: ModelAdapter, loaded: LoadedModel, request: ExecutionRequest) -> Any: ...
    def synchronize(self, adapter: ModelAdapter, loaded: LoadedModel) -> None: ...
    def memory_bytes(self, adapter: ModelAdapter, loaded: LoadedModel) -> int | None: ...
    def token_counts(
        self,
        adapter: ModelAdapter,
        loaded: LoadedModel,
        request: ExecutionRequest,
        result: Any,
    ) -> tuple[int | None, int | None]: ...
    def clock(self) -> float: ...


class InterventionProvider(Plugin, Protocol):
    def validate(self, adapter: ModelAdapter, loaded: LoadedModel, component_ids: tuple[str, ...]) -> None: ...
    def apply(
        self,
        adapter: ModelAdapter,
        loaded: LoadedModel,
        component_ids: tuple[str, ...],
    ) -> AbstractContextManager[None]: ...


class Intervention(Plugin, Protocol):
    def validate(self, provider: InterventionProvider, adapter: ModelAdapter, loaded: LoadedModel) -> None: ...
    def apply(
        self,
        provider: InterventionProvider,
        adapter: ModelAdapter,
        loaded: LoadedModel,
    ) -> AbstractContextManager[None]: ...


class Transformation(Plugin, Protocol):
    def plan(self, adapter: ModelAdapter, model: Any, component_ids: tuple[str, ...]) -> Any: ...
    def apply(self, adapter: ModelAdapter, model: Any, plan: Any) -> Any: ...
    def validate(self, adapter: ModelAdapter, model: Any, plan: Any) -> Any: ...


class EvaluationSuite(Plugin, Protocol):
    def required_record_schema(self) -> Mapping[str, Any]: ...
    def build_request(self, example: NormalizedRecord) -> Any: ...
    def parse(self, result: Any) -> ParseResult: ...
    def score(self, example: NormalizedRecord, output: ParseResult) -> Mapping[str, EvaluationScore]: ...
    def aggregate(self, observations: Iterable[EvaluationObservation]) -> Mapping[str, Any]: ...


class DatasetProvider(Plugin, Protocol):
    def metadata(self) -> DatasetMetadata: ...
    def records(self, split: str) -> Iterable[NormalizedRecord]: ...
    def stable_id(self, record: NormalizedRecord) -> str: ...
    def validate_for(self, suite: EvaluationSuite) -> None: ...
