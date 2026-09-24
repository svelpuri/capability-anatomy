from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from pathlib import Path
from typing import Any, Mapping

from .credentials import CredentialReference


class PolicyStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"


class ExecutionState(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class MetricDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class ObservationStatus(StrEnum):
    COMPLETE = "complete"
    ERROR = "error"
    MISSING = "missing"
    SKIPPED = "skipped"


class ParseStatus(StrEnum):
    PARSED = "parsed"
    ERROR = "error"


@dataclass(frozen=True)
class MetricFraction:
    numerator: float
    denominator: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.numerator) or self.numerator < 0:
            raise ValueError("metric numerator must be finite and non-negative")
        if self.denominator <= 0:
            raise ValueError("metric denominator must be positive")


@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    aggregation_version: str
    direction: MetricDirection
    unit: str
    status: ObservationStatus
    value: float | None = None
    sample_count: int | None = None
    fraction: MetricFraction | None = None
    distribution: Mapping[str, float] | None = None
    uncertainty: Mapping[str, float | str] | None = None

    def __post_init__(self) -> None:
        values = (
            self.value,
            *(self.distribution or {}).values(),
            *(value for value in (self.uncertainty or {}).values() if isinstance(value, (int, float))),
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("metric values must be finite")
        if self.status is ObservationStatus.COMPLETE:
            if self.value is None:
                raise ValueError("complete metric requires a value")
        if self.sample_count is not None and self.sample_count < 0:
            raise ValueError("metric sample count cannot be negative")


@dataclass(frozen=True)
class ArtifactDigest:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class EvidenceManifest:
    schema_version: str
    run_id: str
    config_sha256: str
    execution: "ExecutionRecord"
    artifacts: tuple[ArtifactDigest, ...]
    policy_result: "PolicyResult | None" = None


@dataclass(frozen=True)
class PolicyResult:
    policy_id: str
    status: PolicyStatus
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionRecord:
    state: ExecutionState
    completed_stages: tuple[str, ...]


@dataclass(frozen=True)
class PluginIdentity:
    name: str
    version: str
    api_version: str


@dataclass(frozen=True)
class ArtifactReference:
    uri: str
    digest: str


@dataclass(frozen=True)
class ComponentChange:
    component_id: str
    change: str
    details: Mapping[str, Any]


@dataclass(frozen=True)
class TransformationRecipe:
    schema_version: str
    plugin: PluginIdentity
    parameters: Mapping[str, Any]
    input_artifacts: tuple[ArtifactReference, ...]
    output_artifacts: tuple[ArtifactReference, ...]
    component_changes: tuple[ComponentChange, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    executor: str
    deterministic: bool
    warmup_runs: int
    randomized_execution_order: bool
    repetitions: int
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    plugin: str
    source: str
    revision: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class DatasetMetadata:
    provider: str
    revision: str
    license: str
    sha256: str
    partitions: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedRecord:
    id: str
    partition: str
    input: Any
    expected: Any
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    value: Any = None
    diagnostics: Mapping[str, str] = field(default_factory=dict)


class ScoreReason(StrEnum):
    ABSTENTION_EXPLICIT_FORM = "abstention_explicit_form"
    ABSTENTION_EMPTY_CALL_LIST = "abstention_empty_call_list"
    ABSTENTION_EMPTY_OR_INVALID_TYPE = "abstention_empty_or_invalid_type"
    ABSTENTION_UNRECOGNIZED_OR_MALFORMED = "abstention_unrecognized_or_malformed"
    ABSTENTION_TOOL_INVOCATION = "abstention_tool_invocation"


@dataclass(frozen=True)
class EvaluationScore:
    value: float
    numerator: float | None = None
    denominator: int | None = None
    reason: ScoreReason | None = None

    def __post_init__(self) -> None:
        if self.reason is not None and not isinstance(self.reason, ScoreReason):
            raise ValueError("evaluation score reason must be a supported bounded reason")
        if not math.isfinite(self.value):
            raise ValueError("evaluation score must be finite")
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError("evaluation score fraction must be complete")
        if self.numerator is not None and (not math.isfinite(self.numerator) or self.numerator < 0):
            raise ValueError("evaluation score numerator must be finite and non-negative")
        if self.denominator is not None and self.denominator <= 0:
            raise ValueError("evaluation score denominator must be positive")


@dataclass(frozen=True)
class EvaluationObservation:
    example_id: str
    partition: str
    status: ObservationStatus
    raw_output: Any
    parsed: ParseResult
    scores: Mapping[str, EvaluationScore]
    repetition: int = 0
    elapsed_seconds: float | None = None
    peak_memory_bytes: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class CapabilityConfig:
    evaluation_plugin: str
    suite_version: str
    target_metrics: tuple[str, ...]
    collateral_metrics: tuple[str, ...]


@dataclass(frozen=True)
class DatasetConfig:
    provider: str
    manifest: str
    discovery_partition: str
    validation_partition: str


@dataclass(frozen=True)
class InterventionConfig:
    plugin: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    retain_prompts: bool
    retain_raw_outputs: bool
    prompt_storage: str


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    experiment_id: str
    seed: int
    model: ModelConfig
    runtime: RuntimeConfig
    capability: CapabilityConfig
    dataset: DatasetConfig
    intervention: InterventionConfig
    policy_file: Path | None
    output: OutputConfig
    credentials: Mapping[str, CredentialReference] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        runtime = value["runtime"]
        capability = value["capability"]
        dataset = value["dataset"]
        output = value["output"]
        return cls(
            credentials={key: CredentialReference(**reference) for key, reference in value.get("credentials", {}).items()},
            schema_version=value["schema_version"],
            experiment_id=value["experiment"]["id"],
            seed=value["experiment"]["seed"],
            model=ModelConfig(**value["model"]),
            runtime=RuntimeConfig(
                executor=runtime["executor"],
                deterministic=runtime["deterministic"],
                warmup_runs=runtime["warmup_runs"],
                randomized_execution_order=runtime["randomized_execution_order"],
                repetitions=runtime["repetitions"],
                parameters=runtime["parameters"],
            ),
            capability=CapabilityConfig(
                evaluation_plugin=capability["evaluation_plugin"],
                suite_version=str(capability["suite_version"]),
                target_metrics=tuple(capability["target_metrics"]),
                collateral_metrics=tuple(capability["collateral_metrics"]),
            ),
            dataset=DatasetConfig(**dataset),
            intervention=InterventionConfig(**value["intervention"]),
            policy_file=Path(value["policy"]["file"]) if value.get("policy", {}).get("file") else None,
            output=OutputConfig(
                directory=Path(output["directory"]),
                retain_prompts=output["retain_prompts"],
                retain_raw_outputs=output["retain_raw_outputs"],
                prompt_storage=output["prompt_storage"],
            ),
        )
