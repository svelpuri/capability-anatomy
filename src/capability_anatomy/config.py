from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema

from .authored_inputs import AuthoredInputError, load_mapping
from .credentials import InlineCredentialError, reject_inline_credentials
from .domain import ExperimentConfig
from .errors import InvalidConfigurationError
from .telemetry import DecisionTelemetry


SCHEMA_VERSION = "capability-anatomy/experiment-config/v1"
MUTABLE_REVISIONS = {"head", "latest", "main", "master", "trunk"}


def _schema_text() -> str:
    packaged = files("capability_anatomy").joinpath("_schemas/experiment-config.v1.schema.json")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    return (Path(__file__).resolve().parents[2] / "schemas/experiment-config.v1.schema.json").read_text(encoding="utf-8")


def _load_authored(path: Path) -> Any:
    return load_mapping(path)


class UnsupportedPolicyError(InvalidConfigurationError):
    reason = "generic_policy_unsupported"


def _validate_semantics(value: dict[str, Any]) -> None:
    policy_file = value.get("policy", {}).get("file")
    phase5 = value["runtime"]["parameters"].get("phase5_protocol") is not None
    if policy_file is not None and not phase5:
        raise UnsupportedPolicyError("policy.file is supported only for governed Phase5 authorization; remove it for generic experiments")
    if phase5 and policy_file is None:
        raise InvalidConfigurationError("governed Phase5 requires policy.file authorization")
    revision = value["model"]["revision"].strip().casefold()
    if revision in MUTABLE_REVISIONS or revision.startswith("refs/heads/"):
        raise InvalidConfigurationError("model revision must be immutable")
    dataset = value["dataset"]
    if dataset["discovery_partition"] == dataset["validation_partition"]:
        raise InvalidConfigurationError("discovery and validation partitions must be disjoint")
    capability = value["capability"]
    overlap = set(capability["target_metrics"]) & set(capability["collateral_metrics"])
    if overlap:
        raise InvalidConfigurationError("target and collateral metrics must be disjoint")


def load_experiment_config(path: Path, telemetry: DecisionTelemetry | None = None) -> ExperimentConfig:
    signals = telemetry or DecisionTelemetry.create()
    with signals.tracer.start_as_current_span(
        "capability_anatomy.config.validate",
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        schema_version = "unknown"
        try:
            value = _load_authored(path)
            if not isinstance(value, dict):
                raise InvalidConfigurationError("configuration root must be an object")
            schema_version = SCHEMA_VERSION if value.get("schema_version") == SCHEMA_VERSION else "unknown"
            schema = json.loads(_schema_text())
            jsonschema.Draft202012Validator(schema).validate(value)
            reject_inline_credentials(value)
            _validate_semantics(value)
            config = ExperimentConfig.from_mapping(value)
        except jsonschema.ValidationError as error:
            signals.record(span, outcome="refused", reason="schema_validation_failed", schema_version=schema_version)
            raise InvalidConfigurationError("configuration schema validation failed") from None
        except AuthoredInputError:
            signals.record(span, outcome="refused", reason="authored_config_invalid", schema_version=schema_version)
            raise
        except InlineCredentialError:
            signals.record(span, outcome="refused", reason="inline_credential_refused", schema_version=schema_version)
            raise
        except UnsupportedPolicyError:
            signals.record(span, outcome="refused", reason="generic_policy_unsupported", schema_version=schema_version)
            raise
        except InvalidConfigurationError:
            signals.record(span, outcome="refused", reason="semantic_validation_failed", schema_version=schema_version)
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            signals.record(span, outcome="refused", reason="schema_unavailable", schema_version=schema_version)
            raise InvalidConfigurationError("configuration schema could not be loaded") from error
        signals.record(span, outcome="accepted", reason="schema_and_semantics_valid", schema_version=schema_version)
        return config
