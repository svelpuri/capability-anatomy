from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

from .credentials import RuntimeCredentials, SecretValue, refuse_secret_persistence
from .domain import EvaluationObservation, EvaluationScore, ExperimentConfig


def to_primitive(value: Any) -> Any:
    if isinstance(value, str):
        refuse_secret_persistence(value)
        return value
    if isinstance(value, (SecretValue, RuntimeCredentials)):
        raise TypeError("runtime credentials cannot be serialized")
    if isinstance(value, EvaluationScore):
        return {
            "value": value.value,
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    if is_dataclass(value):
        return {item.name: to_primitive(getattr(value, item.name)) for item in fields(value)
                if not (isinstance(value, ExperimentConfig) and item.name == "credentials" and not value.credentials)}
    if isinstance(value, Enum):
        return to_primitive(value.value)
    if isinstance(value, Path):
        return to_primitive(str(value))
    if isinstance(value, dict):
        return {to_primitive(str(key)): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    encoded = json.dumps(
        to_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    refuse_secret_persistence(encoded.decode("utf-8"))
    return encoded


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def serialize_observation(
    observation: EvaluationObservation,
    *,
    retain_raw_output: bool = True,
    prompt: Any = None,
    retain_prompt: bool = False,
    prompt_storage: str = "hash_only",
) -> dict[str, Any]:
    result = to_primitive(observation)
    if not retain_raw_output:
        result["raw_output"] = None
        result["parsed"]["value"] = None
    if retain_prompt:
        result["prompt"] = (
            to_primitive(prompt) if prompt_storage == "full" else canonical_sha256(prompt)
        )
    result["scores"] = {
        metric: {
            "value": score.value,
            "numerator": score.numerator,
            "denominator": score.denominator,
            **({"reason": score.reason.value} if score.reason is not None else {}),
        }
        for metric, score in sorted(observation.scores.items())
    }
    return result
