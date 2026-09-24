from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from ...domain import EvaluationObservation, EvaluationScore, NormalizedRecord, ObservationStatus, ParseResult, ParseStatus
from ...protocols import CORE_API_VERSION


class ToolCallingSuite:
    name = "reference.tool-calling-json"
    version = "1"
    api_version = CORE_API_VERSION
    capabilities = frozenset({"evaluation.tool_calling", "evaluation.independent_abstention"})

    def required_record_schema(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "required": ["id", "partition", "input", "expected", "metadata"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "partition": {"type": "string", "minLength": 1},
                "input": {"type": "object"},
                "expected": {"type": "array"},
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        }

    def build_request(self, example: NormalizedRecord) -> Any:
        return example.input

    def parse(self, result: Any) -> ParseResult:
        if not isinstance(result, str):
            raise TypeError("tool output must be text")
        value = json.loads(result)
        if not isinstance(value, list) or any(
            not isinstance(call, dict)
            or set(call) != {"name", "arguments"}
            or not isinstance(call["name"], str)
            or not isinstance(call["arguments"], dict)
            for call in value
        ):
            raise ValueError("expected calls with name and arguments")
        return ParseResult(ParseStatus.PARSED, value)

    def score(self, example: NormalizedRecord, output: ParseResult) -> Mapping[str, EvaluationScore]:
        expected = example.expected
        predicted = output.value
        if not expected:
            abstention = float(not predicted)
            return {"abstention": EvaluationScore(abstention, abstention, 1)}
        expected_names = [call["name"] for call in expected]
        predicted_names = [call["name"] for call in predicted]
        selection = float(predicted_names == expected_names)
        unmatched = list(predicted)
        matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for expected_call in expected:
            match_index = next(
                (index for index, call in enumerate(unmatched) if call["name"] == expected_call["name"]),
                None,
            )
            if match_index is not None:
                matched.append((expected_call, unmatched.pop(match_index)))
        bound = sum(predicted_call["arguments"] == expected_call["arguments"] for expected_call, predicted_call in matched)
        full_call = float(predicted == expected)
        scores = {
            "tool_selection": EvaluationScore(selection, selection, 1),
            "full_call": EvaluationScore(full_call, full_call, 1),
        }
        if matched:
            scores["argument_binding"] = EvaluationScore(bound / len(matched), float(bound), len(matched))
        return scores

    def aggregate(self, observations: Iterable[EvaluationObservation]) -> Mapping[str, Any]:
        values: dict[str, list[EvaluationScore]] = {}
        errors = 0
        complete = 0
        for observation in observations:
            if observation.status is not ObservationStatus.COMPLETE:
                errors += 1
                continue
            complete += 1
            for metric, value in observation.scores.items():
                values.setdefault(metric, []).append(value)
        metrics = {}
        sample_counts = {}
        for metric, items in sorted(values.items()):
            if all(item.denominator is not None for item in items):
                numerator = sum(item.numerator or 0 for item in items)
                denominator = sum(item.denominator or 0 for item in items)
                metrics[metric] = numerator / denominator
                sample_counts[metric] = denominator
            else:
                metrics[metric] = sum(item.value for item in items) / len(items)
                sample_counts[metric] = len(items)
        return {
            "metrics": metrics,
            "sample_counts": sample_counts,
            "complete": complete,
            "errors": errors,
        }
