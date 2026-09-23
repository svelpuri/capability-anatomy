from __future__ import annotations

from typing import Any, Iterable, Mapping

from ...domain import EvaluationObservation, EvaluationScore, NormalizedRecord, ObservationStatus, ParseResult, ParseStatus
from ...protocols import CORE_API_VERSION


class SyntheticExactMatchSuite:
    name = "reference.synthetic-exact-match"
    version = "1"
    api_version = CORE_API_VERSION
    capabilities = frozenset({"evaluation.parse", "evaluation.score", "evaluation.aggregate"})

    def required_record_schema(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "required": ["id", "partition", "input", "expected", "metadata"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "partition": {"type": "string", "minLength": 1},
                "input": {},
                "expected": {},
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        }

    def build_request(self, example: NormalizedRecord) -> Any:
        return example.input

    def parse(self, result: Any) -> ParseResult:
        if not isinstance(result, str):
            raise TypeError("synthetic output must be text")
        return ParseResult(ParseStatus.PARSED, result.strip())

    def score(self, example: NormalizedRecord, output: ParseResult) -> Mapping[str, EvaluationScore]:
        value = float(output.value == example.expected)
        return {"exact_match": EvaluationScore(value, value, 1)}

    def aggregate(self, observations: Iterable[EvaluationObservation]) -> Mapping[str, Any]:
        all_observations = tuple(observations)
        complete = [item for item in all_observations if item.status is ObservationStatus.COMPLETE]
        scores = [item.scores["exact_match"] for item in complete]
        numerator = sum(score.numerator or 0 for score in scores)
        denominator = sum(score.denominator or 0 for score in scores)
        return {
            "metrics": {"exact_match": numerator / denominator} if denominator else {},
            "sample_counts": {"exact_match": denominator} if denominator else {},
            "complete": len(complete),
            "errors": sum(item.status is ObservationStatus.ERROR for item in all_observations),
        }
