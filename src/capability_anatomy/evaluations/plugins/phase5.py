from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Mapping

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ...domain import EvaluationObservation, EvaluationScore, NormalizedRecord, ObservationStatus, ParseResult, ParseStatus, ScoreReason
from ...models.base import GenerationRequest, PerplexityRequest
from ...protocols import CORE_API_VERSION
from ...telemetry import OperationTelemetry
from ..reduction import reduce_scores


def _normalize(value: Any) -> Any:
    """Exact typed answers, with outer whitespace ignored only for text.

    Punctuation, case, Unicode code points, and internal whitespace carry
    meaning in identifiers and numbers. Alternative spellings belong in the
    frozen expected-answer list, never in a lossy global normalizer.
    """
    if isinstance(value, str):
        return ("text", value.strip())
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, list):
        return ("list", tuple(_normalize(item) for item in value))
    if isinstance(value, dict):
        return ("object", tuple(sorted((key, _normalize(item)) for key, item in value.items())))
    return (type(value).__name__, value)


# A deliberately narrow, auditable output contract. Other prose is unscorable,
# not evidence that a damaged model deliberately declined a tool invocation.
_ABSTENTION_FORMS = frozenset({
    "no applicable tool", "no applicable tool.",
    "no suitable tool is available", "no suitable tool is available.",
    "i cannot complete this request with the available tools.",
})


def _abstention_decision(raw: Any) -> tuple[float, str]:
    if not isinstance(raw, str) or not raw.strip():
        return 0.0, "abstention_empty_or_invalid_type"
    text = raw.strip()
    if text.casefold() in _ABSTENTION_FORMS:
        return 1.0, "abstention_explicit_form"
    try:
        calls = _parse_calls(text)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return 0.0, "abstention_unrecognized_or_malformed"
    if not calls:
        return 1.0, "abstention_empty_call_list"
    return 0.0, "abstention_tool_invocation"


def _parse_calls(output: str) -> list[dict[str, Any]]:
    tagged = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output, flags=re.DOTALL)
    value = [json.loads(item) for item in tagged] if tagged else json.loads(output.strip())
    if not isinstance(value, list) or any(
        not isinstance(call, dict)
        or set(call) != {"name", "arguments"}
        or not isinstance(call["name"], str)
        or not isinstance(call["arguments"], dict)
        for call in value
    ):
        raise ValueError("tool output is not a list of calls")
    return value


def _arguments_match(arguments: dict[str, Any], expected: dict[str, Any], schema: dict[str, Any]) -> bool:
    _name, allowed = next(iter(expected.items()))
    properties = set(schema["argument_names"])
    required = set(schema["required"])
    if not required.issubset(arguments) or not set(arguments).issubset(properties & set(allowed)):
        return False
    for key, alternatives in allowed.items():
        if key not in arguments:
            if key in required or "" not in alternatives:
                return False
            continue
        if _normalize(arguments[key]) not in [_normalize(item) for item in alternatives if item != ""]:
            return False
    return True


def _format_score(output: str, contract: Mapping[str, Any]) -> float:
    kind = contract["kind"]
    value = contract["value"]
    if kind == "json_keys":
        parsed = json.loads(output)
        return float(isinstance(parsed, dict) and list(parsed) == value)
    if kind == "ordered_list":
        return float(output.splitlines() == [f"{index + 1}. {item}" for index, item in enumerate(value)])
    if kind == "exact_prefix":
        return float(output.startswith(value) and "\n" not in output)
    if kind == "case_transform":
        return float(output == value)
    raise ValueError("unknown format contract")


class Phase5EvaluationSuite:
    name = "reference.phase5-qwen-retention"
    version = "2"
    api_version = CORE_API_VERSION
    capabilities = frozenset({
        "evaluation.phase5", "evaluation.independent_abstention", "evaluation.independent_binding",
        "request.chat_generation", "request.native_tools", "request.perplexity",
    })

    def __init__(self, max_new_tokens: int = 128, telemetry: OperationTelemetry | None = None) -> None:
        self.max_new_tokens = max_new_tokens
        self._telemetry = telemetry or OperationTelemetry.create("evaluation")

    def required_record_schema(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "required": ["id", "partition", "input", "expected", "metadata"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "partition": {"enum": ["discovery", "validation"]},
                "input": {}, "expected": {},
                "metadata": {"type": "object", "required": ["kind", "group_id"]},
            },
            "additionalProperties": False,
        }

    def build_request(self, example: NormalizedRecord) -> GenerationRequest | PerplexityRequest:
        kind = example.metadata["kind"]
        if kind == "perplexity":
            return PerplexityRequest(str(example.input["text"]))
        return GenerationRequest(
            messages=tuple(example.input["messages"]),
            tools=tuple(example.input["tools"]) if example.input.get("tools") is not None else None,
            max_new_tokens=self.max_new_tokens,
        )

    def parse(self, result: Any) -> ParseResult:
        if not isinstance(result, (str, float)):
            raise TypeError("Phase 5 output must be generated text or perplexity")
        return ParseResult(ParseStatus.PARSED, result)

    def score(self, example: NormalizedRecord, output: ParseResult) -> Mapping[str, EvaluationScore]:
        with self._telemetry.tracer.start_as_current_span(
            "capability_anatomy.evaluation.phase5_score",
            record_exception=False, set_status_on_exception=False,
        ) as span:
            span.set_attribute("capability_anatomy.evaluation.version", self.version)
            try:
                scores = self._score(example, output)
            except Exception:
                self._telemetry.record(span, operation="score_phase5", outcome="failed", reason="metric_scoring_failed")
                span.set_status(Status(StatusCode.ERROR, "metric_scoring_failed"))
                raise
            for metric, score in scores.items():
                span.add_event("metric_scored", {"capability_anatomy.metric_id": metric, "capability_anatomy.metric_value": score.value})
            self._telemetry.record(span, operation="score_phase5", outcome="completed", reason="versioned_metric_scores_recorded")
            return scores

    def _score(self, example: NormalizedRecord, output: ParseResult) -> Mapping[str, EvaluationScore]:
        kind = example.metadata["kind"]
        raw = output.value
        if kind == "simple":
            try:
                calls = _parse_calls(raw)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                calls = []
            if len(calls) != 1 or len(example.expected) != 1:
                values = (0.0, 0.0, 0.0)
            else:
                expected_name = next(iter(example.expected[0]))
                selection = float(calls[0]["name"] == expected_name)
                binding = float(_arguments_match(calls[0]["arguments"], example.expected[0], example.metadata["scoring_schema"]))
                values = (selection, binding, selection * binding)
            return {
                metric: EvaluationScore(value, value, 1)
                for metric, value in zip(("tool_selection", "argument_binding", "full_call"), values, strict=True)
            }
        if kind == "abstention":
            value, reason = _abstention_decision(raw)
            self._telemetry.record(trace.get_current_span(), operation="score_abstention", outcome="accepted" if value else "rejected", reason=reason)
            return {"abstention": EvaluationScore(value, value, 1, reason=ScoreReason(reason))}
        if kind == "reasoning":
            expected = example.expected if isinstance(example.expected, list) else [example.expected]
            value = float(_normalize(raw) in {_normalize(item) for item in expected})
            return {"structured_reasoning": EvaluationScore(value, value, 1)}
        if kind == "format":
            try:
                value = _format_score(raw, example.expected)
            except json.JSONDecodeError:
                value = 0.0
            return {"instruction_format": EvaluationScore(value, value, 1)}
        if kind == "perplexity":
            if not isinstance(raw, float) or not math.isfinite(raw) or raw <= 0:
                raise ValueError("perplexity must be finite and positive")
            return {"perplexity": EvaluationScore(raw)}
        raise ValueError("unknown Phase 5 record kind")

    def aggregate(self, observations: Iterable[EvaluationObservation]) -> Mapping[str, Any]:
        values: dict[str, list[EvaluationScore]] = {}
        errors = 0
        complete = 0
        for observation in observations:
            if observation.status is not ObservationStatus.COMPLETE:
                errors += 1
                continue
            complete += 1
            for metric, score in observation.scores.items():
                values.setdefault(metric, []).append(score)
        metrics: dict[str, float] = {}
        fractions: dict[str, dict[str, float | int] | None] = {}
        for metric, scores in sorted(values.items()):
            reduced = reduce_scores({"value": score.value, "numerator": score.numerator,
                                     "denominator": score.denominator} for score in scores)
            metrics[metric] = reduced["value"]
            fractions[metric] = (
                {"numerator": reduced["numerator"], "denominator": reduced["denominator"]}
                if reduced["reducer"] == "ratio_of_sums" else None
            )
        return {"metrics": metrics, "fractions": fractions, "complete": complete, "errors": errors}
