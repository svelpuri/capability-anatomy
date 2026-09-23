from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from ..errors import InvalidEvidenceError
from ..evaluations.reduction import reduce_scores
from ..serialization import canonical_json_bytes, canonical_sha256
from ..review_policy import validate_review_reference
from ..telemetry import OperationTelemetry
from .store import FrozenRunStore
from .phase5_artifacts import Phase5ArtifactReader, artifact_paths, _safe_artifact_path
from .evidence_json import encode_json, validate_tree, parse_json, parse_json_lines
from .evidence_limits import artifact_limit
from . import secure_fs

#: Damage is a difference of rates over small integer denominators, so a cell
#: can land exactly on a threshold: the finest quantum in this protocol is
#: 1/40 and the thresholds are 1/20 and 1/10. The baseline is a float mean of
#: three positions and carries up to ~1e-16 of residue, which is enough to
#: decide a strict comparison at a tie. It decided three labels in the
#: Qwen3-1.7B scan: the same one-example regression fired on tool_selection
#: (baseline 1.0 -> 0.050000000000000044) and not on argument_binding
#: (baseline 0.9499999999999998 -> 0.04999999999999982). This tolerance sits
#: far below any representable signal and far above the residue, so a tie
#: resolves as "not over the threshold", which is what the frozen rule says.
_DAMAGE_TIE_EPSILON = 1e-9


def _require_semantics_version(version: str) -> str:
    if not isinstance(version, str) or version not in {"1", "2"}:
        raise InvalidEvidenceError("Phase 5 evaluation semantics version is unsupported")
    return version


def _manifest_semantics_version(manifest: Mapping[str, Any]) -> str:
    plugins = manifest.get("plugins")
    if not isinstance(plugins, list) or any(not isinstance(item, Mapping) for item in plugins):
        raise InvalidEvidenceError("Phase 5 evaluation identity is missing or unsupported")
    evaluations = [item for item in plugins if item.get("role") == "evaluation"]
    if len(evaluations) != 1 or evaluations[0].get("name") != "reference.phase5-qwen-retention":
        raise InvalidEvidenceError("Phase 5 evaluation identity is missing or unsupported")
    return _require_semantics_version(evaluations[0].get("version"))


def _damage_exceeds_threshold(damage: float, threshold: float) -> bool:
    """Current discovery/validation rule; legacy v1 validation remains strict.

    Offline v1 reconstruction preserves the original stage-specific rules.
    Its old discovery tie tolerance does not retroactively change validation.
    """
    return damage > threshold + _DAMAGE_TIE_EPSILON


def _record_classification(component_id: str, outcome: str, reasons: Sequence[str], operation: str) -> None:
    signals = OperationTelemetry.create("evaluation")
    with signals.tracer.start_as_current_span(f"capability_anatomy.evaluation.{operation}") as span:
        span.set_attribute("capability_anatomy.component_id_sha256", canonical_sha256(component_id))
        for reason in reasons or ["observed_sensitivity_within_limits"]:
            # Prefixes identify metrics in the report; only the bounded reason
            # suffix crosses the telemetry boundary.
            signals.record(span, operation=operation, outcome=outcome, reason=reason.rsplit(":", 1)[-1])

_FORBIDDEN_EVIDENCE_KEYS = frozenset({"prompt", "raw_output", "output_text"})
_METRICS_BY_KIND = {
    "simple": ("tool_selection", "argument_binding", "full_call"),
    "abstention": ("abstention",),
    "reasoning": ("structured_reasoning",),
    "format": ("instruction_format",),
    "perplexity": ("perplexity",),
}


@dataclass(frozen=True)
class CampaignTask:
    id: str
    stage: str
    condition: str
    component_id: str | None = None
    control_position: str | None = None


def build_campaign_plan(
    component_order: Sequence[str], controls: Mapping[str, Any], *, seed: int
) -> tuple[CampaignTask, ...]:
    """Build the result-independent discovery plan frozen before inference."""
    components = tuple(component_order)
    if not components or len(set(components)) != len(components):
        raise InvalidEvidenceError("Phase 5 component plan is incomplete or duplicated")
    positions = tuple(controls.get("repeated_baseline_positions", ()))
    if positions != ("beginning", "middle", "end"):
        raise InvalidEvidenceError("Phase 5 baseline control positions are invalid")
    fraction = controls.get("placement", {}).get("middle_after_fraction")
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise InvalidEvidenceError("Phase 5 middle control placement is invalid")
    random_count = controls.get("matched_random_components")
    if not isinstance(random_count, int) or random_count < 0 or random_count > len(components):
        raise InvalidEvidenceError("Phase 5 matched-control count is invalid")

    random_components = tuple(random.Random(seed).sample(list(components), random_count))
    middle_index = min(len(components) - 1, max(1, math.ceil(len(components) * fraction)))
    tasks = [CampaignTask("baseline.beginning", "baseline", "baseline", control_position="beginning")]
    if controls.get("no_op_hook") is True:
        tasks.append(CampaignTask("control.no-op", "control", "no_op"))
    for index, component in enumerate(components):
        if index == middle_index:
            tasks.append(CampaignTask("baseline.middle", "baseline", "baseline", control_position="middle"))
        tasks.append(CampaignTask(f"scan.{component}", "scan", "bypass", component))
    tasks.extend(
        CampaignTask(f"control.random.{component}", "control", "matched_random", component)
        for component in random_components
    )
    tasks.append(CampaignTask("baseline.end", "baseline", "baseline", control_position="end"))
    if len({task.id for task in tasks}) != len(tasks):
        raise InvalidEvidenceError("Phase 5 campaign task IDs are duplicated")
    return tuple(tasks)


def serialize_campaign_plan(tasks: Iterable[CampaignTask]) -> list[dict[str, Any]]:
    return [asdict(task) for task in tasks]


def aggregate_observations(
    observations: Sequence[Mapping[str, Any]], required_metrics: Iterable[str], *, semantics_version: str = "2"
) -> dict[str, Any]:
    """Aggregate only complete durable rows while retaining reconstructable evidence."""
    _require_semantics_version(semantics_version)
    required = tuple(sorted(set(required_metrics)))
    identities: set[tuple[Any, ...]] = set()
    for row in observations:
        identity = tuple(row.get(key) for key in ("condition", "component_id", "partition", "example_id", "repetition"))
        if None in identity or identity in identities:
            raise InvalidEvidenceError("Phase 5 observation identity is missing or duplicated")
        identities.add(identity)
        expected = row.get("expected_metric_ids")
        scores = row.get("scores")
        if (
            not isinstance(expected, list)
            or not expected
            or len(set(expected)) != len(expected)
            or not isinstance(scores, Mapping)
            or set(scores) != set(expected)
        ):
            raise InvalidEvidenceError("Phase 5 per-record metric evidence is incomplete")
    result: dict[str, Any] = {}
    for metric_id in required:
        values: list[float] = []
        numerators: list[float] = []
        denominators: list[int] = []
        repetition_scores: dict[int, list[Mapping[str, Any]]] = {}
        errors = 0
        for row in observations:
            if row.get("status") != "complete":
                errors += 1
                continue
            score = row.get("scores", {}).get(metric_id)
            if score is None:
                continue
            if not isinstance(score, Mapping) or not isinstance(score.get("value"), (int, float)):
                raise InvalidEvidenceError(f"Phase 5 required metric is missing: {metric_id}")
            value = float(score["value"])
            if not math.isfinite(value):
                raise InvalidEvidenceError("Phase 5 metric is not finite")
            values.append(value)
            repetition_scores.setdefault(int(row["repetition"]), []).append(score)
            numerator, denominator = score.get("numerator"), score.get("denominator")
            if (numerator is None) != (denominator is None):
                raise InvalidEvidenceError("Phase 5 metric fraction is incomplete")
            if numerator is not None:
                if not isinstance(numerator, (int, float)) or not isinstance(denominator, int) or denominator <= 0:
                    raise InvalidEvidenceError("Phase 5 metric fraction is invalid")
                numerators.append(float(numerator))
                denominators.append(denominator)
        if not values:
            raise InvalidEvidenceError(f"Phase 5 required metric is missing complete values: {metric_id}")
        reduced = reduce_scores(score for scores in repetition_scores.values() for score in scores)
        total_denominator = sum(denominators) if denominators else None
        total_numerator = sum(numerators) if numerators else None
        aggregate_value = (
            total_numerator / total_denominator
            if total_denominator is not None
            else sum(values) / len(values)
        )
        if semantics_version == "2":
            aggregate_value = reduced["value"]
        repetition_values = []
        for repetition in sorted(repetition_scores):
            scores = repetition_scores[repetition]
            repeated = reduce_scores(scores)
            if semantics_version == "2":
                repetition_values.append(repeated["value"])
            elif all(score.get("denominator") is not None for score in scores):
                denominator = sum(int(score["denominator"]) for score in scores)
                repetition_values.append(
                    sum(float(score["numerator"]) for score in scores) / denominator
                )
            else:
                repetition_values.append(
                    sum(float(score["value"]) for score in scores) / len(scores)
                )
        if len(repetition_values) > 1:
            repetition_mean = sum(repetition_values) / len(repetition_values)
            variance = sum(
                (value - repetition_mean) ** 2 for value in repetition_values
            ) / (len(repetition_values) - 1)
            half_width = 1.96 * math.sqrt(variance / len(repetition_values))
            uncertainty = [repetition_mean - half_width, repetition_mean + half_width]
        else:
            uncertainty = None
        result[metric_id] = {
            "value": aggregate_value,
            "numerator": total_numerator,
            "denominator": total_denominator,
            "complete_count": len(values),
            "error_count": errors,
            "repetition_values": repetition_values,
            "minimum": min(repetition_values),
            "maximum": max(repetition_values),
            "uncertainty_95": uncertainty,
        }
        if semantics_version == "2":
            contributing = [row for row in observations if row.get("status") == "complete" and metric_id in row.get("scores", {})]
            record_ids = sorted({str(row["example_id"]) for row in contributing})
            group_ids = sorted({str(row["group_id"]) for row in contributing if row.get("group_id") is not None})
            result[metric_id].pop("uncertainty_95")
            result[metric_id].update({
                "semantics_version": "2",
                "record_ids": record_ids,
                "group_ids": group_ids,
                "unique_record_count": len(record_ids),
                "unique_group_count": len(group_ids) if len(group_ids) and all(row.get("group_id") is not None for row in contributing) else None,
                "repetition_standard_deviation": math.sqrt(variance) if len(repetition_values) > 1 else None,
                "sample_uncertainty_95": None,
                "sample_uncertainty_status": "not_estimated_records_not_assumed_independent",
                "retained_competence": "not_established",
            })
    return result


def summarize_measurements(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, float | int | None]]:
    """Preserve comparable runtime measurements without calling allocation a peak."""
    fields = (
        "elapsed_seconds",
        "memory_observation_bytes",
        "input_tokens",
        "output_tokens",
    )
    summary: dict[str, Mapping[str, float | int | None]] = {}
    complete = [row for row in observations if row.get("status") == "complete"]
    for field in fields:
        values = [row[field] for row in complete if isinstance(row.get(field), (int, float))]
        summary[field] = {
            "count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    return summary


def compute_damage_matrix(
    baseline: Mapping[str, Mapping[str, Any]],
    conditions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metric_protocol: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for condition_id, metrics in sorted(conditions.items()):
        row: dict[str, Any] = {}
        for metric_id, specification in sorted(metric_protocol.items()):
            if metric_id not in baseline or metric_id not in metrics:
                raise InvalidEvidenceError(f"Phase 5 damage metric is missing: {metric_id}")
            base, condition = baseline[metric_id], metrics[metric_id]
            base_value, condition_value = float(base["value"]), float(condition["value"])
            if specification["direction"] == "higher_is_better":
                damage = base_value - condition_value
                relative = damage / abs(base_value) if base_value else None
            elif specification["direction"] == "lower_is_better":
                if base_value <= 0:
                    raise InvalidEvidenceError("Phase 5 lower-is-better baseline must be positive")
                damage = (condition_value - base_value) / base_value
                relative = damage
            else:
                raise InvalidEvidenceError("Phase 5 metric direction is invalid")
            row[metric_id] = {
                "role": specification["role"],
                "baseline": dict(base),
                "condition": dict(condition),
                "absolute_damage": damage,
                "relative_damage": relative,
            }
        matrix[condition_id] = row
    return matrix


def aggregate_baseline_controls(
    baselines: Mapping[str, Mapping[str, Mapping[str, Any]]],
    no_op: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Use the mean repeated baseline and maximum repeated/no-op deviation as drift."""
    if set(baselines) != {"beginning", "middle", "end"}:
        raise InvalidEvidenceError("Phase 5 repeated baseline controls are incomplete")
    metric_ids = set(baselines["beginning"])
    if not metric_ids or any(set(value) != metric_ids for value in baselines.values()) or set(no_op) != metric_ids:
        raise InvalidEvidenceError("Phase 5 baseline control metrics disagree")
    reference: dict[str, Any] = {}
    drift: dict[str, float] = {}
    for metric_id in sorted(metric_ids):
        values = [float(baselines[position][metric_id]["value"]) for position in ("beginning", "middle", "end")]
        mean = sum(values) / len(values)
        fractions = [baselines[position][metric_id] for position in ("beginning", "middle", "end")]
        reference[metric_id] = {
            "value": mean,
            "numerator": sum(float(item["numerator"]) for item in fractions) if all(item.get("numerator") is not None for item in fractions) else None,
            "denominator": sum(int(item["denominator"]) for item in fractions) if all(item.get("denominator") is not None for item in fractions) else None,
            "complete_count": sum(int(item.get("complete_count", 0)) for item in fractions),
            "error_count": sum(int(item.get("error_count", 0)) for item in fractions),
            "repetition_values": values,
            "minimum": min(values),
            "maximum": max(values),
            "uncertainty_95": None,
        }
        versions = {item.get("semantics_version", "1") for item in fractions}
        if len(versions) != 1:
            raise InvalidEvidenceError("Phase 5 baseline semantics versions disagree")
        if versions == {"2"}:
            record_ids = sorted({record_id for item in fractions for record_id in item["record_ids"]})
            group_ids = sorted({group_id for item in fractions for group_id in item["group_ids"]})
            reference[metric_id].pop("uncertainty_95")
            reference[metric_id].update({
                "semantics_version": "2", "record_ids": record_ids, "group_ids": group_ids,
                "unique_record_count": len(record_ids),
                "unique_group_count": len(group_ids) if all(item["unique_group_count"] is not None for item in fractions) else None,
                "baseline_position_values": values,
                "baseline_position_standard_deviation": math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)),
                "sample_uncertainty_95": None,
                "sample_uncertainty_status": "not_estimated_records_not_assumed_independent",
                "retained_competence": "not_established",
            })
        drift[metric_id] = max(*(abs(value - mean) for value in values), abs(float(no_op[metric_id]["value"]) - mean))
    return reference, drift


def select_discovery_candidates(
    damage_matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metric_protocol: Mapping[str, Mapping[str, Any]],
    rule: Mapping[str, Any],
    *,
    drift_by_metric: Mapping[str, float],
    matched_random_damage: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Apply the frozen rule; missing stability/control evidence is inconclusive."""
    matched_control_reasons: list[str] = []
    if not matched_random_damage:
        matched_control_reasons.append("matched_random_control:missing")
    for controlled_component, controlled_metrics in matched_random_damage.items():
        scan_metrics = damage_matrix.get(controlled_component)
        if scan_metrics is None:
            matched_control_reasons.append("matched_random_control:component_missing")
            continue
        for metric_id in metric_protocol:
            random_damage = controlled_metrics.get(metric_id)
            scan_damage = scan_metrics.get(metric_id, {}).get("absolute_damage")
            if random_damage is None or scan_damage is None:
                matched_control_reasons.append(f"{metric_id}:matched_control_missing")
            elif abs(float(scan_damage) - float(random_damage)) > float(rule["minimum_drift_margin"]):
                matched_control_reasons.append(f"{metric_id}:matched_control_not_reproduced")
    ranking = []
    for component_id, metrics in sorted(damage_matrix.items()):
        reasons: list[str] = list(matched_control_reasons)
        target_sensitive = collateral_sensitive = False
        for metric_id, specification in metric_protocol.items():
            evidence = metrics.get(metric_id)
            if evidence is None or evidence["condition"].get("error_count") or not evidence["condition"].get("complete_count"):
                reasons.append(f"{metric_id}:incomplete")
                continue
            if evidence.get("baseline", {}).get("semantics_version") == "2":
                baseline = evidence["baseline"]
                if specification["direction"] == "higher_is_better" and float(baseline["value"]) <= 0:
                    reasons.append(f"{metric_id}:baseline_competence_absent")
                if baseline.get("unique_record_count", 0) < 2 or (baseline.get("unique_group_count") or 0) < 2:
                    reasons.append(f"{metric_id}:insufficient_distinct_records_or_groups")
            damage = float(evidence["absolute_damage"])
            threshold = (
                rule["perplexity_relative_damage_max"]
                if specification["direction"] == "lower_is_better"
                else rule["target_absolute_damage_max"]
                if specification["role"] == "target"
                else rule["collateral_absolute_damage_max"]
            )
            if _damage_exceeds_threshold(damage, threshold):
                (reasons.append(f"{metric_id}:target_sensitive") if specification["role"] == "target" else reasons.append(f"{metric_id}:collateral_sensitive"))
                target_sensitive |= specification["role"] == "target"
                collateral_sensitive |= specification["role"] == "collateral"
            drift = float(drift_by_metric.get(metric_id, math.inf))
            if drift > float(rule["minimum_drift_margin"]):
                reasons.append(f"{metric_id}:control_drift")
            if (
                evidence["condition"].get("complete_count", 0) < 2
                or float(evidence["condition"].get("maximum", math.inf))
                - float(evidence["condition"].get("minimum", -math.inf))
                > float(rule["minimum_drift_margin"])
            ):
                reasons.append(f"{metric_id}:unstable_or_single_observation")
        if target_sensitive and collateral_sensitive:
            label = "broadly_sensitive"
        elif target_sensitive:
            label = "target_sensitive"
        elif collateral_sensitive:
            label = "collateral_sensitive"
        elif reasons:
            label = "inconclusive"
        else:
            label = "low_observed_sensitivity"
        ranking.append({"component_id": component_id, "classification": label, "eligible": not reasons, "reasons": sorted(set(reasons))})
        if any(evidence.get("baseline", {}).get("semantics_version") == "2" for evidence in metrics.values()):
            _record_classification(component_id, "accepted" if not reasons else "rejected", sorted(set(reasons)), "classify_candidate")
    return {
        "schema_version": "capability-anatomy/phase5-discovery-ranking/v1",
        "candidates": [item["component_id"] for item in ranking if item["eligible"]],
        "ranking": ranking,
    }


def compute_validation_results(
    candidates: Sequence[str],
    validation_damage: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metric_protocol: Mapping[str, Mapping[str, Any]],
    rule: Mapping[str, Any],
    *, semantics_version: str = "2",
) -> dict[str, Any]:
    _require_semantics_version(semantics_version)
    results = []
    for component in candidates:
        metrics = validation_damage.get(component)
        if metrics is None:
            results.append({"component_id": component, "outcome": "inconclusive", "reasons": ["validation_missing"]})
            if semantics_version == "2":
                _record_classification(component, "inconclusive", ["validation_missing"], "validate_candidate")
            continue
        reasons = []
        for metric_id, specification in metric_protocol.items():
            evidence = metrics.get(metric_id)
            if evidence is None or evidence["condition"].get("error_count"):
                reasons.append(f"{metric_id}:incomplete")
                continue
            threshold = (
                rule["perplexity_relative_damage_max"]
                if specification["direction"] == "lower_is_better"
                else rule["target_absolute_damage_max"]
                if specification["role"] == "target"
                else rule["collateral_absolute_damage_max"]
            )
            if semantics_version == "2":
                baseline = evidence.get("baseline", {})
                if specification["direction"] == "higher_is_better" and float(baseline.get("value", 0)) <= 0:
                    reasons.append(f"{metric_id}:baseline_competence_absent")
                if baseline.get("unique_record_count", 0) < 2 or (baseline.get("unique_group_count") or 0) < 2:
                    reasons.append(f"{metric_id}:insufficient_distinct_records_or_groups")
            exceeds = (float(evidence["absolute_damage"]) > threshold) if semantics_version == "1" else _damage_exceeds_threshold(float(evidence["absolute_damage"]), threshold)
            if exceeds:
                reasons.append(f"{metric_id}:threshold_exceeded")
        outcome = "pass" if not reasons else "inconclusive" if any(reason.endswith(("incomplete", "baseline_competence_absent", "insufficient_distinct_records_or_groups")) for reason in reasons) else "fail"
        results.append({"component_id": component, "outcome": outcome, "reasons": reasons})
        if semantics_version == "2":
            _record_classification(component, outcome, reasons, "validate_candidate")
    return {"schema_version": "capability-anatomy/phase5-validation/v1", "validation_once": True, "results": results}


def render_report_markdown(report: Mapping[str, Any]) -> str:
    candidates = report.get("candidates", [])
    validation = report.get("validation", {}).get("results", [])
    current_semantics = any(item.get("semantics_version") == "2" for item in report.get("baseline", {}).values())
    lines = [
        "# Phase 5 component scan report",
        "",
        f"Status: {report.get('status', 'unknown')}",
        "",
        "## Objective",
        "",
        str(report.get("objective", "unspecified")),
        "",
        f"Claim boundary: {report.get('claim_boundary', 'unspecified')}",
        "",
        f"Discovery candidates: {', '.join(candidates) if candidates else 'none'}",
        "",
    ]
    if current_semantics:
        lines.extend([
            "Evaluation semantics: v2. Repetition and baseline-position dispersion measure run stability only.",
            "Unique records/groups are counted once; their statistical independence is not established.",
            "Sample uncertainty and retained competence are not established. Candidate and validation labels describe only observed sensitivity.",
            "", 
        ])
    for title, key in (
        ("Provenance", "provenance"), ("Baseline and repeat stability" if current_semantics else "Baseline and uncertainty", "baseline"),
        ("Per-component target and collateral damage", "per_component_damage"),
        ("Controls and drift", "controls"), ("Candidate ranking", "candidate_ranking"),
        ("Runtime, memory, and token observations", "measurements"),
        ("Errors", "errors"),
    ):
        lines.extend([f"## {title}", "", "```json", json.dumps(report.get(key), indent=2, sort_keys=True), "```", ""])
    lines.extend([
        "## Validation",
        "",
    ])
    lines.extend(
        f"- {item['component_id']}: {item['outcome']}"
        for item in validation
    )
    if not validation:
        lines.append("- No candidates qualified for validation.")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    lines.extend(["", "## Reproduction commands", ""])
    lines.extend(f"- `{item}`" for item in report.get("reproduction_commands", []))
    return "\n".join(lines) + "\n"


def enforce_budgets(
    budgets: Mapping[str, Any], *, elapsed_seconds: float, rss_bytes: int | None,
    observation_errors: int, task_retries: int,
) -> None:
    checks = (
        (elapsed_seconds > budgets["max_wall_seconds"], "wall_budget_exceeded"),
        (
            rss_bytes is not None
            and rss_bytes > budgets["max_memory_observation_bytes"],
            "memory_budget_exceeded",
        ),
        (observation_errors > budgets["max_observation_errors"], "observation_error_budget_exceeded"),
        (task_retries > budgets["max_task_retries"], "retry_budget_exceeded"),
    )
    for exceeded, reason in checks:
        if exceeded:
            raise InvalidEvidenceError(reason)


def enforce_stop_rules(stop_rules: Iterable[str], state: Mapping[str, bool]) -> None:
    active = set(stop_rules)
    unknown = set(state) - active
    if unknown:
        raise InvalidEvidenceError(f"unfrozen Phase 5 stop condition: {sorted(unknown)}")
    triggered = sorted(reason for reason in active if state.get(reason, False))
    if triggered:
        raise InvalidEvidenceError(f"Phase 5 stop rule triggered: {triggered[0]}")


def write_final_artifacts(
    root: Path,
    artifacts: Mapping[str, Any],
    *,
    required_paths: Iterable[str],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Preflight bounded payloads, then publish under one coherent ownership lease."""
    with secure_fs.ensure_run_ownership(root):
        return _write_final_artifacts_owned(root, artifacts, required_paths=required_paths, identity=identity)


def _write_final_artifacts_owned(root, artifacts, *, required_paths, identity):
    from .evidence_limits import MAX_ARTIFACTS, MAX_BUNDLE_BYTES

    required = set()
    for relative in required_paths:
        _safe_artifact_path(root, relative)
        required.add(relative)
        if len(required) > MAX_ARTIFACTS:
            raise InvalidEvidenceError("Phase 5 required artifact count exceeds its limit")
    encoded = {}
    new_bytes = 0
    for relative, value in artifacts.items():
        _safe_artifact_path(root, relative)
        _reject_sensitive_fields(value)
        payload = _encode_artifact(relative, value)
        if len(payload) > artifact_limit(relative):
            raise InvalidEvidenceError("Phase 5 artifact exceeds its byte limit")
        new_bytes += len(payload)
        if new_bytes > MAX_BUNDLE_BYTES or len(encoded) >= MAX_ARTIFACTS:
            raise InvalidEvidenceError("Phase 5 artifact inventory exceeds its limit")
        _validate_artifact_payload(relative, payload)
        encoded[relative] = payload
    emitted = required | set(encoded) | {path.relative_to(root).as_posix() for path in artifact_paths(root)}
    emitted.discard("evidence-manifest.json")
    if len(emitted) > MAX_ARTIFACTS:
        raise InvalidEvidenceError("Phase 5 artifact inventory exceeds its count limit")
    entries = []
    reader = Phase5ArtifactReader(root)
    total_bytes = 0
    for relative in sorted(emitted):
        payload = encoded[relative] if relative in encoded else reader.read(relative)
        _validate_artifact_payload(relative, payload)
        total_bytes += len(payload)
        if total_bytes > MAX_BUNDLE_BYTES:
            raise InvalidEvidenceError("Phase 5 artifact inventory exceeds the bundle byte limit")
        entries.append({"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "schema_version": "capability-anatomy/phase5-evidence-manifest/v1",
        **dict(identity),
        "required_artifacts": sorted(required),
        "artifacts": entries,
    }
    # Use the same manifest admission contract as the verifier before writing.
    Phase5ArtifactReader(root, manifest)
    manifest_payload = encode_json(manifest)
    if len(manifest_payload) > artifact_limit("evidence-manifest.json"):
        raise InvalidEvidenceError("Phase 5 manifest exceeds its byte limit")
    for relative, payload in encoded.items():
        FrozenRunStore._atomic_write(_safe_artifact_path(root, relative), payload)
    FrozenRunStore._atomic_write(_safe_artifact_path(root, "evidence-manifest.json"), manifest_payload)
    return manifest


def reconstruct_manifest(root: Path, manifest: Mapping[str, Any], required_paths: Iterable[str]) -> None:
    if manifest.get("schema_version") != "capability-anatomy/phase5-evidence-manifest/v1":
        raise InvalidEvidenceError("Phase 5 evidence manifest schema is unsupported")
    reader = Phase5ArtifactReader(root, manifest)
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or len({item.get("path") for item in entries}) != len(entries):
        raise InvalidEvidenceError("Phase 5 evidence manifest paths are duplicated")
    expected = set(required_paths) - {"evidence-manifest.json"}
    if {item.get("path") for item in entries} != expected:
        raise InvalidEvidenceError("Phase 5 evidence manifest is incomplete")
    actual = {
        path.relative_to(root).as_posix()
        for path in artifact_paths(root)
        if path.is_file() and path.name != "evidence-manifest.json"
    }
    if actual != expected:
        raise InvalidEvidenceError("Phase 5 evidence directory contains unbound artifacts")
    for item in entries:
        path = _safe_artifact_path(root, item["path"])
        if not path.is_file() or path.is_symlink():
            raise InvalidEvidenceError("Phase 5 evidence artifact is unavailable")
        payload = reader.read(item["path"])
        if len(payload) != item.get("bytes") or hashlib.sha256(payload).hexdigest() != item.get("sha256"):
            raise InvalidEvidenceError("Phase 5 evidence artifact digest mismatch")
        _validate_artifact_payload(item["path"], payload)


def verify_phase5_aggregates(root: Path, manifest: Mapping[str, Any]) -> None:
    """Recompute controls, damage, ranking, validation, and report claims."""
    reader = Phase5ArtifactReader(root, manifest)
    try:
        observations = parse_json_lines(reader.read("observations.jsonl"))
        metrics = reader.json("metrics.json")
        protocol = reader.json("protocol.json")
        record_plan = reader.json("record-plan.json")
        controls = reader.json("controls.json")
        candidates = reader.json("candidates.json")
        validation = reader.json("validation.json")
        report = reader.json("report.json")
        report_markdown = reader.text("report.md")
        component_order = reader.json("component-scan-order.json")
        discovery_plan = reader.json("discovery-task-plan.json")
        validation_plan = reader.json("validation-task-plan.json")
        topology = reader.json("topology.json")
        configuration = reader.json("configuration.json")
        compatibility = reader.json("compatibility.json")
        provenance = reader.json("provenance.json")
        trace = reader.json("trace.json")
        failures = parse_json_lines(reader.read("failures.jsonl"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidEvidenceError("Phase 5 scientific evidence is unreadable") from error
    semantics_version = _manifest_semantics_version(manifest)
    metric_protocol = protocol.get("metrics")
    rule = protocol.get("candidate_rule")
    metric_ids = sorted(metric_protocol) if isinstance(metric_protocol, Mapping) else []
    if (
        manifest.get("metric_ids") != metric_ids
        or not metric_ids
        or not isinstance(rule, Mapping)
    ):
        raise InvalidEvidenceError("Phase 5 manifest metric contract is missing")
    try:
        planned_rows = [
            row for partition in ("discovery", "validation")
            for row in record_plan["partitions"][partition]
        ]
    except (KeyError, TypeError) as error:
        raise InvalidEvidenceError("Phase 5 bound record plan is invalid") from error
    if (
        record_plan.get("schema_version") != "capability-anatomy/phase5-record-plan/v1"
        or hashlib.sha256(reader.read("record-plan.json")).hexdigest()
        != protocol.get("dataset", {}).get("record_plan", {}).get("sha256")
        or canonical_sha256(planned_rows) != record_plan.get("rows_sha256")
        or len({row.get("source_id") for row in planned_rows}) != len(planned_rows)
        or len({row.get("group_key") for row in planned_rows}) != len(planned_rows)
    ):
        raise InvalidEvidenceError("Phase 5 bound record plan identity is invalid")
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations:
        if not isinstance(row, Mapping) or not isinstance(row.get("condition"), str):
            raise InvalidEvidenceError("Phase 5 observation condition is invalid")
        by_condition.setdefault(row["condition"], []).append(row)
    reconstructed = {
        condition: aggregate_observations(rows, metric_ids, semantics_version=semantics_version)
        for condition, rows in sorted(by_condition.items())
    }
    if metrics.get("per_condition") != reconstructed:
        raise InvalidEvidenceError("Phase 5 aggregate reconstruction mismatch")
    measurements = {
        condition: summarize_measurements(rows)
        for condition, rows in sorted(by_condition.items())
    }
    topology_components = topology.get("components") if isinstance(topology, Mapping) else None
    if (
        not isinstance(component_order, list)
        or not isinstance(topology_components, list)
        or len(component_order) != len(set(component_order))
        or set(component_order) != {item.get("id") for item in topology_components}
        or component_order != _derived_component_order(topology_components, protocol)
    ):
        raise InvalidEvidenceError("Phase 5 topology and component coverage disagree")
    seed = protocol.get("experiment", {}).get("seed")
    if not isinstance(seed, int):
        raise InvalidEvidenceError("Phase 5 protocol seed is missing")
    control_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:controls".encode()).digest()[:8], "big",
    )
    expected_discovery_plan = serialize_campaign_plan(
        build_campaign_plan(component_order, protocol.get("controls", {}), seed=control_seed),
    )
    if discovery_plan != expected_discovery_plan:
        raise InvalidEvidenceError("Phase 5 discovery plan reconstruction mismatch")
    if controls.get("plan") != expected_discovery_plan:
        raise InvalidEvidenceError("Phase 5 control plan evidence disagrees")
    discovery_ids = {task["id"] for task in expected_discovery_plan}
    validation_ids = {
        task.get("id") for task in validation_plan if isinstance(task, Mapping)
    }
    if set(by_condition) != discovery_ids | validation_ids:
        raise InvalidEvidenceError("Phase 5 frozen task observation coverage is incomplete")

    def observation_fingerprints(condition: str) -> set[tuple[Any, ...]]:
        return {
            (
                row.get("partition"), row.get("example_id"), row.get("repetition"),
                row.get("group_id"), tuple(row.get("expected_metric_ids", ())),
                row.get("component_id"),
            )
            for row in by_condition[condition]
        }
    repetitions = protocol.get("runtime", {}).get("repetitions")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise InvalidEvidenceError("Phase 5 repetition contract is invalid")
    record_seed = int.from_bytes(hashlib.sha256(f"{seed}:records".encode()).digest()[:8], "big")
    expected_measurement_controls = {
        "seed": record_seed,
        "warmup_runs": protocol["runtime"]["warmup_runs"],
        "repetitions": repetitions,
        "randomized_order": protocol["runtime"]["randomized_record_order"],
    }
    for condition, rows in by_condition.items():
        try:
            task_payload = reader.json(f"tasks/{condition}.json")
            planned_partition = record_plan["partitions"][rows[0]["partition"]]
        except (OSError, KeyError, IndexError, json.JSONDecodeError) as error:
            raise InvalidEvidenceError("Phase 5 task measurement evidence is incomplete") from error
        expected_order = list(planned_partition)
        if protocol["runtime"]["randomized_record_order"]:
            random.Random(record_seed).shuffle(expected_order)
        if (
            task_payload.get("measurement_controls") != expected_measurement_controls
            or task_payload.get("execution_order") != [item["source_id"] for item in expected_order]
            or task_payload.get("observations") != rows
        ):
            raise InvalidEvidenceError("Phase 5 task measurement evidence disagrees")

    def expected_fingerprints(partition: str, component: str) -> set[tuple[Any, ...]]:
        try:
            planned_rows = record_plan["partitions"][partition]
            return {
                (
                    partition, row["source_id"], repetition, row["group_key"],
                    _METRICS_BY_KIND[row["kind"]], component,
                )
                for row in planned_rows
                for repetition in range(repetitions)
            }
        except (KeyError, TypeError) as error:
            raise InvalidEvidenceError("Phase 5 bound record plan is invalid") from error

    for task in expected_discovery_plan:
        component = (
            "baseline" if task["condition"] == "baseline"
            else component_order[0] if task["condition"] == "no_op"
            else task["component_id"]
        )
        if observation_fingerprints(task["id"]) != expected_fingerprints("discovery", component):
            raise InvalidEvidenceError("Phase 5 sealed discovery record coverage is incomplete")
    for task in validation_plan:
        component = "baseline" if task.get("condition") == "baseline" else task.get("component_id")
        if observation_fingerprints(task["id"]) != expected_fingerprints("validation", component):
            raise InvalidEvidenceError("Phase 5 sealed validation record coverage is incomplete")
    try:
        baseline, drift = aggregate_baseline_controls(
            {
                position: reconstructed[f"baseline.{position}"]
                for position in ("beginning", "middle", "end")
            },
            reconstructed["control.no-op"],
        )
        scan_conditions = {
            rows[0]["component_id"]: reconstructed[condition]
            for condition, rows in by_condition.items()
            if condition.startswith("scan.")
        }
        damage = compute_damage_matrix(baseline, scan_conditions, metric_protocol)
        random_conditions = {
            rows[0]["component_id"]: reconstructed[condition]
            for condition, rows in by_condition.items()
            if condition.startswith("control.random.")
        }
        random_matrix = compute_damage_matrix(baseline, random_conditions, metric_protocol)
        random_damage = {
            component: {
                metric: evidence["absolute_damage"] for metric, evidence in values.items()
            }
            for component, values in random_matrix.items()
        }
        ranking = select_discovery_candidates(
            damage, metric_protocol, rule,
            drift_by_metric=drift, matched_random_damage=random_damage,
        )
        validation_baseline = reconstructed["validation.baseline"]
        validation_conditions = {
            rows[0]["component_id"]: reconstructed[condition]
            for condition, rows in by_condition.items()
            if condition.startswith("validation.") and condition != "validation.baseline"
        }
        validation_damage = compute_damage_matrix(
            validation_baseline, validation_conditions, metric_protocol,
        )
        validation_result = compute_validation_results(
            ranking["candidates"], validation_damage, metric_protocol, rule, semantics_version=semantics_version,
        )
    except (KeyError, IndexError, TypeError) as error:
        raise InvalidEvidenceError("Phase 5 scientific evidence is incomplete") from error
    if (
        report.get("status") != "complete"
        or not isinstance(report.get("limitations"), list)
        or not report["limitations"]
        or not isinstance(report.get("reproduction_commands"), list)
        or not report["reproduction_commands"]
        or metrics.get("baseline") != baseline
        or metrics.get("damage_matrix") != damage
        or metrics.get("validation_damage") != validation_damage
        or metrics.get("measurements") != measurements
        or controls.get("drift") != drift
        or controls.get("matched_random_damage") != random_damage
        or candidates != ranking
        or validation != validation_result
        or report.get("candidates") != ranking["candidates"]
        or report.get("validation") != validation_result
        or report.get("claim_boundary") != protocol.get("claim_boundary")
        or report.get("objective") != "Measure temporary single-block sensitivity for tool calling and collateral capabilities under the frozen Phase 5 protocol."
        or report.get("provenance") != provenance
        or report.get("baseline") != baseline
        or report.get("per_component_damage") != damage
        or report.get("controls") != {"drift": drift, "matched_random_damage": random_damage}
        or report.get("candidate_ranking") != ranking["ranking"]
        or report.get("measurements") != measurements
        or report.get("errors") != failures
        or report_markdown != render_report_markdown(report)
        or validation_plan != [
            {"id": "validation.baseline", "condition": "baseline", "component_id": None},
            *[
                {"id": f"validation.{component}", "condition": "bypass", "component_id": component}
                for component in ranking["candidates"]
            ],
        ]
    ):
        raise InvalidEvidenceError("Phase 5 scientific derivation mismatch")
    _verify_operational_evidence(
        configuration, compatibility, topology, provenance, trace, report,
        protocol, manifest, root,
    )


def _derived_component_order(
    topology_components: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
) -> list[str]:
    """Re-derive the randomized scan order from its seed.

    `freeze_sequence` deliberately lets the stored order win over whatever a
    resumed run proposes, which is what makes a resume stable. Nothing else
    checked that the stored order was the one the seed produces, so an edited
    `component-scan-order.json` would have survived into a fully manifest-bound
    bundle. This closes that by re-deriving it here.
    """
    seed = protocol.get("experiment", {}).get("seed")
    if not isinstance(seed, int):
        return []
    ordered = [component.get("id") for component in topology_components]
    if protocol.get("runtime", {}).get("randomized_execution_order") is not False:
        digest = hashlib.sha256(f"{seed}:components".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(ordered)
    return ordered


def _bypass_hooks_balanced(segments: Sequence[Mapping[str, Any]]) -> bool:
    """Every activated block bypass must reach a recorded terminal cleanup.

    `block_bypass` always runs its release in a `finally`, then records exactly
    one terminal decision per activation: `hooks_removed` when the body
    succeeded, `experiment_body_failed` when it did not, and `cleanup_failed`
    when the release itself raised. So the accounting is exact even across a
    resume -- an interrupted segment reports `experiment_body_failed`, not a
    missing event -- and no tolerance for dangling activations is needed or
    permitted. `cleanup_failed` is the one outcome that means hooks may still be
    installed, so it is prohibited outright. Counting per segment additionally
    prevents a leak in one segment from being masked by a surplus in another.
    """
    for segment in segments:
        counts: dict[str, int] = {}
        for span in segment.get("spans", []):
            for event in span.get("events", []):
                if event.get("name") != "operation.decision":
                    continue
                attributes = event.get("attributes", {})
                if (
                    attributes.get("capability_anatomy.component") != "intervention"
                    or attributes.get("capability_anatomy.operation") != "block_bypass"
                ):
                    continue
                reason = attributes.get("capability_anatomy.reason")
                counts[reason] = counts.get(reason, 0) + 1
        if counts.get("cleanup_failed", 0):
            return False
        if counts.get("scoped_bypass_active", 0) != (
            counts.get("hooks_removed", 0) + counts.get("experiment_body_failed", 0)
        ):
            return False
    return True


def _validate_trace_segment_graph(segment: Mapping[str, Any], *, latest: bool) -> None:
    """Validate completed execution graphs and explicitly declared open parents."""
    from .core_evidence import trace_completeness
    trace_completeness(segment)
    spans = segment.get("spans")
    trace_id = segment.get("trace_id")
    if (not isinstance(trace_id, str) or not re.fullmatch(r"[0-9a-f]{32}", trace_id)
            or not trace_id.strip("0") or not isinstance(spans, list) or not spans
            or any(not isinstance(span, Mapping) for span in spans)):
        raise InvalidEvidenceError("Phase 5 trace graph is invalid")
    ids = [span.get("span_id") for span in spans]
    if (any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value)
            or not value.strip("0") for value in ids) or len(set(ids)) != len(ids)):
        raise InvalidEvidenceError("Phase 5 trace graph is invalid")
    if any(span.get("parent_span_id") is not None and (
            not isinstance(span["parent_span_id"], str)
            or not re.fullmatch(r"[0-9a-f]{16}", span["parent_span_id"])
            or not span["parent_span_id"].strip("0")) for span in spans):
        raise InvalidEvidenceError("Phase 5 trace graph is invalid")
    by_id = {span["span_id"]: span for span in spans}
    declared = "external_parent_span_ids" in segment or "execution_parent_span_id" in segment
    if declared:
        anchors = segment.get("external_parent_span_ids")
        execution_parent = segment.get("execution_parent_span_id")
        if (segment.get("schema_version") != "capability-anatomy/phase5-trace/v2"
                or not isinstance(anchors, list) or not 1 <= len(anchors) <= 2
                or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value)
                       or not value.strip("0") for value in anchors)
                or len(set(anchors)) != len(anchors) or set(anchors) & set(ids)
                or execution_parent not in anchors
                or any(span.get("trace_id") != trace_id for span in spans)):
            raise InvalidEvidenceError("Phase 5 trace graph is invalid")
        root_names = {"capability_anatomy.phase5.scan"}
        if not latest:
            root_names.add("capability_anatomy.phase5.conformance")
        roots = [span for span in spans if span.get("name") in root_names]
        external = {span.get("parent_span_id") for span in spans if span.get("parent_span_id") not in by_id}
        if external != set(anchors) or len(roots) != 1 or roots[0].get("parent_span_id") != execution_parent:
            raise InvalidEvidenceError("Phase 5 trace graph is invalid")
        if roots[0].get("name") == "capability_anatomy.phase5.conformance" and roots[0].get("status") != "ERROR":
            raise InvalidEvidenceError("Phase 5 trace graph is invalid")
        terminals = set(anchors)
    else:
        roots = [span for span in spans if span.get("parent_span_id") is None]
        if len(roots) != 1 or roots[0].get("name") != "capability_anatomy.phase5.scan":
            raise InvalidEvidenceError("Phase 5 trace graph is invalid")
        terminals = {roots[0]["span_id"]}
    if latest and roots[0].get("status") not in {"UNSET", "OK"}:
        raise InvalidEvidenceError("Phase 5 trace graph is invalid")
    for span in spans:
        current = span["span_id"]
        seen = set()
        while current not in terminals:
            if current in seen or current not in by_id:
                raise InvalidEvidenceError("Phase 5 trace graph is invalid")
            seen.add(current)
            current = by_id[current].get("parent_span_id")


def _verify_operational_evidence(
    configuration: Mapping[str, Any], compatibility: Mapping[str, Any],
    topology: Mapping[str, Any], provenance: Mapping[str, Any],
    trace: Mapping[str, Any], report: Mapping[str, Any],
    protocol: Mapping[str, Any], manifest: Mapping[str, Any], root: Path,
) -> None:
    reader = Phase5ArtifactReader(root, manifest)
    model = protocol["model"]
    runtime = protocol["runtime"]
    configured_runtime = configuration.get("runtime", {})
    parameters = configured_runtime.get("parameters", {}) if isinstance(configured_runtime, Mapping) else {}
    protocol_plugins = {item["role"]: item for item in protocol.get("plugins", [])}
    manifest_plugins = {item.get("role"): item for item in manifest.get("plugins", [])}
    expected_plugin_identity = {
        role: {key: specification[key] for key in ("role", "name", "version", "api_version", "capabilities")}
        for role, specification in protocol_plugins.items()
    }
    actual_plugin_identity = {
        role: {key: identity.get(key) for key in ("role", "name", "version", "api_version", "capabilities")}
        for role, identity in manifest_plugins.items()
    }
    authorization = manifest.get("authorization", {})
    validate_review_reference(protocol, authorization, allow_legacy=True)
    expected_dataset = protocol.get("dataset", {})
    immutable_failures = []
    for path in sorted((root / "failures").glob("*.attempt-*.json")) if (root / "failures").exists() else ():
        immutable_failures.append(reader.json(path.relative_to(root).as_posix()))
    summarized_failures = parse_json_lines(reader.read("failures.jsonl"))
    topology_components = topology.get("components", [])
    topology_complete = (
        isinstance(topology.get("architecture"), str)
        and bool(topology["architecture"])
        and bool(topology_components)
        and all(
            set(component) == {"id", "kind", "parent_id", "order", "metadata"}
            and isinstance(component["id"], str)
            and isinstance(component["kind"], str)
            and component["order"] == index
            and component["parent_id"] is None
            and isinstance(component["metadata"].get("module_path"), str)
            and isinstance(component["metadata"].get("parameter_count"), int)
            and component["metadata"].get("parameter_count") > 0
            and component["metadata"].get("original_order") == index
            and component["metadata"].get("architecture") == topology.get("architecture")
            and component["metadata"].get("architecture_profile") == model.get("architecture_profile")
            and component["metadata"].get("architecture_profile_sha256") == model.get("architecture_profile_sha256")
            for index, component in enumerate(topology_components)
        )
    )
    if (
        configuration.get("schema_version") != "capability-anatomy/experiment-config/v1"
        or configuration.get("experiment_id") != protocol["experiment"]["id"]
        or configuration.get("seed") != protocol["experiment"]["seed"]
        or configuration.get("model", {}).get("plugin") != model["plugin"]
        or configuration.get("model", {}).get("source") != model["source"]
        or configuration.get("model", {}).get("revision") != model["revision"]
        or configuration.get("model", {}).get("parameters", {}).get("architecture_profile") != model.get("architecture_profile")
        or configuration.get("model", {}).get("parameters", {}).get("dtype") != runtime.get("dtype")
        or configured_runtime.get("deterministic") != runtime["deterministic"]
        or configured_runtime.get("warmup_runs") != runtime["warmup_runs"]
        or configured_runtime.get("repetitions") != runtime["repetitions"]
        or configured_runtime.get("randomized_execution_order") is not True
        or configured_runtime.get("executor") != protocol_plugins.get("runtime", {}).get("name")
        or parameters.get("device") != runtime["device"]
        or parameters.get("context_length") != runtime["context_length"]
        or configuration.get("capability", {}).get("evaluation_plugin") != protocol_plugins.get("evaluation", {}).get("name")
        or sorted(configuration.get("capability", {}).get("target_metrics", [])) != sorted(key for key, value in protocol["metrics"].items() if value["role"] == "target")
        or sorted(configuration.get("capability", {}).get("collateral_metrics", [])) != sorted(key for key, value in protocol["metrics"].items() if value["role"] == "collateral")
        or configuration.get("dataset", {}).get("provider") != protocol_plugins.get("dataset", {}).get("name")
        or configuration.get("intervention", {}).get("plugin") != protocol_plugins.get("intervention", {}).get("name")
        or configuration.get("intervention", {}).get("parameters", {}).get("component_ids") != "all"
        or configuration.get("output", {}).get("retain_prompts") is not False
        or configuration.get("output", {}).get("retain_raw_outputs") is not False
        or configuration.get("output", {}).get("prompt_storage") != "hash_only"
        or compatibility.get("resolved_topology") != topology
        or compatibility.get("model") != provenance
        or compatibility.get("discovered_plugins") != manifest.get("plugins")
        or compatibility.get("evaluation") != {"plugin": protocol_plugins.get("evaluation", {}).get("name"), "version": protocol_plugins.get("evaluation", {}).get("version")}
        or compatibility.get("intervention") != {"plugin": protocol_plugins.get("intervention", {}).get("name"), "version": protocol_plugins.get("intervention", {}).get("version")}
        or not topology_complete
        or provenance.get("plugin") != model["plugin"]
        or provenance.get("plugin_version") != model["version"]
        or provenance.get("model_source") != model["source"]
        or provenance.get("model_revision") != model["revision"]
        or provenance.get("api_version") != protocol_plugins.get("model", {}).get("api_version")
        or set(provenance) != {"plugin", "plugin_version", "api_version", "architecture", "model_source", "model_revision", "model_class", "implementation_metadata", "libraries"}
        or provenance.get("architecture") != topology.get("architecture")
        or not isinstance(provenance.get("model_class"), str)
        or not provenance.get("model_class")
        or not isinstance(provenance.get("libraries"), Mapping)
        or not provenance.get("libraries")
        or provenance.get("implementation_metadata", {}).get("architecture_profile") != model.get("architecture_profile")
        or provenance.get("implementation_metadata", {}).get("architecture_profile_sha256") != model.get("architecture_profile_sha256")
        or manifest.get("config_sha256") != hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()
        or manifest.get("compatibility_sha256") != hashlib.sha256(canonical_json_bytes(compatibility)).hexdigest()
        or manifest.get("model_revision") != model["revision"]
        or manifest.get("dataset_sha256") != expected_dataset.get("manifest", {}).get("sha256")
        or manifest.get("prompt_sha256") != expected_dataset.get("prompt_templates", {}).get("sha256")
        or manifest.get("scorer_sha256") != expected_dataset.get("scorers", {}).get("sha256")
        or manifest.get("runtime") != runtime
        or manifest.get("execution_state") != "complete"
        or actual_plugin_identity != expected_plugin_identity
        or authorization.get("status") != "approved"
        or authorization.get("conformance_authorized") is not True
        or authorization.get("full_scan_authorized") is not True
        or authorization.get("protocol_sha256") != manifest.get("protocol_sha256")
        or not re.fullmatch(r"[0-9a-f]{40}", str(authorization.get("approved_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(authorization.get("approved_source_sha256", "")))
        or summarized_failures != immutable_failures
    ):
        raise InvalidEvidenceError("Phase 5 configuration or provenance evidence disagrees")

    prior_segments = trace.get("prior_segments", [])
    segments = [*prior_segments, {key: value for key, value in trace.items() if key != "prior_segments"}]
    spans = [span for segment in segments for span in segment.get("spans", [])]
    points = [
        point for segment in segments for metric in segment.get("metrics", [])
        for point in metric.get("points", [])
    ]
    if (
        trace.get("schema_version") not in {"capability-anatomy/phase5-trace/v1", "capability-anatomy/phase5-trace/v2"}
        or not isinstance(trace.get("trace_id"), str)
        or not re.fullmatch(r"[0-9a-f]{32}", trace["trace_id"])
        or not trace["trace_id"].strip("0")
        or report.get("trace_id") != trace["trace_id"]
        or not isinstance(prior_segments, list)
        or not isinstance(spans, list)
        or not spans
        or not points
    ):
        raise InvalidEvidenceError("Phase 5 trace evidence is incomplete")
    trace_ids = []
    for index, segment in enumerate(segments):
        # Historic v1 segments inherited the enclosing trace schema. V2 never
        # grants this omission: its per-segment completeness fields are required.
        if "schema_version" not in segment and trace.get("schema_version") == "capability-anatomy/phase5-trace/v1":
            segment = {"schema_version": "capability-anatomy/phase5-trace/v1", **segment}
        _validate_trace_segment_graph(segment, latest=index == len(segments) - 1)
        trace_ids.append(segment["trace_id"])
    if len(trace_ids) != len(set(trace_ids)):
        raise InvalidEvidenceError("Phase 5 trace graph is invalid")
    required_decisions = {
        ("full_scan_authorized", "gate5a_authorization", "authorize_phase5_operation", "accepted", "capability_anatomy.gate5a.authorization"),
        ("plugin_contract_accepted", "plugin_discovery", "resolve_plugin", "accepted", "capability_anatomy.plugin.resolve"),
        ("resource_budget_available", "execution", "enforce_resource_budget", "accepted", "capability_anatomy.execution.resource_budget"),
        ("task_committed", "execution", "execute_task", "accepted", "capability_anatomy.execution.task"),
        ("runtime_plugin_execution_complete", "runtime", "execute", "accepted", "capability_anatomy.runtime.execute"),
        ("scoped_bypass_active", "intervention", "block_bypass", "accepted", "capability_anatomy.intervention.block_bypass"),
        ("hooks_removed", "intervention", "block_bypass", "completed", "capability_anatomy.intervention.block_bypass"),
        ("identity_no_op_active", "intervention", "identity_no_op", "accepted", "capability_anatomy.intervention.identity_no_op"),
        ("identity_no_op_hooks_removed", "intervention", "identity_no_op", "completed", "capability_anatomy.intervention.identity_no_op"),
        ("result_independent_campaign_plan_persisted", "execution", "freeze_discovery_plan", "accepted", "capability_anatomy.phase5.freeze_discovery_plan"),
        ("repeated_baseline_no_op_and_matched_controls_complete", "execution", "evaluate_controls", "accepted", "capability_anatomy.phase5.evaluate_controls"),
        ("ranking_derived_from_discovery_only", "execution", "freeze_discovery_ranking", "accepted", "capability_anatomy.phase5.freeze_discovery_ranking"),
        ("discovery_ranking_frozen_before_validation", "execution", "open_validation_once", "accepted", "capability_anatomy.phase5.open_validation_once"),
        ("required_artifacts_atomically_written_and_digest_bound", "execution", "finalize_evidence", "accepted", "capability_anatomy.phase5.finalize_evidence"),
        ("task_execution_complete" if trace.get("schema_version") == "capability-anatomy/phase5-trace/v2" else "scan_evidence_validated", "phase5_scan", "phase5_full_scan", "completed", "capability_anatomy.phase5.scan"),
    }
    observed_decisions = set()
    for span in spans:
        for event in span.get("events", []):
            attributes = event.get("attributes", {})
            if event.get("name") == "operation.decision":
                observed_decisions.add((
                    attributes.get("capability_anatomy.reason"), attributes.get("capability_anatomy.component"),
                    attributes.get("capability_anatomy.operation"), attributes.get("capability_anatomy.outcome"), span.get("name"),
                ))
    decision_metrics = [
        metric for segment in segments for metric in segment.get("metrics", [])
        if metric.get("name") == "capability_anatomy.operation.decisions"
    ]
    observed_points = {
        (
            point.get("attributes", {}).get("capability_anatomy.reason"),
            point.get("attributes", {}).get("capability_anatomy.component"),
            point.get("attributes", {}).get("capability_anatomy.operation"),
            point.get("attributes", {}).get("capability_anatomy.outcome"),
        )
        for metric in decision_metrics for point in metric.get("points", []) if point.get("value", 0) > 0
    }
    required_points = {decision[:4] for decision in required_decisions}
    event_counts: dict[tuple[Any, ...], int] = {}
    for decision in observed_decisions:
        event_counts[decision[:4]] = sum(
            1 for span in spans for event in span.get("events", [])
            if event.get("name") == "operation.decision" and (
                event.get("attributes", {}).get("capability_anatomy.reason"),
                event.get("attributes", {}).get("capability_anatomy.component"),
                event.get("attributes", {}).get("capability_anatomy.operation"),
                event.get("attributes", {}).get("capability_anatomy.outcome"),
            ) == decision[:4]
        )
    point_counts: dict[tuple[Any, ...], float] = {}
    for metric in decision_metrics:
        for point in metric.get("points", []):
            attributes = point.get("attributes", {})
            key = tuple(attributes.get(f"capability_anatomy.{name}") for name in ("reason", "component", "operation", "outcome"))
            point_counts[key] = point_counts.get(key, 0) + point.get("value", 0)
    # The execution snapshot explicitly omits its still-open ownership parent.
    # Its one acquire decision is already in SDK counters; all other operation
    # metric points must correspond exactly to retained finished-span events.
    complete_event_counts = dict(event_counts)
    acquisition = ("run_ownership_acquired", "storage", "ownership", "accepted")
    open_owners = sum(bool(segment.get("external_parent_span_ids")) for segment in segments)
    if open_owners:
        complete_event_counts[acquisition] = complete_event_counts.get(acquisition, 0) + open_owners
    if point_counts != complete_event_counts:
        raise InvalidEvidenceError("Phase 5 trace metrics do not match retained decision events")
    task_count = len({task.id for task in build_campaign_plan(
        [item["id"] for item in topology["components"]], protocol["controls"],
        seed=int.from_bytes(hashlib.sha256(f"{protocol['experiment']['seed']}:controls".encode()).digest()[:8], "big"),
    )}) + len(report.get("validation", {}).get("results", [])) + 1
    exact_counts = {
        ("task_committed", "execution", "execute_task", "accepted"): task_count,
    }
    observation_count = sum(bool(line) for line in reader.text("observations.jsonl").splitlines())
    runtime_minimum = observation_count // runtime["repetitions"] * (runtime["warmup_runs"] + runtime["repetitions"])
    bypass_minimum = len(topology["components"]) + protocol["controls"]["matched_random_components"] + len(report.get("validation", {}).get("results", []))
    minimum_counts = {
        ("plugin_contract_accepted", "plugin_discovery", "resolve_plugin", "accepted"): len(protocol_plugins),
        ("resource_budget_available", "execution", "enforce_resource_budget", "accepted"): task_count * 2,
        ("runtime_plugin_execution_complete", "runtime", "execute", "accepted"): runtime_minimum,
        ("scoped_bypass_active", "intervention", "block_bypass", "accepted"): bypass_minimum,
        ("hooks_removed", "intervention", "block_bypass", "completed"): bypass_minimum,
        ("identity_no_op_active", "intervention", "identity_no_op", "accepted"): 1,
        ("identity_no_op_hooks_removed", "intervention", "identity_no_op", "completed"): 1,
    }
    if (
        not required_decisions <= observed_decisions
        or not required_points <= observed_points
        or any(point_counts.get(key) != count or event_counts.get(key) != count for key, count in exact_counts.items())
        or any(point_counts.get(key, 0) < count or event_counts.get(key, 0) < count for key, count in minimum_counts.items())
        or not _bypass_hooks_balanced(segments)
        or event_counts.get(("identity_no_op_active", "intervention", "identity_no_op", "accepted")) != event_counts.get(("identity_no_op_hooks_removed", "intervention", "identity_no_op", "completed"))
        or any(point_counts.get(key) != count for key, count in complete_event_counts.items())
    ):
        raise InvalidEvidenceError("Phase 5 trace decisions or counters are incomplete")




def _encode_artifact(relative: str, value: Any) -> bytes:
    if relative.endswith(".md"):
        if not isinstance(value, str):
            raise InvalidEvidenceError("Phase 5 Markdown artifact must contain text")
        return value.encode("utf-8")
    if relative.endswith(".jsonl"):
        if not isinstance(value, (list, tuple)):
            raise InvalidEvidenceError("Phase 5 JSONL artifact must contain rows")
        validate_tree(value, max_bytes=artifact_limit(relative))
        return b"".join(encode_json(row, max_bytes=artifact_limit(relative)) + b"\n" for row in value)
    return encode_json(value, max_bytes=artifact_limit(relative))


def _validate_artifact_payload(relative: str, payload: bytes) -> None:
    if relative.endswith(".md"):
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise InvalidEvidenceError(f"Phase 5 artifact is not UTF-8: {relative}") from error
        return
    try:
        values = (
            parse_json_lines(payload, max_bytes=artifact_limit(relative))
            if relative.endswith(".jsonl")
            else [parse_json(payload, max_bytes=artifact_limit(relative))]
        )
    except json.JSONDecodeError as error:
        raise InvalidEvidenceError(f"Phase 5 artifact is not valid JSON: {relative}") from error
    for value in values:
        _reject_sensitive_fields(value)


def _reject_sensitive_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_EVIDENCE_KEYS.intersection(value)
        if forbidden:
            raise InvalidEvidenceError(f"Phase 5 evidence contains forbidden fields: {sorted(forbidden)}")
        for child in value.values():
            _reject_sensitive_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_fields(child)
