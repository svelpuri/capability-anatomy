from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .authored_inputs import AuthoredInputError, load_mapping, open_regular_binary
from .errors import InvalidConfigurationError
from .serialization import canonical_json_bytes, canonical_sha256
from .review_policy import PROTOCOL_V1, PROTOCOL_V2, validate_review_policy


PHASE5_PROTOCOL_VERSION = PROTOCOL_V2
PHASE5_RECORD_PLAN_VERSION = "capability-anatomy/phase5-record-plan/v1"
KINDS = ("simple", "abstention", "reasoning", "perplexity", "format")


def _require_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise InvalidConfigurationError(f"Phase 5 {label} fields are incomplete or unknown")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open_regular_binary(path) as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record_plan(source: Path, counts: Mapping[str, int]) -> dict[str, Any]:
    value = load_mapping(source, format="json")
    if set(counts) != set(KINDS) or any(not isinstance(count, int) or count <= 0 for count in counts.values()):
        raise InvalidConfigurationError("Phase 5 record counts are invalid")
    partitions: dict[str, list[dict[str, str]]] = {}
    for partition in ("discovery", "validation"):
        rows = []
        for kind in KINDS:
            selected = [
                row for row in value["rows"]
                if row.get("split") == partition and row.get("kind") == kind
            ][: counts[kind]]
            if len(selected) != counts[kind]:
                raise InvalidConfigurationError("Phase 5 record source is incomplete")
            rows.extend({key: row[key] for key in ("source_id", "group_key", "kind", "split")} for row in selected)
        partitions[partition] = rows
    all_rows = [row for rows in partitions.values() for row in rows]
    if len({row["source_id"] for row in all_rows}) != len(all_rows):
        raise InvalidConfigurationError("Phase 5 record IDs overlap")
    if len({row["group_key"] for row in all_rows}) != len(all_rows):
        raise InvalidConfigurationError("Phase 5 independence groups overlap")
    return {
        "schema_version": PHASE5_RECORD_PLAN_VERSION,
        "source_manifest_sha256": sha256_file(source),
        "counts_per_partition": dict(counts),
        "partitions": partitions,
        "rows_sha256": canonical_sha256(all_rows),
    }


def write_record_plan(source: Path, output: Path, counts: Mapping[str, int]) -> dict[str, Any]:
    plan = build_record_plan(source, counts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(plan) + b"\n")
    return plan


def validate_phase5_protocol(
    path: Path, *, validate_external_artifacts: bool = True,
) -> dict[str, Any]:
    value = load_mapping(path, format="json")
    required = {
        "schema_version", "status", "experiment", "model", "plugins", "runtime",
        "dataset", "metrics", "controls", "execution_semantics", "budgets", "candidate_rule",
        "stop_rules", "retention", "required_artifacts", "claim_boundary",
    }
    if value.get("schema_version") == PROTOCOL_V2:
        required.add("review_policy")
        validate_review_policy(value.get("review_policy"))
    _require_keys(value, required, "protocol")
    if value["schema_version"] not in {PROTOCOL_V1, PROTOCOL_V2} or value["status"] != "frozen_pre_inference":
        raise InvalidConfigurationError("Phase 5 protocol version or status is invalid")
    _require_keys(value["experiment"], {"id", "seed"}, "experiment")
    _require_keys(
        value["model"],
        {"plugin", "version", "source", "revision", "trust_remote_code", "architecture_profile", "architecture_profile_sha256"},
        "model",
    )
    if value["model"]["trust_remote_code"] is not False:
        raise InvalidConfigurationError("Phase 5 remote model code must be disabled")
    if value["model"]["revision"].casefold() in {"main", "master", "head", "latest"}:
        raise InvalidConfigurationError("Phase 5 model revision is mutable")
    if not re.fullmatch(r"[0-9a-f]{64}", value["model"]["architecture_profile_sha256"]):
        raise InvalidConfigurationError("Phase 5 architecture profile digest is invalid")
    runtime = value["runtime"]
    _require_keys(runtime, {
        "device", "dtype", "deterministic", "do_sample", "thinking", "batch_size",
        "context_length", "max_new_tokens", "warmup_runs", "repetitions",
        "synchronization", "memory_semantics", "randomized_record_order",
        "randomized_component_order", "record_seed_derivation", "component_seed_derivation",
    }, "runtime")
    if (
        not runtime["deterministic"] or runtime["do_sample"] or runtime["thinking"]
        or runtime["batch_size"] != 1 or runtime["repetitions"] < 1
        or runtime["warmup_runs"] < 0 or not runtime["randomized_record_order"]
        or not runtime["randomized_component_order"]
    ):
        raise InvalidConfigurationError("Phase 5 runtime controls are invalid")
    if (
        runtime["record_seed_derivation"] != "sha256(seed:records)"
        or runtime["component_seed_derivation"] != "sha256(seed:components)"
    ):
        raise InvalidConfigurationError("record and component seed domains are invalid")
    dataset = value["dataset"]
    _require_keys(dataset, {"manifest", "record_plan", "prompt_templates", "scorers", "final_holdout"}, "dataset")
    _require_keys(dataset["manifest"], {"path", "sha256"}, "dataset manifest")
    _require_keys(dataset["record_plan"], {"path", "sha256"}, "record plan")
    _require_keys(dataset["prompt_templates"], {"path", "sha256"}, "prompt templates")
    _require_keys(dataset["scorers"], {"path", "sha256"}, "scorers")
    if dataset["final_holdout"] != "forbidden":
        raise InvalidConfigurationError("Phase 5 final holdout must be forbidden")
    if validate_external_artifacts:
        root = path.parent
        manifest_path = (root / dataset["manifest"]["path"]).resolve()
        plan_path = (root / dataset["record_plan"]["path"]).resolve()
        prompt_path = (root / dataset["prompt_templates"]["path"]).resolve()
        scorer_path = (root / dataset["scorers"]["path"]).resolve()
        try:
            artifact_digests = {
                "manifest": sha256_file(manifest_path),
                "record_plan": sha256_file(plan_path),
                "prompt_templates": sha256_file(prompt_path),
                "scorers": sha256_file(scorer_path),
            }
        except (OSError, AuthoredInputError) as error:
            raise InvalidConfigurationError("Phase 5 protocol artifact is unavailable") from error
        for name, digest in artifact_digests.items():
            if digest != dataset[name]["sha256"]:
                label = (
                    "dataset manifest" if name == "manifest"
                    else "scorer" if name == "scorers"
                    else name.replace("_", " ")
                )
                raise InvalidConfigurationError(f"Phase 5 {label} digest changed")
        try:
            plan = load_mapping(plan_path, format="json")
            rows = [row for partition in ("discovery", "validation") for row in plan["partitions"][partition]]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise InvalidConfigurationError("Phase 5 record plan is invalid") from error
        if plan.get("schema_version") != PHASE5_RECORD_PLAN_VERSION:
            raise InvalidConfigurationError("Phase 5 record plan version is invalid")
        if len({row["source_id"] for row in rows}) != len(rows) or len({row["group_key"] for row in rows}) != len(rows):
            raise InvalidConfigurationError("Phase 5 record plan is not independent")
        if canonical_sha256(rows) != plan["rows_sha256"]:
            raise InvalidConfigurationError("Phase 5 record plan row identity changed")
    if set(value["metrics"]) != {
        "tool_selection", "argument_binding", "full_call", "abstention",
        "structured_reasoning", "instruction_format", "perplexity",
    }:
        raise InvalidConfigurationError("Phase 5 required metrics are incomplete")
    for metric_id, metric in value["metrics"].items():
        _require_keys(metric, {"role", "direction", "damage_formula"}, f"metric {metric_id}")
        expected_formula = "baseline_minus_condition" if metric["direction"] == "higher_is_better" else "condition_minus_baseline_over_baseline"
        if metric["damage_formula"] != expected_formula:
            raise InvalidConfigurationError("Phase 5 metric damage formula is inconsistent")
    plugin_roles = [plugin.get("role") for plugin in value["plugins"]]
    if set(plugin_roles) != {"model", "intervention", "evaluation", "dataset", "runtime"} or len(plugin_roles) != len(set(plugin_roles)):
        raise InvalidConfigurationError("Phase 5 plugin roles are incomplete or duplicated")
    for plugin in value["plugins"]:
        _require_keys(plugin, {"role", "name", "version", "api_version", "capabilities"}, "plugin")
        if not plugin["capabilities"]:
            raise InvalidConfigurationError("Phase 5 plugin capabilities are missing")
    _require_keys(value["controls"], {"no_op_hook", "repeated_baseline_positions", "matched_random_components", "placement", "random_seed_derivation"}, "controls")
    if set(value["controls"]["repeated_baseline_positions"]) != {"beginning", "middle", "end"}:
        raise InvalidConfigurationError("Phase 5 repeated baseline controls are incomplete")
    semantics = value["execution_semantics"]
    expected_semantics = {
        "no_op_hook": "identity_forward_hook_on_first_frozen_component",
        "baseline_reference": "mean_of_beginning_middle_end",
        "control_drift": "maximum_absolute_repeated_or_no_op_deviation",
        "matched_random": "seeded_single_component_duplicate_reproducibility",
        "task_retries": "retries_after_first_attempt",
        "wall_time": "cumulative_active_execution_across_resume",
        "memory": "synchronized_current_runtime_allocation_observation_not_peak",
        "validation": "freeze_discovery_ranking_then_open_validation_once",
    }
    if semantics != expected_semantics:
        raise InvalidConfigurationError("Phase 5 execution semantics are not frozen")
    _require_keys(value["budgets"], {"max_observation_errors", "max_wall_seconds", "max_memory_observation_bytes", "max_task_retries"}, "budgets")
    if any(value["budgets"][key] < 0 for key in value["budgets"]):
        raise InvalidConfigurationError("Phase 5 resource budget is invalid")
    _require_keys(value["candidate_rule"], {"target_absolute_damage_max", "collateral_absolute_damage_max", "perplexity_relative_damage_max", "minimum_drift_margin", "validation_once"}, "candidate rule")
    if value["retention"] != {"prompts": "sha256_only", "raw_outputs": "sha256_only"}:
        raise InvalidConfigurationError("Phase 5 retention policy is unsafe")
    if not value["stop_rules"] or not value["required_artifacts"]:
        raise InvalidConfigurationError("Phase 5 stop rules or artifacts are missing")
    required_artifacts = {
        "protocol.json", "record-plan.json", "configuration.json", "compatibility.json", "topology.json",
        "component-scan-order.json", "discovery-task-plan.json",
        "validation-task-plan.json", "observations.jsonl", "metrics.json",
        "failures.jsonl", "controls.json", "provenance.json", "candidates.json",
        "validation.json", "trace.json", "evidence-manifest.json", "report.json",
        "report.md",
    }
    if set(value["required_artifacts"]) != required_artifacts:
        raise InvalidConfigurationError("Phase 5 required artifacts are incomplete")
    return value
