from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import jsonschema

from ..authored_inputs import load_mapping, read_regular_bytes
from ..domain import DatasetMetadata, NormalizedRecord
from ..errors import InvalidConfigurationError
from ..protocols import CORE_API_VERSION, EvaluationSuite
from ..serialization import canonical_json_bytes


class Phase5DatasetProvider:
    name = "reference.phase5-records"
    version = "1"
    api_version = CORE_API_VERSION
    capabilities = frozenset({"dataset.frozen_roles", "dataset.strong_independence_groups"})

    def __init__(
        self,
        records: Mapping[str, tuple[NormalizedRecord, ...]],
        plan: Mapping,
        plan_sha256: str | None = None,
    ) -> None:
        self._records = dict(records)
        self._plan = plan
        self._validate_plan_membership()
        self.plan_sha256 = plan_sha256 or hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
        self._digest = hashlib.sha256(canonical_json_bytes({
            "records": [asdict(record) for partition in ("discovery", "validation") for record in self._records[partition]],
            "plan": plan,
        })).hexdigest()

    @classmethod
    def from_manifest(cls, path: Path) -> "Phase5DatasetProvider":
        try:
            value = load_mapping(path, format="json")
            plan_path = (path.parent / value["record_plan"]).resolve()
            plan = load_mapping(plan_path, format="json")
            records = {
                partition: tuple(NormalizedRecord(**record) for record in value["partitions"][partition])
                for partition in ("discovery", "validation")
            }
            return cls(records, plan, hashlib.sha256(read_regular_bytes(plan_path)).hexdigest())
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise InvalidConfigurationError("Phase 5 dataset manifest is invalid") from error

    def _validate_plan_membership(self) -> None:
        if set(self._records) != {"discovery", "validation"} or "final_holdout" in self._records:
            raise InvalidConfigurationError("Phase 5 dataset roles are invalid")
        seen_ids: set[str] = set()
        seen_groups: set[str] = set()
        for partition in ("discovery", "validation"):
            expected = {(row["source_id"], row["group_key"], row["kind"]) for row in self._plan["partitions"][partition]}
            actual = {(record.id, record.metadata.get("group_id"), record.metadata.get("kind")) for record in self._records[partition]}
            ids = {record.id for record in self._records[partition]}
            groups = {str(record.metadata.get("group_id")) for record in self._records[partition]}
            if (
                actual != expected
                or len(actual) != len(self._records[partition])
                or any(record.partition != partition for record in self._records[partition])
                or seen_ids & ids
                or seen_groups & groups
            ):
                raise InvalidConfigurationError("Phase 5 records do not match the frozen role plan")
            seen_ids.update(ids)
            seen_groups.update(groups)

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(self.name, "phase5-record-plan-v1", "mixed-upstream; see source manifest", self._digest, ("discovery", "validation"))

    def records(self, split: str) -> Iterable[NormalizedRecord]:
        if split not in self._records:
            raise InvalidConfigurationError("Phase 5 dataset role is unavailable")
        return iter(self._records[split])

    def stable_id(self, record: NormalizedRecord) -> str:
        return record.id

    def validate_for(self, suite: EvaluationSuite) -> None:
        schema = suite.required_record_schema()
        for records in self._records.values():
            for record in records:
                try:
                    jsonschema.validate(asdict(record), schema)
                except jsonschema.ValidationError as error:
                    raise InvalidConfigurationError("Phase 5 record is incompatible with evaluation suite") from error
