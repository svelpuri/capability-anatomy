from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Iterable, Mapping
from pathlib import Path

import jsonschema

from ..authored_inputs import load_mapping
from ..domain import DatasetMetadata, NormalizedRecord
from ..errors import InvalidConfigurationError
from ..protocols import CORE_API_VERSION, EvaluationSuite


class SyntheticDatasetProvider:
    name = "reference.synthetic-dataset"
    version = "1"
    api_version = CORE_API_VERSION
    capabilities = frozenset({"dataset.normalized_records", "dataset.licensed_fixture"})

    def __init__(self, records: Mapping[str, tuple[NormalizedRecord, ...]], revision: str = "fixture-v1", license_name: str = "CC0-1.0") -> None:
        self._records = {partition: tuple(items) for partition, items in records.items()}
        payload = [asdict(record) for partition in sorted(self._records) for record in self._records[partition]]
        self._digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._revision = revision
        self._license = license_name

    @classmethod
    def from_manifest(cls, path: Path) -> "SyntheticDatasetProvider":
        try:
            value = load_mapping(path, format="json")
            records = {
                partition: tuple(NormalizedRecord(**record) for record in items)
                for partition, items in value["partitions"].items()
            }
            return cls(records, revision=value["revision"], license_name=value["license"])
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise InvalidConfigurationError("synthetic dataset manifest is invalid") from error

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            provider=self.name,
            revision=self._revision,
            license=self._license,
            sha256=self._digest,
            partitions=tuple(sorted(self._records)),
        )

    def records(self, split: str) -> Iterable[NormalizedRecord]:
        if split not in self._records:
            raise InvalidConfigurationError("dataset partition is unavailable")
        return iter(self._records[split])

    def stable_id(self, record: NormalizedRecord) -> str:
        if not record.id:
            raise InvalidConfigurationError("dataset record ID is required")
        return record.id

    def validate_for(self, suite: EvaluationSuite) -> None:
        schema = suite.required_record_schema()
        seen: set[str] = set()
        for partition, records in self._records.items():
            for record in records:
                if record.partition != partition:
                    raise InvalidConfigurationError("dataset record partition is inconsistent")
                if record.id in seen:
                    raise InvalidConfigurationError("dataset record IDs must be unique")
                seen.add(record.id)
                try:
                    jsonschema.validate(asdict(record), schema)
                except jsonschema.ValidationError as error:
                    raise InvalidConfigurationError("dataset record is incompatible with evaluation suite") from error
