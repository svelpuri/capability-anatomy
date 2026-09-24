from __future__ import annotations

from functools import wraps
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from ..errors import InvalidEvidenceError
from . import secure_fs
from .evidence_json import parse_json, encode_json
from .evidence_limits import MAX_ARTIFACT_BYTES


def _encode(value: Any) -> bytes:
    return encode_json(value, max_bytes=MAX_ARTIFACT_BYTES)


def _parse(payload: bytes) -> Any:
    return parse_json(payload, max_bytes=MAX_ARTIFACT_BYTES)


_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _owned(method):
    @wraps(method)
    def operation(self, *args, **kwargs):
        with secure_fs.ensure_run_ownership(self.root):
            return method(self, *args, **kwargs)
    return operation


class FrozenRunStore:
    def __init__(self, root: Path, config: Mapping[str, Any], compatibility: Mapping[str, Any]) -> None:
        self.root = root
        self._config_payload = _encode(config)
        self._compatibility_payload = _encode(compatibility)
        self.config_sha256 = hashlib.sha256(self._config_payload).hexdigest()
        self.compatibility_sha256 = hashlib.sha256(self._compatibility_payload).hexdigest()

    @_owned
    def initialize(self) -> None:
        secure_fs.ensure_directory(self.root)
        manifest = {
            "schema_version": "capability-anatomy/run-state/v1",
            "config_sha256": self.config_sha256,
            "compatibility_sha256": self.compatibility_sha256,
        }
        self._create_once(self.root / "experiment-config.json", self._config_payload)
        self._create_once(self.root / "compatibility.json", self._compatibility_payload)
        path = self.root / "run-state.json"
        if secure_fs.exists(path):
            try:
                existing = _parse(secure_fs.read_bytes(path, max_bytes=MAX_ARTIFACT_BYTES))
            except OSError as error:
                raise InvalidEvidenceError("run state is unreadable") from error
            if existing != manifest:
                raise InvalidEvidenceError("run configuration or compatibility identity changed")
            return
        self._atomic_write(path, _encode(manifest))

    _create_once = staticmethod(secure_fs.create_once)

    def completed(self, task_id: str) -> Mapping[str, Any] | None:
        data_path, marker_path = self._task_paths(task_id)
        if not secure_fs.exists(data_path) or not secure_fs.exists(marker_path):
            return None
        try:
            payload = secure_fs.read_bytes(data_path, max_bytes=MAX_ARTIFACT_BYTES)
            marker = _parse(secure_fs.read_bytes(marker_path, max_bytes=MAX_ARTIFACT_BYTES))
            value = _parse(payload)
        except OSError:
            return None
        expected = {
            "task_id": task_id,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "config_sha256": self.config_sha256,
            "compatibility_sha256": self.compatibility_sha256,
        }
        return value if marker == expected else None

    @_owned
    def commit(self, task_id: str, payload: Mapping[str, Any]) -> None:
        data_path, marker_path = self._task_paths(task_id)
        secure_fs.ensure_directory(data_path.parent)
        encoded = _encode(payload)
        marker = {
            "task_id": task_id,
            "payload_sha256": hashlib.sha256(encoded).hexdigest(),
            "config_sha256": self.config_sha256,
            "compatibility_sha256": self.compatibility_sha256,
        }
        self._atomic_write(data_path, encoded)
        self._atomic_write(marker_path, _encode(marker))

    def failure_attempts(self, task_id: str) -> int:
        value = self.failure_payload(task_id)
        if value is None:
            return 0
        attempt = value.get("_attempt")
        if not isinstance(attempt, int) or attempt < 1:
            raise InvalidEvidenceError("task failure evidence is invalid")
        return attempt

    def failure_payload(self, task_id: str) -> Mapping[str, Any] | None:
        self._task_paths(task_id)
        path = self.root / "failures" / f"{task_id}.json"
        if not secure_fs.exists(path):
            return None
        try:
            value = _parse(secure_fs.read_bytes(path, max_bytes=MAX_ARTIFACT_BYTES))
        except OSError as error:
            raise InvalidEvidenceError("task failure evidence is unreadable") from error
        if not isinstance(value, Mapping):
            raise InvalidEvidenceError("task failure evidence is invalid")
        return value

    @_owned
    def commit_failure(
        self, task_id: str, payload: Mapping[str, Any], *, attempt: int = 1,
    ) -> None:
        self._task_paths(task_id)
        path = self.root / "failures" / f"{task_id}.json"
        encoded = _encode({**payload, "_attempt": attempt})
        attempt_path = self.root / "failures" / f"{task_id}.attempt-{attempt:04d}.json"
        secure_fs.ensure_directory(attempt_path.parent)
        self._create_once(attempt_path, encoded)
        self._atomic_write(path, encoded)

    @_owned
    def freeze_sequence(self, name: str, proposed: tuple[str, ...]) -> tuple[str, ...]:
        if not _TASK_ID.fullmatch(name) or not proposed or len(set(proposed)) != len(proposed):
            raise InvalidEvidenceError("frozen sequence is invalid")
        path = self.root / f"{name}.json"
        if secure_fs.exists(path):
            try:
                existing = _parse(secure_fs.read_bytes(path, max_bytes=MAX_ARTIFACT_BYTES))
            except OSError as error:
                raise InvalidEvidenceError("frozen sequence is unreadable") from error
            if (
                not isinstance(existing, list)
                or any(not isinstance(item, str) for item in existing)
                or len(existing) != len(proposed)
                or set(existing) != set(proposed)
            ):
                raise InvalidEvidenceError("frozen sequence is incompatible")
            return tuple(existing)
        self._create_once(path, _encode(proposed))
        return proposed

    @_owned
    def freeze_payload(self, name: str, proposed: Mapping[str, Any] | list[Any]) -> Any:
        if not _TASK_ID.fullmatch(name):
            raise InvalidEvidenceError("frozen payload name is invalid")
        encoded = _encode(proposed)
        path = self.root / f"{name}.json"
        if secure_fs.exists(path):
            try:
                existing_bytes = secure_fs.read_bytes(path, max_bytes=MAX_ARTIFACT_BYTES)
                existing = _parse(existing_bytes)
            except OSError as error:
                raise InvalidEvidenceError("frozen payload is unreadable") from error
            if existing_bytes != encoded:
                raise InvalidEvidenceError("frozen payload is incompatible")
            return existing
        self._create_once(path, encoded)
        return proposed

    @_owned
    def record_state(self, state: str, reason: str) -> None:
        self._atomic_write(
            self.root / "execution-state.json",
            _encode({"state": state, "reason": reason}),
        )

    def active_wall_seconds(self) -> float:
        path = self.root / "budget-state.json"
        if not secure_fs.exists(path):
            return 0.0
        try:
            value = _parse(secure_fs.read_bytes(path, max_bytes=MAX_ARTIFACT_BYTES))
        except OSError as error:
            raise InvalidEvidenceError("budget state is unreadable") from error
        seconds = value.get("active_wall_seconds")
        if not isinstance(seconds, (int, float)) or seconds < 0:
            raise InvalidEvidenceError("budget state is invalid")
        return float(seconds)

    @_owned
    def add_active_wall_seconds(self, seconds: float) -> None:
        if seconds < 0:
            raise InvalidEvidenceError("active wall duration is invalid")
        self._atomic_write(
            self.root / "budget-state.json",
            _encode({"active_wall_seconds": self.active_wall_seconds() + seconds}),
        )

    def _task_paths(self, task_id: str) -> tuple[Path, Path]:
        if not _TASK_ID.fullmatch(task_id):
            raise InvalidEvidenceError("task ID is invalid")
        directory = self.root / "tasks"
        return directory / f"{task_id}.json", directory / f"{task_id}.complete.json"

    _atomic_write = staticmethod(secure_fs.atomic_write)
