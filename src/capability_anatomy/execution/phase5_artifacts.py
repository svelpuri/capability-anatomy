"""Resource-bounded Phase 5 evidence reads; no earlier path check grants trust."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..errors import InvalidEvidenceError
from . import secure_fs
from .evidence_limits import MAX_ARTIFACT_BYTES, MAX_BUNDLE_BYTES, MAX_ARTIFACTS, artifact_limit
from .evidence_json import parse_json
from .evidence_inventory import inventory_paths

MAX_ARTIFACT_DEPTH = 16


class Phase5ArtifactReader:
    def __init__(self, root: Path, manifest: Mapping[str, Any] | None = None):
        self.root = root
        self.entries = None
        if manifest is not None:
            entries = manifest.get("artifacts")
            if not isinstance(entries, list) or len(entries) > MAX_ARTIFACTS:
                raise InvalidEvidenceError("Phase 5 artifact inventory is invalid or exceeds its limit")
            self.entries = {}
            total = 0
            for item in entries:
                if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                    raise InvalidEvidenceError("Phase 5 artifact entry is invalid")
                name, size = item["path"], item.get("bytes")
                if name in self.entries or type(size) is not int or not 0 <= size <= artifact_limit(name):
                    raise InvalidEvidenceError("Phase 5 artifact size or identity is invalid")
                digest = item.get("sha256")
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise InvalidEvidenceError("Phase 5 artifact digest is invalid")
                self.entries[name] = item
                total += size
                if total > MAX_BUNDLE_BYTES:
                    raise InvalidEvidenceError("Phase 5 artifact inventory exceeds the bundle byte limit")

    def read(self, relative: str) -> bytes:
        path = _safe_artifact_path(self.root, relative)
        entry = None if self.entries is None else self.entries.get(relative)
        if self.entries is not None and entry is None:
            raise InvalidEvidenceError("Phase 5 artifact is not bound by the manifest")
        limit = artifact_limit(relative) if entry is None else entry["bytes"]
        payload = secure_fs.read_bytes(path, max_bytes=limit)
        if entry is not None and (len(payload) != limit or hashlib.sha256(payload).hexdigest() != entry["sha256"]):
            raise InvalidEvidenceError("Phase 5 evidence artifact digest mismatch")
        return payload

    def json(self, relative: str):
        return parse_json(self.read(relative), max_bytes=artifact_limit(relative))

    def text(self, relative: str) -> str:
        return self.read(relative).decode("utf-8")


def artifact_paths(root: Path):
    """Count descriptor directory entries before retaining their names."""
    for relative in inventory_paths(root, max_entries=MAX_ARTIFACTS, max_depth=MAX_ARTIFACT_DEPTH,
                                    error=lambda reason: InvalidEvidenceError("Phase 5 " + reason)):
        yield root / relative


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts or relative.endswith("/"):
        raise InvalidEvidenceError("Phase 5 artifact path is unsafe")
    root = secure_fs._absolute(root)
    path = root.joinpath(*candidate.parts)
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
        raise InvalidEvidenceError("Phase 5 artifact path contains a symlink")
    return path

