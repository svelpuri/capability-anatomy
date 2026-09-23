#!/usr/bin/env python3
"""Export a reviewed source snapshot into a new directory, without Git history.

Use caller-controlled source and destination ancestors that remain stable while
initial directory handles are acquired. This bootstrap tool does not isolate
against a hostile process with the same filesystem authority during acquisition.
After acquisition, pinned roots prevent ancestor replacement redirecting copies.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from fnmatch import fnmatchcase
import hashlib
import json
import os
import stat
from pathlib import Path
import sys


REQUIRED_FILES = (
    "README.md", "pyproject.toml", "uv.lock",
    "configs/examples/synthetic-scan.yaml", "fixtures/synthetic/manifest.json",
    "scripts/export_standalone.py", "scripts/verify_release.py", "scripts/verify_observability.py", ".github/workflows/ci.yml",
    "schemas/experiment-config.v1.schema.json", "schemas/evidence-manifest.v1.schema.json",
    "schemas/metric-result.v1.schema.json", "schemas/transformation-recipe.v1.schema.json",
)
OPTIONAL_FILES = (
    "LICENSE", "NOTICE", "SECURITY.md", "CONTRIBUTING.md", "CHANGELOG.md",
    "docs/operations.md", "docs/research-limits.md", "docs/open-source-alpha.md",
    "docs/release.md", "docs/installation.md", "docs/observability.md", "docs/reasons.md",
    "scripts/release_backend.py", "scripts/ci_acceptance.py", "scripts/update_reason_catalog.py", ".github/actions/acceptance/action.yml",
)
PUBLIC_TESTS = (
    "test_capability_anatomy_phase1.py", "test_capability_anatomy_phase2.py",
    "test_capability_anatomy_phase3.py", "test_capability_anatomy_phase4.py",
    "test_capability_anatomy_phase5_public.py", "test_capability_anatomy_phase5_campaign.py",
    "test_capability_anatomy_gate5m.py",
)
TREE_RULES = {
    "src/capability_anatomy": {".py", ".json"},
    "fixtures/gate5m_external_plugin": {".py", ".txt", ""},
}
FORBIDDEN_NAMES = frozenset({".git", ".env", "AGENTS.md", "CLAUDE.md", "CONTEXT.md", ".session-sync.md", "__pycache__"})


MAX_SOURCE_FILE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_FILES = 10000
MAX_TOTAL_SOURCE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_DEPTH = 32


class ExportError(ValueError):
    pass


def _safe_file(source: Path, relative: str) -> Path:
    path = source / relative
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ExportError("export path must remain inside the selected source")
    cursor = path
    while cursor != source:
        if cursor.name in FORBIDDEN_NAMES or cursor.is_symlink():
            raise ExportError(f"symbolic link or forbidden export input: {relative}")
        cursor = cursor.parent
    if not path.is_file():
        raise ExportError(f"required export input is not a regular file: {relative}")
    return path


@contextmanager
def _root_descriptor(path: Path):
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")) or os.open not in os.supports_dir_fd:
        raise ExportError("export requires POSIX directory-descriptor filesystem support")
    absolute = Path(os.path.abspath(path))
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _parent_descriptor(root: int, relative: str, *, create: bool = False):
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(part in {"..", "."} for part in parts):
        raise ExportError("export path must remain relative to its root")
    descriptor = os.dup(root)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor, parts[-1]
    finally:
        os.close(descriptor)


def _read_file(source: Path, relative: str, root: int) -> bytes:
    _safe_file(source, relative)
    with _parent_descriptor(root, relative) as (parent, name):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ExportError("export source must be a regular file without hard links")
            if metadata.st_size > MAX_SOURCE_FILE_BYTES:
                raise ExportError("export source exceeds the per-file byte limit")
            payload = stream.read(MAX_SOURCE_FILE_BYTES + 1)
            if len(payload) > MAX_SOURCE_FILE_BYTES:
                raise ExportError("export source exceeds the per-file byte limit")
            return payload


def _write_file(destination: int, relative: str, payload: bytes) -> None:
    with _parent_descriptor(destination, relative, create=True) as (parent, name):
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)


def _listed_files(root: int, relative: str, *, recursive: bool) -> list[str]:
    """Enumerate names from the pinned source, without following directories."""
    with _parent_descriptor(root, relative) as (parent, name):
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    paths = []

    visited = 0
    def walk(directory, prefix, depth=0):
        nonlocal visited
        if depth > MAX_SOURCE_DEPTH:
            raise ExportError("export source exceeds the directory depth limit")
        names = []
        with os.scandir(directory) as entries:
            for entry in entries:
                visited += 1
                if visited > MAX_SOURCE_FILES:
                    raise ExportError("export source exceeds the entry limit")
                names.append(entry.name)
        for name in sorted(names):
            if name == "__pycache__":
                continue
            if name in FORBIDDEN_NAMES:
                raise ExportError("forbidden export input name")
            metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
            path = prefix + "/" + name
            if stat.S_ISLNK(metadata.st_mode):
                raise ExportError(f"symbolic link export input: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                if recursive:
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                    try:
                        walk(child, path, depth + 1)
                    finally:
                        os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                paths.append(path)
            else:
                raise ExportError("export source contains a non-regular input")

    try:
        walk(descriptor, relative)
    finally:
        os.close(descriptor)
    return paths


def _source_exists(root: int, relative: str) -> bool:
    try:
        with _parent_descriptor(root, relative) as (parent, name):
            metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise ExportError(f"export input is not a regular file: {relative}")
    except FileNotFoundError:
        return False
    return True


def selected_files(source: Path, root: int, test_files: list[str]) -> dict[str, str]:
    """Destination-to-source projection. No Git index, history, or broad tree copy."""
    selected = {name: name for name in REQUIRED_FILES}
    selected.update({name: name for name in OPTIONAL_FILES if _source_exists(root, name)})
    selected.update({"tests/" + name: "tests/" + name for name in PUBLIC_TESTS})
    for relative in test_files:
        if fnmatchcase(Path(relative).name, "test_release_*.py"):
            selected[relative] = relative
    for directory, suffixes in TREE_RULES.items():
        for relative in _listed_files(root, directory, recursive=True):
            if Path(relative).suffix in suffixes:
                selected[relative] = relative
    for relative in selected.values():
        _safe_file(source, relative)
    return dict(sorted(selected.items()))


def export_snapshot(source: Path, destination: Path, *, allow_unlicensed_preview: bool = False) -> dict:
    if source.is_symlink():
        raise ExportError("symbolic link source root is unsupported")
    source = Path(os.path.abspath(source))
    if not source.is_dir():
        raise ExportError("source must be a directory")
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ExportError("destination already exists; choose a new directory")
    if destination == source or source in destination.resolve().parents:
        raise ExportError("destination must be outside the source tree")
    with _root_descriptor(source) as source_root:
        test_files = _listed_files(source_root, "tests", recursive=False)
        projection = selected_files(source, source_root, test_files)
        licensed = "LICENSE" in projection
        if not licensed and not allow_unlicensed_preview:
            raise ExportError("LICENSE is missing; an unlicensed review requires --allow-unlicensed-preview")
        # Read from one pinned source root, even if an ancestor is renamed.
        if len(projection) > MAX_SOURCE_FILES:
            raise ExportError("export source exceeds the selected-file limit")
        payloads = {}
        total_bytes = 0
        for name, relative in projection.items():
            payload = _read_file(source, relative, source_root)
            total_bytes += len(payload)
            if total_bytes > MAX_TOTAL_SOURCE_BYTES:
                raise ExportError("export source exceeds the total byte limit")
            payloads[name] = payload
    entries = [{"path": name, "source_path": projection[name], "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()} for name, payload in payloads.items()]
    exported_tests = {Path(name).name for name in payloads if name.startswith("tests/")}
    excluded_tests = sorted(Path(path).name for path in test_files
                            if fnmatchcase(Path(path).name, "test_*.py") and Path(path).name not in exported_tests)
    manifest = {
        "schema_version": "capability-anatomy/source-export/v1",
        "license_present": licensed,
        "release_status": "requires_release_review" if licensed else "unlicensed_review_preview_only",
        "source_tree_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "files": entries,
        "tests": {"included_files": sorted(exported_tests), "excluded_private_files": excluded_tests,
                  "boundary": "Generic package contracts and release regressions are included. Historical prototype, data materialization, private inventory and full-model evidence tests remain in the private research suite; no runtime pytest exclusions are used."},
    }
    # Exclusive mkdir guarantees an existing destination is never overwritten.
    # The manifest is written last; an I/O failure leaves an incomplete new
    # directory for inspection and never triggers recursive deletion.
    with _root_descriptor(destination.parent) as parent:
        try:
            os.mkdir(destination.name, mode=0o700, dir_fd=parent)
        except FileExistsError as error:
            raise ExportError("destination already exists; choose a new directory") from error
        destination_root = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            # Refuse observable replacement contents. Emptiness is not proof of
            # creation identity: mkdir cannot atomically return a directory fd.
            if os.listdir(destination_root):
                raise ExportError("new export destination is unexpectedly nonempty")
            for name, payload in payloads.items():
                _write_file(destination_root, name, payload)
            _write_file(destination_root, "EXPORT-MANIFEST.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        finally:
            os.close(destination_root)
    return manifest


@contextmanager
def _export_telemetry(enabled: bool):
    """Optional installed SDK; isolated build backends remain stdlib-only."""
    state = {"enabled": enabled, "trace_id": None, "transport": {"configured": False}}
    if not enabled:
        yield state
        return
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.trace import Status, StatusCode
    from capability_anatomy.run_telemetry import configure_otlp
    from capability_anatomy.telemetry import OperationTelemetry
    provider = TracerProvider()
    meter = None
    transport = None
    try:
        readers = []
        transport = configure_otlp(provider, readers)
        meter = MeterProvider(metric_readers=readers)
        signals = OperationTelemetry.create("source_export", provider.get_tracer("capability_anatomy.source_export"),
                                            meter.get_meter("capability_anatomy.source_export"))
        with signals.tracer.start_as_current_span("capability_anatomy.source_export", record_exception=False,
                                                 set_status_on_exception=False) as span:
            state["trace_id"] = format(span.get_span_context().trace_id, "032x")
            try:
                yield state
            except BaseException:
                signals.record(span, operation="export_snapshot", outcome="refused", reason="source_export_refused")
                span.set_status(Status(StatusCode.ERROR, "source_export_refused"))
                raise
            else:
                span.set_attribute("capability_anatomy.source_tree_sha256", state["source_tree_sha256"])
                span.set_attribute("capability_anatomy.exported_files", state["files"])
                signals.record(span, operation="export_snapshot", outcome="completed", reason="source_export_completed")
    finally:
        try:
            if transport is not None:
                state["transport"] = transport.flush()
        finally:
            provider.shutdown()
            if meter is not None:
                meter.shutdown(timeout_millis=10000)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allow-unlicensed-preview", action="store_true")
    parser.add_argument("--telemetry", action="store_true", help="use the installed Capability Anatomy SDK (required when OTLP is configured)")
    args = parser.parse_args(argv)
    enabled = args.telemetry or any(os.environ.get(key) for key in (
        "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"))
    state = None
    try:
        with _export_telemetry(enabled) as state:
            manifest = export_snapshot(args.source, args.destination, allow_unlicensed_preview=args.allow_unlicensed_preview)
            state.update(source_tree_sha256=manifest["source_tree_sha256"], files=len(manifest["files"]))
    except (ExportError, OSError, RecursionError) as error:
        print(json.dumps({"error": type(error).__name__, "reason": "source_export_refused",
                          "message": "source export refused; inspect selected regular files, license and new destination",
                          "telemetry": state}), file=sys.stderr)
        return 2
    except Exception:
        print(json.dumps({"error": "source_export_failed", "message": "repair the installed SDK or export environment",
                          "telemetry": state}), file=sys.stderr)
        return 3
    result = {"directory": str(args.destination), "source_tree_sha256": manifest["source_tree_sha256"],
              "files": len(manifest["files"]), "release_status": manifest["release_status"], "telemetry": state}
    print(json.dumps(result, sort_keys=True))
    return 3 if state["transport"].get("status") == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
