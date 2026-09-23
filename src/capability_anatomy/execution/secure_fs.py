"""POSIX evidence containment and process ownership, using OS enforcement."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
import errno
import re
import sys
from dataclasses import dataclass
from threading import Lock
from pathlib import Path
import stat
import uuid
from typing import Iterator

from opentelemetry.trace import Status, StatusCode

from ..errors import InvalidEvidenceError, UnsupportedPluginError
from ..telemetry import OperationTelemetry

try:
    import fcntl
except ImportError:  # pragma: no cover - explicitly unsupported hosts
    fcntl = None


class RunBusyError(InvalidEvidenceError):
    reason = "run_ownership_busy"


_REASONS = {
    "storage_path_missing": "evidence path does not exist; check the selected output directory",
    "storage_symlink_refused": "evidence path contains a symbolic link; use a real directory",
    "storage_not_directory": "evidence directory path contains a symbolic link or a non-directory component",
    "storage_not_regular": "evidence file is not a regular file",
    "storage_hardlink_refused": "evidence file has multiple hard links",
    "storage_permission_denied": "evidence path cannot be accessed with the current permissions",
    "storage_temporary_collision": "temporary evidence path already exists; preserve it and use a fresh directory",
    "storage_io_failed": "evidence filesystem operation failed; inspect available space and filesystem health",
    "storage_parent_traversal": "parent traversal is forbidden in evidence paths",
    "storage_outside_root": "evidence path is outside the owned run directory",
    "storage_worker_context_required": "worker storage requires the owning invocation context; propagate it with copy_context",
    "storage_fork_context_refused": "forked workers cannot reuse inherited evidence ownership",
    "storage_readonly_context": "a verification snapshot cannot write evidence",
    "storage_nested_ownership": "run ownership cannot be nested for different roots",
    "storage_recovery_required": "unfinished immutable publication remains; resume under exclusive ownership before verification",
    "storage_temporary_invalid": "unfinished evidence publication has unsafe links; preserve the directory for diagnosis",
    "storage_name_invalid": "evidence file name is invalid",
    "storage_artifact_byte_limit": "evidence artifact exceeds its byte limit",
    "storage_frozen_input_changed": "frozen run input changed; use a new output directory",
}


class StorageError(InvalidEvidenceError):
    def __init__(self, reason: str):
        super().__init__(_REASONS[reason])
        self.reason = reason


class StorageFilesystemUnsupported(UnsupportedPluginError):
    reason = "storage_filesystem_unsupported"

    def __init__(self):
        super().__init__("target filesystem cannot provide required locking, hard-link publication and directory synchronization")


def _os_error(error: OSError) -> StorageError:
    reason = {
        errno.ENOENT: "storage_path_missing", errno.ELOOP: "storage_symlink_refused",
        errno.ENOTDIR: "storage_not_directory", errno.EACCES: "storage_permission_denied",
        errno.EPERM: "storage_permission_denied", errno.EEXIST: "storage_temporary_collision",
    }.get(error.errno, "storage_io_failed")
    return StorageError(reason)


@dataclass(frozen=True)
class _Ownership:
    root: Path
    descriptor: int
    pid: int
    readonly: bool = False


_OWNER: ContextVar[_Ownership | None] = ContextVar("evidence_owner", default=None)
_SIGNALS: ContextVar[OperationTelemetry | None] = ContextVar("storage_signals", default=None)
# This registry does not select an ambient output root. It only refuses unsafe
# context loss while a host has active leases; explicit independent acquisitions
# remain supported, and copy_context carries one chosen invocation to a worker.
_ACTIVE_LOCK = Lock()
_ACTIVE: set[int] = set()


def _after_fork():
    global _ACTIVE_LOCK, _ACTIVE
    _ACTIVE_LOCK, _ACTIVE = Lock(), set()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _current_owner():
    owner = _OWNER.get()
    if owner is not None and owner.pid != os.getpid():
        raise StorageError("storage_fork_context_refused")
    return owner


def _check_platform() -> None:
    if (fcntl is None or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
            or not {os.open, os.mkdir, os.stat, os.unlink, os.rename, os.link} <= os.supports_dir_fd):
        raise UnsupportedPluginError("secure evidence storage requires POSIX dir_fd, O_NOFOLLOW and flock")


def _absolute(path: Path) -> Path:
    # Do not resolve symlinks or erase '..' before enforcing containment.
    value = Path(os.path.abspath(path)) if not path.is_absolute() else path
    if ".." in path.parts:
        raise StorageError("storage_parent_traversal")
    if sys.platform == "darwin":
        aliases = {"/tmp": "/private/tmp", "/var": "/private/var"}
        for alias, target in aliases.items():
            if value == Path(alias) or Path(alias) in value.parents:
                metadata = os.lstat(alias)
                if not stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != 0 or os.readlink(alias) not in (target, target.lstrip("/")):
                    raise StorageError("storage_symlink_refused")
                value = Path(target) / value.relative_to(alias)
                break
    return value


def _open_directory(path: Path, *, create: bool = False, explicit_root: bool = False) -> int:
    _check_platform()
    absolute = _absolute(path)
    owner = _current_owner()
    if owner is None:
        with _ACTIVE_LOCK:
            context_missing = bool(_ACTIVE) and not explicit_root
        if context_missing:
            raise StorageError("storage_worker_context_required")
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        parts = absolute.parts[1:]
    else:
        root, root_descriptor = owner.root, owner.descriptor
        try:
            parts = absolute.relative_to(root).parts
        except ValueError as error:
            raise StorageError("storage_outside_root") from error
        descriptor = os.dup(root_descriptor)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _operation(operation: str, *, success_reason: str = "storage_operation_complete") -> Iterator[None]:
    signals = _SIGNALS.get() or OperationTelemetry.create("storage")
    with signals.tracer.start_as_current_span(
        f"capability_anatomy.storage.{operation}", record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            owner = _current_owner()
            if owner is not None and owner.readonly and operation not in {"read", "exists"}:
                raise StorageError("storage_readonly_context")
            yield
        except (OSError, InvalidEvidenceError, UnsupportedPluginError) as error:
            refused = _os_error(error) if isinstance(error, OSError) else error
            reason = getattr(refused, "reason", "storage_path_refused")
            signals.record(span, operation=operation, outcome="refused", reason=reason)
            span.set_status(Status(StatusCode.ERROR, reason))
            if refused is not error:
                raise refused from error
            raise
        else:
            signals.record(span, operation=operation, outcome="completed", reason=success_reason)


@contextmanager
def _parent(path: Path, *, create: bool = False) -> Iterator[tuple[int, str]]:
    if path.name in ("", ".", ".."):
        raise StorageError("storage_name_invalid")
    descriptor = -1
    try:
        descriptor = _open_directory(path.parent, create=create)
        yield descriptor, path.name
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _regular(metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise StorageError("storage_symlink_refused")
    if not stat.S_ISREG(metadata.st_mode):
        raise StorageError("storage_not_regular")
    if metadata.st_nlink != 1:
        raise StorageError("storage_hardlink_refused")


def _read_at(parent: int, name: str, max_bytes: int | None = None) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        _regular(os.fstat(descriptor))
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
            if max_bytes is not None and len(payload) > max_bytes:
                raise StorageError("storage_artifact_byte_limit")
            return payload
    finally:
        os.close(descriptor)


def read_bytes(path: Path, *, max_bytes: int | None = None) -> bytes:
    with _operation("read"), _parent(path) as (parent, name):
        return _read_at(parent, name, max_bytes)


def exists(path: Path) -> bool:
    with _operation("exists"):
        try:
            with _parent(path) as (parent, name):
                _regular(os.stat(name, dir_fd=parent, follow_symlinks=False))
            return True
        except FileNotFoundError:
            return False


def ensure_directory(path: Path) -> None:
    with _operation("mkdir"):
        descriptor = _open_directory(path, create=True)
        os.close(descriptor)


def _write_at(parent: int, name: str, payload: bytes) -> None:
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(descriptor)
    except BaseException:
        # O_EXCL established ownership of this temporary leaf before any write.
        os.unlink(name, dir_fd=parent)
        raise
    finally:
        os.close(descriptor)


# Only this exact publisher namespace is disposable; arbitrary dotfiles remain
# evidence. Keep generation and recovery classification in the same module.
_PROBE_TEMPORARY = re.compile(r"(?P<base>\.capability-anatomy-probe-[0-9a-f]{32})(?P<link>\.link)?\Z")
_TEMPORARY = re.compile(r"\.(?P<target>[^/\\]+)\.[0-9a-f]{32}\.tmp\Z")


def temporary_target(name: str) -> str | None:
    probe = _PROBE_TEMPORARY.fullmatch(name)
    if probe:
        return probe.group("base") if probe.group("link") else name + ".link"
    match = _TEMPORARY.fullmatch(name)
    target = match.group("target") if match else None
    return target if target not in {None, "", ".", ".."} else None


def _temporary_name(name: str) -> str:
    return f".{name}.{uuid.uuid4().hex}.tmp"


def validate_temporary(parent: int, name: str) -> None:
    """An unpublished leaf, or create_once's published same-inode alias."""
    target = temporary_target(name)
    if target is None:
        raise StorageError("storage_temporary_invalid")
    item = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if item.st_nlink == 2 and stat.S_ISREG(item.st_mode):
        try:
            final = os.stat(target, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            raise StorageError("storage_temporary_invalid") from None
        if (stat.S_ISREG(final.st_mode) and final.st_nlink == 2 and
                (final.st_dev, final.st_ino) == (item.st_dev, item.st_ino)):
            return
        raise StorageError("storage_temporary_invalid")
    _regular(item)


def reclaim_temporary(parent: int, name: str) -> None:
    with _operation("recover_temporary", success_reason="storage_temporary_reclaimed"):
        owner = _current_owner()
        if owner is None or owner.readonly:
            raise StorageError("storage_readonly_context")
        validate_temporary(parent, name)
        # The directory handle and exclusive lease bind recovery to the run.
        # Unlink only the staging name; never promote partially written bytes.
        os.unlink(name, dir_fd=parent)
        os.fsync(parent)


def _recover_temporaries(output: Path) -> None:
    from .evidence_inventory import inventory_paths
    from .evidence_limits import MAX_ARTIFACTS
    from .core_evidence import CoreEvidenceError
    inventory_paths(output, max_entries=MAX_ARTIFACTS, max_depth=16,
                    error=CoreEvidenceError, reclaim_temporaries=True)


def create_once(path: Path, payload: bytes) -> None:
    with _operation("create_once"), _parent(path, create=True) as (parent, name):
        temporary = _temporary_name(name)
        temporary_owned = False
        try:
            _write_at(parent, temporary, payload)
            temporary_owned = True
            try:
                # Publish a fully flushed immutable input without overwriting.
                os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            except FileExistsError:
                if _read_at(parent, name) != payload:
                    raise StorageError("storage_frozen_input_changed")
        finally:
            if temporary_owned:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
        os.fsync(parent)


def atomic_write(path: Path, payload: bytes) -> None:
    with _operation("atomic_write"), _parent(path, create=True) as (parent, name):
        try:
            _regular(os.stat(name, dir_fd=parent, follow_symlinks=False))
        except FileNotFoundError:
            pass
        temporary = _temporary_name(name)
        temporary_owned = False
        try:
            _write_at(parent, temporary, payload)
            temporary_owned = True
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
        finally:
            if temporary_owned:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass


def _probe_volume(descriptor: int) -> None:
    """Exercise the actual target before configuration writes or model loading."""
    name = ".capability-anatomy-probe-" + uuid.uuid4().hex
    linked = name + ".link"
    owned = []
    with _operation("probe_volume"):
        try:
            _write_at(descriptor, name, b"storage capability probe")
            owned.append(name)
            os.link(name, linked, src_dir_fd=descriptor, dst_dir_fd=descriptor, follow_symlinks=False)
            owned.append(linked)
            os.fsync(descriptor)
        except OSError as error:
            raise StorageFilesystemUnsupported() from error
        finally:
            for temporary in reversed(owned):
                try:
                    os.unlink(temporary, dir_fd=descriptor)
                except FileNotFoundError:
                    pass


@contextmanager
def _ownership(output: Path, telemetry: OperationTelemetry | None, *, readonly: bool):
    if _current_owner() is not None:
        raise StorageError("storage_nested_ownership")
    signals = telemetry or OperationTelemetry.create("storage")
    descriptor = -1
    ownership_pid = os.getpid()
    token = signals_token = None
    acquired = False
    failure_reason = None
    with signals.tracer.start_as_current_span(
        "capability_anatomy.storage.ownership", record_exception=False, set_status_on_exception=False,
    ) as span:
        try:
            descriptor = _open_directory(output, create=not readonly, explicit_root=True)
            try:
                fcntl.flock(descriptor, (fcntl.LOCK_SH if readonly else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RunBusyError("output directory is owned by another run; retry after it finishes") from error
            except OSError as error:
                raise StorageFilesystemUnsupported() from error
            acquired = True
            token = _OWNER.set(_Ownership(_absolute(output), descriptor, ownership_pid, readonly))
            signals_token = _SIGNALS.set(signals)
            with _ACTIVE_LOCK:
                _ACTIVE.add(descriptor)
            signals.record(span, operation="ownership", outcome="accepted", reason="run_ownership_acquired")
            if not readonly:
                _recover_temporaries(output)
                _probe_volume(descriptor)
            yield
        except BaseException as error:
            refused = _os_error(error) if isinstance(error, OSError) else error
            reason = "run_ownership_busy" if isinstance(error, RunBusyError) else (
                "owned_run_failed" if acquired else getattr(refused, "reason", "storage_path_refused")
            )
            failure_reason = reason
            signals.record(span, operation="ownership", outcome="failed", reason=reason)
            span.set_status(Status(StatusCode.ERROR, reason))
            if refused is not error:
                raise refused from error
            raise
        finally:
            if signals_token is not None:
                _SIGNALS.reset(signals_token)
            if token is not None:
                _OWNER.reset(token)
            if descriptor >= 0:
                if acquired and os.getpid() == ownership_pid:
                    with _ACTIVE_LOCK:
                        _ACTIVE.discard(descriptor)
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            if acquired and os.getpid() == ownership_pid:
                if failure_reason is None:
                    signals.record(span, operation="ownership", outcome="completed", reason="run_ownership_released")
                else:
                    # Release is a lifecycle event, not successful run completion.
                    span.add_event("ownership.released", {"capability_anatomy.reason": "run_ownership_released"})


@contextmanager
def run_ownership(output: Path, telemetry: OperationTelemetry | None = None) -> Iterator[None]:
    """Hold exclusive ownership and verify target-volume primitives before work."""
    with _ownership(output, telemetry, readonly=False):
        yield


def ownership_active(output: Path) -> bool:
    owner = _current_owner()
    return owner is not None and owner.root == _absolute(output)


@contextmanager
def ensure_run_ownership(output: Path, telemetry: OperationTelemetry | None = None) -> Iterator[None]:
    if ownership_active(output):
        if _current_owner().readonly:
            raise StorageError("storage_readonly_context")
        yield
    else:
        with run_ownership(output, telemetry):
            yield


@contextmanager
def read_ownership(output: Path, telemetry: OperationTelemetry | None = None) -> Iterator[None]:
    """Pin a read-only coherent snapshot, or refuse a writer with a busy reason."""
    if ownership_active(output):
        yield
    else:
        with _ownership(output, telemetry, readonly=True):
            yield


def probe_storage_volume(output: Path) -> dict[str, str]:
    """Explicit target-volume diagnostic; creates a root only when requested."""
    with ensure_run_ownership(output):
        return {"directory_lock": "supported", "hardlink_publication": "supported", "directory_sync": "supported"}
