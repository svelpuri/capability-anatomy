"""Bounded, JSON-compatible authored documents with redacted diagnostics."""
from __future__ import annotations

import json
from contextlib import contextmanager
import math
import os
import stat
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import yaml
from opentelemetry.trace import Status, StatusCode

from .errors import InvalidConfigurationError
from .telemetry import OperationTelemetry

MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_DOCUMENT_DEPTH = 64
MAX_DOCUMENT_NODES = 100_000


class AuthoredInputError(InvalidConfigurationError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        detail = " (not valid JSON or YAML)" if reason == "document_syntax_invalid" else ""
        super().__init__(f"authored input refused: {reason}{detail}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise AuthoredInputError("object_keys_must_be_strings")
        if key in result:
            raise AuthoredInputError("duplicate_object_key")
        result[key] = value
    return result


class _Loader(yaml.SafeLoader):
    pass


def _mapping(loader: _Loader, node: yaml.MappingNode) -> dict[str, Any]:
    # YAML merge tags are refused: implicit overrides are ambiguous.
    return _pairs([(loader.construct_object(k), loader.construct_object(v)) for k, v in node.value])


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _check_tree(value: Any, *, max_depth: int, max_nodes: int) -> None:
    active: set[int] = set()
    pending = [(value, 0, False)]
    count = 0
    while pending:
        item, depth, leaving = pending.pop()
        if leaving:
            active.remove(id(item))
            continue
        count += 1
        if count > max_nodes:
            raise AuthoredInputError("document_node_limit")
        if depth > max_depth:
            raise AuthoredInputError("document_depth_limit")
        if isinstance(item, (dict, list)):
            if id(item) in active:
                raise AuthoredInputError("document_cycle")
            active.add(id(item))
            pending.append((item, depth, True))
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise AuthoredInputError("object_keys_must_be_strings")
                children = item.values()
            else:
                children = item
            pending.extend((child, depth + 1, False) for child in children)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise AuthoredInputError("document_requires_json_values")
        elif isinstance(item, float) and not math.isfinite(item):
            raise AuthoredInputError("document_requires_finite_numbers")


def _preflight(text: str, *, max_depth: int, max_nodes: int) -> None:
    # Streaming events stop excess nesting before recursive construction.
    depth = count = 0
    anchors: list[str | None] = []
    for event in yaml.parse(text, Loader=yaml.SafeLoader):
        count += 1
        if count > max_nodes * 3 + 4:
            raise AuthoredInputError("document_node_limit")
        if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
            anchors.append(event.anchor)
            depth += 1
            if depth > max_depth:
                raise AuthoredInputError("document_depth_limit")
        elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
            anchors.pop()
            depth -= 1
        elif isinstance(event, yaml.AliasEvent) and event.anchor in anchors:
            raise AuthoredInputError("document_cycle")


@contextmanager
def open_regular_binary(path: Path) -> Iterator[BinaryIO]:
    """Nonblocking, no-follow regular-file stream; callers bound their reads."""
    signals = OperationTelemetry.create("authored_input")
    with signals.tracer.start_as_current_span(
        "capability_anatomy.authored_input.read", record_exception=False, set_status_on_exception=False,
    ) as span:
        try:
            flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
            descriptor = os.open(Path(path), flags)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise AuthoredInputError("document_requires_regular_file")
                with os.fdopen(descriptor, "rb", closefd=False) as source:
                    yield source
            finally:
                os.close(descriptor)
        except (OSError, AuthoredInputError) as error:
            reason = error.reason if isinstance(error, AuthoredInputError) else "document_unreadable"
            signals.record(span, operation="read_document", outcome="refused", reason=reason)
            span.set_status(Status(StatusCode.ERROR, reason))
            raise AuthoredInputError(reason) from None
        else:
            signals.record(span, operation="read_document", outcome="accepted", reason="regular_file_read")


def read_regular_bytes(path: Path, *, max_bytes: int = MAX_DOCUMENT_BYTES) -> bytes:
    with open_regular_binary(path) as source:
        data = source.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise AuthoredInputError("document_byte_limit")
        return data


def load_mapping(
    path: Path, *, format: str | None = None,
    max_bytes: int = MAX_DOCUMENT_BYTES, max_depth: int = MAX_DOCUMENT_DEPTH,
    max_nodes: int = MAX_DOCUMENT_NODES, telemetry: OperationTelemetry | None = None,
) -> dict[str, Any]:
    """Read a bounded JSON/YAML object; errors never include input contents."""
    signals = telemetry or OperationTelemetry.create("authored_input")
    with signals.tracer.start_as_current_span(
        "capability_anatomy.authored_input.load", record_exception=False, set_status_on_exception=False,
    ) as span:
        try:
            data = read_regular_bytes(path, max_bytes=max_bytes)
            text = data.decode("utf-8")
            selected = format or ("json" if Path(path).suffix.lower() == ".json" else "yaml")
            if selected == "json":
                value = json.loads(text, object_pairs_hook=_pairs)
            elif selected == "yaml":
                _preflight(text, max_depth=max_depth, max_nodes=max_nodes)
                value = yaml.load(text, Loader=_Loader)
            else:
                raise AuthoredInputError("document_format_unsupported")
            _check_tree(value, max_depth=max_depth, max_nodes=max_nodes)
            if not isinstance(value, dict):
                raise AuthoredInputError("document_root_must_be_object")
        except (OSError, UnicodeError):
            error = AuthoredInputError("document_unreadable")
        except AuthoredInputError as caught:
            error = caught
        except (ValueError, yaml.YAMLError, RecursionError):
            error = AuthoredInputError("document_syntax_invalid")
        else:
            signals.record(span, operation="load_document", outcome="accepted", reason="bounded_object_valid")
            return value
        signals.record(span, operation="load_document", outcome="refused", reason=error.reason)
        span.set_status(Status(StatusCode.ERROR, error.reason))
        raise error from None
