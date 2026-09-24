"""Strict JSON with admission limits enforced before object-graph allocation."""
from __future__ import annotations

import io
import json
import math
from decimal import Decimal, DecimalException
from typing import Any, Mapping

import ijson

from ..errors import InvalidEvidenceError
from .evidence_limits import MAX_ARTIFACT_BYTES, MAX_JSON_DEPTH, json_node_limit

_REASONS = {
    'evidence_json_invalid': 'evidence contains invalid JSON',
    'evidence_json_byte_limit': 'evidence JSON exceeds its declared byte capacity',
    'evidence_json_node_limit': 'evidence JSON exceeds its declared node capacity',
    'evidence_json_depth_limit': 'evidence JSON exceeds the depth limit',
    'evidence_json_nonfinite': 'evidence JSON requires finite numbers',
    'evidence_json_duplicate_key': 'evidence JSON contains a duplicate object key',
    'evidence_json_object_required': 'evidence document must be a JSON object',
}


class EvidenceFormatError(InvalidEvidenceError):
    def __init__(self, reason: str):
        super().__init__(_REASONS[reason])
        self.reason = reason


def _finite(value: Any) -> None:
    if type(value) in (float, int, Decimal):
        try:
            finite = math.isfinite(value)
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise EvidenceFormatError('evidence_json_nonfinite')


def validate_tree(value: Any, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> None:
    """Apply the reader's node/depth rules with only O(depth) traversal state.

    Keys count as nodes in both directions. An iterator frame never queues all
    siblings, and cyclic containers terminate at the depth limit.
    """
    remaining = json_node_limit(max_bytes)
    pending = [iter((value,))]
    while pending:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        remaining -= 1
        if remaining < 0:
            raise EvidenceFormatError('evidence_json_node_limit')
        if len(pending) - 1 > MAX_JSON_DEPTH:
            raise EvidenceFormatError('evidence_json_depth_limit')
        if isinstance(item, Mapping):
            # Reserve keys now; no key list or pending-child list is allocated.
            remaining -= len(item)
            if remaining < 0:
                raise EvidenceFormatError('evidence_json_node_limit')
            if item and len(pending) > MAX_JSON_DEPTH:
                raise EvidenceFormatError('evidence_json_depth_limit')
            pending.append(iter(item.values()))
        elif isinstance(item, (list, tuple)):
            pending.append(iter(item))
        else:
            _finite(item)


def _preflight(payload: bytes, *, max_bytes: int, multiple_values: bool = False) -> None:
    """Consume bounded parser events before json.loads may allocate a tree."""
    remaining = json_node_limit(max_bytes) - int(multiple_values)
    depth = int(multiple_values)
    for event, value in ijson.basic_parse(
        io.BytesIO(payload), buf_size=16 * 1024, multiple_values=multiple_values,
    ):
        if event in ('end_map', 'end_array'):
            depth -= 1
            continue
        remaining -= 1
        if remaining < 0:
            raise EvidenceFormatError('evidence_json_node_limit')
        if depth > MAX_JSON_DEPTH:
            raise EvidenceFormatError('evidence_json_depth_limit')
        if event in ('start_map', 'start_array'):
            depth += 1
        elif event == 'number':
            _finite(value)


def _pairs(pairs):
    # This constructor is reached only after streaming node/depth admission.
    # Keep stdlib Unicode key semantics: YAJL normalizes lone surrogates.
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceFormatError('evidence_json_duplicate_key')
        result[key] = value
    return result


def parse_json(payload: bytes, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> Any:
    if len(payload) > max_bytes:
        raise EvidenceFormatError('evidence_json_byte_limit')
    try:
        _preflight(payload, max_bytes=max_bytes)
        return json.loads(payload, object_pairs_hook=_pairs)
    except (ijson.JSONError, UnicodeError, ValueError, RecursionError, DecimalException, OverflowError) as error:
        # Do not expose parser diagnostics: they include attacker-controlled
        # input fragments. Admission refusals carry fixed reason names above.
        raise EvidenceFormatError('evidence_json_invalid') from error


def parse_json_lines(payload: bytes, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> list[Any]:
    """Admit the whole JSONL artifact as one virtual array before allocation."""
    if len(payload) > max_bytes:
        raise EvidenceFormatError('evidence_json_byte_limit')
    if not payload.strip():
        return []
    try:
        _preflight(payload, max_bytes=max_bytes, multiple_values=True)
        # BytesIO iteration does not duplicate all lines as splitlines() does.
        # stdlib enforces exactly one complete JSON value on every nonempty line.
        return [json.loads(line, object_pairs_hook=_pairs) for line in io.BytesIO(payload) if line.strip()]
    except (ijson.JSONError, UnicodeError, ValueError, RecursionError, DecimalException, OverflowError) as error:
        raise EvidenceFormatError('evidence_json_invalid') from error


def encode_json(value: Any, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> bytes:
    """Apply the same admission contract before serializing or publishing."""
    from ..serialization import canonical_json_bytes

    validate_tree(value, max_bytes=max_bytes)
    payload = canonical_json_bytes(value)
    if len(payload) > max_bytes:
        raise EvidenceFormatError('evidence_json_byte_limit')
    return payload


def parse_mapping(payload: bytes, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> dict[str, Any]:
    value = parse_json(payload, max_bytes=max_bytes)
    if not isinstance(value, dict):
        raise EvidenceFormatError('evidence_json_object_required')
    return value
