"""Shared serialized evidence capacity; distinct from authored-input limits."""
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_ARTIFACTS = 10000
MAX_JSON_DEPTH = 64
# Count keys, scalar values and containers. This permits a full 50,000-span
# generated snapshot while bounding adversarial density independently of bytes.
JSON_BYTES_PER_NODE = 16


def json_node_limit(max_bytes: int) -> int:
    return max(1, max_bytes // JSON_BYTES_PER_NODE)


def is_trace(relative: str) -> bool:
    return relative == 'trace.json' or relative.endswith('/trace.json') or relative.startswith('task-traces/')


def artifact_limit(relative: str) -> int:
    return MAX_TRACE_BYTES if is_trace(relative) else MAX_ARTIFACT_BYTES
