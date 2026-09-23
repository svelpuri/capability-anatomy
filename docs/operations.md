# Operations

`doctor` inspects the installation and built-in plugin discovery. `doctor --output ./volume-probe` also exercises locking, hard-link publication and directory synchronization on the selected filesystem. A normal run performs this volume probe before model loading. Use an isolated Python 3.12 environment on macOS or Linux. Windows filesystem ownership is not supported in this alpha.

## Evidence and resume

A run owns its output directory through final evidence publication. An active reader or writer causes a competing writer to refuse before model loading; nonblocking locks provide no starvation-freedom guarantee. Symlinked output paths and non-regular evidence files refuse. On macOS, the system-owned `/tmp` and `/var` aliases are recognized; arbitrary symlink ancestors still refuse. The owner explicitly unlocks on normal exit. The operating system releases its file descriptors on process exit. Forked workers may compute, but cannot reuse inherited evidence ownership; use a fresh process to acquire a new lease for evidence writes. A hard-killed owner with surviving forked workers can retain inherited descriptors until those workers exit; terminate the experiment-owned workers before resuming. Do not edit or rename an active run directory.

A generic completed run contains a `core-manifest.json`, task payloads and completion markers, the mandatory `task-traces/` checkpoints, frozen configuration and compatibility records, `trace.json`, and JSON/Markdown reports. `verify` and `report` acquire a shared nonblocking read lease and refuse while a writer owns the directory. They are read-only and do not load a model or execute a plugin selected by evidence. Checksums establish consistency, not authenticity.

On interruption, inspect `execution-state.json`, `failures/` and `trace-failures/`. Phase 5 committed tasks without durable completion traces refuse with `phase5_resume_trace_missing` before model loading; task checkpoints alone do not support that recovery in this alpha. Preserve that bundle and start with a fresh output directory. Generic runs support checkpoint-backed resume. Resume with exactly the same inputs and qualified environment. A refused attempt against a completed bundle preserves that bundle; its refusal is visible in configured OTLP telemetry. A hard kill between filesystem publications may leave a publisher temporary named `.<target>.<32 lowercase hex digits>.tmp`. These incomplete publications are never evidence: read-only verification excludes regular staging leaves without changing them, and the next exclusive run owner reclaims them before execution. Recovery never promotes staging bytes. The same rule covers abandoned `.capability-anatomy-probe-<32 lowercase hex digits>` files and their `.link` peers from the filesystem capability check. An interrupted immutable publication with two links is reclaimed only when its staging and final names refer to the same inode. Read-only inspection of an interrupted immutable publication whose final name still has two links returns `storage_recovery_required`; resume first to remove the staging alias under exclusive ownership. Symlinks, special files and unrelated hardlinks refuse and remain untouched. Unknown dotfiles are not automatically deleted. Keep every `task-traces/` checkpoint when archiving a generic bundle; task payloads alone do not preserve its execution evidence.

## Signals

Local `trace.json` records completed execution spans, parent IDs, start/end nanoseconds, event timestamps, reason names and counters. Experiment, task and record identities are hashed. A local invocation receipt is written after ownership release to the sibling `<output-name>.invocations/<invocation-id>/trace.json`. It includes ownership and final publication spans, including failures, and does not alter an already bound evidence bundle. Its scope ends before writing the receipt itself. The receipt records remote delivery counts separately from evidence publication; inspect its status and transport fields. When remote delivery succeeds, these captured spans are available at the collector. Per-task trace checkpoints bind completed observations to their original execution spans across resume. This avoids releasing the lock before final publication.

```sh
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
capability-anatomy run --config demo/experiment.json
```

Configure both traces and metrics pipelines in your collector. Look for `capability_anatomy.operation.decisions`, `run_ownership_acquired`, `run_ownership_released`, `task_committed`, `adapter_execution_failed`, `core_evidence_consistent`, and `storage_temporary_reclaimed` on `capability_anatomy.storage.recover_temporary`. Recovery emits one decision counter per discarded staging leaf, under the ownership span; refusal reasons describe unsafe links or storage failures. Local evidence remains enabled when remote collection is not configured. Remote spans use the OpenTelemetry batch processor. Draining occurs after ownership release. For `run`, failed exports or queue loss cause exit 3 with `telemetry_delivery_incomplete`, while local evidence and its stdout summary are retained. Utility commands preserve their command result and exit status and emit a structured telemetry warning; they have no invocation receipt. Set `OTEL_EXPORTER_OTLP_TIMEOUT` or the `TRACES`/`METRICS` variants in seconds (default 5). The drain has a deadline plus at most one in-flight HTTP call; it is not a hard process deadline. Only `http/protobuf` is supported; unsupported per-signal protocols refuse explicitly. `OTEL_SDK_DISABLED=true` disables SDK initialization for utility commands and refuses evidence-producing runs. Retained traces have a 50,000-span limit and fail explicitly if exceeded.

SIGINT and SIGTERM are cooperative. During post-lease delivery, interruption preserves the invocation receipt, reports evidence and delivery states separately, and returns exit 4. Python handles them at interpreter checkpoints; a native model operation may finish before interruption takes effect. Resource wall-time budgets are soft task-boundary checks, not hard process deadlines. Memory observations are measurements at declared boundaries, not a universal peak-memory guarantee. An enforced memory budget refuses when the runtime cannot measure it; CPU inference does not claim an available CUDA memory measurement.

## Credentials and plugins

Keep plaintext credentials out of configuration and datasets. Use top-level references:

```yaml
credentials:
  service:
    provider: env
    name: SERVICE_API_TOKEN
```

A selected trusted plugin must implement `configure_credentials(provider)` and use `provider.get("service").reveal()` only at its transport consumer. Reference names may be frozen; values are never included in snapshots. Resolved literal credentials are refused if a plugin returns them into evidence. Encoded or transformed secret values cannot be universally detected. Plugins execute with your process privileges and are not sandboxed.

Model loading uses safetensors and a qualified eager attention implementation. Pickle-only checkpoints, remote code and custom attention selectors refuse. Use an immutable model revision and retain the model's own license and provenance.

## Preparing a release snapshot

Use a private, caller-owned build workspace containing reviewed source. Keep source and destination ancestors stable during the exporter's initial directory-handle acquisition; do not run cleanup or renaming concurrently. After handles are acquired, inventory, reads and writes stay bound to those directories. An observed nonempty new destination is refused, but an empty-directory replacement before handle acquisition cannot be distinguished reliably with the supported POSIX creation API. See the [security boundary](../SECURITY.md). Source export is a trusted maintainer operation, not hostile-package isolation.


Generic synthetic runs use the built-in evidence retention behavior; they do not accept a configurable policy input. Phase 5 policy files belong to the advanced governed workflow.

## Commands, bounds and diagnostics

| Command | Purpose |
| --- | --- |
| `doctor [--output PATH]` | Inspect installation; optionally probe the actual storage volume. |
| `example --output PATH` | Create an offline synthetic configuration and dataset. |
| `run --config PATH` | Execute or resume the frozen experiment. |
| `verify --evidence PATH` | Check recorded integrity and reconstruct aggregates without loading a model. |
| `report --evidence PATH [--format json]` | Verify, then display the recorded report. |
| `freeze-phase5-records --source PATH --output PATH` | Freeze Phase 5 record identities from a source document. |
| `validate-phase5 --protocol PATH` | Validate a frozen protocol and its referenced artifacts. |
| `conform-phase5 --config PATH` | Execute bounded conformance under matching source approval. |

Exit codes are 0 success, 2 invalid input/evidence or refusal, 3 runtime or telemetry delivery failure, 4 interruption, and 5 unsupported capability/dependency/filesystem. Code 1 is unused. Errors contain a fixed `error` identifier and a safe actionable `message`; exception content and secret-bearing paths are not printed. Typical codes include `storage_frozen_input_changed`, `run_ownership_busy`, `evidence_digest_mismatch`, `evidence_aggregate_mismatch` and `gate_source_identity_changed`.

Authored JSON/YAML is limited to 8 MiB, 100,000 nodes and depth 64. Generated evidence uses 16 MiB per ordinary artifact, 64 MiB per trace and 256 MiB per bundle; inventories stop at 10,000 entries. Evidence JSON is also limited to depth 64 and one node per 16 bytes of declared artifact capacity: 1,048,576 nodes for ordinary artifacts and 4,194,304 for traces. Keys, values and containers each count; JSONL rows share a single artifact budget. Streaming parser admission applies before object construction, and publication uses the same limits. `evidence_json_node_limit` and `evidence_json_depth_limit` mean the evidence is too structurally dense or deep; split the experiment rather than retrying unchanged input. An overflowing span capture retains a bounded diagnostic prefix and terminal failure, explicitly marked incomplete. Split larger experiments rather than retrying an unchanged oversized plan. Export files are bounded at 8 MiB each and 64 MiB total, with 10,000 selected files overall, directory depth 32 and 10,000 enumerated entries per selected tree. One incoming file can temporarily coexist with the retained budget during admission.

Credential field detection normalizes camel case, underscores and hyphens and examines provider-prefixed authentication-material families, including token, secret, key, passphrase, authorization, auth, signature and assertion. OAuth `code` is reserved as a complete name; it is not a suffix rule, so `trust_remote_code` remains an ordinary model setting. Names for ordinary file references should end in `_file` or `_path`, such as `full_scan_authorization_file`. Recognized JWT and AWS credential forms are refused in authored evidence; this remains a defense for trusted plugins, not a secret scanner for hostile code.

Run `python scripts/verify_observability.py --collector /path/to/otelcol --work-dir /new/probe-directory` from an installed environment to read actual success, configuration refusal and storage refusal signals back from an official collector. The standalone CI executes this gate against the built wheel on Linux and macOS, alongside the install/uninstall/reinstall and source-suite gates.

The Phase 5 commands in the table require optional user-authored governed inputs; no approved pretrained-model campaign is shipped. See [release preparation](release.md). The [observability runbook](observability.md) and [fixed reason index](reasons.md) ship in source distributions.

Current v2 trace evidence requires complete capture with zero dropped spans. Historical v1 bundles can report an unknown completeness guarantee; legacy compatibility is not a new completeness certification.
