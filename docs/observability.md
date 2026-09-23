# Observability

Every public run captures local OpenTelemetry spans and decision counters. The
invocation receipt at `<output>.invocations/<invocation-id>/trace.json` records
the invocation and delivery outcome; `trace.json` and `task-traces/` inside the
run are bound execution evidence. Preserve both the complete run directory and
its sibling receipts. See [operations](operations.md) for interruption, resume,
retention, ownership and limits.

## Read a run

```sh
capability-anatomy example --output demo
capability-anatomy run --config demo/experiment.json
capability-anatomy verify --evidence demo/run
capability-anatomy report --evidence demo/run --format json
```

Inspect stdout for the run directory, invocation receipt and trace ID. A successful
evidence check establishes integrity and reconstruction, not source authenticity
or model quality. Task checkpoints preserve execution provenance across resume;
task payloads alone are insufficient. A receipt can report delivery failure even
when the completed evidence still verifies.

## Collect traces and metrics

Use an operator-owned OTLP/HTTP collector with both pipelines enabled:

```sh
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
capability-anatomy run --config demo/experiment.json
```

The outer `capability_anatomy.invocation` trace covers ownership, execution and
publication. Child spans describe configuration, storage, runtime execution,
interventions, evaluation and tasks. Required decision attributes are
`capability_anatomy.operation`, `capability_anatomy.outcome` and
`capability_anatomy.reason`. `operation.decision` events include the component;
configuration uses `configuration.decision`. Count operations with
`capability_anatomy.operation.decisions` and configuration decisions with
`capability_anatomy.config.decisions`.

Query attributes and events rather than searching status-message text. Examples:

- `reason = observation_complete`: accepted record evaluations.
- `reason = task_committed`: durable task publication.
- `reason = compatible_task_complete`: skipped compatible work during resume.
- `reason = storage_temporary_reclaimed`: abandoned staging leaves removed under exclusive ownership.
- `reason = adapter_execution_failed` with error status: runtime failure.
- `reason = evidence_publication_complete`: final evidence binding succeeded.

See the distributed [fixed reason index](reasons.md) for source locations and
[operations](operations.md) for safe error messages and limits. Identities on
spans are hashes. Prompts, model responses, credential values, raw exception
messages and filesystem paths are not span identities.

Without remote configuration, evidence-producing runs still retain local traces.
Utility commands may opt out of SDK initialization with `OTEL_SDK_DISABLED=true`;
evidence-producing runs refuse that setting. Remote delivery has a bounded drain
after the output lease is released. A live process may therefore no longer own
the run directory. Locks are nonblocking and do not promise fairness between
readers and writers. Remote delivery outcome and local evidence outcome must be
inspected separately.

## Verify the exporter and installed wheel

In an environment with Capability Anatomy installed, supply an official collector
binary you have verified:

```sh
python scripts/verify_observability.py --collector /path/to/otelcol --work-dir /new/collector-check
```

The gate starts and stops its own loopback collector, runs the installed CLI,
checks success and refusal spans by their trace/span IDs, and requires explicit
decision attributes, events and counters. It also exercises source-export success
and refusal. The shared CI action runs this against each built wheel on Linux
and macOS; collector archives are pinned by version and SHA-256.

The bootstrap exporter works with stdlib alone. `--telemetry`, or configured OTLP
endpoints, requires an installed Capability Anatomy SDK and emits
`capability_anatomy.source_export` with `source_export_completed` or
`source_export_refused` plus a decision counter. JSON output includes the trace
ID and delivery status. The isolated PEP 517 backend does not import the runtime
being packaged; archive construction is covered by source manifests and the
release gate. Source export is a trusted maintainer operation with stable
caller-owned ancestors during handle acquisition, not a hostile-source sandbox.
