#!/usr/bin/env python3
"""Read actual stock CLI decisions back from an official OTel Collector.

Requires an installed capability-anatomy distribution and an otelcol binary.
Uses isolated loopback ports and owned temporary outputs; never changes SDK globals.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import time


def collector_spans(content):
    """Parse scoped fields from the pinned collector's detailed debug format."""
    spans = []
    for block in re.split(r"(?m)^Span #\d+\s*$", content)[1:]:
        def field(name):
            match = re.search(r"(?m)^    " + re.escape(name) + r"\s*:\s*(.*?)$", block)
            assert match, "collector span missing " + name
            return match.group(1)
        attributes = block.split("Attributes:\n", 1)[1].split("Events:", 1)[0] if "Attributes:\n" in block else ""
        attrs = dict(re.findall(r"(?m)^     -> ([^:]+): \w+\((.*)\)$", attributes))
        events = []
        for event in re.split(r"(?m)^SpanEvent #\d+\s*$", block)[1:]:
            name = re.search(r"(?m)^     -> Name: (.+)$", event)
            values = dict(re.findall(r"(?m)^          -> ([^:]+): \w+\((.*)\)$", event))
            if name:
                events.append((name.group(1), values))
        spans.append({"trace_id": field("Trace ID"), "span_id": field("ID"), "name": field("Name"),
                      "status": field("Status code"), "attributes": attrs, "events": events})
    return spans


def require_decision(spans, *, name, reason, outcome, trace_id=None, error=False):
    expected = {"capability_anatomy.reason": reason, "capability_anatomy.outcome": outcome}
    matches = [span for span in spans if span["name"] == name and
               (trace_id is None or span["trace_id"] == trace_id) and
               all(span["attributes"].get(key) == value for key, value in expected.items())]
    assert matches, "collector omitted decision attributes: " + reason
    for span in matches:
        assert int(span["trace_id"], 16) and int(span["span_id"], 16)
        assert any(name in {"operation.decision", "configuration.decision"} and
                   all(attrs.get(key) == value for key, value in expected.items())
                   for name, attrs in span["events"]), "collector omitted decision event: " + reason
        assert (span["status"] == "Error") == error, "collector decision status mismatch: " + reason
    return matches


def require_counter(content, *, component, operation, outcome, reason, value):
    expected = {"capability_anatomy.component": component, "capability_anatomy.operation": operation,
                "capability_anatomy.outcome": outcome, "capability_anatomy.reason": reason}
    for metric in re.split(r"(?m)^Metric #\d+\s*$", content)[1:]:
        if not re.search(r"(?m)^     -> Name: capability_anatomy\.operation\.decisions$", metric):
            continue
        for point in re.split(r"(?m)^NumberDataPoints #\d+\s*$", metric)[1:]:
            attrs = dict(re.findall(r"(?m)^     -> ([^:]+): \w+\((.*)\)$", point.split("StartTimestamp:", 1)[0]))
            measured = re.search(r"(?m)^Value: (\d+)$", point)
            if all(attrs.get(key) == expected[key] for key in expected) and measured and int(measured.group(1)) == value:
                return
    raise AssertionError("collector omitted exact decision counter: " + reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collector', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    collector = args.collector.resolve()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    endpoint = f'127.0.0.1:{port}'
    config = {'receivers': {'otlp': {'protocols': {'http': {'endpoint': endpoint}}}},
              'exporters': {'debug': {'verbosity': 'detailed'}},
              'service': {'pipelines': {signal: {'receivers': ['otlp'], 'exporters': ['debug']}
                                        for signal in ('traces', 'metrics')}}}
    config_path = work / 'collector.json'
    config_path.write_text(json.dumps(config))
    log_path = work / 'collector.log'
    env = {key: value for key, value in os.environ.items() if not key.startswith('OTEL_')}
    env.update(OTEL_EXPORTER_OTLP_ENDPOINT='http://' + endpoint,
               OTEL_EXPORTER_OTLP_PROTOCOL='http/protobuf', OTEL_EXPORTER_OTLP_TIMEOUT='3')
    results = {'collector_version': subprocess.check_output([str(collector), '--version'], text=True).strip(),
               'import_path': subprocess.check_output([sys.executable, '-c', 'import capability_anatomy; print(capability_anatomy.__file__)'], text=True).strip()}
    def cli(name, *arguments, expected):
        result = subprocess.run([sys.executable, '-m', 'capability_anatomy.cli', *map(str, arguments)],
                                env=env, capture_output=True, text=True, timeout=30)
        (work / (name + '.stdout')).write_text(result.stdout)
        (work / (name + '.stderr')).write_text(result.stderr)
        results[name] = {'exit': result.returncode}
        assert result.returncode == expected, (name, result.returncode, result.stderr)
        return result
    with log_path.open('w') as log:
        process = subprocess.Popen([str(collector), '--config', str(config_path)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while True:
                assert process.poll() is None, 'collector exited before readiness'
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.1):
                        break
                except OSError:
                    assert time.monotonic() < deadline, 'collector readiness timeout'
                    time.sleep(.05)
            created = cli('example', 'example', '--output', work / 'example', expected=0)
            experiment = Path(json.loads(created.stdout)['configuration'])
            completed = cli('run', 'run', '--config', experiment, expected=0)
            receipt = json.loads(Path(json.loads(completed.stdout)['invocation_trace']).read_text())
            assert receipt['status'] == 'complete' and receipt['transport']['status'] == 'delivered'
            cli('verify', 'verify', '--evidence', experiment.parent / 'run', expected=0)
            cli('missing', 'verify', '--evidence', work / 'missing', expected=2)
            outside = work / 'outside'
            outside.mkdir()
            link = work / 'unsafe'
            link.symlink_to(outside, target_is_directory=True)
            refused = cli('storage-refusal', 'example', '--output', link, expected=2)
            storage_reason = json.loads(refused.stderr)['error']
            assert storage_reason in {'storage_not_directory', 'storage_symlink_refused'}
            assert not list(outside.iterdir()), 'refused path received evidence'
            experiment.write_text('{}')
            cli('config-refusal', 'run', '--config', experiment, expected=2)
            export_script = Path(__file__).with_name('export_standalone.py')
            def export(name, destination, expected):
                result = subprocess.run([sys.executable, str(export_script), str(destination), '--telemetry'],
                                        env=env, capture_output=True, text=True, timeout=30)
                (work / (name + '.stdout')).write_text(result.stdout)
                (work / (name + '.stderr')).write_text(result.stderr)
                assert result.returncode == expected, (name, result.stderr)
                result_data = json.loads(result.stdout if expected == 0 else result.stderr)
                assert result_data['telemetry']['transport']['status'] == 'delivered'
                results[name] = result_data
                return result_data['telemetry']['trace_id']
            export_id = export('source-export', work / 'source-snapshot', 0)
            refused_id = export('source-export-refused', work / 'source-snapshot', 2)
            content = log_path.read_text()
            spans = collector_spans(content)
            require_decision(spans, name='capability_anatomy.evaluation.record', reason='observation_complete',
                             outcome='accepted', trace_id=receipt['trace_id'])
            require_decision(spans, name='capability_anatomy.cli.command', reason='command_failed', outcome='failed', error=True)
            require_decision(spans, name='capability_anatomy.config.validate', reason='schema_and_semantics_valid', outcome='accepted')
            require_decision(spans, name='capability_anatomy.config.validate', reason='schema_validation_failed', outcome='refused', error=True)
            require_decision(spans, name='capability_anatomy.storage.ownership', reason=storage_reason, outcome='failed', error=True)
            require_decision(spans, name='capability_anatomy.source_export', reason='source_export_completed',
                             outcome='completed', trace_id=export_id)
            require_decision(spans, name='capability_anatomy.source_export', reason='source_export_refused',
                             outcome='refused', trace_id=refused_id, error=True)
            for reason, outcome in [('source_export_completed', 'completed'), ('source_export_refused', 'refused')]:
                require_counter(content, component='source_export', operation='export_snapshot', outcome=outcome,
                                reason=reason, value=1)
            received = {(span['trace_id'], span['span_id']) for span in spans}
            assert all((receipt['trace_id'], span['span_id']) in received for span in receipt['spans']), 'collector lost local receipt spans'
            results['receipt_span_count'] = len(receipt['spans'])
            results['required_signals_received'] = True
            results['source_export_decisions_received'] = 2
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            results['collector_exit'] = process.returncode
    assert results['collector_exit'] == 0, results
    results['status'] = 'passed'
    (work / 'verification.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
