"""Actual CLI result, exit status and receipt consistency across delivery failures."""
import errno
import json
import os
from pathlib import Path

import pytest

from capability_anatomy import cli as command
from capability_anatomy.execution import secure_fs
from test_release_review_telemetry import cli, environment, consumer, example, attributes


@pytest.mark.parametrize('shape', ['file', 'symlink'])
def test_doctor_output_preserves_actual_storage_refusal(tmp_path, shape):
    output = tmp_path / 'output'
    if shape == 'file': output.write_text('occupied')
    else: output.symlink_to(tmp_path / 'missing')
    result = cli('doctor', '--output', output, env=environment())
    assert result.returncode == 2
    check = next(item for item in json.loads(result.stdout)['checks'] if item['check'] == 'output_storage')
    assert check['reason'] in secure_fs._REASONS
    assert check['reason'] != 'output_storage_unavailable'
    assert check['exit_code'] == result.returncode


def test_doctor_unsupported_volume_matches_run_exit_and_records_cause(tmp_path, monkeypatch, capsys):
    # Exercise the actual publication probe, injecting the OS's unsupported
    # hard-link result at its consumer. Preserve supports_dir_fd identity.
    original = os.link
    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, 'FAKE-PRIVATE-OS-MESSAGE')
    monkeypatch.setattr(os, 'supports_dir_fd', (os.supports_dir_fd - {original}) | {unsupported})
    monkeypatch.setattr(os, 'link', unsupported)
    assert command.main(['doctor', '--output', str(tmp_path / 'unsupported')]) == 5
    output = capsys.readouterr()
    assert 'FAKE-PRIVATE' not in output.out + output.err
    check = next(item for item in json.loads(output.out)['checks'] if item['check'] == 'output_storage')
    assert check['reason'] == 'storage_filesystem_unsupported'
    assert check['exit_code'] == 5


@pytest.mark.parametrize('operation', ['doctor', 'verify', 'report'])
def test_optional_utility_delivery_does_not_change_success(operation, tmp_path):
    config = example(tmp_path)
    assert cli('run', '--config', config, env=environment()).returncode == 0
    args = [operation] if operation == 'doctor' else [operation, '--evidence', config.parent / 'run']
    if operation == 'report': args += ['--format', 'json']
    for status in (200, 400):
        with consumer(status=status) as (endpoint, spans, _):
            env = environment(endpoint); env['OTEL_EXPORTER_OTLP_TIMEOUT'] = '2'
            result = cli(*args, env=env)
            assert result.returncode == 0, result.stderr
            payload = json.loads(result.stdout)
            assert payload
            if status == 200:
                assert any(attributes(span).get('capability_anatomy.reason') == 'command_complete' for span in spans)
            else:
                diagnostic = json.loads(result.stderr.splitlines()[-1])
                assert diagnostic['warning'] == 'telemetry_delivery_incomplete'
                assert diagnostic['command'] == operation
                assert 'receipt' not in diagnostic['message']


def test_delivery_failure_preserves_run_summary_and_receipt(tmp_path):
    config = example(tmp_path)
    with consumer(status=400) as (endpoint, _, _):
        result = cli('run', '--config', config, env=environment(endpoint))
    assert result.returncode == 3, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    error = json.loads(result.stderr.splitlines()[-1])
    assert summary['status'] == 'complete'
    assert error['error'] == 'telemetry_delivery_incomplete'
    assert error['evidence_status'] == 'complete'
    assert error['trace_id'] == summary['trace_id']
    assert error['invocation_trace'] == summary['invocation_trace']
    receipt = json.loads(Path(error['invocation_trace']).read_text())
    assert receipt['trace_id'] == error['trace_id']
    assert receipt['transport']['spans_not_delivered'] > 0
    assert cli('verify', '--evidence', config.parent / 'run', env=environment()).returncode == 0


def test_receipt_write_failure_reports_completed_evidence_without_fake_locator(tmp_path):
    config = example(tmp_path)
    blocker = config.parent / 'run.invocations'
    blocker.write_text('preserve this regular file')
    result = cli('run', '--config', config, env=environment())
    assert result.returncode == 2, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    error = json.loads(result.stderr.splitlines()[-1])
    assert error['error'] == 'invocation_receipt_unwritable'
    assert summary['status'] == 'complete'
    for document in (summary, error):
        assert document['evidence_status'] == 'complete'
        assert document['receipt_status'] == 'unavailable'
        assert 'invocation_trace' not in document
        assert len(document['trace_id']) == 32
    assert blocker.read_text() == 'preserve this regular file'
    assert cli('verify', '--evidence', config.parent / 'run', env=environment()).returncode == 0
