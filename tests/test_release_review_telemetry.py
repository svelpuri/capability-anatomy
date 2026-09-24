"""Real public CLI and OTLP HTTP consumer regressions for PR59 review."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


def cli(*args, env=None, timeout=30):
    return subprocess.run([sys.executable, '-m', 'capability_anatomy.cli', *map(str,args)],
                          capture_output=True,text=True,env=env,timeout=timeout)


def environment(endpoint=None):
    env={key:value for key,value in os.environ.items() if not key.startswith('OTEL_')}
    if endpoint:
        env.update(OTEL_EXPORTER_OTLP_ENDPOINT=endpoint, OTEL_EXPORTER_OTLP_TIMEOUT='0.3')
    return env


@contextmanager
def consumer(status=200, delay=0):
    spans=[]
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
            requests.append(self.path)
            if delay: time.sleep(delay)
            response_status = status(self.path, raw) if callable(status) else status
            if self.path.endswith('/traces') and response_status == 200:
                request=ExportTraceServiceRequest.FromString(raw)
                spans.extend(span for resource in request.resource_spans for scope in resource.scope_spans for span in scope.spans)
            self.send_response(response_status); self.end_headers()
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try: yield f'http://127.0.0.1:{server.server_port}',spans,requests
    finally: server.shutdown();server.server_close();thread.join(timeout=2)


def example(tmp_path, count=2):
    result=cli('example','--output',tmp_path/'example',env=environment())
    assert result.returncode==0,result.stderr
    config=Path(json.loads(result.stdout)['configuration'])
    dataset=config.parent/'dataset.json';doc=json.loads(dataset.read_text()); row=doc['partitions']['discovery'][0]
    doc['partitions']['discovery']=[{**row,'id':f'record-{index}'} for index in range(count)]
    dataset.write_text(json.dumps(doc));return config


def attributes(span):
    return {item.key:item.value.string_value for item in span.attributes}


def test_actual_slow_otlp_batches_all_spans_and_correlates_refused_config(tmp_path):
    config=example(tmp_path)
    with consumer(delay=.03) as (endpoint,spans,requests):
        # This positive control measures batching and exact delivery, not a
        # sub-second scheduler deadline on shared CI. The unavailable-consumer
        # test separately keeps the 0.3s timeout and bounded-failure assertion.
        env=environment(endpoint)
        env['OTEL_EXPORTER_OTLP_TIMEOUT']='2'
        start=time.monotonic();result=cli('run','--config',config,env=env);elapsed=time.monotonic()-start
        assert result.returncode==0,result.stderr
        receipt=json.loads(Path(json.loads(result.stdout)['invocation_trace']).read_text())
        assert receipt['trace_id'] is not None
        assert receipt['transport']['status']=='delivered'
        assert receipt['transport']['spans_not_delivered']==0
        actual={span.span_id.hex() for span in spans}
        assert {span['span_id'] for span in receipt['spans']} <= actual
        assert len({span.trace_id.hex() for span in spans})==1
        assert len([p for p in requests if p.endswith('/traces')]) < len(spans)/4
        assert elapsed<8
        config.write_text('{}')
        spans.clear()
        refused=cli('run','--config',config,env=env)
        assert refused.returncode==2
        assert any(attributes(s).get('capability_anatomy.reason')=='invocation_failed' for s in spans)
        assert any(s.name=='capability_anatomy.config.validate' or s.name.startswith('capability_anatomy.config') for s in spans)


def test_unavailable_otlp_is_bounded_and_explicit_with_local_evidence(tmp_path):
    config=example(tmp_path)
    with consumer(status=503) as (endpoint,spans,requests):
        start=time.monotonic();result=cli('run','--config',config,env=environment(endpoint));elapsed=time.monotonic()-start
        assert result.returncode==3,result.stdout+result.stderr
        assert 'telemetry_delivery_incomplete' in result.stderr
        assert elapsed<8
        assert len(requests)<10
        receipts=list(config.parent.glob('run.invocations/*/trace.json'));assert len(receipts)==1
        receipt=json.loads(receipts[0].read_text());assert receipt['status']=='failed'
        assert receipt['transport']['spans_not_delivered']>0
        root=next(span for span in receipt['spans'] if span['name']=='capability_anatomy.invocation')
        assert root['status']=='ERROR'
        assert cli('verify','--evidence',config.parent/'run',env=environment()).returncode==0


def test_cli_failure_signal_reaches_actual_consumer(tmp_path):
    with consumer() as (endpoint,spans,_):
        result=cli('verify','--evidence',tmp_path/'missing',env=environment(endpoint))
        assert result.returncode==2
        assert any(attributes(s).get('capability_anatomy.reason')=='command_failed' for s in spans)
        assert any(s.status.code==2 for s in spans)


def test_disabled_sdk_and_per_signal_protocol_refuse_before_compute(tmp_path):
    config=example(tmp_path)
    env=environment();env['OTEL_SDK_DISABLED']='true'
    result=cli('run','--config',config,env=env)
    assert result.returncode==2 and 'telemetry_disabled' in result.stderr
    assert not (config.parent/'run').exists()
    with consumer() as (endpoint,spans,_):
        env=environment(endpoint);env['OTEL_EXPORTER_OTLP_TRACES_PROTOCOL']='grpc'
        result=cli('run','--config',config,env=env)
        assert result.returncode==2
        assert not spans and not (config.parent/'run').exists()


def test_300_record_run_resume_retains_record_trace_checkpoints(tmp_path):
    config=example(tmp_path,300)
    first=cli('run','--config',config,env=environment())
    assert first.returncode==0,first.stderr
    paths=list((config.parent/'run/task-traces').glob('*.json'));assert len(paths)==3
    before={p.name:p.read_bytes() for p in paths}
    second=cli('run','--config',config,env=environment());assert second.returncode==0,second.stderr
    assert before=={p.name:p.read_bytes() for p in paths}
    trace=json.loads((config.parent/'run/trace.json').read_text())
    assert trace['prior_segments']
    assert sum(s['attributes'].get('capability_anatomy.reason')=='observation_complete' for s in trace['prior_segments'][0]['spans'])==900
    assert cli('verify','--evidence',config.parent/'run',env=environment()).returncode==0


def test_storage_operation_refusal_reaches_actual_consumer(tmp_path):
    config = example(tmp_path)
    assert cli('run', '--config', config, env=environment()).returncode == 0
    manifest = config.parent / 'run/core-manifest.json'
    original = manifest.read_bytes()
    outside = tmp_path / 'outside.json'
    outside.write_bytes(original)
    manifest.unlink()
    manifest.symlink_to(outside)
    with consumer() as (endpoint, spans, _):
        result = cli('verify', '--evidence', manifest.parent, env=environment(endpoint))
        assert result.returncode == 2
        assert json.loads(result.stderr)['error'] == 'storage_symlink_refused'
        denied = [s for s in spans if s.name == 'capability_anatomy.storage.exists']
        assert len(denied) == 1
        assert attributes(denied[0])['capability_anatomy.reason'] == 'storage_symlink_refused'
        assert denied[0].status.code == 2
    assert outside.read_bytes() == original


def test_overflow_preserves_bounded_terminal_failure_receipt(tmp_path, monkeypatch):
    import pytest
    from capability_anatomy.execution import orchestrator
    from capability_anatomy.run_telemetry import BoundedSpanExporter, TelemetryCapacityError
    config = example(tmp_path)
    monkeypatch.setattr('capability_anatomy.run_telemetry.BoundedSpanExporter', lambda: BoundedSpanExporter(limit=30))
    with pytest.raises(TelemetryCapacityError):
        orchestrator.run_experiment(config)
    receipt_path, = list((config.parent / 'run.invocations').glob('*/trace.json'))
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'failed'
    assert receipt['complete'] is False
    assert receipt['dropped_spans'] > 0
    assert 0 < len(receipt['spans']) <= 30
    root, = [s for s in receipt['spans'] if s['name'] == 'capability_anatomy.invocation']
    assert root['status'] == 'ERROR'
    assert root['attributes']['capability_anatomy.reason'] == 'invocation_failed'
    assert not (config.parent / 'run/core-manifest.json').exists()


def test_only_terminal_otlp_batch_failure_has_explicit_local_failure_span(tmp_path):
    config = example(tmp_path)
    def status(path, raw):
        if path.endswith('/traces'):
            request = ExportTraceServiceRequest.FromString(raw)
            if any(span.name == 'capability_anatomy.invocation' for resource in request.resource_spans for scope in resource.scope_spans for span in scope.spans):
                return 503
        return 200
    with consumer(status=status) as (endpoint, spans, requests):
        result = cli('run', '--config', config, env=environment(endpoint))
        assert result.returncode == 3
        assert json.loads(result.stderr.splitlines()[-1])['error'] == 'telemetry_delivery_incomplete'
        receipt_path, = list((config.parent / 'run.invocations').glob('*/trace.json'))
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'failed'
        assert receipt['final_delivery_outcome']['status'] == 'incomplete'
        root, = [s for s in receipt['spans'] if s['name'] == 'capability_anatomy.invocation']
        assert root['attributes']['capability_anatomy.outcome_scope'] == 'evidence_publication_before_terminal_export'
        failure, = [s for s in receipt['spans'] if s['name'] == 'capability_anatomy.telemetry.delivery']
        assert failure['status'] == 'ERROR'
        assert failure['parent_span_id'] == root['span_id']
        assert failure['attributes']['capability_anatomy.reason'] == 'telemetry_delivery_incomplete'
        assert failure['span_id'] not in {span.span_id.hex() for span in spans}
        assert receipt['transport']['spans_not_delivered'] == 1
        assert receipt['transport']['spans_delivered'] == len(spans)
        assert len(requests) < 10


def test_phase5_traceless_resume_refuses_before_task_execution(tmp_path, monkeypatch):
    import pytest
    from capability_anatomy.execution.orchestrator import _require_phase5_resume_trace
    from capability_anatomy.execution.secure_fs import run_ownership
    from capability_anatomy.errors import InvalidEvidenceError
    config = example(tmp_path)
    assert cli('run', '--config', config, env=environment()).returncode == 0
    output = config.parent / 'run'
    with run_ownership(output):
        _require_phase5_resume_trace(output)
    trace = output / 'trace.json'
    trace.unlink()
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with run_ownership(output), pytest.raises(InvalidEvidenceError) as caught:
        _require_phase5_resume_trace(output)
    assert caught.value.reason == 'phase5_resume_trace_missing'
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_prior_trace_resume_reads_have_a_cumulative_bound(tmp_path, monkeypatch):
    import pytest
    from capability_anatomy.execution.orchestrator import _prior_trace_segments
    from capability_anatomy.execution.secure_fs import run_ownership
    from capability_anatomy.execution import evidence_limits
    from capability_anatomy.errors import InvalidEvidenceError
    output = tmp_path / 'run'
    output.mkdir()
    first = output / 'trace.json'
    second = output / 'trace-failures/earlier/trace.json'
    second.parent.mkdir(parents=True)
    first.write_text(json.dumps({'trace_id': 'a' * 32, 'spans': []}))
    second.write_text(json.dumps({'trace_id': 'b' * 32, 'spans': []}))
    before = (first.read_bytes(), second.read_bytes())
    with run_ownership(output):
        assert len(_prior_trace_segments(output)) == 2
        monkeypatch.setattr(evidence_limits, 'MAX_BUNDLE_BYTES', len(before[0]) + 1)
        with pytest.raises(InvalidEvidenceError) as caught:
            _prior_trace_segments(output)
        assert caught.value.reason == 'storage_artifact_byte_limit'
    assert before == (first.read_bytes(), second.read_bytes())


def test_authenticated_otlp_error_reason_cannot_leak_credentials(tmp_path):
    import pytest
    token = 'FAKE-OTLP-AUTH-CANARY-41'
    for response in (200, 400):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                received.append(self.headers.get('authorization'))
                self.send_response(response, token if response == 400 else 'OK')
                self.end_headers()
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            env = environment(f'http://127.0.0.1:{server.server_port}')
            env['OTEL_EXPORTER_OTLP_HEADERS'] = 'authorization=' + token
            result = cli('doctor', env=env)
            assert result.returncode == 0
            assert received and all(value == token for value in received)
            assert token not in result.stderr + result.stdout
            if response == 400:
                lines = [json.loads(line) for line in result.stderr.splitlines()]
                assert {line.get('signal') for line in lines if line.get('event') == 'otlp_export_diagnostic'} == {'traces', 'metrics'}
                assert all(line.get('trace_id') for line in lines if line.get('event') == 'otlp_export_diagnostic')
                assert lines[-1]['warning'] == 'telemetry_delivery_incomplete'
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_export_log_scope_preserves_other_threads_and_restores_handlers(caplog):
    import logging
    from capability_anatomy.run_telemetry import _export_diagnostics
    logger = logging.getLogger('opentelemetry.exporter.otlp.proto.http.trace_exporter')
    before = tuple(logger.filters)
    with _export_diagnostics(logger.name, 'traces', 'a' * 32):
        logger.warning('SERVER_CONTROLLED_REASON')
        thread = threading.Thread(target=lambda: logger.warning('ordinary-host-message'))
        thread.start(); thread.join()
    assert tuple(logger.filters) == before
    assert 'SERVER_CONTROLLED_REASON' not in caplog.text
    assert 'ordinary-host-message' in caplog.text
    assert 'otlp_export_diagnostic' in caplog.text


def test_export_exception_logs_are_redacted_before_attached_handler_formatting():
    import logging
    from capability_anatomy.run_telemetry import _export_diagnostics
    logger = logging.getLogger('opentelemetry.exporter.otlp.proto.http.metric_exporter')
    messages = []
    class Handler(logging.Handler):
        def emit(self, record):
            messages.append(self.format(record))
    handler = Handler()
    before = tuple(logger.filters)
    logger.addHandler(handler)
    try:
        with _export_diagnostics(logger.name, 'metrics', 'b' * 32):
            try:
                raise RuntimeError('FAKE-EXCEPTION-CREDENTIAL')
            except RuntimeError:
                logger.exception('server said %s', 'FAKE-RESPONSE-CREDENTIAL', stack_info=True)
        assert len(messages) == 1
        assert 'FAKE-' not in messages[0]
        assert 'Traceback' not in messages[0]
        assert json.loads(messages[0])['trace_id'] == 'b' * 32
        assert tuple(logger.filters) == before
    finally:
        logger.removeHandler(handler)


def test_malformed_otlp_headers_refuse_without_exposing_constructor_credentials():
    import logging
    import pytest
    from opentelemetry.sdk.trace import TracerProvider
    from capability_anatomy.run_telemetry import configure_otlp, OTLPHeaderError
    token = 'FAKE-OTLP-MALFORMED-CREDENTIAL'
    with consumer() as (endpoint, spans, requests):
        env = environment(endpoint)
        env['OTEL_EXPORTER_OTLP_HEADERS'] = token
        result = cli('doctor', env=env)
        assert result.returncode == 2
        assert token not in result.stdout + result.stderr
        assert json.loads(result.stderr.splitlines()[-1])['error'] == 'otlp_headers_invalid'
        assert not requests
        logger = logging.getLogger('opentelemetry.util.re')
        before = logger.disabled
        provider = TracerProvider()
        try:
            logger.disabled = True
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', endpoint)
                patch.setenv('OTEL_EXPORTER_OTLP_HEADERS', token)
                with pytest.raises(OTLPHeaderError):
                    configure_otlp(provider, [])
        finally:
            logger.disabled = before
            provider.shutdown()
