"""Round-two density admission at the parser and resumed task consumer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tracemalloc

import pytest

from capability_anatomy.execution import evidence_json, secure_fs
from capability_anatomy.execution.evidence_json import EvidenceFormatError, parse_json, validate_tree
from capability_anatomy.execution.evidence_limits import MAX_ARTIFACT_BYTES, MAX_TRACE_BYTES, artifact_limit
from capability_anatomy.execution.store import FrozenRunStore


def command(*args):
    return subprocess.run([sys.executable, '-m', 'capability_anatomy.cli', *map(str,args)], capture_output=True,text=True,timeout=60)



_FRESH_LAUNCHER = """import subprocess,sys
raise SystemExit(subprocess.call([sys.executable,'-c',sys.argv[1],*sys.argv[2:]]))
"""


def fresh_process(script, *arguments):
    # A low-resident interpreter launches the measured child. Linux preserves
    # resource watermarks over exec, so measuring a child forked directly from
    # a large pytest/torch process can otherwise include the parent's RSS.
    return subprocess.run([sys.executable,'-c',_FRESH_LAUNCHER,script,*map(str,arguments)],
                          capture_output=True,text=True,timeout=60)

def test_density_refused_before_stdlib_can_allocate(monkeypatch):
    # Small independent budget gives the exact production guard a cheap mutant.
    payload = b'[' + b'{},' * 80 + b'{}]'
    monkeypatch.setattr(evidence_json.json, 'loads', lambda *a, **k: pytest.fail('object construction reached'))
    with pytest.raises(EvidenceFormatError) as failure:
        parse_json(payload, max_bytes=1024)
    assert failure.value.reason == 'evidence_json_node_limit'


def test_depth_refused_before_stdlib_can_allocate(monkeypatch):
    monkeypatch.setattr(evidence_json.json, 'loads', lambda *a, **k: pytest.fail('object construction reached'))
    with pytest.raises(EvidenceFormatError) as failure:
        parse_json(b'[' * 66 + b'0' + b']' * 66)
    assert failure.value.reason == 'evidence_json_depth_limit'


@pytest.mark.parametrize('value', [[0] * 63, {str(i): 0 for i in range(31)}])
def test_reader_writer_node_boundary_matches(value):
    validate_tree(value, max_bytes=1024)
    assert parse_json(json.dumps(value).encode(), max_bytes=1024) == value
    if isinstance(value,list):
        value.append(0)
    else:
        value['additional'] = 0
    for operation in (lambda: validate_tree(value,max_bytes=1024),lambda: parse_json(json.dumps(value).encode(),max_bytes=1024)):
        with pytest.raises(EvidenceFormatError) as failure:
            operation()
        assert failure.value.reason == 'evidence_json_node_limit'


def test_writer_traversal_does_not_queue_wide_children():
    value = [0] * 800_000
    tracemalloc.start()
    try:
        validate_tree(value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024, peak


@pytest.mark.parametrize('payload,reason',[
    (b'{"x":1,"\\u0078":2}', 'evidence_json_duplicate_key'),
    (b'1e9999','evidence_json_nonfinite'),
    (b'1e9999999999999999999999999','evidence_json_invalid'),
    (b'1e-9999999999999999999999999','evidence_json_invalid'),
    (b'NaN','evidence_json_invalid'),
    (b'Infinity','evidence_json_invalid'),
    (b'-Infinity','evidence_json_invalid'),
    (b'01','evidence_json_invalid'),
    (b'[1,]','evidence_json_invalid'),
    (b'{}{}','evidence_json_invalid'),
    (b'/* comment */ {}','evidence_json_invalid'),
    (b'"\xff"','evidence_json_invalid'),
    (b'{"secret":"sensitive', 'evidence_json_invalid'),
])
def test_strict_grammar_refusal_is_safe(payload,reason):
    with pytest.raises(EvidenceFormatError) as failure:
        parse_json(payload)
    assert failure.value.reason == reason
    assert 'sensitive' not in str(failure.value)


@pytest.mark.parametrize('payload',[
    b'{"a":[true,false,null,1.25,123456789012345678901234567890]}',
    b'{"\\ud800":1,"?":2,"\\ud801":3}',
    b'{"\\ud83d\\ude00":"unicode"}',
])
def test_stdlib_value_and_unicode_key_semantics_preserved(payload):
    assert parse_json(payload) == json.loads(payload)


def test_full_50000_span_sdk_snapshot_roundtrips():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from capability_anatomy.run_telemetry import BoundedSpanExporter
    from capability_anatomy.telemetry import OperationTelemetry
    from capability_anatomy.execution.orchestrator import _serialize_phase5_trace
    exporter = BoundedSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    signals = OperationTelemetry.create('evaluation',provider.get_tracer('density-test'),meter_provider.get_meter('density-test'))
    try:
        for _ in range(50_000):
            with signals.tracer.start_as_current_span('capability_anatomy.evaluation.observe') as span:
                signals.record(span,operation='observe',outcome='accepted',reason='observation_complete')
        snapshot = _serialize_phase5_trace(exporter,reader,None)
        assert len(snapshot['spans']) == 50_000
        validate_tree(snapshot,max_bytes=MAX_TRACE_BYTES)
        payload = json.dumps(snapshot,separators=(',',':')).encode()
        assert len(payload) < MAX_TRACE_BYTES
        assert parse_json(payload,max_bytes=MAX_TRACE_BYTES) == snapshot
    finally:
        provider.shutdown()
        meter_provider.shutdown()


def test_task_trace_classification_is_load_bearing():
    assert artifact_limit('task-traces/baseline.json') == 64 * 1024 * 1024
    assert artifact_limit('tasks/baseline.json') == 16 * 1024 * 1024
    assert artifact_limit('trace.json') == 64 * 1024 * 1024


def test_frozen_store_sparse_task_refuses_at_bounded_read(tmp_path):
    store = FrozenRunStore(tmp_path/'run',{}, {})
    store.initialize()
    store.commit('baseline',{'answer':42})
    assert store.completed('baseline') == {'answer':42}
    with (store.root/'tasks/baseline.json').open('r+b') as handle:
        handle.truncate(1024 * 1024 * 1024)
    with pytest.raises(secure_fs.StorageError) as failure:
        store.completed('baseline')
    assert failure.value.reason == 'storage_artifact_byte_limit'


def test_frozen_store_dense_task_refuses_even_with_matching_marker(tmp_path,monkeypatch):
    from capability_anatomy.execution import store as store_module
    store = FrozenRunStore(tmp_path/'run',{}, {})
    store.initialize()
    store.commit('baseline',{'answer':42})
    monkeypatch.setattr(store_module,'MAX_ARTIFACT_BYTES',1024)
    payload = b'{"rows":[' + b'{},' * 80 + b'{}]}'
    (store.root/'tasks/baseline.json').write_bytes(payload)
    marker_path = store.root/'tasks/baseline.complete.json'
    marker = json.loads(marker_path.read_bytes())
    marker['payload_sha256'] = hashlib.sha256(payload).hexdigest()
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(EvidenceFormatError) as failure:
        store.completed('baseline')
    assert failure.value.reason == 'evidence_json_node_limit'


def test_frozen_store_writer_refuses_density_before_publishing(tmp_path,monkeypatch):
    from capability_anatomy.execution import store as store_module
    store = FrozenRunStore(tmp_path/'run',{}, {})
    store.initialize()
    store.commit('baseline',{'answer':42})
    before = (store.root/'tasks/baseline.json').read_bytes()
    monkeypatch.setattr(store_module,'MAX_ARTIFACT_BYTES',1024)
    with pytest.raises(EvidenceFormatError) as failure:
        store.commit('baseline',{'rows':[{} for _ in range(80)]})
    assert failure.value.reason == 'evidence_json_node_limit'
    assert (store.root/'tasks/baseline.json').read_bytes() == before


def test_resealed_64mib_dense_trace_refused_under_memory_ceiling(tmp_path):
    example = tmp_path/'example'
    created = command('example','--output',example)
    assert created.returncode == 0, created.stderr
    run = command('run','--config',example/'experiment.json')
    assert run.returncode == 0,run.stderr
    output = example/'run'
    assert command('verify','--evidence',output).returncode == 0
    trace = output/'trace.json'
    prefix = trace.read_bytes().rstrip()[:-1] + b',"dense_padding":['
    with trace.open('wb') as handle:
        handle.write(prefix)
        chunk = b'[],' * 100_000
        while handle.tell() + len(chunk) + 3 <= MAX_TRACE_BYTES:
            handle.write(chunk)
        handle.write(b'0]}')
        handle.write(b' ' * (MAX_TRACE_BYTES-handle.tell()))
    assert trace.stat().st_size == 64 * 1024 * 1024
    manifest_path = output/'core-manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    for entry in manifest['artifacts']:
        if entry['path'] == 'trace.json':
            entry['bytes'] = trace.stat().st_size
            digest = hashlib.sha256()
            with trace.open('rb') as handle:
                while chunk := handle.read(1024*1024):
                    digest.update(chunk)
            entry['sha256'] = digest.hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    # Normalize native ru_maxrss units and record the pre-import watermark so
    # attribution to the measured CLI is checked, not assumed.
    script = '''import json,resource,sys
def peak_bytes():
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform=='darwin' else rss*1024
baseline=peak_bytes()
from capability_anatomy.cli import main
import ijson
code=main(['verify','--evidence',sys.argv[1]])
print(json.dumps({'code':code,'baseline_peak_bytes':baseline,'peak_bytes':peak_bytes(),'parser_backend':ijson.backend}),file=sys.stderr)
raise SystemExit(code)
'''
    probe = fresh_process(script,output)
    assert probe.returncode == 2,probe.stdout+probe.stderr
    assert 'evidence_json_node_limit' in probe.stdout+probe.stderr
    measurement = json.loads(probe.stderr.splitlines()[-1])
    assert measurement['baseline_peak_bytes'] < 128 * 1024 * 1024,measurement
    assert measurement['peak_bytes'] < 512 * 1024 * 1024,measurement
    print(json.dumps({'dense_verify':measurement}))


def test_store_byte_bound_negative_with_small_real_file(tmp_path,monkeypatch):
    from capability_anatomy.execution import store as store_module
    store = FrozenRunStore(tmp_path/'run',{}, {})
    store.initialize()
    store.commit('baseline',{'answer':42})
    monkeypatch.setattr(store_module,'MAX_ARTIFACT_BYTES',1024)
    (store.root/'tasks/baseline.json').write_bytes(b' ' * 2048)
    with pytest.raises(secure_fs.StorageError) as failure:
        store.completed('baseline')
    assert failure.value.reason == 'storage_artifact_byte_limit'


def test_jsonl_uses_one_cumulative_node_budget_before_allocation(monkeypatch):
    from capability_anatomy.execution.evidence_json import parse_json_lines
    rows = [{"a": 0} for _ in range(21)]
    payload = b"\n".join(json.dumps(row).encode() for row in rows)
    validate_tree(rows,max_bytes=1024)
    assert parse_json_lines(payload,max_bytes=1024) == rows
    monkeypatch.setattr(evidence_json.json, 'loads', lambda *a, **k: pytest.fail('object construction reached'))
    with pytest.raises(EvidenceFormatError) as failure:
        parse_json_lines(payload+b'\n{"a":0}',max_bytes=1024)
    assert failure.value.reason == 'evidence_json_node_limit'


@pytest.mark.parametrize('payload',[b'{} {}',b'{\n"a":1\n}',b'{"a":1,"a":2}'])
def test_jsonl_retains_strict_line_grammar(payload):
    from capability_anatomy.execution.evidence_json import parse_json_lines
    with pytest.raises(EvidenceFormatError):
        parse_json_lines(payload)


def test_memory_probe_excludes_inherited_parent_watermark():
    # Deliberately hold resident memory in an isolated parent, independently of
    # test ordering or optional torch imports. On Linux the direct-exec control
    # must reproduce the contamination; the intermediary must remove it.
    measure = "import resource,sys; r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; print(r if sys.platform=='darwin' else r*1024)"
    polluter = '''import json,resource,subprocess,sys
held=bytearray(192*1024*1024)
measure=sys.argv[1]
launcher=sys.argv[2]
direct=subprocess.run([sys.executable,'-c',measure],capture_output=True,text=True,check=True)
fresh=subprocess.run([sys.executable,'-c',launcher,measure],capture_output=True,text=True,check=True)
print(json.dumps({'direct_peak_bytes':int(direct.stdout),'fresh_peak_bytes':int(fresh.stdout),'held_bytes':len(held),'platform':sys.platform}))
'''
    result = fresh_process(polluter,measure,_FRESH_LAUNCHER)
    assert result.returncode == 0,result.stdout+result.stderr
    measurement = json.loads(result.stdout)
    if sys.platform == 'linux':
        assert measurement['direct_peak_bytes'] >= 192 * 1024 * 1024,measurement
    assert measurement['fresh_peak_bytes'] < 128 * 1024 * 1024,measurement
    print(json.dumps({'memory_probe_control':measurement}))
