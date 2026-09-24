"""Final release trace integrity and invocation lifecycle regressions."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.errors import InvalidEvidenceError
from capability_anatomy.execution import core_evidence
from capability_anatomy.execution.runner import ExperimentRunner,Task
from capability_anatomy.execution.store import FrozenRunStore
from capability_anatomy.serialization import canonical_json_bytes
from capability_anatomy.telemetry import OperationTelemetry
from test_release_cli import _cli,bundle,_rebind
from test_release_review_telemetry import consumer,environment,attributes
from test_release_model_profiles import real_profile,isolated_torch_state


def rewrite(path,value):
    path.write_bytes(canonical_json_bytes(value))


@pytest.mark.parametrize('target',['current','checkpoint','prior'])
@pytest.mark.parametrize('defect',['false','dropped','boolean','missing'])
def test_public_verify_and_report_refuse_incomplete_trace(bundle,target,defect):
    assert _cli('verify','--evidence',bundle).returncode == 0
    path=bundle/('task-traces/baseline.json' if target=='checkpoint' else 'trace.json')
    document=json.loads(path.read_bytes())
    if target=='prior':
        document['prior_segments']=[copy.deepcopy(document)]
        segment=document['prior_segments'][0]
    else: segment=document
    if defect=='false':segment['complete']=False
    elif defect=='dropped':segment['dropped_spans']=999
    elif defect=='boolean':segment['dropped_spans']=False
    else:segment.pop('complete')
    rewrite(path,document);_rebind(bundle)
    for command in ('verify','report'):
        result=_cli(command,'--evidence',bundle)
        assert result.returncode==2,result.stdout+result.stderr
        assert 'evidence_trace_incomplete' in result.stderr


def test_historical_v1_bytes_verify_with_explicit_unknown_completeness(bundle):
    for path in [bundle/'trace.json',*(bundle/'task-traces').glob('*.json')]:
        value=json.loads(path.read_bytes());value['schema_version']='capability-anatomy/phase5-trace/v1'
        value.pop('complete');value.pop('dropped_spans');rewrite(path,value)
    payloads=core_evidence._snapshot(bundle)
    report=core_evidence._report(payloads,report_version='capability-anatomy/core-report/v1')
    rewrite(bundle/'report.json',report);(bundle/'report.md').write_text(core_evidence.render_core_report(report));_rebind(bundle)
    before={p.name:p.read_bytes() for p in (bundle/'report.json',bundle/'report.md')}
    result=_cli('report','--format','json','--evidence',bundle)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['trace_evidence']['status']=='legacy_not_recorded'
    assert before=={p.name:p.read_bytes() for p in (bundle/'report.json',bundle/'report.md')}
    path=bundle/'trace.json';value=json.loads(path.read_bytes());value.update(complete=True,dropped_spans=1);rewrite(path,value);_rebind(bundle)
    assert _cli('verify','--evidence',bundle).returncode==2


@pytest.mark.parametrize('defect',['extra','taskhash','payloadhash','outcome','error','missing'])
def test_checkpoint_guards_refuse_after_full_manifest_rebinding(bundle,defect):
    assert _cli('verify','--evidence',bundle).returncode==0
    path=bundle/'task-traces/baseline.json';value=json.loads(path.read_bytes())
    if defect=='extra':rewrite(bundle/'task-traces/unplanned.json',value)
    elif defect=='missing':path.unlink()
    else:
        if defect=='taskhash':value['task_id_sha256']='0'*64
        elif defect=='payloadhash':value['payload_sha256']='0'*64
        else:
            span=next(s for s in value['spans'] if s['attributes'].get('capability_anatomy.reason')=='observation_complete')
            if defect=='outcome':span['attributes']['capability_anatomy.outcome']='refused'
            else:span['status']='ERROR'
        rewrite(path,value)
    _rebind(bundle)
    result=_cli('verify','--evidence',bundle)
    assert result.returncode==2,result.stdout+result.stderr
    assert 'evidence_trace_coverage_mismatch' in result.stderr


def test_checkpoint_multiplicity_is_not_reduced_to_membership(tmp_path):
    example=tmp_path/'example';assert _cli('example','--output',example).returncode==0
    config=example/'experiment.json';value=json.loads(config.read_bytes());value['runtime']['repetitions']=2;rewrite(config,value)
    result=_cli('run','--config',config);assert result.returncode==0,result.stderr
    output=example/'run';assert _cli('verify','--evidence',output).returncode==0
    path=output/'task-traces/baseline.json';segment=json.loads(path.read_bytes())
    observed=[s for s in segment['spans'] if s['attributes'].get('capability_anatomy.reason')=='observation_complete']
    ids=[s['attributes']['capability_anatomy.record_id_sha256'] for s in observed]
    assert len(ids)>len(set(ids))
    segment['spans'].remove(observed[0]);rewrite(path,segment);_rebind(output)
    result=_cli('verify','--evidence',output)
    assert result.returncode==2,result.stdout+result.stderr
    assert 'evidence_trace_coverage_mismatch' in result.stderr


def test_report_markdown_reconstructs_from_sorted_json_roundtrip(bundle):
    report=json.loads((bundle/'report.json').read_bytes())
    assert report['schema_version']=='capability-anatomy/core-report/v2'
    report['tasks']=dict(reversed(list(report['tasks'].items())))
    assert core_evidence.render_core_report(report)==core_evidence.render_core_report(json.loads(canonical_json_bytes(report)))
    assert core_evidence.render_core_report(report)==(bundle/'report.md').read_text()


def test_plugin_minimal_declared_aggregate_works_and_is_checked_offline(tmp_path,monkeypatch):
    plugin=tmp_path/'plugin';plugin.mkdir()
    (plugin/'minimal.py').write_text('from capability_anatomy.evaluations.plugins.synthetic import SyntheticExactMatchSuite\nclass Suite(SyntheticExactMatchSuite):\n name="review.minimal"\n def aggregate(self, observations):\n  value=super().aggregate(observations)\n  return {"metrics":value["metrics"],"errors":value["errors"]}\n')
    metadata=plugin/'minimal-1.dist-info';metadata.mkdir();(metadata/'METADATA').write_text('Metadata-Version: 2.1\nName: minimal\nVersion: 1\n')
    (metadata/'entry_points.txt').write_text('[capability_anatomy.evaluations]\nreview.minimal = minimal:Suite\n')
    original_pythonpath=os.environ.get('PYTHONPATH')
    monkeypatch.setenv('PYTHONPATH',str(plugin)+(os.pathsep+original_pythonpath if original_pythonpath else ''))
    example=tmp_path/'example';assert _cli('example','--output',example).returncode==0
    config=example/'experiment.json';value=json.loads(config.read_bytes());value['capability']['evaluation_plugin']='review.minimal';rewrite(config,value)
    result=_cli('run','--config',config);assert result.returncode==0,result.stderr
    if original_pythonpath:monkeypatch.setenv('PYTHONPATH',original_pythonpath)
    else:monkeypatch.delenv('PYTHONPATH')
    output=example/'run';assert _cli('verify','--evidence',output).returncode==0
    task=output/'tasks/baseline.json';value=json.loads(task.read_bytes());value['metrics']['metrics']['exact_match']=.123;rewrite(task,value)
    marker=output/'tasks/baseline.complete.json';m=json.loads(marker.read_bytes());m['payload_sha256']=hashlib.sha256(task.read_bytes()).hexdigest();rewrite(marker,m)
    checkpoint=output/'task-traces/baseline.json';c=json.loads(checkpoint.read_bytes());c['payload_sha256']=m['payload_sha256'];rewrite(checkpoint,c);_rebind(output)
    result=_cli('verify','--evidence',output);assert result.returncode==2 and 'evidence_aggregate_mismatch' in result.stderr


def test_incomplete_task_keeps_one_specific_decision_and_metric(tmp_path):
    exporter=InMemorySpanExporter();provider=TracerProvider();provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader=InMemoryMetricReader();meters=MeterProvider(metric_readers=[reader]);signals=OperationTelemetry.create('execution',provider.get_tracer('test'),meters.get_meter('test'))
    store=FrozenRunStore(tmp_path/'run',{},{});runner=ExperimentRunner(store,signals)
    try:
        with pytest.raises(InvalidEvidenceError):runner.run((Task('baseline','baseline',lambda:{'complete':False}),))
        task,=[s for s in exporter.get_finished_spans() if s.name=='capability_anatomy.execution.task']
        assert task.attributes['capability_anatomy.reason']=='task_incomplete'
        assert task.status.status_code.name=='ERROR'
        events=[e for e in task.events if e.name=='operation.decision'];assert len(events)==1
        points=[p for r in reader.get_metrics_data().resource_metrics for scope in r.scope_metrics for metric in scope.metrics for p in metric.data.data_points if p.attributes.get('capability_anatomy.operation')=='execute_task']
        assert len(points)==1 and points[0].value==1
        assert points[0].attributes['capability_anatomy.reason']=='task_incomplete'
    finally:provider.shutdown();meters.shutdown()


def test_finalize_failure_marks_actual_run_span_error(tmp_path):
    example=tmp_path/'example';assert _cli('example','--output',example).returncode==0
    with consumer() as (endpoint,spans,_requests):
        env=environment(endpoint);env['OTEL_EXPORTER_OTLP_TIMEOUT']='2'
        script='''import sys
from capability_anatomy import cli
from capability_anatomy.execution import core_evidence
from capability_anatomy.errors import InvalidEvidenceError
def refused(*a,**k):
 error=InvalidEvidenceError('forced finalize refusal');error.reason='evidence_aggregate_mismatch';raise error
core_evidence.finalize_core_evidence=refused
raise SystemExit(cli.main(['run','--config',sys.argv[1]]))
'''
        result=subprocess.run([sys.executable,'-c',script,str(example/'experiment.json')],env=env,capture_output=True,text=True,timeout=30)
        assert result.returncode==2,result.stdout+result.stderr
        run,=[s for s in spans if s.name=='capability_anatomy.run']
        assert run.status.code==2
        assert attributes(run)['capability_anatomy.reason']=='evidence_aggregate_mismatch'
        assert attributes(run)['capability_anatomy.outcome']=='failed'


@pytest.mark.parametrize('flush_number',[1,2])
def test_sigterm_during_postlease_drain_retains_receipt_and_completed_result(tmp_path,flush_number):
    example=tmp_path/'example';assert _cli('example','--output',example).returncode==0
    with consumer(delay=.03) as (endpoint,spans,_requests):
        env=environment(endpoint);env['OTEL_EXPORTER_OTLP_TIMEOUT']='2'
        script='''import os,signal,sys
from capability_anatomy import cli
from capability_anatomy.run_telemetry import OTLPDelivery
original=OTLPDelivery.flush
count=0
def interrupted(self):
 global count
 count+=1
 if count==int(sys.argv[2]):os.kill(os.getpid(),signal.SIGTERM)
 return original(self)
OTLPDelivery.flush=interrupted
raise SystemExit(cli.main(['run','--config',sys.argv[1]]))
'''
        result=subprocess.run([sys.executable,'-c',script,str(example/'experiment.json'),str(flush_number)],env=env,capture_output=True,text=True,timeout=30)
        assert result.returncode==4,result.stdout+result.stderr
        summary=json.loads(result.stdout);error=json.loads(result.stderr.splitlines()[-1])
        assert summary['status']=='complete' and summary['evidence_status']=='complete'
        assert error['error']=='delivery_interrupted'
        receipt=json.loads(Path(summary['invocation_trace']).read_bytes())
        assert receipt['status']=='interrupted' and receipt['evidence_status']=='complete'
        assert receipt['interruption_requested'] is True
        assert receipt['failure_reason']=='delivery_interrupted'
        assert any(s['attributes'].get('capability_anatomy.reason')=='delivery_interrupted' and s['status']=='ERROR' for s in receipt['spans'])
        assert json.loads((example/'run/execution-state.json').read_bytes())=={'state':'complete','reason':'all_tasks_committed'}
        assert _cli('verify','--evidence',example/'run').returncode==0


@pytest.mark.parametrize('real_profile',['Qwen3'],indirect=True)
def test_phase5_public_consumer_refuses_unbound_metrics_or_trace_loss(tmp_path,real_profile):
    from test_release_portable_campaign import test_public_portable_review_policy_admits_tiny_conformance_and_refuses_identity_defects
    test_public_portable_review_policy_admits_tiny_conformance_and_refuses_identity_defects(real_profile,tmp_path)
    root=tmp_path/'run';positive=_cli('verify','--evidence',root);assert positive.returncode==0,positive.stderr
    path=root/'trace.json';original=json.loads(path.read_bytes())
    manifest_path=root/'evidence-manifest.json';manifest=json.loads(manifest_path.read_bytes())
    for defect in ('rogue_metric','dropped_spans'):
        trace=copy.deepcopy(original)
        if defect=='rogue_metric':
            trace['metrics'][0]['points'].append({'attributes':{'capability_anatomy.reason':'fabricated_metric','capability_anatomy.component':'fabricated','capability_anatomy.operation':'invent','capability_anatomy.outcome':'accepted'},'value':1})
        else:trace.update(complete=True,dropped_spans=999)
        rewrite(path,trace)
        for entry in manifest['artifacts']:
            if entry['path']=='trace.json':entry.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rewrite(manifest_path,manifest)
        result=_cli('verify','--evidence',root)
        assert result.returncode==2,result.stdout+result.stderr
        expected='evidence_trace_incomplete' if defect=='dropped_spans' else 'invalid_evidence'
        assert expected in result.stderr


@pytest.mark.parametrize('transport_failure',[False,True])
def test_unwritable_receipt_never_advertises_a_nonexistent_locator(tmp_path,transport_failure):
    example=tmp_path/'example';assert _cli('example','--output',example).returncode==0
    blocker=example/'run.invocations';blocker.write_bytes(b'preserved operator file')
    with consumer(status=503 if transport_failure else 200) as (endpoint,_spans,_requests):
        result=subprocess.run([sys.executable,'-m','capability_anatomy.cli','run','--config',str(example/'experiment.json')],env=environment(endpoint),capture_output=True,text=True,timeout=30)
        assert result.returncode==(3 if transport_failure else 2),result.stdout+result.stderr
        summary=json.loads(result.stdout)
        assert summary['status']=='complete' and summary['receipt_status']=='unavailable'
        assert 'invocation_trace' not in summary
        error=json.loads(result.stderr.splitlines()[-1])
        assert error['error']==('telemetry_delivery_incomplete' if transport_failure else 'invocation_receipt_unwritable')
        assert 'invocation_trace' not in error.get('context',{})
    assert blocker.read_bytes()==b'preserved operator file'
    assert _cli('verify','--evidence',example/'run').returncode==0


def test_sigterm_after_real_receipt_publication_is_recorded_once(tmp_path):
    example=tmp_path/'example';assert _cli('example','--output',example).returncode==0
    script='''import os,signal,sys
from pathlib import Path
from capability_anatomy import cli
from capability_anatomy.execution.store import FrozenRunStore
original=FrozenRunStore._atomic_write
count=0
def interrupted(path,payload):
 global count
 original(path,payload)
 if '.invocations' in str(path):
  count+=1
  os.kill(os.getpid(),signal.SIGTERM)
FrozenRunStore._atomic_write=staticmethod(interrupted)
code=cli.main(['run','--config',sys.argv[1]])
assert count==2,count
raise SystemExit(code)
'''
    result=subprocess.run([sys.executable,'-c',script,str(example/'experiment.json')],capture_output=True,text=True,timeout=30)
    assert result.returncode==4,result.stdout+result.stderr
    summary=json.loads(result.stdout);receipt=json.loads(Path(summary['invocation_trace']).read_bytes())
    assert summary['interruption_requested'] is True and summary['evidence_status']=='complete'
    assert receipt['status']=='interrupted' and receipt['interruption_requested'] is True
    assert receipt['interruption_phase']=='receipt_publication'
    interrupted=[s for s in receipt['spans'] if s['attributes'].get('capability_anatomy.reason')=='delivery_interrupted']
    assert len(interrupted)==1 and interrupted[0]['status']=='ERROR'
    assert _cli('verify','--evidence',example/'run').returncode==0
