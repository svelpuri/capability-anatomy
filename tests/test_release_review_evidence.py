from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from capability_anatomy import cli
from capability_anatomy.errors import InvalidEvidenceError
from capability_anatomy.execution import core_evidence


def command(*args):
    return subprocess.run([sys.executable, '-m', 'capability_anatomy.cli', *map(str,args)],capture_output=True,text=True,timeout=30)


def example(tmp_path, count=2):
    directory=tmp_path/'example'
    assert command('example','--output',directory).returncode==0
    path=directory/'dataset.json';data=json.loads(path.read_text())
    template=data['partitions']['discovery'][0]
    data['partitions']['discovery']=[dict(template,id=f'discovery-{i}') for i in range(count)]
    path.write_text(json.dumps(data))
    return directory


def test_actual_300_record_run_publishes_and_reads_all_observation_spans(tmp_path):
    directory=example(tmp_path,300)
    result=command('run','--config',directory/'experiment.json')
    assert result.returncode==0,result.stderr
    trace=json.loads((directory/'run/trace.json').read_bytes())
    completed=[span for span in trace['spans'] if span['attributes'].get('capability_anatomy.reason')=='observation_complete']
    assert len(completed)==900
    assert len(trace['spans'])>3850
    resumed=command('run','--config',directory/'experiment.json')
    assert resumed.returncode==0,resumed.stderr
    resumed_trace=json.loads((directory/'run/trace.json').read_bytes())
    assert resumed_trace['prior_segments']
    assert len(list((directory/'run/task-traces').glob('*.json')))==3
    verified=command('verify','--evidence',directory/'run')
    assert verified.returncode==0,verified.stderr
    report=json.loads((directory/'run/report.json').read_bytes())
    assert {task['observations'] for task in report['tasks'].values()}=={300}


def test_generated_trace_capacity_refuses_before_any_publication(tmp_path,monkeypatch):
    directory=example(tmp_path)
    assert command('run','--config',directory/'experiment.json').returncode==0
    output=directory/'run'
    before={p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    trace=json.loads(before[Path('trace.json')])
    trace['extra']='x'*100
    monkeypatch.setattr(core_evidence,'MAX_TRACE_BYTES',len(before[Path('trace.json')])+10)
    with pytest.raises(InvalidEvidenceError):
        core_evidence.finalize_core_evidence(output,trace)
    assert {p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()}==before


def test_evidence_json_uses_byte_bounded_grammar_not_authored_node_cap():
    value={'spans':[{'attributes':{'a':1,'b':2,'c':3}} for _ in range(26000)]}
    assert core_evidence._mapping(json.dumps(value).encode())==value
    for invalid in (b'{"duplicate":1,"duplicate":2}',b'{"score":NaN}',b'{"score":Infinity}', b'['*70+b'0'+b']'*70):
        with pytest.raises(InvalidEvidenceError):core_evidence._json(invalid)


@pytest.mark.parametrize('scores,reason',[
    ([{'value':0.5,'numerator':1,'denominator':2},{'value':0.5}], 'score_fraction_mixed'),
    ([{'value':0.5,'numerator':1}], 'score_fraction_incomplete'),
    ([{'value':0.5,'denominator':2}], 'score_fraction_incomplete'),
    ([{'value':0.5,'numerator':1,'denominator':0}], 'score_denominator_invalid'),
    ([{'value':0.5,'numerator':1,'denominator':3}], 'score_fraction_inconsistent'),
    ([{'value':True}], 'score_value_invalid'),
    ([{'value':float('inf')}], 'score_value_invalid'),
])
def test_shared_reduction_refuses_ambiguous_or_invalid_scores(scores,reason):
    from capability_anatomy.evaluations.reduction import reduce_scores,ScoreReductionError
    with pytest.raises(ScoreReductionError) as failure:reduce_scores(scores)
    assert failure.value.reason==reason


def test_shared_reduction_working_fraction_and_value_controls():
    from capability_anatomy.evaluations.reduction import reduce_scores
    fractional=reduce_scores([{'value':1,'numerator':1,'denominator':1},{'value':.75,'numerator':3,'denominator':4}])
    assert fractional=={'value':.8,'numerator':4.0,'denominator':5.0,'observations':2,'reducer':'ratio_of_sums'}
    plain=reduce_scores([{'value':1},{'value':.75,'numerator':None,'denominator':None}])
    assert plain['value']==.875 and plain['denominator']==2 and plain['reducer']=='mean_value'


def test_actual_third_party_evaluation_aggregate_is_checked_offline(tmp_path,monkeypatch):
    import hashlib,shutil
    from capability_anatomy.serialization import canonical_json_bytes
    plugin=tmp_path/'plugins';plugin.mkdir()
    (plugin/'review_suite.py').write_text("from capability_anatomy.evaluations.plugins.synthetic import SyntheticExactMatchSuite\nclass Suite(SyntheticExactMatchSuite):\n name='review.external-exact'\n")
    metadata=plugin/'review_suite-1.dist-info';metadata.mkdir()
    (metadata/'METADATA').write_text('Metadata-Version: 2.1\nName: review-suite\nVersion: 1\n')
    (metadata/'entry_points.txt').write_text('[capability_anatomy.evaluations]\nreview.external-exact = review_suite:Suite\n')
    monkeypatch.setenv('PYTHONPATH',str(plugin))
    directory=example(tmp_path)
    config=directory/'experiment.json';value=json.loads(config.read_text());value['capability']['evaluation_plugin']='review.external-exact';config.write_text(json.dumps(value))
    result=command('run','--config',config)
    assert result.returncode==0,result.stderr
    output=directory/'run'
    assert 'ratio_of_sums' in (output/'report.md').read_text()
    shutil.rmtree(plugin)
    assert command('verify','--evidence',output).returncode==0
    path=output/'tasks/baseline.json';task=json.loads(path.read_text());task['metrics']['metrics']['exact_match']=.95;payload=canonical_json_bytes(task);path.write_bytes(payload)
    marker=output/'tasks/baseline.complete.json';value=json.loads(marker.read_text());value['payload_sha256']=hashlib.sha256(payload).hexdigest();marker.write_bytes(canonical_json_bytes(value))
    manifest_path=output/core_evidence.MANIFEST;manifest=json.loads(manifest_path.read_text())
    for entry in manifest['artifacts']:
        payload=(output/entry['path']).read_bytes();entry['bytes']=len(payload);entry['sha256']=hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    rebind_bundle(output)
    refused=command('verify','--evidence',output)
    assert refused.returncode==2,refused.stdout+refused.stderr


def rewrite_json(path, value):
    from capability_anatomy.serialization import canonical_json_bytes
    path.write_bytes(canonical_json_bytes(value))


def rebind_bundle(output):
    """Rebuild integrity metadata, deliberately preserving forged semantics."""
    config_hash = hashlib.sha256((output/'experiment-config.json').read_bytes()).hexdigest()
    state_path = output/'run-state.json'; state = json.loads(state_path.read_bytes())
    state['config_sha256'] = config_hash; rewrite_json(state_path, state)
    for marker_path in (output/'tasks').glob('*.complete.json'):
        marker = json.loads(marker_path.read_bytes())
        payload = (output/'tasks'/f"{marker['task_id']}.json").read_bytes()
        marker['payload_sha256'] = hashlib.sha256(payload).hexdigest()
        marker['config_sha256'] = config_hash; rewrite_json(marker_path, marker)
        checkpoint_path = output/'task-traces'/f"{marker['task_id']}.json"
        if checkpoint_path.exists():
            checkpoint=json.loads(checkpoint_path.read_bytes())
            checkpoint['payload_sha256']=marker['payload_sha256']; rewrite_json(checkpoint_path,checkpoint)
    manifest_path=output/core_evidence.MANIFEST; manifest=json.loads(manifest_path.read_bytes())
    manifest['artifacts']=[entry for entry in manifest['artifacts'] if (output/entry['path']).exists()]
    for entry in manifest['artifacts']:
        payload=(output/entry['path']).read_bytes(); entry['bytes']=len(payload); entry['sha256']=hashlib.sha256(payload).hexdigest()
    rewrite_json(manifest_path,manifest)


@pytest.mark.parametrize('defect,reason',[
    ('task_incomplete','evidence_task_incomplete'),
    ('observation_incomplete','evidence_task_incomplete'),
    ('aggregate','evidence_aggregate_mismatch'),
    ('required_metric','evidence_required_metrics_missing'),
    ('negative_denominator','score_denominator_invalid'),
])
def test_public_verify_refuses_semantic_forgery_after_all_hashes_are_rebuilt(tmp_path,defect,reason):
    directory=example(tmp_path); result=command('run','--config',directory/'experiment.json')
    assert result.returncode==0,result.stderr
    output=directory/'run'
    assert command('verify','--evidence',output).returncode==0
    path=output/'tasks/baseline.json'; task=json.loads(path.read_bytes())
    if defect=='task_incomplete': task['complete']=False
    elif defect=='observation_incomplete': task['observations'][0]['status']='error'
    elif defect=='aggregate': task['metrics']['metrics']['exact_match']=.95
    elif defect=='required_metric':
        config_path=output/'experiment-config.json'; config=json.loads(config_path.read_bytes())
        config['capability']['target_metrics'].append('missing_metric'); rewrite_json(config_path,config)
    elif defect=='negative_denominator':
        for observation in task['observations']:
            observation['scores']['exact_match'].update(value=0,numerator=0,denominator=-1)
        count=len(task['observations'])
        task['metrics']['metrics']['exact_match']=-0.0
        task['metrics']['sample_counts']['exact_match']=-count
        report_path=output/'report.json'; report=json.loads(report_path.read_bytes())
        report['tasks']['baseline']['scores']['exact_match'].update(value=-0.0,numerator=0.0,denominator=-float(count))
        report['tasks']={name:report['tasks'][name] for name in ['baseline']+[f"scan.{component.replace('.', '-')}" for component in report['component_scan_order']]}
        rewrite_json(report_path,report)
        (output/'report.md').write_text(core_evidence.render_core_report(report))
    rewrite_json(path,task); rebind_bundle(output)
    refused=command('verify','--evidence',output)
    assert refused.returncode==2,refused.stdout+refused.stderr
    assert json.loads(refused.stderr)['error']==reason


@pytest.mark.parametrize('defect',['missing_record','wrong_record','wrong_payload','truncated_legacy','wrong_trace_id','failed_record','zero_span_id','zero_trace_id','refused_record'])
def test_public_verify_refuses_trace_coverage_forgery(tmp_path,defect):
    directory=example(tmp_path); result=command('run','--config',directory/'experiment.json')
    assert result.returncode==0,result.stderr
    output=directory/'run'; assert command('verify','--evidence',output).returncode==0
    checkpoint_path=output/'task-traces/baseline.json'
    checkpoint=json.loads(checkpoint_path.read_bytes())
    records=[span for span in checkpoint['spans'] if span['attributes'].get('capability_anatomy.reason')=='observation_complete']
    if defect=='missing_record': checkpoint['spans'].remove(records[0])
    elif defect=='wrong_record': records[0]['attributes']['capability_anatomy.record_id_sha256']='f'*64
    elif defect=='wrong_payload': checkpoint['payload_sha256']='f'*64
    elif defect=='wrong_trace_id': records[0]['trace_id']='f'*32
    elif defect=='failed_record': records[0]['status']='ERROR'
    elif defect=='zero_span_id': records[0]['span_id']='0'*16
    elif defect=='zero_trace_id':
        checkpoint['trace_id']='0'*32
        for span in checkpoint['spans']:span['trace_id']='0'*32
    elif defect=='refused_record': records[0]['attributes']['capability_anatomy.outcome']='refused'
    else:
        for path in (output/'task-traces').glob('*.json'): path.unlink()
        (output/'task-traces').rmdir()
        trace_path=output/'trace.json'; trace=json.loads(trace_path.read_bytes())
        trace['spans']=[span for span in trace['spans'] if span['attributes'].get('capability_anatomy.reason')!='observation_complete']
        trace['prior_segments']=[]; rewrite_json(trace_path,trace)
    if defect!='truncated_legacy': rewrite_json(checkpoint_path,checkpoint)
    # Rebind inventory only, so wrong checkpoint payload binding remains wrong.
    manifest_path=output/core_evidence.MANIFEST; manifest=json.loads(manifest_path.read_bytes())
    manifest['artifacts']=[entry for entry in manifest['artifacts'] if (output/entry['path']).exists()]
    for entry in manifest['artifacts']:
        payload=(output/entry['path']).read_bytes(); entry['bytes']=len(payload); entry['sha256']=hashlib.sha256(payload).hexdigest()
    rewrite_json(manifest_path,manifest)
    refused=command('verify','--evidence',output)
    assert refused.returncode==2,refused.stdout+refused.stderr
    assert json.loads(refused.stderr)['error']=='evidence_trace_coverage_mismatch'
