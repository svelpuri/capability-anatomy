"""Bound every Phase5 artifact read before allocation, including re-reads."""
import hashlib
import pytest
from capability_anatomy.errors import InvalidEvidenceError
from capability_anatomy.execution.phase5_artifacts import Phase5ArtifactReader
from capability_anatomy.execution.phase5_campaign import reconstruct_manifest, verify_phase5_aggregates

def _manifest(name, payload):
    return {'schema_version':'capability-anatomy/phase5-evidence-manifest/v1', 'artifacts':[
        {'path':name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}


def test_manifest_size_bounds_the_actual_read_before_allocation(tmp_path, monkeypatch):
    payload=b'{}'; path=tmp_path/'observations.jsonl';path.write_bytes(payload)
    manifest=_manifest(path.name,payload)
    reconstruct_manifest(tmp_path,manifest,[path.name])
    with path.open('r+b') as stream: stream.truncate(1024*1024*1024)
    from capability_anatomy.execution import secure_fs
    original=secure_fs.read_bytes; calls=[]
    def read(path, *, max_bytes=None):
        calls.append(max_bytes)
        assert max_bytes==len(payload)
        return original(path,max_bytes=max_bytes)
    monkeypatch.setattr(secure_fs,'read_bytes',read)
    with pytest.raises(InvalidEvidenceError, match='byte limit'):
        reconstruct_manifest(tmp_path,manifest,[path.name])
    assert calls==[2]
    calls.clear()
    with pytest.raises(InvalidEvidenceError, match='byte limit'):
        verify_phase5_aggregates(tmp_path,manifest)
    assert calls==[2]


@pytest.mark.parametrize('size',[-1,True,1024*1024*1024])
def test_invalid_declared_size_refuses_before_any_artifact_read(tmp_path, monkeypatch, size):
    manifest=_manifest('observations.jsonl',b'{}');manifest['artifacts'][0]['bytes']=size
    from capability_anatomy.execution import secure_fs
    monkeypatch.setattr(secure_fs,'read_bytes',lambda *a,**kw:pytest.fail('invalid size reached I/O'))
    with pytest.raises(InvalidEvidenceError):
        Phase5ArtifactReader(tmp_path,manifest)


from capability_anatomy.execution.phase5_campaign import aggregate_observations
from capability_anatomy.evaluations.reduction import ScoreReductionError


def _row(identity, repetition, score):
    return {'condition':'baseline','component_id':'baseline','partition':'discovery',
            'example_id':identity,'group_id':identity,'repetition':repetition,'status':'complete',
            'expected_metric_ids':['selection'],'scores':{'selection':score}}


@pytest.mark.parametrize('version',['1','2'])
def test_mixed_reducers_refuse_without_reinterpreting_legacy_scores(version):
    rows=[_row('a',0,{'value':.8,'numerator':8,'denominator':10}),
          _row('b',0,{'value':.2,'numerator':None,'denominator':None})]
    with pytest.raises(ScoreReductionError) as error:
        aggregate_observations(rows,['selection'],semantics_version=version)
    assert error.value.reason=='score_fraction_mixed'


@pytest.mark.parametrize('weighted',[False,True])
def test_total_and_repetition_use_the_same_reducer_contract(weighted):
    scores=([{'value':.8,'numerator':8,'denominator':10}, {'value':0.,'numerator':0,'denominator':1}]
            if weighted else [{'value':.5},{'value':1.5}])
    rows=[_row(str(i),repeat,score) for repeat in range(2) for i,score in enumerate(scores)]
    result=aggregate_observations(rows,['selection'])['selection']
    expected=8/11 if weighted else 1
    assert result['value']==pytest.approx(expected)
    assert result['repetition_values']==pytest.approx([expected,expected])
    assert result['complete_count']==4


def test_fraction_value_disagreement_refuses_before_publishing_aggregate():
    with pytest.raises(ScoreReductionError) as error:
        aggregate_observations([_row('a',0,{'value':.8,'numerator':1,'denominator':2})],['selection'])
    assert error.value.reason=='score_fraction_inconsistent'


def test_v1_threshold_reconstruction_preserves_its_historical_stage_rules():
    from capability_anatomy.execution.phase5_campaign import select_discovery_candidates,compute_validation_results
    damage=.05+1e-16
    spec={'selection':{'role':'target','direction':'higher_is_better'}}
    rule={'target_absolute_damage_max':.05,'collateral_absolute_damage_max':.1,
          'perplexity_relative_damage_max':.1,'minimum_drift_margin':.025}
    baseline={'value':1.,'unique_record_count':2,'unique_group_count':2}
    evidence={'block':{'selection':{'baseline':baseline,'condition':{'complete_count':2,'error_count':0,'minimum':.95,'maximum':.95},'absolute_damage':damage}}}
    ranking=select_discovery_candidates(evidence,spec,rule,drift_by_metric={'selection':0},matched_random_damage={'block':{'selection':damage}})
    assert ranking['candidates']==['block']
    legacy=compute_validation_results(['block'],evidence,spec,rule,semantics_version='1')
    current=compute_validation_results(['block'],evidence,spec,rule,semantics_version='2')
    assert legacy['results']==[{'component_id':'block','outcome':'fail','reasons':['selection:threshold_exceeded']}]
    assert current['results']==[{'component_id':'block','outcome':'pass','reasons':[]}]


def test_phase5_inventory_stops_the_actual_directory_iterator_at_limit(tmp_path, monkeypatch):
    from capability_anatomy.execution import evidence_inventory, phase5_artifacts
    from contextlib import contextmanager
    for index in range(30):
        (tmp_path / str(index)).write_text('{}')
    original = evidence_inventory.os.scandir
    consumed = []
    @contextmanager
    def observed(path):
        with original(path) as directory:
            def entries():
                for entry in directory:
                    consumed.append(entry.name)
                    yield entry
            yield entries()
    monkeypatch.setattr(evidence_inventory.os, 'scandir', observed)
    monkeypatch.setattr(phase5_artifacts, 'MAX_ARTIFACTS', 5)
    with pytest.raises(InvalidEvidenceError):
        list(phase5_artifacts.artifact_paths(tmp_path))
    assert len(consumed) == 6
    consumed.clear()
    monkeypatch.setattr(phase5_artifacts, 'MAX_ARTIFACTS', 30)
    assert len(list(phase5_artifacts.artifact_paths(tmp_path))) == 30
    assert len(consumed) == 30


def test_phase5_nested_system_temp_alias_works_but_user_symlinks_refuse(tmp_path):
    import tempfile
    from pathlib import Path
    from capability_anatomy.execution.phase5_campaign import write_final_artifacts
    with tempfile.TemporaryDirectory(prefix='ca-phase5-alias-', dir='/tmp') as location:
        output = Path(location) / 'nested/run'
        manifest = write_final_artifacts(output, {'report.json': {'status':'complete'}},
                                        required_paths=['report.json','evidence-manifest.json'], identity={})
        reconstruct_manifest(output, manifest, ['report.json'])
        assert Phase5ArtifactReader(output, manifest).json('report.json') == {'status':'complete'}
        outside = tmp_path/'outside'; outside.mkdir()
        link = Path(location)/'user-alias'; link.symlink_to(outside, target_is_directory=True)
        with pytest.raises(InvalidEvidenceError):
            write_final_artifacts(link/'run', {'report.json': {}}, required_paths=['report.json'], identity={})
        assert not list(outside.iterdir())


@pytest.mark.parametrize('limit', ['artifact','bundle'])
def test_phase5_publication_preflights_limits_before_changing_existing_bundle(tmp_path, monkeypatch, limit):
    from capability_anatomy.execution import evidence_limits, phase5_artifacts
    from capability_anatomy.execution.phase5_campaign import write_final_artifacts
    original = write_final_artifacts(tmp_path, {'report.json': {'ok':True}},
                                     required_paths=['report.json','evidence-manifest.json'], identity={})
    reconstruct_manifest(tmp_path, original, ['report.json'])
    before = {path.name:path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    if limit == 'artifact':
        monkeypatch.setattr(evidence_limits, 'MAX_ARTIFACT_BYTES', 20)
        artifacts = {'report.json': {'x':'a'*30}}
    else:
        monkeypatch.setattr(evidence_limits, 'MAX_BUNDLE_BYTES', 30)
        monkeypatch.setattr(phase5_artifacts, 'MAX_BUNDLE_BYTES', 30)
        artifacts = {'report.json': {'x':'a'*15}, 'second.json': {'x':'b'*15}}
    with pytest.raises(InvalidEvidenceError):
        write_final_artifacts(tmp_path, artifacts, required_paths=['report.json','evidence-manifest.json'], identity={})
    assert before == {path.name:path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}


def test_direct_phase5_writer_owns_the_run_before_preflight(tmp_path, monkeypatch):
    import subprocess
    import sys
    from capability_anatomy.execution import phase5_campaign
    original = phase5_campaign._encode_artifact
    outcomes = []
    def observe(relative, value):
        code = ('from pathlib import Path\nfrom capability_anatomy.execution.secure_fs import run_ownership, RunBusyError\n'
                'import sys\ntry:\n with run_ownership(Path(sys.argv[1])): print("unexpectedly acquired")\n'
                'except RunBusyError: print("busy")\n')
        contender = subprocess.run([sys.executable,'-c',code,str(tmp_path)], capture_output=True,text=True,timeout=5)
        assert contender.returncode == 0, contender.stderr
        outcomes.append(contender.stdout.strip())
        return original(relative, value)
    monkeypatch.setattr(phase5_campaign, '_encode_artifact', observe)
    phase5_campaign.write_final_artifacts(tmp_path, {'report.json':{}}, required_paths=['report.json'], identity={})
    assert outcomes == ['busy']
