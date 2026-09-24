from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from capability_anatomy.credentials import CredentialReference, RuntimeCredentials, credential_scope
from capability_anatomy.errors import InvalidConfigurationError, InvalidEvidenceError
from capability_anatomy.execution.orchestrator import run_experiment
from capability_anatomy.execution.secure_fs import run_ownership
from capability_anatomy.execution.runner import ExperimentRunner, Task
from capability_anatomy.execution.store import FrozenRunStore
from capability_anatomy.serialization import canonical_json_bytes


def config(tmp_path, output=None):
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / 'configs/examples/synthetic-scan.yaml').read_text())
    value['dataset']['manifest'] = str(root / 'fixtures/synthetic/manifest.json')
    value['output']['directory'] = str(output or tmp_path / 'run')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(value))
    return path


def test_public_contender_refuses_before_actual_model_load(tmp_path):
    path = config(tmp_path)
    sentinel = tmp_path / 'loaded'
    code = '''from pathlib import Path
from capability_anatomy.models.plugins.synthetic import SyntheticModelAdapter
from capability_anatomy.cli import main
import sys
original=SyntheticModelAdapter.load
def load(*args,**kwargs):
 Path(sys.argv[2]).write_text('loaded')
 return original(*args,**kwargs)
SyntheticModelAdapter.load=load
raise SystemExit(main(['run','--config',sys.argv[1]]))
'''
    with run_ownership(tmp_path / 'run'):
        result = subprocess.run([sys.executable, '-c', code, str(path), str(sentinel)],capture_output=True,text=True,timeout=30)
    assert result.returncode == 2, result.stderr
    assert not sentinel.exists()
    assert not list((tmp_path / 'run').iterdir())


@pytest.mark.parametrize('relative', [False, True])
def test_public_output_symlink_refuses_before_load(tmp_path, relative, monkeypatch):
    target = tmp_path / 'outside'; target.mkdir()
    link = tmp_path / 'link'; link.symlink_to(target, target_is_directory=True)
    path = config(tmp_path, 'link' if relative else link)
    called=[]
    monkeypatch.setattr('capability_anatomy.models.plugins.synthetic.SyntheticModelAdapter.load',lambda *a,**k: called.append(True))
    with pytest.raises(InvalidEvidenceError): run_experiment(path)
    assert called == []
    assert list(target.iterdir()) == []


def test_runtime_plaintext_cannot_return_to_evidence_and_scope_restores(monkeypatch):
    monkeypatch.setenv('CA_RELEASE_SECRET','test-credential-canary-unique')
    with credential_scope():
        provider=RuntimeCredentials({'service':CredentialReference('env','CA_RELEASE_SECRET')})
        assert canonical_json_bytes({'ordinary':'value'}) == b'{"ordinary":"value"}'
        value=provider.get('service').reveal()
        with pytest.raises(InvalidConfigurationError,match='resolved credential reached'):
            canonical_json_bytes({'plugin_output': 'echo '+value})
        with pytest.raises(InvalidConfigurationError): canonical_json_bytes({value:'key'})
    assert canonical_json_bytes({'ordinary':'value'}) == b'{"ordinary":"value"}'


def test_unknown_memory_cannot_pass_an_enforced_budget(tmp_path):
    called=[]
    runner=ExperimentRunner(FrozenRunStore(tmp_path/'run',{},{}),max_memory_observation_bytes=1024,memory_bytes=lambda:None)
    with pytest.raises(InvalidEvidenceError,match='cannot be enforced'):
        runner.run((Task('baseline','baseline',lambda:called.append(True)),))
    assert called == []
    assert json.loads((tmp_path/'run/execution-state.json').read_text())['reason']=='memory_budget_unobservable'


def test_sigterm_records_interruption_and_releases_owned_run(tmp_path):
    import signal
    import time
    path = config(tmp_path)
    sentinel = tmp_path / 'executing'
    code = """from pathlib import Path
import sys,time
from capability_anatomy.models.plugins.synthetic import SyntheticModelAdapter
from capability_anatomy.cli import main
original=SyntheticModelAdapter.execute
def execute(*args,**kwargs):
 Path(sys.argv[2]).write_text('executing')
 time.sleep(30)
 return original(*args,**kwargs)
SyntheticModelAdapter.execute=execute
raise SystemExit(main(['run','--config',sys.argv[1]]))
"""
    process=subprocess.Popen([sys.executable,'-c',code,str(path),str(sentinel)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+20
        while not sentinel.exists() and time.monotonic()<deadline and process.poll() is None:
            time.sleep(.02)
        assert sentinel.exists()
        process.send_signal(signal.SIGTERM)
        stdout,stderr=process.communicate(timeout=10)
        assert process.returncode == 4, (stdout,stderr)
        state=json.loads((tmp_path/'run/execution-state.json').read_text())
        assert state['state']=='interrupted'
        assert not (tmp_path/'run/tasks/baseline.complete.json').exists()
        with run_ownership(tmp_path/'run'):
            pass
        assert run_experiment(path)['status']=='complete'
    finally:
        if process.poll() is None:
            process.kill();process.wait()


def test_complete_local_receipt_closes_ownership_and_publication(tmp_path):
    path=config(tmp_path)
    result=run_experiment(path)
    receipt_path,=list((tmp_path/'run.invocations').glob('*/trace.json'))
    receipt=json.loads(receipt_path.read_text())
    spans=receipt['spans']
    reasons={event['attributes'].get('capability_anatomy.reason') for span in spans for event in span['events']}
    assert {'run_ownership_acquired','run_ownership_released','core_evidence_consistent'} <= reasons
    by_id={span['span_id']:span for span in spans}
    assert all(span['parent_span_id'] is None or span['parent_span_id'] in by_id for span in spans)
    assert all(span['end_time_unix_nano'] >= span['start_time_unix_nano'] for span in spans)
    assert result['trace_id'] in {span['trace_id'] for span in spans}
    assert any('capability_anatomy.record_id_sha256' in span['attributes'] for span in spans)
    assert any('capability_anatomy.task_id_sha256' in span['attributes'] for span in spans)


def test_numeric_credential_echo_is_refused_by_actual_public_run(tmp_path, monkeypatch):
    from dataclasses import replace
    from capability_anatomy.models.plugins.synthetic import SyntheticModelAdapter
    secret='98765432101234567890'
    monkeypatch.setenv('CA_RELEASE_SECRET',secret)
    path=config(tmp_path)
    value=yaml.safe_load(path.read_text())
    value['credentials']={'service':{'provider':'env','name':'CA_RELEASE_SECRET'}}
    path.write_text(yaml.safe_dump(value))
    original=SyntheticModelAdapter.provenance
    monkeypatch.setattr(SyntheticModelAdapter,'configure_credentials',lambda self,provider:setattr(self,'credentials',provider),raising=False)
    marker=[7]
    def provenance(self,loaded):
        self.credentials.get('service').reveal()
        return replace(original(self,loaded),implementation_metadata={'request_marker':marker[0]})
    monkeypatch.setattr(SyntheticModelAdapter,'provenance',provenance)
    assert run_experiment(path)['status']=='complete'
    marker[0]=int(secret)
    value['output']['directory']=str(tmp_path/'denied')
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(InvalidConfigurationError,match='resolved credential reached'):
        run_experiment(path)
    for directory in (tmp_path/'denied',tmp_path/'denied.invocations'):
        for artifact in directory.rglob('*'):
            if artifact.is_file(): assert secret.encode() not in artifact.read_bytes()


def test_direct_runner_holds_ownership_while_task_executes(tmp_path):
    import time
    code="""import sys,time
from pathlib import Path
from capability_anatomy.execution.runner import ExperimentRunner,Task
from capability_anatomy.execution.store import FrozenRunStore
from capability_anatomy.errors import CapabilityAnatomyError
root=Path(sys.argv[1]);sentinel=Path(sys.argv[2])
def work():
 sentinel.write_text('executed')
 time.sleep(float(sys.argv[3]))
 return {'complete':True,'owner':sentinel.name}
try: ExperimentRunner(FrozenRunStore(root,{},{})).run((Task('baseline','baseline',work),))
except CapabilityAnatomyError as error: raise SystemExit(int(error.exit_code))
"""
    first=subprocess.Popen([sys.executable,'-c',code,str(tmp_path/'run'),str(tmp_path/'first'),'2'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+15
        while not (tmp_path/'first').exists() and time.monotonic()<deadline and first.poll() is None:
            time.sleep(.02)
        assert (tmp_path/'first').exists()
        second=subprocess.run([sys.executable,'-c',code,str(tmp_path/'run'),str(tmp_path/'second'),'0'],capture_output=True,timeout=10)
        assert second.returncode==2, second.stderr
        assert not (tmp_path/'second').exists()
        first.communicate(timeout=10)
        assert first.returncode==0
        payload=json.loads((tmp_path/'run/tasks/baseline.json').read_text())
        assert payload['owner']=='first'
    finally:
        if first.poll() is None: first.kill();first.wait()


def test_final_publication_failure_has_failed_receipt_without_false_validation(tmp_path,monkeypatch):
    from capability_anatomy.execution import secure_fs
    path=config(tmp_path)
    assert run_experiment(path)['status']=='complete'
    original=secure_fs.atomic_write
    def fail_manifest(path,payload):
        if path.name=='core-manifest.json': raise OSError('PRIVATE_PUBLICATION_FAILURE')
        return original(path,payload)
    monkeypatch.setattr(secure_fs,'atomic_write',fail_manifest)
    path=config(tmp_path,tmp_path/'failed')
    with pytest.raises(InvalidEvidenceError): run_experiment(path)
    receipt_path,=list((tmp_path/'failed.invocations').glob('*/trace.json'))
    receipt=json.loads(receipt_path.read_text())
    assert receipt['status']=='failed'
    root, = [span for span in receipt['spans'] if span['name'] == 'capability_anatomy.invocation']
    assert root['status'] == 'ERROR'
    reasons={event['attributes'].get('capability_anatomy.reason') for span in receipt['spans'] for event in span['events']}
    assert {'invocation_failed','owned_run_failed','run_ownership_released','storage_io_failed'} <= reasons
    assert not {'scan_evidence_validated','evidence_publication_complete'} & reasons
    assert 'PRIVATE_PUBLICATION_FAILURE' not in receipt_path.read_text()
    assert not (tmp_path/'failed/core-manifest.json').exists()


@pytest.mark.parametrize('wrapper',['path','enum'])
def test_converted_multiline_credentials_cannot_reach_actual_store(tmp_path,monkeypatch,wrapper):
    from enum import Enum
    secret='FAKE_MULTILINE_CREDENTIAL\nPRIVATE_SECOND_LINE'
    monkeypatch.setenv('CA_RELEASE_SECRET',secret)
    with credential_scope():
        RuntimeCredentials({'service':CredentialReference('env','CA_RELEASE_SECRET')}).get('service')
        def wrap(value):
            return Path(value) if wrapper=='path' else Enum('CredentialCarrier',{'VALUE':value}).VALUE
        store=FrozenRunStore(tmp_path/'run',{},{});runner=ExperimentRunner(store)
        assert runner.run((Task('baseline','baseline',lambda:{'value':wrap('ordinary')}),))
        other=ExperimentRunner(FrozenRunStore(tmp_path/'denied',{},{}))
        with pytest.raises(InvalidConfigurationError,match='resolved credential reached'):
            other.run((Task('baseline','baseline',lambda:{'value':wrap(secret)}),))
        assert not (tmp_path/'denied/tasks/baseline.complete.json').exists()
        for path in (tmp_path/'denied').rglob('*.json'):
            assert secret not in json.dumps(json.loads(path.read_text()),ensure_ascii=False).replace('\\n','\n')


def test_forked_worker_cannot_retain_or_reuse_parent_lease(tmp_path):
    code="""import os,sys,time,subprocess,signal
from pathlib import Path
from capability_anatomy.execution.secure_fs import run_ownership
from capability_anatomy.execution.store import FrozenRunStore
from capability_anatomy.errors import InvalidEvidenceError
root=Path(sys.argv[1]);r,w=os.pipe();child=-1
try:
 with run_ownership(root):
  child=os.fork()
  if child==0:
   os.close(w);os.read(r,1)
   try: FrozenRunStore(root,{},{}).commit('child',{'complete':True})
   except InvalidEvidenceError: (root.parent/'child-result').write_text('refused')
   else: (root.parent/'child-result').write_text('UNSAFE')
   time.sleep(5);os._exit(0)
 os.close(r);os.write(w,b'x');os.close(w)
 deadline=time.monotonic()+10
 while not (root.parent/'child-result').exists() and time.monotonic()<deadline: time.sleep(.02)
 assert (root.parent/'child-result').read_text()=='refused'
 probe=subprocess.run([sys.executable,'-c','from pathlib import Path; import sys; from capability_anatomy.execution.secure_fs import run_ownership; scope=run_ownership(Path(sys.argv[1]));scope.__enter__();scope.__exit__(None,None,None)',str(root)],capture_output=True,timeout=3)
 assert probe.returncode==0,probe.stderr
 assert not (root/'tasks/child.json').exists()
finally:
 if child>0:
  waited,_=os.waitpid(child,os.WNOHANG)
  if not waited: os.kill(child,signal.SIGTERM);os.waitpid(child,0)
"""
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path/'run')],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr


def test_worker_thread_resolution_is_bound_to_invocation_and_redacted(tmp_path,monkeypatch):
    from dataclasses import replace
    from concurrent.futures import ThreadPoolExecutor
    from capability_anatomy.models.plugins.synthetic import SyntheticModelAdapter
    secret='FAKE_THREADED_CREDENTIAL_CANARY'
    monkeypatch.setenv('CA_RELEASE_SECRET',secret)
    path=config(tmp_path)
    value=yaml.safe_load(path.read_text());value['credentials']={'service':{'provider':'env','name':'CA_RELEASE_SECRET'}}
    path.write_text(yaml.safe_dump(value))
    original=SyntheticModelAdapter.provenance
    monkeypatch.setattr(SyntheticModelAdapter,'configure_credentials',lambda self,provider:setattr(self,'credentials',provider),raising=False)
    echo=[False]
    def provenance(self,loaded):
        with ThreadPoolExecutor(max_workers=1) as executor:
            resolved=executor.submit(lambda:self.credentials.get('service').reveal()).result()
        return replace(original(self,loaded),implementation_metadata={'worker':resolved if echo[0] else 'ordinary'})
    monkeypatch.setattr(SyntheticModelAdapter,'provenance',provenance)
    result=run_experiment(path)
    trace=json.loads((tmp_path/'run/trace.json').read_text())
    resolution,=[span for span in trace['spans'] if span['name']=='capability_anatomy.credentials.resolve']
    assert resolution['trace_id']==result['trace_id']
    assert resolution['parent_span_id'] in {span['span_id'] for span in trace['spans']}
    echo[0]=True;value['output']['directory']=str(tmp_path/'denied');path.write_text(yaml.safe_dump(value))
    with pytest.raises(InvalidConfigurationError,match='resolved credential reached'):
        run_experiment(path)
    for directory in (tmp_path/'denied',tmp_path/'denied.invocations'):
        for artifact in directory.rglob('*'):
            if artifact.is_file():assert secret.encode() not in artifact.read_bytes()
    receipt_path,=list((tmp_path/'denied.invocations').glob('*/trace.json'))
    receipt=json.loads(receipt_path.read_text())
    assert any(event['attributes'].get('capability_anatomy.reason')=='resolved_credential_persistence_refused' for span in receipt['spans'] for event in span['events'])
