"""Adversarial checks at actual OS read, write and snapshot boundaries."""
from contextvars import copy_context
from pathlib import Path
from threading import Thread
import json
import os
import subprocess
import sys
import tempfile
import time

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.execution import secure_fs, core_evidence
from capability_anatomy.telemetry import telemetry_scope


def test_parent_traversal_refuses_at_actual_write_and_regular_child_works(tmp_path):
    output=tmp_path/'run'
    with secure_fs.run_ownership(output):
        secure_fs.atomic_write(output/'control.json',b'working')
        with pytest.raises(secure_fs.StorageError) as error:
            secure_fs.atomic_write(output/'..'/'escaped.json',b'escape')
        assert error.value.reason=='storage_parent_traversal'
    assert (output/'control.json').read_bytes()==b'working'
    assert not (tmp_path/'escaped.json').exists()


def test_leaf_nofollow_refuses_actual_external_bytes_and_preserves_signal(tmp_path):
    target=tmp_path/'external';target.write_bytes(b'FAKE-PRIVATE-READ-CANARY')
    output=tmp_path/'run';output.mkdir();(output/'link').symlink_to(target)
    exporter=InMemorySpanExporter();tp=TracerProvider();tp.add_span_processor(SimpleSpanProcessor(exporter))
    reader=InMemoryMetricReader();mp=MeterProvider(metric_readers=[reader])
    try:
        with telemetry_scope(tp.get_tracer('storage-review'),mp.get_meter('storage-review')):
            with secure_fs.run_ownership(output):
                secure_fs.atomic_write(output/'control',b'working')
                assert secure_fs.read_bytes(output/'control')==b'working'
                with pytest.raises(secure_fs.StorageError) as error:secure_fs.read_bytes(output/'link')
                assert error.value.reason=='storage_symlink_refused'
        refusal=[span for span in exporter.get_finished_spans() if span.name=='capability_anatomy.storage.read' and span.status.status_code.name=='ERROR']
        assert len(refusal)==1
        assert refusal[0].attributes['capability_anatomy.reason']=='storage_symlink_refused'
        points=[p for r in reader.get_metrics_data().resource_metrics for s in r.scope_metrics for m in s.metrics for p in m.data.data_points]
        assert sum(p.value for p in points if p.attributes.get('capability_anatomy.reason')=='storage_symlink_refused')==1
        assert 'FAKE-PRIVATE-READ-CANARY' not in repr(exporter.get_finished_spans())
    finally:tp.shutdown();mp.shutdown()


@pytest.mark.parametrize('kind',['regular','symlink'])
def test_temporary_leaf_exclusive_create_preserves_preexisting_path(tmp_path,monkeypatch,kind):
    output=tmp_path/'run';outside=tmp_path/'outside';outside.write_bytes(b'preserve outside')
    with secure_fs.run_ownership(output):
        temporary=output/'.payload.json.fixed.tmp'
        if kind=='regular':temporary.write_bytes(b'preserve prior temporary file')
        else:temporary.symlink_to(outside)
        class Fixed:
            hex='fixed'
        monkeypatch.setattr(secure_fs.uuid,'uuid4',lambda:Fixed())
        with pytest.raises(secure_fs.StorageError):secure_fs.atomic_write(output/'payload.json',b'new data')
        assert not (output/'payload.json').exists()
        assert temporary.is_symlink() if kind=='symlink' else temporary.read_bytes()==b'preserve prior temporary file'
    assert outside.read_bytes()==b'preserve outside'


def test_worker_requires_explicit_owning_context(tmp_path):
    output=tmp_path/'run';failures=[]
    with secure_fs.run_ownership(output):
        def unscoped():
            try:secure_fs.atomic_write(tmp_path/'escaped.json',b'outside')
            except secure_fs.StorageError as error:failures.append(error.reason)
        worker=Thread(target=unscoped);worker.start();worker.join(3)
        assert not worker.is_alive() and failures==['storage_worker_context_required']
        context=copy_context();worker=Thread(target=lambda:context.run(secure_fs.atomic_write,output/'inside.json',b'working'));worker.start();worker.join(3)
        assert not worker.is_alive() and secure_fs.read_bytes(output/'inside.json')==b'working'
    assert not (tmp_path/'escaped.json').exists()


def test_failed_owned_run_has_release_event_without_completed_metric(tmp_path):
    exporter=InMemorySpanExporter();tp=TracerProvider();tp.add_span_processor(SimpleSpanProcessor(exporter));reader=InMemoryMetricReader();mp=MeterProvider(metric_readers=[reader])
    try:
        with telemetry_scope(tp.get_tracer('storage-review'),mp.get_meter('storage-review')):
            with pytest.raises(RuntimeError):
                with secure_fs.run_ownership(tmp_path/'run'):raise RuntimeError('FAKE-FAILURE')
        owner,=[span for span in exporter.get_finished_spans() if span.name=='capability_anatomy.storage.ownership']
        assert owner.status.status_code.name=='ERROR' and owner.attributes['capability_anatomy.reason']=='owned_run_failed'
        assert any(event.attributes.get('capability_anatomy.reason')=='run_ownership_released' for event in owner.events)
        points=[p for r in reader.get_metrics_data().resource_metrics for s in r.scope_metrics for m in s.metrics for p in m.data.data_points if p.attributes.get('capability_anatomy.operation')=='ownership']
        assert {p.attributes['capability_anatomy.outcome']:p.value for p in points}=={'accepted':1,'failed':1}
    finally:tp.shutdown();mp.shutdown()


def test_default_temp_alias_works_but_user_alias_still_refuses(tmp_path):
    with tempfile.TemporaryDirectory(prefix='ca-review-alias-',dir='/tmp') as location:
        output=Path(location)/'run'
        with secure_fs.run_ownership(output):secure_fs.atomic_write(output/'control',b'working')
        assert secure_fs.read_bytes(output/'control')==b'working'
        outside=tmp_path/'outside';outside.mkdir()
        link=Path(location)/'user-link';link.symlink_to(outside,target_is_directory=True)
        with pytest.raises(secure_fs.StorageError):
            with secure_fs.run_ownership(link/'run'):pass
        assert not list(outside.iterdir())


def test_target_volume_probe_cleans_owned_files_and_refuses_before_execution(tmp_path,monkeypatch):
    output=tmp_path/'run'
    assert secure_fs.probe_storage_volume(output)=={'directory_lock':'supported','hardlink_publication':'supported','directory_sync':'supported'}
    assert not list(output.iterdir())
    import errno
    def no_links(*args,**kwargs):raise OSError(errno.EOPNOTSUPP,'unsupported')
    monkeypatch.setattr(secure_fs.os,'link',no_links)
    monkeypatch.setattr(secure_fs.os,'supports_dir_fd',secure_fs.os.supports_dir_fd|{no_links})
    entered=[]
    with pytest.raises(secure_fs.StorageFilesystemUnsupported):
        with secure_fs.run_ownership(output):entered.append('execution')
    assert entered==[] and not list(output.iterdir())


def test_inventory_stops_scandir_at_bound_without_materializing_directory(tmp_path,monkeypatch):
    from contextlib import contextmanager
    for number in range(4):(tmp_path/f'{number}.json').write_bytes(b'{}')
    monkeypatch.setattr(core_evidence,'MAX_ARTIFACTS',5)
    assert len(core_evidence._inventory(tmp_path))==4
    for number in range(4,12):(tmp_path/f'{number}.json').write_bytes(b'{}')
    original=os.scandir;seen=[]
    @contextmanager
    def scan(descriptor):
        with original(descriptor) as entries:
            def bounded():
                for entry in entries:seen.append(entry.name);yield entry
            yield bounded()
    monkeypatch.setattr(core_evidence.os,'scandir',scan)
    monkeypatch.setattr(core_evidence.os,'listdir',lambda *args:pytest.fail('full listdir materialized'))
    with pytest.raises(core_evidence.CoreEvidenceError) as error:core_evidence._inventory(tmp_path)
    assert error.value.reason=='evidence_entry_limit' and len(seen)==6


def test_verify_refuses_busy_writer_at_partial_publication_then_succeeds(tmp_path):
    from capability_anatomy import cli
    example=tmp_path/'example';cli._example(example)
    initial=subprocess.run([sys.executable,'-m','capability_anatomy.cli','run','--config',str(example/'experiment.json')],capture_output=True,text=True,timeout=20)
    assert initial.returncode==0,initial.stderr
    output=example/'run';ready=tmp_path/'ready';release=tmp_path/'release'
    program='''
import json,sys,time
from pathlib import Path
from capability_anatomy.execution import core_evidence
output,ready,release=map(Path,sys.argv[1:]);trace=json.loads((output/'trace.json').read_bytes());trace['snapshot_probe']='updated'
original=core_evidence.secure_fs.atomic_write
def pause(path,payload):
 original(path,payload)
 if path.name=='trace.json':
  ready.touch();deadline=time.monotonic()+5
  while not release.exists():
   if time.monotonic()>deadline:raise RuntimeError('probe barrier timed out')
   time.sleep(.005)
core_evidence.secure_fs.atomic_write=pause
core_evidence.finalize_core_evidence(output,trace)
'''
    child=subprocess.Popen([sys.executable,'-c',program,str(output),str(ready),str(release)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+5
        while not ready.exists():
            assert child.poll() is None and time.monotonic()<deadline
            time.sleep(.005)
        with pytest.raises(secure_fs.RunBusyError):core_evidence.verify_core_evidence(output)
        release.touch();stdout,stderr=child.communicate(timeout=5)
        assert child.returncode==0,stdout+stderr
        before={p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()}
        assert core_evidence.verify_core_evidence(output)['status']=='complete'
        assert {p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()}==before
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_read_snapshot_is_shared_and_disallows_writes(tmp_path):
    root=tmp_path/'run'
    secure_fs.ensure_directory(root)
    before=list(root.iterdir())
    with secure_fs.read_ownership(root):
        with pytest.raises(secure_fs.StorageError) as error:secure_fs.atomic_write(root/'forbidden',b'write')
        assert error.value.reason=='storage_readonly_context'
        result=subprocess.run([sys.executable,'-c','from pathlib import Path; import sys; from capability_anatomy.execution.secure_fs import read_ownership\nwith read_ownership(Path(sys.argv[1])): print("shared")',str(root)],capture_output=True,text=True,timeout=3)
        assert result.returncode==0 and result.stdout.strip()=='shared'
    assert list(root.iterdir())==before


@pytest.mark.parametrize('method',['create_once','atomic_write'])
def test_real_flush_failure_removes_owned_partial_temporary_file(tmp_path,monkeypatch,method):
    output=tmp_path/'run'
    with secure_fs.run_ownership(output):
        original=secure_fs.os.fsync
        def fail_flush(descriptor):
            raise OSError(5,'injected disk write failure')
        with monkeypatch.context() as fault:
            fault.setattr(secure_fs.os,'fsync',fail_flush)
            with pytest.raises(secure_fs.StorageError) as error:
                getattr(secure_fs,method)(output/'payload',b'complete')
            assert error.value.reason=='storage_io_failed'
        assert list(output.iterdir())==[]
        getattr(secure_fs,method)(output/'payload',b'complete')
        assert secure_fs.read_bytes(output/'payload')==b'complete'
