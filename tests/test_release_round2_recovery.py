"""Exercise interruption at the OS publication boundary, not random delays."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys

import pytest

from capability_anatomy.execution import secure_fs, core_evidence
from capability_anatomy.errors import InvalidEvidenceError
from test_release_cli import _cli, _all_bytes, bundle


@pytest.mark.parametrize('relative', ['task-traces/baseline.json', 'tasks/baseline.json', 'execution-state.json'])
def test_kill_inside_atomic_publication_resumes_twice(tmp_path, relative):
    example = tmp_path / 'example'
    assert _cli('example', '--output', example).returncode == 0
    script = r'''
import os, signal, sys
from pathlib import Path
from capability_anatomy.execution import secure_fs
from capability_anatomy import cli
original = secure_fs.os.replace
suffix = sys.argv[2]
def crash(source, target, **kwargs):
    # Resolve only our OS descriptor for targeting the test window. The actual
    # publisher has already written/fsynced its staging bytes at this point.
    if target == Path(suffix).name:
        fd = kwargs['dst_dir_fd']
        expected = Path(sys.argv[1]).parent / 'run' / Path(suffix).parent
        if not expected.exists(): return original(source, target, **kwargs)
        st = os.fstat(fd); wanted = expected.stat()
        if (st.st_dev, st.st_ino) == (wanted.st_dev, wanted.st_ino):
            os.kill(os.getpid(), signal.SIGKILL)
    return original(source, target, **kwargs)
secure_fs.os.replace = crash
raise SystemExit(cli.main(['run', '--config', sys.argv[1]]))
'''
    child = subprocess.run([sys.executable, '-c', script, str(example/'experiment.json'), relative],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == -signal.SIGKILL, child.stderr
    root = example/'run'
    leftovers = [p for p in root.rglob('*') if secure_fs.temporary_target(p.name)]
    assert len(leftovers) == 1 and leftovers[0].name.startswith('.'+Path(relative).name+'.')
    for _ in range(2):
        result = _cli('run', '--config', example/'experiment.json')
        assert result.returncode == 0, result.stderr
        assert _cli('verify', '--evidence', root).returncode == 0
        assert not [p for p in root.rglob('*') if secure_fs.temporary_target(p.name)]
    manifest = json.loads((root/'core-manifest.json').read_bytes())
    assert not any('.tmp' in entry['path'] for entry in manifest['artifacts'])


def test_readonly_excludes_temporary_bytes_and_resume_reclaims_with_signal(bundle):
    for relative in ('task-traces/.baseline.json.'+'a'*32+'.tmp',
                     'tasks/.baseline.json.'+'b'*32+'.tmp',
                     '.trace.json.'+'c'*32+'.tmp'):
        (bundle/relative).write_bytes(b'incomplete JSON publication')
    before = _all_bytes(bundle)
    assert _cli('verify', '--evidence', bundle).returncode == 0
    assert _cli('report', '--evidence', bundle).returncode == 0
    assert _all_bytes(bundle) == before
    result = _cli('run', '--config', bundle.parent/'experiment.json')
    assert result.returncode == 0, result.stderr
    receipt = json.loads(Path(json.loads(result.stdout)['invocation_trace']).read_bytes())
    recovered = [s for s in receipt['spans'] if s['name'] == 'capability_anatomy.storage.recover_temporary']
    assert len(recovered) == 3
    assert {s['attributes']['capability_anatomy.reason'] for s in recovered} == {'storage_temporary_reclaimed'}
    assert {s['attributes']['capability_anatomy.outcome'] for s in recovered} == {'completed'}
    assert all(s['end_time_unix_nano'] >= s['start_time_unix_nano'] for s in recovered)
    assert not [p for p in bundle.rglob('*') if secure_fs.temporary_target(p.name)]
    assert {k:v for k,v in _all_bytes(bundle).items() if k.startswith('tasks/')} == {
        k:v for k,v in before.items() if k.startswith('tasks/') and not Path(k).name.endswith('.tmp')}


@pytest.mark.parametrize('kind', ['symlink', 'fifo', 'directory', 'unrelated-hardlink'])
def test_recovery_refuses_unsafe_temporary_without_touching_target(tmp_path, kind):
    root = tmp_path/'run'; root.mkdir()
    victim = tmp_path/'victim'; victim.write_bytes(b'preserve')
    temporary = root/('.trace.json.'+'d'*32+'.tmp')
    if kind == 'symlink': temporary.symlink_to(victim)
    elif kind == 'fifo': os.mkfifo(temporary)
    elif kind == 'directory': temporary.mkdir()
    else: os.link(victim, temporary)
    with pytest.raises(secure_fs.StorageError):
        with secure_fs.run_ownership(root):
            pytest.fail('unsafe temporary admitted')
    assert os.path.lexists(temporary)
    assert victim.read_bytes() == b'preserve'


def test_create_once_crash_after_link_preserves_published_inode(tmp_path):
    root=tmp_path/'run'
    child=subprocess.run([sys.executable, '-c', '''
import os, signal, sys
from pathlib import Path
from capability_anatomy.execution import secure_fs
original=secure_fs.os.link
def crash(source,target,**kwargs):
    original(source,target,**kwargs)
    if target=='frozen.json': os.kill(os.getpid(),signal.SIGKILL)
secure_fs.os.link=crash
os.supports_dir_fd = os.supports_dir_fd | {crash}
with secure_fs.run_ownership(Path(sys.argv[1])):
    secure_fs.create_once(Path(sys.argv[1])/'frozen.json',b'{"frozen":true}')
''',str(root)],capture_output=True,text=True,timeout=10)
    assert child.returncode == -signal.SIGKILL, child.stderr
    published=root/'frozen.json'; identity=published.stat().st_ino
    assert published.stat().st_nlink==2
    with secure_fs.read_ownership(root):
        with pytest.raises(secure_fs.StorageError) as caught:
            core_evidence._inventory(root)
        assert caught.value.reason=="storage_recovery_required"
    with secure_fs.run_ownership(root):
        assert secure_fs.read_bytes(published)==b'{"frozen":true}'
        assert published.stat().st_ino==identity and published.stat().st_nlink==1
        assert [p.name for p in root.iterdir()]==['frozen.json']


def test_recovery_is_bounded_and_does_not_discard_arbitrary_dotfiles(tmp_path, monkeypatch):
    root=tmp_path/'run'; root.mkdir()
    (root/'.notes.tmp').write_bytes(b'operator data')
    with secure_fs.run_ownership(root):
        assert '.notes.tmp' in core_evidence._inventory(root)
    from capability_anatomy.execution import evidence_limits
    monkeypatch.setattr(evidence_limits, 'MAX_ARTIFACTS', 2)
    for n in range(3): (root/f'.trace.json.{n:032x}.tmp').write_bytes(b'partial')
    with pytest.raises(InvalidEvidenceError) as caught:
        with secure_fs.run_ownership(root): pass
    assert caught.value.reason=='evidence_entry_limit'
    assert (root/'.notes.tmp').read_bytes()==b'operator data'


@pytest.mark.parametrize('window', ['probe_write', 'probe_link'])
def test_probe_interruption_is_recovered_before_new_probe(tmp_path, window):
    root=tmp_path/'run'
    child=subprocess.run([sys.executable, '-c', r'''
import os, signal, sys
from pathlib import Path
from capability_anatomy.execution import secure_fs
if sys.argv[2]=='probe_write':
    original=secure_fs._write_at
    def crash(parent,name,payload):
        original(parent,name,payload)
        if name.startswith('.capability-anatomy-probe-'): os.kill(os.getpid(),signal.SIGKILL)
    secure_fs._write_at=crash
else:
    original=secure_fs.os.link
    def crash(source,target,**kwargs):
        original(source,target,**kwargs)
        if str(source).startswith('.capability-anatomy-probe-'): os.kill(os.getpid(),signal.SIGKILL)
    secure_fs.os.link=crash
    os.supports_dir_fd=os.supports_dir_fd|{crash}
with secure_fs.run_ownership(Path(sys.argv[1])): pass
''', str(root),window],capture_output=True,text=True,timeout=10)
    assert child.returncode==-signal.SIGKILL, child.stderr
    assert len(list(root.iterdir()))==(1 if window=='probe_write' else 2)
    with secure_fs.read_ownership(root):
        assert core_evidence._inventory(root)==[]
    with secure_fs.run_ownership(root):
        assert list(root.iterdir())==[]
