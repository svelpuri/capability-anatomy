"""Exercise the ordinary builder with private-file canaries beside public source."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import uuid

ROOT=Path(__file__).resolve().parents[1]


def test_direct_sdist_enforces_the_public_allowlist(tmp_path):
    spec=importlib.util.spec_from_file_location('sdist_export_fixture',ROOT/'scripts/export_standalone.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    source=tmp_path/'source';manifest=exporter.export_snapshot(ROOT,source,allow_unlicensed_preview=True)
    canary='private-material-'+uuid.uuid4().hex
    for name in ['notes/positioning.md','data/phase0b/private.json','src/qca_phase0/private.py','mlx-lora/private.txt','config/private.yaml']:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(canary)
    result=subprocess.run(['uv','build','--sdist','--out-dir',str(tmp_path/'dist')],cwd=source,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    allowed={item['path'] for item in manifest['files']}|{'PKG-INFO','EXPORT-MANIFEST.json'}
    with tarfile.open(next((tmp_path/'dist').glob('*.tar.gz'))) as archive:
        files={item.name.split('/',1)[1]:archive.extractfile(item).read() for item in archive.getmembers() if item.isfile()}
    assert set(files)==allowed
    assert all(canary.encode() not in payload for payload in files.values())


def test_private_names_are_excluded_by_direct_sdist_even_inside_fixture_tree(tmp_path):
    spec=importlib.util.spec_from_file_location('sdist_private_fixture',ROOT/'scripts/export_standalone.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    source=tmp_path/'source';exporter.export_snapshot(ROOT,source,allow_unlicensed_preview=True)
    canary='private-'+uuid.uuid4().hex
    for name in [".git", ".env", "AGENTS.md", "CLAUDE.md", "CONTEXT.md", ".session-sync.md", "__pycache__"]:
        path=source/'fixtures/gate5m_external_plugin'/name/'nested.py'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(canary)
    result=subprocess.run(['uv','build','--sdist','--out-dir',str(tmp_path/'dist')],cwd=source,capture_output=True,text=True)
    assert result.returncode != 0
    assert 'forbidden export input name' in result.stderr
    assert not list((tmp_path/'dist').glob('*.tar.gz'))
