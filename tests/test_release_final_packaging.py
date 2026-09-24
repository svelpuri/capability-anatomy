"""Real direct builds enforce the same public boundary as snapshot builds."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('final_export', ROOT / 'scripts/export_standalone.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture
def uncurated(tmp_path):
    # Copy the actual uncurated source, never the output of the exporter.
    source = tmp_path / 'source'
    shutil.copytree(ROOT, source, symlinks=True,
                    ignore=shutil.ignore_patterns('.venv', '__pycache__', '.pytest_cache', 'dist', 'runs'))
    for name in ('.gitignore', 'private-unselected.txt', 'src/capability_anatomy/private-unselected.md',
                 'fixtures/gate5m_external_plugin/private-unselected.bin', 'tests/test_private_unselected.py'):
        (source / name).write_text('PRIVATE_UNSELECTED_' + str(tmp_path) + '\n')
    return source


def build(source, destination, target='sdist'):
    env = {key: value for key, value in os.environ.items()
           if key not in {'PYTHONPATH', 'VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT'} and not key.startswith('OTEL_')}
    return subprocess.run(['uv', 'build', '--' + target, '--out-dir', str(destination)],
                          cwd=source, env=env, capture_output=True, text=True, timeout=60)


def contents(archive):
    with tarfile.open(archive) as package:
        members = package.getmembers()
        assert all(member.isfile() for member in members)
        return {str(Path(member.name).relative_to(Path(member.name).parts[0])):
                package.extractfile(member).read() for member in members}


def test_direct_sdist_uses_export_projection_and_builds_from_extracted_source(uncurated, tmp_path):
    expected = exporter.export_snapshot(uncurated, tmp_path / 'snapshot')
    result = build(uncurated, tmp_path / 'dist')
    assert result.returncode == 0, result.stderr
    payloads = contents(next((tmp_path / 'dist').glob('*.tar.gz')))
    selected = {item['path'] for item in expected['files']}
    assert set(payloads) == selected | {'EXPORT-MANIFEST.json', 'PKG-INFO'}
    canary = ('PRIVATE_UNSELECTED_' + str(tmp_path)).encode()
    assert not any(canary in payload for payload in payloads.values())
    manifest = json.loads(payloads['EXPORT-MANIFEST.json'])
    assert manifest == expected
    for entry in manifest['files']:
        assert hashlib.sha256(payloads[entry['path']]).hexdigest() == entry['sha256']
    extracted = tmp_path / 'extracted'
    extracted.mkdir()
    for name, payload in payloads.items():
        path = extracted / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    wheel = build(extracted, tmp_path / 'wheel', 'wheel')
    assert wheel.returncode == 0, wheel.stderr
    with zipfile.ZipFile(next((tmp_path / 'wheel').glob('*.whl'))) as archive:
        assert archive.read('capability_anatomy/__init__.py') == payloads['src/capability_anatomy/__init__.py']
        assert not any('private-unselected' in name for name in archive.namelist())


@pytest.mark.parametrize('kind', ['symlink_leaf', 'symlink_directory', 'fifo', 'hardlink'])
def test_direct_build_refuses_unsafe_selected_inputs_before_archiving(uncurated, tmp_path, kind):
    outside = tmp_path / 'outside'
    outside.mkdir()
    victim = outside / 'private.py'
    victim.write_text('PRIVATE_EXTERNAL_CANARY')
    selected = uncurated / 'src/capability_anatomy/__init__.py'
    if kind == 'symlink_directory':
        selected = uncurated / 'src/capability_anatomy'
        selected.rename(selected.with_name('original_package'))
        selected.symlink_to(outside, target_is_directory=True)
    else:
        selected.unlink()
        if kind == 'symlink_leaf':
            selected.symlink_to(victim)
        elif kind == 'hardlink':
            os.link(victim, selected)
        else:
            os.mkfifo(selected)
    result = build(uncurated, tmp_path / 'refused')
    assert result.returncode != 0
    assert not list((tmp_path / 'refused').glob('*.tar.gz'))
    assert victim.read_text() == 'PRIVATE_EXTERNAL_CANARY'


def test_standalone_ci_composes_one_gate_and_uploads_expanded_required_artifacts():
    workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    job = workflow['jobs']['acceptance']
    assert set(job['strategy']['matrix']['os']) == {'ubuntu-latest', 'macos-latest'}
    assert job['steps'][1]['uses'] == './.github/actions/acceptance'
    action = yaml.safe_load((ROOT / '.github/actions/acceptance/action.yml').read_text())
    gate = next(step for step in action['runs']['steps'] if step.get('name') == 'Test source, lifecycle and collector')
    assert 'scripts/ci_acceptance.py' in gate['run']
    uploads = [step for step in action['runs']['steps'] if step.get('uses', '').startswith('actions/upload-artifact@')]
    assert all(step['with']['if-no-files-found'] == 'error' for step in uploads)
    artifact = next(step for step in uploads if step['name'] == 'Preserve built distributions')
    assert artifact['with']['path'].splitlines() == ['${{ github.workspace }}/dist/*.whl', '${{ github.workspace }}/dist/*.tar.gz']
    assert '$GITHUB_WORKSPACE' not in artifact['with']['path']


@pytest.mark.parametrize('kind', ['outside_source', 'unselected_source', 'destination_escape', 'second_selection'])
def test_builder_cannot_add_a_second_unreviewed_projection(uncurated, tmp_path, kind):
    project = uncurated / 'pyproject.toml'
    content = project.read_text()
    source = 'schemas/experiment-config.v1.schema.json'
    if kind == 'outside_source':
        outside = tmp_path / 'outside.json'
        outside.write_text('PRIVATE_EXTERNAL_CANARY')
        content = content.replace('"' + source + '" =', '"' + str(outside) + '" =')
    elif kind == 'unselected_source':
        content = content.replace('"' + source + '" =', '"private-unselected.txt" =')
    elif kind == 'destination_escape':
        content = content.replace('"capability_anatomy/_schemas/experiment-config.v1.schema.json"', '"../outside.json"')
    else:
        content += '\n[tool.hatch.build.targets.sdist]\ninclude = ["**"]\n'
    project.write_text(content)
    result = build(uncurated, tmp_path / 'refused')
    assert result.returncode != 0
    assert 'release builder' in result.stderr
    assert not list((tmp_path / 'refused').glob('*.tar.gz'))


def test_documentation_and_fixed_reasons_ship_in_source(uncurated, tmp_path):
    exporter.export_snapshot(uncurated, tmp_path / 'snapshot')
    for relative in ('docs/observability.md', 'docs/reasons.md', 'docs/release.md', 'scripts/update_reason_catalog.py'):
        assert (tmp_path / 'snapshot' / relative).read_bytes() == (ROOT / relative).read_bytes()
    spec = importlib.util.spec_from_file_location('release_reason_catalog', ROOT / 'scripts/update_reason_catalog.py')
    catalog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog)
    document = (ROOT / 'docs/reasons.md').read_text()
    assert all('`' + reason + '`' in document for reason in catalog.reason_catalog())


def release_verifier():
    spec = importlib.util.spec_from_file_location('final_release_verifier', ROOT / 'scripts/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_verifier_refuses_unbound_extra_source_files(uncurated, tmp_path):
    snapshot = tmp_path / 'snapshot'
    exporter.export_snapshot(uncurated, snapshot)
    (snapshot / 'PKG-INFO').write_text('Metadata-Version: 2.4\n')
    verifier = release_verifier()
    assert verifier.verify_export(snapshot)['license_present']
    (snapshot / 'unbound-private.txt').write_text('unreviewed bytes')
    with pytest.raises(ValueError, match='reviewed file selection'):
        verifier.verify_export(snapshot)
    (snapshot / 'unbound-private.txt').unlink()
    assert verifier.verify_export(snapshot)['license_present']


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'duplicate', 'traversal'])
def test_release_verifier_refuses_unsafe_archive_before_installing(tmp_path, kind):
    from io import BytesIO
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    with zipfile.ZipFile(artifacts / 'package.whl', 'w'):
        pass
    with tarfile.open(artifacts / 'package.tar.gz', 'w:gz') as archive:
        member = tarfile.TarInfo('source/README.md')
        member.size = 2
        archive.addfile(member, BytesIO(b'OK'))
        unsafe = tarfile.TarInfo('../outside' if kind == 'traversal' else 'source/link')
        if kind in {'symlink', 'hardlink'}:
            unsafe.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
            unsafe.linkname = 'README.md'
        elif kind == 'duplicate':
            unsafe.name = member.name
        archive.addfile(unsafe, BytesIO(b''))
    with pytest.raises(ValueError, match='unique relative regular files'):
        release_verifier().verify(artifacts, tmp_path / 'work')
    assert not (tmp_path / 'work/installed-wheel').exists()
    assert not (tmp_path / 'outside').exists()


@pytest.mark.parametrize('field', ['readme', 'license-files', 'dynamic', 'metadata-hook'])
def test_direct_build_cannot_read_external_metadata(uncurated, tmp_path, field):
    outside = tmp_path / 'outside.txt'
    outside.write_text('EXTERNAL_METADATA_CANARY')
    project = uncurated / 'pyproject.toml'
    content = project.read_text()
    if field == 'readme':
        content = content.replace('[project]\n', '[project]\nreadme = ' + json.dumps(str(outside)) + '\n', 1)
    elif field == 'license-files':
        content = content.replace('license-files = ["LICENSE", "NOTICE"]', 'license-files = [' + json.dumps(str(outside)) + ']')
    elif field == 'dynamic':
        content = content.replace('[project]\n', '[project]\ndynamic = ["readme"]\n', 1)
    else:
        hook = tmp_path / 'hook.py'
        marker = tmp_path / 'hook-executed'
        hook.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("executed")\n')
        content += '\n[tool.hatch.metadata.hooks.custom]\npath = ' + json.dumps(str(hook)) + '\n'
    project.write_text(content)
    result = build(uncurated, tmp_path / 'refused')
    assert result.returncode != 0
    assert 'release builder' in result.stderr
    assert not list((tmp_path / 'refused').glob('*.tar.gz'))
    assert outside.read_text() == 'EXTERNAL_METADATA_CANARY'
    assert not (tmp_path / 'hook-executed').exists()


def test_export_bounds_cumulative_bytes_before_destination_creation(uncurated, tmp_path, monkeypatch):
    positive = exporter.export_snapshot(uncurated, tmp_path / 'positive')
    total = sum(item['bytes'] for item in positive['files'])
    monkeypatch.setattr(exporter, 'MAX_TOTAL_SOURCE_BYTES', total)
    assert exporter.export_snapshot(uncurated, tmp_path / 'at-limit')['source_tree_sha256'] == positive['source_tree_sha256']
    monkeypatch.setattr(exporter, 'MAX_TOTAL_SOURCE_BYTES', total - 1)
    with pytest.raises(exporter.ExportError, match='total byte limit'):
        exporter.export_snapshot(uncurated, tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()


def test_export_bounds_global_projection_count(uncurated, tmp_path, monkeypatch):
    positive = exporter.export_snapshot(uncurated, tmp_path / 'positive')
    count = len(positive['files'])
    # Each independently bounded source tree is small; the combined projection
    # includes docs, schemas, tests and package files and exceeds any one tree.
    monkeypatch.setattr(exporter, 'MAX_SOURCE_FILES', count)
    assert exporter.export_snapshot(uncurated, tmp_path / 'at-limit')['files'] == positive['files']
    monkeypatch.setattr(exporter, 'MAX_SOURCE_FILES', count - 1)
    with pytest.raises(exporter.ExportError, match='selected-file limit'):
        exporter.export_snapshot(uncurated, tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()


def test_fixed_reason_catalog_covers_conditional_and_public_error_declarations(tmp_path):
    spec = importlib.util.spec_from_file_location('final_reason_catalog', ROOT / 'scripts/update_reason_catalog.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {'credential_reference_missing', 'credential_environment_missing', 'command_complete',
                'command_failed', 'conformance_authorized', 'full_scan_authorized'}
    assert required <= module.reason_catalog().keys()
    package = tmp_path / 'src/capability_anatomy'
    package.mkdir(parents=True)
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/export_standalone.py').write_text('')
    (package / 'declarations.py').write_text("""
class Refused:
    reason = 'class_refused'
_PUBLIC_ERRORS = {'public_refused': (2, 'safe message')}
_PUBLIC_ERRORS['indexed_refused'] = (2, 'safe message')
record(reason=getattr(error, 'reason', 'fallback_refused'))
record(reason='accepted_branch' if accepted else 'refused_branch')
""")
    assert module.reason_catalog(tmp_path).keys() == {
        'class_refused', 'public_refused', 'indexed_refused', 'fallback_refused',
        'accepted_branch', 'refused_branch'}
