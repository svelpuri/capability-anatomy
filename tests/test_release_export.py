"""The exporter creates a bounded, digest-verifiable snapshot without overwrites."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_standalone.py"
spec = importlib.util.spec_from_file_location("standalone_export", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    files = {
        "README.md": "# Example research alpha\n",
        "pyproject.toml": '[project]\nname = "capability-anatomy"\nversion = "0.1.0"\n',
        "uv.lock": "version = 1\n",
        "configs/examples/synthetic-scan.yaml": "experiment_id: synthetic\n",
        "fixtures/synthetic/manifest.json": "{}\n",
        "scripts/export_standalone.py": SCRIPT.read_text(),
        "scripts/verify_release.py": "# artifact acceptance workflow\n",
        "scripts/verify_observability.py": "# collector acceptance workflow\n",
        ".github/workflows/ci.yml": "name: artifact acceptance\n",
        "src/capability_anatomy/__init__.py": 'NAME = "public"\n',
        "schemas/experiment-config.v1.schema.json": "{}\n",
        "schemas/evidence-manifest.v1.schema.json": "{}\n",
        "schemas/metric-result.v1.schema.json": "{}\n",
        "schemas/transformation-recipe.v1.schema.json": "{}\n",
        "schemas/private-legacy.json": "PRIVATE SCHEMA\n",
        "fixtures/gate5m_external_plugin/example.py": "# external plugin fixture\n",
        "tests/test_capability_anatomy_phase1.py": "def test_contract(): pass\n",
        "tests/test_capability_anatomy_phase2.py": "def test_contract(): pass\n",
        "tests/test_capability_anatomy_phase3.py": "def test_contract(): pass\n",
        "tests/test_capability_anatomy_phase4.py": "def test_contract(): pass\n",
        "tests/test_capability_anatomy_phase5_public.py": "def test_contract(): pass\n",
        "tests/test_capability_anatomy_phase5_campaign.py": "def test_contract(): pass\n",
        "tests/test_capability_anatomy_gate5m.py": "def test_contract(): pass\n",
        "tests/test_release_security.py": "def test_regression(): pass\n",
        "tests/test_private_campaign.py": "PRIVATE HISTORY\n",
        "src/private_prototype/private.py": "PRIVATE SOURCE\n",
        "reports/private.txt": "PRIVATE REPORT\n",
        ".env": "PRIVATE CREDENTIAL\n",
        ".git/config": "PRIVATE HISTORY\n",
        ".session-sync.md": "PRIVATE COORDINATION\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_public_snapshot_hashes_and_private_exclusion(source, tmp_path):
    destination = tmp_path / "export"
    manifest = exporter.export_snapshot(source, destination, allow_unlicensed_preview=True)
    assert manifest == json.loads((destination / "EXPORT-MANIFEST.json").read_text())
    assert manifest["release_status"] == "unlicensed_review_preview_only"
    assert manifest["tests"]["excluded_private_files"] == ["test_private_campaign.py"]
    assert "test_release_security.py" in manifest["tests"]["included_files"]
    exported = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
    assert exported == {item["path"] for item in manifest["files"]} | {"EXPORT-MANIFEST.json"}
    for item in manifest["files"]:
        payload = (destination / item["path"]).read_bytes()
        assert payload == (source / item["source_path"]).read_bytes()
        assert len(payload) == item["bytes"]
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
        assert b"PRIVATE" not in payload
    assert (destination / ".github/workflows/ci.yml").read_bytes() == (source / ".github/workflows/ci.yml").read_bytes()
    second = exporter.export_snapshot(source, tmp_path / "second", allow_unlicensed_preview=True)
    assert second["source_tree_sha256"] == manifest["source_tree_sha256"]


def test_license_is_required_unless_preview_was_explicit(source, tmp_path):
    with pytest.raises(exporter.ExportError, match="LICENSE"):
        exporter.export_snapshot(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    (source / "LICENSE").write_text("Operator selected license text\n")
    assert exporter.export_snapshot(source, tmp_path / "output")["license_present"] is True


@pytest.mark.parametrize("kind", ["empty_directory", "directory_with_file", "file", "symlink"])
def test_existing_destination_is_preserved(source, tmp_path, kind):
    destination = tmp_path / "output"
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "sentinel").write_text("preserve")
    if kind == "file":
        destination.write_text("preserve")
    elif kind == "symlink":
        destination.symlink_to(protected, target_is_directory=True)
    else:
        destination.mkdir()
        if kind == "directory_with_file":
            (destination / "sentinel").write_text("preserve")
    with pytest.raises(exporter.ExportError, match="already exists"):
        exporter.export_snapshot(source, destination, allow_unlicensed_preview=True)
    assert (protected / "sentinel").read_text() == "preserve"
    assert destination.exists()
    if kind == "file":
        assert destination.read_text() == "preserve"
    elif kind != "empty_directory":
        assert (destination / "sentinel").read_text() == "preserve"


@pytest.mark.parametrize("target", ["README.md", "src/capability_anatomy/leak.py", "schemas/experiment-config.v1.schema.json"])
def test_selected_symlink_input_cannot_disclose_outside_bytes(source, tmp_path, target):
    outside = tmp_path / "outside-secret"
    outside.write_text("outside sentinel")
    path = source / target
    if path.exists():
        path.unlink()
    path.symlink_to(outside)
    with pytest.raises(exporter.ExportError, match="symbolic link"):
        exporter.export_snapshot(source, tmp_path / "output", allow_unlicensed_preview=True)
    assert outside.read_text() == "outside sentinel"
    assert not (tmp_path / "output").exists()


def test_export_cli_rejects_failure_and_emits_digest_on_success(source, tmp_path):
    command = [sys.executable, str(SCRIPT), str(tmp_path / "output"), "--source", str(source)]
    refusal = subprocess.run(command, capture_output=True, text=True)
    assert refusal.returncode == 2
    assert json.loads(refusal.stderr)["error"] == "ExportError"
    success = subprocess.run([*command, "--allow-unlicensed-preview"], capture_output=True, text=True)
    assert success.returncode == 0
    assert json.loads(success.stdout)["source_tree_sha256"] == json.loads((tmp_path / "output/EXPORT-MANIFEST.json").read_text())["source_tree_sha256"]


@pytest.mark.parametrize("kind", ["symlink_leaf", "symlink_parent", "fifo"])
def test_source_swap_at_the_copy_boundary_is_refused_without_hanging(source, tmp_path, kind):
    # The subprocess has a deadline so a FIFO regression is a test failure, not
    # an indefinitely blocked release gate. Only disposable sentinel files move.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "__init__.py").write_text("outside sentinel")
    program = '''
import importlib.util, os
from pathlib import Path
spec=importlib.util.spec_from_file_location("exporter", SCRIPT)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
source=Path(SOURCE); outside=Path(OUTSIDE); destination=Path(DESTINATION)
original=m._safe_file
calls=0
def swap(root, relative):
 global calls
 path=original(root, relative)
 if relative == "src/capability_anatomy/__init__.py":
  calls += 1
  if calls == 2:
   if KIND == "symlink_parent":
    path.parent.rename(path.parent.with_name("original-package"))
    path.parent.symlink_to(outside, target_is_directory=True)
   else:
    path.unlink()
    if KIND == "fifo": os.mkfifo(path)
    else: path.symlink_to(outside / "__init__.py")
 return path
m._safe_file=swap
try: m.export_snapshot(source,destination,allow_unlicensed_preview=True)
except (OSError,m.ExportError):
 assert calls == 2
 print("REFUSED_AT_COPY_BOUNDARY")
else: raise AssertionError("unsafe source was copied")
'''
    bindings = "\n".join(f"{name}={value!r}" for name, value in {
        "SCRIPT": str(SCRIPT), "SOURCE": str(source), "OUTSIDE": str(outside),
        "DESTINATION": str(tmp_path / "output"), "KIND": kind,
    }.items())
    result = subprocess.run([sys.executable, "-c", bindings + program], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "REFUSED_AT_COPY_BOUNDARY"
    assert (outside / "__init__.py").read_text() == "outside sentinel"
    assert not (tmp_path / "output").exists()


def test_destination_parent_swap_cannot_create_files_outside_export(source, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("preserve")
    original = exporter._write_file
    swapped = False
    def swap(root, relative, payload):
        nonlocal swapped
        if not swapped:
            assert relative == ".github/workflows/ci.yml"
            (tmp_path / "output/.github").symlink_to(outside, target_is_directory=True)
            swapped = True
        original(root, relative, payload)
    monkeypatch.setattr(exporter, "_write_file", swap)
    with pytest.raises(OSError):
        exporter.export_snapshot(source, tmp_path / "output", allow_unlicensed_preview=True)
    assert swapped
    assert list(outside.iterdir()) == [outside / "sentinel"]
    assert (outside / "sentinel").read_text() == "preserve"
    assert not (tmp_path / "output/EXPORT-MANIFEST.json").exists()


@pytest.mark.parametrize("swap_at", ["first_payload", "manifest"])
def test_destination_ancestor_swap_keeps_all_writes_on_original_root(source, tmp_path, monkeypatch, swap_at):
    parent = tmp_path / "destination-parent"
    parent.mkdir()
    outside = tmp_path / "outside"
    (outside / "snapshot").mkdir(parents=True)
    (outside / "snapshot/sentinel").write_text("preserve outside")
    original = exporter._write_file
    swapped = False

    def swap(root, relative, payload):
        nonlocal swapped
        if not swapped and (swap_at == "first_payload" or relative == "EXPORT-MANIFEST.json"):
            parent.rename(tmp_path / "original-parent")
            parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        original(root, relative, payload)

    monkeypatch.setattr(exporter, "_write_file", swap)
    manifest = exporter.export_snapshot(source, parent / "snapshot", allow_unlicensed_preview=True)
    assert swapped
    assert list((outside / "snapshot").iterdir()) == [outside / "snapshot/sentinel"]
    assert (outside / "snapshot/sentinel").read_text() == "preserve outside"
    actual = tmp_path / "original-parent/snapshot"
    assert json.loads((actual / "EXPORT-MANIFEST.json").read_text()) == manifest
    for item in manifest["files"]:
        assert (actual / item["path"]).read_bytes() == (source / item["source_path"]).read_bytes()


@pytest.mark.parametrize("swap_at", ["read", "inventory"])
def test_source_ancestor_swap_cannot_replace_selected_source_bytes(source, tmp_path, monkeypatch, swap_at):
    parent = tmp_path / "source-parent"
    parent.mkdir()
    source = source.rename(parent / "source")
    outside = tmp_path / "outside"
    shutil.copytree(source, outside / "source")
    (outside / "source/README.md").write_text("OUTSIDE SOURCE SENTINEL")
    (outside / "source/tests/test_private_outside_canary.py").write_text("OUTSIDE PRIVATE FILENAME")
    expected = (source / "README.md").read_bytes()
    original = exporter._read_file
    swapped = False

    def replace_ancestor():
        nonlocal swapped
        if not swapped:
            parent.rename(tmp_path / "original-parent")
            parent.symlink_to(outside, target_is_directory=True)
            swapped = True

    def swap(path, relative, root):
        replace_ancestor()
        return original(path, relative, root)

    if swap_at == "read":
        monkeypatch.setattr(exporter, "_read_file", swap)
    else:
        original_list = exporter._listed_files
        def swap_inventory(root, relative, *, recursive):
            replace_ancestor()
            return original_list(root, relative, recursive=recursive)
        monkeypatch.setattr(exporter, "_listed_files", swap_inventory)
    manifest = exporter.export_snapshot(source, tmp_path / "output", allow_unlicensed_preview=True)
    assert swapped
    assert (tmp_path / "output/README.md").read_bytes() == expected
    assert manifest["tests"]["excluded_private_files"] == ["test_private_campaign.py"]
    for item in manifest["files"]:
        assert (tmp_path / "output" / item["path"]).read_bytes() == (tmp_path / "original-parent/source" / item["source_path"]).read_bytes()
    assert (outside / "source/README.md").read_text() == "OUTSIDE SOURCE SENTINEL"


@pytest.mark.parametrize("side", ["source", "destination"])
def test_existing_root_ancestor_symlinks_are_refused(source, tmp_path, side):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    selected_source = alias / "source" if side == "source" else source
    destination = alias / "output" if side == "destination" else tmp_path / "output"
    with pytest.raises(OSError):
        exporter.export_snapshot(selected_source, destination, allow_unlicensed_preview=True)
    assert not (tmp_path / "output").exists()


def test_observed_nonempty_replacement_during_acquisition_is_refused(source, tmp_path, monkeypatch):
    # Tactical detection only: an empty-directory replacement cannot establish
    # creation identity. Initial acquisition requires caller-controlled parents.
    destination = tmp_path / "snapshot"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "sentinel").write_text("preserve unrelated directory")
    original = exporter.os.mkdir
    swapped = False

    def replace_after_mkdir(path, *args, **kwargs):
        nonlocal swapped
        result = original(path, *args, **kwargs)
        if path == "snapshot" and not swapped:
            destination.rename(tmp_path / "created-snapshot")
            unrelated.rename(destination)
            swapped = True
        return result

    monkeypatch.setattr(exporter.os, "mkdir", replace_after_mkdir)
    with pytest.raises(exporter.ExportError, match="unexpectedly nonempty"):
        exporter.export_snapshot(source, destination, allow_unlicensed_preview=True)
    assert swapped
    assert list(destination.iterdir()) == [destination / "sentinel"]
    assert (destination / "sentinel").read_text() == "preserve unrelated directory"
    assert not list((tmp_path / "created-snapshot").iterdir())


@pytest.mark.parametrize('name', ['.git', '.env', 'AGENTS.md', 'CLAUDE.md', 'CONTEXT.md', '.session-sync.md'])
def test_forbidden_names_under_selected_tree_never_export(source, tmp_path, name):
    planted=source/'fixtures/gate5m_external_plugin'/name
    planted.mkdir()
    (planted/'selected.py').write_text('PRIVATE_NAME_SENTINEL')
    with pytest.raises(exporter.ExportError, match='forbidden'):
        exporter.export_snapshot(source,tmp_path/'output',allow_unlicensed_preview=True)
    assert not (tmp_path/'output').exists()


def test_export_bounds_actual_large_file_before_read(source, tmp_path):
    path=source/'fixtures/gate5m_external_plugin/huge.txt'
    with path.open('wb') as stream: stream.truncate(800*1024*1024)
    with pytest.raises(exporter.ExportError,match='byte limit'):
        exporter.export_snapshot(source,tmp_path/'output',allow_unlicensed_preview=True)


def test_export_depth_refuses_before_python_recursion(source,tmp_path):
    path=source/'fixtures/gate5m_external_plugin'
    for _ in range(34): path=path/'nested'
    path.mkdir(parents=True);(path/'selected.py').write_text('pass')
    with pytest.raises(exporter.ExportError,match='depth limit'):
        exporter.export_snapshot(source,tmp_path/'output',allow_unlicensed_preview=True)


def test_actual_env_file_canary_is_refused_and_clean_tree_exports(source, tmp_path):
    clean = exporter.export_snapshot(source, tmp_path / 'clean', allow_unlicensed_preview=True)
    assert clean['files']
    planted = source / 'fixtures/gate5m_external_plugin/.env'
    planted.write_text('HF_TOKEN=private-fixture-canary')
    with pytest.raises(exporter.ExportError, match='forbidden'):
        exporter.export_snapshot(source, tmp_path / 'refused', allow_unlicensed_preview=True)
    assert not (tmp_path / 'refused').exists()
