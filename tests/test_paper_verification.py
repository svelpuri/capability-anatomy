"""Research-snapshot checks; intentionally outside the software-only sdist."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_verifier', ROOT / 'scripts/verify_paper.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def snapshot(tmp_path):
    for entry in json.loads((ROOT / 'PUBLIC-MANIFEST.json').read_text())['files']:
        source = ROOT / entry['path']
        destination = tmp_path / entry['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(ROOT / 'PUBLIC-MANIFEST.json', tmp_path / 'PUBLIC-MANIFEST.json')
    return tmp_path


def test_retained_snapshot_verifies(snapshot):
    assert verifier.verify(snapshot)['observations'] == 6800


def test_changed_nonheadline_result_cannot_pass_default_verification(snapshot):
    path = snapshot / 'analysis/qwen_0_6b_layer_residuals.csv'
    before = path.read_bytes()
    after = before.replace(b'0.8991928600726307', b'0.7991928600726307', 1)
    assert before != after, 'test corruption did not apply'
    path.write_bytes(after)
    with pytest.raises(ValueError, match='snapshot hash mismatch'):
        verifier.verify(snapshot)
