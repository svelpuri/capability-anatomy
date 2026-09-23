"""Research-snapshot checks; intentionally outside the software-only sdist."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

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


@pytest.fixture(scope='module')
def bootstrap_reproduction(tmp_path_factory):
    """Generate real bootstrap draws, omitting plotting in this numerical test."""
    output = tmp_path_factory.mktemp('bootstrap') / 'specificity'
    command = '''
import importlib.util, sys
spec = importlib.util.spec_from_file_location('specificity', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.plot = lambda *args: None
sys.argv = [sys.argv[1], '--output', sys.argv[2]]
module.main()
'''
    subprocess.run([sys.executable, '-c', command,
                    str(ROOT / 'analysis/specificity_analysis.py'), str(output)],
                   check=True, capture_output=True, text=True)
    for path in (ROOT / 'figures').iterdir():
        if path.suffix in {'.png', '.svg'}:
            shutil.copyfile(path, output / path.name)
    return output


def test_roundoff_changed_bootstrap_passes_numerical_verification(bootstrap_reproduction, tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setenv('PYTHONOPTIMIZE', '1')
    output = tmp_path / 'reproduction'
    shutil.copytree(bootstrap_reproduction, output)
    path = output / 'residual_bootstrap.npz'
    before = path.read_bytes()
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays['residuals'] = np.nextafter(arrays['residuals'], np.inf)
    np.savez_compressed(path, **arrays)
    assert path.read_bytes() != before, 'roundoff mutation did not apply'
    result = verifier.verify(reproduced=output)
    assert result['bootstrap']['numerical_verification'] == 'passed'
    assert result['bootstrap']['historical_sha256_match'] is False


@pytest.mark.parametrize('key', ['residuals', 'maximum_standardized_deviation'])
@pytest.mark.parametrize('optimization', ['0', '1'])
def test_numerically_changed_bootstrap_is_rejected(bootstrap_reproduction, tmp_path, key, optimization, monkeypatch):
    import numpy as np
    monkeypatch.setenv('PYTHONOPTIMIZE', optimization)
    output = tmp_path / 'reproduction'
    shutil.copytree(bootstrap_reproduction, output)
    path = output / 'residual_bootstrap.npz'
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    original = arrays[key].flat[-1]
    arrays[key].flat[-1] += .5
    assert arrays[key].flat[-1] != original, 'numerical corruption did not apply'
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='bootstrap numerical verification failed'):
        verifier.verify(reproduced=output)
