#!/usr/bin/env python3
"""Verify retained paper file integrity and independently checked headline values."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('0.6', '1.7')
METRICS = ('tool_selection', 'argument_binding', 'full_call')
SETS = {'0.6': [6, 9, 10, 14, 18, 19, 21, 26], '1.7': [11, 24, 27]}
R2 = {'0.6': (.0839, .0831, .0828), '1.7': (.2789, .2761, .2761)}
RHO = (.277, .272, .290)
CSVS = ('regression_fits.csv', 'qwen_0_6b_layer_residuals.csv', 'qwen_1_7b_layer_residuals.csv',
        'cross_size_descriptive.csv', 'layer_residuals.csv', 'paired_tool_observations.csv',
        'perplexity_observations.csv')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def compare(a, b, path='root'):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), f'keys differ: {path}')
        for key in a:
            compare(a[key], b[key], path + '/' + key)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), f'length differs: {path}')
        for i, (left, right) in enumerate(zip(a, b)):
            compare(left, right, f'{path}/{i}')
    elif isinstance(a, bool) or a is None:
        require(a == b, f'value differs: {path}')
    else:
        try:
            left, right = float(a), float(b)
        except (TypeError, ValueError):
            require(a == b, f'value differs: {path}')
        else:
            require(math.isfinite(left) and math.isfinite(right) and
                    math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10), f'number differs: {path}')


def verify_snapshot(root):
    """Check the Git-bound inventory; this is integrity checking, not a signature."""
    manifest = read(root / 'PUBLIC-MANIFEST.json')
    names = [entry['path'] for entry in manifest['files']]
    require(len(names) == len(set(names)), 'duplicate snapshot manifest entry')
    for entry in manifest['files']:
        relative = Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,
                'invalid snapshot path')
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            require(not cursor.is_symlink(), 'snapshot symlink refused')
        require(cursor.is_file(), 'snapshot file missing')
        payload = cursor.read_bytes()
        require(len(payload) == entry['bytes'] and
                hashlib.sha256(payload).hexdigest() == entry['sha256'],
                'snapshot hash mismatch: ' + entry['path'])
    return len(names)


def verify_bootstrap(root, reproduced):
    """Validate regenerated numbers; the historical compressed hash is provenance."""
    with tempfile.TemporaryDirectory(prefix='paper-bootstrap-') as directory:
        output = Path(directory) / 'verification.json'
        result = subprocess.run(
            [sys.executable, str(root / 'analysis/verify_bootstrap.py'),
             '--analysis', str(reproduced), '--output', str(output)],
            capture_output=True, text=True, check=False,
        )
        require(result.returncode == 0,
                'bootstrap numerical verification failed: ' + result.stderr.strip())
        require(read(output)['status'] == 'passed', 'bootstrap verification incomplete')
    reference = read(root / 'analysis/bootstrap.json')
    digest = hashlib.sha256((reproduced / reference['filename']).read_bytes()).hexdigest()
    return {'numerical_verification': 'passed', 'sha256': digest,
            'historical_sha256_match': digest == reference['sha256']}


def verify(root=ROOT, reproduced=None, splits=None):
    snapshot_files = verify_snapshot(root)
    bootstrap_verification = None
    files = observations = 0
    for model in MODELS:
        bundle = root / 'artifacts' / f'qwen3-{model}b'
        manifest = read(bundle / 'MANIFEST.json')
        entries = manifest['files']
        names = [e['path'] for e in entries]
        require(len(names) == len(set(names)), 'duplicate manifest entry')
        actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
        require(actual == set(names) | {'MANIFEST.json'}, 'paper artifact inventory differs')
        for entry in entries:
            rel = Path(entry['path'])
            require(not rel.is_absolute() and '..' not in rel.parts, 'invalid artifact path')
            path = bundle / rel
            require(not path.is_symlink() and path.is_file(), 'artifact is not an ordinary file')
            payload = path.read_bytes()
            require(len(payload) == entry['bytes'] and hashlib.sha256(payload).hexdigest() == entry['sha256'],
                    f'artifact hash mismatch: {model}/{rel}')
            files += 1
        rows = [json.loads(line) for line in (bundle / 'observations.jsonl').read_text().splitlines()]
        require(len(rows) == 3400 and all(r['status'] == 'complete' for r in rows), 'observation count/status mismatch')
        observations += len(rows)
        require(read(bundle / 'provenance.json')['model_source'] == f'Qwen/Qwen3-{model}B', 'model identity mismatch')
    analysis = root / 'analysis'
    robustness = read(analysis / 'robustness-summary.json')
    for model in MODELS:
        require(robustness[model]['all_three'] == SETS[model], 'conservative block set mismatch')
    fits = {(r['model_b'], r['metric']): r for r in table(analysis / 'regression_fits.csv') if r['general_proxy'] == 'log_ratio'}
    require(len(fits) == 6, 'expected six log regressions')
    for model in MODELS:
        for i, metric in enumerate(METRICS):
            require(abs(float(fits[model, metric]['r_squared']) - R2[model][i]) < .00005, 'headline R2 mismatch')
    cross = {r['metric']: r for r in table(analysis / 'cross_size_descriptive.csv') if r['general_proxy'] == 'log_ratio'}
    for i, metric in enumerate(METRICS):
        require(abs(float(cross[metric]['residual_spearman_across_sizes']) - RHO[i]) < .0005, 'headline cross-size rho mismatch')
    halves = read(analysis / 'split_half_results.json')
    for model, expected in [('0.6', .9307730477801928), ('1.7', .8757614731271047)]:
        result = halves['phase5'][model]['metric_analysis']['tool_selection']['split_half_rank_correlation_distribution']
        require(math.isclose(result['median'], expected, abs_tol=1e-12), 'split-half headline mismatch')
    if reproduced:
        for name in CSVS:
            compare(table(analysis / name), table(reproduced / name), name)
        compare(robustness, read(reproduced / 'robustness-summary.json'), 'robustness')
        reference, generated = read(analysis / 'results.json'), read(reproduced / 'results.json')
        for name in ('python_version', 'numpy_version'):
            reference.pop(name, None); generated.pop(name, None)
        compare(reference, generated, 'results')
        compare(read(analysis / 'verification.json'), read(reproduced / 'verification.json'), 'verification')
        bootstrap_verification = verify_bootstrap(root, reproduced)
        for stem in ('qwen_0_6b_specificity', 'qwen_1_7b_specificity', 'raw_vs_log_sensitivity', 'cross_size_residuals'):
            for suffix in ('.svg', '.png'):
                require((reproduced / (stem + suffix)).stat().st_size > 1000, 'figure missing')
    if splits:
        compare(halves, read(splits / 'statistics.json'), 'split-half analysis')
    return {'status': 'passed', 'snapshot_files': snapshot_files, 'artifact_files': files, 'observations': observations,
            'headline_sets': SETS, 'analysis_reproduced': reproduced is not None,
            'split_half_reproduced': splits is not None,
            'bootstrap': bootstrap_verification,
            'scope': 'Retained artifact integrity and numerical reproduction, not new inference or historical execution certification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduced', type=Path)
    parser.add_argument('--splits', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(reproduced=args.reproduced, splits=args.splits), indent=2))


if __name__ == '__main__':
    main()
