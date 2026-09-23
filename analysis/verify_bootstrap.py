"""Read-only numerical/source verification; writes only a NEW verification file."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--analysis', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
analysis = args.analysis
summary = json.loads((analysis / 'results.json').read_text())
models = ('0.6', '1.7')
metrics = ('tool_selection', 'argument_binding', 'full_call')
specs = ('log_ratio', 'relative')

def rows(name):
    with (analysis / name).open() as f:
        return list(csv.DictReader(f))

records = rows('layer_residuals.csv')
assert len(records) == 336
index = {(r['model_b'], r['metric'], r['general_proxy'], int(r['layer'])): r for r in records}
assert len(index) == 336
point = np.array([[[[float(index[(m, k, s, l)]['residual_probability'])
                    for l in range(28)] for s in specs] for k in metrics] for m in models])
stored = np.load(analysis / 'residual_bootstrap.npz', allow_pickle=False)
boot = stored['residuals']
assert boot.shape == (10000, 2, 3, 2, 28)
se = boot.std(axis=0, ddof=1)
maximum = np.max(np.abs(boot - point) / se, axis=(1, 2, 3, 4))
assert np.allclose(maximum, stored['maximum_standardized_deviation'], atol=1e-12)
critical = float(np.quantile(maximum, .95))
assert math.isclose(critical, summary['simultaneous_critical_value'], abs_tol=1e-12)
for mi, m in enumerate(models):
    for ki, k in enumerate(metrics):
        for si, s in enumerate(specs):
            for layer in range(28):
                r = index[(m, k, s, layer)]
                for sign, suffix in [(-1, 'low'), (1, 'high')]:
                    expected = point[mi, ki, si, layer] + sign * critical * se[mi, ki, si, layer]
                    assert math.isclose(expected, float(r['residual_simultaneous_95_' + suffix]), abs_tol=1e-12)

# Reconstruct the first paired bootstrap draw through scalar Python arithmetic
# and statistics.linear_regression, not the analysis's broadcasting/fit helper.
rng = np.random.default_rng(20260916)
tool_sample = rng.integers(0, 20, (250, 20))[0].tolist()
ppl_sample = rng.integers(0, 5, (250, 5))[0].tolist()
tool_rows = rows('paired_tool_observations.csv')
ppl_rows = rows('perplexity_observations.csv')
assert len(tool_rows) == 3360 and len(ppl_rows) == 280
max_error = 0.0
for mi, m in enumerate(models):
    ratio = []
    for layer in range(28):
        rs = sorted([r for r in ppl_rows if r['model_b'] == m and int(r['layer']) == layer], key=lambda r: r['example_id'])
        ratio.append(statistics.mean(float(rs[i]['intervention_perplexity']) for i in ppl_sample) /
                     statistics.mean(float(rs[i]['baseline_perplexity']) for i in ppl_sample))
    for ki, k in enumerate(metrics):
        damage = []
        for layer in range(28):
            rs = sorted([r for r in tool_rows if r['model_b'] == m and r['metric'] == k and int(r['layer']) == layer], key=lambda r: r['example_id'])
            damage.append(statistics.mean(float(rs[i]['baseline_score']) - float(rs[i]['intervention_score']) for i in tool_sample))
        for si, x in enumerate([[math.log(v) for v in ratio], [v - 1 for v in ratio]]):
            slope, intercept = statistics.linear_regression(x, damage)
            calculated = [y - (intercept + slope * g) for y, g in zip(damage, x)]
            error = max(abs(a - b) for a, b in zip(calculated, boot[0, mi, ki, si]))
            max_error = max(max_error, error)
            assert error < 1e-12

# Reconstruct the report's conjunction without the analysis implementation.
robustness = {}
for m in models:
    robustness[m] = {}
    sets = []
    for k in metrics:
        chosen = []
        for layer in range(28):
            rs = [index[(m, k, s, layer)] for s in specs]
            if (all(float(r['residual_simultaneous_95_low']) > .1 for r in rs)
                and all(float(r['leave_one_layer_out_residual']) >= .1 - 1e-12 for r in rs)
                and float(rs[0]['isotonic_log_residual']) >= .1 - 1e-12):
                chosen.append(layer)
        robustness[m][k] = chosen
        sets.append(set(chosen))
    robustness[m]['all_three'] = sorted(set.intersection(*sets))
assert robustness == json.loads((analysis / 'robustness-summary.json').read_text())

forbidden = [m for m in sys.modules if m.split('.')[0] in {'torch', 'transformers', 'mlx', 'huggingface_hub', 'capability_anatomy'}]
assert not forbidden
result = {'status': 'passed', 'residual_rows_verified': len(records),
          'bootstrap_shape': list(boot.shape), 'simultaneous_bands_reconstructed': True,
          'critical_value': critical, 'first_bootstrap_draw_scalar_reconstruction_max_error': max_error,
          'robustness_conjunction_reconstructed': robustness, 'model_libraries_loaded': forbidden,
          'scope': 'Numerical cross-check of regenerated results; not a new model experiment.'}
with args.output.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(result, indent=2))
