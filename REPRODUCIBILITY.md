# Reproduce the retained paper analysis

Run the README commands from the repository root in a fresh checkout. `uv sync
--frozen` installs the default test group; `--group analysis` adds the fully
pinned numerical/plotting dependencies. These are also recorded in
`analysis/requirements-analysis.txt`. Model packages installed for package
contract tests are not imported by the paper analyses.

1. `uv run python scripts/verify_paper.py` verifies the complete snapshot inventory and public artifact hashes,
   both 3,400-row observation files, required identities, and headline numbers.
2. `analysis/specificity_analysis.py` reads observations, checks stored damage,
   refits twelve regressions, performs 10,000 paired resamples with seed
   20260916, computes 336 residual bands, and regenerates tables and figures.
3. `analysis/split_half_analysis.py` reproduces the Phase 5 discovery-only
   split-half analysis, preserving the original 1,000-resample random sequence.
   The unrelated earlier-experiment analysis is excluded.
4. `analysis/verify_bootstrap.py` reconstructs all simultaneous bands and the
   first bootstrap draw with scalar arithmetic and `statistics.linear_regression`.
5. `scripts/verify_paper.py --reproduced DIR --splits DIR` compares every numeric
   table against the retained reference, both conservative block sets, all six
   R² values, three cross-size correlations, and split-half distributions.

Expected outputs are in `analysis/`. The redundant bootstrap NPZ is omitted;
`analysis/bootstrap.json` records its historical SHA-256. Exact byte equality was
verified on the recorded Python/NumPy environment. Small floating-point
variation across architectures is assessed with a 1e-10 numerical tolerance;
this does not relax any discrete layer-set check. The final verifier also invokes
the independent bootstrap check (all 336 bands, maximum deviations and scalar
first-draw reconstruction) and reports compressed-file hash equality separately.
Linux/macOS floating-point differences can change that hash while the numerical
checks pass. The retained source/evidence hashes still require exact equality.
PNG/SVG figures regenerate
from the same numbers; fonts and rendering libraries may affect figure bytes.
SVG dates are suppressed to avoid embedding execution timestamps.

The public evidence is an explicit subset of the historical run, containing
unchanged observations, metrics, rankings, controls and configuration identities.
`artifacts/*/MANIFEST.json` records hashes for every selected file and the hash
of the excluded original manifest. Historical relative configuration locators
are preserved as evidence, not interpreted as executable public paths.
The public verifier checks integrity and arithmetic. It does not certify
historical execution, approvals, complete task caches, or trace delivery.

## Figures without repeating the statistics

```sh
uv run --group analysis python analysis/render_figures.py --analysis analysis --output .reproduction/figures
```

The four figure pairs are `qwen_0_6b_specificity`, `qwen_1_7b_specificity`,
`raw_vs_log_sensitivity`, and `cross_size_residuals`.

## Optional model experiments

No inference is needed for any paper number above. A new model experiment is
a separate execution, not a regeneration of the historical observations.
Install `uv sync --frozen --extra qwen3`, obtain the pinned model revisions in
PROVENANCE under their upstream terms, and follow `docs/installation.md`,
`docs/operations.md`, and the versioned Phase 5 schemas. Use your own explicitly
reviewed input and policy documents; no historical approval is transferred.

Dataset text and historical authorization files are deliberately not distributed.
The mixed upstream source identities and licenses are in `data/paper/`.
Consequently this snapshot does not provide a one-command exact historical
inference replay. Its supported reproducibility claim is reconstruction of
paper analyses from retained observations. The historical Torch/Transformers
versions also differ from the current release. Do not present fresh results
as byte-identical historical measurements without a separately verified run.

## Source distributions and telemetry

The curated wheel/sdist contain the software and its contract tests. The full
research repository additionally carries evidence and paper analyses; clone or
archive the tagged repository to obtain them. `scripts/ci_acceptance.py` runs
normal/reverse tests, package lifecycle checks and real OpenTelemetry Collector
readback. See `docs/observability.md` for the runtime signals. Numerical scripts
write deterministic verification records and import no model libraries.
