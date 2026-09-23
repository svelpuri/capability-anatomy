# Capability Anatomy

Research code and retained measurements accompanying **Layerwise Behavioral
Sensitivity in LLMs: Stability, Scale, and the Limits of Capability Localization**
by Srinivas Velpuri (Andyur AI).

> The published experiments identify intervention-specific behavioral sensitivity.
> They do not establish that a capability is uniquely stored in a particular
> layer or component.

Capability Anatomy measures behavioral changes when individual transformer
blocks are bypassed. This artifact analyzes retained Qwen3-0.6B and Qwen3-1.7B
measurements of tool selection, argument binding, full-call correctness, and
collateral perplexity. [Paper materials and status](paper/README.md).

## Reproduce the analysis

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Numerical analysis
runs on a CPU without model downloads, credentials, or inference.

```sh
uv sync --frozen
uv run pytest
uv run python scripts/verify_paper.py
uv run --group analysis python analysis/specificity_analysis.py --output .reproduction/specificity
uv run python analysis/split_half_analysis.py --output .reproduction/splits
uv run --group analysis python analysis/verify_bootstrap.py --analysis .reproduction/specificity --output .reproduction/bootstrap-verification.json
uv run python scripts/verify_paper.py --reproduced .reproduction/specificity --splits .reproduction/splits
```

Output directories must be new. Specificity analysis also regenerates the four
figure pairs. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact scope,
verification, and the separate optional inference workflow.

## Results and limits

The conservative excess-sensitivity block sets are **6, 9, 10, 14, 18, 19, 21,
26** for 0.6B and **11, 24, 27** for 1.7B (zero-based). Cross-size log-residual
Spearman correlations are **0.277 / 0.272 / 0.290** for selection / binding /
full-call. These are descriptive same-family comparisons, not independent
replications or evidence that an abstract capability moves between layers.

Each model contributes 20 distinct discovery tool examples and five perplexity
documents. Repeated deterministic measurements are collapsed. The analysis uses
10,000 paired bootstrap draws and approximate simultaneous bands across 336
residuals. Layers are not independent sampled units. The retained validation
partition contains baseline measurements only; it supplies no held-out
intervention map. Binding and full-call coincide throughout the 1.7B data.
Perplexity is a limited proxy for general damage. The post-hoc statistical plan,
linear-model shape, saturation, and tiny empirical sample limit interpretation.

## Contents

| Path | Contents |
|---|---|
| `src/capability_anatomy/` | Intervention, measurement, evidence and telemetry code |
| `artifacts/qwen3-{0.6,1.7}b/` | Byte-preserved paper observations and identity metadata; public hash manifests |
| `configs/paper/` | References to historical configuration and protocol identities |
| `data/paper/` | Dataset provenance, source IDs and redistribution boundaries |
| `analysis/` | Portable analyses, expected numeric tables and verification results |
| `figures/` | Four specificity figure pairs (SVG and PNG) |
| `paper/` | Manuscript materials and compilation status |
| `tests/`, `scripts/`, `docs/` | Package checks, reproduction checks and operational documentation |

Historical model runs used MPS, float16, deterministic generation, one-item
batches, Python 3.12.12, PyTorch 2.8.0 and Transformers 4.55.4. Immutable model
revisions are in [PROVENANCE.md](PROVENANCE.md). The current package dependencies
are release-tested versions, not a claim to recreate the historical software
stack. No model weights are distributed. A supported Apple GPU was used for the
historical runs; the exact hardware model is not established by the retained
identity metadata. Offline analysis requires only ordinary CPU resources.

Project code is [Apache-2.0](LICENSE); see [NOTICE](NOTICE) and
[data licensing](data/paper/LICENSES.md) for artifact and upstream boundaries.
Please cite [CITATION.cff](CITATION.cff). No DOI has been assigned here.
This is a research artifact, with [known release limitations](PUBLIC_RELEASE_AUDIT.md),
not a production-readiness claim.
