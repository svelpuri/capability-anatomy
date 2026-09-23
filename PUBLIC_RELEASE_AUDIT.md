# Public research snapshot audit

**Status: manuscript imported unchanged; repository remains private pending final release review.**
The authoritative archive supplies LaTeX with manual references, five PNG
figures/diagrams and submission notes. Each file is preserved byte-for-byte;
no separate bibliography file is required. Compilation and hosted checks are
tracked separately from source preservation.
The `v0.1.0-paper` tag has not been created.

- Source commit: `6e3ab58739857f46c971342a89c34d71445c003f`.
- Analysis commit: `9c90e4c`.
- Export date: 2026-09-23 UTC; exact timestamp in `provenance/identity.json`.
- Files included: curated research software and tests; 32 unchanged selected
  evidence files plus two public manifests; portable specificity and split-half
  analysis; expected numerical results; four SVG/PNG figure pairs; researcher
  documentation; the seven unchanged manuscript source files and their import
  manifest. The final inventory is `PUBLIC-MANIFEST.json`.
- Files intentionally excluded: unrelated platform source/product documents,
  old experiment families, benchmark text/raw answers, model weights, task
  caches, research/release traces not needed for numerical reproduction,
  original private review manifests and preservation archive, machine inventories,
  session material, temporary images/draft PDF, redundant bootstrap NPZ.
- Secret scan: Gitleaks 8.30.1 returned 202 candidates (exit 1): 200 record-group hashes and two deliberate negative-test fixtures. TruffleHog 3.97.6 returned one candidate (exit 0), exactly a recomputed source-file SHA-256. All are explicitly dispositioned as nonsecrets in `provenance/secret-scan.json`; zero unresolved or real-secret findings. Candidate network verification was disabled; no credentials were sent to providers. No detector suppressions were used.
- License scan: Apache-2.0 project LICENSE/NOTICE/metadata retained. Dataset
  provenance inspected against pinned upstream licenses. WikiText's pinned
  license card is inconsistent; no benchmark text is redistributed. See
  `data/paper/LICENSES.md` for source-by-source disposition.
- Local-path scan: pre-manuscript 186-file snapshot: zero private absolute paths, private review URLs, or private-key markers; broad case-insensitive terms reviewed in 75 files as code, tests, research token-count metadata, or scanner/inventory summaries.
- Model-weight scan: zero weight-format files in the pre-manuscript 186-file snapshot; binary figures inspected.
- Clean-checkout tests: independent detached clean checkout passed 640 tests at the initial preparation commit, zero skips. After the integrity-check regression, the independent detached source suite passed 642 tests in normal order and 642 in reverse order, zero skips. Canonical source and source-distribution suites each passed 640 in normal and reverse order, zero skips; installed wheel lifecycle and official Collector readback passed (104 receipt spans plus two exporter decisions).
- Analysis reproduction: all seven retained CSV tables, robustness JSON and regenerated NPZ matched historical
  bytes; all headline numbers and split-half distributions match. Scalar first
  bootstrap draw discrepancy is 8.9e-16; all 336 bands reconstructed.
- Artifact hashes: 32 selected historical files independently compared byte for
  byte; both original complete bundles passed their historical verifier after
  relocation. Public manifests explicitly limit verification to the selected
  evidence, not full historical execution/approval verification.
- Known limitations: tiny empirical sample, conditional general-damage proxy,
  discovery reuse, no validation interventions, duplicated 1.7B binding/full-call
  outcomes, no one-command historical inference replay. The manuscript also retains
  a historical aggregate Phase0B paragraph whose inputs are outside the primary
  Phase5 reproduction. Original submission notes contain an older affiliation
  instruction; see `paper/README.md`.

No private monorepo Git history is included.

No repository visibility change is authorized by this preparation. Publication
requires a separate explicit approval after final review.

## Review and verification scope

Independent review covered the public projection, source-versus-snapshot diff,
privacy/scanner dispositions, protocol/claim boundaries, lifecycle, operations,
test quality, scale and simplicity. R-PUB-01 found that the initial default paper
verifier did not check every result file. The fix verifies the complete snapshot
manifest before accepting evidence. Its positive and corruption regression tests
pass; removing the exact inventory-check call makes the corruption test fail,
and restoring it returns green. Independent re-review reproduced this result and closed R-PUB-01. Final tested source, analysis, test and lockfile bytes match this snapshot.

All four figure pairs were visually inspected. Rendering again from the saved
CSVs produced eight files byte-identical to the included PNG/SVG files.

The generated CSV CRLF and SVG path whitespace are preserved deliberately.
Three whitespace diagnostics in unchanged upstream source/test files remain;
no statistical data or runtime bytes were rewritten for cosmetic linting.

The full snapshot manifest includes itself only through Git's commit tree;
there is no circular self-hash. A hash manifest provides integrity against
accidental change when compared to the reviewed commit, not an external
signature or certification of historical execution. Numerical scripts emit
verification JSON; they are not claimed to be OTel-instrumented services.
Runtime telemetry is verified through the real collector gate.

Hosted Linux/macOS CI may still be running on the draft PR; local macOS gate
results do not constitute a hosted Linux pass. The workflow now includes a
paper job for integrity, numerical reproduction and two-pass pdflatex compilation
with shell escape disabled. Any unfinished hosted checks remain visible release
limitations.

The first hosted paper run exposed R-CI-01: an architecture-specific bootstrap
binary hash was incorrectly required for numerical reproduction. Downloaded
Linux versus macOS arrays differed by at most 4.44e-16. The verifier now runs the
existing independent numerical bootstrap checks and reports historical hash
equality separately. Regression tests accept single-ULP roundoff and reject
changed residuals and maximum deviations. Manuscript and analysis algorithms
are unchanged.
