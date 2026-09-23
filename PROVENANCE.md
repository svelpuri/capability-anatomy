# Research artifact provenance

- Private source release commit: `6e3ab58739857f46c971342a89c34d71445c003f`.
- Paper-analysis commit: `9c90e4c` (full object ID in `provenance/identity.json`).
- Original curated software export: 108 files, tree SHA-256
  `84476a1dbc168ed11fb2e7aa9eabf2dfea45251bcf925148bd7df212c04c5741`.
  `provenance/source-export.json` describes those original selected bytes,
  before paper-facing README and dependency-group changes. It is provenance,
  not an integrity manifest for this whole research snapshot.
- Each `artifacts/*/MANIFEST.json` records the current public evidence files,
  byte counts, original hashes and hash of the excluded historical manifest.
- Qwen/Qwen3-0.6B revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Qwen/Qwen3-1.7B revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Original 1.7B preservation archive SHA-256:
  `2a67da263a35f34989311ca0620a5301ee1f9f98000f18a0911e460d22a6ed45`.
  Its 101 file contents were verified against the retained bundle before public
  projection. The archive is not distributed because it includes a private
  review locator. Its hash is a preservation reference, not a claim that an
  external reader can verify the undistributed archive.
- Regenerable bootstrap NPZ SHA-256:
  `a68048c9875ae4a11e23baf791d5134dfa2b30d2626e37b1644e1fb8660256d7`.
- Authoritative manuscript archive SHA-256:
  `9d46c62f8fe759d81f54ba65f05c529a0fe588b358ddd985fa895bfe84579d4c`.
  The six manuscript/figure files are unchanged under `paper/`.
  `paper/SOURCE-MANIFEST.json` records their original hashes and the excluded
  internal submission notes, whose author-check instructions are obsolete.
  Manual references remain in `main.tex`.
  No reference PDF was present in the supplied archive.
- Final analysis/release tag: **not created**. The intended `v0.1.0-paper` tag
  awaits final release review and completed hosted checks.
  No tag hash is fabricated or inferred from a preparation commit.

The public tree is a selected source snapshot with new paper-facing documentation
and portable analysis entry points. Statistical arithmetic and resampling order
are retained; paths are relative, the unrelated earlier-experiment analysis is
removed, and SVG timestamps are suppressed. Seven numeric CSVs and the regenerated
bootstrap archive matched historical bytes exactly on the recorded environment.
The software runtime source remains byte-identical to the selected release.

`PUBLIC-MANIFEST.json` binds the prepared snapshot's selected file bytes (excluding
itself). Git identifies the commit carrying that manifest without requiring a
self-referential hash inside the same commit. The initial target repository had
one README-only commit. No private monorepo Git history is included.

See [data/paper/LICENSES.md](data/paper/LICENSES.md) for pinned dataset sources,
rights boundaries and the intentionally omitted benchmark text. Configuration
identities record MPS/float16, historical libraries and frozen scorer hashes;
they do not promise that current dependency versions reproduce fresh inference.
