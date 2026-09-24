# Build and qualify a source release

Use a reviewed checkout and a new destination outside it. Distribute only the
reviewed source selection and its verified artifacts.
A successful build is not publication approval or scientific evidence.

```sh
python scripts/export_standalone.py /new/capability-anatomy-source
cd /new/capability-anatomy-source
uv build --out-dir /new/capability-anatomy-artifacts
python scripts/verify_release.py --artifacts /new/capability-anatomy-artifacts --work-dir /new/acceptance
```

Direct `uv build` and `python -m build` use the same curated file selection as
`export_standalone.py` for wheels and source distributions. Symlinks, unrelated
hardlinks, special files and forbidden names inside selected trees refuse the
build. Unselected research files are not archived. Build metadata must be static; readme and license files must be selected source files, and custom metadata/build hooks and secondary source projections are refused. Editable installs intentionally
use the developer checkout. Hatchling owns archive construction and license
metadata; `EXPORT-MANIFEST.json` identifies selected bytes. Inspect the snapshot,
LICENSE, NOTICE, dependency licenses and model/dataset provenance before release.

The shared CI action executes the full available source suite in ordinary and
reverse order on Linux and macOS, followed by wheel install/run/resume/verify/
uninstall/reinstall, the extracted-sdist suite in both orders and official
collector readback. The artifact upload fails if distributions are missing.

The core synthetic example is self-contained. Phase 5 commands are advanced,
optional research workflows: `freeze-phase5-records`, `validate-phase5` and
`conform-phase5` require your own record-source document, immutable model and
protocol inputs, and separately reviewed authorization where applicable. This
release does not include an approved real-model campaign. Do not substitute fake
approval files or nonexistent paths to make the commands run. See the versioned
schemas and API tests as format references, and [research limits](research-limits.md)
for what those tests demonstrate.
