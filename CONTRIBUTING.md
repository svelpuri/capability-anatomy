# Contributing

Use Python 3.12 and a dedicated environment. Run `uv sync --frozen --extra dev` and `uv run pytest`. Tests use local fixtures and tiny models; they do not need pretrained downloads.

Keep interventions scoped and prove restoration after success and failure. Changes to scoring or architecture contracts need new versions. Preserve historical evidence rather than rewriting results. Add a working positive control and a corresponding negative for security changes; demonstrate that restoring the exact defect makes the regression test fail.

Record decisions as named OpenTelemetry events with counters, safe identifiers and duration. Never record raw prompts, credentials or exception text in telemetry. Verify installed artifacts, including the extracted sdist, rather than relying only on an editable checkout.

Use small pull requests that explain the concrete behavior change, test evidence and remaining claim limits. Keep configuration and evidence fixtures generic and free of personal or organization-internal data.

Contributions intentionally submitted for inclusion in this project are licensed under Apache-2.0 unless explicitly stated otherwise, as described in section 5 of [LICENSE](LICENSE). Preserve applicable third-party notices and identify any material governed by different terms.
