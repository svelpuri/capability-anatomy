# Security

This alpha supports trusted local users and trusted installed plugins on macOS/Linux. Evidence files and model repositories are treated as untrusted data. Plugins are executable Python and have the same privileges as the caller.

Controls include descriptor-based evidence containment, process ownership, bounded authored input parsing, duplicate plugin rejection before import, explicit environment credential references, sanitized built-in exception telemetry, safetensors-only model loading and qualified attention implementations.

Advisory locks do not defend against a malicious same-user process that ignores them or renames active directories. Checksums are integrity records, not signatures. Arbitrary plugin code can bypass application controls. Native kernels are not forcibly preempted by soft budgets. Public examples contain no credentials or pretrained weights.

Report a suspected vulnerability privately through the repository's GitHub security reporting mechanism when enabled. Until it is enabled, do not post working exploits or credentials in a public issue; contact the repository owner privately. Include the affected version, minimal reproduction and expected/actual behavior. No response-time guarantee is currently offered.

## Building a source snapshot

Run the standalone exporter on reviewed source in caller-owned, trusted directories. Source and destination ancestors must remain stable while the exporter initially acquires its directory handles. POSIX directory creation does not atomically return an open handle; another process able to replace the new directory before that handle is acquired is outside the supported boundary. The exporter refuses an observed nonempty destination, but this does not prove creation identity against an empty-directory replacement. After acquisition, pinned descriptors bind inventory, source reads, payload writes and the final manifest to the acquired directories. Use a private build workspace and avoid concurrent cleanup or renaming there. The exporter is not a sandbox for untrusted build code or a hostile same-user process.
