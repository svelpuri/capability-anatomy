# Dataset and model provenance

The paper artifacts contain recorded numeric scores, content hashes, example identifiers, frozen metadata, and analysis results. They do not distribute model weights, benchmark prompt text, raw model answers, or the complete historical execution/approval bundle. Original retained files included in each projection are byte-identical; the public MANIFEST is newly generated and declares this limited scope.

The historical Phase 5 dataset has 50 discovery and 50 validation records. Each partition contains 20 BFCL simple tool-call examples, 10 BFCL irrelevance examples, 10 BIG-bench reasoning examples, five WikiText documents, and five repository-authored format examples. Discovery interventions and a validation baseline were executed; no validation interventions occurred. Dataset manifest SHA-256: `8e95c119a1f509eaa694b8813f77bc39643e16922d1cc7584813035a54a47648`. Record-plan SHA-256: `59edc6c23596c9602d97db2e7c4baddffb166dce29213fcc18fb0d626ea7f2a4`. Historical prompt-template file SHA-256: `5bd28c2c7d1b20299c469f0811db5a540c914791d168573447ea4d41e62a4e6e`.

| Input | Frozen upstream revision | Upstream license evidence |
|---|---|---|
| BFCL V4 | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | [Pinned Apache 2.0 license](https://raw.githubusercontent.com/ShishirPatil/gorilla/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/LICENSE) |
| BIG-bench | `092b196c1f8f14a54bbc62f24759d43bde46dd3b` | [Pinned Apache 2.0 license](https://raw.githubusercontent.com/google/BIG-bench/092b196c1f8f14a54bbc62f24759d43bde46dd3b/LICENSE) |
| Salesforce WikiText, wikitext-2-raw-v1 test | `b08601e04326c79dfdd32d625aee71d232d685c3` | [Pinned dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md): metadata lists CC-BY-SA-3.0/GFDL while its licensing prose says CC-BY-SA-4.0. This discrepancy is not resolved by the project. Benchmark text is omitted. |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | [Pinned model card](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md) identifies Apache 2.0 |
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | [Pinned model card](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md) identifies Apache 2.0 |

GSM8K, SQuAD and RULER occur in the broader historical input inventory; they are not records in the frozen Phase 5 dataset. Do not describe that inventory as the set of evaluated Phase 5 tasks.

Published IDs and hashes support numerical reconstruction and input identity comparison. They do not reconstruct original prompts or raw answers. Obtaining upstream datasets and selecting rows alone does not prove byte-identical historical prompt rendering, scorer equivalence, or equivalent inference on a new software stack. Frozen relative locators inside historical configuration/protocol files document original layout and are not executable public paths. A future model rerun needs the exact prompt construction, licensed upstream inputs, historical scorer semantics and an explicitly reviewed execution configuration.

The project Apache 2.0 license does not relicense third-party datasets, weights, or dependencies. No additional rights are claimed for those inputs.

## Redistribution inventory

| Distributed category | Original source bytes? | License/scope |
|---|---|---|
| Research code, configurations, new manifests, numerical analyses and figures | Project-authored files or derived research outputs | Apache-2.0 project license; no third-party text is relicensed |
| Observation scores and timing/identity fields | Yes, the selected historical research output files are byte-preserved | Project-generated measurements; prompt/output content is represented only by hashes |
| Record IDs, content hashes and benchmark revision identity | Yes in the retained record plan/protocol | Factual provenance; upstream licenses above remain applicable to source material |
| Benchmark prompts, expected answers, documents and raw generated answers | No | Obtain separately under the applicable upstream terms |
| Model weights and tokenizer files | No | Obtain separately under the pinned model's upstream terms |
| Package dependencies | No vendored dependency source | Installed separately; their upstream licenses apply |

The historical library versions and exact dataset/input hashes remain in the
artifact metadata. No ownership certification or broader rights over upstream
content is asserted by this inventory.
