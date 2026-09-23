"""Public governed execution with actual Git source identity and a tiny local model."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from capability_anatomy.cli import _example
from capability_anatomy.discovery import PluginDiscovery
from capability_anatomy.models.plugins.huggingface import ARCHITECTURE_PROFILES
from capability_anatomy.phase5_protocol import sha256_file
from capability_anatomy.review_policy import AUTHORIZATION_V2, PROTOCOL_V2, REVIEW_POLICY_VERSION
from capability_anatomy.serialization import canonical_sha256
from test_release_model_profiles import real_profile, isolated_torch_state


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def _freeze_protocol(tmp_path, model_source):
    plan = tmp_path / "records.json"
    prompts = tmp_path / "prompts.json"
    value = {
        "schema_version": PROTOCOL_V2,
        "status": "frozen_pre_inference",
        "experiment": {"id": "portable-conformance", "seed": 20260904},
        "model": {"plugin": "huggingface.causal-lm", "version": "2", "source": model_source, "revision": "a" * 40, "trust_remote_code": False, "architecture_profile": "qwen3-dense-v2", "architecture_profile_sha256": ARCHITECTURE_PROFILES["qwen3-dense-v2"].sha256},
        "plugins": [
            {"role": role, "name": name, "version": "2" if role == "evaluation" else "1", "api_version": "capability-anatomy/plugin-api/v1", "capabilities": [capability]}
            for role, name, capability in (
                ("model", "huggingface.causal-lm", "execution.autoregressive_text"),
                ("intervention", "huggingface.block-bypass", "intervention.block_bypass"),
                ("evaluation", "reference.phase5-qwen-retention", "evaluation.phase5"),
                ("dataset", "reference.phase5-records", "dataset.frozen_roles"),
                ("runtime", "builtin.local", "runtime.local"),
            )
        ],
        "runtime": {"device": "cpu", "dtype": "float32", "deterministic": True, "do_sample": False, "thinking": False, "batch_size": 1, "context_length": 256, "max_new_tokens": 3, "repetitions": 2, "warmup_runs": 1, "synchronization": "adapter_before_after", "memory_semantics": "synchronized_observation_boundary", "randomized_record_order": True, "randomized_component_order": True, "record_seed_derivation": "sha256(seed:records)", "component_seed_derivation": "sha256(seed:components)"},
        "dataset": {"manifest": {"path": "records.json", "sha256": sha256_file(plan)}, "record_plan": {"path": "records.json", "sha256": sha256_file(plan)}, "prompt_templates": {"path": "prompts.json", "sha256": sha256_file(prompts)}, "scorers": {"path": "prompts.json", "sha256": sha256_file(prompts)}, "final_holdout": "forbidden"},
        "metrics": {
            **{metric: {"role": "target", "direction": "higher_is_better", "damage_formula": "baseline_minus_condition"} for metric in ("tool_selection", "argument_binding", "full_call", "abstention")},
            **{metric: {"role": "collateral", "direction": "higher_is_better", "damage_formula": "baseline_minus_condition"} for metric in ("structured_reasoning", "instruction_format")},
            "perplexity": {"role": "collateral", "direction": "lower_is_better", "damage_formula": "condition_minus_baseline_over_baseline"},
        },
        "controls": {"no_op_hook": True, "repeated_baseline_positions": ["beginning", "middle", "end"], "matched_random_components": 1, "placement": {"middle_after_fraction": 0.5}, "random_seed_derivation": "sha256(seed:controls)"},
        "execution_semantics": {"no_op_hook": "identity_forward_hook_on_first_frozen_component", "baseline_reference": "mean_of_beginning_middle_end", "control_drift": "maximum_absolute_repeated_or_no_op_deviation", "matched_random": "seeded_single_component_duplicate_reproducibility", "task_retries": "retries_after_first_attempt", "wall_time": "cumulative_active_execution_across_resume", "memory": "synchronized_current_runtime_allocation_observation_not_peak", "validation": "freeze_discovery_ranking_then_open_validation_once"},
        "budgets": {"max_observation_errors": 0, "max_wall_seconds": 28800, "max_memory_observation_bytes": 17179869184, "max_task_retries": 2},
        "candidate_rule": {"target_absolute_damage_max": 0.05, "collateral_absolute_damage_max": 0.1, "perplexity_relative_damage_max": 0.1, "minimum_drift_margin": 0.025, "validation_once": True},
        "stop_rules": ["identity_mismatch", "missing_metric", "duplicate_observation", "budget_exceeded", "hook_cleanup_failure", "control_drift"],
        "retention": {"prompts": "sha256_only", "raw_outputs": "sha256_only"},
        "required_artifacts": ["protocol.json", "record-plan.json", "configuration.json", "compatibility.json", "topology.json", "component-scan-order.json", "discovery-task-plan.json", "validation-task-plan.json", "observations.jsonl", "metrics.json", "failures.jsonl", "controls.json", "provenance.json", "candidates.json", "validation.json", "trace.json", "evidence-manifest.json", "report.json", "report.md"],
        "claim_boundary": "temporary component sensitivity only; no removal, compression, or publication claim",
    }
    value["review_policy"] = {
        "schema_version": REVIEW_POLICY_VERSION, "policy_id": "fork-research-review",
        "allowed_locations": [{"origin": "https://review.example.org", "path_prefix": "/independent/anatomy/changes/"}],
    }
    names = {"model": "huggingface.causal-lm", "intervention": "huggingface.block-bypass",
             "evaluation": "reference.phase5-qwen-retention", "dataset": "reference.phase5-records",
             "runtime": "builtin.local"}
    value["plugins"] = []
    for role, name in names.items():
        plugin = PluginDiscovery().resolve(role, name, required_capabilities=frozenset(), required_methods=()).plugin
        value["plugins"].append({"role": role, "name": name, "version": plugin.version,
                                 "api_version": plugin.api_version, "capabilities": sorted(plugin.capabilities)})
    for key, name in (("manifest", "dataset.json"), ("record_plan", "records.json"),
                      ("prompt_templates", "prompts.json"), ("scorers", "../../src/capability_anatomy/evaluations/plugins/phase5.py")):
        value["dataset"][key] = {"path": name, "sha256": sha256_file(tmp_path / name)}
    _write(tmp_path / "protocol.json", value)
    return value


# This subprocess imports the copied source. The load wrapper only counts calls;
# the admitted control executes the actual adapter, safetensors model and cache.
_PUBLIC_COMMAND = r"""
import json,sys
from pathlib import Path
import torch
from capability_anatomy.cli import main
from capability_anatomy.execution import orchestrator
from capability_anatomy.models.plugins.huggingface import HuggingFaceCausalLMAdapter
assert Path(orchestrator.__file__).resolve().is_relative_to(Path.cwd()/"src")
torch.set_num_threads(1)
original=HuggingFaceCausalLMAdapter.load
calls=[]
def load(self,*args,**kwargs):
    calls.append("load")
    return original(self,*args,**kwargs)
HuggingFaceCausalLMAdapter.load=load
result=main([sys.argv[1],"--config","configs/experiments/experiment.json"])
Path("../last-invocation.json").write_text(json.dumps({"exit":result,"model_loads":len(calls),"imported":orchestrator.__file__}))
raise SystemExit(result)
"""


@pytest.mark.parametrize("real_profile", ["Qwen3"], indirect=True)
def test_public_portable_review_policy_admits_tiny_conformance_and_refuses_identity_defects(real_profile, tmp_path):
    _family, _adapter, _provider, loaded, spec, _runtime = real_profile
    loaded.model.generation_config.eos_token_id = None
    loaded.model.save_pretrained(spec.source, safe_serialization=True)
    lab = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(lab / "src/capability_anatomy", checkout / "src/capability_anatomy", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(lab / "schemas", checkout / "schemas")
    # Freeze two explicit test-only implementation choices before source review:
    # three generated tokens keep this software gate cheap; CPU memory uses actual
    # current process RSS rather than claiming an unavailable CUDA allocation.
    scorer = checkout / "src/capability_anatomy/evaluations/plugins/phase5.py"
    scorer_source = scorer.read_text()
    assert scorer_source.count("max_new_tokens: int = 128") == 1
    scorer.write_text(scorer_source.replace("max_new_tokens: int = 128", "max_new_tokens: int = 3"))
    adapter_source = checkout / "src/capability_anatomy/models/plugins/huggingface.py"
    source = adapter_source.read_text()
    marker = "    def memory_bytes(self, loaded: LoadedModel) -> int | None:\n"
    assert source.count(marker) == 1
    source = source.replace(marker, marker + "        if next(loaded.model.parameters()).device.type == 'cpu':\n            import psutil\n            return psutil.Process().memory_info().rss\n")
    adapter_source.write_text(source)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(lab / name, checkout / name)
    selected = checkout / "configs/experiments"
    _example(selected)
    config_path = selected / "experiment.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = {"id": "portable-conformance", "seed": 20260904}
    config["model"] = {"plugin": "huggingface.causal-lm", "source": spec.source, "revision": "a" * 40,
                       "parameters": {"architecture_profile": "qwen3-dense-v2", "dtype": "float32"}}
    config["runtime"].update(repetitions=2, parameters={"device": "cpu", "context_length": 256, "phase5_protocol": "protocol.json"})
    config["capability"] = {"evaluation_plugin": "reference.phase5-qwen-retention", "suite_version": "2",
                            "target_metrics": ["tool_selection", "argument_binding", "full_call", "abstention"],
                            "collateral_metrics": ["structured_reasoning", "instruction_format", "perplexity"]}
    config["dataset"]["provider"] = "reference.phase5-records"
    config["intervention"]["plugin"] = "huggingface.block-bypass"
    config["output"].update(directory=str(tmp_path / "run"), retain_raw_outputs=False)
    config["policy"]["file"] = "approval.json"
    _write(config_path, config)
    kinds = ("reasoning", "simple", "abstention", "format", "perplexity")
    partitions = {partition: [{"source_id": partition + "-" + kind, "group_key": partition + "-" + kind,
                              "kind": kind, "split": partition} for kind in kinds]
                  for partition in ("discovery", "validation")}
    rows = [row for partition in partitions.values() for row in partition]
    _write(selected / "records.json", {"schema_version": "capability-anatomy/phase5-record-plan/v1",
                                       "partitions": partitions, "rows_sha256": canonical_sha256(rows)})
    _write(selected / "prompts.json", {"system": "hello"})
    expected = {"reasoning": "42", "simple": [{"lookup": {"value": ["42"]}}], "abstention": [],
                "format": {"kind": "exact_prefix", "value": "OK"}, "perplexity": None}
    records = {}
    for partition, plan_rows in partitions.items():
        records[partition] = []
        for row in plan_rows:
            kind = row["kind"]
            metadata = {"kind": kind, "group_id": row["group_key"]}
            if kind == "simple": metadata["scoring_schema"] = {"argument_names": ["value"], "required": ["value"]}
            records[partition].append({"id": row["source_id"], "partition": partition,
                "input": {"text": "hello world hello"} if kind == "perplexity" else {"messages": [{"role": "user", "content": "hello world"}], "tools": None},
                "expected": expected[kind], "metadata": metadata})
    _write(selected / "dataset.json", {"schema_version": "capability-anatomy/phase5-dataset/v1",
                                       "record_plan": "records.json", "partitions": records})
    protocol = _freeze_protocol(selected, spec.source)
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "review@example.invalid")
    _git(checkout, "config", "user.name", "Portable Review Test")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "reviewed source and explicit fork policy")
    commit = _git(checkout, "rev-parse", "HEAD")
    env = {**os.environ, "PYTHONPATH": str(checkout / "src"), "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    derive = subprocess.run([sys.executable, "-c", "from pathlib import Path; from capability_anatomy.config import load_experiment_config; from capability_anatomy.execution.orchestrator import _governed_source_identity; from capability_anatomy.phase5_protocol import validate_phase5_protocol; p=Path('configs/experiments/experiment.json').resolve(); q=p.parent/'protocol.json'; print(*_governed_source_identity(p, '" + commit + "', load_experiment_config(p),q,validate_phase5_protocol(q)))"],
                            cwd=checkout, env=env, text=True, capture_output=True, check=True)
    approved, current = derive.stdout.split()
    assert approved == current
    approval = {"schema_version": AUTHORIZATION_V2, "status": "approved", "conformance_authorized": True,
                "full_scan_authorized": False, "protocol_sha256": sha256_file(selected / "protocol.json"),
                "approved_commit": commit, "approved_source_sha256": approved,
                "review_policy_sha256": canonical_sha256(protocol["review_policy"]),
                "review_url": "https://review.example.org/independent/anatomy/changes/17#approved"}

    def invoke(command, authorization):
        _write(selected / "approval.json", authorization)
        result = subprocess.run([sys.executable, "-c", _PUBLIC_COMMAND, command], cwd=checkout, env=env,
                                text=True, capture_output=True, timeout=60)
        receipt = json.loads((tmp_path / "last-invocation.json").read_text())
        assert receipt["exit"] == result.returncode, result.stderr
        assert receipt["imported"].startswith(str(checkout / "src"))
        return result, receipt

    positive, receipt = invoke("conform-phase5", approval)
    assert positive.returncode == 0, positive.stderr
    assert receipt["model_loads"] == 1
    evidence = json.loads(positive.stdout)
    assert evidence["prefill_decode_verified"] is evidence["cleanup_verified"] is True
    assert evidence["normal_output_sha256"] == evidence["restored_output_sha256"]
    assert evidence["normal_invocations"] > 1 and evidence["bypass_identity_invocations"] > 1
    cases = [("full_scope", "run", {}, "gate_authorization_invalid"),
             ("wrong_host", "conform-phase5", {"review_url": "https://other.example.org/independent/anatomy/changes/17"}, "gate_authorization_invalid"),
             ("double_encoded_parent", "conform-phase5", {"review_url": "https://review.example.org/independent/anatomy/changes/%252e%252e/17"}, "gate_authorization_invalid"),
             ("prefix_index", "conform-phase5", {"review_url": "https://review.example.org/independent/anatomy/changes/"}, "gate_authorization_invalid"),
             ("policy_digest", "conform-phase5", {"review_policy_sha256": "0" * 64}, "gate_authorization_invalid"),
             ("fabricated_commit", "conform-phase5", {"approved_commit": "0" * 40}, "gate_approved_source_unavailable"),
             ("source_digest", "conform-phase5", {"approved_source_sha256": "0" * 64}, "gate_source_identity_changed")]
    for case, command, changes, reason in cases:
        result, receipt = invoke(command, {**approval, **changes})
        assert result.returncode == 2, (case, result.stdout, result.stderr)
        assert receipt["model_loads"] == 0, case
        assert json.loads(result.stderr)["error"] == reason, case
        traces = [json.loads(path.read_text()) for path in (tmp_path / "run/trace-failures").glob("*/trace.json")]
        latest = max(traces, key=lambda trace: max(span["end_time_unix_nano"] for span in trace["spans"]))
        gate, = [span for span in latest["spans"] if span["name"] == "capability_anatomy.gate5a.authorization"]
        assert gate["attributes"]["capability_anatomy.reason"] == reason, case
        assert gate["status"] == "ERROR", case
    positive, receipt = invoke("conform-phase5", approval)
    assert positive.returncode == 0, positive.stderr
    assert receipt["model_loads"] == 1

    # Full execution is explicitly authorized only for this separately reviewed
    # local fixture. Its random weights establish no competence/retention claim.
    full_approval = {**approval, "full_scan_authorized": True}
    completed, receipt = invoke("run", full_approval)
    assert completed.returncode == 0, completed.stderr
    assert receipt["model_loads"] == 1
    for command in ("verify", "report"):
        result = subprocess.run([sys.executable, "-m", "capability_anatomy.cli", command, "--evidence", str(tmp_path / "run")],
                                cwd=checkout, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (command, result.stderr)
    trace_path = tmp_path / "run/trace.json"
    trace = json.loads(trace_path.read_text())
    assert len(trace["external_parent_span_ids"]) == 2
    assert trace["execution_parent_span_id"] in trace["external_parent_span_ids"]
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert all(metric["sample_uncertainty_status"] == "not_estimated_records_not_assumed_independent" for metric in report["baseline"].values())
    # A declared anchor may not be fabricated, omitted, or used to hide a cycle.
    from capability_anatomy.execution.phase5_campaign import _validate_trace_segment_graph
    from capability_anatomy.errors import InvalidEvidenceError
    _validate_trace_segment_graph(trace, latest=True)
    for defect in ("missing_anchor", "wrong_execution_parent", "cycle", "wrong_span_trace"):
        tampered = json.loads(json.dumps(trace))
        if defect == "missing_anchor": tampered["external_parent_span_ids"].pop()
        elif defect == "wrong_execution_parent": tampered["execution_parent_span_id"] = "f" * 16
        elif defect == "cycle": tampered["spans"][0]["parent_span_id"] = tampered["spans"][0]["span_id"]
        else: tampered["spans"][0]["trace_id"] = "f" * 32
        with pytest.raises(InvalidEvidenceError, match="trace graph"):
            _validate_trace_segment_graph(tampered, latest=True)
    resumed, _receipt = invoke("run", full_approval)
    assert resumed.returncode == 0, resumed.stderr
    result = subprocess.run([sys.executable, "-m", "capability_anatomy.cli", "verify", "--evidence", str(tmp_path / "run")],
                            cwd=checkout, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

    # Simulate loss of whole-invocation trace after task commit (as with a hard
    # kill). Task checkpoints remain, but this alpha refuses unsupported recovery
    # before loading a model instead of publishing an unverifiable campaign.
    trace_bytes = trace_path.read_bytes()
    trace_path.unlink()
    preserved = {str(path.relative_to(tmp_path / "run")): path.read_bytes()
                 for path in (tmp_path / "run").rglob("*") if path.is_file()}
    refused, receipt = invoke("run", full_approval)
    assert refused.returncode == 2 and receipt["model_loads"] == 0
    assert json.loads(refused.stderr)["error"] == "phase5_resume_trace_missing"
    assert preserved == {str(path.relative_to(tmp_path / "run")): path.read_bytes()
                         for path in (tmp_path / "run").rglob("*") if path.is_file()}
    trace_path.write_bytes(trace_bytes)

    # Mutate actual governed source and commit it: current source must still equal
    # the previously approved snapshot, even when HEAD is a descendant commit.
    source = checkout / "src/capability_anatomy/review_policy.py"
    source.write_text(source.read_text() + "\n# changed after source approval\n")
    _git(checkout, "add", str(source.relative_to(checkout)))
    _git(checkout, "commit", "-qm", "unreviewed source change")
    result, receipt = invoke("conform-phase5", approval)
    assert result.returncode == 2 and receipt["model_loads"] == 0
    assert json.loads(result.stderr)["error"] == "gate_source_identity_changed"
