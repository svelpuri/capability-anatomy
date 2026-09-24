# Fixed reason index

This index covers statically declared operation decisions and typed storage,
evidence and scoring refusals. It is generated from the distributed source by
`python scripts/update_reason_catalog.py`; source module names identify the
consumer to inspect. A reason is a classification, not exception text. Inspect
its correlated span status, duration, decision event and the command's safe
message to distinguish a refusal from a completed operation. Runtime plugins may
emit additional bounded reason names; they are responsible for documenting them.

See [operations](operations.md) for recovery, exit codes and resource limits,
and [observability](observability.md) for collector queries. Do not treat a
historical `complete` state as proof that the current invocation or telemetry
delivery succeeded. Keep the entire evidence directory and adjacent receipts.

| Reason | Source module |
| --- | --- |
| `adapter_execution_complete` | runtime |
| `adapter_execution_failed` | runtime |
| `authored_config_invalid` | config |
| `bounded_object_valid` | authored_inputs |
| `cleanup_failed` | block_bypass |
| `command_complete` | cli |
| `command_failed` | cli |
| `compatible_task_complete` | runner |
| `conformance_authorized` | orchestrator |
| `conformance_execution_complete` | orchestrator |
| `conformance_execution_failed` | orchestrator |
| `control_drift` | orchestrator |
| `core_evidence_consistent` | core_evidence |
| `credential_environment_missing` | credentials |
| `credential_environment_resolved` | credentials |
| `credential_reference_missing` | credentials |
| `delivery_interrupted` | cli, orchestrator |
| `discovery_ranking_frozen_before_validation` | orchestrator |
| `document_byte_limit` | cli |
| `document_cycle` | cli |
| `document_depth_limit` | cli |
| `document_format_unsupported` | cli |
| `document_node_limit` | cli |
| `document_requires_finite_numbers` | cli |
| `document_requires_json_values` | cli |
| `document_requires_regular_file` | cli |
| `document_root_must_be_object` | cli |
| `document_syntax_invalid` | cli |
| `document_unreadable` | authored_inputs, cli |
| `duplicate_object_key` | cli |
| `evidence_aggregate_contract_unsupported` | core_evidence |
| `evidence_aggregate_mismatch` | core_evidence |
| `evidence_artifact_coverage_mismatch` | core_evidence |
| `evidence_bundle_byte_limit` | core_evidence |
| `evidence_completion_invalid` | core_evidence |
| `evidence_digest_mismatch` | core_evidence |
| `evidence_directory_depth_limit` | core_evidence |
| `evidence_entry_limit` | core_evidence |
| `evidence_generated_byte_limit` | core_evidence |
| `evidence_json_byte_limit` | evidence_json |
| `evidence_json_depth_limit` | evidence_json |
| `evidence_json_duplicate_key` | evidence_json |
| `evidence_json_invalid` | evidence_json |
| `evidence_json_node_limit` | evidence_json |
| `evidence_json_nonfinite` | evidence_json |
| `evidence_json_object_required` | evidence_json |
| `evidence_manifest_invalid` | core_evidence |
| `evidence_path_invalid` | core_evidence |
| `evidence_publication_complete` | orchestrator |
| `evidence_report_mismatch` | core_evidence |
| `evidence_required_metrics_missing` | core_evidence |
| `evidence_score_invalid` | core_evidence |
| `evidence_scores_missing` | core_evidence |
| `evidence_structure_invalid` | core_evidence |
| `evidence_task_commit_mismatch` | core_evidence |
| `evidence_task_coverage_mismatch` | core_evidence |
| `evidence_task_incomplete` | core_evidence |
| `evidence_task_plan_invalid` | core_evidence |
| `evidence_trace_coverage_mismatch` | core_evidence |
| `evidence_trace_incomplete` | core_evidence |
| `evidence_trace_missing` | core_evidence |
| `evidence_trace_schema_invalid` | core_evidence |
| `experiment_body_failed` | block_bypass |
| `full_scan_authorized` | orchestrator |
| `gate_approved_source_unavailable` | cli |
| `gate_authorization_invalid` | cli |
| `gate_source_identity_changed` | cli |
| `generic_policy_unsupported` | cli, config |
| `hook_installation_failed` | block_bypass |
| `hooks_removed` | block_bypass |
| `identity_no_op_active` | orchestrator |
| `identity_no_op_hooks_removed` | orchestrator |
| `inline_credential_refused` | config |
| `interrupted` | cli, errors |
| `invalid_configuration` | cli, errors |
| `invalid_evidence` | cli, errors |
| `invocation_failed` | orchestrator |
| `invocation_receipt_unwritable` | cli, orchestrator |
| `memory_budget_exceeded` | runner |
| `memory_budget_unobservable` | runner |
| `metric_scoring_failed` | phase5 |
| `model_execution_failed` | executor |
| `model_loading_selector_refused` | cli, huggingface |
| `object_keys_must_be_strings` | cli |
| `observation_complete` | executor |
| `observation_error_budget_exceeded` | runner |
| `otlp_headers_invalid` | cli, run_telemetry |
| `owned_run_failed` | secure_fs |
| `parse_failure_recorded` | executor |
| `parse_failure_returned` | executor |
| `phase5_resume_trace_available` | orchestrator |
| `phase5_resume_trace_invalid` | orchestrator |
| `phase5_resume_trace_missing` | cli, orchestrator |
| `plugin_contract_accepted` | discovery |
| `plugin_contract_refused` | discovery |
| `ranking_derived_from_discovery_only` | orchestrator |
| `regular_file_read` | authored_inputs |
| `repeated_baseline_no_op_and_matched_controls_complete` | orchestrator |
| `required_artifacts_atomically_written_and_digest_bound` | orchestrator |
| `resolved_credential_persistence_refused` | credentials |
| `resource_budget_available` | runner |
| `result_independent_campaign_plan_persisted` | orchestrator |
| `retry_budget_exceeded` | runner |
| `run_ownership_acquired` | secure_fs |
| `run_ownership_busy` | cli, secure_fs |
| `run_ownership_released` | secure_fs |
| `run_publication_failed` | orchestrator |
| `runtime_failure` | cli, errors |
| `runtime_plugin_execution_complete` | orchestrator |
| `runtime_plugin_execution_failed` | orchestrator |
| `safe_model_loading_refused` | huggingface |
| `safetensors_eager_model_loaded` | huggingface |
| `scan_execution_failed` | orchestrator |
| `schema_and_semantics_valid` | config |
| `schema_unavailable` | config |
| `schema_validation_failed` | config |
| `scoped_bypass_active` | block_bypass |
| `score_denominator_invalid` | reduction |
| `score_failure_recorded` | executor |
| `score_fraction_incomplete` | reduction |
| `score_fraction_inconsistent` | reduction |
| `score_fraction_mixed` | reduction |
| `score_list_empty` | reduction |
| `score_value_invalid` | reduction |
| `semantic_validation_failed` | config |
| `source_export_completed` | export_standalone |
| `source_export_refused` | export_standalone |
| `storage_artifact_byte_limit` | secure_fs |
| `storage_filesystem_unsupported` | cli, secure_fs |
| `storage_fork_context_refused` | secure_fs |
| `storage_frozen_input_changed` | secure_fs |
| `storage_hardlink_refused` | secure_fs |
| `storage_io_failed` | secure_fs |
| `storage_name_invalid` | secure_fs |
| `storage_nested_ownership` | secure_fs |
| `storage_not_directory` | secure_fs |
| `storage_not_regular` | secure_fs |
| `storage_outside_root` | secure_fs |
| `storage_parent_traversal` | secure_fs |
| `storage_path_missing` | secure_fs |
| `storage_path_refused` | secure_fs |
| `storage_permission_denied` | secure_fs |
| `storage_readonly_context` | secure_fs |
| `storage_recovery_required` | secure_fs |
| `storage_symlink_refused` | secure_fs |
| `storage_temporary_collision` | secure_fs |
| `storage_temporary_invalid` | secure_fs |
| `storage_temporary_reclaimed` | secure_fs |
| `storage_worker_context_required` | secure_fs |
| `task_committed` | runner |
| `task_execution_complete` | orchestrator |
| `task_failed` | runner |
| `task_incomplete` | runner |
| `task_interrupted` | runner |
| `telemetry_capacity_exceeded` | cli, run_telemetry |
| `telemetry_delivery_incomplete` | cli, orchestrator, run_telemetry |
| `telemetry_disabled` | cli, orchestrator |
| `unsupported_adapter` | cli, errors |
| `unsupported_plugin` | cli, errors |
| `validation_failed` | block_bypass |
| `versioned_metric_scores_recorded` | phase5 |
| `wall_budget_exceeded` | runner |
