from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Any, Mapping
from opentelemetry.trace import Status, StatusCode

from .errors import InvalidConfigurationError, UnsupportedPluginError
from .protocols import CORE_API_VERSION
from .telemetry import OperationTelemetry


ENTRY_POINT_GROUPS = {
    "model": "capability_anatomy.models",
    "evaluation": "capability_anatomy.evaluations",
    "dataset": "capability_anatomy.datasets",
    "intervention": "capability_anatomy.interventions",
    "runtime": "capability_anatomy.runtimes",
}

BUILTIN_PLUGINS: Mapping[str, Mapping[str, str]] = {
    "model": {
        "builtin.synthetic-model": "capability_anatomy.models.plugins.synthetic:SyntheticModelAdapter",
        "huggingface.causal-lm": "capability_anatomy.models.plugins.huggingface:HuggingFaceCausalLMAdapter",
        "reference.qwen3-transformers": "capability_anatomy.models.plugins.qwen3:Qwen3Adapter",
    },
    "evaluation": {
        "reference.synthetic-exact-match": "capability_anatomy.evaluations.plugins.synthetic:SyntheticExactMatchSuite",
        "reference.chat-exact-match": "capability_anatomy.evaluations.plugins.chat_exact:ChatExactMatchSuite",
        "reference.phase5-qwen-retention": "capability_anatomy.evaluations.plugins.phase5:Phase5EvaluationSuite",
    },
    "dataset": {
        "reference.synthetic-dataset": "capability_anatomy.datasets.synthetic:SyntheticDatasetProvider",
        "reference.phase5-records": "capability_anatomy.datasets.phase5:Phase5DatasetProvider",
    },
    "intervention": {
        "builtin.synthetic-intervention": "capability_anatomy.models.plugins.synthetic:SyntheticInterventionProvider",
        "huggingface.block-bypass": "capability_anatomy.models.plugins.huggingface:HuggingFaceBlockBypassProvider",
        "reference.qwen3-block-bypass": "capability_anatomy.models.plugins.qwen3:Qwen3BlockBypassProvider",
    },
    "runtime": {
        "builtin.local": "capability_anatomy.runtime:LocalRuntimePlugin",
    },
}


@dataclass(frozen=True)
class ResolvedPlugin:
    role: str
    plugin: Any
    distribution: str
    distribution_version: str

    def identity(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.plugin.name,
            "version": self.plugin.version,
            "api_version": self.plugin.api_version,
            "capabilities": sorted(self.plugin.capabilities),
            "distribution": self.distribution,
            "distribution_version": self.distribution_version,
        }


def _load_reference(reference: str) -> Any:
    module_name, attribute = reference.split(":", 1)
    return getattr(import_module(module_name), attribute)


class PluginDiscovery:
    def __init__(self, telemetry: OperationTelemetry | None = None) -> None:
        self._telemetry = telemetry or OperationTelemetry.create("plugin_discovery")

    def resolve(
        self,
        role: str,
        name: str,
        *,
        required_capabilities: frozenset[str],
        required_methods: tuple[str, ...],
    ) -> ResolvedPlugin:
        with self._telemetry.tracer.start_as_current_span(
            "capability_anatomy.plugin.resolve",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                resolved = self._resolve(
                    role, name,
                    required_capabilities=required_capabilities,
                    required_methods=required_methods,
                )
            except (InvalidConfigurationError, UnsupportedPluginError):
                self._telemetry.record(
                    span, operation="resolve_plugin", outcome="refused", reason="plugin_contract_refused",
                )
                span.set_status(Status(StatusCode.ERROR, "plugin_contract_refused"))
                raise
            self._telemetry.record(
                span, operation="resolve_plugin", outcome="accepted", reason="plugin_contract_accepted",
            )
            return resolved

    def _resolve(
        self,
        role: str,
        name: str,
        *,
        required_capabilities: frozenset[str],
        required_methods: tuple[str, ...],
    ) -> ResolvedPlugin:
        if role not in ENTRY_POINT_GROUPS:
            raise InvalidConfigurationError("plugin role is invalid")
        builtin = BUILTIN_PLUGINS.get(role, {}).get(name)
        entry_points = list(metadata.entry_points(group=ENTRY_POINT_GROUPS[role], name=name))
        candidate_count = int(builtin is not None) + len(entry_points)
        if candidate_count == 0:
            raise UnsupportedPluginError(f"{role} plugin is not installed: {name}")
        if candidate_count != 1:
            raise InvalidConfigurationError(f"duplicate {role} plugin name: {name}")
        # Refuse ambiguity before importing any plugin code.
        if builtin is not None:
            factory = _load_reference(builtin)
            distribution, distribution_version = "capability-anatomy", metadata.version("capability-anatomy")
        else:
            entry_point = entry_points[0]
            distribution, distribution_version = entry_point.dist.name, entry_point.dist.version
            factory = entry_point.load()
        plugin = factory if role == "dataset" else factory()
        if plugin.name != name:
            raise InvalidConfigurationError(f"discovered {role} plugin name is inconsistent")
        if plugin.api_version != CORE_API_VERSION:
            raise InvalidConfigurationError(f"discovered {role} plugin API version is incompatible")
        missing = required_capabilities - plugin.capabilities
        if missing:
            raise InvalidConfigurationError(f"discovered {role} plugin lacks required capabilities")
        if any(not callable(getattr(plugin, method, None)) for method in required_methods):
            raise InvalidConfigurationError(f"discovered {role} plugin interface is incomplete")
        return ResolvedPlugin(role, plugin, distribution, distribution_version)
