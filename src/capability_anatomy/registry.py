from __future__ import annotations

import re
from typing import Generic, TypeVar

from .errors import InvalidConfigurationError, UnsupportedPluginError
from .protocols import CORE_API_VERSION, Plugin


PluginT = TypeVar("PluginT", bound=Plugin)
PLUGIN_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]+$")


class PluginRegistry(Generic[PluginT]):
    def __init__(
        self,
        required_capabilities: frozenset[str] = frozenset(),
        required_methods: tuple[str, ...] = (),
    ) -> None:
        self._plugins: dict[str, PluginT] = {}
        self._required_capabilities = required_capabilities
        self._required_methods = required_methods

    def register(self, plugin: PluginT) -> None:
        if not PLUGIN_NAME.fullmatch(plugin.name):
            raise InvalidConfigurationError("plugin name is invalid")
        if plugin.api_version != CORE_API_VERSION:
            raise InvalidConfigurationError("plugin API version is incompatible")
        if not plugin.version:
            raise InvalidConfigurationError("plugin version is required")
        missing = self._required_capabilities - plugin.capabilities
        if missing:
            raise InvalidConfigurationError("plugin lacks required capabilities")
        if any(not callable(getattr(plugin, method, None)) for method in self._required_methods):
            raise InvalidConfigurationError("plugin interface is incomplete")
        if plugin.name in self._plugins:
            raise InvalidConfigurationError("plugin name is already registered")
        self._plugins[plugin.name] = plugin

    def resolve(self, name: str) -> PluginT:
        try:
            return self._plugins[name]
        except KeyError as error:
            raise UnsupportedPluginError(f"plugin is not installed: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))
