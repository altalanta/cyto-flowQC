"""Plugin registry system for discovering and loading cytoflow-qc plugins."""

from __future__ import annotations

import importlib.metadata
from typing import Any, Dict, List, Type

from .base import PluginBase, PluginLoadError

ENTRY_POINT_GROUPS = {
    "gating": "cytoflow_qc.gating_strategies",
    "qc": "cytoflow_qc.qc_methods",
    "stats": "cytoflow_qc.stats_methods",
}

class PluginRegistry:
    """Registry for discovering, loading, and managing cytoflow-qc plugins."""

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        self._plugins: Dict[str, Dict[str, Type[PluginBase]]] = {
            "gating": {},
            "qc": {},
            "stats": {},
        }
        self.discover_plugins()

    def discover_plugins(self) -> None:
        """Discover available plugins via entry points."""
        for plugin_type, group_name in ENTRY_POINT_GROUPS.items():
            for entry_point in importlib.metadata.entry_points(group=group_name):
                try:
                    plugin_class = entry_point.load()
                    if issubclass(plugin_class, PluginBase):
                        self._plugins[plugin_type][entry_point.name] = plugin_class
                except Exception as e:
                    print(f"Warning: Could not load plugin '{entry_point.name}': {e}")

    def get_available_plugins(self, plugin_type: str | None = None) -> Dict[str, List[str]]:
        """Get list of available plugins."""
        if plugin_type:
            if plugin_type not in self._plugins:
                raise ValueError(f"Unknown plugin type: {plugin_type}")
            return {plugin_type: list(self._plugins[plugin_type].keys())}
        return {ptype: list(p.keys()) for ptype, p in self._plugins.items()}

    def get_plugin_class(self, plugin_type: str, plugin_name: str) -> Type[PluginBase]:
        """Get the class for a specific plugin."""
        if plugin_type not in self._plugins or plugin_name not in self._plugins[plugin_type]:
            raise PluginLoadError(f"Plugin '{plugin_name}' of type '{plugin_type}' not found.")
        return self._plugins[plugin_type][plugin_name]

    def load_plugin(
        self, plugin_type: str, plugin_name: str, config: Dict[str, Any] | None = None
    ) -> PluginBase:
        """Load a specific plugin by type and name."""
        plugin_class = self.get_plugin_class(plugin_type, plugin_name)
        return plugin_class(config=config)

# Global registry instance
_registry = PluginRegistry()

def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    return _registry

