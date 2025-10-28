"""Plugin registry system for discovering and loading cytoflow-qc plugins."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .base import PluginBase, PluginError, PluginLoadError, PluginVersionError
from .gating import GatingStrategyPlugin
from .qc import QCMethodPlugin
from .stats import StatsMethodPlugin


class PluginRegistry:
    """Registry for discovering, loading, and managing cytoflow-qc plugins."""

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        self._plugins: dict[str, dict[str, PluginBase]] = {
            "gating_strategy": {},
            "qc_method": {},
            "stats_method": {},
        }
        self._plugin_paths: list[Path] = []
        self._loaded_modules: dict[str, Any] = {}

    def register_plugin_path(self, path: str | Path) -> None:
        """Register a directory to search for plugins.

        Args:
            path: Directory path to search for plugin modules
        """
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_dir():
            self._plugin_paths.append(path_obj)
        else:
            raise ValueError(f"Plugin path does not exist or is not a directory: {path}")

    def discover_plugins(self, plugin_type: str | None = None) -> dict[str, list[str]]:
        """Discover available plugins in registered paths.

        Args:
            plugin_type: Specific plugin type to discover (optional)

        Returns:
            Dictionary mapping plugin types to lists of discovered plugin names
        """
        discovered = {ptype: [] for ptype in self._plugins.keys()}

        for path in self._plugin_paths:
            for plugin_file in path.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue

                try:
                    # Load module spec
                    module_name = f"plugin_{plugin_file.stem}_{id(path)}"
                    spec = importlib.util.spec_from_file_location(module_name, plugin_file)

                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)

                        # Find plugin classes
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                issubclass(obj, PluginBase) and
                                obj != PluginBase and
                                hasattr(obj, 'PLUGIN_TYPE')):

                                if plugin_type is None or obj.PLUGIN_TYPE == plugin_type:
                                    discovered[obj.PLUGIN_TYPE].append(obj.PLUGIN_NAME)

                except Exception as e:
                    print(f"Warning: Could not load plugin from {plugin_file}: {e}")

        return discovered

    def load_plugin(self, plugin_type: str, plugin_name: str, config: dict[str, Any] | None = None) -> PluginBase:
        """Load a specific plugin by type and name.

        Args:
            plugin_type: Type of plugin to load
            plugin_name: Name of the plugin to load
            config: Configuration parameters for the plugin

        Returns:
            Loaded plugin instance

        Raises:
            PluginLoadError: If plugin cannot be loaded
        """
        if plugin_type not in self._plugins:
            raise PluginLoadError(f"Unknown plugin type: {plugin_type}")

        # Check if already loaded
        if plugin_name in self._plugins[plugin_type]:
            return self._plugins[plugin_type][plugin_name]

        # Discover plugins if not already done
        if not any(self._plugins[ptype] for ptype in self._plugins):
            self.discover_plugins()

        # Try to find and load the plugin
        for path in self._plugin_paths:
            for plugin_file in path.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue

                try:
                    module_name = f"plugin_{plugin_file.stem}_{id(path)}"
                    if module_name in sys.modules:
                        module = sys.modules[module_name]
                    else:
                        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)

                    # Find plugin class
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                            issubclass(obj, PluginBase) and
                            obj != PluginBase and
                            hasattr(obj, 'PLUGIN_TYPE') and
                            obj.PLUGIN_TYPE == plugin_type and
                            obj.PLUGIN_NAME == plugin_name):

                            # Validate compatibility
                            plugin_instance = obj(config)
                            plugin_instance.validate_compatibility()

                            # Store in registry
                            self._plugins[plugin_type][plugin_name] = plugin_instance
                            return plugin_instance

                except Exception as e:
                    continue

        raise PluginLoadError(f"Could not find plugin '{plugin_name}' of type '{plugin_type}'")

    def get_available_plugins(self, plugin_type: str | None = None) -> dict[str, list[str]]:
        """Get list of available plugins.

        Args:
            plugin_type: Specific plugin type to query (optional)

        Returns:
            Dictionary mapping plugin types to lists of available plugin names
        """
        if not any(self._plugins[ptype] for ptype in self._plugins):
            self.discover_plugins()

        available = {}
        for ptype, plugins in self._plugins.items():
            if plugin_type is None or ptype == plugin_type:
                available[ptype] = list(plugins.keys())

        return available

    def get_plugin_info(self, plugin_type: str, plugin_name: str) -> dict[str, Any]:
        """Get detailed information about a specific plugin.

        Args:
            plugin_type: Type of plugin
            plugin_name: Name of the plugin

        Returns:
            Dictionary with plugin metadata
        """
        plugin = self.load_plugin(plugin_type, plugin_name)

        return {
            "name": plugin.name,
            "version": plugin.version,
            "description": plugin.description,
            "author": plugin.PLUGIN_AUTHOR,
            "email": plugin.PLUGIN_EMAIL,
            "plugin_type": plugin_type,
            "default_config": plugin.get_default_config(),
            "config_schema": plugin.get_config_schema(),
        }

    def create_plugin_instance(
        self,
        plugin_type: str,
        plugin_name: str,
        config: dict[str, Any] | None = None
    ) -> PluginBase:
        """Create a new instance of a plugin with given configuration.

        Args:
            plugin_type: Type of plugin
            plugin_name: Name of the plugin
            config: Configuration parameters

        Returns:
            Plugin instance
        """
        plugin_class = self._find_plugin_class(plugin_type, plugin_name)
        return plugin_class(config)

    def _find_plugin_class(self, plugin_type: str, plugin_name: str) -> type[PluginBase]:
        """Find the plugin class for given type and name."""
        # This would be implemented to search through loaded modules
        # For now, return a placeholder
        raise NotImplementedError("Plugin class discovery not fully implemented")

    def validate_plugin_compatibility(self, plugin: PluginBase) -> None:
        """Validate that a plugin is compatible with the current environment.

        Args:
            plugin: Plugin instance to validate

        Raises:
            PluginVersionError: If plugin is incompatible
        """
        plugin.validate_compatibility()

    def get_plugin_dependencies(self, plugin: PluginBase) -> list[str]:
        """Get list of dependencies required by a plugin.

        Args:
            plugin: Plugin instance

        Returns:
            List of dependency names
        """
        # This would analyze plugin code to find imports
        # For now, return empty list
        return []

    def unload_plugin(self, plugin_type: str, plugin_name: str) -> None:
        """Unload a plugin from memory.

        Args:
            plugin_type: Type of plugin
            plugin_name: Name of the plugin
        """
        if plugin_name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][plugin_name]

    def clear_registry(self) -> None:
        """Clear all loaded plugins from registry."""
        for plugin_type in self._plugins:
            self._plugins[plugin_type].clear()

    def list_plugin_paths(self) -> list[str]:
        """Get list of registered plugin search paths."""
        return [str(path) for path in self._plugin_paths]


# Global registry instance
_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    return _registry


def register_plugin_path(path: str | Path) -> None:
    """Register a plugin search path with the global registry."""
    _registry.register_plugin_path(path)


def load_plugin(plugin_type: str, plugin_name: str, config: dict[str, Any] | None = None) -> PluginBase:
    """Load a plugin using the global registry."""
    return _registry.load_plugin(plugin_type, plugin_name, config)


def discover_plugins(plugin_type: str | None = None) -> dict[str, list[str]]:
    """Discover available plugins using the global registry."""
    return _registry.discover_plugins(plugin_type)


def get_available_plugins(plugin_type: str | None = None) -> dict[str, list[str]]:
    """Get available plugins using the global registry."""
    return _registry.get_available_plugins(plugin_type)














